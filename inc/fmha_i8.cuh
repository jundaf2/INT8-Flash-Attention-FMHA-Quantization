#pragma once
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <mma.h>
#include <limits>   

#include "fmha_param_i8.h"
using namespace nvcuda;
namespace cg = cooperative_groups;

namespace Detail
{
  float constexpr sqrtNewtonRaphson(float x, float curr, float prev)
    {
        return curr == prev
            ? curr
            : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }
}

/*
* Constexpr version of the square root
* Return value:
*   - For a finite and non-negative value of "x", returns an approximation for the square root of "x"
*   - Otherwise, returns NaN
*/
float constexpr sqrtExpr(float x)
{
    return x >= 0 && x < std::numeric_limits<float>::infinity()
        ? Detail::sqrtNewtonRaphson(x, x, 0)
        : std::numeric_limits<float>::quiet_NaN();
}




namespace gpuImpl {
namespace kernel{


template <int HEAD_DIM, int BASE_SEQ_LEN, int SEQ_LEN, int NUM_WARPS, bool USE_TCU> 
__global__ typename std::enable_if<(USE_TCU==true), void>::type
FMHAInferKernel(const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const int8_t * __restrict__ V, const int8_t *padding_mask, int8_t * __restrict__ O, FMHAParamI8 fmha_param){
  constexpr int TC_SIZE = 16; // int8*int8=int32 WMMA API is [16,16]*[16,16]=[16,16], for details, see https://github.com/jundaf2/CUDA-INT8-GEMM
  constexpr int WARP_SIZE = 32; // Nvidia GPU has 32 threads per warp
  constexpr int CP_SIZE_BYTE = 8; // cosecutive 64 bit
  static_assert(NUM_WARPS == 4 || NUM_WARPS == 8); // warp layout inside a block is [NUM_WARPS,1,1]
  static_assert(NUM_WARPS*TC_SIZE == BASE_SEQ_LEN); // a block of 16 warps is assigned veritically to the S matrix, 16*8 = 128
  

  __shared__ int SLB[BASE_SEQ_LEN*TC_SIZE*2*2*sizeof(int8_t)/sizeof(int)+BASE_SEQ_LEN/sizeof(int)]; // SLB means "shared local buffer" here, 128*16*2*2*sizeof(int8_t) = 8192 bytes
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);


  unsigned batch = blockIdx.x;
  unsigned head = blockIdx.y;
  unsigned head_num = gridDim.y;
  const int warp_id = tile32.meta_group_rank();
  const int lane_id = tile32.thread_rank();

  const int mem_lane_id_x = lane_id % (2); // [0,2]
  const int mem_lane_id_y = lane_id / (2); // [0,16]

  const int reg_lane_id_x = lane_id % (4); // [0,4]
  const int reg_lane_id_y = lane_id / (4); // [0,8]

  constexpr float row_scale = 1.0f/sqrtExpr(static_cast<float>(HEAD_DIM));
  const float scale_qk_out = fmha_param.q_amax * fmha_param.k_amax / (127.f * 127.f);

  int* smem_q[2];
  int* smem_k[2];
  int* smem_p;
  int* smem_v[2];
  smem_q[0] = SLB; 
  smem_q[1] = SLB + 1*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int);
  smem_k[0] = SLB + 2*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int);
  smem_k[1] = SLB + 3*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int);
  smem_p = SLB;
  smem_v[0] = SLB + 2*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int);
  smem_v[1] = SLB + 3*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int);
  int8_t* smem_mask = reinterpret_cast<int8_t*>(SLB + 4*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int));
  
  for(int q_block_start = 0; q_block_start < SEQ_LEN; q_block_start+=BASE_SEQ_LEN){
    int global_start_o = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;
    int global_start_q = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;

    float thread_max_old[2] = {-1e6, -1e6}; 
    float thread_sum_old[2] = {0, 0}; 

    wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> q_int8_frag;
    wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::col_major> k_int8_frag[BASE_SEQ_LEN/TC_SIZE];

    wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> p_int8_frag[BASE_SEQ_LEN/TC_SIZE];
    wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> v_int8_frag;

    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int> s_o_int32_frag[std::max(BASE_SEQ_LEN/TC_SIZE, HEAD_DIM/TC_SIZE)];

    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, float> p_fp32_frag[BASE_SEQ_LEN/TC_SIZE];
    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, float> o_fp32_frag[HEAD_DIM/TC_SIZE];

    #pragma unroll
    for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
      wmma::fill_fragment(o_fp32_frag[xi], 0.f);
    }

    for(int k_block_start = 0; k_block_start < SEQ_LEN; k_block_start+=BASE_SEQ_LEN) 
    {
      int global_start_k = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + k_block_start*HEAD_DIM;
      int global_start_v = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + k_block_start*HEAD_DIM;
      int global_start_mask = batch*SEQ_LEN + k_block_start;

      float thread_max[2] = {-1e6, -1e6};
      float thread_sum[2] = {0, 0};

      
      /***** GEMM 1 *****/
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        wmma::fill_fragment(s_o_int32_frag[xi], 0);
      }

      int k;
      const int8_t *load_gmem_addr_a, *load_gmem_addr_b;
      int store_smem_addr_a, store_smem_addr_b;

      k = 0;
      load_gmem_addr_a = reinterpret_cast<const int8_t*>(Q + global_start_q + (warp_id*TC_SIZE + mem_lane_id_y) * HEAD_DIM + k*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
      load_gmem_addr_b = reinterpret_cast<const int8_t*>(K + global_start_k + (warp_id*TC_SIZE + mem_lane_id_y) * HEAD_DIM + k*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);

      store_smem_addr_a = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_q[k%2]) + (warp_id*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
      store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_k[k%2]) + (warp_id*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
      
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTE));
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));
      
      #pragma unroll
      for(k=1; k<(HEAD_DIM/TC_SIZE); k++){
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        load_gmem_addr_a = reinterpret_cast<const int8_t*>(Q + global_start_q + (warp_id*TC_SIZE + mem_lane_id_y) * HEAD_DIM + k*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
        load_gmem_addr_b = reinterpret_cast<const int8_t*>(K + global_start_k + (warp_id*TC_SIZE + mem_lane_id_y) * HEAD_DIM + k*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);

        store_smem_addr_a = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_q[k%2]) + (warp_id*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
        store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_k[k%2]) + (warp_id*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
        
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTE));
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));

        wmma::load_matrix_sync(q_int8_frag, &(reinterpret_cast<int8_t*>(smem_q[(k-1)%2])[(warp_id*TC_SIZE)*TC_SIZE]), TC_SIZE);
        #pragma unroll
        for(int xi=0; xi<(BASE_SEQ_LEN/TC_SIZE); xi++){
          wmma::load_matrix_sync(k_int8_frag[xi],  &(reinterpret_cast<int8_t*>(smem_k[(k-1)%2])[(xi*TC_SIZE)*TC_SIZE]), TC_SIZE);
          wmma::mma_sync(s_o_int32_frag[xi], q_int8_frag, k_int8_frag[xi], s_o_int32_frag[xi]);
        }
      }

      asm ("cp.async.commit_group;\n" ::);
      asm ("cp.async.wait_group 0;\n" ::);
      __syncthreads();
      k = (HEAD_DIM/TC_SIZE) - 1;
      wmma::load_matrix_sync(q_int8_frag, &(reinterpret_cast<int8_t*>(smem_q[k%2])[(warp_id*TC_SIZE)*TC_SIZE]), TC_SIZE);
      #pragma unroll
      for(int xi=0; xi<(BASE_SEQ_LEN/TC_SIZE); xi++){
        wmma::load_matrix_sync(k_int8_frag[xi],  &(reinterpret_cast<int8_t*>(smem_k[k%2])[(xi*TC_SIZE)*TC_SIZE]), TC_SIZE);
        wmma::mma_sync(s_o_int32_frag[xi], q_int8_frag, k_int8_frag[xi], s_o_int32_frag[xi]);
      }

      /***** Softmax *****/
      // MASK
      #pragma unroll
      for(int i = threadIdx.x; i<BASE_SEQ_LEN/sizeof(int); i+=blockDim.x){
        reinterpret_cast<unsigned int*>(smem_mask)[i] = reinterpret_cast<const unsigned int*>(&padding_mask[global_start_mask])[i];
      } 
      __syncthreads();
      // thread level reduce max
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        #pragma unroll
        for(int tc_yi=0; tc_yi<2; tc_yi++){
          #pragma unroll
          for(int tc_xi=0; tc_xi<2; tc_xi++){
            int tmp_idx = TC_SIZE*xi + tc_xi*8 + reg_lane_id_x*2;
            int8_t mask1 = smem_mask[tmp_idx+0];
            int8_t mask2 = smem_mask[tmp_idx+1];
            int tmp_val1 = mask1 ? -1e6 : s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+0];
            int tmp_val2 = mask2 ? -1e6 : s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+1];
            float tmp_max_val = row_scale*max(tmp_val1, tmp_val2)*scale_qk_out;
            thread_max[tc_yi] = max(thread_max[tc_yi], tmp_max_val);
          }
        }
      }

      // warp level reduce max
      #pragma unroll
      for (int s = 2; s > 0; s >>= 1){
        #pragma unroll
        for(int tc_yi=0; tc_yi<2; tc_yi++){
          thread_max[tc_yi] = max(thread_max[tc_yi], __shfl_xor_sync(0xffffffff, thread_max[tc_yi], s, 4));
        }
      }

      // thread level reduce sum
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        #pragma unroll
        for(int tc_yi=0; tc_yi<2; tc_yi++){
          #pragma unroll
          for(int tc_xi=0; tc_xi<2; tc_xi++){
            int tmp_idx = TC_SIZE*xi + tc_xi*8 + reg_lane_id_x*2;
            int8_t mask1 = smem_mask[tmp_idx+0];
            int8_t mask2 = smem_mask[tmp_idx+1];
            float tmp_sum_val_0 = __expf(row_scale*s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+0]*scale_qk_out - thread_max[tc_yi]);
            float tmp_sum_val_1 = __expf(row_scale*s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+1]*scale_qk_out - thread_max[tc_yi]);
            p_fp32_frag[xi].x[tc_xi*4+tc_yi*2+0] = mask1 ? 0 : tmp_sum_val_0;
            p_fp32_frag[xi].x[tc_xi*4+tc_yi*2+1] = mask2 ? 0 : tmp_sum_val_1;
            thread_sum[tc_yi] += (p_fp32_frag[xi].x[tc_xi*4+tc_yi*2+0] + p_fp32_frag[xi].x[tc_xi*4+tc_yi*2+1]);
          }
        }
      }

      
      // warp level reduce sum
      #pragma unroll
      for (int s = 2; s > 0; s >>= 1){
        #pragma unroll
        for(int tc_yi=0; tc_yi<2; tc_yi++){
          thread_sum[tc_yi] += __shfl_xor_sync(0xffffffff, thread_sum[tc_yi], s, 4);
        }
      }


      // 8*8*2 -> 8*16
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        const float softmax_out_scale = -128.f;
        #pragma unroll
        for(int tc_yi=0; tc_yi<2; tc_yi++){
          #pragma unroll
          for(int tc_xi=0; tc_xi<4; tc_xi++){
            int which_lane_x = (reg_lane_id_x*4+tc_xi)%8/2;
            int which_val_idx = (reg_lane_id_x*4+tc_xi)/8*4 + tc_yi*2 + tc_xi%2;
            float which_val = __shfl_sync(0xffffffff, p_fp32_frag[xi].x[which_val_idx], which_lane_x, 4); 
            int8_t tmp_val = __float2int_rn(max(min(softmax_out_scale * which_val, 0.f), -128.f));
            p_int8_frag[xi].x[tc_yi*4+tc_xi] = tmp_val;
          }
        }
      }

      /***** GEMM 2 *****/
      
      #pragma unroll
      for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
        wmma::fill_fragment(s_o_int32_frag[xi], 0);
      }
      
      k = 0;
      #pragma unroll
      for(int i=warp_id; i<(HEAD_DIM/TC_SIZE); i+=NUM_WARPS){
        load_gmem_addr_b = reinterpret_cast<const int8_t*>(V + global_start_v + (k*TC_SIZE + mem_lane_id_y) * HEAD_DIM + i*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
        store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_v[k%2]) + (i*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));
      }

      #pragma unroll
      for(k=1; k<(BASE_SEQ_LEN/TC_SIZE); k++){
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        #pragma unroll
        for(int i=warp_id; i<(HEAD_DIM/TC_SIZE); i+=NUM_WARPS){
          load_gmem_addr_b = reinterpret_cast<const int8_t*>(V + global_start_v + (k*TC_SIZE + mem_lane_id_y) * HEAD_DIM + i*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
          store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_v[k%2]) + (i*TC_SIZE + mem_lane_id_y)*TC_SIZE + mem_lane_id_x*CP_SIZE_BYTE);
          asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));
        }
        
        for(int xi=0; xi<(HEAD_DIM/TC_SIZE); xi++){
          wmma::load_matrix_sync(v_int8_frag,  &(reinterpret_cast<int8_t*>(smem_v[(k-1)%2])[(xi*TC_SIZE)*TC_SIZE]), TC_SIZE);
          wmma::mma_sync(s_o_int32_frag[xi], p_int8_frag[k-1], v_int8_frag, s_o_int32_frag[xi]);
        }
      }

      k = (BASE_SEQ_LEN/TC_SIZE) - 1;
      asm ("cp.async.commit_group;\n" ::);
      asm ("cp.async.wait_group 0;\n" ::);
      __syncthreads();
      for(int xi=0; xi<(HEAD_DIM/TC_SIZE); xi++){
        wmma::load_matrix_sync(v_int8_frag,  &(reinterpret_cast<int8_t*>(smem_v[k%2])[(xi*TC_SIZE)*TC_SIZE]), TC_SIZE);
        wmma::mma_sync(s_o_int32_frag[xi], p_int8_frag[k], v_int8_frag, s_o_int32_frag[xi]);
      }

      const float scale_o_fp32 = - 1.f;
      #pragma unroll
      for(int tc_yi=0; tc_yi<2; tc_yi++){
        float thread_max_new = max(thread_max_old[tc_yi], thread_max[tc_yi]);
        float exp_max_old = __expf(thread_max_old[tc_yi] - thread_max_new);
        float exp_max = __expf(thread_max[tc_yi] - thread_max_new);
        float thread_sum_new = exp_max_old*thread_sum_old[tc_yi] + exp_max*thread_sum[tc_yi];
        #pragma unroll
        for(int xi=0; xi<(HEAD_DIM/TC_SIZE); xi++){
          #pragma unroll
          for(int tc_xi=0; tc_xi<2; tc_xi++){
            o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+0] = __frcp_rn(thread_sum_new)*(thread_sum_old[tc_yi]*exp_max_old*o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+0] + exp_max*(scale_o_fp32 * s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+0]));
            o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+1] = __frcp_rn(thread_sum_new)*(thread_sum_old[tc_yi]*exp_max_old*o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+1] + exp_max*(scale_o_fp32 * s_o_int32_frag[xi].x[tc_xi*4+tc_yi*2+1]));
          }
        }
        thread_sum_old[tc_yi] = thread_sum_new;
        thread_max_old[tc_yi] = thread_max_new;
      }
    }

    const float scale_v_out = fmha_param.v_amax * __frcp_rn(fmha_param.o_amax * 128.f);
    #pragma unroll
    for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
      #pragma unroll
      for(int tc_yi=0; tc_yi<2; tc_yi++){
        #pragma unroll
        for(int tc_xi=0; tc_xi<2; tc_xi++){
          auto* store_gmem_addr = reinterpret_cast<char2*>(O + global_start_o + (warp_id*TC_SIZE + tc_yi*TC_SIZE/2 +reg_lane_id_y) * HEAD_DIM + xi*TC_SIZE + tc_xi*TC_SIZE/2 + reg_lane_id_x*2);
          char2 tmp_char2;
          tmp_char2.x = __float2int_rn(max(min(scale_v_out * o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+0], 127.f), -127.f));
          tmp_char2.y = __float2int_rn(max(min(scale_v_out * o_fp32_frag[xi].x[tc_xi*4+tc_yi*2+1], 127.f), -127.f));
          *store_gmem_addr = tmp_char2; 
        }
      }
    }
  }
}


template <int HEAD_DIM, int BASE_SEQ_LEN, int SEQ_LEN, int NUM_WARPS, bool USE_TCU> 
__global__ typename std::enable_if<(USE_TCU==false), void>::type
FMHAInferKernel(const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const int8_t * __restrict__ V, const int8_t *padding_mask, int8_t * __restrict__ O, FMHAParamI8 fmha_param) 
{
  
}

  
}
}
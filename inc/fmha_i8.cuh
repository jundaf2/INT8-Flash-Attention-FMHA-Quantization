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
  constexpr int TC_SIZE = 16; // int8*int8=int32 tensor core size is [8,16]*[16,8]=[8,8]
  constexpr int WARP_SIZE_X = 4;
  constexpr int WARP_SIZE = 32; // Nvidia GPU has 32 threads per warp
  constexpr int CP_SIZE_BYTE = 8;
  static_assert(NUM_WARPS == 8); // warp layout inside a block is [8,1,1]
  static_assert(NUM_WARPS*TC_SIZE == BASE_SEQ_LEN); // a block of 16 warps is assigned veritically to the S matrix, 16*8 = 128
  

  __shared__ int SLB[BASE_SEQ_LEN*TC_SIZE*2*2*sizeof(int8_t)/sizeof(int)]; // SLB means "shared local buffer" here, 128*16*2*2*sizeof(int8_t) = 8192 bytes
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);


  unsigned batch = blockIdx.x;
  unsigned head = blockIdx.y;
  unsigned head_num = gridDim.y;
  int warp_id = tile32.meta_group_rank();
  int lane_id = tile32.thread_rank();

  int lane_id_x = lane_id % (WARP_SIZE_X); // [0,4]
  int lane_id_y = lane_id / (WARP_SIZE_X); // [0,8]
  int smem_lane_id_x = lane_id / TC_SIZE; // [0,1]
  int smem_lane_id_y = lane_id % TC_SIZE; // [0,15]
  int gmem_lane_id_x = lane_id % (TC_SIZE / 8); // [0,1]
  int gmem_lane_id_y = lane_id / (TC_SIZE / 8); // [0,15]

  constexpr float row_scale = 1.0f/sqrtExpr(static_cast<float>(HEAD_DIM));
  const float scale_qk_out = fmha_param.q_amax * fmha_param.k_amax / (127.f * 127.f);

  int* smem_q[2];
  int* smem_k[2];
  int* smem_p;
  int* smem_v[2];
  smem_q[0] = SLB; 
  smem_q[1] = SLB + 1*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int32_t);
  smem_k[0] = SLB + 2*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int32_t);
  smem_k[1] = SLB + 3*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int32_t);
  smem_p = SLB;
  smem_v[0] = SLB + 2*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int32_t);
  smem_v[1] = SLB + 3*BASE_SEQ_LEN*TC_SIZE*sizeof(int8_t)/sizeof(int32_t);
  
  for(int q_block_start = 0; q_block_start < SEQ_LEN; q_block_start+=BASE_SEQ_LEN){
    int global_start_o = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;
    int global_start_q = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;

    float thread_max_old = -INFINITY; 
    float thread_sum_old = 0; 

    wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> q_int8_frag;
    wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::col_major> k_int8_frag[BASE_SEQ_LEN/TC_SIZE];
    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int32_t> s_int32_frag[BASE_SEQ_LEN/TC_SIZE];

    wmma::fragment<wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> p_int8_frag[BASE_SEQ_LEN/TC_SIZE];
    wmma::fragment<wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, wmma::row_major> v_int8_frag[HEAD_DIM/TC_SIZE];
    wmma::fragment<wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int32_t> o_int32_frag[HEAD_DIM/TC_SIZE];

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

      float thread_max = -INFINITY;
      float thread_sum = 0; 
      
      /***** GEMM 1 *****/
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        wmma::fill_fragment(s_int32_frag[xi], 0);
      }

      for(int k=0; k<(HEAD_DIM/TC_SIZE); k++){
        const auto* load_gmem_addr_a = reinterpret_cast<const int8_t*>(Q + global_start_q + (warp_id*TC_SIZE + gmem_lane_id_y) * HEAD_DIM + k*TC_SIZE + gmem_lane_id_x*8);
        const auto* load_gmem_addr_b = reinterpret_cast<const int8_t*>(K + global_start_k + (warp_id*TC_SIZE + gmem_lane_id_y) * HEAD_DIM + k*TC_SIZE + gmem_lane_id_x*8);

        int store_smem_addr_a = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_q[k%2]) + (warp_id*TC_SIZE + smem_lane_id_y)*TC_SIZE + smem_lane_id_x*8);
        int store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_k[k%2]) + (warp_id*TC_SIZE + smem_lane_id_y)*TC_SIZE + smem_lane_id_x*8);
        
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_a), "l"(load_gmem_addr_a), "n"(CP_SIZE_BYTE));
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));
        
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        wmma::load_matrix_sync(q_int8_frag, &(reinterpret_cast<int8_t*>(smem_q[k%2])[(warp_id*TC_SIZE)*TC_SIZE]), TC_SIZE);
        for(int xi=0; xi<(BASE_SEQ_LEN/TC_SIZE); xi++){
          wmma::load_matrix_sync(k_int8_frag[xi],  &(reinterpret_cast<int8_t*>(smem_k[k%2])[(warp_id*TC_SIZE)*TC_SIZE]), TC_SIZE);
          wmma::mma_sync(s_int32_frag[xi], q_int8_frag, k_int8_frag[xi], s_int32_frag[xi]);
        }
      }

      /***** Softmax *****/
      // thread level reduce max
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        #pragma unroll
        for(int i=0; i<TC_SIZE/2; i++){
          float tmp_max_val = row_scale*s_int32_frag[xi].x[i];
          thread_max = max(thread_max, tmp_max_val);
        }
      }

      // warp level reduce max
      thread_max = max(thread_max, __shfl_xor_sync(0xffffffff, thread_max, TC_SIZE));

      // thread level reduce sum
      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        #pragma unroll
        for(int i=0; i<TC_SIZE/2; i++){
          float tmp_sum_val = __expf(row_scale*s_int32_frag[xi].x[i] - thread_max);
          p_fp32_frag[xi].x[i] = tmp_sum_val;
          thread_sum += p_fp32_frag[xi].x[i];
        }
      }
      
      // warp level reduce sum
      thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, TC_SIZE);

      #pragma unroll
      for(int xi=0; xi<BASE_SEQ_LEN/TC_SIZE; xi++){
        #pragma unroll
        for(int i=0; i<TC_SIZE/2; i++){
          const float softmax_out_scale = -128.f;
          p_int8_frag[xi].x[i] = __float2int_rn(max(min(softmax_out_scale * p_fp32_frag[xi].x[i], 0.f), -128.f));
        }
      }
      

      /***** GEMM 2 *****/
      #pragma unroll
      for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
        wmma::fill_fragment(o_int32_frag[xi], 0);
      }

      for(int k=0; k<(BASE_SEQ_LEN/TC_SIZE); k++){
        const auto* load_gmem_addr_b = reinterpret_cast<const int8_t*>(V + global_start_v + (warp_id*TC_SIZE + gmem_lane_id_y) * HEAD_DIM + k*TC_SIZE + gmem_lane_id_x*8);

        int store_smem_addr_b = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(smem_v[k%2]) + (warp_id*TC_SIZE + smem_lane_id_y)*TC_SIZE + smem_lane_id_x*8);
        
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" :: "r"(store_smem_addr_b), "l"(load_gmem_addr_b), "n"(CP_SIZE_BYTE));
        
        asm ("cp.async.commit_group;\n" ::);
        asm ("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        for(int xi=0; xi<(HEAD_DIM/TC_SIZE); xi++){
          wmma::load_matrix_sync(v_int8_frag[xi],  &(reinterpret_cast<int8_t*>(smem_v[k%2])[(warp_id*TC_SIZE)*TC_SIZE]), TC_SIZE);
          wmma::mma_sync(o_int32_frag[xi], p_int8_frag[k], v_int8_frag[xi], o_int32_frag[xi]);
        }
      }

      const float scale_o_fp32 = - 1.f;
      float thread_max_new = max(thread_max_old, thread_max);
      float exp_max_old = __expf(thread_max_old-thread_max_new);
      float exp_max = __expf(thread_max-thread_max_new);
      float thread_sum_new = exp_max_old*thread_sum_old + exp_max*thread_sum;

      #pragma unroll
      for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
        #pragma unroll
        for(int i=0; i<TC_SIZE/2; i++){
          o_fp32_frag[xi].x[i] = __frcp_rn(thread_sum_new)*(thread_sum_old*exp_max_old*o_fp32_frag[xi].x[i] + exp_max*(scale_o_fp32 * o_int32_frag[xi].x[i]));
        }
      }

      thread_sum_old = thread_sum_new;
      thread_max_old = thread_max_new;
    }

    uint8_t o_int8_frag[HEAD_DIM/TC_SIZE][TC_SIZE/2];

    const float scale_v_out = fmha_param.v_amax * __frcp_rn(fmha_param.r_amax * 128.f);
    #pragma unroll
    for(int xi=0; xi<HEAD_DIM/TC_SIZE; xi++){
      #pragma unroll
      for(int i=0; i<TC_SIZE/2; i++){
        int8_t tmp_val = __float2int_rn(max(min(scale_v_out * o_fp32_frag[xi].x[i], 127.f), -127.f));
        o_int8_frag[xi][i] = *reinterpret_cast<uint8_t*>(&tmp_val);
      }
      

      auto* store_gmem_addr = reinterpret_cast<uint2*>(O + global_start_o + (warp_id*TC_SIZE + gmem_lane_id_y) * HEAD_DIM + xi*TC_SIZE + gmem_lane_id_x*8);

      uint2 tmp_vr;
      tmp_vr.x = o_int8_frag[xi][0] + (o_int8_frag[xi][1] << 8) + (o_int8_frag[xi][2] << 16) + (o_int8_frag[xi][3] << 24);
      tmp_vr.y = o_int8_frag[xi][4] + (o_int8_frag[xi][5] << 8) + (o_int8_frag[xi][6] << 16) + (o_int8_frag[xi][7] << 24);

      *store_gmem_addr = tmp_vr;
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
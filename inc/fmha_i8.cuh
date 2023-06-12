#pragma once
#include "cuda.h"
#include "fmha_param_i8.h"
#include <limits>   

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
  __shared__ int SLB[6400];

  constexpr int TC_SIZE = 16;
  constexpr int WARP_SIZE = 32;
  static_assert(NUM_WARPS*TC_SIZE == BASE_SEQ_LEN);
  static_assert(NUM_WARPS == 8);

  unsigned batch = blockIdx.x;
  unsigned head = blockIdx.y;
  unsigned head_num = gridDim.y;

  unsigned warp_base = threadIdx.x / WARP_SIZE;
  unsigned warp_id = warp_base >> 5;
  unsigned lane_id = threadIdx.x % WARP_SIZE;

  int lane_id_x = lane_id % TC_SIZE; // [0,15]
  int lane_id_y = lane_id / TC_SIZE; // [0,4]

  constexpr float row_scale = 1.0f/sqrtExpr(static_cast<float>(HEAD_DIM));
  const float scale_qk_out = fmha_param.q_amax * fmha_param.k_amax / (127.f * 127.f);

  int* smem_p[2];
  int* smem_v[2];
  int* smem_q = SLB; 
  int* smem_k = SLB + 2048;
  smem_p[0] = SLB + 2048;
  smem_p[1] = SLB + 3072;
  smem_v[0] = SLB + 4096;
  smem_v[1] = SLB + 5120;
  float* block_sum_max = reinterpret_cast<float*>(SLB + 6144); 

  for(int q_block_start = 0; q_block_start < SEQ_LEN; q_block_start+=BASE_SEQ_LEN){
    int global_start_o = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;
    int global_start_q = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + q_block_start*HEAD_DIM;

    for(int k_block_start = 0; k_block_start < SEQ_LEN; k_block_start+=BASE_SEQ_LEN) 
    {
      int global_start_k = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + k_block_start*HEAD_DIM;
      int global_start_v = (batch * head_num + head) * SEQ_LEN * HEAD_DIM + k_block_start*HEAD_DIM;
      int global_start_mask = batch*SEQ_LEN + k_block_start;

      
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
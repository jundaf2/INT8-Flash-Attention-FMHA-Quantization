#pragma once
#include "cuda.h"
#include "fmha_param_i8.h"

namespace gpuImpl {
namespace kernel{


template <int HEAD_DIM, int BASE_SEQ_LEN, int SEQ_LEN, int NUM_WARPS, bool USE_TCU> 
__global__ typename std::enable_if<(USE_TCU==true), void>::type
FMHAInferKernel(const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const int8_t * __restrict__ V, const int8_t *padding_mask, int8_t * __restrict__ O, FMHAParamI8 fmha_param){

}


template <int HEAD_DIM, int BASE_SEQ_LEN, int SEQ_LEN, int NUM_WARPS, bool USE_TCU> 
__global__ typename std::enable_if<(USE_TCU==false), void>::type
FMHAInferKernel(const int8_t * __restrict__ Q, const int8_t * __restrict__ K, const int8_t * __restrict__ V, const int8_t *padding_mask, int8_t * __restrict__ O, FMHAParamI8 fmha_param) 
{
  
}

  
}
}
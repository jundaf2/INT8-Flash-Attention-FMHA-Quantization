#pragma once
#include <cuda.h>
#include "fmha_param_i8.h"

namespace gpuImpl {

void FMHAInferI8(cudaStream_t stream, 
                  FMHAParamI8 fmha_param,
                  AttnDataDescriptor attn_desc,
                  const void *q,
                  const void *k,
                  const void *v,
                  const void *padding_mask,
                  void *o,
                  const bool use_tcu);

} // namespace gpuImpl
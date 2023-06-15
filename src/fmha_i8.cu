#include "fmha_i8.cuh"
#include "fmha_i8.h"

namespace gpuImpl {

void LaunchFMHAInfer_CC(cudaStream_t stream,
  FMHAParamI8 fmha_param,
  AttnDataDescriptor attn_desc, 
  const int8_t *q,
  const int8_t *k,
  const int8_t *v,
  const int8_t *padding_mask,
  int8_t *o) {

  const dim3 blockDim = {256,1,1}; // 4 warps
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};

  if(attn_desc.seq_len==64)
    kernel::FMHAInferKernel<64, 64, 64, 4, false><<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  else if(attn_desc.seq_len==128)
    kernel::FMHAInferKernel<64, 64, 128, 4, false><<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  else if(attn_desc.seq_len==256)
    kernel::FMHAInferKernel<64, 64, 256, 4, false><<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  else if(attn_desc.seq_len==384)
    kernel::FMHAInferKernel<64, 64, 384, 4, false><<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  else if(attn_desc.seq_len==512)
    kernel::FMHAInferKernel<64, 64, 512, 4, false><<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);

}

void LaunchFMHAInfer_TCU(cudaStream_t stream, 
  FMHAParamI8 fmha_param,
  AttnDataDescriptor attn_desc,
  const int8_t *q,
  const int8_t *k,
  const int8_t *v,
  const int8_t *padding_mask,
  int8_t *o) {
  if(attn_desc.seq_len==128){
  const dim3 blockDim = {256,1,1};
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 128, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }
  else if(attn_desc.seq_len==256){
  const dim3 blockDim = {256,1,1}; 
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 256, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }
  else if(attn_desc.seq_len==384){
  const dim3 blockDim = {256,1,1};
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 384, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }
  else if(attn_desc.seq_len==512){
  const dim3 blockDim = {256,1,1}; 
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 512, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }
  else if(attn_desc.seq_len==1024){
  const dim3 blockDim = {256,1,1}; 
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 1024, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }
  else if(attn_desc.seq_len==2048){
  const dim3 blockDim = {256,1,1}; 
  const dim3 gridDim = {static_cast<unsigned int>(attn_desc.batch_num), static_cast<unsigned int>(attn_desc.head_num),  1};
  kernel::FMHAInferKernel<128, 128, 2048, 8, true> <<<gridDim, blockDim, 0, stream>>>(q,k,v,padding_mask,o,fmha_param);
  }

}

void FMHAInferI8(cudaStream_t stream, 
                  FMHAParamI8 fmha_param,
                  AttnDataDescriptor attn_desc, 
                  const void *q,
                  const void *k,
                  const void *v,
                  const void *padding_mask,
                  void *o,
                  const bool use_tcu) {

  if(!use_tcu){
    LaunchFMHAInfer_CC(stream, fmha_param, attn_desc, reinterpret_cast<const int8_t *>(q), reinterpret_cast<const int8_t *>(k), reinterpret_cast<const int8_t *>(v), reinterpret_cast<const int8_t *>(padding_mask), reinterpret_cast<int8_t *>(o));
  }
  else{
    LaunchFMHAInfer_TCU(stream, fmha_param, attn_desc, reinterpret_cast<const int8_t *>(q), reinterpret_cast<const int8_t *>(k), reinterpret_cast<const int8_t *>(v), reinterpret_cast<const int8_t *>(padding_mask), reinterpret_cast<int8_t *>(o));
  }
}


}
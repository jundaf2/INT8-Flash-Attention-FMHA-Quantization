// System includes
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "cpuGEMM.hpp"
#include "cpuSoftmax.hpp"
#include "utils.hpp"
#include "fmha_i8.h"
#include "fmha_param_i8.h"
using namespace std;

class FMHA {
public:
  FMHA(bool use_tcu, int batch_num, int seq_len, int head_num, int head_dim) {
    this->use_tcu = use_tcu;
    this->batch_num = batch_num;
    this->seq_len = seq_len;
    this->head_num = head_num;
    this->head_dim = head_dim; 

    cout << "compute type=int32" << ", "
          << "data type=int8" << ", "
          << "use_tcu=" << use_tcu << ", "
          << "batch_num=" << batch_num << ", "
          << "seq_len=" << seq_len << ", "
          << "head_num=" << head_num << ", "
          << "head_dim=" << head_dim << ", "
          << endl;

    generateTestData();
  }
  ~FMHA() {
    freeTestData();
  }

  void generateTestData() {
    int len_mask = batch_num*seq_len;
    int len_q = batch_num*seq_len*head_num*head_dim;
    int len_kvo = batch_num*seq_len*head_num*head_dim;
    int len_sp = batch_num*head_num*seq_len*seq_len;

    const auto random_seed = 2023;
    std::mt19937 generator(static_cast<unsigned int>(random_seed));

    h_mat_q = malloc(len_q * sizeof(int8_t));
    h_mat_k = malloc(len_kvo * sizeof(int8_t));
    h_mat_v = malloc(len_kvo * sizeof(int8_t));
    
    h_padding_mask = malloc(len_mask * sizeof(int8_t));

    h_mat_q_ref = malloc(len_q * sizeof(float)); // reference is always float
    h_mat_k_ref = malloc(len_kvo * sizeof(float));
    h_mat_v_ref = malloc(len_kvo * sizeof(float));
    h_padding_mask_ref = (int *) malloc(len_mask * sizeof(int));

    h_mat_s_ref = malloc(len_sp * sizeof(float));

    h_mat_p_ref = malloc(len_sp * sizeof(float));

    h_mat_o = malloc(len_kvo * sizeof(int8_t));
    h_mat_o_ref = malloc(len_kvo * sizeof(float));

    memset(h_mat_s_ref,0,len_sp * sizeof(float));
    memset(h_mat_p_ref,0,len_sp * sizeof(float));
    memset(h_mat_o_ref,0,len_kvo * sizeof(float));

    std::uniform_real_distribution<float> uf_distribution(-1.0f, 1.0f);
    std::bernoulli_distribution b_distribution(0.2); // more 0 than 1

    for(int i=0; i < len_mask; i++){
      reinterpret_cast<int *>(h_padding_mask_ref)[i] = 0;//static_cast<int>(b_distribution(generator)); // 
    }

    for (int i = 0; i < len_q; i++) {
      reinterpret_cast<float *>(h_mat_q_ref)[i] = (i/head_dim%3); // uf_distribution(generator); //uf_distribution(generator); //  (i%4096/64);//1+(i%4096/64)/64.0f;//i%4096/64;//uf_distribution(generator); // // 
    }
    for (int i = 0; i < len_kvo; i++) {
      reinterpret_cast<float *>(h_mat_k_ref)[i] = 1;// uf_distribution(generator); //1+(i%4096/64)/64.0f;// uf_distribution(generator); //  //uf_distribution(generator); //   
    }
    for (int i = 0; i < len_kvo; i++) {
      reinterpret_cast<float *>(h_mat_v_ref)[i] = 1;// uf_distribution(generator); ///32.0f;//  
    }

    for(int i=0; i < len_mask; i++){
      reinterpret_cast<int8_t *>(h_padding_mask)[i] =  reinterpret_cast<int *>(h_padding_mask_ref)[i]; // 
    }

    q_amax = abs_max((float *)h_mat_q_ref, len_q);
    k_amax = abs_max((float *)h_mat_k_ref, len_kvo);
    v_amax = abs_max((float *)h_mat_v_ref, len_kvo);
    
    for(int i = 0; i < len_q; i++){
      float q = ((float *)h_mat_q_ref)[i];
      ((int8_t *)h_mat_q)[i] = float_quant2_int8(q,q_amax);
    }
    for(int i = 0; i < len_kvo; i++){
      float k = ((float *)h_mat_k_ref)[i];
      ((int8_t *)h_mat_k)[i] = float_quant2_int8(k,k_amax);
    }
    for(int i = 0; i < len_kvo; i++){
      float v = ((float *)h_mat_v_ref)[i];
      ((int8_t *)h_mat_v)[i] = float_quant2_int8(v,v_amax);
    }
  }

  void freeTestData() {
    free(h_padding_mask);
    free(h_mat_q);
    free(h_mat_q_ref);
    free(h_mat_k);
    free(h_mat_k_ref);
    free(h_mat_v);
    free(h_mat_v_ref);
    free(h_mat_s_ref);
    free(h_mat_p_ref);
    free(h_mat_o);
    free(h_mat_o_ref);
  }

public:
  void testFMHA() {
    cudaStream_t stream;
    ASSERT_CUDA(cudaStreamCreate(&stream));
    
    int len_mask = batch_num*seq_len;
    int len_q = batch_num*seq_len*head_num*head_dim;
    int len_kvo = batch_num*seq_len*head_num*head_dim;
    int len_sp = batch_num*head_num*seq_len*seq_len;

    int m,n,k;
    int stride_q,stride_k,stride_v,stride_s,stride_p,stride_o;

    // CPU reference
    {
      m = seq_len; n = seq_len; k = head_dim;
      stride_q = seq_len*head_dim; stride_k = seq_len*head_dim; stride_s = seq_len*seq_len;
      cpuGEMM<float, float, float, float>(
          (float *)h_mat_q_ref, (float *)h_mat_k_ref, (float *)h_mat_s_ref, m, n, k,
          stride_q, stride_k, stride_s, batch_num*head_num, static_cast<float>(1), static_cast<float>(0), GEMM_OP_T, GEMM_OP_N,
          nullptr, false);

      cpuAttentionMaskedSoftmax<float, float>(
          (float *)h_mat_s_ref, (float *)h_mat_p_ref, (int *)h_padding_mask_ref, batch_num, seq_len, seq_len, head_num, head_dim);

      s_max = abs_max((float *)h_mat_p_ref, len_sp);
      
      m = seq_len; n = head_dim; k = seq_len;
      stride_p = seq_len*seq_len; stride_v = seq_len*head_dim; stride_o = seq_len*head_dim;
      cpuGEMM<float, float, float, float>(
          (float *)h_mat_p_ref, (float *)h_mat_v_ref, (float *)h_mat_o_ref, m, n, k,
          stride_p, stride_v, stride_o, batch_num*head_num, static_cast<float>(1), static_cast<float>(0), GEMM_OP_T, GEMM_OP_T,
          nullptr, false);

      r_amax = abs_max((float *)h_mat_o_ref, len_kvo);
    }


    ASSERT_CUDA(cudaMalloc(&d_padding_mask, len_mask * sizeof(int8_t))); 
    ASSERT_CUDA(cudaMemcpy(d_padding_mask, h_padding_mask, len_mask * sizeof(int8_t), cudaMemcpyHostToDevice)); 

    ASSERT_CUDA(cudaMalloc(&d_mat_q, len_q * sizeof(int8_t))); 
    ASSERT_CUDA(cudaMalloc(&d_mat_k, len_kvo * sizeof(int8_t)));
    ASSERT_CUDA(cudaMalloc(&d_mat_v, len_kvo * sizeof(int8_t)));
    ASSERT_CUDA(cudaMalloc(&d_mat_o, len_kvo * sizeof(int8_t)));


    ASSERT_CUDA(cudaMemcpy(d_mat_q, h_mat_q, len_q * sizeof(int8_t), cudaMemcpyHostToDevice)); 
    ASSERT_CUDA(cudaMemcpy(d_mat_k, h_mat_k, len_kvo * sizeof(int8_t), cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(d_mat_v, h_mat_v, len_kvo * sizeof(int8_t), cudaMemcpyHostToDevice));

    ASSERT_CUDA(cudaMemset (d_mat_o, 0, len_kvo * sizeof(int8_t)));

    FMHAParamI8 fmha_param;
    fmha_param.q_amax = q_amax;
    fmha_param.k_amax = k_amax;
    fmha_param.v_amax = v_amax;
    fmha_param.r_amax = r_amax;
    fmha_param.s_max = s_max;


    AttnDataDescriptor attn_desc;
    attn_desc.batch_num = batch_num;
    attn_desc.seq_len = seq_len;
    attn_desc.head_num = head_num;
    attn_desc.head_dim = head_dim;


    std::cout << "q_amax: " << q_amax << std::endl;
    std::cout << "k_amax: " << k_amax << std::endl;
    std::cout << "v_amax: " << v_amax << std::endl;
    std::cout << "s_max: " << s_max << std::endl;
    std::cout << "r_amax: " << r_amax << std::endl;

    // warp up the device
    {  
      if(use_tcu) {
        gpuImpl::FMHAInferI8(stream, fmha_param, attn_desc, d_mat_q, d_mat_k, d_mat_v, d_padding_mask, d_mat_o, true);
      }
      else{
        gpuImpl::FMHAInferI8(stream, fmha_param, attn_desc, d_mat_q, d_mat_k, d_mat_v, d_padding_mask, d_mat_o, false);
      }
    }

    // time it
    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    {  
      if(use_tcu) {
        gpuImpl::FMHAInferI8(stream, fmha_param, attn_desc, d_mat_q, d_mat_k, d_mat_v, d_padding_mask, d_mat_o, true);
      }
      else{
        gpuImpl::FMHAInferI8(stream, fmha_param, attn_desc, d_mat_q, d_mat_k, d_mat_v, d_padding_mask, d_mat_o, false);
      }
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds , start, stop);
    
    double   flops = batch_num*head_num*(double)(seq_len*seq_len*head_dim*2+seq_len*seq_len*(4)+seq_len*seq_len*head_dim*2)*1.0;
    double   gigaFlops = (flops * 1.0e-9f) / (milliseconds  / 1000.0f);
    double   bandWidth = batch_num*head_num*(double)(4*seq_len*head_dim)*sizeof(int8_t) / (milliseconds  * 1000 * 1000);
    printf("\033[31;47m FMHA Inference took %.3f ms, %.2f GFlop/s, %.2f GB/s \033[0m\n", milliseconds , gigaFlops, bandWidth);
    ASSERT_CUDA(cudaDeviceSynchronize());
    ASSERT_CUDA(cudaEventDestroy(start));
    ASSERT_CUDA(cudaEventDestroy(stop));
    ASSERT_CUDA(cudaMemcpy(h_mat_o, d_mat_o, len_kvo * sizeof(int8_t), cudaMemcpyDeviceToHost));
    ASSERT_CUDA(cudaFree(d_padding_mask));
    ASSERT_CUDA(cudaFree(d_mat_q));
    ASSERT_CUDA(cudaFree(d_mat_k));
    ASSERT_CUDA(cudaFree(d_mat_v));
    ASSERT_CUDA(cudaFree(d_mat_o));
    ASSERT_CUDA(cudaStreamDestroy(stream));

    print_vec_i8((int8_t *)h_mat_o, "h_mat_o: ", 0, 32*head_dim, head_dim, r_amax);
    print_vec((float *)h_mat_o_ref, "h_mat_o_ref: ", 0, 32*head_dim, head_dim);
    compareResultsWithGoldenI8((int8_t *)h_mat_o, (float *)h_mat_o_ref, len_kvo, r_amax); // o = p*v

  }

protected:

  bool use_tcu;
  int batch_num;
  int seq_len;
  int head_num;
  int head_dim;

  float q_amax = 0.0f;
  float k_amax = 0.0f;
  float v_amax = 0.0f;
  float r_amax = 1.0f;
  float s_max = 1.0f;

  void *h_padding_mask;
  void *h_padding_mask_ref;

  void *h_mat_q;
  void *h_mat_q_ref;
  void *h_mat_k;
  void *h_mat_k_ref;
  void *h_mat_v;
  void *h_mat_v_ref;
  void *h_mat_o;
  void *h_mat_o_ref;

  void *h_mat_s_ref;
  void *h_mat_p_ref;

  void *d_mat_q;
  void *d_mat_k;
  void *d_mat_v;
  void *d_padding_mask;
  void *d_mat_o;
};


int main(int argc, char **argv) {
  bool use_tcu = true;
  int batch_num = 1;
  int seq_len = 512;
  int head_num = 1;
  int head_dim = 128;

  if (argc > 1) {
    use_tcu = atoi(argv[1]);
  }
  if (argc > 2) {
    batch_num = atoi(argv[2]);
  }
  if (argc > 3) {
    seq_len = atoi(argv[3]);
  }
  if (argc > 4) {
    head_num = atoi(argv[4]);
  }
  if (argc > 5) {
    head_dim = atoi(argv[5]);
  }

  FMHA fmha(use_tcu, batch_num, seq_len, head_num, head_dim);
  fmha.testFMHA();

  return 0;
}
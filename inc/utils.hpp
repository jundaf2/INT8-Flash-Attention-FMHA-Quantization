#pragma once
#include <cmath>
#include <iostream>
#include <limits>
#include <assert.h>
#include <cuda.h>

#define ASSERT_CUDA(ret) assert(cudaSuccess==ret)

#define ASSERT_NEAR2(a, b, prec)                                               \
  assert((a != a && b != b) ||                                            \
              (a == std::numeric_limits<typename std::remove_reference<        \
                        decltype(a)>::type>::infinity() &&                     \
                b == std::numeric_limits<typename std::remove_reference<        \
                        decltype(b)>::type>::infinity()) ||                    \
              (-a == std::numeric_limits<typename std::remove_reference<       \
                          decltype(a)>::type>::infinity() &&                    \
                -b == std::numeric_limits<typename std::remove_reference<       \
                          decltype(b)>::type>::infinity()) ||                   \
              (abs(a - b) / abs(a) < prec) || (abs(a - b) / abs(b) < prec) ||  \
              (abs(a - b) < prec))

__inline__ int32_t float_quant2_int8(float x, float x_max) {
  float scale = 127.0f / x_max;
  float i32_f = x * scale;
  int32_t i32 = floorf(i32_f + 0.5);
  i32 = i32 < -127 ? -127 : (i32 > 127 ? 127 : i32);
  return i32;
}

__inline__ float abs_max(float* ptr, int len){
  float max_val = 0.0f;
  for(int i = 0; i < len; i++){
    float val = ptr[i];
    float abs_val  = std::abs(val);
    if(abs_val > max_val) max_val = abs_val;          
  }
  return max_val;
}

void print_vec(const float *outv, std::string outn, int start, int end, int row_size) {
  std::cout << outn << ": ";
  for(int i=start; i<end; i++) {
    std::cout << static_cast<float>(outv[i]) << " ";
    if((i-start+1)%row_size==0) std::cout << std::endl;  
  }
  std::cout << std::endl;
}

void print_vec_i8(const int8_t *outv, std::string outn, int start, int end, int row_size, float r_amax) {
  std::cout << outn << ": ";
  for(int i=start; i<end; i++) {
    std::cout << static_cast<float>(outv[i])/127.0f*r_amax << " ";
    if((i-start+1)%row_size==0) std::cout << std::endl;  
  }
  std::cout << std::endl;
}

void compareResultsWithGoldenI8(const int8_t *res, const float *ref, int len, float r_amax) {
  float deviation_sum = 0;
  for (unsigned int i_row = 0; i_row < len/64; i_row++) {
    float this_row_deviation = 0;
    for (unsigned int i_col = 0; i_col < 64; i_col++) {
      unsigned int i = i_row*64+i_col;
      float res_f = static_cast<float>(res[i])/127.0f*r_amax;
      float ref_f = ref[i];
      float deviation = std::abs(res_f-ref_f);
      deviation_sum += deviation/(std::abs(ref_f)+0.01);
      this_row_deviation += deviation/(std::abs(ref_f)+0.01);
    }
    this_row_deviation = this_row_deviation * 100.0 / 64;
    if(this_row_deviation>20 && false) {
      float temp[64];
      for (unsigned int i_col = 0; i_col < 64; i_col++) {
        unsigned int i = i_row*64+i_col;
        float res_f = static_cast<float>(res[i])/127.0f*r_amax;
        float ref_f = ref[i];
        temp[i_col] = std::abs(res_f-ref_f);
      }
      print_vec_i8(res, "h_mat_o: ", i_row*64, (i_row+1)*64, 64, r_amax);
      print_vec(ref, "h_mat_o_ref: ", i_row*64, (i_row+1)*64, 64);
      print_vec(temp, "abs(h_mat_o - h_mat_o_ref): ", 0, 64, 64);
      printf("\033[31;47m INT8 output absolute deviation at row %d: %f %% \033[0m\n", i_row, this_row_deviation);
    }
  }
  deviation_sum = deviation_sum * 100.0 / len;
  printf("\033[31;47m INT8 average absolute deviation: %f %% \033[0m\n", deviation_sum);
  assert(deviation_sum < 20);
}
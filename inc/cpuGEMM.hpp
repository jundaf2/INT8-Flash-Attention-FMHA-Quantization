#pragma once
#include <cmath>


constexpr bool GEMM_OP_T = true;
constexpr bool GEMM_OP_N = false;

template <typename computeType, typename scaleType, typename inputType,
          typename resultType>
void cpuGEMM(inputType *inputA, inputType *inputB, resultType *resultC, int M,
             int N, int K, int strideA, int strideB, int strideC,
             int batchCount, scaleType alpha, scaleType beta, bool transA,
             bool transB, scaleType *bias = nullptr, bool isColMajorC = true) {
  for (int batch = 0; batch < batchCount; batch++) {
    inputType *A = inputA + batch * strideA;
    inputType *B = inputB + batch * strideB;
    resultType *C = resultC + batch * strideC;
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        computeType sum = 0;
        for (int k = 0; k < K; k++) {
          inputType a = transA ? A[m * K + k] : A[k * M + m];
          inputType b = transB ? B[k * N + n] : B[n * K + k];
          sum += static_cast<computeType>(a) * static_cast<computeType>(b);
        }
        unsigned ci = isColMajorC ? n * M + m : m * N + n;
        C[ci] = static_cast<resultType>(alpha * static_cast<computeType>(sum) +
                                        beta * static_cast<computeType>(C[ci]));
      }
    }
  }
}
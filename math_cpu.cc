#include "math_cpu.h"

// Use BLAS fortran interface
extern "C" {
  void sgemm_(char*, char*, int*, int*, int*, float*, const float*,
              int*, const float*, int*, float*, float*, int*);
  void dgemm_(char*, char*, int*, int*, int*, double*, const double*,
              int*, const double*, int*, double*, double*, int*);
}

template <>
void gemm_cpu<float>(
    char opA, char opB, int m, int n, int k, float alpha,
    const float* A, int lda, const float* B, int ldb, float beta,
    float* C, int ldc) {
  sgemm_(&opB, &opA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

template <>
void gemm_cpu<double>(
    char opA, char opB, int m, int n, int k, double alpha,
    const double* A, int lda, const double* B, int ldb, double beta,
    double* C, int ldc) {
  dgemm_(&opB, &opA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

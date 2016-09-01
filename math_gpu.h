#ifndef RNN2D_MATH_GPU_H_
#define RNN2D_MATH_GPU_H_

#include <cublas_v2.h>

// General matrix multiplication: C = alpha * A * B + beta * C
// handle -> cuBLAS handler
// opA -> operation on A ('N': none, 'T': transpose, 'C': conjugate transpose)
// opB -> operation on B ('N': none, 'T': transpose, 'C': conjugate transpose)
// m -> rows in op(A) and C
// n -> columns in op(B) and C
// k -> columns/rows in op(A)/op(B)
// alpha -> scaling factor for the product of op(A) and op(B)
// A -> row-major matrix A
// lda -> size of the leading dimension (number of columns in a row) in A
// B -> row-major matrix B
// ldb -> size of the leading dimension (number of columns in a row) in B
// beta -> scaling factor for matrix C
// C -> row-major matrix C
// ldc -> size of the leading dimension (number of columns in a row) in C
template <typename T>
inline cublasStatus_t gemm_gpu(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t obB,
    int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb,
    T beta, T* C, int ldc);


/*****************************************************************************
 ** IMPLEMENTATIONS
 *****************************************************************************/

template <>
inline cublasStatus_t gemm_gpu<float>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t obB,
    int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc) {
  return cublasSgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<double>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t obB,
    int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc) {
  return cublasDgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<__half>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t obB,
    int m, int n, int k, __half alpha, const __half* A, int lda,
    const __half* B, int ldb, __half beta, __half* C, int ldc) {
  return cublasHgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

#endif  // RNN2D_MATH_GPU_H_

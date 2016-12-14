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
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb,
    T beta, T* C, int ldc);

template <typename T>
inline cublasStatus_t gemm_gpu(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const T* alpha, const T* A, int lda,
    const T* B, int ldb, const T* beta, T* C, int ldc);

template <typename T>
inline cublasStatus_t gemm_gpu_batched(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, T alpha, const T** A, int lda, const T** B, int ldb,
    T beta, T** C, int ldc, int batch_size);

template <typename T>
inline cublasStatus_t gemm_gpu_batched(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const T* alpha, const T** A, int lda,
    const T** B, int ldb, const T* beta, T** C, int ldc, int batch_size);

template <typename T>
inline cublasStatus_t gemv_gpu(
    cublasHandle_t handle, cublasOperation_t op, int m, int n, T alpha,
    const T* A, int lda, const T* x, int incx, T beta, T* y, int incy);

template <typename T>
inline cublasStatus_t gemv_gpu(
    cublasHandle_t handle, cublasOperation_t op, int m, int n, const T* alpha,
    const T* A, int lda, const T* x, int incx, const T* beta, T* y, int incy);


/*****************************************************************************
 ** IMPLEMENTATIONS
 *****************************************************************************/

template <>
inline cublasStatus_t gemm_gpu<float>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<double>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<__half>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, __half alpha, const __half* A, int lda,
    const __half* B, int ldb, __half beta, __half* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasHgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<float>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemm(handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<double>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemm(handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu<__half>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const __half* alpha, const __half* A, int lda,
    const __half* B, int ldb, const __half* beta, __half* C, int ldc) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasHgemm(handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta,
                     C, ldc);
}

template <>
inline cublasStatus_t gemm_gpu_batched<float>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, float alpha, const float** A, int lda,
    const float** B, int ldb, float beta, float** C, int ldc, int batch_size) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemmBatched(
      handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc,
      batch_size);
}

template <>
inline cublasStatus_t gemm_gpu_batched<double>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, double alpha, const double** A, int lda,
    const double** B, int ldb, double beta, double** C, int ldc,
    int batch_size) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemmBatched(
      handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc,
      batch_size);
}

template <>
inline cublasStatus_t gemm_gpu_batched<float>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const float* alpha, const float** A, int lda,
    const float** B, int ldb, const float* beta, float** C, int ldc,
    int batch_size) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemmBatched(
      handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc,
      batch_size);
}

template <>
inline cublasStatus_t gemm_gpu_batched<double>(
    cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k, const double* alpha, const double** A, int lda,
    const double** B, int ldb, const double* beta, double** C, int ldc,
    int batch_size) {
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemmBatched(
      handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc,
      batch_size);
}

template <>
inline cublasStatus_t gemv_gpu<float>(
    cublasHandle_t handle, cublasOperation_t op, int m, int n, float alpha,
    const float* A, int lda, const float* x, int incx, float beta, float* y,
    int incy) {
  op = op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemv(handle, op, n, m, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline cublasStatus_t gemv_gpu<double>(
    cublasHandle_t handle, cublasOperation_t op, int m, int n, double alpha,
    const double* A, int lda, const double* x, int incx, double beta, double* y,
    int incy) {
  op = op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemv(handle, op, n, m, &alpha, A, lda, x, incx, &beta, y, incy);
}

template <>
inline cublasStatus_t gemv_gpu<float>(
    cublasHandle_t handle, cublasOperation_t op, int m, int n,
    const float* alpha, const float* A, int lda, const float* x, int incx,
    const float* beta, float* y, int incy) {
  op = op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasSgemv(handle, op, n, m, alpha, A, lda, x, incx, beta, y, incy);
}

template <>
inline cublasStatus_t gemv_gpu<double>(
    cublasHandle_t handle, cublasOperation_t op, int m, int n,
    const double* alpha, const double* A, int lda, const double* x, int incx,
    const double* beta, double* y,  int incy) {
  op = op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasStatus_t stat =
      cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  if (stat != CUBLAS_STATUS_SUCCESS) return stat;
  return cublasDgemv(handle, op, n, m, alpha, A, lda, x, incx, beta, y, incy);
}

#endif  // RNN2D_MATH_GPU_H_

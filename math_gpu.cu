#include "math_gpu.h"

#include "utils.h"

template <typename T>
__global__
void kernel_copym_gpu(const int m, const int n, const T* A, const int lda,
                      T* B, const int ldb) {
  if (thGi >= m * n) return;
  const int j = thGi % n;
  const int i = thGi / n;
  B[i * ldb + j] = A[i * lda + j];
}

template <>
void copym_gpu<float>(const int m, const int n, const float* A, const int lda,
                      float* B, const int ldb, cudaStream_t stream) {
  kernel_copym_gpu<float><<<DIV_UP(m * n, 512), 512, 0, stream>>>(
      m, n, A, lda, B, ldb);
}

template <>
void copym_gpu<double>(const int m, const int n, const double* A, const int lda,
                       double* B, const int ldb, cudaStream_t stream) {
  kernel_copym_gpu<double><<<DIV_UP(m * n, 512), 512, 0, stream>>>(
      m, n, A, lda, B, ldb);
}

template <>
void copym_gpu<__half>(const int m, const int n, const __half* A, const int lda,
                       __half* B, const int ldb, cudaStream_t stream) {
  kernel_copym_gpu<__half><<<DIV_UP(m * n, 512), 512, 0, stream>>>(
      m, n, A, lda, B, ldb);
}

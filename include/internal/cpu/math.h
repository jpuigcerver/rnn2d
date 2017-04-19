#ifndef RNN2D_INTERNAL_CPU_MATH_H
#define RNN2D_INTERNAL_CPU_MATH_H

namespace rnn2d {
namespace internal {
namespace cpu {

// General matrix multiplication: C = alpha * A * B + beta * C
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
inline void gemm(
    char opA, char opB, int m, int n, int k, T alpha,
    const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc);

template<typename T>
inline void gemv(
    char op, int m, int n, T alpha, const T* A, int lda, const T* x, int incx,
    T beta, T* y, int incy);


/*****************************************************************************
 ** IMPLEMENTATIONS
 *****************************************************************************/

// Use BLAS fortran interface
extern "C" {
void sgemm_(char*, char*, int*, int*, int*, float*, const float*,
            int*, const float*, int*, float*, float*, int*);
void dgemm_(char*, char*, int*, int*, int*, double*, const double*,
            int*, const double*, int*, double*, double*, int*);
void sgemv_(char*, int*, int*, float*, const float*, int*, const float*,
            int*, float*, float*, int*);
void dgemv_(char*, int*, int*, double*, const double*, int*, const double*,
            int*, double*, double*, int*);
}

template <>
inline void gemm<float>(
    char opA, char opB, int m, int n, int k, float alpha,
    const float* A, int lda, const float* B, int ldb, float beta,
    float* C, int ldc) {
  sgemm_(&opB, &opA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

template <>
inline void gemm<double>(
    char opA, char opB, int m, int n, int k, double alpha,
    const double* A, int lda, const double* B, int ldb, double beta,
    double* C, int ldc) {
  dgemm_(&opB, &opA, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

template <>
inline void gemv<float>(
    char op, int m, int n, float alpha, const float* A, int lda,
    const float* x, int incx, float beta, float* y, int incy) {
  op = op == 'N' ? 'T' : 'N';
  sgemv_(&op, &n, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

template <>
inline void gemv<double>(
    char op, int m, int n, double alpha, const double* A, int lda,
    const double* x, int incx, double beta, double* y, int incy) {
  op = op == 'N' ? 'T' : 'N';
  dgemv_(&op, &n, &m, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif // RNN2D_INTERNAL_CPU_MATH_H

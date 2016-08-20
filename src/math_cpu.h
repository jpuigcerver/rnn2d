#ifndef RNN2D_SRC_MATH_CPU_H_
#define RNN2D_SRC_MATH_CPU_H_

// C = alpha * A * B + beta * C
// m -> rows in op(A) and C
// n -> columns in op(B) and C
// k -> columns/rows in op(A)/op(B)
// alpha -> scaling factor for the product of op(A) and op(B)
// A -> column-major matrix C
// lda -> size of the leading dimension (rows) in op(A)
// B -> column-major matrix B
// ldb -> size of the leading dimension (rows) in op(B)
// beta -> scaling factor for matrix C
// C -> column-major matrix C
// ldc -> size of the leading dimension (rows) in C
template <typename T>
void gemm(
    char opA, char opB, int m, int n, int k, T alpha,
    const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc);

template <>
void gemm<float>(
    char opA, char opB, int m, int n, int k, float alpha,
    const float* A, int lda, const float* B, int ldb, float beta,
    float* C, int ldc);

template <>
void gemm<double>(
    char opA, char opB, int m, int n, int k, double alpha,
    const double* A, int lda, const double* B, int ldb, double beta,
    double* C, int ldc);

#endif  // RNN2D_SRC_MATH_CPU_H_

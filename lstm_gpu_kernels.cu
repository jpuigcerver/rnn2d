#include "lstm_gpu_kernels.h"

#include "activation.h"
#include "lstm_common.h"
#include "utils.h"

#include <cuda_fp16.h>

#ifndef __CUDACC__
#error "This should be compiled with nvcc!"
#endif

template <typename T>
__global__
void kernel_fill(const int n, T* x, const T v) {
  if (thGi >= n) return;
  x[thGi] = v;
}

template <typename T>
__global__
void kernel_init_Q_with_bias(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, T* Q) {
  if (thGi >= 4 * H * W * N * 5 * D) return;
  const int d = thGi % D;                      // d \in [0 ... D-1]
  const int g = (thGi / D) % 5;                // g \in [0 ... 5]
  const int n = (thGi / (5 * D)) % N;          // n \in [0 ... N-1]
  const int x = (thGi / (N * 5 * D)) % W;      // x \in [0 ... W-1]
  const int y = (thGi / (W * N * 5 * D)) % H;  // y \in [0 ... H-1]
  const int z = (thGi / (H * W * N * 5 * D));  // z \in [0 ... 3]
  *Q_ptr(z, y, x, n, g, d) = *B_ptr(z, g, d);
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_fw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, T* Q, T* O) {
  if (thGi >= 4 * Tn * N * D) return;
  const int d = thGi % D;
  const int n = (thGi / D) % N;
  const int e = (thGi / (N * D)) % Tn;
  const int z = (thGi / (Tn * N * D));
  const int i = e + Tmin;
  const int j = t - i;
  const int y  = (z == 0 || z == 1) ? i : H - i - 1;
  const int x  = (z == 0 || z == 2) ? j : W - j - 1;
  const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
  const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
  if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
    const T f_a   = FI::f(*Q_ptr(z, y, x, n, 0, d));  // f_i(input)
    const T f_gi  = FG::f(*Q_ptr(z, y, x, n, 1, d));  // f_g(input gate)
    const T f_go  = FG::f(*Q_ptr(z, y, x, n, 2, d));  // f_g(output gate)
    const T f_gfy = FG::f(*Q_ptr(z, y, x, n, 3, d));  // f_g(forget_y gate)
    const T f_gfx = FG::f(*Q_ptr(z, y, x, n, 4, d));  // f_g(forget_x gate)
    const T C_10  = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 5, d) : 0;
    const T C_01  = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 5, d) : 0;
    *Q_ptr(z, y, x, n, 5, d) = f_gi * f_a + f_gfy * C_10 + f_gfx * C_01;
    *O_ptr(y, x, n, z, d) = f_go * FO::f(*Q_ptr(z, y, x, n, 5, d));
  } else {
    *Q_ptr(z, y, x, n, 5, d) = 0;
    *O_ptr(y, x, n, z, d) = 0;
  }
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_bw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, const T* Q, T* dQ) {
  if (thGi >= 4 * Tn * N * D) return;
  const int d = thGi % D;
  const int n = (thGi / D) % N;
  const int e = (thGi / (N * D)) % Tn;
  const int z = (thGi / (Tn * N * D));
  const int i = e + Tmin;
  const int j = t - i;
  const int y = (z == 0 || z == 1) ? i : H - i - 1;
  const int x = (z == 0 || z == 2) ? j : W - j - 1;
  const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
  const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x
  const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
  const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;  // previous x
  T* dA_00   = dQ_ptr(z, y, x, n, 0, d);
  T* dGi_00  = dQ_ptr(z, y, x, n, 1, d);
  T* dGo_00  = dQ_ptr(z, y, x, n, 2, d);
  T* dGfy_00 = dQ_ptr(z, y, x, n, 3, d);
  T* dGfx_00 = dQ_ptr(z, y, x, n, 4, d);
  T* dC_00   = dQ_ptr(z, y, x, n, 5, d);
  if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
    const T dC_10 = (yn >= 0 && yn < H) ? *dQ_ptr(z, yn, x, n, 5, d) : 0;
    const T dC_01 = (xn >= 0 && xn < W) ? *dQ_ptr(z, y, xn, n, 5, d) : 0;
    const T Gfx_01 = (xn >= 0 && xn < W) ? *Q_ptr(z, y, xn, n, 4, d) : 0;
    const T Gfy_10 = (yn >= 0 && yn < H) ? *Q_ptr(z, yn, x, n, 3, d) : 0;
    const T C_10   = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 5, d) : 0;
    const T C_01   = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 5, d) : 0;
    const T C_00   = *Q_ptr(z, y, x, n, 5, d);
    const T Gfx_00 = *Q_ptr(z, y, x, n, 4, d);
    const T Gfy_00 = *Q_ptr(z, y, x, n, 3, d);
    const T Go_00  = *Q_ptr(z, y, x, n, 2, d);
    const T Gi_00  = *Q_ptr(z, y, x, n, 1, d);
    const T A_00   = *Q_ptr(z, y, x, n, 0, d);
    *dGo_00 = (*dC_00) * FO::f(C_00) * FG::df(Go_00);
    *dC_00  = (*dC_00) * FO::df(C_00) * FG::f(Go_00) +
        dC_10 * FG::f(Gfy_10) + dC_01 * FG::f(Gfx_01);
    *dGfy_00 =
        (yp >= 0 && yp < H) ? (*dC_00) * C_10 * FG::df(Gfy_00) : 0;
    *dGfx_00 =
        (xp >= 0 && xp < W) ? (*dC_00) * C_01 * FG::df(Gfx_00) : 0;
    *dGi_00  = (*dC_00) * FI::f(A_00) * FG::df(Gi_00);
    *dA_00   = (*dC_00) * FI::df(A_00) * FG::f(Gi_00);
  } else {
    *dA_00   = 0;
    *dGi_00  = 0;
    *dGo_00  = 0;
    *dGfy_00 = 0;
    *dGfx_00 = 0;
    *dC_00   = 0;
  }
}

template <typename T>
__global__
void kernel_copy_dO_to_dC(const int H, const int W, const int N, const int D,
                          const int t, const int Tn, const int Tmin,
                          const T* dO, T* dQ) {
  printf("WAKA %d\n", thGi);
  if (thGi >= 4 * Tn * N * D) return;
  const int d = thGi % D;
  const int n = (thGi / D) % N;
  const int e = (thGi / (N * D)) % Tn;
  const int z = (thGi / (Tn * N * D));
  const int i = e + Tmin;
  const int j = t - i;
  const int y = (z == 0 || z == 1) ? i : H - i - 1;
  const int x = (z == 0 || z == 2) ? j : W - j - 1;
  *dQ_ptr(z, y, x, n, 5, d) = *dO_ptr(y, x, n, z, d);
}

#define DEFINE_FILL(T)                                  \
  template <>                                           \
  void fill<T>(const int n, T* x, const T& v) {         \
    kernel_fill<T><<<DIV_UP(n, 512), 512>>>(n, x, v);   \
    CHECK_LAST_CUDA_CALL();                             \
  }

#define DEFINE_INIT_Q_WITH_BIAS(T)                                      \
  template <>                                                           \
  void init_Q_with_bias<T>(                                             \
      const int H, const int W, const int N, const int K, const int D,  \
      const T* P, T* Q) {                                               \
    kernel_init_Q_with_bias<T>                                          \
        <<<DIV_UP(4 * H * W * N * 5 * D, 512), 512>>>(                  \
            H, W, N, K, D, P, Q);                                       \
    CHECK_LAST_CUDA_CALL();                                             \
  }

#define DEFINE_COPY_dO_TO_dC(T)                                         \
  template <>                                                           \
  void copy_dO_to_dC<T>(                                                \
      const int H, const int W, const int N, const int D,               \
      const int t, const int Tn, const int Tmin,                        \
      const T* dO, T* dQ) {                                             \
    kernel_copy_dO_to_dC<T>                                             \
        <<<DIV_UP(4 * Tn * N * D, 512), 512>>>(                         \
            H, W, N, D, t, Tn, Tmin, dO, dQ);                           \
    CHECK_LAST_CUDA_CALL();                                             \
  }

#define DEFINE_FW_ELEMWISE_OPS(T, FG, FI, FO)                           \
  template <>                                                           \
  void fw_elemwise_ops< T, FG<T>, FI<T>, FO<T> >(                       \
      const int H, const int W, const int N, const int D,               \
      const int t, const int Tn, const int Tmin,                        \
      const int* S, T* Q, T* O) {                                       \
    kernel_fw_elemwise_ops< T, FG<T>, FI<T>, FO<T> >                    \
        <<<DIV_UP(4 * Tn * N * D, 512), 512>>>(                         \
            H, W, N, D, t, Tn, Tmin, S, Q, O);                          \
    CHECK_LAST_CUDA_CALL();                                             \
  }

#define DEFINE_BW_ELEMWISE_OPS(T, FG, FI, FO)                           \
  template <>                                                           \
  void bw_elemwise_ops< T, FG<T>, FI<T>, FO<T> >(                       \
      const int H, const int W, const int N, const int D,               \
      const int t, const int Tn, const int Tmin,                        \
      const int* S, const T* Q, T* dQ) {                                \
    kernel_bw_elemwise_ops< T, FG<T>, FI<T>, FO<T> >                    \
        <<<DIV_UP(4 * Tn * N * D, 512), 512>>>(                         \
            H, W, N, D, t, Tn, Tmin, S, Q, dQ);                         \
    CHECK_LAST_CUDA_CALL();                                             \
  }

DEFINE_FILL(float);
DEFINE_FILL(double);
DEFINE_FILL(half);

DEFINE_INIT_Q_WITH_BIAS(float);
DEFINE_INIT_Q_WITH_BIAS(double);
DEFINE_INIT_Q_WITH_BIAS(half);

DEFINE_COPY_dO_TO_dC(float);
DEFINE_COPY_dO_TO_dC(double);
DEFINE_COPY_dO_TO_dC(half);

// ALL LINEAR: USEFUL FOR TESTING
DEFINE_FW_ELEMWISE_OPS(float, Linear, Linear, Linear);
DEFINE_FW_ELEMWISE_OPS(double, Linear, Linear, Linear);
DEFINE_BW_ELEMWISE_OPS(float, Linear, Linear, Linear);
DEFINE_BW_ELEMWISE_OPS(double, Linear, Linear, Linear);

// Regular LSTM: Tanh/Tanh for both input and output activations
DEFINE_FW_ELEMWISE_OPS(float, Sigmoid, Tanh, Tanh);
DEFINE_FW_ELEMWISE_OPS(double, Sigmoid, Tanh, Tanh);
DEFINE_BW_ELEMWISE_OPS(float, Sigmoid, Tanh, Tanh);
DEFINE_BW_ELEMWISE_OPS(double, Sigmoid, Tanh, Tanh);

// Regular LSTM: Linear/Tanh
DEFINE_FW_ELEMWISE_OPS(float, Sigmoid, Linear, Tanh);
DEFINE_FW_ELEMWISE_OPS(double, Sigmoid, Linear, Tanh);
DEFINE_BW_ELEMWISE_OPS(float, Sigmoid, Linear, Tanh);
DEFINE_BW_ELEMWISE_OPS(double, Sigmoid, Linear, Tanh);

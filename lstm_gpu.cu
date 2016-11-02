#include "lstm_gpu.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <glog/logging.h>

#include "activation.h"
#include "cuda_utils.h"
#include "lstm_common.h"
#include "math_gpu.h"

#define MAX_STREAMS 1024

#define STREAMS_CREATE(N)                               \
  for (int z = 0; z < 4; ++z) {                         \
    for (int e = 0; e < (N); ++e) {                     \
      CHECK_CUDA_CALL(cudaStreamCreate(&stream[z][e])); \
    }                                                   \
  }

#define STREAMS_DESTROY(N)                              \
  for (int z = 0; z < 4; ++z) {                         \
    for (int e = 0; e < (N); ++e) {                     \
      CHECK_CUDA_CALL(cudaStreamDestroy(stream[z][e])); \
    }                                                   \
  }

#define STREAMS_SYNCHRONIZE(N)                                  \
  for (int z = 0; z < 4; ++z) {                                 \
    for (int e = 0; e < (N); ++e) {                             \
      CHECK_CUDA_CALL(cudaStreamSynchronize(stream[z][e]));     \
    }                                                           \
  }

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


/* 2D-LSTM forward pass running on the GPU
 * H -> maximum height
 * W -> maximum width
 * N -> batch size
 * K -> input dimensions/channels
 * D -> output dimensions/channels
 * I -> input data (layout: H x W x N x K)
 * S -> input sizes (height and width of each sample, layout: N x 2)
 * P -> parameters (size: 4 * (1 + K + D + D) * 5 * D)
 * O -> output data (layout: H x W x N x 4 x D)
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 6 x D)
 */
template <typename T, typename FG, typename FI, typename FO>
inline void fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, T* O, T* Q) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(Q);
  const int NSZ = std::max(std::min(std::min(H, W), MAX_STREAMS), 4);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));
  cudaStream_t stream[4][MAX_STREAMS];
  STREAMS_CREATE(NSZ);

  // Initialize gates with bias
  kernel_init_Q_with_bias<T><<<DIV_UP(4 * H * W * N * 5 * D, 512), 512>>>(
      H, W, N, K, D, P, Q);
  CHECK_LAST_CUDA_CALL();

  // Multiply inputs by weights.
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][0]);
    CHECK_CUBLAS_CALL(
        gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, H * W * N, 5 * D, K,
                    1.0, I, K,
                    W_ptr(z, 0, 0, 0), 5 * D,
                    1.0, Q_ptr(z, 0, 0, 0, 0, 0), 6 * D));
  }

  // Synchronize streams
  STREAMS_SYNCHRONIZE(1);

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int t = 0; t < H + W - 1; ++t) {
    // Compute number of elements in the u-th diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[z][e % NSZ]));
        // (y, x) coordinates of the e-th element in the z-th diagonal.
        const int i = e + Tmin;
        const int j = t - i;
        const int y  = (z == 0 || z == 1) ? i : H - i - 1;
        const int x  = (z == 0 || z == 2) ? j : W - j - 1;
        const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
        const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
        T* Q_00 = Q_ptr(z, y, x, 0, 0, 0);
        if (yp >= 0 && yp <= H - 1) {
          const T* O_10 = O_ptr(yp, x, 0, z, 0);
          CHECK_CUBLAS_CALL(
              gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                          1.0, O_10, 4 * D,
                          Ry_ptr(z, 0, 0, 0), 5 * D,
                          1.0, Q_00, 6 * D));
        }
        if (xp >= 0 && xp <= W - 1) {
          const T* O_01 = O_ptr(y, xp, 0, z, 0);
          CHECK_CUBLAS_CALL(
              gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                          1.0, O_01, 4 * D,
                          Rx_ptr(z, 0, 0, 0), 5 * D,
                          1.0, Q_00, 6 * D));
        }
      }
    }
    STREAMS_SYNCHRONIZE(Tn);
    kernel_fw_elemwise_ops<T, FG, FI, FO><<<DIV_UP(4 * Tn * N * D, 512), 512>>>(
        H, W, N, D, t, Tn, Tmin, S, Q, O);
  }

  STREAMS_DESTROY(NSZ);
  CHECK_CUBLAS_CALL(cublasDestroy(handle));  // TODO: check for errors
}


/* 2D-LSTM backward pass running on the GPU
 * H -> maximum height
 * W -> maximum width
 * N -> batch size
 * K -> input dimensions/channels
 * D -> output dimensions/channels
 * I -> input data (layout: H x W x N x K)
 * S -> input sizes (height and width of each sample, layout: N x 2)
 * P -> parameters (size: 4 * (1 + K + D + D) * 5 * D)
 * O -> output data (layout: H x W x N x 4 x D)
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 6 x D)
 * dO -> derivative of the loss w.r.t the output
 * dQ -> derivative of the loss w.r.t the internal states
 */
template <typename T, typename FG, typename FI, typename FO>
inline void bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, const T* O, const T* Q, const T* dO,
    T* dQ) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(Q);
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(dQ);
  const int NSZ = std::max(std::min(std::min(H, W), MAX_STREAMS), 4);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));
  cudaStream_t stream[4][MAX_STREAMS];
  STREAMS_CREATE(NSZ);

  // Process the image diagonal-wise, in backwards order (there are H + W - 1
  // diagonals to process)
  for (int t = H + W - 2; t >= 0; --t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;
    kernel_copy_dO_to_dC<T><<<DIV_UP(4 * Tn * N * D, 512), 512>>>(
        H, W, N, D, t, Tn, Tmin, dO, dQ);
    CHECK_LAST_CUDA_CALL();

    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[z][e % NSZ]));
        const int i = e + Tmin;
        const int j = t - i;
        const int y  = (z == 0 || z == 1) ? i : H - i - 1;
        const int x  = (z == 0 || z == 2) ? j : W - j - 1;
        const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
        const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x
        if (yn >= 0 && yn < H) {
          CHECK_CUBLAS_CALL(
              gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                          1.0, dQ_ptr(z, yn, x, 0, 0, 0), 6 * D,
                          Ry_ptr(z, 0, 0, 0), 5 * D,
                          1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D));
        }
        if (xn >= 0 && xn < W) {
          CHECK_CUBLAS_CALL(
              gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                          1.0, dQ_ptr(z, y, xn, 0, 0, 0), 6 * D,
                          Rx_ptr(z, 0, 0, 0), 5 * D,
                          1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D));
        }
      }
    }
    STREAMS_SYNCHRONIZE(Tn);
    kernel_bw_elemwise_ops< T, FG, FI, FO >
        <<<DIV_UP(4 * Tn * N * D, 512), 512>>>(
            H, W, N, D, t, Tn, Tmin, S, Q, dQ);
    CHECK_LAST_CUDA_CALL();
  }

  STREAMS_DESTROY(NSZ);
  CHECK_CUBLAS_CALL(cublasDestroy(handle));
}

template <typename T>
inline void bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, const T* dQ, const T scale, T* dI) {
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(dQ);
  CHECK_NOTNULL(dI);
  // dJ/dI(y,x)
  cublasHandle_t handle;
  cublasCreate(&handle);  // TODO: check for errors
  for (int z = 0; z < 4; ++z) {
    gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, H * W * N, K, 5 * D,
                scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                W_ptr(z, 0, 0, 0), 5 * D,
                1.0, dI, K);
  }
  cublasDestroy(handle);  // TODO: check for errors
}

template <typename T>
inline void bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T* dQ, const T scale, T* dP) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dQ);
  CHECK_NOTNULL(dP);

  cublasHandle_t handle;
  cublasCreate(&handle);  // TODO: check for errors
  cudaStream_t stream[4][4];
  STREAMS_CREATE(4);

  // dJ/db
  T* vOnes = nullptr;
  cudaMalloc(&vOnes, sizeof(T) * H * W * N);
  kernel_fill<T><<<DIV_UP(H * W * N, 512), 512>>>(H * W * N, vOnes, 1);
  CHECK_LAST_CUDA_CALL();
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][0]);
    gemv_gpu<T>(handle, CUBLAS_OP_T, H * W * N, 5 * D,
                scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                vOnes, 1,
                1.0, dB_ptr(z, 0, 0), 1);
  }

  // dJ/dW
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][1]);
    gemm_gpu<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, 5 * D, H * W * N,
                scale, I, K,
                dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                1.0, dW_ptr(z, 0, 0, 0), 5 * D);
  }

  // dJ/dRy
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][2]);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
        if (yp >= 0 && yp < H) {
          gemm_gpu<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, N,
                      scale, O_ptr(yp, x, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRy_ptr(z, 0, 0, 0), 5 * D);
        }
      }
    }
  }

  // dJ/dRx
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][3]);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;  // previous x
        if (xp >= 0 && xp < W) {
          gemm_gpu<T>(handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, N,
                      scale, O_ptr(y, xp, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRx_ptr(z, 0, 0, 0), 5 * D);
        }
      }
    }
  }

  STREAMS_SYNCHRONIZE(4);
  STREAMS_DESTROY(4);
  cublasDestroy(handle);  // TODO: check for errors
}

extern "C" {
  DEFINE_WRAPPERS(gpu, float)
  DEFINE_WRAPPERS(gpu, double)
}  // extern "C"

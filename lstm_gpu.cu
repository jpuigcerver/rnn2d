#include "lstm_gpu.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>

#include <glog/logging.h>
#include <thrust/device_vector.h>

#include "activation.h"
#include "cuda_utils.h"
#include "lstm_common.h"
#include "math_gpu.h"

#define BLOCK_SIZE1D 1024
#define BLOCK_SIZE2D 32

#define STREAMS_CREATE(N)                               \
  for (int i = 0; i < (N); ++i) {                       \
    CHECK_CUDA_CALL(cudaStreamCreate(&stream[i]));      \
  }

#define STREAMS_DESTROY(N)                              \
  for (int i = 0; i < (N); ++i) {                       \
    CHECK_CUDA_CALL(cudaStreamDestroy(stream[i]));      \
  }

#define STREAMS_SYNCHRONIZE(N)                                  \
  for (int i = 0; i < (N); ++i) {                               \
    CHECK_CUDA_CALL(cudaStreamSynchronize(stream[i]));          \
  }

template <typename T>
__global__
void kernel_fill1D(const int n, T* x, const T v) {
  for (int i = thGx; i < n; i += NTGx) {
    x[i] = v;
  }
}

template <typename T>
__global__
void kernel_fill2D(const int n, const int m, T* x, const T v) {
  for (int i = thGy; i < n; i += NTGy) {
    for (int j = thGx; j < m; j += NTGx) {
      x[i * m + j] = v;
    }
  }
}

template <typename T>
__global__
void kernel_init_Q_with_bias(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, T* Q) {
  for (int ii = thGi; ii < 4 * H * W * N * 5 * D; ii += NTG) {
    const int d = ii % D;                      // d \in [0 ... D-1]
    const int g = (ii / D) % 5;                // g \in [0 ... 5]
    const int n = (ii / (5 * D)) % N;          // n \in [0 ... N-1]
    const int x = (ii / (N * 5 * D)) % W;      // x \in [0 ... W-1]
    const int y = (ii / (W * N * 5 * D)) % H;  // y \in [0 ... H-1]
    const int z = (ii / (H * W * N * 5 * D));  // z \in [0 ... 3]
    *Q_ptr(z, y, x, n, g, d) = *B_ptr(z, g, d);
  }
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_fw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, T* Q, T* O) {
  for (int ii = thGi; ii < 4 * Tn * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int e = (ii / (N * D)) % Tn;
    const int z = (ii / (Tn * N * D));
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
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_bw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, T* Q) {
  for (int ii = thGi; ii < 4 * Tn * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int e = (ii / (N * D)) % Tn;
    const int z = (ii / (Tn * N * D));
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
}

template <typename T>
__global__
void kernel_copy_dO_to_dC(const int H, const int W, const int N, const int D,
                          const int t, const int Tn, const int Tmin,
                          const T* dO, T* Q) {
  for (int ii = thGi; ii < 4 * Tn * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int e = (ii / (N * D)) % Tn;
    const int z = (ii / (Tn * N * D));
    const int i = e + Tmin;
    const int j = t - i;
    const int y = (z == 0 || z == 1) ? i : H - i - 1;
    const int x = (z == 0 || z == 2) ? j : W - j - 1;
    *dQ_ptr(z, y, x, n, 5, d) = *dO_ptr(y, x, n, z, d);
  }
}

template <typename T>
__global__
void kernel_copy_Oxp_to_Q(const int H, const int W, const int N, const int D,
                          const T* O, T* Q) {
  for (int ii = thGi; ii < 4 * H * W * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int x = (ii / (N * D)) % W;
    const int y = (ii / (W * N * D)) % H;
    const int z = ii / (H * W * N * D);
    const int xp = (z == 0 || z == 2) ? x - 1 : x + 1; // previous x
    /* Q[(z * D * H * W * N) + (d * H * W * N) + (y * W * N) + (x * N + n)] =
       xp >= 0 && xp < W ? *O_ptr(y, xp, n, z, d) : 0; */
    Q[(z * H * W * N * D) + (y * W * N * D) + (x * N * D) + (n * D) + d] =
        xp >= 0 && xp < W ? *O_ptr(y, xp, n, z, d) : 0;
  }
}

template <typename T>
__global__
void kernel_copy_Oyp_to_Q(const int H, const int W, const int N, const int D,
                          const T* O, T* Q) {
  for (int ii = thGi; ii < 4 * H * W * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int x = (ii / (N * D)) % W;
    const int y = (ii / (W * N * D)) % H;
    const int z = ii / (H * W * N * D);
    const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
    Q[(z * H * W * N * D) + (y * W * N * D) + (x * N * D) + (n * D) + d] =
        yp >= 0 && yp < H ? *O_ptr(yp, x, n, z, d) : 0;
  }
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
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));

  // Initialize gates with bias
  kernel_init_Q_with_bias<T>
      <<<NUM_BLOCKS(4 * H * W * N * 5 * D, BLOCK_SIZE1D), BLOCK_SIZE1D>>>(
          H, W, N, K, D, P, Q);
  CHECK_LAST_CUDA_CALL();

  // Multiply inputs by weights.
  {
    thrust::device_vector<const T*> I_batched_gpu(
        std::vector<const T*>{I, I, I, I});
    thrust::device_vector<const T*> W_batched_gpu(
        std::vector<const T*>{
          W_ptr(0, 0, 0, 0), W_ptr(1, 0, 0, 0),
          W_ptr(2, 0, 0, 0), W_ptr(3, 0, 0, 0)
        });
    thrust::device_vector<T*> Q_batched_gpu(
        std::vector<T*>{
          Q_ptr(0, 0, 0, 0, 0, 0), Q_ptr(1, 0, 0, 0, 0, 0),
          Q_ptr(2, 0, 0, 0, 0, 0), Q_ptr(3, 0, 0, 0, 0, 0)
        });
    CHECK_CUBLAS_CALL(gemm_gpu_batched<T>(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        H * W * N, 5 * D, K,
        1.0, I_batched_gpu.data().get(), K,
        W_batched_gpu.data().get(), 5 * D,
        1.0, Q_batched_gpu.data().get(), 6 * D, 4));
  }

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  {
    std::vector<const T*> Ox_batched, Oy_batched;
    std::vector<const T*> Rx_batched, Ry_batched;
    std::vector<T*> Qx_batched, Qy_batched;
    thrust::device_vector<const T*> Ox_batched_gpu, Oy_batched_gpu;
    thrust::device_vector<const T*> Rx_batched_gpu, Ry_batched_gpu;
    thrust::device_vector<T*> Qx_batched_gpu, Qy_batched_gpu;
    for (int t = 0; t < H + W - 1; ++t) {
      Ox_batched.clear(); Oy_batched.clear();
      Rx_batched.clear(); Ry_batched.clear();
      Qx_batched.clear(); Qy_batched.clear();

      // Compute number of elements in the u-th diagonal
      const int Tmin = std::max(0, t - W + 1);
      const int Tmax = std::min(t, H - 1);
      const int Tn   = (Tmax - Tmin) + 1;
      for (int z = 0; z < 4; ++z) {
        for (int e = 0; e < Tn; ++e) {
          // (y, x) coordinates of the e-th element in the z-th diagonal.
          const int i = e + Tmin;
          const int j = t - i;
          const int y  = (z == 0 || z == 1) ? i : H - i - 1;
          const int x  = (z == 0 || z == 2) ? j : W - j - 1;
          const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
          const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
          if (xp >= 0 && xp <= W - 1) {
            Ox_batched.push_back(O_ptr(y, xp, 0, z, 0));
            Rx_batched.push_back(Rx_ptr(z, 0, 0, 0));
            Qx_batched.push_back(Q_ptr(z, y, x, 0, 0, 0));
          }
          if (yp >= 0 && yp <= H - 1) {
            Oy_batched.push_back(O_ptr(yp, x, 0, z, 0));
            Ry_batched.push_back(Ry_ptr(z, 0, 0, 0));
            Qy_batched.push_back(Q_ptr(z, y, x, 0, 0, 0));
          }
        }
      }

      Rx_batched_gpu.assign(Rx_batched.begin(), Rx_batched.end());
      Ry_batched_gpu.assign(Ry_batched.begin(), Ry_batched.end());
      Ox_batched_gpu.assign(Ox_batched.begin(), Ox_batched.end());
      Oy_batched_gpu.assign(Oy_batched.begin(), Oy_batched.end());
      Qx_batched_gpu.assign(Qx_batched.begin(), Qx_batched.end());
      Qy_batched_gpu.assign(Qy_batched.begin(), Qy_batched.end());
      CHECK_CUBLAS_CALL(
          gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                              1.0, Oy_batched_gpu.data().get(), 4 * D,
                              Ry_batched_gpu.data().get(), 5 * D,
                              1.0, Qy_batched_gpu.data().get(), 6 * D,
                              Qy_batched_gpu.size()));
      CHECK_CUBLAS_CALL(
          gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                              1.0, Ox_batched_gpu.data().get(), 4 * D,
                              Rx_batched_gpu.data().get(), 5 * D,
                              1.0, Qx_batched_gpu.data().get(), 6 * D,
                              Qx_batched_gpu.size()));
      kernel_fw_elemwise_ops<T, FG, FI, FO>
          <<<NUM_BLOCKS(4 * Tn * N * D, BLOCK_SIZE1D), BLOCK_SIZE1D>>>(
              H, W, N, D, t, Tn, Tmin, S, Q, O);
      CHECK_LAST_CUDA_CALL();
    }
  }

  CHECK_CUBLAS_CALL(cublasDestroy(handle));
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
    const T* I, const int* S, const T* P, const T* O, const T* dO,
    T* Q) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(Q);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));

  // Process the image diagonal-wise, in backwards order (there are H + W - 1
  // diagonals to process)
  {
    std::vector<const T*> dQx_batched, dQy_batched;
    std::vector<const T*> Rx_batched, Ry_batched;
    std::vector<T*> dQx2_batched, dQy2_batched;
    thrust::device_vector<const T*> dQx_batched_gpu, dQy_batched_gpu;
    thrust::device_vector<const T*> Rx_batched_gpu, Ry_batched_gpu;
    thrust::device_vector<T*> dQx2_batched_gpu, dQy2_batched_gpu;
    for (int t = H + W - 2; t >= 0; --t) {
      dQx_batched.clear(); dQy_batched.clear();
      Rx_batched.clear(); Ry_batched.clear();
      dQx2_batched.clear(); dQy2_batched.clear();

      // Compute number of elements in the diagonal
      const int Tmin = std::max(0, t - W + 1);
      const int Tmax = std::min(t, H - 1);
      const int Tn   = (Tmax - Tmin) + 1;
      kernel_copy_dO_to_dC<T>
          <<<NUM_BLOCKS(4 * Tn * N * D, BLOCK_SIZE1D), BLOCK_SIZE1D>>>(
              H, W, N, D, t, Tn, Tmin, dO, Q);
      CHECK_LAST_CUDA_CALL();

      for (int z = 0; z < 4; ++z) {
        for (int e = 0; e < Tn; ++e) {
          const int i = e + Tmin;
          const int j = t - i;
          const int y  = (z == 0 || z == 1) ? i : H - i - 1;
          const int x  = (z == 0 || z == 2) ? j : W - j - 1;
          const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
          const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x
          if (xn >= 0 && xn < W) {
            dQx_batched.push_back(dQ_ptr(z, y, xn, 0, 0, 0));
            Rx_batched.push_back(Rx_ptr(z, 0, 0, 0));
            dQx2_batched.push_back(dQ_ptr(z, y, x, 0, 5, 0));
          }
          if (yn >= 0 && yn < H) {
            dQy_batched.push_back(dQ_ptr(z, yn, x, 0, 0, 0));
            Ry_batched.push_back(Ry_ptr(z, 0, 0, 0));
            dQy2_batched.push_back(dQ_ptr(z, y, x, 0, 5, 0));
          }
        }
      }

      Rx_batched_gpu.assign(Rx_batched.begin(), Rx_batched.end());
      Ry_batched_gpu.assign(Ry_batched.begin(), Ry_batched.end());
      dQx_batched_gpu.assign(dQx_batched.begin(), dQx_batched.end());
      dQy_batched_gpu.assign(dQy_batched.begin(), dQy_batched.end());
      dQx2_batched_gpu.assign(dQx2_batched.begin(), dQx2_batched.end());
      dQy2_batched_gpu.assign(dQy2_batched.begin(), dQy2_batched.end());
      CHECK_CUBLAS_CALL(gemm_gpu_batched<T>(
          handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
          1.0, dQx_batched_gpu.data().get(), 6 * D,
          Rx_batched_gpu.data().get(), 5 * D,
          1.0, dQx2_batched_gpu.data().get(), 6 * D, dQx2_batched_gpu.size()));
      CHECK_CUBLAS_CALL(gemm_gpu_batched<T>(
          handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
          1.0, dQy_batched_gpu.data().get(), 6 * D,
          Ry_batched_gpu.data().get(), 5 * D,
          1.0, dQy2_batched_gpu.data().get(), 6 * D, dQy2_batched_gpu.size()));
      kernel_bw_elemwise_ops< T, FG, FI, FO >
          <<<NUM_BLOCKS(4 * Tn * N * D, BLOCK_SIZE1D), BLOCK_SIZE1D>>>(
              H, W, N, D, t, Tn, Tmin, S, Q);
      CHECK_LAST_CUDA_CALL();
    }
  }

  CHECK_CUBLAS_CALL(cublasDestroy(handle));
}

template <typename T>
inline void bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, const T scale, T* dI, T* Q) {
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(dI);
  CHECK_NOTNULL(Q);
  // dJ/dI(y,x)
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, H * W * N, K, 5 * D,
        scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
        W_ptr(z, 0, 0, 0), 5 * D,
        1.0, dI, K));
  }
  CHECK_CUBLAS_CALL(cublasDestroy(handle));
}

template <typename T>
inline void bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T scale, T* dP, T* Q) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dP);
  CHECK_NOTNULL(Q);

  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));
  cudaStream_t stream[4 * 4];
  STREAMS_CREATE(4 * 4);

  // dJ/db
  T* vOnes = Q;
  kernel_fill1D<T>
      <<<DIV_UP(H * W * N, BLOCK_SIZE1D), BLOCK_SIZE1D>>>(H * W * N, vOnes, 1);
  CHECK_LAST_CUDA_CALL();
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[z]));
    CHECK_CUBLAS_CALL(gemv_gpu<T>(
        handle, CUBLAS_OP_T, H * W * N, 5 * D,
        scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
        vOnes, 1,
        1.0, dB_ptr(z, 0, 0), 1));
  }

  // dJ/dW
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[4 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, K, 5 * D, H * W * N,
        scale, I, K,
        dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
        1.0, dW_ptr(z, 0, 0, 0), 5 * D));
  }

  // translate the output tensor in the x-dimension
  T* Oxp = Q + H * W * N;
  kernel_copy_Oxp_to_Q
      <<<DIV_UP(4 * H * W * N * D, BLOCK_SIZE1D), BLOCK_SIZE1D, 0, stream[8]>>>
      (H, W, N, D, O, Oxp);
  // translate the output tensor in the y-dimension
  T* Oyp = Q + H * W * N + 4 * H * W * N * D;
  kernel_copy_Oyp_to_Q
      <<<DIV_UP(4 * H * W * N * D, BLOCK_SIZE1D), BLOCK_SIZE1D, 0, stream[9]>>>
      (H, W, N, D, O, Oyp);
  // wait for data copies
  CHECK_CUDA_CALL(cudaStreamSynchronize(stream[8]));
  CHECK_CUDA_CALL(cudaStreamSynchronize(stream[9]));

  // dJ/dRx
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[8 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, H * W * N,
        scale, Oxp + z * H * W * N * D, D,
        dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
        1.0, dRx_ptr(z, 0, 0, 0), 5 * D));
  }

  // dJ/dRy
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[12 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, H * W * N,
        scale, Oyp + z * H * W * N * D, D,
        dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
        1.0, dRy_ptr(z, 0, 0, 0), 5 * D));
  }

  STREAMS_SYNCHRONIZE(4 * 4);
  STREAMS_DESTROY(4 * 4);
  CHECK_CUBLAS_CALL(cublasDestroy(handle));
}

extern "C" {
  DEFINE_WRAPPERS(gpu, float)
  DEFINE_WRAPPERS(gpu, double)
}  // extern "C"

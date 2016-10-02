#ifndef RNN2D_LSTM_GPU_H_
#define RNN2D_LSTM_GPU_H_

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>

#include "activation.h"
#include "lstm_common.h"
#include "lstm_gpu_kernels.h"
#include "math_gpu.h"
#include "utils.h"

#define MAX_STREAMS 1024

#define STREAMS_CREATE(N)                       \
  for (int z = 0; z < 4; ++z) {                 \
    for (int e = 0; e < (N); ++e) {             \
      cudaStreamCreate(&stream[z][e]);          \
    }                                           \
  }

#define STREAMS_DESTROY(N)                      \
  for (int z = 0; z < 4; ++z) {                 \
    for (int e = 0; e < (N); ++e) {             \
      cudaStreamDestroy(stream[z][e]);          \
    }                                           \
  }

#define STREAMS_SYNCHRONIZE(N)                  \
  for (int z = 0; z < 4; ++z) {                 \
    for (int e = 0; e < (N); ++e) {             \
      cudaStreamSynchronize(stream[z][e]);      \
    }                                           \
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
template < typename T, typename FG, typename FI, typename FO >
void rnn2d_lstm_fw_gpu(const int H, const int W, const int N, const int K,
                       const int D, const T* I, const int* S, const T* P,
                       T* O, T* Q) {
  const int NSZ = std::max(std::min(std::min(H, W), MAX_STREAMS), 4);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  cublasCreate(&handle);  // TODO: check for errors
  cudaStream_t stream[4][MAX_STREAMS];
  STREAMS_CREATE(NSZ);

  // Initialize gates with bias
  init_Q_with_bias<T>(H, W, N, K, D, P, Q);

  // Multiply inputs by weights.
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][0]);
    gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, H * W * N, 5 * D, K,
                1.0, I, K,
                W_ptr(z, 0, 0, 0), 5 * D,
                1.0, Q_ptr(z, 0, 0, 0, 0, 0), 6 * D);
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
        cublasSetStream(handle, stream[z][e % NSZ]);
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
          gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                      1.0, O_10, 4 * D,
                      Ry_ptr(z, 0, 0, 0), 5 * D,
                      1.0, Q_00, 6 * D);
        }
        if (xp >= 0 && xp <= W - 1) {
          const T* O_01 = O_ptr(y, xp, 0, z, 0);
          gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                      1.0, O_01, 4 * D,
                      Rx_ptr(z, 0, 0, 0), 5 * D,
                      1.0, Q_00, 6 * D);
        }
      }
    }
    STREAMS_SYNCHRONIZE(Tn);
    fw_elemwise_ops<T, FG, FI, FO>(H, W, N, D, t, Tn, Tmin, S, Q, O);
  }

  STREAMS_DESTROY(NSZ);
  cublasDestroy(handle);  // TODO: check for errors
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
void rnn2d_lstm_bw_gpu(const int H, const int W, const int N, const int K,
                       const int D, const T* I, const int* S, const T* P,
                       const T* O, const T* Q, const T* dO, T* dQ) {
  const int NSZ = std::max(std::min(std::min(H, W), MAX_STREAMS), 4);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  cublasCreate(&handle);  // TODO: check for errors
  cudaStream_t stream[4][MAX_STREAMS];
  STREAMS_CREATE(NSZ);

  // Process the image diagonal-wise, in backwards order (there are H + W - 1
  // diagonals to process)
  for (int t = H + W - 2; t >= 0; --t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    copy_dO_to_dC<T>(H, W, N, D, t, Tn, Tmin, dO, dQ);

    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        cublasSetStream(handle, stream[z][e % NSZ]);
        const int i = e + Tmin;
        const int j = t - i;
        const int y  = (z == 0 || z == 1) ? i : H - i - 1;
        const int x  = (z == 0 || z == 2) ? j : W - j - 1;
        const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
        const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x
        if (yn >= 0 && yn < H) {
          gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                      1.0, dQ_ptr(z, yn, x, 0, 0, 0), 6 * D,
                      Ry_ptr(z, 0, 0, 0), 5 * D,
                      1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D);
        }
        if (xn >= 0 && xn < W) {
          gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                      1.0, dQ_ptr(z, y, xn, 0, 0, 0), 6 * D,
                      Rx_ptr(z, 0, 0, 0), 5 * D,
                      1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D);
        }
      }
    }
    STREAMS_SYNCHRONIZE(Tn);
    bw_elemwise_ops<T, FG, FI, FO>(H, W, N, D, t, Tn, Tmin, S, Q, dQ);
  }

  STREAMS_DESTROY(NSZ);
  cublasDestroy(handle);  // TODO: check for errors
}

template <typename T>
void rnn2d_lstm_bw_input_gpu(const int H, const int W, const int N, const int K,
                             const int D, const T* P, const T* dQ,
                             const T scale, T* dI) {
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
void rnn2d_lstm_bw_params_gpu(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T* dQ, const T scale, T* dP) {
  cublasHandle_t handle;
  cublasCreate(&handle);  // TODO: check for errors
  cudaStream_t stream[4][4];
  STREAMS_CREATE(4);

  // dJ/db
  T* vOnes = nullptr;
  cudaMalloc(&vOnes, sizeof(T) * H * W * N);
  fill<T>(H * W * N, vOnes, 1);
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

#endif  // RNN2D_LSTM_GPU_H_

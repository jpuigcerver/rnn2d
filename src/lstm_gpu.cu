#include <rnn2d/lstm_gpu.h>

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <iostream>

#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <rnn2d/activation.h>
#include <rnn2d/cuda_utils.h>
#include <rnn2d/lstm_impl.h>
#include <rnn2d/math_gpu.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 128

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
inline size_t get_inference_workspace_size(
    const int H, const int W, const int N, const int D) {
  const size_t tmpd_size = 4 * H * W * N * 5 * D * sizeof(T);
  const size_t ptrs_size = 2 * 3 * 4 * std::min(H, W) * sizeof(T*);
  return tmpd_size + ptrs_size;
}

template <typename T>
inline size_t get_training_workspace_size(
    const int H, const int W, const int N, const int D) {
  const size_t tmpd_size = 3 * 4 * H * W * N * D * sizeof(T);
  const size_t ptrs_size = 2 * 3 * 4 * std::min(H, W) * sizeof(T*);
  return tmpd_size + ptrs_size;
}

template <typename T>
inline size_t get_training_reserve_size(
    const int H, const int W, const int N, const int D) {
  return 4 * H * W * N * 5 * D * sizeof(T);
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
void kernel_init_Q_with_bias(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, T* Q) {
  for (int ii = thGi; ii < 4 * H * W * N * 5 * D; ii += NTG) {
    const int d = ii % D;                      // d \in [0 ... D-1]
    const int g = (ii / D) % 5;                // g \in [0 ... 4]
    const int n = (ii / (5 * D)) % N;          // n \in [0 ... N-1]
    const int x = (ii / (N * 5 * D)) % W;      // x \in [0 ... W-1]
    const int y = (ii / (W * N * 5 * D)) % H;  // y \in [0 ... H-1]
    const int z = (ii / (H * W * N * 5 * D));  // z \in [0 ... 3]
    *Q_ptr(z, y, x, n, g, d) = *B_ptr(P, z, g, d);
  }
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_fw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, T* Q, T* O) {
  for (int ii = thGi; ii < 4 * Tn * N * D; ii += NTG) {
    const int d = ii % D;                // dimension
    const int n = (ii / D) % N;          // batch sample
    const int e = (ii / (N * D)) % Tn;   // element in diagonal
    const int z = (ii / (Tn * N * D));   // direction
    const int i = e + Tmin;
    const int j = t - i;
    const int y  = (z == 0 || z == 2) ? i : H - i - 1;
    const int x  = (z == 0 || z == 1) ? j : W - j - 1;
    if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
      const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
      const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
      const T f_gi  = FG::f(*Q_ptr(z, y, x, n, 0, d));  // input gate
      const T f_gfy = FG::f(*Q_ptr(z, y, x, n, 1, d));  // fgt_y gate
      const T f_gfx = FG::f(*Q_ptr(z, y, x, n, 2, d));  // fgt_x gate
      const T f_go  = FG::f(*Q_ptr(z, y, x, n, 3, d));  // output gate
      const T f_a   = FI::f(*Q_ptr(z, y, x, n, 4, d));  // pre-cell
      const T C_10 = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 4, d) : 0;
      const T C_01 = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 4, d) : 0;
      const T C_00 = f_gi * f_a + f_gfy * C_10 + f_gfx * C_01;  // state
      const T O_00 = f_go * FO::f(C_00);                        // output
      *Q_ptr(z, y, x, n, 0, d) = f_gi;
      *Q_ptr(z, y, x, n, 1, d) = f_gfy;
      *Q_ptr(z, y, x, n, 2, d) = f_gfx;
      *Q_ptr(z, y, x, n, 3, d) = f_go;
      *Q_ptr(z, y, x, n, 4, d) = C_00;
      *O_ptr(O, y, x, n, z, d) = O_00;
    } else {
      *Q_ptr(z, y, x, n, 0, d) = 0;
      *Q_ptr(z, y, x, n, 1, d) = 0;
      *Q_ptr(z, y, x, n, 2, d) = 0;
      *Q_ptr(z, y, x, n, 3, d) = 0;
      *Q_ptr(z, y, x, n, 4, d) = 0;
      *O_ptr(O, y, x, n, z, d) = 0;
    }
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
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 5 x D)
 */
template <typename T, typename FG, typename FI, typename FO>
inline void fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, T* O, void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(wspace);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));

  T* Q = reinterpret_cast<T*>(rspace != nullptr ? rspace : wspace);

  // Initialize gates with bias
  // [A,Gi,Go,Gx,Gy](x,y) = [b_a,b_i,b_o,b_x,b_y]
  kernel_init_Q_with_bias<T><<<GRID_SIZE, BLOCK_SIZE>>>(H, W, N, K, D, P, Q);
  CHECK_LAST_CUDA_CALL();


  // Multiply inputs by weights:
  // [A,Gi,Go,Gx,Gy](x,y) += I(x,y) * [W_a,W_i,W_o,W_x,W_y]
  const T** ptrs_gpu = reinterpret_cast<const T**>(
      (char*)wspace + sizeof(T) * ((rspace != nullptr)
                                   // workspace during training
                                   ? (3 * 4 * H * W * N * D)
                                   // workspace during inference
                                   : (4 * H * W * N * 5 * D)));
  const T** ptrs_cpu = nullptr;
  CHECK_CUDA_CALL(cudaMallocHost(
      &ptrs_cpu, sizeof(const T**) * 2 * 3 * 4 * std::min(H, W)));
  ptrs_cpu[0]  = ptrs_cpu[1] = ptrs_cpu[2] = ptrs_cpu[3] = I;
  ptrs_cpu[4]  = W_ptr(P, 0, 0, 0, 0); ptrs_cpu[5] = W_ptr(P, 1, 0, 0, 0);
  ptrs_cpu[6]  = W_ptr(P, 2, 0, 0, 0); ptrs_cpu[7] = W_ptr(P, 3, 0, 0, 0);
  ptrs_cpu[8]  = Q_ptr(0, 0, 0, 0, 0, 0);
  ptrs_cpu[9]  = Q_ptr(1, 0, 0, 0, 0, 0);
  ptrs_cpu[10] = Q_ptr(2, 0, 0, 0, 0, 0);
  ptrs_cpu[11] = Q_ptr(3, 0, 0, 0, 0, 0);
  CHECK_CUDA_CALL(cudaMemcpy(
      ptrs_gpu, ptrs_cpu, 12 * sizeof(const T**), cudaMemcpyHostToDevice));
  CHECK_CUBLAS_CALL(gemm_gpu_batched<T>(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, H * W * N, 5 * D, K,
      1.0, ptrs_gpu, K,
      ptrs_gpu + 4, 5 * D,
      1.0, const_cast<T**>(ptrs_gpu) + 8, 5 * D, 4));

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  const T** ptrs_cpu_x = ptrs_cpu + 0 * 3 * 4 * std::min(H, W);
  const T** ptrs_cpu_y = ptrs_cpu + 1 * 3 * 4 * std::min(H, W);

  for (int t = 0; t < H + W - 1; ++t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    // Matrix multiplications to compute the input to the gates from the
    // recurrent connections.
    // [A,Gi,Go,Gx,Gy](x,y) += O(x,y-1) * [U_a,U_i,U_o,U_x,U_y]
    // [A,Gi,Go,Gx,Gy](x,y) += O(x-1,y) * [V_a,V_i,V_o,V_x,V_y]
    int batch_mul_size_x = 0, batch_mul_size_y = 0;
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        // (y, x) coordinates of the e-th element in the z-th diagonal.
        const int i = e + Tmin;
        const int j = t - i;
        const int y  = (z == 0 || z == 2) ? i : H - i - 1;
        const int x  = (z == 0 || z == 1) ? j : W - j - 1;
        const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
        const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
        if (yp >= 0 && yp <= H - 1) {
          ptrs_cpu_y[batch_mul_size_y + 0 * 4 * std::min(H, W)] =
              O_ptr(O, yp, x, 0, z, 0);
          ptrs_cpu_y[batch_mul_size_y + 1 * 4 * std::min(H, W)] =
              V_ptr(P, z, 0, 0, 0);
          ptrs_cpu_y[batch_mul_size_y + 2 * 4 * std::min(H, W)] =
              Q_ptr(z, y, x, 0, 0, 0);
          ++batch_mul_size_y;
        }
        if (xp >= 0 && xp <= W - 1) {
          ptrs_cpu_x[batch_mul_size_x + 0 * 4 * std::min(H, W)] =
              O_ptr(O, y, xp, 0, z, 0);
          ptrs_cpu_x[batch_mul_size_x + 1 * 4 * std::min(H, W)] =
              U_ptr(P, z, 0, 0, 0);
          ptrs_cpu_x[batch_mul_size_x + 2 * 4 * std::min(H, W)] =
              Q_ptr(z, y, x, 0, 0, 0);
          ++batch_mul_size_x;
        }
      }
    }
    // Copy pointers to the gpu for batched multiplications
    CHECK_CUDA_CALL(
        cudaMemcpy(ptrs_gpu, ptrs_cpu,
                   sizeof(const T**) * 2 * 3 * 4 * std::min(H, W),
                   cudaMemcpyHostToDevice));
    // [A,Gi,Go,Gx,Gy](x,y) += O(x-1,y) * [V_a,V_i,V_o,V_x,V_y]
    const T** Ox_ptrs = ptrs_gpu + 0 * 4 * std::min(H, W);
    const T** U_ptrs  = ptrs_gpu + 1 * 4 * std::min(H, W);
    T** Qx_ptrs = const_cast<T**>(ptrs_gpu) + 2 * 4 * std::min(H, W);
    CHECK_CUBLAS_CALL(
        gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                            1.0, Ox_ptrs, 4 * D, U_ptrs, 5 * D,
                            1.0, Qx_ptrs, 5 * D, batch_mul_size_x));
    // [A,Gi,Go,Gx,Gy](x,y) += O(x,y-1) * [U_a,U_i,U_o,U_x,U_y]
    const T** Oy_ptrs = ptrs_gpu + (3 + 0) * 4 * std::min(H, W);
    const T** V_ptrs  = ptrs_gpu + (3 + 1) * 4 * std::min(H, W);
    T** Qy_ptrs = const_cast<T**>(ptrs_gpu + (3 + 2) * 4 * std::min(H, W));
    CHECK_CUBLAS_CALL(
        gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                            1.0, Oy_ptrs, 4 * D, V_ptrs, 5 * D,
                            1.0, Qy_ptrs, 5 * D, batch_mul_size_y));
    // Compute cell states
    kernel_fw_elemwise_ops<T, FG, FI, FO>
        <<<GRID_SIZE, BLOCK_SIZE>>>(H, W, N, D, t, Tn, Tmin, S, Q, O);
    CHECK_LAST_CUDA_CALL();
  }
  CHECK_CUBLAS_CALL(cublasDestroy(handle));
  CHECK_CUDA_CALL(cudaFreeHost(ptrs_cpu));
}

template <typename T, typename FG, typename FI, typename FO>
__global__
void kernel_bw_elemwise_ops(const int H, const int W, const int N, const int D,
                            const int t, const int Tn, const int Tmin,
                            const int* S, T* Q, T* Z) {
  for (int ii = thGi; ii < 4 * Tn * N * D; ii += NTG) {
    const int d = ii % D;
    const int n = (ii / D) % N;
    const int e = (ii / (N * D)) % Tn;
    const int z = (ii / (Tn * N * D));
    const int i = e + Tmin;
    const int j = t - i;
    const int y = (z == 0 || z == 2) ? i : H - i - 1;
    const int x = (z == 0 || z == 1) ? j : W - j - 1;
    T* dGi_00 = Q_ptr(z, y, x, n, 0, d);   // currenlty contains f(Gi_00)
    T* dGy_00 = Q_ptr(z, y, x, n, 1, d);   // currently contains f(Gy_00)
    T* dGx_00 = Q_ptr(z, y, x, n, 2, d);   // currently contains f(Gx_00)
    T* dGo_00 = Q_ptr(z, y, x, n, 3, d);   // currently contains f(Go_00)
    T* dA_00  = Q_ptr(z, y, x, n, 4, d);   // currently contains C_00
    if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
      const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
      const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
      const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // previous y
      const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;  // previous x
      const T C_00   = *dA_00;
      const T fGi_00 = *dGi_00;
      const T fGo_00 = *dGo_00;
      const T fGy_00 = *dGy_00;
      const T fGx_00 = *dGx_00;
      const T dO_00  = *Z_ptr(0, z, y, x, n, d);
      const T C_10 = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 4, d) : 0;
      const T C_01 = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 4, d) : 0;
      const T fA_00 = fGi_00 != 0.0 ?
          (C_00 - C_10 * fGy_00 - C_01 * fGx_00) / fGi_00 : 0.0;
      // Z_10 = dC(y+1, x) * f(Gy(y+1, x))
      const T Z_10  = (yn >= 0 && yn < H) ? *Z_ptr(1, z, yn, x, n, d) : 0;
      // Z_01 = dC(y, x+1) * f(Gx(y, x+1))
      const T Z_01  = (xn >= 0 && xn < W) ? *Z_ptr(2, z, y, xn, n, d) : 0;
      const T dC_00  = dO_00 * FO::df(C_00) * fGo_00 + Z_10 + Z_01;
      *dGo_00 = dO_00 * FO::f(C_00) * FG::df2(fGo_00);
      *dGy_00 = (yp >= 0 && yp < H) ? dC_00 * C_10 * FG::df2(fGy_00) : 0;
      *dGx_00 = (xp >= 0 && xp < W) ? dC_00 * C_01 * FG::df2(fGx_00) : 0;
      *dGi_00 = dC_00 * fA_00 * FG::df2(fGi_00);
      *dA_00  = dC_00 * FI::df2(fA_00) * fGi_00;
      *Z_ptr(1, z, y, x, n, d) = dC_00 * fGy_00;
      *Z_ptr(2, z, y, x, n, d) = dC_00 * fGx_00;
    } else {
      *dA_00  = 0;
      *dGi_00 = 0;
      *dGo_00 = 0;
      *dGy_00 = 0;
      *dGx_00 = 0;
      *Z_ptr(1, z, y, x, n, d) = 0;
      *Z_ptr(2, z, y, x, n, d) = 0;
    }
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
    const int xp = (z == 0 || z == 1) ? x - 1 : x + 1; // previous x
    Q[(z * H * W * N * D) + (y * W * N * D) + (x * N * D) + (n * D) + d] =
        xp >= 0 && xp < W ? *O_ptr(O, y, xp, n, z, d) : 0;
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
    const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // previous y
    Q[(z * H * W * N * D) + (y * W * N * D) + (x * N * D) + (n * D) + d] =
        yp >= 0 && yp < H ? *O_ptr(O, yp, x, n, z, d) : 0;
  }
}

template <typename T>
__global__
void kernel_copy_dO_to_Z(const int H, const int W, const int N, const int D,
                         const T* dO, T* Z) {
  for (int ii = thGi; ii < H * W * N * 4 * D; ii += NTG) {
    const int d = ii % D;                    // d \in [0 ... D-1]
    const int z = (ii / D) % 4;              // z \in [0 ... 3]
    const int n = (ii / (4 * D)) % N;        // n \in [0 ... N-1]
    const int x = (ii / (N * 4 * D)) % W;    // x \in [0 ... W-1]
    const int y = (ii / (W * N * 4 * D));    // y \in [0 ... H-1]
    *Z_ptr(0, z, y, x, n, d) = *O_ptr(dO, y, x, n, z, d);
  }
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
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 5 x D)
 * dO -> derivative of the loss w.r.t the output
 * dQ -> derivative of the loss w.r.t the internal states
 */
template <typename T, typename FG, typename FI, typename FO>
inline void bw_data(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, const T* O, const T* dO, T* dI,
    void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(wspace);
  CHECK_NOTNULL(rspace);
  // Prepare cublas handler and streams
  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));

  T* Q = reinterpret_cast<T*>(rspace);
  T* Z = reinterpret_cast<T*>(wspace);

  // Copy errors from the next layer(s) to the workspace
  kernel_copy_dO_to_Z<T><<<GRID_SIZE, BLOCK_SIZE>>>(H, W, N, D, dO, Z);
  CHECK_LAST_CUDA_CALL();

  const T** ptrs_gpu = reinterpret_cast<const T**>(
      (char*)wspace + sizeof(T) * (3 * 4 * H * W * N * D));
  const T** ptrs_cpu = nullptr;
  CHECK_CUDA_CALL(cudaMallocHost(
      &ptrs_cpu, sizeof(const T**) * 2 * 3 * 4 * std::min(H, W)));

  // Process the image diagonal-wise, in backwards order (there are H + W - 1
  // diagonals to process)
  {
    const T** ptrs_cpu_x = ptrs_cpu + 0 * 3 * 4 * std::min(H, W);
    const T** ptrs_cpu_y = ptrs_cpu + 1 * 3 * 4 * std::min(H, W);

    for (int t = H + W - 2; t >= 0; --t) {
      // Compute number of elements in the diagonal
      const int Tmin = std::max(0, t - W + 1);
      const int Tmax = std::min(t, H - 1);
      const int Tn   = (Tmax - Tmin) + 1;

      // Matrix multiplications to compute dJ/dO(x,y).
      // Notice that the loss function is not only affected by the output
      // at time (x,y) but also at times (x+1,y) and (x,y+1)!
      int batch_mul_size_x = 0, batch_mul_size_y = 0;
      for (int z = 0; z < 4; ++z) {
        for (int e = 0; e < Tn; ++e) {
          const int i = e + Tmin;
          const int j = t - i;
          const int y  = (z == 0 || z == 2) ? i : H - i - 1;
          const int x  = (z == 0 || z == 1) ? j : W - j - 1;
          const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
          const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
          if (yn >= 0 && yn < H) {
            ptrs_cpu_y[batch_mul_size_y + 0 * 4 * std::min(H, W)] =
                Q_ptr(z, yn, x, 0, 0, 0);
            ptrs_cpu_y[batch_mul_size_y + 1 * 4 * std::min(H, W)] =
                V_ptr(P, z, 0, 0, 0);
            ptrs_cpu_y[batch_mul_size_y + 2 * 4 * std::min(H, W)] =
                Z_ptr(0, z, y, x, 0, 0);
            ++batch_mul_size_y;
          }
          if (xn >= 0 && xn < W) {
            ptrs_cpu_x[batch_mul_size_x + 0 * 4 * std::min(H, W)] =
                Q_ptr(z, y, xn, 0, 0, 0);
            ptrs_cpu_x[batch_mul_size_x + 1 * 4 * std::min(H, W)] =
                U_ptr(P, z, 0, 0, 0);
            ptrs_cpu_x[batch_mul_size_x + 2 * 4 * std::min(H, W)] =
                Z_ptr(0, z, y, x, 0, 0);
            ++batch_mul_size_x;
          }
        }
      }

      // Copy pointers to the gpu for batched multiplications
      CHECK_CUDA_CALL(
          cudaMemcpy(ptrs_gpu, ptrs_cpu,
                     sizeof(const T**) * 2 * 3 * 4 * std::min(H, W),
                     cudaMemcpyHostToDevice));
      // [A,Gi,Go,Gx,Gy](x,y) += O(x-1,y) * [V_a,V_i,V_o,V_x,V_y]
      const T** dQx_ptrs = ptrs_gpu + 0 * 4 * std::min(H, W);
      const T** U_ptrs   = ptrs_gpu + 1 * 4 * std::min(H, W);
      T** Zx_ptrs = const_cast<T**>(ptrs_gpu) + 2 * 4 * std::min(H, W);
      CHECK_CUBLAS_CALL(
          gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                              1.0, dQx_ptrs, 5 * D, U_ptrs, 5 * D,
                              1.0, Zx_ptrs, D, batch_mul_size_x));
      // [A,Gi,Go,Gx,Gy](x,y) += O(x,y-1) * [U_a,U_i,U_o,U_x,U_y]
      const T** dQy_ptrs = ptrs_gpu + (3 + 0) * 4 * std::min(H, W);
      const T** V_ptrs   = ptrs_gpu + (3 + 1) * 4 * std::min(H, W);
      T** Zy_ptrs = const_cast<T**>(ptrs_gpu + (3 + 2) * 4 * std::min(H, W));
      CHECK_CUBLAS_CALL(
          gemm_gpu_batched<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, D, 5 * D,
                              1.0, dQy_ptrs, 5 * D, V_ptrs, 5 * D,
                              1.0, Zy_ptrs, D, batch_mul_size_y));

      kernel_bw_elemwise_ops<T, FG, FI, FO>
          <<<GRID_SIZE, BLOCK_SIZE>>>(H, W, N, D, t, Tn, Tmin, S, Q, Z);
    }
  }

  // Compute dJ/dI(y,x)
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, H * W * N, K, 5 * D,
        1.0, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
        W_ptr(P, z, 0, 0, 0), 5 * D,
        1.0, dI, K));
  }

  CHECK_CUBLAS_CALL(cublasDestroy(handle));
  CHECK_CUDA_CALL(cudaFreeHost(ptrs_cpu));
}

template <typename T>
inline void bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T scale, T* dP, void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dP);
  CHECK_NOTNULL(wspace);
  CHECK_NOTNULL(rspace);

  T* Q = reinterpret_cast<T*>(rspace);
  T* Z = reinterpret_cast<T*>(wspace);

  cublasHandle_t handle;
  CHECK_CUBLAS_CALL(cublasCreate(&handle));
  cudaStream_t stream[4 * 4];
  STREAMS_CREATE(4 * 4);

  // dJ/db
  T* vOnes = Z;
  kernel_fill1D<T><<<GRID_SIZE, BLOCK_SIZE>>>(H * W * N, vOnes, 1);
  CHECK_LAST_CUDA_CALL();
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[z]));
    CHECK_CUBLAS_CALL(gemv_gpu<T>(
        handle, CUBLAS_OP_T, H * W * N, 5 * D,
        scale, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D, vOnes, 1,
        1.0, B_ptr(dP, z, 0, 0), 1));
  }

  // dJ/dW
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[4 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, K, 5 * D, H * W * N,
        scale, I, K, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
        1.0, W_ptr(dP, z, 0, 0, 0), 5 * D));
  }

  // translate the output tensor in the x-dimension
  T* Oxp = Z + H * W * N;
  kernel_copy_Oxp_to_Q
      <<<GRID_SIZE, BLOCK_SIZE, 0, stream[8]>>>(H, W, N, D, O, Oxp);
  // translate the output tensor in the y-dimension
  T* Oyp = Z + H * W * N + 4 * H * W * N * D;
  kernel_copy_Oyp_to_Q
      <<<GRID_SIZE, BLOCK_SIZE, 0, stream[9]>>>(H, W, N, D, O, Oyp);
  // wait for data copies
  CHECK_CUDA_CALL(cudaStreamSynchronize(stream[8]));
  CHECK_CUDA_CALL(cudaStreamSynchronize(stream[9]));

  // dJ/dRx
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[8 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, H * W * N,
        scale, Oxp + z * H * W * N * D, D,
        Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
        1.0, U_ptr(dP, z, 0, 0, 0), 5 * D));
  }

  // dJ/dRy
  for (int z = 0; z < 4; ++z) {
    CHECK_CUBLAS_CALL(cublasSetStream(handle, stream[12 + z]));
    CHECK_CUBLAS_CALL(gemm_gpu<T>(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, D, 5 * D, H * W * N,
        scale, Oyp + z * H * W * N * D, D,
        Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
        1.0, V_ptr(dP, z, 0, 0, 0), 5 * D));
  }

  STREAMS_SYNCHRONIZE(4 * 4);
  STREAMS_DESTROY(4 * 4);
  CHECK_CUBLAS_CALL(cublasDestroy(handle));
}

extern "C" {
  DEFINE_WRAPPERS(gpu, float)
  DEFINE_WRAPPERS(gpu, double)
}  // extern "C"

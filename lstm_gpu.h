#ifndef RNN2D_LSTM_CPU_H_
#define RNN2D_LSTM_CPU_H_

#include <cmath>
#include <cassert>
#include <cstdio>

#include "activation.h"
#include "lstm_common.h"
#include "math_gpu.h"

#define MAX_DIAG_N 1024

/* === 2D-LSTM EQUATIONS ===
 * Input: I(y,x) is a N x K matrix
 * Output: O(y,x) is a N x D matrix
 *
 * A(y,x)   = I(y,x) * W_a  + O(y-1,x) * R_ay  + O(y,x-1) * R_ax  + B_a
 * Gi(y,x)  = I(y,x) * W_i  + O(y-1,x) * R_iy  + O(y,x-1) * R_ix  + B_i
 * Go(y,x)  = I(y,x) * W_o  + O(y-1,x) * R_oy  + O(y,x-1) * R_ox  + B_o
 * Gfy(y,x) = I(y,x) * W_fy + O(y-1,x) * R_fyy + O(y,x-1) * R_fyx + B_fy
 * Gfx(y,x) = I(y,x) * W_fx + O(y-1,x) * R_fxy + O(y,x-1) * R_fxx + B_fx
 * C(y,x)   = s(Gi(y,x))  · f_i(A(y,x)) +
 *            s(Gfy(y,x)) · C(y-1,x)    +
 *            s(Gfx(y,x)) · C(y,x-1)
 * O(y,x)   = s(Go(y,x))  · f_o(C(y,x))
 *
 * Operator (*) denotes matrix multiplication, operator (·) is element-wise
 * multiplication (or Hadamard product), s(z) is the sigmoid function and,
 * f_i/f_o are any two differentiable activation functions.
 *
 * The previous equations decribe the output when the image is processed in
 * the top-left direction. The equations in the other directions are similar,
 * but the offset for the recurrent connections in each dimension changes:
 *   Top-Left origin:     y,x-offsets = -1, -1
 *   Top-Right origin:    y,x-offsets = -1, +1
 *   Bottom-Left origin:  y,x-offsets = +1, -1
 *   Bottom-Right origin: y,x-offsets = +1, +1
 */

/* 2D-LSTM forward pass running on the CPU
   H -> maximum height
   W -> maximum width
   N -> batch size
   K -> input dimensions/channels
   D -> output dimensions/channels
   I -> input data (layout: H x W x N x K)
   S -> input sizes (height and width of each sample, layout: N x 2)
   P -> 4 x params (layout: [5 x D] (b) + [K x 5 x D] (iW) +
                    [D x 5 x D] (Ry) + [D x 5 x D] (Rx))
   O -> output data (layout: H x W x N x 4 x D)
   Q -> gates pre-activations and cells (layout: 4 x H x W x N x 6 x D)
*/
template < typename T, typename FG, typename FI, typename FO >
void lstm_2d_fw_gpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P[4],
                     T* O, T* Q) {
  // Bias, in each direction
  const T* b[4] = {P[0], P[1], P[2], P[3]};
  // Input weights, in each direction
  const T* iW[4] = {P[0] + 5 * D, P[1] + 5 * D, P[2] + 5 * D, P[3] + 5 * D};
  // Recurrent weights for the y-dimension, in each direction
  const T* Ry[4] = {
    P[0] + 5 * D + K * 5 * D,
    P[1] + 5 * D + K * 5 * D,
    P[2] + 5 * D + K * 5 * D,
    P[3] + 5 * D + K * 5 * D
  };
  // Recurrent weights for the x-dimension, in each direction
  const T* Rx[4] = {
    P[0] + 5 * D + K * 5 * D + D * 5 * D,
    P[1] + 5 * D + K * 5 * D + D * 5 * D,
    P[2] + 5 * D + K * 5 * D + D * 5 * D,
    P[3] + 5 * D + K * 5 * D + D * 5 * D
  };

  cublasHandle_t handle;
  cublasCreate(&handle); // check for errors

  // Streams
  cudaStream_t stream[4][MAX_DIAG_N];
  for (int z = 0; z < 4; ++z) {
    for (int e = 0; e < MAX_DIAG_N; ++e)
      cudaStreamCreate(&stream[z][e]);
  }

  // TODO: Initialize with bias

  // Multiply inputs by weights.
  for (int z = 0; z < 4; ++z) {
    cublasSetStream(handle, stream[z][0]);
    gemm_gpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, H * W * N, 5 * D, K,
                1.0, I, K,
                iW[z], 5 * D,
                1.0, Q_ptr(z, 0, 0, 0, 0, 0), 6 * D);
  }

  // Synchronize streams
  for (int z = 0; z < 4; ++z)
    cudaStreamSynchronize(stream[z][0]);


  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int u = 0; u < H + W - 1; ++u) {
    // Compute number of elements in the u-th diagonal
    const int Umin = std::max(0, u - W + 1);
    const int Umax = std::min(u, H - 1);
    const int Un   = (Umax - Umin) + 1;

    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Un; ++e) {
        cublasSetStream(handle, streams[z][e]);
        // (y, x) coordinates of the e-th element in the z-th diagonal.
        const int i = e + Zmin;
        const int j = u - i;
        const int y  = (z == 0 || z == 1) ? i : H - i - 1;
        const int x  = (z == 0 || z == 2) ? j : W - j - 1;
        const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
        const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
        T* Q_00 = Q_ptr(k, y, x, 0, 0, 0);
        if (yp >= 0 && yp <= H - 1) {
          const T* O_10 = O_ptr(yp, x, 0, k, 0);
          gemm_cpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                      1.0, O_10, 4 * D,
                      Ry[k], 5 * D,
                      1.0, Q_00, 6 * D);
        }
        if (xp >= 0 && xp <= W - 1) {
          const T* O_01 = O_ptr(y, xp, 0, k, 0);
          gemm_cpu<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 5 * D, D,
                      1.0, O_01, 4 * D,
                      Rx[k], 5 * D,
                      1.0, Q_00, 6 * D);
        }
      }
    }
  }
}


/* 2D-LSTM backward pass running on the CPU
   H  -> maximum height
   W  -> maximum width
   N  -> batch size
   K  -> input dimensions/channels
   D  -> output dimensions/channels
   I  -> input data (layout: H x W x N x K)
   S  -> input sizes (height and width of each sample, layout: N x 2)
   P  -> parameters 4 x (layout: [5 x D] (b) + [K x 5 x D] (iW) +
                                 [D x 5 x D] (Ry) + [D x 5 x D] (Rx))
   O  -> output data (layout: H x W x N x 4 x D)
   Q  -> gates pre-activations and cells (layout: H x W x N x 6 x D)
   dO -> derivative of the loss w.r.t the output
   dQ -> derivative of the loss w.r.t the internal states
   dI -> derivative of the loss w.r.t. the input
   dP -> derivative of the loss w.r.t. the parameters
*/
template <typename T, typename FG, typename FI, typename FO>
void lstm_2d_bw_gpu(const int H, const int W, const int N, const int K,
                    const int D, const T* I, const int* S, const T* P[4],
                    const T* O, const T* Q, const T* dO,
                    T* dQ, T* dI, T* dP[4]) {

}

#endif  // RNN2D_LSTM_CPU_H_

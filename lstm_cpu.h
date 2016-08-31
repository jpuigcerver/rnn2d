#ifndef RNN2D_SRC_LSTM_CPU_H_
#define RNN2D_SRC_LSTM_CPU_H_

#include <cmath>
#include <cassert>
#include <cstdio>

#include "activation.h"
#include "lstm_common.h"
#include "math_cpu.h"

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
void lstm_2d_fw_cpu(const int H, const int W, const int N, const int K,
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

  // Initialize input to the block and gates with bias
  #pragma omp parallel for
  for (int i = 0; i < 4 * H * W * N * 5 * D; ++i) {
    const int d   = i % D;
    const int g   = (i / D) % 5;
    const int n   = (i / (5 * D)) % N;
    const int x   = (i / (N * 5 * D)) % W;
    const int y   = (i / (W * N * 5 * D)) % H;
    const int k   = i / (H * W * N * 5 * D);
    *Q_ptr(k, y, x, n, g, d) = b[k][g * D + d];
  }

  // Multiply inputs by weights. Each direction can run in parallel
  for (int k = 0; k < 4; ++k) {
    T* Qk = Q + k * H * W * N * 6 * D;
    gemm_cpu<T>('N', 'N', H * W * N, 5 * D, K,
                1.0, I, K,          /* I reshaped as (H * W * N) x K */
                iW[k], 5 * D,       /* iW reshaped as K x (5 * D) */
                1.0, Qk, 6 * D);    /* Qk reshaped as (H * W * N) x (6 * D),
                                       notice that only first 5 * D columns are
                                       used. */
  }

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int z = 0; z < H + W - 1; ++z) {
    // Compute number of elements in the diagonal
    const int Zmin = std::max(0, z - W + 1);
    const int Zmax = std::min(z, H - 1);
    const int Zn   = (Zmax - Zmin) + 1;

    // Propagate recurrent connections
    #pragma omp parallel for
    for (int e = 0; e < 4 * Zn; ++e) {
      const int k = e / Zn; // Diagonal direction
      // (y, x) coordinates of the e-th element in the z-th diagonal.
      const int i = (e % Zn) + Zmin;
      const int j = z - i;
      const int y  = (k == 0 || k == 1) ? i : H - i - 1;
      const int x  = (k == 0 || k == 2) ? j : W - j - 1;
      const int yp = (k == 0 || k == 1) ? y - 1 : y + 1;
      const int xp = (k == 0 || k == 2) ? x - 1 : x + 1;
      assert(y >= 0 && x >= 0 && y < H && x < W);
      T* Q_00 = Q_ptr(k, y, x, 0, 0, 0);
      if (yp >= 0 && yp <= H - 1) {
        const T* O_10 = O_ptr(yp, x, 0, k, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_10, 4 * D,   /* O_10 reshaped as N x (4 * D). */
                    Ry[k], 5 * D,       /* Ry reshaped as D x (5 * D) */
                    1.0, Q_00, 6 * D);  /* Q reshaped as (H * W * N) x (6 * D),
                                           notice that only first 5 * D columns
                                           are used. */
      }
      if (xp >= 0 && xp <= W - 1) {
        const T* O_01 = O_ptr(y, xp, 0, k, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_01, 4 * D,   /* O_01 reshaped as N x (4 * D). */
                    Rx[k], 5 * D,       /* Rx reshaped as D x (5 * D) */
                    1.0, Q_00, 6 * D);  /* Q reshaped as (H * W * N) x (6 * D),
                                           notice that only first 5 * D columns
                                           are used. */
      }
    }

    // Compute cell and output values
    #pragma omp parallel for
    for (int e = 0; e < 4 * Zn * N * D; ++e) {
      const int d = e % D;
      const int n = (e / D) % N;
      const int k = (e / (Zn * N * D));
      // (y, x) coordinates of the e-th element in the z-th diagonal.
      const int i = ((e / (N * D)) % Zn) + Zmin;
      const int j = z - i;
      const int y  = (k == 0 || k == 1) ? i : H - i - 1;
      const int x  = (k == 0 || k == 2) ? j : W - j - 1;
      const int yp = (k == 0 || k == 1) ? y - 1 : y + 1;
      const int xp = (k == 0 || k == 2) ? x - 1 : x + 1;
      T* C_00 = Q_ptr(k, y, x, n, 5, d);
      T* O_00 = O_ptr(y, x, n, k, d);
      if (y < S[n * 2] && x < S[n * 2 + 1]) {
        const T f_a   = FI::f(*Q_ptr(k, y, x, n, 0, d));  // f(input)
        const T f_gi  = FG::f(*Q_ptr(k, y, x, n, 1, d));  // input gate
        const T f_go  = FG::f(*Q_ptr(k, y, x, n, 2, d));  // output gate
        const T f_gfy = FG::f(*Q_ptr(k, y, x, n, 3, d));  // forget_y gate
        const T f_gfx = FG::f(*Q_ptr(k, y, x, n, 4, d));  // forget_x gate
        const T C_10  = (yp >= 0 && yp < H) ? *Q_ptr(k, yp, x, n, 5, d) : 0;
        const T C_01  = (xp >= 0 && xp < W) ? *Q_ptr(k, y, xp, n, 5, d) : 0;
        *C_00 = f_gi * f_a + f_gfy * C_10 + f_gfx * C_01;
        *O_00 = f_go * FO::f(*C_00);
      } else {
        *C_00 = 0;
        *O_00 = 0;
      }
    }
  }
}

template <typename T>
inline void copym_cpu(int m, int n, const T* A, int lda, T* B, const int ldb) {
  #pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      B[i * ldb + j] = A[i * lda + j];
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
void lstm_2d_bw_cpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P[4],
                     const T* O, const T* Q, const T* dO,
                     T* dQ, T* dI, T* dP[4]) {
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

  // Process the image diagonal-wise, in backwards order
  // (there are H + W - 1 diagonals to process)
  for (int t = 0; t < H + W - 1; ++t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    // Note: All elements in the diagonal could be processed in parallel.
    // However, a OpenMP parallel for was not used here because the internal
    // operations (i.e. copy_matrix and gemm_cpu) are already parallelized.
    for (int e = 0; e < 4 * Tn; ++e) {
      const int i = (e % Tn) + Tmin;
      const int j = t - i;
      const int z = e / Tn;
      const int y  = (z == 0 || z == 1) ? H - i - 1 : i;
      const int x  = (z == 0 || z == 2) ? W - j - 1 : j;
      const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
      const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x
      const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
      const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;  // previous x

      // dJ/dC(y,x)  = dJ/dC(yn,x) dC(yn,x)/dC(y,x) +
      //               dJ/dC(y,xn) dC(y,xn)/dC(y,x) +
      //               ( dJ/dO(y, x) +
      //                 sum_g dJ/dG_g(yn, x) * Ry_g^T +
      //                 sum_g dJ/dG_g(y, xn) * Rx_g^T ) * dO(y,x)/dC(y,x)
      // dJ/dGo(y,x) = ( dJ/dO(y, x) +
      //                 sum_g dJ/dG_g(yn, x) * Ry_g^T +
      //                 sum_g dJ/dG_g(y, xn) * Rx_g^T ) * dO(y,x)/dGo(y,x)

      T* dC_00 = dQ_ptr(z, y, x, 0, 5, 0);
      copym_cpu<T>(N, D, dO_ptr(y, x, 0, z, 0), 4 * D, dC_00, 6 * D);
      if (yn >= 0 && yn < H) {
        for (int g = 0; g < 5; ++g) {
          gemm_cpu<T>('N', 'T', N, D, D,
                      1.0, dQ_ptr(z, yn, x, 0, g, 0), 6 * D,
                      Ry[z] + g * D, 5 * D,
                      1.0, dC_00, 6 * D);
        }
      }
      if (xn >= 0 && xn < W) {
        for (int g = 0; g < 5; ++g) {
          gemm_cpu<T>('N', 'T', N, D, D,
                      1.0, dQ_ptr(z, y, xn, 0, g, 0), 6 * D,
                      Rx[z] + g * D, 5 * D,
                      1.0, dC_00, 6 * D);
        }
      }

      copym_cpu<T>(N, D, dC_00, 6 * D, dQ_ptr(z, y, x, 0, 2, 0), 6 * D);

      #pragma omp parallel for collapse(2)
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          T* dA_00   = dQ_ptr(z, y, x, n, 0, d);
          T* dGi_00  = dQ_ptr(z, y, x, n, 1, d);
          T* dGo_00  = dQ_ptr(z, y, x, n, 2, d);
          T* dGfy_00 = dQ_ptr(z, y, x, n, 3, d);
          T* dGfx_00 = dQ_ptr(z, y, x, n, 4, d);
          T* dC_00   = dQ_ptr(z, y, x, n, 5, d);
          const T dC_10 = (yn >= 0 && yn < H) ? *dQ_ptr(z, yn, x, n, 5, d) : 0;
          const T dC_01 = (xn >= 0 && xn < W) ? *dQ_ptr(z, y, xn, n, 5, d) : 0;
          const T Gfx_01  = (xn >= 0 && xn < W) ? *Q_ptr(z, y, xn, n, 4, d) : 0;
          const T Gfy_10  = (yn >= 0 && yn < H) ? *Q_ptr(z, yn, x, n, 3, d) : 0;
          const T C_10    = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 5, d) : 0;
          const T C_01    = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 5, d) : 0;
          const T C_00    = *Q_ptr(z, y, x, n, 5, d);
          const T Gfx_00  = *Q_ptr(z, y, x, n, 4, d);
          const T Gfy_00  = *Q_ptr(z, y, x, n, 3, d);
          const T Go_00   = *Q_ptr(z, y, x, n, 2, d);
          const T Gi_00   = *Q_ptr(z, y, x, n, 1, d);
          const T A_00    = *Q_ptr(z, y, x, n, 0, d);
          if (y < S[2 * n] && x < S[2 * n + 1]) {
            *dGo_00 =
                (*dGo_00) * FO::f(C_00) * FG::df(Go_00);
            *dC_00  =
                (*dC_00)  * FO::df(C_00) * FG::f(Go_00) +
                dC_10 * FG::f(Gfy_10) +
                dC_01 * FG::f(Gfx_01);
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
    }
  }

  // dJ/dI(y,x)
  for (int z = 0; z < 4; ++z) {
    gemm_cpu<T>('N', 'T', H * W * N, K, 5 * D,
                1.0, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                                      /* dQ reshaped as (H * W * N) x (6 * D),
                                         but only the first 5 * D columns are
                                         used */
                iW[z], 5 * D,             /* iW^T reshaped as (5 * D) x (K) */
                (z == 0 ? 0 : 1), dI, K); /* dI reshaped as (H * W * N) x K */
  }

  // Initialize derivatives to 0
  memset(dP[0], 0x00, sizeof(T) * (5 * D + K * 5 * D + D * 5 * D + D * 5 * D));
  memset(dP[1], 0x00, sizeof(T) * (5 * D + K * 5 * D + D * 5 * D + D * 5 * D));
  memset(dP[2], 0x00, sizeof(T) * (5 * D + K * 5 * D + D * 5 * D + D * 5 * D));
  memset(dP[3], 0x00, sizeof(T) * (5 * D + K * 5 * D + D * 5 * D + D * 5 * D));

  // dJ/db
  T* db[4] = {dP[0], dP[1], dP[2], dP[3]};
  #pragma omp parallel for collapse(2)
  for (int z = 0; z < 4; ++z) {
    for (int d = 0; d < 5 * D; ++d) {
      const T* dQz = dQ + z * H * W * N * 6 * D;
      for (int n = 0; n < H * W * N; ++n) {
        db[z][d] += dQz[n * 6 * D + d];
      }
    }
  }

  // dJ/diW
  T* diW[4] = {dP[0] + 5 * D, dP[1] + 5 * D, dP[2] + 5 * D, dP[3] + 5 * D};
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    gemm_cpu<T>('T', 'N', K, 5 * D, H * W * N,
                1.0, I, K,
                dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                0.0, diW[z], 5 * D);
  }

  // dJ/dRy
  T* dRy[4] = {
    dP[0] + 5 * D + K * 5 * D,
    dP[1] + 5 * D + K * 5 * D,
    dP[2] + 5 * D + K * 5 * D,
    dP[3] + 5 * D + K * 5 * D
  };
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
        if (yp >= 0 && yp < H) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      1.0, O_ptr(yp, x, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRy[z], 5 * D);
        }
      }
    }
  }

  // dJ/dRx
  T* dRx[4] = {
    dP[0] + 5 * D + K * 5 * D + D * 5 * D,
    dP[1] + 5 * D + K * 5 * D + D * 5 * D,
    dP[2] + 5 * D + K * 5 * D + D * 5 * D,
    dP[3] + 5 * D + K * 5 * D + D * 5 * D
  };
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;  // previous x
        if (xp >= 0 && xp < W) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      1.0, O_ptr(y, xp, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRx[z], 5 * D);
        }
      }
    }
  }
}

#endif  // RNN2D_SRC_LSTM_CPU_H_

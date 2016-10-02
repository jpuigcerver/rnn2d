#ifndef RNN2D_LSTM_CPU_H_
#define RNN2D_LSTM_CPU_H_

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>

#include "activation.h"
#include "lstm_common.h"
#include "lstm_cpu_kernels.h"
#include "math_cpu.h"

/* 2D-LSTM forward pass running on the CPU
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
void rnn2d_lstm_fw_cpu(const int H, const int W, const int N, const int K,
                       const int D, const T* I, const int* S, const T* P,
                       T* O, T* Q) {
  // Initialize input to the block and gates with bias
  #pragma omp parallel for
  for (int i = 0; i < 4 * H * W * N * 5 * D; ++i) {
    const int d   = i % D;
    const int g   = (i / D) % 5;
    const int n   = (i / (5 * D)) % N;
    const int x   = (i / (N * 5 * D)) % W;
    const int y   = (i / (W * N * 5 * D)) % H;
    const int k   = i / (H * W * N * 5 * D);
    *Q_ptr(k, y, x, n, g, d) = *B_ptr(k, g, d);
  }

  // Multiply inputs by weights. Each direction can run in parallel
  for (int k = 0; k < 4; ++k) {
    T* Qk = Q + k * H * W * N * 6 * D;
    gemm_cpu<T>('N', 'N', H * W * N, 5 * D, K,
                1.0, I, K,                /* I reshaped as (H * W * N) x K */
                W_ptr(k, 0, 0, 0), 5 * D, /* iW reshaped as K x (5 * D) */
                1.0, Qk, 6 * D);          /* Qk reshaped as
                                             (H * W * N) x (6 * D),
                                             notice that only first 5 * D
                                             columns are used. */
  }

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int t = 0; t < H + W - 1; ++t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    // Propagate recurrent connections
    #pragma omp parallel for
    for (int e = 0; e < 4 * Tn; ++e) {
      const int z = e / Tn; // Diagonal direction
      // (y, x) coordinates of the e-th element in the t-th diagonal.
      const int i = (e % Tn) + Tmin;
      const int j = t - i;
      const int y  = (z == 0 || z == 1) ? i : H - i - 1;
      const int x  = (z == 0 || z == 2) ? j : W - j - 1;
      const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
      const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
      assert(y >= 0 && x >= 0 && y < H && x < W);
      T* Q_00 = Q_ptr(z, y, x, 0, 0, 0);
      if (yp >= 0 && yp <= H - 1) {
        const T* O_10 = O_ptr(yp, x, 0, z, 0);
        /* O_10 reshaped as N x (4 * D). */
        /* Ry reshaped as D x (5 * D) */
        /* Q reshaped as (H * W * N) x (6 * D),
           notice that only first 5 * D columns
           are used. */
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_10, 4 * D,
                    Ry_ptr(z, 0, 0, 0), 5 * D,
                    1.0, Q_00, 6 * D);
      }
      if (xp >= 0 && xp <= W - 1) {
        /* O_01 reshaped as N x (4 * D). */
        /* Rx reshaped as D x (5 * D) */
        /* Q reshaped as (H * W * N) x (6 * D),
           notice that only first 5 * D columns
           are used. */
        const T* O_01 = O_ptr(y, xp, 0, z, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_01, 4 * D,
                    Rx_ptr(z, 0, 0, 0), 5 * D,
                    1.0, Q_00, 6 * D);
      }
    }

    // Compute cell and output values
    #pragma omp parallel for
    for (int e = 0; e < 4 * Tn * N * D; ++e) {
      const int d = e % D;
      const int n = (e / D) % N;
      const int z = (e / (Tn * N * D));
      // (y, x) coordinates of the e-th element in the t-th diagonal.
      const int i = ((e / (N * D)) % Tn) + Tmin;
      const int j = t - i;
      const int y  = (z == 0 || z == 1) ? i : H - i - 1;
      const int x  = (z == 0 || z == 2) ? j : W - j - 1;
      const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;
      const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;
      T* C_00 = Q_ptr(z, y, x, n, 5, d);
      T* O_00 = O_ptr(y, x, n, z, d);
      if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
        const T f_a   = FI::f(*Q_ptr(z, y, x, n, 0, d));  // f(input)
        const T f_gi  = FG::f(*Q_ptr(z, y, x, n, 1, d));  // input gate
        const T f_go  = FG::f(*Q_ptr(z, y, x, n, 2, d));  // output gate
        const T f_gfy = FG::f(*Q_ptr(z, y, x, n, 3, d));  // forget_y gate
        const T f_gfx = FG::f(*Q_ptr(z, y, x, n, 4, d));  // forget_x gate
        const T C_10  = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 5, d) : 0;
        const T C_01  = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 5, d) : 0;
        *C_00 = f_gi * f_a + f_gfy * C_10 + f_gfx * C_01;
        *O_00 = f_go * FO::f(*C_00);
      } else {
        *C_00 = 0;
        *O_00 = 0;
      }
    }
  }
  /*
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        printf("Q(%d,%d,%d,.,.,.) =\n", z, y, x);
        for (int n = 0; n < N; ++n) {
          for (int g = 0; g < 6; ++g)
            for (int d = 0; d < D; ++d) {
              printf(" % .4f", *Q_ptr(z, y, x, n, g, d));
            }
          printf("\n");
        }
        printf("\n");
      }

  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      printf("O(%d,%d,.,.,.) =\n", y, x);
      for (int n = 0; n < N; ++n) {
        for (int z = 0; z < 4; ++z)
          for (int d = 0; d < D; ++d) {
            printf(" % .4f", *O_ptr(y, x, n, z, d));
          }
        printf("\n");
      }
      printf("\n");
    } */
}


/* 2D-LSTM backward pass running on the CPU
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
void rnn2d_lstm_bw_cpu(const int H, const int W, const int N, const int K,
                       const int D, const T* I, const int* S, const T* P,
                       const T* O, const T* Q, const T* dO, T* dQ) {
  printf("bw: sum_I = %.10f\n",
         std::accumulate(I, I + H * W * N * K, 0.0));
  printf("bw: sum_P = %.10f\n",
         std::accumulate(P, P + 4 * (1 + K + D + D) * 5 * D, 0.0));
  printf("bw: sum_O = %.10f\n",
         std::accumulate(O, O + H * W * N * 4 * D, 0.0));
  printf("bw: sum_Q = %.10f\n",
         std::accumulate(Q, Q + 4 * H * W * N * 6 * D, 0.0));
  printf("bw: sum_dO = %.10f\n",
         std::accumulate(dO, dO + H * W * N * 4 * D, 0.0));
  printf("bw: sum_dQ' = %.10f\n",
         std::accumulate(dQ, dQ + 4 * H * W * N * 6 * D, 0.0));

  // Process the image diagonal-wise, in backwards order
  // (there are H + W - 1 diagonals to process)
  for (int t = H + W - 2; t >= 0; --t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    copy_dO_to_dC<T>(H, W, N, D, t, Tn, Tmin, dO, dQ);

    // Note: All elements in the diagonal could be processed in parallel.
    // However, a OpenMP parallel for was not used here because the internal
    // operations (i.e. copy_matrix and gemm_cpu) are already parallelized.
    for (int e = 0; e < 4 * Tn; ++e) {
      const int i = (e % Tn) + Tmin;
      const int j = t - i;
      const int z = e / Tn;
      const int y  = (z == 0 || z == 1) ? i : H - i - 1;
      const int x  = (z == 0 || z == 2) ? j : W - j - 1;
      const int yn = (z == 0 || z == 1) ? y + 1 : y - 1;  // next y
      const int xn = (z == 0 || z == 2) ? x + 1 : x - 1;  // next x

      // dJ/dC(y,x)  = dJ/dC(yn,x) dC(yn,x)/dC(y,x) +
      //               dJ/dC(y,xn) dC(y,xn)/dC(y,x) +
      //               ( dJ/dO(y, x) +
      //                 sum_g dJ/dG_g(yn, x) * Ry_g^T +
      //                 sum_g dJ/dG_g(y, xn) * Rx_g^T ) * dO(y,x)/dC(y,x)
      // dJ/dGo(y,x) = ( dJ/dO(y, x) +
      //                 sum_g dJ/dG_g(yn, x) * Ry_g^T +
      //                 sum_g dJ/dG_g(y, xn) * Rx_g^T ) * dO(y,x)/dGo(y,x)

      if (yn >= 0 && yn < H) {
        gemm_cpu<T>('N', 'T', N, D, 5 * D,
                    1.0, dQ_ptr(z, yn, x, 0, 0, 0), 6 * D,
                    Ry_ptr(z, 0, 0, 0), 5 * D,
                    1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D);
      }
      if (xn >= 0 && xn < W) {
        gemm_cpu<T>('N', 'T', N, D, 5 * D,
                    1.0, dQ_ptr(z, y, xn, 0, 0, 0), 6 * D,
                    Rx_ptr(z, 0, 0, 0), 5 * D,
                    1.0, dQ_ptr(z, y, x, 0, 5, 0), 6 * D);
      }
    }
    bw_elemwise_ops<T, FG, FI, FO>(H, W, N, D, t, Tn, Tmin, S, Q, dQ);
  }

  printf("bw: sum_dQ = %.10f\n",
         std::accumulate(dQ, dQ + 4 * H * W * N * 6 * D, 0.0));
}


template <typename T>
void rnn2d_lstm_bw_input_cpu(const int H, const int W, const int N, const int K,
                             const int D, const T* P, const T* dQ,
                             const T scale, T* dI) {
  // dJ/dI(y,x)
  for (int z = 0; z < 4; ++z) {
    /* dQ reshaped as (H * W * N) x (6 * D),
       but only the first 5 * D columns are
       used */
    /* iW^T reshaped as (5 * D) x (K) */
    /* dI reshaped as (H * W * N) x K */
    gemm_cpu<T>('N', 'T', H * W * N, K, 5 * D,
                scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                W_ptr(z, 0, 0, 0), 5 * D,
                1.0, dI, K);
  }
}

template <typename T>
void rnn2d_lstm_bw_params_cpu(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T* dQ, const T scale, T* dP) {
  // dJ/db
  const std::vector<T> vOnes(H * W * N, 1);
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    gemv_cpu<T>('T', H * W * N, 5 * D,
                scale, dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                vOnes.data(), 1,
                1.0, dB_ptr(z, 0, 0), 1);
  }

  // dJ/dW
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    gemm_cpu<T>('T', 'N', K, 5 * D, H * W * N,
                scale, I, K,
                dQ_ptr(z, 0, 0, 0, 0, 0), 6 * D,
                1.0, dW_ptr(z, 0, 0, 0), 5 * D);
  }

  // dJ/dRy
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int yp = (z == 0 || z == 1) ? y - 1 : y + 1;  // previous y
        if (yp >= 0 && yp < H) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      scale, O_ptr(yp, x, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRy_ptr(z, 0, 0, 0), 5 * D);
        }
      }
    }
  }

  // dJ/dRx
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int xp = (z == 0 || z == 2) ? x - 1 : x + 1;  // previous x
        if (xp >= 0 && xp < W) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      scale, O_ptr(y, xp, 0, z, 0), 4 * D,
                      dQ_ptr(z, y, x, 0, 0, 0), 6 * D,
                      1.0, dRx_ptr(z, 0, 0, 0), 5 * D);
        }
      }
    }
  }
}

#endif  // RNN2D_LSTM_CPU_H_

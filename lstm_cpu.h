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
                    [D x 5 x D] (rWy) + [D x 5 x D] (rWx))
   O -> output data (layout: H x W x N x 4 x D)
   Q -> gates pre-activations and cells (layout: 4 x H x W x N x 6 x D)
*/
template < typename T, typename FG, typename FI, typename FO >
void lstm_2d_fwd_cpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P[4],
                     T* O, T* Q[4]) {
  // Bias, in each direction
  const T* b[4] = {P[0], P[1], P[2], P[3]};
  // Input weights, in each direction
  const T* iW[4] = {P[0] + 5 * D, P[1] + 5 * D, P[2] + 5 * D, P[3] + 5 * D};
  // Recurrent weights for the y-dimension, in each direction
  const T* rWy[4] = {
    P[0] + 5 * D + K * 5 * D,
    P[1] + 5 * D + K * 5 * D,
    P[2] + 5 * D + K * 5 * D,
    P[3] + 5 * D + K * 5 * D
  };
  // Recurrent weights for the x-dimension, in each direction
  const T* rWx[4] = {
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
    const int xyn = (i / (5 * D)) % (H * W * N);
    const int k   = i / (H * W * N * 5 * D);
    Q[k][xyn * 6 * D + g * D + d] = b[k][g * D + d];
  }

  // Multiply inputs by weights. Each direction can run in parallel
  for (int k = 0; k < 4; ++k) {
    gemm_cpu<T>('N', 'N', H * W * N, 5 * D, K,
                1.0, I, K,          /* I reshaped as (H * W * N) x K */
                iW[k], 5 * D,       /* iW reshaped as K x (5 * D) */
                1.0, Q[k], 6 * D);  /* Q reshaped as (H * W * N) x (6 * D),
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
                    rWy[k], 5 * D,      /* rWy reshaped as D x (5 * D) */
                    1.0, Q_00, 6 * D);  /* Q reshaped as (H * W * N) x (6 * D),
                                           notice that only first 5 * D columns
                                           are used. */
      }
      if (xp >= 0 && xp <= W - 1) {
        const T* O_01 = O_ptr(y, xp, 0, k, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_01, 4 * D,   /* O_01 reshaped as N x (4 * D). */
                    rWx[k], 5 * D,      /* rWx reshaped as D x (5 * D) */
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
      const int k = e / (Zn * N * D);
      // (y, x) coordinates of the e-th element in the z-th diagonal.
      const int i = ((e / (N * D)) % Zn) + Zmin;
      const int j = z - i;
      const int y  = (k == 0 || k == 1) ? i : H - i - 1;
      const int x  = (k == 0 || k == 2) ? j : W - j - 1;
      const int yp = (k == 0 || k == 1) ? y - 1 : y + 1;
      const int xp = (k == 0 || k == 2) ? x - 1 : x + 1;
      assert(y >= 0 && x >= 0 && y < H && x < W);
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

/*
// 2D-LSTM backward pass running on the CPU
// H -> maximum height
// W -> maximum width
// N -> batch size
// K -> input dimensions/channels
// D -> output dimensions/channels
// I -> input data (layout: H x W x N x K)
// S -> input sizes (height and width of each sample, layout: N x 2)
// P -> params (layout: [20 * D] (b) + [K x 20 * D] (iW) + [4 * D x 10*D] (rW))
// O -> output data (layout: H x W x N x 4 x D)
// Q -> gates and cells activations (layout: H x W x N x 4 x 6 x D)
template <typename T, typename FI, typename FO>
void lstm_2d_bkw_cpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P,
                     const T* O, const T* Q, T*, const T* dO, T* dQ,
                     T* dI, T* dP) {


  // Process the image diagonal-wise, in backwards order
  // (there are H + W - 1 diagonals to process)
  for (int z = H + W - 2; z >= 0; --z) {
    // Number of elements in the z-th diagonal
    const int Zn = std::min(std::min(H + W - 1 - z, z + 1), std::min(H, W));
    // (y, x) coordinates of the first element in the z-th diagonals
    const int zyx[4][2] = DIAGONAL_INITIAL_COORDS_ARRAY(z);

    // Compute derivatives w.r.t. the cell
    // dJ/dC(y,x) = dJ/dO(y,x) * g_o(y,x) * f_o'(C(y,x)) +
    //              dJ/dC(y+1,x) * g_fy(y+1,x) +
    //              dJ/dC(y,x+1) * g_fx(y,x+1)
    #pragma omp parallel for
    for (int i = 0; i < Zn * N * 4 * D; ++i) {
      const int k = i % D;
      const int d = (i / D) % 4;
      const int n = (i / (4 * D)) % N;
      const int j = (i / (N * 4 * D)) % Zn;
      const int y = zyx[d][0] + ZDIR[d][0] * j, x = zyx[d][1] + ZDIR[d][1] * j;
      assert(!(y < 0 || x < 0 || y >= H || x >= W));
      T* dC_yx = dQ_ptr(y, x, n, d, 5, k);
      if (y < S[n * 2] && x < S[n * 2 + 1]) {
        const T C_yx  =  *Q_ptr(y, x, n, d, 5, k);
        const T dO_yx = *dO_ptr(y, x, n, d, k);
        const T og  = Sigmoid<T>::f(*Q_ptr(y, x, n, d, 2, k));
        *dC_yx  = dO_yx * og * FO::df(C_yx);
        if (y + BNYX[d][0] >= 0 && y + BNYX[d][0] < H - 1) {
          const T fgy_10 = Sigmoid<T>::f(*Q_ptr(y + BNYX[d][0], x, n, d, 3, k));
          const T dC_10  = *dQ_ptr(y + BNYX[d][0], x, n, d, 5, k);
          *dC_yx += dC_10 * fgy_10;
        }
        if (x + BNYX[d][1] >= 0 && x + BNYX[d][1] < W - 1) {
          const T fgx_01 = Sigmoid<T>::f(*Q_ptr(y, x + BNYX[d][1], n, d, 4, k));
          const T dC_01  = *dQ_ptr(y, x + BNYX[d][1], n, d, 5, k);
          *dC_yx += dC_01 * fgx_01;
        }
      } else {
        *dC_yx = 0;
      }
    }
  }

  // Compute derivatives w.r.t. the block input and the gates
  // dJ/dA(y,x) = dJ/dC(y,x) * g_i(y,x)
  // dJ/dGi(y,x) = dJ/dC(y,x) * f_i(a(y,x))
  // dJ/dGo(y,x) = dJ/dO(y,x) * f_o(c(y,x))
  #pragma omp parallel for
  for (int i = 0; i < H * W * N * 4 * D; ++i) {
    const int k = i % D;
    const int d = (i / D) % 4;
    const int n = (i / (4 * D)) % N;
    const int x = (i / (N * 4 * D)) % W;
    const int y = (i / (W * N * 4 * D));
    assert(!(y < 0 || x < 0 || y >= H || x >= W));
    const T A_yx   = *Q_ptr(y, x, n, d, 0, k);  // a(y,x)
    const T Gi_yx  = *Q_ptr(y, x, n, d, 1, k);  // g_i(x,y)
    const T Go_yx  = *Q_ptr(y, x, n, d, 2, k);  // g_o(x,y)
    const T Gfy_yx = *Q_ptr(y, x, n, d, 3, k);  // g_fy(x,y)
    const T Gfx_yx = *Q_ptr(y, x, n, d, 4, k);  // g_fx(x,y)
    const T C_yx   = *Q_ptr(y, x, n, d, 5, k);  // c(y,x)
    const T C_10 =                              // c(y-1,x)
        (y + FNYX[d][0] >= 0 && y + FNYX[d][0] < H - 1) ?
        *Q_ptr(y + FNYX[d][0], x, n, d, 5, k) : 0;
    const T C_01 =                              // c(y,x-1)
        (x + FNYX[d][1] >= 0 && x + FNYX[d][1] < W - 1) ?
        *Q_ptr(y, x + FNYX[d][1], n, d, 5, k) : 0;
    const T dO_yx = *dO_ptr(y, x, n, d, k);     // dJ/dO(y,x)
    const T dC_yx = *dQ_ptr(y, x, n, d, 5, k);  // dJ/dC(y,x)
    *dQ_ptr(y, x, n, d, 0, k) = dC_yx * Sigmoid<T>::f(Gi_yx) * FI::df(A_yx);
    *dQ_ptr(y, x, n, d, 1, k) = dC_yx * FI::f(A_yx) * Sigmoid<T>::df(Gi_yx);
    *dQ_ptr(y, x, n, d, 2, k) = dO_yx * FO::f(C_yx) * Sigmoid<T>::df(Go_yx);
    *dQ_ptr(y, x, n, d, 3, k) = dC_yx * C_10 * Sigmoid<T>::df(Gfy_yx);
    *dQ_ptr(y, x, n, d, 4, k) = dC_yx * C_01 * Sigmoid<T>::df(Gfx_yx);
  }

  // Input weights
  const T* iW[4] = {
    P + 20 * D,
    P + 20 * D +  5 * K * D,
    P + 20 * D + 10 * K * D,
    P + 20 * D + 15 * K * D
  };
  // Compute derivatives w.r.t the layer input (I), across all dimensions
  // dJ/dI(y,x) = dJ/dA(y,x) * W_a +
  //              dJ/dGi(y,x) * W_i +
  //              dJ/dGo(y,x) * W_o +
  //              dJ/dGfy(y,x) * W_fy +
  //              dJ/dGfx(y,x) * W_fx
  for (int d = 0; d < 4; ++d) {
    const T* dQ_d = dQ_ptr(0, 0, 0, d, 0, 0);
    gemm_cpu<T>('N', 'T', H * W * N, K, 5 * D,
                1.0, dQ_d, 4 * 6 * D,
                iW[d], H * W * N,
                (d == 0 ? 0 : 1), dI, K);
  }
}

*/
#endif  // RNN2D_SRC_LSTM_CPU_H_

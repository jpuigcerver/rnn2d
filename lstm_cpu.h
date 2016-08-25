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
void lstm_2d_fw_cpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P[4],
                     T* O, T* Q) {
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


/* 2D-LSTM forward pass running on the CPU
   H  -> maximum height
   W  -> maximum width
   N  -> batch size
   K  -> input dimensions/channels
   D  -> output dimensions/channels
   I  -> input data (layout: H x W x N x K)
   S  -> input sizes (height and width of each sample, layout: N x 2)
   P  -> parameters 4 x (layout: [5 x D] (b) + [K x 5 x D] (iW) +
                                 [D x 5 x D] (rWy) + [D x 5 x D] (rWx))
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

  // Process the image diagonal-wise, in backwards order
  // (there are H + W - 1 diagonals to process)
  for (int z = H + W - 2; z >= 0; --z) {
    // Compute number of elements in the diagonal
    const int Zmin = std::max(0, z - W + 1);
    const int Zmax = std::min(z, H - 1);
    const int Zn   = (Zmax - Zmin) + 1;

    // Compute derivatives w.r.t. the cell
    // dJ/dC(y,x) = dJ/dO(y,x) * FO(g_o(y,x)) * FO'(C(y,x)) +
    //              dJ/dC(y+1,x) * FG(g_fy(y+1,x)) +
    //              dJ/dC(y,x+1) * FG(g_fx(y,x+1))
    #pragma omp parallel for
    for (int e = 0; e < Zn * N * 4 * D; ++e) {
      const int d = e % D;
      const int n = (e / D) % N;
      const int k = e / (Zn * N * D);
      // (y, x) coordinates of the e-th element in the z-th diagonal.
      const int i = ((e / (N * D)) % Zn) + Zmin;
      const int j = z - i;
      const int y  = (k == 0 || k == 1) ? i : H - i - 1;
      const int x  = (k == 0 || k == 2) ? j : W - j - 1;
      const int yn = (k == 0 || k == 1) ? y + 1 : y - 1;  // next y
      const int xn = (k == 0 || k == 2) ? x + 1 : x - 1;  // next x
      T* dC_00 = dQ_ptr(k, y, x, n, 5, d);
      if (y < S[n * 2] && x < S[n * 2 + 1]) {
        // f(output gate(y,x))
        const T f_go  = FG::f(*Q_ptr(k, y, x, n, 2, d));
        // f(forget gate_y(y+1,x)
        const T f_gfy_10 = (yn >= 0 && yn < H) ?
            FG::f(*Q_ptr(k, yn, x, n, 3, d)) : 0;
        // f(forget gate_x(y,x+1)
        const T f_gfx_01 = (xn >= 0 && xn < W) ?
            FG::f(*Q_ptr(k, y, xn, n, 4, d)) : 0;
        // C(y,x)
        const T C_00  = *Q_ptr(k, y, x, n, 5, d);
        // dJ/dO(y,x)
        const T dO_00 = *dO_ptr(y, x, n, k, d);
        // dJ/dC(y+1,x)
        const T dC_10 = (yn >= 0 && yn < H) ? *dQ_ptr(k, yn, x, n, 5, d) : 0;
        // dJ/dC(y,x+1)
        const T dC_01 = (xn >= 0 && xn < W) ? *dQ_ptr(k, y, xn, n, 5, d) : 0;
        *dC_00 =
            dO_00 * f_go * FO::df(C_00) +
            dC_10 * f_gfy_10 +
            dC_01 * f_gfx_01;
      } else {
        *dC_00 = 0;
      }
    }
  }

  // Compute derivatives w.r.t. the block input and the gates
  // dJ/dA(y,x)   = dJ/dC(y,x) * FG(g_i(y,x)) * FI'(a(y,x))
  // dJ/dGi(y,x)  = dJ/dC(y,x) * FI(a(y,x))   * FG'(g_i(y,x))
  // dJ/dGo(y,x)  = dJ/dO(y,x) * FO(c(y,x))   * FG'(g_o(y,x))
  // dJ/dGfy(y,x) = dJ/dC(y,x) * C(y-1,x)     * FG'(g_fy(y,x))
  // dJ/dGfx(y,x) = dJ/dC(y,x) * C(y,x-1)     * FG'(g_fx(y,x))
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < 4 * H * W * N * D; ++i) {
    const int d = i % D;
    const int n = (i / D) % N;
    const int x = (i / (N * D)) % W;
    const int y = (i / (W * N * D)) % H;
    const int k = i / (H * W * N * D);
    const int yp = (k == 0 || k == 1) ? y - 1 : y + 1;  // previous y
    const int xp = (k == 0 || k == 2) ? x - 1 : x + 1;  // previous x
    const T dC_00 = *dQ_ptr(k, y, x, n, 5, d);  // dJ/dC(y,x)
    const T dO_00 = *dO_ptr(y, x, n, k, d);     // dJ/dO(y,x)
    const T a    = *Q_ptr(k, y, x, n, 0, d);    // a(y,x)
    const T g_i  = *Q_ptr(k, y, x, n, 1, d);    // g_i(y,x)
    const T g_o  = *Q_ptr(k, y, x, n, 2, d);    // g_o(y,x)
    const T g_fy = *Q_ptr(k, y, x, n, 3, d);    // g_fy(y,x)
    const T g_fx = *Q_ptr(k, y, x, n, 4, d);    // g_fx(y,x)
    const T C_00 = *Q_ptr(k, y, x, n, 5, d);    // C(y,x)
    const T C_10 = *Q_ptr(k, yp, x, n, 4, d);   // C(y-1,x)
    const T C_01 = *Q_ptr(k, y, xp, n, 4, d);   // C(y,x-1)
    *dQ_ptr(k, y, x, n, 0, d) = dC_00 * FG::f(g_i)  * FI::df(a);
    *dQ_ptr(k, y, x, n, 1, d) = dC_00 * FI::f(a)    * FG::df(g_i);
    *dQ_ptr(k, y, x, n, 2, d) = dO_00 * FI::f(C_00) * FG::df(g_o);
    *dQ_ptr(k, y, x, n, 3, d) = dC_00 * C_10        * FG::df(g_fy);
    *dQ_ptr(k, y, x, n, 4, d) = dC_00 * C_01        * FG::df(g_fx);
  }

  // Compute derivatives w.r.t the layer input (I), across all directions
  // For each direction:
  // dJ/dI(y,x) = dJ/dA(y,x)   * W_a +
  //              dJ/dGi(y,x)  * W_i +
  //              dJ/dGo(y,x)  * W_o +
  //              dJ/dGfy(y,x) * W_fy +
  //              dJ/dGfx(y,x) * W_fx
  // The total derivative of the loss w.r.t. the input (I) is the sum across
  // all directions.
  for (int k = 0; k < 4; ++k) {
    const T* dQk = dQ + k * H * W * N * 6 * D;
    gemm_cpu<T>('N', 'T', H * W * N, K, 5 * D,
                1.0, dQk, 6 * D,      /* dQ reshaped as (H * W * N) x (6 * D),
                                         but only the first 5 * D columns are
                                         used */
                iW[k], K,                 /* iW^T reshaped as (5 * D) x (K) */
                (k == 0 ? 0 : 1), dI, K); /* dI reshaped as (H * W * N) x K */
  }
}

#endif  // RNN2D_SRC_LSTM_CPU_H_

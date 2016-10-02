#ifndef RNN2D_LSTM_CPU_KERNELS_H_

#include "lstm_common.h"

template <typename T>
void copy_dO_to_dC(const int H, const int W, const int N, const int D,
                   const int t, const int Tn, const int Tmin,
                   const T* dO, T* dQ) {
  #pragma omp parallel for collapse(4)
  for (int z = 0; z < 4; ++z) {
    for (int e = 0; e < Tn; ++e) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          const int i = e + Tmin;
          const int j = t - i;
          const int y = (z == 0 || z == 1) ? i : H - i - 1;
          const int x = (z == 0 || z == 2) ? j : W - j - 1;
          *dQ_ptr(z, y, x, n, 5, d) = *dO_ptr(y, x, n, z, d);
        }
      }
    }
  }
}

template <typename T, typename FG, typename FI, typename FO>
void bw_elemwise_ops(const int H, const int W, const int N, const int D,
      const int t, const int Tn, const int Tmin,
      const int* S, const T* Q, T* dQ) {
  #pragma omp parallel for collapse(4)
  for (int z = 0; z < 4; ++z) {
    for (int e = 0; e < Tn; ++e) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
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
          if (S == nullptr || (y < S[2 * n] && x < S[2 * n + 1])) {
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
    }
  }
}

#endif  // LSTM_CPU_KERNELS_H_

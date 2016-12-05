#include "tile_cpu.h"

#include <glog/logging.h>

#include "tile_xxx.h"

template <typename T>
inline void fw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* I, T* O) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  const int o_H = DIV_UP(H, Kh);    // height of the output image
  const int o_W = DIV_UP(W, Kw);    // width of the output image
  const int o_D = D * Kh * Kw;      // depth of the output image

  #pragma omp parallel for collapse(4)
  for (int y = 0; y < o_H; ++y) {
    for (int x = 0; x < o_W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < o_D; ++d) {
          const int dd = d % D;                // channel on the input image
          const int j  = (d / D) % Kw;
          const int i  = (d / (D * Kw)) % Kh;
          const int xx = Kw * x + j;           // x-coordinate on the input
          const int yy = Kh * y + i;           // y-coordinate on the input
          if ((S != nullptr && yy < S[2 * n] && xx < S[2 * n + 1]) ||
              (yy < H && xx < W)) {
            *O_ptr(O, y, x, n, d) = *I_ptr(I, yy, xx, n, dd);
          } else {
            *O_ptr(O, y, x, n, d) = 0;
          }
        }
      }
    }
  }
}

template <typename T>
inline void bw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* dO, T* dI) {
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(dI);
  const int o_H = DIV_UP(H, Kh);    // height of the output image
  const int o_W = DIV_UP(W, Kw);    // width of the output image
  const int o_D = D * Kh * Kw;      // depth of the output image

  #pragma omp parallel for collapse(4)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          if (S == nullptr || (y < S[2 * n] && x < S[2 * n + 1])) {
            const int yy = y / Kh;                      // output y-coordinate
            const int xx = x / Kw;                      // output x-coordinate
            const int dd = d + D * (x % Kw) + D * Kw * (y % Kh);  // output ch
            *I_ptr(dI, y, x, n, d) = *O_ptr(dO, yy, xx, n, dd);
          } else {
            *I_ptr(dI, y, x, n, d) = 0;
          }
        }
      }
    }
  }
}

extern "C" {
  DEFINE_WRAPPERS(cpu, float)
  DEFINE_WRAPPERS(cpu, double)
}  // extern "C"

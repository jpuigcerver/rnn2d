#ifndef RNN2D_INTERNAL_CPU_LSTM2D_CELL_H_
#define RNN2D_INTERNAL_CPU_LSTM2D_CELL_H_

#include <rnn2d/basic_types.h>
#include <rnn2d/activations.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template <
    typename T,
    class FG = internal::Sigmoid<T>,
    class FI = internal::Tanh<T>,
    class FO = internal::Tanh<T>>
class Lstm2dCell {
 public:
  Lstm2dCell() {}

  Lstm2dCell(const FG& fg, const FI& fi, const FO& fo) :
      fg_(fg), fi_(fi), fo_(fo) {}

  static int NumGates() { return 5; }

  template <class Lstm2dImpl>
  rnn2dStatus_t Forward(Lstm2dImpl* lstm, const int t) {
    const int H = lstm->GetH();
    const int W = lstm->GetW();
    const int N = lstm->GetN();
    const int D = lstm->GetD();
    const int L = std::min(H, W);

    // Compute cell and output values
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int y = lstm->GetY(z, t, e), x = lstm->GetX(z, t, e);
            if (y >= 0 && x >= 0 && y < H && x < W) {
              if (y < lstm->GetH(n) && x < lstm->GetW(n)) {
                const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
                const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
                const T fGi = fg_.f(*lstm->Q(z, y, x, n, 0, d));  // input gate
                const T fGy = fg_.f(*lstm->Q(z, y, x, n, 1, d));  // fgt_y gate
                const T fGx = fg_.f(*lstm->Q(z, y, x, n, 2, d));  // fgt_x gate
                const T fGo = fg_.f(*lstm->Q(z, y, x, n, 3, d));  // output gate
                const T fA  = fi_.f(*lstm->Q(z, y, x, n, 4, d));  // pre-cell
                const T C_10 = (yp >= 0 && yp < lstm->GetH())
                               ? *lstm->Q(z, yp, x, n, 4, d) : 0;
                const T C_01 = (xp >= 0 && xp < lstm->GetW())
                               ? *lstm->Q(z, y, xp, n, 4, d) : 0;
                const T C_00 = fGi * fA + fGy * C_10 + fGx * C_01;  // state
                const T O_00 = fGo * fo_.f(C_00);                   // output
                *lstm->Q(z, y, x, n, 0, d) = fGi;
                *lstm->Q(z, y, x, n, 1, d) = fGy;
                *lstm->Q(z, y, x, n, 2, d) = fGx;
                *lstm->Q(z, y, x, n, 3, d) = fGo;
                *lstm->Q(z, y, x, n, 4, d) = C_00;
                *lstm->O(z, y, x, n, d) = O_00;
              } else {
                *lstm->Q(z, y, x, n, 0, d) = 0;
                *lstm->Q(z, y, x, n, 1, d) = 0;
                *lstm->Q(z, y, x, n, 2, d) = 0;
                *lstm->Q(z, y, x, n, 3, d) = 0;
                *lstm->Q(z, y, x, n, 4, d) = 0;
                *lstm->O(z, y, x, n, d) = 0;
              }
            }
          }
        }
      }
    }

    return RNN2D_STATUS_SUCCESS;
  }

  template <class Lstm2dImpl>
  rnn2dStatus_t Backward(Lstm2dImpl* lstm, const int t) {
    const int H = lstm->GetH();
    const int W = lstm->GetW();
    const int N = lstm->GetN();
    const int D = lstm->GetD();
    const int L = std::min(H, W);

    // Compute cell and output values
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int y = lstm->GetY(z, t, e), x = lstm->GetX(z, t, e);
            if (y >= 0 && x >= 0 && y < H && x < W) {
              if (y < lstm->GetH(n) && x < lstm->GetW(n)) {
                const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
                const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
                const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // prev y
                const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;  // prev x
                const T fGi = lstm->Q(z, y, x, n, 0, d);
                const T fGy = lstm->Q(z, y, x, n, 1, d);
                const T fGx = lstm->Q(z, y, x, n, 2, d);
                const T fGo = lstm->Q(z, y, x, n, 3, d);
                const T C_00 = lstm->Q(z, y, x, n, 4, d);
                const T dO_00 = lstm->Z(0, z, y, x, n, d);
                const T C_10 = (yp >= 0 && yp < lstm->GetH())
                               ? lstm->Q(z, yp, x, n, 4, d) : 0;
                const T C_01 = (xp >= 0 && xp < lstm->GetW())
                               ? lstm->Q(z, y, xp, n, 4, d) : 0;
                const T fA = (fGi != 0)
                             ? ((C_00 - fGy * C_10 - fGx * C_01) / fGi) : 0;
                // Z_10 = dC(y+1, x) * f(Gy(y+1, x))
                const T Z_10 = (yn >= 0 && yn < lstm->GetH())
                               ? lstm->Z(1, z, yn, x, n, d) : 0;
                // Z_01 = dC(y, x+1) * f(Gx(y, x+1))
                const T Z_01 = (xn >= 0 && xn < lstm->GetW())
                               ? lstm->Z(2, z, y, xn, n, d) : 0;
                const T dC_00 = dO_00 * fo_.df(C_00) * fGo + Z_10 + Z_01;
                lstm->Q(z, y, x, n, 0, d) = dC_00 * fA * fg_.df2(fGi);
                lstm->Q(z, y, x, n, 1, d) = (yp >= 0 && yp < lstm->GetH())
                                            ? (dC_00 * fg_.df2(fGx) * C_10)
                                            : 0;
                lstm->Q(z, y, x, n, 2, d) = (xp >= 0 && xp < lstm->GetW())
                                            ? (dC_00 * fg_.df2(fGx) * C_01)
                                            : 0;
                lstm->Q(z, y, x, n, 3, d) = dO_00 * fo_.f(C_00) * fg_.df2(fGo);
                lstm->Q(z, y, x, n, 4, d) = dC_00 * fi_.df2(fA) * fGi;
                lstm->Z(1, z, y, x, n, d) = fGy;
                lstm->Z(2, z, y, x, n, d) = fGx;
              } else {
                lstm->Q(z, y, x, n, 0, d) = 0;
                lstm->Q(z, y, x, n, 1, d) = 0;
                lstm->Q(z, y, x, n, 2, d) = 0;
                lstm->Q(z, y, x, n, 3, d) = 0;
                lstm->Q(z, y, x, n, 4, d) = 0;
                lstm->Z(1, z, y, x, n, d) = 0;
                lstm->Z(2, z, y, x, n, d) = 0;
              }
            }
          }
        }
      }
    }

    return RNN2D_STATUS_SUCCESS;
  }

 private:
  FG fg_;
  FI fi_;
  FO fo_;
};

}
}
}

#endif  // RNN2D_INTERNAL_CPU_LSTM2D_CELL_H_

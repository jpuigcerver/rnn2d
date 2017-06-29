#ifndef RNN2D_INTERNAL_CPU_LSTM_STANDARD_CELL_IMPL_H_
#define RNN2D_INTERNAL_CPU_LSTM_STANDARD_CELL_IMPL_H_

#include <rnn2d/internal/activation.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template <
    typename T,
    class FG = internal::Sigmoid<T>,
    class FI = internal::Tanh<T>,
    class FO = internal::Tanh<T>>
class LstmAachenCellInference {
 public:
  template <class LstmImpl>
  void Forward(LstmImpl* lstm, const int t, const int tn, const int tmin) {
    const int* shape = lstm->GetShape();
    // Compute cell and output values
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < tn; ++e) {
        for (int n = 0; n < lstm->GetN(); ++n) {
          for (int d = 0; d < lstm->GetD(); ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int i = e + tmin;
            const int j = t - i;
            const int y  = (z == 0 || z == 2) ? i : lstm->GetH() - i - 1;
            const int x  = (z == 0 || z == 1) ? j : lstm->GetW() - j - 1;
            if (shape == nullptr ||
                (y < shape[n * 2] && x < shape[n * 2 + 1])) {
              const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
              const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
              const T fGi = FG::f(lstm->Q(z, y, x, n, 0, d));  // input gate
              const T fGy = FG::f(lstm->Q(z, y, x, n, 1, d));  // fgt_y gate
              const T fGx = FG::f(lstm->Q(z, y, x, n, 2, d));  // fgt_x gate
              const T fGo = FG::f(lstm->Q(z, y, x, n, 3, d));  // output gate
              const T fA  = FI::f(lstm->Q(z, y, x, n, 4, d));  // pre-cell
              const T C_10 = (yp >= 0 && yp < lstm->GetH())
                             ? lstm->Q(z, yp, x, n, 4, d) : 0;
              const T C_01 = (xp >= 0 && xp < lstm->GetW())
                             ? lstm->Q(z, y, xp, n, 4, d) : 0;
              // Cell state
              const T C_00 = fGi * fA + 0.5 * fGy * C_10 + 0.5 * fGx * C_01;
              // Cell output
              const T O_00 = fGo * FO::f(C_00);
              lstm->Q(z, y, x, n, 0, d) = fGi;
              lstm->Q(z, y, x, n, 1, d) = fGy;
              lstm->Q(z, y, x, n, 2, d) = fGx;
              lstm->Q(z, y, x, n, 3, d) = fGo;
              lstm->Q(z, y, x, n, 4, d) = C_00;
              lstm->O(O, y, x, n, z, d) = O_00;
            } else {
              lstm->Q(z, y, x, n, 0, d) = 0;
              lstm->Q(z, y, x, n, 1, d) = 0;
              lstm->Q(z, y, x, n, 2, d) = 0;
              lstm->Q(z, y, x, n, 3, d) = 0;
              lstm->Q(z, y, x, n, 4, d) = 0;
              lstm->O(O, y, x, n, z, d) = 0;
            }
          }
        }
      }
    }
  }
};


template <
    typename T,
    class FG = internal::Sigmoid<T>,
    class FI = internal::Tanh<T>,
    class FO = internal::Tanh<T>>
class LstmAachenCellTraining : LstmAachenCellInference<T, FG, FI, FO> {
 public:
  template <class LstmImpl>
  void Backward(LstmImpl* lstm, const int t, const int tn, const int tmin) {
    const int* shape = lstm->GetShape();
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        for (int n = 0; n < lstm->GetN(); ++n) {
          for (int d = 0; d < lstm->GetD(); ++d) {
            const int i = e + Tmin;
            const int j = t - i;
            const int y = (z == 0 || z == 2) ? i : lstm->GetH() - i - 1;
            const int x = (z == 0 || z == 1) ? j : lstm->GetW() - j - 1;
            if (shape == nullptr ||
                (y < shape[2 * n] && x < shape[2 * n + 1])) {
              const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
              const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
              const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // previous y
              const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;  // previous x
              const T fGi  = lstm->Q(z, y, x, n, 0, d);
              const T fGy  = lstm->Q(z, y, x, n, 1, d);
              const T fGx  = lstm->Q(z, y, x, n, 2, d);
              const T fGo  = lstm->Q(z, y, x, n, 3, d);
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
              const T dC_00 = dO_00 * FO::df(C_00) * fGo + Z_10 + Z_01;
              lstm->Q(z, y, x, n, 0, d) = dC_00 * fA * FG::df2(fGi);
              lstm->Q(z, y, x, n, 1, d) = (yp >= 0 && yp < lstm->GetH())
                                          ? (dC_00 * FG::df2(fGx) * C_10) : 0;
              lstm->Q(z, y, x, n, 2, d) = (xp >= 0 && xp < lstm->GetW())
                                          ? (dC_00 * FG::df2(fGx) * C_01) : 0;
              lstm->Q(z, y, x, n, 3, d) = dO_00 * FO::f(C_00) * FG::df2(fGo);
              lstm->Q(z, y, x, n, 4, d) = dC_00 * FI::df2(fA) * fGi;
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
};

}
}
}

#endif // RNN2D_INTERNAL_CPU_LSTM_STANDARD_CELL_IMPL_H_

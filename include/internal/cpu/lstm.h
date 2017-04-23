#ifndef RNN2D_INTERNAL_CPU_LSTM_H
#define RNN2D_INTERNAL_CPU_LSTM_H

#include "rnn2d.h"
#include "internal/common.h"
#include "internal/lstm.h"
#include "internal/cpu/math.h"

namespace rnn2d {
namespace internal {
namespace cpu {

template <typename T, class FI, class FO, class FG>
rnn2dStatus_t Lstm


template <typename T, int NG>
class Lstm : public ::rnn2d::internal::Lstm<T, NG> {
 public:
  using Lstm<T, NG>::Lstm;

  rnn2dStatus_t ForwardInference() const override {

    return ForwardImpl<false>();
  }

  rnn2dStatus_t ForwardTraining() const override {

    return ForwardImpl<true>();
  }

  rnn2dStatus_t Backward() const override {

  }

 private:
  template <bool training>
  rnn2dStatus_t ForwardImpl() const {
    if (H_ <= 0 || W_ <= 0 || N_ <= 0 || K_ <= 0 || D_ <= 0) {
      RNN2D_SET_ERROR_MSG("LSTM: Some dimension is non-positive.");
      return RNN2D_STATUS_BAD_PARAM;
    }

    T *Q = reinterpret_cast<T*>(training ? rspace : wspace);

    // Initialize gates with bias
    // [A,Gi,Go,Gx,Gy](x,y) = [b_a,b_i,b_o,b_x,b_y]
    #pragma omp parallel for collapse(6)
    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < H_; ++y)
        for (int x = 0; x < W_; ++x)
          for (int n = 0; n < N_; ++n)
            for (int g = 0; g < NG; ++g)
              for (int d = 0; d < D_; ++d) {
                *PtrQ(Q, z, y, x, n, g, d) = *PtrB(z, g, d);
              }

    // Multiply inputs by weights:
    // [A,Gi,Go,Gx,Gy](x,y) += I(x,y) * [W_a,W_i,W_o,W_x,W_y]
    // Note: Each direction could run in parallel. Not done here because gemm
    // already uses multiple threads.
    for (int z = 0; z < 4; ++z) {
      // I viewed as a (H * W * N) x K matrix
      // W viewed as a K x (NG * D) matrix
      // Q viewed as a (H * W * N) x (NG * D) matrix
      gemm<T>('N', 'N', H * W * N, NG * D, K,
              1.0, input_, K, PtrW(z, 0, 0, 0), NG * D,
              1.0, PtrQ(Q, z, 0, 0, 0, 0, 0), NG * D);
    }

    // Process the image diagonal-wise (there are H + W - 1 diagonals)
    for (int t = 0; t < H + W - 1; ++t) {
      // Compute number of elements in the diagonal
      const int Tmin = std::max(0, t - W + 1);
      const int Tmax = std::min(t, H - 1);
      const int Tn = (Tmax - Tmin) + 1;

      // Matrix multiplications to compute the input to the gates from the
      // recurrent connections.
      // [Gi,Gy,Gx,Go,A](x,y) += O(x,y-1) * [U_i,U_y,U_x,U_o,U_a]
      // [Gi,Gy,Gx,Go,A](x,y) += O(x-1,y) * [V_i,V_y,V_x,V_o,V_a]
      #pragma omp parallel for
      for (int e = 0; e < 4 * Tn; ++e) {
        const int z = e / Tn; // Diagonal direction
        // (y, x) coordinates of the e-th element in the t-th diagonal.
        const int i = (e % Tn) + Tmin;
        const int j = t - i;
        const int y = (z == 0 || z == 2) ? (i) : (H - i - 1);
        const int x = (z == 0 || z == 1) ? (j) : (W - j - 1);
        const int yp = (z == 0 || z == 2) ? (y - 1) : (y + 1);
        const int xp = (z == 0 || z == 1) ? (x - 1) : (x + 1);
        if (yp >= 0 && yp < H) {
          gemm<T>('N', 'N', N, NG * D, D,
                  1.0, PtrO(yp, x, 0, z, 0), 4 * D,
                  V_ptr(P, z, 0, 0, 0), NG * D,
                  1.0, Q_ptr(z, y, x, 0, 0, 0), NG * D);
        }
        if (xp >= 0 && xp < W) {
          gemm<T>('N', 'N', N, NG * D, D,
                  1.0, O_ptr(O, y, xp, 0, z, 0), 4 * D,
                  U_ptr(P, z, 0, 0, 0), NG * D,
                  1.0, Q_ptr(z, y, x, 0, 0, 0), NG * D);
        }
      }

      // Compute cell states
      ForwardElemwise(t, Tn, Tmin, Q);
    }

    return RNN2D_STATUS_SUCCESS;
  }

  virtual void ForwardElemwise(int t, int L, int Lmin, T* Q) = 0;
};


template <typename T, class FG, class FI, class FO>
class LstmFiveGates : public Lstm<T, 5> {
 public:
  using Lstm<T, 5>::Lstm;

 private:
  void ForwardElemwise(int t, int L, int Lmin, T* Q) override {
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int i = e + Lmin;
            const int j = t - i;
            const int y  = (z == 0 || z == 2) ? i : H - i - 1;
            const int x  = (z == 0 || z == 1) ? j : W - j - 1;
            if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
              const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
              const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
              const T fGi = FG::f(GetQ(Q, z, y, x, n, 0, d));  // input gate
              const T fGy = FG::f(GetQ(Q, z, y, x, n, 1, d));  // fgt_y gate
              const T fGx = FG::f(GetQ(Q, z, y, x, n, 2, d));  // fgt_x gate
              const T fGo = FG::f(GetQ(Q, z, y, x, n, 3, d));  // output gate
              const T fA  = FI::f(GetQ(Q, z, y, x, n, 4, d));  // pre-cell
              // previous cell state in the y-direction
              const T C_10 =
                  (yp >= 0 && yp < H) ? GetQ(Q, z, yp, x, n, 4, d) : 0;
              // previous cell state in the x-direction
              const T C_01 =
                  (xp >= 0 && xp < W) ? GetQ(Q, z, y, xp, n, 4, d) : 0;
              const T C_00 = ForwardCell(fA, fGi, fGy, C_10, fGx, C_01);
              const T O_00 = fGo * FO::f(C_00);               // output
              SetQ(Q, z, y, x, n, 0, d) = fGi;
              SetQ(Q, z, y, x, n, 1, d) = fGy;
              SetQ(Q, z, y, x, n, 2, d) = fGx;
              SetQ(Q, z, y, x, n, 3, d) = fGo;
              SetQ(Q, z, y, x, n, 4, d) = C_00;
              *PtrO(y, x, n, z, d) = O_00;
            } else {
              SetQ(Q, z, y, x, n, 0, d) = 0;
              SetQ(Q, z, y, x, n, 1, d) = 0;
              SetQ(Q, z, y, x, n, 2, d) = 0;
              SetQ(Q, z, y, x, n, 3, d) = 0;
              SetQ(Q, z, y, x, n, 4, d) = 0;
              *PtrO(y, x, n, z, d) = 0;
            }
          }
        }
      }
    }
  }

  virtual T ForwardCell(T fA, T fGi, T fGy, T C_10, T fGx, T C_01) = 0;
};


template <typename T, class FG, class FI, class FO>
class LstmStandard : public LstmFiveGates<T, FG, FI, FO> {
 public:
   using LstmFiveGates<T, FG, FI, FO>::LstmFiveGates;

 private:
   T ForwardCell(T fA, T fGi, T fGy, T C_10, T fGx, T C_01) override {
    return fA * fGi + fGy * C_10 + fGx * C_01;
   }
};


template <typename T, class FG, class FI, class FO>
class LstmAachen : public LstmFiveGates<T, FG, FI, FO> {
 public:
   using LstmFiveGates<T, FG, FI, FO>::LstmFiveGates;

 private:
   T ForwardCell(T fA, T fGi, T fGy, T C_10, T fGx, T C_01) override {
    return fA * fGi + 0.5 * fGy * C_10 + 0.5 * fGx * C_01;
   }
};


template <typename T>
class LstmStable : public LstmFiveGates<T, FG, FI, FO> {
 public:
   using LstmFiveGates<T, FG, FI, FO>::LstmFiveGates;

 private:
   T ForwardCell(T fA, T fGi, T fGy, T C_10, T fGx, T C_01) override {
    return fA * fGi + fGy * (fGx * C_10 + (1.0 - fGx) * C_01);
   }
};

template <typename T>
class LstmLeaky : public Lstm<T, 4> {
};


// C(y, x) = A(y, x)     * G(y, x, 1) +
//           C(y - 1, x) * G(y, x, 2) +
//           C(y, x - 1) * G(y, x, 3)
// where G(y, x, i) is a softmax function, which basically controls the
// propagation from the input, the recurrent-y or the recurrent-x in a
// soft way.
template <typename T, class FI, class FO, class FG>
class LstmSoftmax : public Lstm<T, 5> {
 public:
  using Lstm<T, 5>::Lstm;

 private:
  void ForwardElemwise(int t, int L, int Lmin, T* Q) override {
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int i = e + Lmin;
            const int j = t - i;
            const int y  = (z == 0 || z == 2) ? i : H - i - 1;
            const int x  = (z == 0 || z == 1) ? j : W - j - 1;
            if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
              const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
              const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
              // Pre-activation of the input, recurrent-y and recurrent-x gates.
              const T Gi = GetQ(Q, z, y, x, n, 0, d);
              const T Gy = GetQ(Q, z, y, x, n, 1, d);
              const T Gx = GetQ(Q, z, y, x, n, 1, d);
              // Output gate
              const T fGo = FG::f(GetQ(Q, z, y, x, n, 3, d));
              // Pre-cell activation
              const T fA  = FI::f(GetQ(Q, z, y, x, n, 4, d));
              // Previous cell state in the y-direction
              const T C_10 =
                  (yp >= 0 && yp < H) ? GetQ(Q, z, yp, x, n, 4, d) : 0;
              // Previous cell state in the x-direction
              const T C_01 =
                  (xp >= 0 && xp < W) ? GetQ(Q, z, y, xp, n, 4, d) : 0;
              // Compute cell state. Note: this is equivalent to use the
              // softmax as weights to linearly interpolate {fA, C_10, C_01}.
              const T C_00 =
                  fA   / (1 + exp(Gy - Gi) + exp(Gx - Gi)) +
                  C_10 / (exp(Gi - Gy) + 1 + exp(Gx - Gy)) +
                  C_01 / (exp(Gi - Gx) + exp(Gy - Gx) + 1);
              // Cell output
              const T O_00 = fGo * FO::f(C_00);
              SetQ(Q, z, y, x, n, 0, d) = Gi;
              SetQ(Q, z, y, x, n, 1, d) = Gy;
              SetQ(Q, z, y, x, n, 2, d) = Gx;
              SetQ(Q, z, y, x, n, 3, d) = fGo;
              SetQ(Q, z, y, x, n, 4, d) = C_00;
              *PtrO(y, x, n, z, d) = O_00;
            } else {
              SetQ(Q, z, y, x, n, 0, d) = 0;
              SetQ(Q, z, y, x, n, 1, d) = 0;
              SetQ(Q, z, y, x, n, 2, d) = 0;
              SetQ(Q, z, y, x, n, 3, d) = 0;
              SetQ(Q, z, y, x, n, 4, d) = 0;
              *PtrO(y, x, n, z, d) = 0;
            }
          }
        }
      }
    }
  }

  virtual T ForwardCell(T fA, T fGi, T fGy, T C_10, T fGx, T C_01) = 0;
};


}  // namespace lstm
}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif // RNN2D_INTERNAL_CPU_LSTM_H

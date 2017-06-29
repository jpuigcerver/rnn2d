//
// Created by Joan Puigcerver on 25/04/2017.
//

#ifndef RNN2D_INTERNAL_CPU_LSTM_INFERENCE_LOW_MEM_IMPL_H_
#define RNN2D_INTERNAL_CPU_LSTM_INFERENCE_LOW_MEM_IMPL_H_

#include <rnn2d/activations.h>
#include <rnn2d/basic_types.h>
#include <rnn2d/internal/cpu/math.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template <typename T>
class LstmInferenceCpuImpl {
 public:
  LstmInferenceCpuImpl(const int K, const int D) : K_(K), D_(D) {}

  size_t GetNumParameters() const {
    // 4 directions x 5 gates x D hidden units x (1 + K + D + D) connections
    return 4 * 5 * D_ * (1 + K_ + D_ + D_);
  }

  size_t GetSizeWSpace() const {
    return 4 * 2 * 5 * std::min(H_, W_) * N_ * D_ * sizeof(T);
  }

  void SetInput(const int H, const int W, const int N, const int *shape,
                const T *input) {
    H_ = H;
    W_ = W;
    N_ = N;
    shape_ = shape;
    input_ = input;
  }

  void SetOutput(T *output) { output_ = output; }

  void SetParameters(const T *param) { param_ = param; }

  void SetWSpace(void *wspace) { wspace_ = wspace; }

  rnn2dStatus_t Forward() {
    Qp_ = reinterpret_cast<T*>(wspace_);
    Qc_ = reinterpret_cast<T*>(wspace_) + 4 * 5 * std::min(H_, W_) * N_ * D_;
    // Process the image diagonal-wise (there are H + W - 1 diagonals)
    for (int t = 0; t < H + W - 1; ++t) {
      // Compute number of elements in the diagonal
      const int Tmin = std::max(0, t - W + 1);
      const int Tmax = std::min(t, H - 1);
      const int Tn   = (Tmax - Tmin) + 1;

      // Initialize cells with bias
      #pragma omp parallel for collapse(5)
      for (int z = 0; z < 4; ++z)
        for (int l = 0; l < std::min(H_, W_); ++l)
          for (int n = 0; n < N_; ++n)
            for (int g = 0; g < 5; ++g)
              for (int d = 0; d < D_; ++d) {
                Qc(z, l, n, g, d) = B(z, g, d);
              }


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
        const int y  = (z == 0 || z == 2) ? (i    ) : (H - i - 1);
        const int x  = (z == 0 || z == 1) ? (j    ) : (W - j - 1);
        const int yp = (z == 0 || z == 2) ? (y - 1) : (y + 1    );
        const int xp = (z == 0 || z == 1) ? (x - 1) : (x + 1    );
        if (xp >= 0 && xp < W) {
          gemm<T>('N', 'N', N, 5 * D, D,
                  1.0, O(O, y, xp, 0, z, 0), 4 * D,
                  U(z, 0, 0, 0), 5 * D,
                  1.0, Q(z, y, x, 0, 0, 0), 5 * D);
        }
        if (yp >= 0 && yp < H) {
          gemm<T>('N', 'N', N, 5 * D, D,
                  1.0, O(O, yp, x, 0, z, 0), 4 * D,
                  V(P, z, 0, 0, 0), 5 * D,
                  1.0, Q(z, y, x, 0, 0, 0), 5 * D);
        }
      }

      // Compute cell states
      ComputeCells(t, Tn, Tmin);
    }
  }

 protected:
  const int K_, D_;
  int H_, W_, N_;
  const int *shape_;
  const T *input_, *param_;
  T *output_;
  void *wspace_;
  T *Qc_, *Qp_;

  static inline
  size_t Qoffset(const int z, const int l ,const int n, const int g, const int d) {
    return z * std::min(H_, W_) * N_ * 5 * D_ +
           l * N_ * 5 * D_ +
           n * 5 * D_ +
           g * D_ +
           d;
  }

  inline
  T* Qc(const int z, const int l, const int n, const int g, const int d) {
    return Qc_ + Qoffset(z, l, n, g, d);

  }

  inline
  T* Qp(const int z, const int l, const int n, const int g, const int d) {
    return Qp_ + Qoffset(z, l, n, g, d);
  }

  inline
  T& Qc(const int z, const int l, const int n, const int g, const int d) {
    return *Qc(z, l, n, g, d);
  }

  inline
  T& Qp(const int z, const int l, const int n, const int g, const int d) {
    return *Qp(z, l, n, g, d);
  }

  inline
  const T& B(const int z, const int g, const int d) const {
    return param_[
        z * 5 * D_ * (1 + K_ + D_ + D_) +
            g * D_ +
            d
    ];
  }

  inline
  const T* W(const int z, const int k, const int g, const int d) const {
    return param_ +
        5 * D_ +  // offset
        z * 5 * D_ * (1 + K_ + D_ + D_) +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  inline
  const T* U(const int z, const int k, const int g, const int d) {
    return param_ +
        5 * D_ + K_ * 5 * D_ +  // offset
        z * 5 * D_ * (1 + K_ + D_ + D_) +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  inline
  const T* V(const int z, const int k, const int g, const int d) {
    return param_ +
        5 * D_ + K_ * 5 * D_ + D_ * 5 * D_ +  // offset
        z * 5 * D_ * (1 + K_ + D_ + D_) +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  inline
  T* O(const int y, const int x, const int n, const int z, const int d) {
    return output_ +
        y * W_ * N_ * 4 * D_ +
        x * N_ * 4 * D_ +
        n * 4 * D_ +
        z * D_ +
        d;
  }

  inline
  T& O(const int y, const int x, const int n, const int z, const int d) {
    return *O(y, x, n, z, d);
  }

  virtual void ComputeCells(const int t, const int tn, const int tmin) = 0;
};

template <typename T>
class StandardLstmInferenceCpuImpl : public LstmInferenceCpuImpl<T> {
 protected:
  using LstmInferenceCpuImpl<T>::K_;
  using LstmInferenceCpuImpl<T>::D_;
  using LstmInferenceCpuImpl<T>::H_;
  using LstmInferenceCpuImpl<T>::W_;
  using LstmInferenceCpuImpl<T>::N_;
  using LstmInferenceCpuImpl<T>::shape_;

  void ComputeCells(const int t, const int tn, const int tmin) final {
    // Compute cell and output values
    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < tn; ++e) {
        for (int n = 0; n < N_; ++n) {
          for (int d = 0; d < D_; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int i = e + tmin;
            const int j = t - i;
            const int y  = (z == 0 || z == 2) ? i : H_ - i - 1;
            const int x  = (z == 0 || z == 1) ? j : W_ - j - 1;
            if (shape_ == nullptr ||
                (y < shape_[n * 2] && x < shape_[n * 2 + 1])) {
              const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
              const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
              const T fGi = Sigmoid::f(Q(z, y, x, n, 0, d));  // input gate
              const T fGy = Sigmoid::f(Q(z, y, x, n, 1, d));  // fgt_y gate
              const T fGx = Sigmoid::f(Q(z, y, x, n, 2, d));  // fgt_x gate
              const T fGo = Sigmoid::f(Q(z, y, x, n, 3, d));  // output gate
              const T fA  = Tanh::f(Q(z, y, x, n, 4, d));     // pre-cell
              const T C_10 = (yp >= 0 && yp < H_) ? Q(z, yp, x, n, 4, d) : 0;
              const T C_01 = (xp >= 0 && xp < W_) ? Q(z, y, xp, n, 4, d) : 0;
              const T C_00 = fGi * fA + fGy * C_10 + fGx * C_01;  // state
              const T O_00 = fGo * Tanh::f(C_00);                 // output
              Q(z, y, x, n, 0, d) = fGi;
              Q(z, y, x, n, 1, d) = fGy;
              Q(z, y, x, n, 2, d) = fGx;
              Q(z, y, x, n, 3, d) = fGo;
              Q(z, y, x, n, 4, d) = C_00;
              O(O, y, x, n, z, d) = O_00;
            } else {
              Q(z, y, x, n, 0, d) = 0;
              Q(z, y, x, n, 1, d) = 0;
              Q(z, y, x, n, 2, d) = 0;
              Q(z, y, x, n, 3, d) = 0;
              Q(z, y, x, n, 4, d) = 0;
              O(O, y, x, n, z, d) = 0;
            }
          }
        }
      }
    }
  }
};

}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif // RNN2D_INTERNAL_CPU_LSTM_INFERENCE_LOW_MEM_IMPL_H_

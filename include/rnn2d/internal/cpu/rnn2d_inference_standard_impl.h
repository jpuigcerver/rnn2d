#ifndef RNN2D_INTERNAL_CPU_RNN2D_INFERENCE_STANDARD_IMPL_H_
#define RNN2D_INTERNAL_CPU_RNN2D_INFERENCE_STANDARD_IMPL_H_

#include <rnn2d/internal/activation.h>
#include <rnn2d/internal/rnn2d_inference_impl.h>
#include <rnn2d/internal/cpu/math.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template <typename T, class Cell>
class Rnn2dInferenceStandardImpl :
    public ::rnn2d::internal::Rnn2dInferenceImpl<T> {
 public:
  using Rnn2dInferenceImpl<T>::GetX;
  using Rnn2dInferenceImpl<T>::GetY;

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceStandardImpl(const int K, const int D) :
      Rnn2dInferenceImpl<T>(K, D), G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceStandardImpl(const int K, const int D, const Cell& cell) :
      Rnn2dInferenceImpl<T>(K, D), cell_(cell), G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceStandardImpl(const int K, const int D, Cell&& cell) :
      Rnn2dInferenceImpl<T>(K, D), cell_(std::move(cell)), G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceStandardImpl(const Rnn2dInferenceStandardImpl& impl) :
    Rnn2dInferenceImpl<T>(impl), cell_(impl.cell_), G_(impl.G_) {}

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceStandardImpl(Rnn2dInferenceStandardImpl&& impl) :
      Rnn2dInferenceImpl<T>(std::move(impl)), cell_(std::move(impl.cell_)), G_(impl.G_) {}

  CUDA_CALLABLE_MEMBER
  size_t GetNumParameters() const override {
    // 4 directions x G gates x D hidden units x (1 + K + D + D) connections
    return 4 * G_ * D_ * (1 + K_ + D_ + D_);
  }

  CUDA_CALLABLE_MEMBER
  size_t GetSizeWSpace() const override {
    return 4 * H_ * W_ * N_ * G_ * D_ * sizeof(T);
  }

  rnn2dStatus_t Forward() override {
    RNN2D_CHECK_AND_RETURN_ERROR(input_ != nullptr, "Input array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(output_ != nullptr, "Output array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(wspace_ != nullptr, "Workspace array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(param_ != nullptr, "Parameters array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);

    // Initialize gates with bias
    // [A,Gi,Go,Gx,Gy](x,y) = [b_a,b_i,b_o,b_x,b_y]
    #pragma omp parallel for collapse(6)
    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < H_; ++y)
        for (int x = 0; x < W_; ++x)
          for (int n = 0; n < N_; ++n)
            for (int g = 0; g < G_; ++g)
              for (int d = 0; d < D_; ++d) {
                *Q(z, y, x, n, g, d) = *B(z, g, d);
              }

    // Multiply inputs by weights:
    // [A,Gi,Go,Gx,Gy](x,y) += I(x,y) * [W_a,W_i,W_o,W_x,W_y]
    // Note: Each direction could run in parallel, but gemm already uses
    // multiple threads.
    for (int z = 0; z < 4; ++z) {
      // I viewed as a (H * W * N) x K matrix
      // W viewed as a K x (5 * D) matrix
      // Q viewed as a (H * W * N) x (5 * D) matrix
      gemm<T>(
          'N', 'N', H_ * W_ * N_, G_ * D_, K_,
          1.0, input_, K_, W(z, 0, 0, 0), G_ * D_,
          1.0, Q(z, 0, 0, 0, 0, 0), G_ * D_);
    }

    // Process the image diagonal-wise. There are H + W - 1 diagonals, each
    // of them with up to L elements.
    const int L = std::min(H_, W_);
    for (int t = 0; t < H_ + W_ - 1; ++t) {
      // Matrix multiplications to compute the input to the gates from the
      // recurrent connections.
      // [Gi,Gy,Gx,Go,A](x,y) += O(x,y-1) * [U_i,U_y,U_x,U_o,U_a]
      // [Gi,Gy,Gx,Go,A](x,y) += O(x-1,y) * [V_i,V_y,V_x,V_o,V_a]
      #pragma omp parallel for collapse(2)
      for (int z = 0; z < 4; ++z) {
        for (int e = 0; e < L; ++e) {
          // (y, x) coordinates of the e-th element in the t-th diagonal.
          const int y = GetY(z, t, e), x = GetX(z, t, e);
          if (y >= 0 && x >= 0 && y < H_ && x < W_) {
            const int yp = (z == 0 || z == 2) ? (y - 1) : (y + 1);
            const int xp = (z == 0 || z == 1) ? (x - 1) : (x + 1);
            if (xp >= 0 && xp < W_) {
              gemm<T>('N', 'N', N_, G_ * D_, D_,
                      1.0, O(z, y, xp, 0, 0), 4 * D_,
                      U(z, 0, 0, 0), G_ * D_,
                      1.0, Q(z, y, x, 0, 0, 0), G_ * D_);
            }
            if (yp >= 0 && yp < H_) {
              gemm<T>('N', 'N', N_, G_ * D_, D_,
                      1.0, O(z, yp, x, 0, 0), 4 * D_,
                      V(z, 0, 0, 0), G_ * D_,
                      1.0, Q(z, y, x, 0, 0, 0), G_ * D_);
            }
          }
        }
      }

      // Compute cell states and output values
      RNN2D_RETURN_ERROR_IF_FAILED(cell_.Forward(this, t));
    }

    return RNN2D_STATUS_SUCCESS;
  }

  CUDA_CALLABLE_MEMBER static inline
  T* Q(const int H, const int W, const int N, const int G, const int D,
       const int z, const int y, const int x, const int n, const int g,
       const int d, T* ptr) {
    return ptr +
        z * H * W * N * G * D +
        y * W * N * G * D +
        x * N * G * D +
        n * G * D +
        g * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  const T* Q(const int H, const int W, const int N, const int G, const int D,
             const int z, const int y, const int x, const int n, const int g,
             const int d, const T* ptr) {
    return ptr +
        z * H * W * N * G * D +
        y * W * N * G * D +
        x * N * G * D +
        n * G * D +
        g * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  T* O(const int H, const int W, const int N, const int D,
       const int z, const int y, const int x, const int n, const int d,
       T* ptr) {
    return ptr +
        y * W * N * 4 * D +
        x * N * 4 * D +
        n * 4 * D +
        z * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  const T* O(const int H, const int W, const int N, const int D,
             const int z, const int y, const int x, const int n, const int d,
             const T* ptr) {
    return ptr +
        y * W * N * 4 * D +
        x * N * 4 * D +
        n * 4 * D +
        z * D +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T* Q(const int z, const int y, const int x, const int n, const int g,
       const int d) {
    return Q(H_, W_, N_, G_, D_, z, y, x, n, g, d,
             reinterpret_cast<T*>(wspace_));
  }

  CUDA_CALLABLE_MEMBER inline
  const T* Q(const int z, const int y, const int x, const int n, const int g,
             const int d) const {
    return Q(H_, W_, N_, G_, D_, z, y, x, n, g, d,
             reinterpret_cast<const T*>(wspace_));
  }

  CUDA_CALLABLE_MEMBER inline
  T* B(const int z, const int g, const int d) {
    return param_ +
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T* B(const int z, const int g, const int d) const {
    return param_ +
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T* W(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T* W(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T* U(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ + K_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T* U(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ + K_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T* V(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ + K_ * G_ * D_ + D_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T* V(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ + K_ * G_ * D_ + D_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T* O(const int z, const int y, const int x, const int n, const int d) {
    return O(H_, W_, N_, D_, z, y, x, n, d, output_);
  }

 protected:
  using Rnn2dInferenceImpl<T>::H_;
  using Rnn2dInferenceImpl<T>::W_;
  using Rnn2dInferenceImpl<T>::N_;
  using Rnn2dInferenceImpl<T>::K_;
  using Rnn2dInferenceImpl<T>::D_;
  using Rnn2dInferenceImpl<T>::input_;
  using Rnn2dInferenceImpl<T>::shape_;
  using Rnn2dInferenceImpl<T>::param_;
  using Rnn2dInferenceImpl<T>::output_;
  using Rnn2dInferenceImpl<T>::wspace_;

  Cell cell_;
  const int G_;
};

}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_CPU_RNN2D_INFERENCE_STANDARD_IMPL_H_

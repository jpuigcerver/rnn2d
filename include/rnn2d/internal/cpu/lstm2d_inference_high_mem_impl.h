#ifndef RNN2D_INTERNAL_CPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_
#define RNN2D_INTERNAL_CPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

#include <include/rnn2d/activations.h>
#include <rnn2d/internal/lstm2d_inference_high_mem_impl.h>
#include <rnn2d/internal/cpu/math.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template<typename T, class C>
class Lstm2dInferenceHighMemImpl :
    public ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C> {
 public:
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::Lstm2dInferenceHighMemImpl;

  using Rnn2dInferenceImpl<T>::GetX;
  using Rnn2dInferenceImpl<T>::GetY;

  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::Q;
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::B;
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::U;
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::V;
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::W;
  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::O;

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

  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::G_;


  rnn2dStatus_t ForwardBias() override {
    #pragma omp parallel for collapse(6)
    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < H_; ++y)
        for (int x = 0; x < W_; ++x)
          for (int n = 0; n < N_; ++n)
            for (int g = 0; g < G_; ++g)
              for (int d = 0; d < D_; ++d) {
                *Q(z, y, x, n, g, d) = *B(z, g, d);
              }
    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardInput() override {
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
    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardPreviousOutputs(const int L, const int t) override {
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
    return RNN2D_STATUS_SUCCESS;
  }
};

}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_CPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

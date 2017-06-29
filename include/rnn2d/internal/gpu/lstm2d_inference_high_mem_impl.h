#ifndef RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_
#define RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

#include <include/rnn2d/activations.h>
#include <rnn2d/internal/lstm2d_inference_high_mem_impl.h>
#include <rnn2d/internal/gpu/math.h>

namespace rnn2d {
namespace internal {
namespace gpu {

template<typename T, class C>
class Lstm2dInferenceHighMemImpl :
    public ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C> {
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

    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardInput() override {

    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardPreviousOutputs(const int L, const int t) override {

    return RNN2D_STATUS_SUCCESS;
  }
};

}  // namespace gpu
}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

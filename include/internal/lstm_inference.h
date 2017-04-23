#ifndef RNN2D_INTERNAL_LSTM_INFERENCE_H_
#define RNN2D_INTERNAL_LSTM_INFERENCE_H_

#include <algorithm>

#include "internal/common.h"
#include "internal/lstm.h"

namespace rnn2d {
namespace internal {

//
template <typename T>
class LstmInference : public Lstm<T> {
 public:
  bool IsTrainable() const final { return false; }
};


// Base implementation of a LstmInference layer.
//
template <typename T>
class LstmInferenceImpl : public LstmImpl<T> {
 public:
  size_t GetSizeWSpace() const override {
    const size_t tmpd_size = 4 * H_ * W_* N_ * 5 * D_ * sizeof(T);
    const size_t ptrs_size = 2 * 3 * 4 * std::min(H_, W_) * sizeof(T*);
    return tmpd_size + ptrs_size;
  }

  rnn2dStatus_t Forward() final {
    RNN2D_CHECK_AND_RETURN_ERROR(
        H_ > 0, "LstmInference: Height must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        W_ > 0, "LstmInference: Width must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        N_ > 0, "LstmInference: Batch size must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        K_ > 0, "LstmInference: Input depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        D_ > 0, "LstmInference: Output depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_ != nullptr, "LstmInference: Input array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        param_ != nullptr, "LstmInference: Parameters array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_ != nullptr, "LstmInference: Output array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        wspace_ != nullptr, "LstmInference: Work array is null",
        RNN2D_STATUS_BAD_PARAM);
    return ForwardInferenceImpl();
  }

 protected:
  virtual rnn2dStatus_t ForwardInferenceImpl() = 0;
};


}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_LSTM_INFERENCE_H_

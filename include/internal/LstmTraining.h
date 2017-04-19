#ifndef RNN2D_INTERNAL_LSTMTRAINING_H_
#define RNN2D_INTERNAL_LSTMTRAINING_H_

#include <algorithm>

#include "internal/common.h"
#include "internal/Lstm.h"

namespace rnn2d {
namespace internal {

template <typename T>
class LstmTraining : public Lstm<T> {
 public:
  CUDA_CALLABLE_MEMBER
  LstmTraining(int K, int D) : Lstm<T>(K, D), stateDone_(BACKWARD_PARAM) {}

  // Return the required size (in bytes) of the reserved space.
  size_t GetRSpaceSize() const override {
    return 4 * H_ * W_ * N_ * 5 * D_ * sizeof(T);
  }

  // Return the size (in bytes) of the workspace required for training.
  size_t GetWSpaceSize() const override {
    const size_t tmpd_size = 3 * 4 * H_ * W_ * N_ * D_ * sizeof(T);
    const size_t ptrs_size = 2 * 3 * 4 * std::min(H_, W_) * sizeof(T*);
    return tmpd_size + ptrs_size;
  }

  rnn2dStatus_t Forward() override {
    if (H_ <= 0 || W_ <= 0 || N_ <= 0 || K_ <= 0 || D_ <= 0) {
      RNN2D_SET_ERROR_MSG("LSTMTraining: Some dimension is non-positive.");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (input_ == nullptr || param_ == nullptr || output_ == nullptr ||
        wspace_ == nullptr || rspace_ == nullptr) {
      RNN2D_SET_ERROR_MSG(
          "LSTMTraining: Unexpected null pointer in Forward().");
      return RNN2D_STATUS_BAD_PARAM;
    }
    stateDone_ = FORWARD;
    return ForwardImpl();
  }

  rnn2dStatus_t BackwardData() override {
    if (H_ <= 0 || W_ <= 0 || N_ <= 0 || K_ <= 0 || D_ <= 0) {
      RNN2D_SET_ERROR_MSG("LSTM: Some dimension is non-positive.");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (input_ == nullptr || gInput_ == nullptr || output_ == nullptr ||
        gOutput_ == nullptr || param_ == nullptr || wspace_ == nullptr ||
        rspace_ == nullptr) {
      RNN2D_SET_ERROR_MSG(
          "LSTM: Unexpected null pointer in BackwardData().");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (stateDone_ != FORWARD) {
      RNN2D_SET_ERROR_MSG(
          "LSTM: Forward() must run before BackwardData().");
      return RNN2D_STATUS_WRONG_STATE;
    }
    stateDone_ = BACKWARD_DATA;
    return BackwardDataImpl();
  }

  rnn2dStatus_t BackwardParam() override {
    if (H_ <= 0 || W_ <= 0 || N_ <= 0 || K_ <= 0 || D_ <= 0) {
      RNN2D_SET_ERROR_MSG("LSTM: Some dimension is non-positive.");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (input_ == nullptr || output_ == nullptr || gradParam_ == nullptr ||
        wspace_ == nullptr || rspace_ == nullptr) {
      RNN2D_SET_ERROR_MSG(
          "LSTM: Unexpected null pointer in BackwardParam().");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (stateDone_ != BACKWARD_DATA) {
      RNN2D_SET_ERROR_MSG(
          "LSTM: BackwardData() must run before BackwardParam().");
      return RNN2D_STATUS_WRONG_STATE;
    }
    stateDone_ = BACKWARD_PARAM;
    return BackwardParamImpl();
  }

 private:
  virtual rnn2dStatus_t ForwardImpl() = 0;

  virtual rnn2dStatus_t BackwardDataImpl() = 0;

  virtual rnn2dStatus_t BackwardParamImpl() = 0;

  // Keep track of the last performed operation.
  enum {
    FORWARD           = 0,
    BACKWARD_DATA     = 1,
    BACKWARD_PARAM    = 2
  } stateDone_;
};

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_LSTMINFERENCE_H_

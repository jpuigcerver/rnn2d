#ifndef RNN2D_INTERNAL_LSTM_TRAINING_H_
#define RNN2D_INTERNAL_LSTM_TRAINING_H_

#include <algorithm>

#include "internal/common.h"
#include "internal/lstm.h"

namespace rnn2d {
namespace internal {

// Lstm layer with training forward and backward operations.
template <typename T>
class LstmTraining : public Lstm<T> {
 public:
  LstmTraining(int K, int D) :
      Lstm<T>(K, D),
      input_grad_(nullptr), param_grad_(nullptr), output_grad_(nullptr),
      rspace_(nullptr), curr_state_(NONE) {}

  bool IsTrainable() const final { return true; }

  virtual void SetGradInput(T *input_grad) {
    input_grad_ = input_grad;
  }

  virtual void SetGradOutput(const T *output_grad) {
    output_grad_ = output_grad;
  }

  virtual void SetGradParameters(T *param_grad) {
    param_grad_ = param_grad;
  }

  virtual void SetRSpace(void *rspace) {
    rspace_ = rspace;
  }

  rnn2dStatus_t Forward() final {
    RNN2D_CHECK_AND_RETURN_ERROR(
        H_ > 0, "LstmTraining: Height must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        W_ > 0, "LstmTraining: Width must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        N_ > 0, "LstmTraining: Batch size must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        K_ > 0, "LstmTraining: Input depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        D_ > 0, "LstmTraining: Output depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_ != nullptr, "LstmTraining: Input array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        param_ != nullptr, "LstmTraining: Parameters array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_ != nullptr, "LstmTraining: Output array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        wspace_ != nullptr, "LstmTraining: Work array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        rspace_ != nullptr, "LstmTraining: Reserved array is null",
        RNN2D_STATUS_BAD_PARAM);
    curr_state_ = FORWARD;
    return ForwardImpl();
  }

  virtual rnn2dStatus_t BackwardData() {
    RNN2D_CHECK_AND_RETURN_ERROR(
        curr_state_ == FORWARD,
        "LstmTraining: Forward() must run just before BackwardData()",
        RNN2D_STATUS_WRONG_STATE);
    RNN2D_CHECK_AND_RETURN_ERROR(
        H_ > 0, "LstmTraining: Height must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        W_ > 0, "LstmTraining: Width must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        N_ > 0, "LstmTraining: Batch size must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        K_ > 0, "LstmTraining: Input depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        D_ > 0, "LstmTraining: Output depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_ != nullptr, "LstmTraining: Input array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_grad_ != nullptr, "LstmTraining: Input gradient array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        param_ != nullptr, "LstmTraining: Parameters array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_ != nullptr, "LstmTraining: Output array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_grad_ != nullptr, "LstmTraining: Output gradient array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        wspace_ != nullptr, "LstmTraining: Work array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        rspace_ != nullptr, "LstmTraining: Reserved array is null",
        RNN2D_STATUS_BAD_PARAM);
    curr_state_ = BACKWARD_DATA;
    return BackwardDataImpl();
  }

  virtual rnn2dStatus_t BackwardParam() {
    RNN2D_CHECK_AND_RETURN_ERROR(
        curr_state_ == BACKWARD_DATA,
        "LstmTraining: BackwardData() must run just before BackwardParam()",
        RNN2D_STATUS_WRONG_STATE);
    RNN2D_CHECK_AND_RETURN_ERROR(
        H_ > 0, "LstmTraining: Height must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        W_ > 0, "LstmTraining: Width must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        N_ > 0, "LstmTraining: Batch size must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        K_ > 0, "LstmTraining: Input depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        D_ > 0, "LstmTraining: Output depth must be positive",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_ != nullptr, "LstmTraining: Input array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        input_grad_ != nullptr, "LstmTraining: Input gradient array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        param_ != nullptr, "LstmTraining: Parameters array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_ != nullptr, "LstmTraining: Output array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        output_grad_ != nullptr, "LstmTraining: Output gradient array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        wspace_ != nullptr, "LstmTraining: Work array is null",
        RNN2D_STATUS_BAD_PARAM);
    RNN2D_CHECK_AND_RETURN_ERROR(
        rspace_ != nullptr, "LstmTraining: Reserved array is null",
        RNN2D_STATUS_BAD_PARAM);
    curr_state_ = BACKWARD_PARAM;
    return BackwardParamImpl();
  }

  virtual size_t GetSizeRSpace() const = 0;

 protected:
  // Pointer to the gradients of the input and parameters w.r.t. the loss.
  T *input_grad_, *param_grad_;
  // Pointer to the gradients of the output w.r.t. the loss.
  const T *output_grad_;
  // Reserved space used during forward/backward operations.
  void *rspace_;
  // Keep track of the last performed operation.
  enum {
    NONE              = 0,
    FORWARD           = 1,
    BACKWARD_DATA     = 2,
    BACKWARD_PARAM    = 3
  } curr_state_;

 private:
  virtual rnn2dStatus_t ForwardImpl() = 0;
  virtual rnn2dStatus_t BackwardDataImpl() = 0;
  virtual rnn2dStatus_t BackwardParamImpl() = 0;
};

template <class Impl, class Inter = LstmTraining<typename Impl::DateType>>
class ImplToLstmTraining : public ImplToLstm<Impl, Inter> {
 public:
  void SetGradInput(T *input_grad) override {
    GetMutableImpl()->SetGradInput(input_grad);
  }

  void SetGradOutput(const T *output_grad) override {
    GetMutableImpl()->SetGradOutput(output_grad);
  }

  void SetGradParameters(T *param_grad) override {
    GetMutableImpl()->SetGradParameters(param_grad);
  }

  void SetRSpace(void *rspace) override {
    GetMutableImpl()->SetRSpace(rspace);
  }

  size_t GetSizeRSpace() const override {
    return GetImpl()->GetSizeRSpace();
  }

  rnn2dStatus_t BackwardData() override {
    return GetMutableImpl()->BackwardData();
  }

  rnn2dStatus_t BackwardParam() override {
    return GetMutableImpl()->BackwardParam();
  }

 protected:
  ImplToLstmTraining(Impl* impl) : ImplToLstm<Impl, Inter>(impl) {}

  ImplToLstmTraining(const ImplToLstmTraining& other) :
      ImplToLstm<Impl, Inter>(other) {}

};

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_LSTM_TRAINING_H_

#ifndef RNN2D_INTERNAL_LSTMINFERENCE_H_
#define RNN2D_INTERNAL_LSTMINFERENCE_H_

#include <algorithm>

#include "internal/common.h"
#include "internal/Lstm.h"

namespace rnn2d {
namespace internal {

template <typename T>
class LstmInference : public Lstm<T> {
 public:
  using Lstm<T>::Lstm;

  // Return the size (in bytes) of the workspace required for inference.
  size_t GetWSpaceSize() const override {
    const size_t tmpd_size = 4 * H_ * W_ * N_ * 5 * D_ * sizeof(T);
    const size_t ptrs_size = 2 * 3 * 4 * std::min(H_, W_) * sizeof(T*);
    return tmpd_size + ptrs_size;
  }

  rnn2dStatus_t Forward() override {
    if (H_ <= 0 || W_ <= 0 || N_ <= 0 || K_ <= 0 || D_ <= 0) {
      RNN2D_SET_ERROR_MSG("LSTM: Some dimension is non-positive.");
      return RNN2D_STATUS_BAD_PARAM;
    }
    if (input_ == nullptr || param_ == nullptr || output_ == nullptr ||
        wspace_ == nullptr) {
      RNN2D_SET_ERROR_MSG(
          "LSTM: Unexpected null pointer in Forward().");
      return RNN2D_STATUS_BAD_PARAM;
    }
    return ForwardImpl();
  }

  rnn2dStatus_t BackwardData() final {
    RNN2D_SET_ERROR_MSG(
        "LstmInference class does not implement BackwardData().");
    return RNN2D_STATUS_NOT_SUPPORTED;
  }

  rnn2dStatus_t BackwardParam() final {
    RNN2D_SET_ERROR_MSG(
        "LstmInference class does not implement BackwardParam().");
    return RNN2D_STATUS_NOT_SUPPORTED;
  }

 private:
  virtual rnn2dStatus_t ForwardImpl() = 0;
};

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_LSTMINFERENCE_H_

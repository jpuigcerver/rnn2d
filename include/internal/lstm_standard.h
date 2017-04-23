#ifndef RNN2D_INTERNAL_LSTM_STANDARD_H_
#define RNN2D_INTERNAL_LSTM_STANDARD_H_

#include "internal/lstm_inference.h"

namespace rnn2d {
namespace internal {

template <typename T>
class LstmStandardInferenceImpl : public LstmInferenceImpl<T> {
 public:
  size_t GetNumParameters() const override {

  }

  size_t GetSizeWSpace() const override {

  }
};

}  // namespace internal
}  // namespace rnn2d


#endif // RNN2D_INTERNAL_LSTM_STANDARD_H_

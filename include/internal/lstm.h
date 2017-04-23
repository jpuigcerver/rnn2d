#ifndef RNN2D_INTERNAL_LSTM_H
#define RNN2D_INTERNAL_LSTM_H

#include <algorithm>

#include "internal/common.h"

namespace rnn2d {
namespace internal {

// Generic Lstm layer interface, templated with the data type
// (e.g. T = float, T = double, etc).
//
// Typically, one would use LstmInference or LstmTraining instead, depending
// on whether the Lstm layer is used for evaluation or training.
template <typename T>
class Lstm {
 public:
  virtual ~Lstm() {}

  // Return the number of parameters.
  virtual size_t GetNumParameters() const = 0;

  // Return the size (in bytes) of the work array required.
  virtual size_t GetSizeWSpace() const = 0;

  // Check whether or not the class has backward methods.
  // If IsTrainable() returns true, a pointer/reference to the object can be
  // casted to LstmTraining, otherwise it can be casted to LstmInference.
  virtual bool IsTrainable() const = 0;

  // Set the input max height, max width and batch size dimensions, the shape
  // (height and width) of each image in the batch, and the pointer to the
  // data.
  virtual void SetInput(const int H, const int W, const int N,
                        const int *shape, const T *input) = 0;

  // Set the pointer to the output array
  virtual void SetOutput(T *output) = 0;

  // Set the pointer to the parameters array. The parameters array should have
  // GetNumParameters() elements.
  virtual void SetParameters(const T *param) = 0;

  // Set the pointer to the work array. It must point to an array of, at least,
  // GetSizeWSpace() bytes. The same work array can be used by different
  // layers, as long as they do not perform concurrent operations.
  virtual void SetWSpace(void *wspace) = 0;

  // Perform forward pass.
  virtual rnn2dStatus_t Forward() = 0;
};

// Lstm implementation base.
//
// This implements most of setter methods required to perform the forward pass.
template <typename T>
class LstmImpl {
 public:
  LstmImpl(int K, int D) :
      K_(K), D_(D), H_(0), W_(0), N_(0),
      shape_(nullptr), input_(nullptr), param_(nullptr), output_(nullptr),
      wspace_(nullptr) {}

  virtual ~LstmImpl() {}

  virtual rnn2dStatus_t Forward() = 0;

  virtual size_t GetNumParameters() const {
    // This is the default number of parameters for a LstmLayer, but children
    // classes may override this, thus the virtual keyword.
    // 4 directions x 5 gates x D hidden units x (1 + K + D + D) connections
    return 4 * 5 * D_ * (1 + K_ + D_ + D_);
  }

  virtual size_t GetSizeWSpace() const = 0;

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

 protected:
  const int K_, D_;
  int H_, W_, N_;
  const int *shape_;
  const T *input_, *param_;
  T *output_;
  void *wspace_;
};


// Helper class useful to attach a Lstm layer to its implementation.
template <class Impl, class Inter = Lstm<typename Impl::DataType>>
class ImplToLstm : public Inter {
 public:
  void SetInput(const int H, const int W, const int N, const int *shape,
                const T *input) override {
    impl_->SetInput(H, W, N, shape, input);
  }

  void SetOutput(T *output) override { impl_->SetOutput(output); }

  void SetParameters(const T *param) override { impl_->SetParameters(param); }

  void SetWSpace(void *wspace) override { impl_->SetWSpace(wspace); }

  size_t GetNumParameters() const override {
    return impl_->GetNumParameters();
  }

  size_t GetSizeWSpace() const override { return impl_->GetSizeWSpace(); }

  rnn2dStatus_t Forward() override { return impl_->Forward(); }

 protected:
  explicit ImplToLstm(Impl* impl) :
      impl_(impl) {}

  ImplToLstm(const ImplToLstm<Impl, Inter>& other) :
      impl_(other->impl_->Clone()) {}

  const Impl* GetImpl() const { return impl_.get(); }

  Impl* GetMutableImpl() const { return impl_.get(); }

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace internal
}  // namespace rnn2d

#endif //RNN2D_INTERNAL_LSTM_H

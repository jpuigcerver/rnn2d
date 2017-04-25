#ifndef RNN2D_LAYER_H_
#define RNN2D_LAYER_H_

#include <rnn2d/rnn2d.h>

namespace rnnd2 {

// Interface to a generic layer, templated with the data type
// (e.g. T = float, T = double, etc).
//
// Typically, one would use LayerInferenceInterface or LayerTrainingInterface
// instead, depending whether the layer is used for inference or training.
template <typename T>
class LayerInterface {
 public:
  virtual ~LayerInterface() {}

  // Check whether or not the class has backward methods.
  // If IsTrainable() returns true, a pointer/reference to the object can be
  // casted to LayerTraining, otherwise it can be casted to LayerInference.
  virtual bool IsTrainable() const = 0;

  // Set the input max height, max width and batch size dimensions, the shape
  // (height and width) of each image in the batch, and the pointer to the
  // data.
  virtual void SetInput(const int H, const int W, const int N,
                        const int *shape, const T *input) = 0;

  // Set the pointer to the output array
  virtual void SetOutput(T *output) = 0;

  // Perform forward pass.
  virtual rnn2dStatus_t Forward() = 0;
};


// Interface to a generic layer used in inference mode.
//
// Layers used in inference mode do not need to provide a backward pass, and
// thus can have further optimizations (especially memory optimizations) than
// layers implementing the backward pass.
template <typename T>
class LayerInferenceInterface : public LayerInterface<T> {
 public:
  bool IsTrainable() const final { return false; }
};


// Interface to a generic layer used in training mode.
//
// Layers in training mode need to provide two methods for the backward pass:
// BackwardData() and BackwardParams().
template <typename T>
class LayerTrainingInterface : public LayerInterface<T> {
 public:
  bool IsTrainable() const final { return true; }

  virtual void SetGradInput(T *input_grad) = 0;

  virtual void SetGradOutput(const T *output_grad) = 0;

  virtual rnn2dStatus_t BackwardData() = 0;

  virtual rnn2dStatus_t BackwardParams() = 0;
};


// Helper class useful to attach a layer interface to its implementation.
template <class Impl, class Inter = LayerInterface<typename Impl::DataType>>
class ImplToLayer : public Inter {
 public:
  void SetInput(const int H, const int W, const int N, const int *shape,
                const T *input) override {
    impl_->SetInput(H, W, N, shape, input);
  }

  void SetOutput(T *output) override { impl_->SetOutput(output); }

  rnn2dStatus_t Forward() override { return impl_->Forward(); }

 protected:
  explicit ImplToLayer(Impl* impl) :
      impl_(impl) {}

  ImplToLayer(const ImplToLayer<Impl, Inter>& other) :
      impl_(other->impl_->Clone()) {}

  const Impl* GetImpl() const { return impl_.get(); }

  Impl* GetMutableImpl() const { return impl_.get(); }

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace rnn2d

#endif // RNN2D_LAYER_H_

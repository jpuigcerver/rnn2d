#ifndef RNN2D_RNN_H_
#define RNN2D_RNN_H_

#include <rnn2d/layer.h>

namespace rnn2d {

template <typename T>
class Rnn2dInterface : public Layer2dInterface<T> {
 public:
  // Number of input channels.
  virtual int GetK() const = 0;

  // Pointer to the array containg the shape of each image in the batch.
  virtual const int* GetShape() const = 0;

  // Return the number of parameters.
  virtual size_t GetNumParameters() const = 0;

  // Return the required size (in bytes) of the work array.
  virtual size_t GetSizeWSpace() const = 0;

  // Set the pointer to the parameters array. The parameters array should have
  // GetNumParameters() elements.
  virtual void SetParameters(const T *param) = 0;

  // Set the pointer to the work array. It must point to an array of, at least,
  // GetSizeWSpace() bytes. The same work array can be used by different
  // layers, as long as they do not perform concurrent operations.
  virtual void SetWSpace(void *wspace) = 0;
};


template <typename T>
class Rnn2dInferenceInterface :
    public Rnn2dInterface<T>, public LayerInferenceInterface<T> { };


template <typename T>
class Rnn2dTrainingInterface :
    public Rnn2dInterface<T>, public LayerTrainingInterface<T> {
 public:
  // Return the required size (in bytes) for the reserved array.
  virtual size_t GetSizeRSpace() const = 0;

  // Set the pointer to the gradient of the parameters w.r.t. the loss.
  // The gradients array should have GetNumParameters() elements.
  virtual void SetGradParameters(T *param_grad) = 0;

  // Set the pointer to the reserved array. It must point to an array of
  // GetSizeRSpace() bytes. Each Lstm layer must have its own reserved array.
  virtual void SetRSpace(void *rspace) = 0;
};


template <class Impl,
    class Inter = Rnn2dInterface<typename Impl::DataType>>
class ImplToRnn2d : public ImplToLayer2d<Impl, Inter> {
 public:
  int GetK() const override {
    GetImpl()->GetK();
  }

  const int* GetShape() const override {
    GetImpl()->GetShape();
  }

  size_t GetNumParameters() const override {
    return GetImpl()->GetNumParameters();
  }

  size_t GetSizeWSpace() const override {
    return GetImpl()->GetSizeWSpace();
  }

  void SetParameters(const T *param) override {
    GetMutableImpl()->SetParameters(param);
  }

  void SetWSpace(void *wspace) override {
    GetMutableImpl()->SetWSpace(param);
  }
};


template <class Impl,
    class Inter = Rnn2dInferenceInterface<typename Impl::DataType>>
class ImplToRnn2dInference : public ImplToRnn2d<Impl, Inter> { };


template <class Impl,
    class Inter = Rnn2dTrainingInterface<typename Impl::DataType>>
class ImplToRnn2dTraining : public ImplToRnn2d<Impl, Inter> {
 public:
  void SetGradInput(T *input_grad) override {
    GetMutableImpl()->SetGradInput(input_grad);
  }

  void SetGradOutput(const T *output_grad) override {
    GetMutableImpl()->SetGradOutput(output_grad);
  }

  void SetGradParameters(T *param_grad) override {
    return GetMutableImpl()->SetGradParameters(param_grad);
  }

  size_t GetSizeRSpace() const override {
    return GetImpl()->GetSizeRSpace();
  }

  void SetRSpace(void *rspace) override {
    return GetMutableImpl()->SetRSpace(rspace);
  }

  rnn2dStatus_t BackwardData() override {
    return GetMutableImpl()->BackwardData();
  }

  rnn2dStatus_t BackwardParams() override {
    return GetMutableImpl()->BackwardParams();
  }
};

}  // namespace rnn2d

#endif //RNN2D_RNN_H_

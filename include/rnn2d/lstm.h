#ifndef RNN2D_LSTM_H_
#define RNN2D_LSTM_H_

#include <rnn2d/layer.h>

namespace rnn2d {

template <typename T>
class LstmInterface : public LayerInterface<T> {
 public:
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
class LstmInferenceInterface :
    public LstmInterface<T>, public LayerInferenceInterface<T> { };


template <typename T>
class LstmTrainingInterface :
    public LstmInterface<T>, public LayerTrainingInterface<T> {
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
    class Inter = LstmInterface<typename Impl::DataType>>
class ImplToLstm : public ImplToLayer<Impl, Inter> {
 public:
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
    class Inter = LstmInferenceInterface<typename Impl::DataType>>
class ImplToLstmInference : public ImplToLstm<Impl, Inter> { };


template <class Impl,
    class Inter = LstmTrainingInterface<typename Impl::DataType>>
class ImplToLstmTraining : public ImplToLstm<Impl, Inter> {
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

#endif //RNN2D_LSTM_H_

#ifndef RNN2D_INTERNAL_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_
#define RNN2D_INTERNAL_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

#include <rnn2d/internal/rnn2d_inference_impl.h>

namespace rnn2d {
namespace internal {

template<typename T, class C>
class Lstm2dInferenceHighMemImpl :
    public ::rnn2d::internal::Rnn2dInferenceImpl<T> {
 public:
  using typename Rnn2dInferenceImpl<T>::DataType;
  typedef C Cell;

  CUDA_CALLABLE_MEMBER
  Lstm2dInferenceHighMemImpl(const int K, const int D) :
      Rnn2dInferenceImpl<T>(K, D), G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Lstm2dInferenceHighMemImpl(const int K, const int D, const Cell &cell) :
      Rnn2dInferenceImpl<T>(K, D), cell_(cell), G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Lstm2dInferenceHighMemImpl(const int K, const int D, Cell &&cell) :
      Rnn2dInferenceImpl<T>(K, D),
      cell_(std::move(cell)),
      G_(cell_.NumGates()) {}

  CUDA_CALLABLE_MEMBER
  Lstm2dInferenceHighMemImpl(const Lstm2dInferenceHighMemImpl &impl) :
      Rnn2dInferenceImpl<T>(impl), cell_(impl.cell_), G_(impl.G_) {}

  CUDA_CALLABLE_MEMBER
  Lstm2dInferenceHighMemImpl(Lstm2dInferenceHighMemImpl &&impl) :
      Rnn2dInferenceImpl<T>(std::move(impl)),
      cell_(std::move(impl.cell_)),
      G_(impl.G_) {}

  CUDA_CALLABLE_MEMBER
  size_t GetNumParameters() const override {
    // 4 directions x G gates x D hidden units x (1 + K + D + D) connections
    return 4 * G_ * D_ * (1 + K_ + D_ + D_);
  }

  CUDA_CALLABLE_MEMBER
  size_t GetSizeWSpace() const override {
    return 4 * H_ * W_ * N_ * G_ * D_ * sizeof(T);
  }

  inline int GetG() const { return G_; }

  rnn2dStatus_t Forward() override {
    RNN2D_CHECK_AND_RETURN_ERROR(input_ != nullptr, "Input array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(output_ != nullptr, "Output array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(wspace_ != nullptr, "Workspace array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);
    RNN2D_CHECK_AND_RETURN_ERROR(param_ != nullptr, "Parameters array is null",
                                 RNN2D_STATUS_NOT_INITIALIZED);

    // [A,Gi,Go,Gx,Gy](x,y) = [b_a,b_i,b_o,b_x,b_y]
    RNN2D_RETURN_ERROR_IF_FAILED(ForwardBias());

    // [A,Gi,Go,Gx,Gy](x,y) += I(x,y) * [W_a,W_i,W_o,W_x,W_y]
    RNN2D_RETURN_ERROR_IF_FAILED(ForwardInput());

    // Process the image diagonal-wise. There are H + W - 1 diagonals, each
    // of them with up to L elements.
    const int L = std::min(H_, W_);
    for (int t = 0; t < H_ + W_ - 1; ++t) {
      // [Gi,Gy,Gx,Go,A](x,y) += O(x,y-1) * [U_i,U_y,U_x,U_o,U_a]
      // [Gi,Gy,Gx,Go,A](x,y) += O(x-1,y) * [V_i,V_y,V_x,V_o,V_a]
      RNN2D_RETURN_ERROR_IF_FAILED(ForwardPreviousOutputs(L, t));
      // Compute cell states and output values
      RNN2D_RETURN_ERROR_IF_FAILED(cell_.Forward(this, t));
    }

    return RNN2D_STATUS_SUCCESS;
  }

  CUDA_CALLABLE_MEMBER static inline
  T *Q(const int H, const int W, const int N, const int G, const int D,
       const int z, const int y, const int x, const int n, const int g,
       const int d, T *ptr) {
    return ptr +
        z * H * W * N * G * D +
        y * W * N * G * D +
        x * N * G * D +
        n * G * D +
        g * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  const T *Q(const int H, const int W, const int N, const int G, const int D,
             const int z, const int y, const int x, const int n, const int g,
             const int d, const T *ptr) {
    return ptr +
        z * H * W * N * G * D +
        y * W * N * G * D +
        x * N * G * D +
        n * G * D +
        g * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  T *O(const int H, const int W, const int N, const int D,
       const int z, const int y, const int x, const int n, const int d,
       T *ptr) {
    return ptr +
        y * W * N * 4 * D +
        x * N * 4 * D +
        n * 4 * D +
        z * D +
        d;
  }

  CUDA_CALLABLE_MEMBER static inline
  const T *O(const int H, const int W, const int N, const int D,
             const int z, const int y, const int x, const int n, const int d,
             const T *ptr) {
    return ptr +
        y * W * N * 4 * D +
        x * N * 4 * D +
        n * 4 * D +
        z * D +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T *Q(const int z, const int y, const int x, const int n, const int g,
       const int d) {
    return Q(H_, W_, N_, G_, D_, z, y, x, n, g, d,
             reinterpret_cast<T *>(wspace_));
  }

  CUDA_CALLABLE_MEMBER inline
  const T *Q(const int z, const int y, const int x, const int n, const int g,
             const int d) const {
    return Q(H_, W_, N_, G_, D_, z, y, x, n, g, d,
             reinterpret_cast<const T *>(wspace_));
  }

  CUDA_CALLABLE_MEMBER inline
  T *B(const int z, const int g, const int d) {
    return param_ +
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T *B(const int z, const int g, const int d) const {
    return param_ +
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T *W(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T *W(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T *U(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ + K_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T *U(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ + K_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T *V(const int z, const int k, const int g, const int d) {
    return param_ +
        G_ * D_ + K_ * G_ * D_ + D_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  const T *V(const int z, const int k, const int g, const int d) const {
    return param_ +
        G_ * D_ + K_ * G_ * D_ + D_ * G_ * D_ +  // offset
        z * G_ * D_ * (1 + K_ + D_ + D_) +
        k * G_ * D_ +
        g * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER inline
  T *O(const int z, const int y, const int x, const int n, const int d) {
    return O(H_, W_, N_, D_, z, y, x, n, d, output_);
  }

 protected:
  Cell cell_;
  const int G_;

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

  virtual rnn2dStatus_t ForwardBias() = 0;
  virtual rnn2dStatus_t ForwardInput() = 0;
  virtual rnn2dStatus_t ForwardPreviousOutputs(const int L, const int t) = 0;
};

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

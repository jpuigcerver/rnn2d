#ifndef RNN2D_INTERNAL_RNN2D_INFERENCE_IMPL_H_
#define RNN2D_INTERNAL_RNN2D_INFERENCE_IMPL_H_

#include <memory>

#include <rnn2d/rnn2d.h>
#include <rnn2d/internal/common.h>

namespace rnn2d {
namespace internal {

template <typename T>
class Rnn2dInferenceImpl {
 public:
  typedef T DataType;

  CUDA_CALLABLE_MEMBER
  Rnn2dInferenceImpl(const int K, const int D) : K_(K), D_(D) {}

  // Return the depth at the input of the layer.
  CUDA_CALLABLE_MEMBER
  inline int GetK() const { return K_; }

  // Return the depth at the output of the layer.
  CUDA_CALLABLE_MEMBER
  inline int GetD() const { return D_; }

  // Return the number of samples in the batch.
  CUDA_CALLABLE_MEMBER
  inline int GetN() const { return N_; }

  // Return the batch height (i.e. maximum height of all images in the batch)
  CUDA_CALLABLE_MEMBER
  inline int GetH() const { return H_; }

  // Return the batch width (i.e. maximum width of all images in the batch)
  CUDA_CALLABLE_MEMBER
  inline int GetW() const { return W_; }

  // Return the height of a particular sample in the batch.
  CUDA_CALLABLE_MEMBER
  inline int GetH(const int n) const { return shape_ ? shape_[2*n+0] : H_; }

  // Return the width of a particular sample in the batch.
  CUDA_CALLABLE_MEMBER
  inline int GetW(const int n) const { return shape_ ? shape_[2*n+1] : W_; }

  // Get the number of parameters in this layer.
  CUDA_CALLABLE_MEMBER
  virtual size_t GetNumParameters() const = 0;

  // Get the size (in bytes) of the work space required by this layer.
  // Note: The same work space can be used by multiple layers that DO NOT
  // operate concurrently.
  CUDA_CALLABLE_MEMBER
  virtual size_t GetSizeWSpace() const = 0;

  // Set input batch buffer and sizes.
  CUDA_CALLABLE_MEMBER
  inline void SetInput(const int H, const int W, const int N, const int *shape,
                       const T *input) {
    H_ = H;
    W_ = W;
    N_ = N;
    shape_ = shape;
    input_ = input;
  }

  // Set output buffer.
  CUDA_CALLABLE_MEMBER
  inline void SetOutput(T *output) { output_ = output; }

  // Set parameters buffer.
  CUDA_CALLABLE_MEMBER
  inline void SetParameters(T *param) { param_ = param; }

  // Set work space buffer.
  CUDA_CALLABLE_MEMBER
  inline void SetWSpace(void *wspace) { wspace_ = wspace; }

  virtual rnn2dStatus_t Forward() = 0;

  // Get x-coordinate of the e-th element in the t-diagonal (in z-direction).
  CUDA_CALLABLE_MEMBER static inline
  int GetX(const int H, const int W, const int z, const int t, const int e) {
    if (z == 0 || z == 1) {
      return (H > W) ? (        e) : (        t - e);
    } else {
      return (H > W) ? (W - 1 - e) : (W - 1 - t + e);
    }
  }

  // Get y-coordinate of the e-th element in the t-diagonal (in z-direction).
  CUDA_CALLABLE_MEMBER static inline
  int GetY(const int H, const int W, const int z, const int t, const int e) {
    if (z == 0 || z == 2) {
      return (H > W) ? (        t - e) : (        e);
    } else {
      return (H > W) ? (H - 1 - t + e) : (H - 1 - e);
    }
  }

  // Get element from the previous diagonal in the x-direction
  CUDA_CALLABLE_MEMBER static inline
  int GetPrevElemX(const int H, const int W, const int e) {
    return (H > W) ? (e - 1) : (e);
  }

  // Get element from the previous diagonal in the y-direction
  CUDA_CALLABLE_MEMBER static inline
  int GetPrevElemY(const int H, const int W, const int e) {
    return (H > W) ? (e) : (e - 1);
  }

  CUDA_CALLABLE_MEMBER inline
  int GetX(const int z, const int t, const int e) const {
    return GetX(H_, W_, z, t, e);
  }

  CUDA_CALLABLE_MEMBER inline
  int GetY(const int z, const int t, const int e) const {
    return GetY(H_, W_, z, t, e);
  }

 protected:
  const int K_, D_;
  int H_, W_, N_;
  const int *shape_;
  const T *input_;
  T *param_, *output_;
  void *wspace_;
};

}  // namespace internal
}  // namespace rnn2d

#endif // RNN2D_INTERNAL_RNN2D_INFERENCE_IMPL_H_

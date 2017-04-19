#ifndef RNN2D_INTERNAL_LSTM_H
#define RNN2D_INTERNAL_LSTM_H

#include <algorithm>

#include "internal/common.h"

namespace rnn2d {
namespace internal {

// Abstract Lstm layer.
template <typename T>
class Lstm {
 public:
  CUDA_CALLABLE_MEMBER
  Lstm(int K, int D) :
      K_(K), D_(D), H_(0), W_(0), N_(0),
      input_(nullptr), gOutput_(nullptr), param_(nullptr),
      gInput_(nullptr), output_(nullptr), gParam_(nullptr),
      wspace_(nullptr), rspace_(nullptr) {}

  CUDA_CALLABLE_MEMBER
  virtual ~Lstm() {}

  // Return the number of parameters of this layer.
  virtual size_t GetNumParameters() const {
    return 4 * (1 + K_ + D_ + D_) * 5 * D_;
  }

  // Return the required size (in bytes) of the reserved space.
  virtual size_t GetRSpaceSize() const { return 0; }

  // Return the required size (in bytes) of the workspace.
  virtual size_t GetWSpaceSize() const { return 0; }

  // Set the pointer to the array of parameters. It must point to an array of
  // GetNumParameters() contiguous elements.
  void SetParameters(const T *param) {
    param_ = parameters;
  }

  // Set the pointer to the array of the parameters gradients. It must point
  // to an array of GetNumParameters() contiguous elements.
  void SetGradParameters(T *gParam) {
    gParam_ = gParameters;
  }

  // Set the pointer to the reserved space. It must point to an array of
  // GetRSpaceSize() bytes.
  inline void SetReservedSpace(void* rspace) {
    rspace_ = rspace;
  }

  // Set the pointer to the workspace. It must point to an array of
  // GetWSpaceSize() bytes.
  void SetWorkSpace(void* wspace) {
    wspace_ = wspace;
  }

  // Set the input max height, max width and batch size dimensions, the shape
  // (height and width) of each image in the batch, and the pointer to the
  // data.
  void SetInput(const int H, const int W, const int N, const int* shape,
                const T* input) {
    H_ = H;
    W_ = W;
    N_ = N;
    shape_ = shape;
    input_ = input;
  }

  // Set the pointer to the gradient of the input data.
  void SetGradInput(const T* gInput) {
    gInput_ = gInput;
  }

  // Set the pointer to the output array.
  void SetOutput(T* output) {
    output_ = output;
  }

  // Set the pointer to the output gradient array.
  void SetGradOutput(const T* gOutput) {
    gOutput_ = gOutput;
  }

  // Do forward pass.
  virtual rnn2dStatus_t Forward() = 0;

  // Do backward pass through the layer data and compute gradInput.
  virtual rnn2dStatus_t BackwardData() = 0;

  // Do backward pas through the layer parameters and comptue gradParam.
  virtual rnn2dStatus_t BackwardParam() = 0;

 protected:
  // Access input array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrI(int y, int x, int n, int k) const {
    return input_ + y * W_ * N_ * K_ + x * N_ * K_ + n * K_ + k;
  }

  // Access input gradient array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrGI(int y, int x, int n, int k) {
    return gInput_ + y * W_ * N_ * K_ + x * N_ * K_ + n * K_ + k;
  }

  // Access output array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrO(int y, int x, int n, int z, int d) {
    return output_ +
        y * W_ * N_ * 4 * D_ +
        x * N_ * 4 * D_ +
        n * 4 * D_ +
        z * D_ +
        d;
  }

  // Access output gradient array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrGO(int y, int x, int n, int z, int d) const {
    return gOutput_ +
        y * W_ * N_ * 4 * D_ +
        x * N_ * 4 * D_ +
        n * 4 * D_ +
        z * D_ +
        d;
  }

  // Access bias array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrB(int z, int g, int d) const {
    return param_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        g * D_ +
        d;
  }

  // Access bias gradient array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrGB(int z, int g, int d) {
    return gParam_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        g * D_ +
        d;
  }

  // Access input weights array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrW(int z, int k, int g, int d) const {
    return param_ + 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access input weights gradient array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrGW(int z, int k, int g, int d) {
    return gParam_ + 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access recurrent-y weight array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrU(int z, int k, int g, int d) const {
    return param_ + 5 * D_ + K_ * 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access recurrent-y weight gradient array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrGU(int z, int k, int g, int d) {
    return gParam_ + 5 * D_ + K_ * 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access recurrent-x weight array
  CUDA_CALLABLE_MEMBER virtual
  const T* PtrV(int z, int k, int g, int d) const {
    return param_ + 5 * D_ + K_ * 5 * D_ + D_ * 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access recurrent-x weight gradient array
  CUDA_CALLABLE_MEMBER virtual
  T* PtrGV(int z, int k, int g, int d) {
    return gParam_ + 5 * D_ + K_ * 5 * D_ + D_ * 5 * D_ +
        z * (1 + K_ + D_ + D_) * 5 * D_ +
        k * 5 * D_ +
        g * D_ +
        d;
  }

  // Access reserved space
  CUDA_CALLABLE_MEMBER virtual
  T* PtrQ(T* Q, int z, int y, int x, int n, int g, int d) {
    return Q +
        z * H_ * W_ * N_ * 5 * D_ +
        y * W_ * N_ * 5 * D_ +
        x * N_ * 5 * D_ +
        n * 5 * D_ +
        g * D_ +
        d;
  }

  // Access workspace space
  CUDA_CALLABLE_MEMBER virtual
  T* PtrZ(T* Z, int g, int z, int y, int x, int n, int d) {
    return Z +
        g * 4 * H_ * W_ * N_ * D_ +
        z * H_ * W_ * N_ * D_ +
        y * W_ * N_ * D_ +
        x * N_ * D_ +
        n * D_ +
        d;
  }

  CUDA_CALLABLE_MEMBER virtual
  const T& GetQ(const T* Q, int z, int y, int x, int n, int g, int d) const {
    return Q[
        z * H_ * W_ * N_ * 5 * D_ +
        y * W_ * N_ * 5 * D_ +
        x * N_ * 5 * D_ +
        n * 5 * D_ +
        g * D_ +
        d];
  }

  CUDA_CALLABLE_MEMBER virtual
  T& SetQ(T* Q, int z, int y, int x, int n, int g, int d) {
    return Q[
        z * H_ * W_ * N_ * 5 * D_ +
        y * W_ * N_ * 5 * D_ +
        x * N_ * 5 * D_ +
        n * 5 * D_ +
        g * D_ +
        d];
  }

 protected:
  // Input and output fixed dimensions.
  const int K_, D_;
  // Input and output dynamic dimensions.
  int H_, W_, N_;
  // Pointers to the different arrays.
  const T *input_, *gOutput_, *param_;
  T* gInput_, *output_, *gParam_;
  // Workspace and reservec space pointers.
  void *wspace_, *rspace_;
};

}  // namespace internal
}  // namespace rnn2d

#endif //RNN2D_INTERNAL_LSTM_H

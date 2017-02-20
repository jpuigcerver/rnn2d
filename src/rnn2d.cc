#include <rnn2d/rnn2d.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

struct rnn2dContext {
  int device;  // negative: CPU, positive: GPU
#ifdef WITH_CUDA
  cudaStream_t stream;
#endif
};

rnn2dStatus_t rnn2dCreate(rnn2dHandle_t* handle) {
  if (handle == nullptr) return RNN2D_STATUS_BAD_PARAM;
  *handle = new rnn2dContext;
  (*handle)->device = -1;
#ifdef WITH_CUDA
  (*handle)->stream = 0;
#endif
  return RNN2D_STATUS_SUCCESS;
}

rnn2dStatus_t rnn2dDestroy(rnn2dHandle_t handle) {
  if (handle == nullptr) return RNN2D_STATUS_NOT_INITIALIZED;
  delete handle;
  return RNN2D_STATUS_SUCCESS;
}

#ifdef WITH_CUDA

rnn2dStatus_t rnn2dSetDevice(rnn2dHandle_t handle, int device) {
  if (handle == nullptr) return RNN2D_STATUS_NOT_INITIALIZED;
  (*handle)->device = device;
  return RNN2D_STATUS_SUCCESS;
}

rnn2dStatus_t rnn2dGetDevice(rnn2dHandle_t handle, int* device) {
  if (handle == nullptr) return RNN2D_STATUS_NOT_INITIALIZED;
  if (device == nullptr) return RNN2D_STATUS_BAD_PARAM;
  *device = (*handle)->device;
  return RNN2D_STATUS_SUCCESS;
}

rnn2dStatus_t rnn2dSetStream(rnn2dHandle_t handle, cudaStream_t streamId) {
  if (handle == nullptr) return RNN2D_STATUS_NOT_INITIALIZED;
  (*handle)->stream = streamId;
  return RNN2D_STATUS_SUCCESS;
}

rnn2dStatus_t rnn2dGetStream(rnn2dHandle_t handle, cudaStream_t *streamId) {
  if (handle == nullptr) return RNN2D_STATUS_NOT_INITIALIZED;
  if (streamId == nullptr) return RNN2D_STATUS_BAD_PARAM;
  *streamId = (*handle)->stream;
  return RNN2D_STATUS_SUCCESS;
}

#endif // WITH_CUDA

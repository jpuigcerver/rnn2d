#ifndef RNN2D_UTILS_H_
#define RNN2D_UTILS_H_

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif  // USE_CUDA
#include <glog/logging.h>

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#ifdef USE_CUDA
#define CHECK_CUDA_CALL(err) {                                          \
    const cudaError_t e = (err);                                        \
    CHECK_EQ(e, cudaSuccess) << "CUDA error : " << e << " ("            \
                             << cudaGetErrorString(e)  << ")";          \
  }
#define CHECK_LAST_CUDA_CALL CHECK_CUDA_CALL(cudaGetLastError())
#endif  // USE_CUDA

// Some definitions that only make sense when using CUDACC
#ifdef __CUDACC__
// Thread IDs within a block
#define thBx (threadIdx.x)
#define thBy (threadIdx.y)
#define thBz (threadIdx.z)
#define thBi (                                          \
    threadIdx.x +                                       \
    threadIdx.y * blockDim.x +                          \
    threadIdx.z * blockDim.x * blockDim.y)
// Number of threads in a block
#define NTB (blockDim.x * blockDim.y * blockDim.z)

// Thread IDs within the grid (global IDs)
#define thGx (threadIdx.x + blockIdx.x * blockDim.x)
#define thGy (threadIdx.y + blockIdx.y * blockDim.y)
#define thGz (threadIdx.z + blockIdx.z * blockDim.z)
#define thGi (                                                          \
    threadIdx.x +                                                       \
    threadIdx.y * blockDim.x +                                          \
    threadIdx.z * blockDim.x * blockDim.y +                             \
    (blockIdx.x +                                                       \
     blockIdx.y * gridDim.x +                                           \
     blockIdx.z * gridDim.x * gridDim.z) *                              \
    blockDim.x * blockDim.y * blockDim.z)
// Number of blocks in the grid
#define NBG (gridDim.x * gridDim.y * gridDim.z)
// Number of threads in the grid (total number of threads)
#define NTG (blockDim.x * blockDim.y * blockDim.z * \
             gridDim.x * gridDim.y * gridDim.z)
#endif  // __CUDACC__

#endif  // RNN2D_UTILS_H_

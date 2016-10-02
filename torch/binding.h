#ifndef RNN2D_TORCH_BINDING_H_
#define RNN2D_TORCH_BINDING_H_

#include <lua.hpp>
#include <luaT.h>
#include <TH.h>
#ifdef USE_CUDA
#include <THC.h>
#endif  // USE_CUDA

template <typename Tensor>
inline const char* getTorchClass();

#define REGISTER_TORCH_CLASS(Type, ClassName)           \
  template <>                                           \
  inline const char* getTorchClass<Type>() {            \
    static char classname[] = ClassName;                \
    return classname;                                   \
  }

REGISTER_TORCH_CLASS(THFloatTensor, "torch.FloatTensor")
REGISTER_TORCH_CLASS(THDoubleTensor, "torch.DoubleTensor")
REGISTER_TORCH_CLASS(THIntTensor, "torch.IntTensor")

#ifdef USE_CUDA
REGISTER_TORCH_CLASS(THCudaTensor, "torch.CudaTensor")
REGISTER_TORCH_CLASS(THCudaDoubleTensor, "torch.CudaDoubleTensor")
REGISTER_TORCH_CLASS(THCudaHalfTensor, "torch.CudaHalfTensor")
REGISTER_TORCH_CLASS(THCudaIntTensor, "torch.CudaIntTensor")
#endif  // USE_CUDA

// All zeros -> Unknown
// MSB = 0 -> CPU, 1 -> GPU
// LSB = 0 -> Integer number, 1 -> Floating number
enum TensorType {
  UNKNOWN    = 0x00000000,
  CPU_INT    = 0x00000002,
  CPU_FLOAT  = 0x00000003,
  CPU_DOUBLE = 0x00000005,
  GPU_INT    = 0x80000002,
  GPU_FLOAT  = 0x80000003,
  GPU_DOUBLE = 0x80000005,
  GPU_HALF   = 0x80000007
};

#define IS_UNKNOWN(Type) ((Type) == UNKNOWN)
#define IS_CPU_TENSOR(Type) (((Type) & 0x80000000) == 0)
#define IS_GPU_TENSOR(Type) (((Type) & 0x80000000) == 1)
#define IS_INTEGER_TENSOR(Type) (((Type) & 0x01) == 0 && (Type) != UNKNOWN)
#define IS_FLOATING_TENSOR(Type) (((Type) & 0x01) == 1 && (Type) != UNKNOWN)

inline TensorType getTensorType(lua_State* L, int ud) {
  if (luaT_isudata(L, ud, "torch.IntTensor")) {
    return CPU_INT;
  } else if (luaT_isudata(L, ud, "torch.FloatTensor")) {
    return CPU_FLOAT;
  } else if (luaT_isudata(L, ud, "torch.DoubleTensor")) {
    return CPU_DOUBLE;
  } else if (luaT_isudata(L, ud, "torch.CudaIntTensor")) {
    return GPU_INT;
  } else if (luaT_isudata(L, 1, "torch.CudaTensor")) {
    return GPU_FLOAT;
  } else if (luaT_isudata(L, 1, "torch.CudaDoubleTensor")) {
    return GPU_DOUBLE;
  } else if (luaT_isudata(L, 1, "torch.CudaHalfTensor")) {
    return GPU_HALF;
  } else {
    return UNKNOWN;
  }
}

#include <cstdio>
#include <cstring>
#define __FILENAME__                                                    \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define THROW_LUA_ERROR(fmt) do {                          \
    char _msg[256];                                        \
    sprintf(_msg, "ERROR(%s:%d): " fmt,                    \
            __FILENAME__, __LINE__);                       \
    lua_pushfstring((L), _msg);                            \
    lua_error((L));                                        \
  } while(0)

#define THROW_LUA_ERROR_FMT(fmt, ...) do {                      \
    char _msg[256];                                             \
    sprintf(_msg, "ERROR(%s:%d): " fmt,                         \
            __FILENAME__, __LINE__, ##__VA_ARGS__);             \
    lua_pushfstring((L), _msg);                                 \
    lua_error((L));                                             \
  } while(0)

#define CHECK_TENSOR_N_DIMENSIONS(TNAME, EXPECTED, ACTUAL) do {         \
    if ((EXPECTED) != (ACTUAL)) {                                       \
      THROW_LUA_ERROR_FMT("Tensor " TNAME " has a wrong number of dimensions " \
                          "(expected: %d, actual: %d)",                 \
                          (int)(EXPECTED), (int)(ACTUAL));              \
    }                                                                   \
  } while(0)

#define CHECK_TENSOR_NOT_EMPTY(TNAME, TENSOR) do {                      \
    bool empty = (TENSOR)->nDimension < 1;                             \
    for (int d = 0; d < (TENSOR)->nDimension && !empty; ++d)           \
      empty = (TENSOR)->size[d] < 1;                                   \
    if (empty) {                                                       \
      THROW_LUA_ERROR("Tensor " TNAME " is empty!");                   \
    }                                                                  \
  } while(0)

#define CHECK_TENSOR_N_ELEMENTS(TNAME, EXPECTED, ACTUAL) do {           \
    if ((EXPECTED) != (ACTUAL)) {                                       \
      THROW_LUA_ERROR_FMT("Tensor " TNAME " has a wrong number of elements " \
                          "(expected: %d, actual: %d)",                 \
                          (int)(EXPECTED), (int)(ACTUAL));              \
    }                                                                   \
  } while(0)

#define CHECK_TENSOR_SIZE_2(TNAME, TENSOR, S1, S2) do {                 \
    CHECK_TENSOR_N_DIMENSIONS(TNAME, 2, (TENSOR)->nDimension);          \
    if ((TENSOR)->size[0] != (S1) || (TENSOR)->size[1] != (S2)) {       \
      THROW_LUA_ERROR_FMT("Tensor " TNAME " has a wrong size "          \
                          "(expected: %d x %d, actual: %d x %d)",       \
                          (int)(S1), (int)(S2), (int)(TENSOR)->size[0], \
                          (int)(TENSOR)->size[1]);                      \
    }                                                                   \
  } while(0)

#define CHECK_TENSOR_SIZE_4(TNAME, TENSOR, S1, S2, S3, S4) do {         \
    CHECK_TENSOR_N_DIMENSIONS(TNAME, 4, (TENSOR)->nDimension);          \
    if ((TENSOR)->size[0] != (S1) || (TENSOR)->size[1] != (S2) ||       \
        (TENSOR)->size[2] != (S3) || (TENSOR)->size[3] != (S4)) {       \
      THROW_LUA_ERROR_FMT("Tensor " TNAME " has a wrong size "          \
                          "(expected: %d x %d x %d x %d, "              \
                          "actual: %d x %d x %d x %d)",                 \
                          (int)(S1), (int)(S2), (int)(S3), (int)(S4),   \
                          (int)(TENSOR)->size[0], (int)(TENSOR)->size[1], \
                          (int)(TENSOR)->size[2], (int)(TENSOR)->size[3]); \
    }                                                                   \
  } while(0)

#define CHECK_TENSOR_CONTIGUOUS(TNAME, IS_CONTIGUOUS) do {      \
    if (!(IS_CONTIGUOUS)) {                                     \
      THROW_LUA_ERROR("Tensor " TNAME " is not contiguous!");   \
    }                                                           \
  } while(0)

#include "utils.h"
#include "generic/cpu.h"
#include "THGenerateFloatTypes.h"

#ifdef USE_CUDA
#include "generic/gpu.h"
#include "THCGenerateFloatTypes.h"
#endif

#endif  // RNN2D_TORCH_BINDING_H_

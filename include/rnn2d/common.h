#ifndef RNN2D_INTERNAL_COMMON_H_
#define RNN2D_INTERNAL_COMMON_H_

#include <string>

// Use this define in methods of classes that have to be called from CUDA.
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

namespace rnn2d {
namespace internal {

static const char* error_msg = "";

#define RNN2D_SET_ERROR_MSG(m) do {             \
static const std::string ms = (m);              \
::rnn2d::internal::error_msg = ms.c_str();      \
} while(0)

#define RNN2D_CHECK_AND_RETURN_ERROR(c, m, e) do {  \
  if (!(c)) {                                       \
    static const std::string ms = (m);              \
    ::rnn2d::internal::error_msg = ms.c_str();      \
    return (e);                                     \
  }                                                 \
} while(0)

#define RNN2D_CHECK_CUDA_AND_RETURN_ERROR(cuda_status, rnn2d_status, msg_pre) \
do {                                                                          \
  if (cuda_status != cudaSuccess) {                                           \
    static const std::string ms =                                             \
      std::string(msg_pre) + std::string(cudaGetErrorString(cuda_status));    \
    ::rnn2d::internal::error_msg = ms.c_str();                                \
    return (e);                                                               \
  }                                                                           \
} while(0)

#define RNN2D_RETURN_ERROR_IF_FAILED(e) \
  if ((e) != RNN2D_STATUS_SUCCESS) return (e)

#define RNN2D_GET_LAST_ERROR_MSG() ::rnn2d::internal::error_msg

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_COMMON_H_

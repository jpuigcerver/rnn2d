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
error_msg = ms.c_str();                         \
} while(0)

#define RNN2D_CHECK_AND_RETURN_ERROR(c, m, e) do {  \
  if (!(c)) {                                       \
    static const std::string ms = (m);              \
    error_msg = ms.c_str();                         \
    return (e);                                     \
  }                                                 \
} while(0)

}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_COMMON_H_

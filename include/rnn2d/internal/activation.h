#ifndef RNN2D_INTERNAL_ACTIVATION_H_
#define RNN2D_INTERNAL_ACTIVATION_H_

#include <rnn2d/internal/common.h>

namespace rnn2d {
namespace internal {

template <typename T>
class Linear {
 public:
  CUDA_CALLABLE_MEMBER
  static inline T f(const T& x) {
    return x;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x) {
    return 1;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df2(const T& fx) {
    return 1;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x, const T& fx) {
    return 1;
  }
};

template <typename T>
class Sigmoid {
 public:
  CUDA_CALLABLE_MEMBER
  static inline T f(const T& x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x) {
    const T fx = f(x);
    return (1 - fx) * fx;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df2(const T& fx) {
    return (1 - fx) * fx;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x, const T& fx) {
    return (1 - fx) * fx;
  }
};

template <typename T>
class Tanh {
 public:
  CUDA_CALLABLE_MEMBER
  static inline T f(const T& x) {
    return tanh(x);
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x) {
    const T fx = f(x);
    return 1 - fx * fx;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df2(const T& fx) {
    return 1 - fx * fx;
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x, const T& fx) {
    return 1 - fx * fx;
  }
};

template <typename T>
class ReLU {
 public:
  CUDA_CALLABLE_MEMBER
  static inline T f(const T& x) {
    return (x > 0 ? x : 0);
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x) {
    return (x > 0 ? 1 : 0);
  }
  CUDA_CALLABLE_MEMBER
  static inline T df2(const T& fx) {
    return (fx > 0 ? 1 : 0);
  }
  CUDA_CALLABLE_MEMBER
  static inline T df(const T& x, const T& fx) {
    return (x > 0 ? 1 : 0);
  }
};

}  // namespace internal
}  // namespace rnn2d

#endif // RNN2D_INTERNAL_ACTIVATION_H_

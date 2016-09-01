#ifndef RNN2D_ACTIVATION_H_
#define RNN2D_ACTIVATION_H_

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

template <typename T>
class Linear {
 public:
  __host__ __device__
  static inline T f(const T& x) {
    return x;
  }
  __host__ __device__
  static inline T df(const T& x) {
    return 1;
  }
};

template <typename T>
class Sigmoid {
 public:
  __host__ __device__
  static inline T f(const T& x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
  }
  __host__ __device__
  static inline T df(const T& x) {
    const T fx = f(x);
    return (1 - fx) * fx;
  }
};

template <typename T>
class Tanh {
 public:
  __host__ __device__
  static inline T f(const T& x) {
    return tanh(x);
  }
  __host__ __device__
  static inline T df(const T& x) {
    const T fx = f(x);
    return 1 - fx * fx;
  }
};

template <typename T>
class ReLU {
 public:
  __host__ __device__
  static inline T f(const T& x) {
    return (x > 0 ? x : 0);
  }
  __host__ __device__
  static inline T df(const T& x) {
    return (x > 0 ? 1 : 0);
  }
};

#endif  // RNN2D_ACTIVATION_H_

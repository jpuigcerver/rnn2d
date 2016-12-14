#ifndef RNN2D_TILE_IMPL_H_
#define RNN2D_TILE_IMPL_H_

#define I_ptr(I, y, x, n, d)                            \
  ((I) + ((((y) * W + (x)) * N) + (n)) * D + (d))

#define O_ptr(O, y, x, n, d)                            \
  ((O) + ((((y) * o_W + (x)) * N) + (n)) * o_D + (d))

#define DEFINE_WRAPPERS(DEVICE, TYPE)                                   \
  void rnn2d_tile_ ## DEVICE ## _ ## TYPE ## _fw(                       \
      const int H, const int W, const int N, const int D,               \
      const int Kh, const int Kw, const int* shape,                     \
      const TYPE* input, TYPE* output) {                                \
    fw< TYPE >(H, W, N, D, Kh, Kw, shape, input, output);               \
  }                                                                     \
                                                                        \
  void rnn2d_tile_ ## DEVICE ## _ ## TYPE ## _bw(                       \
      const int H, const int W, const int N, const int D,               \
      const int Kh, const int Kw, const int* shape,                     \
      const TYPE* dOutput, TYPE* dInput) {                              \
    bw< TYPE >(H, W, N, D, Kh, Kw, shape, dOutput, dInput);             \
  }

#endif  // RNN2D_TILE_IMPL_H_

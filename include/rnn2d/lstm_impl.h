#ifndef RNN2D_LSTM_IMPL_H_
#define RNN2D_LSTM_IMPL_H_

/* === 2D-LSTM EQUATIONS ===
 * Input: I(y,x) is a N x K matrix
 * Output: O(y,x) is a N x D matrix
 *
 * A(y,x)   = I(y,x) * W_a  + O(y-1,x) * R_ay  + O(y,x-1) * R_ax  + B_a
 * Gi(y,x)  = I(y,x) * W_i  + O(y-1,x) * R_iy  + O(y,x-1) * R_ix  + B_i
 * Go(y,x)  = I(y,x) * W_o  + O(y-1,x) * R_oy  + O(y,x-1) * R_ox  + B_o
 * Gfy(y,x) = I(y,x) * W_fy + O(y-1,x) * R_fyy + O(y,x-1) * R_fyx + B_fy
 * Gfx(y,x) = I(y,x) * W_fx + O(y-1,x) * R_fxy + O(y,x-1) * R_fxx + B_fx
 * C(y,x)   = s(Gi(y,x))  · g(A(y,x)) +
 *            s(Gfy(y,x)) · C(y-1,x)  +
 *            s(Gfx(y,x)) · C(y,x-1)
 * O(y,x)   = s(Go(y,x))  · g(C(y,x))
 *
 * Operator (*) denotes matrix multiplication, operator (·) is the element-wise
 * multiplication (or Hadamard product), s(z) is the sigmoid function and,
 * g is the tanh function.
 *
 * The previous equations decribe the output when the image is processed in
 * the top-left direction. The equations in the other directions are similar,
 * but the offset for the recurrent connections in each dimension changes:
 *   Top-Left origin:     y,x-offsets = -1, -1
 *   Top-Right origin:    y,x-offsets = -1, +1
 *   Bottom-Left origin:  y,x-offsets = +1, -1
 *   Bottom-Right origin: y,x-offsets = +1, +1
 */

// Useful defines to index several arrays:
// Input array
#define I_ptr(y, x, n, d)                       \
  (I  + (((y) * W + (x)) * N + (n)) * K + (d))
#define dI_ptr(y, x, n, d)                      \
  (dI + (((y) * W + (x)) * N + (n)) * K + (d))
// Output array
#define O_ptr(y, x, n, z, d)                                    \
  (O  + ((((y) * W + (x)) * N + (n)) * 4 + (z)) * D + (d))
#define dO_ptr(y, x, n, z, d)                                   \
  (dO + ((((y) * W + (x)) * N + (n)) * 4 + (z)) * D + (d))
// Bias array
#define B_ptr(z, g, d)                                  \
  (P  + (z) * (1 + K + D + D) * 5 * D + (g) * D + (d))
#define dB_ptr(z, g, d)                                 \
  (dP + (z) * (1 + K + D + D) * 5 * D + (g) * D + (d))
// Input weights array
#define W_ptr(z, k, g, d)                                               \
  (P  + ((z) * (1 + K + D + D) + 1 + (k)) * 5 * D + (g) * D + (d))
#define dW_ptr(z, k, g, d)                                              \
  (dP + ((z) * (1 + K + D + D) + 1 + (k)) * 5 * D + (g) * D + (d))
// Recurrent-y array
#define U_ptr(z, d1, g, d2)                                             \
  (P  + ((z) * (1 + K + D + D) + 1 + K + (d1)) * 5 * D + (g) * D + (d2))
#define dU_ptr(z, d1, g, d2)                                            \
  (dP + ((z) * (1 + K + D + D) + 1 + K + (d1)) * 5 * D + (g) * D + (d2))
// Recurrent-x array
#define V_ptr(z, d1, g, d2)                                             \
  (P  + ((z) * (1 + K + D + D) + 1 + K + D + (d1)) * 5 * D + (g) * D + (d2))
#define dV_ptr(z, d1, g, d2)                                            \
  (dP + ((z) * (1 + K + D + D) + 1 + K + D + (d1)) * 5 * D + (g) * D + (d2))

// Reserve array
#define Q_ptr(z, y, x, n, g, d)                                         \
  (Q  + (((((z) * H + (y)) * W + (x)) * N + (n)) * 5 + (g)) * D + (d))
// Workspace array
#define Z_ptr(g, z, y, x, n, d)                                         \
  (Z  + (((((g) * 4 + (z)) * H + (y)) * W + (x)) * N + (n)) * D + (d))

#define DEFINE_WRAPPERS(DEVICE, TYPE)                                   \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(             \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      TYPE* output, void* workspace) {                                  \
    fw_training< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(         \
        H, W, N, K, D, input, shape, param, output,                     \
        workspace, nullptr);                                            \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(              \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      TYPE* output, void* workspace, void* reserve) {                   \
    fw_training< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(         \
        H, W, N, K, D, input, shape, param, output,                     \
        workspace, reserve);                                            \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_workspace(             \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      const TYPE* output, const TYPE* dOutput,                          \
      void* workspace, void* reserve) {                                 \
    bw_workspace< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(        \
        H, W, N, K, D, input, shape, param, output, dOutput,            \
        workspace, reserve);                                            \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_input(                 \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* param, const TYPE scale, TYPE* dInput,                \
      void* workspace, void* reserve) {                                 \
    bw_input< TYPE >(H, W, N, K, D, param, scale, dInput,               \
                     workspace, reserve);                               \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_param(                 \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const TYPE* output, const TYPE scale,          \
      TYPE* dParam, void* workspace, void* reserve) {                   \
    bw_param< TYPE >(H, W, N, K, D, input, output, scale, dParam,       \
                     workspace, reserve);                               \
  }                                                                     \
                                                                        \
  size_t rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _inference_workspace_size( \
      const int H, const int W, const int N, const int D) {             \
    return get_inference_workspace_size<TYPE>(H, W, N, D);              \
  }                                                                     \
                                                                        \
  size_t rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _training_workspace_size( \
      const int H, const int W, const int N, const int D) {             \
    return get_training_workspace_size<TYPE>(H, W, N, D);               \
  }                                                                     \
                                                                        \
  size_t rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _training_reserve_size(  \
      const int H, const int W, const int N, const int D) {             \
    return get_training_reserve_size<TYPE>(H, W, N, D);                 \
  }

#endif  // RNN2D_LSTM_IMPL_H_

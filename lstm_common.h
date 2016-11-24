#ifndef RNN2D_LSTM_COMMON_H_
#define RNN2D_LSTM_COMMON_H_

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

// Useful defines to access specific addresses in the input, output and
// internal state tensors.
#define I_ptr(y, x, n, d)                       \
  (I  + (((y) * W + (x)) * N + (n)) * K + (d))
#define dI_ptr(y, x, n, d)                      \
  (dI + (((y) * W + (x)) * N + (n)) * K + (d))
#define O_ptr(y, x, n, z, d)                                    \
  (O  + ((((y) * W + (x)) * N + (n)) * 4 + (z)) * D + (d))
#define dO_ptr(y, x, n, z, d)                                   \
  (dO + ((((y) * W + (x)) * N + (n)) * 4 + (z)) * D + (d))
#define Q_ptr(z, y, x, n, g, d)                                 \
  (Q  + (((((z) * H + (y)) * W + (x)) * N + (n)) * 6 + (g)) * D + (d))
#define dQ_ptr(z, y, x, n, g, d)                                \
  (Q + (4 * H * W * N * 6 * D) + \
   (((((z) * H + (y)) * W + (x)) * N + (n)) * 6 + (g)) * D + (d))

#define B_ptr(z, g, d)                                  \
  (P  + (z) * (1 + K + D + D) * 5 * D + (g) * D + (d))
#define dB_ptr(z, g, d)                                 \
  (dP + (z) * (1 + K + D + D) * 5 * D + (g) * D + (d))
#define W_ptr(z, k, g, d)                                               \
  (P  + ((z) * (1 + K + D + D) + 1 + (k)) * 5 * D + (g) * D + (d))
#define dW_ptr(z, k, g, d)                                              \
  (dP + ((z) * (1 + K + D + D) + 1 + (k)) * 5 * D + (g) * D + (d))
#define Ry_ptr(z, d1, g, d2)                                            \
  (P  + ((z) * (1 + K + D + D) + 1 + K + (d1)) * 5 * D + (g) * D + (d2))
#define dRy_ptr(z, d1, g, d2)                                           \
  (dP + ((z) * (1 + K + D + D) + 1 + K + (d1)) * 5 * D + (g) * D + (d2))
#define Rx_ptr(z, d1, g, d2)                                            \
  (P  + ((z) * (1 + K + D + D) + 1 + K + D + (d1)) * 5 * D + (g) * D + (d2))
#define dRx_ptr(z, d1, g, d2)                                           \
  (dP + ((z) * (1 + K + D + D) + 1 + K + D + (d1)) * 5 * D + (g) * D + (d2))

#define print_4D(H, W, N, D, T) {                                       \
    for (int y = 0; y < (H); ++y) {                                     \
      for (int x = 0; x < (W); ++x) {                                   \
        for (int n = 0; n < (N); ++n) {                                 \
          printf("%s(%d, %d, %d, :) =", #T, y, x, n);                   \
          for (int d = 0; d < (D); ++d) {                               \
            printf(" %f", T[y * W * N * D + x * N * D + n * D + d]);    \
          }                                                             \
          printf("\n");                                                 \
        }                                                               \
        printf("\n");                                                   \
      }                                                                 \
      printf("\n");                                                     \
    }                                                                   \
  }

#define print_6D(Z, H, W, N, G, D, T) {                                 \
    for(int z = 0; z < (Z); ++z) {                                      \
      for (int y = 0; y < (H); ++y) {                                   \
        for (int x = 0; x < (W); ++x) {                                 \
          for (int n = 0; n < (N); ++n) {                               \
            for (int g = 0; g < (G); ++g) {                             \
              printf("%s(%d, %d, %d, %d, %d, :) =", #T, z, y, x, n, g); \
              for (int d = 0; d < (D); ++d) {                           \
                printf(" %f", T[z * (H) * (W) * (N) * (G) * (D) +       \
                                y * (W) * (N) * (G) * (D) +             \
                                x * (N) * (G) * (D) +                   \
                                n * (D) * (G) +                         \
                                g * (D) + d]);                          \
              }                                                         \
              printf("\n");                                             \
            }                                                           \
            printf("\n");                                               \
          }                                                             \
          printf("\n");                                                 \
        }                                                               \
        printf("\n");                                                   \
      }                                                                 \
      printf("\n");                                                     \
    }                                                                   \
  }

#define DEFINE_WRAPPERS(DEVICE, TYPE)                                   \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(             \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      TYPE* output, TYPE* workspace) {                                  \
    fw_training< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(         \
        H, W, N, K, D, input, shape, param, output, workspace);         \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(              \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      TYPE* output, TYPE* workspace) {                                  \
    fw_training< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(         \
        H, W, N, K, D, input, shape, param, output, workspace);         \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_workspace(             \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const int* shape, const TYPE* param,           \
      const TYPE* output, const TYPE* dOutput, TYPE* workspace) {       \
    bw_workspace< TYPE, Sigmoid<TYPE>, Tanh<TYPE>, Tanh<TYPE> >(        \
        H, W, N, K, D, input, shape, param, output, dOutput,            \
        workspace);                                                     \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_input(                 \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* param, const TYPE scale, TYPE* dInput,                \
      TYPE* workspace) {                                                \
    bw_input< TYPE >(H, W, N, K, D, param, scale, dInput, workspace);   \
  }                                                                     \
                                                                        \
  void rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _bw_param(                 \
      const int H, const int W, const int N, const int K, const int D,  \
      const TYPE* input, const TYPE* output, const TYPE scale,          \
      TYPE* dParam, TYPE* workspace) {                                  \
    bw_param< TYPE >(H, W, N, K, D, input, output, scale, dParam,       \
                     workspace);                                        \
  }

#endif  // RNN2D_LSTM_COMMON_H_

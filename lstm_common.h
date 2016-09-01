#ifndef RNN2D_LSTM_COMMON_H_
#define RNN2D_LSTM_COMMON_H_

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
  (dQ + (((((z) * H + (y)) * W + (x)) * N + (n)) * 6 + (g)) * D + (d))

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

#endif  // RNN2D_LSTM_COMMON_H_

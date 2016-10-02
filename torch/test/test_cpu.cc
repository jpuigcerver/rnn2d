#include <iostream>
#include "../../lstm_cpu.h"

static const int H = 2, W = 3, N = 2, K = 3, D = 2;
static const int S[N * 2] = {2, 3, 2, 3};

template <typename T>
const std::vector<T>& I() {
  static const std::vector<T> I_{
    0.30, 0.68, 0.29, 0.10, 0.70, 0.88, 0.13, 0.18, 0.35, 0.86, 0.66, 0.75,
    0.53, 0.40, 0.48, 0.20, 0.58, 0.66, 0.30, 0.99, 0.64, 0.46, 0.44, 0.65,
    0.82, 0.59, 0.47, 0.18, 0.53, 0.13, 0.68, 0.79, 0.80, 0.32, 0.09, 0.40
  };
  return I_;
}

template <typename T>
const std::vector<T>& P() {
  // Parameters (note, the 4 directions share the same weights)
  static const std::vector<T> P_{
    // TOP-LEFT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // TOP-RIGHT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // BOTTOM-LEFT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // BOTTOM-RIGHT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
  };
  return P_;
}

template <typename T>
const std::vector<T>& dO() {
  static const std::vector<T> dO_{
    0.51,  0.10,  0.21,  0.47, -0.06,  0.26,  0.50, -0.71,
    0.53,  0.65,  0.52,  0.25, -0.39, -0.13,  0.05,  0.07,
    0.44,  0.66,  0.30,  0.98,  0.20,  0.76, -0.93,  0.42,
    0.17,  0.71,  0.16, -0.48,  0.39,  0.92,  0.04,  0.81,
    0.07,  0.98, -0.17,  0.79,  0.57,  0.39,  0.94,  0.40,
    0.81,  0.40,  0.81,  0.34,  0.74,  0.49,  0.68,  0.00,
    0.29,  0.29,  0.50,  0.52, -0.15, -0.63, -0.87,  0.43,
    0.39,  0.59, -0.68,  0.92,  0.43, -0.16, -0.27,  0.19,
   -0.84,  0.13,  0.33,  0.89, -0.47,  0.72, -0.47,  0.27,
    0.85, -0.23,  0.15, -0.61,  0.69,  0.76,  0.47,  0.56,
    0.13,  0.61,  0.71,  0.11, -0.44,  0.11,  0.47,  0.04,
   -0.34,  0.78,  0.80,  0.24,  0.40,  0.49, -0.93,  0.09
  };
  return dO_;
}

template <typename T>
void run() {
  // Allocate space used for the internal states
  std::vector<T> Q(4 * H * W * N * 6 * D);
  std::vector<T> dQ(4 * H * W * N * 6 * D);
  // Output
  std::vector<T> O(H * W * N * 4 * D);
  // Derivative w.r.t. input
  std::vector<T> dI(H * W * N * K);
  // Derivative w.r.t. parameters
  std::vector<T> dP(P<T>().size());
  // Forward pass
  rnn2d_lstm_fw_cpu< T, Sigmoid<T>, Tanh<T>, Tanh<T> >(
      H, W, N, K, D, I<T>().data(), S, P<T>().data(), O.data(), Q.data());
  // Backward pass
  rnn2d_lstm_bw_cpu< T, Sigmoid<T>, Tanh<T>, Tanh<T> >(
      H, W, N, K, D, I<T>().data(), S, P<T>().data(), O.data(), Q.data(),
      dO<T>().data(), dQ.data());
  // Get gradInput
  rnn2d_lstm_bw_input_cpu<T>(
      H, W, N, K, D, P<T>().data(), dQ.data(), static_cast<T>(1.0), dI.data());
  // Get gradParams
  rnn2d_lstm_bw_params_cpu<T>(
      H, W, N, K, D, I<T>().data(), O.data(), dQ.data(), static_cast<T>(1.0),
      dP.data());
  const T sum_I = std::accumulate(I<T>().begin(), I<T>().end(), 0.0);
  const T sum_P = std::accumulate(P<T>().begin(), P<T>().end(), 0.0);
  const T sum_dO = std::accumulate(dO<T>().begin(), dO<T>().end(), 0.0);
  const T sum_Q  = std::accumulate(Q.begin(), Q.end(), 0.0);
  const T sum_O  = std::accumulate(O.begin(), O.end(), 0.0);
  const T sum_dI = std::accumulate(dI.begin(), dI.end(), 0.0);
  const T sum_dP = std::accumulate(dP.begin(), dP.end(), 0.0);
  const T sum_dQ = std::accumulate(dQ.begin(), dQ.end(), 0.0);
  std::cout.precision(18);
  std::cout << "sum_I  = " << sum_I << std::endl;
  std::cout << "sum_P  = " << sum_P << std::endl;
  std::cout << "sum_dO = " << sum_dO << std::endl;
  std::cout << "sum_Q  = " << sum_Q << std::endl;
  std::cout << "sum_O  = " << sum_O << std::endl;
  std::cout << "sum_dQ = " << sum_dQ << std::endl;
  std::cout << "sum_dI = " << sum_dI << std::endl;
  std::cout << "sum_dP = " << sum_dP << std::endl;
}

int main() {
  run<float>();
  return 0;
}

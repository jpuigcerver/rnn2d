#ifndef RNN2D_LSTM_COMMON_H_
#define RNN2D_LSTM_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

inline int rnn2d_lstm_input_nelem(
    const int H, const int W, const int N, const int K) {
  return H * W * N * K;
}

inline int rnn2d_lstm_output_nelem(
    const int H, const int W, const int N, const int D) {
  return H * W * N * 4 * D;
}

inline int rnn2d_lstm_parameters_nelem(const int K, const int D) {
  return 4 * (1 + (K) + (D) + (D)) * 5 * (D);
}

#ifdef __cplusplus
}
#endif

#endif  // RNN2D_LSTM_COMMON_H_

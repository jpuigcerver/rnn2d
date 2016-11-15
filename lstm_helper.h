#ifndef RNN2D_LSTM_HELPER_H_
#define RNN2D_LSTM_HELPER_H_

// Expected size for the LSTM2D input buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_INPUT_SIZE(H, W, N, K) ((H) * (W) * (N) * (K))

// Expected size for the LSTM2D output buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) ((H) * (W) * (N) * (D) * 4)

// Expected size for the LSTM2D parameters buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_PARAMETERS_SIZE(K, D) (4 * (1 + (K) + (D) + (D)) * 5 * (D))

// Expected size for the LSTM2D workspace buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D) (4 * (H) * (W) * (N) * 6 * (D))

#endif  // RNN2D_LSTM_HELPER_H_

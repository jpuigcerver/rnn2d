#ifndef RNN2D_COMMON_H_
#define RNN2D_COMMON_H_

#define DIV_UP(x, y) ((x) == 0 ? 0 : 1 + ((x) - 1) / (y))

// Expected size for the LSTM2D input buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_INPUT_SIZE(H, W, N, K) ((H) * (W) * (N) * (K))

// Expected size for the LSTM2D output buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) ((H) * (W) * (N) * (D) * 4)

// Expected size for the LSTM2D parameters buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_PARAMETERS_SIZE(K, D) (4 * (1 + (K) + (D) + (D)) * 5 * (D))

// Expected size for the LSTM2D workspace buffer, during inference.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_WORKSPACE_INFERENCE_SIZE(H, W, N, D) \
  (4 * (H) * (W) * (N) * 6 * (D))

// Expected size for the LSTM2D workspace buffer, during training.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D)  \
  (2 * 4 * (H) * (W) * (N) * 6 * (D))

// Expected size for the TILE2D input buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_TILE_INPUT_SIZE(H, W, N, D)  ((H) * (W) * (N) * (D))

// Expected size for the TILE2D output buffer.
// IMPORTANT: This is the number of elements, not bytes!
#define RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, KH, KW)              \
  (DIV_UP(H, KH) * DIV_UP(W, KW) * (N) * (KH) * (KW) * (D))

#endif  // RNN2D_COMMON_H_

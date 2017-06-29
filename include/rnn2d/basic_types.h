#ifndef RNN2D_BASIC_TYPES_H_
#define RNN2D_BASIC_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

#define RNN2D_DEVICE_CPU   -1

typedef enum {
  RNN2D_STATUS_SUCCESS         = 0,
  RNN2D_STATUS_NOT_INITIALIZED = 1,
  RNN2D_STATUS_BAD_PARAM       = 2,
  RNN2D_STATUS_NOT_SUPPORTED   = 3,
  RNN2D_STATUS_WRONG_STATE     = 4,
} rnn2dStatus_t;

typedef enum {
  RNN2D_LSTM_CELL_STANDARD  = 0,
  RNN2D_LSTM_CELL_HALVED    = 1,
  RNN2D_LSTM_CELL_STABLE    = 2,
  RNN2D_LSTM_CELL_LEAKYLP   = 3
} rnn2dLstmCell_t;

#ifdef __cplusplus
}
#endif

#endif  // RNN2D_BASIC_TYPES_H_

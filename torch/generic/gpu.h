#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/gpu.h"
#else

TORCH_API int THCTensor_(rnn2d_lstm_fw)(lua_State* L);
TORCH_API int THCTensor_(rnn2d_lstm_bw_workspace)(lua_State* L);
TORCH_API int THCTensor_(rnn2d_lstm_bw_input)(lua_State* L);
TORCH_API int THCTensor_(rnn2d_lstm_bw_params)(lua_State* L);

#endif  // THC_GENERIC_FILE

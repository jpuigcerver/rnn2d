#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.h"
#else

TORCH_API int THTensor_(rnn2d_lstm_fw)(lua_State* L);
TORCH_API int THTensor_(rnn2d_lstm_bw_workspace)(lua_State* L);
TORCH_API int THTensor_(rnn2d_lstm_bw_input)(lua_State* L);
TORCH_API int THTensor_(rnn2d_lstm_bw_params)(lua_State* L);

#endif  // TH_GENERIC_FILE

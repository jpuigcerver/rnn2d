#include "binding.h"

#include "generic/cpu.cc"
#include "THGenerateFloatTypes.h"

#ifdef USE_CUDA
#include "generic/gpu.cc"
#include "THCGenerateFloatTypes.h"
#endif  // USE_CUDA

#define REGISTER_TYPE_FUNCTIONS(TYPE)                                   \
  lua_register(L,                                                       \
               #TYPE"_rnn2d_lstm_fw",                                   \
               TYPE##_rnn2d_lstm_fw);                                   \
  lua_register(L,                                                       \
               #TYPE"_rnn2d_lstm_bw_workspace",                         \
               TYPE##_rnn2d_lstm_bw_workspace);                         \
  lua_register(L,                                                       \
               #TYPE"_rnn2d_lstm_bw_input",                             \
               TYPE##_rnn2d_lstm_bw_input);                             \
  lua_register(L,                                                       \
               #TYPE"_rnn2d_lstm_bw_params",                            \
               TYPE##_rnn2d_lstm_bw_params)

TORCH_API int luaopen_librnn2d_torch(lua_State* L) {
  REGISTER_TYPE_FUNCTIONS(THFloatTensor);
  REGISTER_TYPE_FUNCTIONS(THDoubleTensor);
#ifdef USE_CUDA
  REGISTER_TYPE_FUNCTIONS(THCudaTensor);
  REGISTER_TYPE_FUNCTIONS(THCudaDoubleTensor);
#endif  //USE_CUDA
  return 0;
}

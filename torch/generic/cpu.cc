#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.cc"
#else

#include <rnn2d/lstm_cpu.h>

#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#define RNN2D_LSTM_FW_TRAINING                          \
  TH_CONCAT_3(rnn2d_lstm_cpu_, real, _fw_training)
#define RNN2D_LSTM_BW_WORKSPACE                         \
  TH_CONCAT_3(rnn2d_lstm_cpu_, real, _bw_workspace)
#define RNN2D_LSTM_BW_INPUT                     \
  TH_CONCAT_3(rnn2d_lstm_cpu_, real, _bw_input)
#define RNN2D_LSTM_BW_PARAM                     \
  TH_CONCAT_3(rnn2d_lstm_cpu_, real, _bw_param)

TORCH_API int THTensor_(rnn2d_lstm_fw_training)(lua_State* L) {
  const THIntTensor* shape = static_cast<const THIntTensor*>(
      luaT_checkudata(L, 1, "torch.IntTensor"));
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  THTensor* out = static_cast<THTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  THTensor* workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 5, torch_Tensor));
  // Check input tensor
  CHECK_TENSOR_N_DIMENSIONS("input", 4, inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("input", inp);
  const int H = inp->size[0];
  const int W = inp->size[1];
  const int N = inp->size[2];
  const int K = inp->size[3];
  // Check output tensor
  CHECK_TENSOR_N_DIMENSIONS("output", 4, out->nDimension);
  const int D = out->size[3] / 4;
  CHECK_TENSOR_SIZE_4("output", out, H, W, N, 4 * D);
  // Check whether explicit shape for each sample where given
  const bool explicit_shape = THIntTensor_nElement(shape) > 0;
  if (explicit_shape) { CHECK_TENSOR_SIZE_2("shapes", shape, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THTensor_(nElement)(param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THTensor_(nElement)(workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  if (explicit_shape) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THIntTensor_isContiguous(shape));
  }
  // Run forward pass
  RNN2D_LSTM_FW_TRAINING(
      H, W, N, K, D,
      THTensor_(data)(inp),
      explicit_shape ? THIntTensor_data(shape) : nullptr,
      THTensor_(data)(param),
      THTensor_(data)(out),
      THTensor_(data)(workspace));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_workspace)(lua_State* L){
  const THIntTensor* shape = static_cast<const THIntTensor*>(
      luaT_checkudata(L, 1, "torch.IntTensor"));
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  const THTensor* out = static_cast<const THTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  const THTensor* d_out = static_cast<const THTensor*>(
      luaT_checkudata(L, 5, torch_Tensor));
  THTensor* workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 6, torch_Tensor));
  // Check input tensor
  CHECK_TENSOR_N_DIMENSIONS("input", 4, inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("input", inp);
  const int H = inp->size[0];
  const int W = inp->size[1];
  const int N = inp->size[2];
  const int K = inp->size[3];
  // Check output tensor
  CHECK_TENSOR_N_DIMENSIONS("output", 4, out->nDimension);
  const int D = out->size[3] / 4;
  CHECK_TENSOR_SIZE_4("output", out, H, W, N, 4 * D);
  // Check whether explicit dimensions for each sample where given
  const bool explicit_shape = THIntTensor_nElement(shape) > 0;
  if (explicit_shape) { CHECK_TENSOR_SIZE_2("shapes", shape, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THTensor_(nElement)(param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace",
      RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THTensor_(nElement)(workspace));
  // Check gradOutput tensor
  CHECK_TENSOR_SIZE_4("gradOutput", d_out, H, W, N, 4 * D);

  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  if (explicit_shape) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THIntTensor_isContiguous(shape));
  }
  CHECK_TENSOR_CONTIGUOUS("gradOutput", THTensor_(isContiguous)(d_out));
  // Run backward pass
  RNN2D_LSTM_BW_WORKSPACE(
      H, W, N, K, D,
      THTensor_(data)(inp),
      explicit_shape ? THIntTensor_data(shape) : nullptr,
      THTensor_(data)(param),
      THTensor_(data)(out),
      THTensor_(data)(d_out),
      THTensor_(data)(workspace));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_input)(lua_State* L){
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  const real scale = luaL_checknumber(L, 2);
  THTensor* d_inp = static_cast<THTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  THTensor* workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  // Check gradInput tensor
  CHECK_TENSOR_N_DIMENSIONS("gradInput", 4, d_inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("gradInput", d_inp);
  const int H = d_inp->size[0];
  const int W = d_inp->size[1];
  const int N = d_inp->size[2];
  const int K = d_inp->size[3];
  const int D = THTensor_(nElement)(workspace) / (2 * H * W * N * 4 * 6);
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THTensor_(nElement)(param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THTensor_(nElement)(workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("gradInput", THTensor_(isContiguous)(d_inp));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  // Run backward pass
  RNN2D_LSTM_BW_INPUT(
      H, W, N, K, D,
      THTensor_(data)(param),
      scale,
      THTensor_(data)(d_inp),
      THTensor_(data)(workspace));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_param)(lua_State* L){
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  const THTensor* out = static_cast<const THTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const real scale = luaL_checknumber(L, 3);
  THTensor* d_param = static_cast<THTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  THTensor* workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 5, torch_Tensor));
  // Check input tensor
  CHECK_TENSOR_N_DIMENSIONS("input", 4, inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("input", inp);
  const int H = inp->size[0];
  const int W = inp->size[1];
  const int N = inp->size[2];
  const int K = inp->size[3];
  // Check output tensor
  CHECK_TENSOR_N_DIMENSIONS("output", 4, out->nDimension);
  const int D = out->size[3] / 4;
  CHECK_TENSOR_SIZE_4("output", out, H, W, N, 4 * D);
  // Check gradParameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "gradParameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THTensor_(nElement)(d_param));
  // Check workspace
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THTensor_(nElement)(workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("gradParameters", THTensor_(isContiguous)(d_param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  // Run backward pass
  RNN2D_LSTM_BW_PARAM(
      H, W, N, K, D,
      THTensor_(data)(inp),
      THTensor_(data)(out),
      scale,
      THTensor_(data)(d_param),
      THTensor_(data)(workspace));
  return 0;
}

#undef torch_Tensor
#undef RNN2D_LSTM_FW_TRAINING
#undef RNN2D_LSTM_BW_WORKSPACE
#undef RNN2D_LSTM_BW_INPUT
#undef RNN2D_LSTM_BW_PARAM

#endif  // TH_GENERIC_FILE

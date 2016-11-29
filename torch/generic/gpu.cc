#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/gpu.cc"
#else

#include "THC/THCTensor.h"
#include "../../lstm_gpu.h"

#define torch_Tensor TH_CONCAT_STRING_3(torch.,CReal,Tensor)

#define RNN2D_LSTM_FW_TRAINING                          \
  TH_CONCAT_3(rnn2d_lstm_gpu_, real, _fw_training)
#define RNN2D_LSTM_BW_WORKSPACE                         \
  TH_CONCAT_3(rnn2d_lstm_gpu_, real, _bw_workspace)
#define RNN2D_LSTM_BW_INPUT                     \
  TH_CONCAT_3(rnn2d_lstm_gpu_, real, _bw_input)
#define RNN2D_LSTM_BW_PARAM                     \
  TH_CONCAT_3(rnn2d_lstm_gpu_, real, _bw_param)

TORCH_API int THCTensor_(rnn2d_lstm_fw_training)(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  const THCudaIntTensor* shape = static_cast<const THCudaIntTensor*>(
      luaT_checkudata(L, 1, "torch.CudaIntTensor"));
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  THCTensor* out = static_cast<THCTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  THCTensor* workspace = static_cast<THCTensor*>(
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
  const bool explicit_shape = THCudaIntTensor_nElement(state, shape) > 0;
  if (explicit_shape) { CHECK_TENSOR_SIZE_2("shapes", shape, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THCTensor_(nElement)(state, param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THCTensor_(nElement)(state, workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("workspace",
                          THCTensor_(isContiguous)(state, workspace));
  if (explicit_shape) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THCudaIntTensor_isContiguous(state, shape));
  }
  // Run forward pass
  RNN2D_LSTM_FW_TRAINING(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      explicit_shape ? THCudaIntTensor_data(state, shape) : nullptr,
      THCTensor_(data)(state, param),
      THCTensor_(data)(state, out),
      THCTensor_(data)(state, workspace));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_workspace)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCudaIntTensor* shape = static_cast<const THCudaIntTensor*>(
      luaT_checkudata(L, 1, "torch.CudaIntTensor"));
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  const THCTensor* out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  const THCTensor* d_out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 5, torch_Tensor));
  THCTensor* workspace = static_cast<THCTensor*>(
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
  const bool explicit_shape = THCudaIntTensor_nElement(state, shape) > 0;
  if (explicit_shape) { CHECK_TENSOR_SIZE_2("shapes", shape, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THCTensor_(nElement)(state, param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace",
      RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THCTensor_(nElement)(state, workspace));
  // Check gradOutput tensor
  CHECK_TENSOR_SIZE_4("gradOutput", d_out, H, W, N, 4 * D);

  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("workspace",
                          THCTensor_(isContiguous)(state, workspace));
  if (explicit_shape) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THCudaIntTensor_isContiguous(state, shape));
  }
  CHECK_TENSOR_CONTIGUOUS("gradOutput", THCTensor_(isContiguous)(state, d_out));
  // Run backward pass
  RNN2D_LSTM_BW_WORKSPACE(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      explicit_shape ? THCudaIntTensor_data(state, shape) : nullptr,
      THCTensor_(data)(state, param),
      THCTensor_(data)(state, out),
      THCTensor_(data)(state, d_out),
      THCTensor_(data)(state, workspace));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_input)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  const real scale = luaL_checknumber(L, 2);
  THCTensor* d_inp = static_cast<THCTensor*>(
      luaT_checkudata(L, 3, torch_Tensor));
  THCTensor* workspace = static_cast<THCTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  // Check gradInput tensor
  CHECK_TENSOR_N_DIMENSIONS("gradInput", 4, d_inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("gradInput", d_inp);
  const int H = d_inp->size[0];
  const int W = d_inp->size[1];
  const int N = d_inp->size[2];
  const int K = d_inp->size[3];
  const int D =
      THCTensor_(nElement)(state, workspace) / (2 * H * W * N * 4 * 6);
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", RNN2D_LSTM_PARAMETERS_SIZE(K, D),
      THCTensor_(nElement)(state, param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THCTensor_(nElement)(state, workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("gradInput", THCTensor_(isContiguous)(state, d_inp));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THCTensor_(isContiguous)(state, workspace));
  // Run backward pass
  RNN2D_LSTM_BW_INPUT(
      H, W, N, K, D,
      THCTensor_(data)(state, param),
      scale,
      THCTensor_(data)(state, d_inp),
      THCTensor_(data)(state, workspace));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_param)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, torch_Tensor));
  const THCTensor* out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 2, torch_Tensor));
  const real scale = luaL_checknumber(L, 3);
  THCTensor* d_param = static_cast<THCTensor*>(
      luaT_checkudata(L, 4, torch_Tensor));
  THCTensor* workspace = static_cast<THCTensor*>(
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
      THCTensor_(nElement)(state, d_param));
  // Check workspace
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D),
      THCTensor_(nElement)(state, workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("gradParameters", THCTensor_(isContiguous)(state, d_param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THCTensor_(isContiguous)(state, workspace));
  // Run backward pass
  RNN2D_LSTM_BW_PARAM(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      THCTensor_(data)(state, out),
      scale,
      THCTensor_(data)(state, d_param),
      THCTensor_(data)(state, workspace));
  return 0;
}

#undef torch_Tensor
#undef RNN2D_LSTM_FW_TRAINING
#undef RNN2D_LSTM_BW_WORKSPACE
#undef RNN2D_LSTM_BW_INPUT
#undef RNN2D_LSTM_BW_PARAM

#endif  // THC_GENERIC_FILE

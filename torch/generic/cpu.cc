#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.cc"
#else

#include <TH/THTensor.h>
#include "../../lstm_cpu.h"

TORCH_API int THTensor_(rnn2d_lstm_fw)(lua_State* L) {
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THTensor>()));
  const THIntTensor* dim = static_cast<const THIntTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THIntTensor>()));
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THTensor>()));
  THTensor* out = static_cast<THTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THTensor>()));
  THTensor* workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 5, getTorchClass<THTensor>()));
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
  const bool explicit_dims = THIntTensor_nElement(dim) > 0;
  if (explicit_dims) { CHECK_TENSOR_SIZE_2("shapes", dim, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS(
      "parameters", 4 * (1 + K + D + D) * 5 * D, THTensor_(nElement)(param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS(
      "workspace", 4 * H * W * N * 6 * D, THTensor_(nElement)(workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  if (explicit_dims) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THIntTensor_isContiguous(dim));
  }
  // Run forward pass
  rnn2d_lstm_fw_cpu< real, Sigmoid<real>, Tanh<real>, Tanh<real> >(
      H, W, N, K, D,
      THTensor_(data)(inp),
      explicit_dims ? THIntTensor_data(dim) : nullptr,
      THTensor_(data)(param),
      THTensor_(data)(out),
      THTensor_(data)(workspace));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_workspace)(lua_State* L){
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THTensor>()));
  const THIntTensor* dim = static_cast<const THIntTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THIntTensor>()));
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THTensor>()));
  const THTensor* out = static_cast<const THTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THTensor>()));
  const THTensor* workspace = static_cast<const THTensor*>(
      luaT_checkudata(L, 5, getTorchClass<THTensor>()));
  const THTensor* d_out = static_cast<const THTensor*>(
      luaT_checkudata(L, 6, getTorchClass<THTensor>()));
  THTensor* d_workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 7, getTorchClass<THTensor>()));
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
  const bool explicit_dims = THIntTensor_nElement(dim) > 0;
  if (explicit_dims) { CHECK_TENSOR_SIZE_2("shapes", dim, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS("parameters", 4 * (1 + K + D + D) * 5 * D,
                          THTensor_(nElement)(param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS("workspace", 4 * H * W * N * 6 * D,
                          THTensor_(nElement)(workspace));
  // Check gradOutput tensor
  CHECK_TENSOR_SIZE_4("gradOutput", d_out, H, W, N, 4 * D);
  // Check gradWorkspace tensor
  CHECK_TENSOR_N_ELEMENTS("gradWorkspace", 4 * H * W * N * 6 * D,
                          THTensor_(nElement)(d_workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THTensor_(isContiguous)(workspace));
  if (explicit_dims) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THIntTensor_isContiguous(dim));
  }
  CHECK_TENSOR_CONTIGUOUS("gradOutput", THTensor_(isContiguous)(d_out));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THTensor_(isContiguous)(d_workspace));
  printf("%.18f\n", std::accumulate(
      THTensor_(data)(d_workspace),
      THTensor_(data)(d_workspace) + THTensor_(nElement)(d_workspace), 0.0));
  // Run backward pass
  rnn2d_lstm_bw_cpu< real, Sigmoid<real>, Tanh<real>, Tanh<real> >(
      H, W, N, K, D,
      THTensor_(data)(inp),
      explicit_dims ? THIntTensor_data(dim) : nullptr,
      THTensor_(data)(param),
      THTensor_(data)(out),
      THTensor_(data)(workspace),
      THTensor_(data)(d_out),
      THTensor_(data)(d_workspace));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_input)(lua_State* L){
  const THTensor* param = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THTensor>()));
  const THTensor* d_workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THTensor>()));
  THTensor* d_inp = static_cast<THTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THTensor>()));
  const real scale = luaL_checknumber(L, 4);
  // Check gradInput tensor
  CHECK_TENSOR_N_DIMENSIONS("gradInput", 4, d_inp->nDimension);
  CHECK_TENSOR_NOT_EMPTY("gradInput", d_inp);
  const int H = d_inp->size[0];
  const int W = d_inp->size[1];
  const int N = d_inp->size[2];
  const int K = d_inp->size[3];
  // Check gradWorkspace
  CHECK_TENSOR_NOT_EMPTY("gradWorkspace", d_workspace);
  const int D = THTensor_(nElement)(d_workspace) / (H * W * N * 4 * 6);
  CHECK_TENSOR_N_ELEMENTS("gradWorkspace", 4 * H * W * N * D * 6,
                          THTensor_(nElement)(d_workspace));
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS("parameters", 4 * (1 + K + D + D) * 5 * D,
                          THTensor_(nElement)(param));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("gradInput", THTensor_(isContiguous)(d_inp));
  CHECK_TENSOR_CONTIGUOUS("parameters", THTensor_(isContiguous)(param));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THTensor_(isContiguous)(d_workspace));
  // Run backward pass
  rnn2d_lstm_bw_input_cpu<real>(
      H, W, N, K, D,
      THTensor_(data)(param),
      THTensor_(data)(d_workspace),
      scale, THTensor_(data)(d_inp));
  return 0;
}

TORCH_API int THTensor_(rnn2d_lstm_bw_params)(lua_State* L){
  const THTensor* inp = static_cast<const THTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THTensor>()));
  const THTensor* out = static_cast<const THTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THTensor>()));
  const THTensor* d_workspace = static_cast<THTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THTensor>()));
  THTensor* d_param = static_cast<THTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THTensor>()));
  const real scale = luaL_checknumber(L, 5);
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
  // Check gradWorkspace
  CHECK_TENSOR_N_ELEMENTS("gradWorkspace", 4 * H * W * N * D * 6,
                          THTensor_(nElement)(d_workspace));
  // Check gradParameters tensor
  CHECK_TENSOR_N_ELEMENTS("gradParameters", 4 * (1 + K + D + D) * 5 * D,
                          THTensor_(nElement)(d_param));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THTensor_(isContiguous)(inp));
  CHECK_TENSOR_CONTIGUOUS("output", THTensor_(isContiguous)(out));
  CHECK_TENSOR_CONTIGUOUS("gradParameters", THTensor_(isContiguous)(d_param));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THTensor_(isContiguous)(d_workspace));
  // Run backward pass
  rnn2d_lstm_bw_params_cpu< real >(
      H, W, N, K, D,
      THTensor_(data)(inp),
      THTensor_(data)(out),
      THTensor_(data)(d_workspace),
      scale, THTensor_(data)(d_param));
  return 0;
}

#endif  // TH_GENERIC_FILE

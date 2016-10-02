#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/gpu.cc"
#else

#include <THC/THCTensor.h>
#include "../../lstm_gpu.h"

TORCH_API int THCTensor_(rnn2d_lstm_fw)(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THCTensor>()));
  const THCudaIntTensor* dim = static_cast<const THCudaIntTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THCudaIntTensor>()));
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THCTensor>()));
  THCTensor* out = static_cast<THCTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THCTensor>()));
  THCTensor* workspace = static_cast<THCTensor*>(
      luaT_checkudata(L, 5, getTorchClass<THCTensor>()));
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
  const bool explicit_dims = THCudaIntTensor_nElement(state, dim) > 0;
  if (explicit_dims) { CHECK_TENSOR_SIZE_2("shapes", dim, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS("parameters", 4 * (1 + K + D + D) * 5 * D,
                          THCTensor_(nElement)(state, param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS("workspace", 4 * H * W * N * 6 * D,
                          THCTensor_(nElement)(state, workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THCTensor_(isContiguous)(state, workspace));
  if (explicit_dims) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THCudaIntTensor_isContiguous(state, dim));
  }
  // Run forward pass
  rnn2d_lstm_fw_cpu< real, Sigmoid<real>, Tanh<real>, Tanh<real> >(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      explicit_dims ? THCudaIntTensor_data(state, dim) : nullptr,
      THCTensor_(data)(state, param),
      THCTensor_(data)(state, out),
      THCTensor_(data)(state, workspace));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_workspace)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THCTensor>()));
  const THCudaIntTensor* dim = static_cast<const THCudaIntTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THCudaIntTensor>()));
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THCTensor>()));
  const THCTensor* out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THCTensor>()));
  const THCTensor* workspace = static_cast<const THCTensor*>(
      luaT_checkudata(L, 5, getTorchClass<THCTensor>()));
  const THCTensor* d_out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 6, getTorchClass<THCTensor>()));
  THCTensor* d_workspace = static_cast<THCTensor*>(
      luaT_checkudata(L, 7, getTorchClass<THCTensor>()));
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
  const bool explicit_dims = THCudaIntTensor_nElement(state, dim) > 0;
  if (explicit_dims) { CHECK_TENSOR_SIZE_2("shapes", dim, N, 2); }
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS("parameters", 4 * (1 + K + D + D) * 5 * D,
                          THCTensor_(nElement)(state, param));
  // Check workspace tensor
  CHECK_TENSOR_N_ELEMENTS("workspace", 4 * H * W * N * 6 * D,
                          THCTensor_(nElement)(state, workspace));
  // Check gradOutput tensor
  CHECK_TENSOR_SIZE_4("gradOutput", d_out, H, W, N, 4 * D);
  // Check gradWorkspace tensor
  CHECK_TENSOR_N_ELEMENTS("gradWorkspace", 4 * H * W * N * 6 * D,
                          THCTensor_(nElement)(state, d_workspace));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("workspace", THCTensor_(isContiguous)(state, workspace));
  if (explicit_dims) {
    CHECK_TENSOR_CONTIGUOUS("shapes", THCudaIntTensor_isContiguous(state, dim));
  }
  CHECK_TENSOR_CONTIGUOUS("gradOutput", THCTensor_(isContiguous)(state, d_out));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THCTensor_(isContiguous)(state, d_workspace));
  // Run backward pass
  rnn2d_lstm_bw_cpu< real, Sigmoid<real>, Tanh<real>, Tanh<real> >(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      explicit_dims ? THCudaIntTensor_data(state, dim) : nullptr,
      THCTensor_(data)(state, param),
      THCTensor_(data)(state, out),
      THCTensor_(data)(state, workspace),
      THCTensor_(data)(state, d_out),
      THCTensor_(data)(state, d_workspace));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_input)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCTensor* param = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THCTensor>()));
  const THCTensor* d_workspace = static_cast<THCTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THCTensor>()));
  THCTensor* d_inp = static_cast<THCTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THCTensor>()));
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
  const int D = THCTensor_(nElement)(state, d_workspace) / (H * W * N * 4 * 6);
  CHECK_TENSOR_N_ELEMENTS("gradWorkspace", 4 * H * W * N * D * 6,
                          THCTensor_(nElement)(state, d_workspace));
  // Check parameters tensor
  CHECK_TENSOR_N_ELEMENTS("parameters", 4 * (1 + K + D + D) * 5 * D,
                          THCTensor_(nElement)(state, param));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("gradInput", THCTensor_(isContiguous)(state, d_inp));
  CHECK_TENSOR_CONTIGUOUS("parameters", THCTensor_(isContiguous)(state, param));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THCTensor_(isContiguous)(state, d_workspace));
  // Run backward pass
  rnn2d_lstm_bw_input_cpu<real>(
      H, W, N, K, D,
      THCTensor_(data)(state, param),
      THCTensor_(data)(state, d_workspace),
      scale, THCTensor_(data)(state, d_inp));
  return 0;
}

TORCH_API int THCTensor_(rnn2d_lstm_bw_params)(lua_State* L){
  THCState* state = cutorch_getstate(L);
  const THCTensor* inp = static_cast<const THCTensor*>(
      luaT_checkudata(L, 1, getTorchClass<THCTensor>()));
  const THCTensor* out = static_cast<const THCTensor*>(
      luaT_checkudata(L, 2, getTorchClass<THCTensor>()));
  const THCTensor* d_workspace = static_cast<THCTensor*>(
      luaT_checkudata(L, 3, getTorchClass<THCTensor>()));
  THCTensor* d_param = static_cast<THCTensor*>(
      luaT_checkudata(L, 4, getTorchClass<THCTensor>()));
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
                          THCTensor_(nElement)(state, d_workspace));
  // Check gradParameters tensor
  CHECK_TENSOR_N_ELEMENTS("gradParameters", 4 * (1 + K + D + D) * 5 * D,
                          THCTensor_(nElement)(state, d_param));
  // Check that all tensors are contiguous
  CHECK_TENSOR_CONTIGUOUS("input", THCTensor_(isContiguous)(state, inp));
  CHECK_TENSOR_CONTIGUOUS("output", THCTensor_(isContiguous)(state, out));
  CHECK_TENSOR_CONTIGUOUS("gradParameters", THCTensor_(isContiguous)(state, d_param));
  CHECK_TENSOR_CONTIGUOUS("gradWorkspace",
                          THCTensor_(isContiguous)(state, d_workspace));
  // Run backward pass
  rnn2d_lstm_bw_params_cpu< real >(
      H, W, N, K, D,
      THCTensor_(data)(state, inp),
      THCTensor_(data)(state, out),
      THCTensor_(data)(state, d_workspace),
      scale, THCTensor_(data)(state, d_param));
  return 0;
}

#endif  // TH_GENERIC_FILE

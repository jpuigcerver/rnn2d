#ifndef TEST_LSTM_TEST_GPU_H_
#define TEST_LSTM_TEST_GPU_H_

#include <rnn2d/lstm_gpu.h>

#include "LstmTest.h"
#include "LstmAllocatorGPU.h"

template <>
LstmTest< TypeDefinitions<float, device::GPU> >::LstmTest()
    : allocator_(new LstmAllocatorGPU<float>(H, W, N, K, D)) { }

template <>
LstmTest< TypeDefinitions<double, device::GPU> >::LstmTest()
    : allocator_(new LstmAllocatorGPU<double>(H, W, N, K, D)) { }

template <>
void LstmTest< TypeDefinitions<float, device::GPU> >::DoForwardInference() {
  rnn2d_lstm_gpu_float_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<double, device::GPU> >::DoForwardInference() {
  rnn2d_lstm_gpu_double_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<float, device::GPU> >::DoForwardTraining() {
  rnn2d_lstm_gpu_float_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::GPU> >::DoForwardTraining() {
  rnn2d_lstm_gpu_double_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<float, device::GPU> >::DoBackwardData() {
  rnn2d_lstm_gpu_float_bw_data(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      GradOutput(), GradInput(), Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::GPU> >::DoBackwardData() {
  rnn2d_lstm_gpu_double_bw_data(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      GradOutput(), GradInput(), Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<float, device::GPU> >::DoBackwardInput() {
  rnn2d_lstm_gpu_float_bw_param(
      H, W, N, K, D, Input(), Output(), 1.0, GradParameters(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::GPU> >::DoBackwardInput() {
  rnn2d_lstm_gpu_double_bw_param(
      H, W, N, K, D, Input(), Output(), 1.0, GradParameters(),
      Workspace(), Reserved());
}

#endif  // TEST_LSTM_TEST_GPU_H_

#ifndef TEST_LSTM_TEST_CPU_H_
#define TEST_LSTM_TEST_CPU_H_

#include <rnn2d/lstm_cpu.h>

#include "LstmTest.h"
#include "LstmAllocatorCPU.h"

template <>
LstmTest< TypeDefinitions<float, device::CPU> >::LstmTest()
    : allocator_(new LstmAllocatorCPU<float>(H, W, N, K, D)) { }

template <>
LstmTest< TypeDefinitions<double, device::CPU> >::LstmTest()
    : allocator_(new LstmAllocatorCPU<double>(H, W, N, K, D)) { }

template <>
void LstmTest< TypeDefinitions<float, device::CPU> >::DoForwardInference() {
  rnn2d_lstm_cpu_float_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<double, device::CPU> >::DoForwardInference() {
  rnn2d_lstm_cpu_double_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<float, device::CPU> >::DoForwardTraining() {
  rnn2d_lstm_cpu_float_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::CPU> >::DoForwardTraining() {
  rnn2d_lstm_cpu_double_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<float, device::CPU> >::DoBackwardData() {
  rnn2d_lstm_cpu_float_bw_data(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      GradOutput(), GradInput(), Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::CPU> >::DoBackwardData() {
  rnn2d_lstm_cpu_double_bw_data(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      GradOutput(), GradInput(), Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<float, device::CPU> >::DoBackwardInput() {
  rnn2d_lstm_cpu_float_bw_param(
      H, W, N, K, D, Input(), Output(), 1.0, GradParameters(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, device::CPU> >::DoBackwardInput() {
  rnn2d_lstm_cpu_double_bw_param(
      H, W, N, K, D, Input(), Output(), 1.0, GradParameters(),
      Workspace(), Reserved());
}

#endif  // TEST_LSTM_TEST_CPU_H_

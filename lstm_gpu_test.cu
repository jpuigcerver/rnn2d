#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thrust/device_vector.h>

#include "lstm_gpu.h"
#include "lstm_common_test.h"

#define DEFINE_GPU_TESTS(TYPE)                                          \
  TEST(lstm_test, rnn2d_lstm_gpu_ ## TYPE  ## _fw) {                    \
    using thrust::device_vector;                                        \
    const device_vector<int>  S_gpu(S);                                 \
    const device_vector<TYPE> I_gpu(I<TYPE>());                         \
    const device_vector<TYPE> P_gpu(P<TYPE>());                         \
    device_vector<TYPE> O_gpu(RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D));      \
    device_vector<TYPE> Q_gpu(RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D));   \
    rnn2d_lstm_gpu_ ## TYPE ## _fw_inference(                           \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), Q_gpu.data().get());    \
    const TYPE sum_O = thrust::reduce(                                  \
        O_gpu.begin(), O_gpu.end(), static_cast<TYPE>(0));              \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_gpu_ ## TYPE ## _bw) {                     \
    using thrust::device_vector;                                        \
    const device_vector<int>  S_gpu(S);                                 \
    const device_vector<TYPE> I_gpu(I<TYPE>());                         \
    const device_vector<TYPE> P_gpu(P<TYPE>());                         \
    const device_vector<TYPE> dO_gpu(dO<TYPE>());                       \
    device_vector<TYPE> O_gpu(RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D));      \
    device_vector<TYPE> Q_gpu(RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D));   \
    device_vector<TYPE> dQ_gpu(RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D));  \
    device_vector<TYPE> dI_gpu(RNN2D_LSTM_INPUT_SIZE(H, W, N, K));      \
    device_vector<TYPE> dP_gpu(RNN2D_LSTM_PARAMETERS_SIZE(K, D));       \
    /* First, forward pass in training mode. */                         \
    rnn2d_lstm_gpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), Q_gpu.data().get());    \
    /* Derivative w.r.t. the workspace. */                              \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_workspace(                           \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), Q_gpu.data().get(),     \
        dO_gpu.data().get(), dQ_gpu.data().get());                      \
    /* Derivative w.r.t. the input. */                                  \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_input(                               \
        H, W, N, K, D, P_gpu.data().get(), dQ_gpu.data().get(), 1.0,    \
        dI_gpu.data().get());                                           \
    /* Derivative w.r.t. the parameters. */                             \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_param(                               \
        H, W, N, K, D, I_gpu.data().get(), O_gpu.data().get(),          \
        dQ_gpu.data().get(), 1.0, dP_gpu.data().get());                 \
    const TYPE sum_dI = thrust::reduce(                                 \
        dI_gpu.begin(), dI_gpu.end(), static_cast<TYPE>(0));            \
    EXPECT_NEAR(expected_sum_dI<TYPE>(), sum_dI, MAX_ERROR);            \
    const TYPE sum_dP = thrust::reduce(                                 \
        dP_gpu.begin(), dP_gpu.end(), static_cast<TYPE>(0));            \
    EXPECT_NEAR(expected_sum_dP<TYPE>(), sum_dP, MAX_ERROR);            \
  }

DEFINE_GPU_TESTS(float)
DEFINE_GPU_TESTS(double)

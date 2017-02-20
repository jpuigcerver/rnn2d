#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <rnn2d/lstm_cpu.h>
#include "lstm_common_test.h"

#define DEFINE_CPU_TESTS(TYPE)                                          \
  TEST(lstm_test, rnn2d_lstm_cpu_ ## TYPE  ## _fw_inference) {          \
    std::vector<TYPE> O(H * W * N * D * 4);                             \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_cpu_##TYPE##_inference_workspace_size(H, W, N, D);   \
    std::vector<char> workspace(workspaceSize);                         \
    rnn2d_lstm_cpu_ ## TYPE ## _fw_inference(                           \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), workspace.data());                                    \
    const TYPE sum_O = std::accumulate(                                 \
        O.begin(), O.end(), static_cast<TYPE>(0));                      \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_cpu_ ## TYPE ## _fw_training) {            \
    std::vector<TYPE> O(H * W * N * D * 4);                             \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_cpu_##TYPE##_inference_workspace_size(H, W, N, D);   \
    const size_t reserveSize =                                          \
        rnn2d_lstm_cpu_##TYPE##_training_reserve_size(H, W, N, D);      \
    std::vector<char> workspace(workspaceSize);                         \
    std::vector<char> reserve(reserveSize);                             \
    rnn2d_lstm_cpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), workspace.data(), reserve.data());                    \
    const TYPE sum_O = std::accumulate(                                 \
        O.begin(), O.end(), static_cast<TYPE>(0));                      \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_cpu_ ## TYPE ## _bw) {                     \
    std::vector<TYPE> O(H * W * N * D * 4);                             \
    std::vector<TYPE> dI(I<TYPE>().size());                             \
    std::vector<TYPE> dP(P<TYPE>().size());                             \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_cpu_##TYPE##_inference_workspace_size(H, W, N, D);   \
    const size_t reserveSize =                                          \
        rnn2d_lstm_cpu_##TYPE##_training_reserve_size(H, W, N, D);      \
    std::vector<char> workspace(workspaceSize);                         \
    std::vector<char> reserve(reserveSize);                             \
    /* First, forward pass in training mode. */                         \
    rnn2d_lstm_cpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), workspace.data(), reserve.data());                    \
        /* Derivative w.r.t. the workspace. */                          \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_workspace(                           \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), dO<TYPE>().data(), workspace.data(), reserve.data()); \
    /* Derivative w.r.t. the input. */                                  \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_input(                               \
        H, W, N, K, D, P<TYPE>().data(), 1.0, dI.data(),                \
        workspace.data(), reserve.data());                              \
    /* Derivative w.r.t. the parameters. */                             \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_param(                               \
        H, W, N, K, D, I<TYPE>().data(), O.data(), 1.0, dP.data(),      \
        workspace.data(), reserve.data());                              \
    const TYPE sum_dI = std::accumulate(                                \
        dI.begin(), dI.end(), static_cast<TYPE>(0));                    \
    EXPECT_NEAR(expected_sum_dI<TYPE>(), sum_dI, MAX_ERROR);            \
    const TYPE sum_dP = std::accumulate(                                \
        dP.begin(), dP.end(), static_cast<TYPE>(0));                    \
    EXPECT_NEAR(expected_sum_dP<TYPE>(), sum_dP, MAX_ERROR);            \
  }

DEFINE_CPU_TESTS(float)
DEFINE_CPU_TESTS(double)

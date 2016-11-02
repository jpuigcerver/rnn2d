#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lstm_cpu.h"
#include "lstm_common_test.h"

#define MAX_ERROR 1E-5

#define DEFINE_CPU_TESTS(TYPE)                                          \
  TEST(lstm_test, rnn2d_lstm_cpu_ ## TYPE  ## _fw) {                    \
    std::vector<TYPE> O(H * W * N * 4 * D);                             \
    std::vector<TYPE> Q(4 * H * W * N * 6  * D);                        \
    rnn2d_lstm_cpu_ ## TYPE ## _fw_inference(                           \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), Q.data());                                            \
    const TYPE sum_O = std::accumulate(                                 \
        O.begin(), O.end(), static_cast<TYPE>(0));                      \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_cpu_ ## TYPE ## _bw) {                     \
    std::vector<TYPE> O(H * W * N * 4 * D);                             \
    std::vector<TYPE> Q(4 * H * W * N * 6 * D);                         \
    std::vector<TYPE> dQ(4 * H * W * N * 6 * D);                        \
    std::vector<TYPE> dI(H * W * N * K);                                \
    std::vector<TYPE> dP(4 * (1 + K + D + D) * 5 * D);                  \
    /* First, forward pass in training mode. */                         \
    rnn2d_lstm_cpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), Q.data());                                            \
    /* Derivative w.r.t. the workspace. */                              \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_workspace(                           \
        H, W, N, K, D, I<TYPE>().data(), S.data(), P<TYPE>().data(),    \
        O.data(), Q.data(), dO<TYPE>().data(), dQ.data());              \
    /* Derivative w.r.t. the input. */                                  \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_input(                               \
        H, W, N, K, D, P<TYPE>().data(), dQ.data(), 1.0, dI.data());    \
    /* Derivative w.r.t. the parameters. */                             \
    rnn2d_lstm_cpu_ ## TYPE ## _bw_param(                               \
        H, W, N, K, D, I<TYPE>().data(), O.data(), dQ.data(),           \
        1.0, dP.data());                                                \
    const TYPE sum_dI = std::accumulate(                                \
        dI.begin(), dI.end(), static_cast<TYPE>(0));                    \
    EXPECT_NEAR(expected_sum_dI<TYPE>(), sum_dI, MAX_ERROR);            \
    const TYPE sum_dP = std::accumulate(                                \
        dP.begin(), dP.end(), static_cast<TYPE>(0));                    \
    EXPECT_NEAR(expected_sum_dP<TYPE>(), sum_dP, MAX_ERROR);            \
  }

DEFINE_CPU_TESTS(float)
DEFINE_CPU_TESTS(double)

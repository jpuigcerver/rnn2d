#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <thrust/device_vector.h>

#include <rnn2d/lstm_gpu.h>
#include <rnn2d/cuda_utils.h>
#include "lstm_common_test.h"

using thrust::device_vector;

#define DEFINE_GPU_TESTS(TYPE)                                          \
  TEST(lstm_test, rnn2d_lstm_gpu_ ## TYPE  ## _fw_inference) {          \
    const device_vector<int>  S_gpu(S);                                 \
    const device_vector<TYPE> I_gpu(I<TYPE>());                         \
    const device_vector<TYPE> P_gpu(P<TYPE>());                         \
    device_vector<TYPE> O_gpu(H * W * N * D * 4);                       \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_gpu_ ## TYPE ## _inference_workspace_size(H, W, N, D); \
    void* workspace = nullptr;                                          \
    CHECK_CUDA_CALL(cudaMalloc(&workspace, workspaceSize));             \
    rnn2d_lstm_gpu_ ## TYPE ## _fw_inference(                           \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), workspace);             \
    const TYPE sum_O = thrust::reduce(                                  \
        O_gpu.begin(), O_gpu.end(), static_cast<TYPE>(0));              \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
    CHECK_CUDA_CALL(cudaFree(workspace));                               \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_gpu_ ## TYPE  ## _fw_training) {           \
    const device_vector<int>  S_gpu(S);                                 \
    const device_vector<TYPE> I_gpu(I<TYPE>());                         \
    const device_vector<TYPE> P_gpu(P<TYPE>());                         \
    device_vector<TYPE> O_gpu(H * W * N * D * 4);                       \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_gpu_ ## TYPE ## _training_workspace_size(H, W, N, D); \
    const size_t reserveSize =                                          \
        rnn2d_lstm_gpu_ ## TYPE ## _training_reserve_size(H, W, N, D);  \
    void* workspace = nullptr;                                          \
    CHECK_CUDA_CALL(cudaMalloc(&workspace, workspaceSize));             \
    void* reserve = nullptr;                                            \
    CHECK_CUDA_CALL(cudaMalloc(&reserve, reserveSize));                 \
    rnn2d_lstm_gpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), workspace, reserve);    \
    const TYPE sum_O = thrust::reduce(                                  \
        O_gpu.begin(), O_gpu.end(), static_cast<TYPE>(0));              \
    EXPECT_NEAR(expected_sum_O<TYPE>(), sum_O, MAX_ERROR);              \
    CHECK_CUDA_CALL(cudaFree(workspace));                               \
    CHECK_CUDA_CALL(cudaFree(reserve));                                 \
  }                                                                     \
  TEST(lstm_test, rnn2d_lstm_gpu_ ## TYPE ## _bw) {                     \
    const device_vector<int> S_gpu(S);                                  \
    const device_vector<TYPE> I_gpu(I<TYPE>());                         \
    const device_vector<TYPE> P_gpu(P<TYPE>());                         \
    const device_vector<TYPE> dO_gpu(dO<TYPE>());                       \
    device_vector<TYPE> O_gpu(dO_gpu.size());                           \
    device_vector<TYPE> dI_gpu(I_gpu.size());                           \
    device_vector<TYPE> dP_gpu(P_gpu.size());                           \
    const size_t workspaceSize =                                        \
        rnn2d_lstm_gpu_ ## TYPE ## _training_workspace_size(H, W, N, D); \
    const size_t reserveSize =                                          \
        rnn2d_lstm_gpu_ ## TYPE ## _training_reserve_size(H, W, N, D);  \
    void* workspace = nullptr;                                          \
    CHECK_CUDA_CALL(cudaMalloc(&workspace, workspaceSize));             \
    void* reserve = nullptr;                                            \
    CHECK_CUDA_CALL(cudaMalloc(&reserve, reserveSize));                 \
    /* First, forward pass in training mode. */                         \
    rnn2d_lstm_gpu_ ## TYPE ## _fw_training(                            \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), workspace, reserve);    \
    /* Derivative w.r.t. the workspace. */                              \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_workspace(                           \
        H, W, N, K, D, I_gpu.data().get(), S_gpu.data().get(),          \
        P_gpu.data().get(), O_gpu.data().get(), dO_gpu.data().get(),    \
        workspace, reserve);                                            \
    /* Derivative w.r.t. the input. */                                  \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_input(                               \
        H, W, N, K, D, P_gpu.data().get(), 1.0, dI_gpu.data().get(),    \
        workspace, reserve);                                            \
    /* Derivative w.r.t. the parameters. */                             \
    rnn2d_lstm_gpu_ ## TYPE ## _bw_param(                               \
        H, W, N, K, D, I_gpu.data().get(), O_gpu.data().get(),          \
        1.0, dP_gpu.data().get(), workspace, reserve);                  \
    const TYPE sum_dI = thrust::reduce(                                 \
        dI_gpu.begin(), dI_gpu.end(), static_cast<TYPE>(0));            \
    EXPECT_NEAR(expected_sum_dI<TYPE>(), sum_dI, MAX_ERROR);            \
    const TYPE sum_dP = thrust::reduce(                                 \
        dP_gpu.begin(), dP_gpu.end(), static_cast<TYPE>(0));            \
    EXPECT_NEAR(expected_sum_dP<TYPE>(), sum_dP, MAX_ERROR);            \
    CHECK_CUDA_CALL(cudaFree(workspace));                               \
    CHECK_CUDA_CALL(cudaFree(reserve));                                 \
  }

DEFINE_GPU_TESTS(float)
DEFINE_GPU_TESTS(double)

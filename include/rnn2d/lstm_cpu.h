#ifndef RNN2D_LSTM_CPU_H_
#define RNN2D_LSTM_CPU_H_

#include <rnn2d/lstm_common.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// float

size_t rnn2d_lstm_cpu_float_inference_workspace_size(
    const int H, const int W, const int N, const int D);

size_t rnn2d_lstm_cpu_float_training_workspace_size(
    const int H, const int W, const int N, const int D);

size_t rnn2d_lstm_cpu_float_training_reserve_size(
    const int H, const int W, const int N, const int D);

void rnn2d_lstm_cpu_float_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param, float* output,
    void* workspace);

void rnn2d_lstm_cpu_float_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param, float* output,
    void* workspace, void* reserve);

void rnn2d_lstm_cpu_float_bw_data(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    const float* output, const float* dOutput, float* dInput,
    void* workspace, void* reserve);

void rnn2d_lstm_cpu_float_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const float* output, const float scale,
    float* dParam, void* workspace, void* reserve);

// double

size_t rnn2d_lstm_cpu_double_inference_workspace_size(
    const int H, const int W, const int N, const int D);

size_t rnn2d_lstm_cpu_double_training_workspace_size(
    const int H, const int W, const int N, const int D);

size_t rnn2d_lstm_cpu_double_training_reserve_size(
    const int H, const int W, const int N, const int D);

void rnn2d_lstm_cpu_double_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param, double* output,
    void* workspace);

void rnn2d_lstm_cpu_double_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param, double* output,
    void* workspace, void* reserve);

void rnn2d_lstm_cpu_double_bw_data(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    const double* output, const double* dOutput, double* dInput,
    void* workspace, void* reserve);

void rnn2d_lstm_cpu_double_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const double* output, const double scale,
    double* dParam, void* workspace, void* reserve);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RNN2D_LSTM_CPU_H_

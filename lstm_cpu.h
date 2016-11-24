#ifndef RNN2D_LSTM_CPU_H_
#define RNN2D_LSTM_CPU_H_

#include "lstm_helper.h"

#ifdef __cplusplus
extern "C" {
#endif

// float

void rnn2d_lstm_cpu_float_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_cpu_float_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_cpu_float_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    const float* output, const float* dOutput, float* workspace);

void rnn2d_lstm_cpu_float_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const float* param, const float scale, float* dInput, float* workspace);

void rnn2d_lstm_cpu_float_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const float* output, const float scale,
    float* dParam, float* workspace);

// double

void rnn2d_lstm_cpu_double_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_cpu_double_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_cpu_double_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    const double* output, const double* dOutput, double* workspace);

void rnn2d_lstm_cpu_double_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const double* param, const double scale, double* dInput, double* workspace);

void rnn2d_lstm_cpu_double_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const double* output, const double scale,
    double* dParam, double* workspace);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif  // RNN2D_LSTM_CPU_H_

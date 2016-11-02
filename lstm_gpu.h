#ifndef RNN2D_LSTM_GPU_H_
#define RNN2D_LSTM_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

// float

void rnn2d_lstm_gpu_float_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_gpu_float_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_gpu_float_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    const float* output, const float* workspace, const float* dOutput,
    float* dWorkspace);

void rnn2d_lstm_gpu_float_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const float* param, const float* dWorkspace, const float scale,
    float* dInput);

void rnn2d_lstm_gpu_float_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const float* output, const float* dWorkspace,
    const float scale, float* dParam);

// double

void rnn2d_lstm_gpu_double_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_gpu_double_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_gpu_double_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    const double* output, const double* workspace, const double* dOutput,
    double* dWorkspace);

void rnn2d_lstm_gpu_double_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const double* param, const double* dWorkspace, const double scale,
    double* dInput);

void rnn2d_lstm_gpu_double_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const double* output, const double* dWorkspace,
    const double scale, double* dParam);

#ifdef __cplusplus
}  // extern "C"
#endif


#endif  // RNN2D_LSTM_GPU_H_

#ifndef RNN2D_LSTM_GPU_KERNELS_H_
#define RNN2D_LSTM_GPU_KERNELS_H_

template <typename T>
void fill(const int n, T* x, const T& v);

template <typename T>
void init_Q_with_bias(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, T* Q);

template <typename T>
void copy_dO_to_dC(
    const int H, const int W, const int N, const int D,
    const int t, const int Tn, const int Tmin, const T* dO, T* dQ);

template <typename T, typename FG, typename FI, typename FO>
void fw_elemwise_ops(
    const int H, const int W, const int N, const int D,
    const int t, const int Tn, const int Tmin, const int* S, T* Q, T* O);

template <typename T, typename FG, typename FI, typename FO>
void bw_elemwise_ops(
    const int H, const int W, const int N, const int D,
    const int t, const int Tn, const int Tmin, const int* S, const T* Q, T* dQ);

#endif  // RNN2D_LSTM_GPU_KERNELS_H_

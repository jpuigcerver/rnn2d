#ifndef BENCH_LSTM_COMMON_BENCHMARK_H_
#define BENCH_LSTM_COMMON_BENCHMARK_H_

#include <algorithm>
#include <random>
#include <vector>

#include <glog/logging.h>

#define DEFAULT_H 32    // default image height
#define DEFAULT_W 256   // default image width
#define DEFAULT_N 16    // default batch size
#define DEFAULT_D 16    // default image depth

static std::default_random_engine RNG;

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
using thrust::system::cuda::experimental::pinned_allocator;
static thrust::device_vector<float> DATA_gpu_float;
static thrust::device_vector<double> DATA_gpu_double;
static thrust::host_vector<float, pinned_allocator<float> > DATA_cpu_float;
static thrust::host_vector<double, pinned_allocator<double> > DATA_cpu_double;
#define VECTOR_DATA(v) (v).data().get()
#else
static std::vector<float> DATA_cpu_float;
static std::vector<double> DATA_cpu_double;
#define VECTOR_DATA(v) (v).data()
#endif

static void AllocateData(
    int H = DEFAULT_H, int W = DEFAULT_W, int N = DEFAULT_N,
    int K = DEFAULT_D, int D = DEFAULT_D) {
  const size_t ALL_DATA_SIZE =
      2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +
      2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +
      2 * RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D) +
      2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);
  DATA_cpu_float.resize(ALL_DATA_SIZE);
  DATA_cpu_double.resize(ALL_DATA_SIZE);
  std::normal_distribution<float>  ndist_f(0.0, 0.01);
  std::generate(DATA_cpu_float.begin(), DATA_cpu_float.end(),
                [&ndist_f]{ return ndist_f(RNG); });
  std::normal_distribution<double> ndist_d(0.0, 0.01);
  std::generate(DATA_cpu_double.begin(), DATA_cpu_double.end(),
                [&ndist_d]{ return ndist_d(RNG); });
#ifdef __CUDACC__
  DATA_gpu_float.assign(DATA_cpu_float.begin(), DATA_cpu_float.end());
  DATA_gpu_double.assign(DATA_cpu_double.begin(), DATA_cpu_double.end());
#endif
  LOG(INFO) << "Requested memory = "
            << (4 + 8) * (ALL_DATA_SIZE) / static_cast<float>(2 << 20) << "MB";
}

static void DeallocateData() {
  DATA_cpu_float.clear();
  DATA_cpu_double.clear();
  DATA_cpu_float.shrink_to_fit();
  DATA_cpu_double.shrink_to_fit();
#ifdef __CUDACC__
  DATA_gpu_float.clear();
  DATA_gpu_double.clear();
  // Make sure to free GPU memory.
  DATA_gpu_float.shrink_to_fit();
  DATA_gpu_double.shrink_to_fit();
#endif
}

#define DEFINE_BENCHMARK(DEVICE, TYPE)                                  \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K);       /* param offset */     \
    TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +          \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +      /* param offset */     \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* output offset */    \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +      /* param offset */     \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* output offset */    \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);      /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(              \
          H, W, N, K, D, input, nullptr, param, output, workspace);     \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K);        \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw)                     \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* param offset */      \
    const TYPE* gradOutput = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +      \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D);       /* gradOutput offset */ \
    TYPE* output = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +                \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +      /* gradOutput offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);     /* output offset */     \
    TYPE* workspace = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +             \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +      /* gradOutput offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* output offset */     \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);     /* workspace offset */  \
    TYPE* gradWorkspace = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +         \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +      /* gradOutput offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* output offset */     \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* workspace offset */  \
        RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D);  /* gradWork offset */   \
    TYPE* gradParam = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +             \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +      /* gradOutput offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* output offset */     \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* workspace offset */  \
        RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D) + /* gradWork offset */   \
        RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D);  /* gradParam offset */  \
    TYPE* gradInput = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +             \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* param offset */      \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D) +      /* gradOutput offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* output offset */     \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +    /* workspace offset */  \
        RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D) + /* gradWork offset */   \
        RNN2D_LSTM_WORKSPACE_SIZE(H, W, N, D) + /* gradParam offset */  \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D);      /* gradInput offset */   \
    while (state.KeepRunning()) {                                       \
    rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(                 \
    H, W, N, K, D, input, nullptr, param, output, workspace);           \
        rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_workspace(             \
    H, W, N, K, D, input, nullptr, param, output, workspace,            \
        gradOutput, gradWorkspace);                                     \
            rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_input(             \
    H, W, N, K, D, param, gradWorkspace, 1.0, gradInput);               \
                rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_param(         \
    H, W, N, K, D, input, output, gradWorkspace, 1.0, gradParam);       \
  }                                                                     \
    state.SetItemsProcessed(state.iterations() * H * W * N * K);        \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw)                     \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime()

#endif  // BENCH_LSTM_COMMON_BENCHMARK_H_

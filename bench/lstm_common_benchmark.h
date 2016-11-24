#ifndef BENCH_LSTM_COMMON_BENCHMARK_H_
#define BENCH_LSTM_COMMON_BENCHMARK_H_

#include <algorithm>
#include <random>
#include <vector>

#include <glog/logging.h>

#define DEFAULT_H 32    // default image height
#define DEFAULT_W 256   // default image width
#define DEFAULT_N 16    // default batch size
#define DEFAULT_K 16    // default input depth
#define DEFAULT_D 16    // default output depth

static std::default_random_engine RNG;

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "../lstm_gpu.h"
using thrust::system::cuda::experimental::pinned_allocator;
static thrust::device_vector<float> DATA_gpu_float;
static thrust::device_vector<double> DATA_gpu_double;
static thrust::host_vector<float, pinned_allocator<float> > DATA_cpu_float;
static thrust::host_vector<double, pinned_allocator<double> > DATA_cpu_double;
#define VECTOR_DATA(v) (v).data().get()
#else
#include "../lstm_cpu.h"
static std::vector<float> DATA_cpu_float;
static std::vector<double> DATA_cpu_double;
#define VECTOR_DATA(v) (v).data()
#endif

static void AllocateData(
    int H = DEFAULT_H, int W = DEFAULT_W, int N = DEFAULT_N,
    int K = DEFAULT_K, int D = DEFAULT_D) {
  LOG(INFO) << "Input/output size: H = " << H << ", W = " << W << ", N = "
            << N << ", K = " << K << ", D = " << D;
  const size_t ALL_DATA_SIZE =
      2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +
      2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +
      2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D) +
      RNN2D_LSTM_WORKSPACE_TRAINING_SIZE(H, W, N, D);
  const float ALL_DATA_SIZE_MB =
      (4 + 8) * (ALL_DATA_SIZE) / static_cast<float>(1 << 20);
  LOG(INFO) << "Allocating " << ALL_DATA_SIZE_MB << "MB in the CPU memory...";
  DATA_cpu_float.resize(ALL_DATA_SIZE);
  DATA_cpu_double.resize(ALL_DATA_SIZE);
  LOG(INFO) << "Filling CPU memory with random data...";
  std::normal_distribution<float>  ndist_f(0.0, 0.01);
  std::generate(DATA_cpu_float.begin(), DATA_cpu_float.end(),
                [&ndist_f]{ return ndist_f(RNG); });
  std::normal_distribution<double> ndist_d(0.0, 0.01);
  std::generate(DATA_cpu_double.begin(), DATA_cpu_double.end(),
                [&ndist_d]{ return ndist_d(RNG); });
#ifdef __CUDACC__
  LOG(INFO) << "Allocating " << ALL_DATA_SIZE_MB << "MB in the GPU memory...";
  DATA_gpu_float.resize(DATA_cpu_float.size());
  DATA_gpu_double.resize(DATA_cpu_double.size());
  LOG(INFO) << "Copying data into GPU...";
  DATA_gpu_float.assign(DATA_cpu_float.begin(), DATA_cpu_float.end());
  DATA_gpu_double.assign(DATA_cpu_double.begin(), DATA_cpu_double.end());
#endif
  LOG(INFO) << "Done!";
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
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(         \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* input offset */ \
    TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +          \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* output offset */ \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(              \
          H, W, N, K, D, input, nullptr, param, output, workspace);     \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference)           \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(          \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* input offset */ \
    TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +          \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* output offset */ \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(               \
          H, W, N, K, D, input, nullptr, param, output, workspace);     \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training)            \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_workspace(         \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* input offset */ \
    const TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +    \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* output offset */ \
    const TYPE* gOutput = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +   \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, K, K);        /* gOutput offset */ \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_workspace(               \
          H, W, N, K, D, input, nullptr, param, output, gOutput,        \
          workspace);                                                   \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_workspace)           \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_param(             \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    TYPE* gradParam = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* gradParam offset */ \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* input offset */ \
    const TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +    \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* output offset */ \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_param(                   \
      H, W, N, K, D, input, output, 1.0, gradParam, workspace);         \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_param)               \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_input(             \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    TYPE* gradInput = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +             \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +  /* gradOutput offset */ \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* param offset */      \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_input(                   \
          H, W, N, K, D, param, 1.0, gradInput, workspace);             \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_input)               \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_ALL(               \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_K, D = DEFAULT_D;              \
    const TYPE* param = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);      \
    TYPE* gradParam = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* gradParam offset */ \
    const TYPE* input = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +     \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D);        /* input offset */ \
    TYPE* gradInput = VECTOR_DATA(DATA_##DEVICE##_##TYPE) +             \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +  /* gradOutput offset */ \
        RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* param offset */      \
    const TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +    \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K);      /* output offset */ \
    const TYPE* gOutput = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +   \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D)  +      /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        RNN2D_LSTM_OUTPUT_SIZE(H, W, K, K);        /* gOutput offset */ \
    TYPE* workspace = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +       \
        2 * RNN2D_LSTM_PARAMETERS_SIZE(K, D) +       /* input offset */ \
        2 * RNN2D_LSTM_INPUT_SIZE(H, W, N, K) +     /* output offset */ \
        2 * RNN2D_LSTM_OUTPUT_SIZE(H, W, N, D);  /* workspace offset */ \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_workspace(               \
          H, W, N, K, D, input, nullptr, param, output, gOutput,        \
          workspace);                                                   \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_param(                   \
          H, W, N, K, D, input, output, 1.0, gradParam, workspace);     \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_input(                   \
          H, W, N, K, D, param, 1.0, gradInput, workspace);             \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * K * D);    \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw_ALL)                 \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime()



#endif  // BENCH_LSTM_COMMON_BENCHMARK_H_

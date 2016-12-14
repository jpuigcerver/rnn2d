#ifndef BENCH_TILE_COMMON_BENCHMARK_H_
#define BENCH_TILE_COMMON_BENCHMARK_H_

#include <algorithm>
#include <random>
#include <vector>

#include <glog/logging.h>

#define DEFAULT_H 32    // default image height
#define DEFAULT_W 256   // default image width
#define DEFAULT_N 16    // default batch size
#define DEFAULT_D 16    // default input depth
#define DEFAULT_MAX_KH 5
#define DEFAULT_MAX_KW 5

static std::default_random_engine RNG;

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <rnn2d/tile_gpu.h>
using thrust::system::cuda::experimental::pinned_allocator;
static thrust::device_vector<float> DATA_gpu_float;
static thrust::device_vector<double> DATA_gpu_double;
static thrust::host_vector<float, pinned_allocator<float> > DATA_cpu_float;
static thrust::host_vector<double, pinned_allocator<double> > DATA_cpu_double;
#define VECTOR_DATA(v) (v).data().get()
#else
#include <rnn2d/tile_cpu.h>
static std::vector<float> DATA_cpu_float;
static std::vector<double> DATA_cpu_double;
#define VECTOR_DATA(v) (v).data()
#endif

static void AllocateData(
    int H = DEFAULT_H, int W = DEFAULT_W, int N = DEFAULT_N,
    int D = DEFAULT_D, int kH = DEFAULT_MAX_KH, int kW = DEFAULT_MAX_KW) {
  LOG(INFO) << "Input size: H = " << H << ", W = " << W << ", N = "
            << N << ", D = " << D;
  const size_t ALL_DATA_SIZE =
      RNN2D_TILE_INPUT_SIZE(H, W, N, D) +
      RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW);
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
  static void BM_tile_ ## DEVICE ## _ ## TYPE ## _fw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W, N = DEFAULT_N,              \
        D = DEFAULT_D;                                                  \
    const int kH = state.range(0), kW = state.range(1);                 \
    const TYPE* input  = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);     \
    TYPE* output = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) +          \
        RNN2D_TILE_INPUT_SIZE(H, W, N, D);                              \
    while (state.KeepRunning()) {                                       \
      rnn2d_tile_ ## DEVICE ## _ ## TYPE ## _fw(                        \
          H, W, N, D, kH, kW, nullptr, input, output);                  \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * D);        \
  }                                                                     \
  BENCHMARK(BM_tile_ ## DEVICE ## _ ## TYPE ## _fw)                     \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime()                                                       \
  ->Args({2, 2})->Args({3, 3})->Args({5, 5});                           \
                                                                        \
  static void BM_tile_ ## DEVICE ## _ ## TYPE ## _bw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W, N = DEFAULT_N,              \
        D = DEFAULT_D;                                                  \
    const int kH = state.range(0), kW = state.range(1);                 \
    TYPE* gradInput = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE);        \
    const TYPE* gradOutput = VECTOR_DATA(DATA_ ## DEVICE ## _ ## TYPE) + \
        RNN2D_TILE_INPUT_SIZE(H, W, N, D);                              \
    while (state.KeepRunning()) {                                       \
      rnn2d_tile_## DEVICE ##_## TYPE ##_bw(                            \
          H, W, N, D, kH, kW, nullptr, gradOutput, gradInput);          \
    }                                                                   \
    state.SetItemsProcessed(state.iterations() * H * W * N * D);        \
  }                                                                     \
  BENCHMARK(BM_tile_ ## DEVICE ## _ ## TYPE ## _bw)                     \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime()                                                       \
  ->Args({2, 2})->Args({3, 3})->Args({5, 5})

#endif  // BENCH_TILE_COMMON_BENCHMARK_H_

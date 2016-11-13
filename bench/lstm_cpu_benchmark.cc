#include <algorithm>
#include <random>

#include <benchmark/benchmark.h>

#include "../lstm_cpu.h"

#define DEFAULT_H 32    // default image height
#define DEFAULT_W 256   // default image width
#define DEFAULT_N 16    // default batch size
#define DEFAULT_D 32    // default image depth

static std::default_random_engine rng;

#define DEFINE_CPU_BENCH(TYPE)                                          \
  static void BM_lstm_cpu_ ## TYPE ## _fw(benchmark::State& state) {    \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    std::normal_distribution<TYPE> norm_dist((TYPE)0.0, (TYPE)0.01);    \
    std::vector<TYPE> input(H * W * N * K);                             \
    std::vector<TYPE> output(H * W * N * D * 4);                        \
    std::vector<TYPE> param(4 * (1 + K + D + D) * 5 * D);               \
    std::vector<TYPE> workspace(4 * H * W * N * 6 * D);                 \
    std::generate(input.begin(), input.end(), [&norm_dist]{             \
        return norm_dist(rng);                                          \
      });                                                               \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_cpu_ ## TYPE ## _fw_inference(                         \
          H, W, N, K, D, input.data(), nullptr, param.data(),           \
          output.data(), workspace.data());                             \
    }                                                                   \
    state.SetItemsProcessed(input.size());                              \
  }                                                                     \
  BENCHMARK(BM_lstm_cpu_ ## TYPE ## _fw)                                \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_cpu_ ## TYPE ## _bw(benchmark::State& state) {    \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    std::vector<TYPE> input(H * W * N * K);                             \
    std::vector<TYPE> output(H * W * N * D * 4);                        \
    std::vector<TYPE> param(4 * (1 + K + D + D) * 5 * D);               \
    std::vector<TYPE> workspace(4 * H * W * N * 6 * D);                 \
    std::vector<TYPE> dWorkspace(4 * H * W * N * 6 * D);                \
    std::vector<TYPE> dInput(H * W * N * K);                            \
    std::vector<TYPE> dParam(4 * (1 + K + D + D) * 5 * D);              \
    std::vector<TYPE> dOutput(H * W * N * D * 4);                       \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_cpu_ ## TYPE ## _fw_training(                          \
          H, W, N, K, D, input.data(), nullptr, param.data(),           \
          output.data(), workspace.data());                             \
      rnn2d_lstm_cpu_ ## TYPE ## _bw_workspace(                         \
          H, W, N, K, D, input.data(), nullptr, param.data(),           \
          output.data(), workspace.data(), dOutput.data(),              \
          dWorkspace.data());                                           \
      rnn2d_lstm_cpu_ ## TYPE ## _bw_input(                             \
          H, W, N, K, D, param.data(), dWorkspace.data(), 1.0,          \
          dInput.data());                                               \
      rnn2d_lstm_cpu_ ## TYPE ## _bw_param(                             \
          H, W, N, K, D, input.data(), output.data(),                   \
          dWorkspace.data(), 1.0, dParam.data());                       \
    }                                                                   \
    state.SetItemsProcessed(input.size());                              \
  }                                                                     \
  BENCHMARK(BM_lstm_cpu_ ## TYPE ## _bw)                                \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime()

DEFINE_CPU_BENCH(float);
DEFINE_CPU_BENCH(double);

BENCHMARK_MAIN()

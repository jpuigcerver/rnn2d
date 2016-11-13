#ifndef BECNH_LSTM_COMMON_BENCHMARK_H_
#define BECNH_LSTM_COMMON_BENCHMARK_H_

#define DEFAULT_H 32    // default image height
#define DEFAULT_W 256   // default image width
#define DEFAULT_N 16    // default batch size
#define DEFAULT_D 16    // default image depth

static std::default_random_engine rng;

/*
  std::normal_distribution<TYPE> norm_dist((TYPE)0.0, (TYPE)0.01);      \
  std::generate(input.begin(), input.end(), [&norm_dist]{               \
  return norm_dist(rng);                                                \
  });                                                                   \
*/

// Before using this define, you must define VECTOR_CLASS(type) and
// VECTOR_DATA(vector).
// For instance, for a CPU benchmark:
// #define VECTOR_CLASS(t) std::vector<t>
// #define VECTOR_DATA(v) v.data()
// DEFINE_BENCHMARK(cpu, float)
// DEFINE_BENCHMARK(cpu, double)
#define DEFINE_BENCHMARK(DEVICE, TYPE)                                  \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    VECTOR_CLASS(TYPE) input(H * W * N * K);                            \
    VECTOR_CLASS(TYPE) output(H * W * N * D * 4);                       \
    VECTOR_CLASS(TYPE) param(4 * (1 + K + D + D) * 5 * D);              \
    VECTOR_CLASS(TYPE) workspace(4 * H * W * N * 6 * D);                \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_inference(              \
          H, W, N, K, D, VECTOR_DATA(input), nullptr,                   \
          VECTOR_DATA(param), VECTOR_DATA(output),                      \
          VECTOR_DATA(workspace));                                      \
    }                                                                   \
    state.SetItemsProcessed(input.size());                              \
  }                                                                     \
  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _fw)                     \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  static void BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw(                   \
      benchmark::State& state) {                                        \
    const int H = DEFAULT_H, W = DEFAULT_W;                             \
    const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;              \
    VECTOR_CLASS(TYPE) input(H * W * N * K);                            \
    VECTOR_CLASS(TYPE) output(H * W * N * D * 4);                       \
    VECTOR_CLASS(TYPE) param(4 * (1 + K + D + D) * 5 * D);              \
    VECTOR_CLASS(TYPE) workspace(4 * H * W * N * 6 * D);                \
    VECTOR_CLASS(TYPE) dWorkspace(4 * H * W * N * 6 * D);               \
    VECTOR_CLASS(TYPE) dInput(H * W * N * K);                           \
    VECTOR_CLASS(TYPE) dParam(4 * (1 + K + D + D) * 5 * D);             \
    VECTOR_CLASS(TYPE) dOutput(H * W * N * D * 4);                      \
    while (state.KeepRunning()) {                                       \
      rnn2d_lstm_ ## DEVICE ## _ ## TYPE ## _fw_training(               \
          H, W, N, K, D, VECTOR_DATA(input), nullptr,                   \
          VECTOR_DATA(param), VECTOR_DATA(output),                      \
          VECTOR_DATA(workspace));                                      \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_workspace(               \
          H, W, N, K, D, VECTOR_DATA(input), nullptr,                   \
          VECTOR_DATA(param), VECTOR_DATA(output),                      \
          VECTOR_DATA(workspace), VECTOR_DATA(dOutput),                 \
          VECTOR_DATA(dWorkspace));                                     \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_input(                   \
          H, W, N, K, D, VECTOR_DATA(param), VECTOR_DATA(dWorkspace),   \
          1.0, VECTOR_DATA(dInput));                                    \
      rnn2d_lstm_## DEVICE ## _ ## TYPE ## _bw_param(                   \
          H, W, N, K, D, VECTOR_DATA(input), VECTOR_DATA(output),       \
          VECTOR_DATA(dWorkspace), 1.0, VECTOR_DATA(dParam));           \
    }                                                                   \
    state.SetItemsProcessed(input.size());                              \
  }
/*  BENCHMARK(BM_lstm_ ## DEVICE ## _ ## TYPE ## _bw)                   \
  ->Unit(benchmark::kMillisecond)                                       \
  ->UseRealTime() */

#endif  // BECNH_LSTM_COMMON_BENCHMARK_H_

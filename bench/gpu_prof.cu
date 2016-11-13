#include <algorithm>
#include <random>

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "../lstm_gpu.h"
#include "lstm_common_benchmark.h"

#define VECTOR_CLASS(t) thrust::device_vector<t>
#define VECTOR_DATA(v) (v).data().get()
DEFINE_BENCHMARK(gpu, float);
//DEFINE_BENCHMARK(gpu, double);

int main(int argc, char **argv) {
  const int H = DEFAULT_H, W = DEFAULT_W;
  const int N = DEFAULT_N, K = DEFAULT_D, D = DEFAULT_D;
  VECTOR_CLASS(float) input(H * W * N * K);
  VECTOR_CLASS(float) output(H * W * N * D * 4);
  VECTOR_CLASS(float) param(4 * (1 + K + D + D) * 5 * D);
  VECTOR_CLASS(float) workspace(4 * H * W * N * 6 * D);
  rnn2d_lstm_gpu_float_fw_inference(
      H, W, N, K, D, VECTOR_DATA(input), nullptr,
      VECTOR_DATA(param), VECTOR_DATA(output),
      VECTOR_DATA(workspace));
  return 0;
}

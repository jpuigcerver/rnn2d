#include <algorithm>
#include <random>

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "../lstm_gpu.h"
#include "lstm_common_benchmark.h"

#define VECTOR_CLASS(t) thrust::device_vector<t>
#define VECTOR_DATA(v) (v).data().get()
DEFINE_BENCHMARK(gpu, float);
DEFINE_BENCHMARK(gpu, double);

BENCHMARK_MAIN()

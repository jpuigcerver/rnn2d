#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "../lstm_cpu.h"
#include "lstm_common_benchmark.h"

#define VECTOR_CLASS(t) std::vector<t>
#define VECTOR_DATA(v) (v).data()
DEFINE_BENCHMARK(cpu, float);
DEFINE_BENCHMARK(cpu, double);

BENCHMARK_MAIN()

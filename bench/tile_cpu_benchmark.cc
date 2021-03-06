#include <benchmark/benchmark.h>

#include "tile_common_benchmark.h"

DEFINE_BENCHMARK(cpu, float);
DEFINE_BENCHMARK(cpu, double);

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  benchmark::Initialize(&argc, argv);
  AllocateData();
  benchmark::RunSpecifiedBenchmarks();
  DeallocateData();
  return 0;
}

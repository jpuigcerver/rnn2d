#include <benchmark/benchmark.h>

#include "lstm_common_benchmark.h"
#include "../include/rnn2d/cuda_utils.h"

DEFINE_BENCHMARK(gpu, float);
DEFINE_BENCHMARK(gpu, double);

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  benchmark::Initialize(&argc, argv);
  // Get number of CUDA devices and their names.
  int numDevices = 0;
  CHECK_CUDA_CALL(cudaGetDeviceCount(&numDevices));
  cudaDeviceProp* dp = new cudaDeviceProp[numDevices];
  for (int d = 0; d < numDevices; ++d) {
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&dp[d], d));
  }
  // Set the CUDA device to use
  if (argc > 1) {
    CHECK_CUDA_CALL(cudaSetDevice(atoi(argv[1])));
  }
  // Display the name of the CUDA device being used
  int curDevice = 0;
  CHECK_CUDA_CALL(cudaGetDevice(&curDevice));
  LOG(INFO) << "Found " << numDevices << " CUDA devices, using device "
            << curDevice << ": " << dp[curDevice].name;
  AllocateData();
  benchmark::RunSpecifiedBenchmarks();
  DeallocateData();
  return 0;
}

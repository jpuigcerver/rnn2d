FIND_PACKAGE(GBenchmark)
IF(GBENCH_FOUND)
  # CPU BENCHMARKS
  ADD_EXECUTABLE(lstm_cpu_benchmark lstm_cpu_benchmark.cc)
  TARGET_LINK_LIBRARIES(lstm_cpu_benchmark
    rnn2d_cpu pthread benchmark ${COMMON_LIBRARIES})

  # GPU BENCHMARKS
  IF (CUDA_FOUND AND USE_CUDA)
    CUDA_ADD_EXECUTABLE(lstm_gpu_benchmark lstm_gpu_benchmark.cu)
    TARGET_LINK_LIBRARIES(lstm_gpu_benchmark
      rnn2d_gpu pthread benchmark ${COMMON_LIBRARIES})
  ENDIF()
ENDIF()
FIND_PACKAGE(GBenchmark)
IF(GBENCH_FOUND)

ENDIF()

ADD_EXECUTABLE(lstm_cpu_benchmark lstm_cpu_benchmark.cc)
TARGET_LINK_LIBRARIES(lstm_cpu_benchmark rnn2d_cpu pthread benchmark ${COMMON_LIBRARIES})
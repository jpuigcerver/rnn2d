FIND_PACKAGE(GTest)
FIND_PACKAGE(GMock)

IF(GTEST_FOUND AND GMOCK_FOUND)
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(${GMOCK_INCLUDE_DIRS})

  # LSTM CPU TESTS
  ADD_EXECUTABLE(lstm_cpu_test lstm_cpu_test.cc)
  TARGET_LINK_LIBRARIES(lstm_cpu_test rnn2d_cpu ${GMOCK_MAIN_LIBRARIES}
    ${COMMON_LIBRARIES})
  ADD_TEST(lstm_cpu_test lstm_cpu_test)

  # TILE CPU TESTS
  ADD_EXECUTABLE(tile_cpu_test tile_cpu_test.cc)
  TARGET_LINK_LIBRARIES(tile_cpu_test rnn2d_cpu ${GMOCK_MAIN_LIBRARIES}
    ${COMMON_LIBRARIES})
  ADD_TEST(tile_cpu_test tile_cpu_test)

  IF (CUDA_FOUND AND WITH_CUDA)
    # LSTM GPU TEST
    CUDA_ADD_EXECUTABLE(lstm_gpu_test lstm_gpu_test.cu)
    TARGET_LINK_LIBRARIES(lstm_gpu_test rnn2d_gpu ${GMOCK_MAIN_LIBRARIES}
      ${COMMON_LIBRARIES})
    ADD_TEST(lstm_gpu_test lstm_gpu_test)

    # TILE GPU TEST
    CUDA_ADD_EXECUTABLE(tile_gpu_test tile_gpu_test.cu)
    TARGET_LINK_LIBRARIES(tile_gpu_test rnn2d_gpu ${GMOCK_MAIN_LIBRARIES}
      ${COMMON_LIBRARIES})
    ADD_TEST(tile_gpu_test tile_gpu_test)
  ENDIF()
ENDIF()

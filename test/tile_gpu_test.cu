#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <thrust/device_vector.h>

#include <rnn2d/tile_gpu.h>

static std::default_random_engine RNG;

#define DEFINE_GPU_TESTS(TYPE)                                          \
  TEST(tile_test, rnn2d_tile_gpu_ ## TYPE ## _fw) {                     \
    const int H = 2, W = 3, N = 2, D = 4;                               \
    const std::vector<int> S_cpu{                                       \
      1, 2,                                                             \
      2, 3                                                              \
    };                                                                  \
    const thrust::device_vector<int> S(S_cpu);                          \
    const std::vector<TYPE> I_cpu{                                      \
      /* y = 0, x = 0 */                                                \
      1,  2,  3,  4,                                                    \
      5,  6,  7,  8,                                                    \
      /* y = 0, x = 1 */                                                \
      9,  10, 11, 12,                                                   \
      13, 14, 15, 16,                                                   \
      /* y = 0, x = 3 */                                                \
      0,  0,  0,  0,                                                    \
      17, 18, 19, 20,                                                   \
      /* y = 1, x = 0 */                                                \
      0,  0,  0,  0,                                                    \
      21, 22, 23, 24,                                                   \
      /* y = 1, x = 1 */                                                \
      0,  0,  0,  0,                                                    \
      25, 26, 27, 28,                                                   \
      /* y = 1, x = 2 */                                                \
      0,  0,  0,  0,                                                    \
      29, 30, 31, 32                                                    \
    };                                                                  \
    thrust::device_vector<TYPE> I(I_cpu), O;                            \
    {                                                                   \
      const int kH = 2, kW = 2;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_gpu_ ## TYPE ## _fw(                                   \
          H, W, N, D, kH, kW, S.data().get(),                           \
          I.data().get(), O.data().get());                              \
      const std::vector<TYPE> expected_O_cpu{                           \
        /* y = 0, x = 0 */                                              \
        1,  2,  3,  4,  9, 10, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  \
        5,  6,  7,  8, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28,  \
        /* y = 0, x = 1 */                                              \
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  \
        17, 18, 19, 20, 0,  0,  0,  0, 29, 30, 31, 32,  0,  0,  0,  0   \
      };                                                                \
      const thrust::device_vector<TYPE> expected_O(expected_O_cpu);     \
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
    {                                                                   \
      const int kH = 3, kW = 2;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_gpu_ ## TYPE ## _fw(                                   \
          H, W, N, D, kH, kW, S.data().get(),                           \
          I.data().get(), O.data().get());                              \
      const std::vector<TYPE> expected_O_cpu{                           \
        /* y = 0, x = 0, n = 0 */                                       \
        1,   2,  3,  4,  9, 10, 11, 12,                                 \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        /* y = 0, x = 0, n = 1 */                                       \
        5,   6,  7,  8, 13, 14, 15, 16,                                 \
        21, 22, 23, 24, 25, 26, 27, 28,                                 \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        /* y = 0, x = 1, n = 0 */                                       \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        0,   0,  0,  0,  0,  0,  0,  0,                                 \
        /* y = 0, x = 1, n = 1 */                                       \
        17, 18, 19, 20,  0,  0,  0,  0,                                 \
        29, 30, 31, 32,  0,  0,  0,  0,                                 \
        0,   0,  0,  0,  0,  0,  0,  0                                  \
      };                                                                \
      const thrust::device_vector<TYPE> expected_O(expected_O_cpu);     \
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
    {                                                                   \
      const int kH = 1, kW = 3;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_gpu_ ## TYPE ## _fw(                                   \
          H, W, N, D, kH, kW, S.data().get(),                           \
          I.data().get(), O.data().get());                              \
      const std::vector<TYPE> expected_O_cpu{                           \
        /* y = 0, x = 0 */                                              \
        1,  2,  3,  4,  9, 10, 11, 12,  0,  0,  0,  0,                  \
        5,  6,  7,  8, 13, 14, 15, 16, 17, 18, 19, 20,                  \
        /* y = 1, x = 0 */                                              \
        0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,                 \
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32                  \
      };                                                                \
      const thrust::device_vector<TYPE> expected_O(expected_O_cpu);     \
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
  }                                                                     \
  TEST(tile_test, rnn2d_tile_gpu_ ## TYPE ## _bw) {                     \
  }

DEFINE_GPU_TESTS(float)
DEFINE_GPU_TESTS(double)

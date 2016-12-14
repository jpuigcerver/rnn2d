#include <algorithm>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <rnn2d/tile_cpu.h>

static std::default_random_engine RNG;

#define DEFINE_CPU_TESTS(TYPE)                                          \
  TEST(tile_test, rnn2d_tile_cpu_ ## TYPE ## _fw) {                     \
    const int H = 2, W = 3, N = 2, D = 4;                               \
    const int S[] = {                                                   \
      1, 2,                                                             \
      2, 3                                                              \
    };                                                                  \
    const std::vector<TYPE> I = {                                       \
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
    std::vector<TYPE> O;                                                \
    {                                                                   \
      const int kH = 2, kW = 2;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_cpu_ ## TYPE ## _fw(H, W, N, D, kH, kW, S,             \
                                     I.data(), O.data());               \
      const std::vector<TYPE> expected_O = {                            \
        /* y = 0, x = 0 */                                              \
        1,  2,  3,  4,  9, 10, 11, 12,  0,  0,  0,  0,  0,  0,  0,  0,  \
        5,  6,  7,  8, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28,  \
        /* y = 0, x = 1 */                                              \
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  \
        17, 18, 19, 20, 0,  0,  0,  0, 29, 30, 31, 32,  0,  0,  0,  0   \
      };                                                                \
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
    {                                                                   \
      const int kH = 3, kW = 2;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_cpu_ ## TYPE ## _fw(H, W, N, D, kH, kW, S,             \
                                     I.data(), O.data());               \
      const std::vector<TYPE> expected_O = {                            \
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
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
    {                                                                   \
      const int kH = 1, kW = 3;                                         \
      O.resize(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));             \
      rnn2d_tile_cpu_ ## TYPE ## _fw(H, W, N, D, kH, kW, S,             \
                                     I.data(), O.data());               \
      const std::vector<TYPE> expected_O = {                            \
        /* y = 0, x = 0 */                                              \
        1,  2,  3,  4,  9, 10, 11, 12,  0,  0,  0,  0,                  \
        5,  6,  7,  8, 13, 14, 15, 16, 17, 18, 19, 20,                  \
        /* y = 1, x = 0 */                                              \
        0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,                 \
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32                  \
      };                                                                \
      EXPECT_THAT(O, ::testing::ElementsAreArray(expected_O));          \
    }                                                                   \
  }                                                                     \
  TEST(tile_test, rnn2d_tile_cpu_ ## TYPE ## _bw) {                     \
    std::uniform_int_distribution<int> udist(1, 32);                    \
    std::uniform_int_distribution<int> udist2(1, 4);                    \
    for (int r = 0; r < 10; ++r) {                                      \
      const int H = udist(RNG), W = udist(RNG), N = udist(RNG),         \
          D = udist(RNG), kH = udist2(RNG), kW = udist2(RNG);           \
      std::vector<TYPE> I(H * W * N * D), dI(H * W * N * D);            \
      std::vector<TYPE> O(RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, kH, kW));  \
      size_t n = 0;                                                     \
      std::generate(I.begin(), I.end(), [&n]() { return ++n; });        \
      /* The gradient is really easy to check in this case, since it */ \
      /* reduces to the inverse operation: backpropagate the output  */ \
      /* and you should get the input. */                               \
      rnn2d_tile_cpu_ ## TYPE ## _fw(H, W, N, D, kH, kW, nullptr,       \
                                     I.data(), O.data());               \
      rnn2d_tile_cpu_ ## TYPE ## _bw(H, W, N, D, kH, kW, nullptr,       \
                                     O.data(), dI.data());              \
      EXPECT_THAT(dI, ::testing::ElementsAreArray(I));                  \
    }                                                                   \
  }

DEFINE_CPU_TESTS(float)
DEFINE_CPU_TESTS(double)

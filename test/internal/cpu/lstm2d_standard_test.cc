#include <gmock/gmock.h>

#include <rnn2d/internal/cpu/rnn2d_inference_standard_impl.h>
#include <rnn2d/internal/cpu/lstm2d_cell.h>

using ::rnn2d::internal::Sigmoid;
using ::rnn2d::internal::cpu::Lstm2dCell;
using ::rnn2d::internal::cpu::Rnn2dInferenceStandardImpl;


using ::testing::Each;

template <typename T>
class Lstm2dStandardCpuTest : public ::testing::Test {};

typedef ::testing::Types<float, double> MyTypes;
TYPED_TEST_CASE(Lstm2dStandardCpuTest, MyTypes);

TYPED_TEST(Lstm2dStandardCpuTest, ForwardAllWeightsZero) {
  typedef TypeParam T;
  constexpr int H = 2, W = 3, N = 1, K = 2, D = 2;
  Rnn2dInferenceStandardImpl<T, Lstm2dCell<T>> lstm(K, D);
  std::vector<T> input(H * W * N * K, 1);
  std::vector<T> output(H * W * N * 4 * D, 1);
  std::vector<T> params(lstm.GetNumParameters(), 0);
  lstm.SetInput(H, W, N, nullptr, input.data());
  lstm.SetOutput(output.data());
  lstm.SetParameters(params.data());

  std::vector<T> wspace(lstm.GetSizeWSpace() / sizeof(T));
  lstm.SetWSpace(wspace.data());
  EXPECT_EQ(RNN2D_STATUS_SUCCESS, lstm.Forward())
            << RNN2D_GET_LAST_ERROR_MSG();
  EXPECT_THAT(output, Each(static_cast<T>(0)));
}

TYPED_TEST(Lstm2dStandardCpuTest, ForwardBias) {
  typedef TypeParam T;
  constexpr int H = 2, W = 3, N = 1, K = 2, D = 2;
  Rnn2dInferenceStandardImpl<T, Lstm2dCell<T>> lstm(K, D);
  std::vector<T> input(H * W * N * K, 1);       // Initialized to all ones
  std::vector<T> output(H * W * N * 4 * D, 1);  // Initialized to all ones
  std::vector<T> params(lstm.GetNumParameters(), 0);
  lstm.SetInput(H, W, N, nullptr, input.data());
  lstm.SetOutput(output.data());
  lstm.SetParameters(params.data());

  for (int z = 0; z < 4; ++z) {
    for (int d = 0; d < D; ++d) {
      // Cell state ~= Input activation
      *lstm.B(z, 0, d) = static_cast<T>(+10000);
      *lstm.B(z, 1, d) = static_cast<T>(-10000);
      *lstm.B(z, 2, d) = static_cast<T>(-10000);
      // Output gate ~= 1
      *lstm.B(z, 3, d) = static_cast<T>(+10000);
      // Input bias
      *lstm.B(z, 4, d) = static_cast<T>(z * 0.1 + 0.01 * d);
    }
  }

  std::vector<T> wspace(lstm.GetSizeWSpace() / sizeof(T));
  lstm.SetWSpace(wspace.data());
  EXPECT_EQ(RNN2D_STATUS_SUCCESS, lstm.Forward())
            << RNN2D_GET_LAST_ERROR_MSG();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int z = 0; z < 4; ++z) {
          EXPECT_NEAR(std::tanh(std::tanh(z * 0.1 + 0.00)),
                      *lstm.O(z, y, x, n, 0), 0.00001)
                    << "Failed at y=" << y << ", x=" << x << ", n=" << n
                    << ", z=" << z << ", d=0";
          EXPECT_NEAR(std::tanh(std::tanh(z * 0.1 + 0.01)),
                      *lstm.O(z, y, x, n, 1), 0.00001)
                    << "Failed at y=" << y << ", x=" << x << ", n=" << n
                    << ", z=" << z << ", d=1";
        }
      }
    }
  }
}

TYPED_TEST(Lstm2dStandardCpuTest, ForwardInputWeights) {
  typedef TypeParam T;
  constexpr int H = 2, W = 3, N = 1, K = 2, D = 2;
  Rnn2dInferenceStandardImpl<T, Lstm2dCell<T>> lstm(K, D);
  std::vector<T> input(H * W * N * K, 1);       // Initialized to all ones
  std::vector<T> output(H * W * N * 4 * D, 1);  // Initialized to all ones
  std::vector<T> params(lstm.GetNumParameters(), 0);
  lstm.SetInput(H, W, N, nullptr, input.data());
  lstm.SetOutput(output.data());
  lstm.SetParameters(params.data());

  for (int z = 0; z < 4; ++z) {
    for (int k = 0; k < K; ++k) {
      for (int d = 0; d < D; ++d) {
        *lstm.W(z, k, 0, d) = -2.0 / K;
        *lstm.W(z, k, 1, d) = -1.0 / K;
        *lstm.W(z, k, 2, d) =  0.0 / K;
        *lstm.W(z, k, 3, d) =  1.0 / K;
        *lstm.W(z, k, 4, d) =  2.0 / K;
      }
    }
  }

  std::vector<T> wspace(lstm.GetSizeWSpace() / sizeof(T));
  lstm.SetWSpace(wspace.data());
  EXPECT_EQ(RNN2D_STATUS_SUCCESS, lstm.Forward())
            << RNN2D_GET_LAST_ERROR_MSG();

  T cell_tmp[H][W];
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          cell_tmp[y][x] = std::tanh(2) * Sigmoid<T>::f(-2);
          if (y > 0) {
            cell_tmp[y][x] += cell_tmp[y - 1][x] * Sigmoid<T>::f(-1);
          }
          if (x > 0) {
            cell_tmp[y][x] += cell_tmp[y][x - 1] * Sigmoid<T>::f(0);
          }
          EXPECT_NEAR(std::tanh(cell_tmp[y][x]) * Sigmoid<T>::f(1),
                      *lstm.O(0, y, x, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << x << ", n=" << n
                    << ", z=" << 0 << ", d=" << d;
          EXPECT_NEAR(std::tanh(cell_tmp[y][x]) * Sigmoid<T>::f(1),
                      *lstm.O(1, H - y - 1, x, n, d), 0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << x
                    << ", n=" << n << ", z=" << 1 << ", d=" << d;
          EXPECT_NEAR(std::tanh(cell_tmp[y][x]) * Sigmoid<T>::f(1),
                      *lstm.O(2, y, W - x - 1, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 2 << ", d=" << d;
          EXPECT_NEAR(std::tanh(cell_tmp[y][x]) * Sigmoid<T>::f(1),
                      *lstm.O(3, H - y - 1, W - x - 1, n, d), 0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 3 << ", d=" << d;
        }
      }
    }
  }
}

TYPED_TEST(Lstm2dStandardCpuTest, ForwardRecurrentXWeights) {
  typedef TypeParam T;
  constexpr int H = 2, W = 3, N = 1, K = 2, D = 2;
  Rnn2dInferenceStandardImpl<T, Lstm2dCell<T>> lstm(K, D);
  std::vector<T> input(H * W * N * K, 1);       // Initialized to all ones
  std::vector<T> output(H * W * N * 4 * D, 1);  // Initialized to all ones
  std::vector<T> params(lstm.GetNumParameters(), 0);
  lstm.SetInput(H, W, N, nullptr, input.data());
  lstm.SetOutput(output.data());
  lstm.SetParameters(params.data());

  for (int z = 0; z < 4; ++z) {
    for (int k = 0; k < K; ++k) {
      for (int d = 0; d < D; ++d) {
        // Input activation ~= 1
        *lstm.B(z, 4, d) = static_cast<T>(100000);
        *lstm.U(z, k, 0, d) = -2.0 / K;
        *lstm.U(z, k, 1, d) = -1.0 / K;
        *lstm.U(z, k, 2, d) =  0.5 / K;
        *lstm.U(z, k, 3, d) =  1.0 / K;
        *lstm.U(z, k, 4, d) =  2.0 / K;
      }
    }
  }

  std::vector<T> wspace(lstm.GetSizeWSpace() / sizeof(T));
  lstm.SetWSpace(wspace.data());
  EXPECT_EQ(RNN2D_STATUS_SUCCESS, lstm.Forward())
            << RNN2D_GET_LAST_ERROR_MSG();

  T cell_tmp[H][W];
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          cell_tmp[y][x] =
              1.0 * Sigmoid<T>::f(x > 0 ? -2 * *lstm.O(0, y, x - 1, n, d) : 0);
          if (y > 0) {
            cell_tmp[y][x] += cell_tmp[y - 1][x] *
                Sigmoid<T>::f(x > 0 ? -1 * *lstm.O(0, y, x - 1, n, d) : 0);
          }
          if (x > 0) {
            cell_tmp[y][x] += cell_tmp[y][x - 1] *
                Sigmoid<T>::f(x > 0 ? 0.5 * *lstm.O(0, y, x - 1, n, d) : 0);
          }
          const T expected_output =
              std::tanh(cell_tmp[y][x]) *
                  Sigmoid<T>::f(x > 0 ? 1.0 * *lstm.O(0, y, x - 1, n, d) : 0);
          EXPECT_NEAR(expected_output, *lstm.O(0, y, x, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << x << ", n=" << n
                    << ", z=" << 0 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(1, H - y - 1, x, n, d), 0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << x
                    << ", n=" << n << ", z=" << 1 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(2, y, W - x - 1, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 2 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(3, H - y - 1, W - x - 1, n, d),
                      0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 3 << ", d=" << d;
        }
      }
    }
  }
}

TYPED_TEST(Lstm2dStandardCpuTest, ForwardRecurrentYWeights) {
  typedef TypeParam T;
  constexpr int H = 2, W = 3, N = 1, K = 2, D = 2;
  Rnn2dInferenceStandardImpl<T, Lstm2dCell<T>> lstm(K, D);
  std::vector<T> input(H * W * N * K, 1);       // Initialized to all ones
  std::vector<T> output(H * W * N * 4 * D, 1);  // Initialized to all ones
  std::vector<T> params(lstm.GetNumParameters(), 0);
  lstm.SetInput(H, W, N, nullptr, input.data());
  lstm.SetOutput(output.data());
  lstm.SetParameters(params.data());

  for (int z = 0; z < 4; ++z) {
    for (int k = 0; k < K; ++k) {
      for (int d = 0; d < D; ++d) {
        // Input activation ~= 1
        *lstm.B(z, 4, d) = static_cast<T>(100000);
        *lstm.V(z, k, 0, d) = -2.0 / K;
        *lstm.V(z, k, 1, d) = -1.0 / K;
        *lstm.V(z, k, 2, d) =  0.5 / K;
        *lstm.V(z, k, 3, d) =  1.0 / K;
        *lstm.V(z, k, 4, d) =  2.0 / K;
      }
    }
  }

  std::vector<T> wspace(lstm.GetSizeWSpace() / sizeof(T));
  lstm.SetWSpace(wspace.data());
  EXPECT_EQ(RNN2D_STATUS_SUCCESS, lstm.Forward())
            << RNN2D_GET_LAST_ERROR_MSG();

  T cell_tmp[H][W];
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          cell_tmp[y][x] =
              1.0 * Sigmoid<T>::f(y > 0 ? -2 * *lstm.O(0, y - 1, x, n, d) : 0);
          if (y > 0) {
            cell_tmp[y][x] += cell_tmp[y - 1][x] *
                Sigmoid<T>::f(y > 0 ? -1 * *lstm.O(0, y - 1, x, n, d) : 0);
          }
          if (x > 0) {
            cell_tmp[y][x] += cell_tmp[y][x - 1] *
                Sigmoid<T>::f(y > 0 ? 0.5 * *lstm.O(0, y - 1, x, n, d) : 0);
          }
          const T expected_output = std::tanh(cell_tmp[y][x]) *
              Sigmoid<T>::f(y > 0 ? 1.0 * *lstm.O(0, y - 1, x, n, d) : 0);
          EXPECT_NEAR(expected_output, *lstm.O(0, y, x, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << x << ", n=" << n
                    << ", z=" << 0 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(1, H - y - 1, x, n, d), 0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << x
                    << ", n=" << n << ", z=" << 1 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(2, y, W - x - 1, n, d), 0.00001)
                    << "Failed at y=" << y << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 2 << ", d=" << d;
          EXPECT_NEAR(expected_output, *lstm.O(3, H - y - 1, W - x - 1, n, d),
                      0.00001)
                    << "Failed at y=" << H - y - 1 << ", x=" << W - x - 1
                    << ", n=" << n << ", z=" << 3 << ", d=" << d;
        }
      }
    }
  }
}


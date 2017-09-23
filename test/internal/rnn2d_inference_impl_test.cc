#include <memory>

#include <gmock/gmock.h>

#include <rnn2d/internal/rnn2d_inference_impl.h>

namespace rnn2d {
namespace internal {
namespace testing {

template<typename T>
class Rnn2dInferenceImplFake : public Rnn2dInferenceImpl<T> {
 public:
  using Rnn2dInferenceImpl<T>::Rnn2dInferenceImpl;

  size_t GetNumParameters() const override { return 0; }
  size_t GetSizeWSpace() const override { return 0; }
  rnn2dStatus_t Forward() override { return RNN2D_STATUS_SUCCESS; }
};

}  // namespace testing
}  // namespace internal
}  // namespace rnn2d

using ::rnn2d::internal::testing::Rnn2dInferenceImplFake;

TEST(Rnn2dInferenceImplTest, Constructor) {
  const int K = 3, D = 25;
  Rnn2dInferenceImplFake<float> impl(K, D);
  EXPECT_EQ(K, impl.GetK());
  EXPECT_EQ(D, impl.GetD());

  // Copy constructor
  Rnn2dInferenceImplFake<float> impl2(impl);
  EXPECT_THAT(K, impl2.GetK());
  EXPECT_THAT(D, impl2.GetD());
}

TEST(Rnn2dInferenceImplTest, SetInputEqualSizes) {
  const int K = 3, D = 25;
  Rnn2dInferenceImplFake<float> impl(K, D);
  const int H = 10, W = 20, N = 30;

  float input = 1.0;
  impl.SetInput(H, W, N, nullptr, &input);
  EXPECT_EQ(H, impl.GetH());
  EXPECT_EQ(W, impl.GetW());
  EXPECT_EQ(N, impl.GetN());

  // Shape buffer was nullptr, thus all samples have the same size.
  EXPECT_EQ(H, impl.GetH(0));
  EXPECT_EQ(W, impl.GetW(0));
  EXPECT_EQ(H, impl.GetH(N - 1));
  EXPECT_EQ(W, impl.GetW(N - 1));
}

TEST(Rnn2dInferenceImplTest, SetInputDifferentSizes) {
  const int K = 3, D = 25;
  Rnn2dInferenceImplFake<float> impl(K, D);
  const int H = 10, W = 20, N = 30;

  const int shape[] = {1, 2, 3, 4, 5, 6}; float input = 1.0;
  impl.SetInput(H, W, N, shape, &input);
  EXPECT_EQ(1, impl.GetH(0));
  EXPECT_EQ(2, impl.GetW(0));
  EXPECT_EQ(5, impl.GetH(2));
  EXPECT_EQ(6, impl.GetW(2));
}

TEST(Rnn2dInferenceImplTest, GetXY) {
  // Note: In the following comment regions the asterisks denote the first
  // element in the diagonal (i.e. e = 0).
  // Case: H <= W
  {
    const int H = 2, W = 4;
    // z = 0
    // | 0* | 1* | 2* | 3* |
    // | 1  | 2  | 3  | 4  |
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 0, 0));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 0, 1));
    EXPECT_EQ( 2, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 3, 1));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 4, 0));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 4, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 4, 1));

    // z = 1
    // | 1  | 2  | 3  | 4  |
    // | 0* | 1* | 2* | 3* |
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 0, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 0, 0));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 0, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 0, 1));
    EXPECT_EQ( 2, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 3, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 3, 1));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 4, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 4, 0));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 4, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 4, 1));

    // z = 2
    // | 3* | 2* | 1* | 0* |
    // | 4  | 3  | 2  | 1  |
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 0, 0));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 3, 1));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 4, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 4, 1));

    // z = 3
    // | 4  | 3  | 2  | 1  |
    // | 3* | 2* | 1* | 0* |
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 0, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 0, 0));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 0, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 3, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 3, 1));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 4, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 4, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 4, 1));
  }
  // Case: H > W
  {
    const int H = 4, W = 2;
    // z = 0
    // | 0* | 1 |
    // | 1* | 2 |
    // | 2* | 3 |
    // | 3* | 4 |
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 0, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 0, 1));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 3, 1));
    EXPECT_EQ( 2, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 3, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 4, 0));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 4, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 0, 4, 1));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetY(H, W, 0, 4, 1));

    // z = 1
    // | 3* | 4 |
    // | 2* | 3 |
    // | 1* | 2 |
    // | 0* | 1 |
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 0, 0));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 0, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 0, 1));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 0, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 3, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 4, 0));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 4, 0));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 1, 4, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 1, 4, 1));

    // z = 2
    // | 1 | 0* |
    // | 2 | 1* |
    // | 3 | 2* |
    // | 4 | 3* |
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 0, 1));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 0, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 3, 1));
    EXPECT_EQ( 2, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 4, 0));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 2, 4, 1));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetY(H, W, 2, 4, 1));

    // z = 3
    // | 4 | 3* |
    // | 3 | 2* |
    // | 2 | 1* |
    // | 1 | 0* |
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 0, 0));
    EXPECT_EQ( 3, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 0, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 0, 1));
    EXPECT_EQ( 4, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 0, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 3, 1));
    EXPECT_EQ( 1, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 4, 0));
    EXPECT_EQ(-1, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 4, 0));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetX(H, W, 3, 4, 1));
    EXPECT_EQ( 0, Rnn2dInferenceImplFake<float>::GetY(H, W, 3, 4, 1));
  }
}

TEST(Rnn2dInferenceImplTest, GetPrevElemXY) {
  const int H = 3, W = 4;
  for (int z = 0; z < 4; ++z) {
    for (int t = 0; t < H + W - 1; ++t) {
      for (int e = 0; e < std::min(H, W); ++e) {
        // Get x,y-coordinates of the e-th element in the t-th diagonal.
        const int x = Rnn2dInferenceImplFake<float>::GetX(H, W, z, t, e);
        const int y = Rnn2dInferenceImplFake<float>::GetY(H, W, z, t, e);
        // Check that the element corresponding to the previous diagonal
        // displaced in the x-direction is correct.
        const int px = (z == 0 || z == 1) ? (x - 1) : (x + 1);
        const int ex = Rnn2dInferenceImplFake<float>::GetPrevElemX(H, W, e);
        EXPECT_EQ(px, Rnn2dInferenceImplFake<float>::GetX(H, W, z, t - 1, ex))
                  << "Failed for z = " << z << ", t = " << t << ", e = " << e;
        EXPECT_EQ(y, Rnn2dInferenceImplFake<float>::GetY(H, W, z, t - 1, ex))
                  << "Failed for z = " << z << ", t = " << t << ", e = " << e;
        // Check that the element corresponding to the previous diagonal
        // displaced in the y-direction is correct.
        const int py = (z == 0 || z == 2) ? (y - 1) : (y + 1);
        const int ey = Rnn2dInferenceImplFake<float>::GetPrevElemY(H, W, e);
        EXPECT_EQ(x, Rnn2dInferenceImplFake<float>::GetX(H, W, z, t - 1, ey))
                  << "Failed for z = " << z << ", t = " << t << ", e = " << e;
        EXPECT_EQ(py, Rnn2dInferenceImplFake<float>::GetY(H, W, z, t - 1, ey))
                  << "Failed for z = " << z << ", t = " << t << ", e = " << e;
      }
    }
  }
}
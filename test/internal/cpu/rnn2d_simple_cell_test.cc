#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <rnn2d/internal/cpu/rnn2d_simple_cell.h>

namespace rnn2d {
namespace internal {
namespace cpu {
namespace testing {

template <typename T>
class MockLayer {
 public:
  MOCK_CONST_METHOD0(GetH, int());
  MOCK_CONST_METHOD0(GetW, int());
  MOCK_CONST_METHOD0(GetN, int());
  MOCK_CONST_METHOD0(GetD, int());
  MOCK_CONST_METHOD1(GetH, int(int n));
  MOCK_CONST_METHOD1(GetW, int(int n));
  //MOCK_METHOD1(Q, T& )
};

}  // namespace testing
}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

using ::rnn2d::internal::Sigmoid;
using ::rnn2d::internal::cpu::Rnn2dSimpleCell;
using ::rnn2d::internal::cpu::testing::MockLayer;

using ::testing::Return;

TEST(Rnn2dSimpleCellTest, Constructor) {
  Rnn2dSimpleCell<float> cell1;
  Rnn2dSimpleCell<double, Sigmoid<double>> cell2;
  auto cell3 = Rnn2dSimpleCell<float, Sigmoid<float>>(Sigmoid<float>());
}

TEST(Rnn2dSimpleCell, ForwardNoShape) {
  const int H = 2, W = 3, N = 2, D = 3;
  MockLayer<float> layer;
  EXPECT_CALL(layer, GetH()).WillRepeatedly(Return(H));
  EXPECT_CALL(layer, GetW()).WillRepeatedly(Return(W));
  EXPECT_CALL(layer, GetN()).WillRepeatedly(Return(N));
  EXPECT_CALL(layer, GetD()).WillRepeatedly(Return(D));

  for (int n = 0; n < N; ++n) {
    EXPECT_CALL(layer, GetH(n)).WillRepeatedly(Return(H));
    EXPECT_CALL(layer, GetW(n)).WillRepeatedly(Return(W));
  }

  Rnn2dSimpleCell<float> cell;
  cell.Forward(&layer, 0);

}
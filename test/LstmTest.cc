#include <gtest/gtest-message.h>        // for Message
#include <gtest/gtest-typed-test.h>     // for TYPED_TEST, etc
#include <gtest/gtest.h>                // for ASSERT_TRUE, EXPECT_LT, etc
#include "LstmTest.h"                   // for LstmTest
#include "LstmTestCPU.h"                // for LstmTest<>::LstmTest, etc

#ifdef WITH_CUDA
#include "LstmTestGPU.h"

typedef ::testing::Types <
  TypeDefinitions<float,  device::CPU>,
  TypeDefinitions<double, device::CPU>,
  TypeDefinitions<float,  device::GPU>,
  TypeDefinitions<double, device::GPU>
  > Implementations;

#else

typedef ::testing::Types <
  TypeDefinitions<float,  device::CPU>,
  TypeDefinitions<double, device::CPU>
  > Implementations;

#endif  // WITH_CUDA
TYPED_TEST_CASE(LstmTest, Implementations);


TYPED_TEST(LstmTest, DISABLED_ForwardInference) {
  typedef typename TypeParam::Type Type;
  this->AllocateInference();
  ASSERT_TRUE(this->Input() != nullptr);
  ASSERT_TRUE(this->InputShape() != nullptr);
  ASSERT_TRUE(this->Parameters() != nullptr);
  ASSERT_TRUE(this->Output() != nullptr);
  ASSERT_TRUE(this->Workspace() != nullptr);
  this->DoForwardInference();
  this->CopyToHost();

  // Check output
  ASSERT_TRUE(this->OutputHost() != nullptr);
  ASSERT_TRUE(this->ExpectedOutputHost() != nullptr);
  for (int y = 0, i = 0; y < this->H; ++y) {
    for (int x = 0; x < this->W; ++x) {
      for (int n = 0; n < this->N; ++n) {
        for (int d = 0; d < 4 * this->D; ++d, ++i) {
          const Type expected = this->ExpectedOutputHost()[i];
          const Type actual = this->OutputHost()[i];
          const typename TypeParam::Type err = RelativeError(expected, actual);
          EXPECT_LT(err, 1e-4)
              << "Failed output at "
              << "(" << y << ", " << x << ", " << n << ", " << d << "): "
              << "Expected = " << expected << "  " << "Actual = " << actual;
        }
      }
    }
  }
}


TYPED_TEST(LstmTest, DISABLED_ForwardTraining) {
  typedef typename TypeParam::Type Type;
  this->AllocateTraining();
  this->DoForwardTraining();
  this->CopyToHost();

  // Check output
  ASSERT_TRUE(this->OutputHost() != nullptr);
  ASSERT_TRUE(this->ExpectedOutputHost() != nullptr);
  for (int y = 0, i = 0; y < this->H; ++y) {
    for (int x = 0; x < this->W; ++x) {
      for (int n = 0; n < this->N; ++n) {
        for (int d = 0; d < 4 * this->D; ++d, ++i) {
          const Type expected = this->ExpectedOutputHost()[i];
          const Type actual = this->OutputHost()[i];
          const typename TypeParam::Type err = RelativeError(expected, actual);
          EXPECT_LT(err, 1e-4)
              << "Failed output at "
              << "(" << y << ", " << x << ", " << n << ", " << d << "): "
              << "Expected = " << expected << "  " << "Actual = " << actual;
        }
      }
    }
  }
}


TYPED_TEST(LstmTest, Backward) {
  typedef typename TypeParam::Type Type;
  this->AllocateTraining();
  this->DoForwardTraining();
  this->DoBackwardData();
  this->DoBackwardInput();
  this->CopyToHost();
}

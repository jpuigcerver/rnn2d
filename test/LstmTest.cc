#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <glog/logging.h>

#include <rnn2d/lstm_cpu.h>
#ifdef WITH_CUDA
#include <rnn2d/lstm_gpu.h>
#endif

#include "./LstmAllocator.h"

typedef enum { CPU = 0, GPU = 1 } Device;

template <typename T, Device D>
struct TypeDefinitions {
  typedef T Type;
  static constexpr Device device = D;
};

template <typename T>
class LstmTest : public ::testing::Test {
 public:
  typedef typename T::Type Type;
  static constexpr Device device = T::device;

  static constexpr int H = 11;
  static constexpr int W = 7;
  static constexpr int N = 5;
  static constexpr int K = 3;
  static constexpr int D = 2;

  LstmTest () {
    InitializeAllocator();
    allocator_->AllocateInput();
    allocator_->AllocateOutput();
    allocator_->AllocateParameters();
  }

  void DoForwardInference();
  void DoForwardTraining();
  void DoBackwardData();
  void DoBackwardInput();

  inline Type* Input() {
    //ASSERT_TRUE(allocator_);
    return allocator_->Input();
  }

  inline Type* Output() {
    //ASSERT_TRUE(allocator_);
    return allocator_->Output();
  }

  inline Type* Parameters() {
    //ASSERT_TRUE(allocator_);
    return allocator_->Parameters();
  }

  inline const int* InputShape() const {
    //ASSERT_TRUE(allocator_);
    return allocator_->InputShape();
  }

  inline Type* GradInput() {
    return allocator_->GradInput();
  }

  inline void* Workspace() {
    //ASSERT_TRUE(allocator_);
    return allocator_->Workspace();
  }

  inline void* Reserved() {
    //ASSERT_TRUE(allocator_);
    return allocator_->Reserved();
  }

  void CopyToDevice();
  void CopyFromDevice();

  void PrintOutput() {
    for (int y = 0, i = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        printf("O[%d, %d, :, :] =\n", y, x);
        for (int n = 0; n < N; ++n) {
          printf("  ");
          for (int d = 0; d < 5 * D; ++d, ++i) {
            printf(" % 8.6f", allocator_->Output()[i]);
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }

 protected:
  std::unique_ptr< LstmAllocator<Type> > allocator_;

  void InitializeAllocator();
};

template <>
void LstmTest< TypeDefinitions<float, Device::CPU> >::InitializeAllocator() {
  allocator_.reset(new LstmAllocatorCPU<float>(H, W, N, K, D));
}

template <>
void LstmTest< TypeDefinitions<double, Device::CPU> >::InitializeAllocator() {
  allocator_.reset(new LstmAllocatorCPU<double>(H, W, N, K, D));
}

template <>
void LstmTest< TypeDefinitions<float, Device::CPU> >::DoForwardInference() {
  rnn2d_lstm_cpu_float_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<double, Device::CPU> >::DoForwardInference() {
  rnn2d_lstm_cpu_double_fw_inference(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace());
}

template <>
void LstmTest< TypeDefinitions<float, Device::CPU> >::DoForwardTraining() {
  rnn2d_lstm_cpu_float_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

template <>
void LstmTest< TypeDefinitions<double, Device::CPU> >::DoForwardTraining() {
  rnn2d_lstm_cpu_double_fw_training(
      H, W, N, K, D, Input(), InputShape(), Parameters(), Output(),
      Workspace(), Reserved());
}

#ifdef WITH_CUDA
typedef ::testing::Types <
  TypeDefinitions<float,  Device::CPU>,
  TypeDefinitions<double, Device::CPU>,
  TypeDefinitions<float,  Device::GPU>,
  TypeDefinitions<double, Device::GPU>
  > Implementations;
#else
typedef ::testing::Types <
  TypeDefinitions<float,  Device::CPU>,
  TypeDefinitions<double, Device::CPU>
  > Implementations;
#endif
TYPED_TEST_CASE(LstmTest, Implementations);


TYPED_TEST(LstmTest, ForwardInference) {
  this->allocator_->AllocateInferenceWorkspace();
  this->DoForwardInference();
}


TYPED_TEST(LstmTest, ForwardTraining) {
  this->allocator_->AllocateTrainingWorkspace();
  this->allocator_->AllocateReserved();
  this->DoForwardTraining();
}


TYPED_TEST(LstmTest, Backward) {
  this->allocator_->AllocateTrainingWorkspace();
  this->allocator_->AllocateReserved();

}

TYPED_TEST(LstmTest, CheckGradient) {
  /*
  typedef typename TypeParam::Type Type;
  constexpr Type eps = 1e-6;
  const int H = LstmTest<TypeParam>::H;
  const int W = LstmTest<TypeParam>::W;
  const int N = LstmTest<TypeParam>::N;
  const int K = LstmTest<TypeParam>::K;
  const int D = LstmTest<TypeParam>::D;

  this->allocator_->AllocateTrainingWorkspace();
  this->allocator_->AllocateReserved();


  this->DoForwardTraining();
  this->DoBackwardData();


  for (int y = 0, i = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k, ++i) {
          const auto v = this->Input()[i];
          this->Input()[i] = v + 0.5 * eps;
          this->DoForwardTraining();
          const auto fvp =
              std::accumulate(this->Output(), this->Output() + H * W * N * 4 * D, 0.0);

          this->Input()[i] = v - 0.5 * eps;
          this->DoForwardTraining();
          const auto fvm =
              std::accumulate(this->Output(), this->Output() + H * W * N * 4 * D, 0.0);

          const Type expectedGrad = (fvp - fvm) / eps;
          const Type reldiff = abs(this->GradInput()[i] - expectedGrad) /
              std::max<Type>(this->GradInput()[i], expectedGrad);
        }
      }
    }
  }
  */
}

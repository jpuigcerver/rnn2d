#ifndef TEST_LSTM_TEST_H_
#define TEST_LSTM_TEST_H_

#include <gtest/gtest.h>

#include "LstmAllocator.h"

namespace device {
struct CPU {};
struct GPU {};
}  // namespace

template <typename T, typename D>
struct TypeDefinitions {
  typedef T Type;
  typedef D Device;
};

template <typename T>
T RelativeError(const T&a, const T&b) {
  const T d = fabs(a - b);
  const T m = std::max<T>(a, b);
  if (m > 1e-9)
    return d / m;
  else
    return m;
}

template <typename T>
class LstmTest : public ::testing::Test {
 public:
  typedef typename T::Type Type;

  static constexpr int H = 11;
  static constexpr int W = 7;
  static constexpr int N = 5;
  static constexpr int K = 3;
  static constexpr int D = 2;

  LstmTest();

  void DoForwardInference();
  void DoForwardTraining();
  void DoBackwardData();
  void DoBackwardInput();

  void AllocateInference() {
    allocator_->AllocateInput();
    allocator_->AllocateOutput();
    allocator_->AllocateParameters();
    allocator_->AllocateInferenceWorkspace();
  }

  void AllocateTraining() {
    allocator_->AllocateInput();
    allocator_->AllocateOutput();
    allocator_->AllocateParameters();
    allocator_->AllocateGradInput();
    allocator_->AllocateGradOutput();
    allocator_->AllocateGradParameters();
    allocator_->AllocateTrainingWorkspace();
    allocator_->AllocateReserved();
  }

  void CopyToHost() {
    allocator_->CopyToHost();
  }

  inline Type* Input() {
    return allocator_->Input();
  }

  inline Type* Output() {
    return allocator_->Output();
  }

  inline Type* Parameters() {
    return allocator_->Parameters();
  }

  inline const int* InputShape() const {
    return allocator_->InputShape();
  }

  inline const Type* ExpectedOutputHost() const {
    return allocator_->ExpectedOutputHost();
  }

  inline const Type* OutputHost() const {
    return allocator_->OutputHost();
  }

  inline Type* GradInput() {
    return allocator_->GradInput();
  }

  inline Type* GradParameters() {
    return allocator_->GradParameters();
  }

  inline const Type* GradOutput() {
    return allocator_->GradOutput();
  }

  inline void* Workspace() {
    return allocator_->Workspace();
  }

  inline void* Reserved() {
    return allocator_->Reserved();
  }

 protected:
  std::unique_ptr< LstmAllocator<Type> > allocator_;
};

#endif  // TEST_LSTM_TEST_H_

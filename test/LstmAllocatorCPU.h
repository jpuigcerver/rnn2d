#ifndef TEST_LSTM_TEST_ALLOCATOR_CPU_H_
#define TEST_LSTM_TEST_ALLOCATOR_CPU_H_

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <rnn2d/lstm_cpu.h>

#include "LstmAllocator.h"

template <typename T>
class LstmAllocatorCPU : public LstmAllocator<T> {
 public:
  LstmAllocatorCPU(int H, int W, int N, int K, int D) :
      H(H), W(W), N(N), K(K), D(D), input_shape_(nullptr),
      input_(nullptr), output_(nullptr), parameters_(nullptr),
      gradInput_(nullptr), gradOutput_(nullptr), gradParameters_(nullptr),
      reserved_(nullptr), workspace_(nullptr) { }

  virtual ~LstmAllocatorCPU() override {
    if (input_shape_ != nullptr)
      delete [] input_shape_;
    if (input_ != nullptr)
      delete [] input_;
    if (output_ != nullptr)
      delete [] output_;
    if (parameters_ != nullptr)
      delete [] parameters_;
    if (gradInput_ != nullptr)
      delete [] gradInput_;
    if (gradOutput_ != nullptr)
      delete [] gradOutput_;
    if (gradParameters_ != nullptr)
      delete [] gradParameters_;
    if (reserved_ != nullptr)
      free(reserved_);
    if (workspace_ != nullptr)
      free(workspace_);
  }

  void AllocateInput() override {
    ASSERT_EQ(INPUT.size(), H * W * N * K);
    ASSERT_EQ(INPUT_SHAPE.size(), 2 * N);
    ASSERT_TRUE((input_ = new T[INPUT.size()]) != nullptr);
    std::copy(INPUT.begin(), INPUT.end(), input_);
    ASSERT_TRUE((input_shape_ = new int[INPUT_SHAPE.size()]) != nullptr);
    std::copy(INPUT_SHAPE.begin(), INPUT_SHAPE.end(), input_shape_);
  }

  void AllocateParameters() override {
    ASSERT_EQ(PARAMETERS.size(), 4 * (1 + K + D + D) * 5 * D);
    ASSERT_TRUE((parameters_ = new T[PARAMETERS.size()]) != nullptr);
    std::copy(PARAMETERS.begin(), PARAMETERS.end(), parameters_);
  }

  void AllocateOutput() override {
    ASSERT_EQ(EXPECTED_OUTPUT.size(), H * W * N * D * 4);
    ASSERT_TRUE((output_ = new T[EXPECTED_OUTPUT.size()]) != nullptr);
  }

  void AllocateGradInput() override {
    ASSERT_TRUE((gradInput_ = new T[INPUT.size()]) != nullptr);
  }

  void AllocateGradOutput() override {
    ASSERT_TRUE((gradOutput_ = new T[EXPECTED_OUTPUT.size()]) != nullptr);
  }

  void AllocateGradParameters() override {
    ASSERT_TRUE((gradParameters_ = new T[PARAMETERS.size()]) != nullptr);
  }

  void AllocateReserved() override;

  void AllocateInferenceWorkspace() override;

  void AllocateTrainingWorkspace() override;

  const int* InputShape() const override { return input_shape_; }

  T* Input() override { return input_; }

  T* Output() override { return output_; }

  T* Parameters() override { return parameters_; }

  T* GradInput() override { return gradInput_; }

  T* GradOutput() override { return gradOutput_; }

  T* GradParameters() override { return gradParameters_; }

  void* Workspace() override { return workspace_; }

  void* Reserved() override { return reserved_; }

  const T* OutputHost() const override { return output_; }

  const T* GradInputHost() const override { return gradInput_; }

  const T* GradParametersHost() const override { return gradParameters_; }

 private:
  int H, W, N, K, D;
  int* input_shape_;
  T* input_;
  T* output_;
  T* parameters_;
  T* gradInput_;
  T* gradOutput_;
  T* gradParameters_;
  void* reserved_;
  void* workspace_;
};

template <>
void LstmAllocatorCPU<float>::AllocateReserved() {
  const size_t reserveSize =
      rnn2d_lstm_cpu_float_training_reserve_size(H, W, N, D);
  ASSERT_TRUE((reserved_ = malloc(reserveSize)) != nullptr);
}

template <>
void LstmAllocatorCPU<double>::AllocateReserved() {
  const size_t reserveSize =
      rnn2d_lstm_cpu_double_training_reserve_size(H, W, N, D);
  ASSERT_TRUE((reserved_ = malloc(reserveSize)) != nullptr);
}

template <>
void LstmAllocatorCPU<float>::AllocateInferenceWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_cpu_float_inference_workspace_size(H, W, N, D);
  ASSERT_TRUE((workspace_ = malloc(workspaceSize)) != nullptr);
}

template <>
void LstmAllocatorCPU<double>::AllocateInferenceWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_cpu_double_inference_workspace_size(H, W, N, D);
  ASSERT_TRUE((workspace_ = malloc(workspaceSize)) != nullptr);
}

template <>
void LstmAllocatorCPU<float>::AllocateTrainingWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_cpu_float_training_workspace_size(H, W, N, D);
  ASSERT_TRUE((workspace_ = malloc(workspaceSize)) != nullptr);
}

template <>
void LstmAllocatorCPU<double>::AllocateTrainingWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_cpu_double_training_workspace_size(H, W, N, D);
  ASSERT_TRUE((workspace_ = malloc(workspaceSize)) != nullptr);
}

#endif  // TEST_LSTM_ALLOCATOR_CPU_H_

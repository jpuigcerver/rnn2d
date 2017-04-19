#ifndef TEST_LSTM_TEST_ALLOCATOR_GPU_H_
#define TEST_LSTM_TEST_ALLOCATOR_GPU_H_

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include <rnn2d/lstm_gpu.h>

#include "LstmAllocator.h"

template <typename T>
class LstmAllocatorGPU : public LstmAllocator<T> {
 public:
  LstmAllocatorGPU(int H, int W, int N, int K, int D) :
      H(H), W(W), N(N), K(K), D(D), input_shape_(nullptr),
      input_(nullptr), output_(nullptr), parameters_(nullptr),
      gradInput_(nullptr), gradOutput_(nullptr), gradParameters_(nullptr),
      outputHost_(nullptr), gradInputHost_(nullptr), gradParamHost_(nullptr),
      reserved_(nullptr), workspace_(nullptr) { }

  virtual ~LstmAllocatorGPU() override {
    if (input_shape_ != nullptr)
      cudaFree(input_shape_);
    if (input_ != nullptr)
      cudaFree(input_);
    if (output_ != nullptr)
      cudaFree(output_);
    if (parameters_ != nullptr)
      cudaFree(parameters_);
    if (gradInput_ != nullptr)
      cudaFree(gradInput_);
    if (gradOutput_ != nullptr)
      cudaFree(gradOutput_);
    if (gradParameters_ != nullptr)
      cudaFree(gradParameters_);
    if (reserved_ != nullptr)
      cudaFree(reserved_);
    if (workspace_ != nullptr)
      cudaFree(workspace_);
    if (outputHost_ != nullptr)
      delete [] outputHost_;
    if (gradInputHost_ != nullptr)
      delete [] gradInputHost_;
    if (gradParamHost_ != nullptr)
      delete [] gradParamHost_;
  }

  void AllocateInput() override {
    ASSERT_EQ(INPUT.size(), H * W * N * K);
    ASSERT_EQ(INPUT_SHAPE.size(), 2 * N);
    ASSERT_TRUE(cudaMalloc(&input_, sizeof(T) * INPUT.size()) == cudaSuccess);
    ASSERT_TRUE(cudaMalloc(&input_shape_,
                           sizeof(int) * INPUT_SHAPE.size()) == cudaSuccess);
    // Copy shape directly
    ASSERT_TRUE(cudaMemcpy(input_shape_,
                           INPUT_SHAPE.data(), sizeof(int) * INPUT_SHAPE.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
    // First cast inputs to type T, and then copy to the GPU
    std::vector<T> inputAux(INPUT.begin(), INPUT.end());
    ASSERT_TRUE(cudaMemcpy(input_,
                           inputAux.data(), sizeof(T) * inputAux.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
  }

  void AllocateParameters() override {
    ASSERT_EQ(PARAMETERS.size(), 4 * (1 + K + D + D) * 5 * D);
    ASSERT_TRUE(cudaMalloc(&parameters_,
                           sizeof(T) * PARAMETERS.size()) == cudaSuccess);
    // First cast parameters to type T, and then copy to the GPU
    std::vector<T> parametersAux(PARAMETERS.begin(), PARAMETERS.end());
    ASSERT_TRUE(cudaMemcpy(parameters_, parametersAux.data(),
                           sizeof(T) * parametersAux.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
  }

  void AllocateOutput() override {
    ASSERT_EQ(EXPECTED_OUTPUT.size(), H * W * N * D * 4);
    ASSERT_TRUE(cudaMalloc(&output_,
                           sizeof(T) * EXPECTED_OUTPUT.size()) == cudaSuccess);
    ASSERT_TRUE((outputHost_ = new T[EXPECTED_OUTPUT.size()]) != nullptr);
  }

  void AllocateGradInput() override {
    ASSERT_TRUE(cudaMalloc(&gradInput_,
                           sizeof(T) * INPUT.size()) == cudaSuccess);
    ASSERT_TRUE((gradInputHost_ = new T[INPUT.size()]) != nullptr);
  }

  void AllocateGradOutput() override {
    ASSERT_TRUE(cudaMalloc(&gradOutput_,
                           sizeof(T) * EXPECTED_OUTPUT.size()) == cudaSuccess);
  }

  void AllocateGradParameters() override {
    ASSERT_TRUE(cudaMalloc(&gradParameters_,
                           sizeof(T) * PARAMETERS.size()) == cudaSuccess);
    ASSERT_TRUE((gradParamHost_ = new T[PARAMETERS.size()]) != nullptr);
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

  const T* OutputHost() const override { return outputHost_; }

  const T* GradInputHost() const override { return gradInputHost_; }

  const T* GradParametersHost() const override { return gradParamHost_; }

  void CopyToHost() {
    // Copy output to host
    if (output_ != nullptr) {
      ASSERT_TRUE(outputHost_ != nullptr);
      ASSERT_TRUE(cudaMemcpy(outputHost_, output_,
                             sizeof(T) * EXPECTED_OUTPUT.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    // Copy gradInput to host
    if (gradInput_ != nullptr) {
      ASSERT_TRUE(gradInputHost_ != nullptr);
      ASSERT_TRUE(cudaMemcpy(gradInputHost_, gradInput_,
                             sizeof(T) * INPUT.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
    }

    if (gradParameters_ != nullptr) {
      ASSERT_TRUE(gradParamHost_ != nullptr);
      ASSERT_TRUE(cudaMemcpy(gradParamHost_, gradParameters_,
                             sizeof(T) * PARAMETERS.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
    }
  }

 private:
  const int H, W, N, K, D;
  int* input_shape_;
  T* input_;
  T* output_;
  T* parameters_;
  T* gradInput_;
  T* gradOutput_;
  T* gradParameters_;
  T* outputHost_;
  T* gradInputHost_;
  T* gradParamHost_;
  void* reserved_;
  void* workspace_;
};

template <>
void LstmAllocatorGPU<float>::AllocateReserved() {
  const size_t reserveSize =
      rnn2d_lstm_gpu_float_training_reserve_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&reserved_, reserveSize) == cudaSuccess);
}

template <>
void LstmAllocatorGPU<double>::AllocateReserved() {
  const size_t reserveSize =
      rnn2d_lstm_gpu_double_training_reserve_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&reserved_, reserveSize) == cudaSuccess);
}

template <>
void LstmAllocatorGPU<float>::AllocateInferenceWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_gpu_float_inference_workspace_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&workspace_, workspaceSize) == cudaSuccess);
}

template <>
void LstmAllocatorGPU<double>::AllocateInferenceWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_gpu_double_inference_workspace_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&workspace_, workspaceSize) == cudaSuccess);
}

template <>
void LstmAllocatorGPU<float>::AllocateTrainingWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_gpu_float_training_workspace_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&workspace_, workspaceSize) == cudaSuccess);
}

template <>
void LstmAllocatorGPU<double>::AllocateTrainingWorkspace() {
  const size_t workspaceSize =
      rnn2d_lstm_gpu_double_training_workspace_size(H, W, N, D);
  ASSERT_TRUE(cudaMalloc(&workspace_, workspaceSize) == cudaSuccess);
}

#endif  // TEST_LSTM_ALLOCATOR_GPU_H_

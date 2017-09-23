#ifndef RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_
#define RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

#include <rnn2d/activations.h>
#include <rnn2d/cuda_utils.h>

#include <rnn2d/internal/lstm2d_inference_high_mem_impl.h>

#include <rnn2d/internal/gpu/math.h>

namespace rnn2d {
namespace internal {
namespace gpu {

template<typename T, class C>
__global__
void kernel_init_Q_with_bias(Lstm2dInferenceHighMemImpl<T, C> lstm);

template<typename T, class C>
class Lstm2dInferenceHighMemImpl :
    public ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C> {
 public:
  Lstm2dInferenceHighMemImpl(const int K, const int D,
                             cublasHandle_t cublas_handle) :
  ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>(K, D),
  cublas_handle_(cublas_handle) {}

 protected:
  using Rnn2dInferenceImpl<T>::H_;
  using Rnn2dInferenceImpl<T>::W_;
  using Rnn2dInferenceImpl<T>::N_;
  using Rnn2dInferenceImpl<T>::K_;
  using Rnn2dInferenceImpl<T>::D_;
  using Rnn2dInferenceImpl<T>::input_;
  using Rnn2dInferenceImpl<T>::shape_;
  using Rnn2dInferenceImpl<T>::param_;
  using Rnn2dInferenceImpl<T>::output_;
  using Rnn2dInferenceImpl<T>::wspace_;

  using ::rnn2d::internal::Lstm2dInferenceHighMemImpl<T, C>::G_;

  rnn2dStatus_t ForwardBias() override {
    kernel_init_Q_with_bias<<<GRID_SIZE, BLOCK_SIZE>>>(*this);
    const cudaStatus_t status = cudaPeekAtLastError();
    RNN2D_CHECK_CUDA_AND_RETURN_ERROR(status, RNN2D_STATUS_EXECUTION_FAILED,
                                      "CUDA kernel launch failed: ");
    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardInput() override {
    static std::vector<const T*> ptrs_cpu(12, nullptr);
    ptrs_cpu[0]  = ptrs_cpu[1] = ptrs_cpu[2] = ptrs_cpu[3] = input_;
    ptrs_cpu[4]  = W(0, 0, 0, 0);
    ptrs_cpu[5]  = W(1, 0, 0, 0);
    ptrs_cpu[6]  = W(2, 0, 0, 0);
    ptrs_cpu[7]  = W(3, 0, 0, 0);
    ptrs_cpu[8]  = Q(0, 0, 0, 0, 0, 0);
    ptrs_cpu[9]  = Q(1, 0, 0, 0, 0, 0);
    ptrs_cpu[10] = Q(2, 0, 0, 0, 0, 0);
    ptrs_cpu[12] = Q(3, 0, 0, 0, 0, 0);

    const T** ptrs_gpu =
        reinterpret_cast<const T**>(wspace_ + 4 * H_ * W_ * N_ * G_ * D_);
    status = cudaMemcpy(ptrs_gpu, ptrs_cpu, 12 * sizeof(const T**),
                        cudaMemcpyHostToDevice);
    RNN2D_CHECK_CUDA_AND_RETURN_ERROR(status, RNN2D_STATUS_ALLOC_FAILED,
                                      "CUDA memcpy failed: ");
    const cublasStatus_t status2 =
        gemm_gpu_batched<T>(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            H_ * W_ * N_, G_ * D_, K_, 1.0, ptrs_gpu, K_,
                            ptrs_gpu_ + 4, G_ * D_,
                            1.0, const_cast<T**>(ptrs_gpu) + 8, G_ * D_, 4);
    RNN2D_CHECK_CUBLAS_AND_RETURN_ERROR(status2, RNN2D_STATUS_INTERNAL_ERROR,
                                        "CuBLAS launch error: ");
    return RNN2D_STATUS_SUCCESS;
  }

  rnn2dStatus_t ForwardPreviousOutputs(const int L, const int t) override {

    return RNN2D_STATUS_SUCCESS;
  }

 private:
  constexpr int BLOCK_SIZE = 256;
  constexpr int GRID_SIZE  = 128;
  cublasHandle_t cublas_handle_;
};

template<typename T, class C>
__global__
void kernel_init_Q_with_bias(Lstm2dInferenceHighMemImpl<T, C> lstm) {
  const int H = lstm.GetH();
  const int W = lstm.GetW();
  const int N = lstm.GetN();
  const int D = lstm.GetD();
  const int G = lstm.GetG();

  for (int ii = thGi; ii < 4 * H * W * N * G * D; ii += NTG) {
    const int d = ii % D;                      // d \in [0 ... D-1]
    const int g = (ii / D) % G;                // g \in [0 ... G]
    const int n = (ii / (G * D)) % N;          // n \in [0 ... N-1]
    const int x = (ii / (N * G * D)) % W;      // x \in [0 ... W-1]
    const int y = (ii / (W * N * G * D)) % H;  // y \in [0 ... H-1]
    const int z = (ii / (H * W * N * G * D));  // z \in [0 ... 3]
    *lstm.Q(z, y, x, n, g, d) = *lstm.B(P, z, g, d);
  }
}

}  // namespace gpu
}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_GPU_LSTM2D_INFERENCE_HIGH_MEM_IMPL_H_

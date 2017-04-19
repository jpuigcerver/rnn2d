#include "../include/rnn2d/lstm_gpu.h"
#include "../include/rnn2d/cuda_utils.h"

#include <benchmark/benchmark.h>
#include <curand.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>

using thrust::device_vector;

#define DEFAULT_H 32
#define DEFAULT_W 256
#define DEFAULT_N 16
#define DEFAULT_K 16
#define DEFAULT_D 16

#define CHECK_CURAND(x) CHECK_EQ((x), CURAND_STATUS_SUCCESS)

template <typename T>
class LstmWrapper {
 public:
  LstmWrapper(const int H, const int W, const int N, const int K, const int D) :
      H_(H), W_(W), N_(N), K_(K), D_(D) {
    input_      = data_.data().get();
    output_     = input_      + GetInputSize(H_, W_, N_, K_);
    param_      = output_     + GetOutputSize(H_, W_, N_, D_);
    gradInput_  = param_      + GetParamSize(K_, D_);
    gradOutput_ = gradInput_  + GetInputSize(H_, W_, N_, K_);
    gradParam_  = gradOutput_ + GetOutputSize(H_, W_, N_, K_);
    wspace_ = static_cast<void *>(wrspace_.data().get());
    rspace_ = static_cast<void *>(
        wrspace_.data().get() + std::max(
            GetTrainingWorkspaceSize(H_, W_, N_, D_),
            GetInferenceWorkspaceSize(H_, W_, N_, D_)));
  }

  static void Initialize(const int H, const int W, const int N, const int K,
                         const int D) {
    const size_t data_size =
        2 * (GetInputSize(H, W, N, K) +
             GetOutputSize(H, W, N, D) +
             GetParamSize(K, D));
    const size_t wrspace_size =
        std::max(GetTrainingWorkspaceSize(H, W, N, D),
                 GetInferenceWorkspaceSize(H, W, N, D)) +
        GetReserveSize(H, W, N, D);
    const size_t data_size_mb =
        data_size * sizeof(T) / static_cast<float>(1 << 20);
    const size_t wrspace_size_mb =
        wrspace_size / static_cast<float>(1 << 20);
    LOG(INFO) << "Allocating " << data_size_mb + wrspace_size_mb << "MB in the GPU...";
    data_.resize(data_size);
    wrspace_.resize(wrspace_size);
    LOG(INFO) << "Filling " << data_size_mb << "MB with random numbers...";
    GenerateUniform(data_.data().get(), data_size);
    LOG(INFO) << "Done!";
  }

  static void Destroy() {
    data_.clear();
    wrspace_.clear();
    data_.shrink_to_fit();
    wrspace_.shrink_to_fit();
  }

  inline void ForwardInference();
  inline void ForwardTraining();
  inline void BackwardData();
  inline void BackwardParam();


 private:
  static size_t GetInputSize(const int H, const int W, const int N, const int K) {
    return rnn2d_lstm_input_nelem(H, W, N, K);
  }

  static size_t GetOutputSize(const int H, const int W, const int N, const int D) {
    return rnn2d_lstm_output_nelem(H, W, N, D);
  }

  static size_t GetParamSize(const int K, const int D) {
    return rnn2d_lstm_parameters_nelem(K, D);
  }

  static size_t GetInferenceWorkspaceSize(const int H, const int W, const int N, const int D);
  static size_t GetTrainingWorkspaceSize(const int H, const int W, const int N, const int D);
  static size_t GetReserveSize(const int H, const int W, const int N, const int D);
  static void GenerateUniform(T* data, size_t n);

  static device_vector<T> data_;
  static device_vector<char> wrspace_;
  const int H_, W_, N_, K_, D_;
  T *input_, *output_, *param_, *gradInput_, *gradOutput_, *gradParam_;
  void *wspace_, *rspace_;
};

template <>
device_vector<float> LstmWrapper<float>::data_ = device_vector<float>();
template <>
device_vector<char> LstmWrapper<float>::wrspace_ = device_vector<char>();

template <>
device_vector<double> LstmWrapper<double>::data_ = device_vector<double>();
template <>
device_vector<char> LstmWrapper<double>::wrspace_ = device_vector<char>();

template <>
size_t LstmWrapper<float>::GetInferenceWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_float_inference_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetInferenceWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_double_inference_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<float>::GetTrainingWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_float_training_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetTrainingWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_double_training_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<float>::GetReserveSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_float_training_reserve_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetReserveSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_gpu_double_training_reserve_size(H, W, N, D);
}

template <>
void LstmWrapper<float>::GenerateUniform(float* data, size_t n) {
  curandGenerator_t gen;
  CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  CHECK_CURAND(curandGenerateUniform(gen, data, n));
  CHECK_CUDA_CALL(cudaDeviceSynchronize());
  CHECK_CURAND(curandDestroyGenerator(gen));
}

template <>
void LstmWrapper<double>::GenerateUniform(double* data, size_t n) {
  curandGenerator_t gen;
  CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  CHECK_CURAND(curandGenerateUniformDouble(gen, data, n));
  CHECK_CUDA_CALL(cudaDeviceSynchronize());
  CHECK_CURAND(curandDestroyGenerator(gen));
}

template <>
void LstmWrapper<float>::ForwardInference() {
  rnn2d_lstm_gpu_float_fw_inference(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_);
}

template <>
void LstmWrapper<double>::ForwardInference() {
  rnn2d_lstm_gpu_double_fw_inference(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_);
}

template <>
void LstmWrapper<float>::ForwardTraining() {
  rnn2d_lstm_gpu_float_fw_training(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::ForwardTraining() {
  rnn2d_lstm_gpu_double_fw_training(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_, rspace_);
}

template <>
void LstmWrapper<float>::BackwardData() {
  rnn2d_lstm_gpu_float_bw_data(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, gradOutput_, gradInput_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::BackwardData() {
  rnn2d_lstm_gpu_double_bw_data(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, gradOutput_, gradInput_, wspace_, rspace_);
}

template <>
void LstmWrapper<float>::BackwardParam() {
  rnn2d_lstm_gpu_float_bw_param(H_, W_, N_, K_, D_, input_, output_, 1.0, gradParam_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::BackwardParam() {
  rnn2d_lstm_gpu_double_bw_param(H_, W_, N_, K_, D_, input_, output_, 1.0, gradParam_, wspace_, rspace_);
}

template <typename T>
static void BM_fw_inference(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.ForwardInference();
  while (state.KeepRunning()) {
    lstm.ForwardInference();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}

template <typename T>
static void BM_fw_training(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.ForwardTraining();
  while (state.KeepRunning()) {
    lstm.ForwardTraining();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}

template <typename T>
static void BM_bw_data(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.BackwardData();
  while (state.KeepRunning()) {
    lstm.BackwardData();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}

template <typename T>
static void BM_bw_param(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.BackwardParam();
  while (state.KeepRunning()) {
    lstm.BackwardParam();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}

template <typename T>
static void BM_bw_ALL(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.BackwardData();
  while (state.KeepRunning()) {
    lstm.BackwardData();
    lstm.BackwardParam();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}


#define INSTANTIATE_BENCHMARKS(TYPE)                                    \
  BENCHMARK_TEMPLATE(BM_fw_inference, TYPE)                             \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  BENCHMARK_TEMPLATE(BM_fw_training, TYPE)                              \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  BENCHMARK_TEMPLATE(BM_bw_data, TYPE)                                  \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  BENCHMARK_TEMPLATE(BM_bw_param, TYPE)                                 \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  BENCHMARK_TEMPLATE(BM_bw_ALL, TYPE)                                   \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime()

INSTANTIATE_BENCHMARKS(float);
INSTANTIATE_BENCHMARKS(double);

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  benchmark::Initialize(&argc, argv);
  // Get number of CUDA devices and their names.
  int numDevices = 0;
  CHECK_CUDA_CALL(cudaGetDeviceCount(&numDevices));
  cudaDeviceProp* dp = new cudaDeviceProp[numDevices];
  for (int d = 0; d < numDevices; ++d) {
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&dp[d], d));
  }
  // Set the CUDA device to use
  if (argc > 1) {
    CHECK_CUDA_CALL(cudaSetDevice(atoi(argv[1])));
  }
  // Display the name of the CUDA device being used
  int curDevice = 0;
  CHECK_CUDA_CALL(cudaGetDevice(&curDevice));
  LOG(INFO) << "Found " << numDevices << " CUDA devices, using device "
            << curDevice << ": " << dp[curDevice].name;

  LstmWrapper<float>::Initialize(DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D);
  LstmWrapper<double>::Initialize(DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D);
  benchmark::RunSpecifiedBenchmarks();
  LstmWrapper<float>::Destroy();
  LstmWrapper<double>::Destroy();
  return 0;
}

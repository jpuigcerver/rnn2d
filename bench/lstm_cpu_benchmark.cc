#include "../include/rnn2d/lstm_cpu.h"

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <glog/logging.h>

using std::vector;

static std::default_random_engine RNG;

#define DEFAULT_H 32
#define DEFAULT_W 256
#define DEFAULT_N 16
#define DEFAULT_K 16
#define DEFAULT_D 16

template <typename T, std::size_t N = sizeof(T)>
class aligned_allocator {
 public:
  typedef T value_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T* pointer;
  typedef const T* const_pointer;

  typedef T & reference;
  typedef const T & const_reference;

  inline aligned_allocator() noexcept {}
  inline ~aligned_allocator() noexcept {}

  inline pointer address(reference x) const noexcept { return &x; }
  inline const_pointer address(const_reference x) const noexcept { return &x; }

  inline pointer allocate (size_type n, const_pointer hint=0) {
    return (pointer)::aligned_alloc(N, n);
  }

  inline void deallocate(pointer p, size_type n) {
    free(p);
  }

  inline size_type max_size() const noexcept { return size_type (-1) / sizeof (value_type); }

  template <class U, class... Args>
  inline void construct(U* p, Args&&... args) { ::new ((void*)p) U (std::forward<Args>(args)...); }

  template <class U>
  inline void destroy(U* p) { p->~U(); }

  template <class Type>
  struct rebind {
    typedef aligned_allocator<Type, N> other;
  };
};



template <typename T>
class LstmWrapper {
 public:
  LstmWrapper(const int H, const int W, const int N, const int K, const int D) :
      H_(H), W_(W), N_(N), K_(K), D_(D) {
    input_      = data_.data();
    output_     = input_      + GetInputSize(H_, W_, N_, K_);
    param_      = output_     + GetOutputSize(H_, W_, N_, D_);
    gradInput_  = param_      + GetParamSize(K_, D_);
    gradOutput_ = gradInput_  + GetInputSize(H_, W_, N_, K_);
    gradParam_  = gradOutput_ + GetOutputSize(H_, W_, N_, K_);
    wspace_ = static_cast<void *>(wrspace_.data());
    rspace_ = static_cast<void *>(
        wrspace_.data() + std::max(
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
    LOG(INFO) << "Allocating " << data_size_mb + wrspace_size_mb << "MB in the CPU...";
    data_.resize(data_size);
    wrspace_.resize(wrspace_size);
    LOG(INFO) << "Filling " << data_size_mb << "MB with random numbers...";
    std::uniform_real_distribution<T> udist(0.0, 1.0);
    std::generate(data_.begin(), data_.end(), [&udist]{ return udist(RNG); });
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
  inline void BackwardGates();
  inline void BackwardInput();
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

  static vector<T> data_;
  static vector<char, aligned_allocator<char, 8>> wrspace_;
  const int H_, W_, N_, K_, D_;
  T *input_, *output_, *param_, *gradInput_, *gradOutput_, *gradParam_;
  void *wspace_, *rspace_;
};

template <>
vector<float> LstmWrapper<float>::data_ = vector<float>();
template <>
vector<char, aligned_allocator<char, 8>> LstmWrapper<float>::wrspace_ = vector<char, aligned_allocator<char, 8>>();

template <>
vector<double> LstmWrapper<double>::data_ = vector<double>();
template <>
vector<char, aligned_allocator<char, 8>> LstmWrapper<double>::wrspace_ = vector<char, aligned_allocator<char, 8>>();

template <>
size_t LstmWrapper<float>::GetInferenceWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_float_inference_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetInferenceWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_double_inference_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<float>::GetTrainingWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_float_training_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetTrainingWorkspaceSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_double_training_workspace_size(H, W, N, D);
}

template <>
size_t LstmWrapper<float>::GetReserveSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_float_training_reserve_size(H, W, N, D);
}

template <>
size_t LstmWrapper<double>::GetReserveSize(const int H, const int W, const int N, const int D) {
  return rnn2d_lstm_cpu_double_training_reserve_size(H, W, N, D);
}

template <>
void LstmWrapper<float>::ForwardInference() {
  rnn2d_lstm_cpu_float_fw_inference(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_);
}

template <>
void LstmWrapper<double>::ForwardInference() {
  rnn2d_lstm_cpu_double_fw_inference(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_);
}

template <>
void LstmWrapper<float>::ForwardTraining() {
  rnn2d_lstm_cpu_float_fw_training(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::ForwardTraining() {
  rnn2d_lstm_cpu_double_fw_training(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, wspace_, rspace_);
}

template <>
void LstmWrapper<float>::BackwardGates() {
  rnn2d_lstm_cpu_float_bw_workspace(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, gradOutput_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::BackwardGates() {
  rnn2d_lstm_cpu_double_bw_workspace(H_, W_, N_, K_, D_, input_, nullptr, param_, output_, gradOutput_, wspace_, rspace_);
}

template <>
void LstmWrapper<float>::BackwardInput() {
  rnn2d_lstm_cpu_float_bw_input(H_, W_, N_, K_, D_, param_, 1.0, gradInput_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::BackwardInput() {
  rnn2d_lstm_cpu_double_bw_input(H_, W_, N_, K_, D_, param_, 1.0, gradInput_, wspace_, rspace_);
}

template <>
void LstmWrapper<float>::BackwardParam() {
  rnn2d_lstm_cpu_float_bw_param(H_, W_, N_, K_, D_, input_, output_, 1.0, gradParam_, wspace_, rspace_);
}

template <>
void LstmWrapper<double>::BackwardParam() {
  rnn2d_lstm_cpu_double_bw_param(H_, W_, N_, K_, D_, input_, output_, 1.0, gradParam_, wspace_, rspace_);
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
static void BM_bw_gates(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.BackwardGates();
  while (state.KeepRunning()) {
    lstm.BackwardGates();
  }
  state.SetItemsProcessed(state.iterations() * H * W * N * K * D);
}

template <typename T>
static void BM_bw_input(benchmark::State& state) {
  const int H = state.range(0);
  const int W = state.range(1);
  const int N = state.range(2);
  const int K = state.range(3);
  const int D = state.range(4);
  LstmWrapper<T> lstm(H, W, N, K, D);
  lstm.BackwardInput();
  while (state.KeepRunning()) {
    lstm.BackwardInput();
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
  lstm.BackwardGates();
  while (state.KeepRunning()) {
    lstm.BackwardGates();
    lstm.BackwardInput();
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
  BENCHMARK_TEMPLATE(BM_bw_gates, TYPE)                                 \
  ->Args({DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D})       \
  ->Unit(benchmark::kMicrosecond)                                       \
  ->UseRealTime();                                                      \
                                                                        \
  BENCHMARK_TEMPLATE(BM_bw_input, TYPE)                                 \
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
  LstmWrapper<float>::Initialize(DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D);
  LstmWrapper<double>::Initialize(DEFAULT_H, DEFAULT_W, DEFAULT_N, DEFAULT_K, DEFAULT_D);
  benchmark::RunSpecifiedBenchmarks();
  LstmWrapper<float>::Destroy();
  LstmWrapper<double>::Destroy();
  return 0;
}

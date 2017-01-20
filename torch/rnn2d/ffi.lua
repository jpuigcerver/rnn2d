local ffi = require('ffi')

ffi.cdef[[
void rnn2d_lstm_cpu_float_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_cpu_float_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_cpu_float_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    const float* output, const float* dOutput, float* workspace);

void rnn2d_lstm_cpu_float_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const float* param, const float scale, float* dInput, float* workspace);

void rnn2d_lstm_cpu_float_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const float* output, const float scale,
    float* dParam, float* workspace);

void rnn2d_lstm_cpu_double_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_cpu_double_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_cpu_double_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    const double* output, const double* dOutput, double* workspace);

void rnn2d_lstm_cpu_double_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const double* param, const double scale, double* dInput, double* workspace);

void rnn2d_lstm_cpu_double_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const double* output, const double scale,
    double* dParam, double* workspace);

void rnn2d_lstm_gpu_float_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_gpu_float_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    float* output, float* workspace);

void rnn2d_lstm_gpu_float_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const int* shape, const float* param,
    const float* output, const float* dOutput, float* workspace);

void rnn2d_lstm_gpu_float_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const float* param, const float scale, float* dInput, float* workspace);

void rnn2d_lstm_gpu_float_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const float* input, const float* output, const float scale,
    float* dParam, float* workspace);

void rnn2d_lstm_gpu_double_fw_inference(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_gpu_double_fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    double* output, double* workspace);

void rnn2d_lstm_gpu_double_bw_workspace(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const int* shape, const double* param,
    const double* output, const double* dOutput, double* workspace);

void rnn2d_lstm_gpu_double_bw_input(
    const int H, const int W, const int N, const int K, const int D,
    const double* param, const double scale, double* dInput, double* workspace);

void rnn2d_lstm_gpu_double_bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const double* input, const double* output, const double scale,
    double* dParam, double* workspace);

void rnn2d_tile_cpu_float_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* input,
    float* output);

void rnn2d_tile_cpu_float_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* dOutput,
    float* dInput);

void rnn2d_tile_cpu_double_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* input,
    double* output);

void rnn2d_tile_cpu_double_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* dOutput,
    double* dInput);

void rnn2d_tile_gpu_float_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* input,
    float* output);

void rnn2d_tile_gpu_float_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* dOutput,
    float* dInput);

void rnn2d_tile_gpu_double_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* input,
    double* output);

void rnn2d_tile_gpu_double_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* dOutput,
    double* dInput);
]]

-- Try to find the library with the CPU implementations
local RNN2D_LIBCPU_PATH = os.getenv('RNN2D_LIBCPU_PATH')
if RNN2D_LIBCPU_PATH then
  rnn2d.cpu = ffi.load(RNN2D_LIBCPU_PATH)
else
  local libnames = {'librnn2d_cpu.so', 'librnn2d_cpu.dylib'}
  local ok = false
  for i=1,#libnames do
    ok = pcall(function () rnn2d.cpu = ffi.load(libnames[i]) end)
    if ok then break; end
  end
end

-- Try to find the library with the GPU implementations
local RNN2D_LIBGPU_PATH = os.getenv('RNN2D_LIBGPU_PATH')
if RNN2D_LIBGPU_PATH then
  rnn2d.gpu = ffi.load(RNN2D_LIBGPU_PATH)
else
  local libnames = {'librnn2d_gpu.so', 'librnn2d_gpu.dylib'}
  local ok = false
  for i=1,#libnames do
    ok = pcall(function () rnn2d.gpu = ffi.load(libnames[i]) end)
    if ok then break; end
  end
end

if not rnn2d.cpu and not rnn2d.gpu then
  error([[Neither librnn2d_cpu or librnn2d_gpu were found in your system.
Please, add the directory where these dynamic libraries are located to your
LD_LIBRARY_PATH, or alternatively, set the variables RNN2D_LIBCPU_PATH and
RNN2D_LIBGPU_PATH so that they point the the full path of each library.
For example: export RNN2D_LIBCPU_PATH="$HOME/.local/lib/librnn2d_cpu.so"]])
end

return rnn2d

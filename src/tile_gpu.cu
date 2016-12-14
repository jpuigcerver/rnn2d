#include <rnn2d/tile_gpu.h>

#include <glog/logging.h>

#include <rnn2d/tile_impl.h>
#include <rnn2d/cuda_utils.h>

// IMPORTANT: TILE_H * TILE_W * TILE_Z <= 1024
#define TILE_H 8
#define TILE_W 32
#define TILE_Z 4

template <typename T>
__global__
void kernel_fw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* I, T* O) {
  const int o_W = DIV_UP(W, Kw);                 // width of the output image
  const int o_D = D * Kh * Kw;                   // depth of the output image
  __shared__ T shared[TILE_H][TILE_W][TILE_Z];   // shared memory

  for (int i_y = thGy; i_y < H; i_y += NTGy) {
    for (int i_x = thGx; i_x < W; i_x += NTGx) {
      for (int z = thGz; z < N * D; z += NTGz) {
        const int n   = thGz / D;
        const int i_d = thGz % D;
        // Copy input to shared memory
        if ((S != nullptr && i_y < S[2 * n] && i_x < S[2 * n + 1]) ||
            (i_y < H && i_x < W)) {
          shared[threadIdx.y][threadIdx.x][threadIdx.z] =
              *I_ptr(I, i_y, i_x, n, i_d);
        } else {
          shared[threadIdx.y][threadIdx.x][threadIdx.z] = 0;
        }
        __syncthreads();
        // Copy to the output
        const int o_y = i_y / Kh;
        const int o_x = i_x / Kw;
        const int o_d = i_d + (i_x % Kw) * D + (i_y % Kh) * Kw * D;
        *O_ptr(O, o_y, o_x, n, o_d) =
            shared[threadIdx.y][threadIdx.x][threadIdx.z];
      }
    }
  }
}

template <typename T>
__global__
void kernel_bw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* dO, T* dI) {
  const int o_H = DIV_UP(H, Kh);                 // height of the output image
  const int o_W = DIV_UP(W, Kw);                 // width of the output image
  const int o_D = D * Kh * Kw;                   // depth of the output image
  __shared__ T shared[TILE_H][TILE_W][TILE_Z];   // shared memory

  for (int o_y = thGy; o_y < o_H; o_y += NTGy) {
    for (int o_x = thGx; o_x < o_W; o_x += NTGx) {
      for (int z = thGz; z < N * o_D; z += NTGz) {
        const int n   = thGz / o_D;
        const int o_d = thGz % o_D;
        // Copy dOutput to shared memory
        shared[threadIdx.y][threadIdx.x][threadIdx.z] =
            *O_ptr(dO, o_y, o_x, n, o_d);
        __syncthreads();
        // Copy to the dInput memory
        const int i_d = o_d % D;                // channel on the input image
        const int j  = (o_d / D) % Kw;
        const int i  = (o_d / (D * Kw)) % Kh;
        const int i_x = Kw * o_x + j;           // x-coordinate on the input
        const int i_y = Kh * o_y + i;           // y-coordinate on the input
        if (i_y < H && i_x < W && i_d < D) {
          *I_ptr(dI, i_y, i_x, n, i_d) =
              shared[threadIdx.y][threadIdx.x][threadIdx.z];
        }
      }
    }
  }
}

template <typename T>
inline void fw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* I, T* O) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  // initialize all outputs to 0
  CHECK_CUDA_CALL(
      cudaMemset(O, 0, sizeof(T) * RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, Kh, Kw)));

  // copy all inputs to the appropiate location of the outputs
  const dim3 block_size(TILE_W, TILE_H, TILE_Z);
  const dim3 grid_size(NUM_BLOCKS(W, TILE_W),
                       NUM_BLOCKS(H, TILE_H),
                       NUM_BLOCKS(N * D, TILE_Z));
  kernel_fw<T><<<grid_size, block_size>>>(H, W, N, D, Kh, Kw, S, I, O);
  CHECK_LAST_CUDA_CALL();
}

template <typename T>
inline void bw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* dO, T* dI) {
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(dI);
  // initialize all gradInputs to 0
  CHECK_CUDA_CALL(
      cudaMemset(dI, 0, sizeof(T) * RNN2D_TILE_INPUT_SIZE(H, W, N, D)));

  // copy all inputs to the appropiate location of the outputs
  const int o_H = DIV_UP(H, Kh);               // height of the output image
  const int o_W = DIV_UP(W, Kw);               // width of the output image
  const int o_D = D * Kh * Kw;                 // depth of the output image
  const dim3 block_size(TILE_W, TILE_H, TILE_Z);
  const dim3 grid_size(NUM_BLOCKS(o_W, TILE_W),
                       NUM_BLOCKS(o_H, TILE_H),
                       NUM_BLOCKS(N * o_D, TILE_Z));
  kernel_bw<T><<<grid_size, block_size>>>(H, W, N, D, Kh, Kw, S, dO, dI);
  CHECK_LAST_CUDA_CALL();
}

extern "C" {
  DEFINE_WRAPPERS(gpu, float)
  DEFINE_WRAPPERS(gpu, double)
}  // extern "C"

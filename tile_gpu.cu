#include "tile_gpu.h"

#include <glog/logging.h>

#include "tile_xxx.h"
#include "cuda_utils.h"

// IMPORTANT: TILE_H * TILE_W * TILE_Z <= 1024
#define TILE_H 8
#define TILE_W 32
#define TILE_Z 4


template <typename T>
__global__
void kernel_fw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* I, T* O) {
  const int o_H = DIV_UP(H, Kh);                 // height of the output image
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
inline void fw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* I, T* O) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  const dim3 block_size(TILE_H, TILE_W, TILE_Z);
  const dim3 grid_size(NUM_BLOCKS(H, TILE_H),
                       NUM_BLOCKS(W, TILE_W),
                       NUM_BLOCKS(N * D, TILE_Z));
  kernel_fw<T><<<grid_size, block_size>>>(H, W, N, D, Kh, Kw, S, I, O);
  CHECK_LAST_CUDA_CALL();
}

template <typename T>
inline void bw(const int H, const int W, const int N, const int D,
               const int Kh, const int Kw, const int* S, const T* dO, T* dI) {

}

extern "C" {
  DEFINE_WRAPPERS(gpu, float)
  DEFINE_WRAPPERS(gpu, double)
}  // extern "C"

#include "math_cpu.h"

template <typename T>
inline T sigmoid(const T x) {
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
}

// 2D-LSTM forward pass running on the CPU
// H -> maximum height
// W -> maximum width
// N -> batch size
// K -> number of channels
// I -> input data (layout: H x W x N x K)
// S -> input sizes (height and width of each sample, layout: N x 2)
// D -> number of LSTM cells
// P -> params (layout: [20 * D] (b) + [K x 20 * D] (iW) + [D x 40 * D] (rW))
// O -> output data (layout: H x W x N X (4 * D))
template <typename T, typename FC, typename FO>
void lstm_2d_fwd_cpu(const int H, const int W, const int N, const int K,
                     const T* I, const int* S, const int D, const T* P, T* O) {
  // [d * 0, d * D - 1]         -> input gates  (size: D)
  // [d * D, d * 2 * D - 1]     -> output gates (size: D)
  // [d * 2 * D, d * 4 * D - 1] -> forget gates (size: 2 * D)
  // [d * 4 * D, d * 5 * D - 1] -> cells        (size: D)
  T* tmp = new T[4 * H * W * N * 5 * D];

  // Bias in each direction
  const T** b[4] = {
    P,
    P +  5 * D,
    P + 10 * D,
    P + 15 * D
  };
  // Input weights in each direction
  const T** iW[4] = {
    P + 20 * D,
    P + 20 * D +  5 * D * K,
    P + 20 * D + 10 * D * K,
    P + 20 * D + 15 * D * K
  };
  // Recurrent weights in each direction
  const T*** rW[4][2] = {
    {P + 20 * D + 20 * D * K,
     P + 20 * D + 20 * D * K + 5 * D * D},
    {P + 20 * D + 20 * D * K + 10 * D * D,
     P + 20 * D + 20 * D * K + 15 * D * D},
    {P + 20 * D + 20 * D * K + 20 * D * D,
     P + 20 * D + 20 * D * K + 25 * D * D},
    {P + 20 * D + 20 * D * K + 30 * D * D,
     P + 20 * D + 20 * D * K + 35 * D * D}
  };

  // Initialize bias
  #pragma omp parallel for
  for (int i = 0; i < 4 * H * W * N * 5 * D; ++i) {
    const int k = i % (5 * D);
    const int n = (i / (5 * D)) % N;
    const int x = (i / (5 * D * N)) % W;
    const int y = (i / (5 * D * N * W)) % H;
    const int d = (i / (5 * D * N * W * H));
    tmp[i] = (y < S[2 * n] && x < S[2 * n + 1]) ? b[d][k] : 0.0;
  }

  // Multiply inputs by weights: tmp += W * input
  // Notice that the multiplication is done in column-major order
  // These 4 matrix multiplications are independent and can be performed
  // in parallel.
  #pragma omp parallel for
  for (int d = 0; d < 4; ++d) {
    gemm<T>('N', 'N', 5 * D, H * W * N, K,
            1.0, iW[d], 5 * D,
            I, K,
            1.0, tmp + d * H * W * N * 5 * D, 5 * D);
  }

  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int z = 0; z < H + W - 1; ++z) {
    // (y, x) coordinates of the first element in the z-th diagonals
    const int zyx[4][2] = {
      // origin: top-left
      {z < W ? (0    ) : (z - W + 1    ), z < W ? (z        ) : (W - 1)},
      // origin: top-right
      {z < W ? (0    ) : (z - W + 1    ), z < W ? (W - z - 1) : (0    )},
      // origin: bottom-left
      {z < W ? (H - 1) : (H + W - z - 1), z < W ? (z        ) : (W - 1)},
      // origin: bottom-right
      {z < W ? (H - 1) : (H + W - z - 1), z < W ? (W - z - 1) : (0    )},
    };
    // diagonal directions
    const int zim[4][2] = {
      {+1, -1},  // origin: top-left
      {+1, +1},  // origin: top-right
      {-1, -1},  // origin: bottom-left
      {-1, +1}   // origin: bottom-right
    };
    // neighboring pixels in each direction
    const int zin[4][2] = {
      {-1, -1},  // origin: top-left,     ngbrs(x, y) = {(x, y-1), (x-1, y)}
      {-1, +1},  // origin: top-right,    ngbrs(x, y) = {(x, y-1), (x+1, y)}
      {+1, -1},  // origin: bottom-left,  ngbrs(x, y) = {(x, y+1), (x-1, y)}
      {+1, +1},  // origin: bottom-right, ngbrs(x, y) = {(x, y+1), (x+1, y)}
    };
    // Multiply neighboring pixels by recurrent connections
    #pragma omp parallel for
    for (int i = 0; i < 4 * std::min(H, W); ++i) {
      // d-th direction: top-left, top-right, bottom-left, bottom-right
      const int d = i / std::min(H, W);
      // j-th element in the z-diagonal
      const int j = i % std::min(H, W);
      // (y, x) coords of the j-th element in the z-th diagonal in the
      // d-th direction.
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      if (y < 0 || x < 0 || y >= H || x >= W) continue;
      T* tmp_d_yx = tmp + ((d * H * W) + (y * W + x)) * N * 5 * D;
      if (y + zin[d][0] >= 0 && y + zin[d][0] < H - 1) {
        const T* O_yx_10 =
            O + ((y + zin[d][0]) * W + (x            )) * N * D * 4;
        gemm<T>('N', 'N', N, D * 5, D,
                1.0, O_yx_10, D,
                rW[d][0], D * 5,
                1.0, tmp_d_yx, D * 5);
      }
      if (x + zin[d][1] >= 0 && x + zin[d][1] < W - 1) {
        const T* O_yx_01 =
            O + ((y            ) * W + (x + zin[d][1])) * N * D * 4;
        gemm<T>('N', 'N', N, D * 5, D,
                1.0, O_yx_01, D,
                rW[d][1], D * 5,
                1.0, tmp_d_yx, D * 5);
      }
    }
    // Apply activation functions for gates and cell
    #pragma omp parallel for
    for (int i = 0; i < 4 * std::min(H, W) * N * 5 * D; ++i) {
      const int k = i % D;
      const int g = (i / D) % 5;
      const int n = (i / (5 * D)) % N;
      const int j = (i / (N * 5 * D)) % std::min(H, W);
      const int d = (i / (std::min(H, W) * N * 5 * D));
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      if (y < 0 || x < 0 || y >= H || x >= W) continue;
      // TODO
      T* t =
          tmp[((d * H * W) + (y * W + x)) * N * 5 * D + n * 5 * D + g * D + d];

      *t = g < 4 ? sigmoid(*t) : FC::fn(*t);
    }
    #pragma omp parallel for
    for (int i = 0; i < 4 * std::min(H, W) * N * D; ++i) {
      const int k = i % D;
      const int n = (i / D) % N;
      const int j = (i / (N * D)) % std::min(H, W);
      const int d = (i / (std::min(H, W) * N * D));
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      if (y < 0 || x < 0 || y >= H || x >= W) continue;
      // TODO: c(y,x) =
    }
  }
}

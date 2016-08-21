#ifndef RNN2D_SRC_LSTM_CPU_H_
#define RNN2D_SRC_LSTM_CPU_H_

#include <cmath>
#include <cassert>
#include <cstdio>

#include "activation.h"
#include "math_cpu.h"


template <typename T>
void print_tmp(const int H, const int W, const int N, const int D,
               const T* tmp) {
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int n = 0; n < N; ++n)
          for (int g = 0; g < 5; ++g) {
            fprintf(stderr, "tmp(d=%d,y=%d,x=%d,n=%d,g=%d) =", z, y, x, n, g);
            for (int d = 0; d < D; ++d)
              fprintf(stderr, " %g", tmp[d +
                                         g * D +
                                         z * D * 5 +
                                         n * D * 5 * 4 +
                                         x * D * 5 * 4 * N +
                                         y * D * 5 * 4 * N * W]);
            fprintf(stderr, "\n");
          }
}

// Useful defines to access Q (gates and cells) and output memory arrays
#define Q_ptr(y, x, n, d, g, k)                                       \
  (Q + (((((y) * W + (x)) * N + (n)) * 4 + (d)) * 5 + (g)) * D + (k))
#define O_ptr(y, x, n, d, k)                                    \
  (O + (((((y) * W + (x)) * N + (n)) * 4 + (d)) * D + (k)))

// 2D-LSTM forward pass running on the CPU
// H -> maximum height
// W -> maximum width
// N -> batch size
// K -> input dimensions/channels
// D -> output dimensions/channels
// I -> input data (layout: H x W x N x K)
// S -> input sizes (height and width of each sample, layout: N x 2)
// P -> params (layout: [20 * D] (b) + [K x 20 * D] (iW) + [4 * D x 10*D] (rW))
// O -> output data (layout: H x W x N x 4 x D)
// Q -> gates and cells activations (layout: H x W x N x 4 x 5 x D)
template <typename T, typename FC, typename FO>
void lstm_2d_fwd_cpu(const int H, const int W, const int N, const int K,
                     const int D, const T* I, const int* S, const T* P,
                     T* O, T* Q) {
  // Input weights
  const T* iW = P + 20 * D;    
  // Recurrent weights in each direction
  const T* rW[4][2] = {
    {P + 20 * D + 20 * D * K,
     P + 20 * D + 20 * D * K +  5 * D * D},
    {P + 20 * D + 20 * D * K + 10 * D * D,
     P + 20 * D + 20 * D * K + 15 * D * D},
    {P + 20 * D + 20 * D * K + 20 * D * D,
     P + 20 * D + 20 * D * K + 25 * D * D},
    {P + 20 * D + 20 * D * K + 30 * D * D,
     P + 20 * D + 20 * D * K + 35 * D * D}
  };

  // Initialize with bias
  #pragma omp parallel for
  for (int i = 0; i < H * W * N * 4 * 5 * D; ++i) {
    Q[i] = P[i % (4 * 5 * D)];
  }

  // Multiply inputs by weights
  gemm_cpu<T>('N', 'N', H * W * N, 4 * 5 * D, K, 1.0, I, K, iW, 4 * 5 * D,
              1.0, Q, 4 * 5 * D);

  // diagonal directions (all pixels in the diagonal are independent)
  const int zim[4][2] = {
    {+1, -1},  // origin: top-left      {(y,x), (y+1,x-1), (y+2,x-2), ...}
    {+1, +1},  // origin: top-right     {(y,x), (y+1,x+1), (y+2,x+2), ...}
    {-1, -1},  // origin: bottom-left   {(y,x), (y-1,x-1), (y-2,x-2), ...}
    {-1, +1}   // origin: bottom-right  {(y,x), (y-1,x+1), (y-2,x+2)
  };
  // neighboring pixels in each direction
  const int zin[4][2] = {
    {-1, -1},  // origin: top-left,     ngbrs(y, x) = {(y-1, x), (y, x-1)}
    {-1, +1},  // origin: top-right,    ngbrs(y, x) = {(y-1, x), (y, x+1)}
    {+1, -1},  // origin: bottom-left,  ngbrs(y, x) = {(y+1, x), (y, x-1)}
    {+1, +1},  // origin: bottom-right, ngbrs(y, x) = {(y+1, x), (y, x+1)}
  };  
  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  const int minD = std::min(H, W);  
  for (int z = 0; z < H + W - 1; ++z) {
    // Number of elements in the z-th diagonal
    const int Zn = std::min(std::min(H + W - 1 - z, z + 1), minD);
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
    // Multiply neighboring pixels by recurrent connections
    #pragma omp parallel for
    for (int i = 0; i < Zn * 4; ++i) {
      // d-th direction: top-left, top-right, bottom-left, bottom-right
      const int d = i / Zn;
      // j-th element in the z-diagonal
      const int j = i % Zn;
      // (y, x) coords of the j-th element in the z-th diagonal in the
      // d-th direction.
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      assert(!(y < 0 || x < 0 || y >= H || x >= W));
      //T* Q_yxd = Q + (y * W + x) * N * 4 * 5 * D + d * 5 * D;
      T* Q_yxd = Q_ptr(y, x, 0, d, 0, 0);
      if (y + zin[d][0] >= 0 && y + zin[d][0] < H - 1) {
        //const T* O_yxd = O + ((y + zin[d][0]) * W + x) * N * 4 * D + d * D;
        const T* O_yxd = O_ptr(y + zin[d][0], x, 0, d, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_yxd, 4 * D,
                    rW[d][0], 5 * D,
                    1.0, Q_yxd, 4 * 5 * D);
      }
      if (x + zin[d][1] >= 0 && x + zin[d][1] < W - 1) {
        //const T* O_yxd = O + (y * W + x + zin[d][1]) * N * 4 * D + d * D;
        const T* O_yxd = O_ptr(y, x + zin[d][1], 0, d, 0);
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_yxd, 4 * D,
                    rW[d][1], 5 * D,
                    1.0, Q_yxd, 4 * 5 * D);
      }
    }
    // Apply activation functions for gates and cell
    #pragma omp parallel for
    for (int i = 0; i < Zn * N * 4 * 5 * D; ++i) {
      const int k = i % D;
      const int g = (i / D) % 5;
      const int d = (i / (5 * D)) % 4;
      const int n = (i / (4 * 5 * D)) % N;
      const int j = (i / (N * 4 * 5 * D)) % Zn;
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      
      assert(!(y < 0 || x < 0 || y >= H || x >= W));
      T* t = Q_ptr(y, x, n, d, g, k);
      *t = g < 4 ? Sigmoid<T>::f(*t) : FC::f(*t);
    }
    // Compute cell and output values
    #pragma omp parallel for
    for (int i = 0; i < Zn * N * 4 * D; ++i) {
      const int k = i % D;
      const int d = (i / D) % 4;
      const int n = (i / (4 * D)) % N;
      const int j = (i / (N * 4 * D)) % Zn;
      const int y = zyx[d][0] + zim[d][0] * j, x = zyx[d][1] + zim[d][1] * j;
      assert(!(y < 0 || x < 0 || y >= H || x >= W));
      T* O_yx  =  O_ptr(y, x, n, d, k);
      T* cell  =  Q_ptr(y, x, n, d, 5, k);
      if (y < S[n * 2] && x < S[n * 2 + 1]) {
        const T ig  = *Q_ptr(y, x, n, d, 1, k);
        const T og  = *Q_ptr(y, x, n, d, 2, k);
        const T fgy = *Q_ptr(y, x, n, d, 3, k);
        const T fgx = *Q_ptr(y, x, n, d, 4, k);
        const T cell_10 =
            (y + zin[d][0] >= 0 && y + zin[d][0] < H - 1) ?
            *Q_ptr(y + zin[d][0], x, n, d, 5, k) : 0;
        const T cell_01 =
            (x + zin[d][1] >= 0 && x + zin[d][1] < W - 1) ?
            *Q_ptr(y, x + zin[d][1], n, d, 5, k) : 0;      
        *cell = ig * (*cell) + fgy * cell_10 + fgx * cell_01;
        *O_yx = og * FO::f(*cell);
      } else {
        *cell = 0;
        *O_yx = 0;
      }
    }
  }
}

#endif  // RNN2D_SRC_LSTM_CPU_H_

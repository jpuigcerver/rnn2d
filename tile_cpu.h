#ifndef RNN2D_TILE_CPU_H_
#define RNN2D_TILE_CPU_H_

#include "tile.inc.h"

#ifdef __cplusplus
extern "C" {
#endif

// float

void rnn2d_tile_cpu_float_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* input,
    float* output);

void rnn2d_tile_cpu_float_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const float* dOutput,
    float* dInput);

// double

void rnn2d_tile_cpu_double_fw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* input,
    double* output);

void rnn2d_tile_cpu_double_bw(
    const int H, const int W, const int N, const int D,
    const int Kh, const int Kw, const int* shape, const double* dOutput,
    double* dInput);

#ifdef __cplusplus
}
#endif

#endif  // RNN2D_TILE_CPU_H_

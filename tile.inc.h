#ifndef RNN2D_TILE_INC_H_
#define RNN2D_TILE_INC_H_

#define DIV_UP(x, y) (x == 0 ? 0 : 1 + ((x) - 1) / (y))

#define RNN2D_TILE_OUTPUT_SIZE(H, W, N, D, KH, KW)      \
  (DIV_UP(H, KH) * DIV_UP(W, KW) * N * KH * KW * D)

#endif  // RNN2D_TILE_INC_H_

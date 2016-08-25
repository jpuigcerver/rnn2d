#ifndef RNN2D_LSTM_COMMON_H_
#define RNN2D_LSTM_COMMON_H_

// Useful defines to access specific addresses in the input, output and
// internal state tensors.
#define I_ptr(y, x, n, d)                       \
  (I  + (((y) * W + (x)) * N + (n)) * K + (d))
#define dI_ptr(y, x, n, d)                      \
  (dI + (((y) * W + (x)) * N + (n)) * K + (d))
#define O_ptr(y, x, n, k, d)                                    \
  (O  + ((((y) * W + (x)) * N + (n)) * 4 + (k)) * D + (d))
#define dO_ptr(y, x, n, k, d)                                   \
  (dO + ((((y) * W + (x)) * N + (n)) * 4 + (k)) * D + (d))
#define Q_ptr(k, y, x, n, g, d)                                 \
  (Q  + (((((k) * H + (y)) * W + (x)) * N + (n)) * 6 + (g)) * D + (d))
#define dQ_ptr(k, y, x, n, g, d)                                \
  (dQ + (((((k) * H + (y)) * W + (x)) * N + (n)) * 6 + (g)) * D + (d))

template <typename T>
void print_Q(const int H, const int W, const int N, const int D,
             const T* Q) {
  for (int k = 0; k < 4; ++k) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int n = 0; n < N; ++n) {
          for (int g = 0; g < 6; ++g) {
            fprintf(stderr, "Q(k=%d,y=%d,x=%d,n=%d,g=%d) =", k, y, x, n, g);
            for (int d = 0; d < D; ++d) {
              fprintf(stderr, " %.10g", *Q_ptr(k, y, x, n, g, d));
            }
            fprintf(stderr, "\n");
          }
        }
        fprintf(stderr, "\n");
      }
    }
  }
}

template <typename T>
void print_O(const int H, const int W, const int N, const int D, const T* O) {
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        fprintf(stderr, "O(y=%d,x=%d,n=%d) =", y, x, n);
        for (int k = 0; k < 4; ++k) {
          for (int d = 0; d < D; ++d) {
            fprintf(stderr, " %.20g", *O_ptr(y, x, n, k, d));
          }
        }
        fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
  }
}


template <typename T>
void flip_y(const int H, const int W, const int N, const int D,
            const T* X, T* Y) {
  #pragma omp parallel for schedule(static) collapse(4)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          const int i = (((        y) * W + (        x)) * N + n) * D + d;
          const int j = (((H - y - 1) * W + (        x)) * N + n) * D + d;
          Y[j] = X[i];
        }
      }
    }
  }
}

template <typename T>
void flip_x(const int H, const int W, const int N, const int D,
            const T* X, T* Y) {
  #pragma omp parallel for schedule(static) collapse(4)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          const int i = (((        y) * W + (        x)) * N + n) * D + d;
          const int j = (((        y) * W + (W - x - 1)) * N + n) * D + d;
          Y[j] = X[i];
        }
      }
    }
  }
}

template <typename T>
void flip_yx(const int H, const int W, const int N, const int D,
            const T* X, T* Y) {
  #pragma omp parallel for schedule(static) collapse(4)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          const int i = (((        y) * W + (        x)) * N + n) * D + d;
          const int j = (((H - y - 1) * W + (W - x - 1)) * N + n) * D + d;
          Y[j] = X[i];
        }
      }
    }
  }
}

#endif  // RNN2D_LSTM_COMMON_H_

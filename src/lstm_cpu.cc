#include <rnn2d/lstm_cpu.h>

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <iostream>

#include <glog/logging.h>

#include <rnn2d/activation.h>
#include <rnn2d/lstm_impl.h>
#include <rnn2d/math_cpu.h>

template <typename T>
inline size_t get_inference_workspace_size(
    const int H, const int W, const int N, const int D) {
  const size_t tmpd_size = 4 * H * W * N * 5 * D * sizeof(T);
  const size_t ptrs_size = 2 * 3 * 4 * std::min(H, W) * sizeof(T*);
  return tmpd_size + ptrs_size;
}

template <typename T>
inline size_t get_training_workspace_size(
    const int H, const int W, const int N, const int D) {
  const size_t tmpd_size = 3 * 4 * H * W * N * D * sizeof(T);
  const size_t ptrs_size = 2 * 3 * 4 * std::min(H, W) * sizeof(T*);
  return tmpd_size + ptrs_size;
}

template <typename T>
inline size_t get_training_reserve_size(
    const int H, const int W, const int N, const int D) {
  return 4 * H * W * N * 5 * D * sizeof(T);
}

template <typename T>
inline void copy_dO_to_Z(const int H, const int W, const int N, const int D,
                         const T* dO, T* Z) {
  #pragma omp parallel for collapse(5)
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int n = 0; n < N; ++n)
          for (int d = 0; d < D; ++d)
            *Z_ptr(0, z, y, x, n, d) = *O_ptr(dO, y, x, n, z, d);
}

template <typename T>
inline void init_Q_with_bias(
    const int H, const int W, const int N, const int K, const int D,
    const T* P, T* Q) {
  #pragma omp parallel for collapse(6)
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int n = 0; n < N; ++n)
          for (int g = 0; g < 5; ++g)
            for (int d = 0; d < D; ++d) {
              *Q_ptr(z, y, x, n, g, d) = *B_ptr(P, z, g, d);
            }
}

template <typename T, typename FG, typename FI, typename FO>
inline void fw_elemwise_ops(
    const int H, const int W, const int N, const int D, const int t,
    const int Tn, const int Tmin, const int* S, T* Q, T* O) {
  // Compute cell and output values
  #pragma omp parallel for collapse(4)
  for (int z = 0; z < 4; ++z) {
    for (int e = 0; e < Tn; ++e) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          // (y, x) coordinates of the e-th element in the t-th diagonal.
          const int i = e + Tmin;
          const int j = t - i;
          const int y  = (z == 0 || z == 2) ? i : H - i - 1;
          const int x  = (z == 0 || z == 1) ? j : W - j - 1;
          if (S == nullptr || (y < S[n * 2] && x < S[n * 2 + 1])) {
            const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;
            const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;
            const T f_gi  = FG::f(*Q_ptr(z, y, x, n, 0, d));  // input gate
            const T f_gfy = FG::f(*Q_ptr(z, y, x, n, 1, d));  // fgt_y gate
            const T f_gfx = FG::f(*Q_ptr(z, y, x, n, 2, d));  // fgt_x gate
            const T f_go  = FG::f(*Q_ptr(z, y, x, n, 3, d));  // output gate
            const T f_a   = FI::f(*Q_ptr(z, y, x, n, 4, d));  // pre-cell
            const T C_10 = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 4, d) : 0;
            const T C_01 = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 4, d) : 0;
            const T C_00 = f_gi * f_a + f_gfy * C_10 + f_gfx * C_01;  // state
            const T O_00 = f_go * FO::f(C_00);                        // output
            *Q_ptr(z, y, x, n, 0, d) = f_gi;
            *Q_ptr(z, y, x, n, 1, d) = f_gfy;
            *Q_ptr(z, y, x, n, 2, d) = f_gfx;
            *Q_ptr(z, y, x, n, 3, d) = f_go;
            *Q_ptr(z, y, x, n, 4, d) = C_00;
            *O_ptr(O, y, x, n, z, d) = O_00;
          } else {
            *Q_ptr(z, y, x, n, 0, d) = 0;
            *Q_ptr(z, y, x, n, 1, d) = 0;
            *Q_ptr(z, y, x, n, 2, d) = 0;
            *Q_ptr(z, y, x, n, 3, d) = 0;
            *Q_ptr(z, y, x, n, 4, d) = 0;
            *O_ptr(O, y, x, n, z, d) = 0;
          }
        }
      }
    }
  }
}

template <typename T>
void print_Q(const int H, const int W, const int N, const int D, T* Q) {
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        printf("Q[%d, %d, %d, :, :] =\n", z, y, x);
        for (int n = 0; n < N; ++n) {
          printf("  ");
          for (int d = 0; d < 5 * D; ++d) {
            printf(" % 7.5f", *Q_ptr(z, y, x, n, d / D, d % D));
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }
}

template <typename T>
void print_O(const int H, const int W, const int N, const int D, T* O) {
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        printf("O[%d, %d, %d, :, :] =\n", z, y, x);
        for (int n = 0; n < N; ++n) {
          printf("  ");
          for (int d = 0; d < D; ++d) {
            printf(" % 7.5f", *O_ptr(O, y, x, n, z, d));
          }
          printf("\n");
        }
        printf("\n");
      }
    }
  }
}

template <typename T>
void print_I(const char* name, const int H, const int W, const int N,
             const int K, T* I) {
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      printf("%s[%d, %d, :, :] =\n", name, y, x);
      for (int n = 0; n < N; ++n) {
        printf("  ");
        for (int k = 0; k < K; ++k) {
          printf(" % 7.5f", *I_ptr(I, y, x, n, k));
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

template <typename T>
void print_W(const char* name, const T* ptr, const int K, const int D) {
  for (int z = 0; z < 4; ++z) {
    printf("%s[%d, :, :] =\n", name, z);
    for (int k = 0; k < K; ++k) {
      printf("  ");
      for (int d = 0; d < 5 * D; ++d) {
        printf(" % 7.5f", *W_ptr(ptr, z, k, d / D, d % D));
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void print_U(const char* name, const T* ptr, const int K, const int D) {
  for (int z = 0; z < 4; ++z) {
    printf("%s[%d, :, :] =\n", name, z);
    for (int k = 0; k < D; ++k) {
      printf("  ");
      for (int d = 0; d < 5 * D; ++d) {
        printf(" % 7.5f", *U_ptr(ptr, z, k, d / D, d % D));
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void print_V(const char* name, const T* ptr, const int K, const int D) {
  for (int z = 0; z < 4; ++z) {
    printf("%s[%d, :, :] =\n", name, z);
    for (int k = 0; k < D; ++k) {
      printf("  ");
      for (int d = 0; d < 5 * D; ++d) {
        printf(" % 7.5f", *V_ptr(ptr, z, k, d / D, d % D));
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <typename T>
void print_B(const char* name, const T* ptr, const int K, const int D) {
  for (int z = 0; z < 4; ++z) {
    printf("%s[%d, :, :] =\n", name, z);
    printf("  ");
    for (int d = 0; d < 5 * D; ++d) {
      printf(" % 7.5f", *B_ptr(ptr, z, d / D, d % D));
    }
    printf("\n");
    printf("\n");
  }
}

template <typename T, typename FG, typename FI, typename FO>
inline void bw_elemwise_ops(
    const int H, const int W, const int N, const int D, const int t,
    const int Tn, const int Tmin, const int* S, T* Q, T* Z) {
  #pragma omp parallel for collapse(4)
  for (int z = 0; z < 4; ++z) {
    for (int e = 0; e < Tn; ++e) {
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          const int i = e + Tmin;
          const int j = t - i;
          const int y = (z == 0 || z == 2) ? i : H - i - 1;
          const int x = (z == 0 || z == 1) ? j : W - j - 1;
          T* dGi_00 = Q_ptr(z, y, x, n, 0, d);   // currenlty contains f(Gi_00)
          T* dGy_00 = Q_ptr(z, y, x, n, 1, d);   // currently contains f(Gy_00)
          T* dGx_00 = Q_ptr(z, y, x, n, 2, d);   // currently contains f(Gx_00)
          T* dGo_00 = Q_ptr(z, y, x, n, 3, d);   // currently contains f(Go_00)
          T* dA_00  = Q_ptr(z, y, x, n, 4, d);   // currently contains C_00
          if (S == nullptr || (y < S[2 * n] && x < S[2 * n + 1])) {
            const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
            const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
            const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // previous y
            const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;  // previous x
            const T C_00   = *dA_00;
            const T fGi_00 = *dGi_00;
            const T fGo_00 = *dGo_00;
            const T fGy_00 = *dGy_00;
            const T fGx_00 = *dGx_00;
            const T dO_00  = *Z_ptr(0, z, y, x, n, d);
            const T C_10 = (yp >= 0 && yp < H) ? *Q_ptr(z, yp, x, n, 4, d) : 0;
            const T C_01 = (xp >= 0 && xp < W) ? *Q_ptr(z, y, xp, n, 4, d) : 0;
            const T fA_00 = fGi_00 != 0.0 ?
                (C_00 - C_10 * fGy_00 - C_01 * fGx_00) / fGi_00 : 0.0;
            // Z_10 = dC(y+1, x) * f(Gy(y+1, x))
            const T Z_10  = (yn >= 0 && yn < H) ? *Z_ptr(1, z, yn, x, n, d) : 0;
            // Z_01 = dC(y, x+1) * f(Gx(y, x+1))
            const T Z_01  = (xn >= 0 && xn < W) ? *Z_ptr(2, z, y, xn, n, d) : 0;
            const T dC_00  = dO_00 * FO::df(C_00) * fGo_00 + Z_10 + Z_01;
            *dGo_00 = dO_00 * FO::f(C_00) * FG::df2(fGo_00);
            *dGy_00 = (yp >= 0 && yp < H) ? dC_00 * C_10 * FG::df2(fGy_00) : 0;
            *dGx_00 = (xp >= 0 && xp < W) ? dC_00 * C_01 * FG::df2(fGx_00) : 0;
            *dGi_00 = dC_00 * fA_00 * FG::df2(fGi_00);
            *dA_00  = dC_00 * FI::df2(fA_00) * fGi_00;
            *Z_ptr(1, z, y, x, n, d) = dC_00 * fGy_00;
            *Z_ptr(2, z, y, x, n, d) = dC_00 * fGx_00;
          } else {
            *dA_00  = 0;
            *dGi_00 = 0;
            *dGo_00 = 0;
            *dGy_00 = 0;
            *dGx_00 = 0;
            *Z_ptr(1, z, y, x, n, d) = 0;
            *Z_ptr(2, z, y, x, n, d) = 0;
          }
        }
      }
    }
  }
}

/* 2D-LSTM forward pass running on the CPU
 * H -> maximum height
 * W -> maximum width
 * N -> batch size
 * K -> input dimensions/channels
 * D -> output dimensions/channels
 * I -> input data (layout: H x W x N x K)
 * S -> input sizes (height and width of each sample, layout: N x 2)
 * P -> parameters (size: 4 * (1 + K + D + D) * 5 * D)
 * O -> output data (layout: H x W x N x 4 x D)
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 5 x D)
 */
template <typename T, typename FG, typename FI, typename FO>
inline void fw_training(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, T* O, void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(wspace);

  T* Q = reinterpret_cast<T*>(rspace != nullptr ? rspace : wspace);

  // Initialize gates with bias
  // [A,Gi,Go,Gx,Gy](x,y) = [b_a,b_i,b_o,b_x,b_y]
  init_Q_with_bias<T>(H, W, N, K, D, P, Q);

  // Multiply inputs by weights:
  // [A,Gi,Go,Gx,Gy](x,y) += I(x,y) * [W_a,W_i,W_o,W_x,W_y]
  // Note: Each direction could run in parallel.
  for (int z = 0; z < 4; ++z) {
    // I viewed as a (H * W * N) x K matrix
    // W viewed as a K x (5 * D) matrix
    // Q viewed as a (H * W * N) x (5 * D) matrix
    gemm_cpu<T>('N', 'N', H * W * N, 5 * D, K,
                1.0, I, K, W_ptr(P, z, 0, 0, 0), 5 * D,
                1.0, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D);
  }


  // Process the image diagonal-wise (there are H + W - 1 diagonals to process)
  for (int t = 0; t < H + W - 1; ++t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;

    // Matrix multiplications to compute the input to the gates from the
    // recurrent connections.
    // [Gi,Gy,Gx,Go,A](x,y) += O(x,y-1) * [U_i,U_y,U_x,U_o,U_a]
    // [Gi,Gy,Gx,Go,A](x,y) += O(x-1,y) * [V_i,V_y,V_x,V_o,V_a]
    #pragma omp parallel for
    for (int e = 0; e < 4 * Tn; ++e) {
      const int z = e / Tn; // Diagonal direction
      // (y, x) coordinates of the e-th element in the t-th diagonal.
      const int i = (e % Tn) + Tmin;
      const int j = t - i;
      const int y  = (z == 0 || z == 2) ? (i    ) : (H - i - 1);
      const int x  = (z == 0 || z == 1) ? (j    ) : (W - j - 1);
      const int yp = (z == 0 || z == 2) ? (y - 1) : (y + 1    );
      const int xp = (z == 0 || z == 1) ? (x - 1) : (x + 1    );
      if (yp >= 0 && yp < H) {
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0,  O_ptr(O, yp, x, 0, z, 0), 4 * D,
                    V_ptr(P, z, 0, 0, 0), 5 * D,
                    1.0, Q_ptr(z, y, x, 0, 0, 0), 5 * D);
      }
      if (xp >= 0 && xp < W) {
        gemm_cpu<T>('N', 'N', N, 5 * D, D,
                    1.0, O_ptr(O, y, xp, 0, z, 0), 4 * D,
                    U_ptr(P, z, 0, 0, 0), 5 * D,
                    1.0, Q_ptr(z, y, x, 0, 0, 0), 5 * D);
      }
    }
    // Compute cell states
    fw_elemwise_ops<T, FG, FI, FO>(H, W, N, D, t, Tn, Tmin, S, Q, O);
  }
}


/* 2D-LSTM backward pass running on the CPU
 * H -> maximum height
 * W -> maximum width
 * N -> batch size
 * K -> input dimensions/channels
 * D -> output dimensions/channels
 * I -> input data (layout: H x W x N x K)
 * S -> input sizes (height and width of each sample, layout: N x 2)
 * P -> parameters (size: 4 * (1 + K + D + D) * 5 * D)
 * O -> output data (layout: H x W x N x 4 x D)
 * Q -> gates pre-activations and cells (layout: 4 x H x W x N x 5 x D)
 * dO -> derivative of the loss w.r.t the output
 * dI -> derivative of the loss w.r.t the input
 */
template <typename T, typename FG, typename FI, typename FO>
inline void bw_data(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const int* S, const T* P, const T* O, const T* dO,
    T* dI, void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(P);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dO);
  CHECK_NOTNULL(dI);
  CHECK_NOTNULL(wspace);
  CHECK_NOTNULL(rspace);

  T* Q = reinterpret_cast<T*>(rspace);
  T* Z = reinterpret_cast<T*>(wspace);

  copy_dO_to_Z<T>(H, W, N, D, dO, Z);

  // Process the image diagonal-wise, in backwards order
  // (there are H + W - 1 diagonals to process)
  for (int t = H + W - 2; t >= 0; --t) {
    // Compute number of elements in the diagonal
    const int Tmin = std::max(0, t - W + 1);
    const int Tmax = std::min(t, H - 1);
    const int Tn   = (Tmax - Tmin) + 1;
    // Note: All elements in the diagonal could be processed in parallel.
    // However, a OpenMP parallel for was not used here because the gemm_cpu
    // is already parallelized.
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < Tn; ++e) {
        const int i = e + Tmin;
        const int j = t - i;
        const int y  = (z == 0 || z == 2) ? i : H - i - 1;
        const int x  = (z == 0 || z == 1) ? j : W - j - 1;
        const int yn = (z == 0 || z == 2) ? y + 1 : y - 1;  // next y
        const int xn = (z == 0 || z == 1) ? x + 1 : x - 1;  // next x
        if (xn >= 0 && xn < W) {
          gemm_cpu<T>('N', 'T', N, D, 5 * D,
                      1.0, Q_ptr(z, y, xn, 0, 0, 0), 5 * D,
                      U_ptr(P, z, 0, 0, 0), 5 * D,
                      1.0, Z_ptr(0, z, y, x, 0, 0), D);
        }
        if (yn >= 0 && yn < H) {
          gemm_cpu<T>('N', 'T', N, D, 5 * D,
                      1.0, Q_ptr(z, yn, x, 0, 0, 0), 5 * D,
                      V_ptr(P, z, 0, 0, 0), 5 * D,
                      1.0, Z_ptr(0, z, y, x, 0, 0), D);
        }
      }
    }
    bw_elemwise_ops<T, FG, FI, FO>(H, W, N, D, t, Tn, Tmin, S, Q, Z);
  }

  // Compute dJ/dI(y,x)
  for (int z = 0; z < 4; ++z) {
    // dQ  viewed as (H * W * N) x (5 * D)
    // W^T viewed as (5 * D) x (K)
    // dI  viewed as (H * W * N) x K
    gemm_cpu<T>('N', 'T', H * W * N, K, 5 * D,
                1.0, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
                W_ptr(P, z, 0, 0, 0), 5 * D,
                1.0, dI, K);
  }

  //print_Q(H, W, N, D, Q);
  //print_I("DX", H, W, N, K, dI);
}


template <typename T>
inline void bw_param(
    const int H, const int W, const int N, const int K, const int D,
    const T* I, const T* O, const T scale, T* dP, void* wspace, void* rspace) {
  CHECK_NOTNULL(I);
  CHECK_NOTNULL(O);
  CHECK_NOTNULL(dP);
  CHECK_NOTNULL(wspace);
  CHECK_NOTNULL(rspace);

  T* Q = reinterpret_cast<T*>(rspace);
  T* Z = reinterpret_cast<T*>(wspace);

  // dJ/db
  T* vOnes = Z;
  std::fill(vOnes, vOnes + H * W * N, static_cast<T>(1));
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    gemv_cpu<T>('T', H * W * N, 5 * D,
                scale, Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
                vOnes, 1,
                1.0, B_ptr(dP, z, 0, 0), 1);
  }

  // dJ/dW
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    gemm_cpu<T>('T', 'N', K, 5 * D, H * W * N,
                scale, I, K,
                Q_ptr(z, 0, 0, 0, 0, 0), 5 * D,
                1.0, W_ptr(dP, z, 0, 0, 0), 5 * D);
  }

  // dJ/dU
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int xp = (z == 0 || z == 1) ? x - 1 : x + 1;  // previous x
        if (xp >= 0 && xp < W) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      scale, O_ptr(O, y, xp, 0, z, 0), 4 * D,
                      Q_ptr(z, y, x, 0, 0, 0), 5 * D,
                      1.0, U_ptr(dP, z, 0, 0, 0), 5 * D);
        }
      }
    }
  }


  // dJ/dV
  #pragma omp parallel for
  for (int z = 0; z < 4; ++z) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const int yp = (z == 0 || z == 2) ? y - 1 : y + 1;  // previous y
        if (yp >= 0 && yp < H) {
          gemm_cpu<T>('T', 'N', D, 5 * D, N,
                      scale, O_ptr(O, yp, x, 0, z, 0), 4 * D,
                      Q_ptr(z, y, x, 0, 0, 0), 5 * D,
                      1.0, V_ptr(dP, z, 0, 0, 0), 5 * D);
        }
      }
    }
  }

  /*print_B("DB", dP, K, D);
  print_W("DW", dP, K, D);
  print_U("DU", dP, K, D);
  print_V("DV", dP, K, D);*/
}

extern "C" {
  DEFINE_WRAPPERS(cpu, float)
  DEFINE_WRAPPERS(cpu, double)
}  // extern "C"

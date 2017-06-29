#ifndef RNN2D_INTERNAL_CPU_RNN2D_SIMPLE_CELL_H_
#define RNN2D_INTERNAL_CPU_RNN2D_SIMPLE_CELL_H_

#include <rnn2d/basic_types.h>
#include <include/rnn2d/activations.h>

namespace rnn2d {
namespace internal {
namespace cpu {

template <typename T, class F = internal::Tanh<T>>
class Rnn2dSimpleCell {
 public:
  Rnn2dSimpleCell() {}

  Rnn2dSimpleCell(const F& activation_function) : act_(activation_function) {}

  static int NumGates() { return 1; }

  template <typename Rnn2dImpl>
  rnn2dStatus_t Forward(Rnn2dImpl* layer, const int t) {
    const int H = layer->GetH();
    const int W = layer->GetW();
    const int N = layer->GetN();
    const int D = layer->GetD();
    const int L = std::min(H, W);

    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int y = layer->GetY(t, e), x = layer->GetX(t, e);
            if (y >= 0 && x >= 0 && y < H && x < W) {
              if (y < layer->GetH(n) && x < layer->GetW(n)) {
                // (x, y) coordinates are within the image.
                layer->O(z, y, x, n, d) = act_.f(layer->Q(z, y, x, n, 0, d));
              } else {
                // (x, y) coordinates are NOT within the image, but are within
                // the batch.
                layer->O(z, y, x, n, d) = 0;
              }
            }
          }
        }
      }
    }

    return RNN2D_STATUS_SUCCESS;
  }

  template <typename Rnn2dImpl>
  rnn2dStatus_t Backward(Rnn2dImpl* layer, int t) {
    const int H = layer->GetH();
    const int W = layer->GetW();
    const int N = layer->GetN();
    const int D = layer->GetD();
    const int L = std::min(H, W);

    #pragma omp parallel for collapse(4)
    for (int z = 0; z < 4; ++z) {
      for (int e = 0; e < L; ++e) {
        for (int n = 0; n < N; ++n) {
          for (int d = 0; d < D; ++d) {
            // (y, x) coordinates of the e-th element in the t-th diagonal.
            const int y = layer->GetY(t, e), x = layer->GetX(t, e);
            if (y >= 0 && x >= 0 && y < H && x < W) {
              if (y < layer->GetH(n) && x < layer->GetW(n)) {
                // (x, y) coordinates are within the image.
                layer->O(z, y, x, n, d) = act_.f(layer->Q(z, y, x, n, 0, d));
              } else {
                // (x, y) coordinates are NOT within the image, but are within
                // the batch.
                layer->O(z, y, x, n, d) = 0;
              }
            }
          }
        }
      }
    }

    return RNN2D_STATUS_SUCCESS;
  }


 private:
  F act_;

};

}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d

#endif  // RNN2D_INTERNAL_CPU_RNN2D_SIMPLE_CELL_H_

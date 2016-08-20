
template <typename T>
void blockify_cpu(const int N, const int K, const int H, const int W,
                  const T* input, const int kW, const int kW, T* output) {
  const int outH = (H + kH - 1) / kH, outW = (W + kW - 1) / kW;
  const int outK = kH * kW * K;

  #pragma omp parallel for
  for (int i = 0; i < N * outK * outH * outW; ++i) {
    const int n = i / (outK * outH * outW);
    const int o_k = (i / (outH * outW)) % outK;
    const int o_y = (i / outW) % outH;
    const int o_x = i % outW;
    const int i_k = ;
    const int i_y = ;
    const int i_x = ;

    output[n * outK * outH * outW + o_k * outH * outW + o_y * outW + o_x] =
        input[n * K * H * W + i_k * H * W + i_y * W + i_x];
  }
}

template <typename T>
void deblockify_cpu(const int N, const int K, const int H, const int W,
                  const T* input, const int kW, const int kW, T* output) {
  const int outH = (H + kH - 1) / kH, outW = (W + kW - 1) / kW;
  const int outK = kH * kW * K;

}

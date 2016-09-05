#include <cstring>
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lstm_cpu.h"

static std::default_random_engine RNG = std::default_random_engine();

namespace testing {

template <typename T>
bool FloatRelativeEq(const T r, const T a, const T t) {
  const T d = std::fabs(r - a);
  const T R = std::fabs(r);
  const T A = std::fabs(a);
  const T Min = std::min(A, R);
  const T Max = std::max(A, R);
  return Min > t ? (d <= Max * t) : (d <= t);
}

MATCHER_P(FloatRelativeEqPointwise, tol, "") {
  return FloatRelativeEq<float>(std::get<1>(arg), std::get<0>(arg),
                                tol);
}

MATCHER_P(DoubleRelativeEqPointwise, tol, "") {
  return FloatRelativeEq<double>(std::get<1>(arg), std::get<0>(arg),
                                 tol);
}

MATCHER(FloatEqPointwise, "") {
  return Matcher<float>(FloatEq(get<1>(arg))).Matches(get<0>(arg));
}

MATCHER(DoubleEqPointwise, "") {
  return Matcher<double>(DoubleEq(get<1>(arg))).Matches(get<0>(arg));
}

}  // namespace testing


using ::testing::Pointwise;
using ::testing::FloatRelativeEq;
using ::testing::FloatRelativeEqPointwise;
using ::testing::DoubleRelativeEqPointwise;


static const int H = 2, W = 3, N = 2, K = 3, D = 2;
static const int S[N * 2] = {2, 3, 2, 3};

template <typename T>
const std::vector<T>& I() {
  static const std::vector<T> I_{
    0.30, 0.68, 0.29, 0.10, 0.70, 0.88, 0.13, 0.18, 0.35, 0.86, 0.66, 0.75,
    0.53, 0.40, 0.48, 0.20, 0.58, 0.66, 0.30, 0.99, 0.64, 0.46, 0.44, 0.65,
    0.82, 0.59, 0.47, 0.18, 0.53, 0.13, 0.68, 0.79, 0.80, 0.32, 0.09, 0.40
  };
  return I_;
}

template <typename T>
const std::vector<T>& P() {
  // Parameters (note, the 4 directions share the same weights)
  static const std::vector<T> P_{
    // TOP-LEFT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // TOP-RIGHT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // BOTTOM-LEFT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
    // BOTTOM-RIGHT DIRECTION
    // Bias
    -0.66, -0.56, -0.19,  0.94, -0.22,  0.12, -0.99, -0.08, -0.79, -0.69,
    // Input weights
    -0.69,  0.16,  0.64,  0.11,  0.72, -0.48,  0.67,  0.72,  0.61, -0.34,
     0.72, -0.39, -0.31, -0.76,  0.31, -0.88,  0.24, -0.5,  -0.65, -0.21,
     0.61,  0.95, -0.46,  0.79,  0.98, -0.89,  0.88,  0.62, -0.36,  0.07,
    // Recurrent weights in y-dimension
     0.16,  0.10,  0.01,  0.91, -0.05,  0.38,  0.38, -0.62,  0.99, -0.03,
     0.60,  0.30, -0.47, -0.03,  0.12, -0.77,  0.94,  0.77, -0.79,  0.76,
    // Recurrent weights in x-dimension
    -0.30, -0.80,  0.93,  0.90,  0.95, -0.50,  0.65,  0.23, -0.90,  0.36,
    -0.42,  0.39,  0.54, -0.20,  0.14, -0.16,  0.57,  0.51, -0.30,  0.88,
  };
  return P_;
}

template <typename T>
const std::vector<T>& dO() {
  static const std::vector<T> dO_{
    0.51,  0.10,  0.21,  0.47, -0.06,  0.26,  0.50, -0.71,
    0.53,  0.65,  0.52,  0.25, -0.39, -0.13,  0.05,  0.07,
    0.44,  0.66,  0.30,  0.98,  0.20,  0.76, -0.93,  0.42,
    0.17,  0.71,  0.16, -0.48,  0.39,  0.92,  0.04,  0.81,
    0.07,  0.98, -0.17,  0.79,  0.57,  0.39,  0.94,  0.40,
    0.81,  0.40,  0.81,  0.34,  0.74,  0.49,  0.68,  0.00,
    0.29,  0.29,  0.50,  0.52, -0.15, -0.63, -0.87,  0.43,
    0.39,  0.59, -0.68,  0.92,  0.43, -0.16, -0.27,  0.19,
   -0.84,  0.13,  0.33,  0.89, -0.47,  0.72, -0.47,  0.27,
    0.85, -0.23,  0.15, -0.61,  0.69,  0.76,  0.47,  0.56,
    0.13,  0.61,  0.71,  0.11, -0.44,  0.11,  0.47,  0.04,
   -0.34,  0.78,  0.80,  0.24,  0.40,  0.49, -0.93,  0.09
  };
  return dO_;
}

template <typename T>
const std::vector<T>& expected_O();

template <>
const std::vector<float>& expected_O() {
  static const uint32_t H[] = {
    0x3d09fc7a, 0x3e9aff4f, 0x3d23bac1, 0x3e0c4ec4, 0x3c9ed863, 0x3eb1a948,
    0x3d41cf7a, 0x3d9b63b5, 0xbe5e618b, 0xbce62abb, 0xbee3b645, 0x3f21486d,
    0xbe7d2946, 0xbd610698, 0xbe723b55, 0x3ec929d0, 0xbaa828ae, 0x3c8e63e9,
    0xbb3ddc65, 0x3db414f4, 0xbcd47a41, 0x3c573d97, 0xbb25a916, 0x3e288df6,
    0x3e9fb149, 0xbe67318e, 0x3d9f9426, 0xbeb73ce4, 0x3e9efa69, 0x3e55f5f6,
    0x3df425ae, 0xbddeaadc, 0x3d887e4a, 0x3e08ddf6, 0x3d86bc5e, 0x3e2ff384,
    0x3da0560c, 0x3f23a291, 0x3d8009ac, 0x3e2eeaa5, 0xbedb76a5, 0x3f6ecb84,
    0xbc14e6a7, 0x3e11d2db, 0xbed4702c, 0x3eaf197d, 0xbd2875fe, 0x3e2a0b2b,
    0xbe907b49, 0x3ec88eae, 0xbd25109e, 0xbd4b039d, 0xbe06d1e0, 0x3e9dff8d,
    0xbd270b41, 0xbda3db19, 0x3de8bf7d, 0x3d83f31c, 0x3d940f8f, 0x3f19ac2f,
    0x3d9e5ec1, 0x3d4ef169, 0x3e001405, 0xbe8688c3, 0x3e681949, 0xbe9dd897,
    0xbc9b71fe, 0x3ea88b0f, 0x3df3d583, 0xbe63e98e, 0xbbcfd026, 0x3e956015,
    0xbd850278, 0x3e7c2353, 0xbd0a6d4d, 0x3e00213c, 0xbbff4886, 0x3e84a19a,
    0xbc5f3559, 0x3e232f6d, 0xbf09f64c, 0x3fa12f13, 0x3d1c7646, 0x3daad38d,
    0xbe53f833, 0x3f378f4b, 0x3d0d972f, 0xba9aead0, 0x3e53cab0, 0x3fba5c77,
    0x3d72789d, 0x3dfa29bd, 0x3cfec1d0, 0xbdba6897, 0x3d46b516, 0x3dc06bb9
  };
  static const size_t n = sizeof(H) / sizeof(uint32_t);
  static const float* p = reinterpret_cast<const float*>(H);
  static const std::vector<float> O(p, p + n);
  return O;
}

template <>
const std::vector<double>& expected_O() {
  static const uint64_t H[] = {
    0x3fa13f8f62e419d4L, 0x3fd35fe9f184ecd8L, 0x3fa47757a078cf90L,
    0x3fc189d8a3191152L, 0x3f93db0d1ba1472cL, 0x3fd635292ccb931dL,
    0x3fa839ee6bde3826L, 0x3fb36c776c909561L, 0xbfcbcc31802040aeL,
    0xbf9cc5582094d62cL, 0xbfdc76c89b54ccffL, 0x3fe4290df1127075L,
    0xbfcfa528cff6a5d3L, 0xbfac20d2f5e40326L, 0xbfce476ad05b5c66L,
    0x3fd9253af06b6525L, 0xbf55051979821634L, 0x3f91cc7c7fc95bebL,
    0xbf67bb860feb114eL, 0x3fb6829e8983572cL, 0xbf9a8f48c6cb6d35L,
    0x3f8ae7b2125ebd3bL, 0xbf64b519a457474fL, 0x3fc511bed6fd3e7bL,
    0x3fd3f628faf4606cL, 0xbfcce6319fc3bfb2L, 0x3fb3f283e36d2506L,
    0xbfd6e79cddc06f91L, 0x3fd3df4ce36515fdL, 0x3fcabebea008d9bcL,
    0x3fbe84b499e8bb66L, 0xbfbbd55e145c78beL, 0x3fb10fc95a301ab5L,
    0x3fc11bbe89da3d96L, 0x3fb0d78bab91062cL, 0x3fc5fe701d485f1fL,
    0x3fb40ac2c6d390ffL, 0x3fe47451b5b311a0L, 0x3fb001355e92a32aL,
    0x3fc5dd545b0ffd52L, 0xbfdb6ed47becd4baL, 0x3fedd9708c753cedL,
    0xbf829cd229e59148L, 0x3fc23a5bf4b9d7dcL, 0xbfda8e0586680e28L,
    0x3fd5e32f49dc8240L, 0xbfa50ebee9d96fcfL, 0x3fc54165b4c2d525L,
    0xbfd20f6934d447b8L, 0x3fd911d59a1710fdL, 0xbfa4a215f45dd281L,
    0xbfa960729ec845b7L, 0xbfc0da3c0a0b6a4aL, 0x3fd3bff1a18cab5aL,
    0xbfa4e1689323f2c8L, 0xbfb47b6359990f80L, 0x3fbd17ef8fc0b0e1L,
    0x3fb07e62e94f8ee4L, 0x3fb281ef3b0b1c06L, 0x3fe33586598507fdL,
    0x3fb3cbd821d4e5f3L, 0x3fa9de2b6ada1707L, 0x3fc00280c9cb5629L,
    0xbfd0d118a57b454bL, 0x3fcd0328fc440e7fL, 0xbfd3bb12af28d3c2L,
    0xbf936e3d692bd306L, 0x3fd51161cf73f219L, 0x3fbe7ab04df75859L,
    0xbfcc7d317544504aL, 0xbf79fa015668c595L, 0x3fd2ac02c12666e0L,
    0xbfb0a04edc2bf683L, 0x3fcf846ab6d5d47bL, 0xbfa14da979abadb6L,
    0x3fc0042750632922L, 0xbf7fe91064b36525L, 0x3fd0943384a4ed0dL,
    0xbf8be6ab895480cdL, 0x3fc465edc13b9ca3L, 0xbfe13ec978fa03a3L,
    0x3ff425e21a3d2166L, 0x3fa38ec83a94661fL, 0x3fb55a70e3624214L,
    0xbfca7f066d242566L, 0x3fe6f1e91c9379f8L, 0x3fa1b2e5545df355L,
    0xbf535d5f61c9994dL, 0x3fca79553cdd88d5L, 0x3ff74b8ed87b77f4L,
    0x3fae4f13847bccaeL, 0x3fbf453799c6b62cL, 0x3f9fd8390d76d92cL,
    0xbfb74d1380a84083L, 0x3fa8d6a2ab677f9fL, 0x3fb80d76ea0880c2L,
  };
  static const size_t n = sizeof(H) / sizeof(uint64_t);
  static const double* p = reinterpret_cast<const double*>(H);
  static const std::vector<double> O(p, p + n);
  return O;
}

template <typename T>
inline T expected_sum_dI();

template <>
inline float expected_sum_dI<float>() {
  static const uint32_t H = 0x4193d611;
  return *reinterpret_cast<const float*>(&H);
}

template <>
inline double expected_sum_dI<double>() {
  static const uint64_t H = 0x40327ac23c1e8713L;
  return *reinterpret_cast<const double*>(&H);
}

template <typename T>
inline T expected_sum_dP();

template <>
inline float expected_sum_dP<float>() {
  static const uint32_t H = 0xc270845b;
  return *reinterpret_cast<const float*>(&H);
}

template <>
inline double expected_sum_dP<double>() {
  static const uint64_t H = 0xc04e108a43a607c6L;
  return *reinterpret_cast<const double*>(&H);
}

template <typename T, typename M>
void test_forward(const M& matcher) {
  // Allocate space used for the internal states
  std::vector<T> Q(4 * H * W * N * 6 * D);
  // Output
  std::vector<T> O(H * W * N * 4 * D);
  lstm_2d_fw_cpu< T, Linear<T>, Linear<T>, Linear<T> >(
      H, W, N, K, D, I<T>().data(), S, P<T>().data(), O.data(), Q.data());
  EXPECT_THAT(O, Pointwise(matcher, expected_O<T>()));
}

template <typename T>
void test_backward() {
  // Allocate space used for the internal states
  std::vector<T> Q(4 * H * W * N * 6 * D);
  std::vector<T> dQ(4 * H * W * N * 6 * D);
  // Output
  std::vector<T> O(H * W * N * 4 * D);
  // Derivative w.r.t. input
  std::vector<T> dI(H * W * N * K);
  // Derivative w.r.t. parameters
  std::vector<T> dP(P<T>().size());

  // Forward pass
  lstm_2d_fw_cpu< T, Linear<T>, Linear<T>, Linear<T> >(
      H, W, N, K, D, I<T>().data(), S, P<T>().data(), O.data(), Q.data());
  // Backward pass
  lstm_2d_bw_cpu< T, Linear<T>, Linear<T>, Linear<T> >(
      H, W, N, K, D, I<T>().data(), S, P<T>().data(), O.data(), Q.data(),
      dO<T>().data(), dQ.data(), dI.data(), dP.data());

  // Check dJ/dI
  const T sum_dI = std::accumulate(dI.begin(), dI.end(), static_cast<T>(0));
  EXPECT_TRUE(FloatRelativeEq<T>(expected_sum_dI<T>(), sum_dI, 1E-5))
      << "Expected = " << expected_sum_dI<T>() << " vs. actual = " << sum_dI;

  // Check dJ/dP
  const T sum_dP = std::accumulate(dP.begin(), dP.end(), static_cast<T>(0));
  EXPECT_TRUE(FloatRelativeEq<T>(expected_sum_dP<T>(), sum_dP, 1E-5))
      << "Expected = " << expected_sum_dP<T>() << " vs. actual = " << sum_dP;
}

TEST(lstm_cpu_test, forward) {
  test_forward<float>(FloatRelativeEqPointwise(1E-5));
  test_forward<double>(DoubleRelativeEqPointwise(1E-5));
}

TEST(lstm_cpu_test, backward) {
  test_backward<float>();
  test_backward<double>();
}

#include <gmock/gmock.h>

#include "test/internal/cpu/lstm2d_cell_standard_inference.h"

#include <rnn2d/internal/cpu/lstm2d_cell_standard.h>
#include <include/rnn2d/internal/cpu/lstm2d_inference_high_mem_impl.h>



namespace rnn2d {
namespace internal {
namespace cpu {
namespace testing {

typedef ::testing::Types<
    Lstm2dInferenceHighMemImpl<float,  Lstm2dCell<float>>,
    Lstm2dInferenceHighMemImpl<double, Lstm2dCell<double>>
> MyTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Lstm2dInferenceHighMemImpl, Lstm2dStandardCellInferenceCpuTest, MyTypes);

}  // namespace testing
}  // namespace cpu
}  // namespace internal
}  // namespace rnn2d



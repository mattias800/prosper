// When live compute samples a guest FP16 texture natively rather than through the RGBA8
// conversion (frontends/shared/compute/sampled_float16_view.hpp).
#include "shared/compute/sampled_float16_view.hpp"

#include <gtest/gtest.h>

using prosper::frontend::sample_float16_natively;
using prosper::frontend::SampledFloat16Source;

TEST(SampledFloat16View, GuestBackedNarrowFloat16IsNative) {
    // Signed data such as a DOF circle of confusion: the conversion would clamp it to [0, 1].
    EXPECT_TRUE(sample_float16_natively({1, false, false, false, false}));
    EXPECT_TRUE(sample_float16_natively({2, false, false, false, false}));
}

TEST(SampledFloat16View, GuestBackedRgba16fKeepsTheConversion) {
    EXPECT_FALSE(sample_float16_natively({4, false, false, false, false}));
    EXPECT_FALSE(sample_float16_natively({3, false, false, false, false}));
    EXPECT_TRUE(sample_float16_natively({4, false, false, false, true}));   // a volume
    EXPECT_TRUE(sample_float16_natively({4, true, false, true, false}));    // a renderer import
}

TEST(SampledFloat16View, RendererOwnedNarrowNeedsTheRendererToHoldItNarrow) {
    EXPECT_TRUE(sample_float16_natively({2, true, true, false, false}));
    EXPECT_FALSE(sample_float16_natively({2, true, false, false, false}));
    EXPECT_FALSE(sample_float16_natively({1, true, false, false, false}));
}

// Scalar-buffer bounds are byte bounds, including the established empty stride-zero first dword.
// Shared fold/materializer checks must not turn unrelated bytes into a constant-buffer binding.
#include "gpu/execute/scalar_buffer_bounds.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>
#include <iterator>

namespace prosper::gpu {
namespace {
Rdna2Inst scalar_load(uint32_t offset) {
    const uint32_t words[]{0xf420000cu, 0xfa000000u | offset};
    return rdna2_decode_one(words, std::size(words));
}

TEST(ScalarBufferBounds, NonemptyUnstridedRecordsCountBytes) {
    const uint32_t words[]{0u, 0u, 16u, 0u};
    EXPECT_EQ(scalar_buffer_dword_count(decode_buffer_descriptor(words)), 4u);
}

TEST(ScalarBufferBounds, NonemptyStridedRecordsCountTheirFootprint) {
    const uint32_t words[]{0u, 16u << 16u, 3u, 0u};
    EXPECT_EQ(scalar_buffer_dword_count(decode_buffer_descriptor(words)), 12u);
}

TEST(ScalarBufferBounds, EmptyUnstridedDescriptorKeepsItsFirstDword) {
    const uint32_t words[]{0u, 0u, 0u, 0u};
    const auto descriptor = decode_buffer_descriptor(words);
    EXPECT_EQ(scalar_buffer_dword_count(descriptor), 1u);
    EXPECT_FALSE(sbuffer_access_is_fully_oob(scalar_load(0u), descriptor));
    EXPECT_TRUE(sbuffer_access_is_fully_oob(scalar_load(4u), descriptor));
}

TEST(ScalarBufferBounds, EmptyStridedDescriptorHasNoFirstDword) {
    const uint32_t words[]{0u, 16u << 16u, 0u, 0u};
    const auto descriptor = decode_buffer_descriptor(words);
    EXPECT_EQ(scalar_buffer_dword_count(descriptor), 0u);
    EXPECT_TRUE(sbuffer_access_is_fully_oob(scalar_load(0u), descriptor));
}

TEST(ScalarBufferBounds, UnknownOffsetDoesNotSpecializeANonemptyDescriptor) {
    const uint32_t words[]{0u, 0u, 16u, 0u};
    const auto descriptor = decode_buffer_descriptor(words);
    EXPECT_FALSE(sbuffer_access_is_fully_oob(scalar_load(16u), descriptor));
    EXPECT_TRUE(sbuffer_access_is_fully_oob(scalar_load(0u), descriptor, true, 16u));
}

TEST(ScalarBufferBounds, RawPointerLoadIsNotAnEmptyBufferProof) {
    const uint32_t descriptor_words[]{0u, 0u, 0u, 0u};
    const uint32_t instruction_words[]{0xf400000cu, 0xfa000004u};
    EXPECT_FALSE(sbuffer_access_is_fully_oob(
        rdna2_decode_one(instruction_words, std::size(instruction_words)),
        decode_buffer_descriptor(descriptor_words)));
}
} // namespace
} // namespace prosper::gpu

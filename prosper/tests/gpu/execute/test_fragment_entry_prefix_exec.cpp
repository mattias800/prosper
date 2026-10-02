// #4209: actual shipping-prepared user words -> original guest MOV -> raw EXP on the existing
// logical64 runner. Supplied packet topology is NOT captured guest-wave or live draw admission.
#include "fragment_entry_prefix_fixture.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <cstring>
#include <cstdio>

namespace e = prosper::test::entry_prefix;
namespace {
std::vector<uint32_t> execute(const prosper::gpu::FragmentPacketProgram& program, uint32_t& attempts) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::vector<float> input(program.input_words.size());
    std::memcpy(input.data(), program.input_words.data(), input.size() * sizeof(float));
    ++attempts;
    std::fprintf(stderr, "[entry-prefix-gpu] dispatch_attempt=%u logical_slots=64 output_words=%zu\n",
                 attempts, program.output_words.size());
    const auto out = prosper::test::run_compute(program.spirv, input, 64,
        static_cast<uint32_t>(program.output_words.size()));
    std::vector<uint32_t> words(out.size());
    if (!out.empty()) std::memcpy(words.data(), out.data(), words.size() * sizeof(uint32_t));
    return words;
}
}

TEST(FragmentEntryPrefixExec, PreparedWordsReachRawExports) {
    // Registered only in the Vulkan CMake gate. An absent runtime fails visibly, never passes
    // without executing. This integer program claims no unrelated device's float capabilities.
    const auto subgroup = prosper::test::default_compute_subgroup_properties();
    std::fprintf(stderr, "[entry-prefix-gpu] queried_physical_subgroup_size=%u expected_dispatch_attempts=6\n",
                 subgroup.size);
    ASSERT_GT(subgroup.size, 0u) << "missing runtime is not execution evidence";
    uint32_t attempts = 0;
    prosper::gpu::reset_float_controls_support_for_test();
    for (uint32_t count : {0u, 16u, 17u, 31u, 32u}) {
        SCOPED_TRACE(count);
        const uint32_t source = count ? count - 1 : 128; // count0 uses ISA inline zero, no seed
        auto in = e::inputs(count, source);
        const uint32_t value = count ? in->entry.user_data[source] : 0;
        const auto prepared = prosper::gpu::prepare_fragment_packet_inputs(in, true);
        ASSERT_TRUE(prepared->user_sgpr_count_available);
        const auto p = e::packet(*prepared);
        const auto program = prosper::gpu::recompile_fragment_packet(p);
        ASSERT_FALSE(program.spirv.empty()) << program.rejection;
        EXPECT_EQ(execute(program, attempts), e::expected(value)) << "all64 original EXP records";
    }
    // Present zero is executable authority; absence is tested separately as a compiler refusal.
    const auto prepared = prosper::gpu::prepare_fragment_packet_inputs(e::inputs(1), true);
    const auto program = prosper::gpu::recompile_fragment_packet(e::packet(*prepared));
    ASSERT_FALSE(program.spirv.empty()) << program.rejection;
    EXPECT_EQ(execute(program, attempts), e::expected(0));
    EXPECT_EQ(attempts, 6u);
    std::fprintf(stderr, "[entry-prefix-gpu] dispatch_attempts=%u expected=6\n", attempts);
}

TEST(FragmentEntryPrefixExec, ProducingValueMutationChangesActualSink) {
    const auto subgroup = prosper::test::default_compute_subgroup_properties();
    std::fprintf(stderr, "[entry-prefix-gpu] queried_physical_subgroup_size=%u expected_dispatch_attempts=2\n",
                 subgroup.size);
    ASSERT_GT(subgroup.size, 0u) << "missing runtime is not execution evidence";
    uint32_t attempts = 0;
    prosper::gpu::reset_float_controls_support_for_test();
    auto in = e::inputs(17, 16);
    const auto original = prosper::gpu::prepare_fragment_packet_inputs(in, true);
    const uint32_t first_value = in->entry.user_data[16];
    in->entry.user_data[16] ^= 0x80000001u;
    const uint32_t next_value = in->entry.user_data[16];
    const auto changed = prosper::gpu::prepare_fragment_packet_inputs(in, true);
    const auto first = prosper::gpu::recompile_fragment_packet(e::packet(*original));
    const auto next = prosper::gpu::recompile_fragment_packet(e::packet(*changed));
    ASSERT_FALSE(first.spirv.empty()) << first.rejection;
    ASSERT_FALSE(next.spirv.empty()) << next.rejection;
    EXPECT_EQ(execute(first, attempts), e::expected(first_value)) << "prior owned preparation retains words";
    const auto actual = execute(next, attempts);
    EXPECT_NE(actual, e::expected(first_value)) << "old oracle detects a real producing-word change";
    EXPECT_EQ(actual, e::expected(next_value));
    EXPECT_EQ(attempts, 2u);
    std::fprintf(stderr, "[entry-prefix-gpu] dispatch_attempts=%u expected=2\n", attempts);
}

// SMEM wrappers receive IDs, never guest registers. They must leave selector publication to callers.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/lower/smem_index_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrLowerSmemIndex, BothSourceRoutesKeepImmediateBitsAndComponentUnits) {
    for (uint32_t immediate : {0u, 1u, 3u, 4u, 0xfffffffcu}) {
        Rdna2Inst in{};
        in.literal = immediate;
        SpirvCompute b;
        b.begin(1, nullptr, 64, 1, 1, 64, 0, true, true);
        const uint32_t input = b.load_input(0);
        const uint32_t pc = lower_smem_pcrel_index(b, input, in);
        const uint32_t dynamic = lower_smem_dynamic_index(b, input, in);
        const std::array<uint32_t, 4> values{pc, dynamic, lower_smem_pcrel_component(b, pc, 15),
                                             lower_smem_dynamic_component(b, dynamic, 15)};
        for (uint32_t k = 0; k < values.size(); ++k) b.store_output_word(values[k], 4, k);
        bpermute_oracle::Interpreter vm(b.finish());
        std::vector<uint32_t> inputs;
        inputs.reserve(64);
        for (uint32_t lane = 0; lane < 64; ++lane)
            inputs.push_back(lane < 4 ? std::array{0u, 3u, 0xfffffffeu, 0xffffffffu}[lane]
                                      : 0x81234567u ^ lane * 0x02030405u);
        const auto out =
            vm.run_buffers(64, {{0, inputs}, {1, std::vector<uint32_t>(size_t{64} * 4)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u * 4u);
        for (size_t lane = 0; lane < inputs.size(); ++lane) {
            const uint32_t sum = inputs[lane] + immediate;
            const uint32_t index = sum / 4u;
            EXPECT_EQ(out[lane * 4], index);
            EXPECT_EQ(out[lane * 4 + 1], index);
            EXPECT_EQ(out[lane * 4 + 2], index + 15u);
            EXPECT_EQ(out[lane * 4 + 3], index + 15u);
        }
    }
}
TEST(IrLowerSmemIndex, ZeroComponentDoesNotAppendAnInstructionOrChangeTheId) {
    SpirvCompute b;
    b.begin(1, nullptr, 32, 1, 1, 32, 0, true, true);
    const uint32_t input = b.load_input(0);
    const auto size = b.code.size();
    EXPECT_EQ(lower_smem_pcrel_component(b, input, 0), input);
    EXPECT_EQ(lower_smem_dynamic_component(b, input, 0), input);
    EXPECT_EQ(b.code.size(), size);
}
TEST(IrLowerSmemIndex, TableAndQuotientWrappersDoNotPublishSelectorState) {
    SpirvCompute b;
    b.begin(1, nullptr, 32, 1, 1, 32, 0, true, true);
    const uint32_t input = b.load_input(0);
    b.cbuf_table_selector_value[7] = input;
    b.cbuf_table_selector_block[7] = b.cur_block;
    const auto values_before = b.cbuf_table_selector_value;
    const auto blocks_before = b.cbuf_table_selector_block;
    constexpr std::array<uint32_t, 3> table{0x12345678, 0xabcdef01, 0x56781234};
    const uint32_t word = lower_smem_constant_word(b, input, table);
    const uint32_t selector = lower_smem_table_selector(b, input, 3);
    EXPECT_EQ(b.cbuf_table_selector_value, values_before);
    EXPECT_EQ(b.cbuf_table_selector_block, blocks_before);
    b.store_output_word(word, 2, 0);
    b.store_output_word(selector, 2, 1);
    bpermute_oracle::Interpreter vm(b.finish());
    std::vector<uint32_t> inputs;
    inputs.reserve(32);
    for (uint32_t lane = 0; lane < 32; ++lane)
        inputs.push_back(lane < 4 ? lane : 0xffffffffu - lane);
    const auto out =
        vm.run_buffers(32, {{0, inputs}, {1, std::vector<uint32_t>(size_t{32} * 2)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (size_t lane = 0; lane < inputs.size(); ++lane) {
        const auto x = inputs[lane];
        EXPECT_EQ(out[lane * 2], x < table.size() ? table[x] : 0u);
        EXPECT_EQ(out[lane * 2 + 1], x / 3u);
    }
}
}   // namespace

// SMEM byte offsets wrap before truncation; constant words and runtime table quotients stay pure.
// Hand-worked boundary values constrain semantics independently of evaluator/emitter agreement.
#include "fixtures/smem_index_ir.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/smem_index.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include "gpu/recompiler/ir/ir.hpp"
#include <optional>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace fixture = prosper::test::ir::smem_index;
namespace {
Known value(const SmemIndexBlock& built, Known input) {
    return evaluate(built.block, std::array{input})[built.value.index];
}
TEST(IrSmemIndexValues, DwordIndexAddsSignedImmediateBitsBeforeTruncating) {
    struct Case {
        uint32_t input, immediate, expected;
    };
    constexpr std::array<Case, 10> cases{{{0, 0, 0},
                                          {3, 0, 0},
                                          {3, 1, 1},
                                          {4, 0xfffffffc, 0},
                                          {0, 0xfffffffc, 0x3fffffff},
                                          {0xffffffff, 1, 0},
                                          {0xfffffffe, 1, 0x3fffffff},
                                          {0x80000000, 4, 0x20000001},
                                          {0xfffffff0, 31, 3},
                                          {0, 0x1fffff, 0x7ffff}}};
    for (const auto& c : cases)
        EXPECT_EQ(value(build_smem_dword_index(c.immediate), c.input), Known(c.expected));
}
TEST(IrSmemIndexValues, ComponentZeroIsIdentityAndOtherComponentsWrap) {
    const auto identity = build_smem_component_index(0);
    ASSERT_EQ(identity.block.insts().size(), 1u);
    EXPECT_EQ(value(identity, 0xffffffffu), Known(0xffffffffu));
    EXPECT_EQ(value(build_smem_component_index(1), 0xffffffffu), Known(0u));
    EXPECT_EQ(value(build_smem_component_index(15), 0xfffffff8u), Known(7u));
    EXPECT_EQ(value(build_smem_component_index(0xffffffffu), 0u), Known(0xffffffffu));
}
TEST(IrSmemIndexValues, ConstantWordSelectsEveryEntryAndZeroOutsideTheTable) {
    const auto built = build_smem_constant_word(fixture::words);
    for (uint32_t index = 0; index < fixture::words.size(); ++index)
        EXPECT_EQ(value(built, index), Known(fixture::words[index]));
    EXPECT_EQ(value(built, 5u), Known(0u));
    EXPECT_EQ(value(built, 0xffffffffu), Known(0u));
    EXPECT_EQ(value(build_smem_constant_word({}), 2u), Known(0u));
}
TEST(IrSmemIndexValues, UnsignedTableQuotientUsesFullByteOffsetAndTruncates) {
    struct Case {
        uint32_t input, stride, expected;
    };
    constexpr std::array<Case, 9> cases{{{0, 120, 0},
                                         {119, 120, 0},
                                         {120, 120, 1},
                                         {239, 120, 1},
                                         {240, 120, 2},
                                         {0xffffffff, 3, 0x55555555},
                                         {0xffffffff, 0x80000000, 1},
                                         {0x80000000, 1, 0x80000000},
                                         {0x80000000, 0xffffffff, 0}}};
    for (const auto& c : cases)
        EXPECT_EQ(value(build_smem_table_selector(c.stride), c.input), Known(c.expected));
}
TEST(IrSmemIndexValues, UnknownInputAndZeroDivisorNeverInventAWord) {
    for (const auto& built :
         {build_smem_dword_index(4), build_smem_component_index(3),
          build_smem_constant_word(fixture::words), build_smem_table_selector(120)})
        EXPECT_FALSE(value(built, std::nullopt));
    EXPECT_FALSE(value(build_smem_table_selector(0), 0u));
    EXPECT_FALSE(value(build_smem_table_selector(0), 123u));
    EXPECT_EQ(value(build_smem_constant_word({}), std::nullopt), Known(0u));
}
TEST(IrSmemIndexValues, SharedFixtureExecutesEveryDefinitionAgainstEvalAndIndependentValues) {
    for (uint32_t wave : {32u, 64u}) {
        SpirvCompute b;
        b.begin(1, nullptr, wave, 1, 1, wave, 0, true, true);
        fixture::emit_fixture(b);
        bpermute_oracle::Interpreter vm(b.finish());
        std::vector<uint32_t> inputs;
        inputs.reserve(wave);
        for (uint32_t lane = 0; lane < wave; ++lane)
            inputs.push_back(
                lane < 8 ? std::array{0u, 1u, 2u, 3u, 4u, 5u, 0xfffffffeu, 0xffffffffu}[lane]
                         : 0x91234567u ^ (lane * 0x01020305u));
        const auto out =
            vm.run_buffers(wave, {{0, inputs}, {1, std::vector<uint32_t>(size_t{wave} * 4)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), size_t{wave} * 4);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const uint32_t x = inputs[lane];
            const uint32_t wrapped = x - 4u;
            const std::array<uint32_t, 4> expected{
                wrapped / 4u, x + 3u, x < fixture::words.size() ? fixture::words[x] : 0u, x / 120u};
            const std::array<SmemIndexBlock, 4> definitions{
                build_smem_dword_index(0xfffffffcu), build_smem_component_index(3),
                build_smem_constant_word(fixture::words), build_smem_table_selector(120)};
            for (size_t k = 0; k < expected.size(); ++k) {
                EXPECT_EQ(value(definitions[k], x), Known(expected[k]));
                EXPECT_EQ(out[size_t{lane} * 4 + k], expected[k]) << "lane=" << lane << " k=" << k;
            }
        }
    }
}
TEST(IrSmemIndexValues, EveryDefinitionPreservesInactiveLaneDestination) {
    for (const auto& built :
         {build_smem_dword_index(0xfffffffcu), build_smem_component_index(3),
          build_smem_constant_word(fixture::words), build_smem_table_selector(120)}) {
        constexpr uint32_t wave = 64;
        SpirvCompute b;
        b.begin(3, nullptr, wave, 1, 1, wave, 0, true, true);
        const uint32_t input = b.load_input(0), enabled = b.load_input(1), old = b.load_input(2);
        const uint32_t result = emit_smem_index_value(b, built, input);
        const uint32_t active = b.ucmp(Op_INotEqual, enabled, b.uconst(0));
        b.store_output_word(b.sel(active, result, old), 1, 0);
        bpermute_oracle::Interpreter vm(b.finish());
        std::vector<uint32_t> inputs, expected;
        inputs.reserve(size_t{wave} * 3);
        expected.reserve(wave);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const uint32_t x = lane < 6 ? lane : 0xffffffc0u + lane;
            const uint32_t previous = 0xa5000000u + lane;
            const bool active_lane = lane % 3u != 0;
            inputs.insert(inputs.end(), {x, active_lane ? 1u : 0u, previous});
            const auto evaluated = value(built, x);
            ASSERT_TRUE(evaluated);
            expected.push_back(active_lane ? evaluated.value_or(0) : previous);
        }
        EXPECT_EQ(vm.run_buffers(wave, {{0, inputs}, {1, std::vector<uint32_t>(wave)}}, 1),
                  expected);
        EXPECT_TRUE(vm.error.empty()) << vm.error;
    }
}
TEST(IrSmemIndexValues, UdivPrimitiveHandlesVaryingNonPowerOfTwoDivisors) {
    Block block;
    Builder ir(block);
    const Value a = ir.input(Type::U32), c = ir.input(Type::U32);
    const Value quotient = ir.udiv(a, c);
    for (uint32_t numerator : {0u, 1u, 17u, 0x80000000u, 0xffffffffu})
        for (uint32_t divisor : {1u, 3u, 7u, 120u, 0x80000000u, 0xffffffffu}) {
            const std::array<Known, 2> inputs{numerator, divisor};
            EXPECT_EQ(evaluate(block, inputs)[quotient.index], Known(numerator / divisor));
        }
    EXPECT_FALSE(evaluate(block, std::array<Known, 2>{123u, 0u})[quotient.index]);
    // The same two-input definition also runs emitted OpUDiv, with distinct values per lane.
    SpirvCompute b;
    b.begin(2, nullptr, 64, 1, 1, 64, 0, true, true);
    const std::array<uint32_t, 2> sources{b.load_input(0), b.load_input(1)};
    const auto ids = emit_spirv(b, block, sources);
    b.store_output_word(ids[quotient.index], 1, 0);
    bpermute_oracle::Interpreter vm(b.finish());
    std::vector<uint32_t> inputs, expected;
    inputs.reserve(128);
    expected.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t numerator = 0xffffffffu - lane * 0x01020305u;
        const uint32_t divisor = std::array{1u, 3u, 7u, 120u, 0x80000000u, 0xffffffffu}[lane % 6];
        inputs.insert(inputs.end(), {numerator, divisor});
        expected.push_back(numerator / divisor);
    }
    EXPECT_EQ(vm.run_buffers(64, {{0, inputs}, {1, std::vector<uint32_t>(64)}}, 1), expected);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
}
} // namespace

// Hand-worked U32 arithmetic, including low-address wrap, constrains eval independently.
#include "fixtures/flat_window_ir.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/flat_window.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
TEST(IrFlatWindowValues, SubtractsLowWordsModuloTwoToThe32) {
    const auto built = build_flat_window_byte_offset();
    struct Case {
        uint32_t address, base, expected;
    };
    constexpr std::array<Case, 6> cases{{{0x100004, 0x100000, 4},
                                         {0x100000, 0x100000, 0},
                                         {0xcu, 0xfffffff0u, 0x1cu},
                                         {0, 1, 0xffffffffu},
                                         {0xffffffffu, 0x80000000u, 0x7fffffffu},
                                         {0x80000000u, 0x7fffffffu, 1}}};
    for (const auto& c : cases) {
        const auto result = evaluate(built.block, std::array<Known, 2>{c.address, c.base});
        EXPECT_EQ(result[built.byte_offset.index], Known(c.expected));
    }
}
TEST(IrFlatWindowValues, ComponentsAddBytesOnlyForDwordsBeforeTruncation) {
    struct Case {
        uint32_t byte0, bits, component, address, index;
    };
    constexpr std::array<Case, 8> cases{{{7, 8, 3, 7, 1},
                                         {6, 16, 3, 6, 1},
                                         {7, 32, 0, 7, 1},
                                         {7, 32, 3, 19, 4},
                                         {0xfffffffcu, 32, 1, 0, 0},
                                         {0xffffffffu, 32, 1, 3, 0},
                                         {0x80000000u, 32, 3, 0x8000000cu, 0x20000003u},
                                         {0xffffffffu, 16, 0, 0xffffffffu, 0x3fffffffu}}};
    for (const auto& c : cases) {
        const auto built = build_flat_window_component(c.bits, c.component);
        const auto result = evaluate(built.block, std::array<Known, 1>{c.byte0});
        EXPECT_EQ(result[built.byte_address.index], Known(c.address));
        EXPECT_EQ(result[built.dword_index.index], Known(c.index));
    }
}
TEST(IrFlatWindowValues, UnknownOperandsNeverInventAnOffsetOrIndex) {
    const auto base = build_flat_window_byte_offset();
    for (const auto& inputs : std::array<std::array<Known, 2>, 3>{
             {{std::nullopt, 0u}, {0u, std::nullopt}, {std::nullopt, std::nullopt}}})
        EXPECT_FALSE(evaluate(base.block, inputs)[base.byte_offset.index]);
    for (uint32_t bits : {8u, 16u, 32u}) {
        const auto built = build_flat_window_component(bits, 3);
        const auto result = evaluate(built.block, std::array<Known, 1>{std::nullopt});
        EXPECT_FALSE(result[built.byte_address.index]);
        EXPECT_FALSE(result[built.dword_index.index]);
    }
}
TEST(IrFlatWindowValues, EmittedValuesMatchEvalAndIndependentMathPerLaneUnderPartialExec) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t bits : {8u, 16u, 32u})
            for (uint32_t component : {0u, 1u, 3u}) {
                SpirvCompute b;
                b.begin(4, nullptr, wave, 1, 1, wave, 0, true, true);
                prosper::test::ir::flat_window::emit_fixture(b, bits, component);
                bpermute_oracle::Interpreter vm(b.finish());
                std::vector<uint32_t> input;
                input.reserve(size_t{wave} * 4);
                for (uint32_t lane = 0; lane < wave; ++lane)
                    input.insert(input.end(),
                                 {0xfffffff0u + lane * 0x01020409u, 0x81234567u ^ lane * 0x10203u,
                                  lane % 3u ? 1u : 0u, 0xd0000000u | lane});
                const auto actual = vm.run_buffers(
                    wave, {{0, input}, {1, std::vector<uint32_t>(size_t{wave} * 3)}}, 1);
                ASSERT_TRUE(vm.error.empty()) << vm.error;
                ASSERT_EQ(actual.size(), size_t{wave} * 3);
                const auto offset = build_flat_window_byte_offset();
                const auto comp = build_flat_window_component(bits, component);
                for (uint32_t lane = 0; lane < wave; ++lane) {
                    const uint32_t byte0 = input[lane * 4] - input[lane * 4 + 1];
                    const uint32_t address = byte0 + (bits == 32 ? component * 4 : 0);
                    const std::array expected{byte0, address, address / 4u};
                    const auto a = evaluate(
                        offset.block, std::array<Known, 2>{input[lane * 4], input[lane * 4 + 1]});
                    const auto c = evaluate(comp.block, std::array{a[offset.byte_offset.index]});
                    EXPECT_EQ(a[offset.byte_offset.index], Known(expected[0]));
                    EXPECT_EQ(c[comp.byte_address.index], Known(expected[1]));
                    EXPECT_EQ(c[comp.dword_index.index], Known(expected[2]));
                    for (size_t k = 0; k < expected.size(); ++k)
                        EXPECT_EQ(actual[size_t{lane} * 3 + k],
                                  input[lane * 4 + 2] ? expected[k] : input[lane * 4 + 3]);
                }
            }
}
} // namespace

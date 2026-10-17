// DS byte, paired and component indexes are pure uint32 IR values. Hand arithmetic and the
// interpreter pin wrap, ordering and missing-input behavior independently of the lowering caller.
#include "fixtures/ds_address_ir.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/ds_address.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {
uint32_t only(const DsAddressBlock& built, std::span<const Known> inputs) {
    const auto values = evaluate(built.block, inputs);
    EXPECT_TRUE(values[built.outputs[0].index]);
    return values[built.outputs[0].index].value_or(0);
}

std::vector<uint32_t> execute_fixture(uint32_t lanes, const std::vector<uint32_t>& a,
                                      const std::vector<uint32_t>& c) {
    SpirvCompute b;
    b.begin(2, nullptr, lanes, 1, 1, lanes, 0, true, true);
    prosper::test::ir::ds_address::emit_fixture(b);
    bpermute_oracle::Interpreter vm(b.finish());
    std::vector<uint32_t> inputs;
    inputs.reserve(size_t{lanes} * 2);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        inputs.push_back(a[lane]);
        inputs.push_back(c[lane]);
    }
    auto out =
        vm.run_buffers(lanes, {{0, inputs}, {1, std::vector<uint32_t>(size_t{lanes} * 9)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}

TEST(IrDsAddressValues, DefinitionsMatchHandWorkedUint32Boundaries) {
    constexpr std::array<uint32_t, 8> addresses{0u,          1u,          3u,          4u,
                                                0xfffffffcu, 0xfffffffdu, 0xfffffffeu, 0xffffffffu};
    for (uint32_t literal : {0u, 1u, 3u, 4u, 65535u, 0xffffffffu})
        for (bool gds : {false, true}) {
            const auto built = build_ds_byte_address(literal, gds);
            for (uint32_t address : addresses) {
                const std::array<Known, 1> input{address};
                uint32_t sum = address + literal;
                if (gds) sum &= 0xffffu;
                EXPECT_EQ(only(built, input), sum / 4u)
                    << "literal=" << literal << " gds=" << gds << " address=" << address;
            }
        }

    constexpr std::array<std::array<uint32_t, 2>, 5> offsets{
        {{0u, 1u}, {2u, 2u}, {255u, 256u}, {16320u, 0u}, {0xffffffffu, 2u}}};
    for (const auto [offset0, offset1] : offsets) {
        const auto built = build_ds_pair_address(offset0, offset1);
        for (uint32_t address : addresses) {
            const std::array<Known, 1> input{address};
            const auto values = evaluate(built.block, input);
            const uint32_t base = address / 4u;
            ASSERT_TRUE(values[built.outputs[0].index]);
            ASSERT_TRUE(values[built.outputs[1].index]);
            EXPECT_EQ(values[built.outputs[0].index].value_or(0), base + offset0);
            EXPECT_EQ(values[built.outputs[1].index].value_or(0), base + offset1);
        }
    }

    const auto words = build_ds_pair_words();
    for (uint32_t a : addresses) {
        const std::array<Known, 2> input{a, 0xfffffffeu};
        const auto values = evaluate(words.block, input);
        const std::array<uint32_t, 4> expected{a, a + 1u, 0xfffffffeu, 0xffffffffu};
        for (size_t k = 0; k < expected.size(); ++k)
            EXPECT_EQ(values[words.outputs[k].index], Known(expected[k]));
    }

    for (uint32_t component : {0u, 1u, 2u, 3u, 0xffffffffu}) {
        const auto built = build_ds_word_index(component);
        for (uint32_t index : addresses) {
            const std::array<Known, 1> input{index};
            EXPECT_EQ(only(built, input), index + component);
        }
    }
}

TEST(IrDsAddressValues, MissingInputsStayUnknownAndPairWordsKeepIndependentHalves) {
    const auto byte = build_ds_byte_address(3, false);
    EXPECT_FALSE(evaluate(byte.block, {})[byte.outputs[0].index]);
    const auto pair = build_ds_pair_address(3, 5);
    const auto pair_values = evaluate(pair.block, {});
    EXPECT_FALSE(pair_values[pair.outputs[0].index]);
    EXPECT_FALSE(pair_values[pair.outputs[1].index]);
    const auto words = build_ds_pair_words();
    const std::array<Known, 1> only_a{0xfffffffeu};
    const auto word_values = evaluate(words.block, only_a);
    EXPECT_EQ(word_values[words.outputs[0].index], Known(0xfffffffeu));
    EXPECT_EQ(word_values[words.outputs[1].index], Known(0xffffffffu));
    EXPECT_FALSE(word_values[words.outputs[2].index]);
    EXPECT_FALSE(word_values[words.outputs[3].index]);
    const auto component = build_ds_word_index(3);
    EXPECT_FALSE(evaluate(component.block, {})[component.outputs[0].index]);
}

TEST(IrDsAddressValues, SharedFixtureEmitsNineHandCheckedWordsPerWaveLane) {
    for (uint32_t wave : {32u, 64u}) {
        std::vector<uint32_t> a, c;
        a.reserve(wave);
        c.reserve(wave);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            a.push_back(lane < 8 ? std::array{0u, 1u, 3u, 4u, 0xfffffffcu, 0xfffffffdu, 0xfffffffeu,
                                              0xffffffffu}[lane]
                                 : (0xfedcba98u ^ (lane * 0x01020305u)));
            c.push_back(lane < 8 ? std::array{0xffffffffu, 0xfffffffeu, 3u, 4u, 0x7fffffffu,
                                              0x80000000u, 1u, 0u}[lane]
                                 : (0x91827364u ^ (lane * 0x1020304u)));
        }
        const auto out = execute_fixture(wave, a, c);
        ASSERT_EQ(out.size(), size_t{wave} * 9);
        const auto plain = build_ds_byte_address(3, false);
        const auto gds = build_ds_byte_address(65535, true);
        const auto pair = build_ds_pair_address(3, 5);
        const auto words = build_ds_pair_words();
        const auto component = build_ds_word_index(3);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            uint32_t wrapped = a[lane] + 65535u;
            wrapped &= 0xffffu;
            const std::array<uint32_t, 9> expected{
                (a[lane] + 3u) / 4u, wrapped / 4u, a[lane] / 4u + 3u, a[lane] / 4u + 5u, a[lane],
                a[lane] + 1u,        c[lane],      c[lane] + 1u,      c[lane] + 3u};
            const std::array<Known, 1> address_input{a[lane]};
            const std::array<Known, 2> word_input{a[lane], c[lane]};
            const std::array<Known, 1> component_input{c[lane]};
            EXPECT_EQ(only(plain, address_input), expected[0]);
            EXPECT_EQ(only(gds, address_input), expected[1]);
            const auto pair_eval = evaluate(pair.block, address_input);
            const auto words_eval = evaluate(words.block, word_input);
            EXPECT_EQ(pair_eval[pair.outputs[0].index], Known(expected[2]));
            EXPECT_EQ(pair_eval[pair.outputs[1].index], Known(expected[3]));
            for (size_t k = 0; k < words.outputs.size(); ++k)
                EXPECT_EQ(words_eval[words.outputs[k].index], Known(expected[k + 4]));
            EXPECT_EQ(only(component, component_input), expected[8]);
            for (size_t k = 0; k < expected.size(); ++k)
                EXPECT_EQ(out[size_t{lane} * 9 + k], expected[k])
                    << "wave=" << wave << " lane=" << lane;
        }
    }
}

TEST(IrDsAddressValues, PartialActiveMaskPreservesOldValueInEmittedSpirv) {
    constexpr uint32_t lanes = 32;
    SpirvCompute b;
    b.begin(3, nullptr, lanes, 1, 1, lanes, 0, true, true);
    const uint32_t index = b.load_input(0);
    const uint32_t active_word = b.load_input(1);
    const uint32_t old = b.load_input(2);
    const auto built = build_ds_word_index(3);
    const uint32_t value = emit_ds_address_values(b, built, [&](uint32_t) { return index; })[0];
    const uint32_t active = b.ucmp(Op_INotEqual, active_word, b.uconst(0));
    b.store_output_word(b.sel(active, value, old), 1, 0);
    bpermute_oracle::Interpreter vm(b.finish());
    std::vector<uint32_t> input;
    input.reserve(size_t{lanes} * 3);
    std::vector<uint32_t> expected;
    expected.reserve(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t x = 0xfffffff0u + lane * 3u;
        const bool enabled = lane % 3u != 0;
        const uint32_t old_value = 0xa5000000u + lane;
        input.insert(input.end(), {x, enabled ? 1u : 0u, old_value});
        expected.push_back(enabled ? x + 3u : old_value);
    }
    const auto out = vm.run_buffers(lanes, {{0, input}, {1, std::vector<uint32_t>(lanes)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) EXPECT_EQ(out[lane], expected[lane]);
}

} // namespace

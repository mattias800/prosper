// Packed half binary arithmetic must agree with eval in distinct lanes without waking inactive lanes.
// Hand cases below come from AMD70648 section12.10, independently of the shared definition.
#include "gpu/recompiler/ir/ops/vector_packed_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
// GLSL.std.450 PackHalf2x16/UnpackHalf2x16 use the existing closed RTE/preserved-denormal
// interpreter profile, as other binary16 execution families do. This is not a hardware MODE claim.
LaneRun run_packed(const std::vector<uint32_t>& code, const LaneInputs& in, uint32_t wave = 64) {
    return run_vector_op_per_lane(code, in, wave, true);
}
struct Form {
    uint32_t mode = 0, lo = 0, hi = 7, neg = 0, neg_hi = 0;
    bool clamp = false;
    uint32_t alias = 0;
};
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    // Disjoint nonzero source magnitudes avoid exact-cancellation zeros at the CLAMP NMax.
    // The strict oracle intentionally refuses unspecified zero signs; that domain is checked below.
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto half = [](uint32_t word, uint32_t exponent) {
            return (word & 0x83ffu) | ((exponent + (word >> 16) % 3) << 10);
        };
        const uint32_t a = in.s0[lane], b = in.s1[lane];
        in.s0[lane] = half(a, 12) | (half(a * 1664525u + 1013904223u, 12) << 16);
        in.s1[lane] = half(b, 16) | (half(b * 22695477u + 1, 16) << 16);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t op, Form f = {}) {
    const uint32_t a = f.mode == 1 || f.mode == 3 ? 242
                       : f.mode == 4              ? 255
                       : f.mode == 6              ? 135
                                                  : 256 + (f.alias == 1 ? kVdst : kVsrc0);
    const uint32_t b = f.mode == 2 || f.mode == 3 ? 244
                       : f.mode == 5              ? 255
                                                  : 256 + (f.alias == 2 ? kVdst : kVsrc1);
    std::vector<uint32_t> code = {0xcc000000u | (op << 16) | kVdst | (f.neg_hi << 8) |
                                      (f.lo << 11) | ((f.hi & 4) << 12) | (f.clamp ? 1u << 15 : 0),
                                  a | (b << 9) | ((f.hi & 3) << 27) | (f.neg << 29)};
    if (f.mode == 4 || f.mode == 5) code.push_back(f.mode == 4 ? 0xc1003e00u : 0x4200b800u);
    return code;
}
Known half_result(const TernaryIntegerDef& def, uint32_t a, uint32_t b, Form f, bool high) {
    const uint32_t select = high ? f.hi : f.lo, neg = high ? f.neg_hi : f.neg;
    std::array<Known, 3> src = {std::nullopt, std::nullopt, std::nullopt};
    const uint32_t raw[] = {a, b};
    for (size_t k = 0; k < 2; ++k) {
        auto expanded = evaluate_op(Opcode::UnpackHalf, (select >> k) & 1u, raw[k], 0, 0);
        if (expanded && (neg & (1u << k))) expanded = *expanded ^ 0x80000000u;
        src[k] = expanded;
    }
    auto value = eval_ternary_integer_op(def, src);
    if (f.clamp && value) value = evaluate_op(Opcode::F32Clamp01, 0, *value, 0, 0);
    return value ? evaluate_op(Opcode::PackHalfLo, 0, *value, 0, 0) : std::nullopt;
}
Known expected(const TernaryIntegerDef& def, const LaneInputs& in, size_t lane, Form f) {
    const uint32_t a = f.mode == 1 || f.mode == 3 ? 0x3c00
                       : f.mode == 4              ? 0xc1003e00
                       : f.mode == 6              ? 7
                       : f.alias == 1             ? in.old_d[lane]
                                                  : in.s0[lane];
    const uint32_t b = f.mode == 2 || f.mode == 3 ? 0x4000
                       : f.mode == 5              ? 0x4200b800
                       : f.alias == 2             ? in.old_d[lane]
                                                  : in.s1[lane];
    auto lo = half_result(def, a, b, f, false), hi = half_result(def, a, b, f, true);
    return lo && hi ? Known(*lo | (*hi << 16)) : std::nullopt;
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, Form f = {}) {
    auto run = run_packed(words(def.opcode, f), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (!in.active[lane]) {
            EXPECT_EQ(run.d[lane], in.old_d[lane]) << lane;
            continue;
        }
        auto want = expected(def, in, lane, f);
        if (!want) {
            ADD_FAILURE() << "unknown lane " << lane;
            return;
        }
        EXPECT_EQ(run.d[lane], *want) << lane << " op " << def.opcode << " wave " << wave;
    }
}
Form witness_form(uint32_t form) {
    Form f;
    if (form <= 6)
        f.mode = form;
    else if (form <= 22) {
        const auto x = form - 7;
        f.lo = x & 3;
        f.hi = (x >> 2) | 4;
    } else if (form <= 30) {
        const auto x = form - 23;
        f.neg = x & 3;
        f.neg_hi = (x >> 2) * 3;
        f.clamp = (x & 1) != 0;
    } else
        f.alias = form - 30;
    return f;
}
}   // namespace
TEST(IrVectorPackedFloatExecution, EncodingsAgreeWithThePublishedLlvmRoundTrip) {
    EXPECT_EQ(words(0x0f), (std::vector<uint32_t>{0xcc0f4001, 0x18020702}));
    EXPECT_EQ(words(0x0f, {0, 1, 6, 1, 2, true}), (std::vector<uint32_t>{0xcc0fca01, 0x30020702}));
    for (const auto& def : vector_packed_float_defs()) {
        const auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].vop3p_opsel, 0u);
        EXPECT_EQ(decoded[0].vop3p_opsel_hi, 7u);
        EXPECT_EQ(decoded[0].src[0].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[0].value, kVsrc0);
        EXPECT_EQ(decoded[0].src[1].value, kVsrc1);
        EXPECT_FALSE(decoded[0].has_modifier);
    }
}
TEST(IrVectorPackedFloatExecution, EveryDefinitionMatchesDifferingLanesInBothWaves) {
    for (const auto& def : vector_packed_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, inputs(seed * 5294), wave);
}
TEST(IrVectorPackedFloatExecution, HarnessSeesDifferentHalvesAndInactiveDestinations) {
    auto in = inputs(113);
    auto run = run_packed(words(0x0f), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
}
TEST(IrVectorPackedFloatExecution, HandWorkedHalfArithmeticAndRounding) {
    // Each row: packed S0, packed S1, ADD, MUL, MIN, MAX. Low/high are independent.
    // (1.5,-.5) and (-2.25,2): sums(-.75,1.5), products(-3.375,-1).
    // Tie rounding: 1+2^-11 is halfway and rounds to even1; (1+2^-10)+2^-11 rounds to1+2^-9.
    const std::array<std::array<uint32_t, 6>, 5> cases = {
        {{0xb8003e00, 0x4000c080, 0x3e00ba00, 0xbc00c2c0, 0xb800c080, 0x40003e00},
         {0x3800bc00, 0xbc003800, 0xb800b800, 0xb800b800, 0xbc00bc00, 0x38003800},
         {0x3c013c00, 0x10001000, 0x3c023c00, 0x10011000, 0x10001000, 0x3c013c00},
         {0x04003c00, 0x3c003c00, 0x3c004000, 0x04003c00, 0x04003c00, 0x3c003c00},
         {0x7bff0001, 0x3c003c00, 0x7bff3c00, 0x7bff0001, 0x3c000001, 0x7bff3c00}}};
    for (const auto& e : cases) {
        size_t k = 2;
        for (const auto& def : vector_packed_float_defs()) {
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), e[0]);
            std::fill(in.s1.begin(), in.s1.end(), e[1]);
            const auto want = e[k++];
            EXPECT_EQ(expected(def, in, 0, {}), Known(want));
            for (uint32_t wave : {64u, 32u}) {
                auto run = run_packed(words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
    }
}
TEST(IrVectorPackedFloatExecution, IndependentSelectorsNegationAndClamp) {
    const auto in = inputs(43);
    for (const auto& def : vector_packed_float_defs())
        for (uint32_t lo = 0; lo < 4; ++lo)
            for (uint32_t hi = 0; hi < 4; ++hi)
                for (uint32_t neg = 0; neg < 4; ++neg)
                    for (uint32_t nh = 0; nh < 4; ++nh)
                        for (bool clamp : {false, true})
                            for (uint32_t wave : {64u, 32u})
                                compare(def, in, wave, {0, lo, hi | 4, neg, nh, clamp});
}
TEST(IrVectorPackedFloatExecution, HandWorkedSelectorsAndResultClamp) {
    // S0=(1.5,-2.5), S1=(-.5,3). Low chooses highS0,high chooses lowS1;
    // neg low S0 =>2.5-.5=2; neg high S1 =>-2.5+.5=-2. Clamp yields(1,0).
    auto in = inputs(51);
    std::fill(in.s0.begin(), in.s0.end(), 0xc1003e00);
    std::fill(in.s1.begin(), in.s1.end(), 0x4200b800);
    for (bool clamp : {false, true})
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_packed(words(0x0f, {0, 1, 5, 1, 2, clamp}), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            const uint32_t want = clamp ? 0x00003c00 : 0xc0004000;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
        }
}
TEST(IrVectorPackedFloatExecution, InlineAndLiteralOperandsRetainTheHalfWidth) {
    for (const auto& def : vector_packed_float_defs())
        for (uint32_t mode : {1u, 2u, 4u, 5u, 6u})
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, {mode});
    // Inline1.0 and2.0 contribute low half only, so both high inputs are+0; ADD gives+0.
    auto in = inputs(57);
    for (uint32_t wave : {64u, 32u}) {
        auto run = run_packed(words(0x0f, {3}), in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? 0x00004200u : in.old_d[lane]);
    }
}
TEST(IrVectorPackedFloatExecution, DestinationAliasesAndUnusedThirdFields) {
    auto in = inputs(61);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        in.old_d[lane] = in.s0[(lane + 3) % in.lanes()];
    for (const auto& def : vector_packed_float_defs())
        for (uint32_t wave : {64u, 32u}) {
            for (uint32_t alias : {1u, 2u}) {
                Form f;
                f.alias = alias;
                compare(def, in, wave, f);
            }
            auto code = words(def.opcode);
            code[0] |= (1u << 10) | (1u << 13);
            code[1] |= (1u << 31) | (510u << 18);
            auto base = run_packed(words(def.opcode), in, wave), extra = run_packed(code, in, wave);
            ASSERT_TRUE(base.error.empty()) << base.error;
            ASSERT_TRUE(extra.error.empty()) << extra.error;
            EXPECT_EQ(base.d, extra.d);
        }
}
TEST(IrVectorPackedFloatExecution, EffectsAndUnknownInputsRetainTheirDomains) {
    for (const auto& def : vector_packed_float_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 0x3f800000, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x3f800000, std::nullopt, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x7fc01234, 0x7fc05678, std::nullopt}));
        EXPECT_TRUE(eval_ternary_integer_op(def, {0x3f800000, 0x40000000, std::nullopt}));
    }
    EXPECT_EQ(find_vector_packed_float_op(Rdna2Format::VOP3, 0x0f), nullptr);
    for (auto op : {0x0eu, 0x13u, 0x14u, 0x20u})
        EXPECT_EQ(find_vector_packed_float_op(Rdna2Format::VOP3P, op), nullptr);
}
TEST(IrVectorPackedFloatExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): test reads a fixed process environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "packed-float-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    size_t count = 0;
    for (const auto& def : vector_packed_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t form = 0; form < 33; ++form) {
                const auto program =
                    lane_program(words(def.opcode, witness_form(form)), 64, wave, false);
                const uint64_t address = 0x5294000000ull + count * 0x100;
                const auto stem =
                    dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                bin.write(reinterpret_cast<const char*>(program.code.data()),
                          static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const fs::path file = stem.string() + ".inputs";
                ASSERT_TRUE(write_refused_compute_inputs(file, address, &program.table,
                                                         program.config,
                                                         "synthetic-witness (no live refusal)"));
                const auto read = read_refused_shader_inputs(file);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(program.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                ++count;
            }
    EXPECT_EQ(count, 264u);
}

TEST(IrVectorPackedFloatExecution, HalfConversionsRequireTheExplicitOracleProfile) {
    const auto in = inputs(73);
    const auto closed = run_vector_op_per_lane(words(0x0f), in);
    EXPECT_FALSE(closed.error.empty());
    EXPECT_NE(closed.error.find("ExtInst"), std::string::npos);
    const auto opened = run_packed(words(0x0f), in);
    EXPECT_TRUE(opened.error.empty()) << opened.error;
}

TEST(IrVectorPackedFloatExecution, AmbiguousMinmaxZeroSignsRemainOutsideTheOracle) {
    auto in = inputs(79);
    std::fill(in.s0.begin(), in.s0.end(), 0u);
    std::fill(in.s1.begin(), in.s1.end(), 0x80008000u);
    for (uint32_t op : {0x11u, 0x12u}) {
        const auto* def = find_vector_packed_float_op(Rdna2Format::VOP3P, op);
        ASSERT_NE(def, nullptr);
        EXPECT_FALSE(eval_ternary_integer_op(*def, {0u, 0x80000000u, std::nullopt}));
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_packed(words(op), in, wave);
            EXPECT_NE(run.error.find("unspecified float ALU result"), std::string::npos);
        }
    }
}

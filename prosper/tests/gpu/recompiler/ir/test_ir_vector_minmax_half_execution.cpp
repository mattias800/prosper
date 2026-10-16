// AMD70648 operations 849/852/855: selected half extrema, NaN fallback and partial EXEC.
#include "gpu/recompiler/ir/ops/vector_minmax_half.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
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
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    auto half = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed & 0x83ffu) | ((5u + (seed >> 20) % 20u) << 10);
    };
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = half() | (half() << 16);
        in.s1[lane] = half() | (half() << 16);
        in.s2.push_back(half() | (half() << 16));
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode, uint32_t sel = 0, uint32_t mode = 0, uint32_t abs = 0,
                            uint32_t neg = 0, uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 4 ? 242 : mode == 5 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 4 ? 244 : mode == 6 ? 255 : 256 + kVsrc1;
    const uint32_t c = mode == 3 || mode == 4 ? 241 : mode == 7 ? 255 : 256 + kVsrc2;
    std::vector<uint32_t> code = {0xd4000000u | (opcode << 16) | kVdst | (abs << 8) | (sel << 11) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (c << 18) | (omod << 27) | (neg << 29)};
    if (mode >= 5) code.push_back(mode == 5 ? 0x40003e00 : mode == 6 ? 0x3c00bc00 : 0xc0003800);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t sel = 0, uint32_t mode = 0,
                             uint32_t abs = 0, uint32_t neg = 0) {
    const std::array<uint32_t, 3> raw = {mode == 1 || mode == 4 ? 0x3c00u
                                         : mode == 5            ? 0x40003e00u
                                                                : in.s0[lane],
                                         mode == 2 || mode == 4 ? 0x4000u
                                         : mode == 6            ? 0x3c00bc00u
                                                                : in.s1[lane],
                                         mode == 3 || mode == 4 ? 0xb800u
                                         : mode == 7            ? 0xc0003800u
                                                                : in.s2[lane]};
    std::array<Known, 3> out;
    for (size_t k = 0; k < 3; ++k) {
        auto value = evaluate_op(Opcode::UnpackHalf, (sel >> k) & 1u, raw[k], 0, 0);
        if (!value) continue;
        if (abs & (1u << k)) *value &= 0x7fffffffu;
        if (neg & (1u << k)) value = evaluate_op(Opcode::F32Sub, 0, 0, *value, 0);
        out[k] = value;
    }
    return out;
}
Known result(const TernaryIntegerDef& def, const std::array<Known, 3>& args, uint32_t old,
             uint32_t sel, uint32_t omod = 0, bool clamp = false) {
    auto value = eval_ternary_integer_op(def, args);
    if (!value) return std::nullopt;
    if (omod)
        value = evaluate_op(Opcode::F32Mul, 0, *value,
                            omod == 1   ? 0x40000000
                            : omod == 2 ? 0x40800000
                                        : 0x3f000000,
                            0);
    if (clamp && value) value = evaluate_op(Opcode::F32Clamp01, 0, *value, 0, 0);
    if (!value) return std::nullopt;
    auto h = evaluate_op(Opcode::PackHalfLo, 0, *value, 0, 0);
    if (!h) return std::nullopt;
    return sel & 8 ? (old & 0xffffu) | (*h << 16) : (old & 0xffff0000u) | *h;
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t sel = 0,
             uint32_t mode = 0) {
    auto run = run_vector_op_per_lane(words(def.opcode, sel, mode), in, wave, true);
    // Three high-half inline floats are all +0; NMin/NMax zero-sign selection is
    // unspecified by the SPIR-V oracle. Keep this explicit negative beside the finite controls.
    if (mode == 4 && sel == 15) {
        EXPECT_TRUE(run.d.empty());
        EXPECT_EQ(run.error, "unspecified float ALU result");
        return;
    }
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        auto want = result(def, sources(in, lane, sel, mode), in.old_d[lane], sel);
        if (!want) {
            ADD_FAILURE() << "unknown lane " << lane << " op " << def.opcode;
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << lane << " wave " << wave << " sel " << sel << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorMinmaxHalfExecution, EveryDefinitionInEveryHalfWithDifferingLanes) {
    for (const auto& def : vector_minmax_half_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t sel = 0; sel < 16; ++sel)
                for (uint32_t seed = 1; seed <= 8; ++seed)
                    compare(def, inputs(seed * 5275), wave, sel);
}
TEST(IrVectorMinmaxHalfExecution, HarnessSeesDistinctOperandsAndInactiveDestinations) {
    auto in = inputs(113);
    auto run = run_vector_op_per_lane(words(0x357), in, 64, true);
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
    in.s2.pop_back();
    const auto bad = run_vector_op_per_lane(words(0x357), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorMinmaxHalfExecution, HandWorkedFiniteOrderingAndSingleNanFallback) {
    // IEEE half encodings: -2=c000, 1=3c00, 4=4400. A single NaN returns the other
    // value from each MIN/MAX; MED3 then explicitly chooses MIN3 rather than its finite formula.
    const std::array<std::array<uint32_t, 6>, 4> examples = {
        {{0xc000, 0x3c00, 0x4400, 0xc000, 0x4400, 0x3c00},
         {0x7e12, 0x3c00, 0x4400, 0x3c00, 0x4400, 0x3c00},
         {0xfc00, 0x3c00, 0x7c00, 0xfc00, 0x7c00, 0x3c00},
         {0x3c00, 0x3c00, 0x4000, 0x3c00, 0x4000, 0x3c00}}};
    const std::array<std::array<size_t, 3>, 6> perm = {
        {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
    for (const auto& e : examples)
        for (const auto& order : perm) {
            size_t op = 0;
            for (const auto& def : vector_minmax_half_defs()) {
                uint32_t want = e[3 + op++];
                auto in = inputs(19);
                std::fill(in.s0.begin(), in.s0.end(), e[order[0]] * 0x10001u);
                std::fill(in.s1.begin(), in.s1.end(), e[order[1]] * 0x10001u);
                std::fill(in.s2.begin(), in.s2.end(), e[order[2]] * 0x10001u);
                for (uint32_t sel : {0u, 15u}) {
                    auto args = sources(in, 0, sel);
                    // Unpack's NaN payload is unspecified. The single-NaN extrema result
                    // is payload-independent, so independently evaluate a representative quiet NaN.
                    for (size_t k = 0; k < 3; ++k)
                        if (!args[k]) {
                            ASSERT_EQ(e[order[k]] & 0x7c00u, 0x7c00u);
                            ASSERT_NE(e[order[k]] & 0x3ffu, 0u);
                            args[k] = 0x7fc00000u;
                        }
                    auto got = result(def, args, 0xdeadbeef, sel);
                    if (!got) {
                        ADD_FAILURE() << "unknown hand-worked result";
                        return;
                    }
                    EXPECT_EQ(*got, sel & 8 ? 0xbeefu | (want << 16) : 0xdead0000u | want);
                    for (uint32_t wave : {64u, 32u}) {
                        auto run = run_vector_op_per_lane(words(def.opcode, sel), in, wave, true);
                        // The existing interpreter refuses half NaN transport before arithmetic.
                        // Keep that boundary explicit; the definition above separately checks
                        // the payload-independent result against the hand-worked ISA example.
                        if (e[0] == 0x7e12u) {
                            EXPECT_TRUE(run.d.empty());
                            EXPECT_EQ(run.error, "unspecified half NaN payload");
                            continue;
                        }
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        for (size_t lane = 0; lane < in.lanes(); ++lane) {
                            uint32_t merged = sel & 8 ? (in.old_d[lane] & 0xffffu) | (want << 16)
                                                      : (in.old_d[lane] & 0xffff0000u) | want;
                            EXPECT_EQ(run.d[lane], in.active[lane] ? merged : in.old_d[lane]);
                        }
                    }
                }
            }
        }
}
TEST(IrVectorMinmaxHalfExecution, InlineAndLiteralSourcesSelectTheirOwnHighAndLowHalves) {
    for (const auto& def : vector_minmax_half_defs())
        for (uint32_t mode = 1; mode <= 7; ++mode)
            for (uint32_t sel : {0u, 15u})
                for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, sel, mode);
}
TEST(IrVectorMinmaxHalfExecution, CallerModifiersPrecedeDestinationHalfPacking) {
    auto in = inputs(43);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0xb800b800;
        in.s1[lane] = 0x3c003c00;
        in.s2[lane] = 0x40004000;
    }
    for (const auto& def : vector_minmax_half_defs())
        for (uint32_t abs = 0; abs < 8; ++abs)
            for (uint32_t neg = 0; neg < 8; ++neg)
                for (uint32_t omod = 0; omod < 4; ++omod)
                    for (bool clamp : {false, true})
                        for (uint32_t sel : {0u, 15u}) {
                            auto run = run_vector_op_per_lane(
                                words(def.opcode, sel, 0, abs, neg, omod, clamp), in, 64, true);
                            ASSERT_TRUE(run.error.empty()) << run.error;
                            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                                auto want = result(def, sources(in, lane, sel, 0, abs, neg),
                                                   in.old_d[lane], sel, omod, clamp);
                                if (!want) {
                                    ADD_FAILURE() << "unknown modifier result";
                                    return;
                                }
                                EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                            }
                        }
}
TEST(IrVectorMinmaxHalfExecution, EffectsAndUnknownDomainsStayConservative) {
    for (const auto& def : vector_minmax_half_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_TRUE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 0x3f800000, 0x40000000}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0, 0x80000000, 0x3f800000}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x7fc01234, 0x7fc56789, 0x3f800000}));
    }
    EXPECT_EQ(find_vector_minmax_half_op(Rdna2Format::VOP3, 0x34b), nullptr);
    EXPECT_EQ(find_vector_minmax_half_op(Rdna2Format::VOP3P, 0x351), nullptr);
}

// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorMinmaxHalfExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "minmax-half-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const fs::path dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    size_t count = 0;
    for (const auto& def : vector_minmax_half_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u})
                for (uint32_t sel = 0; sel < 16; ++sel) {
                    if (mode && sel != 0 && sel != 15) continue;
                    const auto program = lane_program(words(def.opcode, sel, mode), 64, wave, true);
                    const uint64_t address = 0x5275000000ull + count * 0x100;
                    const fs::path stem =
                        dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                    std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                    bin.write(reinterpret_cast<const char*>(program.code.data()),
                              static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                    bin.close();
                    ASSERT_TRUE(bin.good());
                    const fs::path inputs = stem.string() + ".inputs";
                    ASSERT_TRUE(write_refused_compute_inputs(
                        inputs, address, &program.table, program.config,
                        "synthetic-witness (no live refusal)"));
                    const auto read = read_refused_shader_inputs(inputs);
                    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                    const auto replay = replay_refused_shader_inputs(program.code, read);
                    ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                    EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                    ++count;
                }
    EXPECT_EQ(count, 180u);
}

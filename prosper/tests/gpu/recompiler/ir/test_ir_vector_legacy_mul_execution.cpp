// AMD70648 legacy multiply: real lane execution pins the DX9 zero rule and preserved modifiers.
// A wrong product corrupts active outputs; a lost predicate overwrites inactive destinations.
#include "gpu/recompiler/ir/ops/vector_legacy_mul.hpp"
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
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0,          0x80000000, 0x3fc00000, 0xc0100000,
                              0x40000000, 0xbf000000, 0xc0000000, 0x3f800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 8 ? edges[lane]
                               : (in.s0[lane] & 0x807fffffu) |
                                     ((110u + static_cast<uint32_t>(lane) % 25) << 23);
        in.s1[lane] = lane < 8 ? edges[(lane * 3 + 1) % 8]
                               : (in.s1[lane] & 0x807fffffu) |
                                     ((112u + static_cast<uint32_t>(lane) % 21) << 23);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t op, uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 3 ? 242
                       : mode == 4            ? 255
                       : mode == 6            ? 135
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 244 : mode == 5 ? 255 : 256 + kVsrc1;
    std::vector<uint32_t> code = {0xd4000000u | (op << 16) | kVdst | (abs << 8) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (omod << 27) | (neg << 29)};
    if (mode == 4 || mode == 5) code.push_back(mode == 4 ? 0x3fc00000 : 0xc0100000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0, uint32_t abs = 0,
                             uint32_t neg = 0) {
    const uint32_t raw[] = {mode == 1 || mode == 3 ? 0x3f800000u
                            : mode == 4            ? 0x3fc00000u
                            : mode == 6            ? 7u
                                                   : in.s0[lane],
                            mode == 2 || mode == 3 ? 0x40000000u
                            : mode == 5            ? 0xc0100000u
                                                   : in.s1[lane]};
    std::array<Known, 3> out = {std::nullopt, std::nullopt, std::nullopt};
    for (size_t k = 0; k < 2; ++k) {
        uint32_t v = raw[k];
        if (abs & (1u << k)) v &= 0x7fffffffu;
        out[k] = neg & (1u << k) ? evaluate_op(Opcode::F32Sub, 0, 0, v, 0) : Known(v);
    }
    return out;
}
Known result(const TernaryIntegerDef& def, const std::array<Known, 3>& args, uint32_t omod = 0,
             bool clamp = false) {
    auto value = eval_ternary_integer_op(def, args);
    if (!value) return std::nullopt;
    if (omod)
        value = evaluate_op(Opcode::F32Mul, 0, *value,
                            omod == 1   ? 0x40000000
                            : omod == 2 ? 0x40800000
                                        : 0x3f000000,
                            0);
    if (clamp && value) value = evaluate_op(Opcode::F32Clamp01, 0, *value, 0, 0);
    return value;
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    auto run = run_vector_op_per_lane(words(def.opcode, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        auto want = result(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown lane " << lane;
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorLegacyMulExecution, EveryDefinitionMatchesFiniteEdgesAndRandomLanes) {
    for (const auto& def : vector_legacy_mul_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, inputs(seed * 5284), wave);
    }
}
TEST(IrVectorLegacyMulExecution, HarnessHasDifferentOperandsAndInactiveDestinations) {
    auto in = inputs(113);
    auto run = run_vector_op_per_lane(words(0x107), in);
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
TEST(IrVectorLegacyMulExecution, HandWorkedDx9ZerosAndExactFiniteProducts) {
 // AMD70648 op7 makes either sign of zero times any finite value +0. The
 // remaining rows are exact binary products: 1.5*-2.25=-3.375 and 2^-20*2^-20=2^-40.
    const std::array<std::array<uint32_t, 3>, 12> examples = {{{0, 0x40000000, 0},
                                                               {0x80000000, 0xc0000000, 0},
                                                               {0xc0000000, 0x80000000, 0},
                                                               {0x3fc00000, 0xc0100000, 0xc0580000},
                                                               {0xc0200000, 0xc0000000, 0x40a00000},
                                                               {0x3f800000, 0, 0},
                                                               {0x35800000, 0x35800000, 0x2b800000},
                                                               {0x80000001, 0, 0},
                                                               {0xbf000000, 0x40000000, 0xbf800000},
                                                               {0x7f7fffff, 0, 0},
                                                               {0x80000000, 0, 0},
                                                               {0x80000000, 0x80000000, 0}}};
    const auto* def = find_vector_legacy_mul_op(Rdna2Format::VOP3, 0x107);
    ASSERT_NE(def, nullptr);
    for (const auto& e : examples) {
        EXPECT_EQ(eval_ternary_integer_op(*def, {e[0], e[1], std::nullopt}), Known(e[2]));
        auto in = inputs(19);
        std::fill(in.s0.begin(), in.s0.end(), e[0]);
        std::fill(in.s1.begin(), in.s1.end(), e[1]);
        for (uint32_t wave : {64u, 32u}) {
            auto run = run_vector_op_per_lane(words(def->opcode), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? e[2] : in.old_d[lane]);
        }
    }
}
TEST(IrVectorLegacyMulExecution, TheZeroRuleDiffersFromOrdinaryMultiply) {
    auto in = inputs(25);
    std::fill(in.s0.begin(), in.s0.end(), 0x80000000);
    std::fill(in.s1.begin(), in.s1.end(), 0x40000000);
    for (uint32_t wave : {64u, 32u}) {
        const auto legacy = run_vector_op_per_lane(words(0x107), in, wave);
        const auto ordinary = run_vector_op_per_lane(words(0x108), in, wave);
        ASSERT_TRUE(legacy.error.empty()) << legacy.error;
        ASSERT_TRUE(ordinary.error.empty()) << ordinary.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            EXPECT_EQ(legacy.d[lane], in.active[lane] ? 0u : in.old_d[lane]);
            EXPECT_EQ(ordinary.d[lane], in.active[lane] ? 0x80000000u : in.old_d[lane]);
        }
    }
}
TEST(IrVectorLegacyMulExecution, InlineFloatIntegerAndLiteralBindingsKeepBothSourcePositions) {
    const auto* def = find_vector_legacy_mul_op(Rdna2Format::VOP3, 0x107);
    ASSERT_NE(def, nullptr);
    for (uint32_t mode = 1; mode <= 6; ++mode)
        for (uint32_t wave : {64u, 32u}) compare(*def, inputs(mode * 31), wave, mode);
}
TEST(IrVectorLegacyMulExecution, SourceAndOutputModifiersKeepTheirOrder) {
    const auto* def = find_vector_legacy_mul_op(Rdna2Format::VOP3, 0x107);
    ASSERT_NE(def, nullptr);
    auto in = inputs(43);
    std::fill(in.s0.begin(), in.s0.end(), 0xbf000000);
    std::fill(in.s1.begin(), in.s1.end(), 0x40000000);
    for (uint32_t abs = 0; abs < 4; ++abs)
        for (uint32_t neg = 0; neg < 4; ++neg)
            for (uint32_t omod = 0; omod < 4; ++omod)
                for (bool clamp : {false, true})
                    for (uint32_t wave : {64u, 32u}) {
                        auto run = run_vector_op_per_lane(
                            words(def->opcode, 0, abs, neg, omod, clamp), in, wave);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        auto want = result(*def, sources(in, 0, 0, abs, neg), omod, clamp);
                        if (!want) {
                            ADD_FAILURE() << "unknown modifier result";
                            return;
                        }
                        for (size_t lane = 0; lane < in.lanes(); ++lane)
                            EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                    }
}
TEST(IrVectorLegacyMulExecution, ExistingUnknownAndNonfiniteInterpreterLimitsStayExplicit) {
    const auto* def = find_vector_legacy_mul_op(Rdna2Format::VOP3, 0x107);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_ternary_integer_op(*def, {std::nullopt, 0x3f800000, std::nullopt}));
    EXPECT_EQ(find_vector_legacy_mul_op(Rdna2Format::VOP2, 7), nullptr);
    EXPECT_EQ(find_vector_legacy_mul_op(Rdna2Format::VOP3, 0x108), nullptr);
 // Select's evaluator still requires both candidate values known. The eager VM likewise
 // refuses the unobserved NaN product; this migration does not broaden either policy.
    for (auto args :
         {std::array<uint32_t, 2>{0, 0x7f800000}, std::array<uint32_t, 2>{0x7fc01234, 0}}) {
        EXPECT_FALSE(eval_ternary_integer_op(*def, {args[0], args[1], std::nullopt}));
        auto in = inputs(79);
        std::fill(in.s0.begin(), in.s0.end(), args[0]);
        std::fill(in.s1.begin(), in.s1.end(), args[1]);
        auto run = run_vector_op_per_lane(words(def->opcode), in);
        EXPECT_TRUE(run.d.empty());
        EXPECT_EQ(run.error, "unspecified float ALU NaN payload");
    }
}

// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorLegacyMulExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "legacy-mul-witness";
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
    for (const auto& def : vector_legacy_mul_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
                const auto program = lane_program(words(def.opcode, mode), 64, wave, false);
                const uint64_t address = 0x5284000000ull + count * 0x100;
                const fs::path stem =
                    dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                bin.write(reinterpret_cast<const char*>(program.code.data()),
                          static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const fs::path inputs = stem.string() + ".inputs";
                ASSERT_TRUE(write_refused_compute_inputs(inputs, address, &program.table,
                                                         program.config,
                                                         "synthetic-witness (no live refusal)"));
                const auto read = read_refused_shader_inputs(inputs);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(program.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                ++count;
            }
    EXPECT_EQ(count, 14u);
}

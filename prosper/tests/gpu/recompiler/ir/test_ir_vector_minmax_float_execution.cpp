// AMD70648 extrema/median: independent per-lane words, partial EXEC, and explicit NaN fallback.
#include "gpu/recompiler/ir/ops/vector_minmax_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edge[] = {0x7fc01234, 0x3f800000, 0xbf800000, 0x3f000000,
                             0xc0200000, 0x3f800001, 0x40000000, 0x47800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 8 ? edge[lane]
                               : (in.s0[lane] & 0x807fffffu) |
                                     ((110u + static_cast<uint32_t>(lane) % 25) << 23);
        in.s1[lane] = lane < 8 ? edge[(lane * 3 + 1) % 8]
                               : (in.s1[lane] & 0x807fffffu) |
                                     ((112u + static_cast<uint32_t>(lane) % 21) << 23);
        seed = seed * 1664525u + 1013904223u;
        in.s2.push_back(lane < 8 ? edge[(lane * 5 + 3) % 8]
                                 : (seed & 0x807fffffu) |
                                       ((108u + static_cast<uint32_t>(lane) % 27) << 23));
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode, uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 4 ? 242 : mode == 5 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 4 ? 244 : mode == 6 ? 255 : 256 + kVsrc1;
    const uint32_t c = mode == 3 || mode == 4 ? 241 : mode == 7 ? 255 : 256 + kVsrc2;
    std::vector<uint32_t> code = {0xd4000000u | (opcode << 16) | kVdst | (abs << 8) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (c << 18) | (omod << 27) | (neg << 29)};
    if (mode >= 5) code.push_back(mode == 5 ? 0x3fc00000 : mode == 6 ? 0x3f7ffffe : 0xbf800000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0) {
    return {mode == 1 || mode == 4 ? 0x3f800000u
            : mode == 5            ? 0x3fc00000u
                                   : in.s0[lane],
            mode == 2 || mode == 4 ? 0x40000000u
            : mode == 6            ? 0x3f7ffffeu
                                   : in.s1[lane],
            mode == 3 || mode == 4 ? 0xbf000000u
            : mode == 7            ? 0xbf800000u
                                   : in.s2[lane]};
}
void compare(const MinmaxFloatDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(def.opcode, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_minmax_float_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown extrema lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorMinmaxFloatExecution, EveryDefinitionMatchesDifferingLanesInBothWaves) {
    for (const auto& def : vector_minmax_float_defs()) {
        const auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, inputs(seed * 5262), wave);
    }
}
TEST(IrVectorMinmaxFloatExecution, HarnessSeesDistinctOperandsAndInactiveDestinations) {
    auto in = inputs(113);
    const auto run = run_vector_op_per_lane(words(0x157), in);
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
    const auto bad = run_vector_op_per_lane(words(0x157), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorMinmaxFloatExecution, HandWorkedOrderingAndNanFallbackInEverySourcePosition) {
 // 70648: finite ordering of(-2,1,4) yields(-2,4,1); any NaN MED3 returns MIN3.
 // Thus (NaN,1,4) yields(1,4,1), unlike the ordinary median expression which gives4.
    const std::array<std::array<uint32_t, 6>, 4> examples = {
        {{0xc0000000, 0x3f800000, 0x40800000, 0xc0000000, 0x40800000, 0x3f800000},
         {0x7fc01234, 0x3f800000, 0x40800000, 0x3f800000, 0x40800000, 0x3f800000},
         {0xff800000, 0x3f800000, 0x7f800000, 0xff800000, 0x7f800000, 0x3f800000},
         {0x3f800000, 0x3f800000, 0x40000000, 0x3f800000, 0x40000000, 0x3f800000}}};
    const std::array<std::array<size_t, 3>, 6> permutations = {
        {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
    for (const auto& e : examples)
        for (const auto& order : permutations) {
            size_t op = 0;
            for (const auto& def : vector_minmax_float_defs()) {
                const auto want = e[3 + op++];
                const std::array<Known, 3> args = {e[order[0]], e[order[1]], e[order[2]]};
                EXPECT_EQ(eval_minmax_float_op(def, args), Known(want));
                auto in = inputs(19);
                std::fill(in.s0.begin(), in.s0.end(), e[order[0]]);
                std::fill(in.s1.begin(), in.s1.end(), e[order[1]]);
                std::fill(in.s2.begin(), in.s2.end(), e[order[2]]);
                for (uint32_t wave : {64u, 32u}) {
                    const auto run = run_vector_op_per_lane(words(def.opcode), in, wave);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                }
            }
        }
}
TEST(IrVectorMinmaxFloatExecution, InlineFloatAndLiteralEachSourceKeepTheirBits) {
    for (const auto& def : vector_minmax_float_defs())
        for (uint32_t mode = 1; mode <= 7; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, mode);
}
TEST(IrVectorMinmaxFloatExecution, CallerSourceAndResultModifiersStayInOrder) {
    auto in =
        inputs(43);   // finite nonzero inputs avoid unspecified signed-zero pairs and NaN payloads.
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0xbf000000;
        in.s1[lane] = 0x3f800000;
        in.s2[lane] = 0x40000000;
    }
    for (const auto& def : vector_minmax_float_defs())
        for (uint32_t abs = 0; abs < 8; ++abs)
            for (uint32_t neg = 0; neg < 8; ++neg)
                for (uint32_t omod = 0; omod < 4; ++omod)
                    for (bool clamp : {false, true}) {
                        auto run =
                            run_vector_op_per_lane(words(def.opcode, 0, abs, neg, omod, clamp), in);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        auto args = sources(in, 0);
                        for (size_t k = 0; k < 3; ++k) {
                            if (abs & (1u << k)) *args[k] &= 0x7fffffffu;
                            if (neg & (1u << k)) *args[k] ^= 0x80000000u;
                        }
                        auto want = eval_minmax_float_op(def, args);
                        ASSERT_TRUE(want);
                        if (omod)
                            want = evaluate_op(Opcode::F32Mul, 0, *want,
                                               omod == 1   ? 0x40000000u
                                               : omod == 2 ? 0x40800000u
                                                           : 0x3f000000u,
                                               0);
                        if (clamp) want = evaluate_op(Opcode::F32Clamp01, 0, *want, 0, 0);
                        ASSERT_TRUE(want);
                        for (size_t lane = 0; lane < in.lanes(); ++lane)
                            EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                    }
}
TEST(IrVectorMinmaxFloatExecution, EffectsUnknownDomainAndEveryValueOpcodeAreCovered) {
    std::set<Opcode> used;
    for (const auto& def : vector_minmax_float_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block b;
        Builder ir(b);
        std::vector<uint32_t> reads;
        def.build(ir, TernaryInputs(ir, reads));
        for (const auto& inst : b.insts()) used.insert(inst.op);
        EXPECT_FALSE(eval_minmax_float_op(def, {std::nullopt, 0x3f800000, 0x40000000}));
        EXPECT_FALSE(eval_minmax_float_op(def, {0, 0x80000000, 0x3f800000}));
        EXPECT_FALSE(eval_minmax_float_op(def, {0x7fc01234, 0x7fc56789, 0x3f800000}));
    }
    EXPECT_EQ(used, (std::set<Opcode>{Opcode::Input, Opcode::F32NMin, Opcode::F32NMax,
                                      Opcode::BoolOr, Opcode::F32UnordNe, Opcode::Select}));
    EXPECT_EQ(find_vector_minmax_float_op(Rdna2Format::VOP3, 0x152), nullptr);
    EXPECT_EQ(find_vector_minmax_float_op(Rdna2Format::VOP3P, 0x151), nullptr);
}
// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorMinmaxFloatExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "minmax-float-witness";
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
    for (const auto& def : vector_minmax_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
                const auto program = lane_program(words(def.opcode, mode), 64, wave, false);
                const uint64_t address = 0x5262000000ull + count * 0x100;
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
    EXPECT_EQ(count, 48u);
}

// AMD70648 RTZ packing: emitted per-lane conversion preserves source class and both output halves.
#include "gpu/recompiler/ir/ops/vector_rtz_pack.hpp"
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
std::vector<uint32_t> words(uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 3 ? 242
                       : mode == 4            ? 255
                       : mode == 6            ? 193
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 244 : mode == 5 ? 255 : 256 + kVsrc1;
    std::vector<uint32_t> code = {0xd4000000u | (0x12fu << 16) | kVdst | (abs << 8) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (omod << 27) | (neg << 29)};
    if (mode == 4 || mode == 5) code.push_back(mode == 4 ? 0x3f803000 : 0xbf803000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0, uint32_t abs = 0,
                             uint32_t neg = 0) {
    const uint32_t raw[] = {mode == 1 || mode == 3 ? 0x3f800000u
                            : mode == 4            ? 0x3f803000u
                            : mode == 6            ? 0xffffffffu
                                                   : in.s0[lane],
                            mode == 2 || mode == 3 ? 0x40000000u
                            : mode == 5            ? 0xbf803000u
                                                   : in.s1[lane]};
    std::array<Known, 3> out = {std::nullopt, std::nullopt, std::nullopt};
    for (size_t k = 0; k < 2; ++k) {
        uint32_t value = raw[k];
        if (abs & (1u << k)) value &= 0x7fffffffu;
        out[k] = neg & (1u << k) ? evaluate_op(Opcode::F32Sub, 0, 0, value, 0) : Known(value);
    }
    return out;
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    auto run = run_vector_op_per_lane(words(mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown RTZ lane " << lane;
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorRtzPackExecution, EveryRawWordClassInBothWaveSizes) {
    for (const auto& def : vector_rtz_pack_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        auto code = words();
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, lane_inputs(seed * 5282), wave);
    }
}
TEST(IrVectorRtzPackExecution, HarnessDistinguishesBothPackedHalvesAndInactiveWords) {
    auto in = lane_inputs(113);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0x3f800000u + static_cast<uint32_t>(lane) * 0x2000u;
        in.s1[lane] = 0xc0000000u + static_cast<uint32_t>(lane) * 0x2000u;
    }
    auto run = run_vector_op_per_lane(words(), in);
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
TEST(IrVectorRtzPackExecution, HandWorkedRtzResultsPinTiesSubnormalsOverflowAndInfinity) {
 // IEEE binary16: 1.5=3e00, -2.25=c080. At 1+3*2^-11 RTZ keeps fraction1,
    // where nearest-even would choose2. Finite overflow truncates to +/-65504, unlike true infinity.
    const std::array<std::array<uint32_t, 3>, 14> examples = {
        {{0, 0x80000000, 0x80000000},
         {0x3fc00000, 0xc0100000, 0xc0803e00},
         {0x3f803000, 0xbf803000, 0xbc013c01},
         {0x33800000, 0xb3800000, 0x80010001},
         {0x337fffff, 0xb37fffff, 0x80000000},
         {0x38800000, 0xb8800000, 0x84000400},
         {0x387fc000, 0xb87fc000, 0x83ff03ff},
         {0x47800000, 0xc7800000, 0xfbff7bff},
         {0x7f7fffff, 0xff7fffff, 0xfbff7bff},
         {0x7f800000, 0xff800000, 0xfc007c00},
         {0x477fffff, 0xc77fffff, 0xfbff7bff},
         {0x3f801000, 0xbf801000, 0xbc003c00},
         {1, 0x80000001, 0x80000000},
         {0x3f7fffff, 0xbf7fffff, 0xbbff3bff}}};
    const auto* def = find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x12f);
    ASSERT_NE(def, nullptr);
    for (const auto& e : examples) {
        EXPECT_EQ(eval_ternary_integer_op(*def, {e[0], e[1], std::nullopt}), Known(e[2]));
        auto in = lane_inputs(19);
        std::fill(in.s0.begin(), in.s0.end(), e[0]);
        std::fill(in.s1.begin(), in.s1.end(), e[1]);
        for (uint32_t wave : {64u, 32u}) {
            auto run = run_vector_op_per_lane(words(), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? e[2] : in.old_d[lane]);
        }
    }
}
TEST(IrVectorRtzPackExecution, EveryExponentWithDifferingSignsAndMantissas) {
    const auto* def = find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x12f);
    ASSERT_NE(def, nullptr);
    for (uint32_t exponent = 0; exponent < 256; ++exponent) {
        auto in = lane_inputs(exponent + 1);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            uint32_t index = static_cast<uint32_t>(lane);
            in.s0[lane] =
                ((index & 1u) << 31) | (exponent << 23) | ((index * 0x1ffffu) & 0x7fffffu);
            in.s1[lane] = ((index & 2u) << 30) | ((255u - exponent) << 23) |
                          ((index * 0x02001u + 0x1000u) & 0x7fffffu);
        }
        for (uint32_t wave : {64u, 32u}) compare(*def, in, wave);
    }
}
TEST(IrVectorRtzPackExecution, InlineFloatIntegerAndLiteralBindingsKeepBothSourcePositions) {
    const auto* def = find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x12f);
    ASSERT_NE(def, nullptr);
    for (uint32_t mode = 1; mode <= 6; ++mode)
        for (uint32_t wave : {64u, 32u}) compare(*def, lane_inputs(mode * 31), wave, mode);
}
TEST(IrVectorRtzPackExecution, RootSourceModifiersApplyAndOutputOmodRemainsIgnored) {
    const auto* def = find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x12f);
    ASSERT_NE(def, nullptr);
    auto in = lane_inputs(43);
    std::fill(in.s0.begin(), in.s0.end(), 0x3fc00000);
    std::fill(in.s1.begin(), in.s1.end(), 0xc0100000);
    for (uint32_t abs = 0; abs < 4; ++abs)
        for (uint32_t neg = 0; neg < 4; ++neg)
            for (uint32_t omod = 0; omod < 4; ++omod)
                for (uint32_t wave : {64u, 32u}) {
                    auto run = run_vector_op_per_lane(words(0, abs, neg, omod), in, wave);
                    ASSERT_TRUE(run.error.empty()) << run.error;
                    auto want = eval_ternary_integer_op(*def, sources(in, 0, 0, abs, neg));
                    if (!want) {
                        ADD_FAILURE() << "unknown modifier result";
                        return;
                    }
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                }
}
TEST(IrVectorRtzPackExecution, CallerClampRefusalHasAnOtherwiseIdenticalAcceptedControl) {
    auto in = lane_inputs(91);
    const auto control = run_vector_op_per_lane(words(), in);
    ASSERT_TRUE(control.error.empty()) << control.error;
    ASSERT_EQ(control.d.size(), in.lanes());
    const auto rejected = run_vector_op_per_lane(words(0, 0, 0, 0, true), in);
    EXPECT_TRUE(rejected.d.empty());
    EXPECT_TRUE(rejected.error.starts_with("refused:")) << rejected.error;
}
TEST(IrVectorRtzPackExecution, UnknownInputsAndOtherEncodingsDoNotBecomeKnown) {
    const auto* def = find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x12f);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_ternary_integer_op(*def, {std::nullopt, 0, std::nullopt}));
    EXPECT_FALSE(eval_ternary_integer_op(*def, {0, std::nullopt, std::nullopt}));
    EXPECT_TRUE(eval_ternary_integer_op(*def, {0x3f800000, 0x40000000, std::nullopt}));
    EXPECT_EQ(find_vector_rtz_pack_op(Rdna2Format::VOP2, 0x2f), nullptr);
    EXPECT_EQ(find_vector_rtz_pack_op(Rdna2Format::VOP3P, 0x12f), nullptr);
    EXPECT_EQ(find_vector_rtz_pack_op(Rdna2Format::VOP3, 0x311), nullptr);
}

// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorRtzPackExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "rtz-pack-witness";
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
    for (const auto& def : vector_rtz_pack_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
                EXPECT_EQ(def.opcode, 0x12fu);
                const auto program = lane_program(words(mode), 64, wave, false);
                const uint64_t address = 0x5282000000ull + count * 0x100;
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

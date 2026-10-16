// AMD70648 normalized low/high conversion: independent RNE examples, per-lane words and EXEC.
#include "gpu/recompiler/ir/ops/vector_pack_norm.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
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
    const std::array<uint32_t, 16> edge = {0,          0x80000000, 0x3f800000, 0xbf800000,
                                           0x40000000, 0xc0000000, 0x3f000000, 0xbf000000,
                                           0x3effffff, 0x3f000001, 0xbeffffff, 0xbf000001,
                                           1,          0x80000001, 0x7f800000, 0xff800000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 16 ? edge[lane]
                                : (in.s0[lane] & 0x807fffffu) |
                                      ((112u + static_cast<uint32_t>(lane) % 18) << 23);
        in.s1[lane] = lane < 16 ? edge[(lane * 5 + seed) % 16]
                                : (in.s1[lane] & 0x807fffffu) |
                                      ((115u + static_cast<uint32_t>(lane) % 15) << 23);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode, uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 3 ? 240 : mode == 4 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 241 : mode == 5 ? 255 : 256 + kVsrc1;
    std::vector<uint32_t> code = {0xd4000000u | (opcode << 16) | kVdst | (abs << 8) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (neg << 29) | (omod << 27)};
    if (mode >= 4) code.push_back(mode == 4 ? 0x3f400000 : 0xbf800000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0) {
    return {mode == 1 || mode == 3 ? 0x3f000000u
            : mode == 4            ? 0x3f400000u
                                   : in.s0[lane],
            mode == 2 || mode == 3 ? 0xbf000000u
            : mode == 5            ? 0xbf800000u
                                   : in.s1[lane],
            std::nullopt};
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(def.opcode, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown normalized-pack lane";
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "opcode " << def.opcode << " lane " << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorPackNormExecution, BothDefinitionsMatchEdgeAndRandomLanesInBothWaves) {
    for (const auto& def : vector_pack_norm_defs()) {
        auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed) compare(def, inputs(seed * 5267), wave);
    }
}
TEST(IrVectorPackNormExecution, HarnessDoesNotHideHalfOrderOrInactiveDestinations) {
    auto in = inputs(37);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = (in.s0[lane] & 0x807fffffu) | (126u << 23);
        in.s1[lane] = (in.s1[lane] & 0x807fffffu) | (125u << 23);
    }
    for (const auto& def : vector_pack_norm_defs()) {
        const auto run = run_vector_op_per_lane(words(def.opcode), in);
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
    in.s1.pop_back();
    const auto bad = run_vector_op_per_lane(words(0x368), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_FALSE(bad.error.empty());
}
TEST(IrVectorPackNormExecution, HandWorkedNormalizedRneAndSaturationExamples) {
    struct X {
        uint32_t a, b, signed_want, unsigned_want;
    };
 // SNORM scales by32767, UNORM by65535; convert each half independently, nearest-even.
 // .5 produces4000/8000; -.5 producesc000/0. Signed-1 maps to8001, not8000.
    const X examples[] = {{0x3f000000, 0xbf000000, 0xc0004000, 0x00008000},
                          {0x3f800000, 0xbf800000, 0x80017fff, 0x0000ffff},
                          {0x40000000, 0xc0000000, 0x80017fff, 0x0000ffff},
                          {0, 0x3f800000, 0x7fff0000, 0xffff0000},
                          {0x80000000, 0x3f000000, 0x40000000, 0x80000000},
                          {0x3e800000, 0x3f400000, 0x5fff2000, 0xbfff4000},
                          {0xbe800000, 0xbf400000, 0xa001e000, 0},
                          {1, 0x80000001, 0, 0},
                          {0x7f800000, 0xff800000, 0x80017fff, 0x0000ffff}};
    for (const auto& def : vector_pack_norm_defs())
        for (auto x : examples) {
            const auto want = def.opcode == 0x368 ? x.signed_want : x.unsigned_want;
            EXPECT_EQ(eval_ternary_integer_op(def, {x.a, x.b, std::nullopt}), Known(want));
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), x.a);
            std::fill(in.s1.begin(), in.s1.end(), x.b);
            for (uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
}
TEST(IrVectorPackNormExecution, InlineFloatsAndBothLiteralSourcesKeepTheirWordValues) {
    for (const auto& def : vector_pack_norm_defs())
        for (uint32_t mode = 1; mode <= 5; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(mode + 31), wave, mode);
}
TEST(IrVectorPackNormExecution, FloatSourceModifiersAndIgnoredIntegerOmodKeepTheirOrder) {
    for (const auto& def : vector_pack_norm_defs())
        for (uint32_t abs = 0; abs < 4; ++abs)
            for (uint32_t neg = 0; neg < 4; ++neg)
                for (uint32_t omod = 0; omod < 4; ++omod) {
                    const auto in = inputs(43);
                    for (uint32_t wave : {64u, 32u}) {
                        const auto run =
                            run_vector_op_per_lane(words(def.opcode, 0, abs, neg, omod), in, wave);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        for (size_t lane = 0; lane < in.lanes(); ++lane) {
                            uint32_t a = in.s0[lane], b = in.s1[lane];
                            if (abs & 1) a &= 0x7fffffffu;
                            if (abs & 2) b &= 0x7fffffffu;
                            if (neg & 1) a ^= 0x80000000u;
                            if (neg & 2) b ^= 0x80000000u;
                            const auto want = eval_ternary_integer_op(def, {a, b, std::nullopt});
                            if (!want) {
                                ADD_FAILURE() << "unknown modified pack";
                                return;
                            }
                            EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                        }
                    }
                }
}
TEST(IrVectorPackNormExecution, CallerClampRefusesAndUnusedThirdSourceModifiersStayIgnored) {
    for (const auto& def : vector_pack_norm_defs()) {
        const auto in = inputs(47);
        const auto bad = run_vector_op_per_lane(words(def.opcode, 0, 0, 0, 0, true), in);
        EXPECT_FALSE(bad.error.empty());
        EXPECT_TRUE(bad.d.empty());
        const auto base = run_vector_op_per_lane(words(def.opcode), in),
                   unused = run_vector_op_per_lane(words(def.opcode, 0, 4, 4), in);
        ASSERT_TRUE(base.error.empty()) << base.error;
        ASSERT_TRUE(unused.error.empty()) << unused.error;
        EXPECT_EQ(unused.d, base.d);
    }
}
TEST(IrVectorPackNormExecution, EffectsUnknownDomainAndEachPrimitiveModeAreCovered) {
    std::set<Opcode> used;
    std::set<uint32_t> modes;
    for (const auto& def : vector_pack_norm_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        Block b;
        Builder ir(b);
        std::vector<uint32_t> reads;
        def.build(ir, TernaryInputs(ir, reads));
        for (const auto& inst : b.insts()) {
            used.insert(inst.op);
            if (inst.op == Opcode::PackNorm16) modes.insert(inst.imm);
        }
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 0, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x7fc01234, 0, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0, 0x7fc01234, std::nullopt}));
    }
    EXPECT_EQ(used, (std::set<Opcode>{Opcode::Input, Opcode::ConstU32, Opcode::PackNorm16,
                                      Opcode::Shl, Opcode::Or}));
    EXPECT_EQ(modes, (std::set<uint32_t>{0, 1}));
    EXPECT_EQ(find_vector_pack_norm_op(Rdna2Format::VOP3, 0x36a), nullptr);
    EXPECT_EQ(find_vector_pack_norm_op(Rdna2Format::VOP3P, 0x368), nullptr);
}
// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorPackNormExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "pack-norm-witness";
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
    for (const auto& def : vector_pack_norm_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u}) {
                const auto program = lane_program(words(def.opcode, mode), 64, wave, false);
                const uint64_t address = 0x5267000000ull + count * 0x100;
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
    EXPECT_EQ(count, 24u);
}

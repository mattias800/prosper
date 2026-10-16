// AMD70648 12.12 u32_to_u16 packing: saturation boundaries, independent per-lane
// operands and partial EXEC. AMD's earlier ISA explicitly clamps overflowing inputs to 0xffff.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/ir/ops/vector_pack_saturate.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
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
    for (size_t lane = 24; lane < in.lanes(); ++lane) {
        if (lane & 1) in.s0[lane] &= 0xffffu;
        if (lane & 2) in.s1[lane] &= 0xffffu;
    }
    return in;
}
std::vector<uint32_t> words(uint32_t mode = 0) {
    const uint32_t a = mode == 1 || mode == 3 ? 135
                       : mode == 4            ? 193
                       : mode == 5            ? 255
                       : mode == 7            ? 242
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 137
                       : mode == 4            ? 194
                       : mode == 6            ? 255
                       : mode == 8            ? 244
                                              : 256 + kVsrc1;
    std::vector<uint32_t> out = {0xd76a0000u | kVdst, a | (b << 9)};
    if (mode == 5 || mode == 6) out.push_back(mode == 5 ? 0x00010000u : 0xffff0000u);
    return out;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0) {
    const uint32_t a = mode == 1 || mode == 3 ? 7
                       : mode == 4            ? 0xffffffffu
                       : mode == 5            ? 0x00010000u
                       : mode == 7            ? 0x3f800000u
                                              : in.s0[lane];
    const uint32_t b = mode == 2 || mode == 3 ? 9
                       : mode == 4            ? 0xfffffffeu
                       : mode == 6            ? 0xffff0000u
                       : mode == 8            ? 0x40000000u
                                              : in.s1[lane];
    return {a, b, std::nullopt};
}
void compare(const TernaryIntegerDef& def, uint32_t seed, uint32_t wave, uint32_t mode = 0) {
    const auto in = inputs(seed);
    const auto run = run_vector_op_per_lane(words(mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown pack lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorPackSaturateExecution, EveryDefinitionMatchesMixedNarrowAndWideLanesInBothWaves) {
    for (const auto& def : vector_pack_saturate_defs()) {
        const auto code = words();
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, seed * 5259, wave);
    }
}
TEST(IrVectorPackSaturateExecution, HarnessDoesNotHideOperandWiringBehindSaturationOrExec) {
    const auto in = inputs(113);
    const auto run = run_vector_op_per_lane(words(), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
}
TEST(IrVectorPackSaturateExecution, HandWorkedIsaUnsignedOverflowAndHalfOrderExamples) {
    struct Example {
        uint32_t a, b, want;
    };
    for (auto x : {Example{0, 0, 0}, Example{1, 2, 0x00020001}, Example{0xffff, 0xffff, 0xffffffff},
                   Example{0x10000, 1, 0x0001ffff}, Example{1, 0x10000, 0xffff0001},
                   Example{0xffffffff, 7, 0x0007ffff}, Example{0x80000000, 0xffff, 0xffffffff},
                   Example{0x1234, 0xabcd, 0xabcd1234}, Example{0xffff0000, 0x12345, 0xffffffff},
                   Example{0xfffe, 0xfffd, 0xfffdfffe}, Example{0x7fffffff, 0, 0x0000ffff},
                   Example{0, 0x80000000, 0xffff0000}}) {
        const auto* def = find_vector_pack_saturate_op(Rdna2Format::VOP3, 0x36a);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, std::nullopt}), Known(x.want));
        auto in = inputs(31);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(words(), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
        }
    }
}
TEST(IrVectorPackSaturateExecution, EveryBoundaryPairInBothWaves) {
    const uint32_t edges[] = {0, 0xfffe, 0xffff, 0x10000, 0x10001, 0xffffffff};
    const auto* def = find_vector_pack_saturate_op(Rdna2Format::VOP3, 0x36a);
    ASSERT_NE(def, nullptr);
    for (uint32_t wave : {64u, 32u}) {
        auto in = inputs(73);
        std::fill(in.active.begin(), in.active.end(), true);
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.s0[lane] = edges[lane % 6];
            in.s1[lane] = edges[(lane / 6) % 6];
        }
        const auto run = run_vector_op_per_lane(words(), in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            const uint32_t want =
                std::min(in.s0[lane], 0xffffu) | (std::min(in.s1[lane], 0xffffu) << 16);
            EXPECT_EQ(run.d[lane], want);
            EXPECT_EQ(eval_ternary_integer_op(*def, {in.s0[lane], in.s1[lane], std::nullopt}),
                      Known(want));
        }
    }
}
TEST(IrVectorPackSaturateExecution, InlineNegativeFloatBusAndLiteralInputsKeepTheirUnsignedWords) {
    for (const auto& def : vector_pack_saturate_defs())
        for (uint32_t mode = 1; mode <= 8; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, 103 + mode, wave, mode);
}
TEST(IrVectorPackSaturateExecution, TheExistingCallerClampRefusalRemains) {
    const auto in = inputs(127);
    auto code = words();
    code[0] |= 1u << 15;
    const auto bad = run_vector_op_per_lane(code, in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
}
TEST(IrVectorPackSaturateExecution, EffectsTwoSourceDemandAndAllValueOpcodesAreCovered) {
    std::set<Opcode> seen;
    for (const auto& def : vector_pack_saturate_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> slots;
        def.build(ir, TernaryInputs(ir, slots));
        EXPECT_EQ((std::set<uint32_t>(slots.begin(), slots.end())), (std::set<uint32_t>{0, 1}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 9u, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {7u, std::nullopt, std::nullopt}));
        EXPECT_EQ(eval_ternary_integer_op(def, {7u, 9u, std::nullopt}), Known(0x00090007));
    }
    for (auto op : {Opcode::UMin, Opcode::Shl, Opcode::Or}) EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_pack_saturate_op(Rdna2Format::VOP3, 0x36b), nullptr);
    EXPECT_EQ(find_vector_pack_saturate_op(Rdna2Format::VOP3P, 0x36a), nullptr);
}
// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorPackSaturateExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "pack-saturate-witness";
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
    for (const auto& def : vector_pack_saturate_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
                const auto program = lane_program(words(mode), 64, wave, false);
                const uint64_t address = 0x5259000000ull + count * 0x100;
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
    EXPECT_EQ(count, 18u);
}

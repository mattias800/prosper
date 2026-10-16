// AMD70648 12.12 numeric data bit utilities: real emitted SPIR-V vs eval per lane,
// independent field/count examples, full wrapped count domains and partial EXEC in both waves.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/ir/ops/vector_bit_utilities.hpp"
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
std::vector<uint32_t> words(const TernaryIntegerDef& def, uint32_t mode = 0) {
    const uint32_t a = mode == 1 || mode == 3 ? 135
                       : mode == 4            ? 255
                       : mode == 6            ? 193
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 137
                       : mode == 5            ? 255
                       : mode == 6            ? 194
                                              : 256 + kVsrc1;
    std::vector<uint32_t> out = {0xd4000000u | (def.opcode << 16) | kVdst, a | (b << 9)};
    if (mode == 4 || mode == 5) out.push_back(0xffff001fu);
    return out;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode) {
    const uint32_t a = mode == 1 || mode == 3 ? 7
                       : mode == 4            ? 0xffff001fu
                       : mode == 6            ? 0xffffffffu
                                              : in.s0[lane];
    const uint32_t b = mode == 2 || mode == 3 ? 9
                       : mode == 5            ? 0xffff001fu
                       : mode == 6            ? 0xfffffffeu
                                              : in.s1[lane];
    return {a, b, std::nullopt};
}
void check(const TernaryIntegerDef& def, uint32_t mode, uint32_t wave, uint32_t seed) {
    const auto in = lane_inputs(seed);
    const auto run = run_vector_op_per_lane(words(def, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown numeric result at lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " op " << def.opcode << " mode " << mode << " wave " << wave;
    }
}
}   // namespace
TEST(IrVectorBitUtilitiesExecution, EveryDefinitionMatchesDifferingLanesInBothWaves) {
    for (const auto& def : vector_bit_utilities_defs()) {
        const auto code = words(def);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) check(def, 0, wave, seed * 5255);
    }
}
TEST(IrVectorBitUtilitiesExecution, InlineNegativeAndLiteralOperandsKeepTheirFullWordValues) {
    for (const auto& def : vector_bit_utilities_defs())
        for (uint32_t mode = 1; mode <= 6; ++mode)
            for (uint32_t wave : {64u, 32u}) check(def, mode, wave, 17 + mode);
}
TEST(IrVectorBitUtilitiesExecution, HandWorkedIsaWrappedFieldsAndAccumulatingCounts) {
    struct Example {
        uint32_t op, a, b, want;
    };
    for (auto x : {Example{0x363, 0, 0, 0}, Example{0x363, 1, 31, 0x80000000},
                   Example{0x363, 31, 1, 0xfffffffe}, Example{0x363, 32, 7, 0},
                   Example{0x363, 33, 33, 2}, Example{0x363, 63, 63, 0x80000000},
                   Example{0x363, 16, 16, 0xffff0000}, Example{0x363, 31, 0, 0x7fffffff},
                   Example{0x363, 0xffffffff, 0xffffffff, 0x80000000}, Example{0x364, 0, 9, 9},
                   Example{0x364, 0xffffffff, 0, 32}, Example{0x364, 0xffffffff, 0xffffffff, 31},
                   Example{0x364, 0x80000000, 0xffffffff, 0}, Example{0x364, 0xaaaaaaaa, 5, 21},
                   Example{0x364, 0x12345678, 7, 20}, Example{0x364, 0xffff0000, 0xfffffff0, 0}}) {
        const auto* def = find_vector_bit_utilities_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, std::nullopt}), Known(x.want));
        auto in = lane_inputs(43);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(words(*def), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
        }
    }
}
TEST(IrVectorBitUtilitiesExecution, AllWidthOffsetPairsWithWrappedHighBitsAndBothWaves) {
    const auto* def = find_vector_bit_utilities_op(Rdna2Format::VOP3, 0x363);
    ASSERT_NE(def, nullptr);
    for (uint32_t wave : {64u, 32u})
        for (uint32_t high : {0u, 0xffffffe0u})
            for (uint32_t batch = 0; batch < 16; ++batch) {
                auto in = lane_inputs(71 + batch);
                std::fill(in.active.begin(), in.active.end(), true);
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    in.s0[lane] = (batch * 2 + static_cast<uint32_t>(lane) / 32) | high;
                    in.s1[lane] = (static_cast<uint32_t>(lane) % 32) | high;
                }
                const auto run = run_vector_op_per_lane(words(*def), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const uint32_t width = in.s0[lane] & 31, offset = in.s1[lane] & 31;
                    const uint32_t want = ((1u << width) - 1u) << offset;
                    EXPECT_EQ(run.d[lane], want);
                    EXPECT_EQ(
                        eval_ternary_integer_op(*def, {in.s0[lane], in.s1[lane], std::nullopt}),
                        Known(want));
                }
            }
}
TEST(IrVectorBitUtilitiesExecution,
     EveryBitPositionCountsWithoutMixingLanesOrMaskingTheAccumulator) {
    const auto* def = find_vector_bit_utilities_op(Rdna2Format::VOP3, 0x364);
    ASSERT_NE(def, nullptr);
    for (uint32_t wave : {64u, 32u})
        for (bool complement : {false, true}) {
            auto in = lane_inputs(83);
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const uint32_t bit = 1u << (static_cast<uint32_t>(lane) & 31);
                in.s0[lane] = complement ? ~bit : bit;
                in.s1[lane] = 0x80000000u | static_cast<uint32_t>(lane);
            }
            const auto run = run_vector_op_per_lane(words(*def), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const uint32_t want = in.s1[lane] + (complement ? 31u : 1u);
                EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
                EXPECT_EQ(eval_ternary_integer_op(*def, {in.s0[lane], in.s1[lane], std::nullopt}),
                          Known(want));
            }
        }
}
TEST(IrVectorBitUtilitiesExecution, ExistingBfmModifierRefusalsAndCallerClampBoundaryRemain) {
    const auto* def = find_vector_bit_utilities_op(Rdna2Format::VOP3, 0x363);
    ASSERT_NE(def, nullptr);
    const auto in = lane_inputs(107);
    for (uint32_t flag = 0; flag < 14; ++flag) {
        auto code = words(*def);
        if (flag < 3)
            code[0] |= 1u << (8 + flag);
        else if (flag < 6)
            code[1] |= 1u << (26 + flag);
        else if (flag == 6)
            code[0] |= 1u << 15;
        else if (flag < 10)
            code[1] |= (flag - 6) << 27;
        else
            code[0] |= 1u << (11 + flag - 10);
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
    auto code = words(*find_vector_bit_utilities_op(Rdna2Format::VOP3, 0x364));
    code[0] |= 1u << 15;
    const auto bad = run_vector_op_per_lane(code, in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_TRUE(bad.error.starts_with("refused:"));
}
TEST(IrVectorBitUtilitiesExecution, EffectsDemandOnlyTwoNumericSourcesAndAllNewIrIsCovered) {
    std::set<Opcode> seen;
    for (const auto& def : vector_bit_utilities_defs()) {
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
        EXPECT_EQ(eval_ternary_integer_op(def, {7u, 9u, std::nullopt}),
                  Known(def.opcode == 0x363 ? 0xfe00 : 12));
    }
    for (auto op : {Opcode::And, Opcode::Shl, Opcode::ISub, Opcode::BitCount, Opcode::IAdd})
        EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_bit_utilities_op(Rdna2Format::VOP3, 0x365), nullptr);
    EXPECT_EQ(find_vector_bit_utilities_op(Rdna2Format::VOP2, 0x364), nullptr);
}
// Export every existing plain operand form using the execution harness and its exact bindings.
// Both waves and each inline/literal pattern stay in a separate synthetic corpus.
TEST(IrVectorBitUtilitiesExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "bit-utilities-witness";
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
    for (const auto& def : vector_bit_utilities_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
                const auto program = lane_program(words(def, mode), 64, wave, false);
                const uint64_t address = 0x5255000000ull + count * 0x100;
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
    EXPECT_EQ(count, 28u);
}

// AMD70648 12.12 raw half packing: emitted SPIR-V vs eval and independent ISA bit examples.
// The per-lane harness preserves complete oldD for inactive lanes and varies both source words.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/ir/ops/vector_pack16.hpp"
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
std::vector<uint32_t> words(const TernaryIntegerDef& def, uint32_t form, uint32_t mode = 0) {
    const uint32_t a = mode == 1 || mode == 5 ? 135
                       : mode == 2            ? 193
                       : mode == 3            ? 242
                       : mode == 4            ? 255
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 1 || mode == 6 ? 137
                       : mode == 2            ? 194
                       : mode == 3            ? 245
                                              : 256 + kVsrc1;
    std::vector<uint32_t> out = {0xd4000000u | (def.opcode << 16) | (form << 11) | kVdst,
                                 a | (b << 9)};
    if (mode == 4) out.push_back(0xcafe7e01u);
    return out;
}
std::array<Known, 3> selected(const LaneInputs& in, size_t lane, uint32_t form, uint32_t mode) {
    const uint32_t a = mode == 1 || mode == 5 ? 7
                       : mode == 2            ? 0xffffffffu
                       : mode == 3            ? 0x3c00
                       : mode == 4            ? 0xcafe7e01u
                                              : in.s0[lane];
    const uint32_t b = mode == 1 || mode == 6 ? 9
                       : mode == 2            ? 0xfffffffeu
                       : mode == 3            ? 0xc000
                                              : in.s1[lane];
    return {(a >> (16 * (form & 1))) & 0xffffu, (b >> (16 * ((form >> 1) & 1))) & 0xffffu,
            std::nullopt};
}
void check(const TernaryIntegerDef& def, uint32_t form, uint32_t mode, uint32_t wave,
           uint32_t seed) {
    const auto in = lane_inputs(seed);
    const auto run = run_vector_op_per_lane(words(def, form, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, selected(in, lane, form, mode));
        if (!want) {
            ADD_FAILURE() << "unknown pack lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " form " << form << " mode " << mode << " wave " << wave;
    }
}
}   // namespace
TEST(IrVectorPack16Execution, EverySourceSelectionAndUnusedSelectBitMatchesBothWaves) {
    for (const auto& def : vector_pack16_defs())
        for (uint32_t form = 0; form < 16; ++form) {
            const auto code = words(def, form);
            std::vector<Rdna2Inst> decoded;
            ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
            ASSERT_EQ(decoded.size(), 1u);
            EXPECT_EQ(decoded[0].opcode, def.opcode);
            EXPECT_EQ(decoded[0].vop3p_opsel, form);
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 8; ++seed) check(def, form, 0, wave, seed * 5253);
        }
}
TEST(IrVectorPack16Execution, InlineHalfBitsLiteralsAndMixedOperandsKeepEveryBit) {
    for (const auto& def : vector_pack16_defs())
        for (uint32_t form = 0; form < 16; ++form)
            for (uint32_t mode = 1; mode <= 6; ++mode)
                for (uint32_t wave : {64u, 32u}) check(def, form, mode, wave, 13 + mode);
}
TEST(IrVectorPack16Execution, HandWorkedIsaHalfOrderZerosSignsAndNanPayloadBits) {
    struct Example {
        uint32_t a, b, form, want;
    };
    // These are raw concatenations, not half conversions: NaN payloads and signed zero survive.
    for (auto x :
         {Example{0x11223344, 0x55667788, 0, 0x77883344},
          Example{0x11223344, 0x55667788, 1, 0x77881122},
          Example{0x11223344, 0x55667788, 2, 0x55663344},
          Example{0x11223344, 0x55667788, 3, 0x55661122},
          Example{0x80000000, 0x00008000, 0, 0x80000000},
          Example{0x80000000, 0x00008000, 1, 0x80008000}, Example{0x80000000, 0x00008000, 2, 0},
          Example{0x80000000, 0x00008000, 3, 0x00008000},
          Example{0x7e013c00, 0xfc008000, 0, 0x80003c00},
          Example{0x7e013c00, 0xfc008000, 1, 0x80007e01},
          Example{0x7e013c00, 0xfc008000, 2, 0xfc003c00},
          Example{0x7e013c00, 0xfc008000, 3, 0xfc007e01}, Example{0xffffffff, 0, 0, 0x0000ffff},
          Example{0, 0xffffffff, 0, 0xffff0000}}) {
        const auto* def = find_vector_pack16_op(Rdna2Format::VOP3, 0x311);
        ASSERT_NE(def, nullptr);
        auto in = lane_inputs(29);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        EXPECT_EQ(eval_ternary_integer_op(*def, selected(in, 0, x.form, 0)), Known(x.want));
        for (uint32_t unused : {0u, 4u, 8u, 12u})
            for (uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(words(*def, x.form | unused), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
            }
    }
}
TEST(IrVectorPack16Execution, ExistingModifierRefusalsAndUnusedThirdSourceBitsRemain) {
    for (const auto& def : vector_pack16_defs()) {
        const auto in = lane_inputs(149);
        for (uint32_t flag = 0; flag < 8; ++flag) {
            auto code = words(def, 0);
            if (flag < 2)
                code[0] |= 1u << (8 + flag);
            else if (flag < 4)
                code[1] |= 1u << (27 + flag);
            else if (flag == 4)
                code[0] |= 1u << 15;
            else
                code[1] |= (flag - 4) << 27;
            const auto bad = run_vector_op_per_lane(code, in);
            EXPECT_TRUE(bad.d.empty());
            EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
        }
        const auto good = run_vector_op_per_lane(words(def, 0), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def, 0);
        code[0] |= 1u << 10;
        code[1] |= 1u << 31;
        const auto same = run_vector_op_per_lane(code, in);
        ASSERT_TRUE(same.error.empty()) << same.error;
        EXPECT_EQ(same.d, good.d);
    }
}
TEST(IrVectorPack16Execution, EffectsDemandOnlyTwoSourcesAndIrCoversTheDefinition) {
    std::set<Opcode> seen;
    for (const auto& def : vector_pack16_defs()) {
        const auto& effect = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_FALSE(effect.vdst_half);
        EXPECT_EQ(effect.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> sources;
        def.build(ir, TernaryInputs(ir, sources));
        EXPECT_EQ((std::set<uint32_t>(sources.begin(), sources.end())), (std::set<uint32_t>{0, 1}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 1u, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {1u, std::nullopt, std::nullopt}));
        EXPECT_EQ(eval_ternary_integer_op(def, {1u, 2u, std::nullopt}), Known(0x00020001));
    }
    EXPECT_TRUE(seen.contains(Opcode::Shl));
    EXPECT_TRUE(seen.contains(Opcode::Or));
    EXPECT_EQ(find_vector_pack16_op(Rdna2Format::VOP3P, 0x311), nullptr);
    EXPECT_EQ(find_vector_pack16_op(Rdna2Format::VOP3, 0x303), nullptr);
}
// The recorded corpus did not observe this family's output. Export every existing plain form
// using exactly the execution harness's program and resource/launch bindings, without touching it.
TEST(IrVectorPack16Execution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "pack16-witness";
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
    for (const auto& def : vector_pack16_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t form = 0; form < 16; ++form)
                for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
                    const auto program = lane_program(words(def, form, mode), 64, wave, false);
                    const uint64_t address = 0x5253000000ull + count * 0x100;
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
    EXPECT_EQ(count, 224u);
}

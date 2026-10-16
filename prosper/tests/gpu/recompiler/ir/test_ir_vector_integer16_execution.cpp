// AMD 70648 12.12 selected-half integer operations: independently worked ISA examples,
// plus emitted SPIR-V vs eval for differing lanes and EXEC-masked opposite-half preservation.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/ir/ops/vector_integer16.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
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
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1103515245u + 12345u;
        in.s2.push_back(seed == in.old_d[lane] ? seed ^ 1u : seed);
    }
    return in;
}
uint32_t forms(const TernaryIntegerDef& def) {
    return 1u << (rdna2_valu_source_count(def.fmt, def.opcode) + 1u);
}
uint32_t opsel(const TernaryIntegerDef& def, uint32_t form) {
    const uint32_t count = rdna2_valu_source_count(def.fmt, def.opcode);
    return (form & ((1u << count) - 1u)) | ((form >> count) << 3);
}
std::vector<uint32_t> words(const TernaryIntegerDef& def, uint32_t form, uint32_t mode = 0) {
    const uint32_t s0 = mode == 1 ? 135 : mode == 2 ? 193 : mode == 3 ? 242 : 256 + kVsrc0;
    const uint32_t s1 = mode == 1 ? 137 : mode == 2 ? 194 : mode == 3 ? 242 : 256 + kVsrc1;
    const uint32_t s2 = mode == 1 ? 139 : mode == 2 ? 195 : mode == 3 ? 242 : 256 + kVsrc2;
    return {0xd4000000u | (def.opcode << 16) | (opsel(def, form) << 11) | kVdst,
            s0 | (s1 << 9) | (s2 << 18)};
}
std::array<Known, 3> selected(const LaneInputs& in, size_t lane, uint32_t select,
                              uint32_t mode = 0) {
    const uint32_t a = mode == 1 ? 7 : mode == 2 ? 0xffffffffu : mode == 3 ? 0x3c00 : in.s0[lane];
    const uint32_t b = mode == 1 ? 9 : mode == 2 ? 0xfffffffeu : mode == 3 ? 0x3c00 : in.s1[lane];
    const uint32_t c = mode == 1 ? 11 : mode == 2 ? 0xfffffffdu : mode == 3 ? 0x3c00 : in.s2[lane];
    return {(a >> (16 * (select & 1))) & 0xffffu, (b >> (16 * ((select >> 1) & 1))) & 0xffffu,
            (c >> (16 * ((select >> 2) & 1))) & 0xffffu};
}
uint32_t merge(uint32_t old, uint32_t value, uint32_t select) {
    return (select & 8) ? (old & 0xffffu) | (value << 16) : (old & 0xffff0000u) | value;
}
void check(const TernaryIntegerDef& def, uint32_t form, uint32_t wave, uint32_t seed,
           uint32_t mode = 0) {
    const auto in = inputs(seed);
    const auto run = run_vector_op_per_lane(words(def, form, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error << " opcode " << def.opcode;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto value = eval_ternary_integer_op(def, selected(in, lane, opsel(def, form), mode));
        if (!value) {
            ADD_FAILURE() << "unknown selected-half result at lane " << lane;
            continue;
        }
        EXPECT_EQ(*value & 0xffff0000u, 0u);
        const auto want =
            in.active[lane] ? merge(in.old_d[lane], *value, opsel(def, form)) : in.old_d[lane];
        EXPECT_EQ(run.d[lane], want)
            << "lane " << lane << " op " << def.opcode << " form " << form << " wave " << wave;
    }
}
} // namespace
TEST(IrVectorInteger16Execution, EveryDefinitionAndHalfSelectionMatchesEachLaneInBothWaves) {
    size_t count = 0;
    for (const auto& def : vector_integer16_defs())
        for (uint32_t form = 0; form < forms(def); ++form) {
            const auto code = words(def, form);
            std::vector<Rdna2Inst> decoded;
            ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
            ASSERT_EQ(decoded.size(), 1u);
            EXPECT_EQ(decoded[0].opcode, def.opcode);
            EXPECT_EQ(decoded[0].vop3p_opsel, opsel(def, form));
            for (uint32_t wave : {64u, 32u})
                for (uint32_t seed = 1; seed <= 8; ++seed) check(def, form, wave, seed * 5252);
            ++count;
        }
    EXPECT_EQ(count, 224u);
}
TEST(IrVectorInteger16Execution, InlineIntegerAndFloatBusBitsHonorSourceHalfSelection) {
    for (const auto& def : vector_integer16_defs())
        for (uint32_t form = 0; form < forms(def); ++form)
            for (uint32_t mode : {1u, 2u, 3u})
                for (uint32_t wave : {64u, 32u}) check(def, form, wave, 79 + mode, mode);
}
TEST(IrVectorInteger16Execution, HandWorkedIsaWrappingShiftCountsAndSignedOrder) {
    struct Example {
        uint32_t op, a, b, c, want;
    };
    // Wrapping arithmetic is modulo65536. Shift counts take only bits[3:0].
    // Signed order interprets 8000=-32768 and ffff=-1; other destination bits survive.
    for (auto x : {Example{0x303, 0xffff, 1, 0, 0},
                   Example{0x30d, 0x7fff, 1, 0, 0x8000},
                   Example{0x304, 0, 1, 0, 0xffff},
                   Example{0x30e, 0x8000, 1, 0, 0x7fff},
                   Example{0x305, 0xffff, 0xffff, 0, 1},
                   Example{0x305, 0x8000, 2, 0, 0},
                   Example{0x305, 0x1234, 0x10, 0, 0x2340},
                   Example{0x314, 16, 0x8001, 0, 0x8001},
                   Example{0x314, 31, 0x8001, 0, 0x8000},
                   Example{0x307, 16, 0x8001, 0, 0x8001},
                   Example{0x307, 31, 0x8001, 0, 1},
                   Example{0x308, 16, 0x8001, 0, 0x8001},
                   Example{0x308, 31, 0x8001, 0, 0xffff},
                   Example{0x309, 0xffff, 0x7fff, 0, 0xffff},
                   Example{0x30a, 0xffff, 0x7fff, 0, 0x7fff},
                   Example{0x30b, 0x8000, 0x7fff, 0, 0x7fff},
                   Example{0x30c, 0x8000, 0x7fff, 0, 0x8000},
                   Example{0x340, 0xffff, 0xffff, 1, 2},
                   Example{0x35e, 0x8000, 2, 0xffff, 0xffff},
                   Example{0x340, 0x1234, 0x10, 3, 0x2343},
                   Example{0x352, 0xffff, 0x8000, 0x7fff, 0x8000},
                   Example{0x353, 0xffff, 0x8000, 0x7fff, 0x7fff},
                   Example{0x355, 0xffff, 0x8000, 0x7fff, 0x7fff},
                   Example{0x356, 0xffff, 0x8000, 0x7fff, 0xffff},
                   Example{0x358, 0xffff, 0x8000, 0x7fff, 0xffff},
                   Example{0x359, 0xffff, 0x8000, 0x7fff, 0x8000}}) {
        const auto* def = find_vector_integer16_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = inputs(19);
        // Both source halves contain the hand value, while oldD differs per lane.
        std::fill(in.s0.begin(), in.s0.end(), x.a * 0x10001u);
        std::fill(in.s1.begin(), in.s1.end(), x.b * 0x10001u);
        std::fill(in.s2.begin(), in.s2.end(), x.c * 0x10001u);
        for (uint32_t form = 0; form < forms(*def); ++form) {
            const auto run = run_vector_op_per_lane(words(*def, form), in);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane]
                                           ? merge(in.old_d[lane], x.want, opsel(*def, form))
                                           : in.old_d[lane]);
        }
    }
}
TEST(IrVectorInteger16Execution, ExistingClampAndAbsNegRefusalsRemainWhileOmodIsIgnored) {
    for (const auto& def : vector_integer16_defs()) {
        const auto in = inputs(113);
        for (uint32_t modifier = 0; modifier < 7; ++modifier) {
            auto code = words(def, 0);
            if (modifier == 0)
                code[0] |= 1u << 15;
            else if (modifier <= 3)
                code[0] |= 1u << (7 + modifier);
            else
                code[1] |= 1u << (25 + modifier);
            const auto bad = run_vector_op_per_lane(code, in);
            EXPECT_TRUE(bad.d.empty());
            EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
        }
        const auto good = run_vector_op_per_lane(words(def, 0), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        for (uint32_t omod : {1u, 2u, 3u}) {
            auto code = words(def, 0);
            code[1] |= omod << 27;
            const auto run = run_vector_op_per_lane(code, in);
            ASSERT_TRUE(run.error.empty()) << run.error;
            EXPECT_EQ(run.d, good.d);
        }
    }
}
TEST(IrVectorInteger16Execution, EffectsSourceDemandAndIrCoverageAreAuthoritative) {
    std::set<Opcode> seen;
    for (const auto& def : vector_integer16_defs()) {
        const auto& effect = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_TRUE(effect.vdst_half);
        EXPECT_EQ(effect.vdst_dwords, 1u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> sources;
        def.build(ir, TernaryInputs(ir, sources));
        const std::set<uint32_t> slots(sources.begin(), sources.end());
        const uint32_t count = rdna2_valu_source_count(def.fmt, def.opcode);
        EXPECT_EQ(slots.size(), count);
        for (uint32_t k = 0; k < count; ++k) EXPECT_TRUE(slots.contains(k));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 1u, 1u}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {1u, std::nullopt, 1u}));
        EXPECT_EQ(eval_ternary_integer_op(def, {1u, 1u, std::nullopt}).has_value(), count == 2);
    }
    for (auto op :
         {Opcode::IAdd, Opcode::ISub, Opcode::IMul, Opcode::And, Opcode::BfeS, Opcode::Shl,
          Opcode::LShr, Opcode::AShr, Opcode::UMin, Opcode::UMax, Opcode::SMin, Opcode::SMax})
        EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_integer16_op(Rdna2Format::VOP3, 0x306), nullptr);
    EXPECT_EQ(find_vector_integer16_op(Rdna2Format::VOP3P, 0x303), nullptr);
}
// The recorded corpus did not observe this family's output. Export every existing plain form
// using exactly the execution harness's program and resource/launch bindings, without touching it.
TEST(IrVectorInteger16Execution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "integer16-witness";
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
    for (const auto& def : vector_integer16_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t form = 0; form < forms(def); ++form)
                for (uint32_t mode : {0u, 1u}) {
                    const auto program = lane_program(words(def, form, mode), 64, wave, true);
                    const uint64_t address = 0x5252000000ull + count * 0x100;
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
    EXPECT_EQ(count, 896u);
}

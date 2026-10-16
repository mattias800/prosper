// Newly admitted e64 F32 rounding runs through the real recompiler and per-lane
// SPIR-V interpreter. Eval checks many inputs; literal ISA examples check both sides.
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <ios>
#include "gpu/recompiler/ir/ops/vector_round_float.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
struct Form {
    bool abs = false, neg = false, clamp = false;
    uint32_t scale = 0, source = 258, literal = 0;
};
std::vector<uint32_t> words(uint32_t op, Form f = {}) {
    std::vector<uint32_t> out{0xd4000001u | (op << 16) | (uint32_t(f.abs) << 8) |
                                  (uint32_t(f.clamp) << 15),
                              f.source | (f.scale << 27) | (uint32_t(f.neg) << 29)};
    if (f.source == 255) out.push_back(f.literal);
    return out;
}
Form form(uint32_t n) {
    if (n < 32) return {bool(n & 1), bool(n & 2), bool(n & 16), (n >> 2) & 3};
    if (n == 32) return {false, false, false, 0, 257};
    if (n == 33) return {false, false, false, 0, 242};
    return {false, false, false, 0, 255, n == 34 ? 0x3fa00000u : 0xbfe00000u};
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const float x = float(int32_t(in.s0[lane] % 65535) - 32767) / 1024.0f;
        in.s0[lane] = std::bit_cast<uint32_t>(x);
        in.old_d[lane] = std::bit_cast<uint32_t>(float(lane + 1) / 16.0f);
    }
    return in;
}
Known expected(const VectorOpDef& def, uint32_t source, Form f, uint32_t rsrc1) {
    if (f.abs) source &= 0x7fffffffu;
    if (f.neg) source ^= 0x80000000u;
    auto result = eval_vector_op(def, source, std::nullopt);
    if (!result) return {};
    float x = std::bit_cast<float>(*result);
    // AMD70648 section 6.2: either output-denorm enable or IEEE suppresses OMOD.
    // The ordinary finite test values scale exactly; enabled OMOD canonicalizes -0.
    if (f.scale && !(rsrc1 & ((0x20u << 12) | (1u << 23)))) {
        x *= f.scale == 1 ? 2.0f : f.scale == 2 ? 4.0f : 0.5f;
        if (x == 0.0f) x = 0.0f;
    }
    if (f.clamp) {
        // With finite nonzero intermediate results, NMax(x,0) then NMin(x,1)
        // is unambiguous. Zero-sign ambiguity is a separate negative below.
        if (x == 0.0f) return {};
        x = std::min(std::max(x, 0.0f), 1.0f);
    }
    return std::bit_cast<uint32_t>(x);
}
void compare_run(const VectorOpDef& def, const LaneInputs& in, uint32_t wave, Form f,
                 uint32_t rsrc1, const LaneRun& run) {
    ASSERT_TRUE(run.error.empty()) << run.error << " opcode " << def.opcode;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        uint32_t source = f.source == 257   ? in.old_d[lane]
                          : f.source == 242 ? 0x3f800000u
                          : f.source == 255 ? f.literal
                                            : in.s0[lane];
        const auto want = in.active[lane] ? expected(def, source, f, rsrc1) : Known(in.old_d[lane]);
        if (!want) {
            ADD_FAILURE() << "opcode " << def.opcode << " lane " << lane;
            return;
        }
        EXPECT_EQ(run.d[lane], *want)
            << "opcode " << def.opcode << " wave " << wave << " lane " << lane;
    }
}
void compare(const VectorOpDef& def, const LaneInputs& in, uint32_t wave, Form f = {},
             uint32_t rsrc1 = kDefaultComputePgmRsrc1) {
    compare_run(def, in, wave, f, rsrc1,
                run_vector_op_per_lane(words(def.opcode, f), in, wave, false, rsrc1));
}
}   // namespace
TEST(IrVectorRoundFloatExecution, EveryNewFormMatchesDistinctActiveAndInactiveLanes) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed) compare(def, inputs(seed * 5304), wave);
}
TEST(IrVectorRoundFloatExecution, SourceAndOutputModifiersMatchTheirNumericOrder) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t n = 0; n < 32; ++n) {
                auto in = inputs(37 + n);
                // Keep all rounded results nonzero for the strict NMax zero-sign domain.
                // Varying magnitudes keep this independent of the constant examples below.
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    in.s0[lane] = std::bit_cast<uint32_t>(float(2 + lane % 16) + 0.25f);
                for (uint32_t rsrc1 : {0u, 0x30u << 12, 1u << 23})
                    compare(def, in, wave, form(n), rsrc1);
            }
}
TEST(IrVectorRoundFloatExecution, DestinationAliasAndInlineOrLiteralInputsExecute) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t n = 32; n < 36; ++n) compare(def, inputs(73), wave, form(n));
}
TEST(IrVectorRoundFloatExecution, HandWorkedIsaExamplesCoverEachNewOperationAndTies) {
    struct Example {
        uint32_t op, source, want;
    };
    // AMD70648 §12.8, independently worked: -1.25 - floor(-1.25) = .75;
    // trunc/ceil(-1.75)=-1; floor(-1.25)=-2; ties 2.5->2, 3.5->4, -2.5->-2.
    for (const auto x :
         {Example{0x1a0, 0xbfa00000, 0x3f400000}, Example{0x1a1, 0xbfe00000, 0xbf800000},
          Example{0x1a2, 0xbfe00000, 0xbf800000}, Example{0x1a3, 0x40200000, 0x40000000},
          Example{0x1a3, 0x40600000, 0x40800000}, Example{0x1a3, 0xc0200000, 0xc0000000},
          Example{0x1a3, 0xbe800000, 0x80000000}, Example{0x1a4, 0xbfa00000, 0xc0000000}}) {
        const auto* def = find_vector_round_float_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.source, std::nullopt), Known(x.want));
        for (uint32_t wave : {64u, 32u}) {
            auto in = inputs(7);
            std::fill(in.s0.begin(), in.s0.end(), x.source);
            const auto run = run_vector_op_per_lane(words(x.op), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
        }
    }
}
TEST(IrVectorRoundFloatExecution, HandWorkedOutputModesDistinguishScalingSuppressionAndZero) {
    struct Example {
        uint32_t op, source, scale, plain, scaled;
    };
    // Worked independently from AMD70648 sections 6.2 and 12.8. FRACT(-1.25)=.75;
    // TRUNC/CEIL(-1.75)=-1; FLOOR(-1.25)=-2; RNDNE(-.25)=-0.
    // Half the minimum normal flushes; twice the maximum finite overflows to +Inf.
    for (const auto x : {Example{0x1a0, 0xbfa00000, 1, 0x3f400000, 0x3fc00000},
                         Example{0x1a0, 0x00800000, 3, 0x00800000, 0},
                         Example{0x1a1, 0x7f7fffff, 1, 0x7f7fffff, 0x7f800000},
                         Example{0x1a1, 0xbfe00000, 1, 0xbf800000, 0xc0000000},
                         Example{0x1a2, 0xbfe00000, 1, 0xbf800000, 0xc0000000},
                         Example{0x1a3, 0xbe800000, 1, 0x80000000, 0},
                         Example{0x1a4, 0xbfa00000, 2, 0xc0000000, 0xc1000000},
                         Example{0x1a4, 0x3fa00000, 3, 0x3f800000, 0x3f000000}})
        for (uint32_t wave : {64u, 32u})
            for (uint32_t rsrc1 : {0u, 0x30u << 12, 1u << 23}) {
                auto in = inputs(5311);
                std::fill(in.s0.begin(), in.s0.end(), x.source);
                Form f;
                f.scale = x.scale;
                const auto run = run_vector_op_per_lane(words(x.op, f), in, wave, false, rsrc1);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane],
                              in.active[lane] ? (rsrc1 ? x.plain : x.scaled) : in.old_d[lane])
                        << "opcode " << x.op << " wave " << wave << " lane " << lane
                        << " PGM_RSRC1 " << rsrc1;
            }
}
TEST(IrVectorRoundFloatExecution, MissingLaunchModeRefusesOnlyAnActiveOutputScale) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u}) {
            Form f;
            f.scale = 1;
            const auto scaled =
                run_vector_op_per_lane(words(def.opcode, f), inputs(7), wave, false, UINT32_MAX);
            EXPECT_NE(scaled.error.find("float-omod-launch-mode-unavailable"), std::string::npos)
                << scaled.error;
            const auto plain =
                run_vector_op_per_lane(words(def.opcode), inputs(7), wave, false, UINT32_MAX);
            EXPECT_TRUE(plain.error.empty()) << plain.error;
        }
}
TEST(IrVectorRoundFloatExecution, HarnessAndEffectsRetainOneOwnedWordPerLane) {
    const auto in = inputs(97);
    const auto run = run_vector_op_per_lane(words(0x1a4), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> distinct;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            distinct.insert(run.d[lane]);
        else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(distinct.size(), 8u);
    EXPECT_GT(inactive, 1u);
    for (const auto& def : vector_round_float_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 1u);
        EXPECT_FALSE(effects.accumulates);
        EXPECT_FALSE(effects.vdst_half);
        EXPECT_FALSE(eval_vector_op(def, std::nullopt, 0x3f800000));
        EXPECT_FALSE(eval_vector_op(def, 0x7fc00001, std::nullopt));
        EXPECT_EQ(eval_vector_op(def, 0xbfa00000, std::nullopt),
                  eval_vector_op(*find_vector_float_unary_op(Rdna2Format::VOP1, def.opcode - 0x180),
                                 0xbfa00000, std::nullopt));
    }
}
TEST(IrVectorRoundFloatExecution, InvalidSourceAndReservedModifiersRemainRefused) {
    for (const auto& def : vector_round_float_defs()) {
        auto code = words(def.opcode);
        code[1] = 251;   // Special source has no numeric service in operand_bits.
        const auto missing = run_vector_op_per_lane(code, inputs(7));
        EXPECT_FALSE(missing.error.empty());
        auto reserved = words(def.opcode);
        reserved[0] |= 1u << 11;
        const auto bad = run_vector_op_per_lane(reserved, inputs(7));
        EXPECT_FALSE(bad.error.empty());
        EXPECT_TRUE(run_vector_op_per_lane(words(def.opcode), inputs(7)).error.empty());
    }
}
TEST(IrVectorRoundFloatExecution, FabricatedScalarWordsCannotBecomeFreshCompareMasks) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (bool restore : {false, true}) {
                auto bad = words(def.opcode);
                bad[1] = 50;   // Missing scalar s50 is hard, unlike an arbitrary unloaded VGPR.
                if (restore) bad.push_back(0xbefe04c1);
                bad.push_back(0x7d8a0280);   // v_cmp_ne_u32_e32 vcc,0,v1
                const auto refused = run_vector_op_per_lane(bad, inputs(7), wave);
                EXPECT_NE(refused.error.find("fabricated-fresh-mask-source"), std::string::npos)
                    << refused.error << " opcode " << def.opcode;
                auto good = words(def.opcode);
                if (restore) good.push_back(0xbefe04c1);
                good.push_back(0x7d8a0280);
                const auto control = run_vector_op_per_lane(good, inputs(7), wave);
                EXPECT_TRUE(control.error.empty()) << control.error;
            }
}
TEST(IrVectorRoundFloatExecution, OneArmScalarDefinitionsKeepTheirHardSourceObligation) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> code;
                if (initialize)
                    code.insert(code.end(), {0xbeb203ff, 0x3f800000});   // s50=1 on both edges
                code.insert(code.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x3fa00000});
                // Dynamic s4==0: only one edge writes s50=1.25.
                const auto op = words(def.opcode, {false, false, false, 0, 50});
                code.insert(code.end(), op.begin(), op.end());
                code.push_back(0x7d8a0280);
                auto program = lane_program(code, 64, wave);
                program.config.user_sgprs.resize(8);   // genuine dynamic scalar branch selector s4
                const auto address = lane_harness_address();
                const auto module =
                    recompile_compute(program.code.data(), program.code.size(), &program.table,
                                      program.config, {RecompileDiagnosticStage::Compute, address});
                const auto reason = last_terminal_reject_reason(address);
                if (initialize)
                    EXPECT_FALSE(module.empty()) << reason;
                else {
                    EXPECT_TRUE(module.empty());
                    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos)
                        << reason;
                }
            }
}
TEST(IrVectorRoundFloatExecution, EncodingsMatchIndependentLlvmChecks) {
    EXPECT_EQ(words(0x1a0), (std::vector<uint32_t>{0xd5a00001, 0x00000102}));
    EXPECT_EQ(words(0x1a4, {true, true, true, 1}), (std::vector<uint32_t>{0xd5a48101, 0x28000102}));
}
TEST(IrVectorRoundFloatExecution, NoncanonicalUnusedFieldsIncludingLiteralSelectorsRefuse) {
    for (const auto& def : vector_round_float_defs())
        for (uint32_t extra : {50u << 9, 510u << 18, 255u << 9}) {
            auto code = words(def.opcode);
            code[1] |= extra;
            if (extra == (255u << 9)) code.push_back(0x3fa00000);
            EXPECT_FALSE(run_vector_op_per_lane(code, inputs(7)).error.empty());
            EXPECT_TRUE(run_vector_op_per_lane(words(def.opcode), inputs(7)).error.empty());
        }
}
TEST(IrVectorRoundFloatExecution, NeighboringUnaryFamiliesKeepTheirCurrentAdmission) {
    for (uint32_t op : {0x1a5u, 0x1abu, 0x1afu, 0x1b7u}) {
        EXPECT_EQ(find_vector_round_float_op(Rdna2Format::VOP3, op), nullptr);
        // Main already admits integer NOT through its own family (#5316).
        EXPECT_EQ(run_vector_op_per_lane(words(op), inputs(7)).error.empty(), op == 0x1b7u);
    }
}
TEST(IrVectorRoundFloatExecution, AmbiguousClampedZeroRemainsAnOracleNegative) {
    auto in = inputs(7);
    std::fill(in.s0.begin(), in.s0.end(), 0u);
    Form f;
    f.clamp = true;
    const auto run = run_vector_op_per_lane(words(0x1a1, f), in);
    EXPECT_NE(run.error.find("unspecified float ALU result"), std::string::npos);
    f.clamp = false;
    EXPECT_TRUE(run_vector_op_per_lane(words(0x1a1, f), in).error.empty());
}
TEST(IrVectorRoundFloatExecution, ExportsEveryNewFormAndItsOldEncodingControls) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed test-process environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "round-float-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    size_t count = 0;
    for (const auto& def : vector_round_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t n = 0; n < 37; ++n) {
                const auto op = n == 36 ? std::vector<uint32_t>{vop1(def.opcode - 0x180)}
                                        : words(def.opcode, form(n));
                const auto program = lane_program(op, 64, wave);
                const uint64_t address = 0x5304000000ull + count * 0x100;
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
                ASSERT_TRUE(replay.produced) << replay.reject;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                // Execute this exact sidecar-replayed module before counting its admission.
                auto in = inputs(37 + n);
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    in.s0[lane] = std::bit_cast<uint32_t>(float(2 + lane % 16) + 0.25f);
                std::vector<uint32_t> buffer;
                for (const auto* array : {&in.s0, &in.s1, &in.old_d})
                    buffer.insert(buffer.end(), array->begin(), array->end());
                for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
                bpermute_oracle::Interpreter vm(replay.spirv);
                auto values = vm.run_buffers(64, {{3, buffer}}, 3);
                ASSERT_TRUE(vm.error.empty())
                    << vm.error << " opcode " << def.opcode << " wave " << wave << " form " << n;
                ASSERT_GE(values.size(), in.lanes());
                values.resize(in.lanes());
                compare_run(def, in, wave, n == 36 ? Form{} : form(n),
                            program.config.compute_pgm_rsrc1, {values, {}});
                ++count;
            }
    EXPECT_EQ(count, 370u);
}

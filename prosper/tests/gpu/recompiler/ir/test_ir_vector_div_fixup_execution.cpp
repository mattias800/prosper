// AMD70648 division-fixup special numerics over real emitted per-lane SPIR-V.
#include "gpu/recompiler/ir/ops/vector_div_fixup.hpp"
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
    const uint32_t edge[] = {0,          0x80000000, 0x7f800000, 0xff800000, 0x7f800123, 0xff800456,
                             0x7fc01234, 0xffc05678, 0x3f800000, 0xbf800000, 0x00800000, 0x80800000,
                             1,          0x80000001, 0x3fc00000, 0x7f7fffff};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 16 ? edge[lane] : in.s0[lane];
        in.s1[lane] = lane < 16 ? edge[(lane * 5 + seed) % 16] : in.s1[lane];
        seed = seed * 1664525u + 1013904223u;
        in.s2.push_back(lane < 16 ? edge[(lane * 7 + 3) % 16] : seed);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 4 ? 242 : mode == 5 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 4 ? 244 : mode == 6 ? 255 : 256 + kVsrc1;
    const uint32_t c = mode == 3 || mode == 4 ? 241 : mode == 7 ? 255 : 256 + kVsrc2;
    std::vector<uint32_t> code = {0xd55f0000u | kVdst | (abs << 8) | (clamp ? 1u << 15 : 0),
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
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown fixup lane " << lane;
            continue;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorDivFixupExecution, EveryDefinitionMatchesRawWordsInBothWaves) {
    for (const auto& def : vector_div_fixup_defs()) {
        const auto code = words();
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed) compare(def, inputs(seed * 5272), wave);
    }
}
TEST(IrVectorDivFixupExecution, HarnessKeepsQuotientSensitivityAndInactiveDestinations) {
    auto in = inputs(43);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = (in.s0[lane] & 0x807fffffu) | (127u << 23);
        in.s1[lane] = 0x40000000;
        in.s2[lane] = 0x40400000;
    }
    const auto run = run_vector_op_per_lane(words(), in);
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
    const auto bad = run_vector_op_per_lane(words(), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorDivFixupExecution, HandWorkedIsaPrecedenceSignAndUnderflowExamples) {
    struct X {
        uint32_t q, den, num, want;
    };
 // AMD70648 op351: numerator NaN first, then denominator NaN, indeterminate cases,
    // signed infinity/zero, underflow, and finally quotient magnitude with denominator XOR numerator sign.
    const X cases[] = {{0x3fc00000, 0x40000000, 0x40400000, 0x3fc00000},
                       {0xbfc00000, 0x40000000, 0x40400000, 0x3fc00000},
                       {0x3fc00000, 0xc0000000, 0x40400000, 0xbfc00000},
                       {0x3fc00000, 0xc0000000, 0xc0400000, 0x3fc00000},
                       {0x3fc00000, 0, 0x3f800000, 0x7f800000},
                       {0x3fc00000, 0x80000000, 0x3f800000, 0xff800000},
                       {0x3fc00000, 0, 0xbf800000, 0xff800000},
                       {0x3fc00000, 0x7f800000, 0x3f800000, 0},
                       {0x3fc00000, 0xff800000, 0x3f800000, 0x80000000},
                       {0x3fc00000, 0x3f800000, 0, 0},
                       {0x3fc00000, 0xbf800000, 0x80000000, 0},
                       {0x3fc00000, 0, 0x80000000, 0xffc00000},
                       {0x3fc00000, 0xff800000, 0x7f800000, 0xffc00000},
                       {0x3fc00000, 0x40000000, 0x7f800000, 0x7f800000},
                       {0x3fc00000, 0xc0000000, 0x7f800000, 0xff800000},
                       {0x3fc00000, 0x7f800123, 0x3f800000, 0x7fc00123},
                       {0x3fc00000, 0x3f800000, 0xff800456, 0xffc00456},
                       {0x3fc00000, 0x7f800123, 0xff800456, 0xffc00456},
                       {0x7fc00033, 0xc0000000, 0x3f800000, 0xffc00033},
                       {0x3fc00000, 0x64800000, 0x19000000, 0},
                       {0x3fc00000, 0xe4800000, 0x19000000, 0x80000000},
                       {0x3fc00000, 0x64800000, 0x19800000, 0x3fc00000},
                       {0x3fc00000, 0x64800000, 0x1a000000, 0x3fc00000}};
    for (const auto& def : vector_div_fixup_defs())
        for (auto x : cases) {
            EXPECT_EQ(eval_ternary_integer_op(def, {x.q, x.den, x.num}), Known(x.want));
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), x.q);
            std::fill(in.s1.begin(), in.s1.end(), x.den);
            std::fill(in.s2.begin(), in.s2.end(), x.num);
            for (uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(words(), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
            }
        }
}
TEST(IrVectorDivFixupExecution, EveryDenominatorExponentCrossesTheUnderflowBoundary) {
    for (const auto& def : vector_div_fixup_defs())
        for (uint32_t den = 1; den < 255; ++den) {
            auto in = inputs(den);
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const int delta = -154 + static_cast<int>(lane % 8);
                const int num = std::clamp(static_cast<int>(den) + delta, 0, 254);
                in.s0[lane] = (in.s0[lane] & 0x807fffffu) | (127u << 23);
                in.s1[lane] = (in.s1[lane] & 0x807fffffu) | (den << 23) | 1u;
                in.s2[lane] = (in.s2[lane] & 0x807fffffu) | (static_cast<uint32_t>(num) << 23) | 1u;
            }
            for (uint32_t wave : {64u, 32u}) compare(def, in, wave);
        }
}
TEST(IrVectorDivFixupExecution, InlineAndLiteralEachSourceKeepTheirRole) {
    for (const auto& def : vector_div_fixup_defs())
        for (uint32_t mode = 1; mode <= 7; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, mode);
}
TEST(IrVectorDivFixupExecution, AllSourceAndResultModifierCombinationsKeepTheCallerOrder) {
    auto in = inputs(43);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane & 1 ? 0xbfc00000 : 0x3fc00000;
        in.s1[lane] = lane & 2 ? 0xc0000000 : 0x40000000;
        in.s2[lane] = lane & 4 ? 0xc0400000 : 0x40400000;
    }
    for (const auto& def : vector_div_fixup_defs())
        for (uint32_t abs = 0; abs < 8; ++abs)
            for (uint32_t neg = 0; neg < 8; ++neg)
                for (uint32_t omod = 0; omod < 4; ++omod)
                    for (bool clamp : {false, true})
                        for (uint32_t wave : {64u, 32u}) {
                            const auto run =
                                run_vector_op_per_lane(words(0, abs, neg, omod, clamp), in, wave);
                            ASSERT_TRUE(run.error.empty()) << run.error;
                            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                                std::array<uint32_t, 3> args = {in.s0[lane], in.s1[lane],
                                                                in.s2[lane]};
                                for (size_t k = 0; k < 3; ++k) {
                                    if (abs & (1u << k)) args[k] &= 0x7fffffffu;
                                    if (neg & (1u << k)) args[k] ^= 0x80000000u;
                                }
                                auto want =
                                    eval_ternary_integer_op(def, {args[0], args[1], args[2]});
                                if (!want) {
                                    ADD_FAILURE();
                                    return;
                                }
                                if (omod)
                                    want = evaluate_op(Opcode::F32Mul, 0, *want,
                                                       omod == 1   ? 0x40000000
                                                       : omod == 2 ? 0x40800000
                                                                   : 0x3f000000,
                                                       0);
                                if (!want) {
                                    ADD_FAILURE();
                                    return;
                                }
                                if (clamp) want = evaluate_op(Opcode::F32Clamp01, 0, *want, 0, 0);
                                if (!want) {
                                    ADD_FAILURE();
                                    return;
                                }
                                EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                            }
                        }
}
TEST(IrVectorDivFixupExecution, EffectsAllValueOpcodesAndUnknownSourcesAreCovered) {
    std::set<Opcode> used;
    for (const auto& def : vector_div_fixup_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block b;
        Builder ir(b);
        std::vector<uint32_t> slots;
        def.build(ir, TernaryInputs(ir, slots));
        for (const auto& inst : b.insts()) used.insert(inst.op);
        for (size_t k = 0; k < 3; ++k) {
            std::array<Known, 3> args = {0x3fc00000, 0x40000000, 0x40400000};
            args[k] = std::nullopt;
            EXPECT_FALSE(eval_ternary_integer_op(def, args));
        }
    }
    EXPECT_EQ(used,
              (std::set<Opcode>{Opcode::Input, Opcode::ConstU32, Opcode::And, Opcode::Or,
                                Opcode::Xor, Opcode::LShr, Opcode::ISub, Opcode::UGt, Opcode::IEq,
                                Opcode::SLt, Opcode::BoolOr, Opcode::BoolAnd, Opcode::Select}));
    EXPECT_EQ(find_vector_div_fixup_op(Rdna2Format::VOP3, 0x16d), nullptr);
    EXPECT_EQ(find_vector_div_fixup_op(Rdna2Format::VOP3, 0x16f), nullptr);
    EXPECT_EQ(find_vector_div_fixup_op(Rdna2Format::VOP3, 0x35f), nullptr);
    EXPECT_EQ(find_vector_div_fixup_op(Rdna2Format::VOP3P, 0x15f), nullptr);
}
// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorDivFixupExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "div-fixup-witness";
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
    for (const auto& def : vector_div_fixup_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
                const auto program = lane_program(words(mode), 64, wave, false);
                const uint64_t address = 0x5272000000ull + count * 0x100;
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
    EXPECT_EQ(count, 16u);
}

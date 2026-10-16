// AMD70648 op866: independently evaluated integral powers of two, raw exponent bus,
// and hand-derived binary32 subnormal tie/overflow examples. No guest folding is expanded.
#include "gpu/recompiler/ir/ops/vector_ldexp.hpp"
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
    const std::array<uint32_t, 16> floats = {0,          0x80000000, 1,          0x80000001,
                                             0x007fffff, 0x00800000, 0x3f800000, 0x3fc00000,
                                             0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
                                             0xff800000, 0x7fc01234, 0xffc12345, 0x7f800001};
    const std::array<uint32_t, 16> exp = {
        0,          1,          0xffffffff, 23,         0xffffffe9, 127,        128, 0xffffff6b,
        0xffffff6a, 0xffffff69, 512,        0xfffffe00, 0x7fffffff, 0x80000000, 150, 0xffffff82};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < 16) {
            in.s0[lane] = floats[lane];
            in.s1[lane] = exp[(lane * 7 + seed) % 16];
        } else
            in.s1[lane] = static_cast<uint32_t>(static_cast<int32_t>(in.s1[lane] % 401) - 200);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t mode = 0, bool abs = false, bool neg = false,
                            bool clamp = false, uint32_t omod = 0) {
    const uint32_t a = mode == 1 || mode == 3 ? 242 : mode == 4 ? 255 : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 195 : mode == 5 ? 255 : 256 + kVsrc1;
    std::vector<uint32_t> code = {0xd7620000u | kVdst | (abs ? 1u << 8 : 0) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (neg ? 1u << 29 : 0) | (omod << 27)};
    if (mode >= 4) code.push_back(mode == 4 ? 0x3f800001 : 0x80000000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0) {
    return {mode == 1 || mode == 3 ? 0x3f800000u
            : mode == 4            ? 0x3f800001u
                                   : in.s0[lane],
            mode == 2 || mode == 3 ? 0xfffffffdu
            : mode == 5            ? 0x80000000u
                                   : in.s1[lane],
            std::nullopt};
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unexpected unknown LDEXP result";
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "wave " << wave << " lane " << lane << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorLdexpExecution, EveryDefinitionMatchesDifferingLanesAndBothWaves) {
    for (const auto& def : vector_ldexp_defs()) {
        auto code = words();
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed) compare(def, inputs(seed * 5265), wave);
    }
}
TEST(IrVectorLdexpExecution, HarnessKeepsInputSensitivityAndInactiveOldDestinations) {
    auto in = inputs(17);
    // Match each lane's magnitude with a small independent exponent, preventing zero/inf saturation from hiding a wiring error.
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] =
            (in.s0[lane] & 0x807fffffu) | ((120u + static_cast<uint32_t>(lane) % 14) << 23);
        in.s1[lane] = static_cast<uint32_t>(static_cast<int32_t>(lane % 11) - 5);
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
    in.s1.pop_back();
    const auto bad = run_vector_op_per_lane(words(), in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_FALSE(bad.error.empty());
}
TEST(IrVectorLdexpExecution, HandWorkedSubnormalTiesOverflowSpecialsAndSignedExponentLimits) {
    struct X {
        uint32_t a, e, want;
    };
    // min subnormal=2^-149. Halfway ties choose the even low significand; 1.5 and2.5 units both round to2.
    const X examples[] = {{0x3f800000, 0xffffff6b, 1},
                          {0x3f800000, 0xffffff6a, 0},
                          {0x3f800001, 0xffffff6a, 1},
                          {0x3fc00000, 0xffffff6b, 2},
                          {0x40200000, 0xffffff6b, 2},
                          {0x007fffff, 1, 0x00fffffe},
                          {1, 23, 0x00800000},
                          {0x7f7fffff, 1, 0x7f800000},
                          {0xbf800000, 0x80000000, 0x80000000},
                          {0x3f800000, 0x7fffffff, 0x7f800000},
                          {0x80000000, 0x7fffffff, 0x80000000},
                          {0x7f800000, 0x80000000, 0x7f800000},
                          {0x7fc01234, 127, 0x7fc01234},
                          {0xff800001, 0xffffff6a, 0xff800001},
                          {0x40c00000, 0xfffffffe, 0x3fc00000}};
    for (const auto& def : vector_ldexp_defs())
        for (auto x : examples) {
            EXPECT_EQ(eval_ternary_integer_op(def, {x.a, x.e, std::nullopt}), Known(x.want));
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), x.a);
            std::fill(in.s1.begin(), in.s1.end(), x.e);
            for (uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(words(), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
            }
        }
}
TEST(IrVectorLdexpExecution, EveryNormalInputExponentBoundaryMatches) {
    for (const auto& def : vector_ldexp_defs())
        for (uint32_t raw_exp = 1; raw_exp < 255; ++raw_exp) {
            auto in = inputs(raw_exp);
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                in.s0[lane] = (in.s0[lane] & 0x807fffffu) | (raw_exp << 23);
                // Center around normal/subnormal transition and around overflow; includes ties and extreme shifts.
                in.s1[lane] =
                    static_cast<uint32_t>((lane & 1 ? 254 : 1) - static_cast<int32_t>(raw_exp) +
                                          static_cast<int32_t>(lane % 9) - 4);
            }
            for (uint32_t wave : {64u, 32u}) compare(def, in, wave);
        }
}
TEST(IrVectorLdexpExecution, InlineFloatsSignedExponentsAndLiteralBothSourcesKeepTheBus) {
    for (const auto& def : vector_ldexp_defs())
        for (uint32_t mode = 1; mode <= 5; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(29 + mode), wave, mode);
}
TEST(IrVectorLdexpExecution, FloatAbsThenNegAndResultClampStayWithTheCaller) {
    for (const auto& def : vector_ldexp_defs())
        for (bool abs : {false, true})
            for (bool neg : {false, true})
                for (bool clamp : {false, true}) {
                    auto in = inputs(43);
                    for (size_t lane = 0; lane < in.lanes(); ++lane) {
                        in.s0[lane] = (in.s0[lane] & 0x807fffffu) | (126u << 23);
                        in.s1[lane] = static_cast<uint32_t>(static_cast<int32_t>(lane % 5) - 2);
                    }
                    for (uint32_t wave : {64u, 32u}) {
                        const auto run =
                            run_vector_op_per_lane(words(0, abs, neg, clamp), in, wave);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        for (size_t lane = 0; lane < in.lanes(); ++lane) {
                            uint32_t a = in.s0[lane];
                            if (abs) a &= 0x7fffffffu;
                            if (neg) a ^= 0x80000000u;
                            auto want =
                                eval_ternary_integer_op(def, {a, in.s1[lane], std::nullopt});
                            if (!want) {
                                ADD_FAILURE() << "unexpected unknown LDEXP result";
                                return;
                            }
                            if (clamp) want = evaluate_op(Opcode::F32Clamp01, 0, *want, 0, 0);
                            if (!want) {
                                ADD_FAILURE() << "unexpected unknown LDEXP result";
                                return;
                            }
                            EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                        }
                    }
                }
}
TEST(IrVectorLdexpExecution, EffectsUnknownsAndExistingModifierRefusalsRemain) {
    std::set<Opcode> used;
    for (const auto& def : vector_ldexp_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        Block b;
        Builder ir(b);
        std::vector<uint32_t> reads;
        def.build(ir, TernaryInputs(ir, reads));
        for (const auto& inst : b.insts()) used.insert(inst.op);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 0, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x3f800000, std::nullopt, std::nullopt}));
    }
    EXPECT_EQ(used, (std::set<Opcode>{Opcode::Input, Opcode::F32Ldexp}));
    EXPECT_EQ(find_vector_ldexp_op(Rdna2Format::VOP3, 0x363), nullptr);
    EXPECT_EQ(find_vector_ldexp_op(Rdna2Format::VOP2, 0x362), nullptr);
    for (uint32_t omod : {1u, 2u, 3u}) {
        const auto bad = run_vector_op_per_lane(words(0, false, false, false, omod), inputs(31));
        EXPECT_FALSE(bad.error.empty());
        EXPECT_TRUE(bad.d.empty());
    }
}
// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorLdexpExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "ldexp-witness";
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
    for (const auto& def : vector_ldexp_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u}) {
                const auto program = lane_program(words(mode), 64, wave, false);
                const uint64_t address = 0x5265000000ull + count * 0x100;
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
    EXPECT_EQ(count, 12u);
}

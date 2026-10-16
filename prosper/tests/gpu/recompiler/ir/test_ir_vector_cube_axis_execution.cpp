// AMD70648 op324/327: independent face/tie/sign examples plus emitted per-lane SPIR-V vs eval.
#include "gpu/recompiler/ir/ops/vector_cube_axis.hpp"
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
    const uint32_t edge[] = {0,          0x80000000, 1,          0x80000001, 0x3f800000, 0xbf800000,
                             0x40000000, 0xc0000000, 0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff,
                             0x007fffff, 0x00800000, 0x3f000000, 0xbf000000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 16 ? edge[lane]
                                : (in.s0[lane] & 0x807fffffu) |
                                      ((110u + static_cast<uint32_t>(lane) % 25) << 23);
        in.s1[lane] = lane < 16 ? edge[(lane * 3 + seed) % 16]
                                : (in.s1[lane] & 0x807fffffu) |
                                      ((112u + static_cast<uint32_t>(lane) % 21) << 23);
        seed = seed * 1664525u + 1013904223u;
        in.s2.push_back(lane < 16 ? edge[(lane * 5 + seed) % 16]
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
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    const auto run = run_vector_op_per_lane(words(def.opcode, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_ternary_integer_op(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown cube lane " << lane;
            return;
        }
        ASSERT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "op " << def.opcode << " wave " << wave << " lane " << lane;
    }
}
}   // namespace
TEST(IrVectorCubeAxisExecution, EveryDefinitionMatchesDistinctLanesInBothWaves) {
    for (const auto& def : vector_cube_axis_defs()) {
        auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed) compare(def, inputs(seed * 5288), wave);
    }
}
TEST(IrVectorCubeAxisExecution, HarnessSeesOperandSensitivityAndInactiveDestinations) {
    for (auto op : {0x144u, 0x147u}) {
        auto in = inputs(131);
        const auto first = run_vector_op_per_lane(words(op), in);
        ASSERT_TRUE(first.error.empty()) << first.error;
        size_t inactive = 0;
        std::set<uint32_t> varied;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            if (in.active[lane])
                varied.insert(first.d[lane]);
            else {
                ++inactive;
                EXPECT_EQ(first.d[lane], in.old_d[lane]);
            }
        EXPECT_GT(varied.size(), op == 0x144 ? 2u : 12u);
        EXPECT_GE(inactive, 2u);
        if (op == 0x144)
            std::swap(in.s0, in.s2);
        else
            for (auto& value : in.s0) value ^= 0x80000000u;
        const auto swapped = run_vector_op_per_lane(words(op), in);
        ASSERT_TRUE(swapped.error.empty()) << swapped.error;
        EXPECT_NE(first.d, swapped.d);
        in.s2.pop_back();
        const auto bad = run_vector_op_per_lane(words(op), in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_FALSE(bad.error.empty());
    }
}
TEST(IrVectorCubeAxisExecution, HandWorkedFacesTiesAndSignedMajorAxes) {
    struct X {
        uint32_t x, y, z, id, ma;
    };
 // 3 dominates1/2; ties prefer Z then Y. Signed zero is not negative for face selection,
 // but doubling an all-negative-zero major coordinate must retain its sign.
    const X cases[] = {{0x40400000, 0x3f800000, 0x40000000, 0, 0x40c00000},
                       {0xc0400000, 0x3f800000, 0x40000000, 0x3f800000, 0xc0c00000},
                       {0x3f800000, 0x40400000, 0x40000000, 0x40000000, 0x40c00000},
                       {0x3f800000, 0xc0400000, 0x40000000, 0x40400000, 0xc0c00000},
                       {0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40c00000},
                       {0x3f800000, 0x40000000, 0xc0400000, 0x40a00000, 0xc0c00000},
                       {0x40000000, 0xc0000000, 0x3f800000, 0x40400000, 0xc0800000},
                       {0x40000000, 0xc0000000, 0x40000000, 0x40800000, 0x40800000},
                       {0xc0000000, 0x40000000, 0xc0000000, 0x40a00000, 0xc0800000},
                       {0, 0, 0, 0x40800000, 0},
                       {0x80000000, 0x80000000, 0x80000000, 0x40800000, 0x80000000},
                       {1, 0, 0, 0, 2},
                       {0, 0x80000001, 0, 0x40400000, 0x80000002},
                       {0x7f800000, 0xff800000, 0, 0x40400000, 0xff800000},
                       {0x7f800000, 0x7f800000, 0xff800000, 0x40a00000, 0xff800000}};
    for (const auto& def : vector_cube_axis_defs())
        for (const auto& x : cases) {
            const uint32_t want = def.opcode == 0x144 ? x.id : x.ma;
            EXPECT_EQ(eval_ternary_integer_op(def, {x.x, x.y, x.z}), Known(want));
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), x.x);
            std::fill(in.s1.begin(), in.s1.end(), x.y);
            std::fill(in.s2.begin(), in.s2.end(), x.z);
            for (uint32_t wave : {64u, 32u}) {
                const auto run = run_vector_op_per_lane(words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
}
TEST(IrVectorCubeAxisExecution, InlineAndLiteralFormsKeepAllThreeSources) {
    for (const auto& def : vector_cube_axis_defs())
        for (uint32_t mode = 1; mode <= 7; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(37 + mode), wave, mode);
}
TEST(IrVectorCubeAxisExecution, SourceAbsNegThenOutputModifiersRemainWithTheCaller) {
    // Explicit ISA arms: scaling enabled, output denorms preserved, or IEEE enabled.
    for (uint32_t rsrc1 : {0u, 0x30u << 12, 1u << 23}) {
        for (const auto& def : vector_cube_axis_defs())
            for (uint32_t abs = 0; abs < 8; ++abs)
                for (uint32_t neg = 0; neg < 8; ++neg)
                    for (uint32_t omod = 0; omod < 4; ++omod)
                        for (bool clamp : {false, true})
                            for (uint32_t wave : {64u, 32u}) {
                                auto in = inputs(43);
                                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                                    in.s0[lane] = (in.s0[lane] & 0x807fffffu) |
                                                  ((lane % 3 == 0 ? 128u : 126u) << 23);
                                    in.s1[lane] = (in.s1[lane] & 0x807fffffu) |
                                                  ((lane % 3 == 1 ? 128u : 126u) << 23);
                                    in.s2[lane] = (in.s2[lane] & 0x807fffffu) |
                                                  ((lane % 3 == 2 ? 128u : 126u) << 23);
                                }
                                // The VM deliberately refuses NMax(+0,+0)'s unspecified zero sign.
                                // Keep clamp-ID comparisons on nonzero faces; X/face0 remains covered
                                // by every unclamped source/result form and the explicit VM-limit arm.
                                if (clamp && def.opcode == 0x144)
                                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                                        if (lane % 3 == 0) {
                                            in.s0[lane] =
                                                (in.s0[lane] & 0x807fffffu) | (126u << 23);
                                            in.s1[lane] =
                                                (in.s1[lane] & 0x807fffffu) | (128u << 23);
                                        }
                                const auto run = run_vector_op_per_lane(
                                    words(def.opcode, 0, abs, neg, omod, clamp), in, wave, false,
                                    rsrc1);
                                ASSERT_TRUE(run.error.empty()) << run.error;
                                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                                    std::array<uint32_t, 3> raw = {in.s0[lane], in.s1[lane],
                                                                   in.s2[lane]};
                                    for (size_t k = 0; k < 3; ++k) {
                                        uint32_t value = raw[k];
                                        if (abs & (1u << k)) value &= 0x7fffffffu;
                                        if (neg & (1u << k)) value ^= 0x80000000u;
                                        raw[k] = value;
                                    }
                                    auto want =
                                        eval_ternary_integer_op(def, {raw[0], raw[1], raw[2]});
                                    if (!want) {
                                        ADD_FAILURE() << "unexpected unknown cube modifier result";
                                        return;
                                    }
                                    if (omod && rsrc1 == 0)
                                        want = evaluate_op(Opcode::F32Mul, 0, *want,
                                                           omod == 1   ? 0x40000000u
                                                           : omod == 2 ? 0x40800000u
                                                                       : 0x3f000000u,
                                                           0);
                                    if (!want) {
                                        ADD_FAILURE() << "unexpected unknown cube modifier result";
                                        return;
                                    }
                                    if (clamp)
                                        want = evaluate_op(Opcode::F32Clamp01, 0, *want, 0, 0);
                                    if (!want) {
                                        ADD_FAILURE() << "unexpected unknown cube modifier result";
                                        return;
                                    }
                                    ASSERT_EQ(run.d[lane],
                                              in.active[lane] ? *want : in.old_d[lane]);
                                }
                            }
    }
}
TEST(IrVectorCubeAxisExecution, ZeroClampKeepsTheReferenceVmLimitWithAnUnclampedControl) {
    auto in = inputs(47);
    std::fill(in.s0.begin(), in.s0.end(), 0x40400000u);
    std::fill(in.s1.begin(), in.s1.end(), 0x3f800000u);
    std::fill(in.s2.begin(), in.s2.end(), 0x40000000u);
    const auto limited = run_vector_op_per_lane(words(0x144, 0, 0, 0, 0, true), in);
    EXPECT_TRUE(limited.d.empty());
    EXPECT_EQ(limited.error, "unspecified float ALU result");
    const auto control = run_vector_op_per_lane(words(0x144), in);
    ASSERT_TRUE(control.error.empty()) << control.error;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        ASSERT_EQ(control.d[lane], in.active[lane] ? 0u : in.old_d[lane]);
}
TEST(IrVectorCubeAxisExecution, EffectsAndUnknownsKeepTheExistingDomain) {
    std::set<Opcode> used;
    for (const auto& def : vector_cube_axis_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> reads;
        def.build(ir, TernaryInputs(ir, reads));
        EXPECT_EQ(reads, (std::vector<uint32_t>{0, 1, 2}));
        for (const auto& inst : block.insts()) used.insert(inst.op);
        for (size_t missing = 0; missing < 3; ++missing) {
            std::array<Known, 3> raw = {0x3f800000u, 0x40000000u, 0x40400000u};
            raw[missing] = std::nullopt;
            EXPECT_FALSE(eval_ternary_integer_op(def, raw));
        }
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x7fc01234, 0x3f800000, 0x40000000}));
    }
    EXPECT_TRUE(used.contains(Opcode::F32Abs));
    EXPECT_TRUE(used.contains(Opcode::F32Lt));
    EXPECT_TRUE(used.contains(Opcode::F32Ge));
    EXPECT_EQ(find_vector_cube_axis_op(Rdna2Format::VOP3, 0x145), nullptr);
    EXPECT_EQ(find_vector_cube_axis_op(Rdna2Format::VOP3, 0x146), nullptr);
    EXPECT_EQ(find_vector_cube_axis_op(Rdna2Format::VOP2, 0x144), nullptr);
}
TEST(IrVectorCubeAxisExecution, ExportsEveryEncodingInBothWavesForCorpusComparison) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "cube-axis-witness";
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
    for (const auto& def : vector_cube_axis_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode = 0; mode <= 9; ++mode) {
                const auto encoding = mode < 8    ? words(def.opcode, mode)
                                      : mode == 8 ? words(def.opcode, 0, 7, 5, 2, true)
                                                  : words(def.opcode, 0, 3, 2, 3, false);
                const auto program = lane_program(encoding, 64, wave, true);
                const uint64_t address = 0x5288000000ull + count * 0x100;
                const fs::path stem =
                    dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                bin.write(reinterpret_cast<const char*>(program.code.data()),
                          static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const fs::path input = stem.string() + ".inputs";
                ASSERT_TRUE(write_refused_compute_inputs(input, address, &program.table,
                                                         program.config,
                                                         "synthetic-witness (no live refusal)"));
                const auto read = read_refused_shader_inputs(input);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(program.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                ++count;
            }
    EXPECT_EQ(count, 40u);
}

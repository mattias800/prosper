// Actual translated instructions, differing lanes and partial EXEC. ISA hand answers in the
// companion core test independently constrain the IR evaluator used by this execution comparison.
#include "gpu/recompiler/ir/ops/float_output_modifier.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
std::vector<uint32_t> multiply(FloatResultWidth width, uint32_t omod, bool clamp = false) {
    if (width == FloatResultWidth::F32)
        return {0xd5080001u | (clamp ? 1u << 15 : 0),
                (256 + kVsrc0) | ((256 + kVsrc1) << 9) | (omod << 27)};
    // v_mul_f16_sdwa v1,v2,v3, dst/src DWORD, UNUSED_PAD; OMOD is in the SDWA word.
    return {vop2(0x35, kVdst, 249),
            kVsrc0 | (6u << 8) | (6u << 16) | (6u << 24) | (omod << 14) | (clamp ? 1u << 13 : 0)};
}
LaneInputs inputs(FloatResultWidth width, uint32_t seed = 1) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1664525u + 1013904223u;
        in.s0[lane] = width == FloatResultWidth::F32
                          ? ((seed & 0x807fffffu) | ((10 + (seed >> 24) % 235) << 23))
                          : ((seed & 0x83ffu) | ((2 + (seed >> 24) % 27) << 10));
        in.s1[lane] = width == FloatResultWidth::F32 ? 0x3f800000 : 0x3c00;
    }
    return in;
}
void compare(FloatResultWidth width, uint32_t omod, uint8_t mode, bool ieee, uint32_t wave,
             const LaneInputs& in) {
    auto run = run_vector_op_per_lane(multiply(width, omod), in, wave, true,
                                      (uint32_t(mode) << 12) | (uint32_t(ieee) << 23));
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    const bool ignored = ieee || (mode & (width == FloatResultWidth::F32 ? 0x20 : 0x80));
    const FloatOutputModifier modifier{
        uint8_t(ignored ? 0 : omod),
        uint8_t((mode >> (width == FloatResultWidth::F32 ? 0 : 2)) & 3), width};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        Known source = in.s0[lane];
        if (width == FloatResultWidth::F16)
            source = evaluate_op(Opcode::UnpackHalf, 0, source.value_or(0), 0, 0);
        ASSERT_TRUE(source.has_value());
        auto want = eval_float_output_modifier(source, modifier);
        ASSERT_TRUE(want.has_value());
        if (width == FloatResultWidth::F16)
            want = evaluate_op(Opcode::PackHalfLo, 0, want.value_or(0), 0, 0);
        ASSERT_TRUE(want.has_value());
        EXPECT_EQ(run.d[lane], in.active[lane] ? want.value_or(0) : in.old_d[lane])
            << "lane " << lane << " wave " << wave << " mode " << uint32_t(mode) << " IEEE " << ieee
            << " scale " << omod;
    }
}
} // namespace
TEST(IrFloatOutputModifierExecution, PreservedOutputAndIeeeSuppressEveryScaleInBothPrecisions) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t scale = 1; scale <= 3; ++scale)
                for (uint32_t seed = 1; seed <= 8; ++seed) {
                    auto in = inputs(width, seed);
                    compare(width, scale, width == FloatResultWidth::F32 ? 0x30 : 0xc0, false, wave,
                            in);
                    compare(width, scale, 0, true, wave, in);
                }
}
TEST(IrFloatOutputModifierExecution, AppliedScalesMatchEvalWithDistinctLanesAndAllRoundModes) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t scale = 1; scale <= 3; ++scale)
                for (uint32_t round = 0; round <= 3; ++round)
                    for (uint32_t seed = 1; seed <= 8; ++seed)
                        compare(width, scale,
                                uint8_t(round << (width == FloatResultWidth::F32 ? 0 : 2)), false,
                                wave, inputs(width, seed));
}
TEST(IrFloatOutputModifierExecution, DenormalResultsAndNegativeZeroFlushOnlyWhenScaleIsApplied) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
        for (uint32_t wave : {32u, 64u})
            for (int kind = 0; kind < 3; ++kind) {
                auto in = inputs(width);
                const uint32_t value = kind == 2
                                           ? (width == FloatResultWidth::F32 ? 0x80000000 : 0x8000)
                                       : width == FloatResultWidth::F32 ? 0x00800000
                                                                        : 0x0400;
                for (auto& x : in.s0) x = value;
                const auto run = run_vector_op_per_lane(multiply(width, kind == 0   ? 3
                                                                        : kind == 1 ? 0
                                                                                    : 1),
                                                        in, wave, true, 0);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane],
                              in.active[lane] ? (kind == 1 ? value : 0) : in.old_d[lane]);
                // Negative arm: no OMOD leaves -0 intact, independently of output flushing.
                if (kind == 2) {
                    const auto plain =
                        run_vector_op_per_lane(multiply(width, 0), in, wave, true, 0);
                    ASSERT_TRUE(plain.error.empty()) << plain.error;
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(plain.d[lane], in.active[lane] ? value : in.old_d[lane]);
                }
            }
}
TEST(IrFloatOutputModifierExecution, ExistingAddReproductionIgnoresScaleAtDefaultFloatMode) {
    auto in = inputs(FloatResultWidth::F32);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = 0x40500000;   // 3.25
        in.s1[lane] = 0x3f800000;   // 1
    }
    for (uint32_t wave : {32u, 64u}) {
        const auto run = run_vector_op_per_lane({0xd5030001, 0x08020702}, in, wave);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane],
                      in.active[lane] ? 0x40880000 : in.old_d[lane]);   // 4.25, not 8.5
    }
}
TEST(IrFloatOutputModifierExecution, BoundaryRoundingAndOverflowExecuteInBothPrecisions) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16}) {
        auto in = inputs(width);
        const std::array<uint32_t, 8> f32 = {0x00800000, 0x00ffffff, 0x80800000, 0x80ffffff,
                                             0x7f7fffff, 0xff7fffff, 0,          0x80000000};
        const std::array<uint32_t, 8> f16 = {0x0400, 0x07ff, 0x8400, 0x87ff,
                                             0x7bff, 0xfbff, 0,      0x8000};
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            in.s0[lane] = (width == FloatResultWidth::F32 ? f32 : f16)[lane % 8];
        for (uint32_t wave : {32u, 64u})
            for (uint32_t scale = 1; scale < 4; ++scale)
                for (uint32_t round = 0; round < 4; ++round)
                    compare(width, scale,
                            uint8_t(round << (width == FloatResultWidth::F32 ? 0 : 2)), false, wave,
                            in);
    }
}
TEST(IrFloatOutputModifierExecution, ClampRemainsAfterAnAppliedOrIgnoredScale) {
    for (auto width : {FloatResultWidth::F32, FloatResultWidth::F16})
        for (uint32_t wave : {32u, 64u})
            for (bool preserve : {false, true}) {
                auto in = inputs(width);
                const uint32_t value = width == FloatResultWidth::F32 ? 0x3ec00000 : 0x3600;
                for (auto& x : in.s0) x = value;   // 0.375; *4 then clamp is 1, ignored is 0.375
                const uint32_t mode = preserve ? (width == FloatResultWidth::F32 ? 0x30 : 0xc0) : 0;
                const auto run =
                    run_vector_op_per_lane(multiply(width, 2, true), in, wave, true, mode << 12);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                const uint32_t expected = preserve                         ? value
                                          : width == FloatResultWidth::F32 ? 0x3f800000
                                                                           : 0x3c00;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? expected : in.old_d[lane]);
            }
}
TEST(IrFloatOutputModifierExecution, Vop1Vop2AndNativeHalfVop3UseTheSameModeContract) {
    for (int form = 0; form < 3; ++form)
        for (uint32_t wave : {32u, 64u})
            for (bool preserve : {false, true}) {
                const bool half = form == 2;
                auto in = inputs(half ? FloatResultWidth::F16 : FloatResultWidth::F32);
                std::vector<uint32_t> op;
                uint32_t plain, scaled;
                if (form == 0) {
                    op = {vop1(0x21, kVdst, 249), kVsrc0 | (6u << 8) | (6u << 16) | (1u << 14)};
                    for (auto& x : in.s0) x = 0x40500000;   // trunc(3.25) = 3
                    plain = 0x40400000;
                    scaled = 0x40c00000;
                } else if (form == 1) {
                    op = {vop2(0x03, kVdst, 249),
                          kVsrc0 | (6u << 8) | (6u << 16) | (6u << 24) | (1u << 14)};
                    for (auto& x : in.s0) x = 0x40500000;   // 3.25+1 = 4.25
                    plain = 0x40880000;
                    scaled = 0x41080000;
                } else {
                    op = {0xd7510001, (256 + kVsrc0) | ((256 + kVsrc1) << 9) |
                                          ((256 + kVsrc2) << 18) | (1u << 27)};
                    for (auto& x : in.s0) x = 0x4200;   // min(3,4,5) = 3
                    for (auto& x : in.s1) x = 0x4400;
                    in.s2.assign(in.lanes(), 0x4500);
                    plain = 0x4200;
                    scaled = 0x4600;
                }
                const uint32_t mode = preserve ? (half ? 0xc0 : 0x30) : 0;
                const auto run = run_vector_op_per_lane(op, in, wave, true, mode << 12);
                ASSERT_TRUE(run.error.empty()) << run.error << " form " << form;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    uint32_t expected = preserve ? plain : scaled;
                    if (half) expected |= in.old_d[lane] & 0xffff0000;
                    EXPECT_EQ(run.d[lane], in.active[lane] ? expected : in.old_d[lane]);
                }
            }
}
TEST(IrFloatOutputModifierExecution, HalfScalingOverflowAndInputInfinityExecuteWithEveryRoundMode) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t round = 0; round < 4; ++round) {
            auto in = inputs(FloatResultWidth::F16);
            const std::array<uint32_t, 4> edge{0x7bff, 0xfbff, 0x7c00, 0xfc00};
            for (size_t lane = 0; lane < in.lanes(); ++lane) in.s0[lane] = edge[lane % 4];
            compare(FloatResultWidth::F16, 1, uint8_t(round << 2), false, wave, in);
        }
}
TEST(IrFloatOutputModifierExecution, NativeVertexCompilerConsumesItsOwnModeCarrier) {
    // Both original operands are explicit; vertex-system values cannot satisfy this sink.
    const auto compile = [](uint32_t omod, GraphicsFloatMode mode) {
        const uint32_t code[] = {0x7e0002ffu, 0x40500000u, 0x7e0202ffu,
                                 0x3f800000u, 0xd5030002u, 256u | (257u << 9) | (omod << 27),
                                 0xf80008cfu, 0x02020202u, 0xbf810000u};
        return recompile_vertex(code, std::size(code), nullptr, nullptr, false, 0, {}, {}, mode);
    };
    const auto plain = compile(0, {});
    ASSERT_FALSE(plain.empty());
    EXPECT_TRUE(compile(1, {}).empty());
    const auto scaled = compile(1, {true, 0, false});
    ASSERT_FALSE(scaled.empty());
    EXPECT_NE(scaled, plain);
    EXPECT_EQ(compile(1, {true, 0x30, false}), plain);
    EXPECT_EQ(compile(1, {true, 0, true}), plain);
}
TEST(IrFloatOutputModifierExecution, NggCompilerExportsTheModeSelectedPerLaneValue) {
    // Dynamic launch v0 reaches the actual raw POS0 record. The physical compute shell's
    // default mode must not override the producing GS policy.
    for (uint32_t lanes : {32u, 64u}) {
        const auto compile = [lanes](uint32_t omod, GraphicsFloatMode mode,
                                     NggExportRecordLayout& layout, std::string& refusal) {
            const uint32_t code[] = {
                0xbefe04c1u, 0x7e100280u, 0xd5030009u, 256u | (242u << 9) | (omod << 27),
                0xb07c3005u, 0xbf900009u, 0xf8000941u, 0x00000008u,
                0xf80000cfu, 0x09090909u, 0xbf810000u};
            NggSubgroupShellConfig config;
            config.wave_lanes = lanes;
            config.float_mode = mode;
            return recompile_ngg_subgroup(code, std::size(code), nullptr, config, &layout,
                                          {RecompileDiagnosticStage::Vertex, 0x53070000u + lanes},
                                          &refusal);
        };
        NggExportRecordLayout layout;
        std::string refusal;
        ASSERT_FALSE(compile(0, {}, layout, refusal).empty()) << refusal;
        EXPECT_TRUE(compile(1, {}, layout, refusal).empty());
        EXPECT_NE(refusal.find("float-omod-launch-mode-unavailable"), std::string::npos);
        for (const auto mode :
             {GraphicsFloatMode{true, 0, false}, GraphicsFloatMode{true, 0x30, false},
              GraphicsFloatMode{true, 0, true}}) {
            const auto module = compile(1, mode, layout, refusal);
            ASSERT_FALSE(module.empty()) << refusal;
            std::vector<uint32_t> launch(size_t(lanes) * kNggLaunchWordsPerLane);
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                launch[size_t(lane) * kNggLaunchWordsPerLane] =
                    lane & 1 ? 0x40500000u : 0x3fa00000u;   // 3.25 or 1.25
                launch[size_t(lane) * kNggLaunchWordsPerLane + kNggLaunchS3Word] =
                    0x10000000u | lanes;
            }
            bpermute_oracle::Interpreter oracle(module);
            oracle.use_ngg_shell_storage();
            const auto output = oracle.run_buffers(
                lanes,
                {{kNggShellLaunchBinding, launch},
                 {kNggShellExportBinding, std::vector<uint32_t>(layout.block_words(1, lanes))}},
                kNggShellExportBinding);
            ASSERT_TRUE(oracle.error.empty()) << oracle.error;
            ASSERT_EQ(output.size(), layout.block_words(1, lanes));
            const bool scaled = !mode.ieee_mode && !(mode.value & 0x20);
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                const uint32_t want = lane & 1 ? (scaled ? 0x41080000u : 0x40880000u)
                                               : (scaled ? 0x40900000u : 0x40100000u);
                const uint32_t record = layout.record_offset(1, 0, lane, lanes);
                ASSERT_EQ(output[record], kNggFlagPrim | kNggFlagPos0) << lane;
                EXPECT_EQ(output[record + kNggRecordPrimWord], 0u) << lane;
                for (uint32_t channel = 0; channel < 4; ++channel)
                    EXPECT_EQ(output[record + kNggRecordPos0Word + channel], want) << lane;
            }
        }
    }
}
TEST(IrFloatOutputModifierExecution, ExportsEveryEncodingAndLaunchModeForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input, never mutated here.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "omod-mode-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    std::ofstream manifest(root / "entries.tsv");
    manifest << "stem\tform\tprecision\twave\trsrc1\tomod\tclamp\n";
    size_t count = 0;
    for (uint32_t form = 0; form < 5; ++form)
        for (uint32_t wave : {32u, 64u})
            for (uint32_t scale = 0; scale < 4; ++scale)
                for (bool clamp : {false, true})
                    for (uint32_t rsrc1 :
                         {0u, 0x1000u, 0x2000u, 0x3000u, 0x4000u, 0x8000u, 0xc000u, 0x10000u,
                          0x20000u, 0x40000u, 0x80000u, 0xf0000u, 0x800000u, 0x4000000u}) {
                        const bool half = form == 1 || form == 4;
                        auto words = multiply(half ? FloatResultWidth::F16 : FloatResultWidth::F32,
                                              scale, clamp);
                        if (form == 2)
                            words = {vop1(0x21, kVdst, 249), kVsrc0 | (6u << 8) | (6u << 16) |
                                                                 (scale << 14) |
                                                                 (clamp ? 1u << 13 : 0)};
                        if (form == 3)
                            words = {vop2(0x03, kVdst, 249), kVsrc0 | (6u << 8) | (6u << 16) |
                                                                 (6u << 24) | (scale << 14) |
                                                                 (clamp ? 1u << 13 : 0)};
                        if (form == 4)
                            words = {0xd7510001u | (clamp ? 1u << 15 : 0),
                                     (256 + kVsrc0) | ((256 + kVsrc1) << 9) |
                                         ((256 + kVsrc2) << 18) | (scale << 27)};
                        auto program = lane_program(words, 64, wave, form == 4);
                        program.config.compute_pgm_rsrc1 = rsrc1;
                        const uint64_t address = 0x5307000000ull + count * 0x100;
                        const auto stem =
                            "cs_" + std::to_string(address) + "_" + std::to_string(count);
                        std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                        bin.write(reinterpret_cast<const char*>(program.code.data()),
                                  std::streamsize(program.code.size() * sizeof(uint32_t)));
                        bin.close();
                        ASSERT_TRUE(bin.good());
                        ASSERT_TRUE(write_refused_compute_inputs(
                            dump / (stem + ".inputs"), address, &program.table, program.config,
                            "synthetic-omod-witness (no live refusal)"));
                        manifest << stem << '\t' << form << '\t' << (half ? 16 : 32) << '\t' << wave
                                 << '\t' << rsrc1 << '\t' << scale << '\t' << clamp << '\n';
                        ++count;
                    }
    manifest.close();
    EXPECT_TRUE(manifest.good());
    EXPECT_EQ(count, 1120u);
}

// E64 F32-to-integer conversions: actual emitted SPIR-V, per-lane EXEC and independent ISA rails.
#include "gpu/recompiler/ir/float_integer_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_float_integer_e64.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
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
namespace cases = prosper::test::ir::float_integer_e64;
namespace {
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
void expect_lanes(const ConversionDef& def, const LaneInputs& in, const LaneRun& run,
                  cases::Form f = {}, uint32_t modifiers = 0) {
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want =
            eval_conversion_op(def, cases::modified(cases::source(f, in, lane), modifiers));
        if (!want) {
            ADD_FAILURE() << "opcode " << def.opcode << " lane " << lane;
            return;
        }
        ASSERT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]) << "lane " << lane;
    }
}
void export_witness(const std::filesystem::path& root, std::ofstream& manifest, size_t index,
                    const ConversionDef& def, uint32_t wave, uint32_t form, uint32_t modifiers,
                    uint32_t omod, uint32_t encoding) {
    const auto f = cases::form(form);
    const auto words = encoding == 2 ? cases::sdwa(def.opcode, modifiers)
                                     : cases::words(def.opcode, f, modifiers, omod, encoding == 0);
    const auto program = lane_program(words, 64, wave);
    const uint64_t address = 0x5312000000ull + index * 0x100;
    const auto stem = root / "refused_shaders_20000101-000000Z_synthetic" /
                      ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    std::ofstream bin(stem.string() + ".bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(program.code.data()),
              static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
    bin.close();
    ASSERT_TRUE(bin.good());
    const std::filesystem::path file = stem.string() + ".inputs";
    ASSERT_TRUE(write_refused_compute_inputs(file, address, &program.table, program.config,
                                             "synthetic-witness (no live refusal)"));
    const auto read = read_refused_shader_inputs(file);
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = replay_refused_shader_inputs(program.code, read);
    ASSERT_TRUE(replay.produced) << replay.reject;
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    auto in = cases::inputs(11);
    if (encoding == 2)
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            in.s0[lane] = (lane & 1u) ? 0xbfe00000 : 0x3fe00000;
    expect_lanes(def, in, execute(replay.spirv, in), encoding == 2 ? cases::Form{} : f, modifiers);
    std::ofstream module(root / "modules" / (std::to_string(index) + ".spv"), std::ios::binary);
    module.write(reinterpret_cast<const char*>(replay.spirv.data()),
                 static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
    module.close();
    ASSERT_TRUE(module.good());
    manifest << stem.filename().string() << '\t' << def.opcode << '\t' << wave << '\t' << form
             << '\t' << modifiers << '\t' << omod << '\t' << encoding << '\n';
}
} // namespace
TEST(IrVectorFloatIntegerE64Execution, EveryDefinitionMatchesEvalAcrossRandomAndBoundaryLanes) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed = 1; seed <= 8; ++seed)
                for (uint32_t modifiers = 0; modifiers < 4; ++modifiers) {
                    const auto in = cases::inputs(seed * 5312);
                    const auto run =
                        run_vector_op_per_lane(cases::words(def.opcode, {}, modifiers), in, wave);
                    expect_lanes(def, in, run, {}, modifiers);
                }
}
TEST(IrVectorFloatIntegerE64Execution, EachSourceEncodingAndOmodCombinationMatchesEveryLane) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t form = 0; form < 8; ++form)
                for (uint32_t modifiers = 0; modifiers < 4; ++modifiers)
                    for (uint32_t omod = 0; omod < 4; ++omod) {
                        SCOPED_TRACE(def.opcode);
                        SCOPED_TRACE(form);
                        SCOPED_TRACE(modifiers);
                        SCOPED_TRACE(omod);
                        const auto in = cases::inputs(11);
                        const auto f = cases::form(form);
                        const auto run = run_vector_op_per_lane(
                            cases::words(def.opcode, f, modifiers, omod), in, wave);
                        expect_lanes(def, in, run, f, modifiers);
                    }
}
TEST(IrVectorFloatIntegerE64Execution, DestinationSourceAliasReadsEachOldLaneBeforeWriting) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t modifiers = 0; modifiers < 4; ++modifiers)
                for (uint32_t omod = 0; omod < 4; ++omod) {
                    auto in = cases::inputs(11);
                    in.old_d = in.s0;
                    cases::Form f;
                    f.source = 257;
                    const auto run = run_vector_op_per_lane(
                        cases::words(def.opcode, f, modifiers, omod), in, wave);
                    expect_lanes(def, in, run, f, modifiers);
                }
}
TEST(IrVectorFloatIntegerE64Execution, IndependentIsaExamplesPinTruncationAndSaturation) {
    struct Example {
        uint32_t input, u32, i32;
    };
    // AMD 70648 12.8 op7/8: truncate toward zero, saturate infinities/overflow, NaN -> 0.
    // 0x4effffff = 2^31-128; 0x4f7fffff = 2^32-256. The adjacent powers reach the rails.
    const Example examples[] = {
        {0, 0, 0},
        {0x80000000, 0, 0},
        {1, 0, 0},
        {0x807fffff, 0, 0},
        {0x3f7fffff, 0, 0},
        {0xbf7fffff, 0, 0},
        {0x3fe00000, 1, 1},
        {0xbfe00000, 0, 0xffffffff},
        {0xc0780000, 0, 0xfffffffd},
        {0x4effffff, 0x7fffff80, 0x7fffff80},
        {0x4f000000, 0x80000000, 0x7fffffff},
        {0xcf000000, 0, 0x80000000},
        {0xcf000001, 0, 0x80000000},
        {0x4f7fffff, 0xffffff00, 0x7fffffff},
        {0x4f800000, 0xffffffff, 0x7fffffff},
        {0x7f800000, 0xffffffff, 0x7fffffff},
        {0xff800000, 0, 0x80000000},
        {0x7fc00001, 0, 0},
        {0xff800001, 0, 0},
        {0x7f7fffff, 0xffffffff, 0x7fffffff},
    };
    for (const auto& def : vector_float_integer_e64_defs())
        for (const auto& x : examples) {
            SCOPED_TRACE(x.input);
            const auto want = def.opcode == 0x187 ? x.u32 : x.i32;
            EXPECT_EQ(eval_conversion_op(def, x.input), Known(want));
            auto in = cases::inputs(11);
            std::fill(in.s0.begin(), in.s0.end(), x.input);
            for (uint32_t wave : {32u, 64u}) {
                const auto run = run_vector_op_per_lane(cases::words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
}
TEST(IrVectorFloatIntegerE64Execution, IndependentAbsThenNegExamplesPinSignOrderAndNanZero) {
    struct Example {
        uint32_t input, modifiers, u32, i32;
    };
    const Example examples[] = {
        {0xbfe00000, 1, 1, 1},
        {0xbfe00000, 2, 1, 1},
        {0xbfe00000, 3, 0, 0xffffffff},
        {0x3fe00000, 2, 0, 0xffffffff},
        {0xff800000, 1, 0xffffffff, 0x7fffffff},
        {0xff800000, 3, 0, 0x80000000},
        {0xff800001, 3, 0, 0},
    };
    for (const auto& def : vector_float_integer_e64_defs())
        for (const auto& x : examples)
            for (uint32_t omod = 0; omod < 4; ++omod) {
                auto in = cases::inputs(11);
                std::fill(in.s0.begin(), in.s0.end(), x.input);
                const auto want = def.opcode == 0x187 ? x.u32 : x.i32;
                EXPECT_EQ(eval_conversion_op(def, cases::modified(x.input, x.modifiers)),
                          Known(want));
                const auto run =
                    run_vector_op_per_lane(cases::words(def.opcode, {}, x.modifiers, omod), in);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
}
TEST(IrVectorFloatIntegerE64Execution, HarnessObservesDifferingLanesAndMaskedOldDestinations) {
    auto in = cases::inputs(11);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        in.s0[lane] = 0x40000000u + (uint32_t(lane) << 21);
    const auto run = run_vector_op_per_lane(cases::words(0x187), in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t masked = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane])
            active.insert(run.d[lane]);
        else {
            ++masked;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 8u);
    EXPECT_GT(masked, 4u);
}
TEST(IrVectorFloatIntegerE64Execution, MissingLaunchModeStillIgnoresIntegerOutputOmod) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode : {0u, 0x30000u, 1u << 23, UINT32_MAX}) {
                const auto in = cases::inputs(11);
                auto program = lane_program(cases::words(def.opcode, {}, 3, 2), 64, wave);
                program.config.compute_pgm_rsrc1 = mode;
                const auto address = lane_harness_address();
                const auto module =
                    recompile_compute(program.code.data(), program.code.size(), &program.table,
                                      program.config, {RecompileDiagnosticStage::Compute, address});
                ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
                expect_lanes(def, in, execute(module, in), {}, 3);
            }
}
TEST(IrVectorFloatIntegerE64Execution, OldE32AndSdwaControlsKeepTheSameResults) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            const auto in = cases::inputs(11);
            for (uint32_t f = 0; f < 8; ++f) {
                const auto current =
                    run_vector_op_per_lane(cases::words(def.opcode, cases::form(f)), in, wave);
                const auto old = run_vector_op_per_lane(
                    cases::words(def.opcode, cases::form(f), 0, 0, false), in, wave);
                ASSERT_TRUE(old.error.empty()) << old.error;
                ASSERT_TRUE(current.error.empty()) << current.error;
                EXPECT_EQ(old.d, current.d);
            }
            // Generic old SDWA FAbs deliberately leaves NaN payloads unspecified in the oracle.
            auto finite = in;
            for (size_t lane = 0; lane < finite.lanes(); ++lane)
                finite.s0[lane] = (lane & 1u) ? 0xbfe00000 : 0x3fe00000;
            for (uint32_t modifiers = 0; modifiers < 4; ++modifiers) {
                const auto old =
                    run_vector_op_per_lane(cases::sdwa(def.opcode, modifiers), finite, wave);
                const auto current =
                    run_vector_op_per_lane(cases::words(def.opcode, {}, modifiers), finite, wave);
                ASSERT_TRUE(old.error.empty()) << old.error;
                ASSERT_TRUE(current.error.empty()) << current.error;
                EXPECT_EQ(old.d, current.d);
            }
        }
}
TEST(IrVectorFloatIntegerE64Execution, ClampAndNoncanonicalUnusedFieldsRemainRefused) {
    for (const auto& def : vector_float_integer_e64_defs()) {
        for (uint32_t word = 0; word < 2; ++word)
            for (uint32_t bit = 9; bit < (word ? 32u : 16u); ++bit) {
                if (word && bit >= 27 && bit <= 29) continue;
                auto bad = cases::words(def.opcode);
                bad[word] |= 1u << bit;
                ASSERT_FALSE(run_vector_op_per_lane(bad, cases::inputs(11)).error.empty());
            }
        ASSERT_TRUE(
            run_vector_op_per_lane(cases::words(def.opcode), cases::inputs(11)).error.empty());
        EXPECT_FALSE(eval_conversion_op(def, std::nullopt));
    }
}
TEST(IrVectorFloatIntegerE64Execution, HardScalarSourcesCannotBecomeFreshMasks) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            auto f = cases::form(7);
            f.initialize = false;
            auto bad = cases::words(def.opcode, f);
            bad.insert(bad.end(),
                       {0xbefe04c1, 0x7d8a0280});   // Restore EXEC, then consume v1 in CMPX.
            const auto refused = run_vector_op_per_lane(bad, cases::inputs(11), wave);
            EXPECT_NE(refused.error.find("fabricated-fresh-mask-source"), std::string::npos)
                << refused.error;
            f.initialize = true;
            auto good = cases::words(def.opcode, f);
            good.insert(good.end(), {0xbefe04c1, 0x7d8a0280});
            ASSERT_TRUE(run_vector_op_per_lane(good, cases::inputs(11), wave).error.empty());
        }
}
TEST(IrVectorFloatIntegerE64Execution, OneArmScalarSourcesRetainTheirMaskObligation) {
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> code;
                if (initialize) code.insert(code.end(), {0xbeb203ff, 0x3fe00000});
                code.insert(code.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x40000000});
                auto f = cases::form(7);
                f.initialize = false;
                const auto op = cases::words(def.opcode, f);
                code.insert(code.end(), op.begin(), op.end());
                code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
                auto program = lane_program(code, 64, wave);
                program.config.user_sgprs.resize(8);
                const auto address = lane_harness_address();
                const auto module =
                    recompile_compute(program.code.data(), program.code.size(), &program.table,
                                      program.config, {RecompileDiagnosticStage::Compute, address});
                const auto reason = last_terminal_reject_reason(address);
                if (initialize)
                    ASSERT_FALSE(module.empty()) << reason;
                else {
                    EXPECT_TRUE(module.empty());
                    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos)
                        << reason;
                }
            }
}
TEST(IrVectorFloatIntegerE64Execution, EncodingsMatchLlvmGfx1030AndUnsupportedNeighborsRefuse) {
    EXPECT_EQ(cases::words(0x187), (std::vector<uint32_t>{0xd5870001, 0x102}));
    EXPECT_EQ(cases::words(0x188, {}, 3, 2), (std::vector<uint32_t>{0xd5880101, 0x30000102}));
    for (uint32_t op : {0x181u, 0x185u, 0x186u, 0x18cu})
        EXPECT_FALSE(run_vector_op_per_lane(cases::words(op), cases::inputs(11)).error.empty());
}
TEST(IrVectorFloatIntegerE64Execution, ExportsAndExecutesEveryNewFormAndOldEncodingControl) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this test process never mutates its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured
                              ? fs::path(configured)
                              : prosper_test::test_scratch_dir() / "float-integer-e64-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "export requires a new witness directory";
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    ASSERT_TRUE(fs::create_directory(root / "refused_shaders_20000101-000000Z_synthetic"));
    std::ofstream manifest(root / "manifest.tsv");
    size_t count = 0;
    for (const auto& def : vector_float_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            for (uint32_t form = 0; form < 8; ++form) {
                for (uint32_t modifiers = 0; modifiers < 4; ++modifiers)
                    for (uint32_t omod = 0; omod < 4; ++omod)
                        export_witness(root, manifest, count++, def, wave, form, modifiers, omod,
                                       0);
                export_witness(root, manifest, count++, def, wave, form, 0, 0, 1);
            }
            for (uint32_t modifiers = 0; modifiers < 4; ++modifiers)
                export_witness(root, manifest, count++, def, wave, 0, modifiers, 0, 2);
        }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 560u);   // 512 newly admitted modules and 48 existing encoding controls.
}

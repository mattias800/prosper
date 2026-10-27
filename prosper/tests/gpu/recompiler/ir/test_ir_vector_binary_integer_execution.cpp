// Newly admitted canonical e64 word-data forms execute real SPIR-V per lane. Shared evaluation
// covers varied inputs; literal AMD70648 §12.7 examples independently check arithmetic and wrapping.
#include "gpu/recompiler/ir/ops/vector_binary_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
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
namespace {
struct Form {
    uint32_t s0 = 258, s1 = 259, literal = 0, scalar = 0;
    bool set_scalar = false;
};
Form form(uint32_t mode) {
    Form f;
    if (mode == 1) f.s0 = 257;
    if (mode == 2) f.s1 = 257;
    if (mode == 3) f.s0 = 128;
    if (mode == 4) f.s1 = 193;
    if (mode == 5) f.s0 = 192;
    if (mode == 6) {
        f.s0 = 255;
        f.literal = 0xdeadbeef;
    }
    if (mode == 7) {
        f.s1 = 255;
        f.literal = 0x80000021;
    }
    if (mode == 8) {
        f.s0 = f.s1 = 255;
        f.literal = 0xffffffff;
    }
    if (mode == 9) {
        f.s0 = 259;
        f.s1 = 258;
    }
    if (mode >= 10 && mode <= 12) {
        f.set_scalar = true;
        f.scalar = mode == 10 ? 0x00ff00ff : mode == 11 ? 33 : 0x80000001;
        if (mode != 11) f.s0 = 50;
        if (mode != 10) f.s1 = 50;
    }
    if (mode == 13) f.s1 = 258;
    if (mode == 14) f.s0 = 242;   // Full-dword inline 1.0 has raw bits 0x3f800000.
    return f;
}
std::vector<uint32_t> words(uint32_t opcode, Form f = {}) {
    std::vector<uint32_t> out;
    if (f.set_scalar) out.insert(out.end(), {0xbeb203ff, f.scalar});
    out.insert(out.end(), {0xd4000001u | (opcode << 16), f.s0 | (f.s1 << 9)});
    if (f.s0 == 255 || f.s1 == 255) out.push_back(f.literal);
    return out;
}
uint32_t source(uint32_t field, const Form& f, const LaneInputs& in, size_t lane) {
    if (field == 257) return in.old_d[lane];
    if (field == 258) return in.s0[lane];
    if (field == 259) return in.s1[lane];
    if (field >= 128 && field <= 192) return field - 128;
    if (field >= 193 && field <= 208) return 192u - field;
    if (field == 255) return f.literal;
    if (field == 242) return 0x3f800000;
    if (field == 50 && f.set_scalar) return f.scalar;
    ADD_FAILURE() << "unmodelled test operand " << field;
    return 0;
}
void compare(const VectorOpDef& def, const LaneInputs& in, uint32_t wave, Form f = {}) {
    const auto run = run_vector_op_per_lane(words(def.opcode, f), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error << " opcode " << def.opcode;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_vector_op(def, source(f.s0, f, in, lane), source(f.s1, f, in, lane));
        if (!want) {
            ADD_FAILURE() << "arithmetic did not produce a known value";
            return;
        }
        ASSERT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "opcode " << def.opcode << " lane " << lane << " wave " << wave;
    }
}
}   // namespace
TEST(IrVectorBinaryIntegerExecution, EveryDefinitionMatchesDifferentLanesInBothWaves) {
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 5; ++seed) compare(def, lane_inputs(seed * 5305), wave);
}
TEST(IrVectorBinaryIntegerExecution, HandWorkedIsaCasesCheckEvalAndExecutionIndependently) {
    struct Example {
        uint32_t op, a, b, result;
    };
    // U24 masks each source; (2^24-1)^2 mod 2^32 = 0xfe000001. Shifts mask to five bits.
    const Example examples[] = {
        {0x10b, 0xff000003, 0xff000005, 15},
        {0x10b, 0xffffffff, 0xffffffff, 0xfe000001},
        {0x111, 0x80000000, 0x7fffffff, 0x80000000},
        {0x111, 0xffffffff, 1, 0xffffffff},
        {0x112, 0x80000000, 0x7fffffff, 0x7fffffff},
        {0x112, 0xffffffff, 1, 1},
        {0x113, 0x80000000, 0x7fffffff, 0x7fffffff},
        {0x113, 0xffffffff, 1, 1},
        {0x114, 0x80000000, 0x7fffffff, 0x80000000},
        {0x114, 0xffffffff, 1, 0xffffffff},
        {0x116, 32, 0x80000001, 0x80000001},
        {0x116, 0xffffffff, 0x80000001, 1},
        {0x118, 1, 0x80000000, 0xc0000000},
        {0x118, 0xffffffff, 0x80000001, 0xffffffff},
        {0x11a, 33, 0x80000001, 2},
        {0x11a, 0xffffffff, 3, 0x80000000},
        {0x11b, 0x00ff00ff, 0x0f0f0f0f, 0x000f000f},
        {0x11b, 0xffffffff, 0x80000000, 0x80000000},
        {0x11c, 0x00ff00ff, 0x0f0f0f0f, 0x0fff0fff},
        {0x11c, 0xffffffff, 0x80000000, 0xffffffff},
        {0x11d, 0x00ff00ff, 0x0f0f0f0f, 0x0ff00ff0},
        {0x11d, 0xffffffff, 0x80000000, 0x7fffffff},
        {0x11e, 0x00ff00ff, 0x0f0f0f0f, 0xf00ff00f},
        {0x11e, 0xffffffff, 0x80000000, 0x80000000},
        {0x125, 0xffffffff, 1, 0},
        {0x125, 0x80000000, 0x80000000, 0},
        {0x126, 0, 1, 0xffffffff},
        {0x126, 0x80000000, 0x7fffffff, 1},
        {0x127, 0, 1, 1},
        {0x127, 0x80000000, 0x7fffffff, 0xffffffff},
    };
    for (const auto& x : examples) {
        const auto* def = find_vector_binary_integer_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_vector_op(*def, x.a, x.b), std::optional<uint32_t>(x.result));
        auto in = lane_inputs(5305);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        for (uint32_t wave : {64u, 32u}) {
            const auto run = run_vector_op_per_lane(words(x.op), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                ASSERT_EQ(run.d[lane], in.active[lane] ? x.result : in.old_d[lane]);
        }
    }
}
TEST(IrVectorBinaryIntegerExecution, CanonicalSourceFormsAndOldDestinationAliasesPreserveWiring) {
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode = 0; mode < 15; ++mode)
                compare(def, lane_inputs(5311), wave, form(mode));
}
TEST(IrVectorBinaryIntegerExecution, ExistingE32ControlsStillExecuteTheSameWordDefinitions) {
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u}) {
            const auto in = lane_inputs(5333);
            const auto old = run_vector_op_per_lane({vop2(def.opcode - 0x100)}, in, wave);
            const auto current = run_vector_op_per_lane(words(def.opcode), in, wave);
            ASSERT_TRUE(old.error.empty()) << old.error;
            ASSERT_TRUE(current.error.empty()) << current.error;
            EXPECT_EQ(old.d, current.d);
        }
}
TEST(IrVectorBinaryIntegerExecution, InactiveLanesRetainIndependentOldDestinationWords) {
    const auto in = lane_inputs(5363);
    std::set<uint32_t> active, inactive;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        (in.active[lane] ? active : inactive).insert(in.old_d[lane]);
    EXPECT_GT(active.size(), 1u);
    EXPECT_GT(inactive.size(), 1u);
    for (const auto& def : vector_binary_integer_defs()) compare(def, in, 64);
}
TEST(IrVectorBinaryIntegerExecution, HardScalarSourcesCannotBecomeMasksAfterRestoringExec) {
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t slot : {0u, 1u}) {
                Form f;
                (slot == 0 ? f.s0 : f.s1) = 50;
                auto bad = words(def.opcode, f);
                bad.insert(bad.end(), {0xbefe04c1, 0x7d8a0280});   // Full EXEC; compare result v1.
                bad.insert(bad.begin(), 0xbeb20300u);   // s50 = unknown possible scratch offset s0
                const auto refused = run_vector_op_per_lane(bad, lane_inputs(7), wave);
                EXPECT_NE(refused.error.find("fabricated-fresh-mask-source"), std::string::npos)
                    << refused.error << " opcode " << def.opcode;
                f.set_scalar = true;
                f.scalar = 0x80000001;
                auto good = words(def.opcode, f);
                good.insert(good.end(), {0xbefe04c1, 0x7d8a0280});
                EXPECT_TRUE(run_vector_op_per_lane(good, lane_inputs(7), wave).error.empty());
            }
}
TEST(IrVectorBinaryIntegerExecution, OneArmWordsKeepTheirHardObligationWithBothEdgeControls) {
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> code;
                if (initialize) code.insert(code.end(), {0xbeb203ff, 0x80000001});
                code.insert(code.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x00000021});
                Form f;
                f.s0 = 50;
                const auto op = words(def.opcode, f);
                code.insert(code.end(), op.begin(), op.end());
                code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
                auto program = lane_program(code, 64, wave);
                seed_lane_hard_scalar(program, 8);   // Genuine dynamic branch selector s4.
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
TEST(IrVectorBinaryIntegerExecution, IntegerModifiersAndNoncanonicalFieldsRemainRefused) {
    for (const auto& def : vector_binary_integer_defs()) {
        for (uint32_t field : {1u << 8, 1u << 11, 1u << 15}) {
            auto bad = words(def.opcode);
            bad[0] |= field;
            EXPECT_FALSE(run_vector_op_per_lane(bad, lane_inputs(7)).error.empty());
        }
        for (uint32_t extra : {50u << 18, 510u << 18, 255u << 18, 1u << 27, 1u << 29}) {
            auto bad = words(def.opcode);
            bad[1] |= extra;
            if (extra == (255u << 18)) bad.push_back(0x00ff00ff);
            EXPECT_FALSE(run_vector_op_per_lane(bad, lane_inputs(7)).error.empty());
        }
        EXPECT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(7)).error.empty());
    }
}
TEST(IrVectorBinaryIntegerExecution, UnimplementedMathAndUnsupportedSpecialWordsStayRefused) {
    for (uint32_t op : {0x109u, 0x10au, 0x10cu})
        EXPECT_FALSE(run_vector_op_per_lane(words(op), lane_inputs(7)).error.empty());
    Form invalid;
    invalid.s0 = 251;   // No numeric service exists for this special operand.
    for (const auto& def : vector_binary_integer_defs()) {
        EXPECT_FALSE(
            run_vector_op_per_lane(words(def.opcode, invalid), lane_inputs(7)).error.empty());
        EXPECT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(7)).error.empty());
    }
}
TEST(IrVectorBinaryIntegerExecution, EncodingsMatchIndependentLlvmChecks) {
    EXPECT_EQ(words(0x10b), (std::vector<uint32_t>{0xd50b0001, 0x00020702}));
    EXPECT_EQ(words(0x125), (std::vector<uint32_t>{0xd5250001, 0x00020702}));
}
TEST(IrVectorBinaryIntegerExecution, ExportsEveryCanonicalFormAndOldEncodingControls) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the test process never changes its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "binary-integer-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    std::ofstream manifest(root / "manifest.tsv");
    size_t count = 0;
    for (const auto& def : vector_binary_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode = 0; mode < 16; ++mode) {
                const auto op = mode == 15 ? std::vector<uint32_t>{vop2(def.opcode - 0x100)}
                                           : words(def.opcode, form(mode));
                const auto program = lane_program(op, 64, wave);
                const uint64_t address = 0x5305000000ull + count * 0x100;
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
                manifest << stem.filename().string() << '\t' << def.opcode << '\t' << wave << '\t'
                         << mode << '\n';
                ++count;
            }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 480u);
}

// Four canonical e64 integer aliases: real SPIR-V per lane, independent ISA examples and
// source-to-mask negatives. Architectural unary decoding does not admit the reserved MOV form.
#include "gpu/recompiler/ir/ops/vector_unary_integer_e64.hpp"
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
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
struct Form {
    uint32_t source = 258, literal = 0, scalar = 0;
    bool initialize = false;
};
Form form(uint32_t mode) {
    Form f;
    if (mode == 1) f.source = 257;
    if (mode == 2) f.source = 128;
    if (mode == 3) f.source = 192;
    if (mode == 4) f.source = 193;
    if (mode == 5) {
        f.source = 255;
        f.literal = 0x80000001;
    }
    if (mode == 6) {
        f.source = 50;
        f.scalar = 0x81234567;
        f.initialize = true;
    }
    if (mode == 7) f.source = 242;
    return f;
}
std::vector<uint32_t> words(uint32_t opcode, Form f = {}) {
    std::vector<uint32_t> out;
    if (f.initialize) out.insert(out.end(), {0xbeb203ff, f.scalar});
    out.insert(out.end(), {0xd4000001u | (opcode << 16), f.source});
    if (f.source == 255) out.push_back(f.literal);
    return out;
}
uint32_t source(Form f, const LaneInputs& in, size_t lane) {
    if (f.source == 258) return in.s0[lane];
    if (f.source == 257) return in.old_d[lane];
    if (f.source >= 128 && f.source <= 192) return f.source - 128;
    if (f.source >= 193 && f.source <= 208) return 192u - f.source;
    if (f.source == 255) return f.literal;
    if (f.source == 242) return 0x3f800000;
    if (f.source == 50 && f.initialize) return f.scalar;
    ADD_FAILURE() << "unmodelled test source " << f.source;
    return 0;
}
void compare(const VectorOpDef& def, const LaneInputs& in, uint32_t wave, Form f = {}) {
    const auto run = run_vector_op_per_lane(words(def.opcode, f), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error << " opcode " << def.opcode;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = eval_vector_op(def, source(f, in, lane), std::nullopt);
        if (!want) {
            ADD_FAILURE() << "arithmetic did not produce a known value";
            return;
        }
        ASSERT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << "opcode " << def.opcode << " lane " << lane << " wave " << wave;
    }
}
std::vector<uint32_t> witness_words(uint32_t opcode, uint32_t mode) {
    if (mode == 8) return {vop1(opcode - 0x180)};
    if (mode >= 9) return {vop1(0x38, kVdst, 249), kVsrc0 | (6u << 8) | ((mode - 9) << 16)};
    return words(opcode, form(mode));
}
} // namespace
TEST(IrVectorUnaryIntegerE64Execution, EveryDefinitionMatchesDifferentLanesInBothWaves) {
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                auto in = lane_inputs(seed * 5310);
                in.active[12] = true; // Explicit active zero catches both scan sentinel rules.
                compare(def, in, wave);
            }
}
TEST(IrVectorUnaryIntegerE64Execution, HandWorkedIsaResultsPinDirectionAndZeroSentinels) {
    struct Example {
        uint32_t op, input, result;
    };
    // AMD70648 12.8: BFREV reverses bits, FFBH counts leading zeroes, FFBL indexes the LSB.
    // Both scans return all ones for zero. These constants are independent of the evaluator.
    const Example examples[] = {
        {0x1b7, 0, 0xffffffff},          {0x1b7, 0x80000000, 0x7fffffff},
        {0x1b7, 0x01234567, 0xfedcba98}, {0x1b8, 1, 0x80000000},
        {0x1b8, 0x01234567, 0xe6a2c480}, {0x1b8, 0x81234567, 0xe6a2c481},
        {0x1b9, 0, 0xffffffff},          {0x1b9, 1, 31},
        {0x1b9, 0x800000ff, 0},          {0x1b9, 0x100000ff, 3},
        {0x1b9, 0x0000ffff, 16},         {0x1ba, 0, 0xffffffff},
        {0x1ba, 0xff000001, 0},          {0x1ba, 0xff000008, 3},
        {0x1ba, 0xffff0000, 16},         {0x1ba, 0x80000000, 31},
    };
    for (const auto& x : examples) {
        const auto* def = find_vector_unary_integer_e64_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        ASSERT_EQ(eval_vector_op(*def, x.input, std::nullopt), Known(x.result));
        auto in = lane_inputs(5310);
        std::fill(in.s0.begin(), in.s0.end(), x.input);
        for (uint32_t wave : {32u, 64u}) {
            const auto run = run_vector_op_per_lane(words(x.op), in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                ASSERT_EQ(run.d[lane], in.active[lane] ? x.result : in.old_d[lane]);
        }
    }
}
TEST(IrVectorUnaryIntegerE64Execution, CanonicalSourceFormsAndOldDestinationAliasPreserveWiring) {
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode = 0; mode < 8; ++mode)
                compare(def, lane_inputs(5311), wave, form(mode));
}
TEST(IrVectorUnaryIntegerE64Execution, ExistingE32ControlsExecuteTheSameDefinitions) {
    const auto in = lane_inputs(5312);
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            const auto old = run_vector_op_per_lane({vop1(def.opcode - 0x180)}, in, wave);
            const auto current = run_vector_op_per_lane(words(def.opcode), in, wave);
            ASSERT_TRUE(old.error.empty()) << old.error;
            ASSERT_TRUE(current.error.empty()) << current.error;
            ASSERT_EQ(old.d, current.d);
        }
}
TEST(IrVectorUnaryIntegerE64Execution, HardScalarWordsCannotBecomeMasksAfterRestoringExec) {
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            Form f;
            f.source = 50;
            auto bad = words(def.opcode, f);
            bad.insert(bad.end(), {0xbefe04c1, 0x7d8a0280});
            const auto refused = run_vector_op_per_lane(bad, lane_inputs(7), wave);
            ASSERT_NE(refused.error.find("fabricated-fresh-mask-source"), std::string::npos)
                << refused.error;
            f.initialize = true;
            f.scalar = 0x80000001;
            auto good = words(def.opcode, f);
            good.insert(good.end(), {0xbefe04c1, 0x7d8a0280});
            const auto run = run_vector_op_per_lane(good, lane_inputs(7), wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
        }
}
TEST(IrVectorUnaryIntegerE64Execution, OneArmWordsKeepTheirObligationWithBothEdgeControls) {
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> code;
                if (initialize) code.insert(code.end(), {0xbeb203ff, 0x80000001});
                code.insert(code.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x00000021});
                Form f;
                f.source = 50;
                const auto op = words(def.opcode, f);
                code.insert(code.end(), op.begin(), op.end());
                code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
                auto program = lane_program(code, 64, wave);
                program.config.user_sgprs.resize(8); // Genuine dynamic s4 branch selector.
                const auto address = lane_harness_address();
                const auto module =
                    recompile_compute(program.code.data(), program.code.size(), &program.table,
                                      program.config, {RecompileDiagnosticStage::Compute, address});
                const auto reason = last_terminal_reject_reason(address);
                if (initialize)
                    ASSERT_FALSE(module.empty()) << reason;
                else {
                    ASSERT_TRUE(module.empty());
                    ASSERT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos)
                        << reason;
                }
            }
}
TEST(IrVectorUnaryIntegerE64Execution, ModifiersAndUnusedRawSourcesRemainRefused) {
    for (const auto& def : vector_unary_integer_e64_defs()) {
        for (uint32_t bit = 8; bit < 16; ++bit) {
            auto bad = words(def.opcode);
            bad[0] |= 1u << bit;
            ASSERT_FALSE(run_vector_op_per_lane(bad, lane_inputs(7)).error.empty());
        }
        for (uint32_t bit = 9; bit < 32; ++bit) {
            auto bad = words(def.opcode);
            bad[1] |= 1u << bit;
            ASSERT_FALSE(run_vector_op_per_lane(bad, lane_inputs(7)).error.empty());
        }
        for (uint32_t shift : {9u, 18u}) {
            auto bad = words(def.opcode);
            bad[1] |= 255u << shift;
            bad.push_back(0x81234567);
            ASSERT_FALSE(run_vector_op_per_lane(bad, lane_inputs(7)).error.empty());
        }
        ASSERT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(7)).error.empty());
    }
}
TEST(IrVectorUnaryIntegerE64Execution, ReservedMovIsArchitecturallyUnaryButStillRefused) {
    const auto in = words(0x181);
    const auto decoded = rdna2_decode_one(in.data(), in.size());
    ASSERT_EQ(decoded.n_src, 1);
    ASSERT_FALSE(run_vector_op_per_lane(in, lane_inputs(7)).error.empty());
    for (const auto& def : vector_unary_integer_e64_defs())
        ASSERT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(7)).error.empty());
}
TEST(IrVectorUnaryIntegerE64Execution, UnsupportedSpecialWordsAndSignedScanStayRefused) {
    Form f;
    f.source = 251;
    for (const auto& def : vector_unary_integer_e64_defs()) {
        ASSERT_FALSE(run_vector_op_per_lane(words(def.opcode, f), lane_inputs(7)).error.empty());
        ASSERT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(7)).error.empty());
        ASSERT_FALSE(eval_vector_op(def, std::nullopt, 0xffffffff));
    }
    ASSERT_FALSE(run_vector_op_per_lane(words(0x1bb), lane_inputs(7)).error.empty());
}
TEST(IrVectorUnaryIntegerE64Execution, EncodingsMatchIndependentLlvmGfx1030Checks) {
    EXPECT_EQ(words(0x1b7), (std::vector<uint32_t>{0xd5b70001, 0x00000102}));
    EXPECT_EQ(words(0x1b8), (std::vector<uint32_t>{0xd5b80001, 0x00000102}));
    EXPECT_EQ(words(0x1b9), (std::vector<uint32_t>{0xd5b90001, 0x00000102}));
    EXPECT_EQ(words(0x1ba), (std::vector<uint32_t>{0xd5ba0001, 0x00000102}));
}
TEST(IrVectorUnaryIntegerE64Execution, ExportsEveryCanonicalFormAndOldEncodingControls) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the test process never changes its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured
                              ? fs::path(configured)
                              : prosper_test::test_scratch_dir() / "unary-integer-e64-witness";
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
    for (const auto& def : vector_unary_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode = 0; mode < (def.opcode == 0x1b8 ? 15u : 9u); ++mode) {
                const auto program = lane_program(witness_words(def.opcode, mode), 64, wave);
                const uint64_t address = 0x5310000000ull + count * 0x100;
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
                ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                manifest << stem.filename().string() << '\t' << def.opcode << '\t' << wave << '\t'
                         << mode << '\n';
                ++count;
            }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 84u);
}

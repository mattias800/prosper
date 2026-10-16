// Canonical low-half e64 conversions: shared eval vs real SPIR-V, hand ISA rails, and
// source/old-destination ownership. NaN payloads retain the existing closed oracle boundary.
#include "gpu/recompiler/ir/ops/vector_half_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
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
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
uint32_t known_half(uint32_t h) {
    h &= 0xffffu;
    return (h & 0x7c00u) == 0x7c00u && (h & 1023u) ? h & 0xfbffu : h;
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = known_half(in.s0[lane]) | (known_half(in.s1[lane]) << 16);
        in.old_d[lane] = (uint32_t(0x3000 + lane * 7) << 16) | known_half(in.s1[lane]);
    }
    return in;
}
struct Form {
    uint32_t source = 258, literal = 0, scalar = 0;
    bool initialize = false;
};
Form form(uint32_t mode) {
    Form f;
    if (mode == 1) f.source = 257;
    if (mode == 2) f.source = 128;
    if (mode == 3) f.source = 192;
    if (mode == 4) f.source = 242;
    if (mode == 5) f.source = 243;
    if (mode == 6) {
        f.source = 255;
        f.literal = 0x40003e00;
    }
    if (mode == 7) {
        f.source = 50;
        f.scalar = 0xc4003e00;
        f.initialize = true;
    }
    return f;
}
std::vector<uint32_t> words(uint32_t op, Form f = {}) {
    std::vector<uint32_t> out;
    if (f.initialize) out.insert(out.end(), {0xbeb203ff, f.scalar});
    out.insert(out.end(), {0xd4000001u | (op << 16), f.source});
    if (f.source == 255) out.push_back(f.literal);
    return out;
}
uint32_t source(Form f, const LaneInputs& in, size_t lane) {
    if (f.source == 258) return in.s0[lane];
    if (f.source == 257) return in.old_d[lane];
    if (f.source == 128) return 0;
    if (f.source == 192) return 64;
    if (f.source == 242) return 0x3c00;
    if (f.source == 243) return 0xbc00;
    if (f.source == 255) return f.literal;
    if (f.source == 50 && f.initialize) return f.scalar;
    ADD_FAILURE() << "unmodelled source";
    return 0;
}
void compare(const ConversionDef& def, const LaneInputs& in, uint32_t wave, Form f = {}) {
    const auto run = run_vector_op_per_lane(words(def.opcode, f), in, wave, true);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto value = eval_conversion_op(def, source(f, in, lane));
        if (!value) {
            ADD_FAILURE() << "unknown numeric value";
            return;
        }
        ASSERT_EQ(run.d[lane],
                  in.active[lane] ? (in.old_d[lane] & 0xffff0000u) | *value : in.old_d[lane])
            << "op " << def.opcode << " lane " << lane << " wave " << wave;
    }
}
std::vector<uint32_t> witness_words(uint32_t op, uint32_t mode) {
    if (mode == 8) return {vop1(op - 0x180)};
    if (mode >= 9) {
        const uint32_t half = mode == 9 ? 4u : 5u;
        return {vop1(op - 0x180, kVdst, 249), kVsrc0 | ((half | 0x10u) << 8) | (half << 16)};
    }
    return words(op, form(mode));
}
void require_mask_result(const std::vector<uint32_t>& code, uint32_t wave, bool missing_old,
                         bool good) {
    auto program = lane_program(code, 64, wave);
    if (missing_old) program.code.erase(program.code.begin() + 4, program.code.begin() + 6);
    program.config.user_sgprs.resize(8);
    const auto address = lane_harness_address();
    const auto module =
        recompile_compute(program.code.data(), program.code.size(), &program.table, program.config,
                          {RecompileDiagnosticStage::Compute, address});
    const auto reason = last_terminal_reject_reason(address);
    if (good)
        ASSERT_FALSE(module.empty()) << reason;
    else {
        ASSERT_TRUE(module.empty());
        ASSERT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    }
}
}   // namespace
TEST(IrVectorHalfIntegerE64Execution, EveryDefinitionMatchesDifferentLanesAndInactiveHighHalves) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed = 1; seed <= 24; ++seed) compare(def, inputs(seed * 5317), wave);
}
TEST(IrVectorHalfIntegerE64Execution, HandWorkedIsaTruncationAndSaturationRails) {
    struct Example {
        uint32_t half, u, i;
    };
    // AMD70648 12.8: truncation toward zero, with the integer type's saturating rails.
    const Example examples[] = {
        {0, 0, 0},
        {0x8000, 0, 0},
        {1, 0, 0},
        {0x3ff, 0, 0},
        {0x83ff, 0, 0},
        {0x400, 0, 0},
        {0x3bff, 0, 0},
        {0x3c00, 1, 1},
        {0x3e00, 1, 1},
        {0x4001, 2, 2},
        {0x43ff, 3, 3},
        {0xbc00, 0, 0xffff},
        {0xbe00, 0, 0xffff},
        {0xc001, 0, 0xfffe},
        {0x77ff, 0x7ff0, 0x7ff0},
        {0x7800, 0x8000, 0x7fff},
        {0xf800, 0, 0x8000},
        {0x7bff, 0xffe0, 0x7fff},
        {0xfbff, 0, 0x8000},
        {0x7c00, 0xffff, 0x7fff},
        {0xfc00, 0, 0x8000},
    };
    for (const auto& def : vector_half_integer_e64_defs())
        for (const auto& x : examples) {
            const uint32_t want = def.opcode == 0x1d2 ? x.u : x.i;
            ASSERT_EQ(eval_conversion_op(def, x.half), Known(want));
            auto in = inputs(5317);
            std::fill(in.s0.begin(), in.s0.end(), 0x7c000000u | x.half);
            for (uint32_t wave : {32u, 64u}) {
                const auto run = run_vector_op_per_lane(words(def.opcode), in, wave, true);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    ASSERT_EQ(run.d[lane], in.active[lane] ? (in.old_d[lane] & 0xffff0000u) | want
                                                           : in.old_d[lane]);
            }
        }
}
TEST(IrVectorHalfIntegerE64Execution, CanonicalSourceFormsAndDestinationAliasPreserveWiring) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode = 0; mode < 8; ++mode) compare(def, inputs(5318), wave, form(mode));
}
TEST(IrVectorHalfIntegerE64Execution, ExistingE32AndSdwaControlsKeepTheirWordPlacement) {
    const auto in = inputs(5319);
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            const auto old = run_vector_op_per_lane(witness_words(def.opcode, 8), in, wave, true);
            const auto current = run_vector_op_per_lane(words(def.opcode), in, wave, true);
            ASSERT_TRUE(old.error.empty()) << old.error;
            ASSERT_TRUE(current.error.empty()) << current.error;
            ASSERT_EQ(old.d, current.d);
            for (uint32_t mode : {9u, 10u}) {
                const auto run =
                    run_vector_op_per_lane(witness_words(def.opcode, mode), in, wave, true);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto d =
                        eval_conversion_op(def, mode == 9 ? in.s0[lane] : in.s0[lane] >> 16);
                    if (!d) {
                        ADD_FAILURE() << "unknown control value";
                        return;
                    }
                    const uint32_t placed = mode == 9 ? (in.old_d[lane] & 0xffff0000u) | *d
                                                      : (in.old_d[lane] & 0xffffu) | (*d << 16);
                    ASSERT_EQ(run.d[lane], in.active[lane] ? placed : in.old_d[lane]);
                }
            }
        }
}
TEST(IrVectorHalfIntegerE64Execution, ArbitraryAndHardOldDestinationsKeepTheirSeparateContracts) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            auto code = words(def.opcode);
            code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
            // #5174/#5193: a never-loaded old word can remain an arbitrary value.
            require_mask_result(code, wave, true, true);
            require_mask_result(code, wave, false, true);
            auto arbitrary =
                std::vector<uint32_t>{0xbf068004, 0xbf840002, vop1(1, 1, 255), 0x3c004000};
            const auto numeric = words(def.opcode);
            arbitrary.insert(arbitrary.end(), numeric.begin(), numeric.end());
            arbitrary.insert(arbitrary.end(), {0xbefe04c1, 0x7d8a0280});
            require_mask_result(arbitrary, wave, true, true);
            arbitrary.insert(arbitrary.begin(), {vop1(1, 1, 255), 0x40003c00});
            require_mask_result(arbitrary, wave, true, true);
            std::vector<uint32_t> bad{vop1(1, 1, 50)};
            const auto op = words(def.opcode);
            bad.insert(bad.end(), op.begin(), op.end());
            bad.insert(bad.end(), {0xbefe04c1, 0x7d8a0280});
            require_mask_result(bad, wave, false, false);
            bad.insert(bad.begin(), {0xbeb203ff, 0x35553c00});
            require_mask_result(bad, wave, false, true);
        }
}
TEST(IrVectorHalfIntegerE64Execution, OneArmHardOldWordKeepsItsObligationWithBothEdgeControl) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> code;
                // A hard scalar supplies a real unknown old word; the other edge replaces it.
                if (initialize) code.insert(code.end(), {0xbeb203ff, 0x40003c00});
                code.push_back(vop1(1, 1, 50));
                code.insert(code.end(), {0xbf068004, 0xbf840002, vop1(1, 1, 255), 0x3c004000});
                const auto op = words(def.opcode);
                code.insert(code.end(), op.begin(), op.end());
                code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
                require_mask_result(code, wave, false, initialize);
            }
}
TEST(IrVectorHalfIntegerE64Execution, MissingAndOneArmSourcesStayUnprovedWithGenuineControls) {
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            Form f;
            f.source = 50;
            auto code = words(def.opcode, f);
            code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
            require_mask_result(code, wave, false, false);
            f.initialize = true;
            f.scalar = 0x35553c00;
            code = words(def.opcode, f);
            code.insert(code.end(), {0xbefe04c1, 0x7d8a0280});
            require_mask_result(code, wave, false, true);
            for (bool initialize : {false, true}) {
                std::vector<uint32_t> branch;
                if (initialize) branch.insert(branch.end(), {0xbeb203ff, 0x35553c00});
                branch.insert(branch.end(), {0xbf068004, 0xbf840002, 0xbeb203ff, 0x35554000});
                f.initialize = false;
                const auto op = words(def.opcode, f);
                branch.insert(branch.end(), op.begin(), op.end());
                branch.insert(branch.end(), {0xbefe04c1, 0x7d8a0280});
                require_mask_result(branch, wave, false, initialize);
            }
        }
}
TEST(IrVectorHalfIntegerE64Execution, ModifiersUnusedRawFieldsAndOtherConversionsRemainRefused) {
    for (const auto& def : vector_half_integer_e64_defs()) {
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9 : 8; bit < (word ? 32u : 16u); ++bit) {
                auto code = words(def.opcode);
                code[word] |= 1u << bit;
                code.push_back(0x3c00);
                ASSERT_FALSE(run_vector_op_per_lane(code, inputs(17), 64, true).error.empty());
            }
        for (uint32_t shift : {9u, 18u}) {
            auto code = words(def.opcode);
            code[1] |= 255u << shift;
            code.push_back(0x3c00);
            ASSERT_FALSE(run_vector_op_per_lane(code, inputs(17), 64, true).error.empty());
        }
        ASSERT_TRUE(run_vector_op_per_lane(words(def.opcode), inputs(17), 64, true).error.empty());
    }
    for (uint32_t op : {0x1d0u, 0x1d1u})
        ASSERT_FALSE(run_vector_op_per_lane(words(op), inputs(17), 64, true).error.empty());
}
TEST(IrVectorHalfIntegerE64Execution, ExistingNaNPayloadBoundaryIsExplicitWithFiniteControls) {
    for (const auto& def : vector_half_integer_e64_defs()) {
        ASSERT_FALSE(eval_conversion_op(def, 0x7e01));
        for (uint32_t raw : {0x3c007e01u, 0x7e013c00u}) {
            auto in = inputs(17);
            std::fill(in.s0.begin(), in.s0.end(), raw);
            const auto run = run_vector_op_per_lane(words(def.opcode), in, 64, true);
            ASSERT_NE(run.error.find("unspecified half NaN payload"), std::string::npos)
                << run.error;
        }
        compare(def, inputs(17), 64);
    }
}
TEST(IrVectorHalfIntegerE64Execution, EncodingsMatchIndependentLlvmGfx1030Checks) {
    EXPECT_EQ(words(0x1d2), (std::vector<uint32_t>{0xd5d20001, 0x00000102}));
    EXPECT_EQ(words(0x1d3), (std::vector<uint32_t>{0xd5d30001, 0x00000102}));
}
TEST(IrVectorHalfIntegerE64Execution, ExportsEveryCanonicalFormAndOldEncodingControl) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the test never modifies its environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured
                              ? fs::path(configured)
                              : prosper_test::test_scratch_dir() / "half-integer-e64-witness";
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
    for (const auto& def : vector_half_integer_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode = 0; mode < 11; ++mode) {
                const auto program = lane_program(witness_words(def.opcode, mode), 64, wave);
                const uint64_t address = 0x5317000000ull + count * 0x100;
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
    EXPECT_EQ(count, 44u);
}

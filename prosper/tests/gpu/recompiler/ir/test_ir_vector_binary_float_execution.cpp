// Existing binary F32 encodings: real emitted lane results, partial EXEC and hand-worked IEEE values.
#include "gpu/recompiler/ir/ops/vector_binary_float.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
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
    const uint32_t edges[] = {0x3fc00000, 0xc0100000, 0x3f800000, 0xbf000000,
                              0x40000000, 0x3f800001, 0xc0200000, 0x3f7fffff};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = lane < 8 ? edges[lane]
                               : (in.s0[lane] & 0x807fffffu) |
                                     ((110u + static_cast<uint32_t>(lane) % 25) << 23);
        in.s1[lane] = lane < 8 ? edges[(lane * 3 + 1) % 8]
                               : (in.s1[lane] & 0x807fffffu) |
                                     ((112u + static_cast<uint32_t>(lane) % 21) << 23);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t op, uint32_t mode = 0, uint32_t abs = 0, uint32_t neg = 0,
                            uint32_t omod = 0, bool clamp = false) {
    const uint32_t a = mode == 1 || mode == 3 ? 242
                       : mode == 4            ? 255
                       : mode == 6            ? 135
                                              : 256 + kVsrc0;
    const uint32_t b = mode == 2 || mode == 3 ? 244 : mode == 5 ? 255 : 256 + kVsrc1;
    std::vector<uint32_t> code = {0xd4000000u | (op << 16) | kVdst | (abs << 8) |
                                      (clamp ? 1u << 15 : 0),
                                  a | (b << 9) | (omod << 27) | (neg << 29)};
    if (mode == 4 || mode == 5) code.push_back(mode == 4 ? 0x3fc00000 : 0xc0100000);
    return code;
}
std::array<Known, 3> sources(const LaneInputs& in, size_t lane, uint32_t mode = 0, uint32_t abs = 0,
                             uint32_t neg = 0) {
    const uint32_t raw[] = {mode == 1 || mode == 3 ? 0x3f800000u
                            : mode == 4            ? 0x3fc00000u
                            : mode == 6            ? 7u
                                                   : in.s0[lane],
                            mode == 2 || mode == 3 ? 0x40000000u
                            : mode == 5            ? 0xc0100000u
                                                   : in.s1[lane]};
    std::array<Known, 3> out = {std::nullopt, std::nullopt, std::nullopt};
    for (size_t k = 0; k < 2; ++k) {
        uint32_t v = raw[k];
        if (abs & (1u << k)) v &= 0x7fffffffu;
        out[k] = neg & (1u << k) ? evaluate_op(Opcode::F32Sub, 0, 0, v, 0) : Known(v);
    }
    return out;
}
Known result(const TernaryIntegerDef& def, const std::array<Known, 3>& args, uint32_t omod = 0,
             bool clamp = false) {
    auto value = eval_ternary_integer_op(def, args);
    if (!value) return std::nullopt;
    if (omod)
        value = evaluate_op(Opcode::F32Mul, 0, *value,
                            omod == 1   ? 0x40000000
                            : omod == 2 ? 0x40800000
                                        : 0x3f000000,
                            0);
    if (clamp && value) value = evaluate_op(Opcode::F32Clamp01, 0, *value, 0, 0);
    return value;
}
void compare(const TernaryIntegerDef& def, const LaneInputs& in, uint32_t wave, uint32_t mode = 0) {
    auto run = run_vector_op_per_lane(words(def.opcode, mode), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        auto want = result(def, sources(in, lane, mode));
        if (!want) {
            ADD_FAILURE() << "unknown lane " << lane;
            return;
        }
        EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
            << lane << " wave " << wave << " mode " << mode;
    }
}
}   // namespace
TEST(IrVectorBinaryFloatExecution, EveryDefinitionMatchesDifferingLanesInBothWaves) {
    for (const auto& def : vector_binary_float_defs()) {
        auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[0].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[1].kind, OperandKind::VGPR);
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, inputs(seed * 5278), wave);
    }
}
TEST(IrVectorBinaryFloatExecution, HarnessSeesDifferentOperandsAndInactiveDestinations) {
    auto in = inputs(113);
    auto run = run_vector_op_per_lane(words(0x104), in);
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
}
TEST(IrVectorBinaryFloatExecution, HandWorkedBinaryArithmeticFromTheIsa) {
    // 70648 ADD/SUB/SUBREV/MUL/MIN/MAX. The chosen operands are exact binary fractions.
    // 1.5 + -2.25=-.75; their differences are +/-3.75 and their product is -3.375.
    const std::array<std::array<uint32_t, 8>, 4> examples = {
        {{0x3fc00000, 0xc0100000, 0xbf400000, 0x40700000, 0xc0700000, 0xc0580000, 0xc0100000,
          0x3fc00000},
         {0xbf000000, 0x40000000, 0x3fc00000, 0xc0200000, 0x40200000, 0xbf800000, 0xbf000000,
          0x40000000},
         {0x3f800000, 0x3f800000, 0x40000000, 0, 0, 0x3f800000, 0x3f800000, 0x3f800000},
         {0x35800000, 0x35800000, 0x36000000, 0, 0, 0x2b800000, 0x35800000, 0x35800000}}};
    for (const auto& e : examples) {
        size_t op = 0;
        for (const auto& def : vector_binary_float_defs()) {
            auto want = e[2 + op++];
            EXPECT_EQ(eval_ternary_integer_op(def, {e[0], e[1], std::nullopt}), Known(want));
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), e[0]);
            std::fill(in.s1.begin(), in.s1.end(), e[1]);
            for (uint32_t wave : {64u, 32u}) {
                auto run = run_vector_op_per_lane(words(def.opcode), in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    EXPECT_EQ(run.d[lane], in.active[lane] ? want : in.old_d[lane]);
            }
        }
    }
}
TEST(IrVectorBinaryFloatExecution, InlineFloatIntegerAndLiteralBindingsKeepSourceOrder) {
    for (const auto& def : vector_binary_float_defs())
        for (uint32_t mode = 1; mode <= 6; ++mode)
            for (uint32_t wave : {64u, 32u}) compare(def, inputs(31 + mode), wave, mode);
}
TEST(IrVectorBinaryFloatExecution, SourceAndResultModifierOrderIsUnchanged) {
    auto in = inputs(43);
    std::fill(in.s0.begin(), in.s0.end(), 0xbf000000);
    std::fill(in.s1.begin(), in.s1.end(), 0x40000000);
    for (const auto& def : vector_binary_float_defs())
        for (uint32_t abs = 0; abs < 4; ++abs)
            for (uint32_t neg = 0; neg < 4; ++neg)
                for (uint32_t omod = 0; omod < 4; ++omod)
                    for (bool clamp : {false, true})
                        for (uint32_t wave : {64u, 32u}) {
                            auto run = run_vector_op_per_lane(
                                words(def.opcode, 0, abs, neg, omod, clamp), in, wave);
                            ASSERT_TRUE(run.error.empty()) << run.error;
                            auto want = result(def, sources(in, 0, 0, abs, neg), omod, clamp);
                            if (!want) {
                                ADD_FAILURE() << "unknown modifier result";
                                return;
                            }
                            for (size_t lane = 0; lane < in.lanes(); ++lane)
                                EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane]);
                        }
}
TEST(IrVectorBinaryFloatExecution, EffectsAndUnknownInputsRetainTheirDomains) {
    for (const auto& def : vector_binary_float_defs()) {
        auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 2u);
        EXPECT_FALSE(eval_ternary_integer_op(def, {std::nullopt, 0x3f800000, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x3f800000, std::nullopt, std::nullopt}));
        EXPECT_FALSE(eval_ternary_integer_op(def, {0x7fc01234, 0x7fc56789, std::nullopt}));
        EXPECT_TRUE(eval_ternary_integer_op(def, {0x3f800000, 0x40000000, std::nullopt}));
    }
    EXPECT_EQ(find_vector_binary_float_op(Rdna2Format::VOP2, 0x103), nullptr);
    EXPECT_EQ(find_vector_binary_float_op(Rdna2Format::VOP3, 0x107), nullptr);
    EXPECT_EQ(find_vector_binary_float_op(Rdna2Format::VOP3, 0x12f), nullptr);
    EXPECT_EQ(find_vector_binary_float_op(Rdna2Format::VOP3, 0x201), nullptr);
}

// Export the existing operand forms in both waves using the execution harness's exact bindings.
TEST(IrVectorBinaryFloatExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "binary-float-witness";
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
    for (const auto& def : vector_binary_float_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
                const auto program = lane_program(words(def.opcode, mode), 64, wave, false);
                const uint64_t address = 0x5278000000ull + count * 0x100;
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
    EXPECT_EQ(count, 84u);
}

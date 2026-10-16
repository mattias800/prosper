// Canonical subword conversions must select the exact integer bits before producing F32.
// Wrong byte direction or signed-nibble bias corrupts unpacked attributes and interpolation.
#include "gpu/recompiler/ir/ops/vector_subword_to_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
Form form(uint32_t kind) {
    Form f;
    if (kind == 1) f.source = 257;
    if (kind == 2) f.source = 128;
    if (kind == 3) f.source = 192;
    if (kind == 4) f.source = 193;
    if (kind == 5) {
        f.source = 255;
        f.literal = 0x81234567;
    }
    if (kind == 6) {
        f.source = 50;
        f.scalar = 0x89abcdef;
        f.initialize = true;
    }
    if (kind == 7) f.source = 242;
    return f;
}
std::vector<uint32_t> words(uint32_t opcode, Form f = {}, bool old = false, uint32_t scale = 0,
                            bool clamp = false, bool sdwa = false) {
    std::vector<uint32_t> out;
    if (f.initialize) out.insert(out.end(), {0xbeb203ff, f.scalar});
    if (sdwa)
        out.insert(out.end(),
                   {vop1(opcode - 0x180, kVdst, 249),
                    kVsrc0 | (6u << 8) | (clamp ? 1u << 13 : 0u) | (scale << 14) | (6u << 16)});
    else if (old)
        out.push_back(vop1(opcode - 0x180, kVdst, f.source));
    else
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
uint32_t unsigned_byte_bits(uint32_t byte) {
    if (!byte) return 0;
    const auto exponent = std::bit_width(byte) - 1u;
    return ((127u + exponent) << 23) | ((byte ^ (1u << exponent)) << (23u - exponent));
}
// AMD70648 op14 lists these sixteen values explicitly: signed low nibble divided by16.
constexpr std::array<uint32_t, 16> kNibbleBits{
    0x00000000, 0x3d800000, 0x3e000000, 0x3e400000, 0x3e800000, 0x3ea00000, 0x3ec00000, 0x3ee00000,
    0xbf000000, 0xbee00000, 0xbec00000, 0xbea00000, 0xbe800000, 0xbe400000, 0xbe000000, 0xbd800000};
LaneInputs modifier_inputs() {
    auto in = lane_inputs(5320);
    // The NMin/NMax oracle declines zero/zero sign ties. Keep the old CLAMP controls
    // inside its defined domain; canonical execution still covers every zero subword.
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto value = static_cast<uint32_t>(lane);
        in.s0[lane] = ((0x01010101u * (1u + value)) & ~15u) | (1u + value % 15u);
    }
    return in;
}
uint32_t isa_result(uint32_t opcode, uint32_t raw) {
    if (opcode == 0x18e) return kNibbleBits[raw & 15u];
    return unsigned_byte_bits((raw >> (8u * (opcode - 0x191u))) & 255u);
}
uint32_t modified_result(uint32_t bits, uint32_t rsrc1, uint32_t scale, bool clamp) {
    if (scale && !(rsrc1 & (0x20000u | 0x800000u)) && bits)
        bits += scale == 1 ? 0x00800000u : (scale == 2 ? 0x01000000u : 0xff800000u);
    if (clamp) bits = bits & 0x80000000u ? 0u : std::min(bits, 0x3f800000u);
    return bits;
}
void check_values(const ConversionDef& def, Form f, const LaneInputs& in,
                  const std::vector<uint32_t>& actual, uint32_t rsrc1 = 0, uint32_t scale = 0,
                  bool clamp = false) {
    ASSERT_EQ(actual.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto raw = source(f, in, lane);
        const auto evaluated = eval_conversion_op(def, raw);
        if (!evaluated) {
            ADD_FAILURE() << "conversion must produce a known value";
            return;
        }
        const auto numeric = isa_result(def.opcode, raw);
        ASSERT_EQ(*evaluated, numeric) << "opcode " << def.opcode << " source " << raw;
        const auto want = modified_result(numeric, rsrc1, scale, clamp);
        ASSERT_EQ(actual[lane], in.active[lane] ? want : in.old_d[lane])
            << "opcode " << def.opcode << " lane " << lane << " RSRC1 " << rsrc1;
    }
}
void compare(const ConversionDef& def, const LaneInputs& in, uint32_t wave, Form f = {},
             uint32_t rsrc1 = 0, bool old = false) {
    const auto run = run_vector_op_per_lane(words(def.opcode, f, old), in, wave, false, rsrc1);
    ASSERT_TRUE(run.error.empty()) << run.error << " opcode " << def.opcode;
    check_values(def, f, in, run.d, rsrc1);
}
std::vector<uint32_t> execute_module(const std::vector<uint32_t>& module, const LaneInputs& in) {
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    auto result = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    if (!vm.error.empty() || result.size() < in.lanes()) return {};
    result.resize(in.lanes());
    return result;
}
} // namespace
TEST(IrVectorSubwordToFloatE64Execution, EveryByteValueIsExactWithUnrelatedOtherBytes) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        if (def.opcode == 0x18e) continue;
        const uint32_t shift = (def.opcode - 0x191u) * 8u;
        for (uint32_t wave : {32u, 64u})
            for (uint32_t batch = 0; batch < 4; ++batch) {
                auto in = lane_inputs(5320 + batch);
                std::fill(in.active.begin(), in.active.end(), true);
                for (size_t lane = 0; lane < in.lanes(); ++lane)
                    in.s0[lane] = ((0x81234567u ^ (static_cast<uint32_t>(lane) * 0x01020408u)) &
                                   ~(255u << shift)) |
                                  ((batch * 64u + static_cast<uint32_t>(lane)) << shift);
                compare(def, in, wave);
            }
    }
}
TEST(IrVectorSubwordToFloatE64Execution, AllSixteenSignedNibblesMatchTheManualWithoutBias) {
    const auto* def = find_vector_subword_to_float_e64_op(Rdna2Format::VOP3, 0x18e);
    ASSERT_NE(def, nullptr);
    auto in = lane_inputs(5320);
    std::fill(in.active.begin(), in.active.end(), true);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        in.s0[lane] = (0x81234560u ^ (static_cast<uint32_t>(lane) << 4)) |
                      (static_cast<uint32_t>(lane) & 15u);
    for (uint32_t nibble = 0; nibble < 16; ++nibble)
        EXPECT_EQ(eval_conversion_op(*def, nibble), Known(kNibbleBits[nibble]));
    for (uint32_t wave : {32u, 64u}) compare(*def, in, wave);
}
TEST(IrVectorSubwordToFloatE64Execution, IndependentByteExamplesPinUnsignednessAndDirection) {
    const std::array<uint32_t, 4> results{0x42ce0000, 0x428a0000, 0x420c0000, 0x43010000};
    for (uint32_t byte = 0; byte < 4; ++byte) {
        const auto* def = find_vector_subword_to_float_e64_op(Rdna2Format::VOP3, 0x191u + byte);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_conversion_op(*def, 0x81234567), Known(results[byte]));
        auto in = lane_inputs(5320);
        std::fill(in.s0.begin(), in.s0.end(), 0x81234567);
        for (uint32_t wave : {32u, 64u}) compare(*def, in, wave);
    }
}
TEST(IrVectorSubwordToFloatE64Execution, DistinctLaneInputsAndPartialExecPreserveOldDestinations) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t seed = 1; seed <= 8; ++seed) compare(def, lane_inputs(5320 * seed), wave);
}
TEST(IrVectorSubwordToFloatE64Execution, OperandFormsAndOldDestinationAliasMatchOldEncodings) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t kind = 0; kind < 8; ++kind)
                for (bool old : {false, true})
                    compare(def, lane_inputs(5320), wave, form(kind), 0, old);
}
TEST(IrVectorSubwordToFloatE64Execution, AllF32RoundingDenormalAndIeeeModesKeepExactResults) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t rounding = 0; rounding < 4; ++rounding)
                for (uint32_t denormals = 0; denormals < 4; ++denormals)
                    for (uint32_t ieee : {0u, 1u})
                        compare(def, lane_inputs(5320), wave, {},
                                (rounding << 12) | (denormals << 16) | (ieee << 23));
}
TEST(IrVectorSubwordToFloatE64Execution, ObservedAndUnavailableModesNeedNoNumericGuess) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t rsrc1 : {0u, 0x20000u, 0x30000u, UINT32_MAX})
                compare(def, lane_inputs(5320), wave, {}, rsrc1);
}
TEST(IrVectorSubwordToFloatE64Execution, ExistingSdwaResultModifiersKeepTheirLaunchPolicy) {
    const auto inputs = lane_inputs(5320);
    const auto clamped_inputs = modifier_inputs();
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u})
            for (uint32_t scale = 1; scale < 4; ++scale)
                for (bool clamp : {false, true})
                    for (uint32_t rsrc1 : {0u, 0x30000u, 0x800000u}) {
                        const auto& in = clamp ? clamped_inputs : inputs;
                        const auto run =
                            run_vector_op_per_lane(words(def.opcode, {}, true, scale, clamp, true),
                                                   in, wave, false, rsrc1);
                        ASSERT_TRUE(run.error.empty()) << run.error;
                        check_values(def, {}, in, run.d, rsrc1, scale, clamp);
                    }
}
TEST(IrVectorSubwordToFloatE64Execution, OracleDeclinesClampZeroTiesWithNonzeroControls) {
    auto zero_inputs = lane_inputs(5320);
    std::fill(zero_inputs.s0.begin(), zero_inputs.s0.end(), 0u);
    const auto nonzero_inputs = modifier_inputs();
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            const auto code = words(def.opcode, {}, true, 1, true, true);
            const auto boundary = run_vector_op_per_lane(code, zero_inputs, wave, false, 0);
            ASSERT_NE(boundary.error.find("unspecified float ALU result"), std::string::npos)
                << boundary.error;
            const auto control = run_vector_op_per_lane(code, nonzero_inputs, wave, false, 0);
            ASSERT_TRUE(control.error.empty()) << control.error;
            check_values(def, {}, nonzero_inputs, control.d, 0, 1, true);
        }
}
TEST(IrVectorSubwordToFloatE64Execution, HardScalarWordsCannotBecomeFreshMasks) {
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            Form f;
            f.source = 50;
            auto bad = words(def.opcode, f);
            bad.insert(bad.end(), {0xbefe04c1, 0x7d8a0280});
            const auto refused = run_vector_op_per_lane(bad, lane_inputs(5320), wave);
            ASSERT_NE(refused.error.find("fabricated-fresh-mask-source"), std::string::npos)
                << refused.error;
            f.initialize = true;
            f.scalar = 0x80000001;
            auto good = words(def.opcode, f);
            good.insert(good.end(), {0xbefe04c1, 0x7d8a0280});
            const auto run = run_vector_op_per_lane(good, lane_inputs(5320), wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
        }
}
TEST(IrVectorSubwordToFloatE64Execution, OneArmSourceKeepsItsObligationWithBothEdgeControls) {
    for (const auto& def : vector_subword_to_float_e64_defs())
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
                program.config.user_sgprs.resize(8);
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
TEST(IrVectorSubwordToFloatE64Execution, ModifiersUnusedFieldsAndSpecialSourcesRemainRefused) {
    for (const auto& def : vector_subword_to_float_e64_defs()) {
        for (uint32_t word : {0u, 1u})
            for (uint32_t bit = word ? 9 : 8; bit < (word ? 32u : 16u); ++bit) {
                auto bad = words(def.opcode);
                bad[word] |= 1u << bit;
                ASSERT_FALSE(run_vector_op_per_lane(bad, lane_inputs(5320)).error.empty());
            }
        for (uint32_t shift : {9u, 18u}) {
            auto bad = words(def.opcode);
            bad[1] |= 255u << shift;
            bad.push_back(0x81234567);
            ASSERT_FALSE(run_vector_op_per_lane(bad, lane_inputs(5320)).error.empty());
        }
        Form special;
        special.source = 251;
        ASSERT_FALSE(
            run_vector_op_per_lane(words(def.opcode, special), lane_inputs(5320)).error.empty());
        ASSERT_FALSE(eval_conversion_op(def, std::nullopt));
        ASSERT_TRUE(run_vector_op_per_lane(words(def.opcode), lane_inputs(5320)).error.empty());
    }
}
TEST(IrVectorSubwordToFloatE64Execution, EncodingsMatchIndependentLlvmGfx1030Checks) {
    EXPECT_EQ(words(0x18e), (std::vector<uint32_t>{0xd58e0001, 0x00000102}));
    EXPECT_EQ(words(0x191), (std::vector<uint32_t>{0xd5910001, 0x00000102}));
    EXPECT_EQ(words(0x192), (std::vector<uint32_t>{0xd5920001, 0x00000102}));
    EXPECT_EQ(words(0x193), (std::vector<uint32_t>{0xd5930001, 0x00000102}));
    EXPECT_EQ(words(0x194), (std::vector<uint32_t>{0xd5940001, 0x00000102}));
}
TEST(IrVectorSubwordToFloatE64Execution, ExportsExecutesAndReplaysAllWitnessAttributes) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input, never mutated here.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "subword-float-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\topcode\twave\tform\trsrc1\told\tscale\tclamp\tsdwa\n";
    size_t count = 0;
    const auto inputs = lane_inputs(5320);
    const auto modified_inputs = modifier_inputs();
    const auto save = [&](const ConversionDef& def, uint32_t wave, uint32_t kind, uint32_t rsrc1,
                          bool old, uint32_t scale = 0, bool clamp = false, bool sdwa = false) {
        const auto f = form(kind);
        auto program = lane_program(words(def.opcode, f, old, scale, clamp, sdwa), 64, wave);
        program.config.compute_pgm_rsrc1 = rsrc1;
        const uint64_t address = 0x5320000000ull + count * 0x100;
        const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
        std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
        bin.write(reinterpret_cast<const char*>(program.code.data()),
                  static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
        bin.close();
        ASSERT_TRUE(bin.good());
        const auto file = dump / (stem + ".inputs");
        ASSERT_TRUE(write_refused_compute_inputs(file, address, &program.table, program.config,
                                                 "synthetic-subword-witness (no live refusal)"));
        const auto read = read_refused_shader_inputs(file);
        ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
        const auto replay = replay_refused_shader_inputs(program.code, read);
        ASSERT_TRUE(replay.produced) << replay.reject;
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        const auto& runtime_inputs = sdwa && clamp ? modified_inputs : inputs;
        check_values(def, f, runtime_inputs, execute_module(replay.spirv, runtime_inputs), rsrc1,
                     scale, clamp);
        manifest << stem << '\t' << def.opcode << '\t' << wave << '\t' << kind << '\t' << rsrc1
                 << '\t' << old << '\t' << scale << '\t' << clamp << '\t' << sdwa << '\n';
        ++count;
    };
    for (const auto& def : vector_subword_to_float_e64_defs())
        for (uint32_t wave : {32u, 64u}) {
            for (uint32_t rsrc1 :
                 {0u, 0x1000u, 0x2000u, 0x3000u, 0x10000u, 0x20000u, 0x30000u, 0xf0000u, 0x800000u})
                for (uint32_t kind = 0; kind < 8; ++kind) save(def, wave, kind, rsrc1, false);
            for (uint32_t kind = 0; kind < 8; ++kind) save(def, wave, kind, 0, true);
            for (uint32_t scale = 1; scale < 4; ++scale)
                for (bool clamp : {false, true})
                    for (uint32_t rsrc1 : {0u, 0x30000u, 0x800000u})
                        save(def, wave, 0, rsrc1, true, scale, clamp, true);
        }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 980u);
}

// ISA70648 §12.10 arithmetic; wrapper-only checks retain explicit MED/LOW uncertainties.
#include "gpu/recompiler/ir/ops/vector_packed_integer.hpp"
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
#include <cstdint>
#include <cstddef>
#include <ios>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    uint32_t lo = 0, hi = 7, neg_lo = 0, neg_hi = 0;
    std::array<uint32_t, 3> src = {256 + kVsrc0, 256 + kVsrc1, 256 + kVsrc2};
    std::optional<uint32_t> literal;
    bool clamp = false;
};
std::vector<uint32_t> words(uint32_t op, const Form& f = {}) {
    std::vector<uint32_t> code = {0xcc000000u | (op << 16) | kVdst | (f.neg_hi << 8) |
                                      (f.lo << 11) | ((f.hi & 4) << 12) | (f.clamp ? 1u << 15 : 0),
                                  f.src[0] | (f.src[1] << 9) | (f.src[2] << 18) |
                                      ((f.hi & 3) << 27) | (f.neg_lo << 29)};
    if (f.literal) code.push_back(*f.literal);
    return code;
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1664525u + 1013904223u;
        in.s2.push_back(seed);
    }
    return in;
}
uint32_t source_word(const LaneInputs& in, size_t lane, const Form& f, uint32_t slot) {
    const uint32_t code = f.src[slot];
    if (code >= 128 && code <= 192) return code - 128;
    if (code >= 193 && code <= 208) return 192u - code;
 // #4407: this is the EXISTING inline-float wrapper contract, not hardware-value proof.
    constexpr uint32_t half[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                 0xc000, 0x4400, 0xc400, 0x3118};
    if (code >= 240 && code <= 248) return half[code - 240];
    if (code == 255) {
        if (!f.literal) std::abort();
        return *f.literal;
    }
    if (code == 256 + kVsrc0) return in.s0[lane];
    if (code == 256 + kVsrc1) return in.s1[lane];
    if (code == 256 + kVsrc2) return in.s2[lane];
    if (code == 256 + kVdst) return in.old_d[lane];
    std::abort();
}
Known expected(const TernaryIntegerDef& def, const LaneInputs& in, size_t lane, const Form& f) {
    std::array<uint32_t, 2> result = {};
    for (uint32_t high = 0; high < 2; ++high) {
        std::array<Known, 3> selected;
        for (uint32_t slot = 0; slot < 3; ++slot) {
            const uint32_t select = ((high ? f.hi : f.lo) >> slot) & 1u;
            uint32_t v = (source_word(in, lane, f, slot) >> (16 * select)) & 0xffffu;
            // Preserve the guest-derived bit15 toggle; CONFIDENCE: MED, no published integer NEG proof.
            if (((high ? f.neg_hi : f.neg_lo) >> slot) & 1u) v ^= 0x8000u;
            selected[slot] = v;
        }
        const auto v = eval_ternary_integer_op(def, selected);
        if (!v) return std::nullopt;
        result[high] = *v;
    }
    return result[0] | (result[1] << 16);
}
::testing::AssertionResult matches(const TernaryIntegerDef& def, const LaneInputs& in,
                                   uint32_t wave, const Form& f = {}, Known hand = std::nullopt) {
    const auto run = run_vector_op_per_lane(words(def.opcode, f), in, wave);
    if (!run.error.empty()) return ::testing::AssertionFailure() << run.error;
    if (run.d.size() != in.lanes()) return ::testing::AssertionFailure() << "wrong lane count";
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto want = hand ? hand : expected(def, in, lane, f);
        if (!want) return ::testing::AssertionFailure() << "unknown expected lane " << lane;
        const auto value = in.active[lane] ? *want : in.old_d[lane];
        if (run.d[lane] != value)
            return ::testing::AssertionFailure()
                   << "op " << def.opcode << " wave " << wave << " lane " << lane << " got "
                   << run.d[lane] << " expected " << value;
    }
    return ::testing::AssertionSuccess();
}
}   // namespace
TEST(IrVectorPackedIntegerExecution, EveryDefinitionMatchesDistinctLanesAndExecInBothWaves) {
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed)
                ASSERT_TRUE(matches(def, inputs(seed * 5293), wave));
}
TEST(IrVectorPackedIntegerExecution, HarnessSeesOperandChangesAndInactiveCanaries) {
    for (const auto& def : vector_packed_integer_defs()) {
        auto in = inputs(531);
        const auto first = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(first.error.empty()) << first.error;
        std::set<uint32_t> varied;
        size_t inactive = 0;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            if (in.active[lane])
                varied.insert(first.d[lane]);
            else {
                ++inactive;
                ASSERT_EQ(first.d[lane], in.old_d[lane]);
            }
        EXPECT_GT(varied.size(), 3u);
        EXPECT_GE(inactive, 2u);
        for (auto& v : in.s0) v ^= 0x80018001u;
        const auto changed = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(changed.error.empty()) << changed.error;
        EXPECT_NE(first.d, changed.d) << def.opcode;
        in.s2.pop_back();
        const auto bad = run_vector_op_per_lane(words(def.opcode), in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_FALSE(bad.error.empty());
    }
}
TEST(IrVectorPackedIntegerExecution, HandWorkedRegisterAndRawLiteralIsaExamples) {
    // Row1 low:2,4,6; high:-1,3,5. Row2 low:15,-2,1; high:1,-32768,-1.
    // Truncate every result to16bits; shifts consume S0&15 and shift S1 (signed for ASHR).
    constexpr std::array<uint32_t, 14> row1 = {
        0x0002000e, 0xfffd0008, 0x00020006, 0xfffcfffe, 0x80000010, 0x00000001, 0x00000001,
        0x00030004, 0xffff0002, 0x0002000e, 0x00020006, 0xfffcfffe, 0xffff0004, 0x00030002};
    constexpr std::array<uint32_t, 14> row2 = {
        0x7fffffe3, 0x8000ffe2, 0x8001000d, 0x80010011, 0,          0x40000001, 0xc000ffff,
        0x0001000f, 0x8000fffe, 0x7fffffe3, 0x8001000d, 0x80010011, 0x8000fffe, 0x0001000f};
    const std::array<std::array<uint32_t, 3>, 2> raw = {
        {{0xffff0002, 0x00030004, 0x00050006}, {0x0001000f, 0x8000fffe, 0xffff0001}}};
    for (const auto& def : vector_packed_integer_defs())
        for (size_t row = 0; row < 2; ++row) {
            auto in = inputs(19);
            std::fill(in.s0.begin(), in.s0.end(), raw[row][0]);
            std::fill(in.s1.begin(), in.s1.end(), raw[row][1]);
            std::fill(in.s2.begin(), in.s2.end(), raw[row][2]);
            const uint32_t hand = (row ? row2 : row1)[def.opcode];
            ASSERT_EQ(expected(def, in, 1, {}), Known(hand));
            const auto sources = rdna2_valu_source_count(def.fmt, def.opcode);
            for (uint32_t mode = 0; mode <= sources; ++mode) {
                Form f;
                if (mode) {
                    f.src[mode - 1] = 255;
                    f.literal = raw[row][mode - 1];
                }
                for (uint32_t wave : {64u, 32u}) ASSERT_TRUE(matches(def, in, wave, f, hand));
            }
        }
}
TEST(IrVectorPackedIntegerExecution, EveryLowHighSelectorPairKeepsItsOwnOperandHalves) {
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t lo = 0; lo < 8; ++lo)
            for (uint32_t hi = 0; hi < 8; ++hi) {
                Form f;
                f.lo = lo;
                f.hi = hi;
                for (uint32_t wave : {64u, 32u})
                    ASSERT_TRUE(matches(def, inputs(541 + lo + 8 * hi), wave, f));
            }
}
TEST(IrVectorPackedIntegerExecution, GuestDerivedNegWrapperPreservationConfidenceMed) {
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t lo = 0; lo < 8; ++lo)
            for (uint32_t hi = 0; hi < 8; ++hi) {
                Form f;
                f.neg_lo = lo;
                f.neg_hi = hi;
                f.lo = (lo + 3) & 7;
                f.hi = (hi + 5) & 7;
                for (uint32_t wave : {64u, 32u})
                    ASSERT_TRUE(matches(def, inputs(547 + lo + 8 * hi), wave, f));
            }
}
TEST(IrVectorPackedIntegerExecution, InlineWrapperPreservationPending4407ConfidenceLow) {
    // This preserves the shared helper's interpretation. It is deliberately separate from ISA
    // arithmetic proof: #4407 has not settled packed-integer inline FLOAT materialization.
    constexpr uint32_t codes[] = {128, 129, 192, 193, 208, 240, 241,
                                  242, 243, 244, 245, 246, 247, 248};
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t slot = 0; slot < rdna2_valu_source_count(def.fmt, def.opcode); ++slot)
            for (auto code : codes)
                for (uint32_t selector : {0u, 7u}) {
                    Form f;
                    f.src[slot] = code;
                    f.lo = selector;
                    f.hi = selector ^ 7u;
                    for (uint32_t wave : {64u, 32u})
                        ASSERT_TRUE(matches(def, inputs(557), wave, f));
                }
}
TEST(IrVectorPackedIntegerExecution, DestinationAliasesUseOldBothHalvesWithoutFeedback) {
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t slot = 0; slot < rdna2_valu_source_count(def.fmt, def.opcode); ++slot)
            for (uint32_t select : {0u, 3u}) {
                Form f;
                f.src[slot] = 256 + kVdst;
                f.lo = select;
                f.hi = select ^ 7u;
                for (uint32_t wave : {64u, 32u}) ASSERT_TRUE(matches(def, inputs(569), wave, f));
            }
}
TEST(IrVectorPackedIntegerExecution, EncodedIntegerClampStillRefusesWithUnclampedControls) {
    // ISA70648 §6.5 requires integer saturation. No saturated implementation is admitted here.
    for (const auto& def : vector_packed_integer_defs()) {
        Form f;
        f.clamp = true;
        const auto code = words(def.opcode, f);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_TRUE(decoded[0].has_modifier);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        for (uint32_t wave : {64u, 32u}) {
            const auto in = inputs(563);
            ASSERT_TRUE(matches(def, in, wave));
            const auto bad = run_vector_op_per_lane(code, in, wave);
            EXPECT_TRUE(bad.d.empty());
            EXPECT_NE(bad.error.find("refused:"), std::string::npos);
        }
    }
}
TEST(IrVectorPackedIntegerExecution, DefinitionsAgreeWithEffectsAndOnlyReadRealOperands) {
    ASSERT_EQ(vector_packed_integer_defs().size(), 14u);
    for (const auto& def : vector_packed_integer_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.vdst_half);
        const auto count = rdna2_valu_source_count(def.fmt, def.opcode);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> reads;
        def.build(ir, TernaryInputs(ir, reads));
        EXPECT_EQ(reads.size(), count);
        std::set<uint32_t> used(reads.begin(), reads.end());
        EXPECT_EQ(used.size(), count);
        for (uint32_t source = 0; source < 3; ++source) {
            std::array<Known, 3> in = {3u, 5u, 7u};
            in[source] = std::nullopt;
            EXPECT_EQ(bool(eval_ternary_integer_op(def, in)), source >= count);
        }
    }
    EXPECT_EQ(find_vector_packed_integer_op(Rdna2Format::VOP3P, 0x0e), nullptr);
    EXPECT_EQ(find_vector_packed_integer_op(Rdna2Format::VOP3, 0x00), nullptr);
}

TEST(IrVectorPackedIntegerExecution, ExportsEveryFormInBothWavesForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "packed-integer-witness";
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
    for (const auto& def : vector_packed_integer_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode = 0; mode < 15; ++mode) {
                Form f;
                if (mode == 1) {
                    f.lo = 5;
                    f.hi = 2;
                }
                if (mode == 2) {
                    f.neg_lo = 3;
                    f.neg_hi = 5;
                    f.lo = 1;
                    f.hi = 6;
                }
                if (mode >= 3 && mode <= 5) {
                    f.src[mode - 3] = 255;
                    f.literal = 0x8001fffe;
                }
                if (mode >= 6 && mode <= 8) f.src[mode - 6] = mode == 7 ? 193 : 135;
                if (mode >= 9 && mode <= 11) f.src[mode - 9] = 242;
                if (mode >= 12) f.src[mode - 12] = 256 + kVdst;
                const auto program = lane_program(words(def.opcode, f), 64, wave, true);
                const uint64_t address = 0x5293000000ull + count * 0x100;
                const fs::path stem =
                    dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
                std::ofstream bin(stem.string() + ".bin", std::ios::binary);
                bin.write(reinterpret_cast<const char*>(program.code.data()),
                          static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const fs::path sidecar = stem.string() + ".inputs";
                ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &program.table,
                                                         program.config,
                                                         "synthetic-witness (no live refusal)"));
                const auto read = read_refused_shader_inputs(sidecar);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(program.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
                EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                ++count;
            }
    EXPECT_EQ(count, 420u);
}

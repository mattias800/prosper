// AMD70648 §12.10 supplies the hand arithmetic; #4407 wrappers remain separate.
#include "gpu/recompiler/ir/ops/vector_integer_dot.hpp"
#include "gpu/recompiler/ir/eval.hpp"
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
    // #4407: existing f32 inline materialization, not proof of packed hardware values.
    constexpr uint32_t bits[] = {0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000,
                                 0xc0000000, 0x40800000, 0xc0800000, 0x3e22f983};
    if (code >= 240 && code <= 248) return bits[code - 240];
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
Known expected(const IntegerDotDef& def, const LaneInputs& in, size_t lane, const Form& f) {
    return eval_integer_dot_op(
        def,
        {source_word(in, lane, f, 0), source_word(in, lane, f, 1), source_word(in, lane, f, 2)},
        static_cast<uint8_t>(f.lo), static_cast<uint8_t>(f.hi));
}
::testing::AssertionResult matches(const IntegerDotDef& def, const LaneInputs& in, uint32_t wave,
                                   const Form& f = {}, Known hand = std::nullopt) {
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
TEST(IrVectorIntegerDotExecution, EveryDefinitionMatchesDistinctLanesAndExecInBothWaves) {
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 16; ++seed)
                ASSERT_TRUE(matches(def, inputs(seed * 5299), wave));
}
TEST(IrVectorIntegerDotExecution, HarnessSeesOperandChangesAndInactiveCanaries) {
    for (const auto& def : vector_integer_dot_defs()) {
        auto in = inputs(537);
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
        for (auto& v : in.s2) v ^= 0x80018001u;
        const auto changed = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(changed.error.empty()) << changed.error;
        EXPECT_NE(first.d, changed.d) << def.opcode;
        in.s2.pop_back();
        const auto bad = run_vector_op_per_lane(words(def.opcode), in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_FALSE(bad.error.empty());
    }
}
TEST(IrVectorIntegerDotExecution, HandWorkedIsaRegisterAndRawLiteralProductsAndWraparound) {
    // Dot2 signed row1:2*4+(-1)*3+0x50006=0x5000b. Unsigned uses65535*3.
    // Row2 dot4 signed:-1*2+127*(-128)+1*(-1)+(-128)*(-1)-16=-16147.
    // Row2 dot8 signed:-2-56-1+8-16=-67. All results wrap modulo2^32.
    // Row3: signed dot2=2-1, dot4=4-1, dot8=8-1; unsigned products are65535^2,
    // 255^2 or15^2, summed2/4/8times and then decremented.
    constexpr std::array<std::array<uint32_t, 6>, 3> hand = {
        {{0x0005000b, 0x0008000b, 0x0005000b, 0x0005030b, 0x0005000b, 0x0005003b},
         {0xc001ffed, 0xc000ffed, 0xffffc0ed, 0x0000c1ed, 0xffffffbd, 0x000000cd},
         {1, 0xfffc0001, 3, 0x0003f803, 7, 0x00000707}}};
    constexpr std::array<std::array<uint32_t, 3>, 3> raw = {{{0xffff0002, 0x00030004, 0x00050006},
                                                             {0x80017fff, 0xffff8002, 0xfffffff0},
                                                             {0xffffffff, 0xffffffff, 0xffffffff}}};
    for (const auto& def : vector_integer_dot_defs())
        for (size_t row = 0; row < raw.size(); ++row) {
            auto in = inputs(23);
            std::fill(in.s0.begin(), in.s0.end(), raw[row][0]);
            std::fill(in.s1.begin(), in.s1.end(), raw[row][1]);
            std::fill(in.s2.begin(), in.s2.end(), raw[row][2]);
            const uint32_t want = hand[row][def.opcode - 0x14];
            ASSERT_EQ(expected(def, in, 1, {}), Known(want));
            for (uint32_t mode = 0; mode <= 3; ++mode) {
                Form f;
                if (mode) {
                    f.src[mode - 1] = 255;
                    f.literal = raw[row][mode - 1];
                }
                for (uint32_t wave : {64u, 32u}) ASSERT_TRUE(matches(def, in, wave, f, want));
            }
        }
}
TEST(IrVectorIntegerDotExecution, ExistingSelectorWrapperPreservesEachDot2HalfAndIgnoredFields) {
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t lo = 0; lo < 4; ++lo)
            for (uint32_t hi = 0; hi < 8; ++hi) {
                if (def.opcode >= 0x16 && lo) continue;
                Form f;
                f.lo = lo;
                f.hi = hi;
                for (uint32_t wave : {64u, 32u})
                    ASSERT_TRUE(matches(def, inputs(571 + lo + 4 * hi), wave, f));
            }
}
TEST(IrVectorIntegerDotExecution, InlineWrapperPreservationPending4407ConfidenceLow) {
    constexpr uint32_t codes[] = {128, 129, 192, 193, 208, 240, 241,
                                  242, 243, 244, 245, 246, 247, 248};
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t slot = 0; slot < rdna2_valu_source_count(def.fmt, def.opcode); ++slot)
            for (uint32_t code : codes) {
                Form f;
                f.src[slot] = code;
                for (uint32_t wave : {64u, 32u}) ASSERT_TRUE(matches(def, inputs(577), wave, f));
            }
}
TEST(IrVectorIntegerDotExecution, DestinationAliasesReadTheOldAccumulatorAndPackedSources) {
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t slot = 0; slot < rdna2_valu_source_count(def.fmt, def.opcode); ++slot) {
            Form f;
            f.src[slot] = 256 + kVdst;
            for (uint32_t wave : {64u, 32u}) ASSERT_TRUE(matches(def, inputs(587), wave, f));
        }
}
TEST(IrVectorIntegerDotExecution, EncodedUnmodeledModifiersStillRefuseWithPlainControls) {
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t mode = 0; mode < 5; ++mode) {
            if (mode == 4 && def.opcode < 0x16) continue;
            Form f;
            if (mode == 0) f.clamp = true;
            if (mode == 1) f.neg_lo = 1;
            if (mode == 2) f.neg_hi = 1;
            if (mode == 3) f.lo = 4;
            if (mode == 4) f.lo = 1;
            const auto code = words(def.opcode, f);
            std::vector<Rdna2Inst> decoded;
            ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
            ASSERT_EQ(decoded.size(), 1u);
            EXPECT_TRUE(decoded[0].has_modifier);
            for (uint32_t wave : {64u, 32u}) {
                const auto in = inputs(593);
                ASSERT_TRUE(matches(def, in, wave));
                const auto bad = run_vector_op_per_lane(code, in, wave);
                EXPECT_TRUE(bad.d.empty());
                EXPECT_NE(bad.error.find("refused:"), std::string::npos);
            }
        }
}
TEST(IrVectorIntegerDotExecution, DefinitionsAgreeWithEffectsAndRequireAllThreeSourceWords) {
    ASSERT_EQ(vector_integer_dot_defs().size(), 6u);
    for (const auto& def : vector_integer_dot_defs()) {
        const auto& effect = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(effect.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effect.vdst_dwords, 1u);
        EXPECT_FALSE(effect.vdst_half);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        for (uint32_t source = 0; source < 3; ++source) {
            std::array<Known, 3> in = {3u, 5u, 7u};
            in[source] = std::nullopt;
            EXPECT_FALSE(eval_integer_dot_op(def, in));
        }
    }
    EXPECT_EQ(find_vector_integer_dot_op(Rdna2Format::VOP3P, 0x13), nullptr);
    EXPECT_EQ(find_vector_integer_dot_op(Rdna2Format::VOP3, 0x14), nullptr);
}
TEST(IrVectorIntegerDotExecution, ExportsEveryFormInBothWavesForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "integer-dot-witness";
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
    for (const auto& def : vector_integer_dot_defs())
        for (uint32_t wave : {64u, 32u})
            for (uint32_t mode = 0; mode < 15; ++mode) {
                Form f;
                if (mode == 1 && def.opcode < 0x16) {
                    f.lo = 3;
                    f.hi = 4;
                }
                if (mode == 2) f.hi = 0;
                if (mode >= 3 && mode <= 5) {
                    f.src[mode - 3] = 255;
                    f.literal = 0x8001fffe;
                }
                if (mode >= 6 && mode <= 8) f.src[mode - 6] = mode == 7 ? 193 : 135;
                if (mode >= 9 && mode <= 11) f.src[mode - 9] = 242;
                if (mode >= 12) f.src[mode - 12] = 256 + kVdst;
                const auto program = lane_program(words(def.opcode, f), 64, wave, true);
                const uint64_t address = 0x5299000000ull + count * 0x100;
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
    EXPECT_EQ(count, 180u);
}

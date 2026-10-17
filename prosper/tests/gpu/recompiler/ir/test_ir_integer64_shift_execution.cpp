// Independent raw-word expectations and real recompiler output for the existing vector shifts.
// Each saved module stores one half; the paired modules observe both destination words.
#include "gpu/recompiler/ir/ops/integer64_shifts.hpp"
#include "gpu/recompiler/ir/integer64_shift_witness.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
std::vector<uint32_t> words(uint32_t op, uint32_t form) {
    const uint32_t count = form == 0 ? 258 : form == 1 ? 50 : form == 2 ? 161 : 255;
    std::vector<uint32_t> out;
    if (form == 1) out = {0xbeb203ff, 0x80000021}; // s_mov_b32 s50, literal
    out.insert(out.end(), {0xd4000001u | (op << 16), count | ((op == 0x2ff ? 129u : 261u) << 9)});
    if (form == 3) out.push_back(0x80000021);
    return out;
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t counts[] = {0, 1, 31, 32, 33, 63, 64, 65, 0xffffffff};
    uint32_t state = seed;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        state = state * 1664525u + 1013904223u;
        in.s0[lane] = lane < std::size(counts) ? counts[lane] : state;
        in.s2.push_back(state ^ 0x9e3779b9);
        in.s3.push_back((state << 13) | (state >> 19));
    }
    return in;
}
LaneProgram program(uint32_t op, uint32_t form, uint32_t wave, uint32_t half) {
    auto lp = lane_program(words(op, form), 64, wave, true, true);
    if (half) lp.code[lp.code.size() - 2] = 0x80020200; // store v2 instead of v1
    return lp;
}
void check_module(const std::vector<uint32_t>& spv, const LaneInputs& in, uint32_t op,
                  uint32_t form, uint32_t wave, uint32_t half) {
    const auto run = execute_integer64_shift_module(spv, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), 64u);
    for (size_t lane = 0; lane < 64; ++lane) {
        const uint32_t count = (form == 0 ? in.s0[lane] : 33) & 63;
        const uint64_t source =
            op == 0x2ff ? 1 : uint64_t(in.s2[lane]) | (uint64_t(in.s3[lane]) << 32);
        const uint64_t right = op == 0x301 && count && (source >> 63)
                                   ? (source >> count) | (UINT64_MAX << (64 - count))
                                   : source >> count;
        const uint64_t result = op == 0x2ff ? source << count : right;
        const uint32_t want = uint32_t(result >> (half * 32));
        EXPECT_EQ(run.d[lane], in.active[lane] ? want
                               : half          ? in.s0[lane]
                                               : in.old_d[lane])
            << "op " << op << " form " << form << " wave " << wave << " half " << half << " lane "
            << lane;
    }
}
void check(uint32_t op, uint32_t form, uint32_t wave, uint32_t half, uint32_t seed) {
    const auto in = inputs(seed);
    const auto lp = program(op, form, wave, half);
    const auto spv = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
    check_module(spv, in, op, form, wave, half);
}

} // namespace
TEST(IrInteger64ShiftExecution, ExistingFormsObserveBothHalvesAndInactiveCanariesInBothWaves) {
    for (const uint32_t op : {0x2ffu, 0x300u, 0x301u})
        for (const uint32_t form : {0u, 1u, 2u, 3u})
            for (const uint32_t wave : {32u, 64u})
                for (const uint32_t half : {0u, 1u})
                    for (const uint32_t seed : {17u, 47u, 71u}) check(op, form, wave, half, seed);
}
TEST(IrInteger64ShiftExecution, EveryValueDefinitionMasksCountsAndPreservesUnknownInputs) {
    for (const auto& def : integer64_shift_defs()) {
        const bool left = def.opcode == 0x2ff || def.opcode == 0x1f;
        const bool arithmetic = def.opcode == 0x301 || def.opcode == 0x23;
        for (const uint32_t count : {0u, 32u, 63u, 64u, 65u, UINT32_MAX}) {
            const uint64_t word = 0x8000000000000001ull;
            const uint32_t n = count & 63;
            const uint64_t right =
                arithmetic && n ? (word >> n) | (UINT64_MAX << (64 - n)) : word >> n;
            EXPECT_EQ(eval_integer64_shift(def, 1, 0x80000000, count),
                      Known64(left ? word << n : right));
        }
        EXPECT_FALSE(eval_integer64_shift(def, std::nullopt, 0, 1));
        EXPECT_FALSE(eval_integer64_shift(def, 0, std::nullopt, 1));
        EXPECT_FALSE(eval_integer64_shift(def, 0, 0, std::nullopt));
    }
}
TEST(IrInteger64ShiftExecution, ExportsExistingFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    const fs::path root = compare_witness_root("integer64-shift-witness");
    // NOLINTNEXTLINE(concurrency-mt-unsafe): launch environment is immutable.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (const uint32_t op : {0x2ffu, 0x300u, 0x301u})
        for (const uint32_t wave : {32u, 64u})
            for (const uint32_t half : {0u, 1u})
                for (uint32_t form = 0; form < 4; ++form) {
                    const auto lp = program(op, form, wave, half);
                    const auto spv = saved_integer64_shift_witness(
                        dump, 0x6450000000ull + count * 0x100, count, lp);
                    check_module(spv, inputs(17), op, form, wave, half);
                    ++count;
                }
    EXPECT_EQ(count, 48u);
}

TEST(IrInteger64ShiftExecution, APreviouslyUnwrittenHighDestinationWorksUnderFullExec) {
    for (uint32_t op : {0x2ffu, 0x300u, 0x301u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t half : {0u, 1u}) {
                auto in = inputs(397);
                std::fill(in.active.begin(), in.active.end(), true);
                auto op_words = words(op, 0);
                op_words[0] = (op_words[0] & ~255u) | 24; // v[24:25] has no initial load
                op_words.push_back(vop1(1, 1, 256 + 24 + half));
                auto lp = lane_program(op_words, 64, wave, true, true);
                lp.code.erase(lp.code.begin() + 13); // six two-word loads, waitcnt, then CMPX
                const auto spv =
                    recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
                check_module(spv, in, op, 0, wave, half);
            }
}
TEST(IrInteger64ShiftExecution, SourceDestinationPairOverlapReadsBothOldWordsFirst) {
    for (uint32_t op : {0x300u, 0x301u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t half : {0u, 1u}) {
                const auto in = inputs(419);
                auto lp = lane_program({0xd4000001u | (op << 16), 259u | (257u << 9)}, 64, wave);
                if (half) lp.code[lp.code.size() - 2] = 0x80020200;
                const auto spv =
                    recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config);
                const auto run = execute_integer64_shift_module(spv, in);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const uint64_t old = uint64_t(in.old_d[lane]) | (uint64_t(in.s0[lane]) << 32);
                    const uint32_t n = in.s1[lane] & 63;
                    const uint64_t result = op == 0x301 && n && (old >> 63)
                                                ? (old >> n) | (UINT64_MAX << (64 - n))
                                                : old >> n;
                    EXPECT_EQ(run.d[lane], in.active[lane] ? uint32_t(result >> (32 * half))
                                           : half          ? in.s0[lane]
                                                           : in.old_d[lane]);
                }
            }
}

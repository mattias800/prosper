// #5436 / #5439 / #5441: a scalar-data pair written in a structured if arm must be phi'd at the
// merge in BOTH words. The structured emitter's phi scan charged the pair only for s_cselect_b64,
// s_lshl_b64 and s_lshr_b64 among the SOP2 rows (#5436: s_bfe_u64, s_ashr_i64 and s_bfm_b64 lost
// their high word), and one word for every SOP1 row (#5439: s_mov_b64 and s_bitreplicate_b64_b32
// lost theirs). The high SGPR kept the arm's SSA value, which does not dominate the merge: a read
// of it after the merge took the arm's value on the edge that never ran it.
//
// Each program runs the production emitter (emit_body) and executes the emitted module per lane
// on the closed SPIR-V interpreter, which is a value oracle, not a live title or GPU claim. Every
// lane has its own sources and its own pre-branch s[6:7], and the expected values are computed
// here from the ISA's pseudo-code (RDNA2 ISA 70648, SOP2 and SOP1), not from the recompiler's
// helpers.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/sop1_phi_witness.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kBfeU64 = 0x29, kAshrI64 = 0x23, kBfmB64 = 0x25;
constexpr uint32_t kMovB64 = 0x04, kBitreplicateB64B32 = 0x3b;
constexpr uint32_t kWqmB64 = 0x0a, kQuadmaskB64 = 0x2d;
constexpr uint32_t kEntry6 = 0x600dcafeu, kEntry7 = 0x7ee1f00du;

// SOP2 `op sdst, ssrc0, ssrc1` with SGPR operands.
constexpr uint32_t sop2(uint32_t op, uint32_t sdst, uint32_t ssrc0, uint32_t ssrc1) {
    return 0x80000000u | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0;
}

// SOP1 `op sdst, ssrc0`.
constexpr uint32_t sop1(uint32_t op, uint32_t sdst, uint32_t ssrc0) {
    return 0xbe800000u | (sdst << 16) | (op << 8) | ssrc0;
}

// The pair op under test into s[6:7]. s_bfe_u64 and s_ashr_i64 read the data pair s[4:5] and a
// 32-bit control; s_bfm_b64 reads two 32-bit operands (width, offset).
constexpr uint32_t pair_op(uint32_t op, uint32_t control, uint32_t other) {
    return op == kBfmB64 ? sop2(op, 6, control, other) : sop2(op, 6, 4, control);
}

// The per-lane launch values the program starts from (set in compile() below).
struct LaneValues {
    uint32_t s4, s5, s8, s9, s6, s7;
};
LaneValues lane_values(uint32_t lane) {
    return {lane ^ 0x89abcdefu,
            (lane << 26) ^ 0x70e1d2c3u,              // both signs of the high word
            lane | (((lane ^ 0x2au) | 1u) << 16),     // bfe: offset = lane, width odd, nonzero
            (lane ^ 0x1fu) | ((lane | 1u) << 16),     // the else arm's control
            lane ^ kEntry6,
            lane ^ kEntry7};
}

// RDNA2 ISA 70648 SOP2 pseudo-code.
uint64_t reference(uint32_t op, uint32_t control, uint32_t other, const LaneValues& v) {
    const uint64_t data = (uint64_t(v.s5) << 32) | v.s4;
    switch (op) {
        case kBfeU64: {   // D.u64 = (S0.u64 >> S1.u[5:0]) & ((1 << S1.u[22:16]) - 1)
            const uint32_t offset = control & 63u, width = (control >> 16) & 127u;
            const uint64_t mask = width >= 64 ? ~0ull : (1ull << width) - 1ull;
            return (data >> offset) & mask;
        }
        case kAshrI64:   // D.i64 = signext(S0.i64) >> S1.u[5:0]
            return uint64_t(int64_t(data) >> (control & 63u));
        case kBfmB64:   // D.u64 = ((1ULL << S0.u[5:0]) - 1) << S1.u[5:0]
            return ((1ull << (control & 63u)) - 1ull) << (other & 63u);
    }
    ADD_FAILURE() << "no reference for op " << op;
    return 0;
}

// RDNA2 ISA 70648 SOP1 S_BITREPLICATE_B64_B32: D.u64[2i] = D.u64[2i + 1] = S0.u32[i].
uint64_t bitreplicate(uint32_t source) {
    uint64_t out = 0;
    for (uint32_t bit = 0; bit < 32; ++bit)
        if ((source >> bit) & 1u) out |= 3ull << (2u * bit);
    return out;
}

// One pair writer: its then-arm and else-arm encodings, and what each leaves in s[6:7].
struct PairCase {
    const char* name;
    uint32_t then_arm, else_arm;
    uint64_t (*expected)(const LaneValues&, bool else_arm);
};

template <uint32_t Op>
uint64_t sop2_expected(const LaneValues& v, bool else_arm) {
    return else_arm ? reference(Op, v.s9, v.s8, v) : reference(Op, v.s8, v.s9, v);
}

constexpr PairCase kSop2Cases[] = {
    {"s_bfe_u64", pair_op(kBfeU64, 8, 9), pair_op(kBfeU64, 9, 8), sop2_expected<kBfeU64>},
    {"s_ashr_i64", pair_op(kAshrI64, 8, 9), pair_op(kAshrI64, 9, 8), sop2_expected<kAshrI64>},
    {"s_bfm_b64", pair_op(kBfmB64, 8, 9), pair_op(kBfmB64, 9, 8), sop2_expected<kBfmB64>},
};

// gtest prints a parameter that fails; name it rather than dump its bytes.
void PrintTo(const PairCase& c, std::ostream* os) {
    *os << c.name;
}

// SOP1 rows whose lowering writes both words as scalar data. S_MOV_B64: D.u64 = S0.u64; an
// inline integer is sign-extended to 64 bits (-5 is SSRC0 197, 7 is SSRC0 135).
constexpr PairCase kSop1Cases[] = {
    {"s_mov_b64", sop1(kMovB64, 6, 4), sop1(kMovB64, 6, 8),
     [](const LaneValues& v, bool else_arm) {
         return else_arm ? (uint64_t(v.s9) << 32) | v.s8 : (uint64_t(v.s5) << 32) | v.s4;
     }},
    {"s_mov_b64_inline", sop1(kMovB64, 6, 197u), sop1(kMovB64, 6, 135u),
     [](const LaneValues&, bool else_arm) { return else_arm ? uint64_t{7} : uint64_t(-5ll); }},
    {"s_bitreplicate_b64_b32", sop1(kBitreplicateB64B32, 6, 8), sop1(kBitreplicateB64B32, 6, 9),
     [](const LaneValues& v, bool else_arm) { return bitreplicate(else_arm ? v.s9 : v.s8); }},
    {"s_wqm_b64_numeric", sop1(kWqmB64, 6, 129u), sop1(kWqmB64, 6, 197u),
     [](const LaneValues&, bool else_arm) { return else_arm ? ~uint64_t{0} : uint64_t{15}; }},
    {"s_quadmask_b64", sop1(kQuadmaskB64, 6, 126u), sop1(kQuadmaskB64, 6, 106u),
     [](const LaneValues&, bool else_arm) {
         return else_arm ? uint64_t{0xaaaa} : uint64_t{0x5555};
     }},
};

// s_cmp_eq_u32 s0, 0 | s_cbranch_scc1 <past the then arm> | then arm [| s_branch | else arm]
// | s_mov_b32 s11, s6 | s_mov_b32 s10, s7 | s_endpgm. The arm runs when s0 != 0; the reads after
// the merge are guest instructions, so they see exactly what the merge left in s6 and s7.
std::vector<uint32_t> program(const PairCase& c, bool has_else) {
    std::vector<uint32_t> code{0xbf068000u, has_else ? 0xbf850002u : 0xbf850001u, c.then_arm};
    if (has_else) code.insert(code.end(), {0xbf820001u, c.else_arm});
    code.insert(code.end(), {0xbe8b0306u, 0xbe8a0307u, 0xbf810000u});
    return code;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint32_t selector) {
    const bool quadmask = ((code[2] >> 8) & 255u) == kQuadmaskB64;
    const bool wqm = ((code[2] >> 8) & 255u) == kWqmB64;
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
    SpirvCompute b;
    b.diagnostic = {RecompileDiagnosticStage::Compute, 0x54360001u};
    b.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    if (quadmask) b.native_subgroup_size = 64;
    RegState state;
    // One active lane in alternating quads, different EXEC/VCC quads. SALU executes even
    // for EXEC-masked lanes; after-merge stores are unconditional interpreter observations.
    const auto quad_lane = b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(7));
    state.exec = b.ucmp(Op_IEqual, quad_lane, b.uconst(0));
    state.vcc = b.ucmp(Op_IEqual, quad_lane, b.uconst(4));
    if (!quadmask && !wqm) {
        state.exec = b.btrue();
        state.vcc = b.bfalse();
    }
    state.scc = b.bfalse();
    state.absent_sgpr_reads_fabricated_zero = true;
    const uint32_t lane = b.linear_localid;
    const auto shl = [&](uint32_t value, uint32_t by) {
        return b.ibin(Op_ShiftLeftLogical, value, b.uconst(by));
    };
    const auto bxor = [&](uint32_t value, uint32_t k) {
        return b.ibin(Op_BitwiseXor, value, b.uconst(k));
    };
    const auto bor = [&](uint32_t x, uint32_t y) { return b.ibin(Op_BitwiseOr, x, y); };
    state.sreg[0] = b.uconst(selector);
    state.sreg[4] = bxor(lane, 0x89abcdefu);
    state.sreg[5] = bxor(shl(lane, 26), 0x70e1d2c3u);
    state.sreg[8] = bor(lane, shl(bor(bxor(lane, 0x2au), b.uconst(1)), 16));
    state.sreg[9] = bor(bxor(lane, 0x1fu), shl(bor(lane, b.uconst(1)), 16));
    state.sreg[6] = bxor(lane, kEntry6);
    state.sreg[7] = bxor(lane, kEntry7);
    if (quadmask) {
        // Both DATA and the saved-mask view describe one incoming architectural pair.
        state.sreg[6] = b.uconst(kEntry6);
        state.sreg[7] = b.uconst(kEntry7);
        const auto word =
            b.sel(b.ucmp(Op_ULessThan, lane, b.uconst(32)), state.sreg[6], state.sreg[7]);
        const auto bit =
            b.ibin(Op_ShiftRightLogical, word, b.ibin(Op_BitwiseAnd, lane, b.uconst(31)));
        state.sreg_bool[6] =
            b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, bit, b.uconst(1)), b.uconst(0));
    }
    if (!emit_body(
            b, state, ins, {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return false; }, code.data(), code.size(), nullptr,
            false))
        return {};
    b.store_output_word(sreg_seed(b, state, 11), 2u, 0u);
    b.store_output_word(sreg_seed(b, state, 10), 2u, 1u);
    return b.finish();
}

// selector != 0 runs the then arm; with an else, selector == 0 runs the else arm, and without
// one the pre-branch s[6:7] must survive.
void expect_merge(const PairCase& c, bool has_else, uint32_t selector) {
    SCOPED_TRACE(std::string(c.name) + (has_else ? " if/else" : " if") + " selector " +
                 std::to_string(selector));
    const auto module = compile(program(c, has_else), selector);
    ASSERT_FALSE(module.empty()) << "the structured emitter refused the program";
    // Opt-in module receipts for the native QUADMASK value oracle. End-to-end captures below
    // cover WQM; native saved-mask word availability outside entry remains a separate guard.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_SOP1_PHI_WITNESS_DIR");
    if (configured && c.then_arm == sop1(kQuadmaskB64, 6, 126)) {
        std::filesystem::create_directories(configured);
        const auto file =
            std::filesystem::path(configured) /
            ("quadmask_" + std::to_string(has_else) + "_" + std::to_string(selector) + ".spv");
        std::ofstream out(file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(module.data()), module.size() * 4);
        out.close();
        ASSERT_TRUE(out.good());
    }
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    const auto actual = vm.run_buffers(64u, {{1u, std::vector<uint32_t>(128u, 0xdeadbeefu)}}, 1u);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(actual.size(), 128u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const LaneValues v = lane_values(lane);
        uint64_t expected = (uint64_t(v.s7) << 32) | v.s6;
        if (c.then_arm == sop1(kQuadmaskB64, 6, 126))
            expected = (uint64_t(kEntry7) << 32) | kEntry6;
        if (selector)
            expected = c.expected(v, false);
        else if (has_else)
            expected = c.expected(v, true);
        EXPECT_EQ(actual[size_t(2) * lane], uint32_t(expected)) << "low word, lane " << lane;
        EXPECT_EQ(actual[size_t(2) * lane + 1], uint32_t(expected >> 32))
            << "high word, lane " << lane;
    }
}

class IfMergeScalarPairWrites : public testing::TestWithParam<PairCase> {};

TEST_P(IfMergeScalarPairWrites, SkippedArmKeepsThePreBranchPair) {
    expect_merge(GetParam(), false, 0u);
}

TEST_P(IfMergeScalarPairWrites, TakenArmPublishesBothWords) {
    expect_merge(GetParam(), false, 1u);
}

TEST_P(IfMergeScalarPairWrites, IfElseMergesBothWordsFromEachArm) {
    expect_merge(GetParam(), true, 0u);
    expect_merge(GetParam(), true, 1u);
}

std::string case_name(const testing::TestParamInfo<PairCase>& info) {
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(Sop2PairResults, IfMergeScalarPairWrites, testing::ValuesIn(kSop2Cases),
                         case_name);
INSTANTIATE_TEST_SUITE_P(Sop1PairResults, IfMergeScalarPairWrites, testing::ValuesIn(kSop1Cases),
                         case_name);

TEST(Sop1PhiWitness, RuntimeBranchReplaysAndKeepsBothWordsWithPartialExec) {
    namespace fs = std::filesystem;
    namespace witness = prosper::test::sop1_phi;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_SOP1_PHI_WITNESS_DIR");
    const auto root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "sop1-phi-witness";
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump) || fs::is_directory(dump));
    uint64_t address = 0x54410000;
    for (bool has_else : {false, true}) {
        auto p = witness::witness(has_else);
        const auto stem = "cs_" + std::to_string(address) + "_0";
        std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
        bin.write(reinterpret_cast<const char*>(p.code.data()), p.code.size() * 4);
        bin.close();
        ASSERT_TRUE(bin.good());
        const auto inputs = dump / (stem + ".inputs");
        ASSERT_TRUE(write_refused_compute_inputs(inputs, address++, &p.table, p.config,
                                                 "synthetic-sop1-phi-witness"));
        const auto read = read_refused_shader_inputs(inputs);
        ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
        const auto replay = replay_refused_shader_inputs(p.code, read);
        ASSERT_TRUE(replay.produced) << replay.reject;
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
        spv.write(reinterpret_cast<const char*>(replay.spirv.data()), replay.spirv.size() * 4);
        spv.close();
        ASSERT_TRUE(spv.good());
        for (uint32_t selector : {0u, 1u}) {
            SCOPED_TRACE(stem + " selector=" + std::to_string(selector));
            bpermute_oracle::Interpreter vm(replay.spirv);
            vm.native_subgroup_width = 64;
            const auto actual = vm.run_buffers(64, {{3, witness::buffer()}}, 3, selector);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            ASSERT_EQ(actual.size(), 256u);
            const auto wanted = witness::expected(has_else, selector);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                EXPECT_EQ(actual[lane], uint32_t(wanted)) << "low, lane " << lane;
                EXPECT_EQ(actual[64 + lane], uint32_t(wanted >> 32)) << "high, lane " << lane;
            }
        }
    }
}

TEST(Sop1PhiWitness, DeadShiftPoisonIsTrackedAndCannotReachStorage) {
    for (uint32_t live : {0u, 1u, 2u}) {
        SpirvCompute b;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        const auto shift = b.ibin(Op_ShiftRightLogical, b.uconst(1), b.uconst(32));
        const auto condition = b.land(b.bfalse(), b.ucmp(Op_INotEqual, shift, b.uconst(0)));
        const auto selected = b.sel(condition, b.uconst(1), b.uconst(2));
        b.store_output_word(live == 1 ? shift : live == 2 ? selected : b.uconst(7), 1, 0);
        bpermute_oracle::Interpreter vm(b.finish());
        const auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(64)}}, 1);
        if (live)
            EXPECT_EQ(vm.error, "poison live storage write") << "live=" << live;
        else {
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            EXPECT_EQ(actual, std::vector<uint32_t>(64, 7));
        }
    }
}
}   // namespace

// #4935: CFG votes/reductions bypass emit_alu. Compare the actual public compiler's modules,
// not storage membership: a zero LOW plus nonzero HIGH is a nonzero B64 source.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/portable_mask_common.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace {
namespace f = prosper::test::portable_mask_slots;
namespace c = prosper::test::portable_mask_common;
namespace base = prosper::test::portable_mask_half;
void observe(const std::vector<uint32_t>& code, const char* name, uint32_t expected,
             [[maybe_unused]] bool vopc = false, bool must_refuse = false) {
    SCOPED_TRACE(name);
    const auto attempt = f::compile(code, true);
    if (must_refuse) {
        EXPECT_TRUE(attempt.module.empty()) << "incomplete half admitted to branch";
        return;
    }
    ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
    bpermute_oracle::Interpreter oracle(attempt.module);
    const auto actual = oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_EQ(actual.size(), 64u);
    uint32_t wrong = 0;
    for (uint32_t lane = 0; lane < 64; ++lane) wrong += actual[lane] != expected;
    EXPECT_EQ(wrong, 0u);
}
void values(const std::vector<uint32_t>& code, const std::vector<uint32_t>& expected) {
    const auto attempt = f::compile(code, true);
    ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
    bpermute_oracle::Interpreter oracle(attempt.module);
    const auto actual = oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    EXPECT_EQ(actual, expected);
    for (size_t pc = 5; pc < attempt.module.size();) {
        ASSERT_NE(attempt.module[pc] >> 16, 0u);
        EXPECT_NE(attempt.module[pc] & 0xffffu, 339u) << "no physical ballot authority";
        pc += attempt.module[pc] >> 16;
    }
}
void refuse(const std::vector<uint32_t>& code, const char* reason) {
    const auto attempt = f::compile(code, true);
    EXPECT_TRUE(attempt.module.empty());
    EXPECT_NE(attempt.refusal.find(reason), std::string::npos) << attempt.refusal;
}
void local_values(const std::vector<uint32_t>& code, uint32_t local, uint32_t expected,
                  bool dispatcher = true) {
    const auto attempt = c::compile(code, local, dispatcher);
    ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
    bpermute_oracle::Interpreter oracle(attempt.module);
    const auto actual =
        oracle.run_buffers(local, {{3, std::vector<uint32_t>(local, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    EXPECT_EQ(actual, std::vector<uint32_t>(local, expected));
}
void partial_refuses(uint32_t local) {
    const std::vector<std::vector<uint32_t>> programs{
        c::branch(c::halves(false, true)),
        c::branch(c::halves(false, true), 20, true),
        c::reduction(c::halves(false, true), 0x10),
        c::reduction(c::halves(false, true), 0x14),
        c::ffbh(c::selected_half(true, 0x1000u), 21),
        c::mbcnt(c::selected_half(true, 0xfffu), 21, true)};
    for (size_t index = 0; index < programs.size(); ++index) {
        SCOPED_TRACE(index);
        const auto attempt = c::compile(programs[index], local);
        EXPECT_TRUE(attempt.module.empty());
        EXPECT_NE(attempt.refusal.find("portable-mask-half-incomplete-wave"), std::string::npos)
            << attempt.refusal;
    }
}
} // namespace
// These nine behavioral assertions and expected values are unchanged from the independent
// exact-5dd probe. Its tenth test authenticated that historical ELF, not compiler semantics.
TEST(PortableMaskCommon, SopcReverseLowZeroHighFull) {
    observe(c::branch(c::halves(false, true)), "sopc_reverse_low_zero_high_full", 9);
}
TEST(PortableMaskCommon, SopcFullPairControl) {
    observe(c::branch(c::halves(true, true)), "sopc_full_pair_control", 9);
}
TEST(PortableMaskCommon, SopcEmptyPairControl) {
    observe(c::branch(c::halves(false, false)), "sopc_empty_pair_control", 0);
}
TEST(PortableMaskCommon, SopcOnlyLowCannotAuthorizePair) {
    observe(c::branch(c::missing(true)), "sopc_missing_high", 0, false, true);
}
TEST(PortableMaskCommon, SopcGenuineBoolControl) {
    std::vector<uint32_t> code{0x7e060280u, base::mov(126, 193, true), base::mov(20, 126, true)};
    observe(c::branch(code), "sopc_genuine_bool_control", 9);
}
TEST(PortableMaskCommon, SopcGenuineNumericControl) {
    std::vector<uint32_t> code{0x7e060280u, base::mov(20, 193, true)};
    observe(c::branch(code), "sopc_genuine_numeric_control", 9);
}
TEST(PortableMaskCommon, VopcReverseLowZeroHighFull) {
    observe(c::branch(c::halves(false, true), 20, true), "vopc_reverse_low_zero_high_full", 9,
            true);
}
TEST(PortableMaskCommon, VopcFullPairControl) {
    observe(c::branch(c::halves(true, true), 20, true), "vopc_full_pair_control", 9, true);
}
TEST(PortableMaskCommon, VopcOnlyLowCannotAuthorizePair) {
    observe(c::branch(c::missing(true), 20, true), "vopc_missing_high", 0, true, true);
}
TEST(PortableMaskCommon, BothReloadOrdersHaveTheSameB64Vote) {
    observe(c::branch(c::halves(false, true, false)), "low_before_high", 9);
    observe(c::branch(c::halves(true, false, false)), "high_zero_low_full", 9);
}
TEST(PortableMaskCommon, RelayedWordsKeepTheirCommonVote) {
    auto code = c::halves(false, true);
    code.insert(code.end(), {base::mov(24, 20), base::mov(25, 21)});
    observe(c::branch(code, 24), "relayed_pair", 9);
}
TEST(PortableMaskCommon, ExecComparisonObservesTheCurrentHighProducer) {
    observe(c::branch(c::halves(true, false), 20, false, 126), "exec_saved_pair", 9);
    observe(c::branch(c::halves(true, true), 20, false, 126), "exec_full_control", 0);
}
TEST(PortableMaskCommon, SavedPairComparisonObservesTheCurrentHighProducer) {
    auto code = c::halves(true, false);
    code.insert(code.end(), {base::mov(126, 193, true), base::mov(24, 126, true)});
    observe(c::branch(code, 20, false, 24), "two_saved_pairs", 9);
}
TEST(PortableMaskCommon, MissingLowCannotAuthorizeTheB64Vote) {
    refuse(c::branch(c::missing(false)), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskCommon, MissingHighCannotAuthorizeExecComparison) {
    refuse(c::branch(c::missing(true), 20, false, 126), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskCommon, BcntComposesBothProducersInEitherReloadOrder) {
    for (bool reverse : {false, true}) {
        observe(c::reduction(c::halves(false, true, reverse), 0x10), "bcnt_high", 32);
        observe(c::reduction(c::halves(true, false, reverse), 0x10), "bcnt_low", 32);
    }
}
TEST(PortableMaskCommon, BcntFullAndEmptyControls) {
    observe(c::reduction(c::halves(true, true), 0x10), "bcnt_full", 64);
    observe(c::reduction(c::halves(false, false), 0x10), "bcnt_empty", 0);
}
TEST(PortableMaskCommon, BcntMissingHighIsNotACompleteSource) {
    refuse(c::reduction(c::missing(true), 0x10), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskCommon, Ff1NormalizesTheHighWordBitPosition) {
    observe(c::reduction(c::halves(false, true), 0x14), "ff1_high", 32);
    observe(c::reduction(c::halves(true, false), 0x14), "ff1_low", 0);
}
TEST(PortableMaskCommon, Ff1EmptyControl) {
    observe(c::reduction(c::halves(false, false), 0x14), "ff1_empty", UINT32_MAX);
}
TEST(PortableMaskCommon, Ff1MissingLowIsNotACompleteSource) {
    refuse(c::reduction(c::missing(false), 0x14), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskCommon, FfbhSelectsTheActualHighWordBits) {
    observe(c::ffbh(c::selected_half(true, 0x80000000u), 21), "ffbh_high_bit31", 0);
    observe(c::ffbh(c::selected_half(true, 0x00001000u), 21), "ffbh_high_bit12", 19);
}
TEST(PortableMaskCommon, FfbhSelectsTheActualLowWordBits) {
    observe(c::ffbh(c::selected_half(false, 0x00001000u), 20), "ffbh_low_bit12", 19);
}
TEST(PortableMaskCommon, FfbhEmptyWordControl) {
    observe(c::ffbh(c::selected_half(true, 0), 21), "ffbh_empty_high", UINT32_MAX);
}
TEST(PortableMaskCommon, FfbhCannotBorrowTheOtherReloadedWord) {
    refuse(c::ffbh(c::missing(true), 21), "portable-mask-half-projection-unavailable");
}
TEST(PortableMaskCommon, MbcntHighUsesItsSelectedWordAndBitPosition) {
    std::vector<uint32_t> expected(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
        expected[lane] = 7u + (lane <= 32 ? 0u : lane >= 44 ? 12u : lane - 32u);
    values(c::mbcnt(c::selected_half(true, 0xfffu), 21, true), expected);
}
TEST(PortableMaskCommon, MbcntLowUsesItsSelectedWordAndBitPosition) {
    constexpr uint32_t bits = 0xaaaa5555u;
    std::vector<uint32_t> expected(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
        expected[lane] = 7u + static_cast<uint32_t>(std::popcount(
                                  bits & (lane >= 32 ? UINT32_MAX : (1u << lane) - 1u)));
    values(c::mbcnt(c::selected_half(false, bits), 20, false), expected);
}
TEST(PortableMaskCommon, MbcntEmptySelectedWordControl) {
    values(c::mbcnt(c::selected_half(true, 0), 21, true), std::vector<uint32_t>(64, 7));
}
TEST(PortableMaskCommon, MbcntCannotUseHighProjectionAsLowData) {
    refuse(c::mbcnt(c::selected_half(true, 0xfffu), 21, false),
           "portable-mask-half-projection-unavailable");
}
TEST(PortableMaskCommon, MbcntCannotUseASymbolicAccumulatorAsData) {
    auto code = c::selected_half(false, UINT32_MAX);
    code.insert(code.end(), {0xd7650003u, 20u | (20u << 9)});
    c::store(code);
    refuse(code, "portable-mask-half-projection-unavailable");
}

TEST(PortableMaskCommon, MbcntCannotBorrowAMissingHighWord) {
    refuse(c::mbcnt(c::missing(true), 21, true), "portable-mask-half-projection-unavailable");
}

TEST(PortableMaskCommon, OracleDeadUndefinedSelectDoesNotReachTheSink) {
    bpermute_oracle::Interpreter oracle(c::poison_select(false));
    const auto result = oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef));
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_EQ(result.size(), 64u);
    EXPECT_EQ(result[0], 9u);
}
TEST(PortableMaskCommon, OracleObservedUndefinedSelectStillRefuses) {
    bpermute_oracle::Interpreter oracle(c::poison_select(true));
    EXPECT_TRUE(oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef)).empty());
    EXPECT_NE(oracle.error.find("poison live storage write"), std::string::npos) << oracle.error;
}
TEST(PortableMaskCommon, OracleUndefinedConditionCannotSteerControlFlow) {
    bpermute_oracle::Interpreter oracle(c::poison_select(false, true));
    EXPECT_TRUE(oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef)).empty());
    EXPECT_NE(oracle.error.find("not defined bool value"), std::string::npos) << oracle.error;
}
TEST(PortableMaskCommon, OracleUnsupportedDefinitionIsNotPoisonAuthority) {
    bpermute_oracle::Interpreter oracle(c::poison_select(false, false, true));
    EXPECT_TRUE(oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef)).empty());
    EXPECT_NE(oracle.error.find("unsupported live opcode 156"), std::string::npos) << oracle.error;
}
TEST(PortableMaskCommon, OracleMissingDefinitionIsNotPoisonAuthority) {
    auto module = c::poison_select(false);
    for (size_t pc = 5; pc < module.size(); pc += module[pc] >> 16)
        if ((module[pc] & 0xffffu) == 169) module[pc + 3] = 13; // no definition for this ID
    bpermute_oracle::Interpreter oracle(module);
    EXPECT_TRUE(oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef)).empty());
    EXPECT_NE(oracle.error.find("unknown or poison value 13"), std::string::npos) << oracle.error;
}

TEST(PortableMaskCommon, Local32CannotSupplyUnlaunchedSymbolicBits) {
    partial_refuses(32);
}
TEST(PortableMaskCommon, Local65CannotSupplyTrailingWaveSymbolicBits) {
    partial_refuses(65);
}
TEST(PortableMaskCommon, CompleteWaveKeepsBothSymbolicComparisonRoutes) {
    local_values(c::branch(c::halves(false, true)), 64, 9);
    local_values(c::branch(c::halves(false, true), 20, true), 64, 9);
}
TEST(PortableMaskCommon, GenuineNumericControlKeepsPartialWaveRoutes) {
    for (uint32_t local : {32u, 65u})
        local_values(c::branch({0x7e060280u, base::mov(20, 193, true)}), local, 9);
}
TEST(PortableMaskCommon, LegacyWholeBoolControlKeepsPartialWaveRoutes) {
    for (uint32_t local : {32u, 65u})
        local_values(c::branch({0x7e060280u, base::mov(126, 193, true), base::mov(20, 126, true)}),
                     local, 9);
}
TEST(PortableMaskCommon, BarrierBeforeReloadKeepsTheCurrentPair) {
    auto code = c::reduction(c::halves(false, true), 0x10);
    code.insert(code.begin() + 7, 0xbf8a0000u);
    local_values(code, 64, 32, false);
}
TEST(PortableMaskCommon, BarrierAfterReloadKeepsTheCurrentPair) {
    auto code = c::reduction(c::halves(false, true), 0x10);
    code.insert(code.begin() + 11, 0xbf8a0000u);
    local_values(code, 64, 32, false);
}

TEST(PortableMaskCommon, ExactPaddedWorkgroupCannotSupplySymbolicBits) {
    for (uint32_t threads : {32u, 65u}) {
        const auto config = c::exact_config(64, 1, 1, threads, 1, 1);
        for (const auto& code :
             {c::branch(c::halves(false, true)), c::branch(c::halves(false, true), 20, true),
              c::reduction(c::halves(false, true), 0x10),
              c::reduction(c::halves(false, true), 0x14),
              c::ffbh(c::selected_half(true, 0x1000u), 21),
              c::mbcnt(c::selected_half(true, 0xfffu), 21, true)}) {
            const auto attempt = c::compile(c::readfirstlane(code), config);
            EXPECT_TRUE(attempt.module.empty());
            EXPECT_NE(attempt.refusal.find("portable-mask-half-incomplete-wave"), std::string::npos)
                << attempt.refusal;
        }
    }
}
TEST(PortableMaskCommon, MultidimensionalExtentDoesNotFlattenParticipation) {
    for (const auto& config :
         {c::exact_config(32, 2, 1, 16, 4, 1), c::exact_config(64, 1, 1, 32, 2, 1)}) {
        const auto attempt =
            c::compile(c::readfirstlane(c::branch(c::halves(false, true))), config);
        EXPECT_TRUE(attempt.module.empty());
        EXPECT_NE(attempt.refusal.find("portable-mask-half-incomplete-wave"), std::string::npos)
            << attempt.refusal;
    }
}
TEST(PortableMaskCommon, CompleteGeometryKeepsTheSymbolicRoute) {
    // These are compiler controls only; the integer oracle does not model these coordinates.
    for (const auto& config :
         {c::exact_config(128, 1, 1, 128, 1, 1), c::exact_config(32, 4, 1, 32, 4, 1),
          c::exact_config(16, 4, 2, 16, 4, 2)}) {
        const auto attempt =
            c::compile(c::readfirstlane(c::branch(c::halves(false, true))), config);
        EXPECT_FALSE(attempt.module.empty()) << attempt.refusal;
    }
}
TEST(PortableMaskCommon, ExactPartialNumericAndWholeBoolKeepTheirSinks) {
    for (uint32_t threads : {32u, 65u})
        for (bool whole_bool : {false, true}) {
            const auto code =
                whole_bool
                    ? c::branch({0x7e060280u, base::mov(126, 193, true), base::mov(20, 126, true)})
                    : c::branch({0x7e060280u, base::mov(20, 193, true)});
            const auto attempt =
                c::compile(c::readfirstlane(code), c::exact_config(64, 1, 1, threads, 1, 1));
            ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
            bpermute_oracle::Interpreter oracle(attempt.module);
            const auto actual = oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}},
                                                   3, threads == 65 ? 1 : 0);
            ASSERT_TRUE(oracle.error.empty()) << oracle.error;
            std::vector<uint32_t> expected(64, 0xdeadbeef);
            std::fill(expected.begin(), expected.begin() + (threads == 65 ? 1 : 32), 9);
            EXPECT_EQ(actual, expected);
        }
}
TEST(PortableMaskCommon, ExactCompleteWaveKeepsItsSelectedHighBits) {
    const auto config = c::exact_config(64, 1, 1, 64, 1, 1);
    const auto attempt =
        c::compile(c::readfirstlane(c::ffbh(c::selected_half(true, 0x1000u), 21)), config);
    ASSERT_FALSE(attempt.module.empty()) << attempt.refusal;
    bpermute_oracle::Interpreter oracle(attempt.module);
    EXPECT_EQ(oracle.run(64, std::vector<uint32_t>(64, 0xdeadbeef)), std::vector<uint32_t>(64, 19));
    EXPECT_TRUE(oracle.error.empty()) << oracle.error;
}
TEST(PortableMaskCommon, ExplicitActiveCallerCannotSupplyAbsentGuestBits) {
    using namespace prosper::gpu;
    const auto run = [](bool incomplete) {
        SpirvCompute b;
        b.begin(1, nullptr, 64, 1, 1, 64);
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0x493504000000ull + incomplete};
        RegState rs;
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.sreg[0] = b.groupid[0];
        // Stop before the store: this pins the public CFG caller's consumed comparison boundary.
        auto code = c::branch(c::halves(false, true));
        code.resize(code.size() - 3);
        code.push_back(base::end);
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        const uint32_t active = incomplete ? b.ucmp(Op_IEqual, b.localid_comp[0], b.uconst(0)) : 0;
        const bool ok = emit_cfg_state_machine(
            b, rs, ins, {}, nullptr, true, true, [](RegState&, const Rdna2Inst&) { return false; },
            code.data(), code.size(), active);
        if (incomplete) {
            EXPECT_FALSE(ok);
            EXPECT_NE(last_terminal_reject_reason(b.diagnostic.program_address)
                          .find("portable-mask-half-incomplete-wave"),
                      std::string::npos);
        } else
            EXPECT_TRUE(ok);
    };
    run(false);
    run(true);
}

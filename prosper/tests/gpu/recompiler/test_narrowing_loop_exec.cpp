// A structured loop's EXEC phi must retain a proved narrowing relationship to its preheader.
// That relationship permits genuine active-lane data, never absent inputs or a poisoned next
// generation. These public-module controls also identify the actual structured emitter route.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr uint32_t kNop = 0xbf800000u;
constexpr uint32_t kEnd = 0xbf810000u;

std::vector<uint32_t> program() {
    return {
        0xbe84047eu, // save actual entry EXEC in s[4:5]
        0x7daa0080u, // CMPX_NE_U32 0,v0: A = entry & actual local-ID comparison
        0x7e020281u, // v1 = 1 in every lane of A
        0x7daa0280u, // loop header: narrow by genuine carried v1
        0xbf880002u, // EXECZ -> saved entry restoration
        0x7e020280u, // genuine next-generation v1 = 0 in the active loop lanes
        0xbf82fffcu, // backedge -> header PC3
        0xbefe0404u, kEnd,
    };
}

struct Result {
    std::vector<uint32_t> module;
    std::string reason;
};

Result compile(const std::vector<uint32_t>& code, uint32_t native_width = 64,
               bool complete_launch = true, uint32_t wave_size = 64) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = wave_size;
    config.native_subgroup_size = native_width;
    config.user_sgprs.assign(1, 0);
    config.tgid_x_en = true; // genuine s1 for matched scalar predecessor controls
    prosper::test::complete_compute_launch(config, 2);
    if (!complete_launch) config.dispatch_launch.known = 0;
    static uint64_t next = 0x5110000100ull;
    const uint64_t address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    if (!module.empty()) {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed before process start; no test mutates it.
        if (const char* directory = std::getenv("PROSPER_NARROW_LOOP_SPV_DIRECTORY")) {
            const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
            const std::string name =
                std::string(test->name()) + "-" + std::to_string(address) + ".spv";
            std::ofstream output(std::filesystem::path(directory) / name, std::ios::binary);
            output.write(reinterpret_cast<const char*>(module.data()),
                         static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
            EXPECT_TRUE(output.good()) << "retain the actual module for strict validation";
        }
    }
    return {std::move(module), last_terminal_reject_reason(address)};
}

void structured(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bool loop = false, dispatch = false;
    for (size_t word = 5; word < result.module.size();) {
        const uint32_t count = result.module[word] >> 16u;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        loop |= (result.module[word] & 0xffffu) == 246u;   // OpLoopMerge
        dispatch |= (result.module[word] & 0xffffu) == 251u;   // OpSwitch
        word += count;
    }
    EXPECT_TRUE(loop);
    EXPECT_FALSE(dispatch);
}

void missing(const Result& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}

DivLoop loop() {
    DivLoop result;
    result.header_pc = 3;
    result.exit_branch_pc = 4;
    result.backedge_pc = 6;
    result.exit_pc = 7;
    return result;
}

std::vector<uint32_t> body_instruction(uint32_t word) {
    auto code = program();
    code.insert(code.begin() + 6, word);
    code[4] = 0xbf880003u;
    code[7] = 0xbf82fffbu;
    return code;
}

DivLoop extended_loop() {
    auto result = loop();
    result.backedge_pc = 7;
    result.exit_pc = 8;
    return result;
}

std::vector<uint32_t> predecessor_writes(bool genuine) {
    // Both predecessors join before the header; one missing producer is not an all-path seed.
    return {
        0xbe84047eu, 0x7daa0080u, 0xbf068001u,
        0xbf840002u,   // genuine TGID X scalar branch
        0x7e020281u, 0xbf820001u, genuine ? 0x7e020281u : kNop,
        kNop,        0x7daa0280u, 0xbf880002u,
        0x7e020280u, 0xbf82fffcu, 0xbefe0404u,
        kEnd,
    };
}
std::vector<uint32_t> scalar_counter(bool observe) {
    // Different nonzero local IDs leave on different iterations. RDNA2 advances s6 for the whole
    // wave; the compute structurizer retains per-invocation exit values. New DATA admission must
    // not expose that scalar result after the wider entry EXEC is restored.
    return {0xbe84047eu,
            0x7daa0080u,
            0x7e020300u,
            0xbe860380u,
            0x7daa0280u,
            0xbf880003u,
            0x80068106u,
            0x4e020281u,
            0xbf82fffbu,
            0xbefe0404u,
            observe ? 0x7e0e0206u : 0x7e0e0280u,
            kEnd};
}

std::vector<uint32_t> scalar_exit_paths(bool both_replace) {
    auto code = scalar_counter(false);
    code.resize(10);
    code.insert(code.end(), {0xbf068001u, 0xbf840002u, 0xbe860380u, 0xbf820001u,
                             both_replace ? 0xbe860380u : kNop, 0x7e0e0206u, kEnd});
    return code;
}

std::vector<uint32_t> scc_counter(bool observe) {
    auto code = scalar_counter(false);
    code[3] = 0xbe860381u;   // s6 = 1
    code.insert(code.begin() + 4, 0xbf068080u);   // genuine initial SCC=true
    code[7] = 0x80868106u;   // s6--: lane 1 leaves SCC=false; later iterations leave it true
    code[11] = observe ? 0x7e0e02fdu : 0x7e0e0280u;
    return code;
}

std::vector<uint32_t> mask_result(bool observe, bool saved = false) {
    if (saved) {
        return {0xbe84047eu, 0xbe88047eu, 0x7e040281u, 0x7daa0080u,
                0x7e020300u, 0x7d840480u, 0x7daa0280u, 0xbf880004u,
                0x7d8a00f9u, 0x06868880u,   // actual body comparison writes s[8:9]
                0x4e020281u, 0xbf82fffau, 0xbefe0404u, observe ? 0xbeea0408u : 0xbeea0480u,
                0x020e0480u, kEnd};
    }
    return {0xbe84047eu, 0x7e040281u, 0x7daa0080u, 0x7e020300u,
            0x7d840480u,   // genuine initial VCC=false
            0x7daa0280u, 0xbf880003u,
            0x7d8a0080u,   // VCC=EXEC in each body; inactive lanes are zero on the guest
            0x4e020281u, 0xbf82fffbu, 0xbefe0404u, observe ? 0x020e0480u : 0x7e0e0280u,
            kEnd};
}
std::vector<uint32_t> in_loop_broadcast(bool readfirst) {
    // For lane i, the real lowest-active-lane broadcasts add 1+...+i. READFIRST's current
    // per-invocation fallback instead adds i*i. Kill s6 before exit: only v7 exposes the error.
    return {0x7e0e0280u, 0xbe84047eu, 0x7daa0080u, 0x7e020300u,
            0xbe860380u, 0x7daa0280u, 0xbf880005u, readfirst ? 0x7e0c0500u : 0xbe860381u,
            0x4a0e0e06u, 0xbe860380u, 0x4e020281u, 0xbf82fff9u,
            0xbefe0404u, 0x7d280e80u, kEnd};
}
}   // namespace

TEST(NarrowingLoopExec, GenuinePreheaderAndBackedgeCompileOnTheStructuredRoute) {
    structured(compile(program()));
}

TEST(NarrowingLoopExec, PortableGenuinePreheaderRetainsTheSameSubset) {
    structured(compile(program(), 0));
}

TEST(NarrowingLoopExec, NativeWave32RetainsItsOwnPreheaderSubset) {
    structured(compile(program(), 32, true, 32));
}

TEST(NarrowingLoopExec, MissingPreheaderWordIsNotCreatedByAnExecPhi) {
    auto code = program();
    code[2] = kNop;
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, MissingPreheaderSourceIsNotCreatedByAnExecPhi) {
    auto code = program();
    code[2] = 0x7e020309u;   // v1 = absent v9
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, APreheaderScalarPlaceholderIsNotLaneData) {
    auto code = program();
    code[2] = 0x7e020209u;   // v1 = absent s9
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, MissingBackedgeSourceRetainsTheCarriedObligation) {
    auto code = program();
    code[5] = 0x7e020309u;
    const auto result = compile(code);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("loop-carried v1 is fabricated on the back edge"),
              std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("header pc=3"), std::string::npos) << result.reason;
}

TEST(NarrowingLoopExec, GenuineBackedgeRelayRetainsThePreheaderDataDomain) {
    auto code = body_instruction(0x7e020302u);   // v1 = current active-lane v2
    code[5] = 0x7e040280u;   // v2 = genuine 0 before the relay
    structured(compile(code));
}

TEST(NarrowingLoopExec, BackedgeRelayCannotLaunderAnAbsentWordIntoTheHeader) {
    auto code = body_instruction(0x7e020302u);
    code[5] = 0x7e040309u;   // v2 = absent v9 before v1 = v2
    const auto result = compile(code);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("loop-carried v1 is fabricated on the back edge"),
              std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("header pc=3"), std::string::npos) << result.reason;
}

TEST(NarrowingLoopExec, BothPredecessorsCanSupplyThePreheaderWord) {
    structured(compile(predecessor_writes(true)));
}

TEST(NarrowingLoopExec, OnePredecessorCannotSupplyThePreheaderWord) {
    missing(compile(predecessor_writes(false)), 8);
}

TEST(NarrowingLoopExec, GenuineRawMaskWordCanSupplyThePreheaderData) {
    auto code = program();
    code[2] = 0x7e02027eu;   // v1 = actual current EXEC low word
    structured(compile(code));
}

TEST(NarrowingLoopExec, UnknownRawLaunchCannotSupplyThePhysicalMaskWord) {
    auto code = program();
    code[2] = 0x7e02027eu;
    const auto result = compile(code, 64, false);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("saved-mask-physical-word-unavailable"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=2 "), std::string::npos) << result.reason;
}

TEST(NarrowingLoopExec, PerLaneDataDoesNotNeedPhysicalMaskWordAuthority) {
    structured(compile(program(), 64, false));
}

TEST(NarrowingLoopExec, DeadScalarLoopResultDoesNotBlockGenuineVectorData) {
    structured(compile(scalar_counter(false)));
}

TEST(NarrowingLoopExec, AScalarLiveoutCannotBorrowNewPerLaneLoopAdmission) {
    missing(compile(scalar_counter(true)), 4);
}

TEST(NarrowingLoopExec, AScalarRelayBeforeAnOverwriteStillObservesTheLoopResult) {
    auto code = scalar_counter(true);
    code.insert(code.end() - 1, 0xbe860380u);
    missing(compile(code), 4);
}

TEST(NarrowingLoopExec, BothExitPathsCanReplaceTheOldScalarResultBeforeReadingIt) {
    structured(compile(scalar_exit_paths(true)));
}

TEST(NarrowingLoopExec, OneExitPathCannotHideASurvivingScalarResult) {
    missing(compile(scalar_exit_paths(false)), 4);
}

TEST(NarrowingLoopExec, DeadSccLoopResultDoesNotBlockGenuineVectorData) {
    structured(compile(scc_counter(false)));
}

TEST(NarrowingLoopExec, AnSccLiveoutCannotBorrowNewPerLaneLoopAdmission) {
    missing(compile(scc_counter(true)), 5);
}

TEST(NarrowingLoopExec, DeadVccLoopResultDoesNotBlockGenuineVectorData) {
    structured(compile(mask_result(false)));
}

TEST(NarrowingLoopExec, AVccLiveoutCannotBorrowNewPerLaneLoopAdmission) {
    missing(compile(mask_result(true)), 5);
}

TEST(NarrowingLoopExec, DeadSavedMaskLoopResultDoesNotBlockGenuineVectorData) {
    structured(compile(mask_result(false, true)));
}

TEST(NarrowingLoopExec, ASavedMaskRestoreCannotBorrowNewPerLaneLoopAdmission) {
    missing(compile(mask_result(true, true)), 6);
}

TEST(NarrowingLoopExec, GenuineInLoopScalarBroadcastKeepsTheObservableVectorResult) {
    structured(compile(in_loop_broadcast(false)));
}

TEST(NarrowingLoopExec, DeadReadfirstScalarCannotExposeAnInLoopVectorRelay) {
    const auto code = in_loop_broadcast(true);
    const auto ins = decode(code);
    ASSERT_EQ(ins[7].fmt, Rdna2Format::VOP1);
    ASSERT_EQ(ins[7].opcode, 0x02u);
    ASSERT_EQ(ins[7].dst.value, 6);
    ASSERT_TRUE(sgpr_dead_at_merge(ins, 12, 6));
    missing(compile(code), 5);
}

TEST(NarrowingLoopExec, APreheaderReadfirstRelayCannotGainUniformityFromItsScalarPhi) {
    auto code = scalar_counter(false);
    code[3] = 0x7e0c0500u;   // genuine local-ID VGPR, existing per-invocation READFIRST scalar
    code.insert(code.begin() + 4,
                0x80068106u);   // relay through scalar ADD does not tag lane_local
    code.insert(code.begin() + 8, 0x7e0e0206u);   // observe the carried scalar in the active body
    code[6] = 0xbf880004u;
    code[10] = 0xbf82fffau;
    code[12] = kNop;   // keep the actual in-loop vector result
    missing(compile(code), 5);
}

TEST(NarrowingLoopExec, AnIndependentSccReplacementBeforeItsReadClosesTheExitPath) {
    auto code = scc_counter(true);
    code.insert(code.end() - 2, 0xbf068080u);   // genuine independent comparison, then v7=SCC
    structured(compile(code));
}

TEST(NarrowingLoopExec, AnImplicitCarryReadCannotKillTheOldSccBeforeObservingIt) {
    auto code = scc_counter(false);
    code.insert(code.end() - 1, 0x82068080u);   // s_addc_u32 s6,0,0 consumes old SCC first
    missing(compile(code), 5);
}

TEST(NarrowingLoopExec, ABodyMaskCannotBecomeAnObservableScalarReduction) {
    // The explicit mask is absent from the preheader map but is published by the actual body
    // comparison. Its scalar reduction is observed by a VGPR before both scalar outputs die.
    const std::vector<uint32_t> code{
        0x7e0e0280u, 0xbe84047eu, 0x7daa0080u, 0x7e020300u,
        0x7daa0280u, 0xbf880007u, 0x7d8400f9u, 0x06869081u,   // v_cmp_eq_u32 s16,1,v0 (SDWAB)
        0xbf078010u,   // s_cmp_lg_u32 s16,0: Wave32 native mask ANY
        0x4a0e0efdu,   // v7 += actual SCC inside this iteration
        0xbe900380u,   // kill the scalar mask before exit
        0x4e020281u, 0xbf82fff7u,   // next counter; back to header4
        0xbefe0404u, 0xbf068080u, 0x7d280e80u, kEnd};
    const auto ins = decode(code);
    ASSERT_EQ(ins[6].fmt, Rdna2Format::VOPC);
    ASSERT_EQ(ins[6].dst.kind, OperandKind::SGPR);
    ASSERT_EQ(ins[6].dst.value, 16);
    ASSERT_TRUE(sgpr_dead_at_merge(ins, 13, 16));
    missing(compile(code, 32, true, 32), 4);
}

TEST(NarrowingLoopExec, ADeadPreheaderMaskCannotReduceOnlySurvivingHostLanes) {
    const std::vector<uint32_t> code{0x7e0e0280u, 0xbe84047eu, 0x7d840081u, 0xbe88046au,
                                     0x7daa0080u, 0x7e020300u, 0x7daa0280u, 0xbf880004u,
                                     0xbf138008u, 0x4a0e0efdu, 0x4e020281u, 0xbf82fffau,
                                     0xbefe0404u, 0xbf068080u, 0x7d280e80u, kEnd};
    missing(compile(code, 32, true, 32), 6);
}

TEST(NarrowingLoopExec, FullExecRestoreInsideTheLoopDoesNotReceiveTheCertificate) {
    const auto code = body_instruction(0xbefe0404u);
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), extended_loop()));
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, PartialExecLowWriteDoesNotReceiveTheCertificate) {
    const auto code = body_instruction(0xbefe0381u);
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), extended_loop()));
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, PartialExecHighWriteDoesNotReceiveTheCertificate) {
    const auto code = body_instruction(0xbeff0380u);
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), extended_loop()));
    missing(compile(code), 3);
}

TEST(NarrowingLoopExec, RestoringEntryAfterTheLoopDoesNotInitializeInactiveWords) {
    auto code = program();
    code.insert(code.end() - 1, 0x7d280280u);   // fresh compare of v1 after wider entry restore
    missing(compile(code), 8);
}

TEST(NarrowingLoopExec, PlainCmpxCertificateUsesActualDecodedExecReduction) {
    const auto ins = decode(program());
    ASSERT_EQ(ins[3].fmt, Rdna2Format::VOPC);
    EXPECT_EQ(ins[3].opcode, 0xd5u);
    EXPECT_TRUE(vopc_is_cmpx(ins[3].opcode));
    EXPECT_EQ(ins[3].src[1].kind, OperandKind::VGPR);
    EXPECT_EQ(ins[3].src[1].value, 1);
    EXPECT_TRUE(structured_loop_only_narrows_exec(ins, loop()));
}

TEST(NarrowingLoopExec, UnknownEncodingCannotCloseTheBackedgeProof) {
    EXPECT_FALSE(
        structured_loop_only_narrows_exec(decode(body_instruction(0xffffffffu)), extended_loop()));
}

TEST(NarrowingLoopExec, ScalarPairOverlappingExecHighIsNotAnUnchangedMask) {
    const auto ins = decode(body_instruction(0xbefd047eu));   // s_mov_b64 s[125:126],exec
    ASSERT_EQ(ins[6].dst.value, 125);
    ASSERT_EQ(scalar_write_width(ins[6]), 2u);
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, extended_loop()));
}

TEST(NarrowingLoopExec, ScalarShiftPairOverlappingExecIsNotAnUnchangedMask) {
    const auto ins = decode(body_instruction(0x91fd817eu));   // s_ashr_i64 s[125:126],exec,1
    ASSERT_EQ(ins[6].fmt, Rdna2Format::SOP2);
    ASSERT_EQ(ins[6].opcode, 0x23u);
    ASSERT_EQ(ins[6].dst.value, 125);
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, extended_loop()));
}

TEST(NarrowingLoopExec, SaveExecFamiliesAreOutsideThePlainCmpxCertificate) {
    for (uint32_t word : {0xbe862404u, 0xbe862704u, 0xbe863704u}) {
        SCOPED_TRACE(word);
        EXPECT_FALSE(
            structured_loop_only_narrows_exec(decode(body_instruction(word)), extended_loop()));
    }
}

TEST(NarrowingLoopExec, ForeignScalarPcTransferCannotReceiveTheCertificate) {
    EXPECT_FALSE(
        structured_loop_only_narrows_exec(decode(body_instruction(0xbe802004u)), extended_loop()));
}

TEST(NarrowingLoopExec, OutsideIndirectEntryCannotBorrowTheEmittedPreheader) {
    auto code = program();
    code[0] = 0xbe802004u;   // SETPC before the owned preheader has an unaccounted destination
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), loop()));
}

TEST(NarrowingLoopExec, UnaccountedBodyBranchCannotReceiveTheCertificate) {
    EXPECT_FALSE(
        structured_loop_only_narrows_exec(decode(body_instruction(0xbf820001u)), extended_loop()));
}

TEST(NarrowingLoopExec, OutsideHeaderEntryCannotBorrowTheEmittedPreheader) {
    auto code = program();
    code[2] = 0xbf820000u;   // another explicit header entry replaces the genuine seed
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), loop()));
}

TEST(NarrowingLoopExec, OutsideInteriorEntryCannotBorrowTheEmittedPreheader) {
    auto code = program();
    code[2] = 0xbf820002u;
    EXPECT_FALSE(structured_loop_only_narrows_exec(decode(code), loop()));
}

TEST(NarrowingLoopExec, UnaccountedBreakAndBottomTestedShapesHaveNoCertificate) {
    const auto ins = decode(program());
    auto candidate = loop();
    candidate.break_pcs.push_back(5);
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
    candidate = loop();
    candidate.bottom_tested = true;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
    candidate = loop();
    candidate.direct_exec_breaks = true;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
    candidate = loop();
    candidate.direct_wave_breaks = true;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
}

TEST(NarrowingLoopExec, CertificateIsNotAPermissionForVccOrSccLoopConditions) {
    const auto ins = decode(program());
    auto candidate = loop();
    candidate.condition = DivLoop::Condition::Vcc;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
    candidate.condition = DivLoop::Condition::Scc;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
    candidate = loop();
    candidate.continue_on_set = false;
    EXPECT_FALSE(structured_loop_only_narrows_exec(ins, candidate));
}

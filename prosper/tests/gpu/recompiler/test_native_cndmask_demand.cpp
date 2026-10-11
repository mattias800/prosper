// #4904: charging V_CNDMASK's proved Bool condition as DATA refuses partial mask logic.
// The selected values must stay correct, numeric operands must still require genuine words,
// and joins/overwrites must not manufacture a MUST Bool fact or a producer ballot.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t a = 0x80000001abcdef01ull, c = 0x5555555512345678ull;
uint32_t logical(uint32_t op = 0x0f, uint32_t destination = 20) {
    return 0x80000200u | (op << 23) | (destination << 16);
}
std::vector<uint32_t> cndmask(uint32_t condition = 20, uint32_t left = 129, uint32_t right = 137) {
    return {0xd5010000u, left | (right << 9) | (condition << 18)};
}
std::vector<uint32_t> program(const std::vector<uint32_t>& middle = {}, uint32_t condition = 20,
                              uint32_t left = 129, uint32_t right = 137,
                              const std::vector<uint32_t>& tail = {}) {
    std::vector<uint32_t> code{logical()};
    code.insert(code.end(), middle.begin(), middle.end());
    const auto select = cndmask(condition, left, right);
    code.insert(code.end(), select.begin(), select.end());
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    return code;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
size_t ballots(const SpirvCompute& b) {
    size_t count = 0;
    for (size_t at = 0; at < b.code.size();) {
        const size_t width = b.code[at] >> 16;
        if (!width || width > b.code.size() - at) return SIZE_MAX;
        count += (b.code[at] & 0xffffu) == 339u;
        at += width;
    }
    return count;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        b.native_subgroup_size = 64;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0xa4904001ull};
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        pair(0, a);
        pair(2, c);
        rs.vreg[0] = b.uconst(0);
    }
    void pair(int root, uint64_t value) {
        rs.sreg[root] = b.uconst(static_cast<uint32_t>(value));
        rs.sreg[root + 1] = b.uconst(static_cast<uint32_t>(value >> 32));
        rs.sreg_bool[root] =
            constant_mask_bit(b, static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32));
    }
    void prove(const std::vector<uint32_t>& code) {
        const auto ins = decode(code);
        // Partial entry forbids producer ballots; actual Bool state stays usable.
        initialize_saved_mask_word_proof(b, ins, code.data(), code.size(), b.btrue(), &rs);
    }
    bool emit(const Rdna2Inst& in) {
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(rs, in);
        bool ok = true;
        EXPECT_TRUE(emit_alu(b, rs, in, ok, true));
        if (ok) {
            record_scalar_write(rs, in, allows_compute_scalar_vcc_bridge(b), before);
            retain_saved_mask_words(b, rs, in);
        }
        return ok;
    }
    bool producer(const std::vector<uint32_t>& code) {
        prove(code);
        return emit(decode(code).front());
    }
    bool demanded() const { return b.native_logical_data_demand_pcs.contains(0); }
};
std::vector<uint32_t> production_program(uint32_t left = 129, uint32_t right = 137,
                                         const std::vector<uint32_t>& tail = {},
                                         const std::vector<uint32_t>& middle = {},
                                         uint32_t condition = 20, bool guarded = true) {
    // CMPX alone keeps every physical worker. An earlier wave-dependent VCC exit
    // bypasses these producers, so their words lack the original-program proof;
    // the dispatcher must still retain the actual saved Bool condition.
    std::vector<uint32_t> code{0x7daa0080u, 0x7d8400f9u, 0x06868081u,
                               0x7d8400f9u, 0x06868281u, logical()};
    code.insert(code.end(), middle.begin(), middle.end());
    const auto select = cndmask(condition, left, right);
    code.insert(code.end(), select.begin(), select.end());
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    if (guarded) {
        const uint32_t exit = 0xbf860000u | static_cast<uint32_t>(code.size() - 1);
        code.insert(code.begin(), {0x7d840080u, exit});
    }
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    ComputeShaderConfig config;
    config.wave_size = config.native_subgroup_size = 64;
    config.local_x = 128;
    config.user_sgprs.resize(8);
    return recompile_compute(code.data(), code.size(), nullptr, config,
                             {RecompileDiagnosticStage::Compute, address});
}
}   // namespace

TEST(NativeCndmaskDemand, MustBoolConditionDoesNotDemandPhysicalWords) {
    Emitter e;
    e.rs.sreg.erase(0);
    e.rs.sreg.erase(1);
    ASSERT_TRUE(e.producer(program()));
    EXPECT_FALSE(e.demanded());
    EXPECT_FALSE(e.rs.sreg.contains(20));
    EXPECT_FALSE(e.rs.sreg.contains(21));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeCndmaskDemand, MarkedPeerKeepsBoolSelectAndActualSelectedValues) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    const auto code = program();
    ASSERT_TRUE(e.producer(code));
    ASSERT_TRUE(e.emit(decode(code)[1]));
    EXPECT_FALSE(e.demanded());
    EXPECT_EQ(ballots(e.b), 0u);
    e.b.store_output_word(e.rs.vreg.at(0), 1, 0);
    bpermute_oracle::Interpreter vm(e.b.finish());
    const auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(64, 0xdeadbeef)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    expected.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane) expected.push_back(((a & c) >> lane) & 1u ? 9u : 1u);
    EXPECT_EQ(actual, expected);
}

TEST(NativeCndmaskDemand, NumericDataSourceZeroStillDemandsWords) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({}, 2, 20, 137)));
    EXPECT_TRUE(e.demanded());
    EXPECT_EQ(e.b.stage_reject_reason, "native-logical-data-words-unavailable");
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeCndmaskDemand, NumericDataSourceOneStillDemandsWords) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({}, 2, 129, 20)));
    EXPECT_TRUE(e.demanded());
    EXPECT_EQ(e.b.stage_reject_reason, "native-logical-data-words-unavailable");
}

TEST(NativeCndmaskDemand, LaterNumericLowReadStillDemandsTheSameGeneration) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({}, 20, 129, 137, {0xbe9e0314u})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, LaterNumericHighReadStillDemandsTheSameGeneration) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({}, 20, 129, 137, {0xbe9e0315u})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, LowOverlapRetiresBoolAuthorityAndChargesUntouchedHigh) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({0xbe940381u})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, HighOverlapRetiresBoolAuthorityAndChargesUntouchedLow) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({0xbe950381u})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, CompleteReplacementDoesNotChargeAnOlderGeneration) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    ASSERT_TRUE(e.producer(program({0xbe940481u})));
    EXPECT_FALSE(e.demanded());
}

TEST(NativeCndmaskDemand, OneArmHighReplacementCannotProveMustBoolAtJoin) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({0xbf060202u, 0xbf840001u, 0xbe950381u})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, LoopCarriedHighReplacementCannotProveMustBool) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({}, 20, 129, 137, {0xbe950381u, 0xbf85fffcu})));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, BoolPreservingLoopRetainsItsMustCondition) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    ASSERT_TRUE(e.producer(program({}, 20, 129, 137, {0xbf800000u, 0xbf85fffcu})));
    EXPECT_FALSE(e.demanded());
}

TEST(NativeCndmaskDemand, MissingConditionStillRejectsAtActualConsumer) {
    Emitter e;
    const auto code = program({}, 24);
    ASSERT_TRUE(e.producer(code));
    EXPECT_FALSE(e.emit(decode(code)[1]));
}

TEST(NativeCndmaskDemand, OrdinaryScalarConditionDoesNotGainBoolAuthority) {
    Emitter e;
    e.rs.sreg[24] = e.b.uconst(1);
    e.rs.sreg[25] = e.b.uconst(0);
    const auto code = program({}, 24);
    ASSERT_TRUE(e.producer(code));
    EXPECT_FALSE(e.emit(decode(code)[1]));
}

TEST(NativeCndmaskDemand, PhysicalSourceWidthAbiRemainsTwoWords) {
    const auto select = decode(cndmask()).front();
    EXPECT_EQ(scalar_alu_source_words(select, 0), 1u);
    EXPECT_EQ(scalar_alu_source_words(select, 1), 1u);
    EXPECT_EQ(scalar_alu_source_words(select, 2), 2u);
}

TEST(NativeCndmaskDemand, VccGapCopyRetainsItsSeparatePhysicalCopyRefusal) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    const auto code = program({0xbeea0414u}, 106);
    ASSERT_TRUE(e.producer(code));
    EXPECT_FALSE(e.demanded());
    // The consumer inventory is fixed, but the independent older MOV guard still
    // requires physical words for this CNDMASK spelling. It remains a named gap.
    EXPECT_FALSE(e.emit(decode(code)[1]));
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-copy-words-unavailable");
    EXPECT_EQ(e.b.stage_reject_pc, 1u);
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeCndmaskDemand, GenuinePhysicalVccCopyKeepsBoolConditionAndBothWords) {
    Emitter e;
    const auto code = program({0xbeea0414u}, 106);
    ASSERT_TRUE(e.producer(code));
    EXPECT_FALSE(e.demanded());
    const auto ins = decode(code);
    ASSERT_TRUE(e.emit(ins[1]));
    EXPECT_EQ(e.rs.sreg.at(106), e.rs.sreg.at(20));
    EXPECT_EQ(e.rs.sreg.at(107), e.rs.sreg.at(21));
    ASSERT_TRUE(e.emit(ins[2]));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeCndmaskDemand, VccHighConditionHasNoUnprovedMaskExemption) {
    Emitter e;
    e.rs.sreg_merge_placeholder.insert(1);
    EXPECT_FALSE(e.producer(program({0xbeea0414u}, 107)));
    EXPECT_TRUE(e.demanded());
}

TEST(NativeCndmaskDemand, GenuineWordsRemainReadableBesideTheBoolCondition) {
    Emitter e;
    const auto code = program({}, 20, 20, 137);
    ASSERT_TRUE(e.producer(code));
    EXPECT_TRUE(e.demanded());
    EXPECT_TRUE(e.rs.sreg.contains(20));
    EXPECT_TRUE(e.rs.sreg.contains(21));
    ASSERT_TRUE(e.emit(decode(code)[1]));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeCndmaskDemand, ProductionPartialExecAcceptsBoolCondition) {
    const auto code = production_program();
    const auto ins = decode(code);
    ASSERT_TRUE(vopc_is_cmpx(ins[2].opcode));
    ASSERT_EQ(ins[3].dst.value, 0);
    ASSERT_EQ(ins[4].dst.value, 2);
    ASSERT_EQ(ins[6].src[2].value, 20);
    EXPECT_FALSE(compile(code, 0xa4904101ull).empty());
}

TEST(NativeCndmaskDemand, ProductionNumericDataSourceZeroStillRejects) {
    EXPECT_TRUE(compile(production_program(20, 137), 0xa4904102ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904102ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904102ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionNumericDataSourceOneStillRejects) {
    EXPECT_TRUE(compile(production_program(129, 20), 0xa4904103ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904103ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904103ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionLaterNumericConditionReadStillRejects) {
    EXPECT_TRUE(compile(production_program(129, 137, {0xbe9e0314u}), 0xa4904104ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904104ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904104ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionLaterNumericHighReadStillRejects) {
    EXPECT_TRUE(compile(production_program(129, 137, {0xbe9e0315u}), 0xa4904105ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904105ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904105ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionLowOverlapStillDemandsTheUntouchedHighWord) {
    EXPECT_TRUE(compile(production_program(129, 137, {}, {0xbe940381u}), 0xa4904106ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904106ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904106ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionHighOverlapStillDemandsTheUntouchedLowWord) {
    EXPECT_TRUE(compile(production_program(129, 137, {}, {0xbe950381u}), 0xa4904107ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904107ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904107ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionCompleteReplacementAcceptsTheNewBoolGeneration) {
    EXPECT_FALSE(compile(production_program(129, 137, {}, {0xbe940481u}), 0xa4904108ull).empty());
}

TEST(NativeCndmaskDemand, ProductionOneArmOverwriteCannotSupplyMustBool) {
    const auto code = production_program(129, 137, {}, {0xbf068080u, 0xbf840001u, 0xbe950381u});
    EXPECT_TRUE(compile(code, 0xa4904109ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa4904109ull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa4904109ull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionLoopOverwriteCannotSupplyMustBool) {
    const auto code = production_program(129, 137, {0xbe950381u, 0xbf85fffcu});
    EXPECT_TRUE(compile(code, 0xa490410aull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0xa490410aull).find("native-logical-data-words-unavailable"),
        std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa490410aull).find("pc=7 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionMissingBoolConditionStillRejects) {
    EXPECT_TRUE(compile(production_program(129, 137, {}, {}, 24), 0xa490410bull).empty());
    EXPECT_NE(last_terminal_reject_reason(0xa490410bull).find("op=0x101"), std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa490410bull).find("pc=8 "), std::string::npos);
}

TEST(NativeCndmaskDemand, ProductionScalarWordsDoNotSupplyBoolCondition) {
    const auto code = production_program(129, 137, {}, {0xbe980381u, 0xbe990380u}, 24);
    EXPECT_TRUE(compile(code, 0xa490410cull).empty());
    EXPECT_NE(last_terminal_reject_reason(0xa490410cull).find("op=0x101"), std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(0xa490410cull).find("pc=10 "), std::string::npos);
}

TEST(NativeCndmaskDemand, StraightCmpxKeepsGenuineNumericDataWithoutAnEarlierExit) {
    EXPECT_FALSE(compile(production_program(20, 137, {}, {}, 20, false), 0xa490410dull).empty());
}
}   // namespace prosper::gpu

// #4847: NGG may preserve a current whole B64 predicate whose own VCC generation
// has no physical DATA consumer. Neither that copy nor its trusted barrier phase
// may grant packed words or recover a proof for arbitrary caller-created slices.
#include "fragment_predicate_oracle.hpp"
#include "fixtures/ngg_bool_copy.hpp"
#include "fixtures/fragment_compare_kill.hpp"
#include "fixtures/ngg_lane_slots.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::ngg_bool_copy;
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t barrier = 0xbf8a0000u;
constexpr uint64_t mask = 0x80000001abcdef01ull;
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
std::vector<uint32_t> program(const std::vector<uint32_t>& tail = {}, bool phase = false) {
    std::vector<uint32_t> code{fixture::mov(30, 106), fixture::mov(106, 20, true), 0xd5010000u,
                               129u | (137u << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    if (phase) code.push_back(barrier);
    code.push_back(end);
    return code;
}
size_t ballots(const SpirvCompute& b) {
    size_t result = 0;
    for (size_t at = 0; at < b.code.size();) {
        const size_t width = b.code[at] >> 16;
        if (!width || width > b.code.size() - at) return SIZE_MAX;
        result += (b.code[at] & 0xffffu) == 339u;
        at += width;
    }
    return result;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.ngg_workgroup_shell = true;
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 192, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.bfalse();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[106] = b.uconst(3);
        rs.sreg[107] = b.uconst(5);
        rs.sreg_bool[20] =
            constant_mask_bit(b, static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32));
        rs.vreg[0] = b.uconst(0);
        rs.vreg[1] = b.uconst(0);
    }
    void prove(const std::vector<uint32_t>& code,
               const std::vector<Rdna2Inst>* rewritten = nullptr) {
        const auto ins = rewritten ? *rewritten : decode(code);
        initialize_saved_mask_word_proof(b, ins, code.data(), code.size(), 0, &rs);
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
    bool copy(const std::vector<uint32_t>& code) {
        prove(code);
        EXPECT_TRUE(b.saved_mask_word_roots.contains(106));
        return emit(decode(code)[1]);
    }
    bool body(const std::vector<uint32_t>& code, uint32_t active = 0,
              const std::function<bool(RegState&, const Rdna2Inst&)>& callback = {}) {
        const auto ins = decode(code);
        return emit_body(b, rs, ins, {}, nullptr, true, false, callback, code.data(), code.size(),
                         nullptr, true, active, true);
    }
    void expect_refusal(uint32_t pc = 1) const {
        EXPECT_EQ(b.stage_reject_reason, "saved-mask-copy-words-unavailable");
        EXPECT_EQ(b.stage_reject_pc, pc);
        EXPECT_EQ(ballots(b), 0u);
    }
};
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, std::string* refusal = nullptr) {
    NggSubgroupShellConfig config;
    config.waves = 3;
    config.wave_lanes = 64;
    config.native_wave64 = true;
    return recompile_ngg_subgroup(code.data(), code.size(), nullptr, config, nullptr,
                                  {RecompileDiagnosticStage::Vertex, 0xa4847001ull}, refusal);
}
} // namespace

TEST(NggSavedVccCopyDemand, OldNumericGenerationDoesNotDemandCurrentPredicateWords) {
    Emitter e;
    ASSERT_TRUE(e.copy(program()));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
    EXPECT_FALSE(e.rs.sreg.contains(106));
    EXPECT_FALSE(e.rs.sreg.contains(107));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NggSavedVccCopyDemand, EmittedCopySelectsTheActualLowAndHighLaneBits) {
    Emitter e;
    const auto code = program();
    ASSERT_TRUE(e.copy(code));
    ASSERT_TRUE(e.emit(decode(code)[2]));
    // This is a lowered SSA value witness with an explicitly supplied exact
    // linear lane value, not NGG entry execution or host subgroup participation.
    fragment_predicate_oracle::Values omitted(e.b);
    EXPECT_FALSE(omitted.evaluate(e.rs.vreg.at(0), 0).has_value());
    fragment_predicate_oracle::Values values(
        e.b, fragment_predicate_oracle::LaneInput::SuppliedComputeLinear);
    EXPECT_FALSE(values.evaluate(UINT32_MAX, 0).has_value());
    EXPECT_FALSE(values.evaluate(e.rs.vreg.at(0), 64).has_value());
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = values.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << values.error;
        EXPECT_EQ(actual.value_or(UINT32_MAX), (mask >> lane) & 1u ? 9u : 1u) << lane;
    }
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NggSavedVccCopyDemand, NewCompleteReplacementOwnsLaterNumericReads) {
    Emitter e;
    ASSERT_TRUE(e.copy(program({fixture::mov(106, 193, true), fixture::mov(31, 107)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NggSavedVccCopyDemand, TrustedOriginalBarrierPhaseRetainsOnlyThePredicateProof) {
    Emitter e;
    ASSERT_TRUE(e.body(program({fixture::mov(106, 193, true), fixture::mov(31, 107)}, true)));
    EXPECT_TRUE(e.b.saved_mask_entry_prefix_pcs.empty());
    EXPECT_TRUE(e.b.saved_mask_reconverged_entry_pcs.empty());
    EXPECT_TRUE(e.b.native_logical_uniform_entry_pcs.empty());
    EXPECT_EQ(ballots(e.b), 0u);
    EXPECT_FALSE(e.b.saved_mask_copy_frame);
}

TEST(NggSavedVccCopyDemand, PublicNggCompilerLowersTheSdwaMaskCopyAndBarrier) {
    for (bool phase : {false, true}) {
        std::string refusal;
        EXPECT_FALSE(compile(fixture::program(phase), &refusal).empty()) << refusal;
    }
}

TEST(NggSavedVccCopyDemand, CurrentGenerationLowAndHighDataStillRefuse) {
    for (uint32_t word : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({fixture::mov(31, word)})));
        e.expect_refusal();
    }
    EXPECT_TRUE(compile(fixture::program(true, true)).empty());
}

TEST(NggSavedVccCopyDemand, GenuineCurrentPhysicalWordsRemainUsable) {
    Emitter e;
    e.rs.sreg[20] = e.b.uconst(static_cast<uint32_t>(mask));
    e.rs.sreg[21] = e.b.uconst(static_cast<uint32_t>(mask >> 32));
    ASSERT_TRUE(e.copy(program({fixture::mov(31, 107)})));
    ASSERT_TRUE(e.emit(decode(program({fixture::mov(31, 107)}))[3]));
    EXPECT_EQ(e.rs.sreg.at(31), e.b.uconst(static_cast<uint32_t>(mask >> 32)));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NggSavedVccCopyDemand, AliasNumericReadsStillDemandTheCopyGeneration) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({fixture::mov(24, 106, true), fixture::mov(31, 25)})));
    e.expect_refusal();
}

TEST(NggSavedVccCopyDemand, PartialReplacementCannotSupplyTheUntouchedWord) {
    for (uint32_t word : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(
            e.copy(program({fixture::mov(word, 129), fixture::mov(31, word == 106 ? 107 : 106)})));
        e.expect_refusal();
    }
}

TEST(NggSavedVccCopyDemand, OneArmAndLoopReadsKeepNumericGenerationLive) {
    for (const auto& tail :
         {std::vector<uint32_t>{0xbf840001u, fixture::mov(106, 193, true), fixture::mov(31, 107)},
          std::vector<uint32_t>{fixture::mov(31, 107), 0xbf85fffau}}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program(tail)));
        e.expect_refusal();
    }
}

TEST(NggSavedVccCopyDemand, B32CurrentSourceDoesNotGainWholeMaskAuthority) {
    Emitter e;
    e.rs.sreg_bool_b32.insert(20);
    EXPECT_FALSE(e.copy(program()));
    e.expect_refusal();
}

TEST(NggSavedVccCopyDemand, NumericPlaceholdersCannotEraseAnIndependentGenuineBool) {
    for (int kind : {0, 1, 2}) {
        Emitter e;
        if (kind == 0) e.rs.unavailable_saved_mask_words.insert(20);
        if (kind == 1) e.rs.sreg_merge_placeholder.insert(20);
        if (kind == 2) e.rs.sreg_memory_pattern.insert(21);
        EXPECT_TRUE(e.copy(program()));
        EXPECT_FALSE(e.rs.sreg.contains(106));
        EXPECT_FALSE(e.rs.sreg.contains(107));
        EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(NggSavedVccCopyDemand, MissingOriginalMaskCannotBeInferredFromCurrentPresence) {
    Emitter e;
    const auto current = e.rs.sreg_bool.at(20);
    e.rs.sreg_bool.erase(20);
    const auto code = program();
    e.prove(code);
    e.rs.sreg_bool[20] = current;
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_refusal();
}

TEST(NggSavedVccCopyDemand, PartialSourceWriteCannotReuseTheEntryMask) {
    for (uint32_t word : {20u, 21u}) {
        Emitter e;
        auto code = program();
        code.insert(code.begin() + 1, fixture::mov(word, 129));
        e.prove(code);
        // Keep the entry predicate present deliberately. Current presence alone
        // cannot undo the original inventory's possible partial-word replacement.
        EXPECT_FALSE(native_vcc_copy_is_bool_only(e.b, decode(code)[2]));
        EXPECT_FALSE(e.emit(decode(code)[2]));
        e.expect_refusal(2);

        Emitter whole;
        code[1] = fixture::mov(20, 193, true);
        whole.prove(code);
        EXPECT_TRUE(native_vcc_copy_is_bool_only(whole.b, decode(code)[2]));
        ASSERT_TRUE(whole.emit(decode(code)[1]));
        EXPECT_TRUE(whole.emit(decode(code)[2]));
    }
}

TEST(NggSavedVccCopyDemand, OneArmSourceOverwriteDoesNotSupplyAMustPredicate) {
    for (bool overwrite : {true, false}) {
        Emitter e;
        auto code = program();
        code.insert(code.begin() + 1,
                    {0xbf840001u, overwrite ? fixture::mov(20, 129) : 0xbf800000u});
        e.prove(code);
        EXPECT_EQ(native_vcc_copy_is_bool_only(e.b, decode(code)[3]), !overwrite);
        EXPECT_EQ(e.emit(decode(code)[3]), !overwrite);
        if (overwrite) e.expect_refusal(3);
    }
}

TEST(NggSavedVccCopyDemand, LoopSourceOverwriteDoesNotSupplyAMustPredicate) {
    for (bool overwrite : {true, false}) {
        Emitter e;
        auto code = program();
        code.insert(code.begin() + 1,
                    {0xbf840002u, overwrite ? fixture::mov(20, 129) : 0xbf800000u, 0xbf82fffdu});
        e.prove(code);
        EXPECT_EQ(native_vcc_copy_is_bool_only(e.b, decode(code)[4]), !overwrite);
        EXPECT_EQ(e.emit(decode(code)[4]), !overwrite);
        if (overwrite) e.expect_refusal(4);
    }
}

TEST(NggSavedVccCopyDemand, IndependentHighWordMoveAndSelectCannotBecomeWholeRoots) {
    for (bool select : {false, true}) {
        Emitter e;
        e.rs.sreg_bool.erase(20);
        e.rs.sreg_bool[107] = e.b.btrue();
        auto code = program();
        if (select)
            code.insert(code.begin() + 1, 0x85940000u | (106u << 8) | 107u);
        else
            code.insert(code.begin() + 1, fixture::mov(20, 107, true));
        e.prove(code);
        ASSERT_TRUE(e.emit(decode(code)[1]));
        EXPECT_FALSE(e.emit(decode(code)[2]));
        EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-copy-words-unavailable");
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(NggSavedVccCopyDemand, SdwaFullPairRetiresBothPreviousScalarAndDescriptorWords) {
    namespace compare = prosper::test::fragment_compare_kill;
    Emitter e;
    e.rs.sreg_bool.erase(20);
    for (int word : {20, 21}) {
        e.rs.sreg[word] = e.b.uconst(0xabc000u + word);
        e.rs.sreg_srt[word] = 128;
    }
    auto code = program();
    code.insert(code.begin() + 1, {compare::compare(), compare::compare_sd(20)});
    e.prove(code);
    ASSERT_TRUE(e.emit(decode(code)[1]));
    for (int word : {20, 21}) {
        EXPECT_FALSE(e.rs.sreg.contains(word));
        EXPECT_FALSE(e.rs.sreg_srt.contains(word));
    }
    ASSERT_TRUE(e.emit(decode(code)[2]));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
}

TEST(NggSavedVccCopyDemand, MissingVectorProducerIsRefusedByTheActualNggAbi) {
    std::string refusal;
    EXPECT_TRUE(compile(fixture::program(true, false, 4), &refusal).empty());
    EXPECT_NE(refusal.find("ngg-abi-read-v4"), std::string::npos) << refusal;
    EXPECT_FALSE(compile(fixture::program()).empty());
}

TEST(NggSavedVccCopyDemand, CallerSliceAndSyntheticPrefixCannotAcquirePhaseAuthority) {
    for (bool prefix : {false, true}) {
        Emitter e;
        const auto code = program({}, true);
        e.prove(code);
        auto sliced = decode(code);
        if (prefix) {
            const auto at = std::find_if(sliced.begin(), sliced.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a;
            });
            ASSERT_NE(at, sliced.end());
            sliced.erase(at, sliced.end());
            Rdna2Inst stop;
            stop.pc = static_cast<uint32_t>(code.size() - 2);
            stop.fmt = Rdna2Format::SOPP;
            stop.opcode = 1;
            stop.len_dwords = 1;
            stop.is_end = stop.synthetic_terminator = true;
            sliced.push_back(stop);
        } else {
            sliced.erase(sliced.begin());
        }
        initialize_saved_mask_word_proof(e.b, sliced, code.data(), code.size(), 0, &e.rs);
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_refusal();
    }
}

TEST(NggSavedVccCopyDemand, ChangedRawSourceAndRestorationCannotReseedTheOldCache) {
    Emitter e;
    const auto code = program();
    e.prove(code);
    auto changed = code;
    changed[1] = fixture::mov(106, 22, true);
    e.prove(changed);
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_refusal();
}

TEST(NggSavedVccCopyDemand, DomainAndB32ModeChangesExpireTheOriginalProof) {
    for (int kind : {0, 1, 2, 3}) {
        Emitter e;
        const auto code = program();
        e.prove(code);
        if (kind == 0) e.b.ngg_workgroup_shell = false;
        if (kind == 1) e.b.native_subgroup_size = 0;
        if (kind == 2) e.b.allow_b32_masks = true;
        if (kind == 3) e.b.physical_domain = PhysicalExecutionDomain::Invocation;
        e.prove(code);
        e.b.ngg_workgroup_shell = true;
        e.b.native_subgroup_size = 64;
        e.b.allow_b32_masks = false;
        e.b.physical_domain = PhysicalExecutionDomain::Workgroup;
        e.prove(code);
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_refusal();
    }
}

TEST(NggSavedVccCopyDemand, PortableWave32AndOneLaneShellsGainNoNewProof) {
    for (int kind : {0, 1, 2}) {
        Emitter e;
        if (kind == 0) e.b.native_subgroup_size = 0;
        if (kind == 1) {
            e.b.wave_size = 32;
            e.b.native_subgroup_size = 32;
            e.b.allow_b32_masks = true;
        }
        if (kind == 2) e.b.ngg_one_lane = true;
        const auto code = program();
        e.prove(code);
        EXPECT_FALSE(native_vcc_copy_is_bool_only(e.b, decode(code)[1]));
    }
}

TEST(NggSavedVccCopyDemand, PartialEntryDoesNotGainTheTrustedBarrierPermit) {
    Emitter e;
    EXPECT_FALSE(e.body(program({}, true), e.b.btrue()));
    e.expect_refusal();
    EXPECT_FALSE(e.b.saved_mask_copy_frame);
}

TEST(NggSavedVccCopyDemand, ChangedEntrySeedCannotBeRestoredToAcquireAContinuation) {
    Emitter e;
    const auto code = program();
    e.prove(code);
    const auto value = e.rs.sreg_bool.at(20);
    e.rs.sreg_bool.erase(20);
    e.prove(code);
    e.rs.sreg_bool[20] = value;
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_refusal();
}

TEST(NggSavedVccCopyDemand, CrossingABarrierCannotCreateATrustedPhase) {
    auto code = fixture::program();
    code.insert(code.begin() + 1, 0xbf820000u);
    const auto at = std::find(code.begin(), code.end(), barrier);
    ASSERT_NE(at, code.end());
    const size_t destination = static_cast<size_t>(at - code.begin()) + 1;
    prosper::test::ngg_slots::target(code, 1, destination);
    std::string refusal;
    EXPECT_TRUE(compile(code, &refusal).empty());
    EXPECT_NE(refusal.find("ngg-barrier-not-uniform"), std::string::npos) << refusal;
}

TEST(NggSavedVccCopyDemand, ExceptionClosesTheOwnedPhaseAndCannotRestoreRevokedAuthority) {
    Emitter e;
    auto code = program({}, true);
    code.insert(code.begin() + 1, {0xf800080fu, 0u});
    EXPECT_THROW(e.body(code, 0,
                        [](RegState&, const Rdna2Inst&) -> bool {
                            throw std::runtime_error("owned phase callback");
                        }),
                 std::runtime_error);
    EXPECT_FALSE(e.b.saved_mask_copy_frame);
    EXPECT_FALSE(native_vcc_copy_is_bool_only(e.b, decode(code)[2]));
    auto public_prefix = decode(code);
    public_prefix.erase(public_prefix.begin());
    initialize_saved_mask_word_proof(e.b, public_prefix, code.data(), code.size(), 0, &e.rs);
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[2]));
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-copy-words-unavailable");
}

namespace {
uint64_t literal_result(uint32_t opcode, uint64_t first, uint64_t second) {
    switch (opcode) {
        case 0x0f: return first & second;
        case 0x11: return first | second;
        case 0x13: return first ^ second;
        case 0x15: return first & ~second;
        case 0x17: return first | ~second;
        case 0x19: return ~(first & second);
        case 0x1b: return ~(first | second);
        case 0x1d: return ~(first ^ second);
        default: ADD_FAILURE() << "unexpected B64 logical"; return 0;
    }
}
void literal_values(uint32_t value, uint32_t opcode, bool first, bool trailing = true) {
    Emitter e;
    const auto code = fixture::literal_logical(value, opcode, first, trailing);
    const auto ins = decode(code);
    ASSERT_GE(ins.size(), 3u);
    ASSERT_EQ(ins[1].fmt, Rdna2Format::SOP2);
    ASSERT_EQ(ins[1].opcode, opcode);
    ASSERT_EQ(ins[1].src[first ? 0 : 1].kind,
              trailing ? OperandKind::Literal : OperandKind::InlineInt);
    if (trailing) {
        ASSERT_TRUE(ins[1].has_literal);
        ASSERT_EQ(ins[1].literal, value);
    }
    e.prove(code);
    EXPECT_FALSE(e.b.native_logical_data_demand_pcs.contains(ins[1].pc));
    ASSERT_TRUE(e.emit(ins[1]));
    ASSERT_TRUE(e.emit(ins[2]));
    const uint64_t immediate = trailing ? uint64_t(value) : UINT64_MAX;
    const uint64_t expected =
        literal_result(opcode, first ? immediate : UINT64_MAX, first ? UINT64_MAX : immediate);
    fragment_predicate_oracle::Values values(
        e.b, fragment_predicate_oracle::LaneInput::SuppliedComputeLinear);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = values.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << values.error;
        EXPECT_EQ(*actual, ((expected >> lane) & 1u) ? 9u : 1u) << lane;
    }
}
} // namespace

TEST(NggSavedVccCopyDemand, TrailingLiteralLogicalRetainsThePublicNggMaskDomain) {
    for (uint32_t value : {0u, 0xffu, 0x80000000u, UINT32_MAX}) {
        SCOPED_TRACE(value);
        Emitter e;
        const auto code = fixture::literal_logical(value);
        e.prove(code);
        EXPECT_FALSE(e.b.native_logical_data_demand_pcs.contains(1u));
        std::string refusal;
        EXPECT_FALSE(compile(code, &refusal).empty()) << refusal;
    }
}

TEST(NggSavedVccCopyDemand, TrailingLiteralLogicalUsesZeroExtendedBitsInEitherOperand) {
    for (uint32_t opcode : {0x0fu, 0x11u, 0x13u, 0x15u, 0x17u, 0x19u, 0x1bu, 0x1du}) {
        SCOPED_TRACE(opcode);
        for (uint32_t value : {0u, 0xffu, 0x80000000u, UINT32_MAX}) {
            SCOPED_TRACE(value);
            for (bool first : {false, true}) {
                SCOPED_TRACE(first);
                literal_values(value, opcode, first);
            }
        }
    }
}

TEST(NggSavedVccCopyDemand, InlineMinusOneRemainsAFullMaskBesideTheLowOnlyLiteral) {
    literal_values(UINT32_MAX, 0x0fu, false);
    literal_values(UINT32_MAX, 0x0fu, false, false);
    std::string refusal;
    EXPECT_FALSE(
        compile(fixture::literal_logical(UINT32_MAX, 0x0fu, false, false), &refusal).empty())
        << refusal;
}

TEST(NggSavedVccCopyDemand, LiteralWithAGenuineSavedPredicateRetainsItsMaskDomain) {
    Emitter e;
    const std::vector<uint32_t> code{0x8798ff14u, 0xffu, 0xd5010000u,
                                     129u | (137u << 9u) | (24u << 18u), end};
    e.prove(code);
    EXPECT_FALSE(e.b.native_logical_data_demand_pcs.contains(0u));
    ASSERT_TRUE(e.emit(decode(code)[0]));
    ASSERT_TRUE(e.emit(decode(code)[1]));
    fragment_predicate_oracle::Values values(
        e.b, fragment_predicate_oracle::LaneInput::SuppliedComputeLinear);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = values.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << values.error;
        EXPECT_EQ(*actual, (((mask & 0xffu) >> lane) & 1u) ? 9u : 1u) << lane;
    }
}

TEST(NggSavedVccCopyDemand, LiteralCannotSupplyMissingOrMarkedNumericPeerWords) {
    for (uint32_t kind : {0u, 1u, 2u, 3u}) {
        SCOPED_TRACE(kind);
        Emitter e;
        const std::vector<uint32_t> code{0x8798ff28u, 0xffu, 0xd5010000u,
                                         129u | (137u << 9u) | (24u << 18u), end};
        e.rs.sreg[40] = e.b.uconst(static_cast<uint32_t>(mask));
        if (kind != 1) e.rs.sreg[41] = e.b.uconst(static_cast<uint32_t>(mask >> 32u));
        if (kind == 2) e.rs.sreg_merge_placeholder.insert(40);
        if (kind == 3) e.rs.unavailable_saved_mask_words.insert(41);
        e.prove(code);
        EXPECT_TRUE(e.b.native_logical_data_demand_pcs.contains(0u));
        EXPECT_EQ(e.emit(decode(code)[0]), kind == 0u);
    }
}

TEST(NggSavedVccCopyDemand, MissingTrailingLiteralDoesNotBecomeAMustMaskSource) {
    Emitter e;
    const auto code = fixture::literal_logical(0xffu);
    auto ins = decode(code);
    ASSERT_TRUE(ins[1].has_literal);
    ins[1].has_literal = false;
    initialize_native_logical_data_demand(e.b, ins, &e.rs);
    EXPECT_TRUE(e.b.native_logical_data_demand_pcs.contains(1u));
    EXPECT_FALSE(e.emit(ins[1]));
}
} // namespace prosper::gpu

// #4918: physical reads of another VCC generation must not force a saved Bool copy
// to reconstruct departed lanes. Current-generation DATA, aliases and unproved paths
// still refuse at the copy; original bytes and the entry Bool domains own that proof.
#include "fragment_predicate_oracle.hpp"
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t mask = 0x80000001abcdef01ull;
constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
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
std::vector<uint32_t> program(const std::vector<uint32_t>& tail = {}, uint32_t left = 129,
                              uint32_t right = 137) {
    // A real numeric use PRECEDES the saved copy. It belongs to the old generation.
    std::vector<uint32_t> code{mov(30, 106), mov(106, 20, true), 0xd5010000u,
                               left | (right << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    return code;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.wave_size = 64;
        b.begin_fragment();
        b.diagnostic = {RecompileDiagnosticStage::Fragment, 0xa4918001ull};
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[106] = b.uconst(3);
        rs.sreg[107] = b.uconst(5);
        rs.sreg_bool[20] =
            constant_mask_bit(b, static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32));
        rs.vreg[0] = b.uconst(0);
    }
    void prove(const std::vector<uint32_t>& code, const RegState* entry = nullptr,
               const std::vector<Rdna2Inst>* rewritten = nullptr) {
        const auto ins = rewritten ? *rewritten : decode(code);
        initialize_saved_mask_word_proof(b, ins, code.data(), code.size(), b.btrue(),
                                         entry ? entry : &rs);
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
        EXPECT_TRUE(b.saved_mask_word_roots.contains(106))
            << "the old whole-root guard must actually be exercised";
        return emit(decode(code)[1]);
    }
    void expect_copy_refusal() const {
        EXPECT_EQ(b.stage_reject_reason, "saved-mask-copy-words-unavailable");
        EXPECT_EQ(b.stage_reject_pc, 1u);
        EXPECT_EQ(ballots(b), 0u);
    }
};
std::vector<uint32_t> production_program(const std::vector<uint32_t>& tail = {},
                                         uint32_t left = 129, uint32_t right = 137) {
    namespace fixture = prosper::test::fragment_compare_kill;
    std::vector<uint32_t> code{
        mov(106, 129, true), mov(30, 106),       0x7e000281u,
        0x7e020280u,         fixture::compare(), fixture::compare_sd(20),
        mov(106, 20, true),  0xd5010000u,        left | (right << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.insert(code.end(), {0xf800080fu, 0x00000000u, end});
    return code;
}
std::vector<uint32_t> partial_production_program(const std::vector<uint32_t>& tail = {}) {
    namespace fixture = prosper::test::fragment_compare_kill;
    std::vector<uint32_t> code{mov(106, 129, true),
                               mov(30, 106),
                               0x7e000281u,
                               0x7e020280u,
                               0x7d840300u,
                               0xbe98246au,
                               0xbf880000u | static_cast<uint32_t>(5 + tail.size()),
                               fixture::compare(),
                               fixture::compare_sd(20),
                               mov(106, 20, true),
                               0xd5010000u,
                               129u | (137u << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.insert(code.end(), {mov(126, 24, true), 0xf800080fu, 0x00000000u, end});
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}
}   // namespace

TEST(FragmentSavedVccCopyDemand, PriorNumericReadDoesNotDemandTheNewCopyGeneration) {
    Emitter e;
    const auto code = program();
    ASSERT_TRUE(e.copy(code));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
    EXPECT_FALSE(e.rs.sreg.contains(106));
    EXPECT_FALSE(e.rs.sreg.contains(107));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, SavedCopySelectsActualLaneValuesWithoutNewWords) {
    Emitter e;
    const auto code = program();
    ASSERT_TRUE(e.copy(code));
    ASSERT_TRUE(e.emit(decode(code)[2]));
    // This evaluates the emitted value graph with supplied lane IDs. It does not execute
    // a fragment entry point, framebuffer operation, or the compute interpreter.
    fragment_predicate_oracle::Values vm(e.b);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = vm.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << vm.error;
        EXPECT_EQ(actual.value_or(UINT32_MAX), (mask >> lane) & 1u ? 9u : 1u) << "lane " << lane;
    }
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, PredicateOracleRejectsUnsupportedInputs) {
    Emitter e;
    const uint32_t unsupported = e.b.ibin(Op_IAdd, e.b.uconst(1), e.b.uconst(2));
    fragment_predicate_oracle::Values vm(e.b);
    EXPECT_FALSE(vm.evaluate(unsupported, 0).has_value());
    EXPECT_FALSE(vm.evaluate(e.rs.sreg_bool.at(20), 64).has_value());
}

TEST(FragmentSavedVccCopyDemand, FullReplacementOwnsLaterNumericReads) {
    for (const auto& tail : {std::vector<uint32_t>{mov(106, 129, true), mov(31, 107)},
                             std::vector<uint32_t>{mov(106, 129), mov(107, 130), mov(31, 107)}}) {
        Emitter e;
        ASSERT_TRUE(e.copy(program(tail)));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(FragmentSavedVccCopyDemand, UnusedCopyCanBeReplacedBeforeAnyNumericRead) {
    Emitter e;
    const std::vector<uint32_t> code{mov(30, 106), mov(106, 20, true), mov(106, 129, true),
                                     mov(31, 107), end};
    ASSERT_TRUE(e.copy(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, CurrentLowAndHighReadsStillRefuse) {
    for (uint32_t word : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({mov(31, word)})));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, NumericCndmaskDataOperandsStillRefuse) {
    for (bool left : {true, false}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({}, left ? 106 : 129, left ? 137 : 106)));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, AliasNumericReadStillDemandsTheCopy) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({mov(24, 106, true), mov(31, 25)})));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, PartialOverwriteStillDemandsUntouchedPeer) {
    for (uint32_t written : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({mov(written, 129), mov(31, written == 106 ? 107 : 106)})));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, OneArmOverwriteCannotKillOldGeneration) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0xbf840001u, mov(106, 129, true), mov(31, 107)})));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, LoopReadKeepsPhysicalDemand) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({mov(31, 107), 0xbf85fffau})));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, ConditionalReplacementCannotKillOldGeneration) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0xbeea0681u, mov(31, 107)})));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, UnknownReadAndEscapingControlSupplyNoDeadnessProof) {
    for (uint32_t unknown : {0xffffffffu, 0xbf827fffu}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({unknown})));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, BoolAliasToExecDoesNotDemandPackedWords) {
    Emitter e;
    ASSERT_TRUE(e.copy(program({mov(24, 106, true), mov(126, 24, true)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, BothBoolCselectRelaysWithoutPhysicalDemand) {
    Emitter e;
    e.rs.sreg_bool[2] = e.b.btrue();
    ASSERT_TRUE(e.copy(program({0x8596026au, mov(126, 22, true)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, MixedCselectStillDemandsPhysicalInputs) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0x8596026au, mov(126, 22, true)})));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, MissingMustSourceCannotBorrowDynamicBool) {
    Emitter e;
    RegState entry = e.rs;
    entry.sreg_bool.erase(20);
    const auto code = program();
    e.prove(code, &entry);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, B32SourceCannotSupplyB64CopyProof) {
    Emitter e;
    e.rs.sreg_bool_b32.insert(20);
    EXPECT_FALSE(e.copy(program()));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, MarkedWordsStayMarkedWhenOnlyBoolIsCopied) {
    Emitter e;
    for (int word : {20, 21}) {
        e.rs.sreg[word] = e.b.uconst(0);
        e.rs.sreg_merge_placeholder.insert(word);
    }
    ASSERT_TRUE(e.copy(program()));
    EXPECT_TRUE(e.rs.sreg_merge_placeholder.contains(106));
    EXPECT_TRUE(e.rs.sreg_merge_placeholder.contains(107));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, GenuinePhysicalCopyStillPreservesBothWords) {
    Emitter e;
    e.rs.sreg[20] = e.b.uconst(static_cast<uint32_t>(mask));
    e.rs.sreg[21] = e.b.uconst(static_cast<uint32_t>(mask >> 32));
    ASSERT_TRUE(e.copy(program({mov(31, 107)})));
    EXPECT_EQ(e.rs.sreg.at(106), e.rs.sreg.at(20));
    EXPECT_EQ(e.rs.sreg.at(107), e.rs.sreg.at(21));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(FragmentSavedVccCopyDemand, IdenticalOriginalRetryRetainsCopyProof) {
    Emitter e;
    const auto code = program();
    e.prove(code);
    e.prove(code);
    ASSERT_TRUE(e.emit(decode(code)[1]));
}

TEST(FragmentSavedVccCopyDemand, ChangedEntryBoolDomainInvalidatesCachedProof) {
    for (bool width : {false, true}) {
        Emitter e;
        const auto code = program();
        e.prove(code);
        RegState changed = e.rs;
        if (width)
            changed.sreg_bool_b32.insert(20);
        else
            changed.sreg_bool.erase(20);
        e.prove(code, &changed);
        e.prove(code);   // invalidation is permanent, never a reseed from a retry
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, ChangedAndTruncatedOriginalInvalidateCachedProof) {
    for (bool truncated : {false, true}) {
        Emitter e;
        const auto code = program();
        e.prove(code);
        auto changed = code;
        if (truncated)
            changed.pop_back();
        else
            changed[3] ^= 1;
        e.prove(changed);
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, FirstSliceAndRecursiveSliceCannotReseedProof) {
    for (bool first : {false, true}) {
        Emitter e;
        const auto code = program();
        auto ins = decode(code);
        ins.erase(ins.begin());
        if (!first) e.prove(code);
        e.prove(code, nullptr, &ins);
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, ChangedDecodedCopyCannotBorrowOriginalPc) {
    for (bool synthetic : {false, true}) {
        Emitter e;
        const auto code = program();
        e.prove(code);
        auto in = decode(code)[1];
        if (synthetic)
            in.synthetic_terminator = true;
        else
            in.src[0].value = 22;
        e.rs.sreg_bool[22] = e.rs.sreg_bool.at(20);
        EXPECT_FALSE(e.emit(in));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, IndependentHighDestinationRemainsOutsideNewProof) {
    Emitter e;
    auto code = program();
    code[0] = mov(30, 107);
    code[1] = mov(107, 20, true);
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, NggShellCannotBorrowOrdinaryComputeCopyProof) {
    Emitter e;
    e.b.ngg_workgroup_shell = true;
    EXPECT_FALSE(e.copy(program()));
    e.expect_copy_refusal();
}

TEST(FragmentSavedVccCopyDemand, ProductionBoolCopyAndLaterReplacementCompile) {
    EXPECT_FALSE(
        compile(production_program({mov(106, 129, true), mov(31, 107)}), 0xa4918101ull).empty());
}

TEST(FragmentSavedVccCopyDemand, ProductionGenuineCurrentPhysicalReadCompiles) {
    EXPECT_FALSE(compile(production_program({mov(31, 107)}), 0xa4918102ull).empty());
}

TEST(FragmentSavedVccCopyDemand, ProductionGenuineNumericCndmaskInputCompiles) {
    EXPECT_FALSE(compile(production_program({}, 106), 0xa4918103ull).empty());
}

TEST(FragmentSavedVccCopyDemand, ProductionPartialBoolCopyCompilesWithoutNewBallot) {
    const auto module = compile(partial_production_program(), 0xa4918201ull);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0xa4918201ull);
    size_t count = 0;
    for (size_t at = 5; at < module.size();) {
        const size_t width = module[at] >> 16;
        ASSERT_NE(width, 0u);
        ASSERT_LE(width, module.size() - at);
        count += (module[at] & 0xffffu) == 339u;
        at += width;
    }
    EXPECT_EQ(count, 0u);
}

TEST(FragmentSavedVccCopyDemand, ProductionPartialPhysicalReadRefusesAtExactCopy) {
    const uint64_t address = 0xa4918202ull;
    EXPECT_TRUE(compile(partial_production_program({mov(31, 107)}), address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("saved-mask-copy-words-unavailable"),
              std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=9 "), std::string::npos);
}

TEST(FragmentSavedVccCopyDemand, ChangedExecutionDomainCannotReuseCachedProof) {
    for (int domain = 0; domain < 4; ++domain) {
        Emitter e;
        const auto code = program();
        e.prove(code);
        if (domain == 0) e.b.wave_size = 32;
        if (domain == 1) e.b.physical_domain = PhysicalExecutionDomain::Workgroup;
        if (domain == 2) e.b.guest_stage = GuestShaderStage::Compute;
        if (domain == 3) e.b.native_subgroup_size = 64;
        e.prove(code);
        e.b.wave_size = 64;
        e.b.physical_domain = PhysicalExecutionDomain::Invocation;
        e.b.guest_stage = GuestShaderStage::Fragment;
        e.b.native_subgroup_size = 0;
        e.prove(code);
        EXPECT_FALSE(e.emit(decode(code)[1]));
        e.expect_copy_refusal();
    }
}

TEST(FragmentSavedVccCopyDemand, PacketAndPortableDomainsCannotBorrowFragmentProof) {
    for (bool packet : {false, true}) {
        Emitter e;
        if (packet) {
            e.b.physical_domain = PhysicalExecutionDomain::Workgroup;
            e.b.is_compute = true;
        } else {
            e.b.is_fragment = false;
            e.b.is_compute = true;
            e.b.guest_stage = GuestShaderStage::Compute;
        }
        e.prove(program());
        EXPECT_FALSE(native_vcc_copy_is_bool_only(e.b, decode(program())[1]));
    }
}

TEST(FragmentSavedVccCopyDemand, OneArmSourceOverwriteCannotBorrowDynamicBool) {
    Emitter e;
    auto code = program();
    code.insert(code.begin() + 1, {0xbf840001u, mov(20, 129)});
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[3]));
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-copy-words-unavailable");
}

TEST(FragmentSavedVccCopyDemand, LoopSourceOverwriteCannotBorrowDynamicBool) {
    Emitter e;
    auto code = program();
    code.insert(code.begin() + 1, {mov(20, 129), 0xbf85ffffu});
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[3]));
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-copy-words-unavailable");
}
}   // namespace prosper::gpu

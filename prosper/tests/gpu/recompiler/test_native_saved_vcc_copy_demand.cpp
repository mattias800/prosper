// #4911: physical reads of another VCC generation must not force a saved Bool copy
// to reconstruct departed lanes. Current-generation DATA, aliases and unproved paths
// still refuse at the copy; original bytes and the entry Bool domains own that proof.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
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
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        b.native_subgroup_size = 64;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0xa4911001ull};
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
    // A wave-dependent early END makes a later packed producer ballot unproved.
    // Keep the complete old #4904 shape, adding only the saved Bool -> VCC move.
    std::vector<uint32_t> code{0x7daa0080u,        0x7d8400f9u, 0x06868081u,
                               0x7d8400f9u,        0x06868281u, 0x87940200u,
                               mov(106, 20, true), 0xd5010000u, left | (right << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    code.insert(code.begin(), {0x7d840080u, 0xbf860000u | static_cast<uint32_t>(code.size() - 1)});
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

TEST(NativeSavedVccCopyDemand, PriorNumericReadDoesNotDemandTheNewCopyGeneration) {
    Emitter e;
    const auto code = program();
    ASSERT_TRUE(e.copy(code));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
    EXPECT_FALSE(e.rs.sreg.contains(106));
    EXPECT_FALSE(e.rs.sreg.contains(107));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, SavedCopySelectsActualLaneValuesWithoutNewWords) {
    Emitter e;
    const auto code = program();
    ASSERT_TRUE(e.copy(code));
    ASSERT_TRUE(e.emit(decode(code)[2]));
    e.b.store_output_word(e.rs.vreg.at(0), 1, 0);
    bpermute_oracle::Interpreter vm(e.b.finish());
    const auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(64, 0xdeadbeef)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    expected.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane) expected.push_back((mask >> lane) & 1u ? 9u : 1u);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, FullReplacementOwnsLaterNumericReads) {
    for (const auto& tail : {std::vector<uint32_t>{mov(106, 129, true), mov(31, 107)},
                             std::vector<uint32_t>{mov(106, 129), mov(107, 130), mov(31, 107)}}) {
        Emitter e;
        ASSERT_TRUE(e.copy(program(tail)));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(NativeSavedVccCopyDemand, UnusedCopyCanBeReplacedBeforeAnyNumericRead) {
    Emitter e;
    const std::vector<uint32_t> code{mov(30, 106), mov(106, 20, true), mov(106, 129, true),
                                     mov(31, 107), end};
    ASSERT_TRUE(e.copy(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, CurrentLowAndHighReadsStillRefuse) {
    for (uint32_t word : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({mov(31, word)})));
        e.expect_copy_refusal();
    }
}

TEST(NativeSavedVccCopyDemand, NumericCndmaskDataOperandsStillRefuse) {
    for (bool left : {true, false}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({}, left ? 106 : 129, left ? 137 : 106)));
        e.expect_copy_refusal();
    }
}

TEST(NativeSavedVccCopyDemand, AliasNumericReadStillDemandsTheCopy) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({mov(24, 106, true), mov(31, 25)})));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, PartialOverwriteStillDemandsUntouchedPeer) {
    for (uint32_t written : {106u, 107u}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({mov(written, 129), mov(31, written == 106 ? 107 : 106)})));
        e.expect_copy_refusal();
    }
}

TEST(NativeSavedVccCopyDemand, OneArmOverwriteCannotKillOldGeneration) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0xbf840001u, mov(106, 129, true), mov(31, 107)})));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, LoopReadKeepsPhysicalDemand) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({mov(31, 107), 0xbf85fffau})));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, ConditionalReplacementCannotKillOldGeneration) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0xbeea0681u, mov(31, 107)})));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, UnknownReadAndEscapingControlSupplyNoDeadnessProof) {
    for (uint32_t unknown : {0xffffffffu, 0xbf827fffu}) {
        Emitter e;
        EXPECT_FALSE(e.copy(program({unknown})));
        e.expect_copy_refusal();
    }
}

TEST(NativeSavedVccCopyDemand, BoolAliasToExecDoesNotDemandPackedWords) {
    Emitter e;
    ASSERT_TRUE(e.copy(program({mov(24, 106, true), mov(126, 24, true)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, BothBoolCselectRelaysWithoutPhysicalDemand) {
    Emitter e;
    e.rs.sreg_bool[2] = e.b.btrue();
    ASSERT_TRUE(e.copy(program({0x8596026au, mov(126, 22, true)})));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, MixedCselectStillDemandsPhysicalInputs) {
    Emitter e;
    EXPECT_FALSE(e.copy(program({0x8596026au, mov(126, 22, true)})));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, MissingMustSourceCannotBorrowDynamicBool) {
    Emitter e;
    RegState entry = e.rs;
    entry.sreg_bool.erase(20);
    const auto code = program();
    e.prove(code, &entry);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, B32SourceCannotSupplyB64CopyProof) {
    Emitter e;
    e.rs.sreg_bool_b32.insert(20);
    EXPECT_FALSE(e.copy(program()));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, MarkedWordsStayMarkedWhenOnlyBoolIsCopied) {
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

TEST(NativeSavedVccCopyDemand, GenuinePhysicalCopyStillPreservesBothWords) {
    Emitter e;
    e.rs.sreg[20] = e.b.uconst(static_cast<uint32_t>(mask));
    e.rs.sreg[21] = e.b.uconst(static_cast<uint32_t>(mask >> 32));
    ASSERT_TRUE(e.copy(program({mov(31, 107)})));
    EXPECT_EQ(e.rs.sreg.at(106), e.rs.sreg.at(20));
    EXPECT_EQ(e.rs.sreg.at(107), e.rs.sreg.at(21));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeSavedVccCopyDemand, IdenticalOriginalRetryRetainsCopyProof) {
    Emitter e;
    const auto code = program();
    e.prove(code);
    e.prove(code);
    ASSERT_TRUE(e.emit(decode(code)[1]));
}

TEST(NativeSavedVccCopyDemand, ChangedEntryBoolDomainInvalidatesCachedProof) {
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

TEST(NativeSavedVccCopyDemand, ChangedAndTruncatedOriginalInvalidateCachedProof) {
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

TEST(NativeSavedVccCopyDemand, FirstSliceAndRecursiveSliceCannotReseedProof) {
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

TEST(NativeSavedVccCopyDemand, ChangedDecodedCopyCannotBorrowOriginalPc) {
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

TEST(NativeSavedVccCopyDemand, IndependentHighDestinationRemainsOutsideNewProof) {
    Emitter e;
    auto code = program();
    code[0] = mov(30, 107);
    code[1] = mov(107, 20, true);
    e.prove(code);
    EXPECT_FALSE(e.emit(decode(code)[1]));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, NggShellCannotBorrowOrdinaryComputeCopyProof) {
    Emitter e;
    e.b.ngg_workgroup_shell = true;
    EXPECT_FALSE(e.copy(program()));
    e.expect_copy_refusal();
}

TEST(NativeSavedVccCopyDemand, ProductionBoolCopyAndLaterReplacementCompile) {
    EXPECT_FALSE(
        compile(production_program({mov(106, 129, true), mov(31, 107)}), 0xa4911101ull).empty());
}

TEST(NativeSavedVccCopyDemand, ProductionCurrentPhysicalReadStillRefuses) {
    const uint64_t address = 0xa4911102ull;
    EXPECT_TRUE(compile(production_program({mov(31, 107)}), address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("native-logical-data-words-unavailable"),
              std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=7 "), std::string::npos);
}

TEST(NativeSavedVccCopyDemand, ProductionNumericCndmaskInputStillRefuses) {
    const uint64_t address = 0xa4911103ull;
    EXPECT_TRUE(compile(production_program({}, 106), address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("native-logical-data-words-unavailable"),
              std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=7 "), std::string::npos);
}
}   // namespace prosper::gpu

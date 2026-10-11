// #4892: native SOP2 B64 DATA must keep departed-lane bits, not reballot an older Bool.
// Execute the production integer words independently; partial and marked mask-only sources stay DATA gaps.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint64_t a = 0x80000001abcdef01ull, c = 0x5555555512345678ull;
constexpr uint32_t end = 0xbf810000u;
uint32_t logical(uint32_t op, uint32_t dst = 20, uint32_t left = 0, uint32_t right = 2) {
    return 0x80000000u | (op << 23) | (dst << 16) | (right << 8) | left;
}
uint64_t expected(uint32_t op, uint64_t left = a, uint64_t right = c) {
    switch (op) {
        case 0x0f: return left & right;
        case 0x11: return left | right;
        case 0x13: return left ^ right;
        case 0x15: return left & ~right;
        case 0x17: return left | ~right;
        case 0x19: return ~(left & right);
        case 0x1b: return ~(left | right);
        case 0x1d: return ~(left ^ right);
        default: ADD_FAILURE(); return 0;
    }
}
size_t ballots(const SpirvCompute& b) {
    size_t count = 0;
    for (size_t n = 0; n < b.code.size();) {
        const size_t width = b.code[n] >> 16;
        if (!width || width > b.code.size() - n) return SIZE_MAX;
        count += (b.code[n] & 0xffffu) == 339u;
        n += width;
    }
    return count;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.native_saved_mask_words = true;
        pair(0, a);
        pair(2, c);
    }
    void pair(int root, uint64_t value) {
        rs.sreg[root] = b.uconst(static_cast<uint32_t>(value));
        rs.sreg[root + 1] = b.uconst(static_cast<uint32_t>(value >> 32));
        // Give the production mask resolver the corresponding per-lane view without requiring
        // a native-device builtin in this closed CPU interpreter. DATA is independently checked.
        rs.sreg_bool[root] =
            constant_mask_bit(b, static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32));
    }
    Rdna2Inst instruction(const std::vector<uint32_t>& code) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        EXPECT_FALSE(ins.empty());
        return ins.empty() ? Rdna2Inst{} : ins.front();
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
    void prove(const std::vector<uint32_t>& code, bool partial = false, bool slice = false) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        if (slice) ins.erase(ins.begin());
        initialize_saved_mask_word_proof(b, ins, code.data(), code.size(), partial ? b.btrue() : 0);
    }
    void no_data(int root) {
        rs.sreg.erase(root);
        rs.sreg.erase(root + 1);
    }
    void expect_gap(int root) {
        EXPECT_FALSE(rs.sreg.contains(root));
        EXPECT_FALSE(rs.sreg.contains(root + 1));
        EXPECT_TRUE(rs.unavailable_saved_mask_words.contains(root));
        EXPECT_TRUE(rs.unavailable_saved_mask_words.contains(root + 1));
        bool ok = true;
        Rdna2Inst read;
        read.pc = 100;
        operand_bits(b, rs, read, {root <= 104 ? OperandKind::SGPR : OperandKind::Special, root},
                     &ok);
        EXPECT_FALSE(ok) << "a later DATA read must not issue a replacement ballot";
    }
    void expect_values(int root, uint64_t value, bool partial_host = false) {
        ASSERT_TRUE(rs.sreg.contains(root) && rs.sreg.contains(root + 1));
        b.store_output_word(rs.sreg.at(root), 2, 0);
        b.store_output_word(rs.sreg.at(root + 1), 2, 1);
        bpermute_oracle::Interpreter vm(b.finish());
        const auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(128, 0xdeadbeef)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        std::vector<uint32_t> want;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            want.push_back(!partial_host || lane < 8 ? static_cast<uint32_t>(value) : 0xdeadbeef);
            want.push_back(!partial_host || lane < 8 ? static_cast<uint32_t>(value >> 32)
                                                     : 0xdeadbeef);
        }
        EXPECT_EQ(actual, want)
            << "complete scalar words include lanes absent from this host region";
    }
};
}   // namespace

TEST(NativeLogicalWords, AllEightOperationsPreserveBothPhysicalWordsWithoutBallots) {
    for (uint32_t op = 0x0f; op <= 0x1d; op += 2) {
        SCOPED_TRACE(op);
        Emitter e;
        ASSERT_TRUE(e.emit(e.instruction({logical(op)})));
        EXPECT_EQ(ballots(e.b), 0u);
        EXPECT_EQ(e.rs.scc, 0u) << "wave-level SCC remains outside this DATA repair";
        e.expect_values(20, expected(op));
    }
}

TEST(NativeLogicalWords, PartialHostRegionUsesRetainedWordsOfDepartedLanes) {
    Emitter e;
    const uint32_t inside = e.b.id(), merge = e.b.id();
    e.b.emit_selmerge(merge);
    e.b.emit_condbranch(e.b.ucmp(Op_ULessThan, e.b.linear_localid, e.b.uconst(8)), inside, merge);
    e.b.emit_label(inside);
    ASSERT_TRUE(e.emit(e.instruction({logical(0x1d)})));
    EXPECT_EQ(ballots(e.b), 0u);
    e.b.store_output_word(e.rs.sreg.at(20), 2, 0);
    e.b.store_output_word(e.rs.sreg.at(21), 2, 1);
    e.b.emit_branch(merge);
    e.b.emit_label(merge);
    // Stores are inside the real host selection; no false publication outside it.
    bpermute_oracle::Interpreter vm(e.b.finish());
    auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(128, 0xdeadbeef)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(actual[lane * 2], lane < 8 ? static_cast<uint32_t>(expected(0x1d)) : 0xdeadbeef);
        EXPECT_EQ(actual[lane * 2 + 1],
                  lane < 8 ? static_cast<uint32_t>(expected(0x1d) >> 32) : 0xdeadbeef);
    }
}

TEST(NativeLogicalWords, InPlaceWriteReadsBothCompleteSourcesBeforeExpiry) {
    for (uint32_t dst : {0u, 2u}) {
        Emitter e;
        ASSERT_TRUE(e.emit(e.instruction({logical(0x15, dst)})));
        EXPECT_EQ(ballots(e.b), 0u);
        e.expect_values(dst, expected(0x15));
    }
}

TEST(NativeLogicalWords, VccDestinationRetainsGenuineWordsForSpecialDataReads) {
    Emitter e;
    ASSERT_TRUE(e.emit(e.instruction({logical(0x11, 106)})));
    EXPECT_EQ(ballots(e.b), 0u);
    bool ok = true;
    EXPECT_EQ(operand_bits(e.b, e.rs, {}, {OperandKind::Special, 107}, &ok), e.rs.sreg.at(107));
    EXPECT_TRUE(ok);
    e.expect_values(106, expected(0x11));
}

TEST(NativeLogicalWords, LiteralZeroExtensionAndInlineSignExtensionKeepBothWords) {
    for (bool literal : {false, true}) {
        Emitter e;
        const auto in =
            e.instruction(literal ? std::vector<uint32_t>{logical(0x13, 20, 0, 255), UINT32_MAX}
                                  : std::vector<uint32_t>{logical(0x13, 20, 0, 193)});
        ASSERT_TRUE(e.emit(in));
        EXPECT_EQ(ballots(e.b), 0u);
        e.expect_values(20, a ^ (literal ? UINT32_MAX : UINT64_MAX));
    }
}

TEST(NativeLogicalWords, GenuineZeroPairIsDefinedData) {
    Emitter e;
    e.pair(0, 0);
    ASSERT_TRUE(e.emit(e.instruction({logical(0x11)})));
    e.expect_values(20, c);
}

TEST(NativeLogicalWords, MaskOnlyOrdinaryDestinationCannotInventScalarWords) {
    Emitter e;
    e.no_data(0);
    ASSERT_TRUE(e.emit(e.instruction({logical(0x0f)})));
    EXPECT_TRUE(e.rs.sreg_bool.contains(20));
    EXPECT_EQ(ballots(e.b), 0u);
    e.expect_gap(20);
}

TEST(NativeLogicalWords, MaskOnlyVccDestinationCannotDeferItsBallotToDataRead) {
    Emitter e;
    e.no_data(0);
    ASSERT_TRUE(e.emit(e.instruction({logical(0x0f, 106)})));
    EXPECT_NE(e.rs.vcc, 0u);
    e.expect_gap(106);
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeLogicalWords, MissingOrMarkedHighPeerDoesNotGrantDataAuthority) {
    for (int mode = 0; mode < 4; ++mode) {
        Emitter e;
        if (mode == 0) e.rs.sreg.erase(1);
        if (mode == 1) e.rs.sreg_merge_placeholder.insert(1);
        if (mode == 2) e.rs.sreg_memory_pattern.insert(1);
        if (mode == 3) e.rs.unavailable_saved_mask_words.insert(1);
        ASSERT_TRUE(e.emit(e.instruction({logical(0x11)})));
        e.expect_gap(20);
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(NativeLogicalWords, OlderInputMapCannotRepairACurrentMaskOnlyPair) {
    Emitter e;
    e.rs.sreg_input[0] = e.rs.sreg.at(0);
    e.rs.sreg_input[1] = e.rs.sreg.at(1);
    e.no_data(0);
    ASSERT_TRUE(e.emit(e.instruction({logical(0x11)})));
    e.expect_gap(20);
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeLogicalWords, OriginalPrefixProofMayCaptureMaskOnlyResultAtItsProducer) {
    Emitter e;
    const std::vector<uint32_t> code{logical(0x0f, 20, 126, 193), end};
    e.prove(code);
    ASSERT_TRUE(e.emit(e.instruction(code)));
    EXPECT_TRUE(e.rs.sreg.contains(20) && e.rs.sreg.contains(21));
    EXPECT_EQ(ballots(e.b), 2u);
}

TEST(NativeLogicalWords, PartialOrSlicedEntryCannotCreateProducerParticipation) {
    for (bool slice : {false, true}) {
        Emitter e;
        const std::vector<uint32_t> code{logical(0x0f, 20, 126, 193), end};
        e.prove(code, !slice, slice);
        ASSERT_TRUE(e.emit(e.instruction(code)));
        e.expect_gap(20);
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(NativeLogicalWords, MarkedPeerCannotUsePrefixProofToLaunderNumericAuthority) {
    Emitter e;
    const std::vector<uint32_t> code{logical(0x0f), end};
    e.prove(code);
    e.rs.sreg_merge_placeholder.insert(1);
    ASSERT_TRUE(e.emit(e.instruction(code)));
    e.expect_gap(20);
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(NativeLogicalWords, DispatcherAcceptsCompleteCurrentDataAndRefusesMaskOnlyData) {
    for (bool mask_only : {false, true}) {
        Emitter e;
        if (mask_only) e.no_data(0);
        const std::vector<uint32_t> code{logical(0x0f), 0xbe9e0314u, end};
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        const bool ok = emit_cfg_state_machine(e.b, e.rs, ins, {}, nullptr, true, false, {},
                                               code.data(), code.size(), e.b.btrue());
        EXPECT_EQ(ok, !mask_only)
            << "dispatcher numeric validity needs both current words on every path";
    }
}

TEST(NativeLogicalWords, DispatcherMaskOnlyVccCannotReloadPlaceholderData) {
    Emitter e;
    e.no_data(0);
    const std::vector<uint32_t> code{logical(0x0f, 106), 0xbf800000u, 0xbe9e036bu, end};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    EXPECT_FALSE(emit_cfg_state_machine(e.b, e.rs, ins, {}, nullptr, true, false, {}, code.data(),
                                        code.size(), e.b.btrue()));
}

TEST(NativeLogicalWords, PortableComputeRetainsItsSeparateMaskOnlyContract) {
    Emitter e;
    e.b.native_subgroup_size = 0;
    ASSERT_TRUE(e.emit(e.instruction({logical(0x0f)})));
    EXPECT_FALSE(e.rs.sreg.contains(20));
    EXPECT_TRUE(e.rs.sreg_bool.contains(20));
    EXPECT_TRUE(e.rs.unavailable_saved_mask_words.empty());
}
}   // namespace prosper::gpu

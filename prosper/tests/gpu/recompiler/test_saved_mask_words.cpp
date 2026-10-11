// #4781: a saved Wave64 mask's untouched physical word must survive a partial overwrite.
// Taking a ballot at a later copy/reload loses lanes that have already left that dynamic region.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t checkpoint[] = {0xf8000000u, 0u};
constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
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
    explicit Emitter(bool fragment = true) {
        b.wave_size = 64;
        if (fragment)
            b.begin_fragment();
        else {
            b.native_subgroup_size = 64;
            b.begin(64);
            b.is_compute = true;
            ComputeShaderConfig launch;
            launch.local_x = launch.wave_size = launch.native_subgroup_size = 64;
            prosper::test::complete_compute_launch(launch);
            prosper::test::bind_checked_compute_launch(b, launch);
        }
        rs.exec = b.btrue();
        rs.vcc = b.btrue();
    }
    bool run(const std::vector<uint32_t>& code,
             const std::function<bool(RegState&, const Rdna2Inst&)>& observer = {},
             bool dispatcher = false, bool slice = false, bool partial_entry = false,
             const std::function<void(std::vector<Rdna2Inst>&)>& rewrite = {}) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(code.data(), code.size(), instructions);
        if (slice) instructions.erase(instructions.begin());
        if (rewrite) rewrite(instructions);
        const auto exp = observer ? observer : [](RegState&, const Rdna2Inst&) { return true; };
        if (dispatcher)
            return emit_cfg_state_machine(b, rs, instructions, {}, nullptr, true, false, exp,
                                          code.data(), code.size(), partial_entry ? b.btrue() : 0);
        return emit_body(b, rs, instructions, {}, nullptr, true, false, exp, code.data(),
                         code.size(), nullptr, true, partial_entry ? b.btrue() : 0);
    }
};
} // namespace

TEST(SavedMaskWords, PhysicalWordsExistAtProducerBeforeLowOverwrite) {
    for (bool fragment : {true, false}) {
        Emitter e(fragment);
        uint32_t low = 0, high = 0;
        size_t at_save = 0;
        const std::vector<uint32_t> code{mov(20, 126, true), checkpoint[0], checkpoint[1],
                                         mov(20, 135),       mov(31, 21),   end};
        ASSERT_TRUE(e.run(code, [&](RegState& rs, const Rdna2Inst&) {
            EXPECT_TRUE(rs.sreg.contains(20) && rs.sreg.contains(21))
                << "both words must exist before any later control or replacement";
            if (rs.sreg.contains(20)) low = rs.sreg.at(20);
            if (rs.sreg.contains(21)) high = rs.sreg.at(21);
            at_save = ballots(e.b);
            return true;
        }));
        ASSERT_NE(low, 0u);
        ASSERT_NE(high, 0u);
        EXPECT_NE(low, high);
        EXPECT_EQ(e.rs.sreg.at(31), high) << "the later read consumes the original high word";
        EXPECT_EQ(ballots(e.b), at_save) << "no late ballot at the physical replacement";
        EXPECT_FALSE(e.rs.sreg_bool.contains(20));
    }
}

TEST(SavedMaskWords, HighOverwriteRetiresCompleteAliasAndRestoresRealPair) {
    Emitter e;
    uint32_t low = 0, old_exec = e.rs.exec;
    const std::vector<uint32_t> code{mov(20, 126, true), checkpoint[0],      checkpoint[1],
                                     mov(21, 135),       mov(126, 20, true), end};
    ASSERT_TRUE(e.run(code, [&](RegState& rs, const Rdna2Inst&) {
        if (rs.sreg.contains(20)) low = rs.sreg.at(20);
        return true;
    }));
    ASSERT_NE(low, 0u);
    EXPECT_EQ(e.rs.sreg.at(20), low);
    EXPECT_FALSE(e.rs.sreg_bool.contains(20)) << "a new high word invalidates the old B64 Bool";
    EXPECT_NE(e.rs.exec, old_exec) << "EXEC must use both current scalar words";
}

TEST(SavedMaskWords, LaterSavedPairCopyReusesOriginalPhysicalWords) {
    for (bool fragment : {true, false}) {
        Emitter e(fragment);
        uint32_t low = 0, high = 0;
        size_t at_save = 0;
        const std::vector<uint32_t> code{mov(20, 126, true),
                                         checkpoint[0],
                                         checkpoint[1],
                                         mov(24, 20, true),
                                         mov(30, 24),
                                         mov(31, 25),
                                         end};
        ASSERT_TRUE(e.run(code, [&](RegState& rs, const Rdna2Inst&) {
            if (rs.sreg.contains(20)) low = rs.sreg.at(20);
            if (rs.sreg.contains(21)) high = rs.sreg.at(21);
            at_save = ballots(e.b);
            return true;
        }));
        ASSERT_NE(low, 0u);
        ASSERT_NE(high, 0u);
        EXPECT_EQ(e.rs.sreg.at(30), low);
        EXPECT_EQ(e.rs.sreg.at(31), high);
        EXPECT_EQ(ballots(e.b), at_save) << "copying an older Bool cannot reballot departed lanes";
    }
}

TEST(SavedMaskWords, WqmCopyReusesBothProducerWords) {
    Emitter e(false);
    uint32_t low = 0, high = 0;
    size_t at_producer = 0;
    const std::vector<uint32_t> code{
        0xbe940a7eu, checkpoint[0], checkpoint[1], mov(24, 20, true), mov(30, 24), mov(31, 25),
        end};
    ASSERT_TRUE(e.run(code, [&](RegState& rs, const Rdna2Inst&) {
        if (rs.sreg.contains(20)) low = rs.sreg.at(20);
        if (rs.sreg.contains(21)) high = rs.sreg.at(21);
        at_producer = ballots(e.b);
        return true;
    }));
    ASSERT_NE(low, 0u);
    ASSERT_NE(high, 0u);
    EXPECT_EQ(e.rs.sreg.at(30), low) << "the low DATA word is the original WQM ballot value";
    EXPECT_EQ(e.rs.sreg.at(31), high) << "the high DATA word is the original WQM ballot value";
    EXPECT_EQ(at_producer, 2u);
    EXPECT_EQ(ballots(e.b), at_producer) << "the later MOV copies bytes without reballoting";
}

TEST(SavedMaskWords, OlderBoolWithoutWordsCannotBecomeNumericByCopy) {
    for (bool fragment : {true, false}) {
        Emitter e(fragment);
        e.rs.sreg_bool[12] = e.b.btrue();
        const std::vector<uint32_t> code{mov(20, 12, true), mov(30, 20), end};
        EXPECT_FALSE(e.run(code)) << "a prefix copy is not the original saved-mask producer";
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(SavedMaskWords, AfterControlOrSlicedEntryCannotMaterializeOlderMask) {
    for (bool slice : {false, true}) {
        Emitter e;
        const std::vector<uint32_t> code{checkpoint[0], checkpoint[1], mov(20, 126, true),
                                         mov(30, 21), end};
        EXPECT_FALSE(e.run(code, {}, false, slice));
        EXPECT_EQ(ballots(e.b), 0u) << "a lexical sliced prefix is not original entry authority";
    }
}

TEST(SavedMaskWords, MarkedScalarCopyNeverAcquiresGenuineBallotWords) {
    Emitter e;
    e.rs.sreg_bool[12] = e.b.btrue();
    e.rs.sreg[12] = e.b.uconst(0);
    e.rs.sreg[13] = e.b.uconst(0);
    e.rs.sreg_merge_placeholder.insert(12);
    e.rs.sreg_merge_placeholder.insert(13);
    const std::vector<uint32_t> code{mov(20, 12, true), mov(30, 20), end};
    EXPECT_FALSE(e.run(code))
        << "copying held placeholders must retain their lack of DATA authority";
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, MaskOnlySaveKeepsExistingWidthContract) {
    Emitter e;
    const std::vector<uint32_t> code{mov(20, 126, true), mov(126, 20, true), end};
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
    EXPECT_EQ(e.b.fragment_required_subgroup_size, 0u)
        << "a pure Bool save/restore adds no physical-word subgroup requirement";
}

TEST(SavedMaskWords, DispatcherCarriesOriginalWordsAfterPartialReplacement) {
    Emitter e;
    const std::vector<uint32_t> code{mov(20, 126, true), mov(21, 135), mov(31, 20), end};
    ASSERT_TRUE(e.run(code, {}, true));
    EXPECT_GE(ballots(e.b), 2u);
    EXPECT_FALSE(e.rs.sreg_bool.contains(20));
    EXPECT_TRUE(e.rs.sreg.contains(20) && e.rs.sreg.contains(21));
}

TEST(SavedMaskWords, PartialDispatcherEntryCannotClaimFullParticipation) {
    Emitter e(false);
    const std::vector<uint32_t> code{mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code, {}, true, false, true));
}
TEST(SavedMaskWords, OriginalEntryHintsDoNotPreventProducerRetention) {
    Emitter e;
    const std::vector<uint32_t> code{0xbfa00003u, mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(code));
    EXPECT_TRUE(e.rs.sreg.contains(20) && e.rs.sreg.contains(21));
    EXPECT_EQ(ballots(e.b), 2u);
}

TEST(SavedMaskWords, OuterExtentGuardIsNotFullEntryParticipation) {
    Emitter e(false);
    e.b.guard_invocation_extent(33, 1, 1);
    const std::vector<uint32_t> code{mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, OlderBoolCopyToVccCannotUseLateBallot) {
    for (bool fragment : {true, false}) {
        Emitter e(fragment);
        e.rs.sreg_bool[12] = e.b.btrue();
        const std::vector<uint32_t> code{checkpoint[0], checkpoint[1], mov(106, 12, true),
                                         mov(31, 106), end};
        EXPECT_FALSE(e.run(code));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(SavedMaskWords, GenuineSavedPairCopyToVccKeepsCapturedWords) {
    Emitter e(false);
    const std::vector<uint32_t> code{mov(20, 126, true), checkpoint[0], checkpoint[1],
                                     mov(106, 20, true), mov(31, 21),   end};
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(e.rs.sreg.at(106), e.rs.sreg.at(20));
    EXPECT_EQ(e.rs.sreg.at(107), e.rs.sreg.at(21));
    EXPECT_EQ(ballots(e.b), 2u);
}

TEST(SavedMaskWords, OlderBoolCopyToVccForMaskOnlyUseAddsNoBallot) {
    for (bool fragment : {true, false}) {
        Emitter e(fragment);
        e.rs.sreg_bool[12] = e.b.btrue();
        const std::vector<uint32_t> code{checkpoint[0], checkpoint[1], mov(106, 12, true),
                                         mov(126, 106, true), end};
        EXPECT_TRUE(e.run(code));
        EXPECT_EQ(ballots(e.b), 0u) << "a Bool consumer does not require or invent physical DATA";
    }
}
TEST(SavedMaskWords, BackedgeCannotReenterAClaimedEntryProducer) {
    Emitter e(false);
    const std::vector<uint32_t> code{mov(20, 126, true), mov(31, 21), 0xbf068080u, 0xbf85fffcu,
                                     end};
    EXPECT_FALSE(e.run(code, {}, true));
    EXPECT_EQ(e.b.stage_reject_pc, 1u);
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, RecursiveSliceCannotReseedProducerParticipation) {
    Emitter e;
    const std::vector<uint32_t> original{checkpoint[0], checkpoint[1], mov(20, 126, true),
                                         mov(31, 21), end};
    EXPECT_FALSE(e.run(original));
    EXPECT_FALSE(e.run({mov(20, 126, true), mov(31, 21), end}));
    EXPECT_EQ(ballots(e.b), 0u) << "a later region cannot replace original entry authority";
}

TEST(SavedMaskWords, StructuredForwardSelectionReconvergesBeforeProducer) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> code{0xbf068000u,        0xbf850001u, mov(8, 129),
                                     mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(e.rs.sreg.at(31), e.rs.sreg.at(21));
    EXPECT_EQ(ballots(e.b), 2u);
}

TEST(SavedMaskWords, ProducerBypassDoesNotProveEntryParticipation) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> code{0xbf068000u,        0xbf850002u, 0xbf800000u,
                                     mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, EarlierEndOnOneArmDoesNotProveEntryParticipation) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> code{0xbf068000u,        0xbf850001u, end,
                                     mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, UnknownControlCannotGrantAnEntrySnapshot) {
    Emitter e;
    const std::vector<uint32_t> code{mov(20, 126, true), 0xbf900000u, mov(31, 21), end};
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, DispatcherGuestReconvergenceIsNotABallotInstanceProof) {
    Emitter e(false);
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> code{0xbf068000u,        0xbf850001u, mov(8, 129),
                                     mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code, {}, true));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, SlicedSelectionCannotReinventOriginalReconvergence) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> code{0xbf068000u,        0xbf850001u, mov(8, 129),
                                     mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code, {}, false, true));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, NativePhysicalSpillRestoreUsesRetainedWordsWithoutAnotherBallot) {
    Emitter e(false);
    const uint32_t old_exec = e.rs.exec;
    const std::vector<uint32_t> code{
        mov(20, 126, true), 0xd7610014u, 0x00010014u, 0xd7610014u,        0x00010215u, 0xd7600014u,
        0x00010114u,        0xd7600015u, 0x00010314u, mov(126, 20, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(code));
    EXPECT_TRUE(e.rs.vgpr_lane_slots.contains(20));
    EXPECT_FALSE(e.rs.vgpr_lane_mask_slots.contains(20));
    EXPECT_EQ(e.rs.sreg.at(31), e.rs.sreg.at(21));
    EXPECT_NE(e.rs.exec, old_exec);
    EXPECT_EQ(ballots(e.b), 2u);
}

TEST(SavedMaskWords, ChangedLiteralCannotReuseProducerParticipation) {
    Emitter e;
    const std::vector<uint32_t> original{mov(8, 255), 1u, mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    auto changed = original;
    changed[1] = 2u;
    EXPECT_FALSE(e.run(changed));
    EXPECT_EQ(ballots(e.b), retained);
}

TEST(SavedMaskWords, TruncatedStreamCannotReuseProducerParticipation) {
    Emitter e;
    const std::vector<uint32_t> original{mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    EXPECT_FALSE(e.run({mov(20, 126, true), mov(31, 21)}));
    EXPECT_EQ(ballots(e.b), retained);
}

TEST(SavedMaskWords, RewrittenProducerCannotReuseOriginalPc) {
    Emitter e;
    const std::vector<uint32_t> original{mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    EXPECT_FALSE(e.run(original, {}, false, false, false,
                       [](auto& decoded) { decoded[0].src[0].value = 106; }));
    EXPECT_EQ(ballots(e.b), retained);
}

TEST(SavedMaskWords, MalformedProducerCannotReuseOriginalPc) {
    Emitter e;
    const std::vector<uint32_t> original{mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    EXPECT_FALSE(
        e.run(original, {}, false, false, false, [](auto& decoded) { decoded[0].words[0] ^= 1u; }));
    EXPECT_EQ(ballots(e.b), retained);
}

TEST(SavedMaskWords, CachedSliceCannotInheritOriginalReconvergence) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(1);
    const std::vector<uint32_t> original{0xbf068000u,        0xbf850001u, mov(8, 129),
                                         mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    e.rs.sreg[0] = e.b.uconst(1);
    EXPECT_FALSE(e.run(original, {}, false, false, false,
                       [](auto& decoded) { decoded.erase(decoded.begin(), decoded.begin() + 3); }));
    EXPECT_EQ(ballots(e.b), retained);
}
TEST(SavedMaskWords, PriorHostKillCannotGrantProducerParticipation) {
    Emitter e;
    // CPU emission control only: a kill in prior host control invalidates entry participation.
    e.b.code.push_back((1u << 16) | Op_Kill);
    const std::vector<uint32_t> code{mov(20, 126, true), mov(31, 21), end};
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b), 0u);
}

TEST(SavedMaskWords, CachedProofCannotCrossNewHostKill) {
    Emitter e;
    const std::vector<uint32_t> original{mov(20, 126, true), mov(31, 21), end};
    ASSERT_TRUE(e.run(original));
    const size_t retained = ballots(e.b);
    e.rs = RegState{};
    e.rs.exec = e.b.btrue();
    e.rs.vcc = e.b.btrue();
    e.b.code.push_back((1u << 16) | Op_Kill);
    EXPECT_FALSE(e.run(original));
    EXPECT_EQ(ballots(e.b), retained);
}

} // namespace prosper::gpu

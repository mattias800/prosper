#include "fixtures/native_mask_participation.hpp"
#include "gpu/recompiler/rdna2_native_participation.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"

#include <gtest/gtest.h>

namespace fixture = prosper::test::native_mask_participation;
using namespace prosper::gpu;
namespace {
struct Entry {
    SpirvCompute b;
    RegState rs;
    Entry() {
        b.native_subgroup_size = 64;
        b.begin(64, nullptr, 64, 1, 1, 64, 13);
        b.native_logical_uniform_launch = true;
        rs.exec = b.btrue();
        rs.vcc = b.btrue();
        rs.vreg[0] = b.localid_comp[0];
        for (int r = 0; r < 12; ++r) rs.sreg_input[r] = b.load_push_constant(r);
        rs.sreg[12] = b.groupid[0];
    }
    std::set<uint32_t> proof(const fixture::Program& p) const {
        std::vector<Rdna2Inst> decoded;
        rdna2_walk(p.words.data(), p.words.size(), decoded);
        return native_mask_participation(b, &rs, decoded);
    }
};
void compiles(fixture::Options options = {}) {
    const auto p = fixture::program(options);
    const auto addr = fixture::address();
    const auto module = fixture::compile(p, fixture::table(), addr);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(addr);
    EXPECT_EQ(compute_spirv_min_subgroup_size(module), 64u);
}
} // namespace

TEST(NativeMaskParticipation, FreshLoopWordsCompileThroughProduction) {
    compiles();
}
TEST(NativeMaskParticipation, ExecOffSelectedPeerStillRoutesGenuineSource) {
    compiles({.exec_off_peer = true});
}
TEST(NativeMaskParticipation, UniformOpaquePrefixDoesNotGrantDataAuthority) {
    compiles({.opaque_prefix = true});
}
TEST(NativeMaskParticipation, InitializedPredicateSurvivesMaskedRealLoads) {
    compiles({.masked_load = true});
}

TEST(NativeMaskParticipation, UnsafeDependenciesCannotCertifyProducer) {
    for (auto unsafe : {fixture::Unsafe::RelativeBranch, fixture::Unsafe::RelativePredicate,
                        fixture::Unsafe::ReadfirstBranch, fixture::Unsafe::Writelane}) {
        const auto p = fixture::program({.unsafe = unsafe});
        Entry e;
        EXPECT_FALSE(e.proof(p).contains(p.producer_pc)) << int(unsafe);
        const auto addr = fixture::address();
        EXPECT_TRUE(fixture::compile(p, fixture::table(), addr).empty()) << int(unsafe);
        EXPECT_FALSE(last_terminal_reject_reason(addr).empty());
    }
}
TEST(NativeMaskParticipation, MissingMarkedAndLaneLocalLaunchWordsStayUnproved) {
    const auto p = fixture::program();
    Entry missing;
    missing.rs.sreg.erase(12);
    EXPECT_FALSE(missing.proof(p).contains(p.producer_pc));
    Entry marked;
    marked.rs.sreg_merge_placeholder.insert(12);
    EXPECT_FALSE(marked.proof(p).contains(p.producer_pc));
    Entry lane_local;
    lane_local.rs.lane_local_scalars.insert(lane_local.rs.sreg.at(12));
    EXPECT_FALSE(lane_local.proof(p).contains(p.producer_pc));
    Entry vector_marked;
    vector_marked.rs.vreg_source_marks[0].hard = true;
    EXPECT_FALSE(vector_marked.proof(p).contains(p.producer_pc));
}
TEST(NativeMaskParticipation, NativeWidthAndStoragePresenceAloneGrantNothing) {
    const auto p = fixture::program();
    for (int domain = 0; domain < 5; ++domain) {
        Entry e;
        if (domain == 0) e.b.native_logical_uniform_launch = false;
        if (domain == 1) e.b.native_subgroup_size = 0;
        if (domain == 2) e.b.local_count = 128;
        if (domain == 3) e.b.ngg_workgroup_shell = true;
        if (domain == 4) e.rs.exec_narrowed = true;
        EXPECT_TRUE(e.proof(p).empty()) << domain;
    }
}
TEST(NativeMaskParticipation, OriginalEntryAuthenticationNeverReseedsSlicesOrMarks) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(p.words.data(), p.words.size(), decoded);
    Entry e;
    initialize_saved_mask_word_proof(e.b, decoded, p.words.data(), p.words.size(), 0, &e.rs);
    ASSERT_TRUE(e.b.native_participating_mask_pcs.contains(p.producer_pc));
    e.rs.sreg_merge_placeholder.insert(12);
    initialize_saved_mask_word_proof(e.b, decoded, p.words.data(), p.words.size(), 0, &e.rs);
    EXPECT_TRUE(e.b.native_participating_mask_pcs.empty());
    e.rs.sreg_merge_placeholder.clear();
    initialize_saved_mask_word_proof(e.b, decoded, p.words.data(), p.words.size(), 0, &e.rs);
    EXPECT_TRUE(e.b.native_participating_mask_pcs.empty());
    Entry slice;
    decoded.erase(decoded.begin());
    initialize_saved_mask_word_proof(slice.b, decoded, p.words.data(), p.words.size(), 0,
                                     &slice.rs);
    EXPECT_TRUE(slice.b.native_participating_mask_pcs.empty());
}
TEST(NativeMaskParticipation, PartialHostEntryCannotInheritCompleteWaveProof) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(p.words.data(), p.words.size(), decoded);
    Entry e;
    initialize_saved_mask_word_proof(e.b, decoded, p.words.data(), p.words.size(), e.b.btrue(),
                                     &e.rs);
    EXPECT_TRUE(e.b.native_participating_mask_pcs.empty());
    Entry guarded;
    guarded.b.invocation_guard_merge = 1;
    initialize_saved_mask_word_proof(guarded.b, decoded, p.words.data(), p.words.size(), 0,
                                     &guarded.rs);
    EXPECT_TRUE(guarded.b.native_participating_mask_pcs.empty());
}
TEST(NativeMaskParticipation, CachedSiteCannotCrossChangedExecutionDomain) {
    const auto p = fixture::program();
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(p.words.data(), p.words.size(), decoded);
    Entry e;
    initialize_saved_mask_word_proof(e.b, decoded, p.words.data(), p.words.size(), 0, &e.rs);
    auto in = rdna2_decode_one(p.words.data() + p.producer_pc, p.words.size() - p.producer_pc);
    in.pc = p.producer_pc;
    ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
    e.b.native_logical_uniform_launch = false;
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    e.b.native_logical_uniform_launch = true;
    e.b.ngg_workgroup_shell = true;
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    e.b.ngg_workgroup_shell = false;
    e.b.local_count = 128;
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
}
TEST(NativeMaskParticipation, RuntimePredicateCheckPreservesUnprovedSourceMarks) {
    Entry e;
    const auto p = fixture::program();
    auto in = rdna2_decode_one(p.words.data() + p.producer_pc, p.words.size() - p.producer_pc);
    e.rs.vreg[4] = e.b.load_push_constant(0);
    ASSERT_TRUE(native_mask_predicate_available(e.rs, in));
    e.rs.vreg_source_marks[4].placeholder = true;
    EXPECT_FALSE(native_mask_predicate_available(e.rs, in));
    EXPECT_TRUE(e.rs.vreg_source_marks[4].placeholder);
    e.rs.vreg_source_marks.clear();
    e.rs.vreg_source_marks[0].hard = true;
    EXPECT_FALSE(native_mask_predicate_available(e.rs, in));
    EXPECT_TRUE(e.rs.vreg_source_marks[0].hard);
}

namespace {
void preservation_control(fixture::Preservation kind) {
    const auto absent = fixture::program({.preservation = kind});
    const auto genuine = fixture::program({.preservation = kind, .initialize_preserved = true});
    Entry e;
    EXPECT_FALSE(e.proof(absent).contains(absent.producer_pc));
    EXPECT_TRUE(e.proof(genuine).contains(genuine.producer_pc));
    const auto addr = fixture::address();
    EXPECT_TRUE(fixture::compile(absent, fixture::table(), addr).empty());
    compiles({.preservation = kind, .initialize_preserved = true});
}
} // namespace
TEST(NativeMaskParticipation, DppBoundaryPreservesUninitializedOldDestination) {
    preservation_control(fixture::Preservation::DppBoundary);
}
TEST(NativeMaskParticipation, InactivePermlanePeerPreservesOldDestination) {
    preservation_control(fixture::Preservation::PermlaneInactive);
}
TEST(NativeMaskParticipation, TransformedMovCannotClaimOriginalLocalLaneIdentity) {
    preservation_control(fixture::Preservation::TransformedLane);
}
TEST(NativeMaskParticipation, ImplicitVccWriteRetiresPriorPhysicalDataGeneration) {
    preservation_control(fixture::Preservation::ImplicitVcc);
}

TEST(NativeMaskParticipation, Orn2ExeczCannotAuthenticateBypassedInitializer) {
    preservation_control(fixture::Preservation::Orn2Execz);
}
TEST(NativeMaskParticipation, OrdinaryOrSaveexecDoesNotBypassGenuineInitializer) {
    Entry e;
    const auto p = fixture::program({.preservation = fixture::Preservation::OrExecz});
    EXPECT_TRUE(e.proof(p).contains(p.producer_pc));
    compiles({.preservation = fixture::Preservation::OrExecz});
}

TEST(NativeMaskParticipation, SaveexecFormsPreserveExactOriginalExeczContract) {
    // RDNA2 SOP1: full OLD_EXEC and zero S0. Empty results must bypass the initializer.
    for (const auto [opcode, nonempty] :
         std::array<std::pair<uint32_t, bool>, 10>{{{0x24, false},
                                                    {0x25, true},
                                                    {0x26, true},
                                                    {0x27, false},
                                                    {0x28, false},
                                                    {0x29, true},
                                                    {0x2a, false},
                                                    {0x2b, false},
                                                    {0x37, true},
                                                    {0x38, true}}}) {
        const auto p = fixture::program({.saveexec_opcode = opcode});
        Entry e;
        EXPECT_EQ(e.proof(p).contains(p.producer_pc), nonempty) << std::hex << opcode;
        const auto genuine =
            fixture::program({.initialize_preserved = true, .saveexec_opcode = opcode});
        EXPECT_TRUE(e.proof(genuine).contains(genuine.producer_pc)) << std::hex << opcode;
    }
}

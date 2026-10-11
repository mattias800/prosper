// #4937: normalized geometry/host width is not genuine guest entry participation.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/native_mask_participation.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_copy_continuation.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_native_participation.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::native_mask_participation;
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t words = 0x80000001abcdef01ull;
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    rdna2_walk(code.data(), code.size(), result);
    return result;
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
    ComputeShaderConfig config = fixture::config();
    explicit Emitter(bool fragment = false, bool scalar_entry = true) {
        if (fragment) {
            b.wave_size = 64;
            b.begin_fragment();
        } else {
            // The closed CPU integer interpreter uses LocalInvocationIndex, not native ballots.
            b.begin(4, nullptr, 64, 1, 1, 64, scalar_entry ? 13 : 0, true, true);
            b.native_subgroup_size = 64;
            b.native_logical_uniform_launch = true;
            bind();
        }
        rs.exec = b.btrue();
        rs.vcc = b.btrue();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.vreg[0] = b.localid_comp[0];
        if (scalar_entry) {
            for (int r = 0; r < 12; ++r) rs.sreg_input[r] = b.load_push_constant(r);
            rs.sreg[12] = b.groupid[0];
        }
    }
    void bind() { prosper::test::bind_checked_compute_launch(b, config); }
    void prove(const std::vector<uint32_t>& code, uint32_t active = 0) {
        initialize_saved_mask_word_proof(b, decode(code), code.data(), code.size(), active, &rs);
    }
    bool body(const std::vector<uint32_t>& code,
              const std::function<bool(RegState&, const Rdna2Inst&)>& observe = {}) {
        return emit_body(b, rs, decode(code), {}, nullptr, true, false, observe, code.data(),
                         code.size());
    }
};
std::vector<uint32_t> prefix_program() {
    return {mov(20, 126, true), mov(31, 21), end};
}
std::vector<uint32_t> reconverged_program() {
    return {0xbf068000u, 0xbf850001u, mov(8, 129), mov(20, 126, true), mov(31, 21), end};
}
std::vector<uint32_t> conditional_logical_program() {
    return {mov(20, 128, true), 0xbf068008u, 0xbf840001u, 0x8794c17eu, mov(31, 20), end};
}
std::vector<uint32_t> conditional_mov_program() {
    return {mov(106, 128, true), 0xbf068008u, 0xbf840001u, mov(106, 126, true), mov(31, 106), end};
}
void loop_rejects(const ComputeShaderConfig& config) {
    const auto p = fixture::program();
    const auto address = fixture::address();
    EXPECT_TRUE(fixture::compile(p, fixture::table(), address, config).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("saved-mask"), std::string::npos)
        << last_terminal_reject_reason(address);
}
void expect_no_physical_sites(const Emitter& e) {
    EXPECT_TRUE(e.b.saved_mask_entry_prefix_pcs.empty());
    EXPECT_TRUE(e.b.saved_mask_reconverged_entry_pcs.empty());
    EXPECT_TRUE(e.b.native_logical_uniform_entry_pcs.empty());
    EXPECT_TRUE(e.b.native_participating_mask_pcs.empty());
#if __has_include("gpu/recompiler/rdna2_native_launch.hpp")
    EXPECT_TRUE(e.b.native_uniform_exec_mov_pcs.empty());
#endif
}
void prefix_refuses(const ComputeShaderConfig& config) {
    Emitter e;
    e.config = config;
    e.bind();
    EXPECT_FALSE(e.body(prefix_program()));
    EXPECT_EQ(e.b.stage_reject_pc, 1u);
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
    EXPECT_EQ(ballots(e.b), 0u);
}
void flip_entry_domain(SpirvCompute& b, int field) {
    switch (field) {
        case 0:
            b.guest_stage = b.guest_stage == GuestShaderStage::Compute ? GuestShaderStage::Vertex
                                                                       : GuestShaderStage::Compute;
            break;
        case 1:
            b.physical_domain = b.physical_domain == PhysicalExecutionDomain::Workgroup
                                    ? PhysicalExecutionDomain::Invocation
                                    : PhysicalExecutionDomain::Workgroup;
            break;
        case 2: b.wave_size = b.wave_size == 64 ? 32 : 64; break;
        case 3: b.native_subgroup_size = b.native_subgroup_size == 64 ? 32 : 64; break;
        case 4: b.is_compute = !b.is_compute; break;
        case 5: b.is_fragment = !b.is_fragment; break;
        case 6: b.is_vertex = !b.is_vertex; break;
        case 7: b.ngg_workgroup_shell = !b.ngg_workgroup_shell; break;
        case 8: b.ngg_one_lane = !b.ngg_one_lane; break;
        case 9: b.allow_b32_masks = !b.allow_b32_masks; break;
    }
}
}   // namespace

TEST(NativeComputeLaunch, CompleteRawLoopRetainsChangingProducerWords) {
    const auto p = fixture::program();
    const auto address = fixture::address();
    EXPECT_FALSE(fixture::compile(p, fixture::table(), address).empty())
        << last_terminal_reject_reason(address);
}
TEST(NativeComputeLaunch, PartialFlagAloneCannotInitializeTheSameLoopData) {
    auto config = fixture::config();
    config.dispatch_launch.initiator |= 2u;
    loop_rejects(config);
}
TEST(NativeComputeLaunch, UnknownLegacyLaunchCannotInitializeTheSameLoopData) {
    auto config = fixture::config();
    config.dispatch_launch = {};
    loop_rejects(config);
}
TEST(NativeComputeLaunch, MissingEachProducingWordCannotInitializeLoopData) {
    for (uint32_t bit = 1; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1) {
        auto config = fixture::config();
        config.dispatch_launch.known &= ~bit;
        loop_rejects(config);
    }
}
TEST(NativeComputeLaunch, CompleteEntryPrefixStillRetainsBothWords) {
    Emitter e;
    ASSERT_TRUE(e.body(prefix_program()));
    EXPECT_TRUE(e.rs.sreg.contains(20) && e.rs.sreg.contains(21));
    EXPECT_EQ(e.rs.sreg.at(31), e.rs.sreg.at(21));
    EXPECT_EQ(ballots(e.b), 2u);
}
TEST(NativeComputeLaunch, UnknownPrefixCannotBypassRawEntryAuthority) {
    auto config = fixture::config();
    config.dispatch_launch = {};
    prefix_refuses(config);
}
TEST(NativeComputeLaunch, PartialPrefixCannotBypassRawEntryAuthority) {
    auto config = fixture::config();
    config.dispatch_launch.initiator |= 2u;
    prefix_refuses(config);
}
TEST(NativeComputeLaunch, RestoreThreadModeUpperAndMissingControlsStayUnproved) {
    for (uint64_t flag : {uint64_t{1u << 5u}, uint64_t{1u << 14u}, uint64_t{1} << 32u}) {
        auto config = fixture::config();
        config.dispatch_launch.initiator |= flag;
        prefix_refuses(config);
    }
    auto config = fixture::config();
    config.dispatch_launch.known |= 128u;
    prefix_refuses(config);
    config = fixture::config();
    config.dispatch_launch.initiator = 0;
    prefix_refuses(config);
}
TEST(NativeComputeLaunch, CompleteReconvergedProducerStillRetainsWords) {
    Emitter e;
    ASSERT_TRUE(e.body(reconverged_program()));
    EXPECT_EQ(ballots(e.b), 2u);
}
TEST(NativeComputeLaunch, ReconvergedProducerCannotBypassUnknownLaunch) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    EXPECT_FALSE(e.body(reconverged_program()));
    EXPECT_EQ(e.b.stage_reject_pc, 4u);
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
    EXPECT_EQ(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, CompleteUniformLogicalProducerStillCapturesWords) {
    Emitter e;
    ASSERT_TRUE(e.body(conditional_logical_program()));
    EXPECT_GT(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, UniformLogicalFallbackCannotBypassUnknownLaunch) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    EXPECT_FALSE(e.body(conditional_logical_program()));
    EXPECT_EQ(e.b.stage_reject_pc, 3u);
    EXPECT_EQ(e.b.stage_reject_reason, "native-logical-data-words-unavailable");
    EXPECT_EQ(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, LoopProofCannotSeedMissingOrMarkedEntryWords) {
    const auto p = fixture::program();
    const auto ins = decode(p.words);
    for (bool marked : {false, true}) {
        Emitter e;
        if (marked)
            e.rs.vreg_source_marks[0].hard = true;
        else
            e.rs.vreg.erase(0);
        EXPECT_FALSE(native_mask_participation(e.b, &e.rs, ins).contains(p.producer_pc));
    }
}
TEST(NativeComputeLaunch, CompleteGroupsAreSeparateFromCompleteGuestWaves) {
    Emitter e;
    e.config.local_x = 65;
    prosper::test::complete_compute_launch(e.config);
    ASSERT_TRUE(compute_dispatch_has_complete_groups(e.config));
    e.b.guest_local_size = {65, 1, 1};
    e.b.local_count = 65;
    e.bind();
    e.prove(prefix_program());
    expect_no_physical_sites(e);
}
TEST(NativeComputeLaunch, MatchingThreeDimensionalCompleteWaveKeepsAuthority) {
    Emitter e;
    e.config.local_x = 2;
    e.config.local_y = 4;
    e.config.local_z = 8;
    prosper::test::complete_compute_launch(e.config, 3, 2, 5);
    e.b.guest_local_size = {2, 4, 8};
    e.bind();
    e.prove(prefix_program());
    EXPECT_TRUE(saved_mask_producer_has_full_participation(e.b, decode(prefix_program())[0]));
}
TEST(NativeComputeLaunch, ExtentMismatchZeroAndProductOverflowDoNotGrantEntry) {
    for (int change = 0; change < 4; ++change) {
        Emitter e;
        if (change == 0) ++e.config.threads_x;
        if (change == 1) ++e.config.dispatch_launch.dimensions[0];
        if (change == 2) e.config.dispatch_launch.num_thread[2] = 0;
        if (change == 3) {
            e.config.local_x = e.config.local_y = e.config.local_z = 65535;
            e.config.threads_x = e.config.threads_y = e.config.threads_z = 65535;
            e.config.dispatch_launch.num_thread = {65535, 65535, 65535};
            e.config.dispatch_launch.dimensions = {1, 1, 1};
            e.b.guest_local_size = {65535, 65535, 65535};
            e.b.local_count = 65535u * 65535u * 65535u;
        }
        e.bind();
        e.prove(prefix_program());
        expect_no_physical_sites(e);
    }
}
TEST(NativeComputeLaunch, UnknownNggPortableWave32AndInconsistentHostWidthDoNotGrantEntry) {
    for (int domain = 0; domain < 5; ++domain) {
        Emitter e;
        if (domain == 0) e.b.ngg_workgroup_shell = true;
        if (domain == 1) e.b.ngg_one_lane = true;
        if (domain == 2) e.b.native_subgroup_size = 0;
        if (domain == 3) e.b.wave_size = 32;
        if (domain == 4) e.b.guest_local_size = {32, 2, 1};
        e.prove(prefix_program());
        EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(prefix_program())[0]));
    }
}
TEST(NativeComputeLaunch, RawWaveWidthAndExactPartialExtentDoNotGrantEntry) {
    for (bool exact : {false, true}) {
        auto config = fixture::config();
        if (exact)
            config.exact_thread_extent = true;
        else
            config.dispatch_launch.initiator |= 1u << 15u;
        prefix_refuses(config);
    }
}
TEST(NativeComputeLaunch, CachedCompleteToPartialToCompleteCannotReseedPhysicalSites) {
    Emitter e;
    const auto code = prefix_program();
    e.prove(code);
    const auto in = decode(code)[0];
    ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
    e.config.dispatch_launch.initiator |= 2u;
    e.bind();
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    e.prove(code);
    expect_no_physical_sites(e);
    e.config.dispatch_launch.initiator &= ~2u;
    e.bind();
    e.prove(code);
    expect_no_physical_sites(e);
}
TEST(NativeComputeLaunch, CachedCompleteToUnknownCannotRetainLogicalFallback) {
    Emitter e;
    const auto code = conditional_logical_program();
    e.prove(code);
    const auto in = decode(code)[3];
    ASSERT_TRUE(native_logical_producer_has_full_participation(e.b, in));
    e.config.dispatch_launch = {};
    e.bind();
    EXPECT_FALSE(native_logical_producer_has_full_participation(e.b, in));
    e.prove(code);
    expect_no_physical_sites(e);
}
TEST(NativeComputeLaunch, IgnoredRawPartialWordsStillBelongToOriginalLaunchIdentity) {
    Emitter e;
    const auto code = prefix_program();
    e.prove(code);
    e.config.dispatch_launch.num_thread[0] |= 31u << 16u;
    ASSERT_TRUE(compute_dispatch_has_complete_groups(e.config));
    e.bind();
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, decode(code)[0]));
    e.prove(code);
    expect_no_physical_sites(e);
}
TEST(NativeComputeLaunch, LoopEntryIdentityIncludesEveryRawObservedWord) {
    Emitter e;
    const auto before = native_participation_entry(e.b, &e.rs);
    ASSERT_FALSE(before.empty());
    e.config.dispatch_launch.num_thread[1] |= 7u << 16u;
    e.bind();
    EXPECT_NE(native_participation_entry(e.b, &e.rs), before);
}
TEST(NativeComputeLaunch, ComputeProofCannotBeRelabelledAsFragmentAuthority) {
    for (int field = 0; field < 10; ++field) {
        SCOPED_TRACE(field);
        Emitter e;
        const auto code = prefix_program();
        e.prove(code);
        const auto in = decode(code)[0];
        ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
        flip_entry_domain(e.b, field);
        EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
        flip_entry_domain(e.b, field);
        EXPECT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
    }
}
TEST(NativeComputeLaunch, FragmentEntryUsesItsSeparateContract) {
    Emitter e(true);
    ASSERT_TRUE(e.body(prefix_program()));
    EXPECT_EQ(ballots(e.b), 2u);
}
TEST(NativeComputeLaunch, RestoredComputeDomainCannotReviveExpiredSpecialRead) {
    for (int field = 0; field < 10; ++field) {
        SCOPED_TRACE(field);
        Emitter e;
        const std::vector<uint32_t> code{mov(31, 106), end};
        e.prove(code);
        flip_entry_domain(e.b, field);
        e.prove(code);
        flip_entry_domain(e.b, field);
        e.prove(code);
        bool ok = true;
        const auto in = decode(code)[0];
        operand_bits(e.b, e.rs, in, in.src[0], &ok);
        EXPECT_FALSE(ok);
        EXPECT_EQ(ballots(e.b), 0u);
        expect_no_physical_sites(e);
    }
}
TEST(NativeComputeLaunch, EntryDomainIsCapturedIndependentlyOfBoolCopyDemand) {
    Emitter e;
    const auto code = prefix_program();
    const auto in = decode(code)[0];
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    e.prove(code);
    ASSERT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
    revoke_saved_mask_copy_proof(e.b);
    EXPECT_TRUE(native_vcc_copy_execution_domain_matches(e.b));
    EXPECT_TRUE(saved_mask_producer_has_full_participation(e.b, in));
}
TEST(NativeComputeLaunch, GenuineLaunchScalarSelectionRetainsCurrentExecMovWords) {
    Emitter e;
    const auto code = conditional_mov_program();
    e.prove(code);
    const auto in = decode(code)[3];
    EXPECT_FALSE(saved_mask_producer_has_full_participation(e.b, in));
    EXPECT_TRUE(saved_mask_mov_defines_words(e.b, in, false, true));
    ASSERT_TRUE(e.body(code));
    EXPECT_EQ(ballots(e.b), 2u);
    EXPECT_TRUE(e.rs.sreg.contains(106) && e.rs.sreg.contains(107));
}
TEST(NativeComputeLaunch, ConditionalMovRequiresActualUnmarkedScalarEntryWords) {
    for (int source = 0; source < 6; ++source) {
        Emitter e;
        if (source == 0) e.rs.sreg_input.erase(8);
        if (source == 1) e.rs.sreg_merge_placeholder.insert(8);
        if (source == 2) e.rs.sreg_memory_pattern.insert(8);
        if (source == 3) e.rs.unavailable_saved_mask_words.insert(8);
        if (source == 4) e.rs.lane_local_scalars.insert(e.rs.sreg_input.at(8));
        if (source == 5) e.rs.sreg_bool[8] = e.b.btrue();
        const auto code = conditional_mov_program();
        e.prove(code);
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true)) << source;
    }
}
TEST(NativeComputeLaunch, ConditionalMovRejectsUnknownPartialAndChangedRawLaunch) {
    for (int launch = 0; launch < 3; ++launch) {
        Emitter e;
        const auto code = conditional_mov_program();
        if (launch == 2) e.prove(code);
        if (launch == 0) e.config.dispatch_launch = {};
        if (launch == 1) e.config.dispatch_launch.initiator |= 2u;
        if (launch == 2) e.config.dispatch_launch.num_thread[0] |= 31u << 16u;
        e.bind();
        e.prove(code);
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true));
    }
}
TEST(NativeComputeLaunch, ConditionalMovCannotUseVectorDerivedScalarCondition) {
    Emitter e;
    auto code = conditional_mov_program();
    code.insert(code.begin(), 0x7e100500u);   // s8=READFIRSTLANE v0, not launch-uniform authority
    e.prove(code);
    EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[4], false, true));
}
TEST(NativeComputeLaunch, ConditionalMovDoesNotInventNoncanonicalPairRootsOrHighSources) {
    for (uint32_t destination : {105u, 107u, 127u}) {
        Emitter e;
        auto code = conditional_mov_program();
        code[3] = mov(destination, 126, true);
        e.prove(code);
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true));
    }
    for (uint32_t source : {106u, 107u, 127u}) {
        Emitter e;
        auto code = conditional_mov_program();
        code[3] = mov(106, source, true);
        e.prove(code);
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true));
    }
}
TEST(NativeComputeLaunch, ConditionalMovCannotKeepAnOlderEntrySourceGeneration) {
    Emitter e;
    const auto code = conditional_mov_program();
    e.prove(code);
    ASSERT_TRUE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true));
    e.rs.sreg_input[8] = e.b.uconst(7);
    e.prove(code);
    EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[3], false, true));
}
TEST(NativeComputeLaunch, ConditionalMovCannotReuseAlteredOrRecursiveInstructions) {
    for (bool slice : {false, true}) {
        Emitter e;
        const auto code = conditional_mov_program();
        e.prove(code);
        const auto original = decode(code);
        auto changed = original[3];
        changed.src[0].value = 106;
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, changed, false, true));
        if (slice) {
            auto suffix = std::vector<Rdna2Inst>(original.begin() + 3, original.end());
            initialize_saved_mask_word_proof(e.b, suffix, code.data(), code.size(), 0, &e.rs);
        } else {
            auto modified = code;
            modified[1] = 0xbf068009u;
            e.prove(modified);
        }
        EXPECT_FALSE(saved_mask_mov_defines_words(e.b, original[3], false, true));
    }
}
TEST(NativeComputeLaunch, ConditionalMovCannotUsePartialOuterOrReenteredBody) {
    for (int control = 0; control < 4; ++control) {
        Emitter e;
        auto code = conditional_mov_program();
        if (control == 1) e.b.invocation_guard_merge = 1;
        if (control == 2) code.insert(code.end() - 1, 0xbf82fffcu);   // edge reenters branch at2
        if (control == 3) code.insert(code.begin(), end);
        e.prove(code, control == 0 ? e.b.btrue() : 0);
        EXPECT_FALSE(
            saved_mask_mov_defines_words(e.b, decode(code)[control == 3 ? 4 : 3], false, true));
    }
}
TEST(NativeComputeLaunch, SpecialDestinationMovCannotBypassUnknownEntry) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    const std::vector<uint32_t> code{mov(106, 126, true), mov(31, 107), end};
    e.prove(code);
    EXPECT_FALSE(saved_mask_mov_defines_words(e.b, decode(code)[0], false, true));
    EXPECT_EQ(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, KnownFullSpecialReadStillUsesItsExistingBallotContract) {
    Emitter e;
    const std::vector<uint32_t> code{mov(31, 106), mov(32, 127), end};
    e.prove(code);
    bool ok = true;
    const auto ins = decode(code);
    operand_bits(e.b, e.rs, ins[0], ins[0].src[0], &ok);
    operand_bits(e.b, e.rs, ins[1], ins[1].src[0], &ok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(ballots(e.b), 2u);
}
TEST(NativeComputeLaunch, SpecialScalarReadCannotBallotUnknownOrPartialEntry) {
    for (bool partial : {false, true}) {
        Emitter e;
        if (partial)
            e.config.dispatch_launch.initiator |= 2u;
        else
            e.config.dispatch_launch = {};
        e.bind();
        const std::vector<uint32_t> code{mov(31, 106), end};
        e.prove(code);
        bool ok = true;
        const auto in = decode(code)[0];
        operand_bits(e.b, e.rs, in, in.src[0], &ok);
        EXPECT_FALSE(ok);
        EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
        EXPECT_EQ(e.b.stage_reject_pc, 0u);
        EXPECT_EQ(ballots(e.b), 0u);
    }
}
TEST(NativeComputeLaunch, KnownFullSpecialDestinationMoveStillCapturesAtEntry) {
    Emitter e;
    const std::vector<uint32_t> code{mov(106, 126, true), mov(31, 107), end};
    e.prove(code);
    EXPECT_TRUE(saved_mask_mov_defines_words(e.b, decode(code)[0], false, true));
}
TEST(NativeComputeLaunch, RestoredRawProfileCannotReviveSpecialReadAfterExpiry) {
    Emitter e;
    const std::vector<uint32_t> code{mov(31, 106), end};
    e.prove(code);
    e.config.dispatch_launch.initiator |= 2u;
    e.bind();
    e.prove(code);
    e.config.dispatch_launch.initiator &= ~2u;
    e.bind();
    e.prove(code);
    bool ok = true;
    const auto in = decode(code)[0];
    operand_bits(e.b, e.rs, in, in.src[0], &ok);
    EXPECT_FALSE(ok);
    EXPECT_EQ(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, GenuineSpecialDataReadStaysUsableAtUnknownEntry) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    e.rs.sreg[106] = e.b.uconst(0xabc123u);
    const std::vector<uint32_t> code{mov(31, 106), end};
    e.prove(code);
    bool ok = true;
    const auto in = decode(code)[0];
    EXPECT_EQ(operand_bits(e.b, e.rs, in, in.src[0], &ok), e.rs.sreg.at(106));
    EXPECT_TRUE(ok);
    EXPECT_EQ(ballots(e.b), 0u);
}
TEST(NativeComputeLaunch, GenuineNumericMovKeepsWordsWithUnknownEntry) {
    Emitter e(false, false);
    e.config.dispatch_launch = {};
    e.bind();
    e.rs.sreg[0] = e.b.uconst(static_cast<uint32_t>(words));
    e.rs.sreg[1] = e.b.uconst(static_cast<uint32_t>(words >> 32u));
    ASSERT_TRUE(e.body({mov(20, 0, true), mov(31, 21), end}));
    EXPECT_EQ(ballots(e.b), 0u);
    e.b.store_output_word(e.rs.sreg.at(20), 2, 0);
    e.b.store_output_word(e.rs.sreg.at(31), 2, 1);
    bpermute_oracle::Interpreter vm(e.b.finish());
    const auto output = vm.run_buffers(64, {{1, std::vector<uint32_t>(128, 0xdeadbeef)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        expected.push_back(static_cast<uint32_t>(words));
        expected.push_back(static_cast<uint32_t>(words >> 32u));
    }
    EXPECT_EQ(output, expected);
}
TEST(NativeComputeLaunch, GenuineNumericLogicalDoesNotRequireProducerBallotAuthority) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    const std::vector<uint32_t> code{0x87940200u, mov(31, 20), end};
    e.prove(code);
    EXPECT_TRUE(saved_mask_logical_defines_words(e.b, decode(code)[0], true));
    EXPECT_FALSE(saved_mask_logical_defines_words(e.b, decode(code)[0], false));
}
TEST(NativeComputeLaunch, UnknownEntryStillPreservesActualBoolOnlyCountedPostlude) {
    Emitter e;
    e.config.dispatch_launch = {};
    e.bind();
    e.rs.sreg[106] = e.b.uconst(3);
    e.rs.sreg[107] = e.b.uconst(5);
    e.rs.sreg_bool[20] =
        constant_mask_bit(e.b, static_cast<uint32_t>(words), static_cast<uint32_t>(words >> 32u));
    const std::vector<uint32_t> code{mov(30, 106),
                                     mov(2, 128),
                                     0xbf0a8302u,
                                     0xbf840002u,
                                     0x80028102u,
                                     0xbf82fffcu,
                                     mov(106, 20, true),
                                     0xd5010000u,
                                     129u | (137u << 9u) | (106u << 18u),
                                     end};
    ASSERT_TRUE(e.body(code));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(20));
    EXPECT_EQ(ballots(e.b), 0u);
    expect_no_physical_sites(e);
}
}   // namespace prosper::gpu

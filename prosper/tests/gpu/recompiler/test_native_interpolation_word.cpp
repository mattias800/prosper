// #5099: a successfully published native fragment Input word is defined in the EXEC lanes
// it writes. P1, inactive lanes and unavailable ordinary relays must not inherit that proof.
// Exercise actual emission and complete fragment modules; no device or raster oracle is used.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t kP1 = 0xc8180014u;   // encoded v20 is not a native varying read
constexpr uint32_t kP2 = 0xc8190014u;
constexpr uint32_t kMov = 0xc81a0002u;  // supported P0 selector, v6 destination
constexpr uint32_t kNarrow = 0x7da40080u;  // CMPX from independently defined v0
constexpr uint32_t kCompare = 0x7c200d06u; // compare v6 with v6
constexpr uint32_t kRestore = 0xbefe04c1u;
constexpr uint32_t kMissingRelay = 0x7e0c0314u; // v6 <- absent v20
constexpr uint32_t kRelay = 0x7e100306u;        // v8 <- v6
constexpr uint32_t kCompareRelay = 0x7c201106u;

Rdna2Inst instruction(uint32_t word) {
    std::vector<Rdna2Inst> decoded;
    EXPECT_EQ(rdna2_walk(&word, 1, decoded), 1u);
    EXPECT_EQ(decoded.size(), 1u);
    return decoded.empty() ? Rdna2Inst{} : decoded.front();
}

struct Harness {
    SpirvCompute b;
    RegState rs;
    FragmentInterpolationLayout layout;
    Rdna2Inst pending;
    SavedB64MaskSnapshot before;
    uint32_t next_pc = 0;

    explicit Harness(uint32_t width) {
        b.wave_size = width;
        b.begin_fragment();
        b.allow_b32_masks = width == 32;
        layout.attribute_mask = layout.smooth_mask = 1;
        b.fragment_interpolation = &layout;
        rs.exec = b.btrue();
        rs.vreg[0] =
            b.subgroup_local_id(); // CMPX keeps lane 0 and genuinely excludes other lanes
        rs.max_vgpr = 20;
        seed_vector_word_definedness(rs, {instruction(kP2), instruction(kCompare)}, true, width);
    }

    bool emit(Rdna2Inst in, bool transfer = true) {
        in.pc = next_pc++;
        pending = in;
        before = snapshot_saved_b64_masks(b, rs, in);
        bool ok = true;
        const bool handled = emit_alu(b, rs, in, ok, true);
        if (handled && ok && transfer) record_scalar_write(rs, in, false, before);
        return handled && ok;
    }
    bool emit(uint32_t word, bool transfer = true) { return emit(instruction(word), transfer); }
    void transfer_pending() { record_scalar_write(rs, pending, false, before); }
    bool defined(uint32_t word = kCompare) {
        return fresh_mask_sources_defined(b, rs, instruction(word));
    }
};

class NativeInterpolationWord : public ::testing::TestWithParam<uint32_t> {};

TEST_P(NativeInterpolationWord, P2DefinesTheMatchingExecLanes) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.rs.vreg_source_marks.at(6).hard);
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_FALSE(h.before.source_marks.placeholder)
        << "the native Input read has no encoded VGPR/P1 accumulator dependency";
    EXPECT_FALSE(h.before.source_marks.hard);
    EXPECT_TRUE(h.rs.vreg_source_marks.at(6).hard)
        << "the old unavailable word survives outside the producing EXEC";
    EXPECT_EQ(h.rs.vreg_source_marks.at(6).outside_exec, h.rs.exec);
    EXPECT_TRUE(h.defined());
    EXPECT_EQ(h.rs.native_interpolation_publication.pc, UINT32_MAX)
        << "the successful transfer consumes its publication before any edge";
}

TEST_P(NativeInterpolationWord, SupportedMovDefinesTheMatchingExecLanes) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kMov));
    EXPECT_FALSE(h.before.source_marks.placeholder);
    EXPECT_TRUE(h.rs.vreg_source_marks.at(6).hard);
    EXPECT_EQ(h.rs.vreg_source_marks.at(6).outside_exec, h.rs.exec);
    EXPECT_TRUE(h.defined());
}

TEST_P(NativeInterpolationWord, ActualCmpxSubsetCanReadThePublishedWord) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    const uint32_t owner = h.rs.exec;
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_NE(h.rs.exec, owner);
    EXPECT_TRUE(h.defined());
}

TEST_P(NativeInterpolationWord, RestoringExecCannotReadPreviouslyInactiveWords) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    ASSERT_TRUE(h.emit(kRestore));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, P1DoesNotDefineAnAbsentDestination) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP1));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, MissingScalarCompareInputRemainsMissing) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_FALSE(h.defined(0x7c200c14u)); // compare absent s20 against v6
}

TEST_P(NativeInterpolationWord, GenuineRelayKeepsTheExecConfinement) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    ASSERT_TRUE(h.emit(kRelay));
    EXPECT_TRUE(h.defined(kCompareRelay));
}

TEST_P(NativeInterpolationWord, MissingOverwriteCannotBorrowAnEarlierPublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    ASSERT_TRUE(h.emit(kMissingRelay));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, RelayedMissingOverwriteCannotBorrowAnEarlierPublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2));
    ASSERT_TRUE(h.emit(kMissingRelay));
    ASSERT_TRUE(h.emit(kRelay));
    EXPECT_FALSE(h.defined(kCompareRelay));
}

TEST_P(NativeInterpolationWord, ChangedSsaGenerationCannotConsumeThePublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    h.rs.vreg[6] = h.b.uconst(9); // a reload/rewrite is not the published generation
    h.transfer_pending();
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, ChangedExecGenerationCannotConsumeThePublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    h.rs.exec = h.b.ucmp(Op_IEqual, h.b.uconst(1), h.b.uconst(2));
    h.transfer_pending();
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AnotherInstructionCannotConsumeThePublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    ++h.pending.pc;
    h.transfer_pending();
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AnotherDestinationCannotConsumeThePublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    h.pending.dst.value = 8;
    h.pending.words[0] = 0xc8210014u;
    h.rs.vreg[8] = h.rs.vreg.at(6);
    h.transfer_pending();
    EXPECT_FALSE(h.defined(0x7c201108u)); // read v8 only, independently of untransferred v6
}

TEST_P(NativeInterpolationWord, PublicationDoesNotAuthenticateArchitecturalM0) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    EXPECT_FALSE(h.emit(0x7e10027cu)) << "untracked architectural M0 has no data word";
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_TRUE(h.defined());
    EXPECT_FALSE(h.emit(0x7e10027cu)) << "native Input publication must not authenticate M0";
}

TEST_P(NativeInterpolationWord, P1RetiresAnUnconsumedPublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    ASSERT_NE(h.rs.native_interpolation_publication.pc, UINT32_MAX);
    ASSERT_TRUE(h.emit(kP1, false));
    EXPECT_EQ(h.rs.native_interpolation_publication.pc, UINT32_MAX);
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AFailedMovReadRetiresThePublication) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    ASSERT_TRUE(h.emit(kP2, false));
    EXPECT_FALSE(h.emit(0xc81a0000u)) << "P10 has no Input on the ordinary native path";
    EXPECT_EQ(h.rs.native_interpolation_publication.pc, UINT32_MAX);
}

TEST_P(NativeInterpolationWord, AnUnsupportedSelectorDoesNotPublishAWord) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    EXPECT_FALSE(h.emit(0xc81a0003u));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AReservedOpcodeDoesNotGainPublicationAuthority) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    // The historical native emitter routes opcode 3 to an Input; this proof must not endorse it.
    ASSERT_TRUE(h.emit(0xc81b0014u));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AnInconsistentDecodedWordDoesNotPublishAProof) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    auto in = instruction(kP2);
    in.words[0] ^= 1;
    ASSERT_TRUE(h.emit(in));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, MissingLayoutOwnershipDoesNotPublishAProof) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    h.layout.attribute_mask = 0;
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, InvalidLayoutDoesNotPublishAProof) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    h.layout.valid = false;
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_FALSE(h.defined());
}

TEST_P(NativeInterpolationWord, AGeometryMovNeedsItsActualParameterLocation) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    h.layout.requires_geometry = true;
    EXPECT_FALSE(h.emit(0xc81a0000u));
    h.layout.parameter_locations[0][0] = 1;
    ASSERT_TRUE(h.emit(0xc81a0000u));
    EXPECT_TRUE(h.defined());
}

TEST_P(NativeInterpolationWord, ArchitecturalOrOwnedInterpolationKeepsItsOwnProof) {
    Harness h(GetParam());
    ASSERT_TRUE(h.emit(kNarrow));
    h.rs.vector_native_interpolation = false;
    ASSERT_TRUE(h.emit(kP2));
    EXPECT_FALSE(h.defined());
    EXPECT_FALSE(vector_write_completes_active_word(instruction(kP2)))
        << "storage effects must not teach architectural P2 to ignore its old accumulator";
}

std::atomic<uint64_t> next_address{0xa5099001u};
std::vector<uint32_t> compile(const std::vector<uint32_t>& body, uint32_t width,
                              std::string& reason) {
    auto code = body;
    code.insert(code.end(), {kRestore, 0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
                             0xf800000fu, 0x03020100u, 0xbf810000u});
    const uint64_t address = next_address.fetch_add(1);
    auto result =
        recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                           width == 32, {RecompileDiagnosticStage::Fragment, address});
    reason = last_terminal_reject_reason(address);
    return result;
}

TEST_P(NativeInterpolationWord, ThePublicFragmentCompilerUsesThePublication) {
    std::string reason;
    for (const uint32_t interpolation : {kP2, kMov}) {
        SCOPED_TRACE(interpolation);
        EXPECT_FALSE(
            compile({0x7e000280u, kNarrow, interpolation, kCompare}, GetParam(), reason).empty())
            << reason;
    }
}

TEST_P(NativeInterpolationWord, AOneArmPublicationCannotFillTheSkippedEdge) {
    std::string reason;
    const auto result =
        compile({0x7e000280u, kNarrow, 0xbe820380u, 0xbf068002u, 0xbf840001u, kP2, kCompare},
                GetParam(), reason);
    EXPECT_TRUE(result.empty());
    EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=6"), std::string::npos) << reason;
}

TEST_P(NativeInterpolationWord, BothPredecessorsCanPublishTheSameExecDomain) {
    std::string reason;
    EXPECT_FALSE(compile({0x7e000280u, kNarrow, 0xbe820380u, 0xbf068002u, 0xbf840002u, kP2,
                          0xbf820001u, kP2, kCompare},
                         GetParam(), reason)
                     .empty())
        << reason;
}

INSTANTIATE_TEST_SUITE_P(WaveWidths, NativeInterpolationWord, ::testing::Values(32u, 64u));
}   // namespace
}   // namespace prosper::gpu

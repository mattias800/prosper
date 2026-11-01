// Entry EXEC is a MUST identity over the original CFG, never a current-participation guess.
// The retained words and their later lane-bit projection keep missing physical inputs Hard.
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/native_entry_exec_words.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_entry_exec_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::native_entry_exec_words;
struct Frame {
    SpirvCompute b;
    fixture::Words words;
    std::vector<Rdna2Inst> ins;
    ComputeDispatchLaunchProfile launch;
    explicit Frame(fixture::Words code = fixture::program()) : words(std::move(code)) {
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        const auto c = fixture::config(true);
        launch = compute_dispatch_launch_profile(c);
        prosper::test::bind_checked_compute_launch(b, c);
        b.saved_mask_original_words = words;
        b.saved_mask_original_compute_launch = launch;
        rdna2_walk(words.data(), words.size(), ins);
        initialize_native_entry_exec_words(b, launch, ins, words.data(), words.size());
        for (const auto& in : ins)
            if (native_entry_exec_word_move(b, in))
                b.native_logical_data_beyond_mask_spill_pcs.insert(in.pc);
    }
    const Rdna2Inst& producer() const {
        for (const auto& in : ins)
            if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                in.dst.value == 106 && in.src[0].value == 126)
                return in;
        throw std::runtime_error("fixture has no VCC entry move");
    }
    bool proven() const { return native_entry_exec_word_proven(b, producer()); }
};
} // namespace

TEST(NativeEntryExecProof, EveryIncomingEdgeMustRestoreTheSameEntryMask) {
    fixture::Words code{0xbe82047eu, 0xbf0a8304u, 0xbf840002u, 0xbefe0402u,
                        0xbf820001u, 0xbefe0480u, 0xbeea047eu, 0xbf810000u};
    Frame mixed(code);
    EXPECT_FALSE(mixed.proven()); // one predecessor sets EXEC=0, not entry
    code[5] = 0xbefe0402u;
    Frame both(code);
    EXPECT_TRUE(both.proven());
}

TEST(NativeEntryExecProof, ABackedgeCannotReuseItsFirstIterationIdentity) {
    fixture::Words code{0xbe82047eu, 0xbeea047eu, 0xbefe0480u, 0xbf85fffdu, 0xbf810000u};
    Frame changed(code);
    EXPECT_FALSE(changed.proven());
    code[2] = 0xbf800000u;
    Frame preserved(code);
    EXPECT_TRUE(preserved.proven());
}

TEST(NativeEntryExecProof, EitherPhysicalHalfOverwriteExpiresTheSavedPairIdentity) {
    for (const uint32_t write : {0xbe820380u, 0xbe830380u}) {
        Frame overwritten({0xbe82047eu, write, 0xbefe0402u, 0xbeea047eu, 0xbf810000u});
        EXPECT_FALSE(overwritten.proven());
    }
    Frame preserved({0xbe82047eu, 0xbf800000u, 0xbefe0402u, 0xbeea047eu, 0xbf810000u});
    EXPECT_TRUE(preserved.proven());
}

TEST(NativeEntryExecProof, APlanCannotLeaveItsOriginalBuilderProgramLaunchOrDomain) {
    for (unsigned variant = 0; variant < 11; ++variant) {
        Frame f;
        ASSERT_TRUE(f.proven());
        switch (variant) {
            case 0: f.b.saved_mask_original_words[0] ^= 1; break;
            case 1: f.b.guest_compute_launch.launch.known = 0; break;
            case 2: f.b.saved_mask_original_compute_launch.threads[0]++; break;
            case 3: f.b.guest_local_size[0] = 32; break;
            case 4: f.b.native_subgroup_size = 32; break;
            case 5: f.b.is_fragment = true; break;
            case 6: f.b.physical_domain = PhysicalExecutionDomain::Invocation; break;
            case 7: f.b.ngg_workgroup_shell = true; break;
            case 8: f.b.native_logical_data_beyond_mask_spill_pcs.clear(); break;
            case 9: f.b.groupid[0] = 0; break;
            case 10: f.b.saved_mask_original_fragment_entry = true; break;
        }
        EXPECT_FALSE(f.proven()) << variant;
    }
    Frame owner, foreign;
    foreign.b.native_entry_exec_word_plan = owner.b.native_entry_exec_word_plan;
    EXPECT_FALSE(foreign.proven());
}

TEST(NativeEntryExecProof, ChangedSelectedInstructionsAndLateBoundariesCannotSeedAPlan) {
    Frame f;
    ASSERT_TRUE(f.proven());
    auto changed = f.ins;
    changed[0].src[0].value = 124;
    EXPECT_FALSE(native_entry_exec_word_proven(f.b, changed[0]));
    constrain_native_entry_exec_words_region(f.b, changed);
    EXPECT_FALSE(f.proven());
    // A failed region binding is permanent; a later sliced phase cannot reseed the root plan.
    initialize_native_entry_exec_words(f.b, f.launch, f.ins, f.words.data(), f.words.size());
    EXPECT_FALSE(f.proven());
    Frame late;
    late.b.native_entry_exec_word_plan.reset();
    late.b.saved_mask_word_analysis_done = true;
    initialize_native_entry_exec_words(late.b, late.launch, late.ins, late.words.data(),
                                       late.words.size());
    EXPECT_FALSE(late.proven());
}

TEST(NativeEntryExecProof, OriginalSubregionsKeepTheirExistingAuthority) {
    Frame f;
    ASSERT_TRUE(f.proven());
    const std::vector<Rdna2Inst> region{f.producer(), f.ins.back()};
    constrain_native_entry_exec_words_region(f.b, region);
    EXPECT_TRUE(f.proven());
    auto changed = f.producer();
    changed.src[0].value = 124;
    EXPECT_FALSE(native_entry_exec_word_proven(f.b, changed));
}

TEST(NativeEntryExecProof, CurrentVccDataNeedsBothGenuinePhysicalWords) {
    Frame f;
    RegState rs;
    rs.vector_wave_size = 64;
    rs.sreg[106] = f.b.uconst(13);
    rs.sreg[107] = f.b.uconst(0x00ffffffu);
    ASSERT_NE(native_compute_vcc_data_mask(f.b, rs), 0u);
    for (const int word : {106, 107}) {
        RegState missing = rs;
        missing.sreg.erase(word);
        EXPECT_EQ(native_compute_vcc_data_mask(f.b, missing), 0u);
        RegState gap = rs;
        gap.unavailable_saved_mask_words.insert(word);
        EXPECT_EQ(native_compute_vcc_data_mask(f.b, gap), 0u);
        RegState fabricated = rs;
        fabricated.sreg_merge_placeholder.insert(word);
        EXPECT_EQ(native_compute_vcc_data_mask(f.b, fabricated), 0u);
    }
    f.b.is_fragment = true;
    EXPECT_EQ(native_compute_vcc_data_mask(f.b, rs), 0u);
}

TEST(NativeEntryExecProof, TheOriginalClosedInstructionInventoryOwnsThePlan) {
    Frame f;
    f.b.native_entry_exec_word_plan.reset();
    auto changed = f.ins;
    changed[0].src[0].value = 124;
    initialize_native_entry_exec_words(f.b, f.launch, changed, f.words.data(), f.words.size());
    EXPECT_FALSE(f.proven());
    Frame escaping({0xbeea047eu, 0xbf820010u, 0xbf810000u});
    EXPECT_FALSE(escaping.proven());
    Frame unknown({0xbeea047eu, 0xffffffffu, 0xbf810000u});
    EXPECT_FALSE(unknown.proven());
}

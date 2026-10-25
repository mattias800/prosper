// #5497: a closed local-index packet writes each cohort lane's selected bank word.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_local_index_waterfall.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_relative_bank_source.hpp"
#include "gpu/recompiler/rdna2_relative_index_values.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "bpermute_spirv_oracle.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <set>
#include <optional>
#include <unordered_set>
#include <vector>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program() {
    return {
        0x7e000f00u,                 // v0 = uint(FragCoord.x)
        0x7e4a023cu,                 // old v37 = absent s60, unobservable outside the cohort
        0x7da6009fu,                 // EXEC = (31 <= v0), both Wave64 halves
        0x7e0802ffu, 0x3f800000u,    // v4 = 1.0 only in this cohort
        0x7e1002ffu, 0x40000000u,    // v8 = 2.0 only in this cohort
        0xbea0047eu,                 // saved s[32:33] = EXEC
        0xbea4047eu,                 // remaining s[36:37] = EXEC
        0x36420081u,                 // v33 = v0 & 1
        0x34444282u,                 // v34 = v33 << 2, bank offsets 0 or 4
        0x7e060522u,                 // L: s3 = readfirstlane(v34)
        0x7da44403u,                 // EXEC &= (s3 == v34)
        0xbefc0303u,                 // M0 = s3
        0x7e4a8704u,                 // v37 = VGPR[4 + M0]
        0x8aa47e24u,                 // remaining &= ~EXEC
        0xbefe0424u,                 // EXEC = remaining
        0xbf85fff9u,                 // s_cbranch_scc1 L
        0xbefe0420u,                 // restore the saved cohort
        0xbefc0380u,                 // independent M0 restore; s3 is dead
        0x7d844b25u,                 // fresh VCC = (v37 == v37)
        0x024c4a80u,                 // v38 = VCC ? v37 : 0
        0xf800180fu, 0x26262626u,    // export v38
        0xbf810000u,
    };
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1u << 8;
    return recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}
std::vector<uint32_t> emitted_body(const std::vector<uint32_t>& code, uint32_t& refused_pc) {
    std::vector<Rdna2Inst> ins;
    if (rdna2_walk(code.data(), code.size(), ins) != code.size()) return {};
    const auto latch = local_index_waterfall_latch(ins, 11);
    if (latch == UINT32_MAX) return {};
    SpirvCompute b;
    b.native_subgroup_size = 64;
    b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
    b.is_compute = false;
    b.is_fragment = true;
    b.saved_mask_original_words = code;
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.vreg[0] = b.cvt_u2f(b.localid_comp[0]);
    rs.max_vgpr = 38;
    seed_vector_word_definedness(rs, ins, true, 64);
    const std::unordered_set<uint32_t> safe{latch};
    for (const auto& in : ins) {
        if (in.fmt == Rdna2Format::EXP || in.is_end) break;
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(b, rs, in);
        bool ok = true;
        if (!emit_alu(b, rs, in, ok, true, &safe) || !ok) {
            refused_pc = in.pc;
            return {};
        }
        record_scalar_write(rs, in, false, before);
        retain_saved_mask_words(b, rs, in);
    }
    b.store_output_word(rs.vreg.at(37), 4, 0, rs.exec);
    b.store_output_word(rs.vreg.at(38), 4, 1, rs.exec);
    b.store_output_word(b.sel(rs.exec, b.uconst(1), b.uconst(0)), 4, 2);
    b.store_output_word(b.sel(rs.vcc, b.uconst(1), b.uconst(0)), 4, 3);
    return b.finish();
}
} // namespace

TEST(RelativeBankSource, CohortOnlyBankAndMissingOldDestinationReachFreshMask) {
    EXPECT_FALSE(compile(program(), 0x5497050100ull).empty())
        << last_terminal_reject_reason(0x5497050100ull);
}
TEST(RelativeBankSource, BothWaveHalvesSelectTheirOwnWordAndKeepInactiveOutputUntouched) {
    uint32_t refused = UINT32_MAX;
    const auto module = emitted_body(program(), refused);
    ASSERT_FALSE(module.empty()) << "refused at PC " << refused;
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    const auto result =
        vm.run_buffers(64, {{1, std::vector<uint32_t>(std::size_t{64} * 4, 0xdeadbeefu)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(result.size(), 64u * 4);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto at = static_cast<size_t>(lane) * 4;
        const uint32_t expected = lane < 31 ? 0xdeadbeefu : lane % 2 ? 0x40000000u : 0x3f800000u;
        EXPECT_EQ(result[at], expected) << "selected word, lane " << lane;
        EXPECT_EQ(result[at + 1], expected) << "fresh mask selection, lane " << lane;
        EXPECT_EQ(result[at + 2], lane >= 31 ? 1u : 0u) << "restored EXEC, lane " << lane;
        EXPECT_EQ(result[at + 3], lane >= 31 ? 1u : 0u) << "fresh VCC, lane " << lane;
    }
}
TEST(RelativeBankSource, MissingSelectedBankStillRefuses) {
    auto code = program();
    code[5] = 0x7e10023cu; // v8 = absent s60
    code[6] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497050200ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497050200ull).find("fabricated-fresh-mask-source pc=20"),
        std::string::npos);
}
TEST(RelativeBankSource, ExpandedReaderStillRefusesTheUnwrittenOldLanes) {
    auto code = program();
    code.insert(code.begin() + 20, 0xbefe04c1u); // EXEC = -1 after the independent M0 restore
    EXPECT_TRUE(compile(code, 0x5497050300ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497050300ull).find("fabricated-fresh-mask-source pc=21"),
        std::string::npos);
}

TEST(RelativeBankSource, UnboundedAndOutOfRangeBanksStillRefuse) {
    for (bool unbounded : {false, true}) {
        auto code = program();
        if (unbounded)
            code[9] = 0x7e420300u; // v33 = v0, with no finite mask
        else
            code[10] = 0x34444288u; // v34 = v33 << 8, includes VGPR 260
        EXPECT_TRUE(compile(code, 0x5497050400ull + unbounded).empty());
        EXPECT_NE(last_terminal_reject_reason(0x5497050400ull + unbounded)
                      .find("fabricated-fresh-mask-source pc=20"),
                  std::string::npos);
    }
}

namespace {
struct BankFixture {
    SpirvCompute b;
    RegState rs;
    Rdna2Inst relative;
    uint32_t cohort = 0;
    uint32_t index = 0;
    BankFixture() {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        b.is_compute = false;
        b.is_fragment = true;
        b.saved_mask_original_words = program();
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(b.saved_mask_original_words.data(), b.saved_mask_original_words.size(),
                   instructions);
        for (const auto& in : instructions)
            if (in.pc == 14) relative = in;
        cohort = b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(33));
        rs.exec = b.bsel(cohort, b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(17)), b.bfalse());
        rs.exec_subset_of[rs.exec] = cohort;
        rs.exec_narrowed = true;
        rs.vector_wave_size = 64;
        rs.sreg_bool[32] = cohort;
        index = b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, b.localid_comp[0], b.uconst(1)),
                       b.uconst(2));
        ScalarSourceMarks scoped;
        scoped.placeholder = scoped.hard = scoped.scoped_complete_word = true;
        for (int reg : {3, 124}) {
            rs.sreg[reg] = index;
            rs.sreg_merge_placeholder.insert(reg);
            scoped.outside_exec = reg == 3 ? cohort : rs.exec;
            rs.sreg_exec_scopes[reg] = {index, scoped};
        }
        scoped.outside_exec = cohort;
        scoped.memory = true;
        scoped.blanket_roots = {0x12340100ull}; // retained inactive generation
        scoped.scoped_blanket_roots = {0x23450104ull};
        for (int reg : {4, 8}) {
            rs.vreg[reg] = b.uconst(static_cast<uint32_t>(reg));
            rs.vreg_source_marks[reg] = scoped;
        }
        ScalarSourceMarks old;
        old.placeholder = old.hard = old.memory = true;
        old.blanket_roots = {0x34560125ull};
        rs.vreg[37] = b.uconst(42);
        rs.vreg_source_marks[37] = old;
    }
};
} // namespace

TEST(RelativeBankSource, SelectedRootsKeepActiveAndInactiveGenerationsSeparate) {
    BankFixture f;
    const auto proof = relative_bank_source_proof(f.b, f.rs, f.relative);
    if (!proof) {
        ADD_FAILURE() << "authenticated fixture did not create a bank proof";
        return;
    }
    EXPECT_FALSE(proof->source.hard);
    EXPECT_TRUE(proof->source.memory);
    EXPECT_EQ(proof->source.blanket_roots, (std::set<uint64_t>{0x23450104ull}));
    publish_relative_bank_word(f.rs, f.relative, proof->source, proof->publication);
    const auto& result = f.rs.vreg_source_marks.at(37);
    EXPECT_TRUE(result.placeholder && result.hard && result.memory);
    EXPECT_FALSE(result.arbitrary);
    EXPECT_EQ(result.blanket_roots, (std::set<uint64_t>{0x34560125ull}));
    EXPECT_EQ(result.outside_exec, f.cohort);
    EXPECT_TRUE(result.scoped_complete_word);
    EXPECT_EQ(result.scoped_blanket_roots, proof->source.blanket_roots);
}

TEST(RelativeBankSource, MissingStaleAndWiderSourcesCannotMintTheProof) {
    for (unsigned mutation = 0; mutation < 13; ++mutation) {
        BankFixture f;
        switch (mutation) {
            case 0: f.rs.sreg_exec_scopes.erase(3); break;
            case 1: f.rs.sreg_exec_scopes.erase(124); break;
            case 2: f.rs.unavailable_saved_mask_words.insert(3); break;
            case 3: f.rs.unavailable_saved_mask_words.insert(124); break;
            case 4: f.rs.sreg[124] = f.b.uconst(0); break;
            case 5: f.rs.sreg_memory_pattern.insert(124); break;
            case 6: f.rs.sreg_exec_scopes.at(3).source.outside_exec = f.rs.exec; break;
            case 7: f.rs.sreg_exec_scopes.at(124).source.outside_exec = f.cohort; break;
            case 8: f.rs.vreg.erase(8); break;
            case 9: f.rs.vreg_source_marks.at(8).outside_exec = f.rs.exec; break;
            case 10: f.rs.exec_subset_of.clear(); break;
            case 11: f.b.wave_size = 32; break;
            case 12: f.b.saved_mask_original_words[13] = 0xbefc0304u; break;
        }
        EXPECT_FALSE(relative_bank_source_proof(f.b, f.rs, f.relative)) << mutation;
    }
}

TEST(RelativeBankSource, ChangedInstructionOrGenerationCannotPublishTheCapturedProof) {
    for (unsigned mutation = 0; mutation < 7; ++mutation) {
        BankFixture f;
        auto proof = relative_bank_source_proof(f.b, f.rs, f.relative);
        if (!proof) {
            ADD_FAILURE() << "authenticated fixture did not create a bank proof";
            return;
        }
        switch (mutation) {
            case 0: ++f.relative.pc; break;
            case 1: ++f.relative.dst.value; break;
            case 2: f.rs.exec = f.cohort; break;
            case 3: f.rs.sreg[3] = f.b.uconst(0); break;
            case 4: f.rs.sreg_memory_pattern.insert(124); break;
            case 5: proof->source.hard = true; break;
            case 6: f.rs.sreg_exec_scopes.clear(); break;
        }
        publish_relative_bank_word(f.rs, f.relative, proof->source, proof->publication);
        EXPECT_EQ(f.rs.vreg_source_marks.at(37).outside_exec, 0u) << mutation;
        EXPECT_FALSE(f.rs.vreg_source_marks.at(37).scoped_complete_word) << mutation;
    }
}

TEST(RelativeBankSource, RangeQueryUsesOnlyExistingCohortFactsAndStaysBounded) {
    BankFixture f;
    const auto poisoned = f.b.sel(f.cohort, f.index, f.b.localid_comp[0]);
    EXPECT_EQ(relative_index_values(f.b, f.rs, poisoned, f.cohort),
              (std::optional<std::set<uint32_t>>{{0, 4}}));
    EXPECT_FALSE(relative_index_values(f.b, f.rs, poisoned, f.b.btrue()));
    const auto unknown = f.b.ibin(Op_IAdd, f.index, f.b.uconst(1));
    EXPECT_FALSE(relative_index_values(f.b, f.rs, unknown, f.cohort));
    const auto wide = f.b.ibin(Op_BitwiseAnd, f.b.localid_comp[0], f.b.uconst(63));
    EXPECT_FALSE(relative_index_values(f.b, f.rs, wide, f.cohort));
    const auto bad_shift = f.b.ibin(Op_ShiftLeftLogical, f.index, f.b.uconst(32));
    EXPECT_FALSE(relative_index_values(f.b, f.rs, bad_shift, f.cohort));
}

TEST(RelativeBankSource, PublicationDoesNotNarrowExistingWiderCoverage) {
    BankFixture f;
    const auto proof = relative_bank_source_proof(f.b, f.rs, f.relative);
    if (!proof) {
        ADD_FAILURE() << "authenticated fixture did not create a bank proof";
        return;
    }
    const auto whole = f.b.btrue();
    f.rs.exec_subset_of[f.cohort] = whole;
    auto& result = f.rs.vreg_source_marks.at(37);
    result.outside_exec = whole;
    publish_relative_bank_word(f.rs, f.relative, proof->source, proof->publication);
    EXPECT_EQ(result.outside_exec, whole);
    EXPECT_FALSE(result.scoped_complete_word);
}

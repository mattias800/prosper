// #5497: a once-through local index waterfall may read a word defined only in its
// current EXEC cohort. The scalar temporary must not invent an undefined lane.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_local_index_waterfall.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> program() {
    return {
        0x7e000f00u,                 // v0 = uint(FragCoord.x)
        0x7e4a02ffu, 0x40400000u,    // v37 = 3.0, retained outside the body
        0x7e0802ffu, 0x3f800000u,    // v4 = 1.0
        0x7e1002ffu, 0x40000000u,    // v8 = 2.0
        0x7da6009fu,                 // EXEC = (31 <= v0), includes both Wave64 halves
        0xbea0047eu,                 // saved cohort s[32:33] = EXEC
        0xbea4047eu,                 // remaining s[36:37] = EXEC
        0x36420081u,                 // v33 = v0 & 1
        0x34444282u,                 // v34 = v33 << 2, local indices 0 or 4
        0x7e060522u,                 // L: s3 = readfirstlane(v34)
        0x7da44403u,                 // EXEC &= (s3 == v34)
        0xbefc0303u,                 // M0 = s3
        0x7e4a8704u,                 // v37 = VGPR[4 + M0]
        0x8aa47e24u,                 // remaining &= ~EXEC
        0xbefe0424u,                 // EXEC = remaining
        0xbf85fff9u,                 // s_cbranch_scc1 L
        0xbefe0420u,                 // restore cohort
        0xbefc0380u,                 // M0 = 0; index s3 is dead
        0xf800180fu, 0x25252525u,    // export v37 to every colour component
        0xbf810000u,
    };
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1u << 8; // genuine varying FragCoord.x
    return recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}

uint32_t latch(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    if (rdna2_walk(code.data(), code.size(), ins) != code.size()) return UINT32_MAX;
    return local_index_waterfall_latch(ins, 12);
}

// The oracle executes a compute transport with fragment ALU policy. Guest input
// v0 has exactly float(lane), and the three stored words expose v37, EXEC and VCC.
// The public fragment compiler above separately checks the real input/export shell.
std::vector<uint32_t> emitted_body(const std::vector<uint32_t>& code, uint32_t& refused_pc) {
    std::vector<Rdna2Inst> ins;
    if (rdna2_walk(code.data(), code.size(), ins) != code.size()) return {};
    SpirvCompute b;
    b.native_subgroup_size = 64;
    b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
    b.is_compute = false;
    b.is_fragment = true;
    b.saved_mask_original_words = code;
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(17));
    rs.scc = b.bfalse();
    rs.vreg[0] = b.cvt_u2f(b.localid_comp[0]);
    rs.max_vgpr = 37;
    seed_vector_word_definedness(rs, ins, true, 64);
    const uint32_t back = local_index_waterfall_latch(ins, 12);
    if (back == UINT32_MAX) return {};
    const std::unordered_set<uint32_t> safe{back};
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
    b.store_output_word(rs.vreg.at(37), 3, 0);
    b.store_output_word(b.sel(rs.exec, b.uconst(1), b.uconst(0)), 3, 1);
    b.store_output_word(b.sel(rs.vcc, b.uconst(1), b.uconst(0)), 3, 2);
    return b.finish();
}
} // namespace

TEST(LocalIndexWaterfall, CurrentCohortIndexReachesTheFragmentExport) {
    EXPECT_FALSE(compile(program(), 0x5497030100ull).empty())
        << last_terminal_reject_reason(0x5497030100ull);
}

TEST(LocalIndexWaterfall, NeverWrittenIndexStillRefusesAtThePredicate) {
    auto code = program();
    code[11] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497030200ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497030200ull).find("fabricated-fresh-mask-source pc=13"),
        std::string::npos);
}

TEST(LocalIndexWaterfall, EveryLaneSelectsItsOwnBankAndPreservesInactiveWordsAndVcc) {
    uint32_t refused = UINT32_MAX;
    const auto module = emitted_body(program(), refused);
    ASSERT_FALSE(module.empty()) << "refused at PC " << refused;
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    const auto result = vm.run_buffers(64, {{1, std::vector<uint32_t>(64 * 3, 0xdeadbeefu)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(result.size(), 64u * 3);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t value = lane < 31 ? 0x40400000u : lane % 2 ? 0x40000000u : 0x3f800000u;
        EXPECT_EQ(result[lane * 3], value) << "bank/old word, lane " << lane;
        EXPECT_EQ(result[lane * 3 + 1], lane >= 31 ? 1u : 0u) << "EXEC, lane " << lane;
        EXPECT_EQ(result[lane * 3 + 2], lane < 17 ? 1u : 0u) << "VCC, lane " << lane;
    }
}

TEST(LocalIndexWaterfall, MissingIndexCannotBecomeAnExecutedPredicate) {
    auto code = program();
    code[11] = 0xbf800000u;
    uint32_t refused = UINT32_MAX;
    EXPECT_TRUE(emitted_body(code, refused).empty());
    EXPECT_EQ(refused, 13u);
}

TEST(LocalIndexWaterfall, ClosedRegisterOnlyBodyHasOneLocalIteration) {
    EXPECT_EQ(latch(program()), 18u);
}

TEST(LocalIndexWaterfall, AReadOfTheScalarIndexAfterTheBodyInvalidatesTheProof) {
    auto code = program();
    code.insert(code.begin() + 21, 0xbe850303u); // s5 = s3, observable outside the body
    EXPECT_EQ(latch(code), UINT32_MAX);
}

TEST(LocalIndexWaterfall, CmpxPreservesVccSoAnOutsideDataReadCanRemainLive) {
    auto code = program();
    code.insert(code.begin() + 21, 0x7e4a026au); // v37 = VCC_LO
    EXPECT_EQ(latch(code), 18u);
}

TEST(LocalIndexWaterfall, DifferentCompareOrRestoredCohortDoesNotProveTheIndex) {
    for (const auto [at, word] : std::vector<std::pair<size_t, uint32_t>>{
             {13, 0x7da44203u}, // compares v33 instead of the RFL source v34
             {9, 0xbea40480u},  // remaining = 0, rather than the current EXEC cohort
             {19, 0xbefe0424u}, // restores the exhausted remainder, rather than saved EXEC
             {20, 0xbefc0303u}, // M0 retains the lane-varying scalar index
             {20, 0xbefc037cu}, // M0 = M0 also retains that index
         }) {
        auto code = program();
        code[at] = word;
        EXPECT_EQ(latch(code), UINT32_MAX) << "mutation at " << at;
    }
}

TEST(LocalIndexWaterfall, ABranchIntoTheBodyOrAnExecChangeAfterSavingInvalidatesIt) {
    auto code = program();
    code[10] = 0xbf880001u; // branch enters the first predicate, bypassing the RFL
    EXPECT_EQ(latch(code), UINT32_MAX);
    code = program();
    code[10] = 0xbefe04c1u; // widen after the saved cohort
    EXPECT_EQ(latch(code), UINT32_MAX);
}

TEST(LocalIndexWaterfall, MemoryOrScalarCopiesInsideTheBodyNeedAnotherProof) {
    for (const uint32_t word : {0xbe850303u, 0xbf800000u}) {
        auto code = program();
        code[15] = word; // a scalar copy, or no relative move at all
        EXPECT_EQ(latch(code), UINT32_MAX);
    }
    auto code = program();
    code.insert(code.begin() + 15, {0xe0302000u, 0x80020000u}); // buffer_load_dword
    code[20] = 0xbf85fff7u; // retain the same RFL branch target after the insertion
    EXPECT_EQ(latch(code), UINT32_MAX);
}

namespace {
struct HardCohort {
    SpirvCompute builder;
    RegState registers;
    ScalarSourceMarks source;
    Rdna2Inst read;
    HardCohort() {
        builder.is_fragment = true;
        builder.wave_size = 64;
        builder.saved_mask_original_words = program();
        read = rdna2_decode_one(builder.saved_mask_original_words.data() + 12, 1);
        read.pc = 12;
        registers.exec = 7;
        registers.exec_narrowed = true;
        registers.vector_wave_size = 64;
        registers.sreg[3] = 101;
        registers.sreg_merge_placeholder.insert(3);
        registers.vreg[1] = 201;
        source.placeholder = source.hard = source.arbitrary = source.scoped_complete_word = true;
        source.outside_exec = registers.exec;
    }
    void record(bool local = true) {
        record_scalar_exec_scope(registers, read, 3, false, source, false, local);
    }
    bool compare() {
        const uint32_t word = 0x7d840203u; // vcc = s3 == v1
        return fresh_mask_sources_defined(builder, registers, rdna2_decode_one(&word, 1));
    }
};
} // namespace

TEST(LocalIndexWaterfall, HardScalarViewNeedsTheExactOriginalFragmentRoute) {
    HardCohort cohort;
    EXPECT_TRUE(scalar_scope_local_index_waterfall(cohort.builder, cohort.read));
    auto changed = cohort.read;
    changed.src[0].value = 33;
    EXPECT_FALSE(scalar_scope_local_index_waterfall(cohort.builder, changed));
    cohort.builder.is_fragment = false;
    cohort.builder.is_compute = true;
    EXPECT_FALSE(scalar_scope_local_index_waterfall(cohort.builder, cohort.read));
    cohort.record(false);
    EXPECT_EQ(current_scalar_exec_scope(cohort.registers, 3), nullptr);
    EXPECT_FALSE(cohort.compare());
}

TEST(LocalIndexWaterfall, HardScalarViewCannotAuthenticateExpandedExec) {
    HardCohort cohort;
    cohort.record();
    const auto* scope = current_scalar_exec_scope(cohort.registers, 3);
    ASSERT_NE(scope, nullptr);
    EXPECT_FALSE(scope->arbitrary);
    EXPECT_TRUE(cohort.compare());
    cohort.registers.exec = 8;
    EXPECT_FALSE(cohort.compare());
}

TEST(LocalIndexWaterfall, HardScalarViewExpiresWithValuePhysicalBackingAndGlobalMarks) {
    for (int mutation = 0; mutation < 4; ++mutation) {
        HardCohort cohort;
        cohort.record();
        ASSERT_NE(current_scalar_exec_scope(cohort.registers, 3), nullptr);
        if (mutation == 0) cohort.registers.sreg[3] = 102;
        if (mutation == 1) cohort.registers.unavailable_saved_mask_words.insert(3);
        if (mutation == 2) cohort.registers.sreg_loop_blanket[3] = {blanket_root(5, 256)};
        if (mutation == 3) cohort.registers.sreg_memory_pattern.insert(3);
        EXPECT_EQ(current_scalar_exec_scope(cohort.registers, 3), nullptr) << mutation;
        if (mutation == 2) {
            // The original scalar policy admits a soft loop root and records its
            // back-edge obligation; invalidating this view must preserve that contract.
            EXPECT_TRUE(cohort.compare());
            EXPECT_TRUE(cohort.builder.loop_blanket_roots_read.contains(blanket_root(5, 256)));
        } else {
            EXPECT_FALSE(cohort.compare()) << mutation;
        }
    }
}

TEST(LocalIndexWaterfall, GeneralAgreementAndScalarRelaysCannotCreateHardViews) {
    HardCohort cohort;
    record_scalar_exec_scope(cohort.registers, cohort.read, 3, false, cohort.source, true);
    EXPECT_EQ(current_scalar_exec_scope(cohort.registers, 3), nullptr);
    const uint32_t word = 0xbe830303u; // s3 = s3
    record_scalar_exec_scope(cohort.registers, rdna2_decode_one(&word, 1), 3, false, cohort.source,
                             false, true);
    EXPECT_EQ(current_scalar_exec_scope(cohort.registers, 3), nullptr);
    EXPECT_FALSE(cohort.compare());
}

TEST(LocalIndexWaterfall, IncompleteUnownedAndBallotSourcesCannotCreateHardViews) {
    for (int mutation = 0; mutation < 3; ++mutation) {
        HardCohort cohort;
        if (mutation == 0) cohort.source.scoped_complete_word = false;
        if (mutation == 1) cohort.source.outside_exec = 8;
        record_scalar_exec_scope(cohort.registers, cohort.read, 3, mutation == 2, cohort.source,
                                 false, true);
        EXPECT_EQ(current_scalar_exec_scope(cohort.registers, 3), nullptr) << mutation;
        EXPECT_FALSE(cohort.compare()) << mutation;
    }
}

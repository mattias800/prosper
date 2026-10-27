#include "gpu/recompiler/rdna2_saved_mask_phi.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
// #5497: an unchanged saved EXEC mask may acquire a self-carried Bool PHI.
// Restoring that exact value must retain its cohort without authenticating missing DATA.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program() {
    return {
        0x7e020300u,                 // v1 = lane, output index
        0x7e080280u,                 // v4 = 0, flag including inactive lanes
        0x7e0a0280u,                 // v5 = 0, observed loop count
        0x7e0e0281u,                 // v7 = 1, fresh-mask selection value
        0x7e040211u,                 // old v2 = absent s17, inactive bits remain unavailable
        0xbe94047eu,                 // s[20:21] = whole EXEC
        0x7da6009fu,                 // EXEC = (31 <= lane)
        0xbe98047eu,                 // s[24:25] = this cohort
        0x7e040287u,                 // v2 = 7 only in this cohort
        0xbea90380u,                 // s41 = 0
        0xbf068110u,                 // L: SCC = (genuine workgroup id s16 == 1)
        0xbf850001u,                 // scalar if selects the general SCC-loop route
        0x7e060282u,                 // scratch v3 = 2 on this arm
        0x4a0a0a81u,                 // v5 += 1 in active lanes
        0x80298129u,                 // s41 += 1
        0xbf0a8329u,                 // SCC = (s41 < 3)
        0xbf85fff9u,                 // SCC1 -> L
        0xbefe0418u,                 // restore the unchanged, PHI-carried saved cohort
        0x7d840487u,                 // fresh VCC = (7 == v2)
        0x02080e80u,                 // v4 = VCC ? v7 : 0
        0xbefe0414u,                 // restore whole EXEC for both output words
        0xe0702000u, 0x80020401u,    // output[lane].flag = v4
        0xe0702004u, 0x80020501u,    // output[lane].count = v5
        0xbf810000u,
    };
}
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    c.tgid_x_en = true;
    prosper::test::complete_compute_launch(c);
    return c;
}
ShaderResourceTable resources() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 8;
    out.size = 64 * 8;
    out.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(out);
    return table;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    const auto c = config();
    const auto table = resources();
    return recompile_compute(code.data(), code.size(), &table, c,
                             {RecompileDiagnosticStage::Compute, address});
}
bool has_op(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == opcode) return true;
        at += width;
    }
    return false;
}
void evaluate(const std::vector<uint32_t>& code, uint64_t address, uint32_t count) {
    const auto module = compile(code, address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    EXPECT_EQ(has_op(module, 246), count != 0);
    EXPECT_FALSE(has_op(module, 251));
    const auto c = config();
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto words =
        vm.run_buffers(64, {{3, std::vector<uint32_t>(std::size_t{64} * 2, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(words.size(), 64u * 2);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto at = static_cast<size_t>(lane) * 2;
        EXPECT_EQ(words[at], lane >= 31 ? 1u : 0u) << "fresh predicate, lane " << lane;
        EXPECT_EQ(words[at + 1], lane >= 31 ? count : 0u) << "loop count, lane " << lane;
    }
}
} // namespace
TEST(SavedMaskPhi, AnUnchangedLoopMaskRestoresTheOriginalDataCohort) {
    EXPECT_FALSE(compile(program(), 0x5497070100ull).empty())
        << last_terminal_reject_reason(0x5497070100ull);
}
TEST(SavedMaskPhi, NativeWave64ObservesBothMaskHalvesAndThreeIterations) {
    evaluate(program(), 0x5497070200ull, 3);
}
TEST(SavedMaskPhi, TheDirectSavedMaskControlHasTheSamePredicate) {
    auto code = program();
    for (size_t at = 10; at < 17; ++at) code[at] = 0xbf800000u;
    evaluate(code, 0x5497070300ull, 0);
}
TEST(SavedMaskPhi, AChangedBackedgeMaskCannotBorrowTheOriginalOwner) {
    auto code = program();
    code[12] = 0xbe9804c1u; // saved mask = -1 on this arm, a real wider generation
    EXPECT_TRUE(compile(code, 0x5497070400ull).empty());
}
TEST(SavedMaskPhi, ABlockWithoutASavedMaskCannotInventOne) {
    auto code = program();
    code[7] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497070500ull).empty());
}
TEST(SavedMaskPhi, AnUnavailableScalarRelayRemainsUnavailableInTheRestoredCohort) {
    auto code = program();
    code[8] = 0x7e040211u; // v2 = absent s17
    EXPECT_TRUE(compile(code, 0x5497070600ull).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x5497070600ull).find("fabricated-fresh-mask-source pc=18"),
        std::string::npos);
}

namespace {
struct PhiFixture {
    SpirvCompute b;
    uint32_t cohort = 0, phi = 0, entry = 0, header = 0, back = 0, exit = 0;
    size_t patch = 0, back_branch = 0;
    PhiFixture() {
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        entry = b.cur_block;
        cohort = b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(33));
        header = b.id();
        back = b.id();
        exit = b.id();
        b.emit_branch(header);
        b.emit_label(header);
        phi = b.emit_phi2(b.t_bool, cohort, entry, patch);
        b.emit_loopmerge(exit, back);
        b.emit_condbranch(b.btrue(), exit, back);
        b.emit_label(back);
        back_branch = b.code.size();
        b.emit_branch(header);
        b.patch_phi(patch, phi, back);
        b.emit_label(exit);
    }
    Rdna2Inst move() {
        const uint32_t word = 0xbefe0418u;
        b.saved_mask_original_words = {word};
        auto in = rdna2_decode_one(&word, 1);
        in.pc = 0;
        return in;
    }
    RegState state() const {
        RegState rs;
        rs.vector_wave_size = 64;
        rs.exec = phi;
        rs.sreg_bool[24] = phi;
        rs.unavailable_saved_mask_words = {24, 25};
        rs.sreg_merge_placeholder = {24, 25};
        return rs;
    }
};
} // namespace
TEST(SavedMaskPhi, CompletedSelfAndIdenticalIncomingValuesRetainTheirSeed) {
    PhiFixture f;
    EXPECT_EQ(saved_mask_phi_identity(f.b, f.phi), f.cohort);
    f.b.patch_phi(f.patch, f.cohort, f.back);
    EXPECT_EQ(saved_mask_phi_identity(f.b, f.phi), f.cohort);
}
TEST(SavedMaskPhi, PendingChangedUntypedAndUnclosedPhisProvideNoIdentity) {
    for (unsigned mutation = 0; mutation < 8; ++mutation) {
        PhiFixture f;
        switch (mutation) {
            case 0: f.b.code[f.patch] = 0; break;
            case 1: f.b.code[f.patch + 1] = 0; break;
            case 2: f.b.patch_phi(f.patch, f.b.btrue(), f.back); break;
            case 3: f.b.code[f.patch - 2] = f.phi; break;
            case 4: f.b.code[f.patch - 4] = f.b.t_u32; break;
            case 5: f.b.code[f.patch + 1] = f.entry; break;
            case 6: f.b.code[f.back_branch + 1] = f.exit; break;
            case 7: f.b.code.push_back(0); break;
        }
        EXPECT_EQ(saved_mask_phi_identity(f.b, f.phi), 0u) << mutation;
    }
}
TEST(SavedMaskPhi, BoundedQueryRejectsMissingAndOversizedGraphs) {
    PhiFixture f;
    EXPECT_EQ(saved_mask_phi_identity(f.b, f.b.id()), 0u);
    f.b.code.resize(65537, 0);
    EXPECT_EQ(saved_mask_phi_identity(f.b, f.phi), 0u);
}
TEST(SavedMaskPhi, AncestryDoesNotSupplyPhysicalWordsOrClearLifetimeMarks) {
    PhiFixture f;
    const auto in = f.move();
    auto rs = f.state();
    const auto unavailable = rs.unavailable_saved_mask_words;
    const auto placeholders = rs.sreg_merge_placeholder;
    record_saved_mask_phi_subset(f.b, rs, in);
    EXPECT_EQ(rs.exec_subset_of.at(f.phi), f.cohort);
    EXPECT_TRUE(rs.sreg.empty());
    EXPECT_EQ(rs.unavailable_saved_mask_words, unavailable);
    EXPECT_EQ(rs.sreg_merge_placeholder, placeholders);
    EXPECT_EQ(rs.exec, f.phi);
}
TEST(SavedMaskPhi, WrongExecutionOrOriginalSourceCannotBorrowTheIdentity) {
    for (unsigned mutation = 0; mutation < 9; ++mutation) {
        PhiFixture f;
        auto in = f.move();
        auto rs = f.state();
        switch (mutation) {
            case 0: f.b.wave_size = 32; break;
            case 1: f.b.native_subgroup_size = 0; break;
            case 2: f.b.is_fragment = true; break;
            case 3: f.b.ngg_one_lane = true; break;
            case 4: f.b.saved_mask_original_words[0] ^= 1u; break;
            case 5: rs.sreg_bool.clear(); break;
            case 6: rs.sreg_bool_b32.insert(24); break;
            case 7: rs.sreg_bool.at(24) = f.cohort; break;
            case 8: in.synthetic_terminator = true; break;
        }
        record_saved_mask_phi_subset(f.b, rs, in);
        EXPECT_TRUE(rs.exec_subset_of.empty()) << mutation;
    }
}
TEST(SavedMaskPhi, ExistingAncestryIsRetained) {
    PhiFixture f;
    const auto in = f.move();
    auto rs = f.state();
    const auto prior = f.b.btrue();
    rs.exec_subset_of[f.phi] = prior;
    record_saved_mask_phi_subset(f.b, rs, in);
    EXPECT_EQ(rs.exec_subset_of.at(f.phi), prior);
}

#include "gpu/recompiler/rdna2_accumulator_cohort.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
// #5497: a same-lane accumulator must retain genuine old lanes outside its narrower write.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program() {
    return {
        0x7e020300u, // v1 = lane, output index
        0x7e0c0280u, // v6 = flag 0 globally
        0x7e0e0281u, // v7 = selection 1 globally
        0x7e04023cu, // old v2 = absent s60
        0x7e06023cu, // old v3 = absent s60
        0x7e08023cu, // old v4 = absent s60
        0xbe84047eu, // s[4:5] = whole EXEC
        0x7da6009fu, // C = lanes 31..63
        0x7e0402ffu, 0x3f800000u, // v2 = 1.0 in C
        0xbe94047eu, // save C in s[20:21]
        0x7da600afu, // D = lanes 47..63, a strict subset of C
        0x7e0602ffu, 0x40000000u, // v3 = 2.0 in D
        0x7e0802ffu, 0x40400000u, // v4 = 3.0 in D
        0x3e040903u, // v2 += v3 * v4 in D; the remaining C lanes retain 1.0
        0xbefe0414u, // restore C
        0x7d840502u, // fresh VCC = (v2 == v2)
        0x020c0e80u, // flag v6 = VCC ? v7 : 0
        0xe0702000u, 0x80020201u, // output[lane].value = v2 in C
        0xbefe0404u, // restore whole EXEC
        0xe0702004u, 0x80020601u, // output[lane].flag = v6 globally
        0xbf810000u,
    };
}
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
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
    ShaderResourceTable t;
    t.resources.push_back(out);
    return t;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    const auto c = config();
    const auto t = resources();
    return recompile_compute(code.data(), code.size(), &t, c,
                             {RecompileDiagnosticStage::Compute, address});
}
}   // namespace
TEST(AccumulatorCohort, APartialAccumulatorRetainsTheOriginalCohort) {
    EXPECT_FALSE(compile(program(), 0x5497090100ull).empty())
        << last_terminal_reject_reason(0x5497090100ull);
}
TEST(AccumulatorCohort, NativeWave64ObservesBothCohortsAndInactiveOutput) {
    const auto module = compile(program(), 0x5497090200ull);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x5497090200ull);
    const auto c = config();
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto words =
        vm.run_buffers(64, {{3, std::vector<uint32_t>(std::size_t{64} * 2, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(words.size(), 128u);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        EXPECT_EQ(words[std::size_t{2} * lane],
                  lane < 31 ? 0xdeadbeefu : (lane < 47 ? 0x3f800000u : 0x40e00000u))
            << lane;
        EXPECT_EQ(words[std::size_t{2} * lane + 1], lane >= 31 ? 1u : 0u) << lane;
    }
}
TEST(AccumulatorCohort, AMissingActiveSourceStillRefuses) {
    auto code = program();
    code[12] = 0x7e06023cu;
    code[13] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497090300ull).empty());
}
TEST(AccumulatorCohort, AMissingOldAccumulatorStillRefuses) {
    auto code = program();
    code[8] = 0x7e04023cu;
    code[9] = 0xbf800000u;
    EXPECT_TRUE(compile(code, 0x5497090400ull).empty());
}
TEST(AccumulatorCohort, AReaderOutsideTheOriginalCohortStillRefuses) {
    auto code = program();
    code[17] = 0xbefe0404u;
    EXPECT_TRUE(compile(code, 0x5497090500ull).empty());
}

TEST(AccumulatorCohort, TheNativeFragmentEntryRetainsTheSameCohort) {
    auto code = program();
    code.resize(20);
    code.insert(code.begin(), 0x7e000f00u);
    code.insert(code.end(), {0xf800180fu, 0x06060606u, 0xbf810000u});
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1u << 8;
    constexpr uint64_t address = 0x5497090600ull;
    EXPECT_FALSE(recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX, nullptr,
                                    false, {RecompileDiagnosticStage::Fragment, address})
                     .empty())
        << last_terminal_reject_reason(address);
}

namespace {
struct WordFrame {
    RegState rs;
    Rdna2Inst in;
    ScalarSourceMarks source;
    uint64_t old_active = blanket_root(5, 258), old_inactive = blanket_root(6, 258);
    uint64_t input_active = blanket_root(7, 259), input_inactive = blanket_root(8, 259);
    WordFrame() {
        const uint32_t word = 0x3e040903u;
        in = rdna2_decode_one(&word, 1);
        in.pc = 16;
        rs.vector_wave_size = 64;
        rs.exec_narrowed = true;
        rs.exec = 20;
        rs.exec_subset_of[20] = 10;
        rs.vreg[2] = 204;
        rs.unavailable_saved_mask_words = {60, 61};
        rs.sreg_merge_placeholder = {60, 61};
        auto& old = rs.vreg_source_marks[2];
        old.placeholder = old.hard = old.memory = old.scoped_complete_word = true;
        old.outside_exec = 10;
        old.blanket_roots = {old_active, old_inactive};
        old.scoped_blanket_roots = {old_active};
        source.placeholder = source.hard = source.memory = source.scoped_complete_word = true;
        source.outside_exec = 20;
        source.blanket_roots = {input_active, input_inactive};
        source.scoped_blanket_roots = {input_active};
    }
};
void expect_same_planes(const ScalarSourceMarks& a, const ScalarSourceMarks& b) {
    EXPECT_EQ(a.placeholder, b.placeholder);
    EXPECT_EQ(a.hard, b.hard);
    EXPECT_EQ(a.memory, b.memory);
    EXPECT_EQ(a.arbitrary, b.arbitrary);
    EXPECT_EQ(a.scoped_complete_word, b.scoped_complete_word);
    EXPECT_EQ(a.blanket_roots, b.blanket_roots);
    EXPECT_EQ(a.scoped_blanket_roots, b.scoped_blanket_roots);
}
}   // namespace
TEST(AccumulatorCohort, ARefinementChangesOnlyTheOrdinaryCoverageOwner) {
    WordFrame f;
    auto ordinary = f.rs;
    record_vector_word_marks(ordinary, f.in, f.source);
    record_vector_word_marks_preserving_cohort(f.rs, f.in, f.source);
    const auto& before = ordinary.vreg_source_marks.at(2);
    const auto& after = f.rs.vreg_source_marks.at(2);
    EXPECT_EQ(before.outside_exec, 20u);
    EXPECT_EQ(after.outside_exec, 10u);
    expect_same_planes(before, after);
    EXPECT_EQ(f.rs.vreg, ordinary.vreg);
    EXPECT_EQ(f.rs.exec, ordinary.exec);
    EXPECT_EQ(f.rs.exec_subset_of, ordinary.exec_subset_of);
    EXPECT_EQ(f.rs.unavailable_saved_mask_words, ordinary.unavailable_saved_mask_words);
    EXPECT_EQ(f.rs.sreg_merge_placeholder, ordinary.sreg_merge_placeholder);
}
TEST(AccumulatorCohort, BothActiveDependenciesRemainLiveAndInactiveRootsStaySeparate) {
    WordFrame f;
    record_vector_word_marks_preserving_cohort(f.rs, f.in, f.source);
    f.rs.exec = 10;
    const uint32_t word = 0x7d840502u;
    const auto compare = rdna2_decode_one(&word, 1);
    SpirvCompute b;
    b.is_compute = true;
    b.wave_size = 64;
    EXPECT_TRUE(fresh_mask_sources_defined(b, f.rs, compare));
    EXPECT_TRUE(b.loop_blanket_roots_read.contains(f.old_active));
    EXPECT_TRUE(b.loop_blanket_roots_read.contains(f.input_active));
    EXPECT_FALSE(b.loop_blanket_roots_read.contains(f.old_inactive));
    EXPECT_FALSE(b.loop_blanket_roots_read.contains(f.input_inactive));
    f.rs.exec = 30;
    EXPECT_FALSE(fresh_mask_sources_defined(b, f.rs, compare));
}
TEST(AccumulatorCohort, ACurrentReaderKeepsTheOrdinaryActiveRootSet) {
    WordFrame f;
    auto ordinary = f.rs;
    record_vector_word_marks(ordinary, f.in, f.source);
    record_vector_word_marks_preserving_cohort(f.rs, f.in, f.source);
    const auto& before = ordinary.vreg_source_marks.at(2);
    const auto& after = f.rs.vreg_source_marks.at(2);
    EXPECT_TRUE(vector_word_defined_in_exec(ordinary, before, 20));
    EXPECT_TRUE(vector_word_defined_in_exec(f.rs, after, 20));
    EXPECT_EQ(before.scoped_blanket_roots, after.scoped_blanket_roots);
    EXPECT_TRUE(vector_word_defined_in_exec(f.rs, after, 10));
    EXPECT_FALSE(vector_word_defined_in_exec(ordinary, before, 10));
}
TEST(AccumulatorCohort, UnsupportedOrIncompleteContextsKeepTheOrdinaryTransfer) {
    for (unsigned mutation = 0; mutation < 17; ++mutation) {
        WordFrame f;
        switch (mutation) {
            case 0: f.rs.vector_wave_size = 32; break;
            case 1: f.rs.exec_narrowed = false; break;
            case 2: f.rs.vector_full_exec = true; break;
            case 3: f.rs.exec_subset_of.clear(); break;
            case 4: f.rs.vreg_source_marks.at(2).scoped_complete_word = false; break;
            case 5: f.rs.vreg_source_marks.at(2).outside_exec = 0; break;
            case 6: f.rs.vreg_source_marks.at(2).hard = false; break;
            case 7: f.rs.vreg_source_marks.at(2).outside_exec = 20; break;
            case 8:
                f.source.scoped_complete_word = false;
                f.source.outside_exec = 0;
                break;
            case 9: f.source.outside_exec = 30; break;
            case 10: f.in.has_modifier = true; break;
            case 11: f.in.has_dpp = true; break;
            case 12: f.in.has_sdwa = true; break;
            case 13: f.in.words[0] ^= 1u; break;
            case 14: f.rs.vreg.clear(); break;
            case 15: f.rs.vreg.at(2) = 0; break;
            case 16: f.in.len_dwords = 0; break;
        }
        auto ordinary = f.rs;
        record_vector_word_marks(ordinary, f.in, f.source);
        record_vector_word_marks_preserving_cohort(f.rs, f.in, f.source);
        ASSERT_EQ(f.rs.vreg_source_marks.size(), ordinary.vreg_source_marks.size()) << mutation;
        for (const auto& [reg, mark] : ordinary.vreg_source_marks) {
            const auto& after = f.rs.vreg_source_marks.at(reg);
            expect_same_planes(mark, after);
            EXPECT_EQ(mark.outside_exec, after.outside_exec) << mutation;
        }
    }
}

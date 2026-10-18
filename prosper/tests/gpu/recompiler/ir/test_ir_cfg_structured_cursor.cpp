// Cursor peeks respect recursive region bounds without consuming them; taking a branch
// retains every detector field and the diagnostic cursor position.
#include "gpu/recompiler/ir/cfg/structured_if_cursor.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/ir/cfg_structured_cases.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
namespace {
ForwardIf branch(uint32_t pc, uint32_t target) {
    ForwardIf f;
    f.found = true;
    f.branch_pc = pc;
    f.target_pc = target;
    return f;
}
TEST(IrCfgStructuredCursor, PeekBoundsDoNotConsumeTheNextBranch) {
    const std::vector<Rdna2Inst> ins;
    const std::vector<ForwardIf> fs{branch(3, 7), branch(8, 12)};
    StructuredIfCursor cursor(ins, fs, {}, false);
    EXPECT_EQ(cursor.next_before(0), 0u);
    EXPECT_EQ(cursor.next_before(3), 3u);
    EXPECT_EQ(cursor.next_pc(), 3u);
    EXPECT_EQ(cursor.next_before(4), 3u);
    EXPECT_EQ(cursor.take().branch_pc, 3u);
    EXPECT_EQ(cursor.next_before(8), 8u);
    EXPECT_EQ(cursor.next_pc(), 8u);
    EXPECT_EQ(cursor.take().branch_pc, 8u);
    EXPECT_EQ(cursor.next_pc(), UINT32_MAX);
}
TEST(IrCfgStructuredCursor, TakingAnElseRetainsItsConditionAndMergeFields) {
    auto f = branch(4, 10);
    f.on_scc0 = false;
    f.on_vcc = true;
    f.uniform_workgroup = true;
    f.has_else = true;
    f.sb_pc = 9;
    f.merge_pc = 15;
    const std::vector<Rdna2Inst> ins;
    const std::vector<ForwardIf> fs{f};
    StructuredIfCursor cursor(ins, fs, {}, false);
    const auto out = cursor.take();
    EXPECT_TRUE(out.found);
    EXPECT_EQ(out.branch_pc, 4u);
    EXPECT_EQ(out.target_pc, 10u);
    EXPECT_TRUE(out.on_vcc);
    EXPECT_FALSE(out.on_exec);
    EXPECT_FALSE(out.on_scc0);
    EXPECT_FALSE(out.early_out);
    EXPECT_TRUE(out.uniform_workgroup);
    EXPECT_TRUE(out.has_else);
    EXPECT_EQ(out.sb_pc, 9u);
    EXPECT_EQ(out.merge_pc, 15u);
}
TEST(IrCfgStructuredCursor, EmptySequencesReturnTheRequestedContinuation) {
    const std::vector<Rdna2Inst> ins;
    const std::vector<ForwardIf> fs;
    StructuredIfCursor cursor(ins, fs, {}, true);
    EXPECT_EQ(cursor.next_pc(), UINT32_MAX);
    EXPECT_EQ(cursor.next_before(0), 0u);
    EXPECT_EQ(cursor.next_before(17), 17u);
    EXPECT_EQ(cursor.next_before(UINT32_MAX), UINT32_MAX);
}
TEST(IrCfgStructuredCursor, TreePreorderConsumesEveryNestedAndSequentialRecord) {
    namespace cases = prosper::test::ir::cfg_structured;
    for (uint32_t stage : {0u, 1u, 2u}) {
        const bool fragment = stage == 1u, compute = stage == 2u;
        for (uint32_t shape = 0; shape < 5; ++shape) {
            auto code = cases::words({shape, 3});
            code.push_back(prosper::test::cfg_streams::kEndpgm);
            const auto ins = prosper::test::cfg_streams::walk(code);
            bool refused = false;
            const auto fs = detect_forward_ifs(ins, !compute, code.data(), code.size(), nullptr,
                                               nullptr, &refused, compute);
            ASSERT_FALSE(refused);
            ASSERT_FALSE(fs.empty());
            RegionPolicy policy;
            policy.allow_vcc = !compute;
            policy.compute_wave_branches = compute;
            policy.fragment_wave_breaks = fragment;
            policy.code = code.data();
            policy.dwords = code.size();
            StructuredIfCursor cursor(ins, fs, policy, refused);
            ASSERT_TRUE(cursor.uses_tree()) << "stage=" << stage << " shape=" << shape;
            for (const auto& expected : fs) {
                EXPECT_EQ(cursor.next_pc(), expected.branch_pc);
                EXPECT_EQ(cursor.next_before(expected.branch_pc - 1u), expected.branch_pc - 1u);
                EXPECT_EQ(cursor.next_before(expected.branch_pc + 1u), expected.branch_pc);
                const auto actual = cursor.take();
                EXPECT_EQ(actual.found, expected.found);
                EXPECT_EQ(actual.branch_pc, expected.branch_pc);
                EXPECT_EQ(actual.target_pc, expected.target_pc);
                EXPECT_EQ(actual.has_else, expected.has_else);
                EXPECT_EQ(actual.sb_pc, expected.sb_pc);
                EXPECT_EQ(actual.merge_pc, expected.merge_pc);
                EXPECT_EQ(actual.early_out, expected.early_out);
                EXPECT_EQ(actual.on_scc0, expected.on_scc0);
                EXPECT_EQ(actual.on_vcc, expected.on_vcc);
                EXPECT_EQ(actual.on_exec, expected.on_exec);
                EXPECT_EQ(actual.uniform_workgroup, expected.uniform_workgroup);
            }
            EXPECT_EQ(cursor.next_pc(), UINT32_MAX);
        }
    }
}
TEST(IrCfgStructuredCursor, ARecordMismatchRetainsTheEntireDetectorSequence) {
    using namespace prosper::test::cfg_streams;
    const std::vector<uint32_t> code{sopp(kScc0, 1), kVNop, kEndpgm};
    const auto ins = walk(code);
    auto fs = detect_forward_ifs(ins, true, code.data(), code.size());
    ASSERT_EQ(fs.size(), 1u);
    ++fs[0].target_pc; // a deliberately different authoritative sequence
    StructuredIfCursor cursor(ins, fs, {}, false);
    EXPECT_FALSE(cursor.uses_tree());
    EXPECT_EQ(cursor.route(), StructuredIfCursor::Route::Mismatch);
    EXPECT_EQ(cursor.take().target_pc, fs[0].target_pc);
}
TEST(IrCfgStructuredCursor, KnownDisagreementsKeepDetectorAdmissionAndOrder) {
    using namespace prosper::test::cfg_streams;
    const std::vector<std::vector<uint32_t>> programs{
        // An inner if targets the else arm, which the detector bounds only by the outer merge.
        {sopp(kScc0, 3), sopp(kScc1, 3), kVNop, sopp(kBranch, 2), kVNop, kVNop, kEndpgm},
        // #5415: no end marker leaves the detector's merge bound unbounded.
        {sopp(kScc0, 2), kVNop, sopp(kBranch, 4), kVNop}};
    for (const auto& code : programs) {
        const auto ins = walk(code);
        bool refused = false;
        const auto fs =
            detect_forward_ifs(ins, true, code.data(), code.size(), nullptr, nullptr, &refused);
        ASSERT_FALSE(refused);
        ASSERT_FALSE(fs.empty());
        RegionPolicy policy;
        policy.code = code.data();
        policy.dwords = code.size();
        StructuredIfCursor cursor(ins, fs, policy, refused);
        EXPECT_FALSE(cursor.uses_tree());
        EXPECT_EQ(cursor.route(), StructuredIfCursor::Route::Unstructured);
        for (const auto& f : fs) {
            const auto actual = cursor.take();
            EXPECT_EQ(actual.branch_pc, f.branch_pc);
            EXPECT_EQ(actual.target_pc, f.target_pc);
            EXPECT_EQ(actual.merge_pc, f.merge_pc);
        }
    }
}
TEST(IrCfgStructuredCursor, WholeProgramLoopsAndDetectorRefusalsUseLegacyRecords) {
    using namespace prosper::test::cfg_streams;
    const auto ins = walk({sopp(kBranch, -1), kEndpgm});
    const std::vector<ForwardIf> fs{branch(3, 7)};
    StructuredIfCursor loop(ins, fs, {}, false);
    EXPECT_EQ(loop.route(), StructuredIfCursor::Route::Loops);
    EXPECT_EQ(loop.take().branch_pc, 3u);
    StructuredIfCursor refused(ins, fs, {}, true);
    EXPECT_EQ(refused.route(), StructuredIfCursor::Route::DetectorRefused);
    EXPECT_EQ(refused.take().branch_pc, 3u);
}
}   // namespace

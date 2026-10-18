// Cursor peeks respect recursive region bounds without consuming them; taking a branch
// retains every detector field and the diagnostic cursor position.
#include "gpu/recompiler/ir/cfg/structured_if_cursor.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
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
} // namespace

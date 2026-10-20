// Step 7b part 1: equal static answers are consumed; disagreements keep the existing answer.
// Corrupting either static relation through a real compile must preserve the complete module.
#include "gpu/recompiler/ir/cfg/exec_relation_consumer.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;
namespace {
constexpr uint32_t kCmp = 0x7D880702u;   // v_cmp_gt_u32 vcc, v2, v3
constexpr uint32_t kCmpx = 0x7DA80702u;   // v_cmpx_gt_u32 exec, v2, v3
constexpr uint32_t kSave = 0xBE80246Au;   // s_and_saveexec_b64 s[0:1], vcc
constexpr uint32_t kOther = 0x8AFE7E00u;   // s_andn2_b64 exec, s[0:1], exec
constexpr uint32_t kRestore = 0x88FE007Eu;   // s_or_b64 exec, exec, s[0:1]
constexpr uint32_t kWrite2 = 0x7E280302u, kWrite3 = 0x7E280303u;   // v_mov_b32 v20,v2/v3
constexpr uint32_t kRead = 0x7E020314u;   // v_mov_b32 v1,v20

TEST(IrExecRelationConsumer, GuardKeepsIdAnswerForStrongerAndWeakerStaticRelations) {
    const auto ins = walk({kCmpx, kEndpgm});
    SpirvCompute b;
    RegState rs;
    ExecRelationScope scope(b);
    scope.begin_program(build_guest_cfg(ins, SplitPolicy{}), ins, false);
    rs.exec = 10;
    scope.observe(b, rs, ins[0], false);
    rs.exec = 20;
    scope.observe(b, rs, ins[0], true);
    ASSERT_EQ(scope.within(20, 10), true);
    ASSERT_EQ(scope.within(10, 20), false);
    ExecRelationCapture capture;
    EXPECT_TRUE(guarded_exec_within(20, 10, true));
    EXPECT_FALSE(guarded_exec_within(10, 20, false));
    EXPECT_FALSE(guarded_exec_within(20, 10, false));   // stronger static answer withheld
    EXPECT_TRUE(guarded_exec_within(10, 20, true));   // weaker static answer withheld
    EXPECT_TRUE(guarded_exec_within(30, 10, true));   // missing map keeps id answer
    EXPECT_EQ(capture.within.equal, 2u);
    EXPECT_EQ(capture.within.differs, 2u);
    EXPECT_EQ(capture.within.unmapped, 1u);
}

TEST(IrExecRelationConsumer, ConflictingIdsAndForeignBuildersCannotSupplyAnAnswer) {
    const auto ins = walk({kCmpx, kEndpgm});
    SpirvCompute b, other;
    RegState rs;
    ExecRelationScope scope(b);
    scope.begin_program(build_guest_cfg(ins, SplitPolicy{}), ins, false);
    rs.exec = 10;
    scope.observe(b, rs, ins[0], false);
    scope.observe(b, rs, ins[0], true);   // same id observed holding different values
    EXPECT_FALSE(scope.within(10, 10).has_value());
    rs.exec = 20;
    scope.observe(other, rs, ins[0], false);
    EXPECT_FALSE(scope.within(20, 20).has_value());
}

TEST(IrExecRelationConsumer, ComplementGuardChecksBothEndpoints) {
    const auto ins = walk({kCmp, kSave, kOther, kEndpgm});
    SpirvCompute b;
    RegState rs;
    ExecRelationScope scope(b);
    scope.begin_program(build_guest_cfg(ins, SplitPolicy{}), ins, false);
    rs.exec = 10;
    scope.observe(b, rs, ins[0], false);
    rs.vcc = 11;
    scope.observe(b, rs, ins[0], true);
    rs.exec = 20;
    scope.observe(b, rs, ins[1], true);
    rs.exec = 30;
    scope.observe(b, rs, ins[2], true);
    const ExecPartition expected{{10, 20}};
    ASSERT_EQ(scope.complement(30), std::optional<ExecPartition>{expected});
    ExecRelationCapture capture;
    EXPECT_EQ(guarded_exec_complement(30, expected), expected);
    EXPECT_FALSE(guarded_exec_complement(30, {}));
    EXPECT_EQ(guarded_exec_complement(30, ExecPartition{{11, 20}}), (ExecPartition{{11, 20}}));
    EXPECT_EQ(guarded_exec_complement(30, ExecPartition{{10, 11}}), (ExecPartition{{10, 11}}));
    EXPECT_EQ(capture.complement.equal, 1u);
    EXPECT_EQ(capture.complement.differs, 3u);
}

TEST(IrExecRelationConsumer, NestedRoutesRestoreCallerAndExpiredScopesHaveNoAnswer) {
    SpirvCompute b;
    {
        ExecRelationScope outer(b);
        {
            ExecRelationScope inner(b);
            EXPECT_EQ(ExecRelationScope::active(), &inner);
        }
        EXPECT_EQ(ExecRelationScope::active(), &outer);
    }
    EXPECT_EQ(ExecRelationScope::active(), nullptr);
    EXPECT_TRUE(guarded_exec_within(1, 2, true));
    EXPECT_EQ(guarded_exec_complement(1, ExecPartition{{2, 3}}), (ExecPartition{{2, 3}}));
}

TEST(IrExecRelationConsumer, WrongStaticAnswersCannotChangeRealModules) {
    const std::vector<std::vector<uint32_t>> programs = {
        {kCmp, kSave, kWrite2, kOther, kWrite3, kRestore, kRead},
        {kCmp, kSave, sopp(kExecz, 1), kWrite2, kOther, sopp(kExecz, 1), kWrite3, kRestore, kRead},
        // Relay an absent scalar first: the lanes outside the subsequent complete write are
        // hard, rather than arbitrary launch leftovers. The fresh compare therefore needs the
        // owner proof. Without the equality guard, corrupt mode refuses at that actual read.
        {0x7E28021Eu, kWrite2, 0x7D842914u, kSave, kRead, 0xBEFE0400u},
    };
    size_t equal_within = 0, equal_complement = 0, wrong_within = 0, wrong_complement = 0;
    for (const auto& ops : programs) {
        for (uint32_t wave : {32u, 64u}) {
            const auto p = prosper::test::ir::lane_program(ops, 64, wave);
            ExecRelationCapture baseline(ExecRelationCapture::Mode::Withhold);
            const auto old = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            ASSERT_FALSE(old.empty());
            {
                ExecRelationCapture capture;
                EXPECT_EQ(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config), old);
                equal_within += capture.within.equal;
                equal_complement += capture.complement.equal;
            }
            {
                ExecRelationCapture corrupt(ExecRelationCapture::Mode::Corrupt);
                EXPECT_EQ(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config), old);
                wrong_within += corrupt.within.differs;
                wrong_complement += corrupt.complement.differs;
            }
        }
    }
    EXPECT_GT(equal_within, 0u);
    EXPECT_GT(equal_complement, 0u);
    EXPECT_GT(wrong_within, 0u);
    EXPECT_GT(wrong_complement, 0u);
}

TEST(IrExecRelationConsumer, MaskedExecutionPreservesLanesOutsideTheOwner) {
    const auto inputs = prosper::test::ir::lane_inputs(0x7143u);
    for (uint32_t wave : {32u, 64u}) {
        const auto result = prosper::test::ir::run_vector_op_per_lane(
            {kCmp, kSave, kWrite2, kOther, kWrite3, kRestore, kRead}, inputs, wave);
        ASSERT_TRUE(result.error.empty()) << result.error;
        ASSERT_EQ(result.d.size(), inputs.lanes());
        for (size_t lane = 0; lane < inputs.lanes(); ++lane) {
            const uint32_t expected =
                inputs.active[lane]
                    ? (inputs.s0[lane] > inputs.s1[lane] ? inputs.s0[lane] : inputs.s1[lane])
                    : inputs.old_d[lane];
            EXPECT_EQ(result.d[lane], expected) << "lane " << lane << " wave " << wave;
        }
    }
}
}   // namespace

// Every vector-op definition (ir/ops/vector_alu.hpp) against the SPIR-V the recompiler emits for it,
// lane by lane: each definition's VOP2 instruction runs through vector_lane_harness.hpp with
// different operands in every lane and a partial EXEC, in Wave64 and in Wave32 (two waves), and
// each lane's VDST must equal eval_vector_op() on that lane's operands -- or, for an EXEC-inactive
// lane, its old VDST.
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <gtest/gtest.h>

#include <set>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

TEST(IrVectorAluExecution, EveryDefinitionEvaluatesWhatEachLaneComputes) {
    for (uint32_t wave : {64u, 32u}) {
        for (const VectorOpDef& def : vector_alu_defs()) {
            const char* name = rdna2_instruction_effects(def.fmt, def.opcode).name;
            SCOPED_TRACE(::testing::Message() << (name ? name : "?") << " wave" << wave);
            const LaneInputs in = lane_inputs(0x5218u + def.opcode);
            const LaneRun run = run_vector_op_per_lane({vop2(def.opcode)}, in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane) {
                const Known want = in.active[lane] ? eval_vector_op(def, in.s0[lane], in.s1[lane])
                                                   : Known(in.old_d[lane]);
                ASSERT_TRUE(want) << "lane " << lane << " evaluated to nothing";
                EXPECT_EQ(run.d[lane], *want)
                    << std::hex << "lane " << lane << " s0=0x" << in.s0[lane] << " s1=0x"
                    << in.s1[lane] << " active=" << in.active[lane];
            }
        }
    }
}

// The harness itself must be able to fail: a lane-independent op would pass a harness that fed
// every lane the same operands, and a harness that ignored EXEC would still match active lanes.
TEST(IrVectorAluExecution, HarnessSeesPerLaneOperandsAndExec) {
    const LaneInputs in = lane_inputs(7);
    std::set<uint32_t> distinct(in.s0.begin(), in.s0.end());
    EXPECT_GT(distinct.size(), in.lanes() / 2);
    const LaneRun run = run_vector_op_per_lane({vop2(0x25)}, in);   // v_add_nc_u32
    ASSERT_TRUE(run.error.empty()) << run.error;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (in.active[lane]) continue;
        ++inactive;
        EXPECT_EQ(run.d[lane], in.old_d[lane]) << lane;
        EXPECT_NE(run.d[lane], in.s0[lane] + in.s1[lane]) << lane;
    }
    EXPECT_GE(inactive, 2u);
}

TEST(IrVectorAluExecution, DefinitionsAreVop2WithAVectorResult) {
    for (const VectorOpDef& def : vector_alu_defs()) {
        const auto& effects = rdna2_instruction_effects(def.fmt, def.opcode);
        SCOPED_TRACE(effects.name ? effects.name : "?");
        EXPECT_EQ(def.fmt, Rdna2Format::VOP2);
        EXPECT_TRUE(effects.describes_vector_result());
        EXPECT_EQ(effects.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        EXPECT_FALSE(effects.accumulates);
    }
}

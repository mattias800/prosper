// test_ir_cfg_structured_loop_execution -- loops whose records come from the region tree
// (structured_loop_cursor.cpp, plan step 4b) still execute per lane as the guest does.
//
// Each program is checked twice. The cursor, run over the complete harness stream with the
// detector's records, must take the tree route, so the records emit_body consumed were the
// tree's. The recompiled module must then hold one OpLoopMerge per guest loop and no OpSwitch:
// the structured route emitted it, not the whole-program dispatcher, which would compute the same
// answer without exercising the loop consumer. Results are checked lane by lane against values
// computed here, independently of the encoder, including EXEC-inactive lanes.
#include "gpu/recompiler/ir/cfg/structured_loop_cursor.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;

namespace {

// v_add_nc_u32 v1, s<n>, v1
uint32_t add_scalar_to_v1(uint32_t sgpr) {
    return (0x25u << 25) | (kVdst << 17) | (kVdst << 9) | sgpr;
}

// s<n> = 0; do { v1 += s<n>; s<n> += 1; s_cmp_lt_u32 s<n>, trips } while (SCC)
// v1 gains 0 + 1 + ... + (trips - 1).
std::vector<uint32_t> scc_do_while(uint32_t sgpr, uint32_t trips) {
    return {
        0xbe800380u | (sgpr << 16),                    // s_mov_b32 s<n>, 0
        add_scalar_to_v1(sgpr),                        // header
        0x81008100u | (sgpr << 16) | sgpr,             // s_add_i32 s<n>, s<n>, 1
        0xbf0a0000u | ((0x80u + trips) << 8) | sgpr,   // s_cmp_lt_u32 s<n>, trips
        0xbf85fffcu,                                   // s_cbranch_scc1 header
    };
}

// s[14:15] = exec; v7 = (v2 & 3) + 1;
// do { v1 += 1; v7 -= 1; vcc = v7 != 0; exec &= vcc } while (exec != 0); exec = s[14:15]
// Each lane iterates (S0 & 3) + 1 times, so v1 gains that many.
std::vector<uint32_t> exec_do_while() {
    return {
        0xbe8e047eu,   // s_mov_b64 s[14:15], exec
        0x360e0483u,   // v_and_b32 v7, 3, v2
        0x4a0e0e81u,   // v_add_nc_u32 v7, 1, v7
        0x4a020281u,   // header: v_add_nc_u32 v1, 1, v1
        0x4a0e0ec1u,   // v_add_nc_u32 v7, -1, v7
        0x7d8a0e80u,   // v_cmp_ne_u32 vcc, 0, v7
        0x87fe6a7eu,   // s_and_b64 exec, exec, vcc
        0xbf89fffbu,   // s_cbranch_execnz header
        0xbefe040eu,   // s_mov_b64 exec, s[14:15]
    };
}

struct Case {
    const char* name;
    std::vector<uint32_t> words;
    bool counted;   // the counted route (one loop) rather than the divergent cursor
    size_t loops;
    uint32_t (*gain)(uint32_t s0);
};

std::vector<Case> cases() {
    std::vector<uint32_t> two = scc_do_while(12, 4);
    for (uint32_t w : scc_do_while(13, 3)) two.push_back(w);
    return {
        {"counted do-while", scc_do_while(12, 4), true, 1, [](uint32_t) { return 6u; }},
        {"two scc do-whiles", two, false, 2, [](uint32_t) { return 9u; }},
        {"exec do-while", exec_do_while(), false, 1, [](uint32_t s0) { return (s0 & 3u) + 1u; }},
    };
}

void expect_tree_route(const Case& c, uint32_t wave) {
    const LaneProgram p = lane_program(c.words, 64, wave);
    std::vector<Rdna2Inst> ins;
    rdna2_walk(p.code.data(), p.code.size(), ins);
    std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    const auto policy = ir::cfg::structured_loop_policy(true, false, &safe, p.code.data(),
                                                        p.code.size());
    if (c.counted) {
        const CountedLoop detector = detect_counted_loop(ins);
        ASSERT_TRUE(detector.found);
        EXPECT_EQ(ir::cfg::counted_loop_record(ins, detector, policy).route,
                  ir::cfg::LoopRecordRoute::Tree);
        return;
    }
    EXPECT_FALSE(detect_counted_loop(ins).found && c.loops == 1);
    const std::vector<DivLoop> loops = detect_divergent_loops(ins, safe);
    ASSERT_EQ(loops.size(), c.loops);
    EXPECT_EQ(ir::cfg::StructuredLoopCursor(ins, loops, policy, false).route(),
              ir::cfg::LoopRecordRoute::Tree);
}

TEST(IrCfgStructuredLoopExecution, TreeRecordsDriveTheStructuredLoops) {
    for (const Case& c : cases())
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(testing::Message() << c.name << " wave=" << wave);
            expect_tree_route(c, wave);
            const LaneProgram p = lane_program(c.words, 64, wave);
            const uint64_t address = lane_harness_address();
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                                  {RecompileDiagnosticStage::Compute, address});
            ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
            size_t loop_merges = 0, switches = 0;
            for (size_t at = 5; at < module.size();) {
                const uint32_t count = module[at] >> 16, op = module[at] & 0xffffu;
                ASSERT_GT(count, 0u);
                ASSERT_LE(at + count, module.size());
                loop_merges += op == 246u;
                switches += op == 251u;
                at += count;
            }
            EXPECT_EQ(loop_merges, c.loops);
            EXPECT_EQ(switches, 0u);
        }
}

TEST(IrCfgStructuredLoopExecution, EveryLaneGetsItsGuestResult) {
    for (const Case& c : cases())
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(testing::Message() << c.name << " wave=" << wave);
            const LaneInputs in = lane_inputs(0x5d54250000u + wave);
            const LaneRun run = run_vector_op_per_lane(c.words, in, wave);
            ASSERT_TRUE(run.error.empty()) << run.error;
            ASSERT_EQ(run.d.size(), in.lanes());
            for (uint32_t lane = 0; lane < in.lanes(); ++lane) {
                const uint32_t want =
                    in.active[lane] ? in.old_d[lane] + c.gain(in.s0[lane]) : in.old_d[lane];
                EXPECT_EQ(run.d[lane], want) << "lane " << lane;
            }
        }
}

}   // namespace

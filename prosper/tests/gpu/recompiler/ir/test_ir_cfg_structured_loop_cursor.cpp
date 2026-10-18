// test_ir_cfg_structured_loop_cursor -- the loop records emit_body consumes
// (structured_loop_cursor.cpp): the tree route over nested, sequential, break and counted loops,
// every fallback, and the mismatch guard. A record that differs in any one field from the tree's
// must keep the detector's whole sequence, never emit from the tree.
// Streams are given as `pc: instruction`; a branch names its target pc.
#include "gpu/recompiler/ir/cfg/structured_loop_cursor.hpp"

#include "gpu/recompiler/ir/cfg_streams.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;

namespace {

const std::unordered_set<uint32_t> kNoSafe;

struct Program {
    std::vector<uint32_t> words;
    std::vector<Rdna2Inst> ins;
    RegionPolicy policy;
    explicit Program(std::vector<uint32_t> w, bool fragment = false)
        : words(std::move(w)), ins(walk(words)),
          policy(structured_loop_policy(/*compute*/ false, fragment, &kNoSafe, words.data(),
                                        words.size())) {}
    std::vector<DivLoop> divergent() const { return detect_divergent_loops(ins, kNoSafe); }
};

void expect_same(const DivLoop& a, const DivLoop& b) {
    EXPECT_EQ(a.header_pc, b.header_pc);
    EXPECT_EQ(a.exit_branch_pc, b.exit_branch_pc);
    EXPECT_EQ(a.backedge_pc, b.backedge_pc);
    EXPECT_EQ(a.exit_pc, b.exit_pc);
    EXPECT_EQ(a.break_pcs, b.break_pcs);
    EXPECT_EQ(a.direct_exec_breaks, b.direct_exec_breaks);
    EXPECT_EQ(a.direct_wave_breaks, b.direct_wave_breaks);
    EXPECT_EQ(a.bottom_tested, b.bottom_tested);
    EXPECT_EQ(a.condition, b.condition);
    EXPECT_EQ(a.continue_on_set, b.continue_on_set);
}

// Walks the cursor the way emit_structured does and checks it yields `expected` in order.
void expect_sequence(StructuredLoopCursor& cursor, const std::vector<DivLoop>& expected) {
    for (const DivLoop& want : expected) {
        EXPECT_EQ(cursor.next_pc(), want.header_pc);
        EXPECT_EQ(cursor.next_before(want.header_pc), want.header_pc);   // not below `hi`
        EXPECT_EQ(cursor.next_before(want.header_pc + 1), want.header_pc);
        expect_same(cursor.take(), want);
    }
    EXPECT_EQ(cursor.next_pc(), UINT32_MAX);
    EXPECT_EQ(cursor.next_before(42), 42u);
}

TEST(IrCfgStructuredLoopCursor, TreeRouteYieldsNestedLoopsInHeaderOrder) {
    // 0: s_mov; 1: execz -> 8; 2: v_nop; 3: execz -> 6; 4: v_nop; 5: s_branch -> 3;
    // 6: v_nop; 7: s_branch -> 1; 8: s_endpgm
    const Program p({kSMovS0Zero, sopp(kExecz, 6), kVNop, sopp(kExecz, 2), kVNop, sopp(kBranch, -3),
                     kVNop, sopp(kBranch, -7), kEndpgm});
    const auto loops = p.divergent();
    ASSERT_EQ(loops.size(), 2u);
    StructuredLoopCursor cursor(p.ins, loops, p.policy, false);
    ASSERT_EQ(cursor.route(), LoopRecordRoute::Tree);
    // The tree route reads its own records, not the detector's vector.
    EXPECT_NE(&cursor.take(), &loops[0]);
    StructuredLoopCursor again(p.ins, loops, p.policy, false);
    expect_sequence(again, loops);
}

TEST(IrCfgStructuredLoopCursor, TreeRouteYieldsSequentialLoopsAndBreaks) {
    // Loop 1 has a break. 0: s_mov; 1: execz -> 6; 2: v_nop; 3: execz -> 6 (break); 4: v_nop;
    // 5: execnz -> 1; 6: execz -> 9; 7: v_nop; 8: s_branch -> 6; 9: s_endpgm
    const Program p({kSMovS0Zero, sopp(kExecz, 4), kVNop, sopp(kExecz, 2), kVNop, sopp(kExecnz, -5),
                     sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm});
    const auto loops = p.divergent();
    ASSERT_EQ(loops.size(), 2u);
    ASSERT_EQ(loops[0].break_pcs.size(), 1u);
    StructuredLoopCursor cursor(p.ins, loops, p.policy, false);
    ASSERT_EQ(cursor.route(), LoopRecordRoute::Tree);
    expect_sequence(cursor, loops);
}

TEST(IrCfgStructuredLoopCursor, NoLoopsReturnsTheRequestedContinuation) {
    const Program p({kVNop, kEndpgm});
    const std::vector<DivLoop> none;
    StructuredLoopCursor cursor(p.ins, none, p.policy, true);
    EXPECT_EQ(cursor.route(), LoopRecordRoute::NoLoops);
    EXPECT_EQ(cursor.next_pc(), UINT32_MAX);
    EXPECT_EQ(cursor.next_before(0), 0u);
    EXPECT_EQ(cursor.next_before(UINT32_MAX), UINT32_MAX);
}

TEST(IrCfgStructuredLoopCursor, DetectorRefusalKeepsTheDetectorRecords) {
    const Program p({kSMovS0Zero, sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm});
    const auto loops = p.divergent();
    ASSERT_EQ(loops.size(), 1u);
    StructuredLoopCursor cursor(p.ins, loops, p.policy, /*detector_refused*/ true);
    EXPECT_EQ(cursor.route(), LoopRecordRoute::DetectorRefused);
    EXPECT_EQ(&cursor.take(), &loops[0]);
}

TEST(IrCfgStructuredLoopCursor, UnstructuredTreeKeepsTheDetectorRecords) {
    // A forward execnz is never an if, so the tree is Unstructured whatever the records say.
    const Program p({sopp(kExecnz, 1), kVNop, kEndpgm});
    DivLoop loop;
    loop.header_pc = 1;
    loop.backedge_pc = 1;
    const std::vector<DivLoop> loops{loop};
    StructuredLoopCursor cursor(p.ins, loops, p.policy, false);
    EXPECT_EQ(cursor.route(), LoopRecordRoute::Unstructured);
    EXPECT_EQ(&cursor.take(), &loops[0]);
}

TEST(IrCfgStructuredLoopCursor, ACountedTreeIsNotTheDivergentRoute) {
    // The tree takes the counted route; records claiming a divergent loop there are a mismatch.
    const Program p({kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm});
    DivLoop loop;
    loop.header_pc = 0;
    loop.exit_branch_pc = 1;
    loop.backedge_pc = 2;
    loop.exit_pc = 3;
    loop.condition = DivLoop::Condition::Scc;
    const std::vector<DivLoop> loops{loop};
    StructuredLoopCursor cursor(p.ins, loops, p.policy, false);
    EXPECT_EQ(cursor.route(), LoopRecordRoute::Mismatch);
    EXPECT_EQ(&cursor.take(), &loops[0]);
}

// The mismatch guard: each field changed alone, plus a missing, an extra and a reordered loop.
TEST(IrCfgStructuredLoopCursor, AnyDifferentFieldFallsBackToTheWholeDetectorSequence) {
    const Program p({kSMovS0Zero, sopp(kExecz, 4), kVNop, sopp(kExecz, 2), kVNop, sopp(kExecnz, -5),
                     sopp(kExecz, 2), kVNop, sopp(kBranch, -3), kEndpgm});
    const auto base = p.divergent();
    ASSERT_EQ(base.size(), 2u);
    ASSERT_EQ(StructuredLoopCursor(p.ins, base, p.policy, false).route(), LoopRecordRoute::Tree);
    const std::vector<std::function<void(std::vector<DivLoop>&)>> mutations = {
        [](auto& v) { ++v[1].header_pc; },
        [](auto& v) { ++v[1].exit_branch_pc; },
        [](auto& v) { ++v[1].backedge_pc; },
        [](auto& v) { ++v[1].exit_pc; },
        [](auto& v) { v[0].break_pcs.clear(); },
        [](auto& v) { v[1].break_pcs.push_back(7); },
        [](auto& v) { v[1].direct_exec_breaks = !v[1].direct_exec_breaks; },
        [](auto& v) { v[1].direct_wave_breaks = !v[1].direct_wave_breaks; },
        [](auto& v) { v[1].bottom_tested = !v[1].bottom_tested; },
        [](auto& v) { v[1].condition = DivLoop::Condition::Vcc; },
        [](auto& v) { v[1].continue_on_set = !v[1].continue_on_set; },
        [](auto& v) { v.pop_back(); },
        [](auto& v) { v.push_back(v[1]); },
        [](auto& v) { std::swap(v[0], v[1]); },
    };
    for (size_t i = 0; i < mutations.size(); ++i) {
        SCOPED_TRACE(i);
        std::vector<DivLoop> loops = base;
        mutations[i](loops);
        StructuredLoopCursor cursor(p.ins, loops, p.policy, false);
        EXPECT_EQ(cursor.route(), LoopRecordRoute::Mismatch);
        EXPECT_FALSE(cursor.uses_tree());
        // The detector's own (here: wrong) records come back, every one of them.
        for (const DivLoop& want : loops) EXPECT_EQ(&cursor.take(), &want);
    }
}

// --- counted_loop_record ---

TEST(IrCfgStructuredLoopCursor, CountedTreeRouteForTopTestedAndDoWhileLoops) {
    // 0: s_mov; 1: scc0 -> 3; 2: s_branch -> 0; 3: s_endpgm   and   0: v_nop; 1: scc0 -> 0; ...
    for (const auto& words :
         {std::vector<uint32_t>{kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm},
          std::vector<uint32_t>{kVNop, sopp(kScc0, -2), kEndpgm}}) {
        const Program p(words);
        const CountedLoop detector = detect_counted_loop(p.ins);
        ASSERT_TRUE(detector.found);
        const CountedLoopRecord record = counted_loop_record(p.ins, detector, p.policy);
        EXPECT_EQ(record.route, LoopRecordRoute::Tree);
        EXPECT_TRUE(record.loop.found);
        EXPECT_EQ(record.loop.header_pc, detector.header_pc);
        EXPECT_EQ(record.loop.exit_branch_pc, detector.exit_branch_pc);
        EXPECT_EQ(record.loop.backedge_pc, detector.backedge_pc);
        EXPECT_EQ(record.loop.exit_pc, detector.exit_pc);
        EXPECT_EQ(record.loop.exit_on_scc0, detector.exit_on_scc0);
    }
}

TEST(IrCfgStructuredLoopCursor, CountedFallbacks) {
    // No counted loop: the tree is not consulted.
    const Program straight({kVNop, kEndpgm});
    const CountedLoop none = detect_counted_loop(straight.ins);
    EXPECT_EQ(counted_loop_record(straight.ins, none, straight.policy).route,
              LoopRecordRoute::NoLoops);
    // An unreachable counted loop: the detector finds it, the tree refuses the s_branch over it.
    // 0: s_branch -> 3; 1: v_nop; 2: scc1 -> 1; 3: s_endpgm
    const Program unreachable({sopp(kBranch, 2), kVNop, sopp(kScc1, -2), kEndpgm});
    const CountedLoop hidden = detect_counted_loop(unreachable.ins);
    ASSERT_TRUE(hidden.found);
    const CountedLoopRecord u = counted_loop_record(unreachable.ins, hidden, unreachable.policy);
    EXPECT_EQ(u.route, LoopRecordRoute::Unstructured);
    EXPECT_EQ(u.loop.header_pc, hidden.header_pc);
    // A loop before the counted one: the tree takes the general route (two divergent loops).
    // 0: v_nop; 1: s_mov_b64 exec, 0; 2: execnz -> 0; 3: s_mov; 4: scc0 -> 6; 5: s_branch -> 3
    const Program general({kVNop, kSMovExecZero, sopp(kExecnz, -3), kSMovS0Zero, sopp(kScc0, 1),
                           sopp(kBranch, -3), kEndpgm});
    const CountedLoop counted = detect_counted_loop(general.ins);
    ASSERT_TRUE(counted.found);
    EXPECT_EQ(counted_loop_record(general.ins, counted, general.policy).route,
              LoopRecordRoute::Mismatch);
}

TEST(IrCfgStructuredLoopCursor, AnyDifferentCountedFieldKeepsTheDetectorRecord) {
    const Program p({kSMovS0Zero, sopp(kScc0, 1), sopp(kBranch, -3), kEndpgm});
    const CountedLoop base = detect_counted_loop(p.ins);
    ASSERT_EQ(counted_loop_record(p.ins, base, p.policy).route, LoopRecordRoute::Tree);
    const std::vector<std::function<void(CountedLoop&)>> mutations = {
        [](auto& L) { ++L.header_pc; },
        [](auto& L) { ++L.exit_branch_pc; },
        [](auto& L) { ++L.backedge_pc; },
        [](auto& L) { ++L.exit_pc; },
        [](auto& L) { L.exit_on_scc0 = !L.exit_on_scc0; },
    };
    for (size_t i = 0; i < mutations.size(); ++i) {
        SCOPED_TRACE(i);
        CountedLoop wrong = base;
        mutations[i](wrong);
        const CountedLoopRecord record = counted_loop_record(p.ins, wrong, p.policy);
        EXPECT_EQ(record.route, LoopRecordRoute::Mismatch);
        EXPECT_EQ(record.loop.header_pc, wrong.header_pc);
        EXPECT_EQ(record.loop.exit_branch_pc, wrong.exit_branch_pc);
        EXPECT_EQ(record.loop.backedge_pc, wrong.backedge_pc);
        EXPECT_EQ(record.loop.exit_pc, wrong.exit_pc);
        EXPECT_EQ(record.loop.exit_on_scc0, wrong.exit_on_scc0);
    }
}

}   // namespace

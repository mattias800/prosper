// CFG counted-loop detection; private to the recompiler.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>
namespace prosper::gpu {

// A recognized COUNTED loop (the game's MSAA-resolve / accumulation shape). Two accepted shapes:
//   (a) TOP-tested: a single backward UNCONDITIONAL s_branch (the back-edge) to a header, with
//       exactly one forward s_cbranch_scc0/scc1 exit inside the body — `for (...) {...}`.
//   (b) BOTTOM-tested (do-while): the back-edge is ITSELF a backward s_cbranch_scc0/scc1; the loop
//       continues while its SCC condition holds and exits by falling through — `do {...} while(...)`.
//       This is what the compiler emits for a fixed-trip texture-accumulation loop whose whole body
//       (including the s_cmp that sets the exit SCC) runs before the test. Alex Kidd's (PPSA02664)
//       light/blur-accumulation pixel shaders use it, and rejecting it dropped their draws (#320).
// Both are lowered by ONE emitter: shape (b) is modeled as exit_branch_pc == backedge_pc, so the
// emitter emits the entire body as the "condition region" (runs every iteration incl. the exiting
// one — exactly do-while), tests SCC at the bottom, and its "body region" (exit_branch+1..backedge)
// is empty. Anything more complex (nested loops, VCC/EXECNZ exits, multiple back-edges, mid-loop
// s_branch) is rejected — the recompiler then falls back to the straight-line path (as before).
struct CountedLoop {
    bool found = false;
    uint32_t header_pc = 0;   // loop header (target of the back-edge; condition eval starts here)
    uint32_t exit_branch_pc =
        0;   // the s_cbranch_scc0/scc1 exit test (== backedge_pc for a do-while)
    uint32_t backedge_pc =
        0;   // the backward branch (unconditional s_branch, or the scc back-edge)
    uint32_t exit_pc = 0;   // first pc after the loop (== backedge_pc + its length)
    bool exit_on_scc0 =
        true;   // exit-test polarity: true ⇔ the loop continues while SCC==1 (exits on SCC==0)
};
bool counted_loop_backedge_candidate(const Rdna2Inst& in,
                                     const std::unordered_set<uint32_t>& waterfalls);
CountedLoop counted_loop_from_backedge(const std::vector<Rdna2Inst>& ins,
                                       const Rdna2Inst& back_edge);
bool counted_route_exec_guard(const Rdna2Inst& in, const CountedLoop& L);
CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins);
}   // namespace prosper::gpu

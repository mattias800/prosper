// Counted-loop detection moved from rdna2_cfg_support.hpp.
#include "gpu/recompiler/rdna2_cfg_counted_loop.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// ---- detect_counted_loop's admission rules, shared with the IR region tree -----------------
// ir/cfg/region_loops.cpp finds back-edges in the CFG and asks these two the detector's
// questions, so each rule lives here once.
//
// A candidate back-edge: a backward s_branch, or a backward s_cbranch_scc0/scc1 that is not a
// waterfall's once-through branch (`waterfalls` is waterfall_branches(ins)).
// A backward s_cbranch_scc0/scc1 whose SCC is a 64-bit wave-mask reduction is a WATERFALL
// (once-through per invocation), NOT a loop back-edge — its cross-lane SCC has no representable
// per-lane value, and every emit path already consumes it via safe_branches/the linearizer.
// Waterfalls are excluded from the back-edge count entirely: SELECTING one would feed the loop
// emitter a poisoned SCC exit and mis-lower the divergent body (T19), and merely COUNTING one
// would reject a counted loop that legitimately coexists with a waterfall elsewhere in the same
// shader — a previously-recompilable combination (kernel 44d locks it).
bool counted_loop_backedge_candidate(const Rdna2Inst& in,
                                     const std::unordered_set<uint32_t>& waterfalls) {
    if (in.fmt != Rdna2Format::SOPP || in.simm16 >= 0) return false;
    return in.opcode == 0x02 ||
           ((in.opcode == 0x04 || in.opcode == 0x05) && !waterfalls.count(in.pc));
}
// The counted loop that `back_edge` closes, when it is the program's ONLY candidate back-edge.
// It must be an element of `ins`. `found` is false when the loop does not fit either shape.
CountedLoop counted_loop_from_backedge(const std::vector<Rdna2Inst>& ins,
                                       const Rdna2Inst& back_edge) {
    CountedLoop L;
    const Rdna2Inst* back = &back_edge;
    const bool do_while = (back->opcode != 0x02);   // conditional back-edge => bottom-tested
    const uint32_t header = branch_target(*back);
    bool header_ok = false;
    for (const auto& in : ins)
        if (in.pc == header && in.pc < back->pc) {
            header_ok = true;
            break;
        }
    if (!header_ok) return L;
    const uint32_t exit_pc = back->pc + back->len_dwords;
    const Rdna2Inst* exitbr = nullptr;
    int nexit = 0;
    for (const auto& in : ins) {   // scan the loop body [header, back-edge]
        if (in.pc < header || in.pc > back->pc || in.fmt != Rdna2Format::SOPP) continue;
        if (&in == back) continue;   // the back-edge is classified above, not here
        switch (in.opcode) {
            case 0x04:
            case 0x05:   // s_cbranch_scc0 / scc1 — must be the single exit
                if (branch_target(in) != exit_pc) return L;
                exitbr = &in;
                nexit++;
                break;
            case 0x02:   // any other s_branch -> reject
                return L;
            case 0x06:
            case 0x07:
            case 0x09:   // vcc / execnz branches -> reject
                return L;
            case 0x08:
            default: break;   // execz (forward, predication-handled) / hints ok
        }
    }
    if (do_while) {
        // Bottom-tested: the conditional back-edge IS the exit test; no separate forward exit is
        // allowed (a do-while with an inner uniform-if stays unsupported and rejects, conservatively).
        // The back-edge is TAKEN (loops) while its SCC condition holds. exit_on_scc0 drives the
        // emitter's loop_cond = (exit_on_scc0 ? SCC : !SCC), i.e. "continue when this is true": an
        // s_cbranch_scc1 back-edge continues while SCC==1 (exit_on_scc0=true); scc0 continues while
        // SCC==0 (exit_on_scc0=false).
        if (nexit != 0) return L;
        L.found = true;
        L.header_pc = header;
        L.exit_branch_pc = back->pc;
        L.backedge_pc = back->pc;
        L.exit_pc = exit_pc;
        L.exit_on_scc0 = (back->opcode == 0x05);
        return L;
    }
    if (nexit != 1) return L;
    L.found = true;
    L.header_pc = header;
    L.exit_branch_pc = exitbr->pc;
    L.backedge_pc = back->pc;
    L.exit_pc = exit_pc;
    L.exit_on_scc0 = (exitbr->opcode == 0x04);
    return L;
}

// The counted route's EXEC guards: an s_cbranch_execz inside the counted loop [header, back-edge],
// which counted_loop_from_backedge admits as predication, or one before the loop whose target lies
// at or past the header (a wave-empty guard spanning the loop, kept out of the prelude's forward-if
// scan). Neither is structured as an if: the counted lowering linearizes a guard it proves
// (mark_counted_loop_exec_guards), and emit_alu proves full EXEC for the rest or refuses.
bool counted_route_exec_guard(const Rdna2Inst& in, const CountedLoop& L) {
    if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x08) return false;
    if (in.pc >= L.header_pc) return in.pc <= L.backedge_pc;
    return branch_target(in) >= L.header_pc;
}

CountedLoop detect_counted_loop(const std::vector<Rdna2Inst>& ins) {
    // The single backward branch that closes the loop. A conditional scc0/scc1 back-edge is a
    // bottom-tested do-while; an unconditional s_branch is a top-tested loop.
    const std::unordered_set<uint32_t> wf = waterfall_branches(ins);
    const Rdna2Inst* back = nullptr;
    int nback = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (counted_loop_backedge_candidate(in, wf)) {
            back = &in;
            nback++;
        }
    }
    if (nback != 1) return {};   // 0 -> no loop; >1 -> nested/multiple (unhandled)
    return counted_loop_from_backedge(ins, *back);
}

}   // namespace prosper::gpu

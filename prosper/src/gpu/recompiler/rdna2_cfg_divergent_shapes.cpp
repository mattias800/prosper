#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>
#include <unordered_set>
#include <cstdint>
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include <algorithm>
#include <cstddef>

namespace prosper::gpu {
// Pass 1's candidate back-edge: each backward s_branch / s_cbranch_execnz / s_cbranch_scc0/scc1
// that is not linearized. A backward SCC branch closes a bottom-tested do-while unless it is a
// waterfall's once-through branch, which the linearizer consumes (`waterfalls` is
// waterfall_branches(ins)).
bool div_loop_backedge_candidate(const Rdna2Inst& in, const std::unordered_set<uint32_t>& safe,
                                 const std::unordered_set<uint32_t>& waterfalls) {
    if (in.fmt != Rdna2Format::SOPP) return false;
    const bool scc_backedge = in.opcode == 0x04 || in.opcode == 0x05;
    if ((in.opcode != 0x02 && in.opcode != 0x09 && !scc_backedge) || in.simm16 >= 0) return false;
    if (safe.count(in.pc)) return false;
    if (scc_backedge && waterfalls.count(in.pc)) return false;
    return true;
}
// The loop a candidate back-edge `in` closes: its header, back-edge and exit pcs.
DivLoopRefusal div_loop_from_backedge(const std::vector<Rdna2Inst>& ins, const Rdna2Inst& in,
                                      uint32_t end_pc, DivLoop* out) {
    DivLoop L;
    L.header_pc = branch_target(in);
    L.backedge_pc = in.pc;
    L.exit_pc = in.pc + in.len_dwords;
    if (L.header_pc >= L.backedge_pc) return DivLoopRefusal{"degenerate-loop-bounds", L.header_pc};
    if (L.exit_pc > end_pc) return DivLoopRefusal{"exit-past-end-of-program", L.header_pc};
    bool hdr_ok = false;
    for (const auto& h : ins) {
        if (h.pc == L.header_pc) {
            hdr_ok = true;
            break;
        }
        if (h.pc > L.header_pc) break;
    }
    if (!hdr_ok) return DivLoopRefusal{"header-pc-is-not-an-instruction", L.header_pc};
    *out = L;
    return {};
}
// Re-sorts the candidates (and their back-edge opcodes) into header-pc order.
void sort_div_loops_by_header(std::vector<DivLoop>& out, std::vector<uint32_t>& backedge_op) {
    // Loops may be strictly DISJOINT (sequential) or PROPERLY NESTED (#590 — DOLL's post-process
    // kernel iterates an inner table loop inside an outer row loop). Pass 1 collected loops in
    // BACK-EDGE pc order (an inner back-edge precedes its outer's); re-sort by header so the list is
    // in emission order — emit_structured consumes loops by header pc, and emitting an outer body
    // recursively reaches the inner loop first because its header lies inside that range. Then
    // validate every pair as either disjoint-in-order or properly nested; partial overlap (an
    // unstructured region) still rejects loudly.
    {
        std::vector<size_t> order(out.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t c) { return out[a].header_pc < out[c].header_pc; });
        std::vector<DivLoop> sorted_loops;
        std::vector<uint32_t> sorted_op;
        sorted_loops.reserve(out.size());
        sorted_op.reserve(out.size());
        for (size_t i : order) {
            sorted_loops.push_back(out[i]);
            sorted_op.push_back(backedge_op[i]);
        }
        out.swap(sorted_loops);
        backedge_op.swap(sorted_op);
    }
}
// Every pair is either disjoint-in-order or properly nested.
DivLoopRefusal div_loop_pairs_refusal(const std::vector<DivLoop>& out) {
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++) {
            const DivLoop& A = out[i];
            const DivLoop& B = out[j];   // A.header_pc <= B.header_pc
            if (A.header_pc == B.header_pc)
                return DivLoopRefusal{"shared-header", A.header_pc};   // shared header: not modeled
            const bool disjoint = B.header_pc >= A.exit_pc;
            const bool nested = B.exit_pc <= A.backedge_pc;   // B entirely inside A's body
            if (!disjoint && !nested)
                return DivLoopRefusal{"partial-overlap",
                                      B.header_pc};   // partial overlap: unstructured
        }
    return {};
}
DivLoopRefusal div_loop_children_refusal(const std::vector<DivLoop>& out) {
    // A nested child must lie entirely within its parent's BODY: after the parent's canonical exit
    // test (the condition region [header, exit_branch) stays branch-free) and before its back-edge.
    for (size_t i = 0; i < out.size(); i++)
        for (size_t j = i + 1; j < out.size(); j++)
            if (out[j].exit_pc <= out[i].backedge_pc &&   // nested per the classification above
                !out[i].bottom_tested && out[j].header_pc <= out[i].exit_branch_pc)
                return DivLoopRefusal{"child-header-inside-parent-condition-region",
                                      out[j].header_pc};
    return {};
}
DivLoopRefusal div_loop_entry_edge_refusal(const std::vector<Rdna2Inst>& ins,
                                           const std::vector<DivLoop>& out) {
    // Pass 3: no branch from OUTSIDE a loop may target its interior (an unstructured entry edge).
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        if (in.opcode < 0x02 || in.opcode > 0x09 || in.opcode == 0x03) continue;
        uint32_t tgt = branch_target(in);
        for (const auto& L : out) {
            const bool inside_br = in.pc >= L.header_pc && in.pc <= L.backedge_pc;
            const bool inside_tgt = tgt > L.header_pc && tgt < L.exit_pc;
            if (!inside_br && inside_tgt)
                return DivLoopRefusal{"entry-edge-from-outside-loop", in.pc};
        }
    }
    return {};
}
// Everything after pass 1, over candidates already in header-pc order.
DivLoopRefusal validate_div_loops(const std::vector<Rdna2Inst>& ins,
                                  const std::unordered_set<uint32_t>& safe,
                                  bool exact_fragment_wave_breaks, std::vector<DivLoop>& out,
                                  const std::vector<uint32_t>& backedge_op) {
    if (const DivLoopRefusal r = div_loop_pairs_refusal(out); r.why) return r;
    for (size_t li = 0; li < out.size(); li++)
        if (const DivLoopRefusal r =
                admit_div_loop(ins, safe, exact_fragment_wave_breaks, out, li, backedge_op[li]);
            r.why)
            return r;
    if (const DivLoopRefusal r = div_loop_children_refusal(out); r.why) return r;
    return div_loop_entry_edge_refusal(ins, out);
}
}   // namespace prosper::gpu

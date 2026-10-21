// Divergent-loop shape inventory and admission helpers.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// MULTI-LOOP EXEC/VCC/SCC EXITS (#273/#615/#1554 — post-process and light-accumulation PSes,
// title-composite content producers, and compute histogram loops). The divergent compiled shape:
//     header: <exec recompute: v_cmpx_.. counter,bound  |  v_cmp..;s_andn2_b64 exec,exec,vcc>
//             s_cbranch_execz EXIT                      ; leave when no lane remains
//     body:   ... (nested forward-execz if regions, saveexec/restore, scalar counter++) ...
//             s_branch header                           ; backward unconditional back-edge
//     EXIT:   s_mov_b64 exec, <saved>
// Fragment semantics are exact: a forced 64-lane native subgroup vote keeps the complete guest wave
// in the loop until every EXEC/VCC bit clears. Cleared lanes remain synchronization participants while
// EXEC predicates their vector writes, so scalar state advances exactly as on RDNA2. Vertex and the
// narrow structured-compute path retain the older per-invocation lowering and its uniformity guards.
//
// A second flavor (DOLL's scalar-indexed unroll): the back-edge is a backward s_cbranch_EXECNZ and
// the header IS the execz exit (empty condition region); the body may carry an extra forward
// vccz/execz BREAK to the exit. A break lowers as a plain forward IF over the remainder of the body
// (skip-to-backedge): the broken lane's EXEC bit is already clear (the compiled break condition
// mirrors the exec recompute), so the next header check exits it. Fragment wave64 additionally
// supports a VCCZ break with an unconditional back-edge by carrying the uniform continue vote to the
// loop latch and branching the complete wave directly to the loop merge.
struct DivLoop {
    enum class Condition : uint8_t { Exec, Vcc, Scc };
    uint32_t header_pc = 0;   // back-edge target; condition region = [header_pc, exit_branch_pc)
    uint32_t exit_branch_pc =
        0;   // canonical forward execz/vccz/scc branch whose target is exit_pc
    uint32_t backedge_pc = 0;   // backward s_branch, s_cbranch_execnz or s_cbranch_scc0/scc1
    uint32_t exit_pc = 0;   // backedge_pc + its length (first pc after the loop)
    std::vector<uint32_t> break_pcs;   // extra forward vccz/execz -> exit_pc (lowered as body ifs)
    bool direct_exec_breaks = false;   // unconditional back-edge: an interior execz exits directly
    bool direct_wave_breaks = false;   // fragment wave64: an interior vccz exits the complete wave
    bool bottom_tested = false;   // the execnz/scc back-edge is the condition (do-while)
    Condition condition = Condition::Exec;
    // EXECZ/VCCZ and SCC0 all continue while their represented predicate is set. SCC1 is the one
    // admitted opposite-polarity canonical exit: it leaves the loop while SCC is set. A bottom-tested
    // SCC loop reads its back-edge instead: scc1 continues while SCC is set, scc0 while it is clear.
    bool continue_on_set = true;
};

// ---- detect_divergent_loops' admission rules, shared with the IR region tree ---------------
// ir/cfg/region_loops.cpp collects back-edges from the CFG instead of a pc scan and validates
// them with these, so each rule lives here once. A refusal names the detector's reason and pc;
// the detector reports it through its PROSPER_DBG reporter, the region tree as its Unstructured
// reason. None of these log.
struct DivLoopRefusal {
    const char* why = nullptr;   // nullptr: admitted
    uint32_t pc = 0;
};

bool div_loop_backedge_candidate(const Rdna2Inst& in, const std::unordered_set<uint32_t>& safe,
                                 const std::unordered_set<uint32_t>& waterfalls);
DivLoopRefusal div_loop_from_backedge(const std::vector<Rdna2Inst>& ins, const Rdna2Inst& in,
                                      uint32_t end_pc, DivLoop* out);
void sort_div_loops_by_header(std::vector<DivLoop>& out, std::vector<uint32_t>& backedge_op);
DivLoopRefusal div_loop_pairs_refusal(const std::vector<DivLoop>& out);
DivLoopRefusal div_loop_children_refusal(const std::vector<DivLoop>& out);
DivLoopRefusal div_loop_entry_edge_refusal(const std::vector<Rdna2Inst>& ins,
                                           const std::vector<DivLoop>& out);
DivLoopRefusal validate_div_loops(const std::vector<Rdna2Inst>& ins,
                                  const std::unordered_set<uint32_t>& safe,
                                  bool exact_fragment_wave_breaks, std::vector<DivLoop>& out,
                                  const std::vector<uint32_t>& backedge_op);
}   // namespace prosper::gpu

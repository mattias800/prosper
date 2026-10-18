// guest_cfg.hpp -- the guest program's control-flow graph, built once from the decoded stream.
//
// IR_CONTROL_FLOW_PLAN.md steps 1 and 5. Plain data: building it allocates no SPIR-V id and calls
// no SpirvCompute, so it cannot change an emitted byte. The region tree (region_tree.hpp) reads the
// control-flow split; the CFG dispatcher reads the service split (dispatch_blocks.hpp). The
// terminator contract is in AGENTS.md next to this file.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu::ir::cfg {

inline constexpr uint32_t kNone = UINT32_MAX;

// How a block ends. Derived from Rdna2InstructionEffects::control; there is no opcode list here.
enum class Terminator : uint8_t {
    Fallthrough,   // the block ends because the next instruction is a leader
    Branch,   // s_branch
    CondScc0,
    CondScc1,
    CondVccz,
    CondVccnz,
    CondExecz,
    CondExecnz,
    End,   // s_endpgm and its variants, s_code_end
    Trap,   // s_trap: the wave ends (no guest trap handler is emulated)
    // s_setpc/s_rfe (no successor), s_swappc/s_call (succ[0] is the return point) and
    // s_cbranch_cdbg* (succ[0] is the not-taken side): the other successor is unknown.
    IndirectUnknown,
};

bool terminator_is_conditional(Terminator term);

// One guest basic block: the instruction range [first, last] of the decoded program.
struct GuestBlock {
    uint32_t first = 0, last = 0;   // indices into the instruction vector, inclusive
    uint32_t pc = 0;   // pc of `first`
    Terminator term = Terminator::Fallthrough;
    // succ[0]: the fallthrough successor (conditional branches, Fallthrough blocks, and the return
    // point of a call).
    // succ[1]: the taken successor (s_branch and conditional branches). kNone when absent, and
    // when the target is not the pc of a decoded instruction (past s_endpgm, mid-instruction or
    // outside the stream): `taken_pc` then still names it.
    std::array<uint32_t, 2> succ = {kNone, kNone};
    uint32_t taken_pc = kNone;   // the branch target pc, for Branch and the conditional forms
    // The pc of the synchronized cross-lane service ending this block, under
    // SplitPolicy::split_at_services; kNone otherwise. Such a block is that one instruction.
    uint32_t service_event = kNone;
};

struct GuestCfg {
    std::vector<GuestBlock> blocks;   // in pc order; block 0 is the entry
    std::vector<std::vector<uint32_t>> preds;   // predecessor block ids, ascending
    // Reverse post-order of the blocks reachable from the entry (DFS visits succ[0] first).
    std::vector<uint32_t> rpo;
    // Immediate dominators; idom[0] == 0, kNone for an unreachable block.
    std::vector<uint32_t> idom;
    // Retreating DFS edges (source, target): the target is a DFS ancestor of the source. In a
    // reducible graph these are exactly the loop back edges.
    std::vector<std::pair<uint32_t, uint32_t>> back_edges;
    std::vector<bool> reachable;

    // The block whose first instruction has pc `pc`, or kNone.
    uint32_t block_at_pc(uint32_t pc) const;
};

// How blocks are split. The structured emitter splits only at control flow: block leaders are the
// entry, every in-stream branch target and the instruction after every terminator.
//
// split_at_services is the CFG dispatcher's split (emit_cfg_state_machine, plan step 5), which
// differs in more than the services; AGENTS.md has the table. The graph covers the instructions up
// to the first s_endpgm (`is_end`) and is empty without one. Terminators are s_endpgm, s_trap and
// the SOPP branches not in `linearized`; s_setpc, s_swappc, s_call, s_cbranch_cdbg* and the other
// s_endpgm forms are ordinary instructions. Leaders are the entry, s_endpgm, every service in
// `services` and every s_trap together with the instruction after it, and for each such branch the
// instruction after it and its target when that is at or before s_endpgm.
struct SplitPolicy {
    bool split_at_services = false;
    // split_at_services only. The pcs the dispatcher ends a block around: its own service
    // predicate, evaluated by emit_cfg_state_machine (there is no service list here). Null: none.
    const std::unordered_set<uint32_t>* services = nullptr;
    // split_at_services only. SOPP pcs the dispatcher linearizes rather than branches on (the
    // graphics stages' `safe` set). Null: none.
    const std::unordered_set<uint32_t>* linearized = nullptr;
};

GuestCfg build_guest_cfg(std::span<const Rdna2Inst> ins, const SplitPolicy& policy);

}   // namespace prosper::gpu::ir::cfg

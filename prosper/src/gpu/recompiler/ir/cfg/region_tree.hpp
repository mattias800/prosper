// region_tree.hpp -- the structured regions of a GuestCfg (IR_CONTROL_FLOW_PLAN.md steps 1, 4).
//
// The tree holds Seq, Block, IfThen, IfElse, Loop and CountedLoop, plus Unstructured for a
// sub-CFG it cannot express. The admission rules are the detectors' own (detect_forward_ifs,
// detect_divergent_loops, detect_counted_loop), called through the shared predicates in
// rdna2_cfg_support.hpp; the loops themselves are found from the CFG's back edges. The contract
// (terminator table, RegionPolicy rules, loop shapes) is in AGENTS.md next to this file.
//
// Shadow only: nothing in emission reads the tree. region_shadow.hpp compares it with the
// detector.
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The condition a structured branch tests: the flag, and whether the branch is taken when it is
// zero (scc0, vccz, execz). A branch is taken to SKIP the then-arm, so the then-arm runs when the
// branch is not taken.
struct BranchCondition {
    enum class Flag : uint8_t { None, Scc, Vcc, Exec } flag = Flag::None;
    bool taken_on_zero = false;
    // Compute vccz/vccnz: the architectural whole-wave flag, decided per guest wave.
    bool wave_vote = false;
    // Proved to decide identically for every wave of the workgroup (#1554 VCC, #3713 SCC).
    bool uniform_workgroup = false;
};

// What a loop's admission rules decided about it (detect_divergent_loops' DivLoop fields, or the
// counted shape). The exit flag is in Region::condition.flag.
struct LoopShape {
    // EXEC/VCC/SCC0 exits and bottom-tested loops: continue while the represented predicate is
    // set; an SCC1 exit leaves while SCC is set. CountedLoop: true when it continues while SCC==1.
    bool continue_on_set = true;
    bool bottom_tested = false;   // the back edge is the exit test (do-while)
    bool direct_exec_breaks = false;   // unconditional back edge: an execz break exits directly
    bool direct_wave_breaks = false;   // fragment wave64: a vccz break exits the complete wave
};

struct Region {
    enum class Kind : uint8_t { Seq, Block, IfThen, IfElse, Loop, CountedLoop, Unstructured };
    Kind kind = Kind::Seq;
    // Block: the block. IfThen/IfElse: the block ending in the branch. Loop/CountedLoop: the loop
    // header (the back edge's target). Seq/Unstructured: the first block of the range.
    uint32_t header = kNone;
    // IfThen: the taken successor. IfElse: the then-arm's s_branch target. kNone for an if that
    // ends the program (an early-out to s_endpgm). An IfElse merge may lie past the enclosing
    // range: the shared-outer-merge cascade, accepted as detect_forward_ifs accepts it.
    // Loop/CountedLoop: the block after the back edge (where every exit and break lands).
    uint32_t merge = kNone;
    BranchCondition condition;
    // Seq: its parts in emission order. IfThen: {then}. IfElse: {then, else}. Loop/CountedLoop:
    // {body}, a Seq over [header, merge) that starts with the header block. An empty arm is kNone.
    std::vector<uint32_t> children;
    // Loop/CountedLoop: the block ending in the back edge (the latch).
    uint32_t continue_block = kNone;
    // Loop/CountedLoop: the block ending in the canonical exit test; the continue block when the
    // loop is bottom-tested.
    uint32_t exit_branch = kNone;
    // Loop: the blocks ending in a break (a further forward vccz/execz to the merge).
    std::vector<uint32_t> breaks;
    LoopShape loop;
    // IfThen: this if is a loop break. Its arm runs to the end of the loop body, and the branch
    // skips to the back-edge instruction (detect_forward_ifs' break lowering).
    bool loop_break = false;
    // IfElse: the then-arm's last block, which ends in the s_branch to the merge.
    uint32_t else_jump = kNone;
    // IfElse/IfThen: the first block of the else-arm or of the code after the then-arm (the
    // branch target); kNone for an early-out.
    uint32_t taken = kNone;
    // Blocks [first_block, end_block) this region covers.
    uint32_t first_block = kNone, end_block = kNone;
    std::string reason;   // Unstructured: why the range could not be structured
};

struct RegionTree {
    std::vector<Region> regions;
    uint32_t root = kNone;
    bool has_unstructured() const;
};

// The detectors' admission inputs (rdna2_cfg_support.hpp). The rules themselves are not restated
// here; build_region_tree calls the shared predicates with these values.
struct RegionPolicy {
    // Graphics stages structure SCC, VCC and EXEC conditions with their vote model.
    bool allow_vcc = true;
    // Compute: VCC branches are exact guest-wave branches, EXEC branches subgroup votes.
    bool compute_wave_branches = false;
    // The stage's linearizable branches (`safe`: safe_execz_branches, waterfall_branches and the
    // fragment mask tests): SCC and EXECZ branches in it are not ifs, and a backward branch in it
    // is not a divergent loop's back edge.
    const std::unordered_set<uint32_t>* linearized = nullptr;
    // detect_divergent_loops' exact_fragment_wave_breaks: the fragment stage's forced wave64 vote
    // admits a VCC exit without the uniformity proof and a vccz break with an s_branch back edge.
    bool fragment_wave_breaks = false;
    // The whole guest code, so an early-out past s_endpgm can be shown to terminate.
    const uint32_t* code = nullptr;
    size_t dwords = 0;
};

// `ins` is the instruction vector `cfg` was built from.
RegionTree build_region_tree(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins,
                             const RegionPolicy& policy);

}   // namespace prosper::gpu::ir::cfg

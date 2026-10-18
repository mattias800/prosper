// region_tree.hpp -- the structured regions of a GuestCfg (IR_CONTROL_FLOW_PLAN.md step 1).
//
// Step 1 builds Seq, Block, IfThen and IfElse, plus Unstructured for a sub-CFG it cannot express.
// Loops are not structured yet (plan step 4): a back edge leaves its range Unstructured. The
// admission rules are detect_forward_ifs' own, called through the shared predicates in
// rdna2_cfg_support.hpp, so the tree structures exactly the ifs the detector accepts. The
// contract (terminator table, RegionPolicy rules) is in AGENTS.md next to this file.
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

struct Region {
    enum class Kind : uint8_t { Seq, Block, IfThen, IfElse, Loop, CountedLoop, Unstructured };
    Kind kind = Kind::Seq;
    // Block: the block. IfThen/IfElse: the block ending in the branch. Seq/Unstructured: the first
    // block of the range.
    uint32_t header = kNone;
    // IfThen: the taken successor. IfElse: the then-arm's s_branch target. kNone for an if that
    // ends the program (an early-out to s_endpgm). An IfElse merge may lie past the enclosing
    // range: the shared-outer-merge cascade, accepted as detect_forward_ifs accepts it.
    uint32_t merge = kNone;
    BranchCondition condition;
    // Seq: its parts in emission order. IfThen: {then}. IfElse: {then, else}. An empty arm is
    // kNone.
    std::vector<uint32_t> children;
    uint32_t continue_block = kNone;   // loops (not built in step 1)
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

// detect_forward_ifs' admission inputs (rdna2_cfg_support.hpp). The rules themselves are not
// restated here; build_region_tree calls the shared predicates with these values.
struct RegionPolicy {
    // Graphics stages structure SCC, VCC and EXEC conditions with their vote model.
    bool allow_vcc = true;
    // Compute: VCC branches are exact guest-wave branches, EXEC branches subgroup votes.
    bool compute_wave_branches = false;
    // The stage's linearizable branches (`safe`: safe_execz_branches, waterfall_branches and the
    // fragment mask tests): SCC and EXECZ branches in it are not ifs.
    const std::unordered_set<uint32_t>* linearized = nullptr;
    // The whole guest code, so an early-out past s_endpgm can be shown to terminate.
    const uint32_t* code = nullptr;
    size_t dwords = 0;
};

// `ins` is the instruction vector `cfg` was built from.
RegionTree build_region_tree(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins,
                             const RegionPolicy& policy);

}   // namespace prosper::gpu::ir::cfg

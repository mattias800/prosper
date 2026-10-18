// region_shadow.hpp -- the step-1 shadow check: the region tree's IfThen/IfElse set against
// detect_forward_ifs' output, at the pc level (IR_CONTROL_FLOW_PLAN.md, "What keeps output
// byte-identical": every new structure is shadow-checked before a consumer switches).
//
// Used by test_region_shadow (committed fixtures) and `shader_inspect --cfg-regions` (the corpus
// census). Never by emission.
//
// Scope: both sides receive the same instruction vector and the same admission inputs, derived
// here the way the stage entry points derive them (safe_execz_branches + waterfall_branches, plus
// the Wave64 fragment mask tests). So this compares the two STRUCTURERS; a stage entry that adds
// branches to its `safe` set (the vertex trailing-VCC gates) or rewrites the stream first (the
// fragment lexical-if tail) is not reproduced, and programs with a recognized loop are reported
// as `loops` and not compared (plan step 4 structures loops).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu::ir::cfg {

// One structured if, in detect_forward_ifs' terms. sb_pc and merge_pc are 0 without an else.
struct ShadowIf {
    uint32_t branch_pc = 0, target_pc = 0, sb_pc = 0, merge_pc = 0;
    bool has_else = false, early_out = false;
    bool on_scc0 = false, on_vcc = false, on_exec = false;
    bool uniform_workgroup = false;
    bool operator==(const ShadowIf&) const = default;
};

enum class ShadowStage : uint8_t { Vertex, Fragment, Compute };

struct ForwardIfShadow {
    enum class Verdict : uint8_t { Loops, Agree, Disagree } verdict = Verdict::Agree;
    bool detector_refused = false;   // detect_forward_ifs' `rejected`
    bool tree_unstructured = false;
    std::vector<ShadowIf> detector, tree;   // sorted by branch_pc
    std::string tree_reason;   // the Unstructured reason, when there is one
    uint32_t blocks = 0;   // GuestCfg block count
    // One line describing the first difference; empty when the sides agree.
    std::string difference;
};

ForwardIfShadow shadow_forward_ifs(const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                                   const uint32_t* code, size_t dwords);

// "agree-structured", "agree-linear", "agree-refused", "loops" or "disagree".
const char* shadow_category(const ForwardIfShadow& shadow);

}   // namespace prosper::gpu::ir::cfg

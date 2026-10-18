// region_shadow.hpp -- the shadow check: the region tree's ifs and loops against
// detect_forward_ifs, detect_divergent_loops and detect_counted_loop, at the pc level
// (IR_CONTROL_FLOW_PLAN.md, "What keeps output byte-identical": every new structure is
// shadow-checked before a consumer switches). Steps 1 (ifs) and 4 (loops).
//
// Used by test_ir_cfg_region_shadow (committed fixtures) and `shader_inspect --cfg-regions` (the
// corpus census). Never by emission.
//
// Scope: both sides receive the same instruction vector and the same admission inputs, derived
// here the way the stage entry points derive them (safe_execz_branches + waterfall_branches, plus
// the Wave64 fragment mask tests). So this compares the STRUCTURERS; a stage entry that adds
// branches to its `safe` set (the vertex trailing-VCC gates) or rewrites the stream first (the
// fragment lexical-if tail) is not reproduced. The detectors are taken in emit_body's order: a
// counted loop first (then only the loop is compared: the counted route structures its prelude
// and postlude with separate detector calls), else the divergent loops and the forward ifs they
// claim branches from.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu::ir::cfg {

// One structured if, in detect_forward_ifs' terms. sb_pc and merge_pc are 0 without an else. A
// loop break is an if whose target is its loop's back edge.
struct ShadowIf {
    uint32_t branch_pc = 0, target_pc = 0, sb_pc = 0, merge_pc = 0;
    bool has_else = false, early_out = false;
    bool on_scc0 = false, on_vcc = false, on_exec = false;
    bool uniform_workgroup = false;
    bool operator==(const ShadowIf&) const = default;
};

// One loop, in the detectors' terms (DivLoop, or CountedLoop with `counted`).
struct ShadowLoop {
    uint32_t header_pc = 0, exit_branch_pc = 0, backedge_pc = 0, exit_pc = 0;
    std::vector<uint32_t> break_pcs;
    enum class Exit : uint8_t { Exec, Vcc, Scc } exit = Exit::Exec;
    bool continue_on_set = true, bottom_tested = false;
    bool direct_exec_breaks = false, direct_wave_breaks = false;
    bool counted = false;
    bool operator==(const ShadowLoop&) const = default;
};

enum class ShadowStage : uint8_t { Vertex, Fragment, Compute };

struct RegionShadow {
    enum class Verdict : uint8_t { Agree, Disagree } verdict = Verdict::Agree;
    // The detectors refused: detect_forward_ifs' `rejected`, or detect_divergent_loops returned
    // nothing although a candidate back edge exists.
    bool detector_refused = false;
    bool counted = false;   // detect_counted_loop found the program's loop
    bool tree_unstructured = false;
    std::vector<ShadowIf> detector, tree;   // sorted by branch_pc; empty when `counted`
    std::vector<ShadowLoop> detector_loops, tree_loops;   // sorted by header_pc
    std::string tree_reason;   // the Unstructured reason, when there is one
    uint32_t blocks = 0;   // GuestCfg block count
    // One line describing the first difference; empty when the sides agree.
    std::string difference;
};

RegionShadow shadow_regions(const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                            const uint32_t* code, size_t dwords);

// "agree-structured", "agree-linear", "agree-loops", "agree-counted", "agree-refused" or
// "disagree". A program with a structured loop is agree-loops whether or not it has ifs.
const char* shadow_category(const RegionShadow& shadow);

}   // namespace prosper::gpu::ir::cfg

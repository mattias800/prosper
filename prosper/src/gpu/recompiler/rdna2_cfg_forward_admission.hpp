#pragma once
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// A single structured uniform IF: exactly ONE control-flow branch in the whole stream, and it is a
// FORWARD s_cbranch_scc0/scc1 (a scalar-uniform conditional — all lanes take the same path). The
// conditional block is [branch_pc+1, target_pc). Anything more (any s_branch/vcc/execz/execnz branch,
// a backward branch/loop, or a target outside the stream) is rejected → falls back to straight-line
// (which then rejects the scc branch, as before). Deliberately conservative: covers the common single
// forward uniform-if without a general structurizer.
struct ForwardIf {
    bool found = false;
    uint32_t branch_pc = 0, target_pc = 0;
    bool on_scc0 =
        true;   // s_cbranch_scc0/vccz/execz (branch taken == skip block when flag==0) vs scc1/vccnz
    bool on_vcc =
        false;   // condition register: false = SCC, true = VCC (both are wave-uniform branches)
    bool early_out =
        false;   // branch target was clamped to end_pc: the conditional block ENDS the shader
    // Divergent-region if (#273): the branch is a forward s_cbranch_execz over an EXEC-narrowed block
    // (v_cmpx … s_cbranch_execz — DOLL's FXAA PS). Fragment reduces EXEC across an enforced wave64;
    // other structured stages use the guarded per-invocation model. EXEC is phi'd at the merge.
    bool on_exec = false;
    // Compute only (#1554): this VCC branch was PROVED to decide identically for every wave in the
    // workgroup, so its region may contain guest barriers and nested wave votes. Left false for every
    // branch whose uniformity is not proved, which keeps the conservative per-wave treatment.
    bool uniform_workgroup = false;
    // IF/ELSE (#273): the then-arm [branch_pc+1, sb_pc) is terminated by `s_branch merge_pc` at sb_pc
    // (the instruction immediately before target_pc); the else-arm starts at target_pc and runs to
    // merge_pc (or to the enclosing region's end when merge_pc is the shared OUTER merge — DOLL's
    // color-grade if/else-if cascade, where every arm's s_branch jumps to the outermost merge).
    bool has_else = false;
    uint32_t sb_pc = 0, merge_pc = 0;
};

// ---- detect_forward_ifs' admission rules, shared with the IR region tree --------------------
// ir/cfg/region_tree.cpp structures the same programs from a CFG and must admit exactly what the
// detector admits, so each rule lives here once and both call it. A rule that refuses fills
// `why` with the route-decline text the detector logs under compute-struct-reject; the region
// tree keeps it as its Unstructured reason. None of these log or touch recompiler state.
struct ForwardIfRefusal {
    char text[384] = {};
};

// What detect_forward_ifs makes of one SOPP instruction before it looks at the target.
enum class ForwardIfBranchClass : uint8_t {
    NotACondition,   // s_branch (validated as an else jump) and every non-branch SOPP
    Claimed,   // linearized through `skip`, or owned by a loop emitter: not an if
    If,   // a forward-if candidate
    UniformVccIf,   // compute vccz/vccnz admitted for exact guest-wave control flow
    Refused,
};

// Which proof, if any, shows a structured if decides identically for every wave of the
// workgroup, so its region may synchronize (#1554 for VCC, #3713 for SCC).
enum class ForwardIfWorkgroupUniform : uint8_t { No, Vcc, Scc };

void forward_if_refuse(ForwardIfRefusal* why, const char* format, ...);
bool div_loop_backedge_at(const std::vector<DivLoop>* loops, uint32_t pc);
bool div_loop_exit_at(const std::vector<DivLoop>* loops, uint32_t pc);
const DivLoop* div_loop_break_at(const std::vector<DivLoop>* loops, uint32_t pc);
ForwardIfBranchClass classify_forward_if_branch(const Rdna2Inst& in, bool allow_vcc,
                                                bool compute_wave_branches,
                                                const std::unordered_set<uint32_t>* skip,
                                                const std::vector<DivLoop>* loops,
                                                ForwardIfRefusal* why);
bool resolve_forward_if_target(const Rdna2Inst& in, uint32_t end_pc, const uint32_t* code,
                               size_t dwords, uint32_t* target, bool* early, ForwardIfRefusal* why);
bool compute_execz_region_admissible(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                     uint32_t region_end, ForwardIfRefusal* why);
bool wave_vcc_region_admissible(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                uint32_t region_end, ForwardIfRefusal* why);
ForwardIfWorkgroupUniform forward_if_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                                       const Rdna2Inst& in,
                                                       bool compute_uniform_vcc,
                                                       bool compute_wave_branches);
std::vector<ForwardIf>
detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc, const uint32_t* code,
                   size_t dwords, const std::unordered_set<uint32_t>* skip = nullptr,
                   const std::vector<DivLoop>* loops = nullptr, bool* rejected = nullptr,
                   bool compute_wave_branches = false, RecompileDiagnosticContext diagnostic = {});
}   // namespace prosper::gpu

// Existing forward-if detector and original closure ownership.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp"
#include "gpu/recompiler/rdna2_cfg_forward_detector_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdio>
#include <functional>
#include <optional>

namespace prosper::gpu {

// MULTIPLE structured uniform IFs — the generalization of detect_forward_if to N forward
// s_cbranch_scc*/vcc* branches that form a properly-NESTED (or disjoint sequential) region tree.
// Real shaders routinely carry several sequential uniform ifs (DOLL's color-grade PS: two
// `s_cbranch_vccz` blocks) or a nested guard (`vccz L1; …; vccz L2` with L2 < L1). Same per-branch
// validity rules as detect_forward_if (forward target inside the stream, or a VERIFIED early-out
// clamped to s_endpgm; scc branches in `skip` are alpha-test kill-masks handled by linearization);
// plus a structural check: blocks must nest or be disjoint — a partial overlap (branch into the
// middle of another block) is not an if-tree and rejects. Returns branches in pc order; empty =
// no/unsupported control flow (the caller falls back to straight-line, which rejects loudly at the
// branch). Compute execz uses a subgroup vote; execnz and unclaimed s_branch still reject wholesale.
// CONFIDENCE: HIGH on the structure (guarded by the phi machinery shared with the single-if path).
std::vector<ForwardIf> detect_forward_ifs(const std::vector<Rdna2Inst>& ins, bool allow_vcc,
                                          const uint32_t* code, size_t dwords,
                                          const std::unordered_set<uint32_t>* skip,
                                          const std::vector<DivLoop>* loops, bool* rejected,
                                          bool compute_wave_branches,
                                          RecompileDiagnosticContext diagnostic) {
    std::vector<ForwardIf> out;
    if (rejected) *rejected = false;
    auto reject = [&]() -> std::vector<ForwardIf> {
        if (rejected) *rejected = true;
        return {};
    };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins)
        if (in.is_end) {
            end_pc = in.pc;
            break;
        }
    // Loop-claimed pcs (#273): back-edges + canonical exits are consumed by the loop emitter; a break
    // lowers as a forward if over the remainder of its loop's body (target clamped to the back-edge).
    auto loop_backedge = [&](uint32_t pc) { return div_loop_backedge_at(loops, pc); };
    auto decline = [&](const ForwardIfRefusal& why) {
        log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline", "%s",
                                 why.text);
        return reject();
    };
    // Pass 1: collect the conditional branches (else-arm s_branch terminators are claimed in pass 2).
    if (auto result = forward_if_collect(ins, allow_vcc, compute_wave_branches, skip, loops,
                                         std::cref(decline), diagnostic, std::cref(reject), out,
                                         end_pc, code, dwords, std::cref(loop_backedge)))
        return *result;
    // Pass 2: every s_branch in the stream must be a claimed else-arm terminator or a loop back-edge —
    // any other unconditional branch is control flow this structurizer doesn't model.
    if (auto result = forward_if_claimed_branches(ins, std::cref(loop_backedge), out, diagnostic,
                                                  std::cref(reject)))
        return *result;
    // Structural nesting check (branches are already in pc order): each region must lie entirely
    // inside the innermost still-open region, or start after it closed. A region spans to its merge
    // (if/else) or target (plain if); a LOOP is an opaque region [header_pc, exit_pc) that must nest
    // strictly. EXCEPTION (the shared-outer-merge cascade): an if/else whose merge_pc escapes the
    // innermost open region is accepted here and re-validated during emission (emit_structured
    // requires the escaping merge to equal the enclosing merge chain's continuation, i.e. no
    // instructions are skipped) — a violation rejects the shader there, fail-visible.
    std::vector<uint32_t> open;
    if (auto result = forward_if_nesting(loops, out, open, diagnostic, std::cref(reject)))
        return *result;
    return out;
}

}   // namespace prosper::gpu

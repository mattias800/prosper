// Forward-if collection, region marking and structural validation.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_cfg_forward_detector_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdio>
#include <functional>
#include <optional>

namespace prosper::gpu {
void forward_if_break_shape(const prosper::gpu::Rdna2Inst& in, const DivLoop*& L,
                            std::vector<ForwardIf>& out) {
    ForwardIf F;
    F.found = true;
    F.branch_pc = in.pc;
    F.target_pc = L->backedge_pc;
    F.on_scc0 = true;   // vccz/execz: skip block when flag==0
    F.on_vcc = (in.opcode == 0x06);
    F.on_exec = (in.opcode == 0x08);
    out.push_back(F);
}

std::optional<std::vector<ForwardIf>>
forward_if_else_shape(bool& early, uint32_t& tgt, uint32_t& end_pc,
                      const std::vector<Rdna2Inst>& ins, const prosper::gpu::Rdna2Inst& in,
                      const std::function<bool(uint32_t)>& loop_backedge,
                      RecompileDiagnosticContext& diagnostic,
                      const std::function<std::vector<ForwardIf>()>& reject, ForwardIf& F) {
    if (!early && tgt != end_pc) {
        for (const auto& sb : ins) {
            if (sb.pc <= in.pc || sb.pc >= tgt) continue;
            if (sb.fmt != Rdna2Format::SOPP || sb.opcode != 0x02) continue;
            if (sb.pc + sb.len_dwords != tgt) continue;   // must be the arm's LAST instruction
            if (loop_backedge(sb.pc)) continue;   // a loop's back-edge, not an else-jump
            if (sb.simm16 <= 0) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "backward else pc=%u branch=%u target=%u", in.pc, sb.pc,
                                         tgt);
                return reject();   // backward else-jump: a loop, not an if
            }
            uint32_t lm = branch_target(sb);
            if (lm <= tgt || lm > end_pc) {
                log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                         "invalid else merge pc=%u branch=%u target=%u merge=%u",
                                         in.pc, sb.pc, tgt, lm);
                return reject();   // merge must be forward, in-stream
            }
            F.has_else = true;
            F.sb_pc = sb.pc;
            F.merge_pc = lm;
            break;
        }
    }
    return std::nullopt;
}

void forward_if_workgroup_mark(const std::vector<Rdna2Inst>& ins, const prosper::gpu::Rdna2Inst& in,
                               bool& compute_uniform_vcc, bool& compute_wave_branches,
                               ForwardIf& F) {
    // A branch whose mask is built only from launch data over a full EXEC decides identically for
    // every wave of the workgroup (#1554), so its region may synchronize. Prove it before the
    // per-wave restriction below applies.
    //
    // The same proof for a SCALAR branch (#3713). A generated uber-shader selects between
    // barrier-separated variants with `s_cbranch_scc0/1` on a constant read once from a uniform
    // address, so its barriers sit inside SCC regions and nothing else. `top_level_pc` already
    // skips a region carrying this flag -- that machinery came in with #1554 and was only ever
    // fed by VCC branches, which is the whole of why those programs were rejected.
    //
    // Both conjuncts this unblocks want the same guarantee and say so in their own comments:
    // `barriers_are_top_level` needs every workgroup invocation to reach a barrier together, and
    // `structured_wave_forward_ifs_ok` needs every workgroup invocation to reach a portable
    // scratch vote's synthesized barriers together. A workgroup-uniform enclosing condition is
    // exactly that: all of them take the region or none does.
    const ForwardIfWorkgroupUniform uniform =
        forward_if_workgroup_uniform(ins, in, compute_uniform_vcc, compute_wave_branches);
    if (uniform != ForwardIfWorkgroupUniform::No) {
        F.uniform_workgroup = true;
        compute_uniform_vcc = false;
        // Existing diagnostic read retained verbatim.
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        if (getenv("PROSPER_DBG"))
            std::fprintf(stderr, "[compute-cfg] workgroup-uniform %s branch pc=%u\n",
                         uniform == ForwardIfWorkgroupUniform::Vcc ? "vcc" : "scc", in.pc);
    }
}

std::optional<std::vector<ForwardIf>>
forward_if_collect(const std::vector<Rdna2Inst>& ins, bool& allow_vcc, bool& compute_wave_branches,
                   const std::unordered_set<uint32_t>*& skip, const std::vector<DivLoop>*& loops,
                   const std::function<std::vector<ForwardIf>(const ForwardIfRefusal&)>& decline,
                   RecompileDiagnosticContext& diagnostic,
                   const std::function<std::vector<ForwardIf>()>& reject,
                   std::vector<ForwardIf>& out, uint32_t& end_pc, const uint32_t*& code,
                   size_t& dwords, const std::function<bool(uint32_t)>& loop_backedge) {
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP) continue;
        ForwardIfRefusal why;
        const ForwardIfBranchClass cls =
            classify_forward_if_branch(in, allow_vcc, compute_wave_branches, skip, loops, &why);
        if (cls == ForwardIfBranchClass::Refused) return decline(why);
        if (cls != ForwardIfBranchClass::If && cls != ForwardIfBranchClass::UniformVccIf) continue;
        // Set when a compute vccz/vccnz if is accepted for exact guest-wave control flow. Its region
        // must remain free of operations that require every workgroup invocation to reach one common
        // synchronized phase (checked after the target/merge is known).
        bool compute_uniform_vcc = cls == ForwardIfBranchClass::UniformVccIf;
        // A loop BREAK (validated vccz/execz -> exit_pc inside an execnz-back-edge loop): a plain
        // forward if skipping the REST of the body — the exec-governed continue then exits the lane.
        // A compute-uniform-accepted vcc branch can never be a break (compute loops require the
        // s_branch back-edge Vcc shape, which has no breaks) — reject the combination defensively.
        if (compute_uniform_vcc && div_loop_break_at(loops, in.pc)) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "uniform vcc loop break pc=%u", in.pc);
            return reject();
        }
        if (const DivLoop* L = div_loop_break_at(loops, in.pc)) {
            forward_if_break_shape(in, L, out);
            continue;
        }
        uint32_t tgt = 0;
        bool early = false;
        if (!resolve_forward_if_target(in, end_pc, code, dwords, &tgt, &early, &why))
            return decline(why);
        ForwardIf F;
        F.found = true;
        F.branch_pc = in.pc;
        F.target_pc = tgt;
        F.early_out = (early || tgt == end_pc);
        F.on_scc0 = (in.opcode == 0x04 || in.opcode == 0x06 || in.opcode == 0x08);
        F.on_vcc = (in.opcode == 0x06 || in.opcode == 0x07);
        F.on_exec = (in.opcode == 0x08);
        // IF/ELSE detection: the instruction immediately preceding target_pc is a FORWARD s_branch —
        // the then-arm's terminator jumping to the merge; the else-arm is [target_pc, merge). (Skip
        // for early-outs: the arm ends the shader.)
        if (auto result =
                forward_if_else_shape(early, tgt, end_pc, ins, in, std::cref(loop_backedge),
                                      diagnostic, std::cref(reject), F))
            return *result;   // NOLINT(bugprone-optional-value-conversion): retain the original vector refusal through delegation.
        const uint32_t region_end = F.has_else ? F.merge_pc : F.target_pc;
        if (!allow_vcc && compute_wave_branches && F.on_exec &&
            !compute_execz_region_admissible(ins, in.pc, region_end, &why))
            return decline(why);
        forward_if_workgroup_mark(ins, in, compute_uniform_vcc, compute_wave_branches, F);
        if (compute_uniform_vcc && !wave_vcc_region_admissible(ins, in.pc, region_end, &why))
            return decline(why);
        out.push_back(F);
    }
    return std::nullopt;
}

std::optional<std::vector<ForwardIf>>
forward_if_claimed_branches(const std::vector<Rdna2Inst>& ins,
                            const std::function<bool(uint32_t)>& loop_backedge,
                            std::vector<ForwardIf>& out, RecompileDiagnosticContext& diagnostic,
                            const std::function<std::vector<ForwardIf>()>& reject) {
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || in.opcode != 0x02) continue;
        if (loop_backedge(in.pc)) continue;
        bool claimed = false;
        for (const auto& F : out)
            if (F.has_else && F.sb_pc == in.pc) {
                claimed = true;
                break;
            }
        if (!claimed) {
            log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                     "unclaimed s_branch pc=%u target=%u", in.pc,
                                     branch_target(in));
            return reject();
        }
    }
    return std::nullopt;
}

std::optional<std::vector<ForwardIf>>
forward_if_nesting(const std::vector<DivLoop>*& loops, std::vector<ForwardIf>& out,
                   std::vector<uint32_t>& open, RecompileDiagnosticContext& diagnostic,
                   const std::function<std::vector<ForwardIf>()>& reject) {
    {
        size_t fi = 0, li = 0;
        const size_t nl = loops ? loops->size() : 0;
        while (fi < out.size() || li < nl) {
            const bool take_loop =
                li < nl && (fi >= out.size() || (*loops)[li].header_pc < out[fi].branch_pc);
            const uint32_t start = take_loop ? (*loops)[li].header_pc : out[fi].branch_pc;
            uint32_t span_end = take_loop
                                    ? (*loops)[li].exit_pc
                                    : (out[fi].has_else ? out[fi].merge_pc : out[fi].target_pc);
            const bool clampable = !take_loop && out[fi].has_else;
            while (!open.empty() && open.back() <= start) open.pop_back();
            if (!open.empty() && span_end > open.back()) {
                if (!clampable) {
                    log_recompile_diagnostic(diagnostic, "compute-struct-reject", "route-decline",
                                             "overlapping region start=%u end=%u parent-end=%u",
                                             start, span_end, open.back());
                    return reject();   // plain if / loop escaping its region: not a tree
                }
                span_end = open.back();   // cascade if/else: clamped to the enclosing
            }   // region for nesting purposes (see above)
            open.push_back(span_end);
            if (take_loop)
                li++;
            else
                fi++;
        }
    }
    return std::nullopt;
}

}   // namespace prosper::gpu

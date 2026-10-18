// region_shadow.cpp -- see region_shadow.hpp.
#include "gpu/recompiler/ir/cfg/region_shadow.hpp"

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_copy_continuation.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

#include <algorithm>
#include <cstdio>

namespace prosper::gpu::ir::cfg {

namespace {

std::vector<ShadowIf> from_detector(const std::vector<ForwardIf>& ifs) {
    std::vector<ShadowIf> out;
    for (const ForwardIf& f : ifs) {
        ShadowIf s;
        s.branch_pc = f.branch_pc;
        s.target_pc = f.target_pc;
        s.has_else = f.has_else;
        s.sb_pc = f.has_else ? f.sb_pc : 0;
        s.merge_pc = f.has_else ? f.merge_pc : 0;
        s.early_out = f.early_out;
        s.on_scc0 = f.on_scc0;
        s.on_vcc = f.on_vcc;
        s.on_exec = f.on_exec;
        s.uniform_workgroup = f.uniform_workgroup;
        out.push_back(s);
    }
    return out;
}

std::vector<ShadowIf> from_tree(const RegionTree& tree, const GuestCfg& cfg,
                                const std::vector<Rdna2Inst>& ins, uint32_t end_pc) {
    std::vector<ShadowIf> out;
    for (const Region& r : tree.regions) {
        if (r.kind != Region::Kind::IfThen && r.kind != Region::Kind::IfElse) continue;
        ShadowIf s;
        s.branch_pc = ins[cfg.blocks[r.header].last].pc;
        s.early_out = r.taken == kNone;
        s.target_pc = s.early_out ? end_pc : cfg.blocks[r.taken].pc;
        s.has_else = r.kind == Region::Kind::IfElse;
        if (s.has_else) {
            s.sb_pc = ins[cfg.blocks[r.else_jump].last].pc;
            s.merge_pc = cfg.blocks[r.merge].pc;
        }
        s.on_scc0 = r.condition.taken_on_zero;
        s.on_vcc = r.condition.flag == BranchCondition::Flag::Vcc;
        s.on_exec = r.condition.flag == BranchCondition::Flag::Exec;
        s.uniform_workgroup = r.condition.uniform_workgroup;
        out.push_back(s);
    }
    return out;
}

std::string describe(const ShadowIf& s) {
    char line[160];
    std::snprintf(line, sizeof line, "{branch=%u target=%u else=%d sb=%u merge=%u early=%d}",
                  s.branch_pc, s.target_pc, s.has_else ? 1 : 0, s.sb_pc, s.merge_pc,
                  s.early_out ? 1 : 0);
    return line;
}

std::string first_difference(const ForwardIfShadow& s) {
    if (s.detector_refused != s.tree_unstructured)
        return s.detector_refused
                   ? "detector refused, tree structured"
                   : "tree unstructured (" + s.tree_reason + "), detector structured";
    if (s.detector_refused) return {};
    const size_t n = std::max(s.detector.size(), s.tree.size());
    for (size_t i = 0; i < n; ++i) {
        const bool d = i < s.detector.size(), t = i < s.tree.size();
        if (d && t && s.detector[i] == s.tree[i]) continue;
        return "if " + std::to_string(i) + ": detector " + (d ? describe(s.detector[i]) : "none") +
               " tree " + (t ? describe(s.tree[i]) : "none");
    }
    return {};
}

bool by_branch(const ShadowIf& a, const ShadowIf& b) {
    return a.branch_pc < b.branch_pc;
}

}   // namespace

ForwardIfShadow shadow_forward_ifs(const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                                   const uint32_t* code, size_t dwords) {
    ForwardIfShadow out;
    const bool compute = stage == ShadowStage::Compute;
    const bool fragment = stage == ShadowStage::Fragment;
    std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    if (fragment)
        for (uint32_t pc : fragment_mask_test_branches(ins, /*wave32*/ false)) safe.insert(pc);
    if (detect_counted_loop(ins).found || !detect_divergent_loops(ins, safe, fragment).empty()) {
        out.verdict = ForwardIfShadow::Verdict::Loops;
        return out;
    }
    out.detector = from_detector(detect_forward_ifs(ins, /*allow_vcc*/ !compute, code, dwords,
                                                    &safe, nullptr, &out.detector_refused,
                                                    /*compute_wave_branches*/ compute));
    const GuestCfg cfg = build_guest_cfg(ins, SplitPolicy{});
    RegionPolicy policy;
    policy.allow_vcc = !compute;
    policy.compute_wave_branches = compute;
    policy.linearized = &safe;
    policy.code = code;
    policy.dwords = dwords;
    const RegionTree tree = build_region_tree(cfg, ins, policy);
    out.blocks = static_cast<uint32_t>(cfg.blocks.size());
    out.tree_unstructured = tree.has_unstructured();
    if (out.tree_unstructured) out.tree_reason = tree.regions[tree.root].reason;
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins)
        if (in.is_end) {
            end_pc = in.pc;
            break;
        }
    if (!out.tree_unstructured) out.tree = from_tree(tree, cfg, ins, end_pc);
    std::sort(out.detector.begin(), out.detector.end(), by_branch);
    std::sort(out.tree.begin(), out.tree.end(), by_branch);
    out.difference = first_difference(out);
    out.verdict = out.difference.empty() ? ForwardIfShadow::Verdict::Agree
                                         : ForwardIfShadow::Verdict::Disagree;
    return out;
}

const char* shadow_category(const ForwardIfShadow& shadow) {
    switch (shadow.verdict) {
        case ForwardIfShadow::Verdict::Loops: return "loops";
        case ForwardIfShadow::Verdict::Disagree: return "disagree";
        case ForwardIfShadow::Verdict::Agree: break;
    }
    if (shadow.detector_refused) return "agree-refused";
    return shadow.detector.empty() ? "agree-linear" : "agree-structured";
}

}   // namespace prosper::gpu::ir::cfg

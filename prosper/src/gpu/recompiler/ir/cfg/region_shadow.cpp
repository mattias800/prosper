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

using Exit = ShadowLoop::Exit;

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

std::vector<ShadowLoop> from_detector(const std::vector<DivLoop>& loops) {
    std::vector<ShadowLoop> out;
    for (const DivLoop& L : loops) {
        ShadowLoop s;
        s.header_pc = L.header_pc;
        s.exit_branch_pc = L.exit_branch_pc;
        s.backedge_pc = L.backedge_pc;
        s.exit_pc = L.exit_pc;
        s.break_pcs = L.break_pcs;
        s.exit = L.condition == DivLoop::Condition::Exec  ? Exit::Exec
                 : L.condition == DivLoop::Condition::Vcc ? Exit::Vcc
                                                          : Exit::Scc;
        s.continue_on_set = L.continue_on_set;
        s.bottom_tested = L.bottom_tested;
        s.direct_exec_breaks = L.direct_exec_breaks;
        s.direct_wave_breaks = L.direct_wave_breaks;
        out.push_back(s);
    }
    return out;
}

ShadowLoop from_counted(const CountedLoop& L) {
    ShadowLoop s;
    s.header_pc = L.header_pc;
    s.exit_branch_pc = L.exit_branch_pc;
    s.backedge_pc = L.backedge_pc;
    s.exit_pc = L.exit_pc;
    s.exit = Exit::Scc;
    s.continue_on_set = L.exit_on_scc0;
    s.bottom_tested = L.exit_branch_pc == L.backedge_pc;
    s.counted = true;
    return s;
}

// The tree's view, in the same terms. `ins` and `cfg` are what the tree was built from.
class TreeView {
public:
    TreeView(const RegionTree& tree, const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins,
             uint32_t end_pc)
        : tree_(tree), cfg_(cfg), ins_(ins), end_pc_(end_pc) {}

    std::vector<ShadowIf> ifs() const {
        std::vector<ShadowIf> out;
        for (const Region& r : tree_.regions)
            if (r.kind == Region::Kind::IfThen || r.kind == Region::Kind::IfElse)
                out.push_back(if_of(r));
        return out;
    }

    std::vector<ShadowLoop> loops() const {
        std::vector<ShadowLoop> out;
        for (const Region& r : tree_.regions)
            if (r.kind == Region::Kind::Loop || r.kind == Region::Kind::CountedLoop)
                out.push_back(loop_of(r));
        return out;
    }

private:
    const Rdna2Inst& last(uint32_t block) const { return ins_[cfg_.blocks[block].last]; }

    ShadowIf if_of(const Region& r) const {
        ShadowIf s;
        s.branch_pc = last(r.header).pc;
        s.early_out = r.taken == kNone;
        s.target_pc = s.early_out    ? end_pc_
                      : r.loop_break ? last(r.taken).pc   // the back edge
                                     : cfg_.blocks[r.taken].pc;
        s.has_else = r.kind == Region::Kind::IfElse;
        if (s.has_else) {
            s.sb_pc = last(r.else_jump).pc;
            s.merge_pc = cfg_.blocks[r.merge].pc;
        }
        s.on_scc0 = r.condition.taken_on_zero;
        s.on_vcc = r.condition.flag == BranchCondition::Flag::Vcc;
        s.on_exec = r.condition.flag == BranchCondition::Flag::Exec;
        s.uniform_workgroup = r.condition.uniform_workgroup;
        return s;
    }

    ShadowLoop loop_of(const Region& r) const {
        ShadowLoop s;
        const Rdna2Inst& back = last(r.continue_block);
        s.header_pc = cfg_.blocks[r.header].pc;
        s.exit_branch_pc = last(r.exit_branch).pc;
        s.backedge_pc = back.pc;
        s.exit_pc = back.pc + back.len_dwords;
        for (uint32_t b : r.breaks) s.break_pcs.push_back(last(b).pc);
        s.exit = r.condition.flag == BranchCondition::Flag::Exec  ? Exit::Exec
                 : r.condition.flag == BranchCondition::Flag::Vcc ? Exit::Vcc
                                                                  : Exit::Scc;
        s.continue_on_set = r.loop.continue_on_set;
        s.bottom_tested = r.loop.bottom_tested;
        s.direct_exec_breaks = r.loop.direct_exec_breaks;
        s.direct_wave_breaks = r.loop.direct_wave_breaks;
        s.counted = r.kind == Region::Kind::CountedLoop;
        return s;
    }

    const RegionTree& tree_;
    const GuestCfg& cfg_;
    const std::vector<Rdna2Inst>& ins_;
    uint32_t end_pc_;
};

std::string describe(const ShadowIf& s) {
    char line[160];
    std::snprintf(line, sizeof line, "{branch=%u target=%u else=%d sb=%u merge=%u early=%d}",
                  s.branch_pc, s.target_pc, s.has_else ? 1 : 0, s.sb_pc, s.merge_pc,
                  s.early_out ? 1 : 0);
    return line;
}

std::string describe(const ShadowLoop& s) {
    char line[200];
    std::snprintf(line, sizeof line,
                  "{header=%u exit-branch=%u back=%u exit=%u breaks=%zu cond=%d set=%d bottom=%d "
                  "counted=%d}",
                  s.header_pc, s.exit_branch_pc, s.backedge_pc, s.exit_pc, s.break_pcs.size(),
                  static_cast<int>(s.exit), s.continue_on_set ? 1 : 0, s.bottom_tested ? 1 : 0,
                  s.counted ? 1 : 0);
    return line;
}

template <class T>
std::string first_mismatch(const char* what, const std::vector<T>& detector,
                           const std::vector<T>& tree) {
    const size_t n = std::max(detector.size(), tree.size());
    for (size_t i = 0; i < n; ++i) {
        const bool d = i < detector.size(), t = i < tree.size();
        if (d && t && detector[i] == tree[i]) continue;
        return std::string(what) + " " + std::to_string(i) + ": detector " +
               (d ? describe(detector[i]) : "none") + " tree " + (t ? describe(tree[i]) : "none");
    }
    return {};
}

std::string first_difference(const RegionShadow& s) {
    if (s.detector_refused != s.tree_unstructured)
        return s.detector_refused
                   ? "detector refused, tree structured"
                   : "tree unstructured (" + s.tree_reason + "), detector structured";
    if (s.detector_refused) return {};
    if (std::string loops = first_mismatch("loop", s.detector_loops, s.tree_loops); !loops.empty())
        return loops;
    return first_mismatch("if", s.detector, s.tree);
}

uint32_t end_pc_of(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins)
        if (in.is_end) return in.pc;
    return UINT32_MAX;
}

// Whether detect_divergent_loops' pass-1 scan finds a candidate back edge before `before_pc`.
// It returns nothing both when there is no loop and when it refused one; this tells them apart.
bool has_divergent_candidate(const std::vector<Rdna2Inst>& ins,
                             const std::unordered_set<uint32_t>& safe,
                             uint32_t before_pc = UINT32_MAX) {
    const std::unordered_set<uint32_t> waterfalls = waterfall_branches(ins);
    for (const auto& in : ins) {
        if (in.is_end || in.pc >= before_pc) break;
        if (div_loop_backedge_candidate(in, safe, waterfalls)) return true;
    }
    return false;
}

// The detectors, in emit_body's order. A counted loop with another loop before it is not the
// counted route: that loop makes emit_body's prelude scan refuse, and it takes the general route.
void run_detectors(RegionShadow& out, const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                   const std::unordered_set<uint32_t>& safe, const uint32_t* code, size_t dwords) {
    const bool compute = stage == ShadowStage::Compute;
    if (const CountedLoop counted = detect_counted_loop(ins);
        counted.found && !has_divergent_candidate(ins, safe, counted.header_pc)) {
        out.counted = true;
        out.detector_loops = {from_counted(counted)};
        return;
    }
    const std::vector<DivLoop> loops =
        detect_divergent_loops(ins, safe, stage == ShadowStage::Fragment);
    out.detector_loops = from_detector(loops);
    bool rejected = false;
    out.detector = from_detector(detect_forward_ifs(ins, /*allow_vcc*/ !compute, code, dwords,
                                                    &safe, &loops, &rejected,
                                                    /*compute_wave_branches*/ compute));
    out.detector_refused = rejected || (loops.empty() && has_divergent_candidate(ins, safe));
}

bool by_branch(const ShadowIf& a, const ShadowIf& b) {
    return a.branch_pc < b.branch_pc;
}
bool by_header(const ShadowLoop& a, const ShadowLoop& b) {
    return a.header_pc < b.header_pc;
}

}   // namespace

RegionShadow shadow_regions(const std::vector<Rdna2Inst>& ins, ShadowStage stage,
                            const uint32_t* code, size_t dwords) {
    RegionShadow out;
    const bool compute = stage == ShadowStage::Compute;
    const bool fragment = stage == ShadowStage::Fragment;
    std::unordered_set<uint32_t> safe = safe_execz_branches(ins);
    for (uint32_t pc : waterfall_branches(ins)) safe.insert(pc);
    if (fragment)
        for (uint32_t pc : fragment_mask_test_branches(ins, /*wave32*/ false)) safe.insert(pc);
    run_detectors(out, ins, stage, safe, code, dwords);
    const GuestCfg cfg = build_guest_cfg(ins, SplitPolicy{});
    RegionPolicy policy;
    policy.allow_vcc = !compute;
    policy.compute_wave_branches = compute;
    policy.linearized = &safe;
    policy.fragment_wave_breaks = fragment;
    policy.code = code;
    policy.dwords = dwords;
    const RegionTree tree = build_region_tree(cfg, ins, policy);
    out.blocks = static_cast<uint32_t>(cfg.blocks.size());
    out.tree_unstructured = tree.has_unstructured();
    if (out.tree_unstructured) out.tree_reason = tree.regions[tree.root].reason;
    if (!out.tree_unstructured) {
        const TreeView view(tree, cfg, ins, end_pc_of(ins));
        out.tree_loops = view.loops();
        if (!out.counted) out.tree = view.ifs();
    }
    if (out.detector_refused) out.detector_loops.clear();
    std::sort(out.detector.begin(), out.detector.end(), by_branch);
    std::sort(out.tree.begin(), out.tree.end(), by_branch);
    std::sort(out.detector_loops.begin(), out.detector_loops.end(), by_header);
    std::sort(out.tree_loops.begin(), out.tree_loops.end(), by_header);
    out.difference = first_difference(out);
    out.verdict =
        out.difference.empty() ? RegionShadow::Verdict::Agree : RegionShadow::Verdict::Disagree;
    return out;
}

const char* shadow_category(const RegionShadow& shadow) {
    if (shadow.verdict == RegionShadow::Verdict::Disagree) return "disagree";
    if (shadow.detector_refused) return "agree-refused";
    if (shadow.counted) return "agree-counted";
    if (!shadow.detector_loops.empty()) return "agree-loops";
    return shadow.detector.empty() ? "agree-linear" : "agree-structured";
}

}   // namespace prosper::gpu::ir::cfg

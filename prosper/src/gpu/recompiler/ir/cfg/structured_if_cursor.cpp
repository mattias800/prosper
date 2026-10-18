// Region traversal is preferred only after its records and preorder equal the detector.
// No SPIR-V, register state or proof facts are created here; a disagreement retains Fs.
#include "gpu/recompiler/ir/cfg/structured_if_cursor.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cstdlib>
#include <tuple>
namespace prosper::gpu::ir::cfg {
RegionPolicy structured_if_policy(bool compute, const std::unordered_set<uint32_t>* linearized,
                                  const uint32_t* code, size_t dwords) {
    // Named fields: RegionPolicy gains members as the tree learns new shapes. The loop-only
    // fragment_wave_breaks stays false; the cursor falls back for any program with loops.
    ir::cfg::RegionPolicy if_policy;
    if_policy.allow_vcc = !compute;
    if_policy.compute_wave_branches = compute;
    if_policy.linearized = linearized;
    if_policy.code = code;
    if_policy.dwords = dwords;
    return if_policy;
}
namespace {
bool same_if(const ForwardIf& a, const ForwardIf& b) {
    return std::tie(a.found, a.branch_pc, a.target_pc, a.on_scc0, a.on_vcc, a.on_exec, a.early_out,
                    a.has_else, a.uniform_workgroup) ==
               std::tie(b.found, b.branch_pc, b.target_pc, b.on_scc0, b.on_vcc, b.on_exec,
                        b.early_out, b.has_else, b.uniform_workgroup) &&
           (!a.has_else || (a.sb_pc == b.sb_pc && a.merge_pc == b.merge_pc));
}
}   // namespace
StructuredIfCursor::StructuredIfCursor(const std::vector<Rdna2Inst>& ins,
                                       const std::vector<ForwardIf>& branches,
                                       const RegionPolicy& policy, bool detector_refused)
    : Fs(branches), ins_(ins) {
    if (Fs.empty()) return;
    if (detector_refused) {
        route_ = Route::DetectorRefused;
        return;
    }
    cfg_ = build_guest_cfg(ins, {});
    if (!cfg_.back_edges.empty()) {
        route_ = Route::Loops;
        return;
    }
    tree_ = build_region_tree(cfg_, ins, policy);
    if (tree_.has_unstructured()) {
        route_ = Route::Unstructured;
        return;
    }
    for (const auto& in : ins)
        if (in.is_end) {
            end_pc_ = in.pc;
            break;
        }
    pending_.reserve(tree_.regions.size());
    restart_tree();
    for (const auto& expected : Fs) {
        ForwardIf candidate;
        if (!read_tree_if(candidate) || !same_if(candidate, expected)) {
            route_ = Route::Mismatch;
            return;
        }
        advance_tree();
    }
    if (malformed_ || current_ != kNone) {
        route_ = Route::Mismatch;
        return;
    }
    route_ = Route::Tree;
    restart_tree();
}
void StructuredIfCursor::restart_tree() {
    pending_.clear();
    if (tree_.root != kNone) pending_.push_back(tree_.root);
    current_ = kNone;
    steps_ = 0;
    malformed_ = false;
    advance_tree();
}
void StructuredIfCursor::advance_tree() {
    current_ = kNone;
    while (!pending_.empty()) {
        const uint32_t id = pending_.back();
        pending_.pop_back();
        if (id >= tree_.regions.size() || ++steps_ > tree_.regions.size()) {
            malformed_ = true;
            return;
        }
        const auto& region = tree_.regions[id];
        if (region.kind == Region::Kind::Loop || region.kind == Region::Kind::CountedLoop ||
            region.kind == Region::Kind::Unstructured) {
            malformed_ = true;
            return;
        }
        for (auto it = region.children.rbegin(); it != region.children.rend(); ++it)
            if (*it != kNone) pending_.push_back(*it);
        if (region.kind == Region::Kind::IfThen || region.kind == Region::Kind::IfElse) {
            current_ = id;
            return;
        }
    }
}
bool StructuredIfCursor::read_tree_if(ForwardIf& result) const {
    if (malformed_ || current_ == kNone || current_ >= tree_.regions.size()) return false;
    const auto& r = tree_.regions[current_];
    if (r.header >= cfg_.blocks.size() || cfg_.blocks[r.header].last >= ins_.size() ||
        (r.taken != kNone && r.taken >= cfg_.blocks.size()))
        return false;
    const bool has_else = r.kind == Region::Kind::IfElse;
    if (has_else && (r.else_jump >= cfg_.blocks.size() || r.merge >= cfg_.blocks.size() ||
                     cfg_.blocks[r.else_jump].last >= ins_.size()))
        return false;
    result.found = true;
    result.branch_pc = ins_[cfg_.blocks[r.header].last].pc;
    result.early_out = r.taken == kNone;
    result.target_pc = result.early_out ? end_pc_ : cfg_.blocks[r.taken].pc;
    result.has_else = has_else;
    if (has_else) {
        result.sb_pc = ins_[cfg_.blocks[r.else_jump].last].pc;
        result.merge_pc = cfg_.blocks[r.merge].pc;
    }
    result.on_scc0 = r.condition.taken_on_zero;
    result.on_vcc = r.condition.flag == BranchCondition::Flag::Vcc;
    result.on_exec = r.condition.flag == BranchCondition::Flag::Exec;
    result.uniform_workgroup = r.condition.uniform_workgroup;
    return true;
}
uint32_t StructuredIfCursor::next_before(uint32_t hi) const {
    if (uses_tree()) {
        const uint32_t pc = next_pc();
        return pc < hi ? pc : hi;
    }
    const uint32_t next_br = (bi < Fs.size() && Fs[bi].branch_pc < hi) ? Fs[bi].branch_pc : hi;
    return next_br;
}
ForwardIf StructuredIfCursor::take() {
    if (uses_tree()) {
        ForwardIf result;
        // The complete immutable walk was validated before any SPIR-V was emitted.
        if (!read_tree_if(result)) std::abort();
        advance_tree();
        return result;
    }
    const ForwardIf F = Fs[bi++];
    return F;
}
uint32_t StructuredIfCursor::next_pc() const {
    if (uses_tree())
        return current_ == kNone ? UINT32_MAX
                                 : ins_[cfg_.blocks[tree_.regions[current_].header].last].pc;
    const uint32_t next_if = bi < Fs.size() ? Fs[bi].branch_pc : UINT32_MAX;
    return next_if;
}
} // namespace prosper::gpu::ir::cfg

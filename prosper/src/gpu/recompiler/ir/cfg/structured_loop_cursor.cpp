// structured_loop_cursor.cpp -- see structured_loop_cursor.hpp. The tree's records replace the
// detector's only after the complete sequence compares equal; every other outcome keeps them.
#include "gpu/recompiler/ir/cfg/structured_loop_cursor.hpp"

#include <cstdlib>
#include <tuple>
#include <utility>

namespace prosper::gpu::ir::cfg {

namespace {

using Kind = Region::Kind;
using Flag = BranchCondition::Flag;

// Every field of the record, so a field added to DivLoop must be added here to be compared.
bool same_loop(const DivLoop& a, const DivLoop& b) {
    return std::tie(a.header_pc, a.exit_branch_pc, a.backedge_pc, a.exit_pc, a.break_pcs,
                    a.direct_exec_breaks, a.direct_wave_breaks, a.bottom_tested, a.condition,
                    a.continue_on_set) ==
           std::tie(b.header_pc, b.exit_branch_pc, b.backedge_pc, b.exit_pc, b.break_pcs,
                    b.direct_exec_breaks, b.direct_wave_breaks, b.bottom_tested, b.condition,
                    b.continue_on_set);
}

bool same_counted(const CountedLoop& a, const CountedLoop& b) {
    return std::tie(a.found, a.header_pc, a.exit_branch_pc, a.backedge_pc, a.exit_pc,
                    a.exit_on_scc0) == std::tie(b.found, b.header_pc, b.exit_branch_pc,
                                                b.backedge_pc, b.exit_pc, b.exit_on_scc0);
}

// The regions in preorder: a parent before its children, children in emission order (Ls is in
// header order, and a nested loop's header follows its parent's). False when an id is out of
// range or the walk visits more regions than the tree holds.
bool preorder(const RegionTree& tree, std::vector<uint32_t>* out) {
    std::vector<uint32_t> pending;
    if (tree.root != kNone) pending.push_back(tree.root);
    while (!pending.empty()) {
        const uint32_t id = pending.back();
        pending.pop_back();
        if (id >= tree.regions.size() || out->size() >= tree.regions.size()) return false;
        out->push_back(id);
        const auto& children = tree.regions[id].children;
        for (auto it = children.rbegin(); it != children.rend(); ++it)
            if (*it != kNone) pending.push_back(*it);
    }
    return true;
}

// A loop region in the detectors' pc terms (region_shadow's TreeView does the same for the
// census). False when a block index or the exit flag is not one a loop can have.
class LoopReader {
public:
    LoopReader(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins) : cfg_(cfg), ins_(ins) {}

    bool read(const Region& r, DivLoop* out) const {
        if (!loop_blocks_ok(r) || r.condition.flag == Flag::None) return false;
        for (uint32_t b : r.breaks)
            if (!block_ok(b)) return false;
        DivLoop L;
        read_bounds(r, &L.header_pc, &L.exit_branch_pc, &L.backedge_pc, &L.exit_pc);
        for (uint32_t b : r.breaks) L.break_pcs.push_back(last(b).pc);
        L.condition = r.condition.flag == Flag::Exec  ? DivLoop::Condition::Exec
                      : r.condition.flag == Flag::Vcc ? DivLoop::Condition::Vcc
                                                      : DivLoop::Condition::Scc;
        L.continue_on_set = r.loop.continue_on_set;
        L.bottom_tested = r.loop.bottom_tested;
        L.direct_exec_breaks = r.loop.direct_exec_breaks;
        L.direct_wave_breaks = r.loop.direct_wave_breaks;
        *out = std::move(L);
        return true;
    }

    bool read(const Region& r, CountedLoop* out) const {
        if (!loop_blocks_ok(r) || r.condition.flag != Flag::Scc || !r.breaks.empty())
            return false;
        CountedLoop L;
        L.found = true;
        read_bounds(r, &L.header_pc, &L.exit_branch_pc, &L.backedge_pc, &L.exit_pc);
        L.exit_on_scc0 = r.loop.continue_on_set;
        *out = L;
        return true;
    }

private:
    bool block_ok(uint32_t b) const {
        return b < cfg_.blocks.size() && cfg_.blocks[b].last < ins_.size();
    }
    bool loop_blocks_ok(const Region& r) const {
        return block_ok(r.header) && block_ok(r.exit_branch) && block_ok(r.continue_block);
    }
    const Rdna2Inst& last(uint32_t b) const { return ins_[cfg_.blocks[b].last]; }
    void read_bounds(const Region& r, uint32_t* header, uint32_t* exit_branch, uint32_t* back,
                     uint32_t* exit) const {
        const Rdna2Inst& edge = last(r.continue_block);
        *header = cfg_.blocks[r.header].pc;
        *exit_branch = last(r.exit_branch).pc;
        *back = edge.pc;
        *exit = edge.pc + edge.len_dwords;
    }

    const GuestCfg& cfg_;
    const std::vector<Rdna2Inst>& ins_;
};

// The tree's loops of kind `T` in preorder, or the route that keeps the detector's records.
template <class T>
LoopRecordRoute tree_loops(const std::vector<Rdna2Inst>& ins, const RegionPolicy& policy,
                           Kind wanted, std::vector<T>* out) {
    const GuestCfg cfg = build_guest_cfg(ins, SplitPolicy{});
    const RegionTree tree = build_region_tree(cfg, ins, policy);
    if (tree.has_unstructured()) return LoopRecordRoute::Unstructured;
    std::vector<uint32_t> order;
    if (!preorder(tree, &order)) return LoopRecordRoute::Mismatch;
    const LoopReader reader(cfg, ins);
    for (uint32_t id : order) {
        const Region& r = tree.regions[id];
        if (r.kind == Kind::Unstructured) return LoopRecordRoute::Unstructured;
        if (r.kind != Kind::Loop && r.kind != Kind::CountedLoop) continue;
        T loop;
        // A loop of the other kind is a different route (the counted route against the
        // divergent detector, or the reverse): not this route's records.
        if (r.kind != wanted || !reader.read(r, &loop)) return LoopRecordRoute::Mismatch;
        out->push_back(std::move(loop));
    }
    return LoopRecordRoute::Tree;
}

}   // namespace

RegionPolicy structured_loop_policy(bool compute, bool fragment,
                                    const std::unordered_set<uint32_t>* linearized,
                                    const uint32_t* code, size_t dwords) {
    RegionPolicy policy;
    policy.allow_vcc = !compute;
    policy.compute_wave_branches = compute;
    policy.linearized = linearized;
    policy.fragment_wave_breaks = fragment;
    policy.code = code;
    policy.dwords = dwords;
    return policy;
}

StructuredLoopCursor::StructuredLoopCursor(const std::vector<Rdna2Inst>& ins,
                                           const std::vector<DivLoop>& loops,
                                           const RegionPolicy& policy, bool detector_refused)
    : detector_(loops) {
    if (detector_.empty()) return;
    if (detector_refused) {
        route_ = LoopRecordRoute::DetectorRefused;
        return;
    }
    std::vector<DivLoop> candidate;
    route_ = tree_loops(ins, policy, Kind::Loop, &candidate);
    if (route_ != LoopRecordRoute::Tree) return;
    bool equal = candidate.size() == detector_.size();
    for (size_t i = 0; equal && i < candidate.size(); ++i)
        equal = same_loop(candidate[i], detector_[i]);
    if (!equal) {
        route_ = LoopRecordRoute::Mismatch;
        return;
    }
    tree_loops_ = std::move(candidate);
}

uint32_t StructuredLoopCursor::next_before(uint32_t hi) const {
    const auto& loops = records();
    return (next_ < loops.size() && loops[next_].header_pc < hi) ? loops[next_].header_pc : hi;
}

uint32_t StructuredLoopCursor::next_pc() const {
    const auto& loops = records();
    return next_ < loops.size() ? loops[next_].header_pc : UINT32_MAX;
}

const DivLoop& StructuredLoopCursor::take() {
    const auto& loops = records();
    // emit_structured takes a loop only after next_before named one.
    if (next_ >= loops.size()) std::abort();
    return loops[next_++];
}

CountedLoopRecord counted_loop_record(const std::vector<Rdna2Inst>& ins,
                                      const CountedLoop& detector, const RegionPolicy& policy) {
    CountedLoopRecord out{detector, LoopRecordRoute::NoLoops};
    if (!detector.found) return out;
    std::vector<CountedLoop> candidate;
    out.route = tree_loops(ins, policy, Kind::CountedLoop, &candidate);
    if (out.route != LoopRecordRoute::Tree) return out;
    if (candidate.size() != 1 || !same_counted(candidate[0], detector)) {
        out.route = LoopRecordRoute::Mismatch;
        return out;
    }
    out.loop = candidate[0];
    return out;
}

}   // namespace prosper::gpu::ir::cfg

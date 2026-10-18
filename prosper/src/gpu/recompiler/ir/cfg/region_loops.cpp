// region_loops.cpp -- Loop and CountedLoop regions (IR_CONTROL_FLOW_PLAN.md step 4). See
// region_tree.hpp and AGENTS.md (loop shapes).
//
// The loops come from the CFG: every back edge must be reducible (its target dominates its
// source) and end in a backward branch. The detectors' admission rules then decide which of
// those loops the structured emitter lowers, through the shared predicates in
// rdna2_cfg_support.hpp: detect_counted_loop's when the program has exactly one counted
// candidate back edge, detect_divergent_loops' otherwise. On top of the rules, the tree checks
// in block terms that each loop is the contiguous block range [header, latch], entered only at
// its header and left only for the block after its latch.
#include "gpu/recompiler/ir/cfg/region_builder.hpp"

#include <algorithm>

namespace prosper::gpu::ir::cfg::detail {

namespace {

using Kind = Region::Kind;
using Flag = BranchCondition::Flag;

std::string at(const char* what, uint32_t pc) {
    return std::string(what) + " pc=" + std::to_string(pc);
}

Flag flag_of(DivLoop::Condition c) {
    switch (c) {
        case DivLoop::Condition::Exec: return Flag::Exec;
        case DivLoop::Condition::Vcc: return Flag::Vcc;
        case DivLoop::Condition::Scc: return Flag::Scc;
    }
    return Flag::None;
}

}   // namespace

bool RegionBuilder::find_loops() {
    const uint32_t n = static_cast<uint32_t>(cfg_.blocks.size());
    loop_at_header_.assign(n, kNone);
    loop_at_break_.assign(n, kNone);
    std::vector<uint32_t> counted, divergent;
    if (!sort_back_edges(&counted, &divergent)) return false;
    if (counted.size() == 1) {
        bool taken = false;
        if (!counted_route(counted[0], divergent, &taken)) return false;
        if (taken) return true;
    }
    return divergent_loops(divergent);
}

// Whether `header` dominates `latch`: the back edge closes a natural loop.
bool RegionBuilder::reducible(uint32_t latch, uint32_t header) const {
    for (uint32_t b = latch;; b = cfg_.idom[b]) {
        if (b == header) return true;
        if (b == 0 || cfg_.idom[b] == kNone) return false;
    }
}

// Each back edge's latch goes to the detectors whose candidate it is. A back edge that is not a
// backward branch, or whose target does not dominate its source, is irreducible. A backward
// branch that is neither detector's candidate and not linearized is refused, as both are.
bool RegionBuilder::sort_back_edges(std::vector<uint32_t>* counted,
                                    std::vector<uint32_t>* divergent) {
    const std::unordered_set<uint32_t> waterfalls = waterfall_branches(ins_);
    for (const auto& [latch, header] : cfg_.back_edges) {
        const Rdna2Inst& br = last_of(latch);
        if (cfg_.blocks[latch].succ[1] != header || br.simm16 >= 0 || !reducible(latch, header))
            return fail(at("irreducible back edge", br.pc));
        const bool as_counted = counted_loop_backedge_candidate(br, waterfalls);
        const bool as_divergent = div_loop_backedge_candidate(br, linearized_, waterfalls);
        if (as_counted) counted->push_back(latch);
        if (as_divergent) divergent->push_back(latch);
        if (!as_counted && !as_divergent && !linearized_.count(br.pc) && !waterfalls.count(br.pc))
            return fail(at("backward branch is not a loop back edge", br.pc));
    }
    return true;
}

// detect_counted_loop's route: the program's only counted candidate, when it fits a counted
// shape, claims the loop structure (emit_body tries it first). Another loop (an execnz back edge,
// which the counted rules do not count) changes that. Before the counted loop it makes
// emit_body's prelude scan refuse, and emit_body then takes the general route: so does the tree.
// After it, emit_body structures it in the counted postlude, which the tree does not model yet:
// refused.
bool RegionBuilder::counted_route(uint32_t latch, const std::vector<uint32_t>& divergent,
                                  bool* taken) {
    const CountedLoop c = counted_loop_from_backedge(ins_, last_of(latch));
    if (!c.found) return true;
    for (uint32_t other : divergent)
        if (last_of(other).pc < c.header_pc) return true;   // the general route
    for (uint32_t other : divergent)
        if (other != latch) return fail(at("loop after a counted loop", last_of(other).pc));
    counted_ = c;
    FoundLoop loop;
    loop.counted = true;
    loop.header = cfg_.block_at_pc(c.header_pc);
    loop.latch = latch;
    loop.exit_branch = block_ending_at(c.exit_branch_pc);
    loop.flag = Flag::Scc;
    loop.shape.continue_on_set = c.exit_on_scc0;
    loop.shape.bottom_tested = c.exit_branch_pc == c.backedge_pc;
    *taken = true;
    return record(loop, c.backedge_pc);
}

// detect_divergent_loops' route: the candidates, validated by the detector's own rules.
bool RegionBuilder::divergent_loops(const std::vector<uint32_t>& latches) {
    std::vector<DivLoop> list;
    std::vector<uint32_t> ops;
    for (uint32_t latch : latches) {
        DivLoop L;
        const DivLoopRefusal r = div_loop_from_backedge(ins_, last_of(latch), end_pc_, &L);
        if (r.why) return fail(at(r.why, r.pc));
        list.push_back(L);
        ops.push_back(last_of(latch).opcode);
    }
    if (list.empty()) return true;
    sort_div_loops_by_header(list, ops);
    const DivLoopRefusal r =
        validate_div_loops(ins_, linearized_, policy_.fragment_wave_breaks, list, ops);
    if (r.why) return fail(at(r.why, r.pc));
    for (const DivLoop& L : list) {
        FoundLoop loop;
        loop.header = cfg_.block_at_pc(L.header_pc);
        loop.latch = block_ending_at(L.backedge_pc);
        loop.exit_branch = block_ending_at(L.exit_branch_pc);
        for (uint32_t pc : L.break_pcs) loop.breaks.push_back(block_ending_at(pc));
        loop.flag = flag_of(L.condition);
        loop.shape = {L.continue_on_set, L.bottom_tested, L.direct_exec_breaks,
                      L.direct_wave_breaks};
        if (!record(loop, L.backedge_pc)) return false;
    }
    div_loops_ = std::move(list);
    return true;
}

bool RegionBuilder::record(const FoundLoop& loop, uint32_t backedge_pc) {
    const bool mapped =
        loop.header != kNone && loop.latch != kNone && loop.exit_branch != kNone &&
        std::find(loop.breaks.begin(), loop.breaks.end(), kNone) == loop.breaks.end();
    if (!mapped) return fail(at("loop branch is not a block terminator", backedge_pc));
    if (!single_entry_and_exit(loop)) return false;
    const auto index = static_cast<uint32_t>(loops_.size());
    loop_at_header_[loop.header] = index;
    for (uint32_t b : loop.breaks) loop_at_break_[b] = index;
    claimed_.insert(loop.latch);
    claimed_.insert(loop.exit_branch);
    latches_.insert(loop.latch);
    loops_.push_back(loop);
    return true;
}

// The loop is the block range [header, latch]: no edge enters it except at the header, and
// every edge leaving it lands on the block after the latch.
bool RegionBuilder::single_entry_and_exit(const FoundLoop& loop) {
    const uint32_t merge = loop.latch + 1;
    auto inside = [&](uint32_t b) { return b >= loop.header && b <= loop.latch; };
    for (uint32_t b = loop.header; b <= loop.latch; ++b) {
        if (!cfg_.reachable[b]) continue;
        if (b != loop.header)
            for (uint32_t p : cfg_.preds[b])
                if (!inside(p)) return fail(at("entry edge into a loop", last_of(p).pc));
        for (uint32_t s : cfg_.blocks[b].succ)
            if (s != kNone && !inside(s) && s != merge)
                return fail(at("edge leaving a loop past its merge", last_of(b).pc));
    }
    return true;
}

// The block whose last instruction has pc `pc`, or kNone.
uint32_t RegionBuilder::block_ending_at(uint32_t pc) const {
    const auto it = std::upper_bound(cfg_.blocks.begin(), cfg_.blocks.end(), pc,
                                     [](uint32_t p, const GuestBlock& b) { return p < b.pc; });
    if (it == cfg_.blocks.begin()) return kNone;
    const auto b = static_cast<uint32_t>(it - cfg_.blocks.begin() - 1);
    return last_of(b).pc == pc ? b : kNone;
}

// The loop whose header is block b, over [b, latch + 1). Its body is a Seq that starts again at
// the header, built with the header open so the header block is a plain part of it.
uint32_t RegionBuilder::loop(uint32_t b, uint32_t end, uint32_t* next) {
    const FoundLoop& found = loops_[loop_at_header_[b]];
    const uint32_t stop = found.latch + 1;
    if (stop > end) {
        fail(at("loop leaves its region", cfg_.blocks[b].pc));
        return kNone;
    }
    Region r;
    r.kind = found.counted ? Kind::CountedLoop : Kind::Loop;
    r.header = b;
    r.continue_block = found.latch;
    r.merge = stop < cfg_.blocks.size() ? stop : kNone;
    r.exit_branch = found.exit_branch;
    r.breaks = found.breaks;
    r.condition.flag = found.flag;
    r.loop = found.shape;
    r.first_block = b;
    r.end_block = stop;
    open_headers_.insert(b);
    r.children = {seq(b, stop)};
    *next = stop;
    if (!failure_.empty()) return kNone;
    return push(std::move(r));
}

// A break: a plain if whose arm is the rest of its loop's body. The branch lands on the merge;
// the lowering skips to the back edge, where the cleared lane leaves at the next exit test.
uint32_t RegionBuilder::loop_break(uint32_t b, uint32_t end, uint32_t* next, bool uniform_vcc) {
    const Rdna2Inst& br = last_of(b);
    if (uniform_vcc) {
        fail(at("uniform vcc loop break", br.pc));
        return kNone;
    }
    const FoundLoop& found = loops_[loop_at_break_[b]];
    if (end != found.latch + 1) {
        fail(at("break leaves its region", br.pc));
        return kNone;
    }
    Region r;
    r.kind = Kind::IfThen;
    r.loop_break = true;
    r.header = b;
    r.first_block = b;
    r.end_block = end;
    r.taken = found.latch;
    r.merge = found.latch;
    r.condition.flag = br.opcode == 0x06 ? Flag::Vcc : Flag::Exec;
    r.condition.taken_on_zero = true;   // only vccz and execz are breaks
    r.children = {seq(b + 1, end)};
    *next = end;
    if (!failure_.empty()) return kNone;
    return push(std::move(r));
}

}   // namespace prosper::gpu::ir::cfg::detail

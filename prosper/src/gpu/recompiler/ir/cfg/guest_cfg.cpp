// guest_cfg.cpp -- see guest_cfg.hpp and AGENTS.md (the terminator table).
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"

#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include <algorithm>
#include <set>
#include <unordered_map>

namespace prosper::gpu::ir::cfg {

namespace {

Terminator terminator_of(const Rdna2Inst& in) {
    if (in.is_end) return Terminator::End;   // the decoder's own s_endpgm flag
    switch (rdna2_instruction_effects(in).control) {
        case Rdna2ControlFlow::None: return Terminator::Fallthrough;
        case Rdna2ControlFlow::Jump: return Terminator::Branch;
        case Rdna2ControlFlow::BranchScc0: return Terminator::CondScc0;
        case Rdna2ControlFlow::BranchScc1: return Terminator::CondScc1;
        case Rdna2ControlFlow::BranchVccz: return Terminator::CondVccz;
        case Rdna2ControlFlow::BranchVccnz: return Terminator::CondVccnz;
        case Rdna2ControlFlow::BranchExecz: return Terminator::CondExecz;
        case Rdna2ControlFlow::BranchExecnz: return Terminator::CondExecnz;
        case Rdna2ControlFlow::End: return Terminator::End;
        case Rdna2ControlFlow::Trap: return Terminator::Trap;
        case Rdna2ControlFlow::Indirect:
        case Rdna2ControlFlow::IndirectCall:
        case Rdna2ControlFlow::Call:
        case Rdna2ControlFlow::DebugBranch: return Terminator::IndirectUnknown;
    }
    return Terminator::IndirectUnknown;
}

bool has_taken_edge(Terminator term) {
    return term == Terminator::Branch || terminator_is_conditional(term);
}

// Whether execution can continue at the next instruction. For an IndirectUnknown terminator that
// is the return of a call and the not-taken side of a debug branch; s_setpc and s_rfe never return.
bool has_fallthrough_edge(const Rdna2Inst& in, Terminator term) {
    if (term != Terminator::IndirectUnknown)
        return term == Terminator::Fallthrough || terminator_is_conditional(term);
    switch (rdna2_instruction_effects(in).control) {
        case Rdna2ControlFlow::IndirectCall:
        case Rdna2ControlFlow::Call:
        case Rdna2ControlFlow::DebugBranch: return true;
        default: return false;
    }
}

// SOPP branch target, in dwords: the instruction after the branch plus SIMM16.
uint32_t sopp_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

// Leader instruction indices, ascending.
std::vector<uint32_t> leaders(std::span<const Rdna2Inst> ins,
                              const std::unordered_map<uint32_t, uint32_t>& index_of_pc) {
    std::vector<bool> lead(ins.size(), false);
    lead[0] = true;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Terminator term = terminator_of(ins[i]);
        if (term == Terminator::Fallthrough) continue;
        if (i + 1 < ins.size()) lead[i + 1] = true;
        if (!has_taken_edge(term)) continue;
        const auto target = index_of_pc.find(sopp_target(ins[i]));
        if (target != index_of_pc.end()) lead[target->second] = true;
    }
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < ins.size(); ++i)
        if (lead[i]) out.push_back(i);
    return out;
}

void link_blocks(GuestCfg& cfg, std::span<const Rdna2Inst> ins) {
    const uint32_t n = static_cast<uint32_t>(cfg.blocks.size());
    for (uint32_t b = 0; b < n; ++b) {
        GuestBlock& block = cfg.blocks[b];
        const Rdna2Inst& last = ins[block.last];
        block.term = terminator_of(last);
        if (has_fallthrough_edge(last, block.term) && b + 1 < n) block.succ[0] = b + 1;
        if (!has_taken_edge(block.term)) continue;
        block.taken_pc = sopp_target(last);
        block.succ[1] = cfg.block_at_pc(block.taken_pc);
    }
}

void link_preds(GuestCfg& cfg) {
    const uint32_t n = static_cast<uint32_t>(cfg.blocks.size());
    cfg.preds.assign(n, {});
    for (uint32_t b = 0; b < n; ++b) {
        const auto& succ = cfg.blocks[b].succ;
        for (size_t k = 0; k < succ.size(); ++k) {
            if (succ[k] == kNone || (k == 1 && succ[1] == succ[0])) continue;
            cfg.preds[succ[k]].push_back(b);
        }
    }
}

// Iterative DFS from the entry: reachability, post-order and retreating edges.
void depth_first(GuestCfg& cfg) {
    const uint32_t n = static_cast<uint32_t>(cfg.blocks.size());
    enum : uint8_t { White, Grey, Black };
    std::vector<uint8_t> colour(n, White);
    std::vector<std::pair<uint32_t, uint32_t>> stack;   // (block, next successor slot)
    std::vector<uint32_t> post;
    cfg.reachable.assign(n, false);
    stack.push_back({0, 0});
    colour[0] = Grey;
    while (!stack.empty()) {
        auto& [b, slot] = stack.back();
        if (slot == cfg.blocks[b].succ.size()) {
            colour[b] = Black;
            post.push_back(b);
            stack.pop_back();
            continue;
        }
        const uint32_t s = cfg.blocks[b].succ[slot++];
        if (s == kNone) continue;
        if (colour[s] == Grey) {
            cfg.back_edges.push_back({b, s});
        } else if (colour[s] == White) {
            colour[s] = Grey;
            stack.push_back({s, 0});
        }
    }
    for (uint32_t b = 0; b < n; ++b) cfg.reachable[b] = colour[b] == Black;
    cfg.rpo.assign(post.rbegin(), post.rend());
}

// Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm".
void dominators(GuestCfg& cfg) {
    const uint32_t n = static_cast<uint32_t>(cfg.blocks.size());
    std::vector<uint32_t> order(n, kNone);   // rpo position
    for (uint32_t i = 0; i < cfg.rpo.size(); ++i) order[cfg.rpo[i]] = i;
    cfg.idom.assign(n, kNone);
    cfg.idom[0] = 0;
    auto intersect = [&](uint32_t a, uint32_t b) {
        while (a != b) {
            while (order[a] > order[b]) a = cfg.idom[a];
            while (order[b] > order[a]) b = cfg.idom[b];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (uint32_t b : cfg.rpo) {
            if (b == 0) continue;
            uint32_t next = kNone;
            for (uint32_t p : cfg.preds[b]) {
                if (cfg.idom[p] == kNone) continue;
                next = next == kNone ? p : intersect(p, next);
            }
            if (next != cfg.idom[b]) {
                cfg.idom[b] = next;
                changed = true;
            }
        }
    }
}

// --- The dispatcher's split (SplitPolicy::split_at_services) ---------------------------------

bool dispatch_linearized(const SplitPolicy& policy, const Rdna2Inst& in) {
    return policy.linearized && in.fmt == Rdna2Format::SOPP && policy.linearized->contains(in.pc);
}

bool is_service(const SplitPolicy& policy, const Rdna2Inst& in) {
    return policy.services && policy.services->contains(in.pc);
}

// emit_cfg_state_machine's `cfg_terminator`: s_endpgm, then s_trap and the branches it does not
// linearize. Every other instruction, the indirect and call forms included, stays inside its case.
Terminator dispatch_terminator_of(const SplitPolicy& policy, const Rdna2Inst& in) {
    if (in.is_end) return Terminator::End;
    if (dispatch_linearized(policy, in)) return Terminator::Fallthrough;
    const Terminator term = terminator_of(in);
    return term == Terminator::Trap || has_taken_edge(term) ? term : Terminator::Fallthrough;
}

// Leader pcs over ins[0..end], where ins[end] is the first s_endpgm. A branch target is kept only
// when it is the pc of a decoded instruction; the dispatcher also starts a block at a target inside
// an instruction, which this graph cannot express (AGENTS.md).
std::set<uint32_t> dispatch_leaders(std::span<const Rdna2Inst> ins, size_t end,
                                    const SplitPolicy& policy) {
    const uint32_t end_pc = ins[end].pc;
    std::set<uint32_t> out{ins.front().pc, end_pc};
    auto split_around = [&](size_t i) {
        out.insert(ins[i].pc);
        if (ins[i + 1].pc <= end_pc) out.insert(ins[i + 1].pc);
    };
    for (size_t i = 0; i < end; ++i) {
        const Rdna2Inst& in = ins[i];
        if (is_service(policy, in)) split_around(i);
        const Terminator term = terminator_of(in);
        if (term == Terminator::Trap) {
            split_around(i);
            continue;
        }
        if (dispatch_linearized(policy, in) || !has_taken_edge(term)) continue;
        if (sopp_target(in) <= end_pc) out.insert(sopp_target(in));
        if (ins[i + 1].pc <= end_pc) out.insert(ins[i + 1].pc);
    }
    return out;
}

GuestCfg build_dispatch_cfg(std::span<const Rdna2Inst> ins, const SplitPolicy& policy) {
    GuestCfg cfg;
    size_t end = 0;
    while (end < ins.size() && !ins[end].is_end) ++end;
    if (end == ins.size()) return cfg;   // the dispatcher refuses a region without s_endpgm
    std::unordered_map<uint32_t, uint32_t> index_of_pc;
    for (uint32_t i = 0; i <= end; ++i) index_of_pc.emplace(ins[i].pc, i);
    std::vector<uint32_t> starts;
    for (uint32_t pc : dispatch_leaders(ins, end, policy))
        if (const auto it = index_of_pc.find(pc); it != index_of_pc.end())
            starts.push_back(it->second);
    for (size_t k = 0; k < starts.size(); ++k) {
        GuestBlock block;
        block.first = starts[k];
        block.last = k + 1 < starts.size() ? starts[k + 1] - 1 : static_cast<uint32_t>(end);
        block.pc = ins[block.first].pc;
        cfg.blocks.push_back(block);
    }
    const uint32_t n = static_cast<uint32_t>(cfg.blocks.size());
    for (uint32_t b = 0; b < n; ++b) {
        GuestBlock& block = cfg.blocks[b];
        const Rdna2Inst& last = ins[block.last];
        block.term = dispatch_terminator_of(policy, last);
        if (is_service(policy, last)) block.service_event = last.pc;
        if (block.term == Terminator::Fallthrough || terminator_is_conditional(block.term))
            block.succ[0] = b + 1;   // the last block is s_endpgm, so b + 1 exists
        if (!has_taken_edge(block.term)) continue;
        block.taken_pc = sopp_target(last);
        if (block.taken_pc <= ins[end].pc) block.succ[1] = cfg.block_at_pc(block.taken_pc);
    }
    return cfg;
}

}   // namespace

bool terminator_is_conditional(Terminator term) {
    switch (term) {
        case Terminator::CondScc0:
        case Terminator::CondScc1:
        case Terminator::CondVccz:
        case Terminator::CondVccnz:
        case Terminator::CondExecz:
        case Terminator::CondExecnz: return true;
        default: return false;
    }
}

uint32_t GuestCfg::block_at_pc(uint32_t pc) const {
    const auto it = std::lower_bound(blocks.begin(), blocks.end(), pc,
                                     [](const GuestBlock& b, uint32_t p) { return b.pc < p; });
    if (it == blocks.end() || it->pc != pc) return kNone;
    return static_cast<uint32_t>(it - blocks.begin());
}

GuestCfg build_guest_cfg(std::span<const Rdna2Inst> ins, const SplitPolicy& policy) {
    GuestCfg cfg;
    if (ins.empty()) return cfg;
    if (policy.split_at_services) {
        cfg = build_dispatch_cfg(ins, policy);
        if (cfg.blocks.empty()) return cfg;
        link_preds(cfg);
        depth_first(cfg);
        dominators(cfg);
        return cfg;
    }
    std::unordered_map<uint32_t, uint32_t> index_of_pc;
    for (uint32_t i = 0; i < ins.size(); ++i) index_of_pc.emplace(ins[i].pc, i);
    const std::vector<uint32_t> starts = leaders(ins, index_of_pc);
    for (size_t k = 0; k < starts.size(); ++k) {
        GuestBlock block;
        block.first = starts[k];
        block.last =
            k + 1 < starts.size() ? starts[k + 1] - 1 : static_cast<uint32_t>(ins.size() - 1);
        block.pc = ins[block.first].pc;
        cfg.blocks.push_back(block);
    }
    link_blocks(cfg, ins);
    link_preds(cfg);
    depth_first(cfg);
    dominators(cfg);
    return cfg;
}

}   // namespace prosper::gpu::ir::cfg

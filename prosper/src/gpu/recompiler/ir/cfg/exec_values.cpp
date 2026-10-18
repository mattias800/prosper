// exec_values.cpp -- EXEC value identities and their static relations; see exec_values.hpp and
// the "EXEC values" section of AGENTS.md.
#include "gpu/recompiler/ir/cfg/exec_values.hpp"

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu::ir::cfg {

int exec_mask_location(const Operand& operand) {
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) return -1;
    if (operand.value == 126 || operand.value == 127) return kExecLocation;
    if (operand.value == 106 || operand.value == 107) return kVccLocation;
    if (operand.kind == OperandKind::SGPR && operand.value >= 0 && operand.value <= 105)
        return operand.value;
    return -1;
}

const char* exec_value_kind_name(ExecValue::Kind kind) {
    switch (kind) {
        case ExecValue::Kind::Entry: return "entry";
        case ExecValue::Kind::Write: return "write";
        case ExecValue::Kind::Phi: return "phi";
        case ExecValue::Kind::DispatchReload: return "reload";
    }
    return "?";
}

uint32_t ExecValues::representative(uint32_t value) const {
    for (int hop = 0; hop < 64 && value < values.size() && values[value].same_as != kNone; ++hop)
        value = values[value].same_as;
    return value;
}

bool ExecValues::within(uint32_t reader, uint32_t owner, std::vector<uint32_t>* path) const {
    if (reader >= values.size() || owner >= values.size()) return false;
    uint32_t current = representative(reader);
    const uint32_t target = representative(owner);
    for (int hop = 0; hop < 256; ++hop) {
        if (path) path->push_back(current);
        if (current == target) return true;
        const uint32_t parent = current < subset_of.size() ? subset_of[current] : kNone;
        if (parent == kNone) return false;
        current = representative(parent);
    }
    return false;
}

ExecComplement ExecValues::complement(uint32_t value) const {
    value = representative(value);
    return value < complement_in.size() ? complement_in[value] : ExecComplement{};
}

namespace {

using Kind = ExecValue::Kind;
using MaskState = std::map<int, uint32_t>;   // mask location -> value

// The SOP2 64-bit bitwise operations the emitter lowers as lane masks (sop2_b64_routes.cpp).
bool sop2_mask_logical(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP2) return false;
    switch (in.opcode) {
        case 0x0f:
        case 0x11:
        case 0x13:
        case 0x15:
        case 0x17:
        case 0x19:
        case 0x1b:
        case 0x1d: return true;
        default: return false;
    }
}

bool sop1_saveexec(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 &&
           (sop1_opcode_is_saveexec_b64(in.opcode) || sop1_opcode_is_saveexec_b32(in.opcode));
}

// s_mov_b64 and s_wqm_b64: the emitter moves the mask's SPIR-V id. CONFIDENCE: MED for s_wqm_b64:
// architecturally it widens the mask to whole quads, and it is a copy only in the emitter's
// per-invocation model, where helper lanes are implicit (sop1_b64_mask_routes.cpp). This pass
// states the emitter's model, which is what the relations it replaces are about.
bool sop1_mask_copy(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 &&
           (in.opcode == kSop1OpcodeMovB64 || in.opcode == kSop1OpcodeWqmB64);
}

// A write to [base, base + width) ends every mask held in an overlapping pair. EXEC is not
// cleared here: its writers define a new value.
void clobber(MaskState& state, int base, uint32_t width) {
    const int end = base + static_cast<int>(width);
    for (auto it = state.begin(); it != state.end();) {
        if (it->first != kExecLocation && it->first < end && it->first + 2 > base)
            it = state.erase(it);
        else
            ++it;
    }
}

uint32_t lookup(const MaskState& state, int location) {
    const auto found = location < 0 ? state.end() : state.find(location);
    return found == state.end() ? kNone : found->second;
}

// The two mask operands a relation reads, from the solve's final pass.
struct Operands {
    uint32_t left = kNone, right = kNone;
};

class Builder {
public:
    Builder(const GuestCfg& cfg, std::span<const Rdna2Inst> ins, const ExecValueOptions& options)
        : cfg_(cfg), ins_(ins), options_(options) {}
    ExecValues run();

private:
    uint32_t add(ExecValue value);
    uint32_t write_value(uint32_t block, uint32_t inst, int location);
    void set_same_as(uint32_t value, uint32_t source);
    uint32_t rep(uint32_t value) const { return out_.representative(value); }
    uint32_t pred_exec(uint32_t pred) const { return lookup(block_out_[pred], kExecLocation); }
    bool forward_edge(uint32_t pred, uint32_t block) const {
        return rpo_index_[pred] < rpo_index_[block];
    }
    std::vector<uint32_t> exec_incoming(uint32_t block, bool forward_only) const;
    MaskState join(uint32_t block);
    void transfer(uint32_t block, uint32_t inst, MaskState& state);
    bool solve_pass();
    void subset(uint32_t value, uint32_t parent);
    void record_mask_subset(uint32_t result, uint32_t left, uint32_t right, uint32_t opcode,
                            uint32_t previous);
    uint32_t defined_at(uint32_t inst, int location) const;
    void relate_phi(uint32_t block);
    void relate_inst(uint32_t inst);
    bool relate_pass();

    const GuestCfg& cfg_;
    std::span<const Rdna2Inst> ins_;
    ExecValueOptions options_;
    ExecValues out_;
    uint32_t entry_ = kNone;
    bool changed_ = false;
    std::vector<uint32_t> phi_, reload_, rpo_index_;
    std::map<std::pair<uint32_t, int>, uint32_t> writes_;
    std::vector<MaskState> block_out_;
    std::vector<bool> has_out_;
    std::vector<Operands> operands_;
    std::set<uint32_t> refused_;   // loop headers whose induction failed
    std::vector<std::pair<uint32_t, uint32_t>> assumed_;   // (header block, parent)
};

uint32_t Builder::add(ExecValue value) {
    out_.values.push_back(value);
    changed_ = true;
    return static_cast<uint32_t>(out_.values.size() - 1);
}

uint32_t Builder::write_value(uint32_t block, uint32_t inst, int location) {
    const auto key = std::make_pair(inst, location);
    const auto found = writes_.find(key);
    if (found != writes_.end()) return found->second;
    const uint32_t value = add({Kind::Write, block, inst, location, kNone});
    writes_.emplace(key, value);
    return value;
}

void Builder::set_same_as(uint32_t value, uint32_t source) {
    // A copy of itself (s_wqm_b64 exec, exec through a loop) names no other value.
    if (source != kNone && rep(source) == value) source = kNone;
    if (out_.values[value].same_as == source) return;
    out_.values[value].same_as = source;
    changed_ = true;
}

std::vector<uint32_t> Builder::exec_incoming(uint32_t block, bool forward_only) const {
    std::vector<uint32_t> incoming;
    if (block == 0) incoming.push_back(entry_);
    for (uint32_t pred : cfg_.preds[block]) {
        if (!cfg_.reachable[pred] || !has_out_[pred]) continue;
        if (forward_only && !forward_edge(pred, block)) continue;
        incoming.push_back(pred_exec(pred));
    }
    return incoming;
}

// EXEC: the one reaching value, or the block's phi once two differ. Another mask location
// survives the join only when every known predecessor holds the same value there.
MaskState Builder::join(uint32_t block) {
    MaskState state;
    const auto incoming = exec_incoming(block, false);
    uint32_t exec = phi_[block];
    if (exec == kNone) {
        exec = incoming.empty() ? entry_ : rep(incoming.front());
        for (uint32_t value : incoming) {
            if (rep(value) == exec) continue;
            exec = phi_[block] = add({Kind::Phi, block, kNone, kExecLocation, kNone});
            break;
        }
    }
    if (block != 0) {
        bool first = true;
        for (uint32_t pred : cfg_.preds[block]) {
            if (!cfg_.reachable[pred] || !has_out_[pred]) continue;
            const MaskState& out = block_out_[pred];
            if (first) {
                state = out;
                first = false;
                continue;
            }
            for (auto it = state.begin(); it != state.end();) {
                const uint32_t other = lookup(out, it->first);
                it = other != kNone && rep(other) == rep(it->second) ? std::next(it)
                                                                     : state.erase(it);
            }
        }
    }
    if (options_.dispatch_reloads) {
        if (reload_[block] == kNone)
            reload_[block] = add({Kind::DispatchReload, block, kNone, kExecLocation, kNone});
        set_same_as(reload_[block], exec);
        exec = reload_[block];
    }
    state[kExecLocation] = exec;
    return state;
}

void Builder::transfer(uint32_t block, uint32_t inst, MaskState& state) {
    const Rdna2Inst& in = ins_[inst];
    const uint32_t before = lookup(state, kExecLocation);
    out_.exec_before[inst] = before;
    auto& defined = out_.defined[inst];
    defined.clear();
    Operands operands;
    const auto value_of = [&](const Operand& o) { return lookup(state, exec_mask_location(o)); };
    const auto define = [&](int location) {
        if (location != kExecLocation) clobber(state, location, 2);
        const uint32_t value = write_value(block, inst, location);
        state[location] = value;
        defined.emplace_back(location, value);
        return value;
    };
    const int dst = exec_mask_location(in.dst);
    if (in.fmt == Rdna2Format::VOPC) {
        const bool sgpr = in.dst.kind == OperandKind::SGPR && in.dst.value <= 105;
        define(vopc_is_cmpx(in.opcode) ? kExecLocation : sgpr ? in.dst.value : kVccLocation);
    } else if (sop2_mask_logical(in) && dst >= 0) {
        operands = {value_of(in.src[0]), value_of(in.src[1])};
        define(dst);
    } else if (sop1_saveexec(in)) {
        operands.left = value_of(in.src[0]);
        if (dst >= 0 && dst != kExecLocation) {   // D = old EXEC: the same value
            clobber(state, dst, 2);
            if (before != kNone) state[dst] = before;
        }
        define(kExecLocation);
    } else if (sop1_mask_copy(in) && dst >= 0) {
        const uint32_t source = value_of(in.src[0]);
        const int from = exec_mask_location(in.src[0]);
        if (dst == kExecLocation) {
            const uint32_t value = define(kExecLocation);
            set_same_as(value, source);
            // CONFIDENCE: HIGH -- S_MOV_B64 copies the pair (ISA 70648 12.3), so after it EXEC and
            // the source pair hold the same lanes. A source holding no known mask names this
            // value from now on (the emitter's id for both is the source's).
            if (source == kNone && from >= 0 && from != kExecLocation) state[from] = value;
        } else {
            clobber(state, dst, 2);
            if (source != kNone) state[dst] = source;
        }
    } else {
        bool exec_written = rdna2_instruction_may_change_exec(in);
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            clobber(state, base, width);
            exec_written |= base <= 127 && base + static_cast<int>(width) > 126;
        });
        if (exec_written) define(kExecLocation);
    }
    out_.exec_after[inst] = lookup(state, kExecLocation);
    operands_[inst] = operands;
}

bool Builder::solve_pass() {
    changed_ = false;
    for (uint32_t block : cfg_.rpo) {
        MaskState state = join(block);
        const GuestBlock& b = cfg_.blocks[block];
        for (uint32_t inst = b.first; inst <= b.last && inst < ins_.size(); ++inst)
            transfer(block, inst, state);
        if (!has_out_[block] || block_out_[block] != state) {
            block_out_[block] = std::move(state);
            has_out_[block] = true;
            changed_ = true;
        }
    }
    return changed_;
}

void Builder::subset(uint32_t value, uint32_t parent) {
    if (value == kNone || parent == kNone) return;
    value = rep(value);
    if (rep(parent) != value) out_.subset_of[value] = parent;
}

// record_exec_mask_subset (rdna2_vector_word_marks.cpp), over static values: A&B is within the
// operand that was the previous EXEC (else the left one), A&~B within A, and A|B within the wider
// operand when one is proved within the other. A&~B with B within A is A's complement of B.
void Builder::record_mask_subset(uint32_t result, uint32_t left, uint32_t right, uint32_t opcode,
                                 uint32_t previous) {
    uint32_t parent = kNone;
    if (opcode == 0x0f)
        parent = right != kNone && previous != kNone && rep(right) == rep(previous) ? right : left;
    else if (opcode == 0x15)
        parent = left;
    else if (opcode == 0x11)
        parent = out_.within(left, right) ? right : out_.within(right, left) ? left : kNone;
    subset(result, parent);
    if (opcode != 0x15 || left == kNone || right == kNone) return;
    const uint32_t r = rep(result), a = rep(left), b = rep(right);
    if (r != a && r != b && a != b && out_.within(right, left))
        out_.complement_in[r] = {left, right};
}

uint32_t Builder::defined_at(uint32_t inst, int location) const {
    for (const auto& [where, value] : out_.defined[inst])
        if (where == location) return value;
    return kNone;
}

// merge_exec_arm's rule at every join: the phi is within an incoming value every other incoming
// value is within. CONFIDENCE: HIGH -- each lane takes one incoming value. The emitter applies it
// at one-armed if merges only; here it holds at every join, which the census counts as
// static-stronger. At a loop header the back edges are not known yet, so the phi is assumed
// within its forward value and the assumption is checked at the back edges once the body's
// relations exist (LoopExecSubset's induction); a failed check refuses that header.
// CONFIDENCE: MED for loops -- the induction is sound for the guest wave's EXEC at every loop. The
// emitter grants it only to admitted loops (the compute certificate and the checked fragment
// loops), because a compute SPIR-V loop is left per invocation, so its header value is not the
// guest wave's. A consumer (step 7b) must keep that admission; this pass does not model it.
void Builder::relate_phi(uint32_t block) {
    const uint32_t phi = phi_[block];
    const auto forward = exec_incoming(block, true);
    uint32_t parent = kNone;
    for (uint32_t candidate : forward) {
        bool all = true;
        for (uint32_t value : forward) all = all && out_.within(value, candidate);
        if (all) {
            parent = candidate;
            break;
        }
    }
    if (parent == kNone) return;
    const bool loop_header = forward.size() != exec_incoming(block, false).size();
    if (loop_header) {
        if (refused_.contains(block)) return;
        assumed_.emplace_back(block, parent);
    }
    subset(phi, parent);
}

void Builder::relate_inst(uint32_t inst) {
    const Rdna2Inst& in = ins_[inst];
    const uint32_t before = out_.exec_before[inst];
    const Operands& ops = operands_[inst];
    if (in.fmt == Rdna2Format::VOPC) {
        // CONFIDENCE: HIGH -- V_CMPX: EXEC' = EXEC & cmp. Any other compare writes EXEC & cmp
        // (ISA 70648 3.9), whether or not EXEC is narrowed; the emitter records it only when it is.
        if (!out_.defined[inst].empty()) subset(out_.defined[inst].front().second, before);
        return;
    }
    const uint32_t exec = defined_at(inst, kExecLocation);
    if (exec == kNone) return;
    if (sop2_mask_logical(in) && exec_mask_location(in.dst) == kExecLocation) {
        if (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x15)
            record_mask_subset(exec, ops.left, ops.right, in.opcode, before);
        return;
    }
    if (in.fmt != Rdna2Format::SOP1 || !sop1_opcode_is_saveexec_b64(in.opcode)) return;
    switch (in.opcode) {   // the emitter's mapping onto record_exec_mask_subset
        case 0x24: record_mask_subset(exec, before, ops.left, 0x0f, before); break;
        case 0x25: record_mask_subset(exec, before, ops.left, 0x11, before); break;
        case 0x27: record_mask_subset(exec, ops.left, before, 0x15, before); break;
        case 0x37: record_mask_subset(exec, before, ops.left, 0x15, before); break;
        default: break;
    }
}

bool Builder::relate_pass() {
    std::fill(out_.subset_of.begin(), out_.subset_of.end(), kNone);
    std::fill(out_.complement_in.begin(), out_.complement_in.end(), ExecComplement{});
    assumed_.clear();
    for (uint32_t block : cfg_.rpo) {
        if (phi_[block] != kNone) relate_phi(block);
        const GuestBlock& b = cfg_.blocks[block];
        for (uint32_t inst = b.first; inst <= b.last && inst < ins_.size(); ++inst)
            relate_inst(inst);
    }
    bool held = true;
    for (const auto& [block, parent] : assumed_) {
        for (uint32_t pred : cfg_.preds[block]) {
            if (!cfg_.reachable[pred] || forward_edge(pred, block)) continue;
            if (out_.within(pred_exec(pred), parent)) continue;
            refused_.insert(block);
            held = false;
            break;
        }
    }
    return held;
}

ExecValues Builder::run() {
    const size_t blocks = cfg_.blocks.size();
    out_.exec_before.assign(ins_.size(), kNone);
    out_.exec_after.assign(ins_.size(), kNone);
    out_.defined.assign(ins_.size(), {});
    operands_.assign(ins_.size(), {});
    phi_.assign(blocks, kNone);
    reload_.assign(blocks, kNone);
    rpo_index_.assign(blocks, kNone);
    block_out_.assign(blocks, {});
    has_out_.assign(blocks, false);
    for (size_t i = 0; i < cfg_.rpo.size(); ++i) rpo_index_[cfg_.rpo[i]] = static_cast<uint32_t>(i);
    if (blocks == 0) return std::move(out_);
    entry_ = add({Kind::Entry, 0, kNone, kExecLocation, kNone});
    // Each pass can only add a phi or a reload link, or drop a mask location, so this converges;
    // the bound is a backstop.
    for (size_t pass = 0; pass < 4 * blocks + 8 && solve_pass(); ++pass) {}
    out_.subset_of.assign(out_.values.size(), kNone);
    out_.complement_in.assign(out_.values.size(), {});
    // Each failed round refuses one more loop header.
    for (size_t round = 0; round <= blocks && !relate_pass(); ++round) {}
    return std::move(out_);
}

}   // namespace

ExecValues build_exec_values(const GuestCfg& cfg, std::span<const Rdna2Inst> ins,
                             const ExecValueOptions& options) {
    return Builder(cfg, ins, options).run();
}

}   // namespace prosper::gpu::ir::cfg

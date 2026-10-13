#include "gpu/recompiler/rdna2_complementary_exec_data.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"

#include <algorithm>

namespace prosper::gpu {
namespace {
using Words = std::set<int>;

Words intersection(Words a, const Words& b) {
    std::erase_if(a, [&](int word) { return !b.contains(word); });
    return a;
}

Words united(Words a, const Words& b) {
    a.insert(b.begin(), b.end());
    return a;
}

bool pair_operand(const Operand& operand) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value >= 0 && operand.value < 126;
}

bool exec_operand(const Operand& operand) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == 126;
}

struct Domain {
    bool known = false;
    Words parents;
};

Domain domain(const ComplementaryExecData& state, const Operand& operand) {
    if (exec_operand(operand)) return {true, state.current_parents};
    if (!pair_operand(operand)) return {};
    const auto found = state.parents.find(operand.value);
    if (found == state.parents.end()) return {};
    auto parents = found->second;
    parents.insert(operand.value); // a current pair value is a subset of itself
    return {true, std::move(parents)};
}

Domain logical_domain(uint32_t op, const Domain& a, const Domain& b) {
    if (!a.known || !b.known) return {};
    switch (op) {
        case 0x0f: return {true, united(a.parents, b.parents)}; // A&B is within either
        case 0x15: return a; // A&~B is within A, with ordered operands
        case 0x11:
        case 0x13: return {true, intersection(a.parents, b.parents)};   // OR/XOR may enable either
        default: return {};
    }
}

bool overlap(int mask, int base, uint32_t width) {
    return mask <= base + static_cast<int>(width) - 1 && mask + 1 >= base;
}

void retire_pair(ComplementaryExecData& state, int base, uint32_t width) {
    const auto affected = [&](int mask) { return overlap(mask, base, width); };
    std::erase_if(state.current_parents, affected);
    for (auto mask = state.parents.begin(); mask != state.parents.end();) {
        if (affected(mask->first))
            mask = state.parents.erase(mask);
        else {
            std::erase_if(mask->second, affected);
            ++mask;
        }
    }
    if (state.parent >= 0 && affected(state.parent)) {
        state.parent = -1;
        state.first_words.clear();
    }
}

void publish_parent_words(ActiveVectorWords& state, const Words& words) {
    const int parent = state.complement.parent;
    if (parent < 0 || !state.complement.parents.contains(parent)) return;
    auto& parent_words = state.masks[parent];
    parent_words.insert(words.begin(), words.end());
    for (const auto& [mask, parents] : state.complement.parents)
        if (parents.contains(parent)) state.masks[mask].insert(words.begin(), words.end());
    if (state.entry_masks.contains(parent)) {
        state.entry_words.insert(words.begin(), words.end());
        for (int mask : state.within_entry_masks)
            state.masks[mask].insert(words.begin(), words.end());
    }
}
}   // namespace

void meet_complementary_exec_data(ComplementaryExecData& into,
                                  const ComplementaryExecData& incoming) {
    into.current_parents = intersection(std::move(into.current_parents), incoming.current_parents);
    for (auto mask = into.parents.begin(); mask != into.parents.end();) {
        const auto other = incoming.parents.find(mask->first);
        if (other == incoming.parents.end())
            mask = into.parents.erase(mask);
        else {
            mask->second = intersection(std::move(mask->second), other->second);
            ++mask;
        }
    }
    if (into.parent != incoming.parent) {
        into.parent = -1;
        into.first_words.clear();
    } else {
        into.first_words = intersection(std::move(into.first_words), incoming.first_words);
    }
}

void advance_complementary_exec_data(ActiveVectorWords& state, const Rdna2Inst& in,
                                     const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                     int mask_write) {
    // CONFIDENCE: HIGH -- set identities refer to the actual current architectural values.
    // A partial/aliasing write expires the old pair and every relation that referred to it.
    auto& proof = state.complement;
    const Domain before{true, proof.current_parents};
    const Domain left = in.n_src > 0 ? domain(proof, in.src[0]) : Domain{};
    const Domain right = in.n_src > 1 ? domain(proof, in.src[1]) : Domain{};
    const bool copy =
        in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 && in.n_src == 1;
    const bool saveexec = in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x24 && in.opcode <= 0x2b;
    const bool compare =
        in.fmt == Rdna2Format::VOPC || (in.fmt == Rdna2Format::VOP3 && mask_write >= 0);
    Domain result;
    if (copy)
        result = left;
    else if (in.fmt == Rdna2Format::SOP2 && in.n_src == 2)
        result = logical_domain(in.opcode, left, right);

    int parent = -1;
    const bool exact_complement =
        (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x15 && in.dst.value == 126 && in.n_src == 2 &&
         exec_operand(in.src[1])) ||
        (saveexec && in.opcode == 0x27);
    if (exact_complement && in.n_src > 0 && pair_operand(in.src[0]) && left.known &&
        before.parents.contains(in.src[0].value))
        parent = in.src[0].value;
    const Words first_words = parent >= 0 ? state.current : Words{};
    const bool current_within_left = in.n_src > 0 && pair_operand(in.src[0]) && left.known &&
                                     before.parents.contains(in.src[0].value);

    for (const auto& [base, width] : scalar_writes) {
        retire_pair(proof, base, width);
        if (parent >= 0 && overlap(parent, base, width)) parent = -1;
    }
    if (mask_write >= 0) {
        retire_pair(proof, mask_write, 2);
        if (parent >= 0 && overlap(parent, mask_write, 2)) parent = -1;
    }
    // Snapshotted supersets are usable only while their architectural pair survives this write.
    const auto live = [&](Words words) {
        std::erase_if(words, [&](int mask) { return !proof.parents.contains(mask); });
        return words;
    };
    const auto changes_exec = rdna2_instruction_may_change_exec(in);
    if (changes_exec) {
        proof.parent = -1;
        proof.first_words.clear();
        Domain next;
        if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode))
            next = before;
        else if (saveexec) {
            switch (in.opcode) {
                case 0x24: next = logical_domain(0x0f, left, before); break;
                case 0x25: next = logical_domain(0x11, left, before); break;
                case 0x26: next = logical_domain(0x13, left, before); break;
                case 0x27: next = logical_domain(0x15, left, before); break;
                default: break;
            }
        } else if (in.dst.value == 126)
            next = result;
        proof.current_parents = live(std::move(next.parents));
    }
    if (saveexec && in.dst.value < 126) {
        proof.parents[in.dst.value] = live(before.parents);
        if (in.opcode == 0x24) proof.current_parents.insert(in.dst.value);
    } else if (mask_write >= 0 && mask_write < 126 && compare) {
        proof.parents[mask_write] = live(before.parents);
        if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode))
            proof.current_parents.insert(mask_write);
    } else if (mask_write >= 0 && mask_write < 126 && result.known) {
        proof.parents[mask_write] = live(result.parents);
    }
    if (copy && in.dst.value < 126 && result.known) {
        proof.parents[in.dst.value] = live(result.parents);
        if (exec_operand(in.src[0]) || current_within_left)
            proof.current_parents.insert(in.dst.value);
    }
    if (parent >= 0 && proof.parents.contains(parent)) {
        proof.parent = parent;
        proof.first_words = first_words;
    }
}

void publish_complementary_exec_word(ActiveVectorWords& state, int word) {
    if (state.complement.first_words.contains(word)) publish_parent_words(state, {word});
}

SpilledMaskHalves active_vector_edge_state(const SpilledMaskHalves& outgoing,
                                           std::span<const Rdna2Inst> instructions, uint32_t lo,
                                           uint32_t hi, uint32_t next,
                                           const std::set<int>& vector_words) {
    auto edge = outgoing;
    // Which SGPR holds a reloaded half is block-local: emit_alu's per-lane mask for a
    // v_readlane'd half does not survive a dispatcher edge, so a move into VCC after the
    // edge would copy data words while the analysis called it a mask (#4603 review). The
    // spill-slot and pair-definition facts describe memory and stay valid across edges.
    edge.sregs.clear();
    if (!edge.active_vectors.enabled) return edge;
    for (const auto& in : instructions) {
        if (in.pc < lo) continue;
        if (in.pc >= hi) break;
        if (in.fmt != Rdna2Format::SOPP ||
            (in.opcode != kSoppOpcodeCbranchExecz && in.opcode != kSoppOpcodeCbranchExecnz))
            continue;
        const uint32_t fallthrough = in.pc + in.len_dwords;
        const uint32_t target = branch_target(in);
        // A zero-displacement branch has one successor containing BOTH condition outcomes.
        if (target == fallthrough) return edge;
        const bool empty =
            in.opcode == kSoppOpcodeCbranchExecz ? next == target : next == fallthrough;
        if (!empty) return edge;
        // CONFIDENCE: HIGH -- no active lane can read missing DATA when actual EXEC is zero.
        // This says nothing about SGPRs, another saved mask, a wave vote or memory authority.
        auto& state = edge.active_vectors;
        state.current.insert(vector_words.begin(), vector_words.end());
        state.current_is_entry = false;
        publish_parent_words(state, state.complement.first_words);
        return edge;
    }
    return edge;
}
}   // namespace prosper::gpu

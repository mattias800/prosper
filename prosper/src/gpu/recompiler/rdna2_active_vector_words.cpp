#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace prosper::gpu {
namespace {
using Words = std::set<int>;

Words intersect(Words left, const Words& right) {
    std::erase_if(left, [&](int word) { return !right.contains(word); });
    return left;
}

Words combine(Words left, const Words& right) {
    left.insert(right.begin(), right.end());
    return left;
}

Words mask_words(const ActiveVectorWords& state, const Operand& operand) {
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) return {};
    if (operand.value == 126) return state.current;
    const auto found = state.masks.find(operand.value);
    return found == state.masks.end() ? Words{} : found->second;
}

struct EntryDomain {
    bool equal = false;
    bool within = false;
};

EntryDomain entry_domain(const ActiveVectorWords& state, const Operand& operand) {
    if (operand.kind == OperandKind::InlineInt && operand.value == 0) return {false, true};
    if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) return {};
    if (operand.value == 126) return {state.current_is_entry, state.current_within_entry};
    return {state.entry_masks.contains(operand.value),
            state.within_entry_masks.contains(operand.value)};
}

EntryDomain logical_domain(uint32_t op, EntryDomain a, EntryDomain b) {
    switch (op) {
        case 0x0f: return {a.equal && b.equal, a.within || b.within}; // AND
        case 0x15: return {false, a.within}; // ANDN2
        case 0x11: return {(a.equal && b.within) || (b.equal && a.within), a.within && b.within};
        case 0x13: return {false, a.within && b.within}; // XOR
        default: return {};
    }
}

void publish_entry_domain(ActiveVectorWords& state, int mask, EntryDomain domain) {
    if (domain.equal) state.entry_masks.insert(mask);
    if (domain.within) {
        state.within_entry_masks.insert(mask);
        auto& words = state.masks[mask];
        words.insert(state.entry_words.begin(), state.entry_words.end());
    }
}

bool mask_operand(const Operand& operand) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value >= 0 && operand.value <= 126;
}
} // namespace

bool tracks_active_vector_words(const SpirvCompute& b, bool barrier_half_phase) {
    return b.is_compute && !b.is_fragment && b.wave_size == 64 && !b.ngg_workgroup_shell &&
           !b.ngg_one_lane && !barrier_half_phase;
}

void seed_active_vector_words(ActiveVectorWords& facts, const RegState& initial,
                              const std::set<int>& relayed_vectors) {
    // CONFIDENCE: HIGH -- this names only the actual producer ENTRY EXEC. It proves neither a
    // full guest wave nor peer participation. Equality/subset transfer follows B64 ISA set logic.
    facts.enabled = true;
    facts.current_is_entry = facts.current_within_entry = true;
    for (const auto& [reg, value] : initial.vreg)
        if (value && !relayed_vectors.contains(reg)) {
            facts.current.insert(reg);
            facts.entry_words.insert(reg);
        }
}

void meet_active_vector_words(ActiveVectorWords& into, const ActiveVectorWords& incoming) {
    into.enabled = into.enabled && incoming.enabled;
    into.current = intersect(std::move(into.current), incoming.current);
    into.current_is_entry = into.current_is_entry && incoming.current_is_entry;
    into.current_within_entry = into.current_within_entry && incoming.current_within_entry;
    into.entry_words = intersect(std::move(into.entry_words), incoming.entry_words);
    into.entry_masks = intersect(std::move(into.entry_masks), incoming.entry_masks);
    into.within_entry_masks =
        intersect(std::move(into.within_entry_masks), incoming.within_entry_masks);
    for (auto mask = into.masks.begin(); mask != into.masks.end();) {
        const auto other = incoming.masks.find(mask->first);
        if (other == incoming.masks.end()) {
            mask = into.masks.erase(mask);
        } else {
            mask->second = intersect(std::move(mask->second), other->second);
            ++mask;
        }
    }
}

void restore_active_vector_words(const std::set<int>& current, RegState& state) {
    if (!state.exec) return;
    for (int word : current) {
        const auto value = state.vreg.find(word);
        const auto mark = state.vreg_source_marks.find(word);
        if (value == state.vreg.end() || !value->second || mark == state.vreg_source_marks.end())
            continue;
        if (mark->second.placeholder && mark->second.hard) mark->second.outside_exec = state.exec;
    }
}

void restore_active_vector_instruction(const std::map<uint32_t, std::set<int>>& after,
                                       const Rdna2Inst& in, RegState& state) {
    if (const auto facts = after.find(in.pc); facts != after.end())
        restore_active_vector_words(facts->second, state);
}

bool active_vector_sources_defined(const ActiveVectorWords& state, const Rdna2Inst& in,
                                   const std::set<int>& relayed_vectors, const std::set<int>& masks,
                                   const std::set<int>& scalar_words,
                                   const std::set<int>& relayed_words, bool definite_scc) {
    const auto vector = [&](int word) {
        return !relayed_vectors.contains(word) || state.current.contains(word);
    };
    if (in.fmt == Rdna2Format::MUBUF) {
        if (in.opcode < 0x0c || in.opcode > 0x0f || in.mubuf_tfe || in.mubuf_lds || in.n_src < 3)
            return false;
        const uint32_t address_words = ((in.literal >> 12) & 1u) + ((in.literal >> 13) & 1u);
        if (address_words && in.src[0].kind != OperandKind::VGPR) return false;
        for (uint32_t word = 0; word < address_words; ++word)
            if (!vector(in.src[0].value + static_cast<int>(word))) return false;
        const auto& offset = in.src[2];
        if (offset.kind == OperandKind::SGPR || offset.kind == OperandKind::Special)
            return offset.value >= 0 && offset.value <= 124 &&
                   scalar_words.contains(offset.value) && !relayed_words.contains(offset.value);
        return offset.kind == OperandKind::InlineInt || offset.kind == OperandKind::Literal;
    }
    if (in.fmt == Rdna2Format::FLAT)
        return in.opcode >= 0x08 && in.opcode <= 0x0f &&
               (in.flat_segment == 0 || in.flat_segment == 2) && !in.flat_lds &&
               in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 &&
               in.src[0].value < 255 && in.src[1].kind == OperandKind::Special &&
               in.src[1].value == 125 && vector(in.src[0].value) && vector(in.src[0].value + 1);
    // Neither a peer operation nor a relative/selected-slot read may borrow lane-wise ownership.
    if (in.has_dpp || in.has_modifier ||
        (in.fmt == Rdna2Format::VOP1 &&
         (in.opcode == 0x02 || (in.opcode >= 0x40 && in.opcode <= 0x47))) ||
        (in.fmt == Rdna2Format::VOP2 &&
         (in.opcode == 0x1f || in.opcode == 0x2b || in.opcode == 0x3c)) ||
        (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x300 && in.opcode != 0x346) ||
        (in.fmt != Rdna2Format::VOP1 && in.fmt != Rdna2Format::VOP2 &&
         in.fmt != Rdna2Format::VOP3 && in.fmt != Rdna2Format::VOPC))
        return false;
    const bool implicit_vcc = in.fmt == Rdna2Format::VOP2 &&
                              (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a));
    if (implicit_vcc && !masks.contains(106)) return false;
    for (uint32_t source = 0; source < in.n_src; ++source) {
        const auto& operand = in.src[source];
        if (operand.kind == OperandKind::VGPR) {
            const auto width = vector_provenance_source_span(in, source);
            for (uint32_t word = 0; word < width; ++word)
                if (!vector(operand.value + static_cast<int>(word))) return false;
        } else if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) {
            if (operand.kind == OperandKind::Special && operand.value == 253) {
                if (!definite_scc || relayed_words.contains(253)) return false;
                continue;
            }
            const auto width = scalar_alu_source_words(in, source);
            if (!width || width == UINT32_MAX || operand.value < 0 || operand.value > 124)
                return false;
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = operand.value + static_cast<int>(word);
                if (!scalar_words.contains(reg) || relayed_words.contains(reg)) return false;
            }
        } else if (operand.kind != OperandKind::InlineInt &&
                   operand.kind != OperandKind::InlineFloat &&
                   operand.kind != OperandKind::Literal) {
            return false;
        }
    }
    return true;
}

void advance_active_vector_words(ActiveVectorWords& state, const Rdna2Inst& in,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write,
                                 bool genuine_active_sources, bool complete_active_word) {
    // Snapshot masks before an aliasing SDST or saveexec destination overwrites them.
    const Words before = state.current;
    const EntryDomain before_domain{state.current_is_entry, state.current_within_entry};
    const EntryDomain left_domain = in.n_src > 0 ? entry_domain(state, in.src[0]) : EntryDomain{};
    const EntryDomain right_domain = in.n_src > 1 ? entry_domain(state, in.src[1]) : EntryDomain{};
    const Words left = in.n_src > 0 ? mask_words(state, in.src[0]) : Words{};
    const Words right = in.n_src > 1 ? mask_words(state, in.src[1]) : Words{};
    Words result;
    bool has_result = false;
    EntryDomain result_domain;
    const bool copy = in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                      in.n_src == 1 && mask_operand(in.src[0]);
    if (copy) {
        result = left;
        result_domain = left_domain;
        has_result = true;
    } else if (in.fmt == Rdna2Format::SOP2 && in.n_src == 2) {
        result_domain = logical_domain(in.opcode, left_domain, right_domain);
        // These guarantees say which DATA words are genuine under the resulting bits.
        // They do not certify either operand as an architectural mask.
        if (in.opcode == 0x0f) {
            result = combine(left, right);   // A&B is a subset of either operand.
            has_result = true;
        } else if (in.opcode == 0x15) {
            result = left;   // A&~B is a subset of A.
            has_result = true;
        } else if (in.opcode == 0x11 || in.opcode == 0x13) {
            result = intersect(left, right);   // A|B and A^B may enable either operand's lanes.
            has_result = true;
        }
    }

    const auto retire_pair = [&](int base, uint32_t width) {
        const int last = base + static_cast<int>(width) - 1;
        const auto overlaps = [&](int mask) { return base <= mask + 1 && last >= mask; };
        std::erase_if(state.entry_masks, overlaps);
        std::erase_if(state.within_entry_masks, overlaps);
    };
    for (const auto& [base, width] : scalar_writes) {
        retire_pair(base, width);
        for (auto mask = state.masks.begin(); mask != state.masks.end();) {
            const int last = base + static_cast<int>(width) - 1;
            if (base <= mask->first + 1 && last >= mask->first)
                mask = state.masks.erase(mask);
            else
                ++mask;
        }
    }
    if (mask_write >= 0) retire_pair(mask_write, 2);
    if (mask_write >= 0)
        std::erase_if(state.masks, [&](const auto& mask) {
            return mask.first <= mask_write + 1 && mask.first + 1 >= mask_write;
        });
    const bool compare =
        in.fmt == Rdna2Format::VOPC || (in.fmt == Rdna2Format::VOP3 && mask_write >= 0);
    const bool saveexec = in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x24 && in.opcode <= 0x2b;
    const bool changes_exec = rdna2_instruction_may_change_exec(in);
    EntryDomain exec_domain;
    if (changes_exec) {
        if (in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)) {
            // Every possible comparison result is masked by the incoming EXEC.
            state.current = before;
            exec_domain = {false, before_domain.within};
        } else if (saveexec && in.opcode == 0x24) {
            state.current = before;   // S0 & EXEC only narrows old EXEC.
            exec_domain = logical_domain(0x0f, left_domain, before_domain);
        } else if (saveexec && in.opcode == 0x27) {
            // RDNA2 ISA 70648 §12.3: S0 & ~EXEC selects a subset of S0, not of old EXEC.
            state.current = left;
            exec_domain = logical_domain(0x15, left_domain, before_domain);
        } else if (saveexec && (in.opcode == 0x25 || in.opcode == 0x26)) {
            state.current = intersect(before, left);
            exec_domain =
                logical_domain(in.opcode == 0x25 ? 0x11 : 0x13, before_domain, left_domain);
        } else if (in.dst.value == 126 && has_result) {
            state.current = result;
            exec_domain = result_domain;
        } else {
            state.current.clear();
        }
    }
    if (changes_exec) {
        state.current_is_entry = exec_domain.equal;
        state.current_within_entry = exec_domain.within;
        if (exec_domain.within)
            state.current.insert(state.entry_words.begin(), state.entry_words.end());
    }
    if (saveexec && in.dst.value < 126) {
        state.masks[in.dst.value] = before;
        publish_entry_domain(state, in.dst.value, before_domain);
    } else if (mask_write >= 0) {
        if (compare) {
            state.masks[mask_write] = before;
            publish_entry_domain(state, mask_write, {false, before_domain.within});
        } else if (has_result) {
            state.masks[mask_write] = result;
            publish_entry_domain(state, mask_write, result_domain);
        }
    }
    if (copy && in.dst.value < 126) {
        state.masks[in.dst.value] = left;
        publish_entry_domain(state, in.dst.value, left_domain);
    }

    if (vector_write_leaves_old_word(in, false)) return;
    for (int word : vector_writes) {
        // No saved-mask value can revive DATA destroyed after it was saved, including a
        // later iteration at the same writer PC. Unknown partial/cross-lane writers also retire it.
        if (!genuine_active_sources) {
            state.current.erase(word);
            state.entry_words.erase(word);
            for (auto& [mask, words] : state.masks) words.erase(word);
        } else if (complete_active_word) {
            state.current.insert(word);
            if (state.current_is_entry) {
                state.entry_words.insert(word);
                for (int mask : state.within_entry_masks) state.masks[mask].insert(word);
            }
        }
    }
}

}   // namespace prosper::gpu

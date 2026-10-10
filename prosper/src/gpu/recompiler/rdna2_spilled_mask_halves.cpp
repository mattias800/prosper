// See rdna2_spilled_mask_halves.hpp.
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"

#include <algorithm>
#include <cstdint>

namespace prosper::gpu {
namespace {

constexpr uint32_t kOpWritelane = 0x361;
constexpr uint32_t kOpReadlane = 0x360;

template <typename Map>
void keep_agreeing(Map& into, const Map& incoming) {
    for (auto it = into.begin(); it != into.end();) {
        const auto other = incoming.find(it->first);
        if (other == incoming.end() || !(other->second == it->second))
            it = into.erase(it);
        else
            ++it;
    }
}

template <typename Map>
void erase_instance(Map& facts, uint32_t def_pc) {
    for (auto it = facts.begin(); it != facts.end();) {
        if (it->second.def_pc == def_pc)
            it = facts.erase(it);
        else
            ++it;
    }
}

void erase_vgpr_slots(SpilledMaskHalves& state, int vgpr) {
    for (auto it = state.slots.begin(); it != state.slots.end();) {
        if (it->first.first == vgpr)
            it = state.slots.erase(it);
        else
            ++it;
    }
}

bool constant_lane(const Rdna2Inst& in) {
    return in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
           in.src[1].value <= 63;
}

}  // namespace

void meet_spilled_mask_halves(SpilledMaskHalves& into, const SpilledMaskHalves& incoming) {
    keep_agreeing(into.pair_def_pc, incoming.pair_def_pc);
    keep_agreeing(into.slots, incoming.slots);
    keep_agreeing(into.sregs, incoming.sregs);
    std::erase_if(into.defined_slots,
                  [&](const auto& slot) { return !incoming.defined_slots.contains(slot); });
    into.relayed_words.insert(incoming.relayed_words.begin(), incoming.relayed_words.end());
    into.lane_written_vgprs.insert(incoming.lane_written_vgprs.begin(),
                                   incoming.lane_written_vgprs.end());
}

namespace {

// Whether `in` reads a relayed word: the same inputs scalar_source_marks() charges in the emitter
// (scalar operands, a read-modify-write destination, SCC), on the state BEFORE `in`. SMEM's
// address words are not data.
bool reads_relayed_word(const std::set<int>& relayed, const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SMEM) return false;
    const auto any = [&](int first, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w)
            if (relayed.contains(first + static_cast<int>(w))) return true;
        return false;
    };
    for (uint32_t k = 0; k < in.n_src && k < 3; ++k) {
        const Operand& o = in.src[k];
        const bool scalar = o.kind == OperandKind::SGPR ||
                            (o.kind == OperandKind::Special && o.value >= 106 && o.value <= 124);
        if (!scalar) continue;
        uint32_t width = scalar_alu_source_words(in, k);
        if (width == 0 || width == UINT32_MAX) width = 2;   // unknown: both words of a pair
        if (width > 2) width = static_cast<uint32_t>(std::max(0, 125 - o.value));   // a range
        if (any(o.value, width)) return true;
    }
    if (any(in.dst.value, scalar_implicit_destination_read_width(in))) return true;
    return scalar_reads_scc(in) && relayed.contains(kRelayedScc);
}

}  // namespace

void advance_defined_slots(SpilledMaskHalves& state, const Rdna2Inst& in,
                           const std::set<int>& masks, const std::set<int>& scalar_words,
                           const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                           const std::vector<int>& vector_writes) {
    auto& defined = state.defined_slots;
    auto& relayed = state.relayed_words;
    // Relayed words first, while `defined` still describes the slots `in` reads. A V_READLANE
    // relays unless it reads a constant-lane slot in `defined`; any other scalar write relays when
    // one of its inputs does. A compare is the only scalar writer certain to rewrite SCC.
    const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpReadlane;
    const bool relays =
        readlane ? state.lane_written_vgprs.contains(in.src[0].value) &&
                       !(constant_lane(in) && defined.contains({in.src[0].value, in.src[1].value}))
                 : reads_relayed_word(relayed, in);
    for (const auto& [base, width] : scalar_writes)
        for (uint32_t word = 0; word < width; ++word) {
            if (relays)
                relayed.insert(base + static_cast<int>(word));
            else
                relayed.erase(base + static_cast<int>(word));
        }
    const bool scalar_alu = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC;
    if (scalar_alu && relays)
        relayed.insert(kRelayedScc);
    else if (in.fmt == Rdna2Format::SOPC)
        relayed.erase(kRelayedScc);
    const auto drop_vgpr = [&](int vgpr) {
        std::erase_if(defined, [&](const auto& slot) { return slot.first == vgpr; });
    };
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == kOpWritelane) {
        state.lane_written_vgprs.insert(in.dst.value);
        if (!constant_lane(in)) return drop_vgpr(in.dst.value);
        const Operand& source = in.src[0];
        const int reg = source.value;
        const bool constant = source.kind == OperandKind::InlineInt ||
                              source.kind == OperandKind::InlineFloat ||
                              source.kind == OperandKind::Literal;
        const bool is_sgpr = source.kind == OperandKind::SGPR;
        const bool is_exec = source.kind == OperandKind::Special && (reg == 126 || reg == 127);
        const bool pair_mask = is_sgpr && (masks.contains(reg) || masks.contains(reg - 1));
        const bool defined_word = (is_sgpr || source.kind == OperandKind::Special) &&
                                  scalar_words.contains(reg) && !relayed.contains(reg);
        const std::pair<int, int> slot{in.dst.value, in.src[1].value};
        if (constant || is_exec || pair_mask || defined_word)
            defined.insert(slot);
        else
            defined.erase(slot);
    } else if (!readlane) {
        for (int vgpr : vector_writes) drop_vgpr(vgpr);
    }
}

void mark_relayed_words(const SpilledMaskHalves& entry, RegState& rs) {
    for (int word : entry.relayed_words)
        if (word == kRelayedScc)
            rs.scc_merge_placeholder = true;
        else
            rs.sreg_merge_placeholder.insert(word);
}

bool reassembles_spilled_mask_pair(const SpilledMaskHalves& state, const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.src[0].kind != OperandKind::SGPR)
        return false;
    const auto low = state.sregs.find(in.src[0].value);
    const auto high = state.sregs.find(in.src[0].value + 1);
    return low != state.sregs.end() && high != state.sregs.end() && low->second.half == 0 &&
           high->second.half == 1 && low->second.pair == high->second.pair &&
           low->second.def_pc == high->second.def_pc;
}

void advance_spilled_mask_halves(SpilledMaskHalves& state, const Rdna2Inst& in,
                                 const std::set<int>& masks, const std::set<int>& mask_slot_keys,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write) {
    const bool writelane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpWritelane;
    const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpReadlane;

    // Vector side. A constant-lane V_WRITELANE replaces exactly one slot; a dynamic lane may
    // replace any slot of that VGPR; every other vector write ends the whole VGPR's slots.
    if (writelane) {
        if (!constant_lane(in)) {
            erase_vgpr_slots(state, in.dst.value);
        } else {
            const std::pair<int, int> key{in.dst.value, in.src[1].value};
            SpilledMaskHalf fact;
            if (in.src[0].kind == OperandKind::SGPR) {
                const int source = in.src[0].value;
                // The low word of a live MUST-mask pair, or the high word of one. The pair must
                // also be a key the dispatcher persists as a mask slot, or emit_alu's Bool
                // would not survive the block edge this fact is about.
                for (int half = 0; half < 2 && fact.pair < 0; ++half) {
                    const int pair = source - half;
                    const auto def = state.pair_def_pc.find(pair);
                    if (pair >= 0 && pair <= 104 && masks.contains(pair) &&
                        mask_slot_keys.contains(pair) && def != state.pair_def_pc.end())
                        fact = {pair, half, def->second};
                }
            }
            if (fact.pair >= 0)
                state.slots[key] = fact;
            else
                state.slots.erase(key);
        }
    } else if (!readlane) {
        for (int vgpr : vector_writes) erase_vgpr_slots(state, vgpr);
    }

    // Scalar side. Any write of a word ends its reloaded-half fact and the definition of every
    // pair that word belongs to.
    for (const auto& [base, width] : scalar_writes)
        for (uint32_t word = 0; word < width; ++word) {
            const int reg = base + static_cast<int>(word);
            state.sregs.erase(reg);
            state.pair_def_pc.erase(reg);
            state.pair_def_pc.erase(reg - 1);
        }
    if (readlane && constant_lane(in)) {
        const auto slot = state.slots.find({in.src[0].value, in.src[1].value});
        if (slot != state.slots.end()) state.sregs[in.dst.value] = slot->second;
    }

    // A new definition at this pc is a new instance: facts about the previous one must not pair
    // with facts about this one.
    if (mask_write >= 0 && mask_write <= 104) {
        erase_instance(state.slots, in.pc);
        erase_instance(state.sregs, in.pc);
        state.pair_def_pc[mask_write] = in.pc;
    }
}

}   // namespace prosper::gpu

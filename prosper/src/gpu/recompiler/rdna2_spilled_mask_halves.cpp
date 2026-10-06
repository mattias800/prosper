// See rdna2_spilled_mask_halves.hpp.
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"

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

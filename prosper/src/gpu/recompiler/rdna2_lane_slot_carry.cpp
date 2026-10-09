// Constant-lane V_WRITELANE spill slots across structured joins; see rdna2_lane_slot_carry.hpp.
#include "gpu/recompiler/rdna2_lane_slot_carry.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {

constexpr uint32_t kOpWritelane = 0x361;
constexpr uint32_t kOpReadlane = 0x360;

bool constant_lane_writelane(const Rdna2Inst& in) {
    return !in.is_end && in.fmt == Rdna2Format::VOP3 && in.opcode == kOpWritelane &&
           in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
           in.src[1].value <= 63;
}

// A source emit_alu always spills into the mask domain: EXEC or VCC, or a proven Wave64 half
// alias. An SGPR source is a mask only when it holds a saved mask at emission time, which no
// static scan can see; such a slot starts in the data domain and finish/patch handle a mismatch.
bool static_mask_source(const SpirvCompute& b, const Rdna2Inst& in) {
    const int source = in.src[0].value;
    return source == 106 || source == 107 || source == 126 || source == 127 ||
           b.wave64_mask_writelane_alias_pcs.contains(in.pc);
}

const uint32_t* find_slot(const std::unordered_map<int, std::unordered_map<int, uint32_t>>& slots,
                          int vgpr, int lane) {
    const auto v = slots.find(vgpr);
    if (v == slots.end()) return nullptr;
    const auto l = v->second.find(lane);
    return l == v->second.end() ? nullptr : &l->second;
}

void erase_slot(std::unordered_map<int, std::unordered_map<int, uint32_t>>& slots, int vgpr,
                int lane) {
    const auto v = slots.find(vgpr);
    if (v == slots.end()) return;
    v->second.erase(lane);
    if (v->second.empty()) slots.erase(v);
}

// The edge ended this spill array: an ordinary write tombstoned it, or an earlier merge left it as
// an EMPTY array (see join_lane_slots). Its lanes hold values no slot names on that edge.
bool array_ended(const LaneSlotEdge& edge, int vgpr) {
    if (edge.invalidated.contains(vgpr)) return true;
    const auto data = edge.data.find(vgpr);
    return data != edge.data.end() && data->second.empty() && !edge.mask.contains(vgpr);
}

void set_slot(RegState& rs, int vgpr, int lane, bool mask, uint32_t value) {
    auto& keep = mask ? rs.vgpr_lane_mask_slots : rs.vgpr_lane_slots;
    auto& drop = mask ? rs.vgpr_lane_slots : rs.vgpr_lane_mask_slots;
    keep[vgpr][lane] = value;
    erase_slot(drop, vgpr, lane);
    rs.invalidated_vgpr_lane_slots.erase(vgpr);
}

}  // namespace

std::set<std::pair<int, int>> lane_slots_written(const std::vector<Rdna2Inst>& ins, uint32_t lo,
                                                 uint32_t hi) {
    std::set<std::pair<int, int>> written;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.pc < lo || in.pc >= hi || !constant_lane_writelane(in)) continue;
        written.insert({in.dst.value, in.src[1].value});
    }
    return written;
}

LaneSlotEdge LaneSlotEdge::of(const RegState& rs, uint32_t block) {
    return LaneSlotEdge{rs.vgpr_lane_slots, rs.vgpr_lane_mask_slots, rs.invalidated_vgpr_lane_slots,
                        block};
}

void join_lane_slots(SpirvCompute& b, RegState& rs, const LaneSlotEdge& first,
                     const LaneSlotEdge& second, uint32_t branch_pc) {
    std::map<int, std::set<int>> lanes;   // ordered: deterministic phi emission
    for (const LaneSlotEdge* edge : {&first, &second})
        for (const auto* slots : {&edge->data, &edge->mask})
            for (const auto& [vgpr, by_lane] : *slots)
                if (by_lane.empty())
                    lanes[vgpr];
                else
                    for (const auto& kv : by_lane) lanes[vgpr].insert(kv.first);
    rs.vgpr_lane_slots.clear();
    rs.vgpr_lane_mask_slots.clear();
    rs.invalidated_vgpr_lane_slots = first.invalidated;
    rs.invalidated_vgpr_lane_slots.insert(second.invalidated.begin(), second.invalidated.end());
    for (const auto& [vgpr, lane_set] : lanes) {
        // An EMPTY spill array is how a merge says "a spill array whose lanes no slot names": the
        // VGPR's vector value is gone on the slot edge (V_WRITELANE erased it), and the slot values
        // are gone on the other. operand_bits refuses an ordinary read of a spill array and
        // V_READLANE refuses a missing lane, so every later read of it refuses.
        auto& data_lanes = rs.vgpr_lane_slots[vgpr];
        // One edge ended this spill array (an ordinary write, or an earlier such merge). Its lanes
        // hold vector data there, which no slot can stand for.
        if (array_ended(first, vgpr) || array_ended(second, vgpr)) {
            rs.invalidated_vgpr_lane_slots.erase(vgpr);
            continue;
        }
        for (int lane : lane_set) {
            const uint32_t* d1 = find_slot(first.data, vgpr, lane);
            const uint32_t* m1 = find_slot(first.mask, vgpr, lane);
            const uint32_t* d2 = find_slot(second.data, vgpr, lane);
            const uint32_t* m2 = find_slot(second.mask, vgpr, lane);
            if ((d1 && m2) || (m1 && d2)) {
                // Data on one edge and a mask on the other has no single phi type. The slot is
                // often dead here (an EXEC spill reusing a lane); drop it, so only a reload after
                // the merge refuses -- visibly, as "slot never written".
                log_recompile_diagnostic(b.diagnostic, "lane-slot-join", "consequent",
                                         "lane slot v%d[%d] is data on one edge and a mask on the "
                                         "other at the merge of branch pc=%u: dropped",
                                         vgpr, lane, branch_pc);
                continue;
            }
            const bool mask = m1 || m2;
            const uint32_t* v1 = mask ? m1 : d1;
            const uint32_t* v2 = mask ? m2 : d2;
            // An edge that never wrote the lane leaves it undefined there: any value is correct.
            const uint32_t placeholder = mask ? b.bfalse() : b.uconst(0);
            const uint32_t a = v1 ? *v1 : placeholder;
            const uint32_t c = v2 ? *v2 : placeholder;
            const uint32_t joined = a == c ? a
                                           : b.emit_phi_2way(mask ? b.t_bool : b.t_u32, a,
                                                             first.block, c, second.block);
            if (mask)
                rs.vgpr_lane_mask_slots[vgpr][lane] = joined;
            else
                data_lanes[lane] = joined;
            // The placeholder is a FABRICATED word on its edge, exactly like an SGPR absent on one
            // edge (merge_edge_word_fabricated): a consumer that projects scalar data onto lanes
            // must refuse it (#4725). The edges' own marks were already joined.
            if (!v1 || !v2) rs.lane_slot_merge_placeholder.insert({vgpr, lane});
        }
        // `data_lanes` stays even when empty: a reload of a dropped lane must find the array and
        // refuse ("slot never written"), not fall through to a shuffle of the VGPR's placeholder.
    }
}

void LaneSlotLoopCarry::open(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
                             uint32_t header_pc, uint32_t backedge_pc, uint32_t preheader) {
    header_pc_ = header_pc;
    slots_.clear();
    std::map<std::pair<int, int>, bool> statically_mask;
    read_in_loop_.clear();
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.pc >= header_pc && in.pc < backedge_pc && in.fmt == Rdna2Format::VOP3 &&
            in.opcode == kOpReadlane && in.src[1].kind == OperandKind::InlineInt)
            read_in_loop_.insert({in.src[0].value, in.src[1].value});
        if (in.pc < header_pc || in.pc >= backedge_pc || !constant_lane_writelane(in)) continue;
        const std::pair<int, int> key{in.dst.value, in.src[1].value};
        const bool mask = static_mask_source(b, in);
        const auto [it, inserted] = statically_mask.emplace(key, mask);
        if (!inserted) it->second = it->second && mask;
    }
    for (const auto& [key, mask_guess] : statically_mask) {
        const auto [vgpr, lane] = key;
        const uint32_t* data = find_slot(rs.vgpr_lane_slots, vgpr, lane);
        const uint32_t* mask = find_slot(rs.vgpr_lane_mask_slots, vgpr, lane);
        const bool is_mask = mask ? true : (data ? false : mask_guess);
        const bool seeded = data || mask;
        const uint32_t seed = mask ? *mask : (data ? *data : (is_mask ? b.bfalse() : b.uconst(0)));
        size_t patch = 0;
        const uint32_t phi = b.emit_phi2(is_mask ? b.t_bool : b.t_u32, seed, preheader, patch);
        set_slot(rs, vgpr, lane, is_mask, phi);
        slots_.push_back({vgpr, lane, is_mask, phi, patch, phi, seed, seeded, false,
                          !seeded || preheader_marks_.contains(key), true});
    }
}

void LaneSlotLoopCarry::record_check(const RegState& rs) {
    for (Slot& slot : slots_) {
        const uint32_t* value = find_slot(slot.mask ? rs.vgpr_lane_mask_slots : rs.vgpr_lane_slots,
                                          slot.vgpr, slot.lane);
        // The condition region may itself end the lifetime or switch the domain; the exit then
        // carries no slot value this record can name.
        slot.at_check = value ? *value : 0u;
    }
}

bool LaneSlotLoopCarry::patch_backedge(SpirvCompute& b, const RegState& rs, uint32_t cont) {
    for (Slot& slot : slots_) {
        const uint32_t* same = find_slot(slot.mask ? rs.vgpr_lane_mask_slots : rs.vgpr_lane_slots,
                                         slot.vgpr, slot.lane);
        if (same) {
            b.patch_phi(slot.patch, *same, cont);
            slot.backedge_marked = rs.lane_slot_merge_placeholder.contains({slot.vgpr, slot.lane});
            continue;
        }
        const bool other = find_slot(slot.mask ? rs.vgpr_lane_slots : rs.vgpr_lane_mask_slots,
                                     slot.vgpr, slot.lane) != nullptr;
        // Either the body ended the spill lifetime with an ordinary VGPR write, or it left the slot
        // in the other data/mask domain (a seeded slot that changed, or an unseeded one whose domain
        // the static scan could not predict). On the hardware the lane then holds a value no slot
        // names -- a V_READLANE of an ordinary VGPR lane is legitimate code -- so a later
        // iteration's reload must not see the seed. Refuse when the loop reloads this lane at all;
        // otherwise the phi is unobserved inside the loop, and the seed (which the preheader makes
        // dominate this block) closes it.
        if (read_in_loop_.contains({slot.vgpr, slot.lane})) {
            log_recompile_diagnostic(
                b.diagnostic, "recompile-reject", "terminal",
                "lane slot v%d[%d] is reloaded in the loop but the body %s "
                "(header pc=%u)",
                slot.vgpr, slot.lane,
                other ? "leaves it in the other domain" : "ends its spill lifetime", header_pc_);
            return false;
        }
        b.patch_phi(slot.patch, slot.seed, cont);
        slot.uncarried = true;
    }
    return true;
}

void LaneSlotLoopCarry::note_preheader_marks(const RegState& rs) {
    preheader_marks_ = rs.lane_slot_merge_placeholder;
}

void LaneSlotLoopCarry::refine_exit_marks(RegState& rs) const {
    for (const Slot& slot : slots_) {
        if (slot.uncarried || slot.at_check != slot.phi) continue;
        if (slot.seed_marked || slot.backedge_marked)
            rs.lane_slot_merge_placeholder.insert({slot.vgpr, slot.lane});
        else
            rs.lane_slot_merge_placeholder.erase({slot.vgpr, slot.lane});
    }
}

bool LaneSlotLoopCarry::finish_exit(SpirvCompute& b, RegState& rs, bool body_edge,
                                    uint32_t check_end, uint32_t body_end) const {
    struct Exit {
        const Slot* slot;
        uint32_t value;
    };
    std::vector<Exit> exits;
    for (const Slot& slot : slots_) {
        if (slot.uncarried || !slot.at_check) continue;
        uint32_t value = slot.at_check;
        if (body_edge) {
            const uint32_t* from_body = find_slot(
                slot.mask ? rs.vgpr_lane_mask_slots : rs.vgpr_lane_slots, slot.vgpr, slot.lane);
            if (!from_body) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "lane slot v%d[%d] has no %s on the loop's break edge "
                                         "(header pc=%u)",
                                         slot.vgpr, slot.lane, slot.mask ? "mask" : "value",
                                         header_pc_);
                return false;
            }
            if (*from_body != value)
                value = b.emit_phi_2way(slot.mask ? b.t_bool : b.t_u32, value, check_end,
                                        *from_body, body_end);
        }
        exits.push_back({&slot, value});
    }
    for (const Exit& exit : exits)
        set_slot(rs, exit.slot->vgpr, exit.slot->lane, exit.slot->mask, exit.value);
    // A slot the condition region left unnamed, or the body left uncarried, has no value on the
    // exit edge that dominates the merge: on a zero-trip exit it still holds its pre-loop value,
    // after a trip it holds whatever the body left. Drop the lane but keep the spill array, EMPTY
    // if nothing else is left (the if-merge convention): erasing the array would let a later
    // V_READLANE shuffle the VGPR's tracked vector value, which is a placeholder on the zero-trip
    // path. operand_bits refuses an ordinary read of the array and V_READLANE the missing lane.
    for (const Slot& slot : slots_) {
        if (!slot.uncarried && slot.at_check) continue;
        erase_slot(rs.vgpr_lane_slots, slot.vgpr, slot.lane);
        erase_slot(rs.vgpr_lane_mask_slots, slot.vgpr, slot.lane);
        rs.vgpr_lane_slots.try_emplace(slot.vgpr);
    }
    return true;
}

}   // namespace prosper::gpu

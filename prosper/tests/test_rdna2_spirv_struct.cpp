// test_rdna2_spirv_struct -- structural checks for RDNA2->SPIR-V output that do not require Vulkan.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;

namespace {

enum : uint32_t {
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
};

bool is_type_decl(uint32_t op) {
    switch (op) {
        case OpTypeVoid:
        case OpTypeBool:
        case OpTypeInt:
        case OpTypeFloat:
        case OpTypeVector:
        case OpTypeArray:
        case OpTypeRuntimeArray:
        case OpTypeStruct:
        case OpTypePointer:
        case OpTypeFunction:
            return true;
        default:
            return false;
    }
}

bool type_result_ids_are_nonzero(const std::vector<uint32_t>& spv, uint32_t* bad_op) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (is_type_decl(op) && (wc < 2 || spv[i + 1] == 0)) {
            if (bad_op) *bad_op = op;
            return false;
        }
        i += wc;
    }
    return true;
}

bool has_signed_i32_type(const std::vector<uint32_t>& spv) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpTypeInt && wc == 4 && spv[i + 1] != 0 && spv[i + 2] == 32 && spv[i + 3] == 1)
            return true;
        i += wc;
    }
    return false;
}

// Whether the module contains an instruction with the given opcode.
bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == opcode) return true;
        i += wc;
    }
    return false;
}

// Whether the module contains the requested GLSL.std.450 extended instruction number.
bool has_glsl_ext_inst(const std::vector<uint32_t>& spv, uint32_t instruction) {
    constexpr uint32_t OpExtInst = 12;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // OpExtInst operands: result type, result, instruction set, instruction, operands...
        if (op == OpExtInst && wc >= 6 && spv[i + 4] == instruction) return true;
        i += wc;
    }
    return false;
}

uint32_t opcode_count(const std::vector<uint32_t>& spv, uint32_t opcode) {
    uint32_t count = 0;
    if (spv.size() < 5) return count;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return 0;
        count += op == opcode;
        i += wc;
    }
    return count;
}

// Prove that a subgroup vote is persisted through the CFG dispatcher's bool register file and then
// consumed as the condition of a later OpSelect.  This is the structural shape of a scalar mask
// comparison followed by s_cselect: opcode presence alone would not prove that SCC survived the
// dispatcher boundary or that the select read the newly-produced condition.
bool wave_vote_reaches_later_select(const std::vector<uint32_t>& spv,
                                    bool expect_inverted) {
    constexpr uint32_t OpLoad = 61, OpStore = 62, OpLogicalNot = 168,
                       OpSelect = 169, OpGroupNonUniformAny = 335;
    struct Select { uint32_t result = 0, condition = 0, yes = 0, no = 0; };
    std::unordered_set<uint32_t> votes, values, variables;
    std::vector<std::array<uint32_t, 2>> logical_nots, stores, loads;
    std::vector<Select> selects;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpGroupNonUniformAny && wc >= 4)
            votes.insert(spv[i + 2]);
        if (op == OpLogicalNot && wc == 4)
            logical_nots.push_back({spv[i + 2], spv[i + 3]});
        if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2]});
        if (op == OpLoad && wc == 4)
            loads.push_back({spv[i + 2], spv[i + 3]});
        if (op == OpSelect && wc == 6)
            selects.push_back({spv[i + 2], spv[i + 3], spv[i + 4], spv[i + 5]});
        i += wc;
    }
    if (expect_inverted) {
        for (const auto& negate : logical_nots)
            if (votes.contains(negate[1])) values.insert(negate[0]);
    } else {
        values = votes;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& select : selects)
            if ((values.contains(select.yes) || values.contains(select.no)) &&
                values.insert(select.result).second)
                changed = true;
        for (const auto& store : stores)
            if (values.contains(store[1]) && variables.insert(store[0]).second)
                changed = true;
        for (const auto& load : loads)
            if (variables.contains(load[1]) && values.insert(load[0]).second)
                changed = true;
    }
    return std::any_of(selects.begin(), selects.end(),
        [&](const Select& select) { return values.contains(select.condition); });
}

// Prove the defect-shaped saved-mask lowering: two distinct static comparisons publish event tags
// from alternate dispatcher cases, while both subgroup votes execute after the switch merge from
// ungated persistent mask loads.  Each event gate must select its own EQ/LG polarity into one SCC
// variable, whose later load controls s_cselect.  A vote left inside a lane-local case, or a vote
// predicate gated by the publishing lane's PC, fails this dataflow/location check.
bool saved_mask_pair_votes_use_uniform_event_phase(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstant = 43, OpLoad = 61, OpStore = 62,
                       OpLabel = 248, OpSelectionMerge = 247, OpSwitch = 251,
                       OpIEqual = 170, OpLogicalNot = 168, OpLogicalAnd = 167,
                       OpSelect = 169, OpGroupNonUniformAny = 335;
    struct Select { uint32_t condition = 0, yes = 0, no = 0; size_t position = 0; };
    struct Pair { uint32_t first = 0, second = 0; };
    struct Vote { uint32_t result = 0, predicate = 0; size_t position = 0; };
    std::unordered_map<uint32_t, uint32_t> constants, loads, logical_nots;
    std::unordered_map<uint32_t, Pair> logical_ands, equals;
    std::unordered_map<uint32_t, Select> selects;
    std::vector<std::array<uint32_t, 2>> stores;
    std::vector<Vote> votes;
    uint32_t last_selection_merge = 0, switch_merge = 0;
    size_t switch_position = 0, common_position = 0;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4)
            constants[spv[i + 2]] = spv[i + 3];
        else if (op == OpLoad && wc == 4)
            loads[spv[i + 2]] = spv[i + 3];
        else if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2]});
        else if (op == OpLogicalNot && wc == 4)
            logical_nots[spv[i + 2]] = spv[i + 3];
        else if (op == OpLogicalAnd && wc == 5)
            logical_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpIEqual && wc == 5)
            equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpSelect && wc == 6)
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5], i};
        else if (op == OpGroupNonUniformAny && wc == 5)
            votes.push_back({spv[i + 2], spv[i + 4], i});
        else if (op == OpSelectionMerge && wc >= 3)
            last_selection_merge = spv[i + 1];
        else if (op == OpSwitch && last_selection_merge) {
            switch_merge = last_selection_merge;
            switch_position = i;
        } else if (op == OpLabel && wc == 2 && switch_merge &&
                   spv[i + 1] == switch_merge && i > switch_position)
            common_position = i;
        i += wc;
    }
    if (votes.size() != 2 || !switch_position || !common_position) return false;

    std::unordered_set<uint32_t> mask_pointers;
    std::unordered_set<uint32_t> event_tags;
    uint32_t event_pointer = 0, pending_pointer = 0, scc_pointer = 0;
    bool saw_direct = false, saw_inverted = false;
    for (const Vote& vote : votes) {
        if (vote.position <= common_position) return false;
        const auto mismatch = selects.find(vote.predicate);
        if (mismatch == selects.end()) return false;
        const auto first = loads.find(mismatch->second.condition);
        const auto second = loads.find(mismatch->second.no);
        const auto negated_second = logical_nots.find(mismatch->second.yes);
        if (first == loads.end() || second == loads.end() ||
            negated_second == logical_nots.end() ||
            negated_second->second != mismatch->second.no ||
            first->second == second->second)
            return false;
        mask_pointers.insert(first->second);
        mask_pointers.insert(second->second);

        const auto inverted = std::find_if(
            logical_nots.begin(), logical_nots.end(),
            [&](const auto& item) { return item.second == vote.result; });
        const uint32_t direct_result = vote.result;
        const uint32_t inverted_result = inverted == logical_nots.end() ? 0 : inverted->first;
        bool selected_vote = false;
        for (const auto& selected : selects) {
            if (selected.second.position <= vote.position) continue;
            const bool direct = selected.second.yes == direct_result;
            const bool inverse = inverted_result && selected.second.yes == inverted_result;
            if (!direct && !inverse) continue;
            const auto gate = logical_ands.find(selected.second.condition);
            if (gate == logical_ands.end()) continue;
            uint32_t pending_load = 0, equality_id = 0;
            if (loads.contains(gate->second.first) && equals.contains(gate->second.second)) {
                pending_load = gate->second.first;
                equality_id = gate->second.second;
            } else if (loads.contains(gate->second.second) &&
                       equals.contains(gate->second.first)) {
                pending_load = gate->second.second;
                equality_id = gate->second.first;
            } else {
                continue;
            }
            const Pair equal = equals.at(equality_id);
            uint32_t event_load = 0, event_constant = 0;
            if (loads.contains(equal.first) && constants.contains(equal.second)) {
                event_load = equal.first;
                event_constant = equal.second;
            } else if (loads.contains(equal.second) && constants.contains(equal.first)) {
                event_load = equal.second;
                event_constant = equal.first;
            } else {
                continue;
            }
            const uint32_t this_event_pointer = loads.at(event_load);
            const uint32_t this_pending_pointer = loads.at(pending_load);
            if ((event_pointer && event_pointer != this_event_pointer) ||
                (pending_pointer && pending_pointer != this_pending_pointer))
                return false;
            event_pointer = this_event_pointer;
            pending_pointer = this_pending_pointer;
            event_tags.insert(constants.at(event_constant));

            const auto stored = std::find_if(
                stores.begin(), stores.end(),
                [&](const auto& store) { return store[1] == selected.first; });
            if (stored == stores.end()) continue;
            if (scc_pointer && scc_pointer != (*stored)[0]) return false;
            scc_pointer = (*stored)[0];
            saw_direct |= direct;
            saw_inverted |= inverse;
            selected_vote = true;
            break;
        }
        if (!selected_vote) return false;
    }
    if (mask_pointers.size() != 4 || event_tags.size() != 2 ||
        !event_tags.contains(1) || !event_tags.contains(2) ||
        !event_pointer || !pending_pointer || event_pointer == pending_pointer ||
        !scc_pointer || !saw_direct || !saw_inverted)
        return false;
    std::unordered_set<uint32_t> published_event_tags;
    for (const auto& store : stores) {
        if (store[0] != event_pointer) continue;
        const auto value = constants.find(store[1]);
        if (value != constants.end()) published_event_tags.insert(value->second);
    }
    if (!published_event_tags.contains(0) || !published_event_tags.contains(1) ||
        !published_event_tags.contains(2))
        return false;

    for (const auto& load : loads) {
        if (load.second != scc_pointer) continue;
        const auto consumed = std::find_if(
            selects.begin(), selects.end(),
            [&](const auto& selected) {
                return selected.second.condition == load.first;
            });
        if (consumed != selects.end()) return true;
    }
    return false;
}

// Prove the complete common-phase lowering for House's in-place DPP minimum reduction. The four
// dispatcher cases must publish shifts 1/2/4/8 into one mailbox; one uniform subgroup phase must use
// that dynamic amount for both the row bound and lane subtraction, gate source participation by
// two persisted bools (pending + EXEC), and shuffle a static event tag beside value/activity. The
// source event must equal the destination event on the write path before UMin reaches a Function
// VGPR. Opcode presence alone would miss direction and divergent-PC cross-site contamination.
bool dpp_min_row_shr_updates_dispatch_vgpr(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstant = 43, OpExtInst = 12, OpLoad = 61, OpStore = 62,
                       OpISub = 130, OpLogicalAnd = 167, OpSelect = 169,
                       OpIEqual = 170, OpUGreaterThanEqual = 174,
                       OpGroupNonUniformShuffle = 345,
                       GlslUMin = 38;
    struct Select { uint32_t condition = 0, yes = 0, no = 0; };
    struct Shuffle { uint32_t result = 0, value = 0, lane = 0; };
    std::unordered_map<uint32_t, uint32_t> constants, load_pointer;
    std::unordered_map<uint32_t, std::array<uint32_t, 2>> logical_ands;
    std::unordered_map<uint32_t, std::array<uint32_t, 2>> equals;
    std::unordered_map<uint32_t, Select> selects;
    std::vector<std::array<uint32_t, 2>> stores;
    std::vector<Shuffle> shuffles;
    std::unordered_set<uint32_t> shuffle_derived, umin_results;
    std::unordered_set<uint32_t> amount_bound_uses, amount_subtract_uses;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4)
            constants[spv[i + 2]] = spv[i + 3];
        else if (op == OpLoad && wc == 4)
            load_pointer[spv[i + 2]] = spv[i + 3];
        else if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2]});
        else if (op == OpLogicalAnd && wc == 5)
            logical_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpIEqual && wc == 5)
            equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpSelect && wc == 6) {
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
            if (shuffle_derived.contains(spv[i + 4]) ||
                shuffle_derived.contains(spv[i + 5]))
                shuffle_derived.insert(spv[i + 2]);
        } else if (op == OpGroupNonUniformShuffle && wc == 6) {
            shuffle_derived.insert(spv[i + 2]);
            shuffles.push_back({spv[i + 2], spv[i + 4], spv[i + 5]});
        } else if (op == OpExtInst && wc == 7 && spv[i + 4] == GlslUMin &&
                   (shuffle_derived.contains(spv[i + 5]) ||
                    shuffle_derived.contains(spv[i + 6]))) {
            umin_results.insert(spv[i + 2]);
        } else if (op == OpUGreaterThanEqual && wc == 5) {
            amount_bound_uses.insert(spv[i + 4]);
        } else if (op == OpISub && wc == 5) {
            amount_subtract_uses.insert(spv[i + 4]);
        }
        i += wc;
    }
    if (opcode_count(spv, OpGroupNonUniformShuffle) != 3 || umin_results.empty())
        return false;

    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> stored_constants;
    for (const auto& store : stores) {
        const auto value = constants.find(store[1]);
        if (value != constants.end()) stored_constants[store[0]].insert(value->second);
    }
    bool exact_amount = false;
    for (const auto& load : load_pointer) {
        const auto values = stored_constants.find(load.second);
        if (values != stored_constants.end() && values->second.contains(1) &&
            values->second.contains(2) && values->second.contains(4) &&
            values->second.contains(8) &&
            amount_bound_uses.contains(load.first) &&
            amount_subtract_uses.contains(load.first)) {
            exact_amount = true;
            break;
        }
    }

    bool pending_and_exec_gate = false;
    for (const auto& shuffle : shuffles) {
        const auto select = selects.find(shuffle.value);
        if (select == selects.end()) continue;
        const auto one = constants.find(select->second.yes);
        const auto zero = constants.find(select->second.no);
        const auto both = logical_ands.find(select->second.condition);
        if (one == constants.end() || one->second != 1 ||
            zero == constants.end() || zero->second != 0 ||
            both == logical_ands.end())
            continue;
        const auto first = load_pointer.find(both->second[0]);
        const auto second = load_pointer.find(both->second[1]);
        if (first != load_pointer.end() && second != load_pointer.end() &&
            first->second != second->second) {
            pending_and_exec_gate = true;
            break;
        }
    }

    // The event mailbox contains one distinct nonzero tag per static DPP site. Its load is shuffled
    // with the exact same source-lane id as value/activity, then compared against the destination
    // lane's unshuffled tag. Track that equality through every LogicalAnd into the final VGPR write.
    uint32_t event_match = 0;
    std::unordered_set<uint32_t> shuffle_lanes;
    for (const auto& shuffle : shuffles) shuffle_lanes.insert(shuffle.lane);
    for (const auto& shuffle : shuffles) {
        const auto event_load = load_pointer.find(shuffle.value);
        if (event_load == load_pointer.end()) continue;
        const auto tags = stored_constants.find(event_load->second);
        if (tags == stored_constants.end() || !tags->second.contains(0) ||
            !tags->second.contains(1) || !tags->second.contains(2) ||
            !tags->second.contains(3) || !tags->second.contains(4))
            continue;
        for (const auto& equal : equals) {
            const bool exact_pair =
                (equal.second[0] == shuffle.result && equal.second[1] == shuffle.value) ||
                (equal.second[1] == shuffle.result && equal.second[0] == shuffle.value);
            if (exact_pair) {
                event_match = equal.first;
                break;
            }
        }
        if (event_match) break;
    }
    std::unordered_set<uint32_t> event_guarded;
    if (event_match) event_guarded.insert(event_match);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& value : logical_ands) {
            if ((event_guarded.contains(value.second[0]) ||
                 event_guarded.contains(value.second[1])) &&
                event_guarded.insert(value.first).second)
                changed = true;
        }
    }

    bool min_reaches_store = false;
    for (const auto& select : selects) {
        if (!umin_results.contains(select.second.yes) ||
            !event_guarded.contains(select.second.condition))
            continue;
        if (std::any_of(stores.begin(), stores.end(), [&](const auto& store) {
                return store[1] == select.first;
            })) {
            min_reaches_store = true;
            break;
        }
    }
    return exact_amount && pending_and_exec_gate && event_match &&
        shuffle_lanes.size() == 1 && min_reaches_store;
}

// Prove GTA V's portable compute DPP add lowering rather than merely finding an OpIAdd. The two
// guest waves publish separate value/metadata scratch planes, address SRC0 with linear_id-amount
// inside a DPP16 row, require the source's static event to match, and select the add over the old
// persistent VGPR only on that guarded path. Instruction positions additionally prove that two
// uniform workgroup barriers bracket the scratch exchange and its later reuse.
bool compute_dpp_add_row_shr_updates_dispatch_vgpr(const std::vector<uint32_t>& spv,
                                                   uint32_t lanes,
                                                   uint32_t expected_dst) {
    constexpr uint32_t OpConstant = 43, OpLoad = 61, OpStore = 62,
                       OpAccessChain = 65, OpIAdd = 128, OpISub = 130,
                       OpShiftRightLogical = 194, OpLogicalAnd = 167,
                       OpSelect = 169, OpIEqual = 170, OpINotEqual = 171,
                       OpUGreaterThanEqual = 174, OpShiftLeftLogical = 196,
                       OpBitwiseOr = 197, OpBitwiseAnd = 199,
                       OpControlBarrier = 224;
    struct Binary { uint32_t first = 0, second = 0; };
    struct Select { uint32_t condition = 0, yes = 0, no = 0; };
    struct Access { uint32_t base = 0, index = 0; };
    struct Store { uint32_t pointer = 0, value = 0; size_t position = 0; };
    struct Load { uint32_t pointer = 0; size_t position = 0; };
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, Binary> iadds, isubs, shifts, shift_lefts,
                                                logical_ands, equals, not_equals,
                                                greater_equals, bitwise_ors,
                                                bitwise_ands;
    std::unordered_map<uint32_t, Select> selects;
    std::unordered_map<uint32_t, Access> accesses;
    std::unordered_map<uint32_t, Load> loads;
    std::vector<Store> stores;
    std::vector<size_t> barriers;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4)
            constants[spv[i + 2]] = spv[i + 3];
        else if (op == OpLoad && wc == 4)
            loads[spv[i + 2]] = {spv[i + 3], i};
        else if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2], i});
        else if (op == OpAccessChain && wc == 5)
            accesses[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpIAdd && wc == 5)
            iadds[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpISub && wc == 5)
            isubs[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpShiftRightLogical && wc == 5)
            shifts[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpShiftLeftLogical && wc == 5)
            shift_lefts[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpLogicalAnd && wc == 5)
            logical_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpIEqual && wc == 5)
            equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpINotEqual && wc == 5)
            not_equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpUGreaterThanEqual && wc == 5)
            greater_equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpBitwiseOr && wc == 5)
            bitwise_ors[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpBitwiseAnd && wc == 5)
            bitwise_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpSelect && wc == 6)
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
        else if (op == OpControlBarrier)
            barriers.push_back(i);
        i += wc;
    }

    auto literal_other = [&](const Binary& binary, uint32_t literal,
                             uint32_t* other) {
        const auto first = constants.find(binary.first);
        const auto second = constants.find(binary.second);
        if (first != constants.end() && first->second == literal) {
            if (other) *other = binary.second;
            return true;
        }
        if (second != constants.end() && second->second == literal) {
            if (other) *other = binary.first;
            return true;
        }
        return false;
    };
    auto pointer_store = [&](uint32_t pointer, uint32_t value, size_t* position) {
        const auto found = std::find_if(stores.begin(), stores.end(), [&](const Store& store) {
            return store.pointer == pointer && (!value || store.value == value);
        });
        if (found == stores.end()) return false;
        if (position) *position = found->position;
        return true;
    };

    // Locate the loaded source event: scratch metadata >> 1 == the destination's event mailbox.
    for (const auto& shifted_event : shifts) {
        const auto one = constants.find(shifted_event.second.second);
        const auto metadata_load = loads.find(shifted_event.second.first);
        if (one == constants.end() || one->second != 1 || metadata_load == loads.end()) continue;
        const auto metadata_access = accesses.find(metadata_load->second.pointer);
        if (metadata_access == accesses.end()) continue;
        const auto metadata_index_add = iadds.find(metadata_access->second.index);
        uint32_t source_index = 0;
        if (metadata_index_add == iadds.end() ||
            !literal_other(metadata_index_add->second, lanes, &source_index))
            continue;

        uint32_t event_match = 0, event_load = 0;
        for (const auto& equal : equals) {
            if (equal.second.first == shifted_event.first && loads.contains(equal.second.second)) {
                event_match = equal.first;
                event_load = equal.second.second;
                break;
            }
            if (equal.second.second == shifted_event.first && loads.contains(equal.second.first)) {
                event_match = equal.first;
                event_load = equal.second.first;
                break;
            }
        }
        if (!event_match) continue;

        std::unordered_set<uint32_t> event_tags;
        const uint32_t event_pointer = loads.at(event_load).pointer;
        for (const auto& store : stores) {
            if (store.pointer != event_pointer) continue;
            const auto constant = constants.find(store.value);
            if (constant != constants.end()) event_tags.insert(constant->second);
        }
        uint32_t nonzero_events = 0;
        for (uint32_t event : event_tags) nonzero_events += event != 0;
        if (!event_tags.contains(0) || nonzero_events < 2) continue;

        const auto source_select = selects.find(source_index);
        if (source_select == selects.end()) continue;
        const auto source_subtract = isubs.find(source_select->second.yes);
        const auto row_bound = greater_equals.find(source_select->second.condition);
        uint32_t linear_lane = 0;
        const auto row_lane = row_bound == greater_equals.end()
            ? bitwise_ands.end() : bitwise_ands.find(row_bound->second.first);
        if (source_subtract == isubs.end() || row_bound == greater_equals.end() ||
            row_lane == bitwise_ands.end() ||
            !literal_other(row_lane->second, 15, &linear_lane) ||
            source_subtract->second.first != linear_lane ||
            source_select->second.no != linear_lane ||
            source_subtract->second.second != row_bound->second.second)
            continue;
        const uint32_t amount = source_subtract->second.second;
        const auto amount_load = loads.find(amount);
        if (amount_load == loads.end()) continue;
        std::unordered_set<uint32_t> amounts;
        for (const auto& store : stores) {
            if (store.pointer != amount_load->second.pointer) continue;
            const auto constant = constants.find(store.value);
            if (constant != constants.end()) amounts.insert(constant->second);
        }
        if (!amounts.contains(1) || !amounts.contains(2) ||
            !amounts.contains(4) || !amounts.contains(8))
            continue;

        // The value plane uses the identical dynamic source index and the same scratch variable.
        uint32_t shifted_value = 0;
        for (const auto& load : loads) {
            const auto access = accesses.find(load.second.pointer);
            if (access == accesses.end() ||
                access->second.base != metadata_access->second.base)
                continue;
            const auto index_add = iadds.find(access->second.index);
            uint32_t index = 0;
            if (index_add != iadds.end() &&
                literal_other(index_add->second, 0, &index) && index == source_index) {
                shifted_value = load.first;
                break;
            }
        }
        if (!shifted_value) continue;

        uint32_t add_result = 0, local_value = 0;
        for (const auto& add : iadds) {
            if (add.second.first == shifted_value || add.second.second == shifted_value) {
                const uint32_t local = add.second.first == shifted_value
                    ? add.second.second : add.second.first;
                if (loads.contains(local)) {
                    add_result = add.first;
                    local_value = local;
                    break;
                }
            }
        }
        if (!add_result) continue;

        // Find both plane publications by their common destination invocation index.
        size_t value_publish = 0, metadata_publish = 0;
        uint32_t metadata_value = 0;
        for (const auto& value_store : stores) {
            const auto value_access = accesses.find(value_store.pointer);
            if (value_access == accesses.end() ||
                value_access->second.base != metadata_access->second.base ||
                value_store.value != local_value)
                continue;
            const auto value_index = iadds.find(value_access->second.index);
            uint32_t lane = 0;
            if (value_index == iadds.end() ||
                !literal_other(value_index->second, 0, &lane))
                continue;
            for (const auto& metadata_store : stores) {
                const auto metadata_publish_access = accesses.find(metadata_store.pointer);
                if (metadata_publish_access == accesses.end() ||
                    metadata_publish_access->second.base != metadata_access->second.base)
                    continue;
                const auto metadata_publish_index =
                    iadds.find(metadata_publish_access->second.index);
                uint32_t metadata_lane = 0;
                if (metadata_publish_index != iadds.end() &&
                    literal_other(metadata_publish_index->second, lanes, &metadata_lane) &&
                    metadata_lane == lane) {
                    value_publish = value_store.position;
                    metadata_publish = metadata_store.position;
                    metadata_value = metadata_store.value;
                    break;
                }
            }
            if (value_publish && metadata_publish) break;
        }
        if (!value_publish || !metadata_publish || !metadata_value) continue;

        // Recover the destination pending and EXEC mailboxes from the publication expression:
        // pending ? ((event << 1) | (active ? 1 : 0)) : 0.
        const auto metadata_select = selects.find(metadata_value);
        if (metadata_select == selects.end() ||
            !loads.contains(metadata_select->second.condition))
            continue;
        const auto metadata_zero = constants.find(metadata_select->second.no);
        const auto encoded = bitwise_ors.find(metadata_select->second.yes);
        if (metadata_zero == constants.end() || metadata_zero->second != 0 ||
            encoded == bitwise_ors.end())
            continue;
        const uint32_t pending = metadata_select->second.condition;
        uint32_t active = 0;
        bool encodes_event = false;
        for (uint32_t encoded_part : {encoded->second.first, encoded->second.second}) {
            const auto event_shift = shift_lefts.find(encoded_part);
            const auto active_select = selects.find(encoded_part);
            if (event_shift != shift_lefts.end()) {
                const auto one = constants.find(event_shift->second.second);
                encodes_event = encodes_event ||
                    (event_shift->second.first == event_load &&
                     one != constants.end() && one->second == 1);
            }
            if (active_select != selects.end()) {
                const auto one = constants.find(active_select->second.yes);
                const auto zero = constants.find(active_select->second.no);
                if (loads.contains(active_select->second.condition) &&
                    one != constants.end() && one->second == 1 &&
                    zero != constants.end() && zero->second == 0)
                    active = active_select->second.condition;
            }
        }
        if (!encodes_event || !active || active == pending) continue;

        // The selected source lane contributes an independent EXEC predicate.
        uint32_t source_active = 0;
        for (const auto& masked : bitwise_ands) {
            uint32_t metadata = 0;
            if (!literal_other(masked.second, 1, &metadata) ||
                metadata != shifted_event.second.first)
                continue;
            for (const auto& comparison : not_equals) {
                uint32_t compared = 0;
                if (literal_other(comparison.second, 0, &compared) &&
                    compared == masked.first) {
                    source_active = comparison.first;
                    break;
                }
            }
            if (source_active) break;
        }
        if (!source_active) continue;

        // The final condition must select the mailbox's physical destination as well.
        uint32_t destination_match = 0;
        for (const auto& equal : equals) {
            uint32_t mailbox = 0;
            if (!literal_other(equal.second, expected_dst, &mailbox) ||
                !loads.contains(mailbox))
                continue;
            const uint32_t pointer = loads.at(mailbox).pointer;
            const bool initialized_for_dst = std::any_of(
                stores.begin(), stores.end(), [&](const Store& store) {
                    const auto value = constants.find(store.value);
                    return store.pointer == pointer && value != constants.end() &&
                        value->second == expected_dst;
                });
            if (initialized_for_dst) {
                destination_match = equal.first;
                break;
            }
        }
        if (!destination_match) continue;

        auto guard_contains = [&](auto&& self, uint32_t root, uint32_t leaf) -> bool {
            if (root == leaf) return true;
            const auto logical_and = logical_ands.find(root);
            return logical_and != logical_ands.end() &&
                (self(self, logical_and->second.first, leaf) ||
                 self(self, logical_and->second.second, leaf));
        };
        size_t persistent_store_position = 0;
        bool preserves_old_vgpr = false;
        for (const auto& selected : selects) {
            if (selected.second.yes != add_result ||
                !loads.contains(selected.second.no) ||
                !guard_contains(guard_contains, selected.second.condition, pending) ||
                !guard_contains(guard_contains, selected.second.condition, active) ||
                !guard_contains(guard_contains, selected.second.condition,
                                source_select->second.condition) ||
                !guard_contains(guard_contains, selected.second.condition, source_active) ||
                !guard_contains(guard_contains, selected.second.condition, event_match) ||
                !guard_contains(guard_contains, selected.second.condition, destination_match))
                continue;
            const uint32_t old_pointer = loads.at(selected.second.no).pointer;
            const uint32_t source_mailbox_pointer = loads.at(local_value).pointer;
            const bool old_reaches_source_mailbox = std::any_of(
                stores.begin(), stores.end(), [&](const Store& store) {
                    const auto source = loads.find(store.value);
                    return store.pointer == source_mailbox_pointer &&
                        source != loads.end() && source->second.pointer == old_pointer;
                });
            if (old_reaches_source_mailbox &&
                pointer_store(old_pointer, selected.first, &persistent_store_position)) {
                preserves_old_vgpr = true;
                break;
            }
        }
        if (!preserves_old_vgpr) continue;

        const size_t publish_end = std::max(value_publish, metadata_publish);
        const auto first_barrier = std::find_if(
            barriers.begin(), barriers.end(), [&](size_t position) {
                return position > publish_end && position < metadata_load->second.position;
            });
        const auto second_barrier = std::find_if(
            barriers.begin(), barriers.end(), [&](size_t position) {
                return position > persistent_store_position;
            });
        if (first_barrier != barriers.end() && second_barrier != barriers.end()) return true;
    }
    return false;
}

// Trace the native exact-wave form independently of the portable scratch path. Each live ladder
// amount must select lane-amount inside a DPP16 row, shuffle both the value and its EXEC bit, add
// only for EXEC-active destinations with an in-bounds active source, and otherwise keep the old
// in-place VGPR before that value is persisted by the dispatcher.
bool compute_dpp_add_native_row_shr_updates_dispatch_vgpr(
    const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstant = 43, OpLoad = 61, OpStore = 62,
                       OpIAdd = 128, OpISub = 130, OpLogicalAnd = 167,
                       OpSelect = 169, OpINotEqual = 171,
                       OpUGreaterThanEqual = 174, OpBitwiseAnd = 199,
                       OpGroupNonUniformShuffle = 345;
    struct Binary { uint32_t first = 0, second = 0; };
    struct Select { uint32_t condition = 0, yes = 0, no = 0; };
    struct Shuffle { uint32_t value = 0, lane = 0; };
    std::unordered_map<uint32_t, uint32_t> constants, load_pointers;
    std::unordered_map<uint32_t, Binary> iadds, isubs, logical_ands,
                                                not_equals, greater_equals,
                                                bitwise_ands;
    std::unordered_map<uint32_t, Select> selects;
    std::unordered_map<uint32_t, Shuffle> shuffles;
    std::vector<std::array<uint32_t, 2>> stores;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4)
            constants[spv[i + 2]] = spv[i + 3];
        else if (op == OpLoad && wc == 4)
            load_pointers[spv[i + 2]] = spv[i + 3];
        else if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2]});
        else if (op == OpIAdd && wc == 5)
            iadds[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpISub && wc == 5)
            isubs[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpLogicalAnd && wc == 5)
            logical_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpINotEqual && wc == 5)
            not_equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpUGreaterThanEqual && wc == 5)
            greater_equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpBitwiseAnd && wc == 5)
            bitwise_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpSelect && wc == 6)
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
        else if (op == OpGroupNonUniformShuffle && wc == 6)
            shuffles[spv[i + 2]] = {spv[i + 4], spv[i + 5]};
        i += wc;
    }

    auto literal_other = [&](const Binary& binary, uint32_t literal,
                             uint32_t* other) {
        const auto first = constants.find(binary.first);
        const auto second = constants.find(binary.second);
        if (first != constants.end() && first->second == literal) {
            if (other) *other = binary.second;
            return true;
        }
        if (second != constants.end() && second->second == literal) {
            if (other) *other = binary.first;
            return true;
        }
        return false;
    };
    auto exact_and = [&](const Binary& binary, uint32_t first, uint32_t second) {
        return (binary.first == first && binary.second == second) ||
               (binary.first == second && binary.second == first);
    };

    std::unordered_set<uint32_t> proven_amounts;
    for (const auto& value_shuffle : shuffles) {
        const uint32_t source_value = value_shuffle.second.value;
        if (!load_pointers.contains(source_value)) continue;
        const auto source_lane = selects.find(value_shuffle.second.lane);
        if (source_lane == selects.end()) continue;
        const auto source_subtract = isubs.find(source_lane->second.yes);
        const auto in_bounds = greater_equals.find(source_lane->second.condition);
        if (source_subtract == isubs.end() || in_bounds == greater_equals.end() ||
            source_lane->second.no != source_subtract->second.first ||
            source_subtract->second.second != in_bounds->second.second)
            continue;
        const uint32_t lane = source_subtract->second.first;
        const uint32_t amount_id = source_subtract->second.second;
        const auto amount = constants.find(amount_id);
        if (amount == constants.end() ||
            (amount->second != 1 && amount->second != 2 &&
             amount->second != 4 && amount->second != 8))
            continue;
        const auto row_lane = bitwise_ands.find(in_bounds->second.first);
        uint32_t row_lane_source = 0;
        if (row_lane == bitwise_ands.end() ||
            !literal_other(row_lane->second, 15, &row_lane_source) ||
            row_lane_source != lane)
            continue;

        uint32_t destination_active = 0, source_active = 0;
        for (const auto& active_shuffle : shuffles) {
            if (active_shuffle.second.lane != value_shuffle.second.lane) continue;
            const auto active_encoding = selects.find(active_shuffle.second.value);
            if (active_encoding == selects.end() ||
                !load_pointers.contains(active_encoding->second.condition))
                continue;
            const auto one = constants.find(active_encoding->second.yes);
            const auto zero = constants.find(active_encoding->second.no);
            if (one == constants.end() || one->second != 1 ||
                zero == constants.end() || zero->second != 0)
                continue;
            for (const auto& comparison : not_equals) {
                uint32_t compared = 0;
                if (literal_other(comparison.second, 0, &compared) &&
                    compared == active_shuffle.first) {
                    destination_active = active_encoding->second.condition;
                    source_active = comparison.first;
                    break;
                }
            }
            if (source_active) break;
        }
        if (!destination_active || !source_active) continue;

        uint32_t valid_source = 0;
        for (const auto& logical_and : logical_ands) {
            if (exact_and(logical_and.second, source_lane->second.condition, source_active)) {
                valid_source = logical_and.first;
                break;
            }
        }
        if (!valid_source) continue;
        uint32_t shifted_or_old = 0;
        for (const auto& selected : selects) {
            if (selected.second.condition == valid_source &&
                selected.second.yes == value_shuffle.first &&
                selected.second.no == source_value) {
                shifted_or_old = selected.first;
                break;
            }
        }
        if (!shifted_or_old) continue;
        uint32_t add_result = 0;
        for (const auto& add : iadds) {
            if (exact_and(add.second, source_value, shifted_or_old)) {
                add_result = add.first;
                break;
            }
        }
        if (!add_result) continue;
        uint32_t write = 0;
        for (const auto& logical_and : logical_ands) {
            if (exact_and(logical_and.second, destination_active, valid_source)) {
                write = logical_and.first;
                break;
            }
        }
        if (!write) continue;
        for (const auto& selected : selects) {
            if (selected.second.condition != write || selected.second.yes != add_result ||
                selected.second.no != source_value)
                continue;
            const uint32_t pointer = load_pointers.at(source_value);
            if (std::any_of(stores.begin(), stores.end(), [&](const auto& store) {
                    return store[0] == pointer && store[1] == selected.first;
                })) {
                proven_amounts.insert(amount->second);
                break;
            }
        }
    }
    return proven_amounts == std::unordered_set<uint32_t>({1, 2, 4, 8});
}

// Prove the exact GTA V identity-QUAD_PERM lowering with ROW_MASK=0xa. The selected rows must
// integer-add two distinct persistent VGPRs, while the other rows keep the old destination. This
// traces the emitted values to their persistent store so mutations of the production row selector,
// add operands, or BC0 fallback cannot be hidden by unrelated SPIR-V instructions.
bool compute_dpp_add_partial_rows_updates_dispatch_vgpr(
    const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstant = 43, OpLoad = 61, OpStore = 62,
                       OpIAdd = 128, OpLogicalAnd = 167, OpSelect = 169,
                       OpINotEqual = 171, OpShiftRightLogical = 194,
                       OpShiftLeftLogical = 196, OpBitwiseAnd = 199;
    struct Binary { uint32_t first = 0, second = 0; };
    struct Select { uint32_t condition = 0, yes = 0, no = 0; };
    std::unordered_map<uint32_t, uint32_t> constants, load_pointers;
    std::unordered_map<uint32_t, Binary> iadds, logical_ands, not_equals,
                                                shift_rights, shift_lefts, bitwise_ands;
    std::unordered_map<uint32_t, Select> selects;
    std::vector<std::array<uint32_t, 2>> stores;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4)
            constants[spv[i + 2]] = spv[i + 3];
        else if (op == OpLoad && wc == 4)
            load_pointers[spv[i + 2]] = spv[i + 3];
        else if (op == OpStore && wc == 3)
            stores.push_back({spv[i + 1], spv[i + 2]});
        else if (op == OpIAdd && wc == 5)
            iadds[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpLogicalAnd && wc == 5)
            logical_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpINotEqual && wc == 5)
            not_equals[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpShiftRightLogical && wc == 5)
            shift_rights[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpShiftLeftLogical && wc == 5)
            shift_lefts[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpBitwiseAnd && wc == 5)
            bitwise_ands[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        else if (op == OpSelect && wc == 6)
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
        i += wc;
    }

    auto literal_other = [&](const Binary& binary, uint32_t literal,
                             uint32_t* other) {
        const auto first = constants.find(binary.first);
        const auto second = constants.find(binary.second);
        if (first != constants.end() && first->second == literal) {
            if (other) *other = binary.second;
            return true;
        }
        if (second != constants.end() && second->second == literal) {
            if (other) *other = binary.first;
            return true;
        }
        return false;
    };

    for (const auto& shift : shift_rights) {
        const auto four = constants.find(shift.second.second);
        if (four == constants.end() || four->second != 4) continue;
        const auto lane_mask = bitwise_ands.find(shift.second.first);
        if (lane_mask == bitwise_ands.end() ||
            !literal_other(lane_mask->second, 63, nullptr))
            continue;

        uint32_t row_bit = 0;
        for (const auto& left : shift_lefts) {
            const auto one = constants.find(left.second.first);
            if (one != constants.end() && one->second == 1 &&
                left.second.second == shift.first) {
                row_bit = left.first;
                break;
            }
        }
        if (!row_bit) continue;

        uint32_t masked_rows = 0;
        for (const auto& masked : bitwise_ands) {
            uint32_t other = 0;
            if (literal_other(masked.second, 0xa, &other) && other == row_bit) {
                masked_rows = masked.first;
                break;
            }
        }
        if (!masked_rows) continue;

        uint32_t row_selected = 0;
        for (const auto& comparison : not_equals) {
            uint32_t other = 0;
            if (literal_other(comparison.second, 0, &other) && other == masked_rows) {
                row_selected = comparison.first;
                break;
            }
        }
        if (!row_selected) continue;

        std::unordered_set<uint32_t> row_guarded{row_selected};
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& logical_and : logical_ands) {
                if ((row_guarded.contains(logical_and.second.first) ||
                     row_guarded.contains(logical_and.second.second)) &&
                    row_guarded.insert(logical_and.first).second)
                    changed = true;
            }
        }

        for (const auto& selected : selects) {
            if (!row_guarded.contains(selected.second.condition)) continue;
            const auto add = iadds.find(selected.second.yes);
            if (add == iadds.end() || selected.second.no != add->second.first) continue;
            const auto old_load = load_pointers.find(add->second.first);
            const auto addend_load = load_pointers.find(add->second.second);
            if (old_load == load_pointers.end() || addend_load == load_pointers.end() ||
                old_load->second == addend_load->second)
                continue;
            if (std::any_of(stores.begin(), stores.end(), [&](const auto& store) {
                    return store[0] == old_load->second && store[1] == selected.first;
                }))
                return true;
        }
    }
    return false;
}

// Trace IMAGE_GET_LOD all the way from two known VGPR bit patterns to the fragment output. This is
// deliberately a dataflow check rather than an opcode-presence check: it proves coordinate
// provenance, AMD/SPIR-V x/y component order, compact dmask-to-VDATA placement, and the EXEC select
// that preserves an inactive lane's old VGPR value.
bool image_query_lod_contract(const std::vector<uint32_t>& spv,
                              uint32_t expected_u_bits, uint32_t expected_v_bits,
                              const std::array<int, 4>& expected_export_components) {
    constexpr uint32_t OpConstant = 43, OpVariable = 59, OpStore = 62,
                       OpCompositeConstruct = 80, OpCompositeExtract = 81,
                       OpBitcast = 124, OpSelect = 169, OpImageQueryLod = 105;
    constexpr uint32_t StorageClassOutput = 3;
    struct SelectOperands { uint32_t condition = 0, true_value = 0, false_value = 0; };
    struct Extract { uint32_t composite = 0, component = 0; };
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, uint32_t> bitcast_sources;
    std::unordered_map<uint32_t, std::vector<uint32_t>> composites;
    std::unordered_map<uint32_t, Extract> extracts;
    std::unordered_map<uint32_t, SelectOperands> selects;
    std::unordered_set<uint32_t> outputs;
    uint32_t query_result = 0, query_coordinate = 0, stored_output = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4) constants[spv[i + 2]] = spv[i + 3];
        if (op == OpVariable && wc >= 4 && spv[i + 3] == StorageClassOutput)
            outputs.insert(spv[i + 2]);
        if (op == OpBitcast && wc == 4) bitcast_sources[spv[i + 2]] = spv[i + 3];
        if (op == OpCompositeConstruct && wc >= 4)
            composites[spv[i + 2]] = std::vector<uint32_t>(spv.begin() + i + 3,
                                                            spv.begin() + i + wc);
        if (op == OpCompositeExtract && wc == 5)
            extracts[spv[i + 2]] = {spv[i + 3], spv[i + 4]};
        if (op == OpSelect && wc == 6)
            selects[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
        if (op == OpImageQueryLod && wc == 5) {
            if (query_result) return false;
            query_result = spv[i + 2];
            query_coordinate = spv[i + 4];
        }
        if (op == OpStore && wc == 3 && outputs.contains(spv[i + 1])) {
            if (stored_output) return false;
            stored_output = spv[i + 2];
        }
        i += wc;
    }
    if (!query_result || !query_coordinate || !stored_output) return false;
    const auto coordinate = composites.find(query_coordinate);
    if (coordinate == composites.end() || coordinate->second.size() != 2u) return false;
    const uint32_t expected_coordinate_bits[2] = {expected_u_bits, expected_v_bits};
    for (uint32_t component = 0; component < 2; ++component) {
        const auto coordinate_bitcast = bitcast_sources.find(coordinate->second[component]);
        if (coordinate_bitcast == bitcast_sources.end()) return false;
        const auto coordinate_constant = constants.find(coordinate_bitcast->second);
        if (coordinate_constant == constants.end() ||
            coordinate_constant->second != expected_coordinate_bits[component])
            return false;
    }
    const auto output_vector = composites.find(stored_output);
    if (output_vector == composites.end() || output_vector->second.size() != 4u) return false;
    for (uint32_t output_component = 0; output_component < 4; ++output_component) {
        const int expected_query_component = expected_export_components[output_component];
        if (expected_query_component < 0) continue;
        const auto output_bitcast = bitcast_sources.find(output_vector->second[output_component]);
        if (output_bitcast == bitcast_sources.end()) return false;
        const auto predicated_write = selects.find(output_bitcast->second);
        if (predicated_write == selects.end() || !predicated_write->second.condition ||
            predicated_write->second.true_value == predicated_write->second.false_value)
            return false;
        const auto query_bitcast = bitcast_sources.find(predicated_write->second.true_value);
        if (query_bitcast == bitcast_sources.end()) return false;
        const auto extract = extracts.find(query_bitcast->second);
        if (extract == extracts.end() || extract->second.composite != query_result ||
            extract->second.component != static_cast<uint32_t>(expected_query_component))
            return false;
    }
    return true;
}

// Whether an instruction's zero-based operand has the requested literal value.
bool has_instruction_operand(const std::vector<uint32_t>& spv, uint32_t opcode,
                             uint32_t operand_index, uint32_t value) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu;
        const uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == opcode && wc > operand_index + 1u &&
            spv[i + 1u + operand_index] == value)
            return true;
        i += wc;
    }
    return false;
}

// Prove that one binary SPIR-V instruction consumes the two requested literal constants. This is
// stronger than checking that the constants and opcode merely coexist elsewhere in the module.
bool binary_uses_literal_operands(const std::vector<uint32_t>& spv, uint32_t opcode,
                                  uint32_t lhs_literal, uint32_t rhs_literal) {
    constexpr uint32_t OpConstant = 43;
    std::unordered_map<uint32_t, uint32_t> constants;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4) constants[spv[i + 2]] = spv[i + 3];
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // Integer binary: {result type, result id, lhs id, rhs id}.
        if (op == opcode && wc == 5) {
            const auto lhs = constants.find(spv[i + 3]);
            const auto rhs = constants.find(spv[i + 4]);
            if (lhs != constants.end() && rhs != constants.end() &&
                lhs->second == lhs_literal && rhs->second == rhs_literal)
                return true;
        }
        i += wc;
    }
    return false;
}

bool has_select_with_false_constant(const std::vector<uint32_t>& spv, uint32_t literal) {
    constexpr uint32_t OpConstant = 43, OpSelect = 169;
    std::unordered_set<uint32_t> matching_constants;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4 && spv[i + 3] == literal)
            matching_constants.insert(spv[i + 2]);
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // OpSelect {result-type, result, condition, true-object, false-object}.
        if (op == OpSelect && wc == 6 && matching_constants.contains(spv[i + 5]))
            return true;
        i += wc;
    }
    return false;
}

bool has_explicit_lod_constant(const std::vector<uint32_t>& spv, uint32_t expected) {
    constexpr uint32_t OpConstant = 43, OpImageSampleExplicitLod = 88, OpBitcast = 124;
    constexpr uint32_t ImageOperandsLod = 0x2;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, uint32_t> bitcasts;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4) constants[spv[i + 2]] = spv[i + 3];
        if (op == OpBitcast && wc == 4) bitcasts[spv[i + 2]] = spv[i + 3];
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // {result type, result, sampled image, coordinate, image-operands mask, lod}
        if (op == OpImageSampleExplicitLod && wc >= 7 &&
            (spv[i + 5] & ImageOperandsLod) != 0) {
            uint32_t value_id = spv[i + 6];
            if (const auto bitcast = bitcasts.find(value_id); bitcast != bitcasts.end())
                value_id = bitcast->second;
            const auto value = constants.find(value_id);
            if (value != constants.end() && value->second == expected) return true;
        }
        i += wc;
    }
    return false;
}

// Resolve the integer coordinate constructed for OpImageFetch and prove its three components are the
// requested literals. The third component is especially important for guest 2D_MSAA: it must be the
// explicit sample VGPR, because the host representation stores each sample plane as an array layer.
bool image_fetch_coord_literals(const std::vector<uint32_t>& spv,
                                uint32_t x, uint32_t y, uint32_t layer) {
    constexpr uint32_t OpConstant = 43, OpCompositeConstruct = 80, OpImageFetch = 95;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, std::array<uint32_t, 3>> vectors;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4) constants[spv[i + 2]] = spv[i + 3];
        if (op == OpCompositeConstruct && wc == 6)
            vectors[spv[i + 2]] = {spv[i + 3], spv[i + 4], spv[i + 5]};
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpImageFetch && wc >= 7) {
            const auto coordinate = vectors.find(spv[i + 4]);
            if (coordinate != vectors.end()) {
                const auto& ids = coordinate->second;
                const auto cx = constants.find(ids[0]);
                const auto cy = constants.find(ids[1]);
                const auto cl = constants.find(ids[2]);
                if (cx != constants.end() && cy != constants.end() && cl != constants.end() &&
                    cx->second == x && cy->second == y && cl->second == layer)
                    return true;
            }
        }
        i += wc;
    }
    return false;
}

struct OutputStoreStats {
    uint32_t stores = 0;
    uint32_t stores_with_one_repeated_source = 0;
};

// Count stores to Output variables and inspect vec4 construction through the final bitcasts. The
// graphics CFG regression deliberately exports four independently-written VGPRs: if its callback
// accidentally reads the entry RegState, every missing VGPR instead resolves to the same zero ID.
OutputStoreStats output_store_stats(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpVariable = 59, OpStore = 62, OpCompositeConstruct = 80, OpBitcast = 124;
    constexpr uint32_t StorageClassOutput = 3;
    std::unordered_set<uint32_t> outputs;
    std::unordered_map<uint32_t, size_t> definitions;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return {};
        if (op == OpVariable && wc >= 4 && spv[i + 3] == StorageClassOutput)
            outputs.insert(spv[i + 2]);
        if ((op == OpCompositeConstruct || op == OpBitcast) && wc >= 3)
            definitions[spv[i + 2]] = i;
        i += wc;
    }
    OutputStoreStats stats;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return {};
        if (op == OpStore && wc == 3 && outputs.contains(spv[i + 1])) {
            ++stats.stores;
            const auto construct = definitions.find(spv[i + 2]);
            if (construct != definitions.end()) {
                const size_t ci = construct->second;
                const uint32_t cop = spv[ci] & 0xffffu, cwc = spv[ci] >> 16u;
                if (cop == OpCompositeConstruct && cwc == 7) {
                    uint32_t source[4]{};
                    bool traced = true;
                    for (uint32_t component = 0; component < 4; ++component) {
                        const auto bitcast = definitions.find(spv[ci + 3 + component]);
                        if (bitcast == definitions.end()) { traced = false; break; }
                        const size_t bi = bitcast->second;
                        if ((spv[bi] & 0xffffu) != OpBitcast || (spv[bi] >> 16u) != 4) {
                            traced = false;
                            break;
                        }
                        source[component] = spv[bi + 3];
                    }
                    if (traced && source[0] == source[1] && source[1] == source[2] &&
                        source[2] == source[3])
                        ++stats.stores_with_one_repeated_source;
                }
            }
        }
        i += wc;
    }
    return stats;
}

bool binary_id_operands_are_nonzero(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        // Integer binary operations are {result type, result id, operand 1 id, operand 2 id}.
        if (op == opcode && (wc != 5 || !spv[i + 1] || !spv[i + 2] ||
                             !spv[i + 3] || !spv[i + 4]))
            return false;
        i += wc;
    }
    return true;
}

// ANDN1_SAVEEXEC must compute old_EXEC & ~source.  With source=false, the emitted boolean graph
// therefore contains LogicalNot(false) as an operand of LogicalAnd.  The formerly reversed
// lowering (~old_EXEC & source) instead fed false directly to the AND and silently killed all lanes.
bool logical_not_of_false_feeds_and(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstantFalse = 42, OpLogicalAnd = 167, OpLogicalNot = 168;
    uint32_t false_id = 0;
    std::unordered_map<uint32_t, uint32_t> logical_not_sources;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstantFalse && wc == 3) false_id = spv[i + 2];
        if (op == OpLogicalNot && wc == 4)
            logical_not_sources[spv[i + 2]] = spv[i + 3];
        i += wc;
    }
    if (!false_id) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpLogicalAnd && wc == 5) {
            const auto left = logical_not_sources.find(spv[i + 3]);
            const auto right = logical_not_sources.find(spv[i + 4]);
            if ((left != logical_not_sources.end() && left->second == false_id) ||
                (right != logical_not_sources.end() && right->second == false_id))
                return true;
        }
        i += wc;
    }
    return false;
}

bool phi_ids_are_nonzero(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpPhi = 245;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpPhi) {
            if (wc < 5 || ((wc - 3) & 1u) != 0) return false;
            for (uint32_t operand = 1; operand < wc; ++operand)
                if (spv[i + operand] == 0) return false;
        }
        i += wc;
    }
    return true;
}

// Whether the module contains OpDecorate (71) with the given decoration (word[i+2]).
bool has_decoration(const std::vector<uint32_t>& spv, uint32_t decoration) {
    enum : uint32_t { OpDecorateL = 71 };
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpDecorateL && wc >= 3 && spv[i + 2] == decoration) return true;
        i += wc;
    }
    return false;
}

bool has_builtin(const std::vector<uint32_t>& spv, uint32_t builtin) {
    enum : uint32_t { OpDecorateL = 71, DecBuiltIn = 11 };
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpDecorateL && wc >= 4 && spv[i + 2] == DecBuiltIn && spv[i + 3] == builtin)
            return true;
        i += wc;
    }
    return false;
}

// The largest OpTypeArray length (resolved through its OpConstant) in the module — for LDS sizing
// (#130), the Workgroup LDS array is the biggest array the compute shell declares.
uint32_t max_array_length(const std::vector<uint32_t>& spv) {
    enum : uint32_t { OpTypeArrayL = 28, OpConstantL = 43 };
    if (spv.size() < 5) return 0;
    std::unordered_map<uint32_t, uint32_t> const_val;   // result id -> u32 constant value
    uint32_t best = 0;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) break;
        if (op == OpConstantL && wc >= 4) const_val[spv[i + 2]] = spv[i + 3];   // {type,result,value}
        if (op == OpTypeArrayL && wc >= 4) {                                    // {result,elem,length-id}
            auto it = const_val.find(spv[i + 3]);
            if (it != const_val.end() && it->second > best) best = it->second;
        }
        i += wc;
    }
    return best;
}

} // namespace

int main() {
    printf("== test_rdna2_spirv_struct ==\n");

    // Uses signed convert/min/max/ashr, which requires a valid signed i32 type declaration.
    const uint32_t signed_kernel[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0x22060300u, 0x24080300u,
        0x4C060704u, 0x30060702u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv = recompile_valu(signed_kernel, sizeof(signed_kernel) / sizeof(signed_kernel[0]), 3, 0);
    if (spv.empty()) {
        printf("  [FAIL] signed kernel did not recompile\n");
        return 1;
    }

    uint32_t bad_op = 0;
    if (!type_result_ids_are_nonzero(spv, &bad_op)) {
        printf("  [FAIL] SPIR-V type declaration has invalid result id (op=%u)\n", bad_op);
        return 1;
    }
    printf("  [ok]   SPIR-V type declaration result ids are nonzero\n");

    if (!has_signed_i32_type(spv)) {
        printf("  [FAIL] signed kernel SPIR-V lacks a nonzero signed i32 type\n");
        return 1;
    }
    printf("  [ok]   signed kernel SPIR-V declares signed i32 with a nonzero id\n");

    // M0 has different layouts for DS_APPEND/DS_CONSUME in the two address domains. Astro Bot's
    // live GDS value must be shifted to obtain M0[31:16]; the same value in an ordinary LDS append
    // must be masked to M0[15:0]. These are mutation controls for each other: swapping either field
    // makes one of the two named checks fail while leaving the instruction present.
    constexpr uint32_t OpShiftRightLogical = 194, OpBitwiseAnd = 199;
    const uint32_t m0_gds_append[] = {
        0xbefc03ffu, 0x0c600020u,
        0xd8fa0010u, 0x00000000u, // ds_append v0 offset:0x10 gds
        0xbf810000u,
    };
    const uint32_t m0_lds_append[] = {
        0xbefc03ffu, 0x0c600020u,
        0xd8f80010u, 0x00000000u, // ds_append v0 offset:0x10 (LDS)
        0xbf810000u,
    };
    ShaderResourceTable m0_gds_table;
    ShaderResource m0_gds_resource;
    m0_gds_resource.cls = ResourceClass::ConstantBuffer;
    m0_gds_resource.format = DataFormat::Uint32;
    m0_gds_resource.num_components = 1;
    m0_gds_resource.binding = kComputeInternalGdsBinding;
    m0_gds_resource.size = 64 * 1024;
    m0_gds_resource.stride = 4;
    m0_gds_table.resources.push_back(m0_gds_resource);
    ComputeShaderConfig m0_config;
    const auto m0_gds_spv = recompile_compute(
        m0_gds_append, std::size(m0_gds_append), &m0_gds_table, m0_config);
    const auto m0_lds_spv = recompile_compute(
        m0_lds_append, std::size(m0_lds_append), nullptr, m0_config);
    if (m0_gds_spv.empty() ||
        !binary_uses_literal_operands(
            m0_gds_spv, OpShiftRightLogical, 0x0c600020u, 16u) ||
        binary_uses_literal_operands(
            m0_gds_spv, OpBitwiseAnd, 0x0c600020u, 0xffffu)) {
        printf("  [FAIL] GDS append did not derive its byte base from M0[31:16]\n");
        return 1;
    }
    printf("  [ok]   GDS append derives its byte base from M0[31:16]\n");
    if (m0_lds_spv.empty() ||
        !binary_uses_literal_operands(
            m0_lds_spv, OpBitwiseAnd, 0x0c600020u, 0xffffu) ||
        binary_uses_literal_operands(
            m0_lds_spv, OpShiftRightLogical, 0x0c600020u, 16u)) {
        printf("  [FAIL] LDS append did not derive its byte base from M0[15:0]\n");
        return 1;
    }
    printf("  [ok]   LDS append keeps its byte base in M0[15:0]\n");

    // Fixed-offset private spill/fill must also produce structurally valid graphics-stage modules.
    // This exercises Function-variable placement in the fragment and vertex shells, independently
    // of the compute execution tests.
    const uint32_t scratch_fragment[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xf800000fu, 0x00000000u, // exp mrt0 v0, v0, v0, v0
        0xBF810000u,
    };
    const auto scratch_fragment_spv = recompile_fragment(
        scratch_fragment, sizeof(scratch_fragment) / sizeof(scratch_fragment[0]));
    uint32_t scratch_fragment_bad_op = 0;
    if (scratch_fragment_spv.empty() || !has_opcode(scratch_fragment_spv, 28) ||
        !type_result_ids_are_nonzero(scratch_fragment_spv, &scratch_fragment_bad_op) ||
        !phi_ids_are_nonzero(scratch_fragment_spv)) {
        printf("  [FAIL] fragment private spill/fill emitted invalid SPIR-V (op=%u)\n",
               scratch_fragment_bad_op);
        return 1;
    }
    printf("  [ok]   fragment private spill/fill emits structurally valid Function storage\n");

    // #2441: the fragment skip diagnostic must say WHICH cross-lane form raised the wave64 contract,
    // because the two have opposite prospects under a wave32 lowering. A vote MAY be width-agnostic
    // (two 32-lane votes union to the same executed-pixel set); a ballot never is, since its bits
    // become guest scalar DATA and half a mask reported as whole is silently wrong -- which
    // rdna2_to_spirv.cpp's own comment on fragment_wave_ballot_half already says. Both used to set
    // kFragmentWaveReasonWaveAny, so the recoverable and unrecoverable cases were a single number:
    // on PPSA04263, 68 of 112 skipped fragment shaders, with no way to ask how it divides.
    //
    // s_quadmask_b64 vcc, vcc (0xbeea2d6a) is the ballot path. It reads VCC as a mask, so the v_cmp
    // ahead of it is load-bearing: without a real per-lane bool in VCC the lowering takes no ballot.
    const uint32_t quadmask_fragment[] = {
        0x7e020280u,              // v_mov_b32 v1, 0
        0x7c2200f0u,              // v_cmp_* -> VCC (per-lane bool)
        0xbeea2d6au,              // s_quadmask_b64 vcc, vcc
        0x7e000280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u, // exp mrt0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto quadmask_spv = recompile_fragment(quadmask_fragment, std::size(quadmask_fragment));
    if (quadmask_spv.empty()) {
        printf("  [FAIL] fragment s_quadmask_b64 did not recompile\n");
        return 1;
    }
    const uint32_t quadmask_reasons = fragment_spirv_required_subgroup_reasons(quadmask_spv);
    if (quadmask_reasons == UINT32_MAX) {
        printf("  [FAIL] fragment s_quadmask_b64 module carries no wave-reason marker\n");
        return 1;
    }
    // The arm that fails without the split: before it, nothing ever set WaveBallot.
    if (!(quadmask_reasons & prosper::gpu::kFragmentWaveReasonWaveBallot)) {
        printf("  [FAIL] s_quadmask_b64 reported its wave64 contract as a vote, not a ballot "
               "(reasons=0x%x)\n", quadmask_reasons);
        return 1;
    }
    // The width itself must be unchanged: this split is diagnostic, not a behavioural relaxation.
    if (fragment_spirv_required_subgroup_size(quadmask_spv) != 64) {
        printf("  [FAIL] ballot module stopped advertising its exact wave64 requirement\n");
        return 1;
    }
    printf("  [ok]   s_quadmask_b64 records a BALLOT reason, distinct from a vote, still wave64\n");

    // The other direction, so that moving the wrong call site cannot pass: a module whose only
    // cross-lane form is a branch-guard vote must NOT claim a ballot raised its contract.
    const uint32_t execz_vote_fragment[] = {
        0x7e020280u,              // v_mov_b32 v1, 0
        0x7c2200f0u,              // v_cmp_* -> VCC
        0xbf880002u,              // s_cbranch_execz +2  (routes through the fragment wave vote)
        0xbe8503f2u, 0x7e020205u,
        0x7e000280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto execz_vote_spv =
        recompile_fragment(execz_vote_fragment, std::size(execz_vote_fragment));
    if (!execz_vote_spv.empty()) {
        const uint32_t vote_reasons = fragment_spirv_required_subgroup_reasons(execz_vote_spv);
        if (vote_reasons != UINT32_MAX &&
            (vote_reasons & prosper::gpu::kFragmentWaveReasonWaveBallot)) {
            printf("  [FAIL] a vote-only fragment module claimed a ballot (reasons=0x%x)\n",
                   vote_reasons);
            return 1;
        }
        printf("  [ok]   a vote-only fragment module does not claim a ballot\n");
    }

    // Astro Bot's Wave32 graphics shaders save/copy/restore active-lane masks through the low
    // halves of EXEC and VCC.  These are the B32 equivalents of the B64 mask moves already modeled
    // below, not scalar-data transfers: treating EXEC_LO as ordinary data rejected the entire draw.
    const uint32_t wave32_fragment_masks[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,            // exp mrt0 v0,v1,v2,v3
        0xbf810000u,
    };
    if (!recompile_fragment(wave32_fragment_masks,
                            std::size(wave32_fragment_masks)).empty()) {
        printf("  [FAIL] Wave64/default graphics shader accepted Wave32 EXEC_LO/VCC_LO masks\n");
        return 1;
    }
    const auto wave32_fragment_spv = recompile_fragment(
        wave32_fragment_masks, std::size(wave32_fragment_masks), nullptr, nullptr,
        UINT32_MAX, nullptr, true);
    uint32_t wave32_fragment_bad_op = 0;
    if (wave32_fragment_spv.empty() ||
        !type_result_ids_are_nonzero(wave32_fragment_spv, &wave32_fragment_bad_op) ||
        !phi_ids_are_nonzero(wave32_fragment_spv)) {
        printf("  [FAIL] Wave32 fragment EXEC_LO/VCC_LO mask moves emitted invalid SPIR-V (op=%u)\n",
               wave32_fragment_bad_op);
        return 1;
    }
    printf("  [ok]   registered Wave32 fragment EXEC_LO/VCC_LO mask moves emit valid SPIR-V\n");

    if (fragment_effective_wave_size_for_test(
            64, 3142, 0x616dd4c0b241fbb1ull) != 32 ||
        fragment_effective_wave_size_for_test(
            64, 3142, 0x616dd4c0b241fbb0ull) != 64) {
        printf("  [FAIL] legacy Astro Wave32 capture did not select one coherent subgroup contract\n");
        return 1;
    }
    printf("  [ok]   legacy Astro Wave32 capture selects a 32-lane subgroup and mask contract\n");

    const uint32_t wave32_compute_masks[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0xbf810000u,
    };
    ComputeShaderConfig wave32_compute_config;
    wave32_compute_config.wave_size = 32;
    if (recompile_compute(wave32_compute_masks, std::size(wave32_compute_masks), nullptr,
                          wave32_compute_config).empty()) {
        printf("  [FAIL] proven Wave32 compute mask moves did not recompile\n");
        return 1;
    }
    // A Wave32 VOPC creates a VCC mask lifetime; one arm then recycles VCC_LO as ordinary scalar
    // data. The mask view is unknown after the merge, so consuming it with cndmask must remain
    // fail-visible instead of silently treating the poisoned edge as architectural false.
    const uint32_t wave32_compute_recycled_vcc[] = {
        0x7d840000u,                         // v_cmp_eq_u32 vcc, v0, v0
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // s_cbranch_scc0 +1
        0xbeea0385u,                         // s_mov_b32 vcc_lo, 5 (scalar-data lifetime)
        0x02020100u,                         // v_cndmask_b32 v1, v0, v0, vcc
        0xbf810000u,
    };
    wave32_compute_config.native_subgroup_size = 32;
    wave32_compute_config.local_x = 32;
    if (!recompile_compute(wave32_compute_recycled_vcc,
                           std::size(wave32_compute_recycled_vcc), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] poisoned Wave32 VCC was consumed after a CFG merge\n");
        return 1;
    }
    printf("  [ok]   poisoned Wave32 VCC consumption after a CFG merge is rejected\n");
    const uint32_t wave32_compute_recycled_vcc_half[] = {
        0x7d840000u,                         // v_cmp_eq_u32 vcc, v0, v0
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // s_cbranch_scc0 +1
        0xbeea0385u,                         // one arm recycles vcc_lo as scalar data
        0x876b0100u,                         // s_and_b32 vcc_hi, s0, s1
        0x02020100u,                         // implicit VCC_LO read remains ambiguous
        0xbf810000u,
    };
    wave32_compute_config.user_sgprs = {1u, 1u};
    if (!recompile_compute(wave32_compute_recycled_vcc_half,
                           std::size(wave32_compute_recycled_vcc_half), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] partial VCC write accepted an unknown prior mask\n");
        return 1;
    }
    printf("  [ok]   partial VCC_HI write cannot validate an ambiguous VCC_LO read\n");

    // A complete scalar-data write to VCC_LO defines every architectural predicate bit in Wave32.
    // Consume it immediately through the implicit e32 cndmask form so this cannot pass merely by
    // treating the new lifetime as dead scalar scratch.
    const uint32_t wave32_compute_scalar_vcc_lo_consumer[] = {
        0x876a8181u,                         // s_and_b32 vcc_lo, 1, 1
        0x02020100u,                         // v_cndmask_b32 v1, v0, v0, vcc
        0xbf810000u,
    };
    if (recompile_compute(wave32_compute_scalar_vcc_lo_consumer,
                          std::size(wave32_compute_scalar_vcc_lo_consumer), nullptr,
                          wave32_compute_config).empty()) {
        printf("  [FAIL] full Wave32 scalar VCC_LO write did not feed implicit cndmask\n");
        return 1;
    }
    printf("  [ok]   full Wave32 scalar VCC_LO write feeds its implicit mask consumer\n");

    // GTA V's exact PC1309/PC1311 producer-consumer packets occupy one dispatcher case, matching the
    // production basic block. PC1310 stays between them. The crossing tail forces the arbitrary-CFG
    // switch, so its discovery/dataflow scan must advertise the scalar cselect's new mask lifetime
    // before that case is emitted; this does not claim a cross-case producer/consumer transfer.
    const uint32_t wave32_compute_scalar_cselect_cfg[] = {
        0xbe8403ffu, 0x80000009u,            // pc0: s_mov_b32 s4, lane-bit mask
        0xbf060000u,                         // pc2: SCC=1
        0x7e020287u,                         // pc3: v_mov_b32 v1, 7
        0x856a8004u,                         // pc4: exact s_cselect_b32 vcc_lo,s4,0
        0x061212f2u,                         // pc5: exact intervening GTA V vector packet
        0x02020290u,                         // pc6: exact implicit VCC consumer
        0x7da40100u,                         // pc7: CMPX changes EXEC only
        0xbf880003u,                         // pc8: execz -> pc12
        0x7d840100u,                         // pc9: fresh VCC mask in one arm
        0x02020100u,                         // pc10: consume the arm-local mask
        0xbf880002u,                         // pc11: execz -> pc14 (crossing region)
        0xbf060000u,                         // pc12: scalar compare at rejoin
        0xbf850001u,                         // pc13: third branch -> pc15
        0x7e040280u,                         // pc14: no mask read before termination
        0xbf810000u,
    };
    const auto scalar_cselect_cfg_spv = recompile_compute(
        wave32_compute_scalar_cselect_cfg,
        std::size(wave32_compute_scalar_cselect_cfg), nullptr,
        wave32_compute_config);
    if (scalar_cselect_cfg_spv.empty() || !has_opcode(scalar_cselect_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(scalar_cselect_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(scalar_cselect_cfg_spv)) {
        printf("  [FAIL] Wave32 CFG did not discover GTA V's scalar-cselect VCC lifetime\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG discovers and emits GTA V's scalar-cselect VCC lifetime\n");

    // The scalar-data bridge is deliberately compute-only. In a Wave32 graphics shader, the exact
    // packet remains an ordinary scalar write to physical VCC_LO and therefore invalidates an older
    // predicate lifetime. The crossing tail routes this through the graphics dispatcher too: if
    // either its static inventory/dataflow or record_scalar_write applies the compute-only bridge to
    // graphics, the stale all-true compare reaches PC6 and this shader is incorrectly accepted.
    const uint32_t wave32_fragment_scalar_cselect_data_cfg[] = {
        0x7e000280u,                         // pc0: v_mov_b32 v0, 0
        0x7d840000u,                         // pc1: all-true VCC predicate
        0xbe840381u,                         // pc2: s_mov_b32 s4, 1 (ordinary scalar data)
        0xbf060000u,                         // pc3: SCC=1
        0x856a8004u,                         // pc4: exact scalar cselect to VCC_LO
        0x061212f2u,                         // pc5: exact intervening GTA V packet
        0x02020290u,                         // pc6: stale implicit VCC read must reject
        0x7da40100u,                         // pc7: crossing CFG tail begins
        0xbf880003u,                         // pc8: execz -> pc12
        0x7d840100u,                         // pc9: fresh VCC mask in one arm
        0x02020100u,                         // pc10: consume the arm-local mask
        0xbf880002u,                         // pc11: execz -> pc14
        0xbf060000u,                         // pc12: scalar compare at rejoin
        0xbf850001u,                         // pc13: third branch -> pc15
        0x7e040280u,                         // pc14: no mask read before export
        0xf800000fu, 0x03020100u,            // pc15: export MRT0
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_fragment_scalar_cselect_data_cfg,
            std::size(wave32_fragment_scalar_cselect_data_cfg)).empty()) {
        printf("  [FAIL] graphics CFG retained compute-only scalar-cselect VCC semantics\n");
        return 1;
    }
    printf("  [ok]   scalar-cselect VCC bridge remains compute-only in graphics CFGs\n");

    // Astro's larger world-map sibling keeps an ordinary scalar in VCC_HI while an implicit VOPC
    // compare replaces the complete Wave32 predicate in VCC_LO. The unused high word must retain
    // its data lifetime; the same sequence is intentionally invalid in Wave64, where VCC_HI is the
    // upper half of the compare mask and is therefore overwritten.
    const uint32_t wave32_compute_vopc_preserves_vcc_hi_data[] = {
        0xbeeb03ffu, 0x3f400000u,            // s_mov_b32 vcc_hi, 0.75f
        0x7d841680u,                         // v_cmp_eq_u32 vcc, 0, v11
        0x7e4e026bu,                         // v_mov_b32 v39, vcc_hi
        0xbf810000u,
    };
    if (recompile_compute(wave32_compute_vopc_preserves_vcc_hi_data,
                          std::size(wave32_compute_vopc_preserves_vcc_hi_data), nullptr,
                          wave32_compute_config).empty()) {
        printf("  [FAIL] implicit Wave32 VOPC clobbered adjacent VCC_HI scalar data\n");
        return 1;
    }
    ComputeShaderConfig wave64_compute_config = wave32_compute_config;
    wave64_compute_config.wave_size = 64;
    wave64_compute_config.native_subgroup_size = 64;
    wave64_compute_config.local_x = 64;
    // In Wave64 the VOPC overwrites the WHOLE of VCC, so the 0.75f placed in VCC_HI is gone and the
    // later read must NOT return it. That requirement is unchanged. What changed is the outcome: this
    // used to be expressed as "the shader must reject", because a per-invocation bool could not
    // express what VCC_HI now holds. It can — after a Wave64 v_cmp, VCC_HI is exactly lanes 32..63 of
    // the compare mask, and `subgroupBallot(vcc).y` is exactly those bits when the subgroup IS the
    // guest wave, which this config establishes (native_subgroup_size = 64). See #2420.
    //
    // Asserting the BALLOT OPCODE rather than mere non-emptiness is what pins the distinction: a
    // module that compiled by some other route — in particular by resurrecting the stale scalar, the
    // exact defect this case exists to catch — would not contain it.
    const std::vector<uint32_t> wave64_vcc_hi = recompile_compute(
        wave32_compute_vopc_preserves_vcc_hi_data,
        std::size(wave32_compute_vopc_preserves_vcc_hi_data), nullptr, wave64_compute_config);
    auto has_opcode = [](const std::vector<uint32_t>& m, uint32_t op) {
        for (size_t i = 5; i < m.size();) {
            const uint32_t len = m[i] >> 16;
            if (len == 0 || i + len > m.size()) break;
            if ((m[i] & 0xffffu) == op) return true;
            i += len;
        }
        return false;
    };
    if (wave64_vcc_hi.empty() || !has_opcode(wave64_vcc_hi, 339)) {   // OpGroupNonUniformBallot
        printf("  [FAIL] Wave64 VCC_HI read did not materialise the mask half via subgroupBallot\n");
        return 1;
    }
    printf("  [ok]   implicit VOPC preserves VCC_HI scalar data only in Wave32\n");

    // GTA V's Wave64 compute kernel 0x205b5e8600 negates the current EXEC mask as a scalar integer
    // pair. The subtraction words below are its exact pc1310/1312 packets: subtract EXEC_LO from
    // zero, then subtract EXEC_HI and the low-word borrow from zero. An exact 64-lane subgroup can
    // materialise each EXEC half with subgroupBallot; a portable or mismatched subgroup must keep
    // rejecting rather than silently treating the architectural mask as zero.
    const uint32_t gta_wave64_exec_integer_negate[] = {
        0xbefe04c1u,                         // establish a complete live EXEC mask
        0x80907e80u,                         // pc1310: s_sub_u32 s16, 0, exec_lo
        0x82917f80u,                         // pc1312: s_subb_u32 s17, 0, exec_hi
        0x7e040210u,                         // v_mov_b32 v2, s16 (keep the result live)
        0x7e060211u,                         // v_mov_b32 v3, s17
        0xbf810000u,
    };
    ComputeShaderConfig gta_wave64_exec_config = wave64_compute_config;
    const std::vector<uint32_t> gta_wave64_exec_spv = recompile_compute(
        gta_wave64_exec_integer_negate, std::size(gta_wave64_exec_integer_negate), nullptr,
        gta_wave64_exec_config);
    if (gta_wave64_exec_spv.empty() || !has_opcode(gta_wave64_exec_spv, 339)) {
        printf("  [FAIL] GTA Wave64 EXEC integer negation did not materialise the mask via subgroupBallot\n");
        return 1;
    }
    ComputeShaderConfig gta_portable_exec_config = gta_wave64_exec_config;
    gta_portable_exec_config.native_subgroup_size = 0;
    if (!recompile_compute(gta_wave64_exec_integer_negate,
                           std::size(gta_wave64_exec_integer_negate), nullptr,
                           gta_portable_exec_config).empty()) {
        printf("  [FAIL] portable GTA EXEC integer negation compiled without an exact subgroup\n");
        return 1;
    }
    ComputeShaderConfig gta_mismatched_exec_config = gta_wave64_exec_config;
    gta_mismatched_exec_config.native_subgroup_size = 32;
    if (!recompile_compute(gta_wave64_exec_integer_negate,
                           std::size(gta_wave64_exec_integer_negate), nullptr,
                           gta_mismatched_exec_config).empty()) {
        printf("  [FAIL] GTA EXEC integer negation compiled for a mismatched subgroup\n");
        return 1;
    }
    printf("  [ok]   GTA Wave64 EXEC integer negation requires and uses an exact-wave ballot\n");

    // Astro's exact PC458 packet explicitly selects physical VCC_HI in Wave32. A typed B32 mask in
    // that word must drive the select independently of VCC_LO; absent or path-dependent HI mask
    // lifetimes must remain fail-visible instead of falling back to the implicit VCC predicate.
    const uint32_t wave32_compute_explicit_vcc_hi_cndmask[] = {
        0xbf060000u,                         // s_cmp_eq_u32 s0,s0
        0x856b807eu,                         // s_cselect_b32 vcc_hi, exec_lo, 0
        0xd5010000u, 0x01ad0280u,            // v_cndmask_b32_e64 v0, 0, 1, vcc_hi
        0xbf810000u,
    };
    const auto explicit_vcc_hi_cndmask_spv = recompile_compute(
        wave32_compute_explicit_vcc_hi_cndmask,
        std::size(wave32_compute_explicit_vcc_hi_cndmask), nullptr,
        wave32_compute_config);
    if (explicit_vcc_hi_cndmask_spv.empty() ||
        !type_result_ids_are_nonzero(explicit_vcc_hi_cndmask_spv, nullptr) ||
        !phi_ids_are_nonzero(explicit_vcc_hi_cndmask_spv)) {
        printf("  [FAIL] explicit Wave32 VCC_HI cndmask source was not preserved\n");
        return 1;
    }
    const uint32_t wave32_compute_absent_vcc_hi_cndmask[] = {
        0xbeeb0380u,                         // scalar-data vcc_hi=0, not a mask
        0xd5010000u, 0x01ad0280u,            // v_cndmask_b32_e64 v0, 0, 1, vcc_hi
        0xbf810000u,
    };
    if (!recompile_compute(wave32_compute_absent_vcc_hi_cndmask,
                           std::size(wave32_compute_absent_vcc_hi_cndmask), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] explicit VCC_HI cndmask accepted an absent mask lifetime\n");
        return 1;
    }
    const uint32_t wave32_compute_ambiguous_vcc_hi_cndmask[] = {
        0xbf060000u,                         // pc0: scalar branch condition
        0xbf840001u,                         // pc1: one edge skips the HI definition
        0x856b807eu,                         // pc2: other edge defines a VCC_HI mask
        0xd5010000u, 0x01ad0280u,            // pc3: joined VCC_HI read
        0xbf810000u,
    };
    if (!recompile_compute(wave32_compute_ambiguous_vcc_hi_cndmask,
                           std::size(wave32_compute_ambiguous_vcc_hi_cndmask), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] explicit VCC_HI cndmask accepted a path-dependent mask lifetime\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VCC_HI cndmask requires its own unambiguous mask\n");

    // GTA V's live Wave32 kernel reaches the exact v_add_co_u32 packet below after joining a path
    // where s0 is a VOPC mask with one where it is scalar data. The instruction has no carry-in:
    // dword1's zero SRC2 field is reserved/non-operand bits, not an architectural read of s0. It
    // replaces that ambiguous lifetime with a fresh carry in s0, which the exact later _co_ci_
    // packet legitimately consumes. The irreducible tail forces the same portable CFG dispatcher
    // that diagnosed the production kernel at this site.
    const uint32_t wave32_compute_vop3b_two_source_join[] = {
        0x7d8a06f9u, 0x06868080u,            // pc0: exact VOPC SDWA packet -> s0 mask
        0xbf060000u,                         // pc2: scalar branch condition
        0xbf840001u,                         // pc3: one edge retains the s0 mask
        0xbe800380u,                         // pc4: other edge replaces s0 with scalar data
        0xd70f0016u, 0x00021f1bu,            // pc5: exact v_add_co_u32 v22,s0,v27,v15
        0xd5286a17u, 0x00024080u,            // pc7: exact v_add_co_ci_u32 consumes s0
        0x7e040280u,                         // pc9: irreducible crossing-CFG tail
        0x7c020300u,                         // pc10: v_cmp_lt_u32_e32 vcc, v0, v1
        0xbf860001u,                         // pc11: s_cbranch_vccz -> pc13
        0x7e040281u,                         // pc12: v_mov_b32_e32 v2, 1
        0x7d840100u,                         // pc13: v_cmp_eq_u32_e32 vcc, v0, v0
        0xbf870001u,                         // pc14: s_cbranch_vccnz -> pc16
        0xbf82fffdu,                         // pc15: s_branch -> pc13
        0x7e040d02u,                         // pc16: v_mov_b32_e32 v2, v2
        0xbf810000u,
    };
    ComputeShaderConfig wave32_vop3b_two_source_config;
    wave32_vop3b_two_source_config.wave_size = 32;
    wave32_vop3b_two_source_config.local_x = 64;
    wave32_vop3b_two_source_config.native_subgroup_size = 0;
    const auto vop3b_two_source_join_spv = recompile_compute(
        wave32_compute_vop3b_two_source_join,
        std::size(wave32_compute_vop3b_two_source_join), nullptr,
        wave32_vop3b_two_source_config);
    if (vop3b_two_source_join_spv.empty() || !has_opcode(vop3b_two_source_join_spv, 251) ||
        !type_result_ids_are_nonzero(vop3b_two_source_join_spv, nullptr) ||
        !phi_ids_are_nonzero(vop3b_two_source_join_spv)) {
        printf("  [FAIL] Wave32 CFG treated two-source VOP3B carry output as reading SRC2\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG admits exact two-source VOP3B carry packet at an s0 join\n");

    // GTA V's rejected Wave32 compute siblings produce a fresh VCC_LO carry at PC79 with the
    // no-carry-in VOP3B family, then consume it at PC83 through implicit VCC. A later crossing CFG
    // forces the portable arbitrary-CFG dispatcher, which must carry that one-word mask lifetime
    // through its production dataflow instead of classifying the VOP3B SDST as a two-word mask.
    // A second fresh carry targets ordinary s4 and narrows/restores EXEC through the explicit B32
    // saveexec domain before another implicit consumer. That operation pins the emitter's runtime
    // `sreg_bool_b32` state: merely updating static dispatcher provenance or the legacy `rs.vcc`
    // shortcut cannot pass it.
    const uint32_t wave32_compute_fresh_carry_cfg[] = {
        0x7d840000u,                         // pc0: one incoming VCC mask lifetime
        0xbf060000u,                         // pc1: scalar branch condition
        0xbf840001u,                         // pc2: one edge skips the scalar-data lifetime
        0xbeea0385u,                         // pc3: other edge recycles vcc_lo as scalar data
        0x7e0202c1u,                         // pc4: v_mov_b32_e32 v1, -1
        0xd70f6a02u, 0x00020300u,            // pc5: v_add_co_u32 v2,vcc_lo,v0,v1
        0x50060080u,                         // pc7: v_add_co_ci_u32 v3,vcc_lo,0,v0,vcc_lo
        0xd70f0402u, 0x00020300u,            // pc8: fresh carry -> ordinary s4 B32 mask
        0xbe863c04u,                         // pc10: s_and_saveexec_b32 s6, s4
        0xbefe0306u,                         // pc11: s_mov_b32 exec_lo, s6 (restore)
        0xbeea0304u,                         // pc12: s_mov_b32 vcc_lo, s4 (explicit mask copy)
        0x50060080u,                         // pc13: consume the copied mask through implicit VCC
        0x7e040280u,                         // pc14: v_mov_b32_e32 v2, 0
        0x7c020300u,                         // pc15: v_cmp_lt_u32_e32 vcc, v0, v1
        0xbf860001u,                         // pc16: s_cbranch_vccz -> pc18
        0x7e040281u,                         // pc17: v_mov_b32_e32 v2, 1
        0x7d840100u,                         // pc18: v_cmp_eq_u32_e32 vcc, v0, v0
        0xbf870001u,                         // pc19: s_cbranch_vccnz -> pc21
        0xbf82fffdu,                         // pc20: s_branch -> pc18
        0x7e040d02u,                         // pc21: v_mov_b32_e32 v2, v2
        0xbf810000u,
    };
    ComputeShaderConfig wave32_carry_cfg_config;
    wave32_carry_cfg_config.wave_size = 32;
    wave32_carry_cfg_config.local_x = 64;
    wave32_carry_cfg_config.native_subgroup_size = 0;
    const auto fresh_carry_cfg_spv = recompile_compute(
        wave32_compute_fresh_carry_cfg, std::size(wave32_compute_fresh_carry_cfg), nullptr,
        wave32_carry_cfg_config);
    if (fresh_carry_cfg_spv.empty() || !has_opcode(fresh_carry_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(fresh_carry_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(fresh_carry_cfg_spv)) {
        printf("  [FAIL] Wave32 CFG lost a fresh VOP3B carry before its implicit VCC consumer\n");
        return 1;
    }
    std::vector<uint32_t> fresh_carry_wrong_sdst(
        std::begin(wave32_compute_fresh_carry_cfg),
        std::end(wave32_compute_fresh_carry_cfg));
    fresh_carry_wrong_sdst[5] = 0xd70f0402u; // same producer writes s4, not consumed VCC_LO
    if (!recompile_compute(fresh_carry_wrong_sdst.data(), fresh_carry_wrong_sdst.size(), nullptr,
                           wave32_carry_cfg_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted an implicit VCC consumer without its carry producer\n");
        return 1;
    }
    std::vector<uint32_t> fresh_carry_tracked_wrong_sdst(
        std::begin(wave32_compute_fresh_carry_cfg),
        std::end(wave32_compute_fresh_carry_cfg));
    fresh_carry_tracked_wrong_sdst[8] = 0xd70f0602u; // producer writes s6; copy still reads s4
    if (!recompile_compute(fresh_carry_tracked_wrong_sdst.data(),
                           fresh_carry_tracked_wrong_sdst.size(), nullptr,
                           wave32_carry_cfg_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted an explicit B32 copy from the wrong carry SDST\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG tracks fresh VOP3B carry output at its exact SDST\n");

    // Exact control/data lifetime from Astro Bot's world-map phase, reduced only to its register-
    // independent instructions: a scalar-data VCC_LO lifetime feeds CMPX (which changes EXEC but
    // preserves VCC), one crossing arm defines and immediately consumes a fresh VCC mask, and the
    // other arm skips that definition. The rejoined VCC lifetime is invalid but dead. Crossing
    // EXECZ regions force the same arbitrary-CFG dispatcher as the 216..306 production phase.
    const uint32_t wave32_compute_dead_vcc_join[] = {
        0xbeea0385u,                         // pc0: scalar-data vcc_lo=5
        0x7da40100u,                         // pc1: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc2: execz -> pc6, skipping the VCC definition
        0x7d840100u,                         // pc3: v_cmp_eq_u32 vcc,v0,v0 (fresh mask)
        0x02020100u,                         // pc4: v_cndmask_b32 v1,v0,v0,vcc
        0xbf880002u,                         // pc5: execz -> pc8 (crosses the pc2 region)
        0xbf060000u,                         // pc6: rejoin; scalar compare, no VCC read
        0xbf850001u,                         // pc7: third branch -> pc9 forces complex CFG
        0x7e040280u,                         // pc8: no mask read before termination
        0xbf810000u,
    };
    const auto dead_vcc_join_spv = recompile_compute(
        wave32_compute_dead_vcc_join, std::size(wave32_compute_dead_vcc_join), nullptr,
        wave32_compute_config);
    if (dead_vcc_join_spv.empty() || !has_opcode(dead_vcc_join_spv, 251) ||
        !type_result_ids_are_nonzero(dead_vcc_join_spv, nullptr) ||
        !phi_ids_are_nonzero(dead_vcc_join_spv)) {
        printf("  [FAIL] Wave32 CFG rejected Astro's dead VCC lifetime at a crossing join\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG carries Astro's invalid-but-dead VCC lifetime across a join\n");

    std::vector<uint32_t> dead_vcc_read_at_join(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_read_at_join[6] = 0x02040500u; // v_cndmask_b32 v2,v0,v2,vcc
    if (!recompile_compute(dead_vcc_read_at_join.data(), dead_vcc_read_at_join.size(),
                           nullptr, wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a VCC read after the ambiguous join\n");
        return 1;
    }
    std::vector<uint32_t> dead_vcc_missing_definition(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_missing_definition[3] = 0x7da40100u; // CMPX does not define VCC
    if (!recompile_compute(dead_vcc_missing_definition.data(),
                           dead_vcc_missing_definition.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a cndmask after replacing its VCC definition\n");
        return 1;
    }
    std::vector<uint32_t> dead_vcc_entered_consumer(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_entered_consumer[2] = 0xbf880001u; // topology now enters pc4, past the definition
    if (!recompile_compute(dead_vcc_entered_consumer.data(),
                           dead_vcc_entered_consumer.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a branch edge entering past the VCC definition\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG rejects VCC-read, definition, and branch-topology deviations\n");

    // Reduced lifetime from Astro Bot's world-map compute PC396/972/1062/1063/2311/2312. An
    // earlier s_mov_b64 makes s[32:33] a saved mask, a dominating multiword SMEM data load ends that
    // lifetime, and s35 then remains numeric on both entries to the CMPX loop header. The crossing
    // loops force the arbitrary-CFG dispatcher used by the production shader.
    const uint32_t wave32_compute_recycled_b64_before_loop[] = {
        0xbea0047eu,                         // pc0:  s_mov_b64 s[32:33], exec
        0xf4280800u, 0xfa000000u,            // pc1:  s_buffer_load_dwordx4 s[32:35],s[0:3],0
        0xbea30320u,                         // pc3:  s_mov_b32 s35,s32
        0x7d820023u,                         // pc4:  CMPX consumes numeric s35 (reduced v0 source)
        0xbf880006u,                         // pc5:  A exit -> pc12
        0x7d820023u,                         // pc6:  crossing B header consumes numeric s35
        0xbf880004u,                         // pc7:  B exit -> pc12
        0x81238123u,                         // pc8:  s_sub_u32 s35,s35,1
        0xbf82fffau,                         // pc9:  backedge -> pc4
        0x81238123u,                         // pc10: unreachable second decrement
        0xbf82fffau,                         // pc11: syntactic crossing backedge -> pc6
        0xbf810000u,                         // pc12: s_endpgm
    };
    ShaderResourceTable recycled_b64_resources;
    ShaderResource recycled_b64_cbuf;
    recycled_b64_cbuf.cls = ResourceClass::ConstantBuffer;
    recycled_b64_cbuf.format = DataFormat::Uint32;
    recycled_b64_cbuf.num_components = 1;
    recycled_b64_cbuf.binding = 2;
    recycled_b64_cbuf.sgpr_base = 0;
    recycled_b64_cbuf.size = 64;
    recycled_b64_resources.resources.push_back(recycled_b64_cbuf);
    ComputeShaderConfig recycled_b64_config = wave32_compute_config;
    recycled_b64_config.user_sgprs.resize(4);
    const auto recycled_b64_loop_spv = recompile_compute(
        wave32_compute_recycled_b64_before_loop,
        std::size(wave32_compute_recycled_b64_before_loop), &recycled_b64_resources,
        recycled_b64_config);
    if (recycled_b64_loop_spv.empty() || !has_opcode(recycled_b64_loop_spv, 251) ||
        !type_result_ids_are_nonzero(recycled_b64_loop_spv, nullptr) ||
        !phi_ids_are_nonzero(recycled_b64_loop_spv)) {
        printf("  [FAIL] dominating scalar overwrite did not end a stale B64 mask lifetime\n");
        return 1;
    }

    std::vector<uint32_t> recycled_b64_missing_overwrite(
        std::begin(wave32_compute_recycled_b64_before_loop),
        std::end(wave32_compute_recycled_b64_before_loop));
    recycled_b64_missing_overwrite[1] = 0xbf800000u; // remove both dominating data writes
    recycled_b64_missing_overwrite[2] = 0xbf800000u;
    if (!recompile_compute(recycled_b64_missing_overwrite.data(),
                           recycled_b64_missing_overwrite.size(), &recycled_b64_resources,
                           recycled_b64_config).empty()) {
        printf("  [FAIL] B64-mask first entry joined numeric loop backedge without rejection\n");
        return 1;
    }

    const uint32_t wave32_compute_b64_overwrite_bypass[] = {
        0xbea0047eu,                         // pc0:  saved B64 mask in s[32:33]
        0xbf060000u,                         // pc1:  scalar branch condition
        0xbf840002u,                         // pc2:  one edge bypasses both data writes -> pc5
        0xf4280800u, 0xfa000000u,            // pc3:  s_buffer_load_dwordx4 s[32:35],s[0:3],0
        0xbea30320u,                         // pc5:  ambiguous s32 -> s35 copy
        0x7d820023u,                         // pc6:  numeric CMPX consumption (reduced v0 source)
        0xbf880006u,                         // pc7:  A exit -> pc14
        0x7d820023u,                         // pc8:  crossing B header
        0xbf880004u,                         // pc9:  B exit -> pc14
        0x81238123u,                         // pc10: numeric decrement
        0xbf82fffau,                         // pc11: backedge -> pc6
        0x81238123u,                         // pc12: unreachable second decrement
        0xbf82fffau,                         // pc13: syntactic crossing backedge -> pc8
        0xbf810000u,                         // pc14: s_endpgm
    };
    if (!recompile_compute(wave32_compute_b64_overwrite_bypass,
                           std::size(wave32_compute_b64_overwrite_bypass),
                           &recycled_b64_resources, recycled_b64_config).empty()) {
        printf("  [FAIL] branch bypass of the B64-to-data overwrite was accepted\n");
        return 1;
    }

    std::vector<uint32_t> recycled_b64_mask_backedge(
        std::begin(wave32_compute_recycled_b64_before_loop),
        std::end(wave32_compute_recycled_b64_before_loop));
    recycled_b64_mask_backedge[8] = 0xbea3037eu; // reachable backedge keeps a real mask in s35
    if (!recompile_compute(recycled_b64_mask_backedge.data(),
                           recycled_b64_mask_backedge.size(), &recycled_b64_resources,
                           recycled_b64_config).empty()) {
        printf("  [FAIL] numeric first entry joined a real-mask backedge without rejection\n");
        return 1;
    }
    printf("  [ok]   Wave32 B64 mask lifetimes end only on dominating scalar-data overwrites\n");

    // The production shader wraps barrier-separated work in a workgroup-uniform early-out. Its
    // first barrier-free phase has the same crossing Wave32 CFG as the fixture above and no guest
    // S_ENDPGM of its own. The phase splitter must supply an emitter-only terminator so the
    // dispatcher can become inactive, rejoin the uniform outer shell, and reach the guest barrier.
    const uint32_t wave32_compute_guarded_cfg_phase[] = {
        0xbf060000u,                         // pc0: uniform s_cmp_eq_u32 s0,s0
        0xbf84000cu,                         // pc1: early-out -> terminal pc14
        0xbeea0385u,                         // pc2: scalar-data vcc_lo=5
        0x7da40100u,                         // pc3: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc4: execz -> pc8
        0x7d840100u,                         // pc5: fresh VCC definition
        0x02020100u,                         // pc6: consume the fresh VCC
        0xbf880002u,                         // pc7: execz -> pc10
        0xbf060000u,                         // pc8: rejoin without reading VCC
        0xbf850001u,                         // pc9: branch -> pc11, skipping pc10
        0x7e040280u,                         // pc10: first arm
        0x7e060280u,                         // pc11: phase-local join
        0xbf8a0000u,                         // pc12: uniform guest barrier
        0x7e080280u,                         // pc13: tail phase
        0xbf810000u,                         // pc14: terminal s_endpgm
    };
    const auto guarded_cfg_phase_spv = recompile_compute(
        wave32_compute_guarded_cfg_phase, std::size(wave32_compute_guarded_cfg_phase), nullptr,
        wave32_compute_config);
    if (guarded_cfg_phase_spv.empty() || !has_opcode(guarded_cfg_phase_spv, 251) ||
        !type_result_ids_are_nonzero(guarded_cfg_phase_spv, nullptr) ||
        !phi_ids_are_nonzero(guarded_cfg_phase_spv)) {
        printf("  [FAIL] barrier phase rejected Astro's crossing Wave32 CFG\n");
        return 1;
    }
    printf("  [ok]   barrier phase terminates and rejoins Astro's crossing Wave32 CFG\n");

    // The terminal workgroup guard may have unrelated lane-local setup before its scalar condition.
    // Only the backwards slice feeding SCC matters: this vector write cannot affect s0 or the SOPC,
    // so all invocations still enter/skip the barrier sequence together. The phase-local crossing
    // CFG is the same dispatcher-requiring shape as above.
    const uint32_t wave32_compute_vector_prefixed_guarded_phase[] = {
        0x7e000280u,                         // pc0: unrelated v_mov_b32 v0,0
        0xbf060000u,                         // pc1: workgroup-uniform s_cmp_eq_u32 s0,s0
        0xbf84000cu,                         // pc2: early-out -> terminal pc15
        0xbeea0385u,                         // pc3: scalar-data vcc_lo=5
        0x7da40100u,                         // pc4: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc5: execz -> pc9
        0x7d840100u,                         // pc6: fresh VCC definition
        0x02020100u,                         // pc7: consume the fresh VCC
        0xbf880002u,                         // pc8: execz -> pc11
        0xbf060000u,                         // pc9: rejoin without reading VCC
        0xbf850001u,                         // pc10: branch -> pc12, skipping pc11
        0x7e040280u,                         // pc11: first arm
        0x7e060280u,                         // pc12: phase-local join
        0xbf8a0000u,                         // pc13: uniform guest barrier
        0x7e080280u,                         // pc14: tail phase
        0xbf810000u,                         // pc15: terminal s_endpgm
    };
    const auto vector_prefixed_guarded_spv = recompile_compute(
        wave32_compute_vector_prefixed_guarded_phase,
        std::size(wave32_compute_vector_prefixed_guarded_phase), nullptr,
        wave32_compute_config);
    if (vector_prefixed_guarded_spv.empty() || !has_opcode(vector_prefixed_guarded_spv, 251) ||
        !type_result_ids_are_nonzero(vector_prefixed_guarded_spv, nullptr) ||
        !phi_ids_are_nonzero(vector_prefixed_guarded_spv)) {
        printf("  [FAIL] lane-local setup hid a scalar-uniform terminal barrier guard\n");
        return 1;
    }

    // A scalar produced by V_READFIRSTLANE can differ between guest waves. Feeding it into the same
    // terminal SCC guard would place the guest barrier under workgroup-divergent control flow, so the
    // relaxed prefix analysis must remain fail-closed.
    std::vector<uint32_t> lane_guarded_phase(
        std::begin(wave32_compute_vector_prefixed_guarded_phase),
        std::end(wave32_compute_vector_prefixed_guarded_phase));
    lane_guarded_phase[0] = 0x7e000500u;     // v_readfirstlane_b32 s0,v0
    lane_guarded_phase[1] = 0xbf068000u;     // s_cmp_eq_u32 s0,0
    if (!recompile_compute(lane_guarded_phase.data(), lane_guarded_phase.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] lane-derived SCC admitted a divergent terminal barrier guard\n");
        return 1;
    }

    // A uniform literal is not a uniform reaching definition when a lane-derived wave branch can
    // skip it. The textual backwards slice must not cross that branch and place the barrier under a
    // condition that can differ between guest waves.
    const uint32_t conditionally_written_guarded_phase[] = {
        0x7d840300u,                         // pc0: v_cmp_eq_u32 vcc,v0,v1
        0xbf860001u,                         // pc1: vccz -> pc3, skipping the scalar write
        0xbe800380u,                         // pc2: s_mov_b32 s0,0
        0xbf068000u,                         // pc3: s_cmp_eq_u32 s0,0
        0xbf84000cu,                         // pc4: early-out -> terminal pc17
        0xbeea0385u,                         // pc5: scalar-data vcc_lo=5
        0x7da40100u,                         // pc6: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc7: execz -> pc11
        0x7d840100u,                         // pc8: fresh VCC definition
        0x02020100u,                         // pc9: consume the fresh VCC
        0xbf880002u,                         // pc10: execz -> pc13
        0xbf060000u,                         // pc11: rejoin without reading VCC
        0xbf850001u,                         // pc12: branch -> pc14, skipping pc13
        0x7e040280u,                         // pc13: first arm
        0x7e060280u,                         // pc14: phase-local join
        0xbf8a0000u,                         // pc15: guest barrier
        0x7e080280u,                         // pc16: tail phase
        0xbf810000u,                         // pc17: terminal s_endpgm
    };
    if (!recompile_compute(conditionally_written_guarded_phase,
                           std::size(conditionally_written_guarded_phase), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] barrier guard slice crossed a lane-derived conditional write\n");
        return 1;
    }
    std::vector<uint32_t> fork_guarded_phase(
        std::begin(conditionally_written_guarded_phase),
        std::end(conditionally_written_guarded_phase));
    fork_guarded_phase[1] = 0xbf830001u; // s_cbranch_i_fork also invalidates textual dominance
    if (!recompile_compute(fork_guarded_phase.data(), fork_guarded_phase.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] barrier guard slice crossed s_cbranch_i_fork\n");
        return 1;
    }
    printf("  [ok]   barrier guard slicing ignores unrelated VALU but rejects lane-derived SCC\n");

    ComputeShaderConfig portable_wave64_compute_config;
    portable_wave64_compute_config.local_x = 128;
    portable_wave64_compute_config.wave_size = 64;

    // GTA V's two live compute failures use this exact in-place V_ADD_NC_U32 DPP row ladder. Two
    // alternate copies let separate guest waves park at different static events, while the appended
    // nested/backward CFG forces the portable dispatcher. The host-independent lowering must use
    // workgroup scratch rather than assuming a Wave64-capable host subgroup.
    std::vector<uint32_t> gta_compute_dpp_add_cfg = {
        0x7e280281u,                         // pc0:  v_mov_b32 v20,1
        0x7d840100u,                         // pc1:  v_cmp_eq_u32 vcc,v0,v0
        0xbf860009u,                         // pc2:  alternate ladder -> pc12
        0x4a2828fau, 0xff011114u,            // pc3:  event 1 row_shr:1
        0x4a2828fau, 0xff011214u,            // pc5:  event 2 row_shr:2
        0x4a2828fau, 0xff011414u,            // pc7:  event 3 row_shr:4
        0x4a2828fau, 0xff011814u,            // pc9:  event 4 row_shr:8
        0xbf820008u,                         // pc11: join -> pc20
        0x4a2828fau, 0xff011114u,            // pc12: event 5 row_shr:1
        0x4a2828fau, 0xff011214u,            // pc14: event 6 row_shr:2
        0x4a2828fau, 0xff011414u,            // pc16: event 7 row_shr:4
        0x4a2828fau, 0xff011814u,            // pc18: event 8 row_shr:8
        // Reduced nested/backward compute CFG from the existing dispatcher coverage fixture.
        0xbe800380u, 0x7e000280u, 0x7e020300u,
        0xd7610013u, 0x00014a7eu, 0xd7610013u, 0x0001507fu,
        0xd760000eu, 0x00014b13u, 0xd760000fu, 0x00015113u, 0xbefe040eu,
        0xe00c2000u, 0x80020400u, 0x7db900f9u, 0x86050007u,
        0x7d020200u, 0xbf860006u, 0xbf0a8204u, 0x360000fdu, 0xbf840001u,
        0x81008100u, 0x81008100u, 0xbf82fff4u, 0xbf810000u,
    };
    ShaderResourceTable gta_compute_dpp_table;
    ShaderResource gta_compute_dpp_vb{};
    gta_compute_dpp_vb.cls = ResourceClass::VertexBuffer;
    gta_compute_dpp_vb.binding = 3;
    gta_compute_dpp_vb.sgpr_base = 8;
    gta_compute_dpp_vb.stride = 16;
    gta_compute_dpp_vb.format = DataFormat::Float32;
    gta_compute_dpp_vb.num_components = 4;
    gta_compute_dpp_table.resources.push_back(gta_compute_dpp_vb);
    const auto gta_compute_dpp_spv = recompile_compute(
        gta_compute_dpp_add_cfg.data(), gta_compute_dpp_add_cfg.size(),
        &gta_compute_dpp_table, portable_wave64_compute_config);
    if (gta_compute_dpp_spv.empty() || !has_opcode(gta_compute_dpp_spv, 251) ||
        !compute_dpp_add_row_shr_updates_dispatch_vgpr(gta_compute_dpp_spv, 128, 20) ||
        !type_result_ids_are_nonzero(gta_compute_dpp_spv, nullptr) ||
        !phi_ids_are_nonzero(gta_compute_dpp_spv)) {
        printf("  [FAIL] portable Wave64 CFG did not event-isolate GTA V DPP add ladders\n");
        return 1;
    }

    ComputeShaderConfig native_gta_compute_dpp_config = portable_wave64_compute_config;
    native_gta_compute_dpp_config.local_x = 64;
    native_gta_compute_dpp_config.native_subgroup_size = 64;
    const auto native_gta_compute_dpp_spv = recompile_compute(
        gta_compute_dpp_add_cfg.data(), gta_compute_dpp_add_cfg.size(),
        &gta_compute_dpp_table, native_gta_compute_dpp_config);
    if (native_gta_compute_dpp_spv.empty() ||
        !has_opcode(native_gta_compute_dpp_spv, 345) ||
        !compute_dpp_add_native_row_shr_updates_dispatch_vgpr(
            native_gta_compute_dpp_spv) ||
        !type_result_ids_are_nonzero(native_gta_compute_dpp_spv, nullptr) ||
        !phi_ids_are_nonzero(native_gta_compute_dpp_spv)) {
        printf("  [FAIL] native Wave64 CFG rejected GTA V DPP add ladders\n");
        return 1;
    }

    std::vector<uint32_t> gta_compute_dpp_distinct_source = gta_compute_dpp_add_cfg;
    std::vector<uint32_t> gta_compute_dpp_bound_ctrl = gta_compute_dpp_add_cfg;
    for (size_t word : {4u, 6u, 8u, 10u, 13u, 15u, 17u, 19u}) {
        gta_compute_dpp_distinct_source[word] =
            (gta_compute_dpp_distinct_source[word] & ~0xffu) | 0x15u; // SRC0=v21
        gta_compute_dpp_bound_ctrl[word] |= 1u << 19u;
    }
    if (!recompile_compute(gta_compute_dpp_distinct_source.data(),
                           gta_compute_dpp_distinct_source.size(),
                           &gta_compute_dpp_table,
                           portable_wave64_compute_config).empty() ||
        !recompile_compute(gta_compute_dpp_bound_ctrl.data(),
                           gta_compute_dpp_bound_ctrl.size(),
                           &gta_compute_dpp_table,
                           portable_wave64_compute_config).empty()) {
        printf("  [FAIL] GTA V DPP add contract admitted distinct-source or BC1 mutations\n");
        return 1;
    }
    printf("  [ok]   GTA V compute DPP add ladders preserve row direction, events, and BC0 VDST\n");

    // Syberia source 87 is a real 960x544 dispatch whose only generic-coverage rejects are eight
    // v_max3_f16 instructions. Keep the first exact low/high pair: OPSEL chooses different source
    // halves for each result and the second instruction writes the high destination half. The decode
    // test pins those bitmasks; require both exact packet shapes here, and let the Vulkan test
    // independently check their packed result.
    const uint32_t syberia_max3_f16_pair[] = {
        0x7e2c02ffu, 0x40003c00u,            // v_mov_b32 v22,{2.0h,1.0h}
        0x7e2e02ffu, 0x38004200u,            // v_mov_b32 v23,{0.5h,3.0h}
        0x7e1202ffu, 0x12345678u,            // v_mov_b32 v9,old packed destination
        0xd7542009u, 0x045a2d17u,            // max3_f16 v9,v23.lo,v22.lo,v22.hi -> v9.lo
        0xd7545809u, 0x045e2d17u,            // max3_f16 v9,v23.hi,v22.hi,v23.lo -> v9.hi
        0xbf810000u,
    };
    const auto syberia_max3_f16_spv = recompile_compute(
        syberia_max3_f16_pair, std::size(syberia_max3_f16_pair), nullptr,
        portable_wave64_compute_config);
    if (syberia_max3_f16_spv.empty() ||
        !type_result_ids_are_nonzero(syberia_max3_f16_spv, nullptr) ||
        !phi_ids_are_nonzero(syberia_max3_f16_spv)) {
        printf("  [FAIL] Syberia v_max3_f16 OPSEL pair did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   Syberia v_max3_f16 OPSEL pair emits valid SPIR-V\n");

    // Exercise the ordinary integer-B64 emitter separately from the mask-domain vote below.
    // Exact llvm-mc gfx1010 encoding: v_cmp_gt_u64_e64 vcc_lo,s[0:1],0.
    const uint32_t wave64_u64_compare[] = {
        0xd4e4006au, 0x00010000u,
        0xbf810000u,
    };
    const auto wave64_u64_compare_spv = recompile_compute(
        wave64_u64_compare, std::size(wave64_u64_compare), nullptr,
        portable_wave64_compute_config);
    if (wave64_u64_compare_spv.empty() ||
        !type_result_ids_are_nonzero(wave64_u64_compare_spv, nullptr)) {
        printf("  [FAIL] e64 unsigned-B64 comparison did not lower structurally\n");
        return 1;
    }

    // Exact reduced Astro Wave64 shape from PC478..1195: an e64 unsigned-B64 comparison broadcasts
    // whether VCC contains any live lane, then a later s_and_b64 publishes SCC=(result mask != 0).
    // Ordinary VALU can be scheduled between the mask producer and its SCC branch. Crossing SCC/
    // EXEC regions force the portable arbitrary-CFG dispatcher, whose common phase must perform
    // both guest-wave votes without placing a workgroup barrier inside one dynamic switch arm.
    const uint32_t wave64_compute_live_b64_mask_scc[] = {
        0x7d840000u,                         // pc0: v_cmp_eq_u32 vcc,v0,v0
        0xd4e4006au, 0x0001006au,            // pc1: v_cmp_gt_u64_e64 vcc,vcc,0
        0xbea0047eu,                         // pc3: s_mov_b64 s[32:33],exec
        0x87ea6a20u,                         // pc4: s_and_b64 vcc,s[32:33],vcc -> live SCC
        0x7e000280u,                         // pc5: SCC-preserving scheduled VALU
        0xbf840003u,                         // pc6: s_cbranch_scc0 -> pc10
        0x7d840100u,                         // pc7: fresh VCC definition
        0x02020100u,                         // pc8: consume the fresh VCC
        0xbf860002u,                         // pc9: s_cbranch_vccz -> pc12 (crossing region)
        0xbf060000u,                         // pc10: later SCC lifetime, after first consumer
        0xbf850001u,                         // pc11: third branch -> pc13 forces complex CFG
        0x7e020280u,                         // pc12: crossing arm
        0xbf810000u,                         // pc13: s_endpgm
    };
    const auto live_b64_mask_scc_spv = recompile_compute(
        wave64_compute_live_b64_mask_scc, std::size(wave64_compute_live_b64_mask_scc),
        nullptr, portable_wave64_compute_config);
    if (live_b64_mask_scc_spv.empty() ||
        !has_opcode(live_b64_mask_scc_spv, 224) || // synchronized guest-wave vote
        !has_opcode(live_b64_mask_scc_spv, 251) || // arbitrary CFG dispatcher
        !type_result_ids_are_nonzero(live_b64_mask_scc_spv, nullptr) ||
        !phi_ids_are_nonzero(live_b64_mask_scc_spv)) {
        printf("  [FAIL] live B64 mask SCC did not reach its crossing dispatcher branch\n");
        return 1;
    }

    // GTA V's compute culling kernels execute this exact in-place V_FFBH_U32 e32 packet inside
    // crossing control flow. Replace the SCC-preserving VALU in the proven Wave64 dispatcher
    // fixture, retaining every branch offset, and require the real FindUMsb lowering to be present.
    std::vector<uint32_t> gta_compute_ffbh_cfg(
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    gta_compute_ffbh_cfg[5] = 0x7e087304u; // exact live v_ffbh_u32_e32 v4,v4
    const auto gta_compute_ffbh_cfg_spv = recompile_compute(
        gta_compute_ffbh_cfg.data(), gta_compute_ffbh_cfg.size(), nullptr,
        portable_wave64_compute_config);
    if (gta_compute_ffbh_cfg_spv.empty() ||
        !has_opcode(gta_compute_ffbh_cfg_spv, 251) || // arbitrary CFG dispatcher
        !has_glsl_ext_inst(gta_compute_ffbh_cfg_spv, 75) || // FindUMsb
        !type_result_ids_are_nonzero(gta_compute_ffbh_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(gta_compute_ffbh_cfg_spv)) {
        printf("  [FAIL] GTA V v_ffbh_u32 did not lower inside crossing Wave64 compute CFG\n");
        return 1;
    }

    // SDWA source NEG is deliberately outside this plain-e32 admission. Without the gate the
    // generic VOP1 prologue would negate in the f32 domain before the integer FFBH operation.
    const uint32_t ffbh_sdwa_neg[] = {
        0x7e0872f9u, 0x00160604u,             // v_ffbh_u32_sdwa v4,v4 dst:dword src:dword neg
        0xbf810000u,
    };
    if (!recompile_compute(ffbh_sdwa_neg, std::size(ffbh_sdwa_neg), nullptr,
                           portable_wave64_compute_config).empty()) {
        printf("  [FAIL] v_ffbh_u32 admitted unsupported SDWA source NEG\n");
        return 1;
    }
    printf("  [ok]   GTA V v_ffbh_u32 lowers in crossing Wave64 CFG; SDWA NEG stays rejected\n");

    // GTA V exec_cs_413d1bf00 stops at pc458 on this exact packet. Require the production words,
    // the two-source decode, and the portable integer-domain lowering. GLSL.std.450 Ldexp (53) is
    // deliberately forbidden here because its overflow/large-exponent result is undefined; the
    // FindUMsb used to normalize a nonzero subnormal input is guarded inside the builder helper.
    const uint32_t gta_ldexp_f32[] = {
        0xd7620000u, 0x0002030du,             // v_ldexp_f32 v0,v13,v1
        0xbf810000u,
    };
    const auto gta_ldexp_f32_spv = recompile_valu(
        gta_ldexp_f32, std::size(gta_ldexp_f32), 14, 0);
    if (gta_ldexp_f32_spv.empty() || has_glsl_ext_inst(gta_ldexp_f32_spv, 53) ||
        !has_glsl_ext_inst(gta_ldexp_f32_spv, 75) ||
        !has_signed_i32_type(gta_ldexp_f32_spv) ||
        !type_result_ids_are_nonzero(gta_ldexp_f32_spv, nullptr)) {
        printf("  [FAIL] GTA V v_ldexp_f32 did not emit its defined integer-domain SPIR-V\n");
        return 1;
    }
    // Same-site modifier mutations remain outside this bounded admission. These exact bits exercise
    // each generic VOP3 modifier field; accepting one would mean the opcode case silently lost it.
    const uint32_t ldexp_abs0[]  = {0xd7620100u, 0x0002030du, 0xbf810000u};
    const uint32_t ldexp_abs1[]  = {0xd7620200u, 0x0002030du, 0xbf810000u};
    const uint32_t ldexp_abs2[]  = {0xd7620400u, 0x0002030du, 0xbf810000u};
    const uint32_t ldexp_neg0[]  = {0xd7620000u, 0x2002030du, 0xbf810000u};
    const uint32_t ldexp_neg1[]  = {0xd7620000u, 0x4002030du, 0xbf810000u};
    const uint32_t ldexp_neg2[]  = {0xd7620000u, 0x8002030du, 0xbf810000u};
    const uint32_t ldexp_clamp[] = {0xd7628000u, 0x0002030du, 0xbf810000u};
    const uint32_t ldexp_omod[]  = {0xd7620000u, 0x0802030du, 0xbf810000u};
    if (!recompile_valu(ldexp_abs0, std::size(ldexp_abs0), 14, 0).empty() ||
        !recompile_valu(ldexp_abs1, std::size(ldexp_abs1), 14, 0).empty() ||
        !recompile_valu(ldexp_abs2, std::size(ldexp_abs2), 14, 0).empty() ||
        !recompile_valu(ldexp_neg0, std::size(ldexp_neg0), 14, 0).empty() ||
        !recompile_valu(ldexp_neg1, std::size(ldexp_neg1), 14, 0).empty() ||
        !recompile_valu(ldexp_neg2, std::size(ldexp_neg2), 14, 0).empty() ||
        !recompile_valu(ldexp_clamp, std::size(ldexp_clamp), 14, 0).empty() ||
        !recompile_valu(ldexp_omod, std::size(ldexp_omod), 14, 0).empty()) {
        printf("  [FAIL] v_ldexp_f32 admitted an unsupported VOP3 modifier mutation\n");
        return 1;
    }
    printf("  [ok]   GTA V v_ldexp_f32 emits defined SPIR-V; modifier mutations reject\n");

    // GTA V follows its row reduction with this identity QUAD_PERM and partial ROW_MASK. Prefix the
    // exact packet to the crossing CFG above so it reaches the dispatcher lowering used by the live
    // shader. Rows 1/3 add distinct v19, rows 0/2 keep the old in-place v20 because BC is zero.
    std::vector<uint32_t> gta_compute_dpp_partial_rows = {
        0x7e280300u,                         // pc0: v_mov_b32 v20,v0
        0x7e260301u,                         // pc1: v_mov_b32 v19,v1
        0x4a2826fau, 0xaf00e414u,            // pc2: exact live identity DPP add, ROW_MASK=0xa
    };
    gta_compute_dpp_partial_rows.insert(
        gta_compute_dpp_partial_rows.end(),
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    gta_compute_dpp_partial_rows.back() = 0x7e000314u; // expose v20 through v0
    gta_compute_dpp_partial_rows.push_back(0xbf810000u);
    const auto gta_compute_dpp_partial_rows_spv = recompile_compute(
        gta_compute_dpp_partial_rows.data(), gta_compute_dpp_partial_rows.size(), nullptr,
        portable_wave64_compute_config);
    if (gta_compute_dpp_partial_rows_spv.empty() ||
        !has_opcode(gta_compute_dpp_partial_rows_spv, 251) ||
        !compute_dpp_add_partial_rows_updates_dispatch_vgpr(
            gta_compute_dpp_partial_rows_spv) ||
        !type_result_ids_are_nonzero(gta_compute_dpp_partial_rows_spv, nullptr) ||
        !phi_ids_are_nonzero(gta_compute_dpp_partial_rows_spv)) {
        printf("  [FAIL] GTA V partial-row DPP add did not preserve masked dispatcher rows\n");
        return 1;
    }
    printf("  [ok]   GTA V partial-row DPP add selects rows 1/3 and preserves BC0 VDST\n");

    // A later SOPC owns SCC, so the old mask producer must not be selected speculatively. The
    // crossing graph remains valid and branches on the ordinary scalar comparison.
    std::vector<uint32_t> overwritten_b64_mask_scc(
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    overwritten_b64_mask_scc[5] = 0xbf060000u; // s_cmp_eq_u32 s0,s0 overwrites SCC
    if (recompile_compute(overwritten_b64_mask_scc.data(), overwritten_b64_mask_scc.size(),
                          nullptr, portable_wave64_compute_config).empty()) {
        printf("  [FAIL] SCC overwrite was confused with the earlier B64 mask producer\n");
        return 1;
    }

    // SCC use by scalar dataflow is deliberately outside this branch-only proof. Broadening the
    // vote to make s_cselect consume the poisoned per-wave result would synchronize speculatively.
    std::vector<uint32_t> nonbranch_b64_mask_scc(
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    nonbranch_b64_mask_scc[6] = 0x85a2807eu; // s_cselect_b64 s[34:35],exec,0
    if (!recompile_compute(nonbranch_b64_mask_scc.data(), nonbranch_b64_mask_scc.size(),
                           nullptr, portable_wave64_compute_config).empty()) {
        printf("  [FAIL] non-branch SCC consumer gained a speculative B64 mask vote\n");
        return 1;
    }

    // A branch target starts a new SCC lifetime: another predecessor can enter without executing
    // the textually preceding mask producer. The branch-only proof must not cross that CFG join.
    std::vector<uint32_t> joined_b64_mask_scc(
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    joined_b64_mask_scc[3] = 0xbf820001u; // pc3: s_branch pc5, making pc5 a block entry
    if (!recompile_compute(joined_b64_mask_scc.data(), joined_b64_mask_scc.size(), nullptr,
                           portable_wave64_compute_config).empty()) {
        printf("  [FAIL] B64 mask SCC producer was associated across a CFG join\n");
        return 1;
    }

    // With two mask producers, only the last architectural SCC writer feeds the branch. The first
    // producer stays ordinary mask dataflow; the second is the single synchronized vote source.
    std::vector<uint32_t> superseded_b64_mask_scc(
        std::begin(wave64_compute_live_b64_mask_scc),
        std::end(wave64_compute_live_b64_mask_scc));
    superseded_b64_mask_scc[5] = 0x88a26a7eu; // s_or_b64 s[34:35],exec,vcc -> newer SCC
    if (recompile_compute(superseded_b64_mask_scc.data(), superseded_b64_mask_scc.size(),
                          nullptr, portable_wave64_compute_config).empty()) {
        printf("  [FAIL] last-live B64 mask producer was not selected uniquely\n");
        return 1;
    }

    // An ordinary scalar B64 logical writes an exact SCC directly. Even in the complex dispatcher,
    // it must not be diverted into the synchronized mask vote merely because a branch consumes SCC.
    const uint32_t ordinary_b64_scalar_scc[] = {
        0xbea00481u,                         // pc0: s_mov_b64 s[32:33],1
        0xbea20483u,                         // pc1: s_mov_b64 s[34:35],3
        0x87a42220u,                         // pc2: s_and_b64 s[36:37],s[32:33],s[34:35]
        0x7e000280u,                         // pc3: SCC-preserving scheduled VALU
        0xbf840003u,                         // pc4: s_cbranch_scc0 -> pc8
        0x7d840100u,                         // pc5: fresh VCC definition
        0x06020100u,                         // pc6: consume the fresh VCC
        0xbf860002u,                         // pc7: s_cbranch_vccz -> pc10
        0xbf060000u,                         // pc8: later SCC lifetime
        0xbf850001u,                         // pc9: third branch -> pc11 forces complex CFG
        0x7e020280u,                         // pc10: crossing arm
        0xbf810000u,                         // pc11: s_endpgm
    };
    const auto ordinary_b64_scalar_scc_spv = recompile_compute(
        ordinary_b64_scalar_scc, std::size(ordinary_b64_scalar_scc), nullptr,
        portable_wave64_compute_config);
    if (ordinary_b64_scalar_scc_spv.empty() ||
        !has_opcode(ordinary_b64_scalar_scc_spv, 251)) {
        printf("  [FAIL] ordinary scalar B64 SCC was diverted into a mask vote\n");
        return 1;
    }
    printf("  [ok]   B64 mask SCC votes are branch-only and select the last live producer\n");

    // Astro Bot's Wave64 world-map kernel selects ordinary LDS-write sequences with
    // S_CBRANCH_VCCZ. The scalar edge is exact per guest wave, but different waves in the workgroup
    // may choose different arms. That is legal for ordinary LDS loads/stores/atomics: unlike a guest
    // barrier or synthesized wave collective, they impose no workgroup-wide reconvergence point.
    // Keep this portable (native_subgroup_size=0) and use two guest waves so the test exercises the
    // exact scratch vote rather than accidentally inheriting the host subgroup's branch domain.
    const uint32_t wave64_compute_vcc_lds_write[] = {
        0x7d840000u,                         // pc0: v_cmp_eq_u32 vcc,v0,v0
        0xbf860002u,                         // pc1: s_cbranch_vccz -> pc4
        0xd8340084u, 0x00000002u,            // pc2: ds_write_b32 v2,v0 offset:0x84
        0xbf810000u,                         // pc4: s_endpgm
    };
    const auto wave64_compute_vcc_lds_spv = recompile_compute(
        wave64_compute_vcc_lds_write, std::size(wave64_compute_vcc_lds_write), nullptr,
        portable_wave64_compute_config);
    if (wave64_compute_vcc_lds_spv.empty() ||
        !has_opcode(wave64_compute_vcc_lds_spv, 224) || // OpControlBarrier: exact scratch vote
        !has_opcode(wave64_compute_vcc_lds_spv, 247) || // OpSelectionMerge: retained guest arm
        !type_result_ids_are_nonzero(wave64_compute_vcc_lds_spv, nullptr) ||
        !phi_ids_are_nonzero(wave64_compute_vcc_lds_spv)) {
        printf("  [FAIL] Wave64 VCC branch rejected an ordinary in-arm LDS write\n");
        return 1;
    }
    printf("  [ok]   Wave64 VCC branch retains ordinary LDS effects after an exact wave vote\n");

    // The acceptance above must not become a blanket escape hatch for operations whose lowering
    // contains a workgroup/subgroup synchronization phase. Those remain fail-visible inside a
    // per-wave arm until a common-phase transformation proves every invocation participates.
    const uint32_t wave64_compute_vcc_barrier[] = {
        0x7d840000u,
        0xbf860002u,                         // vccz -> pc4
        0xbf8a0000u,                         // guest workgroup barrier
        0xbf800000u,
        0xbf810000u,
    };
    const uint32_t wave64_compute_vcc_mbcnt[] = {
        0x7d840000u,
        0xbf860002u,                         // vccz -> pc4
        0xd7650004u, 0x000100c1u,            // v_mbcnt_lo_u32_b32 v4,-1,0
        0xbf810000u,
    };
    const uint32_t wave64_compute_vcc_swizzle[] = {
        0x7d840000u,
        0xbf860002u,                         // vccz -> pc4
        0xd8d4020fu, 0x00000000u,            // ds_swizzle_b32 v0,v0 offset:0x020f
        0xbf810000u,
    };
    if (!recompile_compute(wave64_compute_vcc_barrier,
                           std::size(wave64_compute_vcc_barrier), nullptr,
                           portable_wave64_compute_config).empty() ||
        !recompile_compute(wave64_compute_vcc_mbcnt,
                           std::size(wave64_compute_vcc_mbcnt), nullptr,
                           portable_wave64_compute_config).empty() ||
        !recompile_compute(wave64_compute_vcc_swizzle,
                           std::size(wave64_compute_vcc_swizzle), nullptr,
                           portable_wave64_compute_config).empty()) {
        printf("  [FAIL] synchronized operation escaped through a Wave64 VCC branch\n");
        return 1;
    }
    printf("  [ok]   Wave64 VCC branches still reject synchronized in-arm operations\n");

    // #1554: The Plucky Squire's chapter-one kernel guards a barrier-separated tail with a VCC
    // branch whose mask is built ONLY from launch data over a full EXEC:
    //     s_mov_b64 exec,-1 ; s_cmp_* <entry sgprs> ; s_cselect_b64 <mask>,exec,0 ; s_and_b64 vcc,..
    // recompile_compute seeds entry SGPRs from push-constant user data and WorkGroupId, both of
    // which are identical for every wave of a workgroup, so with EXEC full the branch decides
    // identically for all waves and the arm may synchronize. local_x=128/wave_size=64 means two
    // guest waves, so this genuinely exercises the cross-wave case rather than a single-wave shader.
    const uint32_t wave64_uniform_vcc_barrier[] = {
        0xbefe04c1u,                         // pc0: s_mov_b64 exec, -1
        0xbf088002u,                         // pc1: s_cmp_gt_u32 s2, 0
        0x858a807eu,                         // pc2: s_cselect_b64 s[10:11], exec, 0
        0xbf060403u,                         // pc3: s_cmp_eq_u32 s3, s4
        0x85ea807eu,                         // pc4: s_cselect_b64 vcc, exec, 0
        0x87ea0a6au,                         // pc5: s_and_b64 vcc, vcc, s[10:11]
        0xbf860002u,                         // pc6: s_cbranch_vccz -> pc9
        0xbf8a0000u,                         // pc7: guest workgroup barrier
        0xbf800000u,                         // pc8: s_nop
        0xbf810000u,                         // pc9: s_endpgm
    };
    const auto wave64_uniform_vcc_barrier_spv = recompile_compute(
        wave64_uniform_vcc_barrier, std::size(wave64_uniform_vcc_barrier), nullptr,
        portable_wave64_compute_config);
    if (wave64_uniform_vcc_barrier_spv.empty() ||
        !has_opcode(wave64_uniform_vcc_barrier_spv, 224) ||  // OpControlBarrier: the guest barrier
        !has_opcode(wave64_uniform_vcc_barrier_spv, 247) ||  // OpSelectionMerge: retained guest arm
        !type_result_ids_are_nonzero(wave64_uniform_vcc_barrier_spv, nullptr) ||
        !phi_ids_are_nonzero(wave64_uniform_vcc_barrier_spv)) {
        printf("  [FAIL] launch-uniform VCC branch rejected a barrier it provably enters uniformly\n");
        return 1;
    }
    printf("  [ok]   launch-uniform VCC branch admits an in-arm workgroup barrier\n");

    // Every obligation of that proof must be independently load-bearing. Each variant below changes
    // exactly ONE thing about the accepted kernel and must go back to rejecting the barrier.
    //
    // (a) EXEC is not provably full: VCCZ is then also true for a wave whose EXEC happens to be
    //     empty, which is a per-wave property no amount of scalar uniformity can recover.
    const uint32_t wave64_uniform_vcc_partial_exec[] = {
        0xbefe04c1u,                         // pc0: s_mov_b64 exec, -1
        0x7da80100u,                         // pc1: v_cmpx_eq_u32 v0, v0   (narrows EXEC per lane)
        0xbf088002u, 0x858a807eu, 0xbf060403u, 0x85ea807eu, 0x87ea0a6au,
        0xbf860002u, 0xbf8a0000u, 0xbf800000u, 0xbf810000u,
    };
    // (b) VCC is combined with a VOPC-written mask instead of a second proved select, so the branch
    //     tests lane occupancy rather than a scalar predicate.
    const uint32_t wave64_uniform_vcc_lane_scc[] = {
        0xbefe04c1u,                         // pc0: s_mov_b64 exec, -1
        0x7d840000u,                         // pc1: v_cmp_eq_u32 vcc, v0, v0   (lane-derived mask)
        0xbf088002u,                         // pc2: s_cmp_gt_u32 s2, 0
        0x858a807eu,                         // pc3: s_cselect_b64 s[10:11], exec, 0
        0x87ea0a6au,                         // pc4: s_and_b64 vcc, vcc, s[10:11]
        0xbf860002u,                         // pc5: s_cbranch_vccz -> pc8
        0xbf8a0000u, 0xbf800000u, 0xbf810000u,
    };
    // (c) A contributing compare reads a V_READFIRSTLANE result: uniform per wave, not per workgroup.
    const uint32_t wave64_uniform_vcc_readfirstlane[] = {
        0xbefe04c1u,
        0x7e0a0500u,                         // v_readfirstlane_b32 s5, v0
        0xbf088002u, 0x858a807eu,
        0xbf060503u,                         // s_cmp_eq_u32 s3, s5
        0x85ea807eu, 0x87ea0a6au,
        0xbf860002u, 0xbf8a0000u, 0xbf800000u, 0xbf810000u,
    };
    // (d) The mask is combined with raw EXEC rather than another proved select.
    const uint32_t wave64_uniform_vcc_exec_operand[] = {
        0xbefe04c1u, 0xbf088002u, 0x858a807eu, 0xbf060403u, 0x85ea807eu,
        0x87ea7e6au,                         // s_and_b64 vcc, vcc, exec
        0xbf860002u, 0xbf8a0000u, 0xbf800000u, 0xbf810000u,
    };
    // Report every variant rather than stopping at the first, so a weakened proof shows exactly which
    // obligations stopped carrying weight instead of hiding behind whichever check happens to be first.
    struct UniformVccNegative {
        const char* what;
        const uint32_t* code;
        size_t words;
    };
    const UniformVccNegative uniform_vcc_negatives[] = {
        {"a non-full EXEC", wave64_uniform_vcc_partial_exec,
         std::size(wave64_uniform_vcc_partial_exec)},
        {"a lane-derived mask operand", wave64_uniform_vcc_lane_scc,
         std::size(wave64_uniform_vcc_lane_scc)},
        {"a readfirstlane SCC", wave64_uniform_vcc_readfirstlane,
         std::size(wave64_uniform_vcc_readfirstlane)},
        {"a raw EXEC mask operand", wave64_uniform_vcc_exec_operand,
         std::size(wave64_uniform_vcc_exec_operand)},
    };
    bool uniform_vcc_negatives_held = true;
    for (const auto& negative : uniform_vcc_negatives) {
        if (recompile_compute(negative.code, negative.words, nullptr,
                              portable_wave64_compute_config).empty())
            continue;
        printf("  [FAIL] %s still admitted a barrier-spanning VCC branch\n", negative.what);
        uniform_vcc_negatives_held = false;
    }
    if (!uniform_vcc_negatives_held) return 1;
    printf("  [ok]   every workgroup-uniformity obligation independently rejects the barrier\n");

    wave32_compute_config.wave_size = 64;
    if (!recompile_compute(wave32_compute_masks, std::size(wave32_compute_masks), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave64 compute accepted Wave32 low-half mask semantics\n");
        return 1;
    }
    printf("  [ok]   compute low-half mask semantics require proven Wave32 launch state\n");

    const uint32_t wave32_fragment_wqm[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbefe0900u,                         // s_wqm_b32 exec_lo, s0
        0xbefe097eu,                         // s_wqm_b32 exec_lo, exec_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,            // exp mrt0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto wave32_fragment_wqm_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_wqm, std::size(wave32_fragment_wqm));
    if (wave32_fragment_wqm_spv.empty() ||
        !type_result_ids_are_nonzero(wave32_fragment_wqm_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_wqm_spv)) {
        printf("  [FAIL] Wave32 fragment s_wqm_b32 mask path did not recompile cleanly\n");
        return 1;
    }
    printf("  [ok]   Wave32 fragment s_wqm_b32 mask path emits valid SPIR-V\n");

    // Exact Astro world-map PC1060..1064 shape. LLVM gfx1030 disassembly identifies the compare as
    // `v_cmp_eq_f32_sdwa vcc_hi, 0, v7`; in Wave32 that explicit one-word destination is an
    // independent saved mask consumed by s_andn2_b32 before WQM restores EXEC_LO.
    const uint32_t wave32_fragment_b32_logic[] = {
        0xbec0037eu,                         // s_mov_b32 s64, exec_lo
        0x7e0e0280u,                         // v_mov_b32 v7, 0
        0x7c040ef9u, 0x0686eb80u,            // v_cmp_eq_f32_sdwa vcc_hi, 0, v7
        0x8a406b40u,                         // s_andn2_b32 s64, s64, vcc_hi
        0xbf840008u,                         // s_cbranch_scc0 -> terminal null-export tail
        0xbefe0940u,                         // s_wqm_b32 exec_lo, s64
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
        0xbefe0380u,                         // tail: s_mov_b64 exec, 0
        0xf8001c00u, 0x00000000u,            // exp null off, off, off, off done vm
        0xbf810000u,
    };
    const auto wave32_fragment_b32_logic_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_b32_logic, std::size(wave32_fragment_b32_logic));
    if (wave32_fragment_b32_logic_spv.empty() ||
        !has_opcode(wave32_fragment_b32_logic_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_b32_logic_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_b32_logic_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_b32_logic_spv)) {
        printf("  [FAIL] Wave32 VCC_HI compare/B32 mask logic did not recompile cleanly "
               "(words=%zu vote=%d marker=%d subgroup=%u)\n",
               wave32_fragment_b32_logic_spv.size(),
               has_opcode(wave32_fragment_b32_logic_spv, 335),
               has_opcode(wave32_fragment_b32_logic_spv, 330),
               fragment_spirv_required_subgroup_size(wave32_fragment_b32_logic_spv));
        return 1;
    }
    printf("  [ok]   Wave32 B32 alpha-test vote linearizes its terminal null-export branch\n");

    const auto wave32_mask_branches = mask_test_branches_for_test(
        wave32_fragment_b32_logic, std::size(wave32_fragment_b32_logic), true);
    bool found_mask_branch = false;
    for (const uint32_t pc : wave32_mask_branches)
        if (pc == 5) found_mask_branch = true;
    if (!found_mask_branch) {
        printf("  [FAIL] proven Wave32 B32 mask vote was not recognized at PC5\n");
        return 1;
    }
    const uint32_t wave32_scalar_scc_branch[] = {
        0x87008104u,                         // s_and_b32 s0, s4, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_scalar_scc_branch,
                                     std::size(wave32_scalar_scc_branch), true).empty()) {
        printf("  [FAIL] ordinary Wave32 scalar SCC branch was mistaken for a mask vote\n");
        return 1;
    }
    printf("  [ok]   Wave32 branch linearization requires proven mask provenance\n");

    // A wide scalar write kills every physical SGPR word it covers. Keep the compare in s1
    // deliberately adjacent to s0: if s_mov_b64 only invalidates its low word, the stale s1 mask
    // provenance infects the ordinary s_and_b32 and makes its real SCC branch look disposable.
    const uint32_t wave32_mask_overwritten_by_b64[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0xbe800480u,                         // s_mov_b64 s[0:1], 0
        0x87008101u,                         // s_and_b32 s0, s1, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_mask_overwritten_by_b64,
                                     std::size(wave32_mask_overwritten_by_b64), true).empty()) {
        printf("  [FAIL] B64 scalar overwrite retained stale high-word mask provenance\n");
        return 1;
    }
    printf("  [ok]   B64 scalar writes invalidate every covered Wave32 mask word\n");

    const uint32_t wave32_mask_overwritten_by_saveexec[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0xbe802680u,                         // s_xor_saveexec_b64 s[0:1], 0
        0x87008101u,                         // s_and_b32 s0, s1, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_mask_overwritten_by_saveexec,
                                     std::size(wave32_mask_overwritten_by_saveexec), true).empty()) {
        printf("  [FAIL] B64 SAVEEXEC overwrite retained stale high-word mask provenance\n");
        return 1;
    }
    printf("  [ok]   every supported B64 SAVEEXEC form has a two-word write lifetime\n");

    // The compare executes only on the fall-through predecessor. At the join, s1 is not proven to
    // be a mask on every path, so the following scalar SCC branch must remain real control flow.
    const uint32_t wave32_path_dependent_branch_mask[] = {
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840002u,                         // s_cbranch_scc0 -> join at PC4
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0x87008101u,                         // join: s_and_b32 s0, s1, 1
        0xbf840002u,                         // real s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_path_dependent_branch_mask,
                                     std::size(wave32_path_dependent_branch_mask), true).empty()) {
        printf("  [FAIL] path-dependent B32 mask provenance escaped its predecessor block\n");
        return 1;
    }
    printf("  [ok]   Wave32 mask-branch proof does not cross control-flow joins\n");

    // Reduced Astro world-map PS PC9..33 preamble: s64 is a saved mask before a real EXECZ
    // conditional. The fall-through arm refines it with an explicit VOPC mask and immediately votes
    // through SCC. Provenance on that arm is valid even though the EXECZ target bypasses the block.
    const uint32_t wave32_mask_through_execz_fallthrough[] = {
        0xbec0037eu,                         // pc0: s_mov_b32 s64, exec_lo
        0x7c220b31u,                         // pc1: v_cmpx_lt_f32 vcc, v49, v5
        0xbf880004u,                         // pc2: s_cbranch_execz -> merge at PC7
        0x7c0862f9u, 0x0686eb25u,            // pc3: v_cmp_lt_f32_sdwa vcc_hi, s37, v49
        0x8a406b40u,                         // pc5: s_andn2_b32 s64, s64, vcc_hi
        0xbf840001u,                         // pc6: s_cbranch_scc0 -> terminal tail
        0xbf810000u,                         // pc7: merge
    };
    bool found_fallthrough_vote = false;
    for (uint32_t pc : mask_test_branches_for_test(
             wave32_mask_through_execz_fallthrough,
             std::size(wave32_mask_through_execz_fallthrough), true))
        if (pc == 6) found_fallthrough_vote = true;
    if (!found_fallthrough_vote) {
        printf("  [FAIL] valid fall-through Wave32 mask provenance was lost at EXECZ\n");
        return 1;
    }
    printf("  [ok]   Wave32 mask proof preserves valid conditional fall-through provenance\n");

    // The same live shader selects EXEC_LO or an empty mask into VCC_HI from an ordinary scalar
    // comparison. This is s_cselect_b32's mask-domain form, not a scalar integer selection.
    const uint32_t wave32_fragment_b32_cselect[] = {
        0xbec0037eu,                         // s_mov_b32 s64, exec_lo
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x856b807eu,                         // s_cselect_b32 vcc_hi, exec_lo, 0
        0x8a406b40u,                         // s_andn2_b32 s64, s64, vcc_hi
        0xbefe0940u,                         // s_wqm_b32 exec_lo, s64
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_b32_cselect_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_b32_cselect, std::size(wave32_fragment_b32_cselect));
    if (wave32_fragment_b32_cselect_spv.empty() ||
        fragment_spirv_required_subgroup_size(wave32_fragment_b32_cselect_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_b32_cselect_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_b32_cselect_spv)) {
        printf("  [FAIL] Wave32 s_cselect_b32 did not preserve its VCC_HI mask result\n");
        return 1;
    }
    printf("  [ok]   Wave32 s_cselect_b32 preserves its selected VCC_HI mask\n");

    // Explicit Wave32 VOPC destinations are one-word masks even when they name ordinary SGPRs.
    // The live barycentric block compares into s0 and VCC_HI, writes an independent carry mask to
    // VCC_LO, then ANDs s0 and the still-live VCC_HI mask back into VCC_LO.
    const uint32_t wave32_fragment_explicit_vopc_masks[] = {
        0x7d8654f9u, 0x06868004u,            // v_cmp_le_u32_sdwa s0, s4, v42
        0x7d865cf9u, 0x0686eb04u,            // v_cmp_le_u32_sdwa vcc_hi, s4, v46
        0xd5286a29u, 0x00025880u,            // v_add_co_ci_u32 v41, vcc_lo, 0, v44, s0
        0x876a6b00u,                         // s_and_b32 vcc_lo, s0, vcc_hi
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_explicit_vopc_masks_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_explicit_vopc_masks,
        std::size(wave32_fragment_explicit_vopc_masks));
    if (wave32_fragment_explicit_vopc_masks_spv.empty() ||
        !has_opcode(wave32_fragment_explicit_vopc_masks_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_explicit_vopc_masks_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_explicit_vopc_masks_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_explicit_vopc_masks_spv)) {
        printf("  [FAIL] explicit Wave32 VOPC SGPR destinations lost their B32 mask domain\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VOPC SGPR destinations feed B32 mask logic\n");

    // A VOPC write to s0 must not erase the adjacent, independently-live s1 mask. In Wave32 each
    // explicit compare destination occupies exactly one scalar word; the old generic inventory
    // incorrectly treated both compares as two-word writes.
    const uint32_t wave32_fragment_adjacent_vopc_masks[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0x7d8654f9u, 0x06868004u,            // v_cmp_le_u32_sdwa s0, s4, v42
        0x876a0100u,                         // s_and_b32 vcc_lo, s0, s1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_adjacent_vopc_masks,
            std::size(wave32_fragment_adjacent_vopc_masks)).empty()) {
        printf("  [FAIL] adjacent explicit Wave32 VOPC masks clobbered each other\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VOPC destinations preserve adjacent SGPR masks\n");

    // VCC_LO/HI remain physical scalar scratch registers too. A B32 ALU destination alone does not
    // prove mask-domain use: the live shader builds M0 from ordinary integer data through VCC_LO.
    const uint32_t wave32_fragment_vcc_scratch[] = {
        0x876a8744u,                         // s_and_b32 vcc_lo, s68, 7
        0x936a856au,                         // s_lshl_b32 vcc_lo, vcc_lo, 5
        0xbefc036au,                         // s_mov_b32 m0, vcc_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_vcc_scratch,
            std::size(wave32_fragment_vcc_scratch)).empty()) {
        printf("  [FAIL] Wave32 VCC_LO scalar-scratch chain was mistaken for a wave mask\n");
        return 1;
    }
    printf("  [ok]   Wave32 VCC_LO remains available for ordinary scalar scratch data\n");

    // Inline constants 0/1 use numeric operand values that also happen to name s0/s1. Even while
    // s0 is a saved Wave32 mask, an all-inline cselect into VCC_HI is ordinary scalar data.
    const uint32_t wave32_fragment_inline_cselect_data[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo (mask-domain s0)
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x856b8081u,                         // s_cselect_b32 vcc_hi, 1, 0 (data-domain)
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x8801fd6bu,                         // s_or_b32 s1, vcc_hi, scc
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_inline_cselect_data,
            std::size(wave32_fragment_inline_cselect_data)).empty()) {
        printf("  [FAIL] Wave32 inline cselect constants aliased saved-mask register numbers\n");
        return 1;
    }
    printf("  [ok]   Wave32 inline cselect constants remain ordinary scalar data\n");

    // Exact PC790..803 control idiom: save EXEC_LO into VCC_HI while narrowing through VCC_LO,
    // then restore that independently tracked saved mask through a B32 move.
    const uint32_t wave32_fragment_and_saveexec[] = {
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbeeb3c6au,                         // s_and_saveexec_b32 vcc_hi, vcc_lo
        0xbefe036bu,                         // s_mov_b32 exec_lo, vcc_hi
        0xbeea446au,                         // s_andn1_saveexec_b32 vcc_lo, vcc_lo
        0xbefe036au,                         // s_mov_b32 exec_lo, vcc_lo
        0xbea0037eu,                         // s_mov_b32 s32, exec_lo
        0xbeea4020u,                         // s_orn2_saveexec_b32 vcc_lo, s32
        0xbefe036au,                         // s_mov_b32 exec_lo, vcc_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_and_saveexec_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_and_saveexec, std::size(wave32_fragment_and_saveexec));
    if (wave32_fragment_and_saveexec_spv.empty() ||
        !has_opcode(wave32_fragment_and_saveexec_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_and_saveexec_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_and_saveexec_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_and_saveexec_spv)) {
        printf("  [FAIL] Wave32 B32 saveexec family did not preserve VCC/EXEC masks\n");
        return 1;
    }
    printf("  [ok]   Wave32 AND/ORN2/ANDN1 saveexec family preserves saved EXEC\n");

    // Use a distinct dynamic old EXEC and a false source so the operand order is observable in the
    // emitted SPIR-V, rather than merely checking that the saveexec family can be translated.
    const uint32_t wave32_fragment_andn1_saveexec[] = {
        0x7c220300u,                         // v_cmpx_lt_f32 v0, v1
        0xbe804480u,                         // s_andn1_saveexec_b32 s0, 0
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_andn1_saveexec_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_andn1_saveexec, std::size(wave32_fragment_andn1_saveexec));
    if (wave32_fragment_andn1_saveexec_spv.empty() ||
        !logical_not_of_false_feeds_and(wave32_fragment_andn1_saveexec_spv)) {
        printf("  [FAIL] s_andn1_saveexec_b32 did not compute old_EXEC & ~source\n");
        return 1;
    }
    printf("  [ok]   s_andn1_saveexec_b32 negates its source operand\n");

    const uint32_t wave32_fragment_readlane31[] = {
        0x7e140280u,                         // v_mov_b32 v10, 0
        0xd7600000u, 0x00013f0au,            // v_readlane_b32 s0, v10, 31
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_readlane31_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_readlane31, std::size(wave32_fragment_readlane31));
    if (wave32_fragment_readlane31_spv.empty() ||
        fragment_spirv_required_subgroup_size(wave32_fragment_readlane31_spv) != 32) {
        printf("  [FAIL] Wave32 fragment v_readlane requested the wrong subgroup size\n");
        return 1;
    }
    const uint32_t wave32_fragment_readlane32[] = {
        0x7e140280u,                         // v_mov_b32 v10, 0
        0xd7600000u, 0x0001410au,            // v_readlane_b32 s0, v10, 32 (out of range)
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_fragment_readlane32,
            std::size(wave32_fragment_readlane32)).empty()) {
        printf("  [FAIL] Wave32 fragment v_readlane accepted lane 32\n");
        return 1;
    }
    printf("  [ok]   Wave32 fragment v_readlane retains a 32-lane subgroup contract\n");

    const uint32_t fragment_cvt_i32_word_sdwa[] = {
        0x7e1a10f9u, 0x0006140du,            // v_cvt_i32_f32_sdwa v13,v13 WORD_0/PRESERVE
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto fragment_cvt_i32_word_sdwa_spv = recompile_fragment(
        fragment_cvt_i32_word_sdwa, std::size(fragment_cvt_i32_word_sdwa));
    if (fragment_cvt_i32_word_sdwa_spv.empty() ||
        !type_result_ids_are_nonzero(fragment_cvt_i32_word_sdwa_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cvt_i32_word_sdwa_spv)) {
        printf("  [FAIL] WORD-preserving v_cvt_i32_f32_sdwa did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   WORD-preserving v_cvt_i32_f32_sdwa emits valid SPIR-V\n");

    const uint32_t wave32_vertex_exec[] = {
        0xbefe03c1u,                         // s_mov_b32 exec_lo, -1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf80008cfu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto wave32_vertex_spv = recompile_vertex(
        wave32_vertex_exec, std::size(wave32_vertex_exec));
    if (!wave32_vertex_spv.empty()) {
        printf("  [FAIL] ungated vertex shader accepted Wave32 EXEC_LO restore\n");
        return 1;
    }
    printf("  [ok]   unproven graphics Wave32 mask operations remain fail-visible\n");

    // A one-word Wave32 saved-mask alias ends when that physical SGPR is reused as scalar data.
    // v_cndmask must not prefer the old bool after either an SOPK or SOP2 writer; without a numeric
    // B64 mask representation these deliberately reject instead of silently selecting the stale arm.
    const uint32_t wave32_mask_then_sopk[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xb0000000u,                         // s_movk_i32 s0, 0
        0xb0010000u,                         // s_movk_i32 s1, 0
        0xd5010003u, 0x0001e8f2u,            // v_cndmask_b32_e64 v3, 1.0, 2.0, s[0:1]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_sopk, std::size(wave32_mask_then_sopk)).empty()) {
        printf("  [FAIL] SOPK scalar reuse retained a stale Wave32 mask alias\n");
        return 1;
    }
    const uint32_t wave32_mask_then_sop2[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0x87008080u,                         // s_and_b32 s0, 0, 0
        0xb0010000u,                         // s_movk_i32 s1, 0
        0xd5010003u, 0x0001e8f2u,            // v_cndmask_b32_e64 v3, 1.0, 2.0, s[0:1]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_sop2, std::size(wave32_mask_then_sop2)).empty()) {
        printf("  [FAIL] SOP2 scalar reuse retained a stale Wave32 mask alias\n");
        return 1;
    }
    const uint32_t wave32_sop2_reuse_control[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0x87008080u,                         // s_and_b32 s0, 0, 0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_sop2_reuse_control, std::size(wave32_sop2_reuse_control)).empty()) {
        printf("  [FAIL] SOP2 scalar-reuse control shader did not recompile\n");
        return 1;
    }
    printf("  [ok]   scalar reuse invalidates saved Wave32 mask aliases\n");

    // A folded s_getpc_b64 has no runtime scalar SSA result, but it still overwrites both physical
    // SGPR words. Keep an independent valid embedded-table chain in the shader so the first getpc is
    // accepted, then prove that it kills the old B32 mask rather than letting v_cndmask see it.
    const uint32_t wave32_mask_then_folded_getpc[] = {
        0xbe80037eu,                         // pc0: s_mov_b32 s0, exec_lo
        0xbe801f00u,                         // pc1: s_getpc_b64 s[0:1] (overwrites saved mask)
        0xb0060010u,                         // pc2: s_movk_i32 s6, 16-byte table
        0xbe8703ffu, 0x10005004u,            // pc3: s_mov_b32 s7, V# config
        0xbe841f00u,                         // pc5: s_getpc_b64 s[4:5] (next byte 24)
        0x800404b4u,                         // pc6: s_add_u32 s4, 52, s4 (table byte 76)
        0x82050580u,                         // pc7: s_addc_u32 s5, 0, s5
        0x7e020280u,                         // pc8: v_mov_b32 v1, 0 (table byte offset)
        0xe0301000u, 0x80010101u,            // pc9: buffer_load_dword v1,v1,s[4:7]
        0xbf8c3f70u,                         // pc11: s_waitcnt vmcnt(0)
        0xd5010003u, 0x0001e8f2u,            // pc12: v_cndmask v3,1.0,2.0,s[0:1]
        0x7e000280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
        7u, 11u, 13u, 17u,
    };
    std::vector<uint32_t> wave32_folded_getpc_control(
        std::begin(wave32_mask_then_folded_getpc),
        std::end(wave32_mask_then_folded_getpc));
    wave32_folded_getpc_control[13] = 0x01a9e8f2u; // consume live VCC, not overwritten s[0:1]
    if (recompile_fragment_wave32_for_test(
            wave32_folded_getpc_control.data(),
            wave32_folded_getpc_control.size()).empty()) {
        printf("  [FAIL] folded s_getpc_b64 control shader did not recompile\n");
        return 1;
    }
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_folded_getpc,
            std::size(wave32_mask_then_folded_getpc)).empty()) {
        printf("  [FAIL] folded s_getpc_b64 retained a stale Wave32 mask alias\n");
        return 1;
    }
    printf("  [ok]   folded s_getpc_b64 invalidates saved Wave32 mask aliases\n");

    // A no-else forward arm has a skipped predecessor at its merge. If only the taken arm creates a
    // B32 saved mask, the physical-word validity differs between predecessors and cannot be modeled
    // by a bool-value phi alone. Keep the post-merge mask consumer fail-visible.
    const uint32_t wave32_conditional_mask_save[] = {
        0xbe800380u,                         // pc0: s_mov_b32 s0, 0
        0xbf060000u,                         // pc1: s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // pc2: s_cbranch_scc0 -> pc4
        0xbe82037eu,                         // pc3: s_mov_b32 s2, exec_lo (taken arm only)
        0xd5010003u, 0x0009e8f2u,            // pc4: v_cndmask v3,1.0,2.0,s[2:3]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_conditional_mask_save,
            std::size(wave32_conditional_mask_save)).empty()) {
        printf("  [FAIL] forward-if leaked a path-dependent Wave32 mask lifetime\n");
        return 1;
    }
    printf("  [ok]   forward-if rejects path-dependent Wave32 mask lifetimes\n");

    const uint32_t scalar_abs_compute[] = {
        0xb000ffffu,                         // s_movk_i32 s0, -1
        0xbe813400u,                         // s_abs_i32 s1, s0
        0xbf850001u,                         // s_cbranch_scc1 +1
        0xbf800000u,                         // s_nop 0
        0xbf810000u,
    };
    ComputeShaderConfig scalar_abs_config;
    const auto scalar_abs_spv = recompile_compute(
        scalar_abs_compute, std::size(scalar_abs_compute), nullptr, scalar_abs_config);
    if (scalar_abs_spv.empty() || !type_result_ids_are_nonzero(scalar_abs_spv, nullptr) ||
        !phi_ids_are_nonzero(scalar_abs_spv)) {
        printf("  [FAIL] scalar s_abs_i32/SCC path did not recompile cleanly\n");
        return 1;
    }
    printf("  [ok]   scalar s_abs_i32 writes its result and SCC in valid SPIR-V\n");

    // Prosper does not expose an attached GPU system debugger to guest shaders, so COND_DBG_SYS is
    // permanently clear and s_cbranch_cdbgsys falls through. Astro's world-map NGG wrapper uses this
    // around its position export; rejecting it drops the complete draw despite ordinary hardware also
    // taking the fallthrough path outside a shader-debugging session.
    const uint32_t no_system_debugger_vertex[] = {
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xbf970001u,                         // s_cbranch_cdbgsys +1 (not taken)
        0xf80008cfu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto no_system_debugger_spv = recompile_vertex(
        no_system_debugger_vertex, std::size(no_system_debugger_vertex));
    if (no_system_debugger_spv.empty() ||
        !type_result_ids_are_nonzero(no_system_debugger_spv, nullptr) ||
        !phi_ids_are_nonzero(no_system_debugger_spv)) {
        printf("  [FAIL] s_cbranch_cdbgsys did not take the no-debugger fallthrough path\n");
        return 1;
    }
    printf("  [ok]   s_cbranch_cdbgsys falls through when no GPU debugger is exposed\n");

    const uint32_t unsupported_exec_hi[] = {
        0xbeff03c1u,                         // s_mov_b32 exec_hi, -1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf80008cfu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_vertex(unsupported_exec_hi, std::size(unsupported_exec_hi)).empty()) {
        printf("  [FAIL] unsupported EXEC_HI B32 write was accepted\n");
        return 1;
    }
    printf("  [ok]   EXEC_HI B32 writes remain fail-closed\n");

    const uint32_t scratch_vertex[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xf80008cfu, 0x00000000u, // exp pos0 v0, v0, v0, v0
        0xBF810000u,
    };
    const auto scratch_vertex_spv = recompile_vertex(
        scratch_vertex, sizeof(scratch_vertex) / sizeof(scratch_vertex[0]));
    uint32_t scratch_vertex_bad_op = 0;
    if (scratch_vertex_spv.empty() || !has_opcode(scratch_vertex_spv, 28) ||
        !type_result_ids_are_nonzero(scratch_vertex_spv, &scratch_vertex_bad_op) ||
        !phi_ids_are_nonzero(scratch_vertex_spv)) {
        printf("  [FAIL] vertex private spill/fill emitted invalid SPIR-V (op=%u)\n",
               scratch_vertex_bad_op);
        return 1;
    }
    printf("  [ok]   vertex private spill/fill emits structurally valid Function storage\n");

    // NGG wave packing writes EXEC through s_lshr_b64 before later structured control flow. The
    // instruction is mask-domain only; inserting rs.sreg[EXEC] left an SSA id 0 that a later merge
    // emitted as an OpPhi input. NVIDIA's Windows driver faults in vkCreateGraphicsPipelines on that
    // invalid module, so guard the exact wave-pack -> forward-if shape independently of a Vulkan driver.
    const uint32_t ngg_exec_if[] = {
        0x93EAFF03u, 0x00080008u, 0x876BFF03u, 0x000000FFu, 0x8F6A8C6Au,
        0x887C6A6Bu, 0xBF900009u, 0x906A8803u, 0x81EA6A80u, 0x90FE6AC1u,
        0xF8000941u, 0x00000000u, 0x81EA0380u, 0x90FE6AC1u,
        0xBE80246Au,                         // s_and_saveexec_b64 s[0:1], vcc
        0xBF880001u,                         // s_cbranch_execz +1
        0x8AFE7E00u,                         // s_andn2_b64 exec, s[0:1], exec
        0xBEFE0400u,                         // s_mov_b64 exec, s[0:1]
        0x34040A81u, 0x36060AC2u, 0x7E000280u, 0x7E0202F2u, 0x36040482u,
        0x4A0606C1u, 0x4A0404C1u, 0x7E060B03u, 0x7E040B02u,
        0xF80008CFu, 0x01000302u, 0xBF810000u,
    };
    const auto ngg_exec_if_spv = recompile_vertex(
        ngg_exec_if, sizeof(ngg_exec_if) / sizeof(ngg_exec_if[0]));
    if (ngg_exec_if_spv.empty() || !phi_ids_are_nonzero(ngg_exec_if_spv)) {
        printf("  [FAIL] NGG EXEC wave-pack control flow emitted a zero-id OpPhi\n");
        return 1;
    }
    printf("  [ok]   NGG EXEC wave-pack control flow emits only valid nonzero OpPhi ids\n");

    // A GS_ALLOC_REQ marker alone does not prove the one-lane model. An unrelated NGG shader that
    // reaches a wave population count must remain fail-closed rather than silently counting lane 0.
    const uint32_t ngg_mask_count[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80106Au,                         // s_bcnt1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    if (!recompile_vertex(ngg_mask_count,
                          sizeof(ngg_mask_count) / sizeof(ngg_mask_count[0])).empty()) {
        printf("  [FAIL] unproven NGG accepted one-lane mask population count\n");
        return 1;
    }
    printf("  [ok]   unproven NGG rejects one-lane mask population count\n");

    // In Astro's byte-exact one-lane NGG projection, find-first-one on a wave mask is exact:
    // the represented lane is bit zero, so an active mask returns 0 and an empty mask returns -1.
    // The test-only entry point exercises that lowering without broadening production acceptance.
    const uint32_t ngg_mask_ff1[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80146Au,                         // s_ff1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_mask_ff1_spv = recompile_vertex_ngg_one_lane_for_test(
        ngg_mask_ff1, std::size(ngg_mask_ff1));
    if (ngg_mask_ff1_spv.empty() || !type_result_ids_are_nonzero(ngg_mask_ff1_spv, nullptr)) {
        printf("  [FAIL] one-lane NGG s_ff1_i32_b64 did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   one-lane NGG s_ff1_i32_b64 emits valid SPIR-V\n");

    // Astro's 7f5f world-map wrapper constructs a B64 wave mask in ordinary scalar DATA registers
    // and consumes it through v_cndmask_b32_e64. The one-lane projection represents lane zero, so
    // the condition is exactly bit zero of the pair's low dword. Other vertex shaders still reject
    // this wave-dependent form because they have no proven lane identity.
    const uint32_t ngg_scalar_data_mask[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBE8403FFu, 0xAAAAAAAAu,            // s_mov_b32 s4, 0xaaaaaaaa
        0xBE850304u,                         // s_mov_b32 s5, s4
        0xD5010005u, 0x00120AFFu, 0x00100800u, // exact v_cndmask_b32_e64 v5, lit, v5, s[4:5]
        0x7E000305u,                         // v_mov_b32 v0, v5
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_scalar_data_mask_spv = recompile_vertex_ngg_one_lane_for_test(
        ngg_scalar_data_mask, std::size(ngg_scalar_data_mask));
    if (ngg_scalar_data_mask_spv.empty() ||
        !type_result_ids_are_nonzero(ngg_scalar_data_mask_spv, nullptr)) {
        printf("  [FAIL] one-lane NGG scalar-data mask cndmask did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   one-lane NGG scalar-data mask cndmask emits valid SPIR-V\n");

    // Exercise the output-selection emitter through its explicit test hook. The production entry
    // point restricts every terminal NGG gate to the byte-exact Astro wrapper, as checked below.
    const uint32_t ngg_output_gate[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32 (uniformly narrows no lanes)
        0xBF880002u,                         // s_cbranch_execz -> s_endpgm
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_gate_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_gate, sizeof(ngg_output_gate) / sizeof(ngg_output_gate[0]));
    if (ngg_output_gate_spv.empty() || !phi_ids_are_nonzero(ngg_output_gate_spv)) {
        printf("  [FAIL] terminal NGG compacted-vertex output gate did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG compacted-vertex output gate produces valid SPIR-V\n");

    // The real Astro wrapper does not export directly after its terminal CMPX gate: it computes an
    // LDS address and reloads the compacted vertex first. Those predicated register-only operations
    // are part of the same bounded gate and must not make its position export look unsafe.
    const uint32_t ngg_output_rebuild_gate[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32
        0xBF880005u,                         // s_cbranch_execz -> s_endpgm
        0x7E000280u,                         // v_mov_b32 v0, 0 (address/value setup)
        0xD8D80000u, 0x00000000u,            // ds_read_b32 v0, v0
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_rebuild_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_rebuild_gate, std::size(ngg_output_rebuild_gate));
    if (ngg_output_rebuild_spv.empty() || !phi_ids_are_nonzero(ngg_output_rebuild_spv)) {
        printf("  [FAIL] terminal NGG LDS output-rebuild gate did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG LDS output-rebuild gate produces valid SPIR-V\n");

    // Astro Bot's first world-map wrapper rebuilds its surviving compacted output through a
    // shader-embedded constant table rather than LDS. The terminal suffix is still side-effect
    // free: scalar ALU constructs the PC-relative V#, MUBUF only reads the proven bounded table,
    // and the results feed POS0/PARAM0 before S_ENDPGM. This is the exact captured 54-dword program
    // plus the table tail required by those two loads.
    const uint32_t astro_worldmap_pcrel_output_gate[] = {
        0xbfa00003u, 0x93ebff03u, 0x00040018u, 0xbefe03c1u,
        0x9380ff02u, 0x00090016u, 0x9381ff02u, 0x0009000cu,
        0xbf8a0000u, 0xbf076b80u, 0xbf850004u, 0x8f6a8c00u,
        0x887c6a01u, 0xbf800000u, 0xbf900009u, 0x8f6a856bu,
        0xd7650001u, 0x0000d4c1u, 0x7da80200u, 0xbf880002u,
        0xf8000941u, 0x00000000u, 0xbf8cff0fu, 0xbefe03c1u,
        0x7da80201u, 0xbf88001bu, 0xd56a0000u, 0x00020affu,
        0xaaaaaaabu, 0xbe8303ffu, 0x10005004u, 0xb0020048u,
        0xbe801f00u, 0x800000ffu, 0x000000acu, 0x82010180u,
        0x2c000081u, 0xd7460000u, 0x04010300u, 0x4c000105u,
        0x34000083u, 0xd7460004u, 0x04010300u, 0xe0381000u,
        0x80000004u, 0xe0341010u, 0x80000404u, 0xbf8c3f71u,
        0xf80008cfu, 0x03020100u, 0xbf8c3f70u, 0xf8000203u,
        0x00000504u, 0xbf810000u,
        // Padding to byte offset 304, then the 72-byte constant table.
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u,
        0x00000000u, 0xbf800000u, 0xbf800000u, 0x3f800000u,
        0x3f800000u, 0x00000000u, 0x3f800000u, 0x40400000u,
        0xbf800000u, 0x3f800000u, 0x3f800000u, 0x40000000u,
        0x3f800000u, 0xbf800000u, 0x40400000u, 0x3f800000u,
        0x3f800000u, 0x00000000u, 0xbf800000u,
    };
    const auto astro_worldmap_pcrel_output_spv = recompile_vertex(
        astro_worldmap_pcrel_output_gate, std::size(astro_worldmap_pcrel_output_gate));
    if (astro_worldmap_pcrel_output_spv.empty() ||
        !type_result_ids_are_nonzero(astro_worldmap_pcrel_output_spv, nullptr) ||
        !phi_ids_are_nonzero(astro_worldmap_pcrel_output_spv)) {
        printf("  [FAIL] captured Astro PC-relative NGG output gate did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   captured Astro PC-relative NGG output gate emits valid SPIR-V\n");

    // Astro's second world-map wrapper exports POS for a surviving compacted vertex, then a regular
    // VCC compare conditionally skips only the trailing PARAM exports. Supplying those otherwise-
    // undefined varyings in the one-lane projection is safe and must not drop the complete draw.
    const uint32_t ngg_output_vcc_tail[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32
        0xBF880006u,                         // s_cbranch_execz -> s_endpgm
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0x7D8A5880u,                         // v_cmp_ne_u32 vcc, 0, v44
        0xBF860002u,                         // s_cbranch_vccnz -> s_endpgm
        0xF800020Fu, 0x03020100u,            // exp param0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_vcc_tail_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_vcc_tail, std::size(ngg_output_vcc_tail));
    if (ngg_output_vcc_tail_spv.empty() || !phi_ids_are_nonzero(ngg_output_vcc_tail_spv)) {
        printf("  [FAIL] terminal NGG VCC-gated PARAM tail did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG VCC-gated PARAM tail produces valid SPIR-V\n");

    if (!recompile_vertex(ngg_output_gate, std::size(ngg_output_gate)).empty()) {
        printf("  [FAIL] production accepted an unproven constant NGG output gate\n");
        return 1;
    }
    printf("  [ok]   production rejects unproven constant NGG output gates (including points)\n");

    // A per-vertex CMPX under the same superficial terminal shape can create mixed active/inactive
    // primitives. Without the byte-exact Astro wrapper/topology proof it must remain rejected.
    const uint32_t unproven_ngg_output_gate[] = {
        0xBF900009u,
        0x7DA80300u,                         // data-dependent v_cmpx_*
        0xBF880002u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };
    if (!recompile_vertex(unproven_ngg_output_gate,
                          std::size(unproven_ngg_output_gate)).empty()) {
        printf("  [FAIL] unproven NGG accepted a mixed terminal output gate\n");
        return 1;
    }
    printf("  [ok]   unproven NGG mixed terminal output gate remains fail-closed\n");

    // Small inline B64 masks are also lane-sensitive. Merely resembling an NGG wrapper cannot opt a
    // shader into the captured Astro program's lane-zero projection.
    const uint32_t ngg_inline_mask[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x92EA8081u,                         // s_bfm_b64 vcc, 1, 0 (lane-zero mask)
        0xBEFE0481u,                         // s_mov_b64 exec, 1
        0xBEFE04C1u,                         // s_mov_b64 exec, -1 (restore before export)
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    if (!recompile_vertex(ngg_inline_mask,
                          sizeof(ngg_inline_mask) / sizeof(ngg_inline_mask[0])).empty()) {
        printf("  [FAIL] unproven NGG accepted lane-zero inline-mask semantics\n");
        return 1;
    }
    printf("  [ok]   unproven NGG rejects lane-zero inline-mask semantics\n");

    // The same wave shortcut is not valid for an ordinary vertex shader: without the exact NGG
    // allocation message there is no proof that a Vulkan invocation represents guest lane zero.
    const uint32_t ordinary_vertex_mask_count[] = {
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80106Au,                         // s_bcnt1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };
    if (!recompile_vertex(ordinary_vertex_mask_count,
                          std::size(ordinary_vertex_mask_count)).empty()) {
        printf("  [FAIL] ordinary vertex shader accepted NGG lane-zero mask semantics\n");
        return 1;
    }
    printf("  [ok]   ordinary vertex shader rejects NGG-only lane-zero mask semantics\n");

    // Even an exact GS_ALLOC_REQ marker does not make arbitrary LDS lane-local. A peer-addressed
    // write/barrier/read shape must stay fail-closed unless the complete observed Astro wrapper is
    // selected by its byte-exact fingerprint.
    const uint32_t unproven_ngg_peer_lds[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ
        0x7E000280u, 0x7E020281u,            // v0=0 byte address, v1=1 data
        0xD8340000u, 0x00000100u,            // ds_write_b32 v0, v1
        0xBF8A0000u,                         // s_barrier
        0xD8D80000u, 0x02000000u,            // ds_read_b32 v2, v0
        0x7E060280u, 0x7E0802F2u,
        0xF80008CFu, 0x04030302u,
        0xBF810000u,
    };
    if (!recompile_vertex(unproven_ngg_peer_lds, std::size(unproven_ngg_peer_lds)).empty()) {
        printf("  [FAIL] unproven NGG peer-LDS shader accepted private one-lane LDS semantics\n");
        return 1;
    }
    printf("  [ok]   unproven NGG peer-LDS shader remains fail-closed\n");

    // v_cmpx_* narrows EXEC. A FRAGMENT export under a narrowed EXEC is a discard (alpha test / kill): it
    // now lowers to a per-invocation OpKill of the inactive lanes followed by an export from the survivors,
    // so fragment recompilation ACCEPTS it and emits valid SPIR-V. (A VERTEX shader cannot discard — OpKill
    // is fragment-only — so the vertex cmpx-export case below still rejects.)
    const uint32_t cmpx_fragment[] = {
        0x7DA80300u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    auto frag_spv = recompile_fragment(cmpx_fragment, sizeof(cmpx_fragment) / sizeof(cmpx_fragment[0]));
    if (frag_spv.empty()) {
        printf("  [FAIL] fragment cmpx discard shader was rejected (should lower to OpKill + export)\n");
        return 1;
    }
    { uint32_t bad_op = 0;
      if (!type_result_ids_are_nonzero(frag_spv, &bad_op)) {
          printf("  [FAIL] fragment discard SPIR-V has an invalid result id (op=%u)\n", bad_op);
          return 1;
      } }
    printf("  [ok]   fragment cmpx export lowers to a discard (OpKill + export), valid SPIR-V\n");

    // Astro Bot's title materials contain large reducible pixel shaders whose forward branch
    // regions overlap rather than forming a lexical if-tree. The compact SSA structurizer must
    // remain conservative, but the per-invocation graphics CFG dispatcher can execute the exact
    // basic-block graph. This small crossing-region shape forces that fallback.
    const uint32_t fragment_cfg_dispatch[] = {
        0x7e040280u,                         // pc0:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc1:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc2:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc3:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc4:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc5:  s_cbranch_vccz -> pc9
        0x7c020300u,                         // pc6:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc7:  s_cbranch_vccz -> pc10 (crosses pc5 region)
        0x7e040281u,                         // pc8:  v_mov_b32 v2, 1
        0x7e060281u,                         // pc9:  v_mov_b32 v3, 1
        0x7c020300u,                         // pc10: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc11: s_cbranch_vccz -> alternate export at pc15
        0xf800180fu, 0x05040302u,            // pc12: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc14: s_branch -> verified tail exit at pc18
        0xf800180fu, 0x05040302u,            // pc15: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc17: s_endpgm
        0xbf810000u,                         // pc18: branch-target s_endpgm
    };
    const auto fragment_cfg_spv = recompile_fragment(
        fragment_cfg_dispatch, std::size(fragment_cfg_dispatch));
    if (fragment_cfg_spv.empty() || !has_opcode(fragment_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(fragment_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_spv)) {
        printf("  [FAIL] complex fragment CFG did not lower through a valid OpSwitch dispatcher\n");
        return 1;
    }
    const OutputStoreStats cfg_outputs = output_store_stats(fragment_cfg_spv);
    if (cfg_outputs.stores != 2 || cfg_outputs.stores_with_one_repeated_source != 0) {
        printf("  [FAIL] complex fragment CFG exports stale entry state or suppresses an alternate "
               "site (stores=%u repeated-source=%u)\n",
               cfg_outputs.stores, cfg_outputs.stores_with_one_repeated_source);
        return 1;
    }
    printf("  [ok]   complex fragment CFG exports active state from both alternate sites\n");

    // Astro Bot's second world-map material is Wave32 and carries saved one-word masks through the
    // same non-lexical branch graph. The explicit VOPC writes below intentionally target adjacent
    // s1 then s0: both independent masks must survive every dispatcher case and feed the later EXEC
    // restore. Treating either compare as a two-word write erases s1 or persists it as scalar zero.
    const uint32_t wave32_fragment_cfg_masks[] = {
        0x7d865cf9u, 0x06868104u,            // pc0:  v_cmp_le_u32_sdwa s1, s4, v46
        0x7d8654f9u, 0x06868004u,            // pc2:  v_cmp_le_u32_sdwa s0, s4, v42
        0xbefe097eu,                         // pc4:  s_wqm_b32 exec_lo, exec_lo
        0x7e040280u,                         // pc5:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc6:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc7:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc8:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc10: s_cbranch_vccz -> pc14
        0x7c020300u,                         // pc11: v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc12: s_cbranch_vccz -> pc15 (crosses pc10 region)
        0x7e040281u,                         // pc13: v_mov_b32 v2, 1
        0x7e060281u,                         // pc14: v_mov_b32 v3, 1
        0x7c020300u,                         // pc15: v_cmp_lt_f32 vcc, v0, v1
        0xbf860004u,                         // pc16: s_cbranch_vccz -> alternate restore at pc21
        0x877e0100u,                         // pc17: s_and_b32 exec_lo, s0, s1
        0xf800180fu, 0x05040302u,            // pc18: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820004u,                         // pc20: s_branch -> verified tail exit at pc25
        0x877e0100u,                         // pc21: alternate s_and_b32 exec_lo, s0, s1
        0xf800180fu, 0x05040302u,            // pc22: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc24: s_endpgm
        0xbf810000u,                         // pc25: branch-target s_endpgm
    };
    const auto wave32_fragment_cfg_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_cfg_masks, std::size(wave32_fragment_cfg_masks));
    if (wave32_fragment_cfg_spv.empty() || !has_opcode(wave32_fragment_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(wave32_fragment_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_cfg_spv)) {
        printf("  [FAIL] complex Wave32 fragment CFG lost saved-mask state across cases\n");
        return 1;
    }
    printf("  [ok]   complex Wave32 fragment CFG persists unambiguous saved-mask lifetimes\n");

    // The live world-map PS writes an explicit SGPR mask with VOPC, then compares that whole B64
    // wave mask against zero inside the same dispatcher. SCC is a wave vote, not this invocation's
    // mask bit; fragment pipelines enforce wave64 and can lower it to subgroup-any exactly.
    const uint32_t fragment_mask_compare_prelude[] = {
        0x7e120280u,                         // v_mov_b32 v9, 0
        0x7c0212f9u, 0x06868480u,            // v_cmp_lt_f32_sdwa s[4:5], 0, v9
        0xbf138004u,                         // s_cmp_lg_u64 s[4:5], 0
    };
    std::vector<uint32_t> fragment_cfg_mask_compare(
        std::begin(fragment_mask_compare_prelude), std::end(fragment_mask_compare_prelude));
    fragment_cfg_mask_compare.insert(fragment_cfg_mask_compare.end(),
        std::begin(fragment_cfg_dispatch), std::end(fragment_cfg_dispatch));
    const auto fragment_cfg_mask_compare_spv = recompile_fragment(
        fragment_cfg_mask_compare.data(), fragment_cfg_mask_compare.size());
    if (fragment_cfg_mask_compare_spv.empty() ||
        !has_opcode(fragment_cfg_mask_compare_spv, 251) ||
        !has_opcode(fragment_cfg_mask_compare_spv, 335) ||
        fragment_spirv_required_subgroup_size(fragment_cfg_mask_compare_spv) != 64 ||
        !type_result_ids_are_nonzero(fragment_cfg_mask_compare_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_mask_compare_spv)) {
        printf("  [FAIL] complex fragment CFG rejected a wave-mask zero comparison\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG lowers wave-mask comparisons to exact wave64 votes\n");

    // House of the Dead 2's large scene fragments reach this exact mask relationship inside the
    // graphics CFG dispatcher:
    //   s[44:45] = EXEC & s[30:31]
    //   s[46:47] = VCC  & s[30:31]
    //   SCC = (s[44:45] ==/!= s[46:47]); s[48:49] = SCC ? EXEC : 0
    // Both operands are complete saved masks, so the scalar comparison is one exact whole-wave
    // ANY(mask0 xor mask1) vote. Keep the later s_cselect and EXEC copy: they prove the vote's SCC
    // survives a dispatcher case boundary and is consumed, rather than merely appearing in SPIR-V.
    const uint32_t fragment_saved_mask_pair_prelude[] = {
        0xbe9e047eu,                         // s_mov_b64 s[30:31],exec
        0x87ac1e7eu,                         // s_and_b64 s[44:45],exec,s[30:31]
        0x7c020300u,                         // v_cmp_lt_f32 vcc,v0,v1
        0x87ae1e6au,                         // s_and_b64 s[46:47],vcc,s[30:31]
        0xbf122e2cu,                         // exact live s_cmp_eq_u64 s[44:45],s[46:47]
        0xbf800000u,                         // SCC-preserving scheduled nop
        0x85b0807eu,                         // s_cselect_b64 s[48:49],exec,0
        0xbefe0430u,                         // s_mov_b64 exec,s[48:49]
    };
    // Crossing scalar branch regions force the same arbitrary graphics CFG dispatcher without
    // introducing any additional subgroup vote. That makes the exact one-vote EQ/LG polarity below
    // a self-checking dataflow witness rather than an opcode census hidden behind red siblings.
    const uint32_t fragment_scalar_cfg_dispatch[] = {
        0x7e040280u,                         // pc0:  v_mov_b32 v2,0
        0x7e060280u,                         // pc1:  v_mov_b32 v3,0
        0x7e080280u,                         // pc2:  v_mov_b32 v4,0
        0x7e0a0280u,                         // pc3:  v_mov_b32 v5,0
        0xbf068080u,                         // pc4:  s_cmp_eq_u32 0,0
        0xbf840003u,                         // pc5:  s_cbranch_scc0 -> pc9
        0xbf068080u,                         // pc6:  s_cmp_eq_u32 0,0
        0xbf840002u,                         // pc7:  s_cbranch_scc0 -> pc10 (crossing region)
        0x7e040281u,                         // pc8:  v_mov_b32 v2,1
        0x7e060281u,                         // pc9:  v_mov_b32 v3,1
        0xbf068080u,                         // pc10: s_cmp_eq_u32 0,0
        0xbf840003u,                         // pc11: s_cbranch_scc0 -> alternate export at pc15
        0xf800180fu, 0x05040302u,            // pc12: exp mrt0 v2,v3,v4,v5 done vm
        0xbf820003u,                         // pc14: s_branch -> verified tail exit at pc18
        0xf800180fu, 0x05040302u,            // pc15: alternate exp mrt0 v2,v3,v4,v5 done vm
        0xbf810000u,                         // pc17: s_endpgm
        0xbf810000u,                         // pc18: branch-target s_endpgm
    };
    auto saved_mask_pair_shader = [&]() {
        std::vector<uint32_t> shader(
            std::begin(fragment_saved_mask_pair_prelude),
            std::end(fragment_saved_mask_pair_prelude));
        shader.insert(shader.end(), std::begin(fragment_scalar_cfg_dispatch),
                      std::end(fragment_scalar_cfg_dispatch));
        return shader;
    };
    auto mask_pair_module_is_exact = [&](const std::vector<uint32_t>& spirv,
                                         uint32_t wave_size, bool inverted) {
        return !spirv.empty() && has_opcode(spirv, 251) &&
            opcode_count(spirv, 335) == 1 &&
            fragment_spirv_required_subgroup_size(spirv) == wave_size &&
            wave_vote_reaches_later_select(spirv, inverted) &&
            type_result_ids_are_nonzero(spirv, nullptr) &&
            phi_ids_are_nonzero(spirv);
    };
    std::vector<uint32_t> fragment_saved_mask_pair_eq = saved_mask_pair_shader();
    const auto fragment_saved_mask_pair_eq_spv = recompile_fragment(
        fragment_saved_mask_pair_eq.data(), fragment_saved_mask_pair_eq.size());
    if (!mask_pair_module_is_exact(fragment_saved_mask_pair_eq_spv, 64,
                                   /*inverted EQ vote*/true)) {
        printf("  [FAIL] House saved-mask equality did not feed CFG-persisted SCC in Wave64\n");
        return 1;
    }
    std::vector<uint32_t> fragment_saved_mask_pair_lg = fragment_saved_mask_pair_eq;
    fragment_saved_mask_pair_lg[4] = 0xbf132e2cu; // s_cmp_lg_u64 s[44:45],s[46:47]
    const auto fragment_saved_mask_pair_lg_spv = recompile_fragment(
        fragment_saved_mask_pair_lg.data(), fragment_saved_mask_pair_lg.size());
    if (!mask_pair_module_is_exact(fragment_saved_mask_pair_lg_spv, 64,
                                   /*direct LG vote*/false)) {
        printf("  [FAIL] saved-mask inequality did not feed CFG-persisted SCC in Wave64\n");
        return 1;
    }
    const auto fragment_wave32_saved_mask_pair_eq_spv =
        recompile_fragment_wave32_for_test(
            fragment_saved_mask_pair_eq.data(), fragment_saved_mask_pair_eq.size());
    const auto fragment_wave32_saved_mask_pair_lg_spv =
        recompile_fragment_wave32_for_test(
            fragment_saved_mask_pair_lg.data(), fragment_saved_mask_pair_lg.size());
    if (!mask_pair_module_is_exact(fragment_wave32_saved_mask_pair_eq_spv, 32, true) ||
        !mask_pair_module_is_exact(fragment_wave32_saved_mask_pair_lg_spv, 32, false)) {
        printf("  [FAIL] saved-mask EQ/LG lost the complete-pair contract in Wave32\n");
        return 1;
    }
    printf("  [ok]   saved-mask EQ/LG votes persist into s_cselect in Wave32 and Wave64 CFGs\n");

    // Two alternate saved-pair comparison sites model adjacent fragment lanes parked at distinct
    // dispatcher PCs.  Event 1 is EQ over s[44:45]/s[46:47], while event 2 is LG over the disjoint
    // s[52:53]/s[54:55] pair.  A mismatch bit may live on a lane currently taking the other case,
    // so both complete-pair votes must be unconditional after the switch merge; only the SCC write
    // is selected by the publishing lane's nonzero event tag.
    std::vector<uint32_t> saved_mask_pair_divergent_sites = {
        0xbe9e047eu,                         // pc0:  s_mov_b64 s[30:31],exec
        0x87ac1e7eu,                         // pc1:  s_and_b64 s[44:45],exec,s[30:31]
        0x7c020300u,                         // pc2:  v_cmp_lt_f32 vcc,v0,v1
        0x87ae1e6au,                         // pc3:  s_and_b64 s[46:47],vcc,s[30:31]
        0x87b41e7eu,                         // pc4:  s_and_b64 s[52:53],exec,s[30:31]
        0x87b61e6au,                         // pc5:  s_and_b64 s[54:55],vcc,s[30:31]
        0xbf060100u,                         // pc6:  s_cmp_eq_u32 s0,s1
        0xbf840002u,                         // pc7:  alternate event -> pc10
        0xbf122e2cu,                         // pc8:  event 1: EQ s[44:45],s[46:47]
        0xbf820001u,                         // pc9:  join -> pc11
        0xbf133634u,                         // pc10: event 2: LG s[52:53],s[54:55]
        0xbf800000u,                         // pc11: SCC-preserving join
        0x85b0807eu,                         // pc12: s_cselect_b64 s[48:49],exec,0
        0xbefe0430u,                         // pc13: s_mov_b64 exec,s[48:49]
    };
    saved_mask_pair_divergent_sites.insert(
        saved_mask_pair_divergent_sites.end(), std::begin(fragment_scalar_cfg_dispatch),
        std::end(fragment_scalar_cfg_dispatch));
    const auto saved_mask_pair_divergent_wave64_spv = recompile_fragment(
        saved_mask_pair_divergent_sites.data(), saved_mask_pair_divergent_sites.size());
    const auto saved_mask_pair_divergent_wave32_spv = recompile_fragment_wave32_for_test(
        saved_mask_pair_divergent_sites.data(), saved_mask_pair_divergent_sites.size());
    if (!saved_mask_pair_votes_use_uniform_event_phase(
            saved_mask_pair_divergent_wave64_spv) ||
        !saved_mask_pair_votes_use_uniform_event_phase(
            saved_mask_pair_divergent_wave32_spv)) {
        printf("  [FAIL] divergent-PC saved-mask votes escaped uniform event isolation\n");
        return 1;
    }
    printf("  [ok]   divergent-PC saved-mask bits vote in uniform Wave32/64 event phases\n");

    // Every physical/dataflow obligation is independently load-bearing. A write to either half of
    // a saved pair ends that lifetime; a numeric replacement is not reinterpreted as a mask; and a
    // producer skipped on one reachable predecessor cannot establish a mask at the join.
    std::vector<uint32_t> saved_mask_high_overwrite = fragment_saved_mask_pair_eq;
    saved_mask_high_overwrite.insert(saved_mask_high_overwrite.begin() + 4, 0xbead0380u);
    if (!recompile_fragment(saved_mask_high_overwrite.data(),
                            saved_mask_high_overwrite.size()).empty()) {
        printf("  [FAIL] high-half scalar overwrite retained a stale saved-mask pair\n");
        return 1;
    }
    std::vector<uint32_t> saved_mask_numeric_replacement = fragment_saved_mask_pair_eq;
    saved_mask_numeric_replacement.insert(
        saved_mask_numeric_replacement.begin() + 4,
        {0xbeae0381u, 0xbeaf0380u});          // numeric s[46:47] = 1
    if (!recompile_fragment(saved_mask_numeric_replacement.data(),
                            saved_mask_numeric_replacement.size()).empty()) {
        printf("  [FAIL] numeric SGPR pair was reinterpreted as a saved wave mask\n");
        return 1;
    }
    std::vector<uint32_t> saved_mask_path_dependent = {
        0xbe9e047eu,                         // s_mov_b64 s[30:31],exec
        0x87ac1e7eu,                         // s_and_b64 s[44:45],exec,s[30:31]
        0xbf060100u,                         // pc2: s_cmp_eq_u32 s0,s1
        0xbf840001u,                         // pc3: edge skips s46 -> compare at pc5
        0x87ae1e6au,                         // pc4: s_and_b64 s[46:47],vcc,s[30:31]
        0xbf122e2cu,                         // pc5: joined s[46:47] lifetime is path-dependent
        0xbf800000u,                         // pc6: SCC-preserving nop
        0x85b0807eu,                         // pc7: consumes comparison SCC
        0xbefe0430u,
    };
    saved_mask_path_dependent.insert(
        saved_mask_path_dependent.end(), std::begin(fragment_scalar_cfg_dispatch),
        std::end(fragment_scalar_cfg_dispatch));
    const auto saved_mask_path_dependent_spv = recompile_fragment(
        saved_mask_path_dependent.data(), saved_mask_path_dependent.size());
    if (wave_vote_reaches_later_select(saved_mask_path_dependent_spv,
                                       /*inverted EQ vote*/true)) {
        printf("  [FAIL] path-dependent saved-mask vote reached a later select\n");
        return 1;
    }
    printf("  [ok]   saved-mask pair compare rejects half-overwrite/numeric mutations and "
           "does not vote across joins\n");

    // The exact House shader next performs four in-place unsigned minima across DPP row-right
    // neighbors. Keep them inside the same crossing graphics CFG fixture: the dispatcher must
    // publish each lane's source, shift, and event, execute three subgroup shuffles in its uniform
    // common phase (value + source-activity + event), persist the UMin, and do so for both sizes.
    auto saved_mask_pair_dpp_shader = [&]() {
        std::vector<uint32_t> shader(
            std::begin(fragment_saved_mask_pair_prelude),
            std::end(fragment_saved_mask_pair_prelude));
        shader.insert(shader.end(), {
            0x7e060287u,                         // v_mov_b32 v3,7
            0x260606fau, 0xff011103u,            // v_min_u32_dpp v3,v3,v3 row_shr:1
            0x260606fau, 0xff011203u,            // row_shr:2
            0x260606fau, 0xff011403u,            // row_shr:4
            0x260606fau, 0xff011803u,            // row_shr:8
        });
        shader.insert(shader.end(), std::begin(fragment_scalar_cfg_dispatch),
                      std::end(fragment_scalar_cfg_dispatch));
        return shader;
    };
    auto dpp_min_module_is_exact = [&](const std::vector<uint32_t>& spirv,
                                       uint32_t wave_size) {
        return !spirv.empty() && has_opcode(spirv, 251) &&
            dpp_min_row_shr_updates_dispatch_vgpr(spirv) &&
            fragment_spirv_required_subgroup_size(spirv) == wave_size &&
            (fragment_spirv_required_subgroup_features(spirv) &
             kFragmentSubgroupShuffle) &&
            type_result_ids_are_nonzero(spirv, nullptr) &&
            phi_ids_are_nonzero(spirv);
    };

    // Four alternate static DPP sites make the cross-PC hazard explicit.  A source lane parked at
    // row_shr:2/4/8 must never satisfy a destination executing row_shr:1 (and vice versa), even when
    // both publish active data in the same common phase.  Value, activity, and event identity must
    // use one source-lane shuffle; the event equality is load-bearing on the VGPR write path.
    std::vector<uint32_t> fragment_dpp_min_divergent_sites(
        std::begin(fragment_saved_mask_pair_prelude),
        std::end(fragment_saved_mask_pair_prelude));
    fragment_dpp_min_divergent_sites.insert(
        fragment_dpp_min_divergent_sites.end(), {
            0x7e060287u,                         // pc8:  v_mov_b32 v3,7
            0xbf060100u,                         // pc9:  s_cmp_eq_u32 s0,s1
            0xbf840003u,                         // pc10: alternate event -> pc14
            0x260606fau, 0xff011103u,            // pc11: event 1 row_shr:1
            0xbf820002u,                         // pc13: join -> pc16
            0x260606fau, 0xff011203u,            // pc14: event 2 row_shr:2
            0xbf800000u,                         // pc16: first join
            0xbf060100u,                         // pc17: s_cmp_eq_u32 s0,s1
            0xbf840003u,                         // pc18: alternate event -> pc22
            0x260606fau, 0xff011403u,            // pc19: event 3 row_shr:4
            0xbf820002u,                         // pc21: join -> pc24
            0x260606fau, 0xff011803u,            // pc22: event 4 row_shr:8
            0xbf800000u,                         // pc24: second join
        });
    fragment_dpp_min_divergent_sites.insert(
        fragment_dpp_min_divergent_sites.end(),
        std::begin(fragment_scalar_cfg_dispatch), std::end(fragment_scalar_cfg_dispatch));
    const auto fragment_dpp_min_divergent_wave64_spv = recompile_fragment(
        fragment_dpp_min_divergent_sites.data(), fragment_dpp_min_divergent_sites.size());
    const auto fragment_dpp_min_divergent_wave32_spv = recompile_fragment_wave32_for_test(
        fragment_dpp_min_divergent_sites.data(), fragment_dpp_min_divergent_sites.size());
    if (!dpp_min_module_is_exact(fragment_dpp_min_divergent_wave64_spv, 64) ||
        !dpp_min_module_is_exact(fragment_dpp_min_divergent_wave32_spv, 32)) {
        printf("  [FAIL] divergent-PC DPP sources escaped static-event isolation\n");
        return 1;
    }
    printf("  [ok]   divergent-PC DPP sources require matching Wave32/64 static events\n");

    std::vector<uint32_t> fragment_dpp_min_row = saved_mask_pair_dpp_shader();
    const auto fragment_dpp_min_row_wave64_spv = recompile_fragment(
        fragment_dpp_min_row.data(), fragment_dpp_min_row.size());
    const auto fragment_dpp_min_row_wave32_spv = recompile_fragment_wave32_for_test(
        fragment_dpp_min_row.data(), fragment_dpp_min_row.size());
    if (!dpp_min_module_is_exact(fragment_dpp_min_row_wave64_spv, 64) ||
        !dpp_min_module_is_exact(fragment_dpp_min_row_wave32_spv, 32)) {
        printf("  [FAIL] House DPP minimum did not persist through the Wave32/64 CFG common phase\n");
        return 1;
    }

    std::vector<uint32_t> dpp_min_wrong_opcode = fragment_dpp_min_row;
    dpp_min_wrong_opcode[9] = 0x280606fau; // v_max_u32 is outside the exact contract
    std::vector<uint32_t> dpp_min_distinct_source = fragment_dpp_min_row;
    dpp_min_distinct_source[10] = 0xff011104u; // src0=v4, while VDST/SRC1 remain v3
    std::vector<uint32_t> dpp_min_wrong_amount = fragment_dpp_min_row;
    dpp_min_wrong_amount[10] = 0xff011203u; // duplicates :2; loses exact :1/:2/:4/:8 chain
    const auto wrong_opcode_spv = recompile_fragment(
        dpp_min_wrong_opcode.data(), dpp_min_wrong_opcode.size());
    const auto distinct_source_spv = recompile_fragment(
        dpp_min_distinct_source.data(), dpp_min_distinct_source.size());
    const auto wrong_amount_spv = recompile_fragment(
        dpp_min_wrong_amount.data(), dpp_min_wrong_amount.size());
    if (dpp_min_row_shr_updates_dispatch_vgpr(wrong_opcode_spv) ||
        dpp_min_row_shr_updates_dispatch_vgpr(distinct_source_spv) ||
        dpp_min_row_shr_updates_dispatch_vgpr(wrong_amount_spv)) {
        printf("  [FAIL] DPP minimum admitted an opcode/source/amount mutation\n");
        return 1;
    }
    printf("  [ok]   House DPP minimum preserves row direction, CFG participation, and VGPR dataflow\n");

    // Astro Bot's world-map material PS reaches the same graphics CFG dispatcher with an
    // s_orn2_saveexec_b64 whose destination is VCC. Keep a saved EXEC source live across dispatcher
    // cases, update both the explicit VCC SGPR pair and the implicit VCC condition, then restore EXEC.
    // The crossing branch regions below force the fallback which rejected the live shader at this op.
    const uint32_t fragment_cfg_orn2_saveexec[] = {
        0xbe82047eu,                         // pc0:  s_mov_b64 s[2:3], exec
        0xbeea2802u,                         // pc1:  s_orn2_saveexec_b64 vcc, s[2:3]
        0xbefe046au,                         // pc2:  s_mov_b64 exec, vcc (restore saved EXEC)
        0x7e040280u,                         // pc3:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc4:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc5:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc6:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc7:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc8:  s_cbranch_vccz -> pc12
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc10: s_cbranch_vccz -> pc13 (crossing region)
        0x7e040281u,                         // pc11: v_mov_b32 v2, 1
        0x7e060281u,                         // pc12: v_mov_b32 v3, 1
        0x7c020300u,                         // pc13: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc14: s_cbranch_vccz -> alternate export at pc18
        0xf800180fu, 0x05040302u,            // pc15: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc17: s_branch -> verified tail exit at pc21
        0xf800180fu, 0x05040302u,            // pc18: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc20: s_endpgm
        0xbf810000u,                         // pc21: branch-target s_endpgm
    };
    const auto fragment_cfg_orn2_spv = recompile_fragment(
        fragment_cfg_orn2_saveexec, std::size(fragment_cfg_orn2_saveexec));
    if (fragment_cfg_orn2_spv.empty() || !has_opcode(fragment_cfg_orn2_spv, 251) ||
        !type_result_ids_are_nonzero(fragment_cfg_orn2_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_orn2_spv)) {
        printf("  [FAIL] complex fragment CFG rejected s_orn2_saveexec_b64 VCC form\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG preserves Astro ORN2-saveexec VCC state\n");

    // Astro's world-map material PS folds a lane-local mask across every 16-lane hardware row.
    // These are the four exact live packets following ORN2-saveexec. They require subgroup shuffle
    // (not the derivative-based FLOAT quad-perm approximation) and a 64-lane fragment subgroup.
    const uint32_t fragment_dpp_row_or[] = {
        0x7e1402c1u,                         // v_mov_b32 v10, -1
        0x381414fau, 0xff01110au,            // v_or_b32_dpp v10,v10,v10 row_shr:1
        0x381414fau, 0xff01120au,            // row_shr:2
        0x381414fau, 0xff01140au,            // row_shr:4
        0x381414fau, 0xff01180au,            // row_shr:8
        0xd7781009u, 0x0305830au,            // v_permlanex16 v9,v10,-1,-1 BC=1
        0x3814130au,                         // v_or_b32 v10,v10,v9
        0xd7600000u, 0x00013f0au,            // v_readlane_b32 s0,v10,31
        0xd7600001u, 0x00017f0au,            // v_readlane_b32 s1,v10,63
        0x883f0100u,                         // s_or_b32 s63,s0,s1
        0xf800000fu, 0x0a0a0a0au,            // exp mrt0 v10,v10,v10,v10
        0xbf810000u,
    };
    const auto fragment_dpp_row_or_spv = recompile_fragment(
        fragment_dpp_row_or, std::size(fragment_dpp_row_or));
    if (fragment_dpp_row_or_spv.empty() ||
        !has_opcode(fragment_dpp_row_or_spv, 345) ||
        !has_builtin(fragment_dpp_row_or_spv, 41) ||
        fragment_spirv_required_subgroup_size(fragment_dpp_row_or_spv) != 64 ||
        !type_result_ids_are_nonzero(fragment_dpp_row_or_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_dpp_row_or_spv)) {
        printf("  [FAIL] Astro fragment DPP row-right OR reduction did not emit valid subgroup SPIR-V\n");
        return 1;
    }
    printf("  [ok]   Astro fragment DPP/PERMLANEX/readlane reduction uses exact wave64 shuffles\n");

    const uint32_t fragment_dpp_distinct_row_or[] = {
        0x7e000281u,                         // v_mov_b32 v0, 1
        0x7e020282u,                         // v_mov_b32 v1, 2
        0x7e0402ffu, 0x12345678u,            // v_mov_b32 v2, unique old destination
        0x380402fau, 0xff011100u,            // v_or_b32_dpp v2,v0,v1 row_shr:1 BC:0
        0xf800000fu, 0x02020202u,
        0xbf810000u,
    };
    const auto fragment_dpp_distinct_spv = recompile_fragment(
        fragment_dpp_distinct_row_or, std::size(fragment_dpp_distinct_row_or));
    const uint32_t fragment_dpp_features =
        fragment_spirv_required_subgroup_features(fragment_dpp_distinct_spv);
    if (fragment_dpp_distinct_spv.empty() ||
        !has_select_with_false_constant(fragment_dpp_distinct_spv, 0x12345678u) ||
        !(fragment_dpp_features & kFragmentSubgroupShuffle) ||
        fragment_subgroup_features_supported(
            fragment_dpp_features,
            kFragmentSubgroupVote | kFragmentSubgroupArithmetic)) {
        printf("  [FAIL] unbounded fragment DPP did not preserve VDST/gate subgroup shuffle\n");
        return 1;
    }
    printf("  [ok]   unbounded fragment DPP preserves VDST and requires host shuffle support\n");

    // Syberia: Remastered's Forward+ light-list scalarization reduces an unsigned index across the
    // wave with the SAME unbounded ROW_SHR ladder, differing from Astro only in the VOP2 opcode
    // (V_MIN_U32 instead of V_OR_B32). These are the exact live packets at pc 1825..1841 of the
    // gameplay fragment program 8ab1535bbb28d3bd. DPP16 rewrites SRC0 only and an out-of-row or
    // EXEC-inactive source preserves VDST, so the lowering is opcode-independent; a per-opcode
    // allow-list silently dropped fifteen 1920x1080 HDR draws in one gameplay submit (#1627).
    const uint32_t fragment_dpp_row_min[] = {
        0x7e1402c1u,                         // v_mov_b32 v10, -1
        0x261414fau, 0xff01110au,            // v_min_u32_dpp v10,v10,v10 row_shr:1
        0x261414fau, 0xff01120au,            // row_shr:2
        0x261414fau, 0xff01140au,            // row_shr:4
        0x261414fau, 0xff01180au,            // row_shr:8
        0xd7781009u, 0x0305830au,            // v_permlanex16 v9,v10,-1,-1 BC=1
        0x2614130au,                         // v_min_u32 v10,v10,v9
        0xd7600000u, 0x00013f0au,            // v_readlane_b32 s0,v10,31
        0xd7600001u, 0x00017f0au,            // v_readlane_b32 s1,v10,63
        0x883f0100u,                         // s_or_b32 s63,s0,s1
        0xf800000fu, 0x0a0a0a0au,            // exp mrt0 v10,v10,v10,v10
        0xbf810000u,
    };
    const auto fragment_dpp_row_min_spv = recompile_fragment(
        fragment_dpp_row_min, std::size(fragment_dpp_row_min));
    if (fragment_dpp_row_min_spv.empty() ||
        !has_opcode(fragment_dpp_row_min_spv, 345) ||
        !has_builtin(fragment_dpp_row_min_spv, 41) ||
        fragment_spirv_required_subgroup_size(fragment_dpp_row_min_spv) != 64 ||
        !type_result_ids_are_nonzero(fragment_dpp_row_min_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_dpp_row_min_spv)) {
        printf("  [FAIL] Syberia fragment DPP row-right MIN reduction did not emit valid subgroup SPIR-V\n");
        return 1;
    }
    printf("  [ok]   Syberia fragment DPP row-right V_MIN_U32 reduction lowers like the OR ladder\n");

    // The boundary is deliberate and stays fail-visible: the VOP2 carry trio also writes VCC, and a
    // ROW_SHR-disabled lane would have to preserve its old VCC bit, which the VDST-only epilogue
    // does not model. VCC is defined first so the rejection cannot come from the missing carry-in.
    const uint32_t fragment_dpp_row_carry[] = {
        0x7d840300u,                         // v_cmp_eq_u32 vcc, v0, v1   (defines VCC)
        0x7e1402c1u,                         // v_mov_b32 v10, -1
        0x501414fau, 0xff01110au,            // v_add_co_ci_u32_dpp v10,vcc,v10,v10,vcc row_shr:1
        0xf800000fu, 0x0a0a0a0au,
        0xbf810000u,
    };
    if (!recompile_fragment(fragment_dpp_row_carry,
                            std::size(fragment_dpp_row_carry)).empty()) {
        printf("  [FAIL] unbounded fragment DPP admitted a VOP2 carry-out opcode\n");
        return 1;
    }
    printf("  [ok]   unbounded fragment DPP still rejects the VOP2 VCC carry-out trio\n");
    // #1474: partially-overlapping LOOPS in the fragment shell. Two back-edges whose ranges cross
    // without nesting (B's header lies inside A's body, B's back-edge outside it) are what the narrow
    // pattern structurizer calls unstructured and rejects. Since the graphics CFG dispatcher above
    // exists, that rejection is no longer the end of the road: the per-invocation dispatcher executes
    // the exact block graph, so the region lowers instead of dropping the draw.
    //
    // Two properties of this stream are worth stating, because both are easy to misread:
    //   * The crossing pair is SYNTACTIC only — pc7 is an unconditional s_branch, so pc8/pc9 (B's
    //     back-edge) are unreachable. detect_divergent_loops collects back-edges without a
    //     reachability filter, so the pair check still sees the overlap and rejects. If that pass
    //     ever gains reachability pruning, the accept assertion below fails for a GOOD reason.
    //   * The rejection is OVER-DETERMINED: pc3's execz targets 10, past A's exit_pc of 8 (the dword
    //     after A's back-edge, not A's exit target), so pass 2's "conditional jump past the loop" rule
    //     would reject this stream even with the overlap check deleted. Neither this assertion nor the
    //     pre-#1474 one isolates overlap as the sole cause of rejection.
    //
    // Both directions are pinned here, in the DEVICE-FREE test, rather than only in the
    // Vulkan-execution tests: those are gated on find_package(Vulkan) succeeding, and every CI job
    // that runs ctest either disables Vulkan discovery (Linux, Windows MinGW, macOS) or runs a
    // three-test seam subset (Windows App), so a guard living only there never runs in CI at all.
    const uint32_t overlapping_loops[] = {
        0xbe800380u,               //  0: s_mov_b32 s0, 0
        0x7e020284u,               //  1: v_mov_b32 v1, 4
        0x7da20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880006u,               //  3: s_cbranch_execz +6 -> 10 (export)
        0x7da20200u,               //  4: B_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880006u,               //  5: s_cbranch_execz +6 -> 12 (endpgm)
        0x81008100u,               //  6: s0++
        0xbf82fffau,               //  7: s_branch -6 -> 2  (A back-edge; A = [2,7])
        0x81008100u,               //  8: s0++
        0xbf82fffau,               //  9: s_branch -6 -> 4  (B back-edge; B = [4,9] crosses A)
        0xf800180fu, 0x05020302u,  // 10: exp mrt0
        0xbf810000u,               // 12: s_endpgm
    };
    const auto overlapping_spv =
        recompile_fragment(overlapping_loops, std::size(overlapping_loops));
    if (overlapping_spv.empty() || !has_opcode(overlapping_spv, 251) ||
        !type_result_ids_are_nonzero(overlapping_spv, nullptr) ||
        !phi_ids_are_nonzero(overlapping_spv)) {
        printf("  [FAIL] #1474: partially-overlapping fragment loops did not lower through a valid "
               "OpSwitch dispatcher\n");
        return 1;
    }
    // Lowering is not enough: a dispatcher that emitted the block graph but dropped the export would
    // satisfy every check above, and the device-side test only asserts the readback's size — that is
    // exactly the "silent skip drops real rendered content" failure the charter warns about. This
    // stream has one EXP site, so it must produce exactly one output store. The count is exact rather
    // than >= 1 so a DOUBLED export (which would write MRT0 twice) fails too; the neighbouring
    // two-site test asserting stores == 2 is the cross-check that this counts sites, not components.
    const OutputStoreStats overlapping_outputs = output_store_stats(overlapping_spv);
    if (overlapping_outputs.stores != 1) {
        printf("  [FAIL] #1474: dispatcher-lowered overlapping loops did not export exactly once "
               "(stores=%u)\n", overlapping_outputs.stores);
        return 1;
    }
    printf("  [ok]   #1474: partially-overlapping fragment loops lower through the CFG dispatcher, "
           "exporting exactly once\n");

    // The fail-visible backstop still has to work. A cross-lane MBCNT inside the same region
    // disqualifies the per-invocation dispatcher — inside a dispatcher case the lanes of one subgroup
    // sit at different guest blocks, so MBCNT's subgroup exclusive scan would be answering for a wave
    // that is not there. See the `reason=mbcnt-cross-lane` reject in rdna2_to_spirv.cpp; if graphics
    // ever gains a synchronized common phase that closes the gap, this assertion fails LOUDLY rather
    // than silently losing the reject coverage. With the narrow structurizer already rejecting, a
    // loud reject is the only remaining outcome. The compute-side #590 case keeps an s_barrier in its
    // region for the same purpose (test_rdna2_to_spirv.cpp).
    //
    // Branch offsets are re-based for the MBCNT's two dwords, and the MBCNT sits on the REACHABLE
    // path (pc5's fallthrough), not in the dead region above. Dropping it makes this stream lower,
    // which is what makes the guard real rather than decorative.
    const uint32_t overlapping_loops_cross_lane[] = {
        0xbe800380u,               //  0: s_mov_b32 s0, 0
        0x7e020284u,               //  1: v_mov_b32 v1, 4
        0x7da20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880008u,               //  3: s_cbranch_execz +8 -> 12 (export)
        0x7da20200u,               //  4: B_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880008u,               //  5: s_cbranch_execz +8 -> 14 (endpgm)
        0xd7650004u, 0x000100c1u,  //  6: v_mbcnt_lo_u32_b32 v4, -1, 0  (cross-lane)
        0x81008100u,               //  8: s0++
        0xbf82fff8u,               //  9: s_branch -8 -> 2  (A back-edge; A = [2,9])
        0x81008100u,               // 10: s0++
        0xbf82fff8u,               // 11: s_branch -8 -> 4  (B back-edge; B = [4,11] crosses A)
        0xf800180fu, 0x05020302u,  // 12: exp mrt0
        0xbf810000u,               // 14: s_endpgm
    };
    if (!recompile_fragment(overlapping_loops_cross_lane,
                            std::size(overlapping_loops_cross_lane)).empty()) {
        printf("  [FAIL] #1474: cross-lane MBCNT inside an unstructured fragment region must "
               "REJECT, not lower through the per-invocation dispatcher\n");
        return 1;
    }
    printf("  [ok]   #1474: cross-lane op in an unstructured fragment region still rejects loudly\n");

    // The graphics CFG dispatcher must retain the fragment shell's already-proven alpha-test
    // linearization.  This reduced Astro Bot shape prefixes the crossing-region CFG above with a
    // survivor-mask SCC early-out.  The SCC from s_andn2_b64 is a whole-wave reduction, not an SSA
    // scalar boolean; the per-invocation translation drops that optimization, narrows EXEC, and
    // OpKills failed lanes at either export.  Treating the safe branch as a dispatcher terminator
    // instead poisoned SCC and rejected the otherwise-supported material shader.
    const uint32_t fragment_cfg_kill_dispatch[] = {
        0xbe82047eu,                         // pc0:  s_mov_b64 s[2:3], exec
        0x7c020300u,                         // pc1:  v_cmp_lt_f32 vcc, v0, v1
        0x8a826a02u,                         // pc2:  s_andn2_b64 s[2:3], s[2:3], vcc
        0xbf840012u,                         // pc3:  s_cbranch_scc0 -> pc22 (wave early-out)
        0xbefe0a02u,                         // pc4:  s_wqm_b64 exec, s[2:3]
        0x7e040280u,                         // pc5:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc6:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc7:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc8:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc10: s_cbranch_vccz -> pc14
        0x7c020300u,                         // pc11: v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc12: s_cbranch_vccz -> pc15 (crossing region)
        0x7e040281u,                         // pc13: v_mov_b32 v2, 1
        0x7e060281u,                         // pc14: v_mov_b32 v3, 1
        0x7c020300u,                         // pc15: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc16: s_cbranch_vccz -> alternate export at pc20
        0xf800180fu, 0x05040302u,            // pc17: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc19: s_branch -> verified tail exit at pc23
        0xf800180fu, 0x05040302u,            // pc20: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc22: s_endpgm
        0xbf810000u,                         // pc23: branch-target s_endpgm
    };
    const auto fragment_cfg_kill_spv = recompile_fragment(
        fragment_cfg_kill_dispatch, std::size(fragment_cfg_kill_dispatch));
    if (fragment_cfg_kill_spv.empty() || !has_opcode(fragment_cfg_kill_spv, 251) ||
        !has_opcode(fragment_cfg_kill_spv, 252) ||
        !type_result_ids_are_nonzero(fragment_cfg_kill_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_kill_spv)) {
        printf("  [FAIL] complex fragment CFG lost its proven alpha-test branch linearization\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG retains alpha-test discard linearization\n");

    // Alpha-test discard via the SCALAR-BRANCH form (not v_cmpx): compare a sampled/interpolated value,
    // ANDN2 the survivor mask into a saved EXEC copy (SCC = "any lane survives"), and s_cbranch_scc0 skips
    // the shading if NO lane survives; the block then narrows EXEC (s_wqm exec, survivors) and shades. This
    // is Unity's clip()/cutout text+sprite shape (The Messenger's cutscene text, #102). The recompiler must
    // lower it — drop the wave early-out, run the block, OpKill the failed lanes at export — instead of
    // rejecting the s_cbranch_scc0, which dropped every alpha-tested text/sprite draw. Bytes assembled with
    // llvm-mc gfx1010 (round-trip verified).
    const uint32_t altest_kill_branch[] = {
        0xbe82047eu,               // s_mov_b64  s[2:3], exec
        0x7c020300u,               // v_cmp_lt_f32_e32 vcc_lo, v0, v1      (alpha < ref -> vcc)
        0x8a826a02u,               // s_andn2_b64 s[2:3], s[2:3], vcc      (survivors; SCC = any-survivor)
        0xbf840003u,               // s_cbranch_scc0 +3                    (skip shading if none survive)
        0xbefe0a02u,               // s_wqm_b64  exec, s[2:3]              (narrow EXEC to survivors)
        0x7e040300u,               // v_mov_b32  v2, v0
        0x7e060301u,               // v_mov_b32  v3, v1
        0x7e0802f2u,               // v_mov_b32  v4, 1.0
        0x7e0a02f2u,               // v_mov_b32  v5, 1.0
        0xf800180fu, 0x05040302u,  // exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,               // s_endpgm
    };
    auto altest_spv = recompile_fragment(altest_kill_branch, sizeof(altest_kill_branch) / sizeof(altest_kill_branch[0]));
    if (altest_spv.empty()) {
        printf("  [FAIL] alpha-test scalar-branch discard (s_cbranch_scc0 on kill mask) was rejected\n");
        return 1;
    }
    { uint32_t bad_op = 0;
      if (!type_result_ids_are_nonzero(altest_spv, &bad_op)) {
          printf("  [FAIL] alpha-test discard SPIR-V has an invalid result id (op=%u)\n", bad_op);
          return 1;
      } }
    { bool has_kill = false;                       // the discard must actually emit OpKill (op 252)
      for (size_t i = 5; i < altest_spv.size(); ) { uint32_t wc = altest_spv[i] >> 16, op = altest_spv[i] & 0xffff;
          if (op == 252u) { has_kill = true; break; } i += wc ? wc : 1; }
      if (!has_kill) { printf("  [FAIL] alpha-test discard SPIR-V lacks an OpKill\n"); return 1; } }
    printf("  [ok]   alpha-test scalar-branch (s_andn2+s_cbranch_scc0) lowers to a discard (OpKill), valid SPIR-V\n");

    const uint32_t cmpx_vertex[] = {
        0x7DA80300u, 0xF80008CFu, 0x03020100u, 0xBF810000u,
    };
    if (!recompile_vertex(cmpx_vertex, sizeof(cmpx_vertex) / sizeof(cmpx_vertex[0])).empty()) {
        printf("  [FAIL] vertex cmpx shader was accepted without EXEC-masked export support\n");
        return 1;
    }
    printf("  [ok]   vertex cmpx shader is rejected until EXEC-masked export is modeled\n");

    // Graphics-path resource binding: a vertex shader that fetches its position from a vertex buffer
    //   v_mov v3, 0 ; v_mov v4, 1.0 ; buffer_load_format_xy v[1:2], v0, s[8:11] idxen ; exp pos0 v1..v4
    // The format-load needs a V# descriptor's data format to translate, which lives in the resource
    // table — so recompilation must FAIL without a table and SUCCEED (to valid SPIR-V) with one.
    const uint32_t vs_fetch[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    const size_t vs_fetch_n = sizeof(vs_fetch) / sizeof(vs_fetch[0]);
    if (!recompile_vertex(vs_fetch, vs_fetch_n, nullptr).empty()) {
        printf("  [FAIL] vertex fetch was accepted without a resource table (format unknown)\n");
        return 1;
    }
    printf("  [ok]   vertex fetch is rejected without a resource table\n");

    ShaderResourceTable rt;
    ShaderResource vb{};
    vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
    vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;   // V# placed directly in user-data s[8:11]
    rt.resources.push_back(vb);
    std::vector<uint32_t> vspv = recompile_vertex(vs_fetch, vs_fetch_n, &rt);
    if (vspv.empty() || vspv[0] != 0x07230203u) {
        printf("  [FAIL] vertex fetch did not recompile to valid SPIR-V with a resource table\n");
        return 1;
    }
    printf("  [ok]   vertex fetch recompiles to valid SPIR-V with a resource table (binding 3)\n");

    // Astro's NGG vertex shader fetches Float16x4 records with an SGPR SOFFSET.  Even when the
    // descriptor stride is dword-aligned, that runtime offset can select either half of a dword;
    // all four components must be extracted relative to the complete byte address.  This exact
    // MUBUF shape previously hit [mubuf-unaligned] and dropped the draw.
    const uint32_t vs_fetch_half4_soffset[] = {
        0xb0050002u,                         // s_movk_i32 s5, 2
        0xe00c2000u, 0x05000100u,           // buffer_load_format_xyzw v[1:4], v0, s[0:3], s5 idxen
        0xf80008cfu, 0x04030201u,            // exp pos0 v1,v2,v3,v4
        0xbf810000u,
    };
    ShaderResourceTable rt_half4;
    ShaderResource vb_half4{};
    vb_half4.cls = ResourceClass::VertexBuffer;
    vb_half4.format = DataFormat::Float16;
    vb_half4.num_components = 4;
    vb_half4.binding = 5;
    vb_half4.stride = 36;
    vb_half4.fetch_pc = 1;
    vb_half4.fetch_index_mode = VertexFetchIndexMode::Shader;
    rt_half4.resources.push_back(vb_half4);
    const auto half4_spv = recompile_vertex(
        vs_fetch_half4_soffset, std::size(vs_fetch_half4_soffset), &rt_half4);
    if (half4_spv.empty() || half4_spv[0] != 0x07230203u) {
        printf("  [FAIL] Float16x4 vertex fetch with runtime SOFFSET did not recompile\n");
        return 1;
    }
    printf("  [ok]   Float16x4 vertex fetch handles a runtime SOFFSET\n");

    // Astro's world-map VS uses the same arbitrary byte-address shape for a three-component SNORM16
    // attribute. Its stride is dword-aligned, but the shader-computed SOFFSET can select either half
    // of a dword, and a component beginning at byte three must join the following dword.
    const uint32_t vs_fetch_snorm16x3_soffset[] = {
        0xb0040002u,                         // s_movk_i32 s4, 2
        0xe0082000u, 0x04000405u,            // buffer_load_format_xyz v[4:6], v5, s[0:3], s4 idxen
        0x7e0e02f2u,                         // v_mov_b32 v7, 1.0
        0xf80008cfu, 0x07060504u,            // exp pos0 v4,v5,v6,v7
        0xbf810000u,
    };
    ShaderResourceTable rt_snorm16x3;
    ShaderResource vb_snorm16x3{};
    vb_snorm16x3.cls = ResourceClass::VertexBuffer;
    vb_snorm16x3.format = DataFormat::Snorm16;
    vb_snorm16x3.num_components = 3;
    vb_snorm16x3.binding = 6;
    vb_snorm16x3.stride = 28;
    vb_snorm16x3.fetch_pc = 1;
    vb_snorm16x3.fetch_index_mode = VertexFetchIndexMode::Shader;
    rt_snorm16x3.resources.push_back(vb_snorm16x3);
    const auto snorm16x3_spv = recompile_vertex(
        vs_fetch_snorm16x3_soffset, std::size(vs_fetch_snorm16x3_soffset), &rt_snorm16x3);
    if (snorm16x3_spv.empty() || snorm16x3_spv[0] != 0x07230203u) {
        printf("  [FAIL] SNORM16x3 vertex fetch with runtime SOFFSET did not recompile\n");
        return 1;
    }
    printf("  [ok]   SNORM16x3 vertex fetch handles a runtime SOFFSET\n");

    // image_sample LOD mode per execution model (#151): OpImageSampleImplicitLod is only legal in
    // the Fragment execution model — the compute and vertex shells have no derivatives, so an
    // image_sample there must lower to OpImageSampleExplicitLod (LOD 0) or spirv-val rejects the
    // module and pipeline creation fails.
    enum : uint32_t { OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88,
                      OpImageQuerySizeLod = 103, OpImageQueryLod = 105,
                      OpImageQueryLevels = 106 };
    ShaderResourceTable rt_tex;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1; /*2D*/
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt_tex.resources.push_back(t); }

    // Compute shell: v0,v1 = uv inputs; image_sample v[0:3], v[0:1], s[8:15], s[16:19] dmask:0xf dim:2D.
    const uint32_t cs_sample[] = { 0xf0800f08u, 0x00820000u, 0xbf810000u };
    std::vector<uint32_t> cspv = recompile_valu(cs_sample, sizeof(cs_sample)/sizeof(cs_sample[0]), 2, 0, &rt_tex);
    if (cspv.empty() || cspv[0] != 0x07230203u) {
        printf("  [FAIL] compute-shell image_sample did not recompile\n");
        return 1;
    }
    if (has_opcode(cspv, OpImageSampleImplicitLod) || !has_opcode(cspv, OpImageSampleExplicitLod)) {
        printf("  [FAIL] compute-shell image_sample emitted ImplicitLod (fragment-only) instead of ExplicitLod\n");
        return 1;
    }
    printf("  [ok]   compute-shell image_sample lowers to OpImageSampleExplicitLod (LOD 0)\n");

    // GTA V gameplay's exact single-level IMAGE_LOAD_MIP / IMAGE_STORE_MIP subset. The resource
    // marker is the independent materialization proof; removing it, changing the descriptor to DCC,
    // or changing the storage classification must reject instead of silently treating mip as zero.
    constexpr uint32_t OpImageFetch = 95, OpImageWrite = 99;
    const uint32_t gta_load_mip_2d[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7 (production fold proves zero)
        0xf0043f08u, 0x00050000u,            // IMAGE_LOAD_MIP xyzw 2D, T# s[20:27]
        0xbf810000u,
    };
    ShaderResourceTable rt_zero_mip_2d;
    { ShaderResource texture{}; texture.cls = ResourceClass::Texture;
      texture.format = DataFormat::Uint32; texture.num_components = 1;
      texture.binding = 4; texture.fetch_pc = 1; texture.img_dim = 1;
      texture.width = 4; texture.height = 4; texture.depth = 1; texture.size = 64;
      texture.proven_zero_mip = true; rt_zero_mip_2d.resources.push_back(texture); }
    const std::vector<uint32_t> gta_load_mip_spv = recompile_valu(
        gta_load_mip_2d, std::size(gta_load_mip_2d), 1, 0, &rt_zero_mip_2d);
    if (gta_load_mip_spv.empty() || !has_opcode(gta_load_mip_spv, OpImageFetch)) {
        printf("  [FAIL] proven GTA V IMAGE_LOAD_MIP 2D did not lower to OpImageFetch\n");
        return 1;
    }
    ShaderResourceTable unproven_load_mip = rt_zero_mip_2d;
    unproven_load_mip.resources[0].proven_zero_mip = false;
    ShaderResourceTable compressed_load_mip = rt_zero_mip_2d;
    compressed_load_mip.resources[0].compression_enabled = true;
    ShaderResourceTable multilevel_load_mip = rt_zero_mip_2d;
    multilevel_load_mip.resources[0].declared_mip_levels = 2;
    if (!recompile_valu(gta_load_mip_2d, std::size(gta_load_mip_2d), 1, 0,
                        &unproven_load_mip).empty() ||
        !recompile_valu(gta_load_mip_2d, std::size(gta_load_mip_2d), 1, 0,
                        &compressed_load_mip).empty() ||
        !recompile_valu(gta_load_mip_2d, std::size(gta_load_mip_2d), 1, 0,
                        &multilevel_load_mip).empty()) {
        printf("  [FAIL] IMAGE_LOAD_MIP accepted an unproven, DCC, or multilevel resource\n");
        return 1;
    }
    printf("  [ok]   IMAGE_LOAD_MIP 2D requires the exact one-level uncompressed proof\n");

    const uint32_t gta_load_mip_2da[] = {
        0x7e060206u,                         // v_mov_b32 v3, s6; v2 remains the slice
        0xf0043128u, 0x00050000u,            // IMAGE_LOAD_MIP 2D_ARRAY
        0xbf810000u,
    };
    ShaderResourceTable rt_zero_mip_2da = rt_zero_mip_2d;
    rt_zero_mip_2da.resources[0].img_dim = 5;
    rt_zero_mip_2da.resources[0].depth = 2;
    rt_zero_mip_2da.resources[0].size = 128;
    const std::vector<uint32_t> gta_load_mip_array_spv = recompile_valu(
        gta_load_mip_2da, std::size(gta_load_mip_2da), 1, 0, &rt_zero_mip_2da);
    const DescriptorValidationReport gta_load_mip_array_report =
        validate_spirv_descriptor_interface(
            gta_load_mip_array_spv, &rt_zero_mip_2da, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* gta_load_mip_array_descriptor = nullptr;
    for (const auto& descriptor : gta_load_mip_array_report.descriptors)
        if (descriptor.binding == 4u) gta_load_mip_array_descriptor = &descriptor;
    if (gta_load_mip_array_spv.empty() || !has_opcode(gta_load_mip_array_spv, OpImageFetch) ||
        !gta_load_mip_array_descriptor || !gta_load_mip_array_descriptor->image_arrayed) {
        printf("  [FAIL] IMAGE_LOAD_MIP 2D_ARRAY dropped its slice/view contract\n");
        return 1;
    }
    printf("  [ok]   IMAGE_LOAD_MIP 2D_ARRAY preserves its slice in an arrayed OpImageFetch\n");

    const uint32_t gta_store_mip_2d[] = {
        0x7e0a0206u,                         // v_mov_b32 v5, s6 (production fold proves zero)
        0xf024310au, 0x00030004u, 0x00000503u,
        0xbf810000u,
    };
    ShaderResourceTable rt_zero_store_mip;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
      image.format = DataFormat::Uint32; image.num_components = 1;
      image.binding = 4; image.fetch_pc = 1; image.img_dim = 1;
      image.width = 4; image.height = 4; image.depth = 1; image.size = 64;
      image.proven_zero_mip = true; rt_zero_store_mip.resources.push_back(image); }
    const std::vector<uint32_t> gta_store_mip_spv = recompile_valu(
        gta_store_mip_2d, std::size(gta_store_mip_2d), 1, 0, &rt_zero_store_mip);
    ShaderResourceTable misclassified_store_mip = rt_zero_store_mip;
    misclassified_store_mip.resources[0].cls = ResourceClass::Texture;
    ShaderResourceTable unproven_store_mip = rt_zero_store_mip;
    unproven_store_mip.resources[0].proven_zero_mip = false;
    std::array<uint32_t, std::size(gta_store_mip_2d)> changed_store_packet{};
    std::copy(std::begin(gta_store_mip_2d), std::end(gta_store_mip_2d),
              changed_store_packet.begin());
    changed_store_packet[1] &= ~0x2000u;     // same instruction loses GLC: no longer audited shape
    if (gta_store_mip_spv.empty() || !has_opcode(gta_store_mip_spv, OpImageWrite) ||
        !recompile_valu(gta_store_mip_2d, std::size(gta_store_mip_2d), 1, 0,
                        &misclassified_store_mip).empty() ||
        !recompile_valu(gta_store_mip_2d, std::size(gta_store_mip_2d), 1, 0,
                        &unproven_store_mip).empty() ||
        !recompile_valu(changed_store_packet.data(), changed_store_packet.size(), 1, 0,
                        &rt_zero_store_mip).empty()) {
        printf("  [FAIL] IMAGE_STORE_MIP lowering/classification/packet gate drifted\n");
        return 1;
    }
    printf("  [ok]   IMAGE_STORE_MIP requires storage classification and its exact proven packet\n");

    const uint32_t gta_store_mip_xyzw_2d[] = {
        0x7e0c0206u,                         // v_mov_b32 v6, s6
        0xf0243f0au, 0x00030005u, 0x00000604u, // live pc17: v[0:3], (v5,v4), mip v6
        0xbf810000u,
    };
    ShaderResourceTable rt_zero_store_mip_xyzw = rt_zero_store_mip;
    rt_zero_store_mip_xyzw.resources[0].format = DataFormat::Uint8;
    rt_zero_store_mip_xyzw.resources[0].num_components = 4;
    const std::vector<uint32_t> gta_store_mip_xyzw_spv = recompile_valu(
        gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d), 1, 0,
        &rt_zero_store_mip_xyzw);
    ShaderResourceTable wrong_format_store_mip = rt_zero_store_mip_xyzw;
    wrong_format_store_mip.resources[0].format = DataFormat::Uint32;
    ShaderResourceTable compressed_store_mip = rt_zero_store_mip_xyzw;
    compressed_store_mip.resources[0].compression_enabled = true;
    ShaderResourceTable multilevel_store_mip = rt_zero_store_mip_xyzw;
    multilevel_store_mip.resources[0].declared_mip_levels = 2;
    std::array<uint32_t, std::size(gta_store_mip_xyzw_2d)> partial_store_packet{};
    std::copy(std::begin(gta_store_mip_xyzw_2d), std::end(gta_store_mip_xyzw_2d),
              partial_store_packet.begin());
    partial_store_packet[1] = (partial_store_packet[1] & ~0xf00u) | 0x300u;
    if (gta_store_mip_xyzw_spv.empty() ||
        !has_opcode(gta_store_mip_xyzw_spv, OpImageWrite) ||
        !recompile_valu(gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d), 1, 0,
                        &wrong_format_store_mip).empty() ||
        !recompile_valu(gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d), 1, 0,
                        &compressed_store_mip).empty() ||
        !recompile_valu(gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d), 1, 0,
                        &multilevel_store_mip).empty() ||
        !recompile_valu(partial_store_packet.data(), partial_store_packet.size(), 1, 0,
                        &rt_zero_store_mip_xyzw).empty()) {
        printf("  [FAIL] live dmask-xyzw IMAGE_STORE_MIP widened past its exact format/safety gate\n");
        return 1;
    }
    printf("  [ok]   live dmask-xyzw IMAGE_STORE_MIP keeps exact format/DCC/mip/dmask gates\n");

    // Vertex shell: image_sample then export the result as the position.
    const uint32_t vs_sample[] = { 0xf0800f08u, 0x00820000u, 0xf80008cfu, 0x03020100u, 0xbf810000u };
    std::vector<uint32_t> vsspv = recompile_vertex(vs_sample, sizeof(vs_sample)/sizeof(vs_sample[0]), &rt_tex);
    if (vsspv.empty() || vsspv[0] != 0x07230203u) {
        printf("  [FAIL] vertex-shell image_sample did not recompile\n");
        return 1;
    }
    if (has_opcode(vsspv, OpImageSampleImplicitLod) || !has_opcode(vsspv, OpImageSampleExplicitLod)) {
        printf("  [FAIL] vertex-shell image_sample emitted ImplicitLod (fragment-only) instead of ExplicitLod\n");
        return 1;
    }
    printf("  [ok]   vertex-shell image_sample lowers to OpImageSampleExplicitLod (LOD 0)\n");

    // Fragment shell must KEEP implicit LOD (derivative-based mip selection is the hardware behavior).
    const uint32_t ps_sample[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> pspv = recompile_fragment(ps_sample, sizeof(ps_sample)/sizeof(ps_sample[0]), &rt_tex);
    if (pspv.empty() || !has_opcode(pspv, OpImageSampleImplicitLod)) {
        printf("  [FAIL] fragment image_sample no longer uses OpImageSampleImplicitLod\n");
        return 1;
    }
    printf("  [ok]   fragment image_sample still uses OpImageSampleImplicitLod\n");

    // House of the Dead 2's Unity scene shaders use IMAGE_GET_LOD on ordinary 2D textures before
    // their explicit-LOD samples. The exact instruction returns clamped/raw LOD in the selected
    // VDATA channels. A constant replacement would hide the rejected opcode while choosing the
    // wrong mip, so require the real derivative-consuming SPIR-V image query. SPIR-V permits that
    // query only in a fragment stage; the same bytes must remain fail-visible in compute.
    ShaderResourceTable rt_get_lod;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
      t.width = 1024; t.height = 1024; t.sgpr_base = 32; t.sampler_sgpr_base = 40;
      rt_get_lod.resources.push_back(t); }
    constexpr uint32_t lod_u_bits = 0x3e800000u; // 0.25, distinct provenance sentinels
    constexpr uint32_t lod_v_bits = 0x3f400000u; // 0.75
    auto compile_get_lod = [&](uint32_t dmask) {
        const uint32_t export_mask = dmask == 3u ? 3u : 1u;
        const std::array<uint32_t, 10> shader = {
            0x7e1202ffu, lod_u_bits,           // v_mov_b32 v9, 0.25
            0x7e1402ffu, lod_v_bits,           // v_mov_b32 v10, 0.75
            0x7da40100u,                       // v_cmpx_eq_u32 v0,v0: force an EXEC-predicated write
            0xf1800008u | (dmask << 8), 0x01480809u,
            0xf8000000u | export_mask, 0x0b0a0908u, // export selected VDATA from v8[/v9]
            0xbf810000u,
        };
        return recompile_fragment(shader.data(), shader.size(), &rt_get_lod);
    };
    const std::vector<uint32_t> get_lod_x_spv = compile_get_lod(1u);
    const std::vector<uint32_t> get_lod_y_spv = compile_get_lod(2u);
    const std::vector<uint32_t> get_lod_xy_spv = compile_get_lod(3u);
    if (get_lod_x_spv.empty() ||
        !image_query_lod_contract(get_lod_x_spv, lod_u_bits, lod_v_bits,
                                  std::array<int, 4>{0, -1, -1, -1})) {
        printf("  [FAIL] IMAGE_GET_LOD dmask:x lost coordinate, clamped-LOD, or predicated VDATA flow\n");
        return 1;
    }
    if (get_lod_y_spv.empty() ||
        !image_query_lod_contract(get_lod_y_spv, lod_u_bits, lod_v_bits,
                                  std::array<int, 4>{1, -1, -1, -1})) {
        printf("  [FAIL] IMAGE_GET_LOD dmask:y did not compact raw LOD into VDATA[0]\n");
        return 1;
    }
    if (get_lod_xy_spv.empty() ||
        !image_query_lod_contract(get_lod_xy_spv, lod_u_bits, lod_v_bits,
                                  std::array<int, 4>{0, 1, -1, -1})) {
        printf("  [FAIL] IMAGE_GET_LOD dmask:xy lost clamped/raw order in consecutive VDATA\n");
        return 1;
    }
    printf("  [ok]   IMAGE_GET_LOD preserves uv, clamped/raw x/y, compact VDATA, and EXEC predication\n");

    // Exact ordinary title packet. The compute negative is intentionally isolated from every MIMG
    // control gate: mutating only `!b.is_fragment` makes this named check fail.
    const uint32_t ps_get_lod[] = {
        0xf1800108u, 0x01480809u,           // House pc-99: dmask:x, ordinary FP32 2D form
        0xf8000001u, 0x0b0a0908u,
        0xbf810000u,
    };
    const uint32_t cs_get_lod[] = {
        ps_get_lod[0], ps_get_lod[1],
        0xbf810000u,
    };
    if (!recompile_valu(cs_get_lod, std::size(cs_get_lod), 0, 0, &rt_get_lod).empty()) {
        printf("  [FAIL] compute-shell IMAGE_GET_LOD bypassed the fragment-only derivative contract\n");
        return 1;
    }
    printf("  [ok]   compute shell rejects only the fragment-stage IMAGE_GET_LOD contract\n");

    // Every non-ordinary Table 100 control remains fail-visible in both production and the
    // table-less classifier. NSA/A16/DLC/GLC/SLC/R128/TFE/LWE are exact llvm-mc gfx1030 encodings.
    // UNRM comes directly from Table 100 because LLVM exposes no IMAGE_GET_LOD spelling for it;
    // LLVM rejects D16 for IMAGE_GET_LOD. Their raw fields and every reserved hole still reject.
    struct UnsupportedGetLod {
        const char* name;
        std::vector<uint32_t> instruction;
    };
    const std::vector<UnsupportedGetLod> unsupported_get_lod = {
        {"NSA",      {0xf180010au, 0x01480809u, 0x0000000au}},
        {"UNRM",     {0xf1801108u, 0x01480809u}},
        {"A16",      {0xf1800108u, 0x41480809u}},
        {"DLC",      {0xf1800188u, 0x01480809u}},
        {"GLC",      {0xf1802108u, 0x01480809u}},
        {"SLC",      {0xf3800108u, 0x01480809u}},
        {"R128",     {0xf1808108u, 0x01480809u}},
        {"TFE",      {0xf1810108u, 0x01480809u}},
        {"LWE",      {0xf1820108u, 0x01480809u}},
        {"D16-raw",  {0xf1800108u, 0x81480809u}},
        {"reserved-w0-b6",  {0xf1800148u, 0x01480809u}},
        {"reserved-w0-b14", {0xf1804108u, 0x01480809u}},
        {"reserved-w1-b26", {0xf1800108u, 0x05480809u}},
        {"reserved-w1-b27", {0xf1800108u, 0x09480809u}},
        {"reserved-w1-b28", {0xf1800108u, 0x11480809u}},
        {"reserved-w1-b29", {0xf1800108u, 0x21480809u}},
    };
    for (const auto& form : unsupported_get_lod) {
        std::vector<uint32_t> fragment = form.instruction;
        fragment.insert(fragment.end(), {0xf8000001u, 0x0b0a0908u, 0xbf810000u});
        if (!recompile_fragment(fragment.data(), fragment.size(), &rt_get_lod).empty()) {
            printf("  [FAIL] IMAGE_GET_LOD %s control was accepted by production\n", form.name);
            return 1;
        }
    }
    printf("  [ok]   all unsupported IMAGE_GET_LOD controls reject in production\n");

    // UE4/DOLL volume initialization starts by querying a 3D T# with image_get_resinfo, then uses the
    // xyz result for its dispatch bounds. This was the sole rejected opcode in that captured kernel.
    ShaderResourceTable rt_3d;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 2;
      t.width = 8; t.height = 8; t.sgpr_base = 0; rt_3d.resources.push_back(t); }
    const uint32_t cs_resinfo[] = {
        0x7e060280u,                         // v_mov_b32 v3, 0 (LOD)
        0xf0380710u, 0x00000003u,           // image_get_resinfo v[0:2], v3, s[0:7] dmask:xyz dim:3D
        0xbf810000u,
    };
    std::vector<uint32_t> resinfo_spv = recompile_valu(
        cs_resinfo, sizeof(cs_resinfo)/sizeof(cs_resinfo[0]), 0, 0, &rt_3d);
    if (resinfo_spv.empty() || !has_opcode(resinfo_spv, OpImageQuerySizeLod) ||
        !has_opcode(resinfo_spv, OpImageQueryLevels)) {
        printf("  [FAIL] image_get_resinfo 3D did not lower to SPIR-V image queries\n");
        return 1;
    }
    printf("  [ok]   image_get_resinfo 3D lowers to size/level image queries\n");

    // Astro Bot's exact visibility-image packet: exchange v9 with R32_UINT texel (v0,v1), GLC=1.
    // Both SPIR-V operations are essential: accepting the MIMG without a real texel pointer/atomic
    // would merely hide the rejection while dropping the image side effect.
    const uint32_t ps_image_atomic[] = {
        0x7e000280u, 0x7e020280u, 0x7e120280u,
        0xf03c2108u, 0x00000900u,
        0x7e000280u, 0x7e020309u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResourceTable rt_atomic_image;
    uint32_t atomic_image_pixel = 0;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
      image.format = DataFormat::Uint32; image.num_components = 1;
      image.binding = 4; image.img_dim = 1; image.width = 1; image.height = 1;
      image.depth = 1; image.sgpr_base = 0; image.size = sizeof(atomic_image_pixel);
      image.host_data = reinterpret_cast<uint8_t*>(&atomic_image_pixel);
      image.host_data_size = sizeof(atomic_image_pixel); rt_atomic_image.resources.push_back(image); }
    const std::vector<uint32_t> atomic_image_spv = recompile_fragment(
        ps_image_atomic, std::size(ps_image_atomic), &rt_atomic_image);
    if (atomic_image_spv.empty() || !has_opcode(atomic_image_spv, 60u) ||
        !has_opcode(atomic_image_spv, 229u)) {
        printf("  [FAIL] image_atomic_swap did not lower through OpImageTexelPointer/OpAtomicExchange\n");
        return 1;
    }
    printf("  [ok]   image_atomic_swap lowers to a typed R32_UINT SPIR-V image atomic\n");

    // Astro Bot's world-map visibility kernel uses the adjacent GFX10 IMAGE_ATOMIC_ADD opcode.
    const uint32_t cs_image_atomic_add[] = {
        0x7e000280u, 0x7e020280u, 0x7e120281u,
        0xf0442108u, 0x00000900u,
        0xbf810000u,
    };
    ComputeShaderConfig atomic_add_config;
    atomic_add_config.local_x = 1;
    const std::vector<uint32_t> atomic_add_spv = recompile_compute(
        cs_image_atomic_add, std::size(cs_image_atomic_add),
        &rt_atomic_image, atomic_add_config);
    const DescriptorValidationReport atomic_add_report = validate_spirv_descriptor_interface(
        atomic_add_spv, &rt_atomic_image, 0, SpirvShaderStage::Compute, false);
    if (atomic_add_spv.empty() || has_opcode(atomic_add_spv, 60u) ||
        !has_opcode(atomic_add_spv, 65u) || !has_opcode(atomic_add_spv, 234u) ||
        !has_opcode(atomic_add_spv, 176u) || !has_opcode(atomic_add_spv, 167u) ||
        !has_opcode(atomic_add_spv, 250u) || !has_opcode(atomic_add_spv, 245u) ||
        !atomic_add_report.ok() || atomic_add_report.descriptors.size() != 1 ||
        atomic_add_report.descriptors[0].kind != SpirvDescriptorKind::StorageBuffer ||
        !atomic_add_report.descriptors[0].atomic_access) {
        printf("  [FAIL] compute image_atomic_add lacks its guarded atomic-buffer lowering\n");
        return 1;
    }
    printf("  [ok]   compute image_atomic_add uses a bounds-checked atomic buffer view\n");
    ShaderResourceTable undersized_atomic_image = rt_atomic_image;
    undersized_atomic_image.resources[0].size = sizeof(atomic_image_pixel) - 1;
    const DescriptorValidationReport undersized_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &undersized_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (undersized_atomic_report.ok()) {
        printf("  [FAIL] compute atomic-buffer view accepted an undersized image backing\n");
        return 1;
    }
    printf("  [ok]   compute atomic-buffer view rejects an undersized image backing\n");

    // --- #2265: IMAGE_ATOMIC_ADD on a 2D_ARRAY R32_UINT image ----------------------------------
    // Sonic Racing: CrossWorlds issues this on a default launch against a two-layer 3840x2160 R32
    // image and the whole full-screen dispatch was skipped every frame. The cause was not a missing
    // feature: #2272 widened the LOWERING gate over the array layer while the descriptor validator
    // and the backend materialization kept the single-layer clause, so the recompiler emitted a
    // StorageBuffer binding that its own validator rejected as WrongType.
    //
    // The instruction is the 2D fixture above with DIM changed from 1 to 5. `mimg_dim` is
    // `(word0 >> 3) & 0x7` (rdna2_decode.cpp:675), so bits [5:3] go 001 -> 101 and the low byte goes
    // 0x08 -> 0x28 -- which is exactly the low byte of CrossWorlds' own `f0440128`. Its `len` is 2,
    // not 3: an arrayed atomic reaches its layer through the address VGPRs, not a longer encoding.
    {
        const uint32_t cs_atomic_add_2d_array[] = {
            0x7e000280u, 0x7e020280u, 0x7e120281u,
            0xf0442128u, 0x00000900u,
            0xbf810000u,
        };
        // Height is 97 -- prime, and chosen so its appearance in the constant pool CANNOT be
        // incidental. The first version of this arm used height 3 and was VOID: a 3 occurs in the
        // module for unrelated reasons, so deleting the layer from the index left the arm passing.
        // The discriminator is only as good as the improbability of the constant it looks for.
        // The 2D index is x + y*width and never needs `height`; the arrayed index is
        // x + (z*height + y)*width and cannot avoid it.
        ShaderResourceTable rt_layered;
        std::vector<uint32_t> layered_backing(5u * 97u * 2u, 0u);
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.format = DataFormat::Uint32; image.num_components = 1;
          image.binding = 4; image.img_dim = 5; image.width = 5; image.height = 97;
          image.depth = 2; image.sgpr_base = 0;
          image.size = layered_backing.size() * sizeof(uint32_t);
          image.host_data = reinterpret_cast<uint8_t*>(layered_backing.data());
          image.host_data_size = image.size; rt_layered.resources.push_back(image); }

        const std::vector<uint32_t> layered_spv = recompile_compute(
            cs_atomic_add_2d_array, std::size(cs_atomic_add_2d_array),
            &rt_layered, atomic_add_config);
        const DescriptorValidationReport layered_report = validate_spirv_descriptor_interface(
            layered_spv, &rt_layered, 0, SpirvShaderStage::Compute, false);
        if (layered_spv.empty() || !layered_report.ok() ||
            layered_report.descriptors.size() != 1 ||
            layered_report.descriptors[0].kind != SpirvDescriptorKind::StorageBuffer ||
            !layered_report.descriptors[0].atomic_access) {
            printf("  [FAIL] 2D_ARRAY image_atomic_add did not recompile into an accepted "
                   "atomic-buffer contract (this is the #2265 skip)\n");
            return 1;
        }
        printf("  [ok]   2D_ARRAY image_atomic_add recompiles AND validates as an atomic buffer\n");

        // Is `value` used as a MULTIPLIER -- does some OpIMul take the constant carrying it?
        //
        // Mere PRESENCE of the constant is not a discriminator, and assuming it was is how two
        // earlier versions of this arm passed while the layer was deleted from the index: the
        // bounds check `coords[1] < height` already puts `height` in the constant pool, so it is
        // there either way. Only the arrayed index MULTIPLIES by it, in (z*height + y).
        const auto multiplies_by = [](const std::vector<uint32_t>& spv, uint32_t value) {
            uint32_t constant_id = 0;
            for (size_t w = 5; w + 1 < spv.size();) {
                const uint32_t count = spv[w] >> 16, op = spv[w] & 0xffffu;
                if (!count || w + count > spv.size()) break;
                if (op == 43u && count == 4u && spv[w + 3] == value) { constant_id = spv[w + 2]; break; }
                w += count;
            }
            if (!constant_id) return false;
            for (size_t w = 5; w + 1 < spv.size();) {
                const uint32_t count = spv[w] >> 16, op = spv[w] & 0xffffu;
                if (!count || w + count > spv.size()) break;
                if (op == 132u && count == 5u &&                      // OpIMul
                    (spv[w + 3] == constant_id || spv[w + 4] == constant_id)) return true;
                w += count;
            }
            return false;
        };

        // MUTATION ARM 1 -- the layer must reach the INDEX. Without it the lowering computes
        // x + y*width for every layer, every layer aliases layer 0, and the dispatch produces a
        // silently wrong result rather than being skipped: strictly worse than the bug it replaces.
        // `height` (3) appears only if the layer was multiplied in.
        if (!multiplies_by(layered_spv, 97u)) {
            printf("  [FAIL] the 2D_ARRAY atomic index does not multiply by height -- every layer "
                   "aliases layer 0\n");
            return 1;
        }
        // Paired negative, so the discriminator's validity is checked rather than assumed -- and it
        // must use the SAME distinctive constant, on a NON-arrayed shader, or it tests nothing. An
        // earlier version asked whether the 1x1 fixture multiplies by 1; it does, because 1 is
        // multiplied all over any module, and that arm then failed on correct code. Here the shape
        // is 5x97 and non-arrayed, so 97 is present as a bound and must NOT be a multiplier: the 2D
        // index is x + y*width and reaches only the WIDTH.
        ShaderResourceTable rt_tall_2d;
        std::vector<uint32_t> tall_backing(5u * 97u, 0u);
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.format = DataFormat::Uint32; image.num_components = 1;
          image.binding = 4; image.img_dim = 1; image.width = 5; image.height = 97;
          image.depth = 1; image.sgpr_base = 0;
          image.size = tall_backing.size() * sizeof(uint32_t);
          image.host_data = reinterpret_cast<uint8_t*>(tall_backing.data());
          image.host_data_size = image.size; rt_tall_2d.resources.push_back(image); }
        const std::vector<uint32_t> tall_2d_spv = recompile_compute(
            cs_image_atomic_add, std::size(cs_image_atomic_add), &rt_tall_2d, atomic_add_config);
        if (tall_2d_spv.empty() || multiplies_by(tall_2d_spv, 97u)) {
            printf("  [FAIL] the discriminator is void: a NON-arrayed 5x97 index also multiplies by "
                   "97, so the arrayed result proves nothing\n");
            return 1;
        }
        printf("  [ok]   the layer reaches the index (height constant present only when arrayed)\n");

        // MUTATION ARM 2 -- the layer must reach the SIZE BOUND. A backing sized for one layer must
        // be rejected; the pre-#2265 check was width*height*4 <= size and would accept it.
        ShaderResourceTable one_layer_backing = rt_layered;
        one_layer_backing.resources[0].size = 5u * 97u * sizeof(uint32_t);
        if (validate_spirv_descriptor_interface(layered_spv, &one_layer_backing, 0,
                                                SpirvShaderStage::Compute, false).ok()) {
            printf("  [FAIL] a two-layer atomic image was accepted against one layer of backing\n");
            return 1;
        }
        printf("  [ok]   the size bound counts every layer, not just the first\n");
    }

    ShaderResourceTable compressed_atomic_image = rt_atomic_image;
    compressed_atomic_image.resources[0].compression_enabled = true;
    const DescriptorValidationReport compressed_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &compressed_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (compressed_atomic_report.ok() ||
        !recompile_compute(cs_image_atomic_add, std::size(cs_image_atomic_add),
                           &compressed_atomic_image, atomic_add_config).empty()) {
        printf("  [FAIL] compute atomic-buffer view accepted a DCC-compressed image\n");
        return 1;
    }
    ShaderResourceTable tail_atomic_image = rt_atomic_image;
    tail_atomic_image.resources[0].in_mip_tail = true;
    const DescriptorValidationReport tail_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &tail_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (tail_atomic_report.ok() ||
        !recompile_compute(cs_image_atomic_add, std::size(cs_image_atomic_add),
                           &tail_atomic_image, atomic_add_config).empty()) {
        printf("  [FAIL] compute atomic-buffer view accepted a mip-tail image\n");
        return 1;
    }
    printf("  [ok]   compute atomic-buffer view rejects compressed and mip-tail images\n");

    // GTA V's program 0x205b5e8600 uses the supported RTIP 1.1 BVH instruction at exact pc 1476
    // with eleven NSA address operands and a 64-byte descriptor. It is lowered to ordinary SSBO
    // loads and scalar ALU, so this remains usable on Vulkan devices without a ray-query feature.
    // Keep the gate exact: accepting a nearby MIMG flag combination would silently assign the wrong
    // hardware intersection contract.
    constexpr uint32_t gta_bvh_pc = 1476u;
    std::vector<uint32_t> cs_bvh_intersect(gta_bvh_pc, 0xbf800000u); // s_nop 0
    const uint32_t gta_bvh_packet[] = {
        0xf1989f07u, 0x00060202u, 0x28292c23u, 0x22262725u, 0x00002a24u,
        0xbf810000u,
    };
    cs_bvh_intersect.insert(cs_bvh_intersect.end(), std::begin(gta_bvh_packet),
                            std::end(gta_bvh_packet));
    uint32_t bvh_node_words[16]{};
    ShaderResourceTable rt_bvh;
    { ShaderResource bvh{}; bvh.cls = ResourceClass::ConstantBuffer;
      bvh.format = DataFormat::Uint32; bvh.num_components = 1;
      bvh.binding = 4; bvh.size = sizeof(bvh_node_words); bvh.fetch_pc = gta_bvh_pc;
      bvh.host_data = reinterpret_cast<uint8_t*>(bvh_node_words);
      bvh.host_data_size = sizeof(bvh_node_words); rt_bvh.resources.push_back(bvh); }
    ComputeShaderConfig bvh_config;
    bvh_config.local_x = 1;
    const std::vector<uint32_t> bvh_spv = recompile_compute(
        cs_bvh_intersect.data(), cs_bvh_intersect.size(), &rt_bvh, bvh_config);
    const DescriptorValidationReport bvh_report = validate_spirv_descriptor_interface(
        bvh_spv, &rt_bvh, 0, SpirvShaderStage::Compute, false);
    if (bvh_spv.empty() || !bvh_report.ok() || bvh_report.descriptors.size() != 1 ||
        bvh_report.descriptors[0].kind != SpirvDescriptorKind::StorageBuffer ||
        !has_opcode(bvh_spv, 61u) || !has_opcode(bvh_spv, 129u) ||
        !has_opcode(bvh_spv, 133u) || !has_opcode(bvh_spv, 169u)) {
        printf("  [FAIL] GTA's 64-byte IMAGE_BVH_INTERSECT_RAY lacks its SSBO/ALU lowering\n");
        return 1;
    }
    if (!recompile_compute(cs_bvh_intersect.data(), cs_bvh_intersect.size(),
                           nullptr, bvh_config).empty()) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY was accepted without its BVH bytes\n");
        return 1;
    }
    ShaderResourceTable sorted_bvh = rt_bvh;
    sorted_bvh.resources[0].bvh_sort_enabled = true;
    const std::vector<uint32_t> sorted_bvh_spv = recompile_compute(
        cs_bvh_intersect.data(), cs_bvh_intersect.size(), &sorted_bvh, bvh_config);
    if (sorted_bvh_spv.empty() || sorted_bvh_spv == bvh_spv) {
        printf("  [FAIL] sorted IMAGE_BVH_INTERSECT_RAY lacks distinct box-order lowering\n");
        return 1;
    }
    std::vector<uint32_t> unsupported_bvh = cs_bvh_intersect;
    unsupported_bvh[gta_bvh_pc] &= ~(1u << 15); // R128=0 has a different destination contract.
    if (!recompile_compute(unsupported_bvh.data(), unsupported_bvh.size(),
                           &rt_bvh, bvh_config).empty()) {
        printf("  [FAIL] unverified IMAGE_BVH_INTERSECT_RAY flags were accepted\n");
        return 1;
    }
    ShaderResourceTable short_bvh = rt_bvh;
    short_bvh.resources[0].size = 60u;
    if (!recompile_compute(cs_bvh_intersect.data(), cs_bvh_intersect.size(),
                           &short_bvh, bvh_config).empty()) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY accepted a sub-64-byte BVH\n");
        return 1;
    }
    ShaderResourceTable misaligned_bvh = rt_bvh;
    misaligned_bvh.resources[0].size = 66u;
    if (!recompile_compute(cs_bvh_intersect.data(), cs_bvh_intersect.size(),
                           &misaligned_bvh, bvh_config).empty()) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY accepted a non-dword-aligned BVH\n");
        return 1;
    }
    ShaderResourceTable wrong_class_bvh = rt_bvh;
    wrong_class_bvh.resources[0].cls = ResourceClass::VertexBuffer;
    if (!recompile_compute(cs_bvh_intersect.data(), cs_bvh_intersect.size(),
                           &wrong_class_bvh, bvh_config).empty()) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY accepted a non-constant-buffer resource\n");
        return 1;
    }
    printf("  [ok]   GTA's exact-pc sorted/unsorted IMAGE_BVH_INTERSECT_RAY lowers through a bounded BVH SSBO\n");

    // Astro Bot's visibility kernel sanitizes a generated coordinate with an explicit-SDST
    // v_cmp_class_f32 SDWA (mask 3 = sNaN|qNaN), followed by v_cndmask reading s[8:9]. Rejecting
    // the compare used to discard the entire 1,954-dword world-map compute shader.
    const uint32_t cs_class_nan[] = {
        0x7e0202ffu, 0x7fc00000u,           // v_mov_b32 v1, qNaN
        0x7d1106f9u, 0x86068801u,           // v_cmp_class_f32_sdwa s8, v1, 3
        0xd5010000u, 0x00210101u,           // v_cndmask_b32_e64 v0, v1, 0, s[8:9]
        0xbf810000u,
    };
    const std::vector<uint32_t> class_nan_spv = recompile_valu(
        cs_class_nan, std::size(cs_class_nan), 1, 0, nullptr);
    if (class_nan_spv.empty() || !has_opcode(class_nan_spv, 199u) ||
        !has_opcode(class_nan_spv, 171u) || !has_opcode(class_nan_spv, 169u) ||
        !type_result_ids_are_nonzero(class_nan_spv, nullptr)) {
        printf("  [FAIL] v_cmp_class_f32 did not lower to raw IEEE class selection/compare\n");
        return 1;
    }
    printf("  [ok]   v_cmp_class_f32 lowers raw IEEE classes and feeds its explicit SGPR mask\n");
    const uint32_t cs_class_modifiers[] = {
        0x7e0202ffu, 0x7f800000u,           // v_mov_b32 v1, +Inf
        0x7d1108f9u, 0x86168801u,           // v_cmp_class_f32_sdwa s8, -v1, 4 (-Inf)
        0x7d1108f9u, 0x86268801u,           // v_cmp_class_f32_sdwa s8, |v1|, 4
        0xbf810000u,
    };
    const std::vector<uint32_t> class_modifier_spv = recompile_valu(
        cs_class_modifiers, std::size(cs_class_modifiers), 1, 0, nullptr);
    bool has_abs_sign_mask = false;
    for (uint32_t word : class_modifier_spv) has_abs_sign_mask |= word == 0x7fffffffu;
    if (class_modifier_spv.empty() || !has_opcode(class_modifier_spv, 198u) ||
        !has_abs_sign_mask || !type_result_ids_are_nonzero(class_modifier_spv, nullptr)) {
        printf("  [FAIL] v_cmp_class_f32 dropped raw ABS/NEG sign-bit modifiers\n");
        return 1;
    }
    printf("  [ok]   v_cmp_class_f32 applies ABS/NEG without canonicalizing raw NaN bits\n");

    // Astro Bot's world-map shader samples a 192-layer BC6H 2D array with explicit LOD. The layer
    // coordinate must survive in OpTypeImage so the live backend creates a matching 2D-array view.
    ShaderResourceTable rt_array;
    { ShaderResource texture{}; texture.cls = ResourceClass::Texture;
      texture.format = DataFormat::Bc6; texture.num_components = 3;
      texture.binding = 4; texture.sgpr_base = 0; texture.img_dim = 5;
      texture.width = 4; texture.height = 4; texture.depth = 2;
      texture.gpu_addr = 0x100000; texture.size = 32;
      rt_array.resources.push_back(texture); }
    const uint32_t cs_sample_array_l[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0900f28u, 0x00400000u,         // image_sample_l dim:2D_ARRAY [u,v,slice,lod]
        0xbf810000u,
    };
    const std::vector<uint32_t> array_l_spv = recompile_valu(
        cs_sample_array_l, std::size(cs_sample_array_l), 4, 0, &rt_array);
    const DescriptorValidationReport array_l_report = validate_spirv_descriptor_interface(
        array_l_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_l_descriptor = nullptr;
    for (const auto& descriptor : array_l_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_l_descriptor = &descriptor;
    if (array_l_spv.empty() || !array_l_descriptor ||
        array_l_descriptor->image_dim != 1u || !array_l_descriptor->image_arrayed ||
        !array_l_descriptor->normalized_sampling) {
        printf("  [FAIL] 2D-array image_sample_l dropped its reflected layer contract\n");
        return 1;
    }
    printf("  [ok]   2D-array image_sample_l retains its layer coordinate and reflected view shape\n");

    // Exact title-live IMAGE_LOAD packet: v5/v6 are integer x/y and v7 is the guest sample index.
    // The host descriptor is deliberately arrayed but non-multisampled; unlike the ordinary 2D-array
    // sample above it is texel-space (OpImageFetch), and ShaderResource keeps img_dim/sample_count so
    // backend identity cannot alias these two guest resource kinds.
    const uint32_t cs_msaa_load_sample3[] = {
        0x7e0a0280u,                       // v_mov_b32 v5, 0 (x)
        0x7e0c0280u,                       // v_mov_b32 v6, 0 (y)
        0x7e0e0283u,                       // v_mov_b32 v7, 3 (sample)
        0xf0000130u, 0x00000305u,         // image_load v3, v[5:7], s[0:7] dmask:x dim:2D_MSAA
        0xbf810000u,
    };
    ShaderResourceTable rt_msaa;
    { ShaderResource texture{}; texture.cls = ResourceClass::Texture;
      texture.format = DataFormat::Float32; texture.num_components = 1;
      texture.binding = 4; texture.sgpr_base = 0; texture.img_dim = 6;
      texture.width = 1; texture.height = 1; texture.depth = 1;
      texture.sample_count = 4; texture.declared_mip_levels = 1;
      texture.gpu_addr = 0x200000; texture.size = 4 * sizeof(float);
      rt_msaa.resources.push_back(texture); }
    const std::vector<uint32_t> msaa_spv = recompile_valu(
        cs_msaa_load_sample3, std::size(cs_msaa_load_sample3), 8, 0, &rt_msaa);
    const DescriptorValidationReport msaa_report = validate_spirv_descriptor_interface(
        msaa_spv, &rt_msaa, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* msaa_descriptor = nullptr;
    for (const auto& descriptor : msaa_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            msaa_descriptor = &descriptor;
    if (msaa_spv.empty() || !msaa_descriptor ||
        msaa_descriptor->image_dim != 1u || !msaa_descriptor->image_arrayed ||
        msaa_descriptor->image_multisampled || !msaa_descriptor->texel_access ||
        msaa_descriptor->normalized_sampling ||
        !array_l_descriptor->image_arrayed || !array_l_descriptor->normalized_sampling ||
        array_l_descriptor->texel_access || rt_array.resources[0].sample_count != 1u ||
        rt_msaa.resources[0].sample_count != 4u) {
        printf("  [FAIL] guest 2D_MSAA load did not retain its distinct host-array fetch contract "
               "(words=%zu issues=%zu desc=%zu shape=%u/%d/%d access=%d/%d)\n",
               msaa_spv.size(), msaa_report.issues.size(), msaa_report.descriptors.size(),
               msaa_descriptor ? msaa_descriptor->image_dim : UINT32_MAX,
               msaa_descriptor ? msaa_descriptor->image_arrayed : false,
               msaa_descriptor ? msaa_descriptor->image_multisampled : false,
               msaa_descriptor ? msaa_descriptor->texel_access : false,
               msaa_descriptor ? msaa_descriptor->normalized_sampling : false);
        return 1;
    }
    printf("  [ok]   guest 2D_MSAA fetch is distinct from ordinary 2D-array sampling in reflection and resource identity\n");
    if (!image_fetch_coord_literals(msaa_spv, 0u, 0u, 3u)) {
        printf("  [FAIL] 2D_MSAA IMAGE_LOAD did not use the explicit sample VGPR as its host array layer\n");
        return 1;
    }
    uint32_t cs_msaa_load_sample1[std::size(cs_msaa_load_sample3)];
    std::copy(std::begin(cs_msaa_load_sample3), std::end(cs_msaa_load_sample3),
              cs_msaa_load_sample1);
    cs_msaa_load_sample1[2] = 0x7e0e0281u; // mutate only v7: guest sample 3 -> sample 1
    const std::vector<uint32_t> msaa_sample1_spv = recompile_valu(
        cs_msaa_load_sample1, std::size(cs_msaa_load_sample1), 8, 0, &rt_msaa);
    if (!image_fetch_coord_literals(msaa_sample1_spv, 0u, 0u, 1u) ||
        image_fetch_coord_literals(msaa_sample1_spv, 0u, 0u, 3u)) {
        printf("  [FAIL] changing the explicit guest sample VGPR did not change the host layer coordinate\n");
        return 1;
    }
    printf("  [ok]   the explicit guest sample VGPR independently selects the host array layer\n");

    // The same title shader uses the one-extra-dword NSA encoding for its other three samples.
    // llvm-mc gfx1030 disassembles this exact packet as
    //   image_load v2, [v5, v6, v2], s[0:7] dmask:x dim:2D_MSAA
    // so the sample VGPR is the second byte of the extra word, independently of VADDR adjacency.
    const uint32_t cs_msaa_nsa_sample1[] = {
        0x7e0a0280u,                       // v_mov_b32 v5, 0 (x)
        0x7e0c0280u,                       // v_mov_b32 v6, 0 (y)
        0x7e040281u,                       // v_mov_b32 v2, 1 (sample)
        0x7e060283u,                       // v_mov_b32 v3, 3 (mutation sample)
        0xf0000132u, 0x00000205u, 0x00000206u,
        0xbf810000u,
    };
    const std::vector<uint32_t> msaa_nsa_sample1_spv = recompile_valu(
        cs_msaa_nsa_sample1, std::size(cs_msaa_nsa_sample1), 8, 0, &rt_msaa);
    if (!image_fetch_coord_literals(msaa_nsa_sample1_spv, 0u, 0u, 1u)) {
        printf("  [FAIL] exact title NSA address did not map its third explicit VGPR to the host layer\n");
        return 1;
    }
    uint32_t cs_msaa_nsa_sample3[std::size(cs_msaa_nsa_sample1)];
    std::copy(std::begin(cs_msaa_nsa_sample1), std::end(cs_msaa_nsa_sample1),
              cs_msaa_nsa_sample3);
    cs_msaa_nsa_sample3[6] = 0x00000306u; // mutate only words[2].byte1: sample v2 -> v3
    const std::vector<uint32_t> msaa_nsa_sample3_spv = recompile_valu(
        cs_msaa_nsa_sample3, std::size(cs_msaa_nsa_sample3), 8, 0, &rt_msaa);
    if (!image_fetch_coord_literals(msaa_nsa_sample3_spv, 0u, 0u, 3u) ||
        image_fetch_coord_literals(msaa_nsa_sample3_spv, 0u, 0u, 1u)) {
        printf("  [FAIL] mutating only the NSA sample byte did not change the host layer coordinate\n");
        return 1;
    }
    printf("  [ok]   exact title NSA address bytes independently select the host array layer\n");

    uint32_t unsupported_msaa_nsa_bytes[std::size(cs_msaa_nsa_sample1)];
    std::copy(std::begin(cs_msaa_nsa_sample1), std::end(cs_msaa_nsa_sample1),
              unsupported_msaa_nsa_bytes);
    unsupported_msaa_nsa_bytes[6] |= 0x00010000u; // a nonzero, unmodelled fourth address byte
    const uint32_t unsupported_msaa_two_extra[] = {
        0x7e0a0280u, 0x7e0c0280u, 0x7e040281u,
        0xf0000134u, 0x00000205u, 0x00000206u, 0x00000000u,
        0xbf810000u,
    };
    if (!recompile_valu(unsupported_msaa_nsa_bytes, std::size(unsupported_msaa_nsa_bytes),
                        8, 0, &rt_msaa).empty() ||
        !recompile_valu(unsupported_msaa_two_extra, std::size(unsupported_msaa_two_extra),
                        8, 0, &rt_msaa).empty()) {
        printf("  [FAIL] unproved NSA address bytes/counts did not remain fail-visible\n");
        return 1;
    }
    printf("  [ok]   unproved NSA address bytes and extra-dword counts remain fail-visible\n");

    ShaderResourceTable unsupported_msaa = rt_msaa;
    unsupported_msaa.resources[0].sample_count = 2;
    uint32_t unsupported_msaa_op[std::size(cs_msaa_load_sample3)];
    std::copy(std::begin(cs_msaa_load_sample3), std::end(cs_msaa_load_sample3),
              unsupported_msaa_op);
    unsupported_msaa_op[3] |= 0x00800000u; // IMAGE_SAMPLE opcode with dim:2D_MSAA
    if (!recompile_valu(cs_msaa_load_sample3, std::size(cs_msaa_load_sample3),
                        8, 0, &unsupported_msaa).empty() ||
        !recompile_valu(unsupported_msaa_op, std::size(unsupported_msaa_op),
                        8, 0, &rt_msaa).empty()) {
        printf("  [FAIL] unsupported MSAA count/opcode did not remain fail-visible\n");
        return 1;
    }
    printf("  [ok]   unsupported MSAA counts and sampled opcodes remain fail-visible\n");

    // The same map kernel reads its two-layer RGBA atlas with the NSA SAMPLE_LZ form. Its third
    // coordinate is still a layer, despite the fixed level zero, and must select an array view too.
    const uint32_t cs_sample_array_lz[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u,
        0xf09c0f2au, 0x00400000u, 0x00000201u,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_lz_spv = recompile_valu(
        cs_sample_array_lz, std::size(cs_sample_array_lz), 4, 0, &rt_array);
    const DescriptorValidationReport array_lz_report = validate_spirv_descriptor_interface(
        array_lz_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_lz_descriptor = nullptr;
    for (const auto& descriptor : array_lz_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_lz_descriptor = &descriptor;
    if (array_lz_spv.empty() || !array_lz_descriptor ||
        array_lz_descriptor->image_dim != 1u || !array_lz_descriptor->image_arrayed ||
        !array_lz_descriptor->normalized_sampling) {
        printf("  [FAIL] 2D-array image_sample_lz dropped its reflected layer contract\n");
        return 1;
    }
    printf("  [ok]   2D-array image_sample_lz retains its layer coordinate and reflected view shape\n");

    // A single binding can be sampled with both ordinary SAMPLE and explicit SAMPLE_L. Compute has
    // no derivatives, so both use explicit LOD while preserving the common array coordinate/type.
    const uint32_t cs_sample_array_mixed[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0800f2au, 0x00400000u, 0x00000201u,
        0xf0900f2au, 0x00400000u, 0x00030201u,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_mixed_spv = recompile_valu(
        cs_sample_array_mixed, std::size(cs_sample_array_mixed), 4, 0, &rt_array);
    const DescriptorValidationReport array_mixed_report = validate_spirv_descriptor_interface(
        array_mixed_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_mixed_descriptor = nullptr;
    size_t array_mixed_texture_count = 0;
    for (const auto& descriptor : array_mixed_report.descriptors) {
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler) {
            array_mixed_descriptor = &descriptor;
            array_mixed_texture_count++;
        }
    }
    if (array_mixed_spv.empty() || array_mixed_texture_count != 1u ||
        !array_mixed_descriptor || !array_mixed_descriptor->image_arrayed) {
        printf("  [FAIL] mixed compute SAMPLE/SAMPLE_L produced incompatible image types\n");
        return 1;
    }
    printf("  [ok]   mixed compute SAMPLE/SAMPLE_L shares one 2D-array image contract\n");

    // The graphics renderer still exposes DIM=5 textures through its established base-slice 2D
    // view. Keep that descriptor non-arrayed, but consume SAMPLE_L's fourth address as the LOD
    // rather than mistaking the discarded third (slice) address for the LOD.
    const uint32_t ps_sample_array_l[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0900f28u, 0x00400000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    const std::vector<uint32_t> graphics_array_l_spv = recompile_fragment(
        ps_sample_array_l, std::size(ps_sample_array_l), &rt_array);
    const DescriptorValidationReport graphics_array_l_report =
        validate_spirv_descriptor_interface(
            graphics_array_l_spv, &rt_array, 0, SpirvShaderStage::Fragment);
    const SpirvDescriptorBinding* graphics_array_l_descriptor = nullptr;
    for (const auto& descriptor : graphics_array_l_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            graphics_array_l_descriptor = &descriptor;
    if (graphics_array_l_spv.empty() || !graphics_array_l_descriptor ||
        graphics_array_l_descriptor->image_arrayed ||
        !has_explicit_lod_constant(graphics_array_l_spv, 0u)) {
        printf("  [FAIL] graphics DIM=5 image_sample_l violated its base-slice 2D contract\n");
        return 1;
    }
    printf("  [ok]   graphics DIM=5 image_sample_l keeps a 2D view and its fourth-address LOD\n");

    // The visibility half of the same kernel comparison-samples a sixteen-layer shadow array.
    // Compute keeps its slice and performs the compare manually over a color-sampled array image.
    ShaderResourceTable rt_array_dref = rt_array;
    rt_array_dref.resources[0].depth_compare = true;
    rt_array_dref.resources[0].depth_compare_func = 4;
    const uint32_t cs_sample_array_c_lz[] = {
        0x7e1402f0u, 0x7e1602f0u, 0x7e1802f0u,
        0x7e1a02ffu, 0x3f800000u,
        0xf0bc012au, 0x0040050au, 0x000d0c0bu,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_c_lz_spv = recompile_valu(
        cs_sample_array_c_lz, std::size(cs_sample_array_c_lz), 4, 0, &rt_array_dref);
    const DescriptorValidationReport array_c_lz_report = validate_spirv_descriptor_interface(
        array_c_lz_spv, &rt_array_dref, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_c_lz_descriptor = nullptr;
    for (const auto& descriptor : array_c_lz_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_c_lz_descriptor = &descriptor;
    if (array_c_lz_spv.empty() || !array_c_lz_descriptor ||
        array_c_lz_descriptor->image_dim != 1u || !array_c_lz_descriptor->image_arrayed ||
        array_c_lz_descriptor->image_depth ||
        has_opcode(array_c_lz_spv, 90u)) {
        printf("  [FAIL] compute 2D-array image_sample_c_lz lost its manual array contract\n");
        return 1;
    }
    printf("  [ok]   compute 2D-array image_sample_c_lz preserves slice without a Dref sampler\n");

    // LDS array is sized from the shader's real allocation (#130), not a hardcoded 16 KB. A compute
    // kernel that uses ds_write/ds_read declares a Workgroup array; its length must be 4096 dwords
    // (16 KB) by default and rise to the requested size (clamped to the RDNA2 64 KB / 16384-dword max)
    // when lds_bytes is plumbed. code32 = lane i writes lds[i], barrier, reads lds[63-i].
    const uint32_t code_lds[] = {
        0x7e020f00u, 0x34040282u, 0x34060281u, 0x4a060681u, 0xd8340000u, 0x00000302u, 0xbf8a0000u,
        0x4c0a02bfu, 0x340c0a82u, 0xd8d80000u, 0x07000006u, 0x7e000d07u, 0xbf810000u,
    };
    const size_t n_lds = sizeof(code_lds)/sizeof(code_lds[0]);
    std::vector<uint32_t> lds_def = recompile_valu(code_lds, n_lds, 1, 0, nullptr);
    std::vector<uint32_t> lds_32k = recompile_valu(code_lds, n_lds, 1, 0, nullptr, 32 * 1024);
    std::vector<uint32_t> lds_big = recompile_valu(code_lds, n_lds, 1, 0, nullptr, 128 * 1024);   // > 64 KB
    if (lds_def.empty() || lds_32k.empty() || lds_big.empty()) {
        printf("  [FAIL] LDS kernel did not recompile\n");
        return 1;
    }
    if (max_array_length(lds_def) != 4096u) {
        printf("  [FAIL] default LDS array length = %u dwords, want 4096 (16 KB)\n", max_array_length(lds_def));
        return 1;
    }
    printf("  [ok]   default LDS array is 4096 dwords (16 KB)\n");
    if (max_array_length(lds_32k) != 8192u) {
        printf("  [FAIL] lds_bytes=32K -> array length = %u dwords, want 8192\n", max_array_length(lds_32k));
        return 1;
    }
    printf("  [ok]   lds_bytes=32 KB -> 8192-dword LDS array\n");
    if (max_array_length(lds_big) != 16384u) {
        printf("  [FAIL] lds_bytes=128K -> array length = %u dwords, want 16384 (clamped to 64 KB)\n",
               max_array_length(lds_big));
        return 1;
    }
    printf("  [ok]   lds_bytes>64 KB clamps to the RDNA2 max (16384 dwords)\n");

    // v_interp_mov explicit-parameter reads (#152/#897). P0-only retains the cheap Flat varying.
    // Mixed smooth+P0 and P10/P20 use the portable generated geometry stage, which publishes the
    // coefficient values through separate Flat locations. Encodings llvm-mc gfx1030 verified.
    enum : uint32_t { OpDecorate = 71, DecFlat = 14 };
    // Flat-only: v_interp_mov v3, p0, attr0.x ; exp mrt0 v3,v3,v3,v3 ; s_endpgm
    const uint32_t ps_flat[] = { 0xc80e0002u, 0xf800000fu, 0x03030303u, 0xbf810000u };
    std::vector<uint32_t> flat_spv = recompile_fragment(ps_flat, sizeof(ps_flat)/sizeof(ps_flat[0]));
    if (flat_spv.empty() || !has_decoration(flat_spv, DecFlat)) {
        printf("  [FAIL] v_interp_mov attribute is not decorated Flat (would smooth-interpolate a flat read)\n");
        return 1;
    }
    printf("  [ok]   v_interp_mov attribute varying is decorated Flat\n");
    // Smooth-only: v_interp_p1 + v_interp_p2 on attr0 ; exp mrt0 v4 -> NOT Flat.
    const uint32_t ps_smooth[] = { 0xc8080000u, 0xc8110002u, 0xf800000fu, 0x04040404u, 0xbf810000u };
    std::vector<uint32_t> smooth_spv = recompile_fragment(ps_smooth, sizeof(ps_smooth)/sizeof(ps_smooth[0]));
    if (smooth_spv.empty() || has_decoration(smooth_spv, DecFlat)) {
        printf("  [FAIL] a smooth-interpolated (v_interp_p2) attribute must NOT be decorated Flat\n");
        return 1;
    }
    printf("  [ok]   v_interp_p2-only attribute stays smooth (no Flat decoration)\n");
    // Mixed: attr0 read via BOTH P0 and smooth interpolation needs separate interface locations.
    const uint32_t ps_mixed[] = { 0xc80e0002u, 0xc8110002u, 0xf800000fu, 0x03030303u, 0xbf810000u };
    FragmentInterpolationLayout mixed_layout = fragment_interpolation_layout(
        ps_mixed, sizeof(ps_mixed)/sizeof(ps_mixed[0]));
    std::vector<uint32_t> mixed_spv = recompile_fragment(
        ps_mixed, sizeof(ps_mixed)/sizeof(ps_mixed[0]), nullptr, nullptr, UINT32_MAX, &mixed_layout);
    std::vector<uint32_t> mixed_gs = recompile_interpolation_geometry(mixed_layout);
    if (!mixed_layout.valid || !mixed_layout.requires_geometry || mixed_spv.empty() ||
        mixed_gs.empty() || !has_opcode(mixed_gs, 218) || !has_opcode(mixed_gs, 219)) {
        printf("  [FAIL] mixed P0+smooth interpolation did not generate a geometry fallback\n");
        return 1;
    }
    printf("  [ok]   mixed P0+smooth interpolation generates coefficient pass-through SPIR-V\n");

    // Transform feedback must decorate the final pre-rasterization stage. When interpolation needs
    // the generated GS, decorating only the VS produces real pixels but writes zero probe vertices.
    const std::vector<uint32_t> mixed_xfb_gs =
        recompile_interpolation_geometry(mixed_layout, true);
    constexpr uint32_t OpCapability = 17, OpExecutionMode = 16, OpMemberDecorate = 72;
    constexpr uint32_t CapTransformFeedback = 53, ExecutionModeXfb = 11;
    constexpr uint32_t DecOffset = 35, DecXfbBuffer = 36, DecXfbStride = 37;
    if (mixed_xfb_gs.empty() ||
        !has_instruction_operand(mixed_xfb_gs, OpCapability, 0, CapTransformFeedback) ||
        !has_instruction_operand(mixed_xfb_gs, OpExecutionMode, 1, ExecutionModeXfb) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecXfbBuffer) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecXfbStride) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecOffset)) {
        printf("  [FAIL] generated interpolation geometry does not own transform-feedback output\n");
        return 1;
    }
    printf("  [ok]   generated interpolation geometry can capture its final positions\n");

    // Explicit P10/P20/P0 values are the three AMD parameters used by Astro Bot's rejected title
    // composite PS. All three fragment inputs are Flat outputs of the generated geometry stage.
    const uint32_t ps_parameters[] = {
        0xc80e0000u, 0xc8120001u, 0xc8160002u,
        0xf800000fu, 0x05050403u, 0xbf810000u,
    };
    FragmentInterpolationLayout parameter_layout = fragment_interpolation_layout(
        ps_parameters, sizeof(ps_parameters)/sizeof(ps_parameters[0]));
    std::vector<uint32_t> parameter_spv = recompile_fragment(
        ps_parameters, sizeof(ps_parameters)/sizeof(ps_parameters[0]), nullptr, nullptr,
        UINT32_MAX, &parameter_layout);
    std::vector<uint32_t> parameter_gs = recompile_interpolation_geometry(parameter_layout);
    if (!parameter_layout.valid || !parameter_layout.requires_geometry || parameter_spv.empty() ||
        parameter_gs.empty() || !has_decoration(parameter_spv, DecFlat) ||
        !has_decoration(parameter_gs, DecFlat)) {
        printf("  [FAIL] P10/P20/P0 explicit parameters did not lower through Flat geometry outputs\n");
        return 1;
    }
    printf("  [ok]   P10/P20/P0 lower through the portable Flat geometry interface\n");

    // Pixel-system VGPR initialization: with perspective sample/center enabled, disabled
    // centroid/pull-model slots reserved by ADDR, and position X/Y enabled, the packed destinations
    // are v12/v13. The fragment shell must source those values from gl_FragCoord rather than the
    // old undefined-register zero. Encodings: exp mrt0 v12,v13,0,1; s_endpgm. BuiltIn FragCoord=15.
    const uint32_t ps_position[] = {
        0x7e1c0280u, 0x7e1e02f2u, // v_mov v14,0; v_mov v15,1
        0xf800000fu, 0x0f0e0d0cu, 0xbf810000u,
    };
    PixelSystemInputMapping sys{};
    sys.ena = (1u << 0) | (1u << 1) | (1u << 8) | (1u << 9);
    sys.addr = sys.ena | (1u << 2) | (1u << 3) | (1u << 6) | (1u << 7);
    std::vector<uint32_t> position_spv = recompile_fragment(
        ps_position, sizeof(ps_position)/sizeof(ps_position[0]), nullptr, &sys);
    if (position_spv.empty() || !has_builtin(position_spv, 15)) {
        printf("  [FAIL] enabled POS_X/Y_FLOAT system VGPRs do not materialize gl_FragCoord\n");
        return 1;
    }
    printf("  [ok]   packed POS_X/Y_FLOAT system VGPRs source gl_FragCoord\n");

    printf("== PASS ==\n");
    return 0;
}

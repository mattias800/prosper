#pragma once

// A deliberately narrow SSA value evaluator for fragment lowering tests. Supplied
// subgroup lane IDs are inputs, not an implementation of fragment entry execution.
// Unsupported types, loads, opcodes, malformed values and cycles fail closed.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <map>
#include <optional>
#include <set>
#include <string>

namespace prosper::gpu::fragment_predicate_oracle {
enum class LaneInput { FragmentSubgroup, SuppliedComputeLinear };
class Values {
    struct Node {
        uint32_t opcode;
        uint32_t type;
        std::vector<uint32_t> operands;
    };
    std::map<uint32_t, Node> nodes;
    std::map<uint32_t, uint32_t> memo;
    std::set<uint32_t> visiting;
    uint32_t uint_type, bool_type, lane_pointer, supplied_lane_value = 0;
    bool valid = true;

    void collect(const std::vector<uint32_t>& words) {
        for (size_t at = 0; at < words.size();) {
            const size_t width = words[at] >> 16;
            if (!width || width > words.size() - at) {
                valid = false;
                return;
            }
            const uint32_t op = words[at] & 0xffffu;
            const bool known = op == Op_Constant || op == Op_ConstantTrue ||
                               op == Op_ConstantFalse || op == Op_Load || op == Op_BitwiseAnd ||
                               op == Op_ShiftRightLogical || op == Op_INotEqual ||
                               op == Op_UGreaterThanEqual || op == Op_Select ||
                               op == Op_LogicalNot || op == Op_LogicalAnd || op == Op_LogicalOr;
            if (known) {
                if (width < 3 || !nodes
                                      .emplace(words[at + 2], Node{op,
                                                                   words[at + 1],
                                                                   {words.begin() + at + 3,
                                                                    words.begin() + at + width}})
                                      .second) {
                    valid = false;
                    return;
                }
            }
            at += width;
        }
    }
    uint32_t type_of(uint32_t id) const {
        if (id == supplied_lane_value && supplied_lane_value) return uint_type;
        const auto at = nodes.find(id);
        return at == nodes.end() ? 0 : at->second.type;
    }
    std::optional<uint32_t> read(uint32_t id, uint32_t lane) {
        if (id == supplied_lane_value && supplied_lane_value) return lane;
        if (memo.contains(id)) return memo.at(id);
        if (!nodes.contains(id) || !visiting.insert(id).second || visiting.size() > 128) {
            error = "unsupported or cyclic SSA input";
            return {};
        }
        const auto& n = nodes.at(id);
        const auto& a = n.operands;
        std::optional<uint32_t> result;
        if (n.opcode == Op_Constant && n.type == uint_type && a.size() == 1)
            result = a[0];
        else if ((n.opcode == Op_ConstantTrue || n.opcode == Op_ConstantFalse) &&
                 n.type == bool_type && a.empty())
            result = n.opcode == Op_ConstantTrue;
        else if (n.opcode == Op_Load && n.type == uint_type && a.size() == 1 &&
                 a[0] == lane_pointer && lane_pointer)
            result = lane;
        else if (n.opcode == Op_LogicalNot && n.type == bool_type && a.size() == 1) {
            const auto x = read(a[0], lane);
            if (x && type_of(a[0]) == bool_type) result = !*x;
        } else if (a.size() == 2) {
            const auto x = read(a[0], lane), y = read(a[1], lane);
            if (x && y) {
                const bool integers = type_of(a[0]) == uint_type && type_of(a[1]) == uint_type;
                const bool booleans = type_of(a[0]) == bool_type && type_of(a[1]) == bool_type;
                if (n.opcode == Op_BitwiseAnd && n.type == uint_type && integers) result = *x & *y;
                if (n.opcode == Op_ShiftRightLogical && n.type == uint_type && integers && *y < 32)
                    result = *x >> *y;
                if (n.opcode == Op_INotEqual && n.type == bool_type && integers) result = *x != *y;
                if (n.opcode == Op_UGreaterThanEqual && n.type == bool_type && integers)
                    result = *x >= *y;
                if (n.opcode == Op_LogicalAnd && n.type == bool_type && booleans) result = *x && *y;
                if (n.opcode == Op_LogicalOr && n.type == bool_type && booleans) result = *x || *y;
            }
        } else if (n.opcode == Op_Select && a.size() == 3) {
            const auto condition = read(a[0], lane);
            const auto yes = read(a[1], lane), no = read(a[2], lane);
            if (condition && yes && no && type_of(a[0]) == bool_type && type_of(a[1]) == n.type &&
                type_of(a[2]) == n.type && (n.type == uint_type || n.type == bool_type))
                result = *condition ? *yes : *no;
        }
        visiting.erase(id);
        if (result)
            memo.emplace(id, *result);
        else if (error.empty())
            error = "unsupported typed SSA operation";
        return result;
    }

public:
    std::string error;
    explicit Values(const SpirvCompute& b, LaneInput input = LaneInput::FragmentSubgroup)
        : uint_type(b.t_u32), bool_type(b.t_bool), lane_pointer(b.v_subgroup_localid) {
        collect(b.types);
        collect(b.code);
        if (input == LaneInput::SuppliedComputeLinear) {
            // The caller supplies this exact builder-owned scalar input. This
            // does not evaluate launch topology or reinterpret arbitrary loads.
            valid = valid && b.is_compute && !b.is_fragment && b.linear_localid;
            supplied_lane_value = b.linear_localid;
        }
    }
    std::optional<uint32_t> evaluate(uint32_t id, uint32_t lane) {
        memo.clear();
        visiting.clear();
        error.clear();
        if (!valid || lane >= 64) {
            error = "malformed graph or lane outside Wave64";
            return {};
        }
        return read(id, lane);
    }
};
} // namespace prosper::gpu::fragment_predicate_oracle

#pragma once

// Evaluate only a typed constant 32-bit SSA graph feeding one vec4 fragment Output store.
// Unsupported operations, types, pointers and cycles refuse; no CFG, subgroup,
// fragment input, arithmetic or raster execution is simulated. spirv-val remains separate.
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace prosper::test {
inline std::optional<std::vector<uint32_t>>
fragment_constant_output(std::span<const uint32_t> words) {
    if (words.size() < 5 || words[0] != 0x07230203u || !words[3]) return {};
    struct Type {
        uint32_t opcode, component = 0, width = 1;
    };
    struct Node {
        uint32_t opcode, type;
        std::vector<uint32_t> operands;
    };
    std::map<uint32_t, Type> types;
    std::map<uint32_t, Node> nodes;
    std::map<uint32_t, uint32_t> pointer_types, output_pointers;
    uint32_t value = 0, output_type = 0;
    for (size_t at = 5; at < words.size();) {
        const uint32_t width = words[at] >> 16, op = words[at] & 0xffffu;
        if (!width || width > words.size() - at) return {};
        const auto in = words.subspan(at, width);
        if ((op == 20 && width == 2) || (op == 21 && width == 4 && in[2] == 32 && in[3] <= 1) ||
            (op == 22 && width == 3 && in[2] == 32)) {
            if (!types.emplace(in[1], Type{op}).second) return {};
        }
        if (op == 23 && width == 4 && types.contains(in[2]) &&
            (types.at(in[2]).opcode == 21 || types.at(in[2]).opcode == 22) && in[3] >= 2 &&
            in[3] <= 4) {
            if (!types.emplace(in[1], Type{op, in[2], in[3]}).second) return {};
        }
        if (op == 32 && width == 4 && in[2] == 3 && types.contains(in[3]))
            pointer_types[in[1]] = in[3];
        if (op == 59 && width == 4 && in[3] == 3 && pointer_types.contains(in[1]))
            output_pointers[in[2]] = pointer_types.at(in[1]);
        if (op == 62 && width == 3 && output_pointers.contains(in[1])) {
            if (value) return {};
            value = in[2];
            output_type = output_pointers.at(in[1]);
        }
        if (op == 41 || op == 42 || op == 43 || op == 80 || op == 124 || op == 169 || op == 170) {
            if (width < 3 || !types.contains(in[1]) || !in[2] || in[2] >= words[3] ||
                !nodes.emplace(in[2], Node{op, in[1], {in.begin() + 3, in.end()}}).second)
                return {};
        }
        at += width;
    }
    if (!value || !nodes.contains(value) || nodes.at(value).type != output_type ||
        types.at(output_type).opcode != 23 || types.at(output_type).width != 4 ||
        types.at(types.at(output_type).component).opcode != 22)
        return {};
    std::set<uint32_t> visiting;
    const auto read = [&](auto&& self, uint32_t id) -> std::optional<std::vector<uint32_t>> {
        const auto it = nodes.find(id);
        if (it == nodes.end() || !visiting.insert(id).second || visiting.size() > 128) return {};
        const auto& [op, type, args] = it->second;
        const auto& t = types.at(type);
        std::optional<std::vector<uint32_t>> result;
        const auto has_type = [&](uint32_t operand, uint32_t expected) {
            return nodes.contains(operand) && nodes.at(operand).type == expected;
        };
        if ((op == 41 || op == 42) && t.opcode == 20 && args.empty())
            result = std::vector<uint32_t>{op == 41};
        else if (op == 43 && (t.opcode == 21 || t.opcode == 22) && args.size() == 1)
            result = args;
        else if (op == 124 && args.size() == 1 && nodes.contains(args[0])) {
            const auto& source = types.at(nodes.at(args[0]).type);
            if (t.opcode != 20 && source.opcode != 20 && t.width == source.width)
                result = self(self, args[0]);
        } else if (op == 80 && t.opcode == 23 && args.size() == t.width) {
            std::vector<uint32_t> values;
            bool valid = true;
            for (uint32_t operand : args) {
                const auto part = self(self, operand);
                if (!part || part->size() != 1 || !has_type(operand, t.component)) {
                    valid = false;
                    break;
                }
                values.push_back(part->front());
            }
            if (valid) result = std::move(values);
        } else if (op == 170 && t.opcode == 20 && args.size() == 2 && nodes.contains(args[0])) {
            const auto operand_type = nodes.at(args[0]).type;
            const auto a = self(self, args[0]), b = self(self, args[1]);
            if (types.at(operand_type).opcode == 21 && has_type(args[1], operand_type) && a && b &&
                a->size() == 1 && b->size() == 1)
                result = std::vector<uint32_t>{*a == *b};
        } else if (op == 169 && args.size() == 3 && nodes.contains(args[0])) {
            const auto condition = self(self, args[0]);
            const auto yes = self(self, args[1]), no = self(self, args[2]);
            if (types.at(nodes.at(args[0]).type).opcode == 20 && condition &&
                condition->size() == 1 && condition->front() <= 1 && yes && no &&
                has_type(args[1], type) && has_type(args[2], type))
                result = condition->front() ? yes : no;
        }
        visiting.erase(id);
        return result && result->size() == t.width ? result : std::nullopt;
    };
    return read(read, value);
}
} // namespace prosper::test

#include "gpu/recompiler/fragment_raster_interface.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace prosper::gpu {
namespace {
FragmentRasterOutputInterface interface_words(std::span<const uint32_t> words,
                                              uint32_t execution_model, uint32_t storage) {
    if (words.size() < 5 || words[0] != 0x07230203u) return {};
    std::map<uint32_t, std::span<const uint32_t>> types, variables;
    std::map<uint32_t, uint32_t> constants, locations, builtins, output_vertex_modes;
    std::map<uint32_t, std::set<uint32_t>> builtin_members;
    std::map<uint32_t, std::optional<uint64_t>> component_memo;
    std::set<uint32_t> interfaces, visiting, position_types;
    uint32_t function = 0;
    for (size_t pc = 5; pc < words.size();) {
        const uint32_t count = words[pc] >> 16, op = words[pc] & 0xffffu;
        if (!count || count > words.size() - pc) return {};
        const auto in = words.subspan(pc, count);
        if (op == 15 && count >= 5 && in[1] == execution_model) {
            // The selected project modules have one main entry. Do not mix other entry interfaces.
            if (function || in[3] != 0x6e69616du || in[4] != 0) return {};
            function = in[2];
            for (uint32_t index = 5; index < count; ++index)
                if (!interfaces.insert(in[index]).second) return {};
        }
        if (op >= 20 && op <= 32 && count >= 2)
            if (!types.emplace(in[1], in).second) return {};
        if (op == 43 && count == 4) constants[in[2]] = in[3];
        if (op == 59 && count >= 4)
            if (!variables.emplace(in[2], in).second) return {};
        if (op == 71 && count == 4) {
            if (in[2] == 30) locations[in[1]] = in[3];
            if (in[2] == 11) builtins[in[1]] = in[3];
            if (in[2] == 31 && in[3] != 0) return {};   // selected producer uses whole locations
        }
        if (op == 72 && count == 5 && in[3] == 11) {
            builtin_members[in[1]].insert(in[2]);
            if (in[2] == 0 && in[4] == 0) position_types.insert(in[1]);
        }
        if (op == 16 && count == 4 && in[2] == 26) output_vertex_modes[in[1]] = in[3];
        pc += count;
    }
    if (!function) return {};
    // Count actual scalar components recursively, with checked arithmetic and cycle rejection.
    std::function<std::optional<uint64_t>(uint32_t)> components =
        [&](uint32_t id) -> std::optional<uint64_t> {
        if (const auto cached = component_memo.find(id); cached != component_memo.end())
            return cached->second;
        const auto found = types.find(id);
        if (found == types.end() || !visiting.insert(id).second) return {};
        const auto in = found->second;
        const uint32_t op = in[0] & 0xffffu;
        std::optional<uint64_t> result;
        if (op == 20 && in.size() == 2) result = 1;
        if ((op == 21 || op == 22) && in.size() >= 3 && (in[2] == 16 || in[2] == 32 || in[2] == 64))
            result = (in[2] + 31u) / 32u;
        if ((op == 23 || op == 24 || op == 28) && in.size() == 4) {
            const auto element = components(in[2]);
            const auto length = op == 28 ? constants.find(in[3]) : constants.end();
            const uint64_t arity =
                op == 28 ? (length == constants.end() ? 0 : length->second) : in[3];
            if (element && arity && *element <= UINT64_MAX / arity) result = *element * arity;
        }
        if (op == 30) {
            uint64_t sum = 0;
            bool complete = true;
            for (size_t index = 2; index < in.size(); ++index) {
                const auto member = components(in[index]);
                if (!member || *member > UINT64_MAX - sum) {
                    complete = false;
                    break;
                }
                sum += *member;
            }
            if (complete) result = sum;
        }
        visiting.erase(id);
        component_memo[id] = result;
        return result;
    };
    FragmentRasterOutputInterface result;
    if (output_vertex_modes.contains(function))
        result.output_vertices = output_vertex_modes.at(function);
    for (uint32_t id : interfaces) {
        const auto variable = variables.find(id);
        if (variable == variables.end()) return {};
        const auto in = variable->second;
        if (in[3] != storage) continue;
        const auto pointer = types.find(in[1]);
        if (pointer == types.end() || pointer->second.size() != 4 ||
            (pointer->second[0] & 0xffffu) != 32 || pointer->second[2] != storage)
            return {};
        const uint32_t type = pointer->second[3];
        const auto size = components(type);
        if (!size || *size > UINT64_MAX - result.components) return {};
        result.components += *size;
        if (position_types.contains(type) || (builtins.contains(id) && builtins.at(id) == 0))
            result.position = true;
        if (!locations.contains(id)) {
            if (builtins.contains(id)) continue;
            const auto structure = types.find(type);
            if (structure == types.end() || (structure->second[0] & 0xffffu) != 30 ||
                !builtin_members.contains(type) ||
                builtin_members.at(type).size() != structure->second.size() - 2)
                return {};   // no uncounted member-location interface
            continue;
        }
        const uint64_t end = uint64_t(locations.at(id)) * 4 + *size;
        result.location_end = std::max(result.location_end, end);
        const auto vector = types.find(type);
        if (vector == types.end() || vector->second.size() != 4 ||
            (vector->second[0] & 0xffffu) != 23 || vector->second[3] < 2 || vector->second[3] > 4 ||
            (storage == 3 && vector->second[3] != 4))
            return {};   // selected ABI has whole vectors, not implicit member/array packing
        const auto scalar = types.find(vector->second[2]);
        if (scalar == types.end() || scalar->second.size() != 3 ||
            (scalar->second[0] & 0xffffu) != 22 || scalar->second[2] != 32)
            return {};
        if (vector->second[3] == 4 && locations.at(id) < 32)
            result.float4_locations |= uint32_t{1} << locations.at(id);
    }
    result.available = storage == 1 || result.position;
    return result;
}
}   // namespace
FragmentRasterOutputInterface fragment_raster_output_interface(std::span<const uint32_t> words,
                                                               uint32_t execution_model) {
    return interface_words(words, execution_model, 3);
}
FragmentRasterOutputInterface fragment_raster_input_interface(std::span<const uint32_t> words,
                                                              uint32_t execution_model) {
    return interface_words(words, execution_model, 1);
}
}   // namespace prosper::gpu

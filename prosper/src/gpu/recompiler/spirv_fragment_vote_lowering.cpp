#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#include "gpu/recompiler/spirv_fragment_uniform_trace.hpp"
#include "gpu/recompiler/spirv_fragment_neutral_selection.hpp"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace prosper::gpu {
namespace {

struct Instruction {
    size_t at;
    uint32_t op;
    uint32_t count;
    bool in_function;
};

// Explicit, total core operations only. In particular, integer divide/variable shifts,
// float-to-integer conversion, arbitrary GLSL operations, derivatives and collectives do not
// inherit a proof from uniform inputs. The same domain bounds the dead-SSA graph.
std::vector<uint32_t> pure_operands(const std::vector<uint32_t>& words,
                                  const Instruction& in, bool preserve_f32) {
    const auto at = in.at;
    switch (in.op) {
        case 83: case 124: case 126: case 168: case 200: // copy, bitcast, integer/Boolean NOT
        case 204: case 205: // bit reverse/count
            if (in.count == 4) return {words[at + 3]};
            break;
        case 128: case 130: case 132: // wrapping integer arithmetic
        case 164: case 165: case 166: case 167: // Boolean comparisons/combinations
        case 170: case 171: case 172: case 173: case 174: case 175:
        case 176: case 177: case 178: case 179: // integer comparisons
        case 197: case 198: case 199: // integer bitwise operations
            if (in.count == 5) return {words[at + 3], words[at + 4]};
            break;
        case 180: case 181: case 182: case 183: case 184: case 185:
        case 186: case 187: case 188: case 189: case 190: case 191:
            // Preserve is required for arbitrary buffer NaNs, not inferred from finite fixtures.
            if (preserve_f32 && in.count == 5)
                return {words[at + 3], words[at + 4]};
            break;
        case 169: // select: condition AND both alternatives must be uniform
            if (in.count == 6)
                return {words[at + 3], words[at + 4], words[at + 5]};
            break;
        case 80: // composite construction
            if (in.count >= 4)
                return {words.begin() + at + 3, words.begin() + at + in.count};
            break;
        case 81: // literal composite indexes are not SSA uses
            if (in.count >= 5) return {words[at + 3]};
            break;
        default: break;
    }
    return {};
}

bool no_result(uint32_t op) {
    switch (op) {
        case 0: case 8: case 56: case 62: case 63: case 64: case 71: case 72:
        case 99: case 224: case 225: case 228: case 246: case 247: case 249:
        case 250: case 251: case 252: case 253: case 254: case 255: case 317:
        case 319: case 4416: case 5380:
            return true;
        default: return false;
    }
}

std::string_view instruction_string(const std::vector<uint32_t>& words,
                                    const Instruction& in, uint32_t start) {
    if (start >= in.count) return {};
    const char* data = reinterpret_cast<const char*>(words.data() + in.at + start);
    const size_t bytes = (in.count - start) * sizeof(uint32_t);
    const char* end = static_cast<const char*>(std::memchr(data, 0, bytes));
    if (!end) return {};
    // Reject hidden trailing data in a load-bearing marker/import.
    for (const char* p = end; p != data + bytes; ++p)
        if (*p != 0) return {};
    return {data, static_cast<size_t>(end - data)};
}

bool read_marker(std::string_view text, std::string_view prefix,
                 uint32_t expected, bool& seen) {
    if (!text.starts_with(prefix)) return true;
    uint32_t value = 0;
    const auto number = text.substr(prefix.size());
    const auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
    if (number.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != number.data() + number.size() || value != expected || seen)
        return false;
    seen = true;
    return true;
}

bool is_fragment_builtin(uint32_t builtin) {
    switch (builtin) {
        case 7: case 9: case 10: case 15: case 16: case 17:
        case 18: case 19: case 20: case 22: case 23:
            return true;
        default: return false; // subgroup IDs/masks and unclassified extension builtins fail closed
    }
}

void append_marker(std::vector<uint32_t>& words, std::string_view marker) {
    const size_t count = 1 + (marker.size() + 1 + 3) / 4;
    const size_t at = words.size();
    words.resize(at + count, 0);
    words[at] = (static_cast<uint32_t>(count) << 16) | 330;
    std::memcpy(words.data() + at + 1, marker.data(), marker.size());
}

// Actual function value uses only. Unknown instructions scan conservatively; a literal collision
// can cost admission but cannot grant it. Known image literals/debug metadata are excluded.
std::vector<uint32_t> value_uses(const std::vector<uint32_t>& words,
                               const Instruction& in, bool preserve_f32) {
    if (!in.in_function || in.op == 8 || in.op == 317 || in.op == 248 ||
        in.op == 249 || in.op == 246 || in.op == 247 || in.op == 253 || in.op == 56)
        return {};
    auto pure = pure_operands(words, in, preserve_f32);
    if (!pure.empty()) return pure;
    if (in.op == 335 && in.count == 5) return {words[in.at + 4]};
    if (in.op == 250 && in.count >= 4) return {words[in.at + 1]};
    if (in.op == 245 && in.count >= 5) {
        std::vector<uint32_t> values;
        for (uint32_t i = 3; i + 1 < in.count; i += 2) values.push_back(words[in.at + i]);
        return values;
    }
    const uint32_t start = no_result(in.op) ? 1 : 3;
    std::vector<uint32_t> values;
    for (uint32_t i = start; i < in.count; ++i) {
        if ((in.op == 12 && i == 4) ||
            (in.op == 99 && i == 4) ||
            ((in.op == 87 || in.op == 88 || in.op == 91 || in.op == 92 ||
              in.op == 95 || in.op == 98) && i == 5) ||
            ((in.op == 89 || in.op == 90 || in.op == 93 || in.op == 94 ||
              in.op == 96 || in.op == 97) && i == 6))
            continue;
        values.push_back(words[in.at + i]);
    }
    return values;
}

std::unordered_set<uint32_t> neutral_selection_votes(
    const std::vector<uint32_t>& words, const std::vector<Instruction>& instructions,
    uint32_t entry_id, const std::unordered_set<uint32_t>& bool_types,
    const std::unordered_set<uint32_t>& int_types, const std::unordered_set<uint32_t>& float_types,
    const std::unordered_set<uint32_t>& glsl_sets, bool preserve_f32,
    const std::unordered_set<uint32_t>& explicit_transport,
    const std::unordered_set<uint32_t>& frozen_leaves,
    const std::unordered_map<uint32_t, std::vector<size_t>>& users) {
    FragmentNeutralSelection graph;
    graph.preserve_f32 = preserve_f32;
    std::unordered_map<uint32_t, std::vector<size_t>> blocks;
    std::unordered_map<uint32_t, std::vector<uint32_t>> predecessors;
    std::unordered_map<size_t, uint32_t> instruction_blocks;
    std::unordered_set<uint32_t> stable_builtin_pointers;
    for (const auto& in : instructions)
        if (in.op == 71 && in.count == 4 && words[in.at + 2] == 11 &&
            (words[in.at + 3] == 15 || words[in.at + 3] == 23))
            stable_builtin_pointers.insert(words[in.at + 1]); // FragCoord/HelperInvocation
    uint32_t function = 0, block = 0, entry_block = 0;
    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& in = instructions[i];
        const auto at = in.at;
        if ((in.op == 16 || in.op == 331) && in.count >= 3 && words[at + 2] == 6028)
            return {}; // FPFastMathDefault overrides the implicit finite-domain contract
        if (in.op == 54) { function = words[at + 2]; block = 0; }
        if (function == entry_id && in.op == 248) {
            block = words[at + 1];
            if (!entry_block) entry_block = block;
        }
        if (function == entry_id && block) {
            blocks[block].push_back(i);
            instruction_blocks.emplace(i, block);
            if (in.op == 249) predecessors[words[at + 1]].push_back(block);
            if (in.op == 250) {
                predecessors[words[at + 2]].push_back(block);
                predecessors[words[at + 3]].push_back(block);
            }
            if (in.op == 251) {
                predecessors[words[at + 2]].push_back(block);
                for (uint32_t n = 4; n < in.count; n += 2)
                    predecessors[words[at + n]].push_back(block);
            }
        }
        if (in.op == 56) { function = block = 0; }
        // Ordinary definitions and constants only; type IDs, labels and metadata are not values.
        if (in.count < 3 || no_result(in.op) || in.op == 248 ||
            !(in.in_function || in.op == 1 || (in.op >= 41 && in.op <= 52))) continue;
        FragmentNeutralValue value;
        value.opcode = in.op;
        value.explicit_transport = explicit_transport.contains(words[at + 2]);
        const uint32_t type = words[at + 1];
        value.type = bool_types.contains(type) ? FragmentNeutralType::Boolean :
            int_types.contains(type) ? FragmentNeutralType::Int32 :
            float_types.contains(type) ? FragmentNeutralType::Float32 : FragmentNeutralType::Other;
        value.operands = pure_operands(words, in, true);
        value.stable_input = in.op == 61 && in.count == 4 &&
            stable_builtin_pointers.contains(words[at + 3]);
        if ((in.op == 127 && in.count == 4) ||
            ((in.op == 129 || in.op == 131 || in.op == 133) && in.count == 5))
            value.operands.assign(words.begin() + at + 3, words.begin() + at + in.count);
        if (in.op == 12 && in.count == 6 && glsl_sets.contains(words[at + 3])) {
            value.extended_opcode = words[at + 4];
            value.operands = {words[at + 5]};
        }
        if (((in.op == 41 || in.op == 42) && in.count == 3 && bool_types.contains(type)) ||
            (in.op == 43 && in.count == 4 && (int_types.contains(type) || float_types.contains(type)))) {
            value.constant = true;
            value.literal = in.op == 41 ? 1 : in.op == 42 ? 0 : words[at + 3];
        }
        graph.values.emplace(words[at + 2], std::move(value));
    }
    std::unordered_set<uint32_t> proved;
    if (!entry_block || !predecessors[entry_block].empty()) return proved;
    const auto& header = blocks.at(entry_block);
    if (header.size() < 3) return proved;
    const auto& branch = instructions[header.back()];
    const auto& selection = instructions[header[header.size() - 2]];
    if (branch.op != 250 || branch.count != 4 || selection.op != 247 || selection.count != 3)
        return proved;
    const uint32_t controller = words[branch.at + 1], body = words[branch.at + 2],
                   merge = words[branch.at + 3];
    if (merge != words[selection.at + 1] || body == merge || !blocks.contains(body) ||
        !blocks.contains(merge) || predecessors[body] != std::vector<uint32_t>{entry_block} ||
        predecessors[merge].size() != 2) return proved;
    const auto vote = std::find_if(header.begin(), header.end(), [&](size_t i) {
        return instructions[i].op == 335 && words[instructions[i].at + 2] == controller;
    });
    if (vote == header.end() || !users.contains(controller) ||
        users.at(controller) != std::vector<size_t>{header.back()}) return proved;
    graph.predicate = words[instructions[*vote].at + 4];
    if (frozen_leaves.contains(graph.predicate)) return proved; // retain the older Copy(P) certificate
    const auto& body_instructions = blocks.at(body);
    const auto& exit = instructions[body_instructions.back()];
    if (exit.op != 249 || exit.count != 2 || words[exit.at + 1] != merge) return proved;
    for (size_t i : body_instructions) {
        const auto& in = instructions[i];
        if (in.op == 248 || in.op == 249 || in.op == 8 || in.op == 317) continue;
        if (in.count < 3 || no_result(in.op)) return proved;
        graph.body.push_back(words[in.at + 2]);
        // Every cross-region use must be a boundary Phi incoming from this body. Even an unused
        // outside address/selector is not silently treated as a color-only export.
        const auto found = users.find(words[in.at + 2]);
        if (found == users.end()) continue;
        for (size_t use : found->second) {
            if (instruction_blocks.contains(use) && instruction_blocks.at(use) == body) continue;
            const auto& consumer = instructions[use];
            if (!instruction_blocks.contains(use) || instruction_blocks.at(use) != merge ||
                consumer.op != 245) return proved;
            for (uint32_t n = 3; n + 1 < consumer.count; n += 2)
                if (words[consumer.at + n] == words[in.at + 2] && words[consumer.at + n + 1] != body)
                    return proved;
        }
    }
    for (size_t i : blocks.at(merge)) {
        const auto& phi = instructions[i];
        if (phi.op != 245) continue;
        if (phi.count != 7) return proved;
        const uint32_t id = words[phi.at + 2];
        if (!users.contains(id) || users.at(id).empty()) continue;
        FragmentNeutralExport output;
        for (uint32_t n : {3u, 5u}) {
            if (words[phi.at + n + 1] == entry_block) output.skipped = words[phi.at + n];
            else if (words[phi.at + n + 1] == body) output.executed = words[phi.at + n];
            else return proved;
        }
        if (!output.skipped || !output.executed) return proved;
        graph.exports.push_back(output);
    }
    if (prove_fragment_neutral_selection(graph, frozen_leaves)) proved.insert(controller);
    return proved;
}

} // namespace

FragmentVoteLowering lower_fragment_votes(const std::vector<uint32_t>& source,
                                         bool immutable_storage_inputs,
                                         bool deterministic_storage_reads) {
    FragmentVoteLowering result;
    if (source.size() < 5 || source[0] != 0x07230203u || !source[3] || source[4] ||
        source[1] < 0x00010300u || source[1] > 0x00010600u || (source[1] & 0xffu))
        return result;
    std::vector<Instruction> instructions;
    bool in_function = false;
    for (size_t at = 5; at < source.size();) {
        const uint32_t count = source[at] >> 16, op = source[at] & 0xffffu;
        if (!count || count > source.size() - at) return result;
        if (op == 54) in_function = true;
        instructions.push_back({at, op, count, in_function});
        if (op == 56) in_function = false;
        at += count;
    }
    if (in_function) return result;

    bool size_seen = false, why_seen = false, entry_seen = false, preserve_f32 = false;
    uint32_t entry_id = 0, preserve_entry = 0;
    uint32_t denorm_entry = 0;
    bool denorm_defined = false;
    bool writes_memory = false;
    std::unordered_set<uint32_t> bool_types, int32_types, float32_types, glsl_sets;
    std::unordered_set<uint32_t> float32_vectors, explicit_transport, no_contraction;
    bool transport_capability = false, transport_extension = false;
    std::unordered_map<uint32_t, uint32_t> value_types, pointer_classes;
    std::unordered_map<uint32_t, uint32_t> pointer_elements, runtime_elements, struct_members, strides, constants;
    std::unordered_set<uint32_t> offset_zero_structs;
    std::vector<Instruction> votes;
    for (const auto& in : instructions) {
        const auto at = in.at;
        if (in.op == 15) {
            if (in.count < 4 || source[at + 1] != 4 || entry_seen) return result;
            entry_seen = true;
            entry_id = source[at + 2];
        }
        if (in.op == 20 && in.count == 2) bool_types.insert(source[at + 1]);
        if (in.op == 21 && in.count == 4 && source[at + 2] == 32)
            int32_types.insert(source[at + 1]);
        if (in.op == 22 && in.count == 3 && source[at + 2] == 32)
            float32_types.insert(source[at + 1]);
        if (in.op == 23 && in.count == 4 && float32_types.contains(source[at + 2]) &&
            source[at + 3] >= 2 && source[at + 3] <= 4)
            float32_vectors.insert(source[at + 1]);
        if (in.op == 17 && in.count == 2 && source[at + 1] == 6029)
            transport_capability = true;
        if (in.op == 10 && instruction_string(source, in, 1) == "SPV_KHR_float_controls2")
            transport_extension = true;
        if ((in.op == 16 || in.op == 331) && in.count >= 3 && source[at + 2] == 6028) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result; // this contract supports only exact per-instruction transport None
        }
        if (in.op == 71 && in.count >= 3 && source[at + 2] == 40) {
            if (in.count != 4 || source[at + 3] != 0 ||
                !explicit_transport.insert(source[at + 1]).second) {
                result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
                return result;
            }
        }
        if (in.op == 71 && in.count == 3 && source[at + 2] == 42)
            no_contraction.insert(source[at + 1]);
        if ((in.in_function && !no_result(in.op) && in.op != 248 && in.count >= 3) ||
            ((in.op == 1 || (in.op >= 41 && in.op <= 52)) && in.count >= 3))
            value_types.emplace(source[at + 2], source[at + 1]);
        if (in.op == 59 && in.count >= 4) pointer_classes.emplace(source[at + 2], source[at + 3]);
        if (in.op == 32 && in.count == 4 && source[at + 2] == 12)
            pointer_elements.emplace(source[at + 1], source[at + 3]);
        if (in.op == 29 && in.count == 3) runtime_elements.emplace(source[at + 1], source[at + 2]);
        if (in.op == 30 && in.count == 3) struct_members.emplace(source[at + 1], source[at + 2]);
        if (in.op == 71 && in.count == 4 && source[at + 2] == 6)
            strides.emplace(source[at + 1], source[at + 3]);
        if (in.op == 72 && in.count == 5 && source[at + 2] == 0 &&
            source[at + 3] == 35 && source[at + 4] == 0)
            offset_zero_structs.insert(source[at + 1]);
        if (in.op == 43 && in.count == 4 && int32_types.contains(source[at + 1]))
            constants.emplace(source[at + 2], source[at + 3]);
        if (in.op == 16 && in.count == 4 && source[at + 2] == 4461 && source[at + 3] == 32) {
            if (preserve_f32) return result;
            preserve_f32 = true;
            preserve_entry = source[at + 1];
        }
        if (in.op == 16 && in.count == 4 && source[at + 2] == 4459 &&
            source[at + 3] == 32) {
            if (denorm_defined) return result;
            denorm_defined = true;
            denorm_entry = source[at + 1];
        }
        if (in.op == 330) {
            const auto text = instruction_string(source, in, 1);
            if (text.empty() || !read_marker(text, "Prosper.FragmentSubgroupSize=", 64, size_seen) ||
                !read_marker(text, "Prosper.FragmentSubgroupWhy=", 2, why_seen)) {
                result.refusal = FragmentVoteRefusal::InconsistentContract;
                return result;
            }
        }
        if (in.op == 11 && instruction_string(source, in, 2) != "GLSL.std.450") {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if (in.op == 11 && in.count >= 3) glsl_sets.insert(source[at + 1]);
        if ((in.op >= 333 && in.op <= 366 && in.op != 335) ||
            (in.op >= 259 && in.op <= 271) || in.op == 73 || in.op == 74 || in.op == 75 ||
            (in.op > 400 && in.op != 4416 && in.op != 5380)) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if ((in.op == 71 && in.count >= 4 && source[at + 2] == 11 &&
             !is_fragment_builtin(source[at + 3])) ||
            (in.op == 72 && in.count >= 5 && source[at + 3] == 11 &&
             !is_fragment_builtin(source[at + 4])) ||
            (in.op == 71 && in.count >= 3 && (source[at + 2] == 0 ||
             source[at + 2] == 4469 || source[at + 2] == 4470)) ||
            (in.op == 72 && in.count >= 4 && (source[at + 3] == 0 || source[at + 3] == 40 ||
             source[at + 3] == 4469 || source[at + 3] == 4470))) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if (in.op == 335) votes.push_back(in);
    }
    if (!entry_seen || !size_seen || !why_seen || votes.empty()) {
        result.refusal = FragmentVoteRefusal::InconsistentContract;
        return result;
    }
    if (preserve_f32 && preserve_entry != entry_id) return result;
    if (denorm_defined && denorm_entry != entry_id) return result;
    for (const auto& vote : votes) {
        if (vote.count != 5 || !bool_types.contains(source[vote.at + 1]) ||
            !value_types.contains(source[vote.at + 4]) ||
            value_types.at(source[vote.at + 4]) != source[vote.at + 1])
            return result;
        bool subgroup_scope = false;
        for (const auto& in : instructions)
            if (in.op == 43 && in.count == 4 && int32_types.contains(source[in.at + 1]) &&
                source[in.at + 2] == source[vote.at + 3] &&
                source[in.at + 3] == 3)
                subgroup_scope = true;
        if (!subgroup_scope) return result;
    }

    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& in : instructions) {
            if ((in.op != 65 && in.op != 66 && in.op != 67 && in.op != 70 && in.op != 83) ||
                in.count < 4 || !pointer_classes.contains(source[in.at + 3]))
                continue;
            if (pointer_classes.emplace(source[in.at + 2],
                                        pointer_classes.at(source[in.at + 3])).second)
                changed = true;
        }
    }
    for (const auto& in : instructions) {
        // No residual subgroup-scoped synchronization may survive erasing the width contract.
        // Device/workgroup/invocation memory scopes do not acquire subgroup semantics from Any.
        uint32_t scope_id = 0;
        if (in.op == 224) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if (in.op == 225 && in.count >= 3) scope_id = source[in.at + 1];
        if (((in.op >= 227 && in.op <= 242) || in.op == 318) && in.count >= 6)
            scope_id = source[in.at + 4];
        if ((in.op == 228 || in.op == 319) && in.count >= 4) scope_id = source[in.at + 2];
        if (scope_id && (!constants.contains(scope_id) || constants.at(scope_id) == 3)) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if (in.op == 12 && (in.count < 5 || !glsl_sets.contains(source[in.at + 3]))) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        if ((in.op == 62 || in.op == 63 || in.op == 64) && in.count >= 2) {
            const auto storage = pointer_classes.find(source[in.at + 1]);
            // Function/Output writes cannot change immutable input-buffer contents.
            writes_memory |= storage == pointer_classes.end() ||
                (storage->second != 7 && storage->second != 3);
        } else if (in.op == 12 && in.count >= 7 &&
                   (source[in.at + 4] == 35 || source[in.at + 4] == 51)) {
            // GLSL Modf/Frexp have an output pointer as well as their returned value.
            const auto storage = pointer_classes.find(source[in.at + 6]);
            writes_memory |= storage == pointer_classes.end() ||
                (storage->second != 7 && storage->second != 3);
        } else if (in.op == 57 || in.op == 99 || (in.op >= 227 && in.op <= 242) ||
                   in.op == 318 || in.op == 319) {
            writes_memory = true;
        }
    }
    // Allow only the producing transport slice, not general explicit FP environments.
    // Strict module validity alone does not prove the stronger neutral theorem's premises.
    if (!explicit_transport.empty() &&
        (!transport_capability || !transport_extension || !preserve_f32)) {
        result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
        return result;
    }
    std::unordered_set<uint32_t> validated_transport;
    for (const auto& in : instructions) {
        if (!in.in_function || in.count < 3 || !explicit_transport.contains(source[in.at + 2]))
            continue;
        const size_t at = in.at;
        const uint32_t id = source[at + 2], type = source[at + 1];
        const bool cross_float_bitcast = in.op == 124 && in.count == 4 &&
            value_types.contains(source[at + 3]) &&
            ((float32_types.contains(type) && int32_types.contains(value_types.at(source[at + 3]))) ||
             (int32_types.contains(type) && float32_types.contains(value_types.at(source[at + 3]))));
        const bool fragment_input_load = in.op == 61 && in.count == 4 &&
            (float32_types.contains(type) || float32_vectors.contains(type)) &&
            pointer_classes.contains(source[at + 3]) && pointer_classes.at(source[at + 3]) == 1;
        if ((!cross_float_bitcast && !fragment_input_load) || no_contraction.contains(id)) {
            result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
            return result;
        }
        validated_transport.insert(id);
    }
    if (validated_transport != explicit_transport) {
        result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
        return result;
    }
    // Any is also an execution rendezvous. A dead result does not prove it may be erased around
    // externally observable memory effects, atomics or an opaque callee. Do not confuse the
    // absence of subgroup-scoped barriers with the absence of that ordering obligation.
    if (writes_memory) {
        result.refusal = FragmentVoteRefusal::UnsupportedWaveOperation;
        return result;
    }
    const auto preserves_operands = [&](const Instruction& in) {
        if (in.op < 180 || in.op > 191) return false;
        return preserve_f32 && denorm_defined && in.count == 5 &&
            value_types.contains(source[in.at + 3]) && value_types.contains(source[in.at + 4]) &&
            float32_types.contains(value_types.at(source[in.at + 3])) &&
            float32_types.contains(value_types.at(source[in.at + 4]));
    };

    std::unordered_set<uint32_t> uniform, uniform_pointers, bounded_word_indexes;
    std::unordered_map<uint32_t, uint32_t> word_buffer_roots;
    for (const auto& in : instructions) {
        if (((in.op == 41 || in.op == 42 || in.op == 46) && in.count == 3) ||
            (in.op == 43 && in.count >= 4))
            uniform.insert(source[in.at + 2]);
        if (in.op == 43 && in.count == 4 && int32_types.contains(source[in.at + 1]) &&
            source[in.at + 3] <= 0x3fffffffu)
            bounded_word_indexes.insert(source[in.at + 2]);
        if (in.op == 59 && in.count == 4 && source[in.at + 3] == 12 &&
            immutable_storage_inputs && deterministic_storage_reads && !writes_memory) {
            const auto pointer = pointer_elements.find(source[in.at + 1]);
            if (pointer == pointer_elements.end() || !offset_zero_structs.contains(pointer->second)) continue;
            const auto member = struct_members.find(pointer->second);
            if (member == struct_members.end() || !strides.contains(member->second) ||
                strides.at(member->second) != 4 || !runtime_elements.contains(member->second)) continue;
            const auto element = runtime_elements.at(member->second);
            if (int32_types.contains(element) || (float32_types.contains(element) && preserve_f32))
                word_buffer_roots.emplace(source[in.at + 2], element);
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& in : instructions) {
            const auto at = in.at;
            if (in.count < 3 || no_result(in.op)) continue;
            const uint32_t id = source[at + 2];
            // Robust2 defines descriptor bounds, not wrapping address calculations. The guest
            // word layout has zero member offset and stride4: prove 0 <= index <= 2^30-1.
            // Masks admit dynamic uniform indexes too, without treating arbitrary SGPRs as bounds.
            if (value_types.contains(id) && int32_types.contains(value_types.at(id))) {
                bool bounded = false;
                if ((in.op == 83 || in.op == 124) && in.count == 4)
                    bounded = bounded_word_indexes.contains(source[at + 3]);
                if (in.op == 199 && in.count == 5)
                    bounded = bounded_word_indexes.contains(source[at + 3]) ||
                              bounded_word_indexes.contains(source[at + 4]);
                if (in.op == 169 && in.count == 6)
                    bounded = bounded_word_indexes.contains(source[at + 4]) &&
                              bounded_word_indexes.contains(source[at + 5]);
                if (bounded && bounded_word_indexes.insert(id).second) changed = true;
            }
            if (uniform.contains(id)) continue;
            std::vector<uint32_t> operands;
            if (in.op == 65 && in.count == 6) {
                if (!word_buffer_roots.contains(source[at + 3]) ||
                    !pointer_elements.contains(source[at + 1]) ||
                    pointer_elements.at(source[at + 1]) != word_buffer_roots.at(source[at + 3]) ||
                    !constants.contains(source[at + 4]) || constants.at(source[at + 4]) != 0 ||
                    !uniform.contains(source[at + 5]) || !value_types.contains(source[at + 5]) ||
                    !int32_types.contains(value_types.at(source[at + 5])) ||
                    !bounded_word_indexes.contains(source[at + 5])) continue;
                if (uniform_pointers.insert(id).second) changed = true;
                continue;
            }
            if (in.op == 61 && in.count == 4 && uniform_pointers.contains(source[at + 3])) {
                if (uniform.insert(id).second) changed = true;
                continue;
            }
            if (in.op == 44 && in.count >= 4)
                operands.assign(source.begin() + at + 3, source.begin() + at + in.count);
            else if (in.op == 335 && in.count == 5) operands = {source[at + 4]};
            else if (in.op == 245 && in.count >= 5 && (in.count % 2) == 1) {
                bool identical = true;
                for (uint32_t i = 5; i + 1 < in.count; i += 2)
                    identical &= source[at + i] == source[at + 3];
                if (identical) operands = {source[at + 3]};
            } else operands = pure_operands(source, in, preserves_operands(in));
            // Even preserved float operations may quiet a signaling NaN. Float value equality
            // is sufficient for ordered/unordered comparisons, not for a raw-bit predicate.
            // Only scalar integer-to-integer bitcasts carry the uniform certificate. Arbitrary
            // integer bits reinterpreted as floats do not supply a finite-value domain proof.
            if (in.op == 124 && (in.count != 4 || !value_types.contains(source[at + 3]) ||
                                !int32_types.contains(value_types.at(source[at + 3])) ||
                                !int32_types.contains(source[at + 1])))
                operands.clear();
            if (!operands.empty() && std::all_of(operands.begin(), operands.end(),
                                               [&](uint32_t op) { return uniform.contains(op); }))
                if (uniform.insert(id).second) changed = true;
        }
    }

    std::unordered_map<uint32_t, std::vector<size_t>> users;
    for (size_t i = 0; i < instructions.size(); ++i)
        for (uint32_t value : value_uses(source, instructions[i], preserves_operands(instructions[i])))
            users[value].push_back(i);
    std::unordered_set<uint32_t> dead_vote_ids;
    for (const auto& vote : votes) {
        if (uniform.contains(source[vote.at + 4])) continue;
        std::vector<uint32_t> pending{source[vote.at + 2]};
        std::unordered_set<uint32_t> seen;
        bool dead = true;
        while (!pending.empty() && dead) {
            const uint32_t value = pending.back();
            pending.pop_back();
            if (!seen.insert(value).second) continue;
            for (size_t user : users[value]) {
                const auto& in = instructions[user];
                if (pure_operands(source, in, preserves_operands(in)).empty()) { dead = false; break; }
                pending.push_back(source[in.at + 2]);
            }
        }
        if (dead) dead_vote_ids.insert(source[vote.at + 2]);
    }

    const auto neutral_vote_ids = neutral_selection_votes(
        source, instructions, entry_id, bool_types, int32_types, float32_types, glsl_sets,
        preserve_f32, explicit_transport, uniform, users);
    // These are facts about the transaction's EFFECTIVE controller, never about source P. The
    // frozen source leaves above still own all load/address authority.
    auto effective_leaves = uniform;
    effective_leaves.insert(neutral_vote_ids.begin(), neutral_vote_ids.end());

    // Recurrences require a separate simultaneous trace certificate. Never feed its provisional
    // facts into pointer/load admission: uniform induction is NOT a non-wrapping address proof.
    FragmentUniformTrace trace;
    uint32_t function = 0, block = 0;
    bool trace_supported = true;
    for (const auto& in : instructions) {
        const auto at = in.at;
        if (in.op == 54) { function = source[at + 2]; block = 0; }
        if (function != entry_id) continue;
        if (in.op == 248) {
            block = source[at + 1];
            trace.successors.emplace(block, std::vector<uint32_t>{});
            if (!trace.entry_block) trace.entry_block = block;
        }
        if (in.op == 56) { function = block = 0; continue; }
        if (!block) continue;
        auto& successors = trace.successors.at(block);
        if (in.op == 249) successors.push_back(source[at + 1]);
        if (in.op == 250) {
            trace.selectors.push_back(source[at + 1]);
            successors.insert(successors.end(), {source[at + 2], source[at + 3]});
        }
        if (in.op == 251) {
            trace.selectors.push_back(source[at + 1]);
            if (!value_types.contains(source[at + 1]) ||
                !int32_types.contains(value_types.at(source[at + 1]))) { trace_supported = false; continue; }
            successors.push_back(source[at + 2]);
            for (uint32_t i = 4; i < in.count; i += 2) successors.push_back(source[at + i]);
        }
        if (in.count < 3 || no_result(in.op) || in.op == 248 ||
            (!bool_types.contains(source[at + 1]) && !int32_types.contains(source[at + 1]))) continue;
        FragmentTraceValue value;
        value.block = block;
        if (in.op == 245 && in.count >= 5 && (in.count % 2) == 1) {
            for (uint32_t i = 3; i + 1 < in.count; i += 2) {
                value.operands.push_back(source[at + i]);
                value.phi_predecessors.push_back(source[at + i + 1]);
            }
        } else if (in.op == 335 && in.count == 5) value.operands = {source[at + 4]};
        else value.operands = pure_operands(source, in, preserves_operands(in));
        if (in.op == 124 && (in.count != 4 || !value_types.contains(source[at + 3]) ||
                            !int32_types.contains(value_types.at(source[at + 3])) ||
                            !int32_types.contains(source[at + 1]))) value.operands.clear();
        if (!value.operands.empty()) trace.values.emplace(source[at + 2], std::move(value));
    }
    if (trace_supported) {
        // A failed fallback must not revoke the older constant/dead-SSA certificates, even if
        // unrelated control flow is varying. Only predicates missing those facts need this proof.
        std::vector<uint32_t> predicates;
        for (const auto& vote : votes)
            if (!uniform.contains(source[vote.at + 4]) && !dead_vote_ids.contains(source[vote.at + 2]) &&
                !neutral_vote_ids.contains(source[vote.at + 2]))
                predicates.push_back(source[vote.at + 4]);
        if (!predicates.empty()) {
            const auto proved = prove_fragment_uniform_trace(trace, effective_leaves, predicates);
            uniform.insert(proved.begin(), proved.end());
        }
    }

    for (const auto& vote : votes) {
        if (neutral_vote_ids.contains(source[vote.at + 2])) {
            ++result.neutral_votes;
            continue;
        }
        if (uniform.contains(source[vote.at + 4])) {
            ++result.uniform_votes;
            continue;
        }
        if (!dead_vote_ids.contains(source[vote.at + 2])) {
            result.refusal = FragmentVoteRefusal::UnprovedVote;
            result.uniform_votes = result.dead_votes = result.neutral_votes = 0;
            result.failed_vote = {true, vote.at, source[vote.at + 2], source[vote.at + 4]};
            // Failure-only lookup: retain the actual typed definition, never a literal/label ID
            // collision. This adds no facts to the proof and never rewrites SOURCE or EFFECTIVE.
            for (const auto& in : instructions) {
                if (((in.in_function && !no_result(in.op) && in.op != 248 && in.count >= 3) ||
                     ((in.op == 1 || (in.op >= 41 && in.op <= 52)) && in.count >= 3)) &&
                    source[in.at + 2] == result.failed_vote.predicate_id) {
                    result.failed_vote.predicate_opcode = in.op;
                    break;
                }
            }
            return result;
        }
        ++result.dead_votes;
    }

    result.words.assign(source.begin(), source.begin() + 5);
    // Create one ordinary TRUE per Boolean type. ID bound belongs to the owned effective module;
    // no source constant/result is mutated and no shader identity/title participates in admission.
    std::unordered_map<uint32_t, uint32_t> true_ids;
    for (const auto& vote : votes)
        if (neutral_vote_ids.contains(source[vote.at + 2])) true_ids.emplace(source[vote.at + 1], 0);
    for (auto& [type, id] : true_ids) id = result.words[3]++;
    bool constants_inserted = false;
    for (const auto& in : instructions) {
        if (in.op == 54 && !constants_inserted) {
            for (const auto& [type, id] : true_ids)
                result.words.insert(result.words.end(), {(3u << 16) | 41u, type, id});
            constants_inserted = true;
        }
        if (in.op == 335) {
            result.words.insert(result.words.end(), {(4u << 16) | 83u, source[in.at + 1],
                source[in.at + 2], neutral_vote_ids.contains(source[in.at + 2]) ?
                    true_ids.at(source[in.at + 1]) : source[in.at + 4]});
        } else if (in.op == 17 && in.count == 2 && source[in.at + 1] >= 61 && source[in.at + 1] <= 68) {
            // Inventory above excludes all other subgroup instructions/builtins/imports.
            continue;
        } else if (in.op == 330 && instruction_string(source, in, 1).starts_with("Prosper.FragmentSubgroupSize=")) {
            append_marker(result.words, "Prosper.FragmentSubgroupSize=0");
        } else if (in.op == 330 && instruction_string(source, in, 1).starts_with("Prosper.FragmentSubgroupWhy=")) {
            append_marker(result.words, "Prosper.FragmentSubgroupWhy=0");
        } else {
            result.words.insert(result.words.end(), source.begin() + in.at,
                                source.begin() + in.at + in.count);
        }
    }
    result.refusal = FragmentVoteRefusal::None;
    return result;
}

} // namespace prosper::gpu

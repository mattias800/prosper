#include "gpu/recompiler/spirv_fragment_neutral_selection.hpp"
#include <algorithm>

namespace prosper::gpu {
namespace {
using Type = FragmentNeutralType;

// This domain promises no execution-side UB, not a defined numerical result. Vulkan precision
// rules can produce poison for mathematically undefined FP results. Do not promote these math
// results to nonpoison just because their operands are nonpoison. No division, variable shifts,
// conversions, loads, derivatives, collectives, pointer intrinsics or opaque callees here.
bool executable(const FragmentNeutralSelection& graph, const FragmentNeutralValue& value) {
    const auto all = [&](Type type, size_t count) {
        return value.operands.size() == count &&
            std::all_of(value.operands.begin(), value.operands.end(), [&](uint32_t id) {
                const auto found = graph.values.find(id);
                return found != graph.values.end() && found->second.type == type;
            });
    };
    switch (value.opcode) {
        case 1: return value.type != Type::Other && value.operands.empty(); // Undef, not poison
        case 83: return all(value.type, 1) && value.type != Type::Other;
        case 124:
            if (value.operands.size() != 1 || !graph.values.contains(value.operands[0])) return false;
            return (value.type == Type::Int32 || value.type == Type::Float32) &&
                (graph.values.at(value.operands[0]).type == Type::Int32 ||
                 graph.values.at(value.operands[0]).type == Type::Float32);
        case 126: case 200: case 204: case 205:
            return value.type == Type::Int32 && all(Type::Int32, 1);
        case 128: case 130: case 132: case 197: case 198: case 199:
            return value.type == Type::Int32 && all(Type::Int32, 2);
        case 164: case 165: case 166: case 167:
            return value.type == Type::Boolean && all(Type::Boolean, 2);
        case 168: return value.type == Type::Boolean && all(Type::Boolean, 1);
        case 169:
            return value.type != Type::Other && value.operands.size() == 3 &&
                graph.values.contains(value.operands[0]) &&
                graph.values.at(value.operands[0]).type == Type::Boolean &&
                graph.values.contains(value.operands[1]) && graph.values.contains(value.operands[2]) &&
                graph.values.at(value.operands[1]).type == value.type &&
                graph.values.at(value.operands[2]).type == value.type;
        case 170: case 171: case 172: case 173: case 174: case 175:
        case 176: case 177: case 178: case 179:
            return value.type == Type::Boolean && all(Type::Int32, 2);
        case 180: case 181: case 182: case 183: case 184: case 185:
        case 186: case 187: case 188: case 189: case 190: case 191:
            return graph.preserve_f32 && value.type == Type::Boolean && all(Type::Float32, 2);
        case 127: return graph.preserve_f32 && value.type == Type::Float32 && all(Type::Float32, 1);
        case 129: case 131: case 133:
            return graph.preserve_f32 && value.type == Type::Float32 && all(Type::Float32, 2);
        case 12:
            // GLSL.std.450 FAbs/Floor/Fract/Cos. No special-value numerical identity is assumed;
            // SignedZeroInfNanPreserve does NOT govern GLSL extended instructions.
            return value.type == Type::Float32 && all(Type::Float32, 1) &&
                (value.extended_opcode == 4 || value.extended_opcode == 8 ||
                 value.extended_opcode == 10 || value.extended_opcode == 14);
        default: return false;
    }
}

struct Abstract {
    bool nonpoison = false;
    bool constant = false;
    uint32_t literal = 0;
    uint32_t atom = 0; // nonzero only for an independently stable Boolean/Int32 value
    bool finite_bits = false; // IEEE32 exponent is not all ones, including raw Int32 carriers
    bool operator==(const Abstract&) const = default;
};
} // namespace

bool prove_fragment_neutral_selection(const FragmentNeutralSelection& graph,
                                     const std::unordered_set<uint32_t>& frozen_leaves) {
    if (!graph.predicate || !graph.values.contains(graph.predicate) ||
        graph.values.at(graph.predicate).type != Type::Boolean) return false;
    std::unordered_set<uint32_t> nonpoison = frozen_leaves;
    std::unordered_set<uint32_t> finite_bits;
    // Reusing a domain fact requires stable values, not just a favorable earlier consumption of
    // Undef. This closure has no backwards inference and no arbitrary Input-load authority.
    std::unordered_set<uint32_t> stable = frozen_leaves;
    for (const auto& [id, value] : graph.values)
        if (value.constant || value.stable_input) stable.insert(id);
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto& [id, value] : graph.values) {
            const bool extract = value.opcode == 81 && value.operands.size() == 1;
            if (stable.contains(id) || (!extract && !executable(graph, value)) ||
                value.operands.empty()) continue;
            if (std::all_of(value.operands.begin(), value.operands.end(),
                    [&](uint32_t input) { return stable.contains(input); }))
                if (stable.insert(id).second) changed = true;
        }
    }
    // In the newly executed case the complete guest-wave Any was stable FALSE. Undefined or
    // poison branch conditions are UB; a FALSE OR cannot mask a Boolean still capable of TRUE.
    // Thus P itself is stable FALSE, without stabilizing its undefined dependencies. This is NOT
    // an arbitrary host Any argument: Vulkan may omit helpers from non-quad group operations.
    // Its scalar predicate cannot have been poison: propagation into the branch would be UB. Trace
    // only scalar strict dependencies. A scalar extract does NOT certify its whole source vector;
    // Select/Phi certify neither unselected inputs nor untaken incoming edges.
    std::vector<uint32_t> pending{graph.predicate};
    while (!pending.empty()) {
        const uint32_t id = pending.back(); pending.pop_back();
        if (!nonpoison.insert(id).second) continue;
        const auto found = graph.values.find(id);
        if (found == graph.values.end() || found->second.opcode == 169 ||
            found->second.opcode == 245 || !executable(graph, found->second)) continue;
        const auto& value = found->second;
        // Vulkan's default environment assumes NotInf/NotNaN for cross-float Bitcast and GLSL
        // math even under SignedZeroInfNanPreserve. A nonpoison strict predicate dependency on
        // one therefore proves its scalar operand/result finite; a random nonpoison float does
        // NOT. Explicit overrides were excluded by the caller, not guessed from captures.
        if (value.opcode == 12 || (value.opcode == 124 &&
            (value.type == Type::Float32 || graph.values.at(value.operands[0]).type == Type::Float32))) {
            if (stable.contains(id)) finite_bits.insert(id);
            for (uint32_t operand : value.operands)
                if (stable.contains(operand)) finite_bits.insert(operand);
        }
        pending.insert(pending.end(), found->second.operands.begin(), found->second.operands.end());
    }

    std::unordered_map<uint32_t, Abstract> facts;
    for (const auto& [id, value] : graph.values) {
        Abstract fact;
        fact.nonpoison = nonpoison.contains(id) || value.constant || value.opcode == 1;
        fact.finite_bits = finite_bits.contains(id) || (value.constant &&
            (value.type == Type::Int32 || value.type == Type::Float32) &&
            (value.literal & 0x7f800000u) != 0x7f800000u);
        if (value.type == Type::Boolean || value.type == Type::Int32) {
            fact.constant = value.constant;
            fact.literal = value.literal;
            if (!value.constant && frozen_leaves.contains(id)) fact.atom = id;
        }
        facts.emplace(id, fact);
    }
    facts.at(graph.predicate) = {true, true, 0, 0};
    for (uint32_t id : graph.body) {
        const auto found = graph.values.find(id);
        if (found == graph.values.end() || !executable(graph, found->second)) return false;
        const auto& value = found->second;
        Abstract result;
        const auto arg = [&](size_t n) { return facts.at(value.operands[n]); };
        const bool all_nonpoison = std::all_of(value.operands.begin(), value.operands.end(),
            [&](uint32_t operand) { return facts.at(operand).nonpoison; });
        const bool exact_type = value.type == Type::Boolean || value.type == Type::Int32;
        if (value.opcode == 169) {
            const auto condition = arg(0);
            if (condition.constant && condition.nonpoison) result = arg(condition.literal ? 1 : 2);
            else if (condition.nonpoison && arg(1) == arg(2)) result = arg(1);
            else result.nonpoison = all_nonpoison;
        } else if (value.opcode == 83 && exact_type) result = arg(0);
        else if (value.opcode == 124) {
            result.nonpoison = all_nonpoison && (arg(0).finite_bits ||
                (value.type == Type::Int32 && graph.values.at(value.operands[0]).type == Type::Int32));
            result.finite_bits = arg(0).finite_bits && result.nonpoison;
        }
        else if (value.opcode == 168 && arg(0).constant && arg(0).nonpoison)
            result = {true, true, !arg(0).literal, 0};
        else if ((value.opcode == 166 || value.opcode == 167) && all_nonpoison) {
            result.nonpoison = true;
            const auto a = arg(0), b = arg(1);
            const uint32_t absorbing = value.opcode == 167 ? 0 : 1;
            if ((a.constant && a.literal == absorbing) || (b.constant && b.literal == absorbing))
                result = {true, true, absorbing, 0};
            else if (a.constant && b.constant)
                result = {true, true, value.opcode == 167 ? (a.literal && b.literal) :
                                                          (a.literal || b.literal), 0};
        } else {
            // FP arithmetic/intrinsics can introduce poison. Preserved scalar comparisons
            // propagate it but introduce none; cross-float bitcasts need finite bits above.
            result.nonpoison = all_nonpoison && value.opcode != 12 &&
                value.opcode != 127 && value.opcode != 129 && value.opcode != 131 && value.opcode != 133;
        }
        // F32 identity is never a bit identity (sNaN quieting); only its nonpoison fact survives.
        if (!exact_type) { result.constant = false; result.literal = 0; result.atom = 0; }
        facts.at(id) = result;
    }
    for (const auto& output : graph.exports) {
        if (!graph.values.contains(output.skipped) || !graph.values.contains(output.executed)) return false;
        const auto type = graph.values.at(output.skipped).type;
        if ((type != Type::Boolean && type != Type::Int32) ||
            type != graph.values.at(output.executed).type) return false;
        const auto old = facts.at(output.skipped), next = facts.at(output.executed);
        // Finite-domain metadata is not part of the observable Boolean/Int32 value identity.
        if (!old.nonpoison || (!old.constant && !old.atom) || !next.nonpoison ||
            old.constant != next.constant || old.literal != next.literal || old.atom != next.atom) return false;
    }
    return true;
}
} // namespace prosper::gpu

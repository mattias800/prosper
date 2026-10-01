#include "gpu/recompiler/spirv_fragment_uniform_trace.hpp"
#include <algorithm>

namespace prosper::gpu {
namespace {
std::unordered_set<uint32_t> reachable_without(const FragmentUniformTrace& trace, uint32_t excluded) {
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> pending{trace.entry_block};
    while (!pending.empty()) {
        const uint32_t block = pending.back();
        pending.pop_back();
        if (block == excluded || !seen.insert(block).second) continue;
        const auto next = trace.successors.find(block);
        if (next == trace.successors.end()) return {};
        pending.insert(pending.end(), next->second.begin(), next->second.end());
    }
    return seen;
}
} // namespace

std::unordered_set<uint32_t> prove_fragment_uniform_trace(
    const FragmentUniformTrace& trace, const std::unordered_set<uint32_t>& leaves,
    const std::vector<uint32_t>& predicates) {
    if (!trace.entry_block || reachable_without(trace, 0).size() != trace.successors.size()) return {};
    std::vector<uint32_t> pending = predicates;
    pending.insert(pending.end(), trace.selectors.begin(), trace.selectors.end());
    std::unordered_set<uint32_t> candidates;
    while (!pending.empty()) {
        const uint32_t value = pending.back();
        pending.pop_back();
        if (leaves.contains(value) || !candidates.insert(value).second) continue;
        const auto def = trace.values.find(value);
        if (def == trace.values.end() || def->second.operands.empty() ||
            !trace.successors.contains(def->second.block)) return {};
        pending.insert(pending.end(), def->second.operands.begin(), def->second.operands.end());
    }

    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> avoiding_header;
    std::unordered_map<uint32_t, uint32_t> dependencies;
    std::unordered_map<uint32_t, std::vector<uint32_t>> users;
    std::vector<uint32_t> ready;
    for (uint32_t id : candidates) {
        const auto& value = trace.values.at(id);
        const bool phi = !value.phi_predecessors.empty();
        if (phi && value.phi_predecessors.size() != value.operands.size()) return {};
        if (phi && !avoiding_header.contains(value.block))
            avoiding_header.emplace(value.block, reachable_without(trace, value.block));
        uint32_t count = 0;
        bool initialized = !phi;
        for (size_t i = 0; i < value.operands.size(); ++i) {
            if (phi) {
                const uint32_t predecessor = value.phi_predecessors[i];
                const auto edge = trace.successors.find(predecessor);
                if (edge == trace.successors.end() ||
                    std::find(edge->second.begin(), edge->second.end(), value.block) == edge->second.end()) return {};
                // If no entry path reaches this predecessor without the Phi block, this is a
                // dominance back edge. Its value was STILL closed above, but is not an initializer.
                if (!avoiding_header.at(value.block).contains(predecessor)) continue;
                initialized = true;
            }
            const uint32_t operand = value.operands[i];
            if (leaves.contains(operand)) continue;
            ++count;
            users[operand].push_back(id);
        }
        if (!initialized) return {};
        dependencies.emplace(id, count);
        if (!count) ready.push_back(id);
    }
    size_t grounded = 0;
    while (!ready.empty()) {
        const uint32_t id = ready.back();
        ready.pop_back();
        ++grounded;
        for (uint32_t user : users[id])
            if (--dependencies.at(user) == 0) ready.push_back(user);
    }
    if (grounded != candidates.size()) return {};
    return candidates;
}
} // namespace prosper::gpu

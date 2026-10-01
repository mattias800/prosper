#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// Internal proof graph, not a general SPIR-V validator. The caller admits only total scalar
// operations from valid SSA, supplies ALL entry-function selectors, and freezes its independently
// defined uniform leaves. A vote node depends on its predicate; it is never a uniform root.
struct FragmentTraceValue {
    uint32_t block = 0;
    std::vector<uint32_t> operands;
    // Empty for ordinary operations; otherwise parallel to the Phi's incoming values.
    std::vector<uint32_t> phi_predecessors;
};

struct FragmentUniformTrace {
    uint32_t entry_block = 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> successors;
    std::unordered_map<uint32_t, FragmentTraceValue> values;
    std::vector<uint32_t> selectors;
};

// Equal initialization + equal total updates + equal selectors imply equal values at MATCHING
// dynamic visits, not equality between iterations. All dependencies, including back-edge values,
// must be certified. Only for the initialization check are dominance back edges removed; the
// remaining dependency graph must be grounded and acyclic. Merely visiting an SCC is no proof.
// Returns new certificates transactionally; refusal never modifies the frozen leaves.
// This is a logical guest-wave trace, including helpers, NOT an arbitrary host-subgroup trace.
// Ground initialization first; each guest scalar reduction of equal defined P equals P. All Any
// results become Copy(P) in the same transaction, preserving selectors and subsequent updates.
// Native helper participation/results are never initialization authority, including for an
// already-certified leaf derived from an Any (that Any must also be rewritten transactionally).
std::unordered_set<uint32_t> prove_fragment_uniform_trace(
    const FragmentUniformTrace& trace, const std::unordered_set<uint32_t>& leaves,
    const std::vector<uint32_t>& predicates);

} // namespace prosper::gpu

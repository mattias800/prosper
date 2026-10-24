#pragma once

#include <cstdint>
#include <optional>
#include <set>

namespace prosper::gpu {
struct Rdna2Inst;
struct RegState;
struct ScalarSourceMarks;

using PredicateRootReader = const std::set<uint64_t>& (*)(const RegState&, const ScalarSourceMarks&,
                                                          uint32_t);

// Refine an already admitted predicate's root observations, never its definedness.
// The legacy reader supplies each actual word's current-EXEC dependencies before
// combining operands can lose their distinct owners. Unsupported forms return nullopt.
std::optional<std::set<uint64_t>> predicate_operand_roots(const RegState& rs, const Rdna2Inst& in,
                                                          PredicateRootReader read_roots);
} // namespace prosper::gpu

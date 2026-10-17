// eval_lanes.hpp — evaluate an IR Block once per invocation of a subgroup, for the cross-lane
// opcodes (LaneId, Shuffle) whose result depends on other invocations.
//
// `evaluate` (eval.hpp) sees one invocation, so a cross-lane value is always unknown there and the
// constant folder never folds one. This evaluates every invocation of a subgroup together: the
// invocation count is `inputs.size()`, every invocation executes the Block (SPIR-V's "active"
// invocations: guest EXEC is a separate Bool value, not host control flow), and value `i` of lane
// `l` is `result[l][i]`. A Shuffle whose lane is not an invocation of the subgroup is unknown,
// which is SPIR-V's "undefined" (OpGroupNonUniformShuffle). Blocks with U64 values are not
// covered: their entries are unknown.
#pragma once

#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <span>
#include <vector>

namespace prosper::gpu::ir {

using InvocationInputs = std::vector<Known>;   // one invocation's input slots

std::vector<std::vector<Known>> evaluate_lanes(const Block& block,
                                               std::span<const InvocationInputs> inputs);

} // namespace prosper::gpu::ir

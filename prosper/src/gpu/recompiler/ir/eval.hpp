// eval.hpp — evaluate an IR Block on inputs that may or may not be known.
//
// This is the constant folder's semantics: a value is known when every argument it depends on is
// known and the operation is defined for those arguments (a shift by 32 or more is not). Booleans
// evaluate to 0 or 1.
#pragma once

#include "gpu/recompiler/ir/ir.hpp"

#include <optional>
#include <span>
#include <vector>

namespace prosper::gpu::ir {

using Known = std::optional<uint32_t>;

// One entry per instruction of `block`. `inputs[i]` is input slot i; a missing slot is unknown.
std::vector<Known> evaluate(const Block& block, std::span<const Known> inputs);

// One instruction on known arguments; nullopt when the operation is undefined for them.
Known evaluate_op(Opcode op, uint32_t imm, uint32_t a, uint32_t b, uint32_t c);

} // namespace prosper::gpu::ir

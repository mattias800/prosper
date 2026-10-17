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
using Known64 = std::optional<uint64_t>;   // any Value, including a U64 word

// One entry per instruction of `block`. `inputs[i]` is input slot i; a missing slot is unknown.
// A U64 instruction has no 32-bit view, so its entry is always unknown here: read it through
// evaluate_wide, which is the same evaluation without the narrowing.
std::vector<Known> evaluate(const Block& block, std::span<const Known> inputs);
// The same, with every value held in 64 bits. A U32 or Bool input wider than 32 bits is unknown.
std::vector<Known64> evaluate_wide(const Block& block, std::span<const Known64> inputs);

// One instruction on known arguments; nullopt when the operation is undefined for them.
Known evaluate_op(Opcode op, uint32_t imm, uint32_t a, uint32_t b, uint32_t c);

} // namespace prosper::gpu::ir

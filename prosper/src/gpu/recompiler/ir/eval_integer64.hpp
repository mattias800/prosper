// eval_integer64.hpp — the evaluation half of the 64-bit word opcodes (PackU64 .. U64Ge, ir.hpp).
#pragma once

#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <cstdint>

namespace prosper::gpu::ir {

// True for an opcode whose result or an argument is a U64 word; evaluate() routes those here.
bool is_integer64_opcode(Opcode op);

// One 64-bit-word instruction on known arguments, each held in a uint64_t (a U32 argument is
// zero-extended). A Bool result is 0 or 1.
Known64 evaluate_integer64(Opcode op, uint64_t a, uint64_t b);

} // namespace prosper::gpu::ir

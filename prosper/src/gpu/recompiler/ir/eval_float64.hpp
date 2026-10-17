// eval_float64.hpp — the evaluation half of the binary64 relations (F64Lt .. F64UnordGe, ir.hpp).
#pragma once

#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"

#include <cstdint>

namespace prosper::gpu::ir {

// True for an opcode that reads U64 words as IEEE binary64 values; evaluate_wide routes those here.
bool is_float64_opcode(Opcode op);

// One binary64 relation on two known U64 words. The result is 0 or 1. Evaluated on the host with
// unflushed subnormals, which is what the emitter's DenormPreserve 64 asks the device for; an input
// flush the guest MODE requests is the lowering's job, before the relation. Tests only: the
// constant folder never evaluates an F64 opcode.
Known64 evaluate_float64(Opcode op, uint64_t a, uint64_t b);

} // namespace prosper::gpu::ir

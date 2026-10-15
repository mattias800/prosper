// Reference evaluation of the retained F32 unary lowering; no guest folder caller.
#pragma once
#include <cstdint>
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
namespace prosper::gpu::ir {
Known evaluate_float_unary(Opcode op, uint32_t word);
}

// Evaluation of the binary32 relations F32Le .. F32UnordGe (eval_float_compare.cpp).
#pragma once
#include <cstdint>
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
namespace prosper::gpu::ir {
Known evaluate_float_compare(Opcode op, uint32_t a, uint32_t b);
}

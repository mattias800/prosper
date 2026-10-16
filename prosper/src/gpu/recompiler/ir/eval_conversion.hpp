// Nominal conversion reference; unknown half NaN payloads are never invented.
#pragma once
#include <cstdint>
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
namespace prosper::gpu::ir {
Known evaluate_conversion(Opcode op, uint32_t imm, uint32_t a, uint32_t b, uint32_t c);
}

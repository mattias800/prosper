// A separately admitted alias; the original VOP1 registry and builder stay unchanged.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
const ConversionDef* find_vector_cvt_rpi_exact_e64_op(Rdna2Format fmt, uint32_t opcode);
} // namespace prosper::gpu::ir

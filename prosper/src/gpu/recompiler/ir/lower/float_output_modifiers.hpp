#pragma once
#include "gpu/recompiler/ir/ops/float_output_modifier.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
namespace prosper::gpu::ir {
FloatOutputModifier float_output_modifier(SpirvCompute& b, const Rdna2Inst& in,
                                          FloatResultWidth width, bool& ok);
uint32_t modify_float_result(SpirvCompute& b, const Rdna2Inst& in, uint32_t bits, bool& ok);
} // namespace prosper::gpu::ir

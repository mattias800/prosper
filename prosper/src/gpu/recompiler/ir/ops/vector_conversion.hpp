// Shared numeric definitions; the lowering retains source selection, modifiers and placement.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct ConversionIn {
    Value s0;
    bool high;
};
struct ConversionDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder&, const ConversionIn&);
};
std::span<const ConversionDef> vector_conversion_defs();
std::span<const ConversionDef> vector_half_integer_conversion_defs();
const ConversionDef* find_vector_conversion_op(Rdna2Format fmt, uint32_t opcode);
const ConversionDef* find_vector_half_integer_conversion_op(Rdna2Format fmt, uint32_t opcode);
Known eval_conversion_op(const ConversionDef& def, Known s0, bool high = false);
uint32_t emit_conversion_op(SpirvCompute& b, const ConversionDef& def, uint32_t s0,
                            bool high = false);
}   // namespace prosper::gpu::ir

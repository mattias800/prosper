// Per-lane f16 definitions: unpacked F32 sources -> low half result; packed RTZ -> full dword.
// Source selects, abs/neg, the preserved destination half and EXEC remain in the lowering.
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
struct HalfOpIn {
    Value s0, s1;
    bool clamp;
};
struct HalfOpDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder&, const HalfOpIn&);
};
// Shared integer-backed value graph; source reads remain with each encoding.
Value build_half_rtz(Builder& ir, Value a, Value b);
std::span<const HalfOpDef> vector_half_defs();
const HalfOpDef* find_vector_half_op(Rdna2Format fmt, uint32_t opcode);
Known eval_half_op(const HalfOpDef& def, Known s0, Known s1, bool clamp = false);
uint32_t emit_half_op(SpirvCompute& b, const HalfOpDef& def, uint32_t s0, uint32_t s1,
                      bool clamp = false);
uint32_t emit_half_rtz(SpirvCompute& b, uint32_t s0, uint32_t s1);
} // namespace prosper::gpu::ir

// Producing MODE policy for integer-valued F32 rounding, shared by instruction encodings.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
struct VectorOpDef;
uint32_t emit_f32_rounding_op(SpirvCompute& b, const Rdna2Inst& in, const VectorOpDef& def,
                              uint32_t source, bool& ok);
} // namespace prosper::gpu::ir

// The root retains opcode matching, output modifiers and final EXEC publication.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
struct Operand;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using NativeHalfFmaSource = std::function<uint32_t(const Operand&)>;
using NativeHalfFmaResult = std::function<uint32_t(uint32_t)>;
void lower_native_half_fma(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                           bool allow_wave, uint32_t old_d, const NativeHalfFmaSource& val,
                           const NativeHalfFmaResult& fresult);
}   // namespace prosper::gpu::ir

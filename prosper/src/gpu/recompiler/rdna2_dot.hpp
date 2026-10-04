#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;

// Lowers RDNA2 VOP3P integer dot product instructions:
// 0x14: v_dot2_i32_i16 (2x16-bit signed dot product + 32-bit accumulator)
// 0x15: v_dot2_u32_u16 (2x16-bit unsigned dot product + 32-bit accumulator)
// 0x16: v_dot4_i32_i8  (4x8-bit signed dot product + 32-bit accumulator)
// 0x17: v_dot4_u32_u8  (4x8-bit unsigned dot product + 32-bit accumulator)
// 0x18: v_dot8_i32_i4  (8x4-bit signed dot product + 32-bit accumulator)
// 0x19: v_dot8_u32_u4  (8x4-bit unsigned dot product + 32-bit accumulator)
uint32_t emit_v_dot(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);

}  // namespace prosper::gpu

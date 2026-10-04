#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;

// Lowers RDNA2 V_DIV_FIXUP_F32 (opcode 0x15f): IEEE 754 division special-case fixup.
// S0: quotient, S1: denominator, S2: numerator.
uint32_t emit_v_div_fixup_f32(SpirvCompute& b, uint32_t s0, uint32_t s1, uint32_t s2);

}  // namespace prosper::gpu

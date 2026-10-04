#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;

// Lowers RDNA2 V_PERM_B32 (opcode 0x344): byte permutation across high (src0) and low (src1)
// words controlled by selector bytes in src2.
uint32_t emit_v_perm_b32(SpirvCompute& b, uint32_t high, uint32_t low, uint32_t sel_dword);

}  // namespace prosper::gpu

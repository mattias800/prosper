#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;

// Lowers RDNA2 VOP3 subword Sum of Absolute Differences (SAD) instructions:
// 0x15A: v_sad_u8    - sum of 4 byte |a-b| + accumulator
// 0x15B: v_sad_hi_u8 - (sum of 4 byte |a-b| << 16) + accumulator
// 0x15C: v_sad_u16   - sum of 2 16-bit |a-b| + accumulator
// 0x171: v_msad_u8   - masked sum of 4 byte |a-b| (where b_byte != 0) + accumulator
uint32_t emit_v_sad_subword(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);

}  // namespace prosper::gpu

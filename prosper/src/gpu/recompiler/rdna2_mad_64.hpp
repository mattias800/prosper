#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;

// The 64-bit multiply-add shared by VOP3B v_mad_u64_u32 (0x176) and v_mad_i64_i32 (0x177):
// {carry, D} = S0 * S1 + S2, with S2 given as its two dwords. Returns SPIR-V ids for the low and
// high result dwords and the per-lane carry-out bool (before the EXEC carry-mask rule).
struct Mad64Result {
    uint32_t lo;
    uint32_t hi;
    uint32_t carry;
};
Mad64Result emit_mad_64_32(SpirvCompute& b, uint32_t s0, uint32_t s1, uint32_t add_lo,
                           uint32_t add_hi, bool is_signed);

}  // namespace prosper::gpu

// carry_mask_write.hpp — the shared lane-mask write of a carry op (carry_mask_write.cpp).
#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {

// ISA 3.9: an EXEC-inactive lane's carry bit is written 0, never its raw carry. `carry` is the
// definition's unmasked carry Bool; the result is what the mask destination receives.
uint32_t exec_masked_carry(SpirvCompute& b, RegState& rs, uint32_t carry);

// VOP2 (e32) v_add/sub/subrev_co_ci_u32: the masked carry replaces VCC.
void write_vop2_carry_mask(SpirvCompute& b, RegState& rs, uint32_t carry);

// VOP3B: the masked carry goes to SDST (VCC, an SGPR pair, or in a Wave32 B32-mask stage one SGPR
// word). Any other SDST refuses (`ok = false`).
void write_vop3b_carry_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            uint32_t carry);

}   // namespace prosper::gpu::ir

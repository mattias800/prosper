// vop3_mask_lowering.hpp — the VOP3 operations that read or write a lane mask: v_cndmask_b32_e64
// (reads SRC2's mask) and the VOP3B carry ops (write the SDST carry mask; the _co_ci forms also read
// a carry-in mask from SRC2). Each returns false when the instruction is not its opcode. The caller
// owns the old-VDST capture, CLAMP and the predicated write.
#pragma once

#include <cstdint>
#include <functional>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {

using Vop3MaskSource = std::function<uint32_t(int)>;   // emit_alu's `fv`

// v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 (VOP3B 0x30F / 0x310 / 0x319).
bool lower_vop3_carry_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave);
// v_add_co_ci_u32 / v_sub_co_ci_u32 / v_subrev_co_ci_u32 (VOP3B 0x128 / 0x129 / 0x12A).
bool lower_vop3_carry_in_out(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave);
// v_cndmask_b32_e64 (0x101).
bool lower_vop3_cndmask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                        bool allow_wave, const Vop3MaskSource& fv);

}   // namespace prosper::gpu::ir

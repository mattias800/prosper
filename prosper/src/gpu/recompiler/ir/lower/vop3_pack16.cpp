// The existing whole-dword halfword pack, mechanically extracted without changing source order.
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_pack16_lowering_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3Pack16Lowering::lower() {
    if (in.opcode != 0x311) return false;
    // v_pack_b32_f16: D.u32 = {S1.f16, S0.f16} — S0's selected half becomes bits
    // [15:0] and S1's becomes [31:16]. OPSEL[k] picks which half of each source is
    // read; the destination is a whole dword, so OPSEL[3] has nothing to select.
    // This is the instruction that re-assembles the two halves the 16-bit convert
    // family produces, so it appears immediately behind them (Sonic Racing:
    // CrossWorlds, `d7110020,00020500` = `v_pack_b32_f16 v32, v0, v2`, #2013).
    // VERIFIED(round-trip llvm-mc gfx1030). It is a bit move, so ABS/NEG (which would
    // be f16 sign operations on the selected halves) and CLAMP/OMOD are not modeled
    // and reject rather than being silently dropped.
    if (in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1] || in.clamp || in.omod) {
        ok = false;
    } else {
        auto packed_half = [&](int k) {
            const Operand& operand = in.src[k];
            // OPSEL[k] selects a half of an inline constant exactly as it does of a
            // register — see inline_16bit_operand_dword. Both inline branches used to
            // ignore it, so `1.0` read 0x3c00 and inline `1` read 1 from the HIGH half
            // where hardware reads 0 (#2119, kernel 32r14d).
            const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
            uint32_t inline_dword;
            if (inline_16bit_operand_dword(operand, inline_dword))
                return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
            return b.bfe_u(val(operand), b.uconst(16u * sel), b.uconst(16));
        };
        vreg[in.dst.value] = b.ibin(Op_BitwiseOr, packed_half(0),
                                    b.ibin(Op_ShiftLeftLogical, packed_half(1), b.uconst(16)));
    }
    return true;
}
bool lower_vop3_pack16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                       bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Pack16Lowering({b, rs, in, ok, allow_wave}).lower();
}
} // namespace prosper::gpu::ir

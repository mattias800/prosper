// The original five word-result SAD cases and subword helper, moved mechanically.
#include "gpu/recompiler/ir/lower/vop3_sad_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_sad_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
namespace {
uint32_t emit_v_sad_subword(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    const uint32_t s0 = operand_bits(b, rs, in, in.src[0], &ok);
    const uint32_t s1 = operand_bits(b, rs, in, in.src[1], &ok);
    const uint32_t s2 = operand_bits(b, rs, in, in.src[2], &ok);
    if (!ok) return b.uconst(0);

    // RDNA2 SAD family (AMD doc 70648):
    // 0x15A: v_sad_u8    - sum of four unsigned byte differences plus S2, wrapping mod 2^32.
    // 0x15B: v_sad_hi_u8 - (sum of four unsigned byte differences << 16) plus S2.
    // 0x15C: v_sad_u16   - sum of two unsigned 16-bit differences plus S2.
    // 0x171: v_msad_u8   - masked sum of byte differences, skipping byte k when S1[k] == 0.
    //                      Verified against RDNA3/CDNA3 pseudocode and D3D msad lowering.
    //                      CONFIDENCE: MED (operand role from CDNA3 / compiler lowering; 70648
    //                      states only "Masked Byte SAD with accum_lo(S0, S1, S2)").
    const bool is_u16 = (in.opcode == 0x15C);
    const bool masked = (in.opcode == 0x171);
    const uint32_t field_bits = is_u16 ? 16u : 8u;

    uint32_t sum = b.uconst(0);
    for (uint32_t off = 0; off < 32u; off += field_bits) {
        const uint32_t a_elem = b.bfe_u(s0, b.uconst(off), b.uconst(field_bits));
        const uint32_t b_elem = b.bfe_u(s1, b.uconst(off), b.uconst(field_bits));
        uint32_t diff =
            b.ibin(Op_ISub, b.uext2(Glsl_UMax, a_elem, b_elem), b.uext2(Glsl_UMin, a_elem, b_elem));
        if (masked) {
            const uint32_t nz = b.ucmp(Op_INotEqual, b_elem, b.uconst(0));
            diff = b.sel(nz, diff, b.uconst(0));
        }
        sum = b.ibin(Op_IAdd, sum, diff);
    }
    if (in.opcode == 0x15B) { sum = b.ibin(Op_ShiftLeftLogical, sum, b.uconst(16)); }
    return b.ibin(Op_IAdd, sum, s2);
}
} // namespace
bool Vop3SadLowering::lower() {
    // Temporary sentinel keeps the extracted else-if chain verbatim.
    if (false) {
    } else if (in.opcode == 0x15A || in.opcode == 0x15B || in.opcode == 0x15C ||
               in.opcode == 0x171) {   // v_sad_u8 / v_sad_hi_u8 / v_sad_u16 / v_msad_u8
        vreg[in.dst.value] = emit_v_sad_subword(b, rs, in, ok);
    } else if (in.opcode == 0x15D) {   // v_sad_u32 = |s0-s1| (unsigned) + s2
        // RDNA2 ISA (document 70648), V_SAD_U32: D.u32 = abs(S0.u32 - S1.u32) + S2.u32.
        // The absolute difference is the UNSIGNED magnitude, so max-min is exact and cannot
        // wrap (a signed abs would be wrong for operands straddling 0x80000000). Worms
        // Armageddon's PSSL compiler emits the src1=0 accumulate form `v_sad_u32 vN, sM, 0,
        // vN` as its vertex-shader index prologue; its shipped .ags shader assets carry the
        // identical word. VERIFIED(llvm-mc gfx1030: VOP3 0x15d = v_sad_u32).
        // CONFIDENCE: HIGH.
        const uint32_t s0 = val(in.src[0]), s1 = val(in.src[1]);
        const uint32_t diff =
            b.ibin(Op_ISub, b.uext2(Glsl_UMax, s0, s1), b.uext2(Glsl_UMin, s0, s1));
        vreg[in.dst.value] = b.ibin(Op_IAdd, diff, val(in.src[2]));
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_sad(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3SadLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir

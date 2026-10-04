#include "gpu/recompiler/rdna2_sad.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"

namespace prosper::gpu {

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

}  // namespace prosper::gpu

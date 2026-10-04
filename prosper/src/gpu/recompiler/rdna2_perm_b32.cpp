#include "gpu/recompiler/rdna2_perm_b32.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t emit_v_perm_b32(SpirvCompute& b, uint32_t high, uint32_t low, uint32_t sel_dword) {
    // RDNA2 V_PERM_B32 (AMD doc 70648): byte permute across {src0 (high), src1 (low)}
    // controlled by 4 selector bytes in src2.
    //   sel 0..7: selects byte 0..7 from {src0, src1} (bytes 0..3 from src1, 4..7 from src0)
    //   sel 8..11: replicates bit 7 of bytes 1, 3, 5, 7 of {src0, src1} (S1.b1, S1.b3, S0.b1, S0.b3)
    //   sel 12: constant 0x00
    //   sel 13..255: constant 0xFF
    // VERIFIED(llvm-mc gfx1030: VOP3 0x144 = v_perm_b32). CONFIDENCE: HIGH.
    uint32_t result = b.uconst(0);
    for (uint32_t byte = 0; byte < 4u; ++byte) {
        const uint32_t selector = b.bfe_u(sel_dword, b.uconst(byte * 8u), b.uconst(8));
        const uint32_t sign_index = b.ibin(
            Op_IAdd, b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, selector, b.uconst(3)), b.uconst(2)),
            b.uconst(1));
        const uint32_t index =
            b.sel(b.ucmp(Op_ULessThan, selector, b.uconst(8)), selector, sign_index);
        const uint32_t word = b.sel(b.ucmp(Op_ULessThan, index, b.uconst(4)), low, high);
        const uint32_t shift =
            b.ibin(Op_IMul, b.ibin(Op_BitwiseAnd, index, b.uconst(3)), b.uconst(8));
        const uint32_t val_byte = b.bfe_u(word, shift, b.uconst(8));
        const uint32_t sign =
            b.sel(b.ucmp(Op_UGreaterThan, val_byte, b.uconst(127)), b.uconst(255), b.uconst(0));
        const uint32_t selected =
            b.sel(b.ucmp(Op_ULessThan, selector, b.uconst(8)), val_byte, sign);
        const uint32_t fill =
            b.sel(b.ucmp(Op_IEqual, selector, b.uconst(12)), b.uconst(0), b.uconst(255));
        const uint32_t out_byte =
            b.sel(b.ucmp(Op_ULessThan, selector, b.uconst(12)), selected, fill);
        result = b.ibin(Op_BitwiseOr, result,
                        b.ibin(Op_ShiftLeftLogical, out_byte, b.uconst(byte * 8u)));
    }
    return result;
}

}  // namespace prosper::gpu

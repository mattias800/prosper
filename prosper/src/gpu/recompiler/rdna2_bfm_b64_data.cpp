#include "gpu/recompiler/rdna2_bfm_b64_data.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

bool record_bfm_b64_data(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    // Recorded only when both source words are defined DATA: #4749's mark (or a
    // non-data operand kind) withholds it, so a fabricated word can never become
    // a tracked zero here, while genuinely undefined sources keep the mask-only
    // behavior they always had.
    auto data_defined = [&](const Operand& o) -> bool {
        if (o.kind == OperandKind::InlineInt || o.kind == OperandKind::Literal) return true;
        if (o.kind == OperandKind::Special && o.value == 125)
            return true;   // SGPR_NULL: the one Special whose data is 0
        if (o.kind == OperandKind::SGPR) return !sreg_word_may_be_fabricated(rs, o.value);
        return false;      // VGPR, masks-as-data and everything else
    };
    if (!data_defined(in.src[0]) || !data_defined(in.src[1])) return false;
    bool data_ok = true;
    const uint32_t width = operand_bits(b, rs, in, in.src[0], &data_ok);
    const uint32_t offset = data_ok ? operand_bits(b, rs, in, in.src[1], &data_ok) : b.uconst(0);
    if (!data_ok) return false;
    // D.u64 from the same width/offset the mask path used, so the two views agree
    // bit-for-bit on every input by construction. Split with every emitted shift
    // amount masked below 32: a 32-bit SPIR-V shift of 32 or more must select,
    // never execute.
    // CONFIDENCE: HIGH on the bitfield formula; MED on operand field widths above
    // 31, shared verbatim with the mask preparation (one place to fix if the ISA's
    // S0/S1 field widths say otherwise).
    const uint32_t w_ge_32 = b.ucmp(Op_UGreaterThanEqual, width, b.uconst(32));
    const uint32_t w_sub =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, width, b.uconst(32)), b.uconst(31));
    const uint32_t w_low = b.ibin(Op_BitwiseAnd, width, b.uconst(31));
    const uint32_t ones_lo =
        b.sel(w_ge_32, b.uconst(0xFFFFFFFFu),
              b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), w_low), b.uconst(1)));
    const uint32_t ones_hi = b.sel(
        w_ge_32, b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), w_sub), b.uconst(1)),
        b.uconst(0));
    const uint32_t o_ge_32 = b.ucmp(Op_UGreaterThanEqual, offset, b.uconst(32));
    const uint32_t o_low = b.ibin(Op_BitwiseAnd, offset, b.uconst(31));
    const uint32_t o_sub =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, offset, b.uconst(32)), b.uconst(31));
    const uint32_t data_lo =
        b.sel(o_ge_32, b.uconst(0), b.ibin(Op_ShiftLeftLogical, ones_lo, o_low));
    const uint32_t o_is_zero = b.ucmp(Op_IEqual, offset, b.uconst(0));
    const uint32_t hi_low_part = b.sel(
        o_is_zero, b.uconst(0),
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftRightLogical, ones_lo,
                      b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), offset), b.uconst(31))),
               b.ibin(Op_ShiftLeftLogical, ones_hi, o_low)));
    const uint32_t data_hi = b.sel(o_ge_32,
                                   b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, ones_lo, o_sub),
                                          b.ibin(Op_ShiftLeftLogical, ones_hi, o_sub)),
                                   hi_low_part);
    rs.sreg[in.dst.value] = data_lo;
    rs.sreg[in.dst.value + 1] = data_hi;
    return true;
}

}  // namespace prosper::gpu

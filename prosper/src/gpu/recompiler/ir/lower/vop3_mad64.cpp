// MAD64 bookkeeping relocated with extract_case_routes.py; arithmetic is unchanged.
#include "gpu/recompiler/ir/lower/vop3_mad64_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3_mad64_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/carry_mask_write.hpp"
#include "gpu/recompiler/rdna2_mad_64.hpp"
namespace prosper::gpu::ir {
std::optional<bool> Vop3Mad64Lowering::route_value() {
    // {carry,D} = S0*S1 + S2 (rdna2_mad_64.cpp). A positive inline constant used as the
    // 64-bit addend is zero-extended, a negative integer inline sign-extended; register
    // addends consume the consecutive high register. A literal addend is zero-extended
    // for 0x176 and refused for 0x177 until its extension is proved.
    if (in.dst.value >= 255) {
        ok = false;
    } else {
        const bool is_signed = (in.opcode == 0x177);
        auto high_half = [&](const Operand& operand) -> uint32_t {
            if (operand.kind == OperandKind::InlineInt)
                return b.uconst(operand.value < 0 ? 0xFFFFFFFFu : 0u);
            // An inline FLOAT 64-bit addend supplies the DOUBLE bit pattern (high dword
            // carries the exponent/mantissa; 1/(2*pi) has a nonzero LOW dword too) —
            // the previous {f32-bits, 0} model was wrong in both halves. No compiler
            // emits a float inline as an integer-mad addend: reject, stay fail-visible.
            if (operand.kind == OperandKind::InlineFloat) {
                ok = false;
                return b.uconst(0);
            }
            if (operand.kind == OperandKind::Literal) {
                ok = ok && !is_signed;
                return b.uconst(0);
            }
            if (operand.kind == OperandKind::Special && operand.value == 125)
                return b.uconst(0);   // null pair
            if (operand.kind == OperandKind::VGPR || operand.kind == OperandKind::SGPR ||
                (operand.kind == OperandKind::Special && operand.value >= 106 &&
                 operand.value < 124)) {
                Operand next = operand;
                ++next.value;
                return val(next);
            }
            ok = false;
            return b.uconst(0);
        };

        const uint32_t a = val(in.src[0]), c = val(in.src[1]);
        const uint32_t add_lo = val(in.src[2]), add_hi = high_half(in.src[2]);
        const Mad64Result mad = emit_mad_64_32(b, a, c, add_lo, add_hi, is_signed);
        const uint32_t result_lo = mad.lo, result_hi = mad.hi, carry_out = mad.carry;

        const int hi_dst = in.dst.value + 1;
        const uint32_t old_hi = vreg_old(b, rs, hi_dst);
        vreg[in.dst.value] = result_lo;
        vreg[hi_dst] = result_hi;
        predicate_write(b, rs, hi_dst, old_hi);
        ir::write_mad64_carry_mask(b, rs, in, ok, carry_out);
    }
    return std::nullopt;
}

bool lower_vop3_mad64(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                      bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3 || (in.opcode != 0x176 && in.opcode != 0x177)) return false;
    Vop3Mad64Lowering lower({b, rs, in, ok, allow_wave});
    lower.route_value();
    return true;
}
}   // namespace prosper::gpu::ir

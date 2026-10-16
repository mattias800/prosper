// Existing packed-integer wrapper around shared per-half operation definitions.
#include "gpu/recompiler/ir/lower/vop3p_packed_integer_lowering.hpp"
#include "gpu/recompiler/ir/lower/vop3p_packed_integer_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_packed_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
// The packed 16-bit INTEGER family. Sonic Racing: CrossWorlds' post chain rejects on
// v_pk_add_u16 (#2013); the rest of the family is the same lowering behind the same
// half-select wrapper. Each result half k is computed from source halves selected by
// OPSEL[src] (low result) / OPSEL_HI[src] (high result), and BOTH halves are always
// written — unlike the scalar 16-bit VOP3 ops, a packed op has no preserved half.
//   0x00 v_pk_mad_i16      D.i16[k] = S0*S1 + S2
//   0x01 v_pk_mul_lo_u16   D.u16[k] = low16(S0 * S1)
//   0x02/0x03 v_pk_add_i16 / v_pk_sub_i16
//   0x04/0x05/0x06 v_pk_lshlrev_b16 / v_pk_lshrrev_b16 / v_pk_ashrrev_i16
//                  (REVERSED operands: D = S1 <op> (S0 & 0xf))
//   0x07/0x08 v_pk_max_i16 / v_pk_min_i16
//   0x09 v_pk_mad_u16      (identical 16-bit bit pattern to v_pk_mad_i16)
//   0x0a/0x0b v_pk_add_u16 / v_pk_sub_u16 (identical bit result to the i16 forms)
//   0x0c/0x0d v_pk_max_u16 / v_pk_min_u16
// VERIFIED(round-trip llvm-mc gfx1030, every opcode above); the live encoding
// `cc0a0002,18020504` disassembles to `v_pk_add_u16 v2, v4, v2`.
// CONFIDENCE: HIGH on the operations themselves.
//
// NEG_LO/NEG_HI on a packed INTEGER source: applied as a flip of bit 15 of the
// selected 16-bit half — physically the same shared sign-negate the packed f16 path
// above already models with fneg, sitting between the operand read and the ALU and
// indifferent to how the ALU then interprets the bits. CONFIDENCE: MED — the
// derivation below is LIVE evidence, and no published statement confirms it.
//
// DO NOT RAISE THIS TO HIGH ON THE LLVM CITATION. It was tried and withdrawn. Two
// things were offered as "published support" and neither holds: (1) LLVM's AMDGPU
// modifier reference says of neg_lo/neg_hi "This modifier is valid for
// floating-point operands only" — that is a RESTRICTION, and if anything it cuts
// AGAINST applying the bit to an integer opcode, which is exactly the unpublished
// step this MED is hedging; (2) the claim that LLVM folds `xor 0x80008000` into
// neg_lo/neg_hi for integer packed ops traces to llvm-project PR #130234, which was
// merged and REVERTED the same day and covered the dot-product family, not
// v_pk_add_u16/v_pk_max_i16. Under the charter's evidence hierarchy that is a single
// secondary implementation — tier 4, hypothesis only.
//
// What actually carries the reading is the live derivation, and it is strong. All
// FOUR sites in this title that set the bits use the same literal
// `0x00007fff` and set NEG_LO[0] and NEG_HI[0] together, with OPSEL/OPSEL_HI
// selecting that literal's LOW half:
//   * `v_pk_add_u16 v4, -(0x7fff), v2 op_sel:[0,1] op_sel_hi:[0,0]` produces the
//     base of a gather whose per-lane offsets are then +0/+2 (a 2x2 quad), so the
//     base must be (x-1, y-1);
//   * `v_pk_max_i16 v4, -(0x7fff), v3` sits beside `v_min_i16 2, …` and
//     `v_pk_min_i16 1, …`, i.e. an offset clamped into a small range around zero.
// Bit-15 flip gives 0xffff = -1, which satisfies both. The alternative reading,
// two's-complement negation, gives -32767 — which would make that `v_pk_max_i16` a
// no-op sitting next to a `min` of +1, and no compiler emits that. If a later title
// contradicts this, the discriminating input is any negated source whose selected
// half is not 0x7fff.

// Inline-float materialization is retained unchanged pending #4407: CONFIDENCE: LOW.
uint32_t Vop3pPackedIntegerLowering::u16src(int k, bool high_result) {
    const Operand& operand = in.src[k];
    const uint8_t selectors = high_result ? in.vop3p_opsel_hi : in.vop3p_opsel;
    const uint32_t half = (selectors >> k) & 1u;
    uint32_t v, inline_dword;
    if (inline_16bit_operand_dword(operand, inline_dword))
        v = b.uconst((inline_dword >> (16u * half)) & 0xFFFFu);
    else
        v = b.bfe_u(val(operand), b.uconst(16u * half), b.uconst(16));
    const bool negate = high_result ? ((in.vop3p_neg_hi >> k) & 1u) != 0 : in.src_neg[k];
    return negate ? b.ibin(Op_BitwiseXor, v, b.uconst(0x8000u)) : v;
}
uint32_t Vop3pPackedIntegerLowering::operation(bool high, const TernaryIntegerDef& def) {
    const uint32_t s0 = u16src(0, high), s1 = u16src(1, high);
    return emit_ternary_integer_op(b, def, [&](uint32_t source) {
        return source == 0 ? s0 : source == 1 ? s1 : u16src(2, high);
    });
}
bool Vop3pPackedIntegerLowering::lower() {
    const auto* def = find_vector_packed_integer_op(in.fmt, in.opcode);
    if (!def) return false;
    const uint32_t old_d = vreg_old(b, rs, in.dst.value);
    rs.vreg[in.dst.value] =
        b.ibin(Op_BitwiseOr, operation(false, *def),
               b.ibin(Op_ShiftLeftLogical, operation(true, *def), b.uconst(16)));
    predicate_write(b, rs, in.dst.value, old_d);
    return true;
}
bool lower_vop3p_packed_integer(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave) {
    if (in.fmt != Rdna2Format::VOP3P) return false;
    return Vop3pPackedIntegerLowering({b, rs, in, ok, allow_wave}).lower();
}
}   // namespace prosper::gpu::ir

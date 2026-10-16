// Existing selected-half integer value stage, mechanically extracted with its source handling.
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering_internal.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer16_lowering.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
bool Vop3Integer16Lowering::lower() {
    // Temporary sentinel preserves the extracted else-if chain verbatim.
    if (false) {
    } else if ((in.opcode >= 0x303 && in.opcode <= 0x30E && in.opcode != 0x306) ||
               in.opcode == 0x314 || in.opcode == 0x340 || in.opcode == 0x35E ||
               in.opcode == 0x352 || in.opcode == 0x353 || in.opcode == 0x355 ||
               in.opcode == 0x356 || in.opcode == 0x358 || in.opcode == 0x359) {
        // The 16-bit integer VALU family. Sonic Racing: CrossWorlds' post chain rejects on
        // v_lshrrev_b16 at thirty sites and v_add_nc_u16 at twenty-four per boot (#2013);
        // the rest of the family is the same lowering behind the same half-select wrapper.
        // Every one of them selects each source half with OPSEL[k] and the destination half
        // with OPSEL[3], PRESERVING the opposite destination half.
        //   0x303 v_add_nc_u16   D.u16 = S0.u16 + S1.u16
        //   0x304 v_sub_nc_u16   D.u16 = S0.u16 - S1.u16
        //   0x305 v_mul_lo_u16   D.u16 = (S0.u16 * S1.u16) & 0xffff
        //   0x307 v_lshrrev_b16  D.u16 = S1.u16 >> S0.u[3:0]      (REVERSED operands)
        //   0x308 v_ashrrev_i16  D.i16 = S1.i16 >> S0.u[3:0]      (REVERSED operands)
        //   0x309/0x30a/0x30b/0x30c  v_max_u16 / v_max_i16 / v_min_u16 / v_min_i16
        //   0x30d/0x30e  v_add_nc_i16 / v_sub_nc_i16 (identical bit result to the u16 forms)
        //   0x314 v_lshlrev_b16  D.u16 = S1.u16 << S0.u[3:0]      (REVERSED operands)
        //   0x340/0x35e  v_mad_u16 / v_mad_i16   D = S0*S1 + S2 (identical 16-bit result)
        //   0x352/0x353/0x355/0x356/0x358/0x359
        //                v_min3_i16 / v_min3_u16 / v_max3_i16 / v_max3_u16 /
        //                v_med3_i16 / v_med3_u16
        // VERIFIED(round-trip llvm-mc gfx1030, every opcode above; the live encoding
        // `d7074001,00020081` disassembles to `v_lshrrev_b16 v1, 1, v0 op_sel:[0,0,1]`,
        // which writes the HIGH half). 0x306 is not an instruction on gfx10.3, and
        // v_lshlrev_b16 is 0x314 — NOT 0x305, which is the 16-bit multiply.
        // CONFIDENCE: HIGH.
        // ABS/NEG are float-only modifiers with no meaning on a 16-bit integer op; reject
        // rather than silently drop them.
        if (in.src_abs[0] || in.src_abs[1] || in.src_abs[2] || in.src_neg[0] || in.src_neg[1] ||
            in.src_neg[2]) {
            ok = false;
        } else {
            auto u16src = [&](int k) {
                const Operand& operand = in.src[k];
                // A 16-bit operand position carries the 16-bit encoding of an inline
                // constant, not the f32 pattern val() would materialize — and OPSEL[k]
                // selects a half of THAT dword (see inline_16bit_operand_dword). Both
                // inline branches used to ignore the select, so a positive inline int read
                // its own value from the HIGH half where hardware reads 0 (#2119, kernel
                // 32r14e). Negative inline ints were already right by accident: -N is
                // sign-extended, so both halves are 0xffff.
                const uint32_t sel = (in.vop3p_opsel >> k) & 1u;
                uint32_t inline_dword;
                if (inline_16bit_operand_dword(operand, inline_dword))
                    return b.uconst((inline_dword >> (16u * sel)) & 0xFFFFu);
                const uint32_t raw = val(operand);
                return b.ibin(Op_BitwiseAnd,
                              sel ? b.ibin(Op_ShiftRightLogical, raw, b.uconst(16)) : raw,
                              b.uconst(0xFFFFu));
            };
            // Sign-extend a 16-bit value into the full 32-bit signed domain.
            auto s16 = [&](uint32_t v) { return b.bfe_s(v, b.uconst(0), b.uconst(16)); };
            const uint32_t s0 = u16src(0), s1 = u16src(1);
            // Three-source forms only; u16src(2) is not evaluated for the two-source ops.
            auto s2 = [&] { return u16src(2); };
            // Integer median of three, the same formula the u32 form uses.
            auto med3 = [&](uint32_t x, uint32_t y, uint32_t z, bool is_signed) {
                const uint32_t lo = is_signed ? (uint32_t)Glsl_SMin : (uint32_t)Glsl_UMin;
                const uint32_t hi = is_signed ? (uint32_t)Glsl_SMax : (uint32_t)Glsl_UMax;
                auto mn = [&](uint32_t p, uint32_t q) {
                    return is_signed ? b.sext2(lo, p, q) : b.uext2(lo, p, q);
                };
                auto mx = [&](uint32_t p, uint32_t q) {
                    return is_signed ? b.sext2(hi, p, q) : b.uext2(hi, p, q);
                };
                return mx(mn(x, y), mx(mn(x, z), mn(y, z)));
            };
            uint32_t r16;
            switch (in.opcode) {
                case 0x340:
                case 0x35E: r16 = b.ibin(Op_IAdd, b.ibin(Op_IMul, s0, s1), s2()); break;
                case 0x353: r16 = b.uext2(Glsl_UMin, b.uext2(Glsl_UMin, s0, s1), s2()); break;
                case 0x356: r16 = b.uext2(Glsl_UMax, b.uext2(Glsl_UMax, s0, s1), s2()); break;
                case 0x352:
                    r16 = b.sext2(Glsl_SMin, b.sext2(Glsl_SMin, s16(s0), s16(s1)), s16(s2()));
                    break;
                case 0x355:
                    r16 = b.sext2(Glsl_SMax, b.sext2(Glsl_SMax, s16(s0), s16(s1)), s16(s2()));
                    break;
                case 0x359: r16 = med3(s0, s1, s2(), false); break;
                case 0x358: r16 = med3(s16(s0), s16(s1), s16(s2()), true); break;
                case 0x303:
                case 0x30D: r16 = b.ibin(Op_IAdd, s0, s1); break;
                case 0x304:
                case 0x30E: r16 = b.ibin(Op_ISub, s0, s1); break;
                case 0x305: r16 = b.ibin(Op_IMul, s0, s1); break;
                case 0x309: r16 = b.uext2(Glsl_UMax, s0, s1); break;
                case 0x30B: r16 = b.uext2(Glsl_UMin, s0, s1); break;
                case 0x30A: r16 = b.sext2(Glsl_SMax, s16(s0), s16(s1)); break;
                case 0x30C: r16 = b.sext2(Glsl_SMin, s16(s0), s16(s1)); break;
                case 0x314:
                    r16 =
                        b.ibin(Op_ShiftLeftLogical, s1, b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu)));
                    break;
                case 0x307:
                    r16 =
                        b.ibin(Op_ShiftRightLogical, s1, b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu)));
                    break;
                default:
                    r16 = b.sbin(Op_ShiftRightArithmetic, s16(s1),
                                 b.ibin(Op_BitwiseAnd, s0, b.uconst(0xFu)));
                    break;   // 0x308 v_ashrrev_i16
            }
            r16 = b.ibin(Op_BitwiseAnd, r16, b.uconst(0xFFFFu));
            vreg[in.dst.value] =
                (in.vop3p_opsel & 8u)
                    ? b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0x0000FFFFu)),
                             b.ibin(Op_ShiftLeftLogical, r16, b.uconst(16)))
                    : b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, old_d, b.uconst(0xFFFF0000u)),
                             r16);
        }
    } else {
        return false;
    }
    return true;
}
bool lower_vop3_integer16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, uint32_t old_d) {
    if (in.fmt != Rdna2Format::VOP3) return false;
    return Vop3Integer16Lowering({b, rs, in, ok, allow_wave}, old_d).lower();
}
}   // namespace prosper::gpu::ir

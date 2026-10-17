// vopc_float64.cpp — operand bookkeeping for the V_CMP(X)_*_F64 compares (ops/vector_compare_f64).
//
// The relation is the definition's; what reaches it is decided here: whether the instruction may
// be emitted at all (device gate, MODE), and each source's two words. Every case this file cannot
// settle refuses by name rather than guessing (HOW_TO_ADD_A_FAMILY.md, "F64 values").
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/lower/launch_float_mode.hpp"
#include "gpu/recompiler/ir/lower/vopc_lowering_internal.hpp"

#include <bit>

namespace prosper::gpu::ir {

namespace {
// AMD 70648 Table 20: inline floats 240-248 are "single, double, or half-precision"; in a
// binary64 operand they are these doubles (1/(2*pi) is 0x3fc45f306dc9c882). CONFIDENCE: HIGH.
uint64_t inline_float_f64_bits(int code) {
    constexpr double kValues[] = {0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0};
    if (code >= 240 && code <= 247) return std::bit_cast<uint64_t>(kValues[code - 240]);
    return 0x3fc45f306dc9c882ull;   // 248
}
} // namespace

void VopcLowering::refuse(const char* reason) {
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = reason;
    ok = false;
}

bool VopcLowering::float64_admission() {
    // SDWA and DPP select sub-dword lanes of 32-bit operands; neither describes a register pair.
    if (in.has_sdwa || in.has_dpp) {
        refuse("f64-compare-sdwa-dpp");
        return false;
    }
    // F and TRU read no value, so they need neither the device feature nor the MODE.
    if (eff == 0x20u || eff == 0x2Fu) return true;
    if (!float64_admitted()) {
        refuse("f64-device-unavailable");
        return false;
    }
    // AMD 70648 6.4, Table 24: FP_DENORM[7:6] is the double/half denormal mode; bit 6 set
    // (modes 1 and 3) allows input denormals, clear (0 and 2) flushes them. A compare has no
    // floating result, so the output half and FP_ROUND cannot affect it. CONFIDENCE: MED that the
    // input flush applies to V_CMP sources; the F32 EQ/NEQ zero relation assumes the same.
    const LaunchFloatMode launch = launch_float_mode(b);
    if (!launch.mode_available) {
        refuse("f64-float-mode-unavailable");
        return false;
    }
    float64_flush_inputs = (launch.mode & 0x40u) == 0;
    return true;
}

VopcLowering::Integer64Halves VopcLowering::float64_source(int k, uint32_t raw) {
    const Operand& operand = in.src[k];
    Integer64Halves h{raw, b.uconst(0)};
    if (operand.kind == OperandKind::VGPR || operand.kind == OperandKind::SGPR ||
        (operand.kind == OperandKind::Special && operand.value >= 106 && operand.value <= 123)) {
        Operand high = operand;
        ++high.value;
        h.hi = val(high);
    } else if (operand.kind == OperandKind::InlineFloat) {
        const uint64_t bits = inline_float_f64_bits(operand.value);
        h = {b.uconst(static_cast<uint32_t>(bits)), b.uconst(static_cast<uint32_t>(bits >> 32))};
    } else if (operand.kind == OperandKind::Literal) {
        // AMD 70648 6.2.1, "Literal Expansion to 64 bits": for a 64-bit float the literal is the
        // HIGH word and the low 32 bits are zero. CONFIDENCE: HIGH.
        h = {b.uconst(0), raw};
    } else if ((operand.kind == OperandKind::InlineInt && operand.value == 0) ||
               (operand.kind == OperandKind::Special && operand.value == 125)) {
        h = {b.uconst(0), b.uconst(0)};   // integer 0 and SGPR_NULL: +0.0 under any reading
    } else {
        // A nonzero inline integer in a binary64 operand: the ISA does not say whether it is the
        // sign-extended bit pattern or the converted value. Refuse rather than pick one.
        refuse(operand.kind == OperandKind::InlineInt ? "f64-inline-integer-unspecified"
                                                      : "f64-source-unsupported");
        return h;
    }
    // ABS then NEG, on the sign bit of the high word (hardware order, as for F32).
    if (in.src_abs[k]) h.hi = b.ibin(Op_BitwiseAnd, h.hi, b.uconst(0x7fffffffu));
    if (in.src_neg[k]) h.hi = b.ibin(Op_BitwiseXor, h.hi, b.uconst(0x80000000u));
    if (float64_flush_inputs) {
        // A zero exponent field is a zero or a subnormal; both become the signed zero.
        const uint32_t exponent = b.ibin(Op_BitwiseAnd, h.hi, b.uconst(0x7ff00000u));
        const uint32_t subnormal_or_zero = b.ucmp(Op_IEqual, exponent, b.uconst(0));
        h.hi = b.sel(subnormal_or_zero, b.ibin(Op_BitwiseAnd, h.hi, b.uconst(0x80000000u)), h.hi);
        h.lo = b.sel(subnormal_or_zero, b.uconst(0), h.lo);
    }
    return h;
}

} // namespace prosper::gpu::ir

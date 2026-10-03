#include "gpu/recompiler/fragment_packet_services.hpp"

namespace prosper::gpu {
namespace {
struct IntegerF32 {
    SpirvCompute& b;
    uint32_t op64(uint32_t op, uint32_t a, uint32_t c) {
        const auto r = b.id(); b.put(b.code, op, {b.t_u64(), r, a, c}); return r;
    }
    uint32_t cmp64(uint32_t op, uint32_t a, uint32_t c) {
        const auto r = b.id(); b.put(b.code, op, {b.t_bool, r, a, c}); return r;
    }
    uint32_t wide(uint32_t a) {
        const auto r = b.id(); b.put(b.code, Op_UConvert, {b.t_u64(), r, a}); return r;
    }
    uint32_t sh(uint32_t op, uint32_t a, uint32_t n) {
        return op64(op, a, wide(b.uext2(Glsl_UMin, n, b.uconst(63))));
    }
    uint32_t and_(uint32_t a, uint32_t c) { return b.ucmp(Op_LogicalAnd, a, c); }
    uint32_t or_(uint32_t a, uint32_t c) { return b.ucmp(Op_LogicalOr, a, c); }
    uint32_t nz(uint32_t a) { return b.ucmp(Op_INotEqual, a, b.uconst(0)); }
    struct Decoded { uint32_t sign, magnitude, mantissa, exponent, finite; };
    Decoded decode(uint32_t raw, FragmentFloatMode mode) {
        const auto sign = b.ibin(Op_BitwiseAnd, raw, b.uconst(0x80000000u));
        auto mag = b.ibin(Op_BitwiseAnd, raw, b.uconst(0x7fffffffu));
        auto exp = b.ibin(Op_ShiftRightLogical, mag, b.uconst(23));
        if (!mode.preserves_f32_inputs())
            mag = b.sel(b.ucmp(Op_IEqual, exp, b.uconst(0)), b.uconst(0), mag);
        exp = b.ibin(Op_ShiftRightLogical, mag, b.uconst(23));
        const auto frac = b.ibin(Op_BitwiseAnd, mag, b.uconst(0x7fffffu));
        const auto normal = nz(exp);
        const auto sig = b.sel(normal, b.ibin(Op_BitwiseOr, frac, b.uconst(0x800000u)), frac);
        // value = significand * 2^(exponent-512); subnormals use exactly -149.
        const auto exponent = b.sel(normal, b.ibin(Op_IAdd, exp, b.uconst(362)), b.uconst(363));
        return {sign, mag, sig, exponent, b.ucmp(Op_INotEqual, exp, b.uconst(255))};
    }
    PacketF32Result pack(uint32_t n, uint32_t exponent, uint32_t sign,
                         uint32_t valid, FragmentFloatMode mode) {
        const auto zero = cmp64(Op_IEqual, n, b.uconst64(0));
        const auto hi = b.u64_hi(n), lo = b.u64_lo(n);
        // FindUMsb(0) is deliberately avoided, including on an unselected SPIR-V arm.
        const auto nonzero_hi = nz(hi);
        const auto high_bit = b.ibin(Op_IAdd, b.find_umsb(b.sel(nonzero_hi, hi, b.uconst(1))), b.uconst(32));
        const auto low_bit = b.find_umsb(b.sel(nz(lo), lo, b.uconst(1)));
        const auto msb = b.sel(nonzero_hi, high_bit, low_bit);
        const auto scale = b.ibin(Op_IAdd, exponent, msb);
        const auto normal = b.ucmp(Op_UGreaterThanEqual, scale, b.uconst(386));
        const auto normal_right = b.ucmp(Op_UGreaterThanEqual, msb, b.uconst(23));
        const auto sub_right = b.ucmp(Op_ULessThan, exponent, b.uconst(363));
        const auto right = b.bsel(normal, normal_right, sub_right);
        const auto shift = b.sel(normal,
            b.sel(normal_right, b.ibin(Op_ISub, msb, b.uconst(23)), b.ibin(Op_ISub, b.uconst(23), msb)),
            b.sel(sub_right, b.ibin(Op_ISub, b.uconst(363), exponent), b.ibin(Op_ISub, exponent, b.uconst(363))));
        const auto big_shift = b.ucmp(Op_UGreaterThanEqual, shift, b.uconst(64));
        const auto executed_shift = b.uext2(Glsl_UMin, shift, b.uconst(63));
        const auto divisor = sh(Op_ShiftLeftLogical, b.uconst64(1), executed_shift);
        const auto mask = op64(Op_ISub, divisor, b.uconst64(1));
        const auto remainder = b.sel64(big_shift, n, op64(Op_BitwiseAnd, n, mask));
        const auto lost = and_(right, cmp64(Op_INotEqual, remainder, b.uconst64(0)));
        auto truncated = b.sel64(right,
            b.sel64(big_shift, b.uconst64(0), sh(Op_ShiftRightLogical, n, shift)),
            sh(Op_ShiftLeftLogical, n, shift));
        const auto half = sh(Op_ShiftRightLogical, divisor, b.uconst(1));
        const auto halfway = cmp64(Op_IEqual, remainder, half);
        const auto greater_half = cmp64(Op_UGreaterThan, remainder, half);
        const auto odd = cmp64(Op_INotEqual, op64(Op_BitwiseAnd, truncated, b.uconst64(1)), b.uconst64(0));
        uint32_t increment;
        switch (mode.value & 3u) {
            case 0: increment = and_(lost, and_(b.logical_not(big_shift), or_(greater_half, and_(halfway, odd)))); break;
            case 1: increment = and_(lost, b.ucmp(Op_IEqual, sign, b.uconst(0))); break;
            case 2: increment = and_(lost, nz(sign)); break;
            default: increment = b.bfalse(); break;
        }
        truncated = op64(Op_IAdd, truncated, b.sel64(increment, b.uconst64(1), b.uconst64(0)));
        auto sig = b.u64_lo(truncated);
        const auto carry = and_(normal, b.ucmp(Op_UGreaterThanEqual, sig, b.uconst(0x1000000u)));
        sig = b.sel(carry, b.ibin(Op_ShiftRightLogical, sig, b.uconst(1)), sig);
        auto encoded_exp = b.ibin(Op_IAdd, b.ibin(Op_ISub, scale, b.uconst(385)), b.sel(carry, b.uconst(1), b.uconst(0)));
        const auto overflow = and_(b.logical_not(zero), and_(normal, b.ucmp(Op_UGreaterThanEqual, encoded_exp, b.uconst(255))));
        const auto raw_normal = b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, encoded_exp, b.uconst(23)),
            b.ibin(Op_BitwiseAnd, sig, b.uconst(0x7fffffu)));
        // A rounded subnormal may become the smallest normal (sig=0x800000).
        auto magnitude = b.sel(normal, raw_normal, sig);
        const auto output_subnormal = and_(nz(magnitude), b.ucmp(Op_ULessThan, magnitude, b.uconst(0x800000u)));
        const auto flushed = ((mode.value & 0x20u) == 0) ? output_subnormal : b.bfalse();
        magnitude = b.sel(or_(zero, flushed), b.uconst(0), magnitude);
        const auto exact = and_(b.logical_not(lost), b.logical_not(flushed));
        return {b.ibin(Op_BitwiseOr, sign, magnitude), and_(valid, b.logical_not(overflow)),
                exact, b.bfalse(), overflow};
    }
};
} // namespace

PacketF32Result packet_f32_mul(SpirvCompute& b, uint32_t a, uint32_t c, FragmentFloatMode mode) {
    IntegerF32 f{b}; const auto x = f.decode(a, mode), y = f.decode(c, mode);
    const auto product = f.op64(Op_IMul, f.wide(x.mantissa), f.wide(y.mantissa));
    auto result = f.pack(product, b.ibin(Op_ISub, b.ibin(Op_IAdd, x.exponent, y.exponent), b.uconst(512)),
        b.ibin(Op_BitwiseXor, x.sign, y.sign), f.and_(x.finite, y.finite), mode);
    result.nonfinite = b.logical_not(f.and_(x.finite, y.finite));
    return result;
}

PacketF32Result packet_f32_add(SpirvCompute& b, uint32_t a, uint32_t c, FragmentFloatMode mode) {
    IntegerF32 f{b}; const auto x = f.decode(a, mode), y = f.decode(c, mode);
    const auto exponent = b.uext2(Glsl_UMax, x.exponent, y.exponent);
    // Zero operands have no exponent significance. Select the real nonzero word's scale first.
    const auto scale = b.sel(b.ucmp(Op_IEqual, x.mantissa, b.uconst(0)), y.exponent,
        b.sel(b.ucmp(Op_IEqual, y.mantissa, b.uconst(0)), x.exponent, exponent));
    uint32_t discarded = b.bfalse();
    const auto align = [&](const IntegerF32::Decoded& operand) {
        const auto n = f.sh(Op_ShiftLeftLogical, f.wide(operand.mantissa), b.uconst(32));
        const auto distance = b.sel(f.nz(operand.mantissa), b.ibin(Op_ISub, scale, operand.exponent), b.uconst(0));
        const auto saturated = b.ucmp(Op_UGreaterThanEqual, distance, b.uconst(64));
        const auto mask = f.op64(Op_ISub, f.sh(Op_ShiftLeftLogical, b.uconst64(1), distance), b.uconst64(1));
        const auto remainder = b.sel64(saturated, n, f.op64(Op_BitwiseAnd, n, mask));
        const auto lost = f.cmp64(Op_INotEqual, remainder, b.uconst64(0));
        discarded = f.or_(discarded, lost);
        // A distant nonzero word MUST remain a sticky bit: +tiny and -tiny have observable
        // directed rounding. No executed shift can equal/exceed64, even on an unselected arm.
        return f.op64(Op_BitwiseOr,
            b.sel64(saturated, b.uconst64(0), f.sh(Op_ShiftRightLogical, n, distance)),
            b.sel64(lost, b.uconst64(1), b.uconst64(0)));
    };
    const auto nx = align(x), ny = align(y);
    const auto same_sign = b.ucmp(Op_IEqual, x.sign, y.sign);
    const auto greater = f.cmp64(Op_UGreaterThanEqual, nx, ny);
    const auto difference = f.op64(Op_ISub, b.sel64(greater, nx, ny), b.sel64(greater, ny, nx));
    const auto sum = b.sel64(same_sign, f.op64(Op_IAdd, nx, ny), difference);
    const auto cancel_sign = b.uconst((mode.value & 3u) == 2 ? 0x80000000u : 0u);
    const auto sign = b.sel(same_sign, x.sign,
        b.sel(f.cmp64(Op_IEqual, sum, b.uconst64(0)), cancel_sign, b.sel(greater, x.sign, y.sign)));
    // 24+32 significand bits plus one carry fit uint64. Cancellation requiring precise low
    // bits occurs only at adjacent exponents, where no alignment remainder was discarded.
    auto result = f.pack(sum, b.ibin(Op_ISub, scale, b.uconst(32)), sign,
        f.and_(x.finite, y.finite), mode);
    result.exact = f.and_(result.exact, b.logical_not(discarded));
    result.nonfinite = b.logical_not(f.and_(x.finite, y.finite));
    return result;
}
} // namespace prosper::gpu

// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t SpirvCompute::ffbh_u32(uint32_t a) {
        const uint32_t safe = ibin(Op_BitwiseOr, a, uconst(1));
        const uint32_t leading_zeroes = ibin(Op_ISub, uconst(31), find_umsb(safe));
        return sel(ucmp(Op_INotEqual, a, uconst(0)),
                   leading_zeroes, uconst(0xffffffffu));
    }

uint32_t SpirvCompute::ldexp_f32_bits(uint32_t bits, uint32_t exponent) {
        const uint32_t sign = ibin(Op_BitwiseAnd, bits, uconst(0x80000000u));
        const uint32_t magnitude = ibin(Op_BitwiseAnd, bits, uconst(0x7fffffffu));
        const uint32_t raw_exp = ibin(Op_ShiftRightLogical, magnitude, uconst(23));
        const uint32_t fraction = ibin(Op_BitwiseAnd, magnitude, uconst(0x007fffffu));

        // Normalize a subnormal input into a 24-bit significand with bit 23 set. FindUMsb is
        // undefined at zero, so substitute one for that otherwise-dead calculation; signed zero
        // is selected back unchanged at the end.
        const uint32_t fraction_zero = ucmp(Op_IEqual, fraction, uconst(0));
        const uint32_t safe_fraction = sel(fraction_zero, uconst(1), fraction);
        const uint32_t sub_shift = ibin(Op_ISub, uconst(23), find_umsb(safe_fraction));
        const uint32_t sub_significand =
            ibin(Op_ShiftLeftLogical, fraction, sub_shift); // sub_shift is in [1,23]
        const uint32_t normal_significand =
            ibin(Op_BitwiseOr, fraction, uconst(0x00800000u));
        const uint32_t input_subnormal = ucmp(Op_IEqual, raw_exp, uconst(0));
        const uint32_t significand =
            sel(input_subnormal, sub_significand, normal_significand);
        const uint32_t normal_unbiased = ibin(Op_ISub, raw_exp, uconst(127));
        const uint32_t sub_unbiased = ibin(Op_ISub, uconst(static_cast<uint32_t>(-126)),
                                           sub_shift);
        const uint32_t unbiased = sel(input_subnormal, sub_unbiased, normal_unbiased);

        const uint32_t bounded_exponent = sext2(
            Glsl_SMin,
            sext2(Glsl_SMax, exponent, uconst(static_cast<uint32_t>(-512))),
            uconst(512));
        const uint32_t adjusted = ibin(Op_IAdd, unbiased, bounded_exponent);

        const uint32_t normal_exp = ibin(
            Op_ShiftLeftLogical, ibin(Op_IAdd, adjusted, uconst(127)), uconst(23));
        const uint32_t normal_result = ibin(
            Op_BitwiseOr, normal_exp,
            ibin(Op_BitwiseAnd, significand, uconst(0x007fffffu)));

        // For adjusted<-126, shift the normalized 24-bit significand into the subnormal range.
        // The calculations exist in SSA for every input, so clamp the working shift to [1,24]
        // even when the normal/overflow result is ultimately selected. shift>=25 is strictly below
        // half the least subnormal (shift==24 retains the exact halfway tie and its RNE decision).
        const uint32_t under_shift =
            ibin(Op_ISub, uconst(static_cast<uint32_t>(-126)), adjusted);
        const uint32_t safe_under_shift = uext2(
            Glsl_UMax, uconst(1), uext2(Glsl_UMin, under_shift, uconst(24)));
        const uint32_t truncated =
            ibin(Op_ShiftRightLogical, significand, safe_under_shift);
        const uint32_t remainder_mask = ibin(
            Op_ISub, ibin(Op_ShiftLeftLogical, uconst(1), safe_under_shift), uconst(1));
        const uint32_t remainder = ibin(Op_BitwiseAnd, significand, remainder_mask);
        const uint32_t half = ibin(
            Op_ShiftLeftLogical, uconst(1), ibin(Op_ISub, safe_under_shift, uconst(1)));
        const uint32_t remainder_gt_half = ucmp(Op_UGreaterThan, remainder, half);
        const uint32_t remainder_eq_half = ucmp(Op_IEqual, remainder, half);
        const uint32_t truncated_odd = ucmp(
            Op_INotEqual, ibin(Op_BitwiseAnd, truncated, uconst(1)), uconst(0));
        const uint32_t round_up = lor(remainder_gt_half,
                                      land(remainder_eq_half, truncated_odd));
        const uint32_t rounded = ibin(
            Op_IAdd, truncated, sel(round_up, uconst(1), uconst(0)));
        const uint32_t too_small = scmp(Op_SGreaterThanEqual, under_shift, uconst(25));
        const uint32_t subnormal_result = sel(too_small, uconst(0), rounded);

        const uint32_t is_underflow =
            scmp(Op_SLessThan, adjusted, uconst(static_cast<uint32_t>(-126)));
        const uint32_t is_overflow = scmp(Op_SGreaterThan, adjusted, uconst(127));
        uint32_t finite_result = sel(is_underflow, subnormal_result, normal_result);
        finite_result = sel(is_overflow, uconst(0x7f800000u), finite_result);
        finite_result = ibin(Op_BitwiseOr, sign, finite_result);

        const uint32_t is_special = ucmp(Op_IEqual, raw_exp, uconst(255));
        const uint32_t is_zero = ucmp(Op_IEqual, magnitude, uconst(0));
        return sel(is_special, bits, sel(is_zero, bits, finite_result));
    }

uint32_t SpirvCompute::cvt_f2u(uint32_t bits) {
        uint32_t f = bcf(bits);
        uint32_t nan = id();  put(code, Op_FUnordNotEqual, {t_bool, nan, f, f});   // true iff NaN
        uint32_t safe = id(); put(code, Op_Select, {t_f32, safe, nan, fconstf(0.0f), f});
        uint32_t lo = id();   putv(code, Op_ExtInst, {t_f32, lo, glsl, Glsl_FMax, safe, fconstf(0.0f)});
        uint32_t cl = id();   putv(code, Op_ExtInst, {t_f32, cl, glsl, Glsl_FMin, lo, fconstf(4294967040.0f)});
        uint32_t r = id();    put(code, Op_ConvertFToU, {t_u32, r, cl});
        uint32_t big = id();  put(code, Op_FOrdGreaterThanEqual, {t_bool, big, f, fconstf(4294967296.0f)});
        return sel(big, uconst(0xFFFFFFFFu), r);
    }

uint32_t SpirvCompute::bsel(uint32_t cond, uint32_t tval, uint32_t fval) {
        uint32_t r = id();
        put(code, Op_Select, {t_bool, r, cond, tval, fval});
        add_fragment_wave_vote_dependency(r, cond);
        add_fragment_wave_vote_dependency(r, tval);
        add_fragment_wave_vote_dependency(r, fval);
        return r;
    }

uint32_t SpirvCompute::umul_hi(uint32_t a, uint32_t b_) {
        if (!t_u32pair) { t_u32pair = id(); put(types, Op_TypeStruct, {t_u32pair, t_u32, t_u32}); }
        uint32_t r = id(); put(code, Op_UMulExtended, {t_u32pair, r, a, b_});
        uint32_t hi = id(); put(code, Op_CompositeExtract, {t_u32, hi, r, 1}); return hi;
    }

uint32_t SpirvCompute::smul_hi(uint32_t a, uint32_t b_) {
        if (!t_i32pair) { t_i32pair = id(); put(types, Op_TypeStruct, {t_i32pair, t_i32, t_i32}); }
        uint32_t r = id(); put(code, Op_SMulExtended, {t_i32pair, r, bcs(a), bcs(b_)});
        uint32_t hi = id(); put(code, Op_CompositeExtract, {t_i32, hi, r, 1}); return i2u(hi);
    }

uint32_t SpirvCompute::cvt_f2i(uint32_t bits) {
        uint32_t f = bcf(bits);
        uint32_t nan = id();  put(code, Op_FUnordNotEqual, {t_bool, nan, f, f});   // true iff NaN
        uint32_t safe = id(); put(code, Op_Select, {t_f32, safe, nan, fconstf(0.0f), f});
        uint32_t magnitude = id();
        putv(code, Op_ExtInst, {t_f32, magnitude, glsl, Glsl_FAbs, safe});
        uint32_t bounded = id();
        putv(code, Op_ExtInst,
             {t_f32, bounded, glsl, Glsl_FMin, magnitude, fconstf(2147483648.0f)});
        uint32_t unsigned_magnitude = id();
        put(code, Op_ConvertFToU, {t_u32, unsigned_magnitude, bounded});
        uint32_t negative = id();
        put(code, Op_FOrdLessThan, {t_bool, negative, f, fconstf(0.0f)});
        uint32_t negated = ibin(Op_ISub, uconst(0), unsigned_magnitude);
        uint32_t signed_bits = sel(negative, negated, unsigned_magnitude);
        uint32_t big = id();  put(code, Op_FOrdGreaterThanEqual, {t_bool, big, f, fconstf(2147483648.0f)});
        return sel(big, uconst(0x7FFFFFFFu), signed_bits);
    }

uint32_t SpirvCompute::u64_from_lohi(uint32_t lo, uint32_t hi) {   // (u64)hi<<32 | (u64)lo  — combine an SGPR pair
        uint32_t l = id(); put(code, Op_UConvert, {t_u64(), l, lo});
        uint32_t h = id(); put(code, Op_UConvert, {t_u64(), h, hi});
        uint32_t hs = id(); put(code, Op_ShiftLeftLogical, {t_u64(), hs, h, uconst64(32)});
        uint32_t r = id(); put(code, Op_BitwiseOr, {t_u64(), r, l, hs}); return r;
    }

uint32_t SpirvCompute::u64_shift(uint32_t op, uint32_t value, uint32_t amount) {
        // Keep the shift operand u64-typed for the same cross-driver reason as uconst64 above.
        uint32_t shift = id(); put(code, Op_UConvert, {t_u64(), shift, amount});
        uint32_t result = id(); put(code, op, {t_u64(), result, value, shift}); return result;
    }

uint32_t SpirvCompute::bfe_u64(uint32_t base64, uint32_t off, uint32_t cnt) {   // 64-bit unsigned bitfield extract
        // res = (base << (64-off-cnt)) >> (64-cnt), all logical u64 (portable — OpBitFieldUExtract on a
        // 64-bit base isn't reliably supported, e.g. llvmpipe returns 0). The 7-bit width field
        // legally encodes 0 and values past the register end; SPIR-V shifts >= 64 are undefined
        // VALUES, so clamp like the 32-bit helper (#455): effective cnt = min(cnt, 64-off) (bits
        // past bit 63 read as 0), and a zero width selects the architectural result 0 explicitly.
        // Requires off <= 63 (both callers mask offset to [5:0]).
        uint32_t cnt_c = uext2(Glsl_UMin, cnt, ibin(Op_ISub, uconst(64), off));
        uint32_t total = ibin(Op_IAdd, off, cnt_c);
        uint32_t lsh32 = ibin(Op_ISub, uconst(64), total);
        uint32_t rsh32 = ibin(Op_ISub, uconst(64), cnt_c);
        uint32_t lsh = id(); put(code, Op_UConvert, {t_u64(), lsh, lsh32});
        uint32_t rsh = id(); put(code, Op_UConvert, {t_u64(), rsh, rsh32});
        uint32_t sl  = id(); put(code, Op_ShiftLeftLogical,  {t_u64(), sl, base64, lsh});
        uint32_t r   = id(); put(code, Op_ShiftRightLogical, {t_u64(), r, sl, rsh});
        uint32_t nz  = ucmp(Op_INotEqual, cnt, uconst(0));
        uint32_t rs  = id(); put(code, Op_Select, {t_u64(), rs, nz, r, uconst64(0)}); return rs; }

uint32_t SpirvCompute::u64_bit(uint32_t v64, uint32_t bit) {
        uint32_t shift = id(); put(code, Op_UConvert, {t_u64(), shift, bit});
        uint32_t shifted = id(); put(code, Op_ShiftRightLogical, {t_u64(), shifted, v64, shift});
        return ucmp(Op_INotEqual,
                    ibin(Op_BitwiseAnd, u64_lo(shifted), uconst(1)), uconst(0));
    }

uint32_t SpirvCompute::pack_half2x16(uint32_t a, uint32_t b) {
        uint32_t vec = id(); put(code, Op_CompositeConstruct, {t_v2f(), vec, bcf(a), bcf(b)});
        uint32_t r = id(); putv(code, Op_ExtInst, {t_u32, r, glsl, Glsl_PackHalf2x16, vec}); return r;
    }

uint32_t SpirvCompute::pack_half2x16_rtz(uint32_t a, uint32_t b) {
        // GLSL PackHalf2x16 does not guarantee RTZ, and its preceding floating-point min/max can
        // turn a NaN or true infinity into a finite endpoint. V_CVT_PKRTZ_F16_F32 always rounds
        // finite operands toward zero. Work on the source bits so host float controls and GLSL
        // exceptional-value min/max do not change the input class.
        const auto half_rtz = [&](uint32_t bits) {
            const uint32_t sign = ibin(Op_BitwiseAnd,
                                       ibin(Op_ShiftRightLogical, bits, uconst(16)),
                                       uconst(0x8000u));
            const uint32_t exponent = ibin(Op_BitwiseAnd,
                                           ibin(Op_ShiftRightLogical, bits, uconst(23)),
                                           uconst(0xffu));
            const uint32_t fraction = ibin(Op_BitwiseAnd, bits, uconst(0x7fffffu));

            const uint32_t normal_exponent = ibin(
                Op_ShiftLeftLogical, ibin(Op_ISub, exponent, uconst(112)), uconst(10));
            const uint32_t normal_fraction = ibin(Op_ShiftRightLogical, fraction, uconst(13));
            const uint32_t normal = ibin(Op_BitwiseOr, normal_exponent, normal_fraction);

            // For f32 exponents 103..112, the hidden bit contributes to a half subnormal.
            // The shift expression is evaluated in SSA for every input, including exponents
            // outside that range. Bound it before OpShiftRightLogical (shift >= 32 is undefined).
            const uint32_t sub_shift = uext2(
                Glsl_UMin, ibin(Op_ISub, uconst(126), exponent), uconst(31));
            const uint32_t subnormal = ibin(
                Op_ShiftRightLogical,
                ibin(Op_BitwiseOr, fraction, uconst(0x800000u)), sub_shift);
            const uint32_t has_subnormal = land(
                ucmp(Op_UGreaterThanEqual, exponent, uconst(103)),
                ucmp(Op_ULessThan, exponent, uconst(113)));
            uint32_t magnitude = sel(has_subnormal, subnormal, uconst(0));
            const uint32_t has_normal = land(
                ucmp(Op_UGreaterThanEqual, exponent, uconst(113)),
                ucmp(Op_ULessThan, exponent, uconst(143)));
            magnitude = sel(has_normal, normal, magnitude);
            const uint32_t finite_overflow = land(
                ucmp(Op_UGreaterThanEqual, exponent, uconst(143)),
                ucmp(Op_ULessThan, exponent, uconst(255)));
            magnitude = sel(finite_overflow, uconst(0x7bffu), magnitude);
            // True infinities stay infinite; NaN payloads become a signed quiet half NaN.
            const uint32_t special = sel(
                ucmp(Op_INotEqual, fraction, uconst(0)),
                uconst(0x7e00u), uconst(0x7c00u));
            magnitude = sel(ucmp(Op_IEqual, exponent, uconst(255)), special, magnitude);
            return ibin(Op_BitwiseOr, sign, magnitude);
        };
        return ibin(Op_BitwiseOr, half_rtz(a),
                    ibin(Op_ShiftLeftLogical, half_rtz(b), uconst(16)));
    }

uint32_t SpirvCompute::unpack_norm(uint32_t dword, uint32_t bit_off, uint32_t bits, bool is_signed, float norm) {
        uint32_t fbits_ = is_signed ? cvt_i2f(bfe_s(dword, uconst(bit_off), uconst(bits)))
                                    : cvt_u2f(bfe_u(dword, uconst(bit_off), uconst(bits)));
        uint32_t v = fbin(Op_FDiv, fbits_, bcu(fconstf(norm)));
        if (is_signed) v = fext2(Glsl_FMax, v, bcu(fconstf(-1.0f)));
        return v;
    }

uint32_t SpirvCompute::unpack_half(uint32_t dword, uint32_t which) {
        uint32_t vec = id(); putv(code, Op_ExtInst, {t_v2f(), vec, glsl, Glsl_UnpackHalf2x16, dword});
        uint32_t f = id(); put(code, Op_CompositeExtract, {t_f32, f, vec, which}); return bcu(f);
    }

uint32_t SpirvCompute::unpack_ufloat(uint32_t dword, uint32_t bit_off, uint32_t bits) {
        uint32_t raw = bfe_u(dword, uconst(bit_off), uconst(bits));
        uint32_t half = ibin(Op_ShiftLeftLogical, raw, uconst(bits == 11 ? 4u : 5u));
        return unpack_half(half, 0);
    }

uint32_t SpirvCompute::round_shift_even_u32(uint32_t value, uint32_t shift) {
        const uint32_t safe_shift = uext2(Glsl_UMin, shift, uconst(31));
        const uint32_t nonzero_shift = uext2(Glsl_UMax, safe_shift, uconst(1));
        const uint32_t rounded = ibin(Op_ShiftRightLogical, value, nonzero_shift);
        const uint32_t one_shifted = ibin(Op_ShiftLeftLogical, uconst(1), nonzero_shift);
        const uint32_t remainder = ibin(
            Op_BitwiseAnd, value, ibin(Op_ISub, one_shifted, uconst(1)));
        const uint32_t halfway_shift = ibin(Op_ISub, nonzero_shift, uconst(1));
        const uint32_t halfway = ibin(Op_ShiftLeftLogical, uconst(1), halfway_shift);
        const uint32_t above = ucmp(Op_UGreaterThan, remainder, halfway);
        const uint32_t tie = land(
            ucmp(Op_IEqual, remainder, halfway),
            ucmp(Op_INotEqual, ibin(Op_BitwiseAnd, rounded, uconst(1)), uconst(0)));
        const uint32_t incremented = ibin(Op_IAdd, rounded, sel(lor(above, tie), uconst(1), uconst(0)));
        const uint32_t finite = sel(ucmp(Op_IEqual, shift, uconst(0)), value, incremented);
        return sel(ucmp(Op_UGreaterThanEqual, shift, uconst(32)), uconst(0), finite);
    }

uint32_t SpirvCompute::pack_ufloat(uint32_t bits, uint32_t mantissa_bits) {
        const uint32_t exponent = bfe_u(bits, uconst(23), uconst(8));
        const uint32_t mantissa = ibin(Op_BitwiseAnd, bits, uconst(0x7fffff));
        const uint32_t sign = ibin(Op_ShiftRightLogical, bits, uconst(31));
        const uint32_t exponent_bits = uconst(0x1fu << mantissa_bits);
        const uint32_t mantissa_mask = uconst((1u << mantissa_bits) - 1u);

        const uint32_t payload_shifted = ibin(
            Op_ShiftRightLogical, mantissa, uconst(23u - mantissa_bits));
        const uint32_t payload = sel(
            ucmp(Op_IEqual, payload_shifted, uconst(0)), uconst(1), payload_shifted);
        const uint32_t special = sel(
            ucmp(Op_IEqual, mantissa, uconst(0)), exponent_bits,
            ibin(Op_BitwiseOr, exponent_bits, payload));

        const uint32_t significand = ibin(Op_BitwiseOr, uconst(0x800000), mantissa);
        const uint32_t sub_shift = ibin(
            Op_ISub, uconst(136u - mantissa_bits), exponent);
        const uint32_t sub_rounded = round_shift_even_u32(significand, sub_shift);
        const uint32_t subnormal = sel(
            ucmp(Op_UGreaterThanEqual, sub_rounded, uconst(1u << mantissa_bits)),
            uconst(1u << mantissa_bits), sub_rounded);

        const uint32_t normal_rounded = round_shift_even_u32(
            significand, uconst(23u - mantissa_bits));
        const uint32_t carry = ucmp(
            Op_IEqual, normal_rounded, uconst(1u << (mantissa_bits + 1u)));
        const uint32_t target_exponent = ibin(Op_ISub, exponent, uconst(112));
        const uint32_t carried_exponent = ibin(
            Op_IAdd, target_exponent, sel(carry, uconst(1), uconst(0)));
        const uint32_t carried_mantissa = sel(
            carry, uconst(0), ibin(Op_BitwiseAnd, normal_rounded, mantissa_mask));
        const uint32_t normal_finite = ibin(
            Op_BitwiseOr,
            ibin(Op_ShiftLeftLogical, carried_exponent, uconst(mantissa_bits)),
            carried_mantissa);
        const uint32_t normal = sel(
            ucmp(Op_UGreaterThanEqual, carried_exponent, uconst(31)),
            exponent_bits, normal_finite);

        const uint32_t finite = sel(
            ucmp(Op_ULessThanEqual, exponent, uconst(112)), subnormal,
            sel(ucmp(Op_UGreaterThanEqual, exponent, uconst(143)), exponent_bits, normal));
        const uint32_t invalid = lor(
            ucmp(Op_INotEqual, sign, uconst(0)), ucmp(Op_IEqual, exponent, uconst(0)));
        const uint32_t ordinary = sel(invalid, uconst(0), finite);
        // Infinity and NaN are handled before the sign clamp, matching the guest conversion.
        return sel(ucmp(Op_IEqual, exponent, uconst(0xff)), special, ordinary);
    }

uint32_t SpirvCompute::pack_norm(uint32_t fbits, uint32_t bits, bool is_signed, float norm) {
        uint32_t lo = bcu(fconstf(is_signed ? -1.0f : 0.0f)), hi = bcu(fconstf(1.0f));
        uint32_t clamped = fext2(Glsl_FMax, fext2(Glsl_FMin, fbits, hi), lo);
        uint32_t scaled  = fbin(Op_FMul, clamped, bcu(fconstf(norm)));
        uint32_t rounded = fext1(Glsl_RoundEven, scaled);
        uint32_t ival    = is_signed ? cvt_f2i(rounded) : cvt_f2u(rounded);
        uint32_t mask    = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        return ibin(Op_BitwiseAnd, ival, uconst(mask));
    }

uint32_t SpirvCompute::u64_buf_for_binding(uint32_t binding) {
        if (auto found = cbuf_u64_var.find(binding); found != cbuf_u64_var.end())
            return found->second;
        if (!t_ptr_sb_struct_u64) {
            const uint32_t runtime_array = id(), block = id();
            t_ptr_sb_struct_u64 = id();
            t_ptr_sb_u64 = id();
            put(deco, Op_Decorate, {runtime_array, Dec_ArrayStride, 8});
            put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
            put(deco, Op_Decorate, {block, Dec_Block});
            put(types, Op_TypeRuntimeArray, {runtime_array, t_u64()});
            put(types, Op_TypeStruct, {block, runtime_array});
            put(types, Op_TypePointer, {t_ptr_sb_struct_u64, SC_StorageBuffer, block});
            put(types, Op_TypePointer, {t_ptr_sb_u64, SC_StorageBuffer, t_u64()});
        }
        const uint32_t variable = id();
        put(deco, Op_Decorate, {variable, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {variable, Dec_Binding, binding});
        declare_external_storage_buffer(t_ptr_sb_struct_u64, variable);
        cbuf_u64_var[binding] = variable;
        return variable;
    }

SpirvCompute::U64PairAdd SpirvCompute::add_u64_pair_u32(uint32_t lo, uint32_t hi, uint32_t addend) {
        U64PairAdd result;
        result.lo = ibin(Op_IAdd, lo, addend);
        const uint32_t carry = ucmp(Op_ULessThan, result.lo, lo);
        result.hi = ibin(Op_IAdd, hi, sel(carry, uconst(1), uconst(0)));
        result.overflow = ucmp(Op_ULessThan, result.hi, hi);
        return result;
    }

uint32_t SpirvCompute::u64_pair_ule(uint32_t lhs_lo, uint32_t lhs_hi,
                          uint32_t rhs_lo, uint32_t rhs_hi) {
        const uint32_t high_less = ucmp(Op_ULessThan, lhs_hi, rhs_hi);
        const uint32_t high_equal = ucmp(Op_IEqual, lhs_hi, rhs_hi);
        const uint32_t low_less_equal = ucmp(Op_ULessThanEqual, lhs_lo, rhs_lo);
        return lor(high_less, land(high_equal, low_less_equal));
    }

uint32_t SpirvCompute::atomic_fminmax_bits(uint32_t resident, uint32_t value, bool is_min) {
        const uint32_t denorm_mode =
            (compute_pgm_rsrc1 >>
             prosper::agc::Pm4::COMPUTE_PGM_RSRC1_FP32_DENORM_SHIFT) &
            prosper::agc::Pm4::COMPUTE_PGM_RSRC1_FP32_DENORM_MASK;
        const bool flush_inputs = denorm_mode == 0u || denorm_mode == 2u;
        auto flush_subnormal = [&](uint32_t bits) {
            const uint32_t absolute = ibin(Op_BitwiseAnd, bits, uconst(0x7fffffffu));
            const uint32_t nonzero = ucmp(Op_INotEqual, absolute, uconst(0));
            const uint32_t below_normal = ucmp(Op_ULessThan, absolute, uconst(0x00800000u));
            const uint32_t subnormal = land(nonzero, below_normal);
            const uint32_t signed_zero = ibin(Op_BitwiseAnd, bits, uconst(0x80000000u));
            return sel(subnormal, signed_zero, bits);
        };
        const uint32_t resident_compare = flush_inputs ? flush_subnormal(resident) : resident;
        const uint32_t value_compare = flush_inputs ? flush_subnormal(value) : value;
        auto ordered_key = [&](uint32_t bits) {
            const uint32_t negative = ucmp(
                Op_INotEqual, ibin(Op_BitwiseAnd, bits, uconst(0x80000000u)), uconst(0));
            return sel(negative, iun(Op_Not, bits),
                       ibin(Op_BitwiseXor, bits, uconst(0x80000000u)));
        };
        const uint32_t resident_abs = ibin(
            Op_BitwiseAnd, resident, uconst(0x7fffffffu));
        const uint32_t value_abs = ibin(
            Op_BitwiseAnd, value, uconst(0x7fffffffu));
        const uint32_t resident_nan = ucmp(
            Op_UGreaterThan, resident_abs, uconst(0x7f800000u));
        const uint32_t value_nan = ucmp(
            Op_UGreaterThan, value_abs, uconst(0x7f800000u));
        const uint32_t resident_snan = land(
            resident_nan,
            ucmp(Op_IEqual,
                 ibin(Op_BitwiseAnd, resident, uconst(0x00400000u)), uconst(0)));
        const uint32_t value_snan = land(
            value_nan,
            ucmp(Op_IEqual,
                 ibin(Op_BitwiseAnd, value, uconst(0x00400000u)), uconst(0)));
        const uint32_t ordered = ucmp(
            is_min ? Op_ULessThan : Op_UGreaterThan,
            ordered_key(value_compare), ordered_key(resident_compare));
        const uint32_t numeric = sel(ordered, value, resident);
        const uint32_t resident_number = sel(value_nan, resident, numeric);
        const uint32_t quiet_resident = ibin(
            Op_BitwiseOr, resident, uconst(0x00400000u));
        const uint32_t quiet_value = ibin(
            Op_BitwiseOr, value, uconst(0x00400000u));
        return sel(
            resident_snan, quiet_resident,
            sel(value_snan, quiet_value,
                sel(resident_nan, sel(value_nan, quiet_resident, value), resident_number)));
    }

}  // namespace prosper::gpu

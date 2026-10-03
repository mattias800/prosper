#include "gpu/recompiler/fragment_packet_services.hpp"

namespace prosper::gpu {
namespace {
// All numerical intermediates are integers. No host/driver FP rounding, denormal, contraction,
// or approximation-unit behavior is authority. Invalid inputs still execute defined integer
// instructions so physical participants cannot abandon the logical-wave rendezvous.
struct SpecialF32 {
    SpirvCompute& b;
    uint32_t op(uint32_t opcode, uint32_t x, uint32_t y) {
        const auto r = b.id();
        b.put(b.code, opcode, {b.t_u64(), r, x, y});
        return r;
    }
    uint32_t cmp(uint32_t opcode, uint32_t x, uint32_t y) {
        const auto r = b.id();
        b.put(b.code, opcode, {b.t_bool, r, x, y});
        return r;
    }
    uint32_t wide(uint32_t x) {
        const auto r = b.id();
        b.put(b.code, Op_UConvert, {b.t_u64(), r, x});
        return r;
    }
    uint32_t both(uint32_t x, uint32_t y) { return b.ucmp(Op_LogicalAnd, x, y); }
    uint32_t either(uint32_t x, uint32_t y) { return b.ucmp(Op_LogicalOr, x, y); }
    uint32_t nz(uint32_t x) { return b.ucmp(Op_INotEqual, x, b.uconst(0)); }
    uint32_t increment(uint32_t lost, uint32_t above, uint32_t tie, uint32_t odd, uint32_t sign,
                       FragmentFloatMode mode) {
        switch (mode.value & 3u) {
            case 0: return both(lost, either(above, both(tie, odd)));
            case 1: return both(lost, b.ucmp(Op_IEqual, sign, b.uconst(0)));
            case 2: return both(lost, nz(sign));
            default: return b.bfalse();
        }
    }
    uint32_t rounded_quotient(uint32_t numerator, uint32_t denominator, uint32_t sign,
                              FragmentFloatMode mode) {
        const auto q = op(Op_UDiv, numerator, denominator);
        const auto r = op(Op_UMod, numerator, denominator);
        const auto twice = op(Op_IAdd, r, r);
        const auto up = increment(
            cmp(Op_INotEqual, r, b.uconst64(0)), cmp(Op_UGreaterThan, twice, denominator),
            cmp(Op_IEqual, twice, denominator),
            cmp(Op_INotEqual, op(Op_BitwiseAnd, q, b.uconst64(1)), b.uconst64(0)), sign, mode);
        return b.u64_lo(op(Op_IAdd, q, b.sel64(up, b.uconst64(1), b.uconst64(0))));
    }
    uint32_t root_floor(uint32_t n) {
        // Trial-bit integer square root, independently checked against rational binary searches
        // in the test oracle. N<2^48; every trial squared fits in uint64, including unused arms.
        auto q = b.uconst64(0);
        for (int bit = 23; bit >= 0; --bit) {
            const auto trial = op(Op_BitwiseOr, q, b.uconst64(uint64_t{1} << bit));
            q = b.sel64(cmp(Op_ULessThanEqual, op(Op_IMul, trial, trial), n), trial, q);
        }
        return q;
    }
    uint32_t pack_normal(uint32_t sig, uint32_t exponent) {
        const auto carry = b.ucmp(Op_UGreaterThanEqual, sig, b.uconst(0x1000000));
        const auto fraction = b.sel(carry, b.ibin(Op_ShiftRightLogical, sig, b.uconst(1)), sig);
        const auto e = b.ibin(Op_IAdd, exponent, b.sel(carry, b.uconst(1), b.uconst(0)));
        return b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, e, b.uconst(23)),
                      b.ibin(Op_BitwiseAnd, fraction, b.uconst(0x7fffff)));
    }
};
}   // namespace

bool packet_special_f32_opcode(uint32_t opcode) {
    return opcode == 0x2a || opcode == 0x2e || opcode == 0x33;
}

PacketF32Result packet_f32_special(SpirvCompute& b, uint32_t raw, uint32_t opcode,
                                   FragmentFloatMode mode) {
    SpecialF32 f{b};
    const auto sign = b.ibin(Op_BitwiseAnd, raw, b.uconst(0x80000000));
    const auto magnitude = b.ibin(Op_BitwiseAnd, raw, b.uconst(0x7fffffff));
    const auto e = b.ibin(Op_ShiftRightLogical, magnitude, b.uconst(23));
    const auto frac = b.ibin(Op_BitwiseAnd, raw, b.uconst(0x7fffff));
    const auto zero = b.ucmp(Op_IEqual, e, b.uconst(0));   // these opcodes ALWAYS flush input
    const auto infinity = b.ucmp(Op_IEqual, magnitude, b.uconst(0x7f800000));
    const auto nan = f.both(b.ucmp(Op_IEqual, e, b.uconst(255)), f.nz(frac));
    const auto normal = f.both(f.nz(e), b.ucmp(Op_ULessThan, e, b.uconst(255)));
    // Substitute safe *internal* arithmetic operands only. Any unresolved observable guest input
    // produces sticky failure and the production decoder discards ALL EXP records, not zeros.
    const auto safe_e = b.sel(normal, e, b.uconst(127));
    const auto m =
        b.sel(normal, b.ibin(Op_BitwiseOr, frac, b.uconst(0x800000)), b.uconst(0x800000));
    const auto denominator = f.wide(m);
    uint32_t result;
    if (opcode == 0x2a) {
        const auto q = f.rounded_quotient(b.uconst64(uint64_t{1} << 47), denominator, sign, mode);
        const auto packed = f.pack_normal(q, b.ibin(Op_ISub, b.uconst(253), safe_e));
        // Our software contract rounds in subnormal units BEFORE opcode-specific output flushing.
        // At E=253 upward rounding can cross back to the smallest normal. This interprets the
        // published MODE/IEEE operation then result flush; it is not a measured AMD-unit result.
        const auto sub_numerator =
            b.sel64(b.ucmp(Op_IEqual, safe_e, b.uconst(253)), b.uconst64(uint64_t{1} << 46),
                    b.uconst64(uint64_t{1} << 45));
        const auto sub_q = f.rounded_quotient(sub_numerator, denominator, sign, mode);
        const auto sub_packed = b.sel(b.ucmp(Op_UGreaterThanEqual, sub_q, b.uconst(0x800000)),
                                      b.uconst(0x800000), b.uconst(0));
        result = b.sel(b.ucmp(Op_UGreaterThanEqual, safe_e, b.uconst(253)), sub_packed, packed);
        result = b.sel(zero, b.uconst(0x7f800000), b.sel(infinity, b.uconst(0), result));
        result = b.ibin(Op_BitwiseOr, sign, result);
    } else {
        // e+129 represents the unbiased exponent with an even +256 bias, so both parity and
        // floor-half remain exact for NEGATIVE exponents without signed division ambiguity.
        const auto biased = b.ibin(Op_IAdd, safe_e, b.uconst(129));
        const auto parity = b.ibin(Op_BitwiseAnd, biased, b.uconst(1));
        const auto half = b.ibin(Op_ShiftRightLogical, biased, b.uconst(1));
        uint32_t q, lost, above, tie, exponent;
        if (opcode == 0x33) {
            const auto n = f.op(Op_ShiftLeftLogical, denominator,
                                f.wide(b.ibin(Op_IAdd, b.uconst(23), parity)));
            q = f.root_floor(n);
            lost = f.cmp(Op_INotEqual, f.op(Op_IMul, q, q), n);
            const auto middle = f.op(Op_IAdd, f.op(Op_IAdd, q, q), b.uconst64(1));
            const auto four_n = f.op(Op_ShiftLeftLogical, n, b.uconst64(2));
            const auto square = f.op(Op_IMul, middle, middle);
            above = f.cmp(Op_UGreaterThan, four_n, square);
            tie = f.cmp(Op_IEqual, four_n, square);
            exponent = b.ibin(Op_ISub, half, b.uconst(1));
        } else {
            // DIRECT reciprocal square root: normalized q^2 = 2^(71-parity)/M.
            // Divide the 71/70-bit numerator in two exact chunks; never overflow uint64 and
            // never round sqrt first. Q=floor(A/M), R=A%M are exact mathematical integers.
            const auto high =
                b.sel64(f.nz(parity), b.uconst64(uint64_t{1} << 46), b.uconst64(uint64_t{1} << 47));
            const auto tail =
                f.op(Op_ShiftLeftLogical, f.op(Op_UMod, high, denominator), b.uconst64(24));
            const auto quotient =
                f.op(Op_IAdd,
                     f.op(Op_ShiftLeftLogical, f.op(Op_UDiv, high, denominator), b.uconst64(24)),
                     f.op(Op_UDiv, tail, denominator));
            const auto remainder = f.op(Op_UMod, tail, denominator);
            // The boundary M=2^23, parity=0 has q=2^24, outside the trial-bit floor domain.
            const auto boundary = f.cmp(Op_IEqual, quotient, b.uconst64(uint64_t{1} << 48));
            q = b.sel64(boundary, b.uconst64(1u << 24), f.root_floor(quotient));
            const auto delta = f.op(Op_ISub, quotient, f.op(Op_IMul, q, q));
            lost = f.either(f.cmp(Op_INotEqual, delta, b.uconst64(0)),
                            f.cmp(Op_INotEqual, remainder, b.uconst64(0)));
            // (q+.5)^2=q^2+q+.25. Compare the exact remainder when delta==q;
            // this is the load-bearing rational midpoint, not sqrt(Q)'s rounded midpoint.
            const auto at_q = f.cmp(Op_IEqual, delta, q);
            const auto four_r = f.op(Op_ShiftLeftLogical, remainder, b.uconst64(2));
            above = f.either(f.cmp(Op_UGreaterThan, delta, q),
                             f.both(at_q, f.cmp(Op_UGreaterThan, four_r, denominator)));
            tie = f.both(at_q, f.cmp(Op_IEqual, four_r, denominator));
            exponent = b.ibin(Op_ISub, b.uconst(254), half);
        }
        const auto up =
            f.increment(lost, above, tie,
                        f.cmp(Op_INotEqual, f.op(Op_BitwiseAnd, q, b.uconst64(1)), b.uconst64(0)),
                        b.uconst(0), mode);
        const auto rounded = b.u64_lo(f.op(Op_IAdd, q, b.sel64(up, b.uconst64(1), b.uconst64(0))));
        result = f.pack_normal(rounded, exponent);
        result =
            b.sel(zero, opcode == 0x33 ? sign : b.ibin(Op_BitwiseOr, sign, b.uconst(0x7f800000)),
                  b.sel(infinity, b.uconst(opcode == 0x33 ? 0x7f800000 : 0), result));
    }
    const auto negative_root =
        opcode == 0x2a ? b.bfalse() : f.both(f.nz(sign), b.logical_not(zero));
    const auto unresolved = f.either(nan, negative_root);
    return {result, b.logical_not(unresolved), b.bfalse(), unresolved, b.bfalse()};
}
}   // namespace prosper::gpu

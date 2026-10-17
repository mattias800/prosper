// A finite half is sign * significand * 2^(shift-24), with significand <= 2047
// and shift <= 29. Thus (a*b+c)*2^48 has fewer than 81 magnitude bits.
// Three U32 limbs retain every bit before the sole round-to-nearest-even operation.
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/ops/half_fma_finite.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <array>
#include <cstddef>
namespace prosper::gpu::ir {
namespace {
using Wide = std::array<Value, 3>;
struct Half {
    Value sig, shift, sign, finite;
};
class Kernel {
public:
    explicit Kernel(Builder& builder) : ir(builder) {}
    Half decode(Value raw) {
        const auto exp = ir.and_(ir.lshr(raw, ir.c32(10)), ir.c32(31));
        const auto normal = ir.ine(exp, ir.c32(0));
        return {ir.or_(ir.and_(raw, ir.c32(1023)), ir.select(normal, ir.c32(1024), ir.c32(0))),
                ir.select(normal, ir.isub(exp, ir.c32(1)), ir.c32(0)),
                ir.and_(ir.lshr(raw, ir.c32(15)), ir.c32(1)), ir.ine(exp, ir.c32(31))};
    }
    // The decoded shifts, including exceptional inputs, are <= 60. Every SPIR-V
    // shift count is masked even on an unselected path (OpSelect is not short circuit).
    Wide shifted(Value word, Value shift) {
        const auto bit = ir.and_(shift, ir.c32(31));
        const auto lo = ir.shl(word, bit);
        const auto hi = ir.select(ir.ieq(bit, ir.c32(0)), ir.c32(0),
                                  ir.lshr(word, ir.and_(ir.isub(ir.c32(32), bit), ir.c32(31))));
        const auto first = ir.ult(shift, ir.c32(32));
        return {ir.select(first, lo, ir.c32(0)), ir.select(first, hi, lo),
                ir.select(first, ir.c32(0), hi)};
    }
    Value less(const Wide& a, const Wide& b) {
        return ir.boolor_(
            ir.ult(a[2], b[2]),
            ir.booland_(ir.ieq(a[2], b[2]),
                        ir.boolor_(ir.ult(a[1], b[1]),
                                   ir.booland_(ir.ieq(a[1], b[1]), ir.ult(a[0], b[0])))));
    }
    Wide add(const Wide& a, const Wide& b) {
        Wide out;
        Value carry = ir.c32(0);
        for (size_t k = 0; k < 3; ++k) {
            const auto sum = ir.iadd(a[k], b[k]);
            out[k] = ir.iadd(sum, carry);
            carry =
                ir.select(ir.boolor_(ir.ult(sum, a[k]), ir.ult(out[k], sum)), ir.c32(1), ir.c32(0));
        }
        return out;
    }
    Wide subtract(const Wide& a, const Wide& b) {
        Wide out;
        Value borrow = ir.c32(0);
        for (size_t k = 0; k < 3; ++k) {
            const auto diff = ir.isub(a[k], b[k]);
            out[k] = ir.isub(diff, borrow);
            borrow = ir.select(ir.boolor_(ir.ult(a[k], b[k]), ir.ult(diff, borrow)), ir.c32(1),
                               ir.c32(0));
        }
        return out;
    }
    Wide choose(Value condition, const Wide& yes, const Wide& no) {
        return {ir.select(condition, yes[0], no[0]), ir.select(condition, yes[1], no[1]),
                ir.select(condition, yes[2], no[2])};
    }
    Value nonzero(const Wide& x) { return ir.ine(ir.or_(ir.or_(x[0], x[1]), x[2]), ir.c32(0)); }
    Value top(const Wide& x) {
        // FfbhU32 is count-leading-zero, not the index of the high bit. For a zero
        // word the subtraction wraps, but never becomes an unbounded shift count.
        const auto low = ir.isub(ir.c32(31), ir.ffbh_u32(x[0]));
        const auto mid = ir.isub(ir.c32(63), ir.ffbh_u32(x[1]));
        const auto high = ir.isub(ir.c32(95), ir.ffbh_u32(x[2]));
        return ir.select(ir.ine(x[2], ir.c32(0)), high,
                         ir.select(ir.ine(x[1], ir.c32(0)), mid,
                                   ir.select(ir.ine(x[0], ir.c32(0)), low, ir.c32(0))));
    }
    Value right_low(const Wide& x, Value count) {
        const auto word = ir.lshr(count, ir.c32(5));
        const auto bit = ir.and_(count, ir.c32(31));
        const auto current = ir.select(ir.ieq(word, ir.c32(0)), x[0],
                                       ir.select(ir.ieq(word, ir.c32(1)), x[1], x[2]));
        const auto next = ir.select(ir.ieq(word, ir.c32(0)), x[1],
                                    ir.select(ir.ieq(word, ir.c32(1)), x[2], ir.c32(0)));
        const auto carry = ir.select(ir.ieq(bit, ir.c32(0)), ir.c32(0),
                                     ir.shl(next, ir.and_(ir.isub(ir.c32(32), bit), ir.c32(31))));
        return ir.or_(ir.lshr(current, bit), carry);
    }
    Value any_low(const Wide& x, Value count) {
        auto bits = ir.c32(0);
        for (uint32_t k = 0; k < 3; ++k) {
            const auto part = ir.isub(count, ir.c32(32 * k));
            const auto mask = ir.select(
                ir.uge(count, ir.c32(32 * (k + 1))), ir.c32(0xffffffffu),
                ir.select(ir.ugt(count, ir.c32(32 * k)),
                          ir.isub(ir.shl(ir.c32(1), ir.and_(part, ir.c32(31))), ir.c32(1)),
                          ir.c32(0)));
            bits = ir.or_(bits, ir.and_(x[k], mask));
        }
        return ir.ine(bits, ir.c32(0));
    }
    HalfFmaFiniteResult finish(const Wide& magnitude, Value sign, Value finite) {
        const auto high = top(magnitude);
        const auto shift =
            ir.select(ir.ugt(high, ir.c32(34)), ir.isub(high, ir.c32(10)), ir.c32(24));
        auto retained = right_low(magnitude, shift);
        const auto guard_position = ir.isub(shift, ir.c32(1));
        const auto guard =
            ir.ine(ir.and_(right_low(magnitude, guard_position), ir.c32(1)), ir.c32(0));
        const auto odd = ir.ine(ir.and_(retained, ir.c32(1)), ir.c32(0));
        const auto increment =
            ir.booland_(guard, ir.boolor_(odd, any_low(magnitude, guard_position)));
        retained = ir.iadd(retained, ir.select(increment, ir.c32(1), ir.c32(0)));
        const auto carry = ir.ieq(retained, ir.c32(2048));
        const auto rounded = ir.select(carry, ir.lshr(retained, ir.c32(1)), retained);
        const auto rounded_shift = ir.iadd(shift, ir.select(carry, ir.c32(1), ir.c32(0)));
        const auto exponent = ir.isub(rounded_shift, ir.c32(23));
        const auto subnormal =
            ir.booland_(ir.ieq(rounded_shift, ir.c32(24)), ir.ult(rounded, ir.c32(1024)));
        const auto normal_bits =
            ir.or_(ir.shl(exponent, ir.c32(10)), ir.and_(rounded, ir.c32(1023)));
        const auto magnitude_bits = ir.select(ir.uge(exponent, ir.c32(31)), ir.c32(0x7c00),
                                              ir.select(subnormal, rounded, normal_bits));
        const auto eligible =
            ir.booland_(finite, ir.booland_(nonzero(magnitude),
                                            ir.booland_(ir.ine(magnitude_bits, ir.c32(0)),
                                                        ir.ule(magnitude_bits, ir.c32(0x7bff)))));
        return {ir.or_(ir.shl(sign, ir.c32(15)), magnitude_bits), eligible};
    }
    Builder& ir;
};
}   // namespace
HalfFmaFiniteResult build_half_fma_finite(Builder& ir, Value raw_a, Value raw_b, Value raw_c) {
    Kernel k(ir);
    const auto a = k.decode(raw_a), b = k.decode(raw_b), c = k.decode(raw_c);
    const auto product = k.shifted(ir.imul(a.sig, b.sig), ir.iadd(a.shift, b.shift));
    const auto addend = k.shifted(c.sig, ir.iadd(c.shift, ir.c32(24)));
    const auto product_sign = ir.xor_(a.sign, b.sign);
    const auto same_sign = ir.ieq(product_sign, c.sign);
    const auto product_less = k.less(product, addend);
    const auto difference =
        k.choose(product_less, k.subtract(addend, product), k.subtract(product, addend));
    const auto magnitude = k.choose(same_sign, k.add(product, addend), difference);
    const auto sign =
        ir.select(same_sign, product_sign, ir.select(product_less, c.sign, product_sign));
    return k.finish(magnitude, sign, ir.booland_(a.finite, ir.booland_(b.finite, c.finite)));
}
std::array<Known, 2> eval_half_fma_finite(Known a, Known b, Known c) {
    Block block;
    Builder ir(block);
    const auto x = ir.input(Type::U32), y = ir.input(Type::U32), z = ir.input(Type::U32);
    const auto out = build_half_fma_finite(ir, x, y, z);
    const auto values = evaluate(block, std::array<Known, 3>{a, b, c});
    return {values[out.bits.index], values[out.eligible.index]};
}
std::array<uint32_t, 2> emit_half_fma_finite(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t d) {
    Block block;
    Builder ir(block);
    const auto x = ir.input(Type::U32), y = ir.input(Type::U32), z = ir.input(Type::U32);
    const auto out = build_half_fma_finite(ir, x, y, z);
    const auto values = emit_spirv(b, block, std::array<uint32_t, 3>{a, c, d});
    return {values[out.bits.index], values[out.eligible.index]};
}
}   // namespace prosper::gpu::ir

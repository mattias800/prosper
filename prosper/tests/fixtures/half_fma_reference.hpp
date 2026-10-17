// Test-only bit-serial exact rational oracle, independent of the production limb rounder.
#pragma once
#include <cstddef>
#include <bitset>
#include <cstdint>
namespace prosper::test::half_fma {
using Exact = std::bitset<128>;
inline bool less(const Exact& a, const Exact& b) {
    for (int k = 127; k >= 0; --k)
        if (a[k] != b[k]) return b[k];
    return false;
}
inline Exact add(const Exact& a, const Exact& b) {
    Exact out;
    bool carry = false;
    for (size_t k = 0; k < 128; ++k) {
        const bool x = a[k], y = b[k];
        out[k] = x ^ y ^ carry;
        carry = (x && y) || (carry && (x || y));
    }
    return out;
}
inline Exact subtract(const Exact& a, const Exact& b) {
    Exact out;
    bool borrow = false;
    for (size_t k = 0; k < 128; ++k) {
        const bool x = a[k], y = b[k];
        out[k] = x ^ y ^ borrow;
        borrow = (!x && y) || (borrow && (!x || y));
    }
    return out;
}
inline uint64_t scaled_half(uint16_t half) {
    const uint32_t exponent = (half >> 10) & 31;
    return uint64_t((half & 1023) | (exponent ? 1024 : 0)) << (exponent ? exponent - 1 : 0);
}
inline Exact scaled_candidate(uint32_t bits) {
    // 0x7c00 is the virtual next finite value 65536 for the overflow midpoint,
    // never a claim that the architectural infinity itself is finite.
    return bits == 0x7c00 ? Exact(1) << 64 : Exact(scaled_half(bits)) << 24;
}
struct Reference {
    uint16_t bits;
    bool eligible;
};
inline Reference reference(uint16_t a, uint16_t b, uint16_t c) {
    if ((a & 0x7c00) == 0x7c00 || (b & 0x7c00) == 0x7c00 || (c & 0x7c00) == 0x7c00)
        return {0, false};
    const auto aa = Exact(scaled_half(a));
    const auto bb = scaled_half(b);
    Exact product;
    for (uint32_t k = 0; k < 40; ++k)
        if ((bb >> k) & 1) product = add(product, aa << k);
    const auto addend = Exact(scaled_half(c)) << 24;
    const bool product_sign = ((a ^ b) & 0x8000) != 0, addend_sign = (c & 0x8000) != 0;
    Exact magnitude;
    bool sign;
    if (product_sign == addend_sign) {
        magnitude = add(product, addend);
        sign = product_sign;
    } else if (less(product, addend)) {
        magnitude = subtract(addend, product);
        sign = addend_sign;
    } else {
        magnitude = subtract(product, addend);
        sign = product_sign;
    }
    if (magnitude.none()) return {0, false};
    if (!less(magnitude, scaled_candidate(0x7c00))) return {0x7c00, false};
    uint32_t low = 0, high = 0x7c00;
    while (high - low > 1) {
        const auto mid = (low + high) / 2;
        if (less(magnitude, scaled_candidate(mid)))
            high = mid;
        else
            low = mid;
    }
    const auto below = subtract(magnitude, scaled_candidate(low));
    const auto above = subtract(scaled_candidate(high), magnitude);
    const auto rounded = less(below, above) || (below == above && !(low & 1)) ? low : high;
    return {uint16_t(rounded | (sign ? 0x8000 : 0)), rounded != 0 && rounded < 0x7c00};
}
}   // namespace prosper::test::half_fma

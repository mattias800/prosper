#pragma once
#include "fragment_resource_packet_fixture.hpp"
#include <array>
#include <stdexcept>

namespace prosper::test::fragment_special_f32 {
using namespace prosper::gpu;
// Independent test oracle: binary-search the ordered positive F32 bit space, comparing exact
// mathematical rationals (y*x, y*y, or y*y*x). It does NOT use the emitter's normalized quotient,
// trial-root, exponent-parity, or reciprocal-square-root remainder construction, nor host FP.
struct Integer {
    std::array<uint32_t, 40> words{};
    static Integer one() {
        Integer n;
        n.words[0] = 1;
        return n;
    }
    void multiply(uint32_t m) {
        uint64_t carry = 0;
        for (auto& word : words) {
            const uint64_t v = uint64_t(word) * m + carry;
            word = uint32_t(v);
            carry = v >> 32;
        }
        if (carry) throw std::runtime_error("special oracle product overflow");
    }
    void shift(uint32_t bits) {
        if (bits >= 32 * words.size()) throw std::runtime_error("special oracle shift overflow");
        Integer n;
        for (uint32_t i = 0; i < words.size(); ++i)
            if (words[i]) {
                const uint32_t j = i + bits / 32, r = bits % 32;
                if (j >= words.size()) throw std::runtime_error("special oracle shift overflow");
                n.words[j] |= words[i] << r;
                if (r) {
                    const auto high = words[i] >> (32 - r);
                    if (j + 1 == words.size() && high)
                        throw std::runtime_error("special oracle shift overflow");
                    if (j + 1 < words.size()) n.words[j + 1] |= high;
                }
            }
        *this = n;
    }
    void add(const Integer& b) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < words.size(); ++i) {
            const uint64_t v = uint64_t(words[i]) + b.words[i] + carry;
            words[i] = uint32_t(v);
            carry = v >> 32;
        }
        if (carry) throw std::runtime_error("special oracle sum overflow");
    }
    int compare(const Integer& b) const {
        for (size_t i = words.size(); i-- > 0;)
            if (words[i] != b.words[i]) return words[i] < b.words[i] ? -1 : 1;
        return 0;
    }
};
struct Rational {
    Integer n;
    int exponent = 0;
};
inline Rational decode(uint32_t bits) {
    const uint32_t e = bits >> 23, m = bits & 0x7fffff;
    Rational r{Integer::one(), e ? int(e) - 150 : -149};
    r.n.multiply(e ? m | 0x800000u : m);
    return r;
}
inline Rational product(Rational a, uint32_t bits) {
    const uint32_t e = bits >> 23, m = bits & 0x7fffff;
    a.n.multiply(e ? m | 0x800000u : m);
    a.exponent += e ? int(e) - 150 : -149;
    return a;
}
inline int compare(Rational a, Rational b) {
    const int base = std::min(a.exponent, b.exponent);
    a.n.shift(uint32_t(a.exponent - base));
    b.n.shift(uint32_t(b.exponent - base));
    return a.n.compare(b.n);
}
inline Rational square(Rational a) {
    // Only midpoint significands need squaring; generic limb schoolbook multiplication is
    // independent of production's bounded uint64 square. Keep exact discarded carry checks.
    Integer result;
    uint32_t count = static_cast<uint32_t>(a.n.words.size());
    while (count && !a.n.words[count - 1]) --count;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < count; ++j) {
            if (j + i >= result.words.size())
                throw std::runtime_error("special oracle square overflow");
            const uint64_t v = uint64_t(a.n.words[i]) * a.n.words[j] + result.words[i + j] + carry;
            result.words[i + j] = uint32_t(v);
            carry = v >> 32;
        }
        if (count + i >= result.words.size()) {
            if (carry) throw std::runtime_error("special oracle square overflow");
        } else
            result.words[count + i] = static_cast<uint32_t>(carry);
    }
    return {result, a.exponent * 2};
}
inline int relation(uint32_t op, Rational y, uint32_t x) {
    if (op == 0x2a) return compare(product(y, x), {Integer::one(), 0});
    if (op == 0x33) return compare(square(y), decode(x));
    return compare(product(square(y), x), {Integer::one(), 0});
}
inline uint32_t oracle(uint32_t op, uint32_t raw, uint8_t mode) {
    const uint32_t sign = raw & 0x80000000u, x = raw & 0x7fffffffu;
    if ((x >> 23) == 255 && (x & 0x7fffff))
        throw std::runtime_error("NaN oracle deliberately unavailable");
    if ((x >> 23) == 0) return op == 0x33 ? sign : sign | 0x7f800000u;
    if (op != 0x2a && sign)
        throw std::runtime_error("negative root oracle deliberately unavailable");
    if (x == 0x7f800000u) return op == 0x33 ? 0x7f800000u : sign;
    uint32_t lo = 0, hi = 0x7f7fffff;
    while (lo < hi) {
        const uint32_t middle = lo + (hi - lo + 1) / 2;
        if (relation(op, decode(middle), x) <= 0)
            lo = middle;
        else
            hi = middle - 1;
    }
    if (relation(op, decode(lo), x) != 0) {
        bool up = false;
        if ((mode & 3) == 0) {
            auto a = decode(lo), b = decode(lo + 1);
            const int base = std::min(a.exponent, b.exponent);
            a.n.shift(uint32_t(a.exponent - base));
            b.n.shift(uint32_t(b.exponent - base));
            a.n.add(b.n);
            a.exponent = base - 1;
            const int midpoint = relation(op, a, x);
            up = midpoint < 0 || (!midpoint && (lo & 1));
        } else if ((mode & 3) == 1)
            up = op == 0x2a ? !sign : true;
        else if ((mode & 3) == 2)
            up = op == 0x2a && sign;
        if (up) ++lo;
    }
    if (lo < 0x800000u) lo = 0; // opcode-specific output flush, AFTER rounding
    return (op == 0x2a ? sign : 0) | lo;
}
inline FragmentResourcePacket packet(uint32_t op, const std::array<uint32_t, 64>& inputs,
                                     uint8_t mode = 0x30, bool inactive40 = false) {
    auto p = fragment_resource_packet::base();
    p.entry_facts.observed = true;
    p.entry_facts.rsrc2_available = true; // genuine fixture known zero
    p.invocation.float_mode.value = mode;
    p.launch_rsrc1.value = (uint32_t(mode) << 12) | (1u << 23);
    if (inactive40) p.invocation.exec_mask &= ~(uint64_t{1} << 40);
    p.invocation.vgprs[0].words = inputs;
    auto& code = p.invocation.guest_code;
    code.push_back(0x7e000000u | (8u << 17) | (op << 9) | 256u); // ORIGINAL VOP1 v8,v0
    fragment_packet::exp(code, 1, 8);
    code.insert(code.end(), {0xd760003cu, 264u | (168u << 9)}); // READLANE s60,v8,40
    fragment_packet::vmov(code, 24, 60);
    fragment_packet::exp(code, 1, 24);
    code.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> expected(const FragmentResourcePacket& p, uint32_t op) {
    const auto& g = p.invocation;
    const uint32_t old = fragment_packet::poison_sentinel;
    const uint32_t peer =
        (g.exec_mask >> 40) & 1 ? oracle(op, g.vgprs[0].words[40], g.float_mode.value) : old;
    std::vector<uint32_t> words(64 * 24, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t active = (g.exec_mask >> lane) & 1;
        for (uint32_t event = 0; event < 2; ++event) {
            const uint32_t offset = lane * 24 + event * 12;
            const uint32_t header[]{1, active, g.export_enabled[lane], 0, 1, 0, 1, 1};
            std::copy(std::begin(header), std::end(header), words.begin() + offset);
            words[offset + 8] =
                active ? (event ? peer : oracle(op, g.vgprs[0].words[lane], g.float_mode.value))
                       : old;
        }
    }
    return words;
}
inline std::array<uint32_t, 64> rails(uint32_t op = 0x33) {
    constexpr uint32_t r[]{0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
                           0x00800000, 0x00800001, 0x00ffffff, 0x3eaaaaab, 0x3f000000, 0x3f7fffff,
                           0x3f800000, 0x3f800001, 0x3fffffff, 0x40000000, 0x4009f038, 0x40400000,
                           0x407fffff, 0x40800000, 0x7e800000, 0x7e800001, 0x7effffff, 0x7f000000,
                           0x7f7fffff, 0x7f800000};
    std::array<uint32_t, 64> words{};
    for (uint32_t i = 0; i < 64; ++i) words[i] = r[i % std::size(r)];
    if (op == 0x2a) {
        constexpr uint32_t signed_rails[]{0xfe800000u, 0xfe800001u, 0xfeffffffu,
                                          0xff000000u, 0xff7fffffu, 0xff800000u};
        std::copy(std::begin(signed_rails), std::end(signed_rails), words.begin() + 28);
    }
    return words;
}
inline FragmentResourcePacket high_pc_failure() {
    auto inputs = rails();
    inputs[63] = 0x7fc00001;
    auto p = packet(0x2e, inputs);
    // A real direct branch skips a same-program NOP region. Full original DWORD PC257 must
    // survive status publication; neither a byte-PC nor an 8-bit dispatcher ordinal is correct.
    p.invocation.guest_code.insert(p.invocation.guest_code.begin(), 256, 0xbf800000u);
    p.invocation.guest_code.insert(p.invocation.guest_code.begin(), 0xbf820100u);
    return p;
}
}   // namespace prosper::test::fragment_special_f32

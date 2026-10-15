// RDNA2 scalar-ALU instruction words for the IR tests (ISA 70648 chapter 13 encodings).
#pragma once

#include <cstdint>

namespace prosper::test::ir {

inline constexpr uint32_t kLiteral = 0xff;   // SSRC value: a 32-bit literal follows
inline constexpr uint32_t kInline0 = 0x80;   // SSRC inline integer 0; 0x81 = 1, ...

inline uint32_t sop2(uint32_t op, uint32_t d, uint32_t s0, uint32_t s1) {
    return 0x80000000u | (op << 23) | (d << 16) | (s1 << 8) | s0;
}
inline uint32_t sop1(uint32_t op, uint32_t d, uint32_t s0) {
    return 0xbe800000u | (d << 16) | (op << 8) | s0;
}
inline uint32_t sopc(uint32_t op, uint32_t s0, uint32_t s1) {
    return 0xbf000000u | (op << 16) | (s1 << 8) | s0;
}
inline uint32_t sopk(uint32_t op, uint32_t d, uint32_t simm16) {
    return 0xb0000000u | (op << 23) | (d << 16) | (simm16 & 0xffffu);
}

} // namespace prosper::test::ir

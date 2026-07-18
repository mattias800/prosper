#pragma once
#include <cstdint>

// AMD SSE4a bitfield semantics (INSERTQ / EXTRQ), extracted as pure functions so the signal-handler
// emulator in exec_image_linux.cpp can be unit-tested (tests/test_sse4a.cpp). PS5 guest code is
// compiled for Zen2 and emits these; they #UD (SIGILL) on Intel hosts, so we emulate them.
// Reference: AMD64 Architecture Programmer's Manual, Volume 4, publication 26568
// (INSERTQ/EXTRQ). Length/index are 6-bit fields; length 0 means 64; results with
// index+length > 64 are architecturally undefined (we let the 64-bit shifts drop them).
namespace prosper {

// INSERTQ: replace `len` bits of `dst` starting at bit `idx` with the low `len` bits of `src`.
// Only the low 64 bits participate; bits of dst outside [idx, idx+len) are preserved.
inline uint64_t sse4a_insertq(uint64_t dst, uint64_t src, uint32_t len, uint32_t idx) {
    if (len == 0) len = 64;
    uint64_t mask  = (len >= 64) ? ~0ull : ((1ull << len) - 1);
    uint64_t field = src & mask;
    uint64_t sh    = (idx >= 64) ? 0 : (mask  << idx);
    uint64_t sf    = (idx >= 64) ? 0 : (field << idx);
    return (dst & ~sh) | sf;
}

// EXTRQ: extract `len` bits of `src` starting at bit `idx`, zero-extended into the low 64 bits.
inline uint64_t sse4a_extrq(uint64_t src, uint32_t len, uint32_t idx) {
    if (len == 0) len = 64;
    uint64_t mask = (len >= 64) ? ~0ull : ((1ull << len) - 1);
    return (idx >= 64 ? 0 : (src >> idx)) & mask;
}

} // namespace prosper

// test_sse4a — golden vectors for the AMD SSE4a INSERTQ/EXTRQ emulation used by the SIGILL handler
// in exec_image_linux.cpp (PS5 Zen2 guest code #UDs these on Intel hosts). Values hand-computed
// from the Intel SDM bitfield semantics; this locks the arithmetic independently of the signal path.
#include "../src/host/sse4a.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;
static int g_fail = 0, g_pass = 0;
#define CHECK(expr, expect) do { uint64_t _v = (expr); if (_v == (uint64_t)(expect)) g_pass++; else { \
    g_fail++; printf("  [FAIL] %s:%d  %s = 0x%llx, expected 0x%llx\n", __FILE__, __LINE__, #expr, \
    (unsigned long long)_v, (unsigned long long)(uint64_t)(expect)); } } while (0)

int main() {
    printf("== test_sse4a ==\n");

    // --- INSERTQ: replace `len` bits of dst at bit `idx` with low `len` bits of src ---
    CHECK(sse4a_insertq(0x0, 0xAB, 8, 8), 0xAB00);                    // insert byte at bit 8
    CHECK(sse4a_insertq(0x12, 0x12, 8, 8), 0x1212);                  // the real crash form: byte1 := byte0
    CHECK(sse4a_insertq(0x0, 0xF, 4, 32), 0x0000000F00000000ull);    // nibble at bit 32
    CHECK(sse4a_insertq(~0ull, 0x0, 4, 0), 0xFFFFFFFFFFFFFFF0ull);   // clear low nibble, keep rest
    CHECK(sse4a_insertq(~0ull, 0x123456789ABCDEF0ull, 0, 0),         // len==0 => 64: full replace
                        0x123456789ABCDEF0ull);
    CHECK(sse4a_insertq(0xFF00, 0xFF, 8, 0), 0xFFFF);                // low byte := src, high byte kept

    // --- EXTRQ: extract `len` bits at bit `idx`, zero-extend into low 64 ---
    CHECK(sse4a_extrq(0xAB00, 8, 8), 0xAB);                          // pull byte 1
    CHECK(sse4a_extrq(0x123456789ABCDEF0ull, 16, 16), 0x9ABC);      // bits [31:16]
    CHECK(sse4a_extrq(0xFF, 4, 0), 0xF);                            // low nibble
    CHECK(sse4a_extrq(0x123456789ABCDEF0ull, 0, 0),                 // len==0 => 64: whole qword
                      0x123456789ABCDEF0ull);
    CHECK(sse4a_extrq(~0ull, 1, 63), 0x1);                         // top bit -> bit 0

    // Round-trip: insert then extract the same field recovers it.
    CHECK(sse4a_extrq(sse4a_insertq(0x0, 0x3C, 8, 24), 8, 24), 0x3C);

    printf("== test_sse4a: %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

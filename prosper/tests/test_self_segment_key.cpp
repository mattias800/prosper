// test_self_segment_key (#339) — the SELF data-segment map key is the SCE self-segment id in flag
// bits [31:20], masked to 12 bits. Without the mask, a data segment whose flags carry any bit >= 32
// keys the map by a garbage index, so every phdr lookup misses and the segment silently falls back to
// `elf_base + p_offset` (mapping garbage bytes). This pins the mask math (matches Kyty Elf.cpp).
#include "../src/self/module.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_self_segment_key ==\n");

    // Real Messenger-style flags (data flag 0x800 + id in [31:20], no upper bits): key == the id.
    CHECK(self_segment_key(0x002804) == 0, "flags 0x002804 -> id 0");
    CHECK(self_segment_key(0x102804) == 1, "flags 0x102804 -> id 1");
    CHECK(self_segment_key(0x302804) == 3, "flags 0x302804 -> id 3");
    CHECK(self_segment_key(0xb02804) == 0xb, "flags 0xb02804 -> id 0xb");

    // The id field is 12 bits [31:20]; its max is 0xFFF.
    CHECK(self_segment_key(0xFFF00000) == 0xFFF, "max 12-bit id 0xFFF");

    // The bug: a flag bit >= 32 must NOT leak into the key. Bit 40 set alongside id 1 -> still id 1
    // (unmasked this would be 0x100001, a garbage index that misses every phdr lookup).
    CHECK(self_segment_key(0x10000102804ull) == 1, "bit 40 set + id 1 -> masked to id 1 (not 0x100001)");
    CHECK((0x10000102804ull >> 20) == 0x100001ull, "  (sanity: the UNMASKED key would be garbage 0x100001)");
    CHECK(self_segment_key(0xFFFFFFFF00000000ull | 0x702804) == 7, "all upper bits set + id 7 -> id 7");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

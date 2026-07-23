// test_self_bounds — hardening guard for SELF/ELF/loader parsing against malformed/truncated dumps.
// A corrupt or partially-copied dump must be handled gracefully, never read out of bounds. Three
// latent memory-safety bugs are fixed and guarded here (all reachable only on malformed input, so
// current well-formed dumps are unaffected):
//   A) rd()'s bounds check was `off + sizeof(T) <= size`, which WRAPS for a near-2^64 offset (an
//      unvalidated e_phoff feeds one straight in) and then passes -> OOB read. Now uses the
//      overflow-safe self_read_ok(). Guarded directly by test_self_read_ok().
//   B) the linker init-array read (linker.cpp) did `memcpy(&fn, p, 8)` after only `if(!p)`, so a
//      DT_INIT_ARRAY landing in the final <8 bytes of the image read past mem. Now has the same
//      `p+8 > end` guard its sibling write64 already had. The invariant it relies on (at() returns a
//      valid pointer for the LAST byte, which +8 overruns) is guarded by test_at_last_byte_needs_guard().
//   C) str_at() could return a pointer with no NUL before EOF -> a consumer's strlen/std::string
//      reads past the buffer. Now requires an in-range NUL. Guarded by test_str_at().
// No game dump needed: pure in-memory construction.
#include "../src/self/module.hpp"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

using namespace prosper;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; \
    printf("  [FAIL] %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// A) The overflow-safe bounds primitive rd() now uses. The naive `off + need <= total` returns TRUE
// for the last two cases (the addition wraps), which is exactly the OOB-read bug; self_read_ok rejects.
static void test_self_read_ok() {
    CHECK(self_read_ok(0, 8, 100),        "0+8 fits in 100");
    CHECK(self_read_ok(92, 8, 100),       "92+8==100 exact fit");
    CHECK(!self_read_ok(93, 8, 100),      "93+8>100 rejected");
    CHECK(!self_read_ok(0, 200, 100),     "need>total rejected");
    CHECK(self_read_ok(100, 0, 100),      "0 bytes at end-of-buffer is in-bounds");
    CHECK(!self_read_ok(UINT64_MAX - 2, 8, 100),
          "near-2^64 offset must NOT wrap past the check (the rd() OOB bug)");
    CHECK(!self_read_ok(UINT64_MAX, 1, 100), "max offset rejected");
    CHECK(!self_read_ok(50, 8, 4),        "buffer smaller than the read is rejected");
}

// C) str_at must (1) reject an out-of-range/overflowing offset and (2) never return a pointer whose
// string runs past EOF with no terminator. A PT_LOAD maps strtab_va 0x1000 -> file offset 0.
static Module strtab_module(const std::vector<uint8_t>& bytes) {
    Module m;
    m.file = bytes;
    Segment s; s.type = PT_LOAD; s.vaddr = 0x1000; s.filesz = bytes.size(); s.memsz = bytes.size(); s.file_off = 0;
    m.loads.push_back(s);
    m.strtab_va = 0x1000;
    return m;
}
static void test_str_at() {
    // Well-formed strtab: two NUL-terminated names back to back.
    {
        std::vector<uint8_t> b = {'h','e','l','l','o',0,'w','o','r','l','d',0};
        Module m = strtab_module(b);
        CHECK(strcmp(m.str_at(0), "hello") == 0, "str_at(0) -> 'hello'");
        CHECK(strcmp(m.str_at(6), "world") == 0, "str_at(6) -> 'world'");
    }
    // Malformed: 16 non-zero bytes, NO terminator before EOF. Pre-fix this returned a live pointer
    // and a consumer strlen would run off the end; now it must return "" (empty).
    {
        std::vector<uint8_t> b(16, 'A');
        Module m = strtab_module(b);
        CHECK(m.str_at(0)[0] == '\0', "unterminated strtab -> empty string (no OOB strlen)");
        CHECK(strlen(m.str_at(0)) == 0, "unterminated strtab -> strlen 0");
    }
    // Offset at/after EOF, and an overflowing offset, both -> "".
    {
        std::vector<uint8_t> b = {'x',0};
        Module m = strtab_module(b);
        CHECK(m.str_at(2)[0] == '\0',            "offset == size -> empty");
        CHECK(m.str_at(1000)[0] == '\0',         "offset past EOF -> empty");
        CHECK(m.str_at(UINT64_MAX)[0] == '\0',   "overflowing offset -> empty (no wrap)");
    }
}

// B) The invariant the linker init-array guard relies on: at() returns a valid pointer for the LAST
// byte of the image, so a raw `memcpy(p, 8)` there reads 7 bytes past mem — hence the p+8>end guard.
static void test_at_last_byte_needs_guard() {
    LoadedImage img;
    img.base = 0x10000; img.min_vaddr = 0; img.max_vaddr = 0x100;
    img.mem.assign(0x100, 0);
    const uint8_t* first = img.at(0x10000);
    const uint8_t* last  = img.at(0x100FF);            // last in-image byte
    const uint8_t* end   = img.mem.data() + img.mem.size();
    CHECK(first == img.mem.data(),                    "at(base) maps to mem start");
    CHECK(last == img.mem.data() + 0xFF,              "at(last va) maps to last byte");
    CHECK(img.at(0x10100) == nullptr,                 "at(base+max) is out of image");
    // The whole point of B: at() alone does NOT guarantee 8 readable bytes.
    CHECK(last != nullptr && last + 8 > end,          "last-byte pointer +8 overruns mem -> guard required");
}

// A (bulk memcpy): build_image maps each PT_LOAD's filesz bytes. A malformed near-2^64 filesz wraps
// the naive `dst + filesz <= mem.size()` check and drives a huge OOB memcpy (a hard segfault) — the
// overflow-safe check must skip it. A valid segment must still map exactly.
static void test_build_image_bounds() {
    {   // valid PT_LOAD maps its bytes at the right image offset
        Module m;
        m.file = {1,2,3,4,5,6,7,8};
        Segment s; s.type = PT_LOAD; s.vaddr = 0x4000; s.filesz = 8; s.memsz = 0x4000; s.file_off = 0; s.flags = 4;
        m.segments.push_back(s);
        LoadedImage img = build_image(m, 0x100000000ull);
        const uint8_t* p = img.at(0x100000000ull + 0x4000);
        CHECK(p && p[0] == 1 && p[7] == 8, "valid PT_LOAD maps its filesz bytes");
    }
    {   // malformed huge filesz that WRAPS BOTH naive checks (dest and source) so they pass -> a huge
        // OOB memcpy (hard segfault). The overflow-safe check must skip it. vaddr=0x4001 -> dst=1;
        // with filesz=UINT64_MAX and file_off=1, both `1 + (2^64-1)` wrap to 0, which the OLD
        // `a + b <= size` accepts; the memcpy size itself stays 2^64-1.
        Module m;
        m.file.assign(64, 0xAB);
        Segment bad; bad.type = PT_LOAD; bad.vaddr = 0x4001; bad.filesz = UINT64_MAX; bad.memsz = 0x4000; bad.file_off = 1; bad.flags = 4;
        m.segments.push_back(bad);
        LoadedImage img = build_image(m, 0x100000000ull);   // reaching here at all means no OOB memcpy
        CHECK(img.mem.size() > 0, "wrapping huge filesz skipped, no OOB memcpy (image still sized)");
    }
    {   // malformed huge memsz: vaddr+memsz does not overflow (passes the extent-skip guard) but
        // align_up(hi) wraps to 0, giving max_vaddr(0) < min_vaddr(0x8000). Without the clamp,
        // mem.assign(0 - 0x8000) requests ~2^64 bytes -> bad_alloc/terminate. The clamp keeps it empty.
        Module m;
        m.file = {1,2,3,4};
        Segment s; s.type = PT_LOAD; s.vaddr = 0x8000; s.filesz = 4; s.memsz = UINT64_MAX - 0x8000; s.file_off = 0; s.flags = 4;
        m.segments.push_back(s);
        LoadedImage img = build_image(m, 0x100000000ull);   // must not attempt a ~2^64 mem.assign
        CHECK(img.mem.size() == 0, "align_up-wrapped extent clamped to empty (no giant allocation)");
    }
}

int main() {
    printf("== test_self_bounds (SELF/ELF/loader malformed-input hardening) ==\n");
    test_self_read_ok();
    test_str_at();
    test_at_last_byte_needs_guard();
    test_build_image_bounds();
    printf("%s  (%d passed, %d failed)\n", g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

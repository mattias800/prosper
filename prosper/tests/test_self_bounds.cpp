// test_self_bounds — hardening guard for SELF/ELF/loader parsing against malformed/truncated dumps.
// A corrupt or partially-copied dump must be handled gracefully, never read out of bounds. Three
// latent memory-safety bugs are fixed and guarded here (all reachable only on malformed input, so
// current well-formed dumps are unaffected):
//   A) rd()'s bounds check was `off + sizeof(T) <= size`, which WRAPS for a near-2^64 offset (an
//      unvalidated e_phoff feeds one straight in) and then passes -> OOB read. Now uses the
//      overflow-safe self_read_ok(). Guarded directly by SelfReadOk.
//   B) the linker init-array read (linker.cpp) did `memcpy(&fn, p, 8)` after only `if(!p)`, so a
//      DT_INIT_ARRAY landing in the final <8 bytes of the image read past mem. Now has the same
//      `p+8 > end` guard its sibling write64 already had. The invariant it relies on (at() returns a
//      valid pointer for the LAST byte, which +8 overruns) is guarded by AtLastByteNeedsGuard.
//   C) str_at() could return a pointer with no NUL before EOF -> a consumer's strlen/std::string
//      reads past the buffer. Now requires an in-range NUL. Guarded by the StrAt* arms.
//   D) build_image() allowed a huge non-wrapping PT_LOAD span to reach vector::assign; the uncaught
//      length_error/bad_alloc terminated linking. Image construction is now fallible and reports it.
//   E) build_image() SKIPPED any PT_LOAD it could not copy in full - a filesz past EOF, a segment
//      outside the image it had just sized - and returned true anyway, so the module mapped a
//      zero-filled hole where its own bytes belonged and the caller was told it had loaded (#2631).
//      Every such segment is now a refusal with a message naming the program header and the
//      shortfall. The arms below are the malformed-input half; tests/misc/test_loader_synth_reject.cpp
//      carries the truncated-module half, where the module otherwise parses perfectly.
// No game dump needed: pure in-memory construction.
#include "../src/self/module.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

namespace {
// A) The overflow-safe bounds primitive rd() now uses. The naive `off + need <= total` returns TRUE
// for the last two cases (the addition wraps), which is exactly the OOB-read bug; self_read_ok rejects.
TEST(SelfBounds, SelfReadOk) {
    EXPECT_TRUE(self_read_ok(0, 8, 100)) << "0+8 fits in 100";
    EXPECT_TRUE(self_read_ok(92, 8, 100)) << "92+8==100 exact fit";
    EXPECT_FALSE(self_read_ok(93, 8, 100)) << "93+8>100 rejected";
    EXPECT_FALSE(self_read_ok(0, 200, 100)) << "need>total rejected";
    EXPECT_TRUE(self_read_ok(100, 0, 100)) << "0 bytes at end-of-buffer is in-bounds";
    EXPECT_FALSE(self_read_ok(UINT64_MAX - 2, 8, 100))
        << "near-2^64 offset must NOT wrap past the check (the rd() OOB bug)";
    EXPECT_FALSE(self_read_ok(UINT64_MAX, 1, 100)) << "max offset rejected";
    EXPECT_FALSE(self_read_ok(50, 8, 4)) << "buffer smaller than the read is rejected";
}

// C) str_at must (1) reject an out-of-range/overflowing offset and (2) never return a pointer whose
// string runs past EOF with no terminator. A PT_LOAD maps strtab_va 0x1000 -> file offset 0.
Module strtab_module(const std::vector<uint8_t>& bytes) {
    Module m;
    m.file = bytes;
    Segment s; s.type = PT_LOAD; s.vaddr = 0x1000; s.filesz = bytes.size(); s.memsz = bytes.size(); s.file_off = 0;
    m.loads.push_back(s);
    m.strtab_va = 0x1000;
    return m;
}
}   // namespace

TEST(SelfBounds, StrAtReadsTerminatedNames) {
    // Well-formed strtab: two NUL-terminated names back to back.
    const std::vector<uint8_t> b = {'h', 'e', 'l', 'l', 'o', 0, 'w', 'o', 'r', 'l', 'd', 0};
    const Module m = strtab_module(b);
    EXPECT_STREQ(m.str_at(0), "hello");
    EXPECT_STREQ(m.str_at(6), "world");
}

TEST(SelfBounds, StrAtRefusesAnUnterminatedString) {
    // Malformed: 16 non-zero bytes, NO terminator before EOF. Pre-fix this returned a live pointer
    // and a consumer strlen would run off the end; now it must return "" (empty).
    const Module m = strtab_module(std::vector<uint8_t>(16, 'A'));
    EXPECT_EQ(m.str_at(0)[0], '\0') << "unterminated strtab -> empty string (no OOB strlen)";
    EXPECT_EQ(std::strlen(m.str_at(0)), 0u) << "unterminated strtab -> strlen 0";
}

TEST(SelfBounds, StrAtRefusesAnOutOfRangeOffset) {
    // Offset at/after EOF, and an overflowing offset, all -> "".
    const Module m = strtab_module({'x', 0});
    EXPECT_EQ(m.str_at(2)[0], '\0') << "offset == size -> empty";
    EXPECT_EQ(m.str_at(1000)[0], '\0') << "offset past EOF -> empty";
    EXPECT_EQ(m.str_at(UINT64_MAX)[0], '\0') << "overflowing offset -> empty (no wrap)";
}

// B) The invariant the linker init-array guard relies on: at() returns a valid pointer for the LAST
// byte of the image, so a raw `memcpy(p, 8)` there reads 7 bytes past mem — hence the p+8>end guard.
TEST(SelfBounds, AtLastByteNeedsGuard) {
    LoadedImage img;
    img.base = 0x10000; img.min_vaddr = 0; img.max_vaddr = 0x100;
    img.mem.assign(0x100, 0);
    const uint8_t* first = img.at(0x10000);
    const uint8_t* last  = img.at(0x100FF);            // last in-image byte
    const uint8_t* end   = img.mem.data() + img.mem.size();
    EXPECT_EQ(first, img.mem.data()) << "at(base) maps to mem start";
    EXPECT_EQ(last, img.mem.data() + 0xFF) << "at(last va) maps to last byte";
    EXPECT_EQ(img.at(0x10100), nullptr) << "at(base+max) is out of image";
    // The whole point of B: at() alone does NOT guarantee 8 readable bytes.
    ASSERT_NE(last, nullptr);
    EXPECT_GT(last + 8, end) << "last-byte pointer +8 overruns mem -> guard required";
}

// A (bulk memcpy): build_image maps each PT_LOAD's filesz bytes. A malformed near-2^64 filesz wraps
// the naive `dst + filesz <= mem.size()` check and drives a huge OOB memcpy (a hard segfault) — the
// overflow-safe check must skip it. A valid segment must still map exactly.
TEST(SelfBounds, BuildImageMapsAValidSegment) {
    Module m;
    m.file = {1, 2, 3, 4, 5, 6, 7, 8};
    Segment s;
    s.type = PT_LOAD;
    s.vaddr = 0x4000;
    s.filesz = 8;
    s.memsz = 0x4000;
    s.file_off = 0;
    s.flags = 4;
    m.segments.push_back(s);
    LoadedImage img;
    ASSERT_TRUE(build_image(m, 0x100000000ull, img)) << "valid PT_LOAD image builds";
    const uint8_t* p = img.at(0x100000000ull + 0x4000);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p[0], 1) << "valid PT_LOAD maps its filesz bytes";
    EXPECT_EQ(p[7], 8) << "valid PT_LOAD maps its filesz bytes";
}

TEST(SelfBounds, BuildImageRefusesASegmentOneByteLongerThanItsFile) {
    // E) the razor for #2631: the SAME segment as the control above, differing in p_filesz alone
    // (9 rather than 8), against the same 8-byte file. One byte of the segment does not exist.
    // Pre-fix, the copy was skipped in full and build_image returned true, so the eight bytes
    // that DO exist were replaced by zeros and the caller was told the module had loaded.
    Module m;
    m.file = {1, 2, 3, 4, 5, 6, 7, 8};
    Segment s;
    s.type = PT_LOAD;
    s.vaddr = 0x4000;
    s.filesz = 9;
    s.memsz = 0x4000;
    s.file_off = 0;
    s.flags = 4;
    m.segments.push_back(s);
    LoadedImage img;
    std::string err;
    EXPECT_FALSE(build_image(m, 0x100000000ull, img, &err))
        << "a PT_LOAD one byte longer than its file is refused";
    EXPECT_NE(err.find("declares 0x9 bytes at file offset 0x0"), std::string::npos)
        << "the refusal quotes the declared length and offset";
    EXPECT_NE(err.find("the file holds only 0x8 bytes"), std::string::npos)
        << "the refusal quotes what the file actually holds";
}

TEST(SelfBounds, BuildImageRefusesAWrappingFilesz) {
    // malformed huge filesz that WRAPS BOTH naive checks (dest and source) so they pass -> a huge
    // OOB memcpy (hard segfault). The overflow-safe check must catch it. vaddr=0x4001 -> dst=1;
    // with filesz=UINT64_MAX and file_off=1, both `1 + (2^64-1)` wrap to 0, which the OLD
    // `a + b <= size` accepts; the memcpy size itself stays 2^64-1.
    //
    // Reaching this line at all IS the memory-safety assertion - the pre-fix code segfaults here
    // and never returns, whatever the return value. What changed with #2631 is only the verdict:
    // the segment used to be dropped and the load called successful, which mapped nothing of a
    // module the caller then linked. A segment this malformed is unloadable, so the honest answer
    // is a refusal that says which header and why.
    Module m;
    m.file.assign(64, 0xAB);
    Segment bad;
    bad.type = PT_LOAD;
    bad.vaddr = 0x4001;
    bad.filesz = UINT64_MAX;
    bad.memsz = 0x4000;
    bad.file_off = 1;
    bad.flags = 4;
    m.segments.push_back(bad);
    LoadedImage img;
    std::string err;
    EXPECT_FALSE(build_image(m, 0x100000000ull, img, &err))
        << "wrapping filesz is refused, not silently skipped (no OOB memcpy either way)";
    // #4344: ELF requires p_filesz <= p_memsz, and build_image now refuses that first, before
    // sizing the image or reaching the copy. The wrapping copy check above still guards a
    // segment whose filesz fits its memsz; this case pins that the earlier rule names it.
    EXPECT_NE(err.find("filesz 0xffffffffffffffff exceeds memsz 0x4000"), std::string::npos)
        << "wrapping filesz is refused as larger than its memsz";
    EXPECT_NE(err.find("program header 0"), std::string::npos)
        << "wrapping filesz names the offending program header";
}

TEST(SelfBounds, BuildImageRefusesAnAlignUpWrappedExtent) {
    // malformed huge memsz: vaddr+memsz does not overflow (passes the extent-skip guard) but
    // align_up(hi) wraps to 0, giving max_vaddr(0) < min_vaddr(0x8000). Unguarded,
    // mem.assign(0 - 0x8000) requests ~2^64 bytes -> bad_alloc/terminate; the wrap is detected
    // before the allocation, which is what keeps this arm from being a crash.
    //
    // It used to be detected and then CLAMPED to an empty image with a `true` return - a module
    // declaring a 16-exabyte span "loaded" with nothing mapped (#2631). The wrap is now the
    // refusal it always was, and it is reported before any allocation is attempted.
    Module m;
    m.file = {1, 2, 3, 4};
    Segment s;
    s.type = PT_LOAD;
    s.vaddr = 0x8000;
    s.filesz = 4;
    s.memsz = UINT64_MAX - 0x8000;
    s.file_off = 0;
    s.flags = 4;
    m.segments.push_back(s);
    LoadedImage img;
    std::string err;
    EXPECT_FALSE(build_image(m, 0x100000000ull, img, &err))
        << "align-up-wrapped extent is refused (no giant allocation, no empty-image success)";
    EXPECT_NE(err.find("overflows the 64-bit address space"), std::string::npos)
        << "align-up-wrapped extent reports the address-space overflow by name";
    EXPECT_EQ(img.mem.size(), 0u) << "a refused image leaves the caller's LoadedImage untouched";
}

TEST(SelfBounds, BuildImageRefusesAHugeNonWrappingExtent) {
    // A huge but non-wrapping extent passes the arithmetic guards above and is still far below a
    // 64-bit vector::max_size(). Before #1299 it reached vector::assign; Linux overcommit could
    // accept the virtual allocation, then OOM-kill the process while assign zero-filled it.
    Module m;
    Segment s;
    s.type = PT_LOAD;
    s.vaddr = 0x4000;
    s.memsz = kMaxLoadedImageBytes + 0x4000;
    s.flags = 4;
    m.segments.push_back(s);
    LoadedImage img;
    std::string err;
    EXPECT_LT(s.memsz, static_cast<uint64_t>(img.mem.max_size()))
        << "regression extent is below vector::max_size (exercises loader policy, not STL limit)";
    EXPECT_FALSE(build_image(m, 0x100000000ull, img, &err))
        << "huge non-wrapping PT_LOAD extent is rejected without throwing";
    EXPECT_FALSE(err.empty()) << "rejected huge PT_LOAD reports a load error";
}

// clamp_table_bytes bounds a guest-declared reloc/symbol table size to the file (#1219). Without it a
// malformed DT_RELASZ/DT_PLTRELSZ/symtab size drives an enormous relocs/symbols vector -> OOM.
TEST(SelfBounds, ClampTableBytesKeepsTablesInsideTheFile) {
    EXPECT_EQ(clamp_table_bytes(0, 240, 1000), 240u) << "table fully inside file: unchanged";
    EXPECT_EQ(clamp_table_bytes(100, 900, 1000), 900u) << "table exactly reaching EOF: unchanged";
    EXPECT_EQ(clamp_table_bytes(500, 100, 1000), 100u) << "small table well inside file: unchanged";
    EXPECT_EQ(clamp_table_bytes(999, 240, 1000), 1u) << "offset one byte before EOF -> 1 byte";
    EXPECT_EQ(clamp_table_bytes(1000, 240, 1000), 0u) << "table offset == EOF -> 0 bytes";
    EXPECT_EQ(clamp_table_bytes(2000, 240, 1000), 0u)
        << "table offset past EOF -> 0 bytes (no underflow)";
}

TEST(SelfBounds, ClampTableBytesBoundsAHugeDeclaredSize) {
    EXPECT_EQ(clamp_table_bytes(100, UINT64_MAX, 1000), 900u)
        << "huge declared size clamped to file remainder";
    // The concrete OOM guard: a UINT64_MAX DT_RELASZ over a 240-byte file yields at most 240/24 = 10
    // relocs, not ~7.7e17 — the value the RELA loop actually iterates.
    EXPECT_EQ(clamp_table_bytes(0, UINT64_MAX, 240) / 24, 10u)
        << "huge DT_RELASZ -> bounded reloc count";
    EXPECT_EQ(clamp_table_bytes(0, 0, 1000), 0u) << "zero declared size stays zero";
}
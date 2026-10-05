// libSceCesCs — single-character CP1252<->UTF-8 conversion (sceCesSbcToUtf8,
// sceCesUtf8ToSbc, sceCesRefersUcsProfileCp1252).
//
// All three were unregistered, so the dispatcher answered `0`. For the two converters 0 is
// success, and their whole job is filling a caller buffer — the unwritten-buffer shape again,
// except here the buffer is 1-3 bytes rather than an image or a struct.
//
// The conversion itself is real: CP1252 is a fixed public table (256 entries below), and UTF-8
// encode/decode is pure arithmetic. The profile the Refers call hands out IS the table the
// converters read, so a guest cannot observe a disagreement between them; a foreign profile
// pointer is refused. Error convention: 0 on success, -1 on any failure. No CesCs error
// facility is evidenced anywhere (neither secondary source names one), so the value is generic
// while the polarity — nonzero on every failure — is what guests branch on. CONFIDENCE: LOW on
// the -1 value, HIGH on the conversion results and the polarity.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_copy.hpp"

#include <cstdint>

namespace prosper {

using host::guest_read_exact;
using host::guest_write_exact;

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

namespace {

// Windows-1252 to Unicode. 0x00-0x7F identity; 0xA0-0xFF identity to U+00A0-U+00FF; the five
// undefined bytes (0x81, 0x8D, 0x8F, 0x90, 0x9D) map to U+FFFD and so never round-trip.
constexpr uint16_t kCp1252[256] = {
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000a, 0x000b,
    0x000c, 0x000d, 0x000e, 0x000f, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017,
    0x0018, 0x0019, 0x001a, 0x001b, 0x001c, 0x001d, 0x001e, 0x001f, 0x0020, 0x0021, 0x0022, 0x0023,
    0x0024, 0x0025, 0x0026, 0x0027, 0x0028, 0x0029, 0x002a, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037, 0x0038, 0x0039, 0x003a, 0x003b,
    0x003c, 0x003d, 0x003e, 0x003f, 0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004a, 0x004b, 0x004c, 0x004d, 0x004e, 0x004f, 0x0050, 0x0051, 0x0052, 0x0053,
    0x0054, 0x0055, 0x0056, 0x0057, 0x0058, 0x0059, 0x005a, 0x005b, 0x005c, 0x005d, 0x005e, 0x005f,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067, 0x0068, 0x0069, 0x006a, 0x006b,
    0x006c, 0x006d, 0x006e, 0x006f, 0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007a, 0x007b, 0x007c, 0x007d, 0x007e, 0x007f, 0x20ac, 0xfffd, 0x201a, 0x0192,
    0x201e, 0x2026, 0x2020, 0x2021, 0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0xfffd, 0x017d, 0xfffd,
    0xfffd, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x02dc, 0x2122, 0x0161, 0x203a,
    0x0153, 0xfffd, 0x017e, 0x0178, 0x00a0, 0x00a1, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7,
    0x00a8, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x00af, 0x00b0, 0x00b1, 0x00b2, 0x00b3,
    0x00b4, 0x00b5, 0x00b6, 0x00b7, 0x00b8, 0x00b9, 0x00ba, 0x00bb, 0x00bc, 0x00bd, 0x00be, 0x00bf,
    0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7, 0x00c8, 0x00c9, 0x00ca, 0x00cb,
    0x00cc, 0x00cd, 0x00ce, 0x00cf, 0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7,
    0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df, 0x00e0, 0x00e1, 0x00e2, 0x00e3,
    0x00e4, 0x00e5, 0x00e6, 0x00e7, 0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef,
    0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7, 0x00f8, 0x00f9, 0x00fa, 0x00fb,
    0x00fc, 0x00fd, 0x00fe, 0x00ff,
};
static_assert(sizeof(kCp1252) == 512, "256-entry UCS profile");

// Zero-extended -1: these contracts return int, and writing eax zeroes upper rax on real
// hardware, so 0x00000000FFFFFFFF is the failure word (low 32 read as -1 either way).
constexpr uint64_t kCesErr = 0xFFFFFFFFull;

bool ces_profile_ok(uint64_t profile) {
    return profile == (uint64_t)(uintptr_t)kCp1252;
}

// Strict single-sequence UTF-8 decode: rejects overlongs, surrogates and >U+FFFF (the table
// is BMP-only, so astral input is unmappable anyway). Returns consumed bytes, 0 if invalid.
size_t ces_utf8_decode(const uint8_t* s, size_t avail, uint32_t* out) {
    if (avail == 0) return 0;
    const uint8_t b0 = s[0];
    if (b0 < 0x80) {
        *out = b0;
        return 1;
    }
    if ((b0 & 0xE0) == 0xC0) {
        if (avail < 2 || (s[1] & 0xC0) != 0x80) return 0;
        const uint32_t cp = ((uint32_t)(b0 & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
        if (cp < 0x80) return 0;  // overlong
        *out = cp;
        return 2;
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (avail < 3 || (s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        const uint32_t cp = ((uint32_t)(b0 & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) |
                            (uint32_t)(s[2] & 0x3F);
        if (cp < 0x800) return 0;  // overlong
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        *out = cp;
        return 3;
    }
    return 0;
}

}  // namespace

// sceCesRefersUcsProfileCp1252() -> the CP1252 UCS profile. The converters below read this
// same table, so profile and conversion can never disagree.
HLE(ces_refers_ucs_profile_cp1252) {
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return (uint64_t)(uintptr_t)kCp1252;
}

// sceCesSbcToUtf8(profile, sbc, utf8*, utf8max, utf8_len*) -> 0; *len = bytes written.
HLE(ces_sbc_to_utf8) {
    (void)a5;
    if (!ces_profile_ok(a0) || !a2 || !a4 || a3 == 0) return kCesErr;
    const uint32_t cp = kCp1252[a1 & 0xFFu];
    uint8_t enc[3]{};
    size_t n = 0;
    if (cp < 0x80) {
        enc[0] = (uint8_t)cp;
        n = 1;
    } else if (cp < 0x800) {
        enc[0] = (uint8_t)(0xC0 | (cp >> 6));
        enc[1] = (uint8_t)(0x80 | (cp & 0x3F));
        n = 2;
    } else {
        enc[0] = (uint8_t)(0xE0 | (cp >> 12));
        enc[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        enc[2] = (uint8_t)(0x80 | (cp & 0x3F));
        n = 3;
    }
    if (n > (size_t)a3) return kCesErr;
    if (!guest_write_exact(a2, enc, n)) return kCesErr;
    const uint32_t len = (uint32_t)n;
    if (!guest_write_exact(a4, &len, sizeof len)) return kCesErr;
    return 0;
}

// sceCesUtf8ToSbc(utf8*, utf8max, utf8_len*, profile, sbc*) -> 0; *len = bytes consumed.
HLE(ces_utf8_to_sbc) {
    (void)a5;
    if (!a0 || a1 == 0 || !a2 || !ces_profile_ok(a3) || !a4) return kCesErr;
    // Read the lead byte first: the sequence length is known only after it, and reading a
    // fixed 4 up front would fault past a short-but-valid mapped buffer.
    uint8_t head[4]{};
    if (!guest_read_exact(a0, head, 1)) return kCesErr;
    size_t want = 1;
    if (head[0] >= 0x80) {
        if ((head[0] & 0xE0) == 0xC0)
            want = 2;
        else if ((head[0] & 0xF0) == 0xE0)
            want = 3;
        else
            return kCesErr;
    }
    if (want > (size_t)a1) return kCesErr;  // truncated input
    if (want > 1 && !guest_read_exact(a0 + 1, head + 1, want - 1)) return kCesErr;
    uint32_t cp = 0;
    const size_t used = ces_utf8_decode(head, want, &cp);
    if (used == 0) return kCesErr;
    int slot = -1;
    for (int i = 0; i < 256; i++) {
        if (kCp1252[i] == cp) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return kCesErr;  // unmappable (astral, or U+FFFD-class without a byte)
    const uint8_t b = (uint8_t)slot;
    if (!guest_write_exact(a4, &b, 1)) return kCesErr;
    const uint32_t n = (uint32_t)used;
    if (!guest_write_exact(a2, &n, sizeof n)) return kCesErr;
    return 0;
}

void register_cescs_hle() {
    Hle::register_fn("LPzYZ+FR0BI", (HleFn)ces_refers_ucs_profile_cp1252,
                     "sceCesRefersUcsProfileCp1252");
    Hle::register_fn("xTd54EEL1Ao", (HleFn)ces_sbc_to_utf8, "sceCesSbcToUtf8");
    Hle::register_fn("3Q1gOWWarcw", (HleFn)ces_utf8_to_sbc, "sceCesUtf8ToSbc");
}

}  // namespace prosper

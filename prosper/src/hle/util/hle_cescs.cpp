// libSceCesCs — single-character CP1252<->UTF-8 conversion (sceCesSbcToUtf8,
// sceCesUtf8ToSbc, sceCesRefersUcsProfileCp1252).
//
// All three were unregistered, so the dispatcher answered `0`. For the two converters 0 is
// success, and their whole job is filling a caller buffer — the unwritten-buffer shape again,
// except here the buffer is 1-3 bytes rather than an image or a struct.
//
// Evidence: the shipped 3.20 module (libSceCesCs.sprx, plaintext; offsets are image RVAs).
//   sceCesRefersUcsProfileCp1252 0x1280 returns the profile at .data 0x845b0. A profile is 0x20
//     bytes: u16 code page (1252) @0, u16 0x0700 @2, and @0x18 a pointer to the single-byte table
//     descriptor at 0x48e70 — two optional converter functions (both null here), u16 first byte
//     (0x80), u16 count (128), then `count` u16 UCS values. A zero entry is a byte with no
//     character: the five CP1252 holes 0x81/0x8D/0x8F/0x90/0x9D.
//   sceCesSbcToUtf8 0x6e0: byte->UCS through the profile (0x2d0), then UTF-8 encode (0x1c7d0).
//   sceCesUtf8ToSbc 0x750: UTF-8 decode (0x1bd80, forms of 1-6 bytes), then UCS->byte (0x470).
// Every behaviour below — error codes, which out-parameter is written on which failure, the
// optional length pointers, the u32 length arguments — was read from those functions.
// CONFIDENCE: HIGH for profiles of the module's table shape, which includes the one this file
// hands out. A guest-built profile that supplies its own converter function is not run (that
// would mean calling guest code from inside an HLE handler); it is refused fail-visibly.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/guest_memory_copy.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace prosper {

using host::guest_read_exact;
using host::guest_write_exact;

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

namespace {

// The module's 0x805C00xx facility. Returned zero-extended, as `mov $0x805c00xx,%eax` leaves it.
constexpr uint32_t kCesErrProfile = 0x805C0004u;    // null profile
constexpr uint32_t kCesErrNoInput = 0x805C0010u;    // null or zero-length UTF-8 input
constexpr uint32_t kCesErrTruncated = 0x805C0011u;  // input ends inside a sequence
constexpr uint32_t kCesErrInvalid = 0x805C0014u;    // bad lead/continuation; byte past the table
constexpr uint32_t kCesErrIllegal = 0x805C0015u;    // overlong form or surrogate
constexpr uint32_t kCesErrUnmappable = 0x805C0020u; // no byte for the character / no character
constexpr uint32_t kCesErrNullOutput = 0x805C0030u;
constexpr uint32_t kCesErrOutputSmall = 0x805C0031u;

struct CesSbcTable {
    uint64_t to_ucs;    // optional custom byte->UCS converter; null for table profiles
    uint64_t from_ucs;  // optional custom UCS->byte converter; null for table profiles
    uint16_t first;     // bytes below this map to themselves
    uint16_t count;
    uint16_t ucs[128];  // ucs[b - first]; 0 = the byte has no character
};
static_assert(offsetof(CesSbcTable, first) == 0x10 && offsetof(CesSbcTable, ucs) == 0x14,
              "descriptor layout read by the module at 0x2e8/0x336/0x4fa");

struct CesProfile {
    uint16_t code_page;
    uint16_t tag;  // 0x0700 in every shipped profile; no converter reads it
    uint8_t reserved[0x14];
    const CesSbcTable* table;
};
static_assert(offsetof(CesProfile, table) == 0x18 && sizeof(CesProfile) == 0x20,
              "profile layout read by the module at 0x2df");

// Windows-1252 0x80-0xFF, byte for byte the module's table at 0x48e84 (0x00-0x7F map to
// themselves through `first`).
const CesSbcTable kCp1252Table = {
    0, 0, 0x80, 128,
    {
        0x20ac, 0x0000, 0x201a, 0x0192, 0x201e, 0x2026, 0x2020, 0x2021,  // 0x80
        0x02c6, 0x2030, 0x0160, 0x2039, 0x0152, 0x0000, 0x017d, 0x0000,  // 0x88
        0x0000, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,  // 0x90
        0x02dc, 0x2122, 0x0161, 0x203a, 0x0153, 0x0000, 0x017e, 0x0178,  // 0x98
        0x00a0, 0x00a1, 0x00a2, 0x00a3, 0x00a4, 0x00a5, 0x00a6, 0x00a7,  // 0xA0
        0x00a8, 0x00a9, 0x00aa, 0x00ab, 0x00ac, 0x00ad, 0x00ae, 0x00af,  // 0xA8
        0x00b0, 0x00b1, 0x00b2, 0x00b3, 0x00b4, 0x00b5, 0x00b6, 0x00b7,  // 0xB0
        0x00b8, 0x00b9, 0x00ba, 0x00bb, 0x00bc, 0x00bd, 0x00be, 0x00bf,  // 0xB8
        0x00c0, 0x00c1, 0x00c2, 0x00c3, 0x00c4, 0x00c5, 0x00c6, 0x00c7,  // 0xC0
        0x00c8, 0x00c9, 0x00ca, 0x00cb, 0x00cc, 0x00cd, 0x00ce, 0x00cf,  // 0xC8
        0x00d0, 0x00d1, 0x00d2, 0x00d3, 0x00d4, 0x00d5, 0x00d6, 0x00d7,  // 0xD0
        0x00d8, 0x00d9, 0x00da, 0x00db, 0x00dc, 0x00dd, 0x00de, 0x00df,  // 0xD8
        0x00e0, 0x00e1, 0x00e2, 0x00e3, 0x00e4, 0x00e5, 0x00e6, 0x00e7,  // 0xE0
        0x00e8, 0x00e9, 0x00ea, 0x00eb, 0x00ec, 0x00ed, 0x00ee, 0x00ef,  // 0xE8
        0x00f0, 0x00f1, 0x00f2, 0x00f3, 0x00f4, 0x00f5, 0x00f6, 0x00f7,  // 0xF0
        0x00f8, 0x00f9, 0x00fa, 0x00fb, 0x00fc, 0x00fd, 0x00fe, 0x00ff,  // 0xF8
    },
};
const CesProfile kCp1252Profile = {1252, 0x0700, {}, &kCp1252Table};

void put_u32(uint64_t addr, uint32_t v) {
    if (addr) guest_write_exact(addr, &v, sizeof v);
}
void put_u8(uint64_t addr, uint8_t v) {
    if (addr) guest_write_exact(addr, &v, sizeof v);
}

void warn_custom_converter() {
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        std::fprintf(stderr, "[prosper] libSceCesCs: profile supplies its own converter function; "
                             "not supported, answering 0x805C0020\n");
    }
}

// The profile's table descriptor header, read through guest memory so a profile the guest built
// itself in the module's layout is honoured too. False when there is no descriptor (identity).
struct CesTableHeader {
    uint64_t to_ucs, from_ucs;
    uint16_t first, count;
};
bool ces_table_header(uint64_t profile, uint64_t* table_addr, CesTableHeader* hdr) {
    uint64_t table = 0;
    if (!guest_read_exact(profile + offsetof(CesProfile, table), &table, sizeof table)) return false;
    if (!table || !guest_read_exact(table, hdr, sizeof *hdr)) return false;
    *table_addr = table;
    return true;
}

// Byte -> UCS (0x2d0). The caller writes the zero/len outputs on failure.
uint32_t ces_sbc_to_ucs(uint64_t profile, uint8_t b, uint32_t* ucs) {
    *ucs = 0;
    if (!profile) return kCesErrProfile;
    uint64_t table = 0;
    CesTableHeader hdr{};
    if (!ces_table_header(profile, &table, &hdr) || b < hdr.first) {
        *ucs = b;
        return 0;
    }
    if (hdr.to_ucs) {
        warn_custom_converter();
        return kCesErrUnmappable;
    }
    const uint32_t idx = b - hdr.first;
    if (idx >= hdr.count) return kCesErrInvalid;
    uint16_t entry = 0;
    if (!guest_read_exact(table + offsetof(CesSbcTable, ucs) + idx * 2ull, &entry, sizeof entry)) {
        return kCesErrUnmappable;
    }
    if (entry == 0) return kCesErrUnmappable;
    *ucs = entry;
    return 0;
}

// UCS -> byte (0x470). Writes *sbc = 0 on every failure but a null output.
uint32_t ces_ucs_to_sbc(uint32_t ucs, uint64_t profile, uint64_t sbc_addr) {
    uint32_t err = 0;
    if ((int32_t)ucs < 0) err = kCesErrInvalid;
    else if ((ucs & 0xFFFFF800u) == 0xD800u) err = kCesErrIllegal;
    else if (!profile) err = kCesErrProfile;
    if (err) {
        put_u8(sbc_addr, 0);
        return err;
    }
    uint64_t table = 0;
    CesTableHeader hdr{};
    uint32_t byte = 0;
    if (!ces_table_header(profile, &table, &hdr)) {
        if (ucs >= 0x100) err = kCesErrUnmappable;
        byte = ucs;
    } else if (hdr.first > ucs) {
        byte = ucs;
    } else if (hdr.from_ucs) {
        warn_custom_converter();
        err = kCesErrUnmappable;
    } else {
        err = kCesErrUnmappable;
        for (uint32_t i = 0; i < hdr.count; i++) {
            uint16_t entry = 0;
            if (!guest_read_exact(table + offsetof(CesSbcTable, ucs) + i * 2ull, &entry,
                                  sizeof entry)) {
                break;
            }
            if (entry == ucs) {
                byte = i + hdr.first;
                err = 0;
                break;
            }
        }
    }
    if (err) {
        put_u8(sbc_addr, 0);
        return err;
    }
    if (!sbc_addr) return kCesErrNullOutput;
    if (!guest_write_exact(sbc_addr, &byte, 1)) return kCesErrNullOutput;
    return 0;
}

uint32_t ces_utf8_length(uint32_t ucs) {
    if (ucs < 0x80) return 1;
    if (ucs < 0x800) return 2;
    if (ucs < 0x10000) return 3;
    if (ucs < 0x200000) return 4;
    if (ucs < 0x4000000) return 5;
    return 6;
}

// UTF-8 encode (0x1c7d0). `*len` (optional) receives the encoded length, or the length that
// was needed when the output is null or too small.
uint32_t ces_encode_utf8(uint32_t ucs, uint64_t out, uint32_t max, uint64_t len_addr) {
    if ((int32_t)ucs < 0) {
        put_u32(len_addr, 0);
        return kCesErrInvalid;
    }
    const uint32_t n = ces_utf8_length(ucs);
    const bool surrogate = (ucs & 0xFFFFF800u) == 0xD800u;
    if (!out || max == 0) {
        put_u32(len_addr, n);
        if (out) return kCesErrOutputSmall;
        return surrogate ? kCesErrIllegal : kCesErrNullOutput;
    }
    if (n > max) {
        put_u32(len_addr, n);
        return kCesErrOutputSmall;
    }
    static const uint8_t kLead[7] = {0, 0x00, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC};
    uint8_t enc[6];
    uint32_t v = ucs;
    for (uint32_t i = n - 1; i > 0; i--) {
        enc[i] = (uint8_t)(0x80 | (v & 0x3F));
        v >>= 6;
    }
    enc[0] = (uint8_t)(kLead[n] | v);
    if (!guest_write_exact(out, enc, n)) return kCesErrNullOutput;
    put_u32(len_addr, n);
    return surrogate ? kCesErrIllegal : 0;
}

// UTF-8 decode of one character (0x1bd80). `*used` (optional) is the sequence length on success,
// on an overlong/surrogate form and on truncation; the index of the offending byte for a bad
// continuation (one more than that when the input was also short); 0 for a bad lead byte.
uint32_t ces_decode_utf8(uint64_t src, uint32_t avail, uint64_t used_addr, uint32_t* ucs) {
    *ucs = 0;
    uint8_t b[6] = {};
    if (!src || avail == 0 || !guest_read_exact(src, b, 1)) {
        put_u32(used_addr, 0);
        return kCesErrNoInput;
    }
    if (b[0] < 0x80) {
        put_u32(used_addr, 1);
        *ucs = b[0];
        return 0;
    }
    uint32_t n = 0;
    if (b[0] < 0xC0) n = 0;
    else if (b[0] < 0xE0) n = 2;
    else if (b[0] < 0xF0) n = 3;
    else if (b[0] < 0xF8) n = 4;
    else if (b[0] < 0xFC) n = 5;
    else if (b[0] < 0xFE) n = 6;
    if (n == 0) {
        put_u32(used_addr, 0);
        return kCesErrInvalid;
    }
    const uint32_t have = avail < n ? avail : n;
    if (have > 1 && !guest_read_exact(src + 1, b + 1, have - 1)) {
        put_u32(used_addr, 0);
        return kCesErrNoInput;
    }
    for (uint32_t i = 1; i < have; i++) {
        if ((b[i] & 0xC0) != 0x80) {
            put_u32(used_addr, avail < n ? i + 1 : i);
            return kCesErrInvalid;
        }
    }
    if (avail < n) {
        put_u32(used_addr, n);
        return kCesErrTruncated;
    }
    uint32_t cp = b[0] & (0x7Fu >> n);
    for (uint32_t i = 1; i < n; i++) cp = (cp << 6) | (b[i] & 0x3Fu);
    put_u32(used_addr, n);
    static const uint32_t kMin[7] = {0, 0, 0x80, 0x800, 0x10000, 0x200000, 0x4000000};
    *ucs = cp;
    if (cp < kMin[n] || (n == 3 && (cp & 0xFFFFF800u) == 0xD800u)) return kCesErrIllegal;
    return 0;
}

}  // namespace

// sceCesRefersUcsProfileCp1252() -> the CP1252 UCS profile.
HLE(ces_refers_ucs_profile_cp1252) {
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return (uint64_t)(uintptr_t)&kCp1252Profile;
}

// sceCesSbcToUtf8(profile, sbc, utf8*, u32 utf8max, u32* utf8_len) -> 0; *len = bytes written.
HLE(ces_sbc_to_utf8) {
    (void)a5;
    uint32_t ucs = 0;
    if (const uint32_t err = ces_sbc_to_ucs(a0, (uint8_t)a1, &ucs)) {
        put_u32(a4, 0);
        return err;
    }
    return ces_encode_utf8(ucs, a2, (uint32_t)a3, a4);
}

// sceCesUtf8ToSbc(utf8*, u32 utf8max, u32* utf8_used, profile, sbc*) -> 0; *used = bytes read.
HLE(ces_utf8_to_sbc) {
    (void)a5;
    uint32_t ucs = 0;
    if (const uint32_t err = ces_decode_utf8(a0, (uint32_t)a1, a2, &ucs)) {
        put_u8(a4, 0);
        return err;
    }
    return ces_ucs_to_sbc(ucs, a3, a4);
}

void register_cescs_hle() {
    Hle::register_fn("LPzYZ+FR0BI", (HleFn)ces_refers_ucs_profile_cp1252,
                     "sceCesRefersUcsProfileCp1252");
    Hle::register_fn("xTd54EEL1Ao", (HleFn)ces_sbc_to_utf8, "sceCesSbcToUtf8");
    Hle::register_fn("3Q1gOWWarcw", (HleFn)ces_utf8_to_sbc, "sceCesUtf8ToSbc");
}

}  // namespace prosper

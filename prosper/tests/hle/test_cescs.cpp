// test_cescs — libSceCesCs single-character CP1252<->UTF-8 conversion must convert, and fail,
// the way the shipped module does.
//
// All three exports were unregistered, so the dispatcher answered `0` — success — while
// writing nothing: a title converting text for display or storage proceeds on untouched
// 1-3 byte buffers. Every arm drives the real NIDs. Expected codes, out-parameter writes on
// failure and the optional length pointers come from the module's disassembly (offsets cited in
// src/hle/util/hle_cescs.cpp).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

static constexpr uint64_t kErrProfile = 0x805C0004ull;
static constexpr uint64_t kErrNoInput = 0x805C0010ull;
static constexpr uint64_t kErrTruncated = 0x805C0011ull;
static constexpr uint64_t kErrInvalid = 0x805C0014ull;
static constexpr uint64_t kErrIllegal = 0x805C0015ull;
static constexpr uint64_t kErrUnmappable = 0x805C0020ull;
static constexpr uint64_t kErrNullOutput = 0x805C0030ull;
static constexpr uint64_t kErrOutputSmall = 0x805C0031ull;

// The five bytes Windows-1252 leaves undefined.
static constexpr uint8_t kHoles[5] = {0x81, 0x8D, 0x8F, 0x90, 0x9D};

namespace {

struct Api {
    HleFn profile = nullptr, to_utf8 = nullptr, to_sbc = nullptr;
    uint64_t cp1252 = 0;
};
Api registered() {
    register_builtin_hle();
    Api api;
    api.profile = Hle::lookup(nid_hash("sceCesRefersUcsProfileCp1252"));
    api.to_utf8 = Hle::lookup(nid_hash("sceCesSbcToUtf8"));
    api.to_sbc = Hle::lookup(nid_hash("sceCesUtf8ToSbc"));
    if (api.profile) api.cp1252 = api.profile(0, 0, 0, 0, 0, 0);
    return api;
}
bool api_ok(const Api& api) {
    return api.profile && api.to_utf8 && api.to_sbc && api.cp1252;
}

}  // namespace

TEST(CesCs, NidsBound) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("sceCesRefersUcsProfileCp1252")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("sceCesSbcToUtf8")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("sceCesUtf8ToSbc")), nullptr);
}

TEST(CesCs, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("sceCesRefersUcsProfileCp1252"), "LPzYZ+FR0BI");
    EXPECT_EQ(nid_hash("sceCesSbcToUtf8"), "xTd54EEL1Ao");
    EXPECT_EQ(nid_hash("sceCesUtf8ToSbc"), "3Q1gOWWarcw");
    EXPECT_NE(nid_hash("sceCesSbcToUtf8"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

// The profile has the module's layout: code page 1252, and at +0x18 a descriptor whose table
// covers 0x80..0xFF.
TEST(CesCs, ProfileHasTheModulesLayout) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    EXPECT_EQ(api.profile(0, 0, 0, 0, 0, 0), api.cp1252) << "the profile is stable";
    uint16_t code_page = 0;
    std::memcpy(&code_page, (const void*)(uintptr_t)api.cp1252, 2);
    EXPECT_EQ(code_page, 1252u);
    uint64_t table = 0;
    std::memcpy(&table, (const void*)(uintptr_t)(api.cp1252 + 0x18), 8);
    ASSERT_NE(table, 0u);
    uint16_t first = 0, count = 0, euro = 0;
    std::memcpy(&first, (const void*)(uintptr_t)(table + 0x10), 2);
    std::memcpy(&count, (const void*)(uintptr_t)(table + 0x12), 2);
    std::memcpy(&euro, (const void*)(uintptr_t)(table + 0x14), 2);
    EXPECT_EQ(first, 0x80u);
    EXPECT_EQ(count, 128u);
    EXPECT_EQ(euro, 0x20ACu);
}

TEST(CesCs, SbcToUtf8Converts) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    // 'A' -> 1 byte; U+00E9 (0xE9) -> 2 bytes; U+20AC (0x80) -> 3 bytes.
    uint8_t out[4]{};
    uint32_t len = 0xDEADu;
    ASSERT_EQ(api.to_utf8(api.cp1252, 'A', addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 1u);
    EXPECT_EQ(out[0], 'A');
    ASSERT_EQ(api.to_utf8(api.cp1252, 0xE9, addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 2u);
    EXPECT_EQ(out[0], 0xC3);
    EXPECT_EQ(out[1], 0xA9);
    ASSERT_EQ(api.to_utf8(api.cp1252, 0x80, addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 3u);
    EXPECT_EQ(out[0], 0xE2);
    EXPECT_EQ(out[1], 0x82);
    EXPECT_EQ(out[2], 0xAC);
    // The byte is the low 8 bits of the register.
    ASSERT_EQ(api.to_utf8(api.cp1252, 0x1241, addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(out[0], 'A');
}

// C3: the length pointer is optional.
TEST(CesCs, SbcToUtf8LengthIsOptional) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    uint8_t out[4]{};
    ASSERT_EQ(api.to_utf8(api.cp1252, 0x80, addr(out), sizeof out, 0, 0), 0u);
    EXPECT_EQ(out[0], 0xE2);
    EXPECT_EQ(out[2], 0xAC);
}

// C2/C4/C5: failure codes and what each failure writes.
TEST(CesCs, SbcToUtf8Failures) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    uint8_t out[4];
    uint32_t len = 0;

    // Too small: the needed length is reported, the output untouched.
    std::memset(out, 0xBB, sizeof out);
    len = 0xDEADu;
    EXPECT_EQ(api.to_utf8(api.cp1252, 0x80, addr(out), 2, addr(&len), 0), kErrOutputSmall);
    EXPECT_EQ(len, 3u);
    EXPECT_EQ(out[0], 0xBB);
    // utf8max is a u32: 2^32 + 4 is 4, and 2^32 is 0.
    EXPECT_EQ(api.to_utf8(api.cp1252, 0x80, addr(out), (1ull << 32) | 4, addr(&len), 0), 0u);
    len = 0xDEADu;
    EXPECT_EQ(api.to_utf8(api.cp1252, 0x80, addr(out), 1ull << 32, addr(&len), 0), kErrOutputSmall);
    EXPECT_EQ(len, 3u);
    // Null output: the needed length is reported too.
    len = 0xDEADu;
    EXPECT_EQ(api.to_utf8(api.cp1252, 0xE9, 0, sizeof out, addr(&len), 0), kErrNullOutput);
    EXPECT_EQ(len, 2u);
    // Null profile: *len = 0.
    len = 0xDEADu;
    EXPECT_EQ(api.to_utf8(0, 'A', addr(out), sizeof out, addr(&len), 0), kErrProfile);
    EXPECT_EQ(len, 0u);
}

// C1: the five undefined CP1252 bytes have no character, in either direction.
TEST(CesCs, UndefinedBytesAreUnmappableBothWays) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    for (uint8_t b : kHoles) {
        uint8_t out[4];
        std::memset(out, 0xBB, sizeof out);
        uint32_t len = 0xDEADu;
        EXPECT_EQ(api.to_utf8(api.cp1252, b, addr(out), sizeof out, addr(&len), 0), kErrUnmappable)
            << "byte " << (int)b;
        EXPECT_EQ(len, 0u) << "byte " << (int)b;
        EXPECT_EQ(out[0], 0xBB) << "byte " << (int)b;

        // ...and its Latin-1 code point (U+0081 etc.) has no byte.
        const uint8_t utf8[2] = {0xC2, b};
        uint32_t used = 0;
        uint8_t sbc = 0xEE;
        EXPECT_EQ(api.to_sbc(addr(utf8), 2, addr(&used), api.cp1252, addr(&sbc), 0), kErrUnmappable)
            << "U+00" << std::hex << (int)b;
        EXPECT_EQ(sbc, 0u);
        EXPECT_EQ(used, 2u);
    }
    // U+FFFD, which a careless table uses for those holes, is not a byte either.
    const uint8_t fffd[3] = {0xEF, 0xBF, 0xBD};
    uint32_t used = 0;
    uint8_t sbc = 0xEE;
    EXPECT_EQ(api.to_sbc(addr(fffd), 3, addr(&used), api.cp1252, addr(&sbc), 0), kErrUnmappable);
    EXPECT_EQ(sbc, 0u);
    EXPECT_EQ(used, 3u);
}

TEST(CesCs, Utf8ToSbcConverts) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    auto run = [&](const uint8_t* in, uint64_t max, uint32_t* used, uint8_t* sbc) {
        return api.to_sbc(addr(in), max, addr(used), api.cp1252, addr(sbc), 0);
    };
    uint32_t used = 0xDEADu;
    uint8_t sbc = 0;
    const uint8_t a[] = {'A', 'B'};
    ASSERT_EQ(run(a, 2, &used, &sbc), 0u);
    EXPECT_EQ(sbc, 'A');
    EXPECT_EQ(used, 1u);
    const uint8_t e_acute[] = {0xC3, 0xA9};
    ASSERT_EQ(run(e_acute, 2, &used, &sbc), 0u);
    EXPECT_EQ(sbc, 0xE9);
    EXPECT_EQ(used, 2u);
    const uint8_t euro[] = {0xE2, 0x82, 0xAC};
    ASSERT_EQ(run(euro, 3, &used, &sbc), 0u);
    EXPECT_EQ(sbc, 0x80);
    EXPECT_EQ(used, 3u);
    // C3: the used pointer is optional.
    sbc = 0;
    ASSERT_EQ(api.to_sbc(addr(euro), 3, 0, api.cp1252, addr(&sbc), 0), 0u);
    EXPECT_EQ(sbc, 0x80);
    // C5: the length is a u32 — 2^32 + 3 is 3.
    ASSERT_EQ(run(euro, (1ull << 32) | 3, &used, &sbc), 0u);
    EXPECT_EQ(sbc, 0x80);
}

// C2/C4: every failure class, its code, *used, and *sbc = 0.
TEST(CesCs, Utf8ToSbcFailures) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    uint32_t used = 0;
    uint8_t sbc = 0;
    auto run = [&](const uint8_t* in, uint64_t max) {
        used = 0xDEADu;
        sbc = 0xEE;
        return api.to_sbc(addr(in), max, addr(&used), api.cp1252, addr(&sbc), 0);
    };
    const uint8_t a[] = {'A'};
    EXPECT_EQ(run(nullptr, 1), kErrNoInput) << "null input";
    EXPECT_EQ(used, 0u);
    EXPECT_EQ(sbc, 0u);
    EXPECT_EQ(run(a, 0), kErrNoInput) << "empty input";
    EXPECT_EQ(used, 0u);
    EXPECT_EQ(run(a, 1ull << 32), kErrNoInput) << "a u32 length of 0";

    const uint8_t cont[] = {0x80};
    EXPECT_EQ(run(cont, 1), kErrInvalid) << "continuation byte as lead";
    EXPECT_EQ(used, 0u);
    EXPECT_EQ(sbc, 0u);
    const uint8_t ff[] = {0xFF};
    EXPECT_EQ(run(ff, 1), kErrInvalid) << "0xFF lead";
    EXPECT_EQ(used, 0u);

    const uint8_t bad2[] = {0xC3, 0x41};
    EXPECT_EQ(run(bad2, 2), kErrInvalid) << "bad continuation";
    EXPECT_EQ(used, 1u) << "index of the offending byte";
    const uint8_t bad3[] = {0xE2, 0x82, 0x41};
    EXPECT_EQ(run(bad3, 3), kErrInvalid);
    EXPECT_EQ(used, 2u);

    const uint8_t e_acute[] = {0xC3, 0xA9};
    EXPECT_EQ(run(e_acute, 1), kErrTruncated);
    EXPECT_EQ(used, 2u) << "the expected sequence length";
    EXPECT_EQ(sbc, 0u);
    const uint8_t euro[] = {0xE2, 0x82, 0xAC};
    EXPECT_EQ(run(euro, 2), kErrTruncated);
    EXPECT_EQ(used, 3u);
    EXPECT_EQ(run(bad3, 2), kErrTruncated) << "the bytes present are valid";
    const uint8_t short_bad[] = {0xE2, 0x41};
    EXPECT_EQ(run(short_bad, 2), kErrInvalid) << "truncated AND a bad continuation";
    EXPECT_EQ(used, 2u);

    const uint8_t overlong[] = {0xC0, 0xAF};
    EXPECT_EQ(run(overlong, 2), kErrIllegal);
    EXPECT_EQ(used, 2u);
    EXPECT_EQ(sbc, 0u);
    const uint8_t overlong3[] = {0xE0, 0x81, 0x81};
    EXPECT_EQ(run(overlong3, 3), kErrIllegal);
    const uint8_t surrogate[] = {0xED, 0xA0, 0x80};
    EXPECT_EQ(run(surrogate, 3), kErrIllegal);
    EXPECT_EQ(used, 3u);

    const uint8_t cjk[] = {0xE4, 0xB8, 0xAD};  // U+4E2D, no CP1252 byte
    EXPECT_EQ(run(cjk, 3), kErrUnmappable);
    EXPECT_EQ(used, 3u);
    EXPECT_EQ(sbc, 0u);
    const uint8_t astral[] = {0xF0, 0x9F, 0x98, 0x80};  // U+1F600: decoded, then unmappable
    EXPECT_EQ(run(astral, 4), kErrUnmappable);
    EXPECT_EQ(used, 4u);
    const uint8_t five[] = {0xF8, 0x88, 0x80, 0x80, 0x80};  // 0x200000, a 5-byte form
    EXPECT_EQ(run(five, 5), kErrUnmappable);
    EXPECT_EQ(used, 5u);
    const uint8_t five_overlong[] = {0xF8, 0x80, 0x80, 0x80, 0x80};
    EXPECT_EQ(run(five_overlong, 5), kErrIllegal);
    EXPECT_EQ(run(five, 3), kErrTruncated);
    EXPECT_EQ(used, 5u);

    // Null profile is checked after the decode; null output after the mapping.
    used = 0;
    sbc = 0xEE;
    EXPECT_EQ(api.to_sbc(addr(a), 1, addr(&used), 0, addr(&sbc), 0), kErrProfile);
    EXPECT_EQ(sbc, 0u);
    EXPECT_EQ(used, 1u);
    EXPECT_EQ(api.to_sbc(addr(ff), 1, addr(&used), 0, addr(&sbc), 0), kErrInvalid)
        << "a decode failure wins over a null profile";
    EXPECT_EQ(api.to_sbc(addr(a), 1, addr(&used), api.cp1252, 0, 0), kErrNullOutput);
    EXPECT_EQ(api.to_sbc(addr(euro), 3, addr(&used), api.cp1252, 0, 0), kErrNullOutput);
}

TEST(CesCs, RoundTripAllMappableBytes) {
    const Api api = registered();
    ASSERT_TRUE(api_ok(api));
    int round_trips = 0;
    for (int b = 0; b < 256; b++) {
        bool hole = false;
        for (uint8_t h : kHoles) hole |= b == h;
        uint8_t enc[4]{};
        uint32_t len = 0;
        const uint64_t r = api.to_utf8(api.cp1252, (uint64_t)b, addr(enc), sizeof enc, addr(&len), 0);
        if (hole) {
            EXPECT_EQ(r, kErrUnmappable) << "byte " << b;
            continue;
        }
        ASSERT_EQ(r, 0u) << "byte " << b;
        uint32_t used = 0;
        uint8_t back = 0;
        ASSERT_EQ(api.to_sbc(addr(enc), len, addr(&used), api.cp1252, addr(&back), 0), 0u)
            << "byte " << b;
        EXPECT_EQ(back, (uint8_t)b) << "round trip of byte " << b;
        EXPECT_EQ(used, len);
        round_trips++;
    }
    EXPECT_EQ(round_trips, 251);
}

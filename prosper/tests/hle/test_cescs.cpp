// test_cescs — libSceCesCs single-character CP1252<->UTF-8 conversion must convert.
//
// All three exports were unregistered, so the dispatcher answered `0` — success — while
// writing nothing: a title converting text for display or storage proceeds on untouched
// 1-3 byte buffers. Every arm drives the real NIDs and asserts WRITTEN bytes against the
// public CP1252 table, plus refusal arms for the failure polarity.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static uint64_t addr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}
static constexpr uint64_t kCesErr = 0xFFFFFFFFull;

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

TEST(CesCs, SbcToUtf8Converts) {
    register_builtin_hle();
    HleFn profile_fn = Hle::lookup(nid_hash("sceCesRefersUcsProfileCp1252"));
    HleFn convert = Hle::lookup(nid_hash("sceCesSbcToUtf8"));
    ASSERT_NE(profile_fn, nullptr);
    ASSERT_NE(convert, nullptr);
    const uint64_t profile = profile_fn(0, 0, 0, 0, 0, 0);
    EXPECT_NE(profile, 0u) << "the profile is a real table address";
    EXPECT_EQ(profile_fn(0, 0, 0, 0, 0, 0), profile) << "the profile is stable";

    // 'A' -> 1 byte; U+00E9 (0xE9) -> 2 bytes; U+20AC (0x80) -> 3 bytes.
    uint8_t out[4]{};
    uint32_t len = 0xDEADu;
    ASSERT_EQ(convert(profile, 'A', addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 1u);
    EXPECT_EQ(out[0], 'A');
    ASSERT_EQ(convert(profile, 0xE9, addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 2u);
    EXPECT_EQ(out[0], 0xC3);
    EXPECT_EQ(out[1], 0xA9);
    ASSERT_EQ(convert(profile, 0x80, addr(out), sizeof out, addr(&len), 0), 0u);
    EXPECT_EQ(len, 3u);
    EXPECT_EQ(out[0], 0xE2);
    EXPECT_EQ(out[1], 0x82);
    EXPECT_EQ(out[2], 0xAC);
    // Too small a buffer fails WITHOUT touching the output.
    std::memset(out, 0xBB, sizeof out);
    len = 0xDEADu;
    EXPECT_EQ(convert(profile, 0x80, addr(out), 2, addr(&len), 0), kCesErr);
    EXPECT_EQ(out[0], 0xBB);
    EXPECT_EQ(len, 0xDEADu);
    // Nulls and a foreign profile fail.
    EXPECT_EQ(convert(0xDEADu, 'A', addr(out), sizeof out, addr(&len), 0), kCesErr);
    EXPECT_EQ(convert(profile, 'A', 0, sizeof out, addr(&len), 0), kCesErr);
    EXPECT_EQ(convert(profile, 'A', addr(out), sizeof out, 0, 0), kCesErr);
}

TEST(CesCs, Utf8ToSbcConverts) {
    register_builtin_hle();
    HleFn profile_fn = Hle::lookup(nid_hash("sceCesRefersUcsProfileCp1252"));
    HleFn convert = Hle::lookup(nid_hash("sceCesUtf8ToSbc"));
    ASSERT_NE(profile_fn, nullptr);
    ASSERT_NE(convert, nullptr);
    const uint64_t profile = profile_fn(0, 0, 0, 0, 0, 0);

    auto run = [&](const uint8_t* in, uint32_t max, uint32_t* used, uint8_t* sbc) {
        return convert(addr(in), max, addr(used), profile, addr(sbc), 0);
    };
    uint32_t used = 0xDEADu;
    uint8_t sbc = 0;
    const uint8_t a[] = {'A'};
    ASSERT_EQ(run(a, 1, &used, &sbc), 0u);
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
    // Invalid, overlong, truncated and unmappable input all fail.
    const uint8_t bad[] = {0xFF};
    sbc = 0xEE;
    EXPECT_EQ(run(bad, 1, &used, &sbc), kCesErr);
    EXPECT_EQ(sbc, 0xEE) << "failed conversion writes no byte";
    const uint8_t overlong[] = {0xC0, 0xAF};
    EXPECT_EQ(run(overlong, 2, &used, &sbc), kCesErr) << "overlong rejected";
    EXPECT_EQ(run(e_acute, 1, &used, &sbc), kCesErr) << "truncated input rejected";
    const uint8_t cjk[] = {0xE4, 0xB8, 0xAD};  // U+4E2D, no CP1252 byte
    EXPECT_EQ(run(cjk, 3, &used, &sbc), kCesErr) << "unmappable rejected";
    EXPECT_EQ(run(a, 1, &used, nullptr), kCesErr) << "null out-byte";
    EXPECT_EQ(run(nullptr, 1, &used, &sbc), kCesErr) << "null input";
}

TEST(CesCs, RoundTripAllMappableBytes) {
    register_builtin_hle();
    HleFn profile_fn = Hle::lookup(nid_hash("sceCesRefersUcsProfileCp1252"));
    HleFn to_utf8 = Hle::lookup(nid_hash("sceCesSbcToUtf8"));
    HleFn to_sbc = Hle::lookup(nid_hash("sceCesUtf8ToSbc"));
    ASSERT_NE(profile_fn, nullptr);
    ASSERT_NE(to_utf8, nullptr);
    ASSERT_NE(to_sbc, nullptr);
    const uint64_t profile = profile_fn(0, 0, 0, 0, 0, 0);
    // Every byte except the five undefined ones must survive sbc->utf8->sbc.
    for (int b = 0; b < 256; b++) {
        if (b == 0x81 || b == 0x8D || b == 0x8F || b == 0x90 || b == 0x9D) continue;
        uint8_t enc[4]{};
        uint32_t len = 0;
        ASSERT_EQ(to_utf8(profile, (uint64_t)b, addr(enc), sizeof enc, addr(&len), 0), 0u)
            << "byte " << b;
        uint32_t used = 0;
        uint8_t back = 0;
        ASSERT_EQ(to_sbc(addr(enc), len, addr(&used), profile, addr(&back), 0), 0u) << "byte " << b;
        EXPECT_EQ(back, (uint8_t)b) << "round trip of byte " << b;
        EXPECT_EQ(used, len);
    }
}

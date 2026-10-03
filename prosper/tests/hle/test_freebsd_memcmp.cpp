// test_freebsd_memcmp — FreeBSD libc contract for memcmp/bcmp, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/memcmp_test.c (ATF) and
// bcmp_test.c (which reuses memcmp_test.c with RES(x) = (x != 0)).
// Original authors: Jilles Tjoelker, Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: src/hle/libc/hle_libc.cpp forwards h_memcmp and
// h_bcmp straight to the host memcmp. The guest-visible contract is only the
// SIGN of the result (C17 7.24.4.1: less/equal/greater than zero), except
// bcmp which is 0/non-0. A host memcmp returning the raw byte difference
// would still satisfy this; one returning an inverted sign would corrupt
// every guest memcmp/parse path. These arms pin the sign, the zero-length
// rule, and unsigned-byte comparison (0xFF > 0x00).
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

using namespace prosper;

static int sign_of(int x) {
    return (x > 0) - (x < 0);
}

static unsigned failures = 0;
#define CHECK_SIGN(got, want, msg)                                                                 \
    do {                                                                                           \
        if (sign_of(got) != sign_of(want)) {                                                       \
            ++failures;                                                                            \
            ADD_FAILURE() << (msg) << ": got " << (got) << " want sign " << sign_of(want);         \
        }                                                                                          \
    } while (0)

TEST(FreebsdMemcmp, ZeroLengthIsEqual) {
    // Port of ATF zero: any pointers with len 0 compare equal.
    CHECK_SIGN(memcmp("a", "b", 0), 0, "memcmp len 0");
    CHECK_SIGN(memcmp("", "", 0), 0, "memcmp empty len 0");
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdMemcmp, EqualBlocksAllLengths) {
    // Port of ATF eq: identical buffers compare 0 at every length and offset.
    unsigned char data1[256], data2[256];
    for (int i = 0; i < 256; ++i) data1[i] = data2[i] = (unsigned char)(i ^ 0x55);
    for (int i = 1; i < 256; ++i) CHECK_SIGN(memcmp(data1, data2, (size_t)i), 0, "equal prefix");
    for (int i = 1; i < 256; ++i)
        CHECK_SIGN(memcmp(data1 + i, data2 + i, (size_t)(256 - i)), 0, "equal suffix");
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdMemcmp, FirstMismatchDecidesSign) {
    // Port of ATF neq: buffers differing everywhere take the first byte's sign.
    unsigned char data1[256], data2[256];
    for (int i = 0; i < 256; ++i) {
        data1[i] = (unsigned char)i;
        data2[i] = (unsigned char)(i ^ 0x55);
    }
    for (int i = 1; i < 256; ++i)
        CHECK_SIGN(memcmp(data1, data2, (size_t)i), -0x55, "mismatch sign");
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdMemcmp, UnsignedByteComparison) {
    // Port of ATF diff: bytes compare UNSIGNED, so 255 > 0 and 'c' < 'e'.
    unsigned char data1[256], data2[256];
    memset(data1, 'a', sizeof data1);
    memset(data2, 'a', sizeof data2);
    data1[128] = 255;
    data2[128] = 0;
    for (int i = 1; i < 66; ++i) {
        CHECK_SIGN(memcmp(data1 + 128, data2 + 128, (size_t)i), 255, "0xFF sorts after 0x00");
        CHECK_SIGN(memcmp(data2 + 128, data1 + 128, (size_t)i), -255, "0x00 sorts before 0xFF");
    }
    data1[128] = 'c';
    data2[128] = 'e';
    for (int i = 1; i < 66; ++i) {
        CHECK_SIGN(memcmp(data1 + 128, data2 + 128, (size_t)i), -2, "'c' < 'e'");
        CHECK_SIGN(memcmp(data2 + 128, data1 + 128, (size_t)i), 2, "'e' > 'c'");
    }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdMemcmp, PositiveControl) {
    // The harness must be able to see a difference: hand-built mismatch outside
    // the generator above. A discriminator that cannot show its lever moved is void.
    EXPECT_NE(sign_of(memcmp("abc", "abd", 3)), 0) << "harness sees a real mismatch";
    EXPECT_EQ(sign_of(memcmp("abc", "abc", 3)), 0) << "harness sees equality";
}

TEST(FreebsdMemcmp, BcmpIsEqualityOnly) {
    // bcmp_test.c reuses memcmp_test.c with RES(x) = (x != 0): bcmp reports
    // 0 for equal and nonzero for different, with no sign promise.
    EXPECT_EQ(memcmp("abc", "abc", 3) == 0, true) << "equal blocks: bcmp 0";
    EXPECT_EQ(memcmp("abc", "abd", 3) == 0, false) << "different blocks: bcmp nonzero";
    EXPECT_EQ(memcmp("a", "b", 0) == 0, true) << "zero length: bcmp 0";
}

TEST(FreebsdHle, StringMemoryHandlersRegistered) {
    // Unregistered, these NIDs fall to the dispatcher's return-0 default,
    // which for memcmp/strcmp reads as "equal" over bytes never compared.
    register_builtin_hle();
    const char* names[] = {"memcmp", "bcmp",   "memcpy",  "memmove", "memset",
                           "memchr", "strlen", "strnlen", "strcmp",  "strncmp"};
    for (const char* n : names)
        EXPECT_NE(Hle::lookup(nid_hash(n)), nullptr) << n << " is registered";
}

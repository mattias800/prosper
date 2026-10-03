// test_freebsd_memset — FreeBSD libc contract for memset, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/memset_test.c (ATF).
// Original author: Strahinja Stanisic / FreeBSD.
//
// Why this pins prosper: h_memset forwards to the host memset, and the one
// rule the ATF suite pins is the one guest zeroing depends on: the fill is
// the int argument CONVERTED TO unsigned char, so memset(p, 0xDEADBEEF, n)
// fills 0xEF, not a trap, not a truncation error. Plus the two C-standard
// rules the original leaves implicit: memset returns its destination, and
// a zero length stores nothing.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

TEST(FreebsdMemset, IntConvertsToUnsignedChar) {
    // Port of ATF int_char_conv: 0xDEADBEEF fills 0xEF (C17 7.24.6.1).
    char b[64];
    memset(b, 0xDEADBEEF, sizeof b);
    for (size_t i = 0; i < sizeof b; ++i)
        EXPECT_EQ(b[i], (char)0xEF) << "byte " << i << " is low byte of fill";
}

TEST(FreebsdMemset, FillsAllLengthsAndAlignments) {
    // Every length 0..128 at every start alignment 0..15 over a canary field:
    // exactly the length bytes change, neighbours never do.
    for (size_t off = 0; off < 16; ++off)
        for (size_t len = 0; len <= 128; ++len) {
            char buf[16 + 128 + 16];
            memset(buf, 'Q', sizeof buf);
            EXPECT_EQ(memset(buf + off, 'Z', len), buf + off) << "memset returns dst";
            for (size_t i = 0; i < sizeof buf; ++i) {
                char want = (i >= off && i < off + len) ? 'Z' : 'Q';
                EXPECT_EQ(buf[i], want) << "off " << off << " len " << len << " byte " << i;
                if (buf[i] != want) break;
            }
        }
}

TEST(FreebsdMemset, ZeroLengthStoresNothing) {
    char b[8] = "abcdefg";
    memset(b + 3, 'X', 0);
    EXPECT_STREQ(b, "abcdefg") << "zero length changes nothing";
}

TEST(FreebsdMemset, PositiveControl) {
    char b[4] = "aaa";
    memset(b, 'b', 3);
    EXPECT_STREQ(b, "bbb") << "harness sees the fill";
}

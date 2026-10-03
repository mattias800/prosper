// test_freebsd_strrchr — FreeBSD libc contract for strrchr, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/strrchr_test.c (ATF,
// adapted from memrchr_test.c).
// Original author: Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: h_strrchr forwards to the host strrchr. Guest path
// handling (extension splitting, basename) depends on LAST-match semantics,
// on finding the terminating NUL itself, and on not seeing bytes past it.
// The nul/not_found/found arms are ported whole; the values arm (every byte
// value 0..255, catching SWAR overflow) runs over a sampled alignment grid
// instead of the full 16x64 cross product, which is hours of release-mode
// strrchr calls for identical coverage of the value dimension.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

static unsigned failures = 0;
#define CHECK_PTR(got, want, msg)                                                                  \
    do {                                                                                           \
        if ((got) != (want)) {                                                                     \
            ++failures;                                                                            \
            ADD_FAILURE() << (msg) << ": got " << (const void*)(got) << " want "                   \
                          << (const void*)(want);                                                  \
        }                                                                                          \
    } while (0)

TEST(FreebsdStrrchr, NulFindsTerminator) {
    // Port of ATF nul: searching for NUL finds the FIRST terminator even
    // when more NULs (and nonzero tails) follow it.
    char buf[1 + 15 + 64];
    buf[0] = '\0';
    memset(buf + 1, '-', sizeof buf - 1);

    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = j; k < 64; ++k) {
                buf[i + j + 1] = '\0';
                buf[i + k + 1] = '\0';
                CHECK_PTR(strrchr(buf + i + 1, '\0'), buf + i + j + 1,
                          "NUL search stops at terminator");
                buf[i + j + 1] = '-';
                buf[i + k + 1] = '-';
            }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrrchr, NotFoundIgnoresOutsideBytes) {
    // Port of ATF not_found: an 'X' before and after the string must not
    // leak in; the answer is NULL.
    char buf[1 + 15 + 64 + 2];
    buf[0] = 'X';
    memset(buf + 1, '-', sizeof buf - 1);

    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 64; ++j) {
            buf[i + j + 1] = '\0';
            buf[i + j + 2] = 'X';
            CHECK_PTR(strrchr(buf + i + 1, 'X'), nullptr, "absent char is NULL");
            buf[i + j + 1] = '-';
            buf[i + j + 2] = '-';
        }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrrchr, LastMatchWins) {
    // Port of ATF found: with two hits, the LATER one is returned.
    char buf[1 + 15 + 64 + 2];
    buf[0] = 'X';
    memset(buf + 1, '-', sizeof buf - 1);

    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 64; ++j)
            for (size_t k = 0; k < j; ++k)
                for (size_t l = 0; l <= k; ++l) {
                    buf[i + j + 1] = '\0';
                    buf[i + j + 2] = 'X';
                    char* base = buf + i + 1;
                    base[l] = 'X';
                    base[k] = 'X';
                    CHECK_PTR(strrchr(base, 'X'), base + k, "last match wins");
                    base[l] = '-';
                    base[k] = '-';
                    buf[i + j + 1] = '-';
                    buf[i + j + 2] = '-';
                }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrrchr, AllByteValues) {
    // Port of the ATF values arm over a sampled grid: every byte value must
    // be found regardless of value (catches SWAR sign/overflow bugs), at the
    // string start, middle and end, on several alignments and lengths.
    char buf[1 + 15 + 64 + 2];
    const size_t aligns[] = {0, 1, 7, 15};
    const size_t lens[] = {1, 8, 33, 64};

    for (size_t ai = 0; ai < sizeof aligns / sizeof aligns[0]; ++ai)
        for (size_t li = 0; li < sizeof lens / sizeof lens[0]; ++li)
            for (int c = 0; c <= 255; ++c) {
                char* s = buf + aligns[ai] + 1;
                size_t len = lens[li];
                s[-1] = (char)c;  // sentinel: must never match through it
                s[len] = '\0';
                s[len + 1] = 'c';
                int fill = (c == 255) ? c - 1 : c + 1;
                memset(s, fill, len);
                for (size_t pos : {size_t{0}, len / 2, len > 0 ? len - 1 : 0}) {
                    if (pos >= len) continue;
                    s[pos] = (char)c;
                    CHECK_PTR(strrchr(s, c), s + pos, "byte value found");
                    s[pos] = (char)((c == 255) ? c - 1 : c + 1);
                }
                // Absent value: a byte that is neither the fill nor NUL, so it
                // can only miss. (Probing 0 would find the terminator; probing
                // the fill would hit every byte.)
                int probe = (fill == 42) ? 43 : 42;
                CHECK_PTR(strrchr(s, probe), nullptr, "absent value misses");
            }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrrchr, PositiveControl) {
    const char* s = "abca";
    EXPECT_EQ(strrchr(s, 'a'), s + 3) << "harness takes the last hit";
    EXPECT_EQ(strrchr("abc", 'z'), nullptr) << "harness sees a miss";
    EXPECT_EQ(strrchr(s, '\0'), s + 4) << "harness finds the terminator";
}

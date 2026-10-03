// test_freebsd_strcmp — FreeBSD libc contract for strcmp, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/strcmp_test.c (ATF).
// Original author: Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: src/hle/libc/hle_libc.cpp forwards h_strcmp to the
// host strcmp. Guest string tables, symbol lookup and path comparison all
// depend on the sign contract across misaligned operands (the original test
// sweeps the 16x16 SSE alignment grid). Only the sign is asserted — the C
// standard promises less/equal/greater, never a magnitude.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

static int sign_of(int x) {
    return (x > 0) - (x < 0);
}

static unsigned failures = 0;
static void check_case(char* a, char* b, int want, const char* /*ctx*/) {
    int res = strcmp(a, b);
    if (sign_of(res) != want) {
        ++failures;
        ADD_FAILURE() << "strcmp(\"" << a << "\", \"" << b << "\") sign " << sign_of(res)
                      << " != " << want;
    }
}

// One alignment cell of the original ATF sweep: two NUL-terminated strings
// built at (a_off, b_off) with a difference injected at pos.
static void check_alignments(char a[], char b[], size_t a_off, size_t b_off, size_t len,
                             size_t pos) {
    a[a_off] = '\0';
    b[b_off] = '\0';

    char* a_str = a + a_off + 1;
    char* b_str = b + b_off + 1;

    a_str[len] = '\0';
    b_str[len] = '\0';
    a_str[len + 1] = 'A';
    b_str[len + 1] = 'B';

    char a_orig = a_str[pos];
    char b_orig = b_str[pos];

    check_case(a_str, b_str, 0, "equal");

    if (pos < len) {
        a_str[pos] = '\0';
        check_case(a_str, b_str, -1, "a shorter");
        a_str[pos] = a_orig;
        b_str[pos] = '\0';
        check_case(a_str, b_str, 1, "b shorter");
        b_str[pos] = b_orig;
    }

    a_str[pos] = 'X';
    check_case(a_str, b_str, 1, "a greater");
    a_str[pos] = a_orig;
    b_str[pos] = 'X';
    check_case(a_str, b_str, -1, "b greater");
    b_str[pos] = b_orig;

    a[a_off] = '-';
    b[b_off] = '-';
    a_str[len] = '-';
    b_str[len] = '-';
    a_str[len + 1] = '-';
    b_str[len + 1] = '-';
}

TEST(FreebsdStrcmp, BasicVectors) {
    EXPECT_LT(sign_of(strcmp("", "")), 1) << "empty strings equal";
    EXPECT_EQ(sign_of(strcmp("", "")), 0);
    EXPECT_EQ(sign_of(strcmp("abc", "abc")), 0);
    EXPECT_EQ(sign_of(strcmp("abc", "abd")), -1);
    EXPECT_EQ(sign_of(strcmp("abd", "abc")), 1);
    EXPECT_EQ(sign_of(strcmp("abc", "abcd")), -1) << "prefix is less";
    EXPECT_EQ(sign_of(strcmp("abcd", "abc")), 1);
    EXPECT_EQ(sign_of(strcmp("ABC", "abc")), -1) << "byte order, not collation";
}

TEST(FreebsdStrcmp, AlignmentSweep) {
    // Full 16x16 offset grid x lengths 1..64 of the ATF original. ~2.5M
    // strcmp calls; each is nanoseconds, total well under a second.
    char a[64 + 16 + 3], b[64 + 16 + 3];
    memset(a, '-', sizeof a);
    memset(b, '-', sizeof b);
    a[sizeof a - 1] = '\0';
    b[sizeof b - 1] = '\0';

    for (size_t a_off = 0; a_off < 16; ++a_off)
        for (size_t b_off = 0; b_off < 16; ++b_off)
            for (size_t len = 1; len <= 64; ++len)
                for (size_t pos = 0; pos <= len; ++pos)
                    check_alignments(a, b, a_off, b_off, len, pos);
    EXPECT_EQ(failures, 0u) << "alignment sweep sign mismatches";
}

TEST(FreebsdStrcmp, PositiveControl) {
    // Hand-built outside the sweep generator: the harness must redden on a
    // real difference, else a clean sweep proves nothing.
    EXPECT_NE(sign_of(strcmp("X", "Y")), 0);
    EXPECT_EQ(sign_of(strcmp("same", "same")), 0);
}

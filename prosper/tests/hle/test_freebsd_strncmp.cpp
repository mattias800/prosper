// test_freebsd_strncmp — FreeBSD libc contract for strncmp, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/strncmp_test.c (ATF).
// Original author: Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: h_strncmp forwards to the host strncmp, and guest
// version checks, magic compares and fixed-field parsing depend on two rules
// beyond strcmp's: comparison stops after n bytes, and hits the NUL first.
// The ATF alignment sweep is ported whole (n = len, len+1, len+16, pos,
// pos+1 per cell). The strncmp_null arm (strncmp(NULL, NULL, 0)) is
// deliberately NOT ported: passing NULL is undefined behaviour even when n
// is 0, so a green arm would pin one libc's UB tolerance, not a contract.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

using namespace prosper;

static int sign_of(int x) {
    return (x > 0) - (x < 0);
}

static unsigned failures = 0;
static void check_case(char* a, char* b, int want, size_t n) {
    int res = strncmp(a, b, n);
    if (sign_of(res) != want) {
        ++failures;
        ADD_FAILURE() << "strncmp(\"" << a << "\", \"" << b << "\", " << n << ") sign "
                      << sign_of(res) << " != " << want;
    }
}

static void check_cell(char a[], char b[], size_t a_off, size_t b_off, size_t len, size_t pos) {
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

    check_case(a_str, b_str, 0, len + 16);
    check_case(a_str, b_str, 0, len + 1);
    check_case(a_str, b_str, 0, len);

    if (pos < len) {
        a_str[pos] = '\0';
        check_case(a_str, b_str, -1, len + 16);
        check_case(a_str, b_str, -1, len + 1);
        check_case(a_str, b_str, -1, len);
        check_case(a_str, b_str, -1, pos + 1);
        check_case(a_str, b_str, 0, pos);
        a_str[pos] = a_orig;

        b_str[pos] = '\0';
        check_case(a_str, b_str, 1, len + 16);
        check_case(a_str, b_str, 1, len + 1);
        check_case(a_str, b_str, 1, len);
        check_case(a_str, b_str, 1, pos + 1);
        check_case(a_str, b_str, 0, pos);
        b_str[pos] = b_orig;
    }

    a_str[pos] = 'X';
    check_case(a_str, b_str, 1, len + 16);
    check_case(a_str, b_str, 0, pos);
    check_case(a_str, b_str, 1, pos + 1);
    if (pos < len) {
        check_case(a_str, b_str, 1, len);
        check_case(a_str, b_str, 1, len + 1);
    }
    a_str[pos] = a_orig;

    b_str[pos] = 'X';
    check_case(a_str, b_str, -1, len + 16);
    check_case(a_str, b_str, 0, pos);
    check_case(a_str, b_str, -1, pos + 1);
    if (pos < len) {
        check_case(a_str, b_str, -1, len);
        check_case(a_str, b_str, -1, len + 1);
    }
    b_str[pos] = b_orig;

    a[a_off] = '-';
    b[b_off] = '-';
    a_str[len] = '-';
    b_str[len] = '-';
    a_str[len + 1] = '-';
    b_str[len + 1] = '-';
}

TEST(FreebsdStrncmp, AlignmentSweep) {
    char a[64 + 16 + 16 + 3], b[64 + 16 + 16 + 3];
    memset(a, '-', sizeof a);
    memset(b, '-', sizeof b);
    a[sizeof a - 1] = '\0';
    b[sizeof b - 1] = '\0';

    for (size_t a_off = 0; a_off < 16; ++a_off)
        for (size_t b_off = 0; b_off < 16; ++b_off)
            for (size_t len = 1; len <= 64; ++len)
                for (size_t pos = 0; pos <= len; ++pos) check_cell(a, b, a_off, b_off, len, pos);
    EXPECT_EQ(failures, 0u) << "alignment sweep sign mismatches";
}

TEST(FreebsdStrncmp, BoundRules) {
    // n = 0 compares nothing: even different strings are equal.
    EXPECT_EQ(strncmp("abc", "xyz", 0), 0);
    // The bound cuts before the difference ...
    EXPECT_EQ(strncmp("abc", "abd", 2), 0);
    // ... and the NUL short-circuits inside the bound: both sides end at
    // [2], so the late 'c' is never reached and the answer is equal.
    EXPECT_EQ(strncmp("ab", "ab\0c", 4), 0) << "NUL ends the comparison";
    EXPECT_EQ(strncmp("ab", "ab", 99), 0) << "n past the NUL still equal";
}

TEST(FreebsdStrncmp, PositiveControl) {
    EXPECT_NE(sign_of(strncmp("abX", "abY", 3)), 0) << "harness sees a difference";
    EXPECT_EQ(sign_of(strncmp("abX", "abY", 2)), 0) << "harness honours the bound";
}

TEST(FreebsdHle, StrncmpHandlersRegistered) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("strncmp")), nullptr) << "strncmp is registered";
    EXPECT_NE(Hle::lookup(nid_hash("strnlen")), nullptr) << "strnlen is registered";
}

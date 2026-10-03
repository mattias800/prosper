// test_freebsd_printbasic — FreeBSD libc contract for integer printf
// formats, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/printbasic_test.c (ATF).
// Original author: David Schultz / FreeBSD.
//
// Why this pins prosper: h_snprintf/h_sprintf/h_printf forward to the host
// printf family, and guest logging, trophy text and save-data formatting
// depend on length modifiers (hh/h/l/ll/j/z/t) and on INT_MIN surviving
// negation. Both the narrow and the wide rendering are checked, as in the
// ATF original. LC_NUMERIC=C throughout: grouping/decimal rules belong to
// the locale, not to this contract.
#include <gtest/gtest.h>

#include <cinttypes>
#include <climits>
#include <clocale>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace {

constexpr int kBuf = 100;
unsigned failures = 0;

void smash_stack() {
    static uint32_t junk = 0xdeadbeef;
    uint32_t buf[512];
    for (size_t i = 0; i < sizeof buf / sizeof buf[0]; ++i) buf[i] = junk;
}

// Mirrors the ATF _testfmt: one vsnprintf plus the parallel vswprintf arm,
// over a smashed stack so a formatter reading past its arguments shows.
void check_fmt(const char* want, const char* fmt, ...) {
    char narrow[kBuf];
    wchar_t wide[kBuf], wfmt[kBuf], wwant[kBuf];
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    smash_stack();
    vsnprintf(narrow, sizeof narrow, fmt, ap);
    if (strcmp(want, narrow) != 0) {
        ++failures;
        ADD_FAILURE() << "printf(\"" << fmt << "\") ==> [" << narrow << "], expected [" << want
                      << "]";
    }
    smash_stack();
    mbstowcs(wide, narrow, kBuf - 1);
    mbstowcs(wfmt, fmt, kBuf - 1);
    mbstowcs(wwant, want, kBuf - 1);
    vswprintf(wide, sizeof wide / sizeof wide[0], wfmt, ap2);
    if (wcscmp(wwant, wide) != 0) {
        ++failures;
        ADD_FAILURE() << "wprintf wide arm mismatch for fmt \"" << fmt << "\"";
    }
    va_end(ap);
    va_end(ap2);
}

#define TESTFMT(want, fmt, ...) check_fmt((want), (fmt), __VA_ARGS__)

void require_c_locale() {
    ASSERT_NE(setlocale(LC_NUMERIC, "C"), nullptr) << "C locale must exist";
}

}  // namespace

TEST(FreebsdPrintBasic, IntWithinLimits) {
    require_c_locale();
    ASSERT_EQ(UINTMAX_MAX, UINT64_MAX);
    ASSERT_EQ(UINT_MAX, UINT32_MAX);

    TESTFMT("-1", "%jd", (intmax_t)-1);
    TESTFMT("18446744073709551615", "%ju", UINT64_MAX);

    TESTFMT("-1", "%td", (ptrdiff_t)-1);
    TESTFMT("18446744073709551615", "%tu", (size_t)-1);

    TESTFMT("-1", "%zd", (ssize_t)-1);
    TESTFMT("18446744073709551615", "%zu", (ssize_t)-1);

    TESTFMT("-1", "%ld", (long)-1);
    // LLP64 (Windows): long is 32 bits; LP64 (Linux/macOS): 64 bits.
    TESTFMT(sizeof(unsigned long) == 8 ? "18446744073709551615" : "4294967295", "%lu", ULONG_MAX);

    TESTFMT("-1", "%lld", (long long)-1);
    TESTFMT("18446744073709551615", "%llu", ULLONG_MAX);

    TESTFMT("-1", "%d", -1);
    TESTFMT("4294967295", "%u", UINT32_MAX);

    TESTFMT("-1", "%hd", -1);
    TESTFMT("65535", "%hu", USHRT_MAX);

    TESTFMT("-1", "%hhd", -1);
    TESTFMT("255", "%hhu", UCHAR_MAX);

    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintBasic, IntLimits) {
    require_c_locale();
    // The largest negative must survive negation without overflow.
    TESTFMT("-2147483648", "%d", INT_MIN);
    TESTFMT("-9223372036854775808", "%jd", INTMAX_MIN);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintBasic, PositiveControl) {
    char buf[32];
    EXPECT_EQ(snprintf(buf, sizeof buf, "%d", 42), 2);
    EXPECT_STREQ(buf, "42") << "harness formats integers";
}

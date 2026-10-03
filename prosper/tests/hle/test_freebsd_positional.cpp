// test_freebsd_positional — FreeBSD libc contract for positional printf
// parameters (%n$), ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/print_positional_test.c
// (ATF, itself from OpenBSD). Original author: Theo de Raadt.
//
// Why this pins prosper: positional parameters reorder one argument list
// across many conversions — the exact machinery a mistranslated guest
// variadic frame breaks first (cf. test_printf.cpp issue #122). The ATF
// positional_normal arm is itself defective: it renders into `buf` but
// asserts on two untouched wide buffers, so it passes with ANY renderer.
// This port asserts `buf` against the expected text, which is what the
// original meant to check.
//
// Windows note: UCRT has no %n$ parameters (measured: the format renders
// literally), so every case skips loudly there and runs on Linux/macOS CI.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <cwchar>

namespace {

// UCRT (all Windows CRTs) has no %n$ positional parameters: the format
// renders literally. prosper's forwarded printf inherits the gap, so these
// cases run on Linux/macOS CI and skip loudly on Windows.
#ifdef _WIN32
#define SKIP_NO_POSITIONAL()                                                                       \
    GTEST_SKIP() << "UCRT printf has no %n$ positional parameters "                                \
                    "(measured: format renders literally); prosper inherits "                      \
                    "the gap on Windows"
#else
#define SKIP_NO_POSITIONAL()
#endif

constexpr const char* kCorrect = "|xx 01 02 03 04\n"
                                 "|xx 05 06 07 08\n"
                                 "|xx 09 10 11 12\n"
                                 "|xx 13 14 15 16\n"
                                 "|xx 17 18 19 20\n"
                                 "|xx 21 22 23 24\n"
                                 "|xx 25 26 27 28\n"
                                 "|xx 29 30 31 32\n"
                                 "|xx 33 34 35 36\n"
                                 "|xx 37 38 39 40\n"
                                 "|xx 41 42 43 44\n"
                                 "|xx 45 -1 1 -1 1\n";

constexpr const char* kFormat = "|xx %1$s %2$s %3$s %4$s\n"
                                "|xx %5$s %6$s %7$s %8$s\n"
                                "|xx %9$s %10$s %11$s %12$s\n"
                                "|xx %13$s %14$s %15$s %16$s\n"
                                "|xx %17$s %18$s %19$s %20$s\n"
                                "|xx %21$s %22$s %23$s %24$s\n"
                                "|xx %25$s %26$s %27$s %28$s\n"
                                "|xx %29$s %30$s %31$s %32$s\n"
                                "|xx %33$s %34$s %35$s %36$s\n"
                                "|xx %37$s %38$s %39$s %40$s\n"
                                "|xx %41$s %42$s %43$s %44$s\n"
                                "|xx %45$d %46$ld %47$lld %48$d %49$lld\n";

}  // namespace

TEST(FreebsdPositional, FortyNineOutOfOrder) {
    SKIP_NO_POSITIONAL();
    // 44 string parameters consumed out of order plus mixed int/long/long
    // long tails: every %n$ must resolve to the right argument.
    char buf[1024];
    snprintf(buf, sizeof buf, kFormat, "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
             "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
             "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38",
             "39", "40", "41", "42", "43", "44", 45, -1L, 1LL, -1, 1LL);
    EXPECT_STREQ(buf, kCorrect) << "positional parameters resolve in order";
}

TEST(FreebsdPositional, Wide) {
    SKIP_NO_POSITIONAL();
    // Port of ATF positional_wide: the same 49-argument table through the
    // wide renderer, compared against the converted expectation.
    wchar_t wbuf[1024], want[1024];
    const char* src = kCorrect;
    mbsrtowcs(want, &src, 1024, nullptr);
    swprintf(wbuf, 1024,
             L"|xx %1$s %2$s %3$s %4$s\n"
             L"|xx %5$s %6$s %7$s %8$s\n"
             L"|xx %9$s %10$s %11$s %12$s\n"
             L"|xx %13$s %14$s %15$s %16$s\n"
             L"|xx %17$s %18$s %19$s %20$s\n"
             L"|xx %21$s %22$s %23$s %24$s\n"
             L"|xx %25$s %26$s %27$s %28$s\n"
             L"|xx %29$s %30$s %31$s %32$s\n"
             L"|xx %33$s %34$s %35$s %36$s\n"
             L"|xx %37$s %38$s %39$s %40$s\n"
             L"|xx %41$s %42$s %43$s %44$s\n"
             L"|xx %45$d %46$ld %47$lld %48$d %49$lld\n",
             "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14",
             "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28",
             "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
             "43", "44", 45, -1L, 1LL, -1, 1LL);
    EXPECT_EQ(wcscmp(wbuf, want), 0) << "wide positional table matches";
}

TEST(FreebsdPositional, PrecisionFromArguments) {
    SKIP_NO_POSITIONAL();
    // Port of ATF positional_precision(_wide): %.*n$ takes the width from
    // another parameter out of order.
    char buf[1024];
    snprintf(buf, sizeof buf, "%2$.*4$s %2$.*3$s %1$s", "BSD", "bsd", 2, 1);
    EXPECT_STREQ(buf, "b bs BSD") << "positional precision selects widths";

    wchar_t wbuf[1024], want[1024];
    const char* src = "b bs BSD";
    mbsrtowcs(want, &src, 1024, nullptr);
    swprintf(wbuf, 1024, L"%2$.*4$s %2$.*3$s %1$s", "BSD", "bsd", 2, 1);
    EXPECT_EQ(wcscmp(wbuf, want), 0) << "wide positional precision matches";
}

TEST(FreebsdPositional, PositiveControl) {
    SKIP_NO_POSITIONAL();
    char buf[32];
    snprintf(buf, sizeof buf, "%2$s %1$s", "a", "b");
    EXPECT_STREQ(buf, "b a") << "harness reorders parameters";
}

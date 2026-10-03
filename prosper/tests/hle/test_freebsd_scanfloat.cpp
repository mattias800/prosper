// test_freebsd_scanfloat — FreeBSD libc contract for floating-point
// scanf formats, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/scanfloat_test.c (ATF).
// Original author: David Schultz / FreeBSD.
//
// Why this pins prosper: h_sscanf/h_vsscanf forward to the host, and guest
// config/version parsing reads floats through them. The load-bearing rules
// are termination (where the number ends and the tail begins), Inf/NaN
// spellings, and directed-rounding of the last bit. The ru_RU locale arms
// are NOT ported (no CI host provides that locale); the suite runs under
// LC_NUMERIC=C and every rounding arm restores FE_TONEAREST.
#include <gtest/gtest.h>

#include <cfenv>
#include <cfloat>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

using std::isinf;
using std::isnan;

void require_c_locale() {
    ASSERT_NE(setlocale(LC_NUMERIC, "C"), nullptr) << "C locale must exist";
}

bool nearly_equal(long double eps, long double a, long double b) {
    return fabsl(a - b) <= eps;
}
#define EQ_FLT(a, b) nearly_equal(FLT_EPSILON, (a), (b))
#define EQ_DBL(a, b) nearly_equal(DBL_EPSILON, (a), (b))
#define EQ_LDBL(a, b) nearly_equal(LDBL_EPSILON, (a), (b))

}  // namespace

TEST(FreebsdScanFloat, NormalizedNumbers) {
    require_c_locale();
    char buf[128];
    long double ld = 0.0;
    double d = 0.0;
    float f = 0.0;

    EXPECT_EQ(sscanf("3.141592", "%e", &f), 1);
    EXPECT_TRUE(EQ_FLT(f, 3.141592f));

    EXPECT_EQ(sscanf("3.141592653589793", "%lf", &d), 1);
    EXPECT_TRUE(EQ_DBL(d, 3.141592653589793));

    EXPECT_EQ(sscanf("1.234568e+06", "%E", &f), 1);
    EXPECT_TRUE(EQ_FLT(f, 1.234568e+06f));

    EXPECT_EQ(sscanf("-1.234568e6", "%lF", &d), 1);
    EXPECT_TRUE(EQ_DBL(d, -1.234568e6));

    EXPECT_EQ(sscanf("+1.234568e-52", "%LG", &ld), 1);
    EXPECT_TRUE(EQ_LDBL(ld, 1.234568e-52L));

    EXPECT_EQ(sscanf("0.1", "%la", &d), 1);
    EXPECT_TRUE(EQ_DBL(d, 0.1));

    EXPECT_EQ(sscanf("00.2", "%lA", &d), 1);
    EXPECT_TRUE(EQ_DBL(d, 0.2));

    EXPECT_EQ(sscanf("123456", "%5le%s", &d, buf), 2);
    EXPECT_TRUE(EQ_DBL(d, 12345.));
    EXPECT_STREQ(buf, "6");

    EXPECT_EQ(sscanf("1.0Q", "%*5le%s", buf), 1);
    EXPECT_STREQ(buf, "Q");

    EXPECT_EQ(sscanf("1.23E4E5", "%le%s", &d, buf), 2);
    EXPECT_TRUE(EQ_DBL(d, 1.23e4));
    EXPECT_STREQ(buf, "E5");

    EXPECT_EQ(sscanf("12e6", "%le", &d), 1);
    EXPECT_TRUE(EQ_DBL(d, 12e6));

    EXPECT_EQ(sscanf("1.a", "%le%s", &d, buf), 2);
    EXPECT_TRUE(EQ_DBL(d, 1.0));
    EXPECT_STREQ(buf, "a");

    EXPECT_EQ(sscanf(".0p4", "%le%s", &d, buf), 2);
    EXPECT_TRUE(EQ_DBL(d, 0.0));
    EXPECT_STREQ(buf, "p4");

    d = 0.25;
    EXPECT_EQ(sscanf(".", "%le", &d), 0);
    EXPECT_EQ(d, 0.25) << "failed conversion stores nothing";

    EXPECT_EQ(sscanf("0x08", "%le", &d), 1);
    EXPECT_EQ(d, 0x8p0);

    EXPECT_EQ(sscanf("0x90a.bcdefP+09a", "%le%s", &d, buf), 2);
    EXPECT_EQ(d, 0x90a.bcdefp+09);
    EXPECT_STREQ(buf, "a");
    // "0xg" lives in PartialScanTermination below (host-divergent arm).

#if (LDBL_MANT_DIG > DBL_MANT_DIG) && !defined(__i386__)
    EXPECT_EQ(sscanf("3.14159265358979323846", "%Lg", &ld), 1);
    EXPECT_TRUE(EQ_LDBL(ld, 3.14159265358979323846L));

    EXPECT_EQ(sscanf("  0X.0123456789abcdefffp-3g", "%Le%s", &ld, buf), 2);
    EXPECT_EQ(ld, 0x0.0123456789abcdefffp-3L);
    EXPECT_STREQ(buf, "g");
#endif
}

TEST(FreebsdScanFloat, PartialScanTermination) {
#ifdef _WIN32
    // Measured UCRT divergences in where a partial number ends: "-1.23e"
    // and "1.25e+" convert nothing further on UCRT (returns 1, tail
    // untouched) where FreeBSD converts the head and leaves "e"/"e+";
    // "0xg" matches nothing (returns 0) where FreeBSD matches 0.0 leaving
    // "xg". prosper's forwarded sscanf inherits the host split, so these
    // run on Linux/macOS CI and skip loudly on Windows.
    GTEST_SKIP() << "UCRT partial-scan termination differs (measured)";
#endif
    require_c_locale();
    char buf[128];
    double d = 0.0;
    float f = 0.0;

    EXPECT_EQ(sscanf("-1.23e", "%e%s", &f, buf), 2);
    EXPECT_TRUE(EQ_FLT(f, -1.23f));
    EXPECT_STREQ(buf, "e");

    EXPECT_EQ(sscanf("1.25e+", "%le%s", &d, buf), 2);
    EXPECT_TRUE(EQ_DBL(d, 1.25));
    EXPECT_STREQ(buf, "e+");

    EXPECT_EQ(sscanf("0xg", "%le%s", &d, buf), 2);
    EXPECT_EQ(d, 0.0);
    EXPECT_STREQ(buf, "xg");
}

TEST(FreebsdScanFloat, InfinitiesAndNans) {
    require_c_locale();
    char buf[128];
    long double ld = 0.0;
    double d = 0.0;
    float f = 0.0;

    EXPECT_EQ(sscanf("-Inf", "%le", &d), 1);
    EXPECT_TRUE(d < 0.0 && isinf(d));

    EXPECT_EQ(sscanf("NaN", "%le", &d), 1);
    EXPECT_TRUE(isnan(d));

    EXPECT_EQ(sscanf("NAN(123Y", "%le%s", &d, buf), 2);
    EXPECT_TRUE(isnan(d));
    EXPECT_STREQ(buf, "(123Y");

    EXPECT_EQ(sscanf("-nan", "%le", &d), 1);
    EXPECT_TRUE(isnan(d));

    // Only quiet NaNs come back, and producing them raises no INVALID.
    EXPECT_EQ(sscanf("NaN", "%e", &f), 1);
    EXPECT_EQ(sscanf("nan", "%le", &d), 1);
    EXPECT_EQ(sscanf("nan", "%Le", &ld), 1);
    ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    EXPECT_TRUE(f != f);
    EXPECT_TRUE(d != d);
    EXPECT_TRUE(ld != ld);
    EXPECT_EQ(fetestexcept(FE_INVALID), 0);
    EXPECT_EQ(sscanf("nan(1234)", "%e", &f), 1);
    EXPECT_EQ(sscanf("nan(1234)", "%le", &d), 1);
    EXPECT_EQ(sscanf("nan(1234)", "%Le", &ld), 1);
    ASSERT_EQ(feclearexcept(FE_ALL_EXCEPT), 0);
    EXPECT_TRUE(f != f);
    EXPECT_TRUE(d != d);
    EXPECT_TRUE(ld != ld);
    EXPECT_EQ(fetestexcept(FE_INVALID), 0) << "POSIX: only quiet NaNs";
}

TEST(FreebsdScanFloat, InfTailAndNanPayload) {
#ifdef _WIN32
    // Measured UCRT divergences in tail splitting: "iNfInItY and beyond"
    // leaves "and" (UCRT %s skips the space; FreeBSD keeps " and beyond"),
    // and "nan(f00f)plugh" leaves "(f00f)plugh" (UCRT does not scan the
    // n-char NaN payload). prosper's forwarded sscanf inherits the host
    // split. Runs on Linux/macOS CI, skips loudly on Windows.
    GTEST_SKIP() << "UCRT inf-tail/NaN-payload termination differs (measured)";
#endif
    require_c_locale();
    char buf[128];
    double d = 0.0;

    EXPECT_EQ(sscanf("iNfInItY and beyond", "%le%s", &d, buf), 2);
    EXPECT_TRUE(d > 0.0 && isinf(d));
    EXPECT_STREQ(buf, " and beyond");

    EXPECT_EQ(sscanf("nan(f00f)plugh", "%le%s", &d, buf), 2);
    EXPECT_TRUE(isnan(d));
    EXPECT_STREQ(buf, "plugh");
}

TEST(FreebsdScanFloat, DirectedRounding) {
#ifdef _WIN32
    // Measured: UCRT strtod/scanf ignore fesetround (overflow is always INF,
    // ties always round the same way), so prosper inherits round-to-nearest
    // on Windows. Skipped loudly; runs on Linux/macOS CI.
    GTEST_SKIP() << "UCRT strtod ignores the FP rounding mode (measured)";
#endif
    require_c_locale();
    long double ld = 0.0;
    double d = 0.0;

    ASSERT_EQ(fesetround(FE_DOWNWARD), 0);
    EXPECT_EQ(sscanf("1.999999999999999999999999999999999", "%le", &d), 1);
    EXPECT_LT(d, 2.0);
    EXPECT_EQ(sscanf("0x1.ffffffffffffffp0", "%le", &d), 1);
    EXPECT_LT(d, 2.0);
    EXPECT_EQ(sscanf("1.999999999999999999999999999999999", "%Le", &ld), 1);
    EXPECT_LT(ld, 2.0L);

    EXPECT_EQ(sscanf("1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("-1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, -0x1.0ea3f4af0dc5ap0);
    EXPECT_EQ(sscanf("1.0571892669084010", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc5ap0);

    EXPECT_EQ(sscanf("0x1.23p-5000", "%le", &d), 1);
    EXPECT_EQ(d, 0.0);
    EXPECT_EQ(sscanf("0x1.2345678p-1050", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.234567p-1050);

    ASSERT_EQ(fesetround(FE_UPWARD), 0);
    EXPECT_EQ(sscanf("1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc5ap0);
    EXPECT_EQ(sscanf("-1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, -0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("1.0571892669084010", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc5bp0);
    EXPECT_EQ(sscanf("0x1.23p-5000", "%le", &d), 1);
    EXPECT_EQ(d, 0x1p-1074);
    EXPECT_EQ(sscanf("0x1.2345678p-1050", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.234568p-1050);

    ASSERT_EQ(fesetround(FE_TOWARDZERO), 0);
    EXPECT_EQ(sscanf("1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("-1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, -0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("1.0571892669084010", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc5ap0);
    EXPECT_EQ(sscanf("0x1.23p-5000", "%le", &d), 1);
    EXPECT_EQ(d, 0.0);
    EXPECT_EQ(sscanf("0x1.2345678p-1050", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.234567p-1050);

    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    EXPECT_EQ(sscanf("1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("-1.0571892669084007", "%le", &d), 1);
    EXPECT_EQ(d, -0x1.0ea3f4af0dc59p0);
    EXPECT_EQ(sscanf("1.0571892669084010", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.0ea3f4af0dc5bp0);
    EXPECT_EQ(sscanf("0x1.23p-5000", "%le", &d), 1);
    EXPECT_EQ(d, 0.0);
    EXPECT_EQ(sscanf("0x1.2345678p-1050", "%le", &d), 1);
    EXPECT_EQ(d, 0x1.234568p-1050);
}

TEST(FreebsdScanFloat, StrtodOverflowFollowsFenv) {
#ifdef _WIN32
    // Same UCRT fesetround blindness as DirectedRounding (measured: overflow
    // clamps to INF in every mode). Skipped loudly.
    GTEST_SKIP() << "UCRT strtod ignores the FP rounding mode (measured)";
#endif
    require_c_locale();
    char* endp = nullptr;
    EXPECT_EQ(strtod("0xy", &endp), 0);
    EXPECT_STREQ(endp, "xy");

    // Overflow clamps to the mode's extreme, and the direction decides which
    // extreme (this once looped forever and rounded the wrong way).
    ASSERT_EQ(fesetround(FE_DOWNWARD), 0);
    EXPECT_EQ(strtof("3.5e38", &endp), FLT_MAX);
    EXPECT_EQ(strtod("2e308", &endp), DBL_MAX);
    ASSERT_EQ(fesetround(FE_UPWARD), 0);
    EXPECT_EQ(strtof("3.5e38", &endp), INFINITY);
    EXPECT_EQ(strtod("2e308", &endp), INFINITY);
    ASSERT_EQ(fesetround(FE_TOWARDZERO), 0);
    EXPECT_EQ(strtof("3.5e38", &endp), FLT_MAX);
    EXPECT_EQ(strtod("2e308", &endp), DBL_MAX);
    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    EXPECT_EQ(strtof("3.5e38", &endp), INFINITY);
    EXPECT_EQ(strtod("2e308", &endp), INFINITY);
}

TEST(FreebsdScanFloat, PositiveControl) {
    double d = 0.0;
    EXPECT_EQ(sscanf("abc", "%le", &d), 0) << "harness sees a failed scan";
    EXPECT_EQ(sscanf("2.5x", "%le", &d), 1);
    EXPECT_EQ(d, 2.5) << "harness sees a value";
}

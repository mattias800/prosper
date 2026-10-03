// test_freebsd_printfloat — FreeBSD libc contract for floating-point
// printf formats, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/printfloat_test.c (ATF).
// Original author: David Schultz / FreeBSD.
//
// Why this pins prosper: the guest formats floats for HUD text, settings
// screens and telemetry through the forwarded printf family, and float
// rendering is where C libraries diverge most (rounding, NaN/Inf spelling,
// hex floats, subnormals). Narrow + wide arms, LC_NUMERIC=C, smashed stack,
// as in the ATF original.
//
// Deliberately NOT ported: thousands_separator_and_other_locale_tests — it
// requires hi_IN/ru_RU/el_GR locales no CI host provides. The rounding arms
// restore FE_TONEAREST so no state leaks into sibling cases (each ctest case
// is its own process, but one binary run without a filter shares it).
#include <gtest/gtest.h>

#include <cfenv>
#include <cfloat>
#include <clocale>
#include <cmath>
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

TEST(FreebsdPrintFloat, WithinLimits) {
    require_c_locale();
    TESTFMT(" 1.000000E+00", "%13E", 1.0);
    TESTFMT("     1.000000", "%13f", 1.0);
    TESTFMT("            1", "%13G", 1.0);
    TESTFMT(" 1.000000E+00", "%13LE", 1.0L);
    TESTFMT("     1.000000", "%13Lf", 1.0L);
    TESTFMT("            1", "%13LG", 1.0L);

    TESTFMT("2.718282", "%.*f", -2, 2.7182818);

    TESTFMT("1.234568e+06", "%e", 1234567.8);
    TESTFMT("1234567.800000", "%f", 1234567.8);
    TESTFMT("1.23457E+06", "%G", 1234567.8);
    TESTFMT("1.234568e+06", "%Le", 1234567.8L);
    TESTFMT("1234567.800000", "%Lf", 1234567.8L);
    TESTFMT("1.23457E+06", "%LG", 1234567.8L);

#if (LDBL_MANT_DIG > DBL_MANT_DIG) && !defined(__i386__)
    TESTFMT("123456789.864210", "%Lf", 123456789.8642097531L);
    TESTFMT("-1.23457E+08", "%LG", -123456789.8642097531L);
    TESTFMT("123456789.8642097531", "%.10Lf", 123456789.8642097531L);
    // NOT ported: the ATF's "%L27.18Le" case (length modifier BEFORE the
    // width). C requires width-then-length; FreeBSD's parser tolerates the
    // swap while glibc/UCRT print the format literally. That pins FreeBSD
    // parser leniency, not a portable contract.
#endif
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, InfinitiesAndNans) {
    require_c_locale();
    TESTFMT("nan", "%e", NAN);
    TESTFMT("NAN", "%F", NAN);
    TESTFMT("nan", "%g", NAN);
    TESTFMT("NAN", "%LE", (long double)NAN);
    TESTFMT("  nan", "%05e", NAN);

    TESTFMT("INF", "%E", HUGE_VAL);
    TESTFMT("-inf", "%f", -HUGE_VAL);
    TESTFMT("+inf", "%+g", HUGE_VAL);
    TESTFMT(" inf", "%4.2Le", HUGE_VALL);
    TESTFMT("-inf", "%Lf", -HUGE_VALL);
    TESTFMT("  inf", "%05e", HUGE_VAL);
    TESTFMT(" -inf", "%05e", -HUGE_VAL);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, Padding) {
    require_c_locale();
    TESTFMT("0.000000e+00", "%e", 0.0);
    TESTFMT("0.000000", "%F", (double)0.0);
    TESTFMT("0", "%G", 0.0);
    TESTFMT("  0", "%3.0Lg", 0.0L);
    TESTFMT("    0", "%5.0f", 0.001);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, PrecisionSpecifiers) {
    require_c_locale();
    TESTFMT("1.0123e+00", "%.4e", 1.0123456789);
    TESTFMT("1.0123", "%.4f", 1.0123456789);
    TESTFMT("1.012", "%.4g", 1.0123456789);
    TESTFMT("1.2346e-02", "%.4e", 0.0123456789);
    TESTFMT("0.0123", "%.4f", 0.0123456789);
    TESTFMT("0.01235", "%.4g", 0.0123456789);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, SignedConversions) {
    require_c_locale();
    TESTFMT("+2.500000e-01", "%+e", 0.25);
    TESTFMT("+0.000000", "%+F", 0.0);
    TESTFMT("-1", "%+g", -1.0);
    TESTFMT("-1.000000e+00", "% e", -1.0);
    TESTFMT("+1.000000", "% +f", 1.0);
    TESTFMT(" 1", "% g", 1.0);
    TESTFMT(" 0", "% g", 0.0);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, AlternateForm) {
    require_c_locale();
    TESTFMT("1.250e+00", "%#.3e", 1.25);
    TESTFMT("123.000000", "%#f", 123.0);
    TESTFMT(" 12345.", "%#7.5g", 12345.0);
    TESTFMT(" 1.00000", "%#8g", 1.0);
    TESTFMT("0.0", "%#.2g", 0.0);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, PaddingAndPointPlacement) {
    require_c_locale();
    TESTFMT("03.2E+00", "%08.1E", 3.25);
    TESTFMT("003.25", "%06.2F", 3.25);
    TESTFMT("0003.25", "%07.4G", 3.25);
    TESTFMT("3.14159e-05", "%g", 3.14159e-5);
    TESTFMT("0.000314159", "%g", 3.14159e-4);
    TESTFMT("3.14159e+06", "%g", 3.14159e6);
    TESTFMT("314159", "%g", 3.14159e5);
    TESTFMT("314159.", "%#g", 3.14159e5);
    TESTFMT(" 9.000000e+03", "%13e", 9000.0);
    TESTFMT(" 9000.000000", "%12f", 9000.0);
    TESTFMT(" 9000", "%5g", 9000.0);
    TESTFMT(" 900000.", "%#8g", 900000.0);
    TESTFMT(" 9e+06", "%6g", 9000000.0);
    TESTFMT(" 9.000000e-04", "%13e", 0.0009);
    TESTFMT(" 0.000900", "%9f", 0.0009);
    TESTFMT(" 0.0009", "%7g", 0.0009);
    TESTFMT(" 9e-05", "%6g", 0.00009);
    TESTFMT(" 9.00000e-05", "%#12g", 0.00009);
    TESTFMT(" 9.e-05", "%#7.1g", 0.00009);
    TESTFMT(" 0.0", "%4.1f", 0.0);
    TESTFMT("90.0", "%4.1f", 90.0);
    TESTFMT(" 100", "%4.0f", 100.0);
    TESTFMT("9.0e+01", "%4.1e", 90.0);
    TESTFMT("1e+02", "%4.0e", 100.0);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, DecimalRoundingFollowsFenv) {
#ifdef _WIN32
    // Measured divergence: UCRT printf ignores fesetround for decimal
    // conversion (%.3f of 4.4375 prints 4.438 even under FE_DOWNWARD), so
    // prosper's forwarded printf inherits round-to-nearest on Windows.
    // Skipped loudly; runs on Linux/macOS CI.
    GTEST_SKIP() << "UCRT printf ignores the FP rounding mode (measured); "
                    "glibc/FreeBSD honor it";
#endif
    require_c_locale();
    ASSERT_EQ(fesetround(FE_DOWNWARD), 0);
    TESTFMT("4.437", "%.3f", 4.4375);
    TESTFMT("-4.438", "%.3f", -4.4375);
    TESTFMT("4.437", "%.3Lf", 4.4375L);
    TESTFMT("-4.438", "%.3Lf", -4.4375L);

    ASSERT_EQ(fesetround(FE_UPWARD), 0);
    TESTFMT("4.438", "%.3f", 4.4375);
    TESTFMT("-4.437", "%.3f", -4.4375);
    TESTFMT("4.438", "%.3Lf", 4.4375L);
    TESTFMT("-4.437", "%.3Lf", -4.4375L);

    ASSERT_EQ(fesetround(FE_TOWARDZERO), 0);
    TESTFMT("4.437", "%.3f", 4.4375);
    TESTFMT("-4.437", "%.3f", -4.4375);
    TESTFMT("4.437", "%.3Lf", 4.4375L);
    TESTFMT("-4.437", "%.3Lf", -4.4375L);

    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    TESTFMT("4.438", "%.3f", 4.4375);
    TESTFMT("-4.438", "%.3f", -4.4375);
    TESTFMT("4.438", "%.3Lf", 4.4375L);
    TESTFMT("-4.438", "%.3Lf", -4.4375L);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, HexFloat) {
    require_c_locale();
    TESTFMT("0x0p+0", "%a", 0x0.0p0);
    TESTFMT("0X0.P+0", "%#LA", 0x0.0p0L);
    TESTFMT("inf", "%La", (long double)INFINITY);
    TESTFMT("+INF", "%+A", INFINITY);
    TESTFMT("nan", "%La", (long double)NAN);
    TESTFMT("NAN", "%A", NAN);

    TESTFMT(" 0x1.23p+0", "%10a", 0x1.23p0);
    TESTFMT(" 0x1.23p-500", "%12a", 0x1.23p-500);
    TESTFMT(" 0x1.2p+40", "%10.1a", 0x1.23p40);
    TESTFMT(" 0X1.230000000000000000000000P-4", "%32.24A", 0x1.23p-4);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, HexFloatExtendedAndDenormal) {
#ifdef _WIN32
    // Measured divergences, two of them. (1) UCRT prints denormals in
    // 0x0.xxxp-1022 form where FreeBSD/glibc print 0x1p-1074. (2) UCRT never
    // normalizes the hex mantissa to [1,2): %La of 0x3.243f6a8885a308dp0
    // prints 0xc.90fdaa22168c234p-2. Both flow into prosper's forwarded
    // printf on Windows. Skipped loudly; runs on Linux/macOS CI.
    GTEST_SKIP() << "UCRT hex-float style differs (measured): denormal form "
                    "and unnormalized mantissas";
#endif
    require_c_locale();
    TESTFMT("0x1p-1074", "%a", 0x1p-1074);
    TESTFMT("0x1.2345p-1024", "%a", 0x1.2345p-1024);

#if (LDBL_MANT_DIG == 64)
    TESTFMT("0x1.921fb54442d18468p+1", "%La", 0x3.243f6a8885a308dp0L);
    TESTFMT("0x1p-16445", "%La", 0x1p-16445L);
    TESTFMT("0x1.30ecap-16381", "%La", 0x9.8765p-16384L);
#elif (LDBL_MANT_DIG == 113)
    TESTFMT("0x1.921fb54442d18469898cc51701b8p+1", "%La", 0x3.243f6a8885a308d313198a2e037p0L);
    TESTFMT("0x1p-16494", "%La", 0x1p-16494L);
    TESTFMT("0x1.2345p-16384", "%La", 0x1.2345p-16384L);
#else
    TESTFMT("0x1.921fb54442d18p+1", "%La", 0x3.243f6a8885a31p0L);
    TESTFMT("0x1p-1074", "%La", 0x1p-1074L);
    TESTFMT("0x1.30ecap-1021", "%La", 0x9.8765p-1024L);
#endif
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, HexFloatRounding) {
#ifdef _WIN32
    // Same fesetround blindness as DecimalRoundingFollowsFenv (measured:
    // %.11A under FE_TOWARDZERO rounds up on UCRT). Skipped loudly.
    GTEST_SKIP() << "UCRT printf ignores the FP rounding mode (measured)";
#endif
    require_c_locale();
    ASSERT_EQ(fesetround(FE_TOWARDZERO), 0);
    TESTFMT("0X1.23456789ABCP+0", "%.11A", 0x1.23456789abcdep0);
    TESTFMT("-0x1.23456p+0", "%.5a", -0x1.23456789abcdep0);
    TESTFMT("0x1.23456p+0", "%.5a", 0x1.23456789abcdep0);
    TESTFMT("0x1.234567p+0", "%.6a", 0x1.23456789abcdep0);
    TESTFMT("-0x1.234566p+0", "%.6a", -0x1.23456689abcdep0);

    ASSERT_EQ(fesetround(FE_DOWNWARD), 0);
    TESTFMT("0X1.23456789ABCP+0", "%.11A", 0x1.23456789abcdep0);
    TESTFMT("-0x1.23457p+0", "%.5a", -0x1.23456789abcdep0);
    TESTFMT("0x1.23456p+0", "%.5a", 0x1.23456789abcdep0);
    TESTFMT("0x1.234567p+0", "%.6a", 0x1.23456789abcdep0);
    TESTFMT("-0x1.234567p+0", "%.6a", -0x1.23456689abcdep0);

    ASSERT_EQ(fesetround(FE_UPWARD), 0);
    TESTFMT("0X1.23456789ABDP+0", "%.11A", 0x1.23456789abcdep0);
    TESTFMT("-0x1.23456p+0", "%.5a", -0x1.23456789abcdep0);
    TESTFMT("0x1.23457p+0", "%.5a", 0x1.23456789abcdep0);
    TESTFMT("0x1.234568p+0", "%.6a", 0x1.23456789abcdep0);
    TESTFMT("-0x1.234566p+0", "%.6a", -0x1.23456689abcdep0);

    ASSERT_EQ(fesetround(FE_TONEAREST), 0);
    TESTFMT("0x1.23456789abcdep+4", "%a", 0x1.23456789abcdep4);
    TESTFMT("0x1.83p+0", "%.2a", 1.51);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, SubnormalDecimal) {
    // The %g/%G half of the FreeBSD 253847 regression: exact 20-digit
    // rendering of the denormal minimum. Portable; runs everywhere.
    require_c_locale();
    double dpos = __DBL_DENORM_MIN__;
    TESTFMT("4.9406564584124654418e-324", "%20.20g", dpos);
    TESTFMT("4.9406564584124654418E-324", "%20.20G", dpos);
    double dneg = -__DBL_DENORM_MIN__;
    TESTFMT("-4.9406564584124654418e-324", "%20.20g", dneg);
    TESTFMT("-4.9406564584124654418E-324", "%20.20G", dneg);

    float fpos = __FLT_DENORM_MIN__;
    TESTFMT("1.4012984643248170709e-45", "%20.20g", fpos);
    TESTFMT("1.4012984643248170709E-45", "%20.20G", fpos);
    float fneg = -__FLT_DENORM_MIN__;
    TESTFMT("-1.4012984643248170709e-45", "%20.20g", fneg);
    TESTFMT("-1.4012984643248170709E-45", "%20.20G", fneg);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, SubnormalHex) {
#ifdef _WIN32
    // UCRT %a of denormals (0x0.0000000000001p-1022 style, see
    // HexFloatExtendedAndDenormal). Skipped loudly.
    GTEST_SKIP() << "UCRT denormal hex-float style differs (measured)";
#endif
    require_c_locale();
    double dpos = __DBL_DENORM_MIN__;
    TESTFMT("0x1p-1074", "%a", dpos);
    TESTFMT("0X1P-1074", "%A", dpos);
    double dneg = -__DBL_DENORM_MIN__;
    TESTFMT("-0x1p-1074", "%a", dneg);
    TESTFMT("-0X1P-1074", "%A", dneg);

    float fpos = __FLT_DENORM_MIN__;
    TESTFMT("0x1p-149", "%a", fpos);
    TESTFMT("0X1P-149", "%A", fpos);
    float fneg = -__FLT_DENORM_MIN__;
    TESTFMT("-0x1p-149", "%a", fneg);
    TESTFMT("-0X1P-149", "%A", fneg);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdPrintFloat, PositiveControl) {
    char buf[32];
    EXPECT_EQ(snprintf(buf, sizeof buf, "%.2f", 3.14159), 4);
    EXPECT_STREQ(buf, "3.14") << "harness formats floats";
}

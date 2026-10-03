// test_hle_printf_family — the guest's printf family, called through prosper's REGISTERED handlers.
//
// Contract: C11 (N1570) §7.21.6.1 (flags, field width, precision, length modifiers, conversions),
// §7.21.6.5/.6 (snprintf bound and return value, sprintf) and §7.21.6.12/.13 (vsnprintf/vsprintf);
// POSIX fprintf for positional `%n$` conversions. Expected strings are derived from those texts, not
// from any other implementation's test suite.
//
// Every case calls the handler a guest import resolves to — `snprintf`/`sprintf` through the
// guest-ABI pointer `Hle::lookup_guest_abi` builds, `vsnprintf`/`vsprintf` through `Hle::lookup` with
// a GENUINE System V va_list — never the host's own formatter. What is prosper's here, and what a
// wrong answer would cost the guest:
//   * the guest-ABI handlers capture a real variadic frame. On Windows they re-express the System V
//     list for the host CRT by classifying every conversion (host/abi/guest_varargs.cpp), and one
//     mis-classified length modifier or `*` shifts every later argument — a `%s` behind it then
//     dereferences a non-pointer. So every case ends with a sentinel conversion AFTER the ones under
//     test, and enough arguments to spill past both register files into the overflow area.
//   * the v* handlers receive the guest's va_list POINTER and must read it as the System V structure
//     it is, not as whatever the host's va_list happens to be.
//   * positional conversions and x87 `long double` are refused by the Windows model, which formats
//     only the prefix in front of them rather than reading a wrong slot; every other host formats
//     them in full. Both behaviours are asserted, per platform.
//
// Deliberately not asserted: the exact `%a` spelling (the leading hex digit is unspecified,
// N1570 §7.21.6.1p8 — a round trip is asserted instead), `%p` (implementation-defined) and `%n`.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>

#include <bit>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

using namespace prosper;

namespace {

using GuestSnprintf = Hle::GuestAbiFn<int, char*, size_t, const char*>;
using GuestSprintf = Hle::GuestAbiFn<int, char*, const char*>;

struct Handlers {
    GuestSnprintf snprintf_fn = nullptr;
    GuestSprintf sprintf_fn = nullptr;
    HleFn vsnprintf_fn = nullptr;
    HleFn vsprintf_fn = nullptr;
};

const Handlers& handlers() {
    static const Handlers h = [] {
        register_builtin_hle();
        Handlers r;
        r.snprintf_fn =
            Hle::lookup_guest_abi<int, char*, size_t, const char*>(nid_hash("snprintf"));
        r.sprintf_fn = Hle::lookup_guest_abi<int, char*, const char*>(nid_hash("sprintf"));
        r.vsnprintf_fn = Hle::lookup(nid_hash("vsnprintf"));
        r.vsprintf_fn = Hle::lookup(nid_hash("vsprintf"));
        return r;
    }();
    return h;
}

// The producer of a guest va_list must build a SYSTEM V frame, which on Windows needs the same tag
// the real handlers carry; everywhere else an ordinary variadic function already does.
#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#define GUEST_VA_LIST __builtin_sysv_va_list
#define GUEST_VA_START __builtin_sysv_va_start
#define GUEST_VA_END __builtin_sysv_va_end
#else
#define GUEST_VA_LIST va_list
#define GUEST_VA_START va_start
#define GUEST_VA_END va_end
#endif

// What a guest's own `vsnprintf(buf, n, fmt, ap)` call hands the import: the list's ADDRESS. These
// frames own no object with a destructor and call nothing that could be inlined carrying one (the
// PROSPER_GUEST_ABI rule, dispatch.hpp), so the handlers come from plain globals the fixture sets
// rather than from handlers()'s guarded static initialisation.
HleFn g_vsnprintf = nullptr;
HleFn g_vsprintf = nullptr;

PROSPER_GUEST_ABI int guest_vsnprintf(char* buf, size_t n, const char* fmt, ...) {
    GUEST_VA_LIST ap;
    GUEST_VA_START(ap, fmt);
    const uint64_t r = g_vsnprintf((uint64_t)(uintptr_t)buf, (uint64_t)n, (uint64_t)(uintptr_t)fmt,
                                   (uint64_t)(uintptr_t)&ap, 0, 0);
    GUEST_VA_END(ap);
    return (int)(int64_t)r;
}
PROSPER_GUEST_ABI int guest_vsprintf(char* buf, const char* fmt, ...) {
    GUEST_VA_LIST ap;
    GUEST_VA_START(ap, fmt);
    const uint64_t r = g_vsprintf((uint64_t)(uintptr_t)buf, (uint64_t)(uintptr_t)fmt,
                                  (uint64_t)(uintptr_t)&ap, 0, 0, 0);
    GUEST_VA_END(ap);
    return (int)(int64_t)r;
}

class PrintfFamily : public ::testing::Test {
protected:
    void SetUp() override {
        const Handlers& h = handlers();
        ASSERT_NE(h.snprintf_fn, nullptr) << "snprintf is not registered as a guest-ABI handler";
        ASSERT_NE(h.sprintf_fn, nullptr) << "sprintf is not registered as a guest-ABI handler";
        ASSERT_NE(h.vsnprintf_fn, nullptr) << "vsnprintf is not registered";
        ASSERT_NE(h.vsprintf_fn, nullptr) << "vsprintf is not registered";
        g_vsnprintf = h.vsnprintf_fn;
        g_vsprintf = h.vsprintf_fn;
        std::memset(buf, 'Z', sizeof buf);
    }
    GuestSnprintf snf() const { return handlers().snprintf_fn; }
    GuestSprintf spf() const { return handlers().sprintf_fn; }
    char buf[512];
};

// C allows "inf" or "infinity" (any case per the conversion's case) for an infinity, and "nan" or
// "nan(n-char-sequence)" for a NaN (N1570 §7.21.6.1p8). Accept exactly those spellings.
bool is_inf_spelling(const std::string& s, bool upper) {
    return s == (upper ? "INF" : "inf") || s == (upper ? "INFINITY" : "infinity");
}
bool is_nan_spelling(const std::string& s, bool upper) {
    const std::string stem = upper ? "NAN" : "nan";
    if (s == stem) return true;
    return s.size() > stem.size() + 1 && s.compare(0, stem.size() + 1, stem + "(") == 0 &&
           s.back() == ')';
}

std::string field(const std::string& s, size_t index) {
    size_t begin = 0;
    for (size_t i = 0; i < index; ++i) {
        begin = s.find('|', begin);
        if (begin == std::string::npos) return "<missing>";
        ++begin;
    }
    const size_t end = s.find('|', begin);
    return s.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

}   // namespace

// Every length modifier C defines for integers, each with an extreme of its own type, and the
// narrowing hh/h conversions given a value OUTSIDE the narrow type: the value is converted to the
// narrow type before printing (§7.21.6.1p7), which separates them from a plain %u. Fourteen
// arguments: three in registers, eleven in the overflow area.
TEST_F(PrintfFamily, IntegerLengthModifiersKeepLaterArgumentsInStep) {
    using ssize = std::make_signed_t<size_t>;
    const int r =
        snf()(buf, sizeof buf, "%hhd %hhu %hd %hu %d %u %lld %llu %jd %ju %zu %zd %td|%hhu %hu|%s",
              -128, (unsigned char)255, (short)-32768, (unsigned short)65535, INT_MIN, UINT_MAX,
              LLONG_MIN, ULLONG_MAX, INTMAX_MIN, UINTMAX_MAX, SIZE_MAX, (ssize)-1, PTRDIFF_MIN,
              0x1ff, 0x1ffff, "end");
    const char* expected =
        "-128 255 -32768 65535 -2147483648 4294967295 -9223372036854775808 18446744073709551615 "
        "-9223372036854775808 18446744073709551615 18446744073709551615 -1 -9223372036854775808"
        "|255 65535|end";
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected)) << "snprintf returns the formatted length";
}

// `*` takes the width or precision from an int argument that comes BEFORE the converted value; a
// negative width is the '-' flag plus its magnitude, and a negative precision is taken as if it
// were omitted (§7.21.6.1p5). The trailing %s proves each `*` consumed exactly one argument.
TEST_F(PrintfFamily, StarWidthAndPrecisionConsumeTheirArguments) {
    const int r = snf()(buf, sizeof buf, "[%*d][%-*d][%*d][%.*f][%*.*e][%.*f]|%s", 6, 42, 6, 42, -6,
                        42, 3, 3.14159, 12, 2, 1234.5, -1, 2.5, "end");
    const char* expected = "[    42][42    ][42    ][3.142][    1.23e+03][2.500000]|end";
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));
}

// The flag characters and precision rules of §7.21.6.1p6 and p8, with integer and floating
// arguments interleaved so the two System V register files are consumed out of step.
TEST_F(PrintfFamily, FlagsAndPrecisionFollowTheStandard) {
    const int r = snf()(buf, sizeof buf,
                        "%+d|% d|%-5d|%05d|%#o|%#x|%#X|%#x|%.3d|%.0d|%5.1s|%c|%%|"
                        "%#.0f|%.0f|%+.1e|%08.3f|%-8.2f|%s",
                        5, 5, 42, -42, 8, 255, 255, 0, 7, 0, "abc", 'A', 3.0, 3.0, 12345.0, -1.5,
                        2.25, "end");
    const char* expected =
        "+5| 5|42   |-0042|010|0xff|0XFF|0|007||    a|A|%|3.|3|+1.2e+04|-001.500|2.25    |end";
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));
}

// %f, %e, %g and their upper-case forms, including %g's choice between the two styles (exponent
// against precision) and its trailing-zero removal, undone by '#'. Ten doubles: eight in xmm
// registers and two in the overflow area, behind which the sentinel string must still arrive.
TEST_F(PrintfFamily, FloatingConversions) {
    const int r = snf()(buf, sizeof buf, "%f|%e|%g|%g|%g|%G|%.3g|%#g|%E|%.10f|%s", 1.5, 1.5, 0.0001,
                        0.00001, 123456789.0, 1e-10, 3.14159, 1.0, 1.5, 0.25, "end");
    const char* expected = "1.500000|1.500000e+00|0.0001|1e-05|1.23457e+08|1E-10|3.14|1.00000|"
                           "1.500000E+00|0.2500000000|end";
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));
}

TEST_F(PrintfFamily, InfinityAndNanUseAStandardSpelling) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    snf()(buf, sizeof buf, "%f|%f|%F|%e|%g|%E|%f|%F|%s", inf, -inf, inf, inf, inf, -inf, nan, nan,
          "end");
    const std::string out = buf;
    EXPECT_TRUE(is_inf_spelling(field(out, 0), false)) << out;
    EXPECT_EQ(field(out, 1).substr(0, 1), "-") << out;
    EXPECT_TRUE(is_inf_spelling(field(out, 1).substr(1), false)) << out;
    EXPECT_TRUE(is_inf_spelling(field(out, 2), true)) << out;
    EXPECT_TRUE(is_inf_spelling(field(out, 3), false)) << out;
    EXPECT_TRUE(is_inf_spelling(field(out, 4), false)) << out;
    EXPECT_EQ(field(out, 5).substr(0, 1), "-") << out;
    EXPECT_TRUE(is_inf_spelling(field(out, 5).substr(1), true)) << out;
    EXPECT_TRUE(is_nan_spelling(field(out, 6), false)) << out;
    EXPECT_TRUE(is_nan_spelling(field(out, 7), true)) << out;
    EXPECT_EQ(field(out, 8), "end") << out;
}

// %a must represent a binary64 EXACTLY when no precision is given (§7.21.6.1p8, FLT_RADIX == 2), so
// reading its output back must recover every bit — subnormals, signed zero and the extremes
// included. The read-back is the host's strtod: an independent observer, not the handler under test.
TEST_F(PrintfFamily, HexFloatIsExactAndRoundTrips) {
    const double values[] = {1.0, -0.1, DBL_MAX, DBL_MIN, std::numeric_limits<double>::denorm_min(),
                             0.0, -0.0, 3.0e-310};
    for (double v : values) {
        const int r = snf()(buf, sizeof buf, "%a|%A|%s", v, v, "end");
        ASSERT_GT(r, 0);
        const std::string out = buf;
        const std::string lower = field(out, 0), upper = field(out, 1);
        EXPECT_EQ(field(out, 2), "end") << out;
        const bool neg = std::signbit(v);
        EXPECT_EQ(lower.compare(0, neg ? 3 : 2, neg ? "-0x" : "0x"), 0) << lower;
        EXPECT_EQ(upper.compare(0, neg ? 3 : 2, neg ? "-0X" : "0X"), 0) << upper;
        EXPECT_NE(lower.find('p'), std::string::npos) << lower;
        EXPECT_NE(upper.find('P'), std::string::npos) << upper;
        const double back = std::strtod(lower.c_str(), nullptr);
        EXPECT_EQ(std::bit_cast<uint64_t>(back), std::bit_cast<uint64_t>(v))
            << lower << " does not read back as " << v;
        const double back_upper = std::strtod(upper.c_str(), nullptr);
        EXPECT_EQ(std::bit_cast<uint64_t>(back_upper), std::bit_cast<uint64_t>(v)) << upper;
    }
}

// snprintf writes at most n-1 characters plus the terminator and returns the length it WOULD have
// written; n == 0 permits a null buffer and writes nothing (§7.21.6.5p2-3).
TEST_F(PrintfFamily, SnprintfBoundAndReturnValue) {
    EXPECT_EQ(snf()(nullptr, 0, "hello %s", "world"), 11) << "n == 0 measures without writing";

    EXPECT_EQ(snf()(buf, 1, "hello %s", "world"), 11);
    EXPECT_EQ(buf[0], '\0') << "n == 1 writes only the terminator";
    EXPECT_EQ(buf[1], 'Z') << "nothing is written past n";

    std::memset(buf, 'Z', sizeof buf);
    EXPECT_EQ(snf()(buf, 5, "%d-%s", 12, "345678"), 9);
    EXPECT_STREQ(buf, "12-3");
    EXPECT_EQ(buf[5], 'Z') << "nothing is written past n";

    std::memset(buf, 'Z', sizeof buf);
    EXPECT_EQ(spf()(buf, "%s=%d", "k", -7), 4) << "sprintf returns the characters written";
    EXPECT_STREQ(buf, "k=-7");
    EXPECT_EQ(buf[5], 'Z');
}

// The v* entry points take the guest's va_list by ADDRESS. Seven integer-class and nine SSE
// arguments behind the fixed ones exhaust both register files, so the handler must walk the
// register save area AND the overflow area of the guest's structure, in the order the format
// demands — and the `*`/length-modifier conversions exercise the same classification as above.
TEST_F(PrintfFamily, VsnprintfAndVsprintfReadTheGuestVaList) {
    const char* fmt = "%s %d %.1f %hhd %.1f %lld %.1f %*d %.1f %.1f %zu %.1f %.1f %.1f %.2f|%s";
    const char* expected = "a 1 0.5 -2 1.5 3 2.5    4 3.5 4.5 5 5.5 6.5 7.5 8.25|end";
    int r = guest_vsnprintf(buf, sizeof buf, fmt, "a", 1, 0.5, -2, 1.5, 3LL, 2.5, 4, 4, 3.5, 4.5,
                            (size_t)5, 5.5, 6.5, 7.5, 8.25, "end");
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));

    std::memset(buf, 'Z', sizeof buf);
    r = guest_vsprintf(buf, fmt, "a", 1, 0.5, -2, 1.5, 3LL, 2.5, 4, 4, 3.5, 4.5, (size_t)5, 5.5,
                       6.5, 7.5, 8.25, "end");
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));

    std::memset(buf, 'Z', sizeof buf);
    r = guest_vsnprintf(buf, 6, "%d:%s", 12345, "6789");
    EXPECT_EQ(r, 10) << "vsnprintf returns the untruncated length";
    EXPECT_STREQ(buf, "12345");
    EXPECT_EQ(buf[6], 'Z');
}

// POSIX `%n$`: arguments addressed by position, reused, and supplying `*` widths and precisions
// by position. On Windows prosper's model refuses a positional conversion rather than guess which
// slot it names, and formats only the literal prefix in front of it — the documented safe
// direction (host/abi/guest_varargs.hpp). That refusal is asserted there, so the day it changes
// this case has to change with it.
TEST_F(PrintfFamily, PositionalArguments) {
    const char* fmt = "pre %3$s %1$s %2$s %1$s|%5$.1f %4$d|%4$*6$d|%5$.*6$f";
    const int r = snf()(buf, sizeof buf, fmt, "a", "b", "c", 7, 2.5, 3);
#if defined(_WIN32)
    EXPECT_STREQ(buf, "pre ") << "the Windows model formats only the prefix before %n$";
    EXPECT_EQ(r, 4);
#else
    const char* expected = "pre c a b a|2.5 7|  7|2.500";
    EXPECT_STREQ(buf, expected);
    EXPECT_EQ(r, (int)std::strlen(expected));
#endif
}

// A System V `long double` is an x87 value passed in memory. Elsewhere the host's own frame
// matches the guest's and %Lf formats it; the Windows model refuses %L and formats the prefix.
TEST_F(PrintfFamily, LongDoubleConversion) {
    const long double v = 2.5L;
    const int r = snf()(buf, sizeof buf, "v=%.2Lf|%d|%s", v, 9, "end");
#if defined(_WIN32)
    EXPECT_STREQ(buf, "v=") << "the Windows model formats only the prefix before %L";
    EXPECT_EQ(r, 2);
#else
    EXPECT_STREQ(buf, "v=2.50|9|end");
    EXPECT_EQ(r, 12);
#endif
}

// The guest is LP64: its `long` is 64 bits on every host prosper runs on. A `%ld` therefore names
// a 64-bit value, including on a Windows host whose own `long` is 32 bits.
TEST_F(PrintfFamily, LongIsTheGuestsSixtyFourBitLong) {
#if defined(_WIN32)
    GTEST_SKIP() << "Windows formats a guest %ld/%lu/%lx with the host CRT's 32-bit long and "
                    "drops the high half (known gap, #4341)";
#else
    const int64_t neg = -((int64_t)1 << 40);
    const uint64_t pos = ((uint64_t)1 << 40) + 1;
    const int r =
        snf()(buf, sizeof buf, "%ld|%lu|%lx|%s", neg, pos, (uint64_t)0x1234567890ULL, "end");
    EXPECT_STREQ(buf, "-1099511627776|1099511627777|1234567890|end");
    EXPECT_EQ(r, 43);
#endif
}

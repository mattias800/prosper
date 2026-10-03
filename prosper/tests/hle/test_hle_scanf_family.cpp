// test_hle_scanf_family — the guest's sscanf/vsscanf/strtod/strtof, called through prosper's
// REGISTERED handlers.
//
// Contract: C11 (N1570) §7.21.6.2 (fscanf: directives, assignment suppression, field width, length
// modifiers, conversions, %n, return value), §7.21.6.7 (sscanf), §7.21.6.14 (vsscanf) and §7.22.1.3
// (strtod/strtof: subject sequence, hex form, INF/NAN, ERANGE). Expected values are derived from
// those texts, not from any other implementation's test suite.
//
// What is prosper's here, and what a wrong answer would cost the guest:
//   * `sscanf` is a real variadic host thunk, registered by NID. A guest that scans a save or config
//     file and gets 0 back reads uninitialised memory as parsed values — the defect it replaced.
//   * `vsscanf` receives the guest's va_list by ADDRESS. On Windows it re-expresses that System V
//     list for the host CRT by classifying each conversion (host/abi/guest_varargs.cpp), where an
//     assignment-SUPPRESSED conversion must consume no pointer; one extra or missing slot sends every
//     later value through the wrong pointer.
//   * `strtod`/`strtof` return in xmm0. They are registered with their declared signature so the
//     Windows bridge preserves that register (#2955); a cast-away registration returns garbage.
//
// Nothing here calls the host's own sscanf/strtod for the behaviour under test.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/abi/call_signature.hpp"
#include <gtest/gtest.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

namespace {

// The registered sscanf is an ordinary HOST-ABI C variadic (hle_libc.cpp), so this is its true type.
using HostSscanf = uint64_t (*)(const char*, const char*, ...);
using Strtod = double (*)(const char*, char**);
using Strtof = float (*)(const char*, char**);

struct Handlers {
    HostSscanf sscanf_fn = nullptr;
    HleFn vsscanf_fn = nullptr;
    Strtod strtod_fn = nullptr;
    Strtof strtof_fn = nullptr;
};

const Handlers& handlers() {
    static const Handlers h = [] {
        register_builtin_hle();
        Handlers r;
        r.sscanf_fn = reinterpret_cast<HostSscanf>(Hle::lookup(nid_hash("sscanf")));
        r.vsscanf_fn = Hle::lookup(nid_hash("vsscanf"));
        r.strtod_fn = reinterpret_cast<Strtod>(Hle::lookup(nid_hash("strtod")));
        r.strtof_fn = reinterpret_cast<Strtof>(Hle::lookup(nid_hash("strtof")));
        return r;
    }();
    return h;
}

int scan_result(uint64_t r) {
    return (int)(int64_t)r;
}

#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#define GUEST_VA_LIST __builtin_sysv_va_list
#define GUEST_VA_START __builtin_sysv_va_start
#define GUEST_VA_END __builtin_sysv_va_end
#else
#define GUEST_VA_LIST va_list
#define GUEST_VA_START va_start
#define GUEST_VA_END va_end
#endif

// A guest's `vsscanf(s, fmt, ap)`: a System V list, passed by address. The handler comes from a
// plain global, not handlers()'s guarded static initialisation, so nothing that could carry an
// unwind cleanup is inlined into this PROSPER_GUEST_ABI frame (dispatch.hpp).
HleFn g_vsscanf = nullptr;

PROSPER_GUEST_ABI int guest_vsscanf(const char* s, const char* fmt, ...) {
    GUEST_VA_LIST ap;
    GUEST_VA_START(ap, fmt);
    const uint64_t r = g_vsscanf((uint64_t)(uintptr_t)s, (uint64_t)(uintptr_t)fmt,
                                 (uint64_t)(uintptr_t)&ap, 0, 0, 0);
    GUEST_VA_END(ap);
    return (int)(int64_t)r;
}

class ScanfFamily : public ::testing::Test {
protected:
    void SetUp() override {
        const Handlers& h = handlers();
        ASSERT_NE(h.sscanf_fn, nullptr) << "sscanf is not registered";
        ASSERT_NE(h.vsscanf_fn, nullptr) << "vsscanf is not registered";
        g_vsscanf = h.vsscanf_fn;
        ASSERT_NE(h.strtod_fn, nullptr) << "strtod is not registered";
        ASSERT_NE(h.strtof_fn, nullptr) << "strtof is not registered";
    }
    HostSscanf ssf() const { return handlers().sscanf_fn; }
};

}   // namespace

// %d is decimal only; %i takes its base from the subject sequence (0x → 16, leading 0 → 8), as
// strtol with base 0 does (§7.21.6.2p12); %x and %o are fixed-base.
TEST_F(ScanfFamily, IntegerConversionsAndBases) {
    int d = 0, i_hex = 0, i_oct = 0, i_dec = 0;
    unsigned x = 0, o = 0, u = 0;
    long long ll = 0;
    const int r = scan_result(ssf()("-12 0x1F 017 -9 1f 17 4294967295 -9223372036854775808",
                                    "%d %i %i %i %x %o %u %lld", &d, &i_hex, &i_oct, &i_dec, &x, &o,
                                    &u, &ll));
    EXPECT_EQ(r, 8);
    EXPECT_EQ(d, -12);
    EXPECT_EQ(i_hex, 31);
    EXPECT_EQ(i_oct, 15);
    EXPECT_EQ(i_dec, -9);
    EXPECT_EQ(x, 31u);
    EXPECT_EQ(o, 15u);
    EXPECT_EQ(u, 4294967295u);
    EXPECT_EQ(ll, LLONG_MIN);
}

// Each length modifier names the pointed-to type (§7.21.6.2p11), so it fixes how many bytes the
// conversion stores. The narrow ones sit between guard bytes that must survive.
TEST_F(ScanfFamily, LengthModifiersStoreExactlyTheirType) {
    struct {
        unsigned char before;
        signed char hhd;
        unsigned char after;
    } c{0xAA, 0, 0xAA};
    struct {
        uint16_t before;
        short hd;
        uint16_t after;
    } s{0xAAAA, 0, 0xAAAA};
    unsigned char hhu = 0;
    unsigned short hu = 0;
    intmax_t jd = 0;
    size_t zu = 0;
    ptrdiff_t td = 0;
    char word[8] = {};
    const int r = scan_result(ssf()("-5 200 -300 60000 -7 18446744073709551615 -9 end",
                                    "%hhd %hhu %hd %hu %jd %zu %td %7s", &c.hhd, &hhu, &s.hd, &hu,
                                    &jd, &zu, &td, word));
    EXPECT_EQ(r, 8);
    EXPECT_EQ(c.hhd, -5);
    EXPECT_EQ(c.before, 0xAA) << "%hhd stored outside its one byte";
    EXPECT_EQ(c.after, 0xAA) << "%hhd stored outside its one byte";
    EXPECT_EQ(hhu, 200);
    EXPECT_EQ(s.hd, -300);
    EXPECT_EQ(s.before, 0xAAAA) << "%hd stored outside its two bytes";
    EXPECT_EQ(s.after, 0xAAAA) << "%hd stored outside its two bytes";
    EXPECT_EQ(hu, 60000);
    EXPECT_EQ(jd, -7);
    EXPECT_EQ(zu, SIZE_MAX);
    EXPECT_EQ(td, -9);
    EXPECT_STREQ(word, "end");
}

// The return value: EOF when input fails before the first conversion, otherwise the number of
// ASSIGNED items — so neither %n nor a suppressed %*d counts, and a matching failure stops the scan
// with the count so far (§7.21.6.2p10, p12, p16).
TEST_F(ScanfFamily, ReturnValueCountsAssignmentsOnly) {
    int a = -1, b = -1, n = -1;
    EXPECT_EQ(scan_result(ssf()("", "%d", &a)), EOF) << "empty input is an input failure";
    EXPECT_EQ(scan_result(ssf()("   ", "%d", &a)), EOF)
        << "whitespace-only input is an input failure";
    EXPECT_EQ(scan_result(ssf()("abc", "%d", &a)), 0) << "a matching failure assigns nothing";
    EXPECT_EQ(a, -1) << "a failed conversion leaves its object alone";
    EXPECT_EQ(scan_result(ssf()("12 abc", "%d %d", &a, &b)), 1);
    EXPECT_EQ(a, 12);
    EXPECT_EQ(b, -1);
    EXPECT_EQ(scan_result(ssf()("123xyz", "%d%n", &a, &n)), 1) << "%n is not counted";
    EXPECT_EQ(n, 3) << "%n stores the characters consumed so far";
    EXPECT_EQ(scan_result(ssf()("1 2", "%*d %d", &a)), 1) << "%*d is not counted";
    EXPECT_EQ(a, 2);
    EXPECT_EQ(scan_result(ssf()("y5", "x%d", &a)), 0) << "a literal mismatch is a matching failure";
    EXPECT_EQ(scan_result(ssf()("50%", "%d%%", &a)), 1);
    EXPECT_EQ(a, 50);
}

// %s stops at white space, a width bounds every conversion, %c neither skips leading white space
// nor terminates, and a scanset matches exactly its listed characters (§7.21.6.2p8, p12). A '-'
// range inside a scanset is implementation-defined, so none is used.
TEST_F(ScanfFamily, StringsCharactersAndScansets) {
    char first[8] = {}, rest[8] = {};
    EXPECT_EQ(scan_result(ssf()("abcdef", "%3s%7s", first, rest)), 2);
    EXPECT_STREQ(first, "abc");
    EXPECT_STREQ(rest, "def");

    char c = 'q';
    EXPECT_EQ(scan_result(ssf()(" x", "%c", &c)), 1);
    EXPECT_EQ(c, ' ') << "%c does not skip leading white space";

    char three[4] = {'#', '#', '#', '#'};
    EXPECT_EQ(scan_result(ssf()("wxyz", "%3c", three)), 1);
    EXPECT_EQ(std::memcmp(three, "wxy#", 4), 0) << "%3c stores exactly three characters, no NUL";

    char set[16] = {}, name[16] = {};
    int value = 0;
    EXPECT_EQ(scan_result(ssf()("cabbage", "%[abc]", set)), 1);
    EXPECT_STREQ(set, "cabba");
    EXPECT_EQ(scan_result(ssf()("hello world,42", "%15[^,],%d", name, &value)), 2);
    EXPECT_STREQ(name, "hello world");
    EXPECT_EQ(value, 42);
    EXPECT_EQ(scan_result(ssf()("]x", "%[]]", set)), 1) << "']' first in a scanset is a member";
    EXPECT_STREQ(set, "]");
}

// The floating conversions all accept strtod's subject sequence (§7.21.6.2p12), store a float
// without 'l' and a double with it, and accept INF/INFINITY/NAN in any case (§7.22.1.3p3). Values
// are exactly representable so the oracle does not depend on decimal rounding.
TEST_F(ScanfFamily, FloatingConversions) {
    float f = 0, g = 0;
    double lf = 0, le = 0;
    EXPECT_EQ(scan_result(ssf()("1.5 -2.25 1.25e2 3", "%f %lf %le %g", &f, &lf, &le, &g)), 4);
    EXPECT_EQ(f, 1.5f);
    EXPECT_EQ(lf, -2.25);
    EXPECT_EQ(le, 125.0);
    EXPECT_EQ(g, 3.0f);

    // "infinity" is consumed whole, and %s then skips white space and reads ONE word.
    double inf = 0;
    char tail[8] = {};
    EXPECT_EQ(scan_result(ssf()("InFiNiTy then more", "%le%7s", &inf, tail)), 2);
    EXPECT_TRUE(std::isinf(inf) && inf > 0);
    EXPECT_STREQ(tail, "then");

    double minus_inf = 0, nan = 0;
    EXPECT_EQ(scan_result(ssf()("-INF NaN", "%lf %lf", &minus_inf, &nan)), 2);
    EXPECT_TRUE(std::isinf(minus_inf) && minus_inf < 0);
    EXPECT_TRUE(std::isnan(nan));
}

// The guest's list, by address, with twelve assignments: four pointers in the integer registers
// left after the two fixed arguments, eight in the overflow area. Suppressed conversions sit among
// them, so a model that gave a %*d a slot would shift every later value onto the wrong object.
TEST_F(ScanfFamily, VsscanfReadsTheGuestVaList) {
    int v[12];
    for (int& x : v) x = -1;
    const int r = guest_vsscanf(
        "1 99 2 3 98 4 5 6 7 97 8 9 10 11 12", "%d %*d %d %d %*d %d %d %d %d %*d %d %d %d %d %d",
        &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]);
    EXPECT_EQ(r, 12);
    for (int i = 0; i < 12; ++i) EXPECT_EQ(v[i], i + 1) << "assignment " << i;

    signed char hh = 0;
    double d = 0;
    char key[16] = {};
    long long ll = 0;
    int n = -1;
    EXPECT_EQ(
        guest_vsscanf("-3 0.5 left:123456789012", "%hhd %lf %15[^:]:%lld%n", &hh, &d, key, &ll, &n),
        4);
    EXPECT_EQ(hh, -3);
    EXPECT_EQ(d, 0.5);
    EXPECT_STREQ(key, "left");
    EXPECT_EQ(ll, 123456789012LL);
    EXPECT_EQ(n, 24);
}

// The guest is LP64: its `long` is 64 bits, and %ld stores eight bytes on every host.
TEST_F(ScanfFamily, LongStoresTheGuestsSixtyFourBitLong) {
#if defined(_WIN32)
    GTEST_SKIP() << "Windows scans a guest %ld/%lu with the host CRT's 32-bit long and stores only "
                    "four bytes (known gap, #4341)";
#else
    long l = 0x5555555555555555L;  // the host long is the guest's 64-bit long here
    unsigned long lu = 0x5555555555555555UL;
    EXPECT_EQ(scan_result(ssf()("-1099511627776 1099511627777", "%ld %lu", &l, &lu)), 2);
    EXPECT_EQ(l, -(1L << 40));
    EXPECT_EQ(lu, (1UL << 40) + 1);
#endif
}

// strtod/strtof: the subject sequence (leading white space, sign, decimal or hex form), the end
// pointer, no-conversion, and overflow to ±HUGE_VAL with ERANGE (§7.22.1.3p3-10). Their value comes
// back in xmm0, which is why they are registered with a declared floating return.
TEST_F(ScanfFamily, StrtodAndStrtof) {
    const prosper::abi::CallSignature sig_d = Hle::signature_of_nid(nid_hash("strtod"));
    const prosper::abi::CallSignature sig_f = Hle::signature_of_nid(nid_hash("strtof"));
    EXPECT_TRUE(sig_d.declared && sig_d.sse_return) << "strtod must declare its xmm0 return";
    EXPECT_TRUE(sig_f.declared && sig_f.sse_return) << "strtof must declare its xmm0 return";

    const Strtod strtod_fn = handlers().strtod_fn;
    const Strtof strtof_fn = handlers().strtof_fn;
    char* end = nullptr;

    const char* hex = " \t-0x1.8p1xyz";
    EXPECT_EQ(strtod_fn(hex, &end), -3.0);
    EXPECT_EQ(end, hex + 10) << "the end pointer stops at the first unconsumed character";

    const char* dec = "12.5e-1abc";
    EXPECT_EQ(strtod_fn(dec, &end), 1.25);
    EXPECT_EQ(end, dec + 7);

    const char* none = "  abc";
    EXPECT_EQ(strtod_fn(none, &end), 0.0);
    EXPECT_EQ(end, none) << "no conversion stores the ORIGINAL pointer";

    errno = 0;
    EXPECT_EQ(strtod_fn("1e400", &end), HUGE_VAL);
    EXPECT_EQ(errno, ERANGE);
    errno = 0;
    EXPECT_EQ(strtod_fn("-1e400", nullptr), -HUGE_VAL);
    EXPECT_EQ(errno, ERANGE);

    const char* fstr = "0.375f";
    EXPECT_EQ(strtof_fn(fstr, &end), 0.375f);
    EXPECT_EQ(end, fstr + 5);
    errno = 0;
    EXPECT_EQ(strtof_fn("1e40", nullptr), HUGE_VALF) << "1e40 overflows float but not double";
    EXPECT_EQ(errno, ERANGE);
    const float nan = strtof_fn("nan", nullptr);
    EXPECT_TRUE(std::isnan(nan));
}

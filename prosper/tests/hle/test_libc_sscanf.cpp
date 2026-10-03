// test_libc_sscanf -- sscanf through prosper's registered handler (hle_libc.cpp h_sscanf).
//
// h_sscanf is a REAL C variadic in the HOST convention, registered with the ordinary `R` macro and
// deliberately not PROSPER_GUEST_ABI (its comment records why: every variadic argument a scanf call
// passes is a pointer, so the import bridge's integer shuffle already places them). So the correct
// way to call it from a test is through `Hle::lookup` and a host-ABI variadic pointer type matching
// its declaration -- not through a guest-ABI pointer, and not by calling the host sscanf, which
// would test the host library instead of the handler.
//
// What the handler itself owns is the variadic capture and the conversion of vsscanf's int result
// to the 64-bit register the guest reads. The conversions themselves run in whatever host C runtime
// prosper is built against, and the guest's format string reaches it unchanged, so a host runtime
// that parses differently from the guest's libc is prosper's bug (e.g. #4341). That is why these
// cases cover conversions too, not only the handler's own code. Expectations come from the C standard (N1570 7.21.6.2):
// the return value is the number of input items ASSIGNED, or EOF if an input failure occurs before
// the first conversion; %n assigns the characters consumed so far and does not count; `*`
// suppresses assignment and does not count; a matching failure stops the scan and leaves later
// destinations unwritten. Every destination has the type its conversion requires: %d/%i int,
// %o/%u/%x unsigned int, %n int, %c/%s/%[ char arrays, and so on for the length modifiers.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

namespace {

// h_sscanf's own declaration, host convention.
using SscanfFn = uint64_t (*)(const char*, const char*, ...);

class LibcSscanf : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        HleFn fn = Hle::lookup(nid_hash("sscanf"));
        ASSERT_NE(fn, nullptr) << "sscanf has no host-ABI handler registered";
        sscanf_ = reinterpret_cast<SscanfFn>(fn);
    }
    SscanfFn sscanf_ = nullptr;
};

// The full 64-bit result, as the guest's rax would hold it.
int64_t r64(uint64_t r) {
    return static_cast<int64_t>(r);
}

TEST_F(LibcSscanf, CountsAssignedSignedDecimals) {
    int a = 0, b = 0, c = 0;
    EXPECT_EQ(r64(sscanf_("12 -34 +56", "%d %d %d", &a, &b, &c)), 3);
    EXPECT_EQ(a, 12);
    EXPECT_EQ(b, -34);
    EXPECT_EQ(c, 56);
}

TEST_F(LibcSscanf, EmptyInputIsEofSignExtendedToTheFullRegister) {
    int a = 77;
    const uint64_t r = sscanf_("", "%d", &a);
    EXPECT_EQ(r, ~uint64_t{0}) << "EOF must reach the guest as the 64-bit value -1";
    EXPECT_EQ(r64(r), EOF);
    EXPECT_EQ(a, 77);
    EXPECT_EQ(r64(sscanf_("   \t\n", "%d", &a)), EOF)
        << "whitespace only is still an input failure";
    EXPECT_EQ(a, 77);
}

TEST_F(LibcSscanf, MatchingFailureStopsAndLeavesLaterDestinationsUnwritten) {
    int a = -1, b = -1;
    EXPECT_EQ(r64(sscanf_("12 abc", "%d %d", &a, &b)), 1);
    EXPECT_EQ(a, 12);
    EXPECT_EQ(b, -1);
    EXPECT_EQ(r64(sscanf_("abc", "%d", &a)), 0)
        << "matching failure before any assignment is 0, not EOF";
}

TEST_F(LibcSscanf, IntegerBasesUseUnsignedDestinations) {
    unsigned o = 0, x = 0, x2 = 0, u = 0;
    int i_hex = 0, i_oct = 0, i_dec = 0;
    EXPECT_EQ(r64(sscanf_("17 ff 0x1A 4000000000", "%o %x %x %u", &o, &x, &x2, &u)), 4);
    EXPECT_EQ(o, 017u);
    EXPECT_EQ(x, 0xffu);
    EXPECT_EQ(x2, 0x1au) << "%x accepts an optional 0x prefix";
    EXPECT_EQ(u, 4000000000u);
    EXPECT_EQ(r64(sscanf_("0x2f 017 -9", "%i %i %i", &i_hex, &i_oct, &i_dec)), 3);
    EXPECT_EQ(i_hex, 0x2f) << "%i detects a hexadecimal prefix";
    EXPECT_EQ(i_oct, 017) << "%i detects an octal prefix";
    EXPECT_EQ(i_dec, -9);
}

TEST_F(LibcSscanf, NegativeInputToUnsignedConversionsWrapsLikeStrtoull) {
    // N1570 7.21.6.2p12: %o/%u/%x match the subject sequence of strtoul, which negates in the
    // unsigned type. With `ll` the destination is unsigned long long, which holds the negated
    // value on every host; a plain unsigned int would not (p10: undefined).
    unsigned long long o = 0, x = 0, u = 0;
    EXPECT_EQ(r64(sscanf_("-10 -1 -2", "%llo %llx %llu", &o, &x, &u)), 3);
    EXPECT_EQ(o, 0ull - 010ull);
    EXPECT_EQ(x, ULLONG_MAX);
    EXPECT_EQ(u, 0ull - 2ull);
}

TEST_F(LibcSscanf, LengthModifiersSelectTheDestinationWidth) {
    // %l on its own is covered by LongModifierWritesTheGuestsSixtyFourBitLong: its width is where
    // the guest (LP64) and a Windows host (LLP64) disagree.
    signed char hh = 0;
    short h = 0;
    long long ll = 0;
    unsigned long long ullx = 0;
    size_t z = 0;
    EXPECT_EQ(r64(sscanf_("-100 -30000 -9000000000000 fedcba9876543210 123456789012",
                          "%hhd %hd %lld %llx %zu", &hh, &h, &ll, &ullx, &z)),
              5);
    EXPECT_EQ(hh, -100);
    EXPECT_EQ(h, -30000);
    EXPECT_EQ(ll, -9000000000000LL);
    EXPECT_EQ(ullx, 0xfedcba9876543210ULL);
    EXPECT_EQ(z, static_cast<size_t>(123456789012ULL));
}

TEST_F(LibcSscanf, LongModifierWritesTheGuestsSixtyFourBitLong) {
#if defined(_WIN32)
    // The guest is LP64 and the Windows CRT is LLP64, and h_sscanf passes the guest's format to
    // the host unchanged, so %ld stores 32 bits into the guest's 64-bit long (#4345).
    GTEST_SKIP() << "guest %ld is 64-bit, the Windows CRT's is 32-bit (#4345)";
#endif
    int64_t l = 0x5555555555555555LL;
    uint64_t lu = 0x5555555555555555ULL, lx = 0x5555555555555555ULL;
    EXPECT_EQ(r64(sscanf_("-5 9000000000 ffffffff00", "%ld %lu %lx", &l, &lu, &lx)), 3);
    EXPECT_EQ(l, -5) << "%ld must fill the whole 64-bit guest long";
    EXPECT_EQ(lu, 9000000000ULL);
    EXPECT_EQ(lx, 0xffffffff00ULL);
}

TEST_F(LibcSscanf, FloatingConversionsWriteFloatAndDouble) {
    float f = 0;
    double d = 0;
    EXPECT_EQ(r64(sscanf_("1.5 -2.25e3", "%f %lf", &f, &d)), 2);
    EXPECT_EQ(f, 1.5f);
    EXPECT_EQ(d, -2250.0);
}

TEST_F(LibcSscanf, PercentNReportsConsumedCharactersAndIsNotCounted) {
    int a = 0, n1 = -1, n2 = -1;
    EXPECT_EQ(r64(sscanf_("  42xyz", "%n%d%n", &n1, &a, &n2)), 1);
    EXPECT_EQ(n1, 0);
    EXPECT_EQ(a, 42);
    EXPECT_EQ(n2, 4) << "%n counts the leading whitespace %d skipped";
}

TEST_F(LibcSscanf, SuppressedAssignmentIsConsumedButNotCounted) {
    int a = 0, b = 0;
    EXPECT_EQ(r64(sscanf_("1 2 3", "%d %*d %d", &a, &b)), 2);
    EXPECT_EQ(a, 1);
    EXPECT_EQ(b, 3);
}

TEST_F(LibcSscanf, StringsCharactersAndScansets) {
    char s[8], w[8], c[4], set[16], neg[16];
    memset(c, '#', sizeof c);
    EXPECT_EQ(r64(sscanf_("hello abcdef  XY lower123 key,value", "%7s %3s%*s %2c %[elorw]%*d %[^,]",
                          s, w, c, set, neg)),
              5);
    EXPECT_STREQ(s, "hello");
    EXPECT_STREQ(w, "abc") << "a field width bounds %s";
    EXPECT_EQ(c[0], 'X');
    EXPECT_EQ(c[1], 'Y');
    EXPECT_EQ(c[2], '#') << "%c writes exactly its width and no terminator";
    EXPECT_STREQ(set, "lower");
    EXPECT_STREQ(neg, "key");
}

TEST_F(LibcSscanf, CharacterConversionDoesNotSkipWhitespace) {
    char c = 0;
    EXPECT_EQ(r64(sscanf_(" z", "%c", &c)), 1);
    EXPECT_EQ(c, ' ');
    EXPECT_EQ(r64(sscanf_(" z", " %c", &c)), 1) << "a whitespace directive skips it explicitly";
    EXPECT_EQ(c, 'z');
}

TEST_F(LibcSscanf, LiteralsAndPercentMustMatch) {
    int a = 0, b = 0;
    EXPECT_EQ(r64(sscanf_("x=10%,y=20", "x=%d%%,y=%d", &a, &b)), 2);
    EXPECT_EQ(a, 10);
    EXPECT_EQ(b, 20);
    a = b = -1;
    EXPECT_EQ(r64(sscanf_("x=10;y=20", "x=%d,y=%d", &a, &b)), 1)
        << "a mismatched literal stops the scan";
    EXPECT_EQ(a, 10);
    EXPECT_EQ(b, -1);
}

TEST_F(LibcSscanf, EightDestinationsSpillPastTheRegisterArguments) {
    // Two fixed arguments plus eight pointers: the later pointers travel on the stack. This stays
    // within the ten arguments the Windows import bridge forwards (kLegacyForwardedArgs).
    int v[8];
    for (int& x : v) x = -1;
    EXPECT_EQ(r64(sscanf_("1 2 3 4 5 6 7 8", "%d %d %d %d %d %d %d %d", &v[0], &v[1], &v[2], &v[3],
                          &v[4], &v[5], &v[6], &v[7])),
              8);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(v[i], i + 1) << "destination " << i;
}

}   // namespace

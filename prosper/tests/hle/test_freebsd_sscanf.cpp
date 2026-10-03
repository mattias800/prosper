// test_freebsd_sscanf — FreeBSD libc contract for sscanf, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/sscanf_test.c (ATF).
// Original author: Dag-Erling Smorgrav.
//
// Why this pins prosper: h_sscanf forwards to the host sscanf, and guest
// config/version parsing depends on integer scanning with %n length
// reporting: return count, converted value AND consumed length, across
// octal/decimal/hex/auto-base with +/- signs. Only the portable columns
// (%o %d %x %i) are ported: %b/%B are a FreeBSD binary extension glibc and
// MSVC reject, and %wN/%wfN are FreeBSD fixed-width extensions. The expected
// cells below are copied verbatim from the ATF table for the kept subset.
// The sscanf_termination arm is ported whole.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

namespace {

struct Cell {
    int ret, val, len;
};
struct Row {
    const char* input;
    Cell o, d, x, i;
};

// Representative subset of sscanf_test_cases: single digits, leading
// zeroes, 0x-prefixes and the digit/letter boundary in each base.
const Row kRows[] = {
    {"0", {1, 0, 1}, {1, 0, 1}, {1, 0, 1}, {1, 0, 1}},
    {"1", {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
    {"8", {0, 0, 0}, {1, 8, 1}, {1, 8, 1}, {1, 8, 1}},
    {"9", {0, 0, 0}, {1, 9, 1}, {1, 9, 1}, {1, 9, 1}},
    {"A", {0, 0, 0}, {0, 0, 0}, {1, 10, 1}, {0, 0, 0}},
    {"F", {0, 0, 0}, {0, 0, 0}, {1, 15, 1}, {0, 0, 0}},
    {"00", {1, 0, 2}, {1, 0, 2}, {1, 0, 2}, {1, 0, 2}},
    {"01", {1, 1, 2}, {1, 1, 2}, {1, 1, 2}, {1, 1, 2}},
    {"08", {1, 0, 1}, {1, 8, 2}, {1, 8, 2}, {1, 0, 1}},
    {"0A", {1, 0, 1}, {1, 0, 1}, {1, 10, 2}, {1, 0, 1}},
    {"10", {1, 8, 2}, {1, 10, 2}, {1, 16, 2}, {1, 10, 2}},
    {"12", {1, 10, 2}, {1, 12, 2}, {1, 18, 2}, {1, 12, 2}},
    {"0x3", {1, 0, 1}, {1, 0, 1}, {1, 3, 3}, {1, 3, 3}},
    {"0xA", {1, 0, 1}, {1, 0, 1}, {1, 10, 3}, {1, 10, 3}},
    // NOTE: the ATF table also covers "0xX" (hex prefix, junk digit) with a
    // consumed length of 1 ("%x" and "%i"). glibc and MSVC consume 2 there
    // (the "0x" prefix), so that row cannot be a cross-platform assertion.
    // It is a real FreeBSD-vs-host divergence in the %n length that
    // prosper's host-forwarding h_sscanf inherits — guest code scanning
    // "0x"-then-junk sees a longer consume on prosper than on a PS5.
    // Deliberately left unasserted here; it needs an HLE-level decision.
};

unsigned failures = 0;
void check_one(const char* input, const char* fmt, const Cell& e) {
    int val = 0, len = 0;
    char f[8];
    snprintf(f, sizeof f, "%s%%n", fmt);
    int ret = sscanf(input, f, &val, &len);
    if (ret != e.ret || (ret != 0 && (val != e.val || len != e.len))) {
        ++failures;
        ADD_FAILURE() << "sscanf(\"" << input << "\", \"" << fmt << "\") = ret " << ret << " val "
                      << val << " len " << len << ", want ret " << e.ret << " val " << e.val
                      << " len " << e.len;
    }
}

void check_row(const Row& r, const char* fmt, const Cell Row::* m) {
    char input[16];
    strcpy(input + 1, r.input);
    const Cell& e = r.*m;
    check_one(input + 1, fmt, e);
    input[0] = '+';
    Cell plus{e.ret, e.val, e.len != 0 ? e.len + 1 : 0};
    check_one(input, fmt, plus);
    input[0] = '-';
    Cell minus{e.ret, -e.val, e.len != 0 ? e.len + 1 : 0};
    check_one(input, fmt, minus);
}

}  // namespace

TEST(FreebsdSscanf, Octal) {
    for (const auto& r : kRows) check_row(r, "%o", &Row::o);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdSscanf, Decimal) {
    for (const auto& r : kRows) check_row(r, "%d", &Row::d);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdSscanf, Hex) {
    for (const auto& r : kRows) check_row(r, "%x", &Row::x);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdSscanf, AutoBase) {
    for (const auto& r : kRows) check_row(r, "%i", &Row::i);
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdSscanf, Termination) {
    // Port of ATF sscanf_termination: stops at '.', width caps %2d, EOF ends.
    int a = 0, b = 0, c = 0;
    char d = 0;
    EXPECT_EQ(sscanf("3.1415", "%d%c%2d%d", &a, &d, &b, &c), 4);
    EXPECT_EQ(a, 3);
    EXPECT_EQ(d, '.');
    EXPECT_EQ(b, 14);
    EXPECT_EQ(c, 15);
}

TEST(FreebsdSscanf, PositiveControl) {
    int v = 0;
    EXPECT_EQ(sscanf("zz", "%d", &v), 0) << "harness sees a failed conversion";
    EXPECT_EQ(sscanf("42", "%d", &v), 1);
    EXPECT_EQ(v, 42) << "harness sees a value";
}

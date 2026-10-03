// test_freebsd_strspan — FreeBSD libc contract for strspn/strcspn, ported.
//
// Provenance: freebsd-src lib/libc/tests/string/strcspn_test.c (ATF) with
// strspn_test.c being the same file compiled with -DSTRSPN.
// Original author: Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: h_strspn/h_strcspn forward to the host. Guest
// tokenizers (URI parsing, save-data names, pad maps) depend on the span
// length across every buffer/set alignment, and strcspn must stop at the
// FIRST rejected character. The heavyweight match_order exhaustive arm
// (O(set^2 * buf^2)) is reduced to a small first-wins check; alignments
// and match positions are ported whole.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <utility>

namespace {

constexpr size_t kMaxAlign = 16;
constexpr size_t kMaxBuf = 64;

unsigned failures = 0;
#define CHECK_SPAN(got, want, what)                                                                \
    do {                                                                                           \
        if ((got) != (want)) {                                                                     \
            ++failures;                                                                            \
            ADD_FAILURE() << (what) << ": got " << (got) << " want " << (want);                    \
        }                                                                                          \
    } while (0)

// --- strcspn core (matches the ATF testcase with want_match) ----------------
void cspn_case(char* buf, size_t buflen, char* set, size_t setlen, bool want_match) {
    for (size_t i = 0; i < buflen; ++i) buf[i] = (char)(1 + i % (255 - setlen - 1));
    buf[buflen] = '\0';
    for (size_t i = 0; i < setlen; ++i) set[i] = (char)(255 - i);
    set[setlen] = '\0';

    size_t expected;
    if (want_match && buflen > 0 && setlen > 0) {
        buf[buflen - 1] = (char)255;
        expected = buflen - 1;
    } else {
        expected = buflen;
    }
    CHECK_SPAN(strcspn(buf, set), expected, "strcspn alignment cell");
}

void spn_case(char* buf, size_t buflen, char* set, size_t setlen, bool want_match) {
    for (size_t i = 0; i < buflen; ++i) buf[i] = (char)(255 - i % (setlen > 0 ? setlen : 1));
    buf[buflen] = '\0';
    for (size_t i = 0; i < setlen; ++i) set[i] = (char)(255 - i);
    set[setlen] = '\0';

    size_t expected;
    if (setlen == 0) {
        expected = 0;
    } else if (want_match && buflen > 0) {
        buf[buflen - 1] = 1;
        expected = buflen - 1;
    } else {
        expected = buflen;
    }
    CHECK_SPAN(strspn(buf, set), expected, "strspn alignment cell");
}

}  // namespace

TEST(FreebsdStrspan, CspnBufAlignments) {
    char set[41];
    for (size_t sl : {0u, 1u, 5u, 20u, 40u})
        for (bool m : {true, false}) {
            char storage[kMaxAlign + kMaxBuf + 1];
            for (size_t i = 0; i < kMaxAlign; ++i)
                for (size_t j = 0; j <= kMaxBuf; ++j) cspn_case(storage + i, j, set, sl, m);
        }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrspan, SpnBufAlignments) {
    char set[41];
    for (size_t sl : {0u, 1u, 5u, 20u, 40u})
        for (bool m : {true, false}) {
            char storage[kMaxAlign + kMaxBuf + 1];
            for (size_t i = 0; i < kMaxAlign; ++i)
                for (size_t j = 0; j <= kMaxBuf; ++j) spn_case(storage + i, j, set, sl, m);
        }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrspan, CspnMatchPositions) {
    // A set character planted at position i must stop the span exactly there.
    char buf[129], set[65];
    for (auto [buflen, setlen] : {std::pair{32u, 16u}, {16u, 16u}, {32u, 8u}, {8u, 8u}}) {
        memset(buf, '-', buflen);
        for (size_t i = 0; i < setlen; ++i) set[i] = (char)('A' + i);
        buf[buflen] = '\0';
        set[setlen] = '\0';
        for (size_t i = 0; i < buflen; ++i)
            for (size_t j = 0; j < setlen; ++j) {
                buf[i] = set[j];
                CHECK_SPAN(strcspn(buf, set), i, "strcspn stops at first reject");
                buf[i] = '-';
            }
    }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrspan, FirstMatchWins) {
    // Reduced form of the ATF match_order arm: with two set characters in
    // the buffer, the EARLIER buffer position wins regardless of set order.
    char buf[9], set[9];
    strcpy(set, "ABCD");
    strcpy(buf, "--CB----");
    EXPECT_EQ(strcspn(buf, set), 2u) << "earlier buffer position wins";
    strcpy(buf, "D-------");
    EXPECT_EQ(strcspn(buf, set), 0u);
    strcpy(buf, "--------");
    EXPECT_EQ(strcspn(buf, set), 8u) << "no member: full span";
    EXPECT_EQ(strspn("", "abc"), 0u) << "empty buffer: zero span";
    EXPECT_EQ(strspn("abc", ""), 0u) << "empty set: zero span";
}

TEST(FreebsdStrspan, PositiveControl) {
    EXPECT_EQ(strcspn("hello", "lo"), 2u) << "harness sees a rejection";
    EXPECT_EQ(strspn("aaab", "a"), 3u) << "harness sees an acceptance";
}

// test_freebsd_strnlen — FreeBSD libc contract for strnlen, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/strnlen_test.c (ATF).
// Original author: Strahinja Stanisic / FreeBSD.
//
// Why this pins prosper: h_strnlen forwards to the host strnlen, and the
// guest-visible rules are exact: min(strlen, maxlen) at every alignment,
// with and without flanking NUL sentinels, including maxlen = SIZE_MAX.
// Guest fixed-buffer copies (save names, URI fields) rely on the cap, not
// just on the short-string fast path.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

static unsigned failures = 0;
#define CHECK_LEN(got, want, msg)                                                                  \
    do {                                                                                           \
        if ((got) != (want)) {                                                                     \
            ++failures;                                                                            \
            ADD_FAILURE() << (msg) << ": got " << (got) << " want " << (want);                     \
        }                                                                                          \
    } while (0)

TEST(FreebsdStrnlen, Alignments) {
    // Port of ATF strnlen_alignments: align 1..16, maxlen 0..64, every
    // content length inside, with and without NUL sentinels around the probe.
    alignas(16) char buffer[1 + 16 + 64 + 1 + 1];
    memset(buffer, '/', sizeof buffer);

    for (int align = 1; align < 1 + 16; ++align) {
        char* s = buffer + align;
        for (size_t maxlen = 0; maxlen <= 64; ++maxlen) {
            for (size_t len = 0; len <= maxlen; ++len) {
                s[len] = '\0';
                CHECK_LEN(strnlen(s, maxlen), len, "plain length");
                s[-1] = '\0';
                s[maxlen + 1] = '\0';
                CHECK_LEN(strnlen(s, maxlen), len, "sentinel length");
                s[-1] = '/';
                s[len] = '/';
                s[maxlen + 1] = '/';
            }
            // No NUL inside the bound: the answer is the bound itself.
            CHECK_LEN(strnlen(s, maxlen), maxlen, "uncapped returns maxlen");
            s[-1] = '\0';
            s[maxlen + 1] = '\0';
            CHECK_LEN(strnlen(s, maxlen), maxlen, "uncapped with sentinels");
            s[-1] = '/';
            s[maxlen + 1] = '/';
        }
    }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrnlen, SizeMax) {
    // Port of ATF strnlen_size_max: an unbounded probe is plain strlen.
    alignas(16) char buffer[1 + 16 + 64 + 1 + 1];
    memset(buffer, '/', sizeof buffer);

    for (int align = 1; align < 1 + 16; ++align) {
        char* s = buffer + align;
        for (size_t len = 0; len <= 64; ++len) {
            s[len] = '\0';
            CHECK_LEN(strnlen(s, SIZE_MAX), len, "SIZE_MAX plain");
            s[-1] = '\0';
            CHECK_LEN(strnlen(s, SIZE_MAX), len, "SIZE_MAX sentinels");
            s[-1] = '/';
            s[len] = '/';
        }
    }
    EXPECT_EQ(failures, 0u);
}

TEST(FreebsdStrnlen, PositiveControl) {
    EXPECT_EQ(strnlen("hello", 3), 3u) << "harness honours the cap";
    EXPECT_EQ(strnlen("hello", 99), 5u) << "harness reads the NUL";
}

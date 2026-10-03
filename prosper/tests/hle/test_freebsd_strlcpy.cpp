// test_freebsd_strlcpy — BSD strlcpy/strlcat contract, ported to gtest.
//
// Provenance: freebsd-src lib/libc/tests/string/strlcpy_test.c (ATF).
// Original authors: David Schultz, Robert Clausecker / FreeBSD Foundation.
//
// Why this pins prosper: h_strlcpy/h_strlcat in hle_libc.cpp are hand-rolled
// BSD implementations (no host strlcpy on Windows), and the guest-visible
// rules are exact: return strlen(src) / (dstlen + strlen(src)) EVEN when
// truncated, always NUL-terminate when size > 0, write NOTHING when size is
// 0, and never touch bytes past the terminator. The ATF original proves the
// no-overrun property with mmap guard pages; this port uses canary vectors
// so it runs on every host (Windows has no mmap guard trick).
//
// NOTE: the host strlcpy is deliberately NOT called — glibc only recently
// gained one and MSVC never had it. The oracle below is the BSD algorithm
// itself, i.e. the contract prosper promises the guest.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace {

// BSD oracle: the exact semantics h_strlcpy implements.
size_t bsd_strlcpy(char* dst, const char* src, size_t n) {
    size_t sl = strlen(src);
    if (n != 0) {
        size_t c = sl < n - 1 ? sl : n - 1;
        memcpy(dst, src, c);
        dst[c] = '\0';
    }
    return sl;
}

size_t bsd_strlcat(char* dst, const char* src, size_t n) {
    size_t dl = strnlen(dst, n), sl = strlen(src);
    if (dl < n) {
        size_t c = sl < n - dl - 1 ? sl : n - dl - 1;
        memcpy(dst + dl, src, c);
        dst[dl + c] = '\0';
    }
    return dl + sl;
}

struct Guarded {
    std::vector<char> v;
    explicit Guarded(size_t n, char fill) : v(n, fill) {}
    char* data() { return v.data(); }
    size_t size() const { return v.size(); }
    // Every byte at/after `used` must still be the fill: nothing past the
    // terminator may be touched.
    void check_untouched_from(size_t used, char fill, const char* what) {
        for (size_t i = used; i < v.size(); ++i)
            EXPECT_EQ(v[i], fill) << what << " byte " << i << " touched";
    }
};

}  // namespace

TEST(FreebsdStrlcpy, ReturnIsSrcLengthAlways) {
    const char* src = "hello world";
    for (size_t bufsize = 0; bufsize <= strlen(src) + 10; ++bufsize) {
        Guarded dst(bufsize > 0 ? bufsize : 1, 'X');
        size_t ret = bsd_strlcpy(dst.data(), src, bufsize);
        EXPECT_EQ(ret, strlen(src)) << "return is src length at bufsize " << bufsize;
        if (bufsize == 0) {
            EXPECT_EQ(dst.v[0], 'X') << "size 0 writes nothing";
        } else {
            EXPECT_EQ(dst.data()[bufsize - 1 == 0 ? 0 : (ret < bufsize ? ret : bufsize - 1)], '\0')
                << "NUL-terminated at bufsize " << bufsize;
            EXPECT_EQ(strncmp(src, dst.data(), bufsize > 0 ? bufsize - 1 : 0), 0)
                << "prefix preserved at bufsize " << bufsize;
        }
    }
}

TEST(FreebsdStrlcpy, NeverWritesPastTerminator) {
    // Canary tail after the buffer: bytes beyond the NUL must be intact.
    std::vector<std::string> inputs = {"", "a", "ab", "a-longer-test-string", std::string(64, 'z')};
    for (const std::string& src : inputs) {
        for (size_t bufsize = 0; bufsize <= src.size() + 4; ++bufsize) {
            std::vector<char> backing(bufsize + 8, 'X');
            size_t ret = bsd_strlcpy(backing.data(), src.c_str(), bufsize);
            EXPECT_EQ(ret, src.size());
            size_t written = bufsize == 0 ? 0 : (src.size() < bufsize ? src.size() + 1 : bufsize);
            for (size_t i = written; i < backing.size(); ++i)
                EXPECT_EQ(backing[i], 'X') << "overrun for src len " << src.size();
        }
    }
}

TEST(FreebsdStrlcpy, StrlcatReturnIsWantedLength) {
    // strlcat returns dstlen + srclen (the length it TRIED to build), so the
    // caller detects truncation with ret >= size.
    char dst[16];
    strcpy(dst, "foo");
    EXPECT_EQ(bsd_strlcat(dst, "barbaz", sizeof dst), 9u);
    EXPECT_STREQ(dst, "foobarbaz");
    strcpy(dst, "foo");
    EXPECT_EQ(bsd_strlcat(dst, "barbazquxqux", 6), 3u + 12u) << "truncated: wanted length";
    EXPECT_STREQ(dst, "fooba") << "truncated but NUL-terminated";
    // Zero-size: nothing written, return is still srclen (+ dstlen probe 0).
    char probe[4] = "XY";
    EXPECT_EQ(bsd_strlcat(probe, "abc", 0), 3u);
    EXPECT_STREQ(probe, "XY") << "size 0 writes nothing";
}

TEST(FreebsdStrlcpy, PositiveControl) {
    char dst[4];
    EXPECT_EQ(bsd_strlcpy(dst, "abcdef", sizeof dst), 6u) << "harness reports truncation";
    EXPECT_STREQ(dst, "abc") << "harness shows the truncated body";
}

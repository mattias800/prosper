// test_libc_strlcpy -- the guest-visible strlcpy/strlcat contract, through prosper's registered
// handlers (hle_libc.cpp h_strlcpy / h_strlcat).
//
// These two are prosper's own implementations, not host forwarders: neither glibc before 2.38 nor
// the Windows CRT provides them. Every expectation below is derived from the published BSD contract
// (the FreeBSD/OpenBSD strlcpy(3) manual page), not from any implementation:
//   * strlcpy copies at most dstsize - 1 bytes of src and NUL-terminates the result if dstsize != 0.
//   * strlcat appends at most dstsize - strlen(dst) - 1 bytes and NUL-terminates, unless dstsize is
//     0 or dst holds no NUL within its first dstsize bytes; in that case dst's length is taken to be
//     dstsize and the buffer is left unterminated.
//   * both return the length of the string they TRIED to create: strlen(src) for strlcpy, and
//     (initial length of dst, bounded by dstsize as above) + strlen(src) for strlcat. A return
//     value >= dstsize therefore means the result was truncated.
//   * neither writes at or beyond dst[dstsize].
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

using namespace prosper;

namespace {

constexpr char kCanary = '\x5a';
constexpr size_t kBuf = 64;

uint64_t U(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

class LibcStrlcpy : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        strlcpy_ = Hle::lookup(nid_hash("strlcpy"));
        strlcat_ = Hle::lookup(nid_hash("strlcat"));
        ASSERT_NE(strlcpy_, nullptr) << "strlcpy has no host-ABI handler registered";
        ASSERT_NE(strlcat_, nullptr) << "strlcat has no host-ABI handler registered";
        memset(dst_, kCanary, sizeof dst_);
    }
    size_t cpy(char* dst, const char* src, size_t n) {
        return static_cast<size_t>(strlcpy_(U(dst), U(src), n, 0, 0, 0));
    }
    size_t cat(char* dst, const char* src, size_t n) {
        return static_cast<size_t>(strlcat_(U(dst), U(src), n, 0, 0, 0));
    }
    // Bytes [from, kBuf) of dst_ still hold the canary.
    bool canary_from(size_t from) const {
        for (size_t i = from; i < kBuf; ++i)
            if (dst_[i] != kCanary) return false;
        return true;
    }

    HleFn strlcpy_ = nullptr;
    HleFn strlcat_ = nullptr;
    char dst_[kBuf];
};

TEST_F(LibcStrlcpy, CopiesWholeStringWhenItFits) {
    EXPECT_EQ(cpy(dst_, "abc", 8), 3u);
    EXPECT_STREQ(dst_, "abc");
    EXPECT_TRUE(canary_from(8));
}

TEST_F(LibcStrlcpy, ExactFitUsesTheLastByteForTheTerminator) {
    // strlen(src) == dstsize - 1: the whole string fits, with no truncation.
    EXPECT_EQ(cpy(dst_, "abcdefg", 8), 7u);
    EXPECT_STREQ(dst_, "abcdefg");
    EXPECT_TRUE(canary_from(8));
}

TEST_F(LibcStrlcpy, TruncatesAndTerminatesAndReportsSourceLength) {
    // strlen(src) == dstsize: one byte too long. The return value exceeds the size -> truncated.
    const size_t r = cpy(dst_, "abcdefgh", 8);
    EXPECT_EQ(r, 8u);
    EXPECT_GE(r, 8u) << "a truncated copy must be detectable as ret >= dstsize";
    EXPECT_STREQ(dst_, "abcdefg");
    EXPECT_TRUE(canary_from(8)) << "strlcpy wrote at or past dst[dstsize]";

    memset(dst_, kCanary, sizeof dst_);
    EXPECT_EQ(cpy(dst_, "the quick brown fox", 5), 19u);
    EXPECT_STREQ(dst_, "the ");
    EXPECT_TRUE(canary_from(5));
}

TEST_F(LibcStrlcpy, SizeOneWritesOnlyTheTerminator) {
    EXPECT_EQ(cpy(dst_, "xyz", 1), 3u);
    EXPECT_EQ(dst_[0], '\0');
    EXPECT_TRUE(canary_from(1));
}

TEST_F(LibcStrlcpy, SizeZeroWritesNothingButStillMeasuresTheSource) {
    EXPECT_EQ(cpy(dst_, "measured", 0), 8u);
    EXPECT_TRUE(canary_from(0)) << "strlcpy with dstsize 0 must not touch dst";
}

TEST_F(LibcStrlcpy, EmptySourceYieldsEmptyString) {
    EXPECT_EQ(cpy(dst_, "", 8), 0u);
    EXPECT_EQ(dst_[0], '\0');
    EXPECT_TRUE(canary_from(8));
}

// Every (source length, dstsize) pair in a small grid, against the manual page's rule written out
// directly: copy min(len, size - 1) bytes, terminate if size != 0, never write past size.
TEST_F(LibcStrlcpy, MatchesTheContractOverALengthBySizeGrid) {
    const std::string alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
    for (size_t len = 0; len <= 20; ++len) {
        const std::string src = alphabet.substr(0, len);
        for (size_t size = 0; size <= 24; ++size) {
            memset(dst_, kCanary, sizeof dst_);
            const size_t r = cpy(dst_, src.c_str(), size);
            SCOPED_TRACE("len=" + std::to_string(len) + " size=" + std::to_string(size));
            EXPECT_EQ(r, len);
            if (size == 0) {
                EXPECT_TRUE(canary_from(0));
                continue;
            }
            const size_t copied = len < size - 1 ? len : size - 1;
            EXPECT_EQ(memcmp(dst_, src.data(), copied), 0);
            EXPECT_EQ(dst_[copied], '\0');
            EXPECT_TRUE(canary_from(size));
        }
    }
}

TEST_F(LibcStrlcpy, CatAppendsWhenItFits) {
    cpy(dst_, "ab", kBuf);
    EXPECT_EQ(cat(dst_, "cd", 8), 4u);
    EXPECT_STREQ(dst_, "abcd");
}

TEST_F(LibcStrlcpy, CatTruncatesAndReportsTheLengthItTriedToCreate) {
    memset(dst_, kCanary, sizeof dst_);
    memcpy(dst_, "ab", 3);
    const size_t r = cat(dst_, "cdefgh", 5);
    EXPECT_EQ(r, 8u);
    EXPECT_STREQ(dst_, "abcd");
    EXPECT_TRUE(canary_from(5)) << "strlcat wrote at or past dst[dstsize]";
}

TEST_F(LibcStrlcpy, CatIntoAFullBufferAppendsNothing) {
    memset(dst_, kCanary, sizeof dst_);
    memcpy(dst_, "abcd", 5);   // strlen(dst) == dstsize - 1
    EXPECT_EQ(cat(dst_, "xyz", 5), 7u);
    EXPECT_STREQ(dst_, "abcd");
    EXPECT_TRUE(canary_from(5));
}

TEST_F(LibcStrlcpy, CatWithNoTerminatorInsideTheSizeTreatsLengthAsSize) {
    // dst holds six non-NUL bytes and dstsize is 4: dst's length is taken to be 4, nothing is
    // appended, the buffer stays unterminated, and the return is 4 + strlen(src).
    memset(dst_, kCanary, sizeof dst_);
    memcpy(dst_, "abcdef", 6);
    EXPECT_EQ(cat(dst_, "xyz", 4), 7u);
    EXPECT_EQ(memcmp(dst_, "abcdef", 6), 0) << "strlcat modified an unterminated destination";
    EXPECT_TRUE(canary_from(6));
}

TEST_F(LibcStrlcpy, CatSizeZeroWritesNothing) {
    memset(dst_, kCanary, sizeof dst_);
    EXPECT_EQ(cat(dst_, "abc", 0), 3u);
    EXPECT_TRUE(canary_from(0));
}

TEST_F(LibcStrlcpy, CatEmptySourceLeavesDestinationAndReturnsItsLength) {
    cpy(dst_, "hello", kBuf);
    EXPECT_EQ(cat(dst_, "", 16), 5u);
    EXPECT_STREQ(dst_, "hello");
}

// Every (initial dst length, source length, dstsize) triple in a small grid, against the manual
// page's rule written out directly.
TEST_F(LibcStrlcpy, CatMatchesTheContractOverAGrid) {
    const std::string head = "ABCDEFGHIJKL";
    const std::string tail = "0123456789";
    for (size_t dl = 0; dl <= head.size(); ++dl) {
        for (size_t sl = 0; sl <= tail.size(); ++sl) {
            for (size_t size = 0; size <= 24; ++size) {
                memset(dst_, kCanary, sizeof dst_);
                memcpy(dst_, head.data(), dl);
                dst_[dl] = '\0';
                const std::string src = tail.substr(0, sl);
                const size_t r = cat(dst_, src.c_str(), size);
                SCOPED_TRACE("dl=" + std::to_string(dl) + " sl=" + std::to_string(sl) +
                             " size=" + std::to_string(size));
                if (dl >= size) {
                    // No NUL within the first dstsize bytes: length is dstsize, nothing written.
                    EXPECT_EQ(r, size + sl);
                    EXPECT_EQ(memcmp(dst_, head.data(), dl), 0);
                    EXPECT_EQ(dst_[dl], '\0');
                    EXPECT_TRUE(canary_from(dl + 1));
                    continue;
                }
                EXPECT_EQ(r, dl + sl);
                const size_t room = size - dl - 1;
                const size_t appended = sl < room ? sl : room;
                EXPECT_EQ(memcmp(dst_, head.data(), dl), 0);
                EXPECT_EQ(memcmp(dst_ + dl, src.data(), appended), 0);
                EXPECT_EQ(dst_[dl + appended], '\0');
                EXPECT_TRUE(canary_from(size > dl + 1 ? size : dl + 1));
            }
        }
    }
}

}   // namespace

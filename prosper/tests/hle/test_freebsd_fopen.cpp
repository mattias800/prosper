// test_freebsd_fopen — FreeBSD libc contract for fopen modes, ported to
// gtest.
//
// Provenance: freebsd-src lib/libc/tests/stdio/fopen_test.c (ATF).
// Original author: Jilles Tjoelker.
//
// Why this pins prosper: f_fopen (hle_file.cpp) translates the guest path
// (/app0, save mounts) and forwards the MODE STRING VERBATIM to host fopen,
// so the guest-visible mode contract is the host's — including its gaps.
// The ATF original opens /dev/null and inspects fd flags via fcntl; this
// port uses scratch files (repo rule: never fixed paths, never /tmp) and
// splits the fcntl/CLOEXEC half behind a POSIX gate — Windows has neither
// fcntl nor the 'e' mode letter, and asserting them there would pin a CRT
// gap as a prosper defect. The 'e' arms run on Linux/macOS CI.
#include <gtest/gtest.h>

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "../fixtures/test_scratch.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace prosper;
using prosper_test::test_scratch_file;

namespace {

void write_text(const std::string& path, const char* content) {
    FILE* f = fopen(path.c_str(), "w");
    ASSERT_NE(f, nullptr) << "fixture setup write of " << path;
    ASSERT_NE(fputs(content, f), EOF);
    ASSERT_EQ(fclose(f), 0);
}

std::string read_all(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    EXPECT_NE(f, nullptr) << "fixture read of " << path;
    if (!f) return {};
    std::string out;
    char buf[256];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

}  // namespace

TEST(FreebsdFopen, ModesOpenAndBehave) {
    // Every basic mode opens and the content matches the mode. Two things
    // are deliberately NOT asserted: (1) writing to a read stream or
    // reading a write stream — glibc refuses, UCRT allows, and C leaves
    // cross-direction access without an intervening flush/seek undefined
    // (measured: UCRT materializes a stray NUL into the file); (2) any
    // read/write adjacency without fseek — same UB class, so every switch
    // below goes through one.
    const std::string p = test_scratch_file("fopen-modes.txt");

    write_text(p, "hello");
    FILE* r = fopen(p.c_str(), "r");
    ASSERT_NE(r, nullptr) << "fopen r of existing file";
    EXPECT_EQ(fgetc(r), 'h');
    EXPECT_EQ(fclose(r), 0);

    FILE* w = fopen(p.c_str(), "w");
    ASSERT_NE(w, nullptr) << "fopen w";
    EXPECT_NE(fputs("abc", w), EOF);
    EXPECT_EQ(fclose(w), 0);
    EXPECT_EQ(read_all(p), "abc") << "w truncates then writes";

    write_text(p, "AB");
    FILE* a = fopen(p.c_str(), "a");
    ASSERT_NE(a, nullptr) << "fopen a";
    EXPECT_NE(fputs("CD", a), EOF);
    EXPECT_EQ(fclose(a), 0);
    EXPECT_EQ(read_all(p), "ABCD") << "a appends";

    FILE* rp = fopen(p.c_str(), "r+");
    ASSERT_NE(rp, nullptr) << "fopen r+";
    EXPECT_EQ(fgetc(rp), 'A');
    EXPECT_EQ(fseek(rp, 1, SEEK_SET), 0);
    EXPECT_EQ(fputc('x', rp), 'x');
    EXPECT_EQ(fclose(rp), 0);
    EXPECT_EQ(read_all(p), "AxCD") << "r+ overwrites in place";

    write_text(p, "stale-content-here");
    FILE* wp = fopen(p.c_str(), "w+");
    ASSERT_NE(wp, nullptr) << "fopen w+";
    EXPECT_NE(fputs("new", wp), EOF);
    EXPECT_EQ(fseek(wp, 0, SEEK_SET), 0);
    char back[8] = {};
    EXPECT_NE(fgets(back, sizeof back, wp), nullptr);
    EXPECT_STREQ(back, "new");
    EXPECT_EQ(fclose(wp), 0);
    EXPECT_EQ(read_all(p), "new") << "w+ truncates";

    write_text(p, "AB");
    FILE* ap = fopen(p.c_str(), "a+");
    ASSERT_NE(ap, nullptr) << "fopen a+";
    EXPECT_EQ(fgetc(ap), 'A') << "a+ reads from the start";
    EXPECT_EQ(fseek(ap, 0, SEEK_END), 0) << "seek between read and write";
    EXPECT_NE(fputs("CD", ap), EOF) << "write after seek appends";
    EXPECT_EQ(fclose(ap), 0);
    EXPECT_EQ(read_all(p), "ABCD") << "a+ write lands at end";
}

TEST(FreebsdFopen, MissingFileReadFails) {
    const std::string p = test_scratch_file("fopen-absent.txt");
    std::remove(p.c_str());
    errno = 0;
    EXPECT_EQ(fopen(p.c_str(), "r"), nullptr) << "read of absent file fails";
    EXPECT_EQ(errno, ENOENT) << "absent file reports ENOENT, not a silent NULL";
}

TEST(FreebsdFopen, BinaryModePreservesBytes) {
    // Text mode on Windows translates \n <-> \r\n and stops at 0x1A; game
    // archives and save blobs need byte-exact round trips, hence "b".
    const std::string p = test_scratch_file("fopen-bin.dat");
    const unsigned char payload[] = {0x0A, 0x0D, 0x1A, 0x00, 0xFF, 0x0A};
    FILE* w = fopen(p.c_str(), "wb");
    ASSERT_NE(w, nullptr) << "fopen wb";
    EXPECT_EQ(fwrite(payload, 1, sizeof payload, w), sizeof payload);
    EXPECT_EQ(fclose(w), 0);
    FILE* r = fopen(p.c_str(), "rb");
    ASSERT_NE(r, nullptr) << "fopen rb";
    unsigned char back[16] = {};
    EXPECT_EQ(fread(back, 1, sizeof payload, r), sizeof payload);
    EXPECT_EQ(fclose(r), 0);
    EXPECT_EQ(memcmp(back, payload, sizeof payload), 0)
        << "binary round trip is byte-exact (no CRLF/0x1A mangling)";
}

TEST(FreebsdFopen, FdFlagsAndCloexec) {
#ifdef _WIN32
    // Port of the ATF runtest()'s fcntl half: Windows has no fcntl(2) and
    // its fopen rejects the 'e' letter, so neither half can even compile
    // there. Runs on Linux/macOS CI; prosper's forwarded mode string means
    // the guest sees exactly this host behaviour.
    GTEST_SKIP() << "no fcntl(2) and no 'e' fopen mode on Windows (measured); "
                    "POSIX hosts assert below";
#else
    const std::string p = test_scratch_file("fopen-flags.txt");
    write_text(p, "x");

    struct Case {
        const char* mode;
        int access;  // O_RDONLY / O_WRONLY / O_RDWR, plus O_APPEND
        bool cloexec;
    };
    const Case cases[] = {
        {"r", O_RDONLY, false},                     //
        {"w", O_WRONLY, false},                     //
        {"a", O_WRONLY | O_APPEND, false},          //
        {"r+", O_RDWR, false},                      //
        {"w+", O_RDWR, false},                      //
        {"a+", O_RDWR | O_APPEND, false},           //
        {"re", O_RDONLY, true},                     //
        {"we", O_WRONLY, true},                     //
        {"ae", O_WRONLY | O_APPEND, true},          //
        {"r+e", O_RDWR, true},                      //
        {"w+e", O_RDWR, true},                      //
        {"a+e", O_RDWR | O_APPEND, true},           //
        {"re+", O_RDWR, true},                      //
        {"we+", O_RDWR, true},                      //
        {"ae+", O_RDWR | O_APPEND, true},           //
    };
    for (const auto& c : cases) {
        FILE* f = fopen(p.c_str(), c.mode);
        ASSERT_NE(f, nullptr) << "fopen " << c.mode;
        const int fd = fileno(f);
        ASSERT_GE(fd, 0);
        const int fdflags = fcntl(fd, F_GETFD);
        EXPECT_EQ(fdflags & FD_CLOEXEC, c.cloexec ? FD_CLOEXEC : 0)
            << "mode " << c.mode << " CLOEXEC bit";
        const int flags = fcntl(fd, F_GETFL);
        EXPECT_EQ(flags & (O_ACCMODE | O_APPEND), c.access) << "mode " << c.mode << " access bits";
        EXPECT_EQ(fclose(f), 0);
    }
#endif
}

TEST(FreebsdFopen, HleFileHandlersRegistered) {
    // Unregistered, these NIDs fall to the dispatcher's return-0 default: a
    // NULL FILE* the guest reads as "missing file" on every open.
    register_builtin_hle();
    const char* names[] = {"fopen", "fclose", "fread", "fwrite", "fseek",  "ftell",
                           "fgets", "fflush", "feof",  "ferror", "rewind", "setvbuf"};
    for (const char* n : names)
        EXPECT_NE(Hle::lookup(nid_hash(n)), nullptr) << n << " is registered";
}

TEST(FreebsdFopen, PositiveControl) {
    // A write/read round trip the harness must see: if this ever reports
    // equal for mismatched content, the mode arms above prove nothing.
    const std::string p = test_scratch_file("fopen-positive.txt");
    write_text(p, "ping");
    EXPECT_EQ(read_all(p), "ping");
    EXPECT_NE(read_all(p), "pong") << "harness distinguishes content";
}

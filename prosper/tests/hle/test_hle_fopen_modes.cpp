// test_hle_fopen_modes — the guest's fopen, through prosper's REGISTERED stdio handlers.
//
// Contract: C11 (N1570) §7.21.5.3 (fopen modes: truncate/create, append forcing writes to the end
// regardless of fseek, update streams needing a positioning call between output and input,
// exclusive "wx") and FreeBSD fopen(3) — the guest's libc — for the mode letters it adds ('e' close-on-
// exec, 'b' ignored) and EINVAL for an invalid mode. Derived from those texts; no other suite's code
// or vectors.
//
// What is prosper's here: `fopen` translates the guest path (/app0 → the title root) and the guest's
// mode string, and every later operation is a registered handler too (`fread`, `fwrite`, `fseek`,
// `ftell`, `fgetc`, `fflush`, `fclose`, and `__error` for the guest's errno). So each case opens a
// GUEST path through the handler and drives the stream through the handlers; the host's own stdio
// appears only to build a fixture or to observe the bytes that reached the disk.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#if defined(_WIN32)
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#endif

using namespace prosper;

namespace {

uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

uint64_t call(const char* name, uint64_t a = 0, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0) {
    HleFn fn = Hle::lookup(nid_hash(name));
    if (!fn) {
        ADD_FAILURE() << name << " is not registered";
        return 0;
    }
    return fn(a, b, c, d, 0, 0);
}

class FopenModes : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        register_builtin_hle();
        root_ = (prosper_test::test_scratch_dir() / "fopen-app0").string();
        std::filesystem::create_directories(root_);
        set_app0_root(root_);
    }

    static std::string host(const std::string& name) { return root_ + "/" + name; }
    static std::string guest(const std::string& name) { return "/app0/" + name; }

    static void put(const std::string& name, const std::string& bytes) {
        std::ofstream out(host(name), std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), (std::streamsize)bytes.size());
    }
    static std::string get(const std::string& name) {
        std::ifstream in(host(name), std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    static bool exists(const std::string& name) { return std::filesystem::exists(host(name)); }
    static void remove(const std::string& name) { std::filesystem::remove(host(name)); }

    static FILE* open(const std::string& name, const char* mode) {
        const std::string path = guest(name);
        return (FILE*)(uintptr_t)call("fopen", ptr(path.c_str()), ptr(mode));
    }
    static int guest_errno() {
        const int* slot = (const int*)(uintptr_t)call("__error");
        return slot ? *slot : -1;
    }
    static void set_guest_errno(int value) {
        int* slot = (int*)(uintptr_t)call("__error");
        if (slot) *slot = value;
    }
    static size_t write(FILE* f, const std::string& bytes) {
        return (size_t)call("fwrite", ptr(bytes.data()), 1, bytes.size(), ptr(f));
    }
    static std::string read(FILE* f, size_t n) {
        std::string out(n, '\0');
        out.resize((size_t)call("fread", ptr(out.data()), 1, n, ptr(f)));
        return out;
    }
    static int seek(FILE* f, int64_t offset, int whence) {
        return (int)(int64_t)call("fseek", ptr(f), (uint64_t)offset, (uint64_t)whence);
    }
    static int64_t tell(FILE* f) { return (int64_t)call("ftell", ptr(f)); }
    static int getc(FILE* f) { return (int)(int64_t)call("fgetc", ptr(f)); }
    static int flush(FILE* f) { return (int)(int64_t)call("fflush", ptr(f)); }
    static int close(FILE* f) { return (int)(int64_t)call("fclose", ptr(f)); }

    static inline std::string root_;
};

} // namespace

// "r" opens an existing file at its start and fails on a missing one; "r+" never creates either.
// The path is a GUEST path: it resolves only because the handler maps /app0 to the title root.
TEST_F(FopenModes, ReadOpensExistingAndFailsOnMissing) {
    put("read.bin", "hello");
    FILE* f = open("read.bin", "r");
    ASSERT_NE(f, nullptr) << "a guest /app0 path opens through the handler";
    EXPECT_EQ(read(f, 16), "hello");
    EXPECT_EQ(getc(f), EOF);
    EXPECT_NE(call("feof", ptr(f)), 0u);
    EXPECT_EQ(close(f), 0);

    remove("absent.bin");
    for (const char* mode : {"r", "rb", "r+", "r+b"}) {
        set_guest_errno(0);
        EXPECT_EQ(open("absent.bin", mode), nullptr) << mode;
        EXPECT_EQ(guest_errno(), ENOENT) << mode;
        EXPECT_FALSE(exists("absent.bin")) << mode << " must not create";
    }
}

// "w" truncates at OPEN time, before anything is written, and creates a missing file.
TEST_F(FopenModes, WriteTruncatesOrCreates) {
    put("trunc.bin", "0123456789");
    FILE* f = open("trunc.bin", "w");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(std::filesystem::file_size(host("trunc.bin")), 0u) << "truncated by the open itself";
    EXPECT_EQ(write(f, "ab"), 2u);
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("trunc.bin"), "ab");

    remove("created.bin");
    f = open("created.bin", "wb");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(write(f, "new"), 3u);
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("created.bin"), "new");
}

// Update streams: "r+" keeps the contents and overwrites in place; "w+" truncates and can read
// back what it wrote. Input after output needs an intervening positioning call (§7.21.5.3p7).
TEST_F(FopenModes, UpdateModesReadAndWrite) {
    put("rplus.bin", "0123456789");
    FILE* f = open("rplus.bin", "r+");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(write(f, "AB"), 2u);
    EXPECT_EQ(seek(f, 0, SEEK_SET), 0);
    EXPECT_EQ(read(f, 4), "AB23");
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("rplus.bin"), "AB23456789") << "r+ neither truncates nor appends";

    put("wplus.bin", "old contents");
    f = open("wplus.bin", "w+");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(write(f, "xyz"), 3u);
    EXPECT_EQ(seek(f, 0, SEEK_SET), 0);
    EXPECT_EQ(read(f, 16), "xyz");
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("wplus.bin"), "xyz");
}

// Append forces every write to the then-current end of file regardless of fseek (§7.21.5.3p6). The
// stream is positioned AWAY from the end before each write, so an ordinary positioned write would
// land elsewhere. Where "a+" first reads from is not specified, so it seeks before reading.
TEST_F(FopenModes, AppendForcesWritesToTheEnd) {
    put("append.bin", "base");
    FILE* f = open("append.bin", "a");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(seek(f, 0, SEEK_SET), 0);
    EXPECT_EQ(write(f, "X"), 1u);
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("append.bin"), "baseX");

    put("aplus.bin", "base");
    f = open("aplus.bin", "a+");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(seek(f, 0, SEEK_SET), 0);
    EXPECT_EQ(getc(f), 'b') << "a+ reads from wherever it is positioned";
    EXPECT_EQ(seek(f, 1, SEEK_SET), 0);
    EXPECT_EQ(write(f, "Y"), 1u);
    EXPECT_EQ(flush(f), 0);
    EXPECT_EQ(tell(f), 5) << "after an append write the stream is at the new end";
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("aplus.bin"), "baseY");

    remove("acreate.bin");
    f = open("acreate.bin", "a");
    ASSERT_NE(f, nullptr) << "append creates a missing file";
    EXPECT_EQ(write(f, "z"), 1u);
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("acreate.bin"), "z");
}

// "wx" fails if the file exists, leaving it untouched, and creates it otherwise (§7.21.5.3p5).
TEST_F(FopenModes, ExclusiveCreate) {
    put("excl.bin", "keep");
    for (const char* mode : {"wx", "wbx", "w+x", "w+bx"}) {
        set_guest_errno(0);
        EXPECT_EQ(open("excl.bin", mode), nullptr) << mode;
        EXPECT_EQ(guest_errno(), EEXIST) << mode;
        EXPECT_EQ(get("excl.bin"), "keep") << mode << " must not truncate an existing file";
    }
    remove("excl-new.bin");
    FILE* f = open("excl-new.bin", "wx");
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(write(f, "fresh"), 5u);
    EXPECT_EQ(close(f), 0);
    EXPECT_EQ(get("excl-new.bin"), "fresh");
}

// A mode that does not begin with 'r', 'w' or 'a' is EINVAL (fopen(3)) and opens nothing.
TEST_F(FopenModes, InvalidLeadingCharacterIsEinval) {
    remove("invalid.bin");
    for (const char* mode : {"", "z", "+r", "br", "x", "W"}) {
        set_guest_errno(0);
        EXPECT_EQ(open("invalid.bin", mode), nullptr) << "'" << mode << "'";
        EXPECT_EQ(guest_errno(), EINVAL) << "'" << mode << "'";
        EXPECT_FALSE(exists("invalid.bin")) << "'" << mode << "' must not create";
    }
}

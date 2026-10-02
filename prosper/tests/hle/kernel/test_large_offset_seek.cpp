// test_large_offset_seek — lseek/fseek/ftell must address offsets at and above 2 GiB.
//
// THE DEFECT. MinGW's `long` and `off_t` are 32 bits. f_lseek cast the guest offset to `off_t` and
// called ::lseek, and f_fseek/f_ftell used the `long` forms, so any seek to or past 2 GiB was
// truncated or refused. Assassin's Creed Black Flag Resynced's DataPS5_boot.forge is ~30 GiB and its
// file allocation table sits far beyond that, so the title's `fseek` failed and it aborted itself with
// "fseek error while reading the fat" (a deliberate write to address 2, host exit 0xC0000005).
//
// WHAT EACH ARM KILLS (Windows):
//   M1  f_lseek casts to off_t / calls ::lseek again          -> §1 absolute and SEEK_CUR arms
//   M2  f_fseek goes back to fseek((long)offset)              -> §2 fseek arm
//   M3  f_ftell goes back to ftell()                          -> §2 ftell arm
//   M4  the failure logger does not restore the seek errno  -> failing-stderr arm
// The configured per-process scratch fixture is exclusively created and automatically cleaned.
// On Windows explicitly mark it sparse before
// SetEndOfFile; extending an ordinary NTFS file is not a sparse-allocation guarantee.
// Unsupported sparse/sizing operations are reported with GTEST_SKIP, not success.
#include <gtest/gtest.h>
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "../../fixtures/test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace prosper;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

struct TempStream {
    const std::string path = prosper_test::test_scratch_file("large-offset.bin");
    FILE* file = nullptr;
    TempStream() {
#ifdef _WIN32
        const int fd = _open(path.c_str(), _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY,
                             _S_IREAD | _S_IWRITE);
        if (fd >= 0) {
            file = _fdopen(fd, "w+b");
            if (!file) _close(fd);
        }
#else
        const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) {
            file = fdopen(fd, "w+b");
            if (!file) close(fd);
        }
#endif
    }
    ~TempStream() { if (file) std::fclose(file); }
};

TEST(LargeOffsetSeek, Contract) {
    printf("== test_large_offset_seek ==\n");
#ifdef _WIN32
    _putenv_s("PROSPER_FILELOG", "1");
#else
    setenv("PROSPER_FILELOG", "1", 1);
#endif
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    register_builtin_hle();

    HleFn lseek_fn = Hle::lookup(nid_hash("lseek"));
    HleFn fseek_fn = Hle::lookup(nid_hash("fseek"));
    HleFn ftell_fn = Hle::lookup(nid_hash("ftell"));
    CHECK(lseek_fn && fseek_fn && ftell_fn, "lseek, fseek and ftell are registered");
    if (!lseek_fn || !fseek_fn || !ftell_fn) return;

    constexpr int64_t kSize = (int64_t)6 * 1024 * 1024 * 1024;            // logical 6 GiB
    constexpr int64_t kPast2GiB = (int64_t)0x80000000LL + 0x1234;         // first bytes past 2 GiB
    constexpr int64_t kPast4GiB = (int64_t)0x100000000LL + 0x1234;

    TempStream owned;
    FILE* file = owned.file;
    CHECK(file != nullptr, "exclusive auto-cleaned temporary stream opens");
    if (!file) return;
#ifdef _WIN32
    const int fd = _fileno(file);
    const HANDLE handle = (HANDLE)_get_osfhandle(fd);
    DWORD returned = 0;
    if (!DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr)) {
        GTEST_SKIP() << "temporary filesystem cannot mark sparse storage (Windows error "
                     << (unsigned long)GetLastError() << ")";
    }
    LARGE_INTEGER size;
    size.QuadPart = kSize;
    const bool sized = SetFilePointerEx(handle, size, nullptr, FILE_BEGIN) && SetEndOfFile(handle);
#else
    const int fd = fileno(file);
    const bool sized = ftruncate(fd, (off_t)kSize) == 0;
#endif
    if (!sized) {
        GTEST_SKIP() << "temporary filesystem cannot size a logical 6 GiB extent";
    }

    // --- §1 lseek through the registered handler -------------------------------------------------
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)kPast2GiB, 0 /*SEEK_SET*/, 0, 0, 0) == kPast2GiB,
          "lseek(SEEK_SET) to 2 GiB + 0x1234 returns that offset (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, 0, 1 /*SEEK_CUR*/, 0, 0, 0) == kPast2GiB,
          "SEEK_CUR 0 reads the same position back (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)kPast4GiB, 0, 0, 0, 0) == kPast4GiB,
          "lseek(SEEK_SET) past 4 GiB returns that offset (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)(int64_t)-0x100, 2 /*SEEK_END*/, 0, 0, 0) ==
              kSize - 0x100,
          "lseek(SEEK_END, -0x100) lands 0x100 before the end (M1)");

    // --- §2 fseek / ftell through the registered handlers ----------------------------------------
    CHECK((int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)kPast2GiB, 0, 0, 0, 0) == 0,
          "fseek to 2 GiB + 0x1234 succeeds (M2)");
    CHECK((int64_t)ftell_fn((uint64_t)(uintptr_t)file, 0, 0, 0, 0, 0) == kPast2GiB,
          "ftell reports the position past 2 GiB (M3)");
    CHECK((int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)kPast4GiB, 0, 0, 0, 0) == 0 &&
              (int64_t)ftell_fn((uint64_t)(uintptr_t)file, 0, 0, 0, 0, 0) == kPast4GiB,
          "fseek/ftell round-trip past 4 GiB (M2, M3)");
    CHECK(std::fputc(0x5a, file) == 0x5a && std::fflush(file) == 0,
          "one sentinel byte writes at the above-4-GiB position");
    CHECK((int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)kPast4GiB, 0, 0, 0, 0) == 0 &&
          std::fgetc(file) == 0x5a, "above-4-GiB sentinel reads back through guest seek");

    // A failing diagnostic sink must not replace the guest's seek EINVAL with the sink's errno.
#ifdef _WIN32
    const int saved_stderr = _dup(_fileno(stderr));
    const int readonly_sink = _open(owned.path.c_str(), _O_RDONLY | _O_BINARY);
    const bool redirected = saved_stderr >= 0 && readonly_sink >= 0 &&
                            _dup2(readonly_sink, _fileno(stderr)) == 0;
#else
    const int saved_stderr = dup(fileno(stderr));
    const int readonly_sink = open(owned.path.c_str(), O_RDONLY);
    const bool redirected = saved_stderr >= 0 && readonly_sink >= 0 &&
                            dup2(readonly_sink, fileno(stderr)) >= 0;
#endif
    CHECK(redirected, "stderr can be redirected to a valid read-only diagnostic sink");
    if (redirected) {
        errno = 0;
        // The native write is the independent sink oracle. Some MinGW fprintf implementations
        // return their formatted character count even though this write fails and changes errno.
#ifdef _WIN32
        const int probe = _write(_fileno(stderr), "x", 1);
#else
        const int probe = (int)write(fileno(stderr), "x", 1);
#endif
        const int sink_error = errno;
        std::clearerr(stderr);
        errno = 0;
        const int64_t rc = (int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)(int64_t)-1,
                                            SEEK_SET, 0, 0, 0);
        const int seek_error = errno;
        printf("  sink probe: rc=%d errno=%d; guest seek: rc=%lld errno=%d\n",
               probe, sink_error, (long long)rc, seek_error);
        CHECK(probe < 0 && sink_error != 0 && sink_error != EINVAL,
              "diagnostic sink independently refuses with a different errno");
        CHECK(rc != 0 && seek_error == EINVAL, "failed seek preserves EINVAL despite logging (M4)");
    }
#ifdef _WIN32
    if (saved_stderr >= 0) { _dup2(saved_stderr, _fileno(stderr)); _close(saved_stderr); }
    if (readonly_sink >= 0) _close(readonly_sink);
#else
    if (saved_stderr >= 0) { dup2(saved_stderr, fileno(stderr)); close(saved_stderr); }
    if (readonly_sink >= 0) close(readonly_sink);
#endif
    std::clearerr(stderr);

}

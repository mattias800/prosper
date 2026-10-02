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
// The file is made by extending a temp file (no 3 GiB of data is written; NTFS allocates lazily).
// Where the filesystem refuses to extend the file the test says so and skips, rather than passing.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_large_offset_seek ==\n");
    register_builtin_hle();

    HleFn lseek_fn = Hle::lookup(nid_hash("lseek"));
    HleFn fseek_fn = Hle::lookup(nid_hash("fseek"));
    HleFn ftell_fn = Hle::lookup(nid_hash("ftell"));
    CHECK(lseek_fn && fseek_fn && ftell_fn, "lseek, fseek and ftell are registered");
    if (!lseek_fn || !fseek_fn || !ftell_fn) return 1;

    constexpr int64_t kSize = (int64_t)3 * 1024 * 1024 * 1024;            // 3 GiB
    constexpr int64_t kPast2GiB = (int64_t)0x80000000LL + 0x1234;         // first bytes past 2 GiB
    constexpr int64_t kPast4GiB = kSize - 0x100;                          // inside the file, > 2 GiB

    const char* tmp = getenv("TEMP");
    std::string path = std::string(tmp ? tmp : ".") + "/prosper_large_seek_test.bin";

#ifdef _WIN32
    int fd = _open(path.c_str(), _O_RDWR | _O_CREAT | _O_TRUNC | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
#endif
    CHECK(fd >= 0, "temp file opens");
    if (fd < 0) return 1;
#ifdef _WIN32
    const bool sized = _chsize_s(fd, kSize) == 0;
#else
    const bool sized = ftruncate(fd, (off_t)kSize) == 0;
#endif
    if (!sized) {
        printf("  [skip] cannot extend a temp file to 3 GiB on this filesystem; nothing asserted\n");
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        std::remove(path.c_str());
        return 0;
    }

    // --- §1 lseek through the registered handler -------------------------------------------------
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)kPast2GiB, 0 /*SEEK_SET*/, 0, 0, 0) == kPast2GiB,
          "lseek(SEEK_SET) to 2 GiB + 0x1234 returns that offset (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, 0, 1 /*SEEK_CUR*/, 0, 0, 0) == kPast2GiB,
          "SEEK_CUR 0 reads the same position back (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)kPast4GiB, 0, 0, 0, 0) == kPast4GiB,
          "lseek(SEEK_SET) near 3 GiB returns that offset (M1)");
    CHECK((int64_t)lseek_fn((uint64_t)fd, (uint64_t)(int64_t)-0x100, 2 /*SEEK_END*/, 0, 0, 0) ==
              kSize - 0x100,
          "lseek(SEEK_END, -0x100) lands 0x100 before the end (M1)");
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    // --- §2 fseek / ftell through the registered handlers ----------------------------------------
    FILE* file = fopen(path.c_str(), "rb");
    CHECK(file != nullptr, "temp file opens through stdio");
    if (file) {
        CHECK((int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)kPast2GiB, 0, 0, 0, 0) == 0,
              "fseek to 2 GiB + 0x1234 succeeds (M2)");
        CHECK((int64_t)ftell_fn((uint64_t)(uintptr_t)file, 0, 0, 0, 0, 0) == kPast2GiB,
              "ftell reports the position past 2 GiB (M3)");
        CHECK((int64_t)fseek_fn((uint64_t)(uintptr_t)file, (uint64_t)kPast4GiB, 0, 0, 0, 0) == 0 &&
                  (int64_t)ftell_fn((uint64_t)(uintptr_t)file, 0, 0, 0, 0, 0) == kPast4GiB,
              "fseek/ftell round-trip near 3 GiB (M2, M3)");
        fclose(file);
    }
    std::remove(path.c_str());

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}

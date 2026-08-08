// Guest fd I/O must preserve every byte of binary game content on every host.
// In particular, Windows CRT text mode treats 0x1a as EOF and translates CRLF.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // setenv/_putenv_s (PROSPER_DENY_SUBSTR arming)
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

// FreeBSD/Orbis fcntl ABI values. Host constants are deliberately not used here: Linux and
// Windows assign different bits to several status flags even when command numbers coincide.
static constexpr uint64_t kGuestFDupFd = 0;
static constexpr uint64_t kGuestFGetFd = 1;
static constexpr uint64_t kGuestFSetFd = 2;
// FreeBSD's ENOTSUP, spelled out because the host's <errno.h> is the WRONG source here: the
// guest reads prosper's errno slot through __error() and compares against FreeBSD's numbering, so
// an arm written as `errno == ENOTSUP` asserts the host value and passes on exactly the code #2296
// is about. Host values measured: Linux 95, MinGW-w64 UCRT 129, Darwin 45 (see the coincidence
// note printed below -- on Darwin these arms cannot distinguish, and say so rather than pretending).
static constexpr int kFreeBsdENotSupp = 45;
static constexpr uint64_t kGuestFGetFl = 3;
static constexpr uint64_t kGuestFSetFl = 4;
static constexpr uint64_t kGuestFdCloExec = 1;
static constexpr uint64_t kGuestONonblock = 0x0004;
static constexpr uint64_t kGuestOAppend = 0x0008;
static constexpr uint64_t kGuestOSync = 0x0080;

struct GuestTimeval {
    int64_t sec;
    int64_t usec;
};
static_assert(sizeof(GuestTimeval) == 0x10, "SceKernelTimeval ABI");

int main() {
    std::printf("== test_file_binary ==\n");
    // Say up front whether the ENOTSUP arms below can distinguish a FreeBSD publish from a host
    // one. They cannot on a host whose ENOTSUP is already 45 (Darwin) -- a green run there is
    // silent about #2296 rather than evidence for it, and a reader should not have to guess which.
    std::printf("  [note] host ENOTSUP=%d, FreeBSD=%d -- ENOTSUP arms %s here\n",
                ENOTSUP, kFreeBsdENotSupp,
                ENOTSUP == kFreeBsdENotSupp ? "CANNOT distinguish" : "discriminate");
    // PROSPER_DENY_SUBSTR (#1237): armed for the whole process BEFORE any guest file op (the
    // substring list is cached on first use). The unique marker cannot collide with any other
    // path this test opens; the case-variant open below proves matching is case-insensitive
    // (the guest namespace resolves case-insensitively since #1233, so a casing variant must
    // not defeat the deny knob).
#ifdef _WIN32
    _putenv_s("PROSPER_DENY_SUBSTR", ".TMPDENY");
#else
    setenv("PROSPER_DENY_SUBSTR", ".TMPDENY", 1);
#endif
    const char* path = "prosper-test-file-binary.tmp";
    std::array<uint8_t, 512> expected{};
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (uint8_t)(i * 37u + 11u);
    expected[7] = 0x1a;
    expected[8] = '\r';
    expected[9] = '\n';

    FILE* out = std::fopen(path, "wb");
    CHECK(out != nullptr, "create binary fixture");
    if (out) {
        CHECK(std::fwrite(expected.data(), 1, expected.size(), out) == expected.size(),
              "write binary fixture");
        std::fclose(out);
    }

    register_file_hle();
    HleFn posix_open_fn = Hle::lookup(nid_hash("open"));
    HleFn open_fn = Hle::lookup(nid_hash("sceKernelOpen"));
    HleFn posix_read_fn = Hle::lookup(nid_hash("read"));
    HleFn read_fn = Hle::lookup(nid_hash("sceKernelRead"));
    HleFn posix_write_fn = Hle::lookup(nid_hash("write"));
    HleFn write_fn = Hle::lookup(nid_hash("sceKernelWrite"));
    HleFn posix_pread_fn = Hle::lookup(nid_hash("pread"));
    HleFn pread_fn = Hle::lookup(nid_hash("sceKernelPread"));
    HleFn posix_pwrite_fn = Hle::lookup(nid_hash("pwrite"));
    HleFn pwrite_fn = Hle::lookup(nid_hash("sceKernelPwrite"));
    HleFn posix_readv_fn = Hle::lookup(nid_hash("readv"));
    HleFn readv_fn = Hle::lookup(nid_hash("sceKernelReadv"));
    HleFn posix_writev_fn = Hle::lookup(nid_hash("writev"));
    HleFn writev_fn = Hle::lookup(nid_hash("sceKernelWritev"));
    HleFn posix_preadv_fn = Hle::lookup(nid_hash("preadv"));
    HleFn preadv_fn = Hle::lookup(nid_hash("sceKernelPreadv"));
    HleFn posix_pwritev_fn = Hle::lookup(nid_hash("pwritev"));
    HleFn pwritev_fn = Hle::lookup(nid_hash("sceKernelPwritev"));
    HleFn posix_lseek_fn = Hle::lookup(nid_hash("lseek"));
    HleFn lseek_fn = Hle::lookup(nid_hash("sceKernelLseek"));
    HleFn posix_close_fn = Hle::lookup(nid_hash("close"));
    HleFn close_fn = Hle::lookup(nid_hash("sceKernelClose"));
    HleFn dup_fn = Hle::lookup(nid_hash("dup"));
    HleFn kernel_dup_fn = Hle::lookup(nid_hash("sceKernelDup"));
    HleFn dup2_fn = Hle::lookup(nid_hash("dup2"));
    HleFn kernel_dup2_fn = Hle::lookup(nid_hash("sceKernelDup2"));
    HleFn mkdir_fn = Hle::lookup(nid_hash("mkdir"));
    HleFn kernel_mkdir_fn = Hle::lookup(nid_hash("sceKernelMkdir"));
    HleFn rmdir_fn = Hle::lookup(nid_hash("rmdir"));
    HleFn kernel_rmdir_fn = Hle::lookup(nid_hash("sceKernelRmdir"));
    HleFn unlink_fn = Hle::lookup(nid_hash("unlink"));
    HleFn kernel_unlink_fn = Hle::lookup(nid_hash("sceKernelUnlink"));
    HleFn rename_fn = Hle::lookup(nid_hash("rename"));
    HleFn kernel_rename_fn = Hle::lookup(nid_hash("sceKernelRename"));
    HleFn kernel_truncate_fn = Hle::lookup(nid_hash("sceKernelTruncate"));
    HleFn kernel_ftruncate_fn = Hle::lookup(nid_hash("sceKernelFtruncate"));
    HleFn fsync_fn = Hle::lookup(nid_hash("fsync"));
    HleFn kernel_fsync_fn = Hle::lookup(nid_hash("sceKernelFsync"));
    HleFn fdatasync_fn = Hle::lookup(nid_hash("fdatasync"));
    HleFn kernel_fdatasync_fn = Hle::lookup(nid_hash("sceKernelFdatasync"));
    HleFn kernel_sync_fn = Hle::lookup(nid_hash("sceKernelSync"));
    HleFn kernel_reachability_fn = Hle::lookup(nid_hash("sceKernelCheckReachability"));
    HleFn kernel_utimes_fn = Hle::lookup(nid_hash("sceKernelUtimes"));
    HleFn chmod_fn = Hle::lookup(nid_hash("chmod"));
    HleFn kernel_chmod_fn = Hle::lookup(nid_hash("sceKernelChmod"));
    HleFn fchmod_fn = Hle::lookup(nid_hash("fchmod"));
    HleFn kernel_fchmod_fn = Hle::lookup(nid_hash("sceKernelFchmod"));
    HleFn getdents_fn = Hle::lookup(nid_hash("getdents"));
    HleFn kernel_getdents_fn = Hle::lookup(nid_hash("sceKernelGetdents"));
    HleFn getdirentries_fn = Hle::lookup(nid_hash("getdirentries"));
    HleFn kernel_getdirentries_fn = Hle::lookup(nid_hash("sceKernelGetdirentries"));
    HleFn stat_fn = Hle::lookup(nid_hash("stat"));
    HleFn kernel_stat_fn = Hle::lookup(nid_hash("sceKernelStat"));
    HleFn fstat_fn = Hle::lookup(nid_hash("fstat"));
    HleFn kernel_fstat_fn = Hle::lookup(nid_hash("sceKernelFstat"));
    HleFn lstat_fn = Hle::lookup(nid_hash("lstat"));
    HleFn kernel_lstat_fn = Hle::lookup(nid_hash("sceKernelLstat"));
    HleFn fcntl_fn = Hle::lookup(nid_hash("fcntl"));
    HleFn kernel_fcntl_fn = Hle::lookup(nid_hash("sceKernelFcntl"));
    CHECK(posix_open_fn && open_fn && posix_read_fn && read_fn && posix_write_fn && write_fn &&
              posix_pread_fn && pread_fn && posix_pwrite_fn && pwrite_fn && posix_lseek_fn &&
              posix_readv_fn && readv_fn && posix_writev_fn && writev_fn &&
              posix_preadv_fn && preadv_fn && posix_pwritev_fn && pwritev_fn &&
              lseek_fn && posix_close_fn && close_fn && dup_fn && kernel_dup_fn &&
              dup2_fn && kernel_dup2_fn && mkdir_fn && kernel_mkdir_fn && rmdir_fn &&
              kernel_rmdir_fn && unlink_fn && kernel_unlink_fn && rename_fn &&
              kernel_rename_fn && kernel_truncate_fn && kernel_ftruncate_fn && fsync_fn &&
              kernel_fsync_fn && fdatasync_fn && kernel_fdatasync_fn && kernel_sync_fn &&
              kernel_reachability_fn && kernel_utimes_fn && chmod_fn && kernel_chmod_fn &&
              fchmod_fn && kernel_fchmod_fn && getdents_fn &&
              kernel_getdents_fn && getdirentries_fn && kernel_getdirentries_fn && stat_fn &&
              kernel_stat_fn && fstat_fn && kernel_fstat_fn && lstat_fn && kernel_lstat_fn &&
              fcntl_fn && kernel_fcntl_fn,
          "file HLE functions registered");
    CHECK(kernel_sync_fn && kernel_sync_fn(0, 0, 0, 0, 0, 0) == 0,
          "sceKernelSync performs the host-wide/process-file flush and returns success");
    std::array<uint8_t, 64> dir_buffer{};

    // Creating an existing directory is a failure, not an idempotent success. Linux previously
    // suppressed EEXIST while Windows propagated it, so save-directory control flow differed by host.
    const char* dir_path = "prosper-test-mkdir-eexist.tmp";
    std::error_code remove_error;
    std::filesystem::remove_all(dir_path, remove_error);
    int64_t mkdir_first = mkdir_fn
        ? (int64_t)mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : -1;
    errno = 0;
    int64_t mkdir_duplicate = mkdir_fn
        ? (int64_t)mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : 0;
    int duplicate_errno = errno;
    uint64_t kernel_mkdir_duplicate = kernel_mkdir_fn
        ? kernel_mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : 0;
    CHECK(mkdir_first == 0, "mkdir creates a new directory");
    CHECK(mkdir_duplicate == -1, "mkdir reports an existing directory as failure");
    CHECK(duplicate_errno == EEXIST, "mkdir preserves EEXIST for the caller");
    CHECK(kernel_mkdir_duplicate == 0x80020011u,
          "sceKernelMkdir returns SCE_KERNEL_ERROR_EEXIST directly");
    set_app0_root(std::filesystem::current_path().string());
    const std::string reachable_file = std::string("/app0/") + path;
    const std::string reachable_directory = std::string("/app0/") + dir_path;
    CHECK(kernel_reachability_fn &&
              kernel_reachability_fn((uint64_t)(uintptr_t)reachable_file.c_str(), 0, 0, 0, 0, 0) == 0,
          "sceKernelCheckReachability finds a translated guest file");
    CHECK(kernel_reachability_fn &&
              kernel_reachability_fn((uint64_t)(uintptr_t)reachable_directory.c_str(), 0, 0, 0, 0, 0) == 0,
          "sceKernelCheckReachability finds a translated guest directory");
#ifdef _WIN32
    CHECK(SetFileAttributesA(path, FILE_ATTRIBUTE_NORMAL),
          "prepare a normal-attribute Windows chmod fixture");
    auto read_chmod_times = [&](FILETIME* access, FILETIME* modify) {
        HANDLE file = CreateFileA(path, FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, 0, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;
        const BOOL result = GetFileTime(file, nullptr, access, modify);
        CloseHandle(file);
        return result != FALSE;
    };
    FILETIME chmod_access_before{}, chmod_modify_before{};
    CHECK(read_chmod_times(&chmod_access_before, &chmod_modify_before),
          "read Windows fixture timestamps before chmod");
#endif
    CHECK(kernel_chmod_fn &&
              kernel_chmod_fn((uint64_t)(uintptr_t)reachable_file.c_str(), 0444, 0, 0, 0, 0) == 0,
          "sceKernelChmod makes a translated guest file read-only");
    CHECK(kernel_chmod_fn &&
              kernel_chmod_fn((uint64_t)(uintptr_t)reachable_directory.c_str(), 0555, 0, 0, 0, 0) == 0,
          "sceKernelChmod applies a mode to a translated guest directory");
#ifdef _WIN32
    DWORD readonly_attributes = GetFileAttributesA(path);
    CHECK((readonly_attributes & FILE_ATTRIBUTE_READONLY) != 0 &&
              (readonly_attributes & FILE_ATTRIBUTE_NORMAL) == 0,
          "Windows sceKernelChmod normalizes the read-only attribute set");
#endif
    CHECK(kernel_chmod_fn &&
              kernel_chmod_fn((uint64_t)(uintptr_t)reachable_file.c_str(), 0644, 0, 0, 0, 0) == 0 &&
              kernel_chmod_fn((uint64_t)(uintptr_t)reachable_directory.c_str(), 0755, 0, 0, 0, 0) == 0,
          "sceKernelChmod restores writable fixture modes");
#ifdef _WIN32
    FILETIME chmod_access_after{}, chmod_modify_after{};
    CHECK(GetFileAttributesA(path) == FILE_ATTRIBUTE_NORMAL &&
              read_chmod_times(&chmod_access_after, &chmod_modify_after) &&
              CompareFileTime(&chmod_access_before, &chmod_access_after) == 0 &&
              CompareFileTime(&chmod_modify_before, &chmod_modify_after) == 0,
          "Windows sceKernelChmod restores normal attributes without changing timestamps");
#endif
    int64_t chmod_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)reachable_file.c_str(), 0, 0, 0, 0, 0)
        : -1;
    CHECK(chmod_fd >= 3, "open a descriptor for sceKernelFchmod");
    CHECK(chmod_fd >= 0 && kernel_fchmod_fn &&
              kernel_fchmod_fn((uint64_t)chmod_fd, 0400, 0, 0, 0, 0) == 0,
          "sceKernelFchmod makes an open file read-only");
#ifdef _WIN32
    readonly_attributes = GetFileAttributesA(path);
    CHECK((readonly_attributes & FILE_ATTRIBUTE_READONLY) != 0 &&
              (readonly_attributes & FILE_ATTRIBUTE_NORMAL) == 0,
          "Windows sceKernelFchmod normalizes the descriptor's read-only attribute");
#endif
    CHECK(chmod_fd >= 0 && kernel_fchmod_fn &&
              kernel_fchmod_fn((uint64_t)chmod_fd, 0600, 0, 0, 0, 0) == 0,
          "sceKernelFchmod restores a writable descriptor mode");
#ifdef _WIN32
    FILETIME fchmod_access_after{}, fchmod_modify_after{};
    CHECK(GetFileAttributesA(path) == FILE_ATTRIBUTE_NORMAL &&
              read_chmod_times(&fchmod_access_after, &fchmod_modify_after) &&
              CompareFileTime(&chmod_access_before, &fchmod_access_after) == 0 &&
              CompareFileTime(&chmod_modify_before, &fchmod_modify_after) == 0,
          "Windows sceKernelFchmod restores normal attributes without changing timestamps");
#endif
    if (chmod_fd >= 0 && close_fn) close_fn((uint64_t)chmod_fd, 0, 0, 0, 0, 0);
#ifndef _WIN32
    // WSL's /mnt/c metadata bridge reports synthetic permission bits. Verify exact POSIX chmod
    // behavior on a unique native-filesystem fixture while retaining /app0 translation above.
    // Deliberately NOT tests/test_scratch.h (#1613). What this fixture needs is a filesystem that
    // keeps real POSIX metadata, and under WSL the build directory is the /mnt/c mount whose bridge
    // is exactly what it guards against — routing it through the scratch helper would break it
    // there unconditionally. mkstemp already makes the name unique and the file is empty, so
    // neither the tmpfs quota nor a concurrent ctest case is exposed.
    // Caveat: temp_directory_path() honors TMPDIR, so pointing TMPDIR at a metadata-lossy mount
    // does reach this fixture. That is fail-visible (the assertions below fail; they cannot
    // silently pass), which is why it is left as a caveat rather than pinned to a literal path.
    std::string mode_path =
        (std::filesystem::temp_directory_path() / "prosper-test-chmod-mode-XXXXXX").string();
    const int mode_fd = ::mkstemp(mode_path.data());
    CHECK(mode_fd >= 0, "create unique native-filesystem mode fixture");
    std::array<uint8_t, 0x78> mode_stat{};
    CHECK(mode_fd >= 0 && kernel_chmod_fn && kernel_stat_fn &&
              kernel_chmod_fn((uint64_t)(uintptr_t)mode_path.c_str(), 0444, 0, 0, 0, 0) == 0 &&
              kernel_stat_fn((uint64_t)(uintptr_t)mode_path.c_str(),
                             (uint64_t)(uintptr_t)mode_stat.data(), 0, 0, 0, 0) == 0 &&
              (*(const uint16_t*)(mode_stat.data() + 0x08) & 0777) == 0444,
          "POSIX sceKernelChmod preserves requested permission bits");
    mode_stat.fill(0);
    CHECK(mode_fd >= 0 && kernel_fchmod_fn && kernel_fstat_fn &&
              kernel_fchmod_fn((uint64_t)mode_fd, 0600, 0, 0, 0, 0) == 0 &&
              kernel_fstat_fn((uint64_t)mode_fd, (uint64_t)(uintptr_t)mode_stat.data(),
                              0, 0, 0, 0) == 0 &&
              (*(const uint16_t*)(mode_stat.data() + 0x08) & 0777) == 0600,
          "POSIX sceKernelFchmod preserves descriptor permission bits");
    if (mode_fd >= 0) ::close(mode_fd);
    std::filesystem::remove(mode_path, remove_error);
#endif
    GuestTimeval explicit_times[2]{{1700000001, 123456}, {1700000002, 654321}};
    CHECK(kernel_utimes_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)reachable_file.c_str(),
                                (uint64_t)(uintptr_t)explicit_times, 0, 0, 0, 0) == 0,
          "sceKernelUtimes applies explicit timestamps to a translated guest file");
    CHECK(kernel_utimes_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)reachable_directory.c_str(),
                                (uint64_t)(uintptr_t)explicit_times, 0, 0, 0, 0) == 0,
          "sceKernelUtimes applies explicit timestamps to a translated guest directory");
    std::array<uint8_t, 0x78> timestamp_stat{};
    CHECK(kernel_stat_fn &&
              kernel_stat_fn((uint64_t)(uintptr_t)reachable_file.c_str(),
                             (uint64_t)(uintptr_t)timestamp_stat.data(), 0, 0, 0, 0) == 0 &&
              *(const int64_t*)(timestamp_stat.data() + 0x18) == explicit_times[0].sec &&
              *(const int64_t*)(timestamp_stat.data() + 0x28) == explicit_times[1].sec,
          "sceKernelUtimes preserves explicit access/modify seconds");
#ifdef _WIN32
    HANDLE timestamp_handle = CreateFileA(path, FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    FILETIME access_time{}, modify_time{};
    const BOOL read_times = timestamp_handle != INVALID_HANDLE_VALUE &&
        GetFileTime(timestamp_handle, nullptr, &access_time, &modify_time);
    auto filetime_ticks = [](const FILETIME& value) {
        return (uint64_t)value.dwLowDateTime | ((uint64_t)value.dwHighDateTime << 32);
    };
    constexpr uint64_t kUnixToWindowsEpochSeconds = 11644473600ull;
    constexpr uint64_t kTicksPerSecond = 10000000ull;
    CHECK(read_times &&
              filetime_ticks(access_time) ==
                  ((uint64_t)explicit_times[0].sec + kUnixToWindowsEpochSeconds) * kTicksPerSecond +
                      (uint64_t)explicit_times[0].usec * 10 &&
              filetime_ticks(modify_time) ==
                  ((uint64_t)explicit_times[1].sec + kUnixToWindowsEpochSeconds) * kTicksPerSecond +
                      (uint64_t)explicit_times[1].usec * 10,
          "sceKernelUtimes preserves explicit Windows microseconds");
    if (timestamp_handle != INVALID_HANDLE_VALUE) CloseHandle(timestamp_handle);
#else
    // The source tree can live on WSL's /mnt/c mount, whose metadata bridge truncates subsecond
    // timestamps. Exercise microsecond preservation on the native temporary filesystem instead.
    // Deliberately NOT tests/test_scratch.h — see the chmod fixture above for why (#1613).
    std::string precision_path =
        (std::filesystem::temp_directory_path() / "prosper-test-utimes-precision-XXXXXX").string();
    const int precision_fd = ::mkstemp(precision_path.data());
    CHECK(precision_fd >= 0, "create unique native-filesystem timestamp fixture");
    if (precision_fd >= 0) ::close(precision_fd);
    timestamp_stat.fill(0);
    CHECK(precision_fd >= 0 && kernel_utimes_fn && kernel_stat_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)precision_path.c_str(),
                                (uint64_t)(uintptr_t)explicit_times, 0, 0, 0, 0) == 0 &&
              kernel_stat_fn((uint64_t)(uintptr_t)precision_path.c_str(),
                             (uint64_t)(uintptr_t)timestamp_stat.data(), 0, 0, 0, 0) == 0 &&
              *(const int64_t*)(timestamp_stat.data() + 0x20) == explicit_times[0].usec * 1000 &&
              *(const int64_t*)(timestamp_stat.data() + 0x30) == explicit_times[1].usec * 1000,
          "sceKernelUtimes preserves explicit POSIX microseconds");
    std::filesystem::remove(precision_path, remove_error);
#endif
    GuestTimeval invalid_times[2]{{1700000001, 1000000}, {1700000002, 0}};
    CHECK(kernel_utimes_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)reachable_file.c_str(),
                                (uint64_t)(uintptr_t)invalid_times, 0, 0, 0, 0) == 0x80020016u,
          "sceKernelUtimes rejects an out-of-range microsecond field");
    timestamp_stat.fill(0);
    CHECK(kernel_stat_fn &&
              kernel_stat_fn((uint64_t)(uintptr_t)reachable_file.c_str(),
                             (uint64_t)(uintptr_t)timestamp_stat.data(), 0, 0, 0, 0) == 0 &&
              *(const int64_t*)(timestamp_stat.data() + 0x28) == explicit_times[1].sec,
          "rejected sceKernelUtimes leaves the modify time unchanged");
    CHECK(kernel_utimes_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)reachable_file.c_str(), 0, 0, 0, 0, 0) == 0,
          "sceKernelUtimes supports null-times touch-now semantics");
    timestamp_stat.fill(0);
    CHECK(kernel_stat_fn &&
              kernel_stat_fn((uint64_t)(uintptr_t)reachable_file.c_str(),
                             (uint64_t)(uintptr_t)timestamp_stat.data(), 0, 0, 0, 0) == 0 &&
              *(const int64_t*)(timestamp_stat.data() + 0x28) > explicit_times[1].sec,
          "touch-now sceKernelUtimes advances the modify time");
#ifdef _WIN32
    // Windows has no CRT directory-open API, but a directory HANDLE can be attached to a CRT fd.
    // The zero-entry stub must still publish the defined starting offset through basep.
    HANDLE directory_handle = CreateFileA(dir_path, GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    int directory_fd = directory_handle != INVALID_HANDLE_VALUE
        ? ::_open_osfhandle((intptr_t)directory_handle, _O_RDONLY | _O_BINARY)
        : -1;
    int64_t directory_base = 0x12345678;
    uint64_t directory_result = directory_fd >= 0 && kernel_getdirentries_fn
        ? kernel_getdirentries_fn((uint64_t)directory_fd,
                                  (uint64_t)(uintptr_t)dir_buffer.data(), dir_buffer.size(),
                                  (uint64_t)(uintptr_t)&directory_base, 0, 0)
        : ~uint64_t{0};
    CHECK(directory_result == 0 && directory_base == 0,
          "Windows sceKernelGetdirentries initializes basep on success");
    if (directory_fd >= 0) ::_close(directory_fd);
    else if (directory_handle != INVALID_HANDLE_VALUE) CloseHandle(directory_handle);

    // A pipe is a valid CRT descriptor, but not a directory. Do not conflate an unsupported
    // handle type with EBADF after _get_osfhandle has established descriptor validity.
    int pipe_fds[2]{-1, -1};
    int pipe_result = ::_pipe(pipe_fds, 256, _O_BINARY);
    uint64_t pipe_getdents = pipe_result == 0 && kernel_getdents_fn
        ? kernel_getdents_fn((uint64_t)pipe_fds[0], (uint64_t)(uintptr_t)dir_buffer.data(),
                             dir_buffer.size(), 0, 0, 0)
        : 0;
    CHECK(pipe_result == 0 && (uint32_t)pipe_getdents == 0x80020014u,
          "Windows valid non-directory handles return SCE_KERNEL_ERROR_ENOTDIR");
    if (pipe_fds[0] >= 0) ::_close(pipe_fds[0]);
    if (pipe_fds[1] >= 0) ::_close(pipe_fds[1]);
#endif
    std::filesystem::remove_all(dir_path, remove_error);

    // PS5 directory enumeration uses the ordinary open/getdents/close descriptor flow. The
    // Windows CRT refuses to open a directory at all, so the HLE must retain an emulator-owned
    // directory cursor instead of reporting a successful empty enumeration. Astro Bot discovers
    // each intro cinematic sublevel through this exact contract.
    const std::filesystem::path dents_path = "prosper-test-getdents.tmp";
    std::filesystem::remove_all(dents_path, remove_error);
    std::filesystem::create_directories(dents_path / "c01");
    std::filesystem::create_directories(dents_path / "c02b");
    FILE* dents_file = std::fopen((dents_path / "cinematics.cx").string().c_str(), "wb");
    CHECK(dents_file != nullptr, "create getdents file fixture");
    if (dents_file) { std::fputc(0x5a, dents_file); std::fclose(dents_file); }

    const std::string dents_string = dents_path.string();
    constexpr uint64_t kGuestDirectory = 0x00020000ull;
    int64_t dir_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)dents_string.c_str(),
                           kGuestDirectory, 0, 0, 0, 0)
        : -1;
    CHECK(dir_fd >= 3, "sceKernelOpen returns a guest directory descriptor");
    std::array<uint8_t, 4096> dents{};
    int64_t dent_bytes = dir_fd >= 0 && kernel_getdents_fn
        ? (int64_t)kernel_getdents_fn((uint64_t)dir_fd, (uint64_t)(uintptr_t)dents.data(),
                                      dents.size(), 0, 0, 0)
        : -1;
    CHECK(dent_bytes > 0, "sceKernelGetdents returns directory records");
    std::set<std::string> dent_names;
    std::map<std::string, uint8_t> dent_types;
    bool valid_records = dent_bytes > 0 && dent_bytes <= (int64_t)dents.size();
    for (size_t offset = 0; valid_records && offset < (size_t)dent_bytes;) {
        const uint16_t record_size = *(const uint16_t*)(dents.data() + offset + 4);
        const uint8_t name_size = dents[offset + 7];
        if (record_size < 8 || offset + record_size > (size_t)dent_bytes ||
            name_size + 9 > record_size) {
            valid_records = false;
            break;
        }
        std::string name((const char*)dents.data() + offset + 8, name_size);
        dent_names.insert(name);
        dent_types[name] = dents[offset + 6];
        offset += record_size;
    }
    CHECK(valid_records, "getdents records use bounded FreeBSD layout");
    CHECK(dent_names.count("c01") && dent_names.count("c02b") &&
              dent_names.count("cinematics.cx"),
          "getdents exposes cinematic directories and files");
    CHECK(dent_types["c01"] == 4 && dent_types["c02b"] == 4 &&
              dent_types["cinematics.cx"] == 8,
          "getdents reports directory/file d_type values");
    int64_t dents_eof = dir_fd >= 0 && kernel_getdents_fn
        ? (int64_t)kernel_getdents_fn((uint64_t)dir_fd, (uint64_t)(uintptr_t)dents.data(),
                                      dents.size(), 0, 0, 0)
        : -1;
    CHECK(dents_eof == 0, "getdents preserves its per-open end cursor");

    std::array<uint8_t, 0x78> dir_stat{};
    int64_t dir_stat_result = dir_fd >= 0 && kernel_fstat_fn
        ? (int64_t)kernel_fstat_fn((uint64_t)dir_fd, (uint64_t)(uintptr_t)dir_stat.data(),
                                   0, 0, 0, 0)
        : -1;
    CHECK(dir_stat_result == 0 &&
              ((*(const uint16_t*)(dir_stat.data() + 8)) & 0xf000u) == 0x4000u,
          "sceKernelFstat identifies an open directory descriptor");

#ifdef _WIN32
    int64_t rewind_dir = dir_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)dir_fd, 0, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t rewound_bytes = dir_fd >= 0 && kernel_getdents_fn
        ? (int64_t)kernel_getdents_fn((uint64_t)dir_fd,
                                      (uint64_t)(uintptr_t)dents.data(), dents.size(), 0, 0, 0)
        : -1;
    CHECK(rewind_dir == 0 && rewound_bytes > 0,
          "Windows directory descriptors can be rewound and enumerated again");
    errno = 0;
    int64_t clear_dir_cloexec = dir_fd >= 0 && fcntl_fn
        ? (int64_t)fcntl_fn((uint64_t)dir_fd, kGuestFSetFd, 0, 0, 0, 0)
        : 0;
    int clear_dir_errno = errno;
    uint64_t dir_fd_flags = dir_fd >= 0 && fcntl_fn
        ? fcntl_fn((uint64_t)dir_fd, kGuestFGetFd, 0, 0, 0, 0)
        : 0;
    CHECK(clear_dir_cloexec == -1 && clear_dir_errno == kFreeBsdENotSupp &&
              dir_fd_flags == kGuestFdCloExec,
          "Windows directory fcntl rejects an unrepresentable close-on-exec change "
          "with FreeBSD's ENOTSUP");
#endif

    // Darwin's getdents compatibility path owns a duplicated descriptor behind DIR*. A raw
    // lseek on the original descriptor cannot portably rewind that cached stream, so exercise
    // getdirentries from a fresh open while retaining the Windows synthetic-cursor rewind check.
    int64_t direntries_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)dents_string.c_str(),
                           kGuestDirectory, 0, 0, 0, 0)
        : -1;
    CHECK(direntries_fd >= 3, "open a fresh directory cursor for getdirentries");
    int64_t directory_cursor_base = -1;
    int64_t entry_bytes = direntries_fd >= 0 && kernel_getdirentries_fn
        ? (int64_t)kernel_getdirentries_fn(
              (uint64_t)direntries_fd, (uint64_t)(uintptr_t)dents.data(), dents.size(),
              (uint64_t)(uintptr_t)&directory_cursor_base, 0, 0)
        : -1;
    CHECK(entry_bytes > 0 && directory_cursor_base == 0,
          "getdirentries reports and advances a fresh directory cursor");
    if (direntries_fd >= 0 && close_fn)
        close_fn((uint64_t)direntries_fd, 0, 0, 0, 0, 0);
    if (dir_fd >= 0 && close_fn) close_fn((uint64_t)dir_fd, 0, 0, 0, 0, 0);
    uint64_t closed_dents = kernel_getdents_fn
        ? kernel_getdents_fn((uint64_t)dir_fd, (uint64_t)(uintptr_t)dents.data(),
                             dents.size(), 0, 0, 0)
        : 0;
    CHECK((uint32_t)closed_dents == 0x80020009u,
          "closed directory descriptor returns EBADF instead of silent EOF");
    std::filesystem::remove_all(dents_path, remove_error);

    std::array<uint8_t, 512> actual{};
    const char* missing_path = "prosper-test-file-binary-missing.tmp";
    std::error_code missing_remove_error;
    std::filesystem::remove(missing_path, missing_remove_error);
    errno = 0;
    int64_t posix_missing = posix_open_fn
        ? (int64_t)posix_open_fn((uint64_t)(uintptr_t)missing_path, 0, 0, 0, 0, 0)
        : 0;
    int posix_missing_errno = errno;
    uint64_t kernel_missing = open_fn
        ? open_fn((uint64_t)(uintptr_t)missing_path, 0, 0, 0, 0, 0)
        : 0;
    CHECK(posix_missing == -1 && posix_missing_errno == ENOENT,
          "libc open retains the -1 plus errno contract");
    CHECK((uint32_t)kernel_missing == 0x80020002u,
          "sceKernelOpen returns SCE_KERNEL_ERROR_ENOENT directly");

    // PROSPER_DENY_SUBSTR case-insensitivity (#1237): the fixture EXISTS on disk, but its name
    // contains a CASE VARIANT of the armed ".tmpdeny" substring — the deny must still fire.
    {
        const char* deny_fixture = "prosper-deny-fixture.TmpDeny";
        FILE* deny_out = std::fopen(deny_fixture, "wb");
        CHECK(deny_out != nullptr, "create deny fixture");
        if (deny_out) { std::fputs("x", deny_out); std::fclose(deny_out); }
        const uint64_t denied = open_fn
            ? open_fn((uint64_t)(uintptr_t)deny_fixture, 0, 0, 0, 0, 0) : 0;
        CHECK((uint32_t)denied == 0x80020002u,
              "PROSPER_DENY_SUBSTR denies a case-variant spelling (ENOENT despite the file existing)");
        std::remove(deny_fixture);
    }

    // Virtual-root clamp (#1234): the jailed title's "/app0/.." is its readable root on real
    // hardware (ArcRunner opens it during boot). It must resolve (directory open succeeds), a
    // mount re-entry through the root must reach the real file, and a traversal to an unmounted
    // sibling must KEEP the #1205 deny (ENOENT) — never fall through to host passthrough.
    {
        const uint64_t root_fd = open_fn
            ? open_fn((uint64_t)(uintptr_t)"/app0/..", 0, 0, 0, 0, 0) : 0;
        CHECK((int64_t)root_fd >= 3, "'/app0/..' opens as the virtual sandbox root");
        if ((int64_t)root_fd >= 3) {
            HleFn close_fn = Hle::lookup(nid_hash("sceKernelClose"));
            if (close_fn) close_fn(root_fd, 0, 0, 0, 0, 0);
        }
        const uint64_t reentry_fd = open_fn
            ? open_fn((uint64_t)(uintptr_t)"/app0/../app0/prosper-test-file-binary.tmp",
                      0, 0, 0, 0, 0)
            : 0;
        CHECK((int64_t)reentry_fd >= 3, "'/app0/../app0/<file>' re-enters the mount");
        if ((int64_t)reentry_fd >= 3) {
            HleFn close_fn = Hle::lookup(nid_hash("sceKernelClose"));
            if (close_fn) close_fn(reentry_fd, 0, 0, 0, 0, 0);
        }
        const uint64_t sibling = open_fn
            ? open_fn((uint64_t)(uintptr_t)"/app0/../etc", 0, 0, 0, 0, 0) : 0;
        CHECK((uint32_t)sibling == 0x80020002u,
              "'/app0/../etc' keeps the sandbox-traversal deny (ENOENT)");
    }
    const std::string unreachable_file = std::string("/app0/") + missing_path;
    std::string overlong_path(256, 'a');
    CHECK(kernel_reachability_fn &&
              kernel_reachability_fn((uint64_t)(uintptr_t)unreachable_file.c_str(), 0, 0, 0, 0, 0) ==
                  0x80020002u,
          "sceKernelCheckReachability returns SCE_KERNEL_ERROR_ENOENT for a missing path");
    CHECK(kernel_reachability_fn && kernel_reachability_fn(0, 0, 0, 0, 0, 0) == 0x80020016u,
          "sceKernelCheckReachability returns SCE_KERNEL_ERROR_EINVAL for a null path");
    CHECK(kernel_reachability_fn &&
              kernel_reachability_fn((uint64_t)(uintptr_t)overlong_path.c_str(), 0, 0, 0, 0, 0) ==
                  0x8002003fu,
          "sceKernelCheckReachability enforces the 255-byte console path bound");
    CHECK(kernel_utimes_fn &&
              kernel_utimes_fn((uint64_t)(uintptr_t)unreachable_file.c_str(), 0, 0, 0, 0, 0) ==
                  0x80020002u,
          "sceKernelUtimes returns SCE_KERNEL_ERROR_ENOENT for a missing path");
    CHECK(kernel_utimes_fn && kernel_utimes_fn(0, 0, 0, 0, 0, 0) == 0x8002000eu,
          "sceKernelUtimes returns SCE_KERNEL_ERROR_EFAULT for a null path");
    errno = 0;
    int64_t missing_chmod = chmod_fn
        ? (int64_t)chmod_fn((uint64_t)(uintptr_t)unreachable_file.c_str(), 0600, 0, 0, 0, 0)
        : 0;
    int missing_chmod_errno = errno;
    CHECK(missing_chmod == -1 && missing_chmod_errno == ENOENT,
          "libc chmod retains -1 plus ENOENT");
    CHECK(kernel_chmod_fn &&
              kernel_chmod_fn((uint64_t)(uintptr_t)unreachable_file.c_str(),
                              0600, 0, 0, 0, 0) == 0x80020002u,
          "sceKernelChmod returns SCE_KERNEL_ERROR_ENOENT for a missing path");
    CHECK(kernel_chmod_fn && kernel_chmod_fn(0, 0600, 0, 0, 0, 0) == 0x8002000eu,
          "sceKernelChmod returns SCE_KERNEL_ERROR_EFAULT for a null path");
    errno = 0;
    int64_t invalid_fchmod = fchmod_fn
        ? (int64_t)fchmod_fn(~uint64_t{0}, 0600, 0, 0, 0, 0)
        : 0;
    int invalid_fchmod_errno = errno;
    CHECK(invalid_fchmod == -1 && invalid_fchmod_errno == EBADF,
          "libc fchmod retains -1 plus EBADF");
    CHECK(kernel_fchmod_fn &&
              kernel_fchmod_fn(~uint64_t{0}, 0600, 0, 0, 0, 0) == 0x80020009u,
          "sceKernelFchmod returns SCE_KERNEL_ERROR_EBADF for an invalid descriptor");

    // The stat-family libc names preserve -1 plus errno, while their kernel siblings return
    // a 32-bit SCE error directly. Neither contract may overwrite the guest buffer on failure.
    auto check_missing_stat_contract = [&](const char* operation, HleFn libc_fn,
                                           HleFn kernel_fn) {
        std::array<uint8_t, 0x78> libc_buffer;
        std::array<uint8_t, 0x78> kernel_buffer;
        libc_buffer.fill(0x5a);
        kernel_buffer.fill(0x5a);
        const auto untouched = libc_buffer;
        errno = 0;
        uint64_t libc_result = libc_fn
            ? libc_fn((uint64_t)(uintptr_t)missing_path,
                      (uint64_t)(uintptr_t)libc_buffer.data(), 0, 0, 0, 0)
            : 0;
        const int libc_error = errno;
        uint64_t kernel_result = kernel_fn
            ? kernel_fn((uint64_t)(uintptr_t)missing_path,
                        (uint64_t)(uintptr_t)kernel_buffer.data(), 0, 0, 0, 0)
            : 0;
        const std::string libc_message = std::string("libc ") + operation +
                                         " retains -1 plus ENOENT";
        const std::string kernel_message = std::string("sceKernel") + operation +
                                           " returns SCE_KERNEL_ERROR_ENOENT directly";
        const std::string buffer_message = std::string(operation) +
                                            " failure leaves stat buffers untouched";
        CHECK((int64_t)libc_result == -1 && libc_error == ENOENT, libc_message.c_str());
        CHECK(kernel_result == 0x80020002u, kernel_message.c_str());
        CHECK(libc_buffer == untouched && kernel_buffer == untouched, buffer_message.c_str());
    };
    check_missing_stat_contract("Stat", stat_fn, kernel_stat_fn);
    check_missing_stat_contract("Lstat", lstat_fn, kernel_lstat_fn);

    auto check_missing_path_contract = [&](const char* operation, HleFn libc_fn,
                                           HleFn kernel_fn) {
        errno = 0;
        uint64_t libc_result = libc_fn
            ? libc_fn((uint64_t)(uintptr_t)missing_path, 0, 0, 0, 0, 0)
            : 0;
        const int libc_error = errno;
        uint64_t kernel_result = kernel_fn
            ? kernel_fn((uint64_t)(uintptr_t)missing_path, 0, 0, 0, 0, 0)
            : 0;
        const std::string libc_message = std::string("libc ") + operation +
                                         " retains -1 plus ENOENT";
        const std::string kernel_message = std::string("sceKernel") + operation +
                                           " returns SCE_KERNEL_ERROR_ENOENT directly";
        CHECK((int64_t)libc_result == -1 && libc_error == ENOENT, libc_message.c_str());
        CHECK(kernel_result == 0x80020002u, kernel_message.c_str());
    };
    check_missing_path_contract("Rmdir", rmdir_fn, kernel_rmdir_fn);
    check_missing_path_contract("Unlink", unlink_fn, kernel_unlink_fn);

    const char* missing_rename_target = "prosper-test-file-binary-missing-renamed.tmp";
    std::filesystem::remove(missing_rename_target, missing_remove_error);
    errno = 0;
    int64_t libc_missing_rename = rename_fn
        ? (int64_t)rename_fn((uint64_t)(uintptr_t)missing_path,
                             (uint64_t)(uintptr_t)missing_rename_target, 0, 0, 0, 0)
        : 0;
    const int libc_missing_rename_error = errno;
    uint64_t kernel_missing_rename = kernel_rename_fn
        ? kernel_rename_fn((uint64_t)(uintptr_t)missing_path,
                           (uint64_t)(uintptr_t)missing_rename_target, 0, 0, 0, 0)
        : 0;
    CHECK(libc_missing_rename == -1 && libc_missing_rename_error == ENOENT,
          "libc rename retains -1 plus ENOENT");
    CHECK(kernel_missing_rename == 0x80020002u,
          "sceKernelRename returns SCE_KERNEL_ERROR_ENOENT directly");

    uint64_t kernel_missing_truncate = kernel_truncate_fn
        ? kernel_truncate_fn((uint64_t)(uintptr_t)missing_path, 3, 0, 0, 0, 0)
        : 0;
    CHECK(kernel_missing_truncate == 0x80020002u,
          "sceKernelTruncate returns SCE_KERNEL_ERROR_ENOENT directly");

    const char* truncate_path = "prosper-test-kernel-truncate.tmp";
    std::filesystem::remove(truncate_path, missing_remove_error);
    FILE* truncate_out = std::fopen(truncate_path, "wb");
    bool truncate_fixture_written = false;
    if (truncate_out) {
        const std::array<uint8_t, 8> truncate_bytes{{0, 1, 2, 3, 4, 5, 6, 7}};
        truncate_fixture_written =
            std::fwrite(truncate_bytes.data(), 1, truncate_bytes.size(), truncate_out) ==
            truncate_bytes.size();
        std::fclose(truncate_out);
    }
    uint64_t kernel_truncate_result = truncate_fixture_written && kernel_truncate_fn
        ? kernel_truncate_fn((uint64_t)(uintptr_t)truncate_path, 3, 0, 0, 0, 0)
        : ~uint64_t{0};
    std::error_code truncate_size_error;
    uintmax_t truncated_size = std::filesystem::file_size(truncate_path, truncate_size_error);
    CHECK(truncate_fixture_written && kernel_truncate_result == 0 && !truncate_size_error &&
              truncated_size == 3,
          "sceKernelTruncate resizes a path on both hosts");
    std::filesystem::remove(truncate_path, missing_remove_error);

    const char* descriptor_resize_path = "prosper-test-kernel-ftruncate.tmp";
    std::filesystem::remove(descriptor_resize_path, missing_remove_error);
    FILE* descriptor_resize_out = std::fopen(descriptor_resize_path, "wb");
    bool descriptor_resize_fixture_written = false;
    if (descriptor_resize_out) {
        const std::array<uint8_t, 8> resize_bytes{{0, 1, 2, 3, 4, 5, 6, 7}};
        descriptor_resize_fixture_written =
            std::fwrite(resize_bytes.data(), 1, resize_bytes.size(), descriptor_resize_out) ==
            resize_bytes.size();
        std::fclose(descriptor_resize_out);
    }
    int64_t descriptor_resize_fd = descriptor_resize_fixture_written && open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)descriptor_resize_path, 2, 0, 0, 0, 0)
        : -1;
    uint64_t kernel_ftruncate_result = descriptor_resize_fd >= 0 && kernel_ftruncate_fn
        ? kernel_ftruncate_fn((uint64_t)descriptor_resize_fd, 3, 0, 0, 0, 0)
        : ~uint64_t{0};
    uint64_t kernel_fsync_result = descriptor_resize_fd >= 0 && kernel_fsync_fn
        ? kernel_fsync_fn((uint64_t)descriptor_resize_fd, 0, 0, 0, 0, 0)
        : ~uint64_t{0};
    uint64_t kernel_fdatasync_result = descriptor_resize_fd >= 0 && kernel_fdatasync_fn
        ? kernel_fdatasync_fn((uint64_t)descriptor_resize_fd, 0, 0, 0, 0, 0)
        : ~uint64_t{0};
    if (descriptor_resize_fd >= 0 && close_fn)
        close_fn((uint64_t)descriptor_resize_fd, 0, 0, 0, 0, 0);
    std::error_code descriptor_resize_size_error;
    uintmax_t descriptor_resize_size =
        std::filesystem::file_size(descriptor_resize_path, descriptor_resize_size_error);
    CHECK(descriptor_resize_fixture_written && kernel_ftruncate_result == 0 &&
              kernel_fsync_result == 0 && kernel_fdatasync_result == 0 &&
              !descriptor_resize_size_error &&
              descriptor_resize_size == 3,
          "sceKernelFtruncate/Fsync/Fdatasync persist a resized descriptor on both hosts");
    std::filesystem::remove(descriptor_resize_path, missing_remove_error);

    constexpr uint64_t kGuestOWriteOnly = 0x0001;
    constexpr uint64_t kGuestOCreate = 0x0200;
    constexpr uint64_t kGuestOExclusive = 0x0800;
    uint64_t kernel_existing = open_fn
        ? open_fn((uint64_t)(uintptr_t)path,
                  kGuestOWriteOnly | kGuestOCreate | kGuestOExclusive, 0600, 0, 0, 0)
        : 0;
    CHECK((uint32_t)kernel_existing == 0x80020011u,
          "sceKernelOpen translates an exclusive-create collision to EEXIST");

#ifndef _WIN32
    // Linux ELOOP is 40, while FreeBSD/Orbis ELOOP is 62. A real symlink cycle guards
    // against accidentally embedding the host errno in a kernel result.
    const char* loop_a = "prosper-test-open-loop-a.tmp";
    const char* loop_b = "prosper-test-open-loop-b.tmp";
    std::error_code symlink_error;
    std::filesystem::remove(loop_a, symlink_error);
    std::filesystem::remove(loop_b, symlink_error);
    std::filesystem::create_symlink(loop_b, loop_a, symlink_error);
    if (!symlink_error) std::filesystem::create_symlink(loop_a, loop_b, symlink_error);
    uint64_t kernel_loop = !symlink_error && open_fn
        ? open_fn((uint64_t)(uintptr_t)loop_a, 0, 0, 0, 0, 0)
        : 0;
    CHECK(!symlink_error && (uint32_t)kernel_loop == 0x8002003eu,
          "sceKernelOpen translates host ELOOP to the divergent Orbis value");
    std::filesystem::remove(loop_a, symlink_error);
    std::filesystem::remove(loop_b, symlink_error);
#endif

    int64_t fd = open_fn ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0) : -1;
    CHECK(fd >= 0, "open fixture through guest fd HLE");

    // fcntl was unregistered, so the generic missing-import path returned false success (zero).
    // Exercise descriptor discovery and status updates with guest values that catch raw Linux
    // passthrough: O_NONBLOCK is 0x4 on FreeBSD/Orbis, but 0x800 on Linux.
#ifndef _WIN32
    uint64_t guest_flags = fcntl_fn && fd >= 0
        ? fcntl_fn((uint64_t)fd, kGuestFGetFl, 0, 0, 0, 0)
        : ~uint64_t{0};
    CHECK(guest_flags == 0, "fcntl F_GETFL reports guest O_RDONLY flags");
    int64_t set_flags = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, kGuestFSetFl,
                            kGuestONonblock | kGuestOAppend, 0, 0, 0)
        : -1;
    uint64_t changed_flags = kernel_fcntl_fn && fd >= 0
        ? kernel_fcntl_fn((uint64_t)fd, kGuestFGetFl, 0, 0, 0, 0)
        : ~uint64_t{0};
    CHECK(set_flags == 0, "fcntl F_SETFL accepts translated guest status flags");
    CHECK((changed_flags & (3 | kGuestONonblock | kGuestOAppend)) ==
              (kGuestONonblock | kGuestOAppend),
          "sceKernelFcntl F_GETFL returns translated non-blocking and append bits");
    errno = 0;
    int64_t unsupported_sync_change = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, kGuestFSetFl,
                            changed_flags | kGuestOSync, 0, 0, 0)
        : 0;
    uint64_t flags_after_sync_change = fcntl_fn && fd >= 0
        ? fcntl_fn((uint64_t)fd, kGuestFGetFl, 0, 0, 0, 0)
        : ~uint64_t{0};
    CHECK(unsupported_sync_change == -1 && errno == kFreeBsdENotSupp &&
              (flags_after_sync_change & kGuestOSync) == 0,
          "fcntl rejects an unsupported O_SYNC change with FreeBSD's ENOTSUP, not the host's");
#else
    // UCRT exposes neither F_GETFL nor F_SETFL. Until the secondary Windows host grows an
    // equivalent descriptor-status layer, report that limitation instead of fake success.
    errno = 0;
    int64_t guest_flags = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, kGuestFGetFl, 0, 0, 0, 0)
        : 0;
    int getfl_errno = errno;
    errno = 0;
    int64_t set_flags = kernel_fcntl_fn && fd >= 0
        ? (int64_t)kernel_fcntl_fn((uint64_t)fd, kGuestFSetFl,
                                   kGuestONonblock | kGuestOAppend, 0, 0, 0)
        : 0;
    CHECK(guest_flags == -1 && getfl_errno == kFreeBsdENotSupp &&
              set_flags == -1 && errno == kFreeBsdENotSupp,
          "Windows fcntl status commands fail explicitly with FreeBSD's ENOTSUP, not the host's");
#endif

    int64_t set_fd_flags = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, kGuestFSetFd, kGuestFdCloExec, 0, 0, 0)
        : -1;
    uint64_t fd_flags = fcntl_fn && fd >= 0
        ? fcntl_fn((uint64_t)fd, kGuestFGetFd, 0, 0, 0, 0)
        : ~uint64_t{0};
    CHECK(set_fd_flags == 0 && fd_flags == kGuestFdCloExec,
          "fcntl translates the close-on-exec descriptor flag");

    int64_t fcntl_duplicate = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, kGuestFDupFd, 8, 0, 0, 0)
        : -1;
    CHECK(fcntl_duplicate >= 8, "fcntl F_DUPFD honors the guest minimum descriptor");
    if (fcntl_duplicate >= 0 && close_fn)
        close_fn((uint64_t)fcntl_duplicate, 0, 0, 0, 0, 0);

    errno = 0;
    int64_t unsupported_fcntl = fcntl_fn && fd >= 0
        ? (int64_t)fcntl_fn((uint64_t)fd, 0x7fffffff, 0, 0, 0, 0)
        : 0;
    CHECK(unsupported_fcntl == -1 && errno == EINVAL,
          "fcntl rejects unsupported guest commands instead of returning false success");

    // libc and sceKernel expose the same enumeration operation with different error conventions.
    // The old shared handler returned a positive-looking SCE error to libc; Windows returned false
    // EOF for every call. Exercise both invalid arguments and a valid non-directory descriptor.
    int64_t base = 0x12345678;
    errno = 0;
    int64_t libc_bad_buffer = getdents_fn
        ? (int64_t)getdents_fn((uint64_t)fd, 0, dir_buffer.size(), 0, 0, 0)
        : 0;
    int libc_bad_buffer_errno = errno;
    uint64_t kernel_bad_buffer = kernel_getdents_fn
        ? kernel_getdents_fn((uint64_t)fd, 0, dir_buffer.size(), 0, 0, 0)
        : 0;
    CHECK(libc_bad_buffer == -1 && libc_bad_buffer_errno == EINVAL,
          "libc getdents returns -1 with EINVAL for an invalid buffer");
    CHECK((uint32_t)kernel_bad_buffer == 0x80020016u,
          "sceKernelGetdents returns SCE_KERNEL_ERROR_EINVAL directly");

    errno = 0;
    int64_t libc_not_directory = getdirentries_fn
        ? (int64_t)getdirentries_fn((uint64_t)fd, (uint64_t)(uintptr_t)dir_buffer.data(),
                                   dir_buffer.size(), (uint64_t)(uintptr_t)&base, 0, 0)
        : 0;
    int libc_not_directory_errno = errno;
    uint64_t kernel_not_directory = kernel_getdirentries_fn
        ? kernel_getdirentries_fn((uint64_t)fd, (uint64_t)(uintptr_t)dir_buffer.data(),
                                  dir_buffer.size(), (uint64_t)(uintptr_t)&base, 0, 0)
        : 0;
    CHECK(libc_not_directory == -1 && libc_not_directory_errno == ENOTDIR,
          "libc getdirentries returns -1 with ENOTDIR for a regular file");
    CHECK((uint32_t)kernel_not_directory == 0x80020014u,
          "sceKernelGetdirentries returns SCE_KERNEL_ERROR_ENOTDIR directly");
    CHECK(base == 0x12345678,
          "failed getdirentries calls leave basep untouched");

    // A failure must be determined before touching basep. Address 1 is deliberately inaccessible;
    // this also exercises Windows' guarded _get_osfhandle invalid-descriptor path.
    errno = 0;
    int64_t libc_bad_fd = getdirentries_fn
        ? (int64_t)getdirentries_fn(~uint64_t{0}, (uint64_t)(uintptr_t)dir_buffer.data(),
                                   dir_buffer.size(), 1, 0, 0)
        : 0;
    int libc_bad_fd_errno = errno;
    uint64_t kernel_bad_fd = kernel_getdirentries_fn
        ? kernel_getdirentries_fn(~uint64_t{0}, (uint64_t)(uintptr_t)dir_buffer.data(),
                                  dir_buffer.size(), 1, 0, 0)
        : 0;
    CHECK(libc_bad_fd == -1 && libc_bad_fd_errno == EBADF,
          "libc getdirentries rejects an invalid descriptor without touching basep");
    CHECK((uint32_t)kernel_bad_fd == 0x80020009u,
          "sceKernelGetdirentries returns EBADF without touching basep");

    int64_t n = fd >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)fd, (uint64_t)(uintptr_t)actual.data(), actual.size(), 0, 0, 0)
        : -1;
    CHECK(n == (int64_t)expected.size(), "read continues through embedded 0x1a");
    CHECK(n == (int64_t)expected.size() && actual == expected,
          "read preserves binary bytes including CRLF");
    if (fd >= 0 && close_fn) close_fn((uint64_t)fd, 0, 0, 0, 0, 0);

    // libc descriptor I/O reports -1 and errno; the sceKernel siblings return the translated
    // error directly in their signed 64-bit result. A closed descriptor exercises the same EBADF
    // contract deterministically on Linux and the secondary Windows host without touching buffers.
    const uint64_t closed_io_fd = (uint64_t)fd;
    std::array<uint8_t, 1> io_byte{{0x5a}};
    auto check_bad_fd_contract = [&](const char* operation, HleFn libc_fn, HleFn kernel_fn,
                                     uint64_t arg1, uint64_t arg2, uint64_t arg3) {
        errno = 0;
        uint64_t libc_result = libc_fn
            ? libc_fn(closed_io_fd, arg1, arg2, arg3, 0, 0)
            : 0;
        const int libc_error = errno;
        uint64_t kernel_result = kernel_fn
            ? kernel_fn(closed_io_fd, arg1, arg2, arg3, 0, 0)
            : 0;
        const std::string libc_message = std::string("libc ") + operation +
                                         " retains -1 plus EBADF";
        const std::string kernel_message = std::string("sceKernel") + operation +
                                           " returns sign-extended SCE_KERNEL_ERROR_EBADF";
        CHECK((int64_t)libc_result == -1 && libc_error == EBADF, libc_message.c_str());
        CHECK(kernel_result == 0xffffffff80020009ull, kernel_message.c_str());
    };
    check_bad_fd_contract("Read", posix_read_fn, read_fn,
                          (uint64_t)(uintptr_t)io_byte.data(), io_byte.size(), 0);
    check_bad_fd_contract("Write", posix_write_fn, write_fn,
                          (uint64_t)(uintptr_t)io_byte.data(), io_byte.size(), 0);
    check_bad_fd_contract("Pread", posix_pread_fn, pread_fn,
                          (uint64_t)(uintptr_t)io_byte.data(), io_byte.size(), 0);
    check_bad_fd_contract("Pwrite", posix_pwrite_fn, pwrite_fn,
                          (uint64_t)(uintptr_t)io_byte.data(), io_byte.size(), 0);
    check_bad_fd_contract("Lseek", posix_lseek_fn, lseek_fn, 0, SEEK_SET, 0);
    struct GuestIovec { void* base; size_t len; };
    GuestIovec io_vector{io_byte.data(), io_byte.size()};
    check_bad_fd_contract("Readv", posix_readv_fn, readv_fn,
                          (uint64_t)(uintptr_t)&io_vector, 1, 0);
    check_bad_fd_contract("Writev", posix_writev_fn, writev_fn,
                          (uint64_t)(uintptr_t)&io_vector, 1, 0);
    check_bad_fd_contract("Preadv", posix_preadv_fn, preadv_fn,
                          (uint64_t)(uintptr_t)&io_vector, 1, 0);
    check_bad_fd_contract("Pwritev", posix_pwritev_fn, pwritev_fn,
                          (uint64_t)(uintptr_t)&io_vector, 1, 0);
    errno = 0;
    int64_t libc_close_result = posix_close_fn
        ? (int64_t)posix_close_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    const int libc_close_error = errno;
    uint64_t kernel_close_result = close_fn
        ? close_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    CHECK(libc_close_result == -1 && libc_close_error == EBADF,
          "libc close retains -1 plus EBADF");
    CHECK(kernel_close_result == 0x80020009u,
          "sceKernelClose returns SCE_KERNEL_ERROR_EBADF directly");
    CHECK(close_fn && close_fn(0, 0, 0, 0, 0, 0) == 0,
          "sceKernelClose preserves reserved stdio descriptors as no-op success");

    std::array<uint8_t, 0x78> libc_fstat_buffer;
    std::array<uint8_t, 0x78> kernel_fstat_buffer;
    libc_fstat_buffer.fill(0xa5);
    kernel_fstat_buffer.fill(0xa5);
    const auto untouched_fstat = libc_fstat_buffer;
    errno = 0;
    int64_t libc_fstat_result = fstat_fn
        ? (int64_t)fstat_fn(closed_io_fd, (uint64_t)(uintptr_t)libc_fstat_buffer.data(),
                            0, 0, 0, 0)
        : 0;
    const int libc_fstat_error = errno;
    uint64_t kernel_fstat_result = kernel_fstat_fn
        ? kernel_fstat_fn(closed_io_fd, (uint64_t)(uintptr_t)kernel_fstat_buffer.data(),
                          0, 0, 0, 0)
        : 0;
    CHECK(libc_fstat_result == -1 && libc_fstat_error == EBADF,
          "libc fstat retains -1 plus EBADF");
    CHECK(kernel_fstat_result == 0x80020009u,
          "sceKernelFstat returns SCE_KERNEL_ERROR_EBADF directly");
    CHECK(libc_fstat_buffer == untouched_fstat && kernel_fstat_buffer == untouched_fstat,
          "fstat failure leaves stat buffers untouched");

    errno = 0;
    int64_t libc_fsync_result = fsync_fn
        ? (int64_t)fsync_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    const int libc_fsync_error = errno;
    uint64_t kernel_fsync_failure = kernel_fsync_fn
        ? kernel_fsync_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    errno = 0;
    int64_t libc_fdatasync_result = fdatasync_fn
        ? (int64_t)fdatasync_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    const int libc_fdatasync_error = errno;
    uint64_t kernel_fdatasync_failure = kernel_fdatasync_fn
        ? kernel_fdatasync_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    uint64_t kernel_ftruncate_failure = kernel_ftruncate_fn
        ? kernel_ftruncate_fn(closed_io_fd, 0, 0, 0, 0, 0)
        : 0;
    CHECK(libc_fsync_result == -1 && libc_fsync_error == EBADF,
          "libc fsync retains -1 plus EBADF");
    CHECK(kernel_fsync_failure == 0x80020009u,
          "sceKernelFsync returns SCE_KERNEL_ERROR_EBADF directly");
    CHECK(libc_fdatasync_result == -1 && libc_fdatasync_error == EBADF,
          "libc fdatasync retains -1 plus EBADF");
    CHECK(kernel_fdatasync_failure == 0x80020009u,
          "sceKernelFdatasync returns SCE_KERNEL_ERROR_EBADF directly");
    CHECK(kernel_ftruncate_failure == 0x80020009u,
          "sceKernelFtruncate returns SCE_KERNEL_ERROR_EBADF directly");

    // A duplicate is a distinct descriptor for the same open file description: it shares the
    // current offset and remains usable after the original descriptor is closed.
    int64_t dup_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t duplicated = dup_source >= 0 && kernel_dup_fn
        ? (int64_t)kernel_dup_fn((uint64_t)dup_source, 0, 0, 0, 0, 0)
        : -1;
    CHECK(duplicated >= 3 && duplicated != dup_source,
          "sceKernelDup returns a distinct non-stdio descriptor");
    std::array<uint8_t, 16> first_chunk{};
    std::array<uint8_t, 16> second_chunk{};
    int64_t first_n = dup_source >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)dup_source,
                           (uint64_t)(uintptr_t)first_chunk.data(), first_chunk.size(), 0, 0, 0)
        : -1;
    int64_t second_n = duplicated >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)duplicated,
                           (uint64_t)(uintptr_t)second_chunk.data(), second_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(first_n == (int64_t)first_chunk.size() &&
              std::memcmp(first_chunk.data(), expected.data(), first_chunk.size()) == 0,
          "original descriptor reads the first chunk");
    CHECK(second_n == (int64_t)second_chunk.size() &&
              std::memcmp(second_chunk.data(), expected.data() + first_chunk.size(),
                          second_chunk.size()) == 0,
          "sceKernelDup shares the original file offset");
    if (dup_source >= 0 && close_fn) close_fn((uint64_t)dup_source, 0, 0, 0, 0, 0);
    int64_t duplicate_seek = duplicated >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)duplicated, 0, SEEK_SET, 0, 0, 0)
        : -1;
    CHECK(duplicate_seek == 0, "duplicate remains usable after closing the original");
    if (duplicated >= 0 && close_fn) close_fn((uint64_t)duplicated, 0, 0, 0, 0, 0);

#ifdef _WIN32
    // Force fd 0 free while duplicating an already-open guest file. _dup chooses the lowest free
    // CRT descriptor, so the wrapper must hold that temporary result and return one above stdio.
    int saved_stdin = ::_dup(0);
    int stdin_filler = -1;
    if (saved_stdin < 0) stdin_filler = ::_open("NUL", _O_RDONLY | _O_BINARY);
    CHECK(saved_stdin >= 0 || stdin_filler == 0, "occupy fd 0 before low-slot dup test");
    int64_t low_slot_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_source >= 3, "open dup source outside the stdio range");
    CHECK(::_close(0) == 0, "free fd 0 for deterministic dup reuse");
    int64_t low_slot_duplicate = low_slot_source >= 0 && kernel_dup_fn
        ? (int64_t)kernel_dup_fn((uint64_t)low_slot_source, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_duplicate >= 3,
          "sceKernelDup lifts a recycled Windows stdio descriptor");
    if (low_slot_duplicate >= 0 && close_fn)
        close_fn((uint64_t)low_slot_duplicate, 0, 0, 0, 0, 0);
    if (low_slot_source >= 0 && close_fn)
        close_fn((uint64_t)low_slot_source, 0, 0, 0, 0, 0);

    // With fd 0 still free, the Windows CRT will allocate it for the next open. The HLE must move
    // that live file into the guest-visible range before returning it, without losing its contents.
    int64_t low_slot_open = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_open >= 3, "sceKernelOpen lifts a recycled Windows stdio descriptor");
    std::array<uint8_t, 16> low_slot_open_chunk{};
    int64_t low_slot_open_n = low_slot_open >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)low_slot_open,
                           (uint64_t)(uintptr_t)low_slot_open_chunk.data(),
                           low_slot_open_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(low_slot_open_n == (int64_t)low_slot_open_chunk.size() &&
              std::memcmp(low_slot_open_chunk.data(), expected.data(),
                          low_slot_open_chunk.size()) == 0,
          "lifted sceKernelOpen descriptor reads the original file");
    if (low_slot_open >= 3 && close_fn)
        close_fn((uint64_t)low_slot_open, 0, 0, 0, 0, 0);
    else if (low_slot_open >= 0)
        ::_close((int)low_slot_open);
    if (saved_stdin >= 0) {
        CHECK(::_dup2(saved_stdin, 0) == 0, "restore fd 0 after low-slot descriptor tests");
        ::_close(saved_stdin);
    }
#endif

    // dup2 must replace an already-open target, share the source offset, and return the target
    // descriptor. The Windows CRT returns zero on success, so this catches a missing ABI translation.
    int64_t dup2_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t dup2_target = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t source_offset = 73;
    constexpr int64_t old_target_offset = 211;
    int64_t source_seek = dup2_source >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)dup2_source, source_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t target_seek = dup2_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)dup2_target, old_target_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t dup2_result = dup2_source >= 0 && dup2_target >= 0 && kernel_dup2_fn
        ? (int64_t)kernel_dup2_fn((uint64_t)dup2_source, (uint64_t)dup2_target, 0, 0, 0, 0)
        : -1;
    CHECK(source_seek == source_offset && target_seek == old_target_offset,
          "position dup2 source and old target independently");
    CHECK(dup2_result == dup2_target, "sceKernelDup2 returns the replaced target descriptor");
    if (dup2_source >= 0 && close_fn) close_fn((uint64_t)dup2_source, 0, 0, 0, 0, 0);
    std::array<uint8_t, 16> dup2_chunk{};
    int64_t dup2_n = dup2_target >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)dup2_target,
                           (uint64_t)(uintptr_t)dup2_chunk.data(), dup2_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(dup2_n == (int64_t)dup2_chunk.size() &&
              std::memcmp(dup2_chunk.data(), expected.data() + source_offset,
                          dup2_chunk.size()) == 0,
          "sceKernelDup2 replaces the target and shares the source offset");
    if (dup2_target >= 0 && close_fn) close_fn((uint64_t)dup2_target, 0, 0, 0, 0, 0);

#ifdef _WIN32
    // UCRT reports invalid descriptors through its invalid-parameter handler before returning -1.
    // Guest inputs must not reach the default terminating handler, and a failed dup2 must not close
    // or replace its valid target.
    int64_t invalid_duplicate = kernel_dup_fn
        ? (int64_t)kernel_dup_fn(~uint64_t{0}, 0, 0, 0, 0, 0)
        : 0;
    CHECK(invalid_duplicate == -1, "sceKernelDup safely rejects an invalid source descriptor");
    int64_t preserved_target = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t preserved_offset = 137;
    int64_t preserved_seek = preserved_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)preserved_target, preserved_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t invalid_dup2 = preserved_target >= 0 && kernel_dup2_fn
        ? (int64_t)kernel_dup2_fn(~uint64_t{0}, (uint64_t)preserved_target, 0, 0, 0, 0)
        : 0;
    int64_t preserved_position = preserved_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)preserved_target, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    std::array<uint8_t, 16> preserved_chunk{};
    int64_t preserved_n = preserved_target >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)preserved_target,
                           (uint64_t)(uintptr_t)preserved_chunk.data(), preserved_chunk.size(),
                           0, 0, 0)
        : -1;
    CHECK(preserved_seek == preserved_offset, "position the invalid-dup2 target");
    CHECK(invalid_dup2 == -1, "sceKernelDup2 safely rejects an invalid source descriptor");
    CHECK(preserved_position == preserved_offset &&
              preserved_n == (int64_t)preserved_chunk.size() &&
              std::memcmp(preserved_chunk.data(), expected.data() + preserved_offset,
                          preserved_chunk.size()) == 0,
          "failed sceKernelDup2 leaves the target descriptor unchanged");
    if (preserved_target >= 0 && close_fn)
        close_fn((uint64_t)preserved_target, 0, 0, 0, 0, 0);
#endif

#ifdef _WIN32
    // Unity asks for a whole 64 KiB cache block even when boot.config is only a few hundred bytes.
    // Guest allocators may leave later pages reserved until first touch. Windows _read validates the
    // entire requested destination range before discovering EOF, so a direct host read fails EINVAL;
    // PS5/Linux instead copy the available prefix and return its short byte count.
    void* sparse = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(sparse != nullptr, "reserve sparse guest-style read buffer");
    void* committed = sparse ? VirtualAlloc(sparse, 0x4000, MEM_COMMIT, PAGE_READWRITE) : nullptr;
    CHECK(committed == sparse, "commit only the first guest page");
    int64_t sparse_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(sparse_fd >= 0, "reopen fixture for sparse-buffer short read");
    int64_t sparse_n = sparse_fd >= 0 && read_fn && sparse
        ? (int64_t)read_fn((uint64_t)sparse_fd, (uint64_t)(uintptr_t)sparse, 0x10000, 0, 0, 0)
        : -1;
    CHECK(sparse_n == (int64_t)expected.size(),
          "short read does not probe reserved destination pages past EOF");
    CHECK(sparse_n == (int64_t)expected.size() &&
              std::memcmp(sparse, expected.data(), expected.size()) == 0,
          "short read preserves the available file prefix");
    if (sparse_fd >= 0 && close_fn) close_fn((uint64_t)sparse_fd, 0, 0, 0, 0, 0);
    if (sparse) VirtualFree(sparse, 0, MEM_RELEASE);

    // Destination validation must happen before a sequential read advances the shared file offset.
    // Repeating the reviewer's invalid-first-chunk probe verifies both EFAULT and retry position.
    void* invalid_first = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(invalid_first != nullptr, "reserve inaccessible invalid-read destination");
    int64_t invalid_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t invalid_n = invalid_fd >= 0 && read_fn && invalid_first
        ? (int64_t)read_fn((uint64_t)invalid_fd, (uint64_t)(uintptr_t)invalid_first,
                           100, 0, 0, 0)
        : 0;
    int64_t invalid_pos = invalid_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)invalid_fd, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    CHECK((uint64_t)invalid_n == 0xffffffff8002000eull,
          "invalid first destination chunk returns SCE_KERNEL_ERROR_EFAULT");
    CHECK(invalid_pos == 0, "failed first destination chunk consumes no file bytes");
    if (invalid_fd >= 0 && close_fn) close_fn((uint64_t)invalid_fd, 0, 0, 0, 0, 0);
    if (invalid_first) VirtualFree(invalid_first, 0, MEM_RELEASE);

    // The same rule applies after earlier chunks were delivered. The fixture is one full bounce
    // chunk plus 512 bytes; only the committed first chunk may be consumed/reported.
    const char* large_path = "prosper-test-file-binary-large.tmp";
    FILE* large_out = std::fopen(large_path, "wb");
    bool large_written = large_out != nullptr;
    if (large_out) {
        for (int block = 0; block < 129 && large_written; ++block)
            large_written = std::fwrite(expected.data(), 1, expected.size(), large_out) ==
                            expected.size();
        std::fclose(large_out);
    }
    CHECK(large_written, "create partial-copy offset fixture");
    void* partial = VirtualAlloc(nullptr, 0x20000, MEM_RESERVE, PAGE_NOACCESS);
    void* partial_committed = partial
        ? VirtualAlloc(partial, 0x10000, MEM_COMMIT, PAGE_READWRITE) : nullptr;
    CHECK(partial && partial_committed == partial,
          "commit only first chunk of partial-copy destination");
    int64_t partial_fd = open_fn && large_written
        ? (int64_t)open_fn((uint64_t)(uintptr_t)large_path, 0, 0, 0, 0, 0)
        : -1;
    int64_t partial_n = partial_fd >= 0 && read_fn && partial
        ? (int64_t)read_fn((uint64_t)partial_fd, (uint64_t)(uintptr_t)partial,
                           0x20000, 0, 0, 0)
        : -1;
    int64_t partial_pos = partial_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)partial_fd, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    CHECK(partial_n == 0x10000, "partial read reports only the delivered first chunk");
    CHECK(partial_pos == 0x10000, "failed later destination chunk consumes no extra bytes");
    CHECK(partial_n == 0x10000 &&
              std::memcmp(partial, expected.data(), expected.size()) == 0,
          "partial read preserves delivered prefix bytes");
    if (partial_fd >= 0 && close_fn) close_fn((uint64_t)partial_fd, 0, 0, 0, 0, 0);
    if (partial) VirtualFree(partial, 0, MEM_RELEASE);
    std::remove(large_path);

    // A read whose current position is already past EOF must return zero without validating or
    // touching any byte of the requested destination range. Keep the entire range reserved and
    // inaccessible so a host read that probes it reproduces the Windows CRT failure deterministically.
    void* past_eof = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(past_eof != nullptr, "reserve inaccessible past-EOF destination");
    int64_t past_eof_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t past_eof_offset = 4096;
    int64_t seek_result = past_eof_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)past_eof_fd, (uint64_t)past_eof_offset,
                            SEEK_SET, 0, 0, 0)
        : -1;
    CHECK(seek_result == past_eof_offset, "seek beyond fixture EOF");
    int64_t past_eof_n = past_eof_fd >= 0 && read_fn && past_eof
        ? (int64_t)read_fn((uint64_t)past_eof_fd, (uint64_t)(uintptr_t)past_eof,
                           0x10000, 0, 0, 0)
        : -1;
    CHECK(past_eof_n == 0, "read past EOF returns zero without probing destination");
    MEMORY_BASIC_INFORMATION past_eof_info{};
    CHECK(past_eof && VirtualQuery(past_eof, &past_eof_info, sizeof past_eof_info) &&
              past_eof_info.State == MEM_RESERVE,
          "past-EOF destination remains uncommitted and untouched");
    if (past_eof_fd >= 0 && close_fn) close_fn((uint64_t)past_eof_fd, 0, 0, 0, 0, 0);
    if (past_eof) VirtualFree(past_eof, 0, MEM_RELEASE);

    // Tracked guest virtual/direct memory is also committed on first touch. The bounce-buffer copy
    // must take the emulator's lazy-commit path before publishing data into such a destination.
    register_kernel_mem_hle();
    HleFn reserve_fn = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    HleFn unmap_fn = Hle::lookup(nid_hash("sceKernelMunmap"));
    uint64_t reserved = 0;
    CHECK(reserve_fn && unmap_fn, "guest virtual-memory HLE functions registered");
    uint64_t reserve_result = reserve_fn
        ? reserve_fn((uint64_t)(uintptr_t)&reserved, 0x8000, 0, 0x4000, 0, 0)
        : ~uint64_t{0};
    CHECK(reserve_result == 0 && reserved != 0, "reserve untouched guest read destination");
    if (reserved) {
        fd = open_fn ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0) : -1;
        n = fd >= 0 && read_fn
            ? (int64_t)read_fn((uint64_t)fd, reserved, expected.size(), 0, 0, 0)
            : -1;
        CHECK(n == (int64_t)expected.size(), "read materializes reserved guest pages");
        CHECK(n == (int64_t)expected.size() &&
                  std::memcmp((const void*)(uintptr_t)reserved, expected.data(), expected.size()) == 0,
              "materialized guest destination receives every byte");
        if (fd >= 0 && close_fn) close_fn((uint64_t)fd, 0, 0, 0, 0, 0);
        if (unmap_fn) unmap_fn(reserved, 0x8000, 0, 0, 0, 0);
    }
#endif
    std::remove(path);

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

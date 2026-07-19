// hle_file.cpp — HLE file I/O. Guest paths like "/app0/..." (the game's own data) are
// translated to the host dump directory; stdio FILE* and POSIX fd calls thunk to the
// host. Set the app0 root via set_app0_root() or the PROSPER_APP0 env var.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // pthread_getattr_np (bound the PREADLOG/DEEPTRACE stack walk to the real stack)
#endif
#include "dispatch.hpp"
#include "nid.hpp"
#include "heap_mutex.hpp"   // #707: keep the APR mutex off macOS __DATA
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <string>
#include <mutex>
#include <atomic>        // sceKernelAio* submit-id/state table
#include <map>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
#include <dirent.h>      // opendir/readdir: savedata0_list_dirs (sceSaveDataDirNameSearch, #299)
#include <sys/uio.h>     // process_vm_readv: fault-safe guest-memory reads for the APR diagnostics
#include <sys/mman.h>    // PROSPER_APR_ALTDEST: prosper-owned guest read buffer
#include <pthread.h>
#include <thread>        // PROSPER_APR_DEFER: async (real-hardware-timing) APR fill
#else
#include <direct.h>
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // ReadFile/WriteFile + OVERLAPPED for positioned IO (pread/pwrite equivalents)
#endif
#include "../host/posix_shim.hpp"   // Darwin: process_vm_*, pthread_getattr_np, st_*tim, prosper_mincore

#if defined(_WIN32) && defined(__MINGW32__)
// MinGW's UCRT import library exposes this API, but its current headers omit the declaration.
extern "C" _invalid_parameter_handler __cdecl
_set_thread_local_invalid_parameter_handler(_invalid_parameter_handler);
#endif

namespace prosper {

#ifdef _WIN32
extern "C" int prosper_reserved_range_state(uint64_t addr);
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);
extern "C" int prosper_try_commit_reserved_placeholder(uint64_t addr, uint64_t len);
#endif

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))
#define CS(x) ((const char*)(uintptr_t)(x))

namespace {
    std::string g_app0;   // host directory backing guest "/app0"
    bool filelog() { static int v = getenv("PROSPER_FILELOG") ? 1 : 0; return v; }
    std::mutex g_filelog_fd_mx;
    std::map<int, std::string> g_filelog_fd_paths;

#ifdef _WIN32
    // Return the writable prefix of a guest destination, materializing sparse direct-memory or
    // tracked reserved pages as the guest VEH would on first touch.  Host I/O bypasses that VEH, so
    // every byte handed to _read/ReadFile must already be committed and writable.  Reporting a
    // prefix (rather than a boolean) is important for sequential reads: a valid committed prefix may
    // precede an inaccessible tail, and the file offset must advance only by bytes actually delivered.
    uint64_t windows_prepare_guest_write_prefix(uint64_t dst, uint64_t bytes) {
        if (!bytes) return 0;
        if (!dst || dst + bytes < dst) return 0;
        prosper_try_commit_dmem(dst, bytes, 1);
        const uint64_t end = dst + bytes;
        for (uint64_t p = dst; p < end;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery((const void*)(uintptr_t)p, &mbi, sizeof mbi)) return p - dst;
            if (mbi.State == MEM_RESERVE && prosper_reserved_range_state(p) == 1) {
                const uint64_t page = p & ~uint64_t{0x3fff};
                if (!prosper_try_commit_reserved_placeholder(page, 0x4000) &&
                    !VirtualAlloc((void*)(uintptr_t)page, 0x4000,
                                  MEM_COMMIT, PAGE_READWRITE))
                    return p - dst;
                continue;
            }
            const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY |
                                   PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
            if (mbi.State != MEM_COMMIT || (mbi.Protect & PAGE_GUARD) ||
                !(mbi.Protect & writable)) return p - dst;
            const uint64_t region_end = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
            if (region_end <= p) return p - dst;
            p = region_end < end ? region_end : end;
        }
        return bytes;
    }

    bool windows_prepare_guest_write(uint64_t dst, uint64_t bytes) {
        return !bytes || windows_prepare_guest_write_prefix(dst, bytes) == bytes;
    }

    void __cdecl ignore_crt_invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*,
                                               unsigned int, uintptr_t) {}

    class ScopedCrtInvalidParameterHandler {
    public:
        ScopedCrtInvalidParameterHandler()
            : previous_(_set_thread_local_invalid_parameter_handler(
                  ignore_crt_invalid_parameter)) {}
        ~ScopedCrtInvalidParameterHandler() {
            _set_thread_local_invalid_parameter_handler(previous_);
        }
        ScopedCrtInvalidParameterHandler(const ScopedCrtInvalidParameterHandler&) = delete;
        ScopedCrtInvalidParameterHandler& operator=(const ScopedCrtInvalidParameterHandler&) = delete;

    private:
        _invalid_parameter_handler previous_;
    };

    int windows_duplicate_at_least(int source_fd, int minimum_fd) {
        if (minimum_fd < 0) {
            errno = EINVAL;
            return -1;
        }
        ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
        int fd = ::_dup(source_fd);
        std::vector<int> low_fds;
        // Windows has no F_DUPFD. Keep lower recycled slots occupied until _dup reaches the
        // requested guest-visible range, then release the temporary descriptors.
        while (fd >= 0 && fd < minimum_fd) {
            low_fds.push_back(fd);
            fd = ::_dup(source_fd);
        }
        int duplicate_errno = fd < 0 ? errno : 0;
        for (int low_fd : low_fds) ::_close(low_fd);
        if (fd < 0) errno = duplicate_errno;
        return fd;
    }

    int windows_duplicate_above_stdio(int source_fd) {
        return windows_duplicate_at_least(source_fd, 3);
    }

    // The Windows CRT refuses to open directories as file descriptors. PS5 guests nevertheless
    // use the normal open/getdents/close sequence, so represent directory opens with emulator-owned
    // descriptors and a per-open cursor. Enumerating eagerly gives each descriptor an immutable
    // snapshot and keeps getdents deterministic and thread-safe without retaining Win32 search
    // handles. The descriptor range is far above UCRT's file table and remains positive to guests.
    struct WindowsDirEntry {
        std::string name;
        uint32_t ino = 0;
        uint8_t type = 0;
    };
    struct WindowsDirState {
        std::string path;
        std::vector<WindowsDirEntry> entries;
        size_t cursor = 0;
    };
    std::mutex g_windows_dir_mx;
    std::map<int, WindowsDirState> g_windows_dirs;
    std::atomic<int> g_windows_next_dir_fd{0x40000000};

    bool windows_host_path_is_directory(const std::string& path) {
        const DWORD attrs = GetFileAttributesA(path.c_str());
        return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }

    uint32_t windows_dir_ino(const std::string& name) {
        uint32_t hash = 2166136261u;
        for (unsigned char c : name) hash = (hash ^ c) * 16777619u;
        return hash ? hash : 1;
    }

    int windows_directory_errno(DWORD error) {
        switch (error) {
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:
        case ERROR_INVALID_NAME:
        case ERROR_DIRECTORY:
            return ENOENT;
        case ERROR_ACCESS_DENIED:
        case ERROR_SHARING_VIOLATION:
            return EACCES;
        case ERROR_NOT_ENOUGH_MEMORY:
        case ERROR_OUTOFMEMORY:
            return ENOMEM;
        default:
            return EIO;
        }
    }

    int windows_open_directory(const std::string& path) {
        WindowsDirState state;
        state.path = path;
        std::string pattern = path;
        if (!pattern.empty() && pattern.back() != '/' && pattern.back() != '\\') pattern += '\\';
        pattern += '*';

        WIN32_FIND_DATAA data{};
        HANDLE search = FindFirstFileA(pattern.c_str(), &data);
        if (search != INVALID_HANDLE_VALUE) {
            DWORD enumeration_error = ERROR_SUCCESS;
            for (;;) {
                WindowsDirEntry entry;
                entry.name = data.cFileName;
                entry.ino = windows_dir_ino(entry.name);
                entry.type = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 4 : 8;
                state.entries.push_back(std::move(entry));
                if (FindNextFileA(search, &data)) continue;
                enumeration_error = GetLastError();
                break;
            }
            FindClose(search);
            if (enumeration_error != ERROR_NO_MORE_FILES) {
                errno = windows_directory_errno(enumeration_error);
                return -1;
            }
        } else {
            const DWORD enumeration_error = GetLastError();
            // With a directory already established by GetFileAttributesA, no wildcard match is
            // the normal representation of an empty directory. Every other failure is real.
            if (enumeration_error != ERROR_FILE_NOT_FOUND) {
                errno = windows_directory_errno(enumeration_error);
                return -1;
            }
        }

        std::lock_guard<std::mutex> lk(g_windows_dir_mx);
        int fd = g_windows_next_dir_fd.fetch_add(1);
        while (fd < 3 || g_windows_dirs.count(fd)) fd = g_windows_next_dir_fd.fetch_add(1);
        g_windows_dirs.emplace(fd, std::move(state));
        return fd;
    }

    bool windows_close_directory(int fd, int* result) {
        std::lock_guard<std::mutex> lk(g_windows_dir_mx);
        auto it = g_windows_dirs.find(fd);
        if (it == g_windows_dirs.end()) return false;
        g_windows_dirs.erase(it);
        if (result) *result = 0;
        return true;
    }

    bool windows_directory_path(int fd, std::string* path) {
        std::lock_guard<std::mutex> lk(g_windows_dir_mx);
        auto it = g_windows_dirs.find(fd);
        if (it == g_windows_dirs.end()) return false;
        if (path) *path = it->second.path;
        return true;
    }

    bool windows_seek_directory(int fd, int64_t offset, int whence, int64_t* result) {
        std::lock_guard<std::mutex> lk(g_windows_dir_mx);
        auto it = g_windows_dirs.find(fd);
        if (it == g_windows_dirs.end()) return false;
        int64_t base = whence == SEEK_SET ? 0
                     : whence == SEEK_CUR ? (int64_t)it->second.cursor
                     : whence == SEEK_END ? (int64_t)it->second.entries.size() : -1;
        if (base < 0 || offset < -base || offset > (int64_t)it->second.entries.size() - base) {
            errno = EINVAL;
            if (result) *result = -1;
            return true;
        }
        it->second.cursor = (size_t)(base + offset);
        if (result) *result = (int64_t)it->second.cursor;
        return true;
    }

    uint64_t windows_getdents(int fd, uint64_t guest_buffer, uint64_t capacity) {
        if (!guest_buffer || capacity < 32) return 0x80020016ull;
        if (!windows_prepare_guest_write(guest_buffer, capacity)) return 0x8002000eull;
        std::lock_guard<std::mutex> lk(g_windows_dir_mx);
        auto it = g_windows_dirs.find(fd);
        if (it == g_windows_dirs.end()) return 0x80020009ull;

        uint8_t* out = (uint8_t*)(uintptr_t)guest_buffer;
        size_t written = 0;
        while (it->second.cursor < it->second.entries.size()) {
            const WindowsDirEntry& entry = it->second.entries[it->second.cursor];
            const size_t name_len = entry.name.size();
            const size_t record_size = (8 + name_len + 1 + 3) & ~(size_t)3;
            if (written + record_size > capacity) break;
            uint8_t* record = out + written;
            memset(record, 0, record_size);
            *(uint32_t*)(record + 0) = entry.ino;
            *(uint16_t*)(record + 4) = (uint16_t)record_size;
            record[6] = entry.type;
            record[7] = (uint8_t)name_len;
            memcpy(record + 8, entry.name.c_str(), name_len + 1);
            written += record_size;
            it->second.cursor++;
        }
        return written;
    }
#endif

    void filelog_remember_fd(int fd, const std::string& path) {
        if (!filelog() || fd < 0) return;
        std::lock_guard<std::mutex> lk(g_filelog_fd_mx);
        g_filelog_fd_paths[fd] = path;
    }

    std::string filelog_fd_path(int fd) {
        if (!filelog()) return {};
        std::lock_guard<std::mutex> lk(g_filelog_fd_mx);
        auto it = g_filelog_fd_paths.find(fd);
        return it == g_filelog_fd_paths.end() ? std::string() : it->second;
    }

    void filelog_forget_fd(int fd) {
        if (!filelog()) return;
        std::lock_guard<std::mutex> lk(g_filelog_fd_mx);
        g_filelog_fd_paths.erase(fd);
    }

    void filelog_fd_io(const char* op, int fd, int64_t off, uint64_t count,
                       int64_t result, int err) {
        if (!filelog()) return;
        std::string path = filelog_fd_path(fd);
        fprintf(stderr,
                "[file] %s fd=%d path='%s' off=0x%llx count=0x%llx -> %lld error=%d\n",
                op, fd, path.empty() ? "?" : path.c_str(),
                (unsigned long long)off, (unsigned long long)count,
                (long long)result, err);
    }

    void filelog_fd_stat(int fd, int result, int err, int64_t size) {
        if (!filelog()) return;
        std::string path = filelog_fd_path(fd);
        fprintf(stderr, "[file] fstat fd=%d path='%s' -> %d size=%lld error=%d\n",
                fd, path.empty() ? "?" : path.c_str(), result, (long long)size, err);
    }
    // Host directory backing guest "/temp0" — the app temp-data area that
    // sceAppContentTemporaryDataMount2 (hle_service.cpp) reports to the game. Created on first
    // use; override with PROSPER_TEMP0. A scratch dir outside the dump keeps the game's writes
    // out of /app0 (which is read-only content).
    std::string temp0_root() {
        static std::string root;
        if (root.empty()) {
            const char* e = getenv("PROSPER_TEMP0");
            root = e ? e : "/tmp/prosper-temp0";
#ifdef _WIN32
            _mkdir(root.c_str());
#else
            ::mkdir(root.c_str(), 0777);
#endif
        }
        return root;
    }
    // Host directory backing guest "/savedata0" — the mounted save-data area that
    // sceSaveDataMount3 (hle_service.cpp) reports to the game. One save dir is mounted at a
    // time (DOLL's wrapper umounts id 0 before the next mount); the current mount's host dir
    // is swapped in here. Root override: PROSPER_SAVE0.
    std::mutex g_save0_mx;
    std::string g_save0;   // host dir for the CURRENT /savedata0 mount ("" = nothing mounted)
    std::string save0_base() {
        static std::string base;
        if (base.empty()) {
            const char* e = getenv("PROSPER_SAVE0");
            base = e ? e : "/tmp/prosper-savedata0";
#ifdef _WIN32
            _mkdir(base.c_str());
#else
            ::mkdir(base.c_str(), 0777);
#endif
        }
        return base;
    }
    // PROSPER_DENY_SUBSTR: comma-separated substrings; any guest path containing one is
    // redirected to a guaranteed-missing host path, so open/stat fail with ENOENT.
    // Diagnostic knob (off by default) — used to A/B whether the title's boot flow gates on a
    // file class we can't service yet (e.g. .usm movies through the stubbed CRI/AJM decoders).
    bool deny_path(const std::string& guest) {
        static std::vector<std::string> subs = [] {
            std::vector<std::string> v;
            if (const char* e = getenv("PROSPER_DENY_SUBSTR")) {
                std::string s = e; size_t pos = 0;
                while (pos <= s.size()) {
                    size_t c = s.find(',', pos);
                    if (c == std::string::npos) c = s.size();
                    if (c > pos) v.push_back(s.substr(pos, c - pos));
                    pos = c + 1;
                }
            }
            return v;
        }();
        if (subs.empty()) return false;
        for (const auto& sub : subs)
            if (guest.find(sub) != std::string::npos) return true;
        return false;
    }
    std::string translate(const char* guest) {
        if (!guest) return {};
        std::string p = guest;
        if (g_app0.empty()) { if (const char* e = getenv("PROSPER_APP0")) g_app0 = e; }
        if (deny_path(p)) {
            if (filelog()) fprintf(stderr, "[file] DENIED (PROSPER_DENY_SUBSTR) '%s'\n", guest);
            return "/prosper-denied" + p;
        }
        // Map /app0[/...] -> <root>[/...], /temp0[/...] -> scratch dir,
        // /savedata0[/...] -> the currently mounted save dir; other paths as-is.
        std::string save0;
        if (p.rfind("/savedata0", 0) == 0) { std::lock_guard<std::mutex> lk(g_save0_mx); save0 = g_save0; }
        std::string h = (p.rfind("/app0", 0) == 0)       ? g_app0 + p.substr(5)
                      : (p.rfind("/temp0", 0) == 0)      ? temp0_root() + p.substr(6)
                      : !save0.empty()                   ? save0 + p.substr(10)
                      : p;
        if (filelog()) fprintf(stderr, "[file] open '%s' -> '%s'\n", guest, h.c_str());
        return h;
    }

    struct HostStat {
        uint32_t dev = 0, ino = 0, mode = 0, nlink = 0, uid = 0, gid = 0, rdev = 0;
        int64_t atime_sec = 0, atime_nsec = 0;
        int64_t mtime_sec = 0, mtime_nsec = 0;
        int64_t ctime_sec = 0, ctime_nsec = 0;
        int64_t size = 0, blocks = 0;
        uint32_t blksize = 0;
    };

    int64_t blocks_512_for_size(int64_t size) {
        return size > 0 ? (size + 511) / 512 : 0;
    }

    void write_sce_stat(const HostStat& s, uint8_t* out) {
        memset(out, 0, 0x78);
        *(uint32_t*)(out + 0x00) = s.dev;
        *(uint32_t*)(out + 0x04) = s.ino;
        *(uint16_t*)(out + 0x08) = (uint16_t)s.mode;    // type+perm bits match Linux/MinGW
        *(uint16_t*)(out + 0x0a) = (uint16_t)s.nlink;
        *(uint32_t*)(out + 0x0c) = s.uid;
        *(uint32_t*)(out + 0x10) = s.gid;
        *(uint32_t*)(out + 0x14) = s.rdev;
        *(int64_t*)(out + 0x18)  = s.atime_sec; *(int64_t*)(out + 0x20) = s.atime_nsec;
        *(int64_t*)(out + 0x28)  = s.mtime_sec; *(int64_t*)(out + 0x30) = s.mtime_nsec;
        *(int64_t*)(out + 0x38)  = s.ctime_sec; *(int64_t*)(out + 0x40) = s.ctime_nsec;
        *(int64_t*)(out + 0x48)  = s.size;
        *(int64_t*)(out + 0x50)  = s.blocks;
        *(uint32_t*)(out + 0x58) = s.blksize;
    }

#ifndef _WIN32
    HostStat from_host_stat(const struct stat& s) {
        HostStat h;
        h.dev = (uint32_t)s.st_dev; h.ino = (uint32_t)s.st_ino; h.mode = (uint32_t)s.st_mode;
        h.nlink = (uint32_t)s.st_nlink; h.uid = (uint32_t)s.st_uid; h.gid = (uint32_t)s.st_gid;
        h.rdev = (uint32_t)s.st_rdev;
        h.atime_sec = s.st_atim.tv_sec; h.atime_nsec = s.st_atim.tv_nsec;
        h.mtime_sec = s.st_mtim.tv_sec; h.mtime_nsec = s.st_mtim.tv_nsec;
        h.ctime_sec = s.st_ctim.tv_sec; h.ctime_nsec = s.st_ctim.tv_nsec;
        h.size = (int64_t)s.st_size; h.blocks = (int64_t)s.st_blocks; h.blksize = (uint32_t)s.st_blksize;
        return h;
    }
#else
    HostStat from_host_stat(const struct stat& s) {
        HostStat h;
        h.dev = (uint32_t)s.st_dev; h.ino = (uint32_t)s.st_ino; h.mode = (uint32_t)s.st_mode;
        h.nlink = (uint32_t)s.st_nlink; h.uid = (uint32_t)s.st_uid; h.gid = (uint32_t)s.st_gid;
        h.rdev = (uint32_t)s.st_rdev;
        h.atime_sec = (int64_t)s.st_atime;
        h.mtime_sec = (int64_t)s.st_mtime;
        h.ctime_sec = (int64_t)s.st_ctime;
        h.size = (int64_t)s.st_size;
        h.blocks = blocks_512_for_size(h.size);
        h.blksize = 4096;
        return h;
    }

    HostStat from_host_stat(const struct _stat64& s) {
        HostStat h;
        h.dev = (uint32_t)s.st_dev; h.ino = (uint32_t)s.st_ino; h.mode = (uint32_t)s.st_mode;
        h.nlink = (uint32_t)s.st_nlink; h.uid = (uint32_t)s.st_uid; h.gid = (uint32_t)s.st_gid;
        h.rdev = (uint32_t)s.st_rdev;
        h.atime_sec = (int64_t)s.st_atime;
        h.mtime_sec = (int64_t)s.st_mtime;
        h.ctime_sec = (int64_t)s.st_ctime;
        h.size = (int64_t)s.st_size;
        h.blocks = blocks_512_for_size(h.size);
        h.blksize = 4096;
        return h;
    }
#endif
}

void set_app0_root(const std::string& root) { g_app0 = root; }
std::string resolve_guest_path(const char* guest_path) {
    if (!guest_path || !*guest_path) return {};
    // Sony media APIs accept content paths relative to the title's application root. Unlike the
    // guest libc, a native host backend has no guest current-working-directory state, so root the
    // relative spelling explicitly before applying the shared mount translation.
    if (guest_path[0] != '/') {
        const std::string app_path = std::string("/app0/") + guest_path;
        return translate(app_path.c_str());
    }
    return translate(guest_path);
}

// Mount / unmount the guest "/savedata0" area onto a host dir named by the save's dirName
// (sceSaveDataMount3 HLE, hle_service.cpp). create=true makes the host dir (CREATE-mode mount);
// create=false requires it to already exist (open-mode) and fails otherwise ("no such save").
// Returns true on success with /savedata0 translation active.
bool savedata0_mount(const char* dirname, bool create) {
    if (!dirname || !*dirname) return false;
    std::string d = save0_base() + "/" + dirname;
#ifdef _WIN32
    if (create) _mkdir(d.c_str());
    struct _stat st{};
    if (_stat(d.c_str(), &st) != 0 || !(st.st_mode & _S_IFDIR)) return false;
#else
    if (create) ::mkdir(d.c_str(), 0777);
    struct stat st{};
    if (::stat(d.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
#endif
    std::lock_guard<std::mutex> lk(g_save0_mx);
    g_save0 = d;
    return true;
}
void savedata0_umount() { std::lock_guard<std::mutex> lk(g_save0_mx); g_save0.clear(); }
// List the save-dir names that exist under the host save root (each subdir is one save the guest created
// via sceSaveDataMount3 create-mode). sceSaveDataDirNameSearch reports these so a prior session's saves
// appear in the game's load/continue list (#299 — the saves persisted but were invisible).
std::vector<std::string> savedata0_list_dirs() {
    std::vector<std::string> out;
#ifndef _WIN32
    std::string base = save0_base();
    if (DIR* dp = opendir(base.c_str())) {
        while (struct dirent* de = readdir(dp)) {
            if (de->d_name[0] == '.') continue;
            struct stat st{};
            if (::stat((base + "/" + de->d_name).c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                out.emplace_back(de->d_name);
        }
        closedir(dp);
    }
#endif   // Windows host is secondary; report no saves there.
    return out;
}

// Translate a host (Linux) struct stat into the FreeBSD/Orbis SceKernelStat layout the
// guest expects: 0x78 bytes, different field order. Writing the host layout (144 bytes,
// different offsets) both gives wrong values AND overruns the guest's 0x78-byte buffer
// (smashing an adjacent stack canary). Fields per FreeBSD 9 <sys/stat.h>.
// Translate a host `struct stat` into the guest's FreeBSD-layout SceKernelStat (0x78 bytes).
// Exposed (not file-local) so tests can guard the layout — writing the wrong size/offsets here
// once smashed a guest stack canary (st_size must land at 0x48, st_mode at 0x08, total 0x78).
void to_sce_stat(const struct stat& s, uint8_t* out) {
    write_sce_stat(from_host_stat(s), out);
}
#ifdef _WIN32
void to_sce_stat64(const struct _stat64& s, uint8_t* out) {
    write_sce_stat(from_host_stat(s), out);
}
#endif

// --- stdio FILE* ---
HLE(f_fopen)   { std::string h = translate(CS(a0)); const char* mode = CS(a1);
                  FILE* f = fopen(h.c_str(), mode);
                  if (filelog()) fprintf(stderr, "[file] fopen host='%s' mode='%s' -> %p error=%d\n",
                                         h.c_str(), mode ? mode : "(null)", (void*)f, f ? 0 : errno);
                  return (uint64_t)(uintptr_t)f; }
HLE(f_fclose)  { return a0 ? (uint64_t)(int64_t)fclose((FILE*)P(a0)) : 0; }
HLE(f_fread)   { if (!a3) return 0;
                  FILE* f = (FILE*)P(a3); size_t n = fread(P(a0), a1, a2, f);
                  if (filelog()) fprintf(stderr,
                      "[file] fread dst=%p size=%llu count=%llu stream=%p -> %llu eof=%d error=%d errno=%d\n",
                      P(a0), (unsigned long long)a1, (unsigned long long)a2, (void*)f,
                      (unsigned long long)n, feof(f), ferror(f), errno);
                  return (uint64_t)n; }
HLE(f_fwrite)  { return a3 ? (uint64_t)fwrite(P(a0), a1, a2, (FILE*)P(a3)) : 0; }
HLE(f_fseek)   { return a0 ? (uint64_t)(int64_t)fseek((FILE*)P(a0), (long)a1, (int)a2) : -1; }
HLE(f_ftell)   { return a0 ? (uint64_t)(int64_t)ftell((FILE*)P(a0)) : -1; }
HLE(f_fgets)   { return (uint64_t)(uintptr_t)(a2 ? fgets((char*)P(a0), (int)a1, (FILE*)P(a2)) : nullptr); }
HLE(f_fflush)  { return (uint64_t)(int64_t)fflush(a0 ? (FILE*)P(a0) : nullptr); }
HLE(f_feof)    { return a0 ? (uint64_t)feof((FILE*)P(a0)) : 1; }
HLE(f_ferror)  { return a0 ? (uint64_t)ferror((FILE*)P(a0)) : 0; }
HLE(f_setvbuf) { return 0; }
HLE(f_rewind)  { if (a0) rewind((FILE*)P(a0)); return 0; }
HLE(f_fgetc)   { return a0 ? (uint64_t)(int64_t)fgetc((FILE*)P(a0)) : (uint64_t)-1; }

// --- POSIX fd ---
// PROSPER_PREADLOG: log fd-lifecycle ops (open/close/read/pread/lseek) with fd->path + the guest
// call chain (first eboot-range return addrs on the stack). Used to trace Unity's FileCacher block
// fetches — which offsets are read vs which blocks are skipped, and who closes a file mid-load.
#ifndef _WIN32
static bool fdlog_on() { static int on = getenv("PROSPER_PREADLOG") ? 1 : 0; return on; }
// How many 8-byte words we can read UP from `sp` before running off the top of THIS thread's stack.
// Worker threads have small stacks, so an unbounded scan (1200-1600 words) SIGSEGVs past the top —
// which faulted inside preadlog itself and destabilized the very load it was tracing. Bound it.
static int stack_words_above(uint64_t* sp) {
    pthread_attr_t a; void* base = nullptr; size_t sz = 0;
    if (pthread_getattr_np(pthread_self(), &a) == 0) {
        pthread_attr_getstack(&a, &base, &sz);
        pthread_attr_destroy(&a);
        uintptr_t top = (uintptr_t)base + sz;                 // highest address of the stack
        if (top > (uintptr_t)sp) {
            size_t words = (top - (uintptr_t)sp) / 8;
            if (words > 4) words -= 2;                          // stop short of the very top guard page
            return words > 4096 ? 4096 : (int)words;
        }
    }
    return 256;   // conservative fallback if the stack extent is unavailable
}
static void preadlog(const char* fn, uint64_t fd, uint64_t off, uint64_t cnt) {
    if (!fdlog_on()) return;
    // resolve fd -> path
    char path[256] = "?"; char lp[64]; snprintf(lp, sizeof lp, "/proc/self/fd/%lld", (long long)fd);
    ssize_t k = readlink(lp, path, sizeof path - 1); if (k > 0) path[k] = 0; else path[0] = '?', path[1] = 0;
    const char* base = strrchr(path, '/'); base = base ? base + 1 : path;
    // top N distinct eboot-range return addresses on the stack (guest call chain)
    uint64_t cc[8] = {0,0,0,0,0,0,0,0}; int nc = 0; uint64_t* sp = (uint64_t*)__builtin_frame_address(0);
    int maxw = stack_words_above(sp);
    for (int i = 0; i < 1200 && i < maxw && nc < 8; i++) { uint64_t v = sp[i];
        if (v >= 0x400000000ull && v < 0x420000000ull) { uint64_t o = v - 0x400000000ull;
            if (nc == 0 || cc[nc-1] != o) cc[nc++] = o; } }
    fprintf(stderr, "[preadlog] %-6s %-24s fd=%lld off=0x%llx(blk %lld) cnt=0x%llx tid=%ld callers=eboot+0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx\n",
            fn, base, (long long)fd, (unsigned long long)off, (long long)(off / 0x10000),
            (unsigned long long)cnt, (long)syscall(SYS_gettid),
            (unsigned long long)cc[0], (unsigned long long)cc[1], (unsigned long long)cc[2], (unsigned long long)cc[3],
            (unsigned long long)cc[4], (unsigned long long)cc[5], (unsigned long long)cc[6], (unsigned long long)cc[7]);
    static int deep_budget = getenv("PROSPER_DEEPTRACE") ? 30 : 0;
    if (deep_budget > 0 && !strcmp(base, "resources.assets") && off >= 0x3c10000ull && off <= 0x3c60000ull) {
        deep_budget--;
        char line[1024]; int p = snprintf(line, sizeof line, "[deeptrace] blk %lld frames:", (long long)(off / 0x10000));
        int nf = 0; uint64_t last = 0;
        for (int i = 0; i < 1600 && i < maxw && nf < 20 && p < (int)sizeof line - 24; i++) {
            uint64_t v = sp[i];
            const char* m = nullptr; uint64_t o = 0;
            if (v >= 0x400000000ull && v < 0x420000000ull) { m = "eb"; o = v - 0x400000000ull; }
            else if (v >= 0x440000000ull && v < 0x4c0000000ull) { m = "il"; o = v - 0x440000000ull; }
            if (m && o != last) { p += snprintf(line + p, sizeof line - p, " %s+%llx", m, (unsigned long long)o); last = o; nf++; }
        }
        fprintf(stderr, "%s\n", line);
    }
}
#else
static bool fdlog_on() { return false; }
static void preadlog(const char*, uint64_t, uint64_t, uint64_t) {}
#endif
// PS5 fd semantics: the guest never receives fds 0-2 from sceKernelOpen (they are process-reserved),
// and the game exploits that — Unity teardown paths issue close(0) as a "close invalid-handle
// sentinel" no-op. On the host, stdin CAN be closed and fd 0 recycled to a game file; the next
// spurious guest close(0) then kills a live file mid-load (proven: unity_builtin_extra's block-2
// pread hit EBADF -> Unity silently kept a stale 64KB cache block -> deser crash at eboot+0x46beb4).
// So: (a) never return fd<3 from open (dup up); (b) refuse guest closes of fd 0-2.
// FreeBSD/Orbis open(2) flags share only the access mode with Linux: SCE O_CREAT=0x0200,
// O_TRUNC=0x0400, O_EXCL=0x0800 vs Linux 0x40/0x200/0x80. Passing the guest word raw turned
// "O_WRONLY|O_CREAT" (0x201) into Linux "O_WRONLY|O_TRUNC" — write-path opens failed with
// ENOENT on new files and silently truncated existing ones. Translate bit by bit (same
// FreeBSD-vs-Linux reason the stat struct is already translated below).
static int host_open_flags(uint64_t f) {
    int h = (int)(f & 3);                       // O_RDONLY/O_WRONLY/O_RDWR coincide
    if (f & 0x0008) h |= O_APPEND;              // FreeBSD O_APPEND
    if (f & 0x0200) h |= O_CREAT;
    if (f & 0x0400) h |= O_TRUNC;
    if (f & 0x0800) h |= O_EXCL;
#ifdef _WIN32
    // Game content is binary. Without O_BINARY, the Windows CRT treats 0x1a as EOF and
    // translates CRLF during _read; global-metadata.dat's first 0x1a is at byte 440, so
    // IL2CPP received 440 bytes from a valid 10.7 MiB file and rejected initialization.
    h |= O_BINARY;
#else
    if (f & 0x0004) h |= O_NONBLOCK;
    if (f & 0x0080) h |= O_SYNC;                // FreeBSD O_FSYNC
    if (f & 0x0100) h |= O_NOFOLLOW;
    if (f & 0x00020000) h |= O_DIRECTORY;
#endif
    return h;
}

// sceKernel filesystem calls return a FreeBSD/Orbis errno directly, while their libc siblings
// return -1 and leave the host errno for libc to expose. Most common errno values coincide, but
// the values after ERANGE diverge on Linux (for example ELOOP and ENAMETOOLONG), so translate
// host errors instead of copying their numbers into an SCE result.
static uint32_t file_sce_error(int error) {
    int guest_error = 5;  // EIO is the conservative fallback for an unmapped host failure.
    if (error == EPERM) guest_error = 1;
    else if (error == ENOENT) guest_error = 2;
    else if (error == EINTR) guest_error = 4;
    else if (error == EIO) guest_error = 5;
    else if (error == ENXIO) guest_error = 6;
    else if (error == EBADF) guest_error = 9;
    else if (error == ENOMEM) guest_error = 12;
    else if (error == EACCES) guest_error = 13;
    else if (error == EFAULT) guest_error = 14;
    else if (error == EBUSY) guest_error = 16;
    else if (error == EEXIST) guest_error = 17;
    else if (error == EXDEV) guest_error = 18;
    else if (error == ENODEV) guest_error = 19;
    else if (error == ENOTDIR) guest_error = 20;
    else if (error == EISDIR) guest_error = 21;
    else if (error == EINVAL) guest_error = 22;
    else if (error == ENFILE) guest_error = 23;
    else if (error == EMFILE) guest_error = 24;
    else if (error == ENOTTY) guest_error = 25;
#ifdef ETXTBSY
    else if (error == ETXTBSY) guest_error = 26;
#endif
    else if (error == EFBIG) guest_error = 27;
    else if (error == ENOSPC) guest_error = 28;
    else if (error == ESPIPE) guest_error = 29;
    else if (error == EROFS) guest_error = 30;
    else if (error == EMLINK) guest_error = 31;
    else if (error == EPIPE) guest_error = 32;
    else if (error == EAGAIN) guest_error = 35;
#ifdef EWOULDBLOCK
    else if (error == EWOULDBLOCK) guest_error = 35;
#endif
#ifdef ELOOP
    else if (error == ELOOP) guest_error = 62;
#endif
#ifdef ENAMETOOLONG
    else if (error == ENAMETOOLONG) guest_error = 63;
#endif
#ifdef EDQUOT
    else if (error == EDQUOT) guest_error = 69;
#endif
#ifdef EOVERFLOW
    else if (error == EOVERFLOW) guest_error = 84;
#endif
    return 0x80020000u | (uint32_t)guest_error;
}

HLE(f_open)  { std::string h = translate(CS(a0)); int host_flags = host_open_flags(a1);
#ifdef _WIN32
               int fd;
               if (windows_host_path_is_directory(h)) {
                   if ((a1 & 3) != 0) { errno = EISDIR; fd = -1; }
                   else fd = windows_open_directory(h);
               } else {
                   fd = (int)::open(h.c_str(), host_flags, (mode_t)a2);
               }
#else
               int fd = (int)::open(h.c_str(), host_flags, (mode_t)a2);
#endif
#ifdef _WIN32
               if (fd >= 0 && fd < 3) {
                   int low_fd = fd;
                   fd = windows_duplicate_above_stdio(low_fd);
                   int duplicate_errno = fd < 0 ? errno : 0;
                   ::_close(low_fd);
                   if (fd < 0) errno = duplicate_errno;
               }
#else
               while (fd >= 0 && fd < 3) { int nfd = fcntl(fd, F_DUPFD, 3); ::close(fd); fd = nfd; }
#endif
               int err = fd < 0 ? errno : 0;
               filelog_remember_fd(fd, h);
               if (filelog()) fprintf(stderr,
                   "[file] open-result host='%s' guest-flags=0x%llx host-flags=0x%x -> fd=%d error=%d\n",
                   h.c_str(), (unsigned long long)a1, host_flags, fd, err);
               if (fd >= 0) preadlog("open", (uint64_t)fd, 0, 0);
               else errno = err;
               return (uint64_t)(int64_t)fd; }
HLE(k_open)  { uint64_t result = f_open(a0, a1, a2, a3, a4, a5);
               return (int64_t)result < 0 ? file_sce_error(errno) : result; }
#ifdef __APPLE__
int getdents_close_fd(int fd);   // #843/#847: atomic cache invalidation + host close
#endif
HLE(f_close) { if (a0 < 3) { preadlog("close-lo-ignored", a0, 0, 0); return 0; }
               preadlog("close", a0, 0, 0);
               int fd = (int)a0;
#ifdef __APPLE__
               int r = getdents_close_fd(fd);
#elif defined(_WIN32)
               ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
               int r = -1;
               if (!windows_close_directory(fd, &r)) r = ::close(fd);
#else
               int r = ::close(fd);
#endif
               int err = r < 0 ? errno : 0;
               filelog_fd_io("close", fd, 0, 0, r, err);
               if (r == 0) filelog_forget_fd(fd);
               else errno = err;
               return (uint64_t)(int64_t)r; }
HLE(k_close) { uint64_t result = f_close(a0, a1, a2, a3, a4, a5);
               int error = errno;
               return (int64_t)result < 0 ? file_sce_error(error) : result; }
// dup/dup2 were MISSING -> the return-0 stub handed back fd 0 (a valid-looking descriptor that is actually
// stdin), so the guest read/closed stdin thinking it was its duplicate -> the fd-0 hazard this file guards
// against elsewhere. Back with host dup/dup2; dup keeps the result above fd 2 (same as f_open).
#ifndef _WIN32
HLE(f_dup)  { int fd = ::dup((int)a0); while (fd >= 0 && fd < 3) { int n = fcntl(fd, F_DUPFD, 3); ::close(fd); fd = n; } return (uint64_t)(int64_t)fd; }
HLE(f_dup2) { return (uint64_t)(int64_t)::dup2((int)a0, (int)a1); }
#else
HLE(f_dup)  { return (uint64_t)(int64_t)windows_duplicate_above_stdio((int)a0); }
HLE(f_dup2) {
    ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
    int r = ::_dup2((int)a0, (int)a1);
    // The Windows CRT reports success as zero; the guest/POSIX contract returns the
    // destination descriptor.
    return (uint64_t)(int64_t)(r < 0 ? -1 : (int)a1);
}
#endif

namespace {
constexpr uint64_t kGuestFDupFd = 0;
constexpr uint64_t kGuestFGetFd = 1;
constexpr uint64_t kGuestFSetFd = 2;
constexpr uint64_t kGuestFGetFl = 3;
constexpr uint64_t kGuestFSetFl = 4;
constexpr uint64_t kGuestFGetOwn = 5;
constexpr uint64_t kGuestFSetOwn = 6;
constexpr uint64_t kGuestFdCloExec = 1;
constexpr uint64_t kGuestONonblock = 0x0004;
constexpr uint64_t kGuestOAppend = 0x0008;
constexpr uint64_t kGuestOAsync = 0x0040;
constexpr uint64_t kGuestOSync = 0x0080;
constexpr uint64_t kGuestODirect = 0x00010000;

#ifndef _WIN32
uint64_t guest_status_flags_from_host(int flags) {
    uint64_t guest = (uint64_t)(flags & O_ACCMODE);
    if (flags & O_NONBLOCK) guest |= kGuestONonblock;
    if (flags & O_APPEND) guest |= kGuestOAppend;
#ifdef O_ASYNC
    if (flags & O_ASYNC) guest |= kGuestOAsync;
#endif
#ifdef O_SYNC
    if ((flags & O_SYNC) == O_SYNC) guest |= kGuestOSync;
#endif
#ifdef O_DIRECT
    if (flags & O_DIRECT) guest |= kGuestODirect;
#endif
    return guest;
}

int host_settable_status_flags(uint64_t guest) {
    int host = 0;
    if (guest & kGuestONonblock) host |= O_NONBLOCK;
    if (guest & kGuestOAppend) host |= O_APPEND;
#ifdef O_ASYNC
    if (guest & kGuestOAsync) host |= O_ASYNC;
#endif
#ifdef O_DIRECT
    if (guest & kGuestODirect) host |= O_DIRECT;
#endif
    return host;
}

int host_settable_status_mask() {
    int host = O_NONBLOCK | O_APPEND;
#ifdef O_ASYNC
    host |= O_ASYNC;
#endif
#ifdef O_DIRECT
    host |= O_DIRECT;
#endif
    return host;
}
#endif
} // namespace

// FreeBSD/Orbis and host fcntl ABIs only coincide for a subset of commands, and their O_* status
// bits differ substantially on Linux. Translate the descriptor/status operations used by libc and
// game runtimes; reject record-lock commands until their FreeBSD flock layout is translated too.
HLE(f_fcntl) {
    if (a0 > INT_MAX) {
        errno = EBADF;
        return (uint64_t)-1;
    }
    const int fd = (int)a0;
    switch (a1) {
    case kGuestFDupFd: {
        if (a2 > INT_MAX) {
            errno = EINVAL;
            return (uint64_t)-1;
        }
        const int minimum = (int)a2 < 3 ? 3 : (int)a2;
#ifdef _WIN32
        if (windows_directory_path(fd, nullptr)) {
            errno = ENOTSUP;
            return (uint64_t)-1;
        }
        return (uint64_t)(int64_t)windows_duplicate_at_least(fd, minimum);
#else
        return (uint64_t)(int64_t)::fcntl(fd, F_DUPFD, minimum);
#endif
    }
    case kGuestFGetFd:
#ifdef _WIN32
        if (windows_directory_path(fd, nullptr)) return kGuestFdCloExec;
        {
            ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
            const intptr_t raw = ::_get_osfhandle(fd);
            if (raw == -1) {
                errno = EBADF;
                return (uint64_t)-1;
            }
            DWORD flags = 0;
            if (!GetHandleInformation((HANDLE)raw, &flags)) {
                errno = GetLastError() == ERROR_INVALID_HANDLE ? EBADF : EACCES;
                return (uint64_t)-1;
            }
            return (flags & HANDLE_FLAG_INHERIT) ? 0 : kGuestFdCloExec;
        }
#else
        {
            const int flags = ::fcntl(fd, F_GETFD);
            return flags < 0 ? (uint64_t)-1
                             : (uint64_t)((flags & FD_CLOEXEC) ? kGuestFdCloExec : 0);
        }
#endif
    case kGuestFSetFd:
        if (a2 & ~kGuestFdCloExec) {
            errno = EINVAL;
            return (uint64_t)-1;
        }
#ifdef _WIN32
        if (windows_directory_path(fd, nullptr)) {
            // Synthetic directory descriptors never escape to a child process. Their fixed
            // CLOEXEC state cannot be cleared without introducing descriptor inheritance.
            if (a2 & kGuestFdCloExec) return 0;
            errno = ENOTSUP;
            return (uint64_t)-1;
        }
        {
            ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
            const intptr_t raw = ::_get_osfhandle(fd);
            if (raw == -1) {
                errno = EBADF;
                return (uint64_t)-1;
            }
            const DWORD inherit = (a2 & kGuestFdCloExec) ? 0 : HANDLE_FLAG_INHERIT;
            if (!SetHandleInformation((HANDLE)raw, HANDLE_FLAG_INHERIT, inherit)) {
                errno = GetLastError() == ERROR_INVALID_HANDLE ? EBADF : EACCES;
                return (uint64_t)-1;
            }
            return 0;
        }
#else
        return (uint64_t)(int64_t)::fcntl(
            fd, F_SETFD, (a2 & kGuestFdCloExec) ? FD_CLOEXEC : 0);
#endif
    case kGuestFGetFl:
#ifdef _WIN32
        // UCRT has no status-flag query. Do not preserve the old missing-import false success.
        errno = ENOTSUP;
        return (uint64_t)-1;
#else
        {
            const int flags = ::fcntl(fd, F_GETFL);
            return flags < 0 ? (uint64_t)-1 : guest_status_flags_from_host(flags);
        }
#endif
    case kGuestFSetFl:
#ifdef _WIN32
        // Likewise, UCRT cannot update O_NONBLOCK/O_APPEND on an existing descriptor.
        errno = ENOTSUP;
        return (uint64_t)-1;
#else
        {
            const int old_flags = ::fcntl(fd, F_GETFL);
            if (old_flags < 0) return (uint64_t)-1;
            const uint64_t old_guest_flags = guest_status_flags_from_host(old_flags);
            // FreeBSD permits O_SYNC in F_SETFL, but Linux and Darwin do not reliably change it
            // after open. Reject a requested transition instead of claiming success while the
            // host descriptor remains unchanged.
            if ((old_guest_flags ^ a2) & kGuestOSync) {
                errno = ENOTSUP;
                return (uint64_t)-1;
            }
            const int new_flags = (old_flags & ~host_settable_status_mask()) |
                                  host_settable_status_flags(a2);
            return (uint64_t)(int64_t)::fcntl(fd, F_SETFL, new_flags);
        }
#endif
    case kGuestFGetOwn:
#ifdef _WIN32
        errno = ENOTSUP;
        return (uint64_t)-1;
#else
        return (uint64_t)(int64_t)::fcntl(fd, F_GETOWN);
#endif
    case kGuestFSetOwn:
#ifdef _WIN32
        errno = ENOTSUP;
        return (uint64_t)-1;
#else
        return (uint64_t)(int64_t)::fcntl(fd, F_SETOWN, (int)a2);
#endif
    default:
        errno = EINVAL;
        return (uint64_t)-1;
    }
}

#ifndef _WIN32
// FULL read: loop until `count` bytes are read (or EOF/error). POSIX read()/pread() may return FEWER
// bytes than requested (a "short read") for a perfectly valid regular file — at internal buffer
// boundaries, under memory pressure, or on a signal. sceKernelRead/Pread on PS5 return the full count
// for a regular file, and Unity's asset streamer (FileCacher/CachedReader) assumes it: a short read
// leaves its 64 KB cache block PARTIALLY filled, the tail stays stale, and the object deserialized from
// that block gets a corrupt/null managed reference — which a later per-frame GC/finalizer pump then
// null-derefs (the intermittent Il2cpp crash on the cutscene-scene load; see CUTSCENE_PROGRESSION.md).
// Looping restores the full-read contract. Returns bytes read (== count on success), or -1/errno.
static int64_t read_full(int fd, void* buf, size_t count, bool positioned, off_t off) {
    size_t done = 0;
    while (done < count) {
        ssize_t r = positioned ? ::pread(fd, (char*)buf + done, count - done, off + (off_t)done)
                               : ::read(fd, (char*)buf + done, count - done);
        if (r < 0) { if (errno == EINTR) continue; return done ? (int64_t)done : (int64_t)-1; }
        if (r == 0) break;   // EOF — return the partial count the file actually had
        done += (size_t)r;
    }
    return (int64_t)done;
}
// Symmetric FULL write: sceKernelWrite/Pwrite on PS5 return the full count for a regular file, and the
// same save/asset code assumes it. A short host ::write (signal, buffer boundary, near-ENOSPC) returned
// verbatim would silently truncate a save; the guest, assuming the full contract, never re-issues the
// remainder. Loop to the full-write contract. Returns bytes written (== count on success), or -1/errno.
static int64_t write_full(int fd, const void* buf, size_t count, bool positioned, off_t off) {
    size_t done = 0;
    while (done < count) {
        ssize_t w = positioned ? ::pwrite(fd, (const char*)buf + done, count - done, off + (off_t)done)
                               : ::write(fd, (const char*)buf + done, count - done);
        if (w < 0) { if (errno == EINTR) continue; return done ? (int64_t)done : (int64_t)-1; }
        if (w == 0) break;   // no progress on a regular file — avoid an infinite loop
        done += (size_t)w;
    }
    return (int64_t)done;
}
#endif
HLE(f_read)  { int fd = (int)a0; int64_t off = -1;
#ifdef _WIN32
               ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
#endif
               if (filelog() || fdlog_on()) off = (int64_t)::lseek(fd, 0, SEEK_CUR);
               if (fdlog_on()) preadlog("read", a0, (uint64_t)off, a2);
               auto logged_return = [&](int64_t r) -> uint64_t {
                   const int error = r < 0 ? errno : 0;
                   filelog_fd_io("read", fd, off, a2, r, error);
                   if (r < 0) errno = error;
                   return (uint64_t)r;
               };
#ifndef _WIN32
               return logged_return(read_full(fd, P(a1), (size_t)a2, false, 0));
#else
               // Full sequential read (loop) — same full-count contract as read_full: a short ::read
               // would leave Unity's asset cache block partially filled and deserialize corrupt data.
               // Windows validates the entire destination range before _read discovers a short EOF.
               // Guest allocators can leave later pages reserved for lazy commit, so asking _read for
               // the whole range can fail even when the available file prefix fits in committed pages.
               // Validate the largest writable prefix first and read directly into it.  A normal
               // multi-megabyte asset allocation is one committed region, so this is one validation
               // and one host read instead of hundreds of 64 KiB bounce-buffer cycles.  An inaccessible
               // tail still limits the request before any file bytes are consumed, preserving partial
               // read and retry-offset behavior.
               { size_t done = 0, cnt = (size_t)a2, readable = cnt; char* b = (char*)P(a1);
                 const __int64 pos = ::_lseeki64((int)a0, 0, SEEK_CUR);
                 struct _stat64 st{};
                 if (pos >= 0 && ::_fstat64((int)a0, &st) == 0 &&
                     (st.st_mode & _S_IFMT) == _S_IFREG) {
                     const uint64_t len = st.st_size > 0 ? (uint64_t)st.st_size : 0;
                     const uint64_t remaining = (uint64_t)pos < len
                         ? len - (uint64_t)pos : 0;
                     if (remaining < readable) readable = (size_t)remaining;
                 }
                 while (done < readable) {
                     size_t left = readable - done;
                     const size_t request = left < 0x40000000u ? left : 0x40000000u;
                     const uint64_t prefix = windows_prepare_guest_write_prefix(
                         (uint64_t)(uintptr_t)(b + done), request);
                     if (!prefix) {
                         errno = EFAULT;
                         return logged_return(done ? (int64_t)done : (int64_t)-1);
                     }
                     const unsigned want = (unsigned)prefix;
                     int r = ::read((int)a0, b + done, want);
                     if (r < 0) return logged_return(done ? (int64_t)done : (int64_t)-1);
                     if (r == 0) break;   // EOF
                     done += (size_t)r;
                 }
                 return logged_return((int64_t)done); }
#endif
             }
HLE(f_write) {
#ifndef _WIN32
               return (uint64_t)write_full((int)a0, P(a1), (size_t)a2, false, 0);
#else
               ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
               return (uint64_t)(int64_t)::write((int)a0, P(a1), (size_t)a2);
#endif
             }
HLE(f_lseek) { if (fdlog_on() && ((int)a2 != SEEK_CUR || a1 != 0)) preadlog("lseek", a0, a1, (uint64_t)a2);
#ifdef _WIN32
               ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
               int64_t directory_result = -1;
               if (windows_seek_directory((int)a0, (int64_t)a1, (int)a2, &directory_result))
                   return (uint64_t)directory_result;
#endif
               return (uint64_t)(int64_t)::lseek((int)a0, (off_t)a1, (int)a2); }
#ifndef _WIN32
HLE(f_pread)  { preadlog("pread", a0, a3, a2); int64_t r = read_full((int)a0, P(a1), (size_t)a2, true, (off_t)a3);
                const int error = r < 0 ? errno : 0;
                filelog_fd_io("pread", (int)a0, (int64_t)a3, a2, r, error);
                if (r < 0) errno = error;
                return (uint64_t)r; }
HLE(f_pwrite) { return (uint64_t)write_full((int)a0, P(a1), (size_t)a2, true, (off_t)a3); }
// Vectored IO — OrbisKernelIovec == host iovec { void* iov_base; size_t iov_len }, and guest pointers are
// identity-mapped, so the guest iovec array and its buffers pass straight to host (p)readv/(p)writev.
// These were MISSING -> the return-0 stub read as "0 bytes" (silent EOF for readv/preadv) / "0 written",
// a corruption trap for any module that uses them (e.g. UE4's IO stack).
HLE(f_readv)  { return (uint64_t)(int64_t)::readv((int)a0, (const struct iovec*)P(a1), (int)a2); }
HLE(f_writev) { return (uint64_t)(int64_t)::writev((int)a0, (const struct iovec*)P(a1), (int)a2); }
HLE(f_preadv) { return (uint64_t)(int64_t)::preadv((int)a0, (const struct iovec*)P(a1), (int)a2, (off_t)a3); }
HLE(f_pwritev){ return (uint64_t)(int64_t)::pwritev((int)a0, (const struct iovec*)P(a1), (int)a2, (off_t)a3); }
#else
// Windows positioned/vectored IO. MinGW has no pread/pwrite/*v, and these previously returned -1
// (error) — which broke Unity's async asset streamer (FileCacher/CachedReader does POSITIONED reads
// via sceKernelPread from the Loading.AsyncRead thread), so no assets loaded and the boot stalled
// after VideoOut setup. Back them with ReadFile/WriteFile + an OVERLAPPED offset: atomic positioned
// IO that is thread-safe even with concurrent reads on the SAME fd (it doesn't touch the shared file
// position). Full-read/write loop to honor the PS5 regular-file full-count contract (a short read
// leaves Unity's 64 KB cache block partially filled -> corrupt deserialization). CONFIDENCE: HIGH.
namespace {
int windows_io_errno(DWORD error) {
    switch (error) {
    case ERROR_INVALID_HANDLE:      return EBADF;
    case ERROR_ACCESS_DENIED:
    case ERROR_SHARING_VIOLATION:
    case ERROR_LOCK_VIOLATION:      return EACCES;
    case ERROR_INVALID_PARAMETER:
    case ERROR_INVALID_USER_BUFFER: return EINVAL;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:         return ENOMEM;
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:    return ENOSPC;
    case ERROR_WRITE_PROTECT:       return EROFS;
    case ERROR_BROKEN_PIPE:
    case ERROR_NO_DATA:             return EPIPE;
    case ERROR_OPERATION_ABORTED:   return EINTR;
    default:                        return EIO;
    }
}

int64_t win_pio(int fd, void* buf, size_t count, uint64_t off, bool write) {
    ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
    HANDLE h = (HANDLE)(intptr_t)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    size_t done = 0;
    while (done < count) {
        OVERLAPPED ov; memset(&ov, 0, sizeof ov);
        uint64_t o = off + done;
        ov.Offset = (DWORD)(o & 0xffffffffu); ov.OffsetHigh = (DWORD)(o >> 32);
        DWORD want = (count - done) > 0x40000000u ? 0x40000000u : (DWORD)(count - done);
        DWORD n = 0;
        BOOL ok = write ? WriteFile(h, (const char*)buf + done, want, &n, &ov)
                        : ReadFile(h, (char*)buf + done, want, &n, &ov);
        if (!ok) {
            const DWORD error = GetLastError();
            if (!write && error == ERROR_HANDLE_EOF) break;   // read past EOF -> partial
            errno = windows_io_errno(error);
            return done ? (int64_t)done : -1;
        }
        if (n == 0) break;   // EOF (read) / no progress (write)
        done += n;
    }
    return (int64_t)done;
}
// Guest OrbisKernelIovec == host layout { void* base; size_t len }.
struct GIovec { void* base; size_t len; };
} // namespace
HLE(f_pread)  { int64_t r = win_pio((int)a0, P(a1), (size_t)a2, (uint64_t)a3, false);
                const int error = r < 0 ? errno : 0;
                filelog_fd_io("pread", (int)a0, (int64_t)a3, a2, r, error);
                if (r < 0) errno = error;
                return (uint64_t)r; }
HLE(f_pwrite) { return (uint64_t)win_pio((int)a0, P(a1), (size_t)a2, (uint64_t)a3, true); }
HLE(f_readv)  { ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
                auto* v = (GIovec*)P(a1); int nc = (int)a2; int64_t tot = 0;
                for (int i = 0; i < nc; i++) { int64_t r = (int64_t)(int)::read((int)a0, v[i].base, (unsigned)v[i].len);
                    if (r < 0) return tot ? (uint64_t)tot : (uint64_t)-1; tot += r; if ((size_t)r < v[i].len) break; }
                return (uint64_t)tot; }
HLE(f_writev) { ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
                auto* v = (GIovec*)P(a1); int nc = (int)a2; int64_t tot = 0;
                for (int i = 0; i < nc; i++) { int64_t w = (int64_t)(int)::write((int)a0, v[i].base, (unsigned)v[i].len);
                    if (w < 0) return tot ? (uint64_t)tot : (uint64_t)-1; tot += w; if ((size_t)w < v[i].len) break; }
                return (uint64_t)tot; }
HLE(f_preadv) { auto* v = (GIovec*)P(a1); int nc = (int)a2; uint64_t off = (uint64_t)a3; int64_t tot = 0;
                for (int i = 0; i < nc; i++) { int64_t r = win_pio((int)a0, v[i].base, v[i].len, off + (uint64_t)tot, false);
                    if (r < 0) return tot ? (uint64_t)tot : (uint64_t)-1; tot += r; if ((size_t)r < v[i].len) break; }
                return (uint64_t)tot; }
HLE(f_pwritev){ auto* v = (GIovec*)P(a1); int nc = (int)a2; uint64_t off = (uint64_t)a3; int64_t tot = 0;
                for (int i = 0; i < nc; i++) { int64_t w = win_pio((int)a0, v[i].base, v[i].len, off + (uint64_t)tot, true);
                    if (w < 0) return tot ? (uint64_t)tot : (uint64_t)-1; tot += w; if ((size_t)w < v[i].len) break; }
                return (uint64_t)tot; }
#endif

// The ssize_t/off_t-returning sceKernel exports sign-extend their 32-bit SCE error values to 64
// bits. Their libc siblings retain -1 + errno. Keep separate wrappers so successful byte counts
// and offsets pass through unchanged while failures preserve the guest ABI on every host.
static uint64_t kernel_io_result(uint64_t result, int error) {
    if ((int64_t)result >= 0) return result;
    return (uint64_t)(int64_t)(int32_t)file_sce_error(error);
}

HLE(k_read)   { uint64_t r = f_read(a0, a1, a2, a3, a4, a5);   int e = errno; return kernel_io_result(r, e); }
HLE(k_write)  { uint64_t r = f_write(a0, a1, a2, a3, a4, a5);  int e = errno; return kernel_io_result(r, e); }
HLE(k_lseek)  { uint64_t r = f_lseek(a0, a1, a2, a3, a4, a5);  int e = errno; return kernel_io_result(r, e); }
HLE(k_pread)  { uint64_t r = f_pread(a0, a1, a2, a3, a4, a5);  int e = errno; return kernel_io_result(r, e); }
HLE(k_pwrite) { uint64_t r = f_pwrite(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_io_result(r, e); }
HLE(k_readv)  { uint64_t r = f_readv(a0, a1, a2, a3, a4, a5);  int e = errno; return kernel_io_result(r, e); }
HLE(k_writev) { uint64_t r = f_writev(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_io_result(r, e); }
HLE(k_preadv) { uint64_t r = f_preadv(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_io_result(r, e); }
HLE(k_pwritev){ uint64_t r = f_pwritev(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_io_result(r, e); }

// sceKernelFtruncate(fd, length): resize an open file. Was MISSING -> the return-0 stub faked success
// without truncating, so the near-universal save idiom "overwrite with fewer bytes, then ftruncate to
// drop the old tail" left stale trailing bytes -> a longer-than-expected, corrupt save on reload (and a
// preallocate-by-ftruncate followed by mmap would SIGBUS). NID VW3TVZiM4-E is imported by the Messenger
// eboot directly. Returns 0 / negative on failure.
#ifndef _WIN32
HLE(f_ftruncate) { return (uint64_t)(int64_t)::ftruncate((int)a0, (off_t)a1); }
// sceKernelTruncate(path, length): the path-based sibling of ftruncate. Same fake-success -> stale-tail
// corruption class (NID WlyEA-sLDf0, reached via libc.prx). Translate the path through the mount layer.
HLE(f_truncate)  { std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::truncate(h.c_str(), (off_t)a1); }
#else
HLE(f_ftruncate) { ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
                    int error = ::_chsize_s((int)a0, (__int64)a1);
                    if (error) { errno = error; return (uint64_t)(int64_t)-1; }
                    return 0; }
HLE(f_truncate)  { std::string h = translate(CS(a0));
                    ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
                    int fd = ::_open(h.c_str(), _O_RDWR | _O_BINARY);
                    if (fd < 0) return (uint64_t)(int64_t)-1;
                    int resize_error = ::_chsize_s(fd, (__int64)a1);
                    int close_result = ::_close(fd);
                    if (resize_error) { errno = resize_error; return (uint64_t)(int64_t)-1; }
                    return (uint64_t)(int64_t)close_result; }
#endif

// --- sceKernelAio* — the kernel async-IO command API (issue #312 suspect list). -----------------
// PS4-inherited contract, reference shadPS4 core/libraries/kernel/aio.cpp (the PS5 3.20 libkernel
// exports the IDENTICAL NID set — verified against ../PS5-3.20_Libs/libkernel.c). The kernel model
// we implement (same as shadPS4): commands complete synchronously at submit — prosper's guest fds
// ARE host fds, so each element performs a positioned read/write immediately; each request's
// result out-struct {s64 returnValue; u32 state} is filled per element; *id receives a nonzero
// submit id whose state Poll/Wait return; Delete/Cancel write their out-params and recycle the id.
// The previous unimplemented stubs returned 0 WITHOUT writing *id, *state, or any result — the
// guest (DOLL's SaveLoadUpdate worker submits its GameSaveData writes here; CRI FS submits movie
// reads) then consumed uninitialized ids and poll states. DOLL calls the singular variants; the
// plural/Multiple ones are registered to the same helpers where the ABI is element-wise.
// AioSetParam is init-tuning (no out-params) -> honest success. CONFIDENCE: HIGH on the ABI
// (shadPS4 + identical PS5 NIDs); MED on error constants (0x8002000e EFAULT, shadPS4's choice).
namespace {
constexpr uint64_t kAioErrFault = 0x8002000eull;             // SCE_KERNEL_ERROR_EFAULT
enum : int32_t { kAioProcessing = 2, kAioCompleted = 3, kAioAborted = 4 };
struct AioResult { int64_t return_value; uint32_t state; };
struct AioReq { int64_t offset; int64_t nbyte; void* buf; AioResult* result; int32_t fd; };
constexpr uint32_t kAioQueue = 512;                          // shadPS4 MAX_QUEUE
std::atomic<int32_t> g_aio_state[kAioQueue];                 // id -> state (0 = never used)
std::atomic<uint32_t> g_aio_next{1};
uint64_t aio_submit(uint64_t reqs, int32_t n, bool is_write, uint64_t out_id) {
    if (!reqs || !out_id) return kAioErrFault;
    auto* req = (AioReq*)P(reqs);
    uint32_t raw = g_aio_next.fetch_add(1) % kAioQueue;
    if (!raw) raw = g_aio_next.fetch_add(1) % kAioQueue;     // id 0 is "no request"
    int32_t id = (int32_t)raw;
    g_aio_state[raw].store(kAioProcessing);
    for (int32_t i = 0; i < n; i++) {
        int64_t r;
#ifndef _WIN32
        r = is_write ? write_full(req[i].fd, req[i].buf, (size_t)req[i].nbyte, true, (off_t)req[i].offset)
                     : read_full(req[i].fd, req[i].buf, (size_t)req[i].nbyte, true, (off_t)req[i].offset);
#else
        // Windows host (secondary): emulate positioned IO with lseek+read/write on the CRT fd.
        long long prev = ::_lseeki64(req[i].fd, 0, SEEK_CUR);
        ::_lseeki64(req[i].fd, req[i].offset, SEEK_SET);
        r = is_write ? (int64_t)::_write(req[i].fd, req[i].buf, (unsigned)req[i].nbyte)
                     : (int64_t)::_read(req[i].fd, req[i].buf, (unsigned)req[i].nbyte);
        if (prev >= 0) ::_lseeki64(req[i].fd, prev, SEEK_SET);
#endif
        if (req[i].result) {
            req[i].result->return_value = r;
            req[i].result->state = (r < 0) ? kAioAborted : kAioCompleted;
        }
        if (filelog()) fprintf(stderr, "[file] aio-%s fd=%d off=0x%llx n=0x%llx -> %lld\n",
                               is_write ? "write" : "read", req[i].fd,
                               (unsigned long long)req[i].offset, (unsigned long long)req[i].nbyte,
                               (long long)r);
    }
    g_aio_state[raw].store(kAioCompleted);
    *(int32_t*)P(out_id) = id;
    return 0;
}
}
HLE(k_aio_init)         { return 0; }                        // InitializeImpl / InitializeParam / SetParam
HLE(k_aio_submit_read)  { return aio_submit(a0, (int32_t)a1, false, a3); }   // (req[], n, prio, id*)
HLE(k_aio_submit_write) { return aio_submit(a0, (int32_t)a1, true,  a3); }
HLE(k_aio_wait)  {   // (id, s32* state, u32* usec) — everything completed at submit: report and return
    if (!a1) return kAioErrFault;
    *(int32_t*)P(a1) = g_aio_state[(uint32_t)a0 % kAioQueue].load();
    return 0;
}
HLE(k_aio_poll)  {   // (id, s32* state)
    if (!a1) return kAioErrFault;
    *(int32_t*)P(a1) = g_aio_state[(uint32_t)a0 % kAioQueue].load();
    return 0;
}
HLE(k_aio_delete) {  // (id, s32* ret)
    if (!a1) return kAioErrFault;
    g_aio_state[(uint32_t)a0 % kAioQueue].store(kAioAborted);
    *(int32_t*)P(a1) = 0;
    return 0;
}
HLE(k_aio_cancel) {  // (id, s32* state): nothing is in flight to cancel — report current/aborted
    if (!a1) return kAioErrFault;
    if (a0) { g_aio_state[(uint32_t)a0 % kAioQueue].store(kAioAborted);
              *(int32_t*)P(a1) = kAioAborted; }
    else    *(int32_t*)P(a1) = kAioProcessing;   // shadPS4: id 0 -> "processing"
    return 0;
}

#ifndef _WIN32
HLE(f_stat)  { std::string h = translate(CS(a0)); struct stat st; int r = ::stat(h.c_str(), &st); if (r == 0 && a1) to_sce_stat(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
HLE(f_fstat) { struct stat st; int r = ::fstat((int)a0, &st); int err = r < 0 ? errno : 0;
               if (r == 0 && a1) to_sce_stat(st, (uint8_t*)P(a1));
               filelog_fd_stat((int)a0, r, err, r == 0 ? (int64_t)st.st_size : -1);
               if (r < 0) errno = err;
               return (uint64_t)(int64_t)r; }
// lstat: was MISSING -> the return-0 stub reported success while leaving the caller's stat buffer as
// uninitialized garbage (wrong file type/size). We have no symlinks in the dump, so ::lstat == ::stat, but
// the key fix is WRITING the buffer. fsync: was fake-success; flush for real save durability.
HLE(f_lstat) { std::string h = translate(CS(a0)); struct stat st; int r = ::lstat(h.c_str(), &st); if (r == 0 && a1) to_sce_stat(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
HLE(f_fsync) { return (uint64_t)(int64_t)::fsync((int)a0); }
HLE(f_fdatasync) {
#if defined(__APPLE__)
    // Darwin does not expose fdatasync. fsync provides the required durability as a stronger
    // operation while preserving the descriptor/error contract.
    return (uint64_t)(int64_t)::fsync((int)a0);
#else
    return (uint64_t)(int64_t)::fdatasync((int)a0);
#endif
}
#else
HLE(f_stat)  { std::string h = translate(CS(a0)); struct _stat64 st; int r = ::_stat64(h.c_str(), &st); if (r == 0 && a1) to_sce_stat64(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
HLE(f_fstat) { struct _stat64 st; std::string directory_path;
               ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
               int r = windows_directory_path((int)a0, &directory_path)
                     ? ::_stat64(directory_path.c_str(), &st)
                     : ::_fstat64((int)a0, &st);
               int err = r < 0 ? errno : 0;
               if (r == 0 && a1) to_sce_stat64(st, (uint8_t*)P(a1));
               filelog_fd_stat((int)a0, r, err, r == 0 ? (int64_t)st.st_size : -1);
               if (r < 0) errno = err;
               return (uint64_t)(int64_t)r; }
HLE(f_lstat) { return f_stat(a0,a1,a2,a3,a4,a5); }
HLE(f_fsync) { ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
               return (uint64_t)(int64_t)::_commit((int)a0); }
// The Windows CRT has no data-only durability primitive. _commit is the same stronger fallback
// used for fsync and, unlike the missing-import stub, reports invalid descriptors truthfully.
HLE(f_fdatasync) { return f_fsync(a0,a1,a2,a3,a4,a5); }
#endif

// These file APIs have signed 32-bit results. Kernel exports return the translated SCE error
// directly, while the libc names retain -1 plus errno. Keep the base host operation shared so
// successful behavior stays identical across both namespaces.
static uint64_t kernel_file_result32(uint64_t result, int error) {
    return (int64_t)result < 0 ? file_sce_error(error) : result;
}
HLE(k_stat)     { uint64_t r = f_stat(a0, a1, a2, a3, a4, a5);     int e = errno; return kernel_file_result32(r, e); }
HLE(k_fstat)    { uint64_t r = f_fstat(a0, a1, a2, a3, a4, a5);    int e = errno; return kernel_file_result32(r, e); }
HLE(k_lstat)    { uint64_t r = f_lstat(a0, a1, a2, a3, a4, a5);    int e = errno; return kernel_file_result32(r, e); }
HLE(k_truncate) { uint64_t r = f_truncate(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }
HLE(k_ftruncate){ uint64_t r = f_ftruncate(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }
HLE(k_fsync)    { uint64_t r = f_fsync(a0, a1, a2, a3, a4, a5);    int e = errno; return kernel_file_result32(r, e); }
HLE(k_fdatasync){ uint64_t r = f_fdatasync(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }
HLE(f_access){ std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::access(h.c_str(), (int)a1); }
HLE(f_mkdir) { std::string h = translate(CS(a0));   // sceKernelMkdir(path, mode)
#ifdef _WIN32
    return (uint64_t)(int64_t)::_mkdir(h.c_str());
#else
    return (uint64_t)(int64_t)::mkdir(h.c_str(), (mode_t)(a1 ? a1 : 0777));
#endif
}
HLE(f_rmdir) { std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::rmdir(h.c_str()); }
HLE(k_mkdir) { uint64_t r = f_mkdir(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }
HLE(k_rmdir) { uint64_t r = f_rmdir(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }

// sceKernelGetdents(int fd, char* buf, size_t nbytes) — fill FreeBSD dirent records
// {u32 fileno; u16 reclen; u8 type; u8 namlen; char name[]} (4-aligned). Backed by the host's
// getdents64 on the SAME fd so the kernel keeps the directory cursor; Linux and BSD DT_* type
// values match. Returns bytes written (0 = end of directory), or an SCE error. UE4 (PPSA17942)
// enumerates its pak directory with this during IO-stack init.
static uint64_t directory_sce_error(int error) {
    return 0x80020000ull | (uint64_t)(error & 0xff);
}

static bool directory_result_is_error(uint64_t result) {
    return (result & ~0xffull) == 0x80020000ull;
}

// libc's getdents/getdirentries use the POSIX -1 + errno contract, while their sceKernel siblings
// return an SCE_KERNEL_ERROR value directly. They used to share the kernel handler, so libc saw a
// large positive byte count on failure and could walk an untouched directory buffer as valid data.
static uint64_t directory_result_to_posix(uint64_t result) {
    if (!directory_result_is_error(result)) return result;
    errno = (int)(result & 0xff);
    return (uint64_t)(int64_t)-1;
}

#ifndef _WIN32
#ifdef __APPLE__
// #843: sceKernelGetdents caches one DIR* per directory fd (Darwin has no getdents-on-fd). The cache
// MUST be dropped when the guest closes that fd — otherwise a reused fd number hands back a STALE DIR*
// for a previously-enumerated directory. .NET's Interop.Sys.EnumerateFilesRecursively (OpenDir ->
// ReadDirR -> GetDirectoryEntryFullPath) opens+closes many directories, reusing fd numbers, so a stale
// DIR* returns a consumed/closed directory -> a garbage DirectoryEntry -> strcmp on a bad name pointer,
// crashing scene load (the macOS Blasphemous 2 first-level wall). File-scope so f_close can invalidate.
// The mutex covers both the map and every DIR* operation: invalidation must not closedir a stream
// while getdents is using it, and two reads of one stream must not race its cursor (#847).
static std::mutex g_getdents_mx;
static std::map<int, DIR*> g_getdents_dirs;
int getdents_close_fd(int fd) {
    std::lock_guard<std::mutex> lk(g_getdents_mx);
    auto it = g_getdents_dirs.find(fd);
    if (it != g_getdents_dirs.end()) { if (it->second) closedir(it->second); g_getdents_dirs.erase(it); }
    // Keep the guest-fd close under the same guard. Otherwise getdents can repopulate this key
    // after erase but before close, leaving the replacement DIR* stale when the fd is reused.
    return ::close(fd);
}
#endif
HLE(k_getdents) {
    if (!a1 || a2 < 32) return 0x80020016ull;   // EINVAL
#ifdef __APPLE__
    // Darwin has no getdents-on-fd (getdirentries is unavailable with 64-bit inodes). Keep the
    // per-fd kernel cursor by caching one DIR* per directory fd (fdopendir owns a dup, so the
    // guest's fd number stays valid); seekdir puts back the first entry that doesn't fit.
    // Hold the cache mutex through the complete read so f_close cannot invalidate/free dp
    // underneath us; this also serializes concurrent reads of the same cursor.
    std::lock_guard<std::mutex> lk(g_getdents_mx);
    DIR* dp = nullptr;
    auto it = g_getdents_dirs.find((int)a0);
    if (it != g_getdents_dirs.end()) dp = it->second;
    else {
        int d2 = dup((int)a0);
        dp = d2 >= 0 ? fdopendir(d2) : nullptr;
        if (!dp) {
            int error = errno;
            if (d2 >= 0) ::close(d2);
            return directory_sce_error(error);
        }
        g_getdents_dirs[(int)a0] = dp;
    }
    uint8_t* out = (uint8_t*)P(a1);
    size_t w = 0;
    for (;;) {
        long pos = telldir(dp);
        struct dirent* d = readdir(dp);
        if (!d) break;
        size_t namlen = d->d_namlen;
        size_t rec = (8 + namlen + 1 + 3) & ~(size_t)3;   // 4-byte header + name + NUL, 4-aligned
        if (w + rec > a2) { seekdir(dp, pos); break; }    // didn't fit: rewind so nothing is lost
        uint8_t* r = out + w;
        *(uint32_t*)(r + 0) = (uint32_t)d->d_ino;
        *(uint16_t*)(r + 4) = (uint16_t)rec;
        r[6] = d->d_type;
        r[7] = (uint8_t)namlen;
        memcpy(r + 8, d->d_name, namlen + 1);
        w += rec;
    }
    return (uint64_t)w;
#else
    struct LinuxDirent64 { uint64_t ino; int64_t off; uint16_t reclen; uint8_t type; char name[]; };
    uint8_t tmp[4096];
    size_t want = a2 < sizeof tmp ? (size_t)a2 : sizeof tmp;
    long n = syscall(SYS_getdents64, (int)a0, tmp, (unsigned)want);
    if (n < 0)  return directory_sce_error(errno);
    if (n == 0) return 0;
    uint8_t* out = (uint8_t*)P(a1);
    size_t o = 0, w = 0;
    while (o < (size_t)n) {
        auto* d = (const LinuxDirent64*)(tmp + o);
        size_t namlen = strlen(d->name);
        size_t rec = (8 + namlen + 1 + 3) & ~(size_t)3;   // 4-byte header + name + NUL, 4-aligned
        if (w + rec > a2) break;
        uint8_t* r = out + w;
        *(uint32_t*)(r + 0) = (uint32_t)d->ino;
        *(uint16_t*)(r + 4) = (uint16_t)rec;
        r[6] = d->type;
        r[7] = (uint8_t)namlen;
        memcpy(r + 8, d->name, namlen + 1);
        w += rec;
        o += d->reclen;
    }
    return (uint64_t)w;
#endif
}
// sceKernelGetdirentries(fd, buf, nbytes, long* basep): like getdents but also writes the pre-call
// directory offset to *basep. Was MISSING -> returned 0 = "empty directory". Delegate to f_getdents and
// stamp basep with the cursor position captured before the read.
HLE(k_getdirentries) {
    off_t base = ::lseek((int)a0, 0, SEEK_CUR);
    uint64_t r = k_getdents(a0, a1, a2, 0, 0, 0);
    if (!directory_result_is_error(r) && a3) *(int64_t*)P(a3) = (int64_t)base;
    return r;
}
#else
HLE(k_getdents) {
    if (!a1 || a2 < 32) return directory_sce_error(EINVAL);
    if (windows_directory_path((int)a0, nullptr))
        return windows_getdents((int)a0, a1, a2);

    // Preserve support for directory HANDLEs attached to CRT descriptors by callers outside the
    // normal guest open path. Guest opens use the emulator-owned descriptor table above because
    // UCRT itself refuses to create such descriptors.
    ScopedCrtInvalidParameterHandler suppress_invalid_parameter;
    intptr_t handle = ::_get_osfhandle((int)a0);
    if (handle == -1) return directory_sce_error(EBADF);
    if (GetFileType((HANDLE)handle) != FILE_TYPE_DISK) return directory_sce_error(ENOTDIR);
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle((HANDLE)handle, &info)) return directory_sce_error(ENOTDIR);
    if (!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) return directory_sce_error(ENOTDIR);
    // Directory descriptors are not currently produced by the Windows CRT-backed open path.
    return 0;
}
HLE(k_getdirentries) {
    int64_t base = -1;
    if (windows_seek_directory((int)a0, 0, SEEK_CUR, &base)) {
        if (a3 && !windows_prepare_guest_write(a3, sizeof(int64_t)))
            return directory_sce_error(EFAULT);
        uint64_t result = windows_getdents((int)a0, a1, a2);
        if (!directory_result_is_error(result) && a3)
            *(int64_t*)(uintptr_t)a3 = base;
        return result;
    }

    uint64_t result = k_getdents(a0, a1, a2, a3, a4, a5);
    if (directory_result_is_error(result)) return result;
    if (a3 && !windows_prepare_guest_write(a3, sizeof(int64_t)))
        return directory_sce_error(EFAULT);
    if (a3) *(int64_t*)(uintptr_t)a3 = 0;
    return result;
}
#endif
HLE(f_getdents) {
    return directory_result_to_posix(k_getdents(a0, a1, a2, a3, a4, a5));
}
HLE(f_getdirentries) {
    return directory_result_to_posix(k_getdirentries(a0, a1, a2, a3, a4, a5));
}
HLE(f_unlink){ std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::
#ifdef _WIN32
    _unlink
#else
    unlink
#endif
    (h.c_str()); }
// sceKernelRename(from, to): move/rename a file. Was MISSING -> the return-0 stub faked success without
// moving anything, breaking the near-universal atomic-save idiom (write "save.tmp", then rename it over
// "save.dat"): the real save file was never produced/updated while the guest believed it saved. BOTH
// paths go through the mount-path translation (a rename inside /savedata0 must not hit raw guest paths).
// NID 52NcYU9+lEo, reached via libc.prx. ::rename is standard C (Windows + Linux).
HLE(f_rename){ std::string from = translate(CS(a0)), to = translate(CS(a1));
               return (uint64_t)(int64_t)::rename(from.c_str(), to.c_str()); }
HLE(k_unlink){ uint64_t r = f_unlink(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }
HLE(k_rename){ uint64_t r = f_rename(a0, a1, a2, a3, a4, a5); int e = errno; return kernel_file_result32(r, e); }

// --- APR (Async Page Read) file resolution -----------------------------------------------------
// sceKernelAprResolveFilepathsToIdsAndFileSizes(const char** paths, int count, uint32_t* outIds,
//   uint64_t* outSizes, uint32_t* outFlags, int reserved) — the entry point of UE4's IoStore/APR
// pipeline. Live capture (PPSA17942): called once per pak container with a /app0-translated path
// (global.utoc/.ucas, pakchunkN-ps5.pak/.utoc/.ucas). Stubbing it to EINVAL made APR proceed on
// garbage ids/sizes and wild-write over the allocator (the "MallocBinned unrecognized block" /
// canary corruption). Resolve each path for real: stat the host file, assign a stable id, record
// id->host-path so the read path can pread by id. CONFIDENCE: MED (arg roles from live capture;
// outFlags semantics unknown -> 0).
namespace {
    PROSPER_HEAP_MUTEX(g_apr_mx);   // #707: heap-backed on macOS (APR registry mutex near the corrupted __DATA cluster)
    struct AprFile { std::string path; uint64_t size; };
    std::vector<AprFile> g_apr_files;   // id (1-based index) -> {host path, size}
}
// Exposed to the read path (Ampr page-read) to map an APR id back to its host file.
std::string prosper_apr_path_for_id(uint32_t id) {
    std::lock_guard lk(g_apr_mx);
    return (id >= 1 && id <= g_apr_files.size()) ? g_apr_files[id - 1].path : std::string();
}
// Register a resolved container and return its stable 1-based id. Re-resolving the same host path
// returns the existing id (updated size) instead of a duplicate entry — a duplicate would make
// every size-keyed read of that file look ambiguous. Exposed (not static) for the unit test.
uint32_t prosper_apr_register(const std::string& path, uint64_t size) {
    std::lock_guard lk(g_apr_mx);
    for (size_t i = 0; i < g_apr_files.size(); i++)
        if (g_apr_files[i].path == path) { g_apr_files[i].size = size; return (uint32_t)(i + 1); }
    g_apr_files.push_back({ path, size });
    return (uint32_t)g_apr_files.size();
}
// Test hook: drop all registered containers (the registry is process-global).
void prosper_apr_reset_for_test() {
    std::lock_guard lk(g_apr_mx);
    g_apr_files.clear();
}
// Find resolved host paths whose TOTAL size equals `size`. Returns the match count and sets
// *out_path to the first match. The read path may only act on an unambiguous (count==1) match:
// the APR read-request object carries the total byte count at obj+0x30 but the file id is NOT
// legible in the captured request layout (docs/UE4_APR_IOSTORE_BRINGUP.md field map, +0x00..+0x40),
// so size is the only correlation currently available. Exposed (not static) for the unit test.
int prosper_apr_match_by_size(uint64_t size, std::string* out_path) {
    std::lock_guard lk(g_apr_mx);
    int n = 0;
    for (auto& f : g_apr_files)
        if (f.size == size) { if (n++ == 0 && out_path) *out_path = f.path; }
    return n;
}
HLE(f_apr_resolve) {
    const char** paths = (const char**)P(a0);
    int count = (int)(int64_t)a1;
    uint32_t* out_ids   = (uint32_t*)P(a2);
    uint64_t* out_sizes = (uint64_t*)P(a3);
    uint32_t* out_flags = (uint32_t*)P(a4);
    if (!paths || count <= 0) return 0x80020016ull;   // EINVAL
    for (int i = 0; i < count; i++) {
        const char* gp = paths[i];
        std::string host = gp ? translate(gp) : std::string();
        uint64_t size = 0; uint32_t id = 0;
#ifndef _WIN32
        struct stat st;
        if (!host.empty() && ::stat(host.c_str(), &st) == 0) size = (uint64_t)st.st_size;
#else
        struct _stat64 st;
        if (!host.empty() && ::_stat64(host.c_str(), &st) == 0) size = (uint64_t)st.st_size;
#endif
        else { if (out_ids) out_ids[i] = 0; if (out_sizes) out_sizes[i] = 0; if (out_flags) out_flags[i] = 0;
               if (filelog()) fprintf(stderr, "[apr] resolve MISS %s\n", gp ? gp : "(null)"); continue; }
        // Warn loudly (unconditionally) when a DIFFERENT container shares this size: the read
        // path is size-keyed (see f_apr_read_submit) and will refuse such reads as ambiguous.
        std::string clash; int same_size = prosper_apr_match_by_size(size, &clash);
        id = prosper_apr_register(host, size);
        if (same_size > 0 && clash != host)
            fprintf(stderr, "[apr] WARNING: %s and %s share byte size %llu — size-keyed reads of "
                    "either will be refused as ambiguous (issue #62)\n",
                    host.c_str(), clash.c_str(), (unsigned long long)size);
        if (out_ids)   out_ids[i]   = id;
        if (out_sizes) out_sizes[i] = size;
        if (out_flags) out_flags[i] = 0;
        if (filelog()) fprintf(stderr, "[apr] resolve %s -> id=%u size=%llu\n", gp, id, (unsigned long long)size);
    }
    return 0;
}

// libSceAmpr::mQ16-QdKv7k — the APR read SUBMIT (identified by tracing readFile eboot 0x59b6110 ->
// this import). Call shape: mQ16(reqFrame, outScratchPtr /*a1*/, outRecord /*a2*/, fileId /*a3*/,
// descBuf /*a4*/, descSize /*a5: 0x90, 0x90, 0xdd in the three boot captures*/).
//
// CORRECT CONTRACT (established 2026-07-08 on this branch, by A/B experiment over three live
// reads): a3 is the APR file id from sceKernelAprResolveFilepathsToIdsAndFileSizes (read1 id=1
// global.utoc, read2 id=3 pakchunk2-ps5.utoc, read3 id=4 pakchunk1-ps5.pak), and a2 points at a
// COMPLETION RECORD the LIBRARY must fill:
//   a2+0x00  OUT data pointer  — WHERE THE LIBRARY PUT THE BYTES (library-chosen pages)
//   a2+0x08  OUT status        — 0 = success; on failure {low32 err, high32 CB offset}, exactly
//            what the guest's "Apr read failure %x at CB offset %d" fatal prints (read1 24|40,
//            read2 2b|48); checked at eboot 0x22738a5 after the guest's completion wait
//   a2+0x10  OUT bytes transferred
// Every one of those stack slots is FRAME RESIDUE at submit (read3's record held code addresses
// and stack pointers), so nothing in the record is a usable input; the byte count is the whole
// resolved file (the engine got sizes from resolve).
//
// The READ DESTINATION IS LIBRARY-CHOSEN, not an engine input. The old model treated the residue
// at a2+0 as the destination and pread() the TOC into it in place — that pointer was the freed
// resolve-path-string block sitting as the LIVE HEAD of a guest pool freelist class
// (PROSPER_APR_POOLSCAN: "dest-on-list=YES(head-eq)"; the block still held UTF-16
// "…ll/content/paks/global.utoc" residue), so the TOC magic landed in a free node's next-pointer
// and the pool pop dereferenced it -> the crash at eboot+0x2316c91. Publishing a prosper-owned
// buffer through the record instead: the engine parses the TOC from the published pointer, the
// crash vanishes, and boot advances through the next containers — proving the record-output
// model both ways. We model the Ampr engine's page pool with per-read host mmaps that are never
// freed (the engine treats the pages as library-owned). Zero-byte files (the dump's empty
// pakchunk2 placeholders) complete trivially with size 0.
// CONFIDENCE: HIGH on id=a3 + record-output via a2 (three callsites, A/B-tested); MED on
// whole-file semantics — a partial/offset read has not been observed yet (fine for TOC/pak-header
// reads; revisit at the first .ucas chunk read).
// --- Stack-argument capture for sceAmprAprCommandBufferReadFile ---------------------------------
// The NID reverses to the real Sony name (brute-forced against nid_hash):
//   mQ16-QdKv7k = sceAmprAprCommandBufferReadFile   (and the rest of the flow:
//   8aI7R7WaOlc = sceAmprCommandBufferConstructor, a8uLzYY--tM = sceAmprAprCommandBufferConstructor,
//   N-FSPA4S3nI = sceAmprCommandBufferSetBuffer, baQO9ez2gL4 = sceAmprCommandBufferReset,
//   Qs1xtplKo0U = sceAmprAprCommandBufferDestructor, GuchCTefuZw = sceAmprCommandBufferDestructor).
// ReadFile is a command-BUILDER: every read parameter is an argument of this call. Six land in
// registers (cb, out1, out2, fileId, dst?, size) — the FILE OFFSET is the 7th argument and lives
// ON THE GUEST STACK, which the plain HleFn signature never saw (that blind spot is exactly why
// pak footer reads looked offset-less). The asm entry below snapshots rsp into a TLS slot so the
// C handler can read the stack args. Both stub paths run on the HOST %fs (the swap stub switches
// before the call; the tail-jmp path never left it), so TLS is safe here.
// Stack layout at our entry (after master #61 taught the swap stub to FORWARD the first two stack
// args, re-pushing arg7/arg8 right below the call): BOTH stub paths now put the stack args at the
// same slots — arg7 at [rsp+8], arg8 at [rsp+0x10]:
//   swap stub ("push r11; push arg8; push arg7; call rax"): [rsp]=stub ret (0x6xxxxxxxx),
//                                           [rsp+8]=arg7 (forwarded), [rsp+0x10]=arg8 (forwarded)
//   tail-jmp ("jmp rax"):                   [rsp]=guest ret, [rsp+8]=arg7, [rsp+0x10]=arg8
// (Before #61 the swap path had arg7 at [rsp+0x18] behind saved-r11/guest-ret — reading that slot
// post-#61 returns the saved guest-fs base, which clamped every offset to EOF and broke the mount.)
// Ampr command-buffer address/size captured at init (hle_kernel_mem.cpp). Declared at namespace
// scope (NOT inside the extern "C" handler, where a block-scope extern would take C linkage and
// miss the mangled prosper:: definition).
extern uint64_t g_apr_last_cb, g_apr_last_cb_size;
// Mark an APR request frame eventful (async streaming read -> equeue completion event) or sync;
// consumed by the ASoW5WE-UPo submit handler (hle_kernel_mem.cpp, issue #180).
void prosper_apr_mark_eventful(uint64_t req, bool eventful);
// Tracked-mapping state probe (hle_kernel_mem.cpp): 1 = addr is inside a guest-RESERVED range
// whose pages lazy-commit on first touch. Used by the read path's dst-write to replicate the
// fault handler's commit-on-write semantics for kernel-side (process_vm_writev) copies.
extern "C" int prosper_reserved_range_state(uint64_t addr);

#ifndef _WIN32
// The entry shim passes its %rsp to the C handler as a SEVENTH argument (on the stack, per SysV).
// History: this was a plain global (`g_apr_entry_rsp`) — NOT `__thread`, because an initial-exec
// TLS slot grows the HOST binary's static TLS block and prosper's non-GUEST_FS boot aliases the
// host TCB through %fs (issue #89: one static-TLS slot crashed The Messenger's minimal boot 9/10).
// The global relied on APR reads never overlapping across threads. With APR completion EVENTS
// delivered (issue #115) the engine's loader/precacher threads DO submit concurrently, and a
// cross-thread overwrite makes the handler read arg7 (the FILE OFFSET) from another thread's
// frame — wrong-offset reads served as "OK" corrupt whatever the engine parses. Passing rsp as a
// real argument is race-free with no TLS. Alignment: at shim entry rsp%16==8; the single push
// re-aligns for the call and lands the 7th arg at the callee's [rsp+8], exactly where SysV wants
// it. CONFIDENCE: HIGH.
extern "C" uint64_t f_apr_read_submit_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5,
                                        uint64_t entry_rsp);
PROSPER_ASM_TRAMPOLINE(f_apr_read_submit_entry, f_apr_read_submit_c)
extern "C" void f_apr_read_submit_entry();
// Fetch the Nth stack argument (0-based: arg7 is n=0) of the in-flight ReadFile call.
// Both stub paths land the forwarded/natural stack args at [entry_rsp+8] (layout note above).
static uint64_t apr_stack_arg(uint64_t entry_rsp, int n) {
    if (!entry_rsp) return 0;
    return *(uint64_t*)(entry_rsp + 0x8 + (uint64_t)n * 8);
}
#endif

#ifndef _WIN32
// Cached host fd per resolved APR container path. The engine issues hundreds of small reads
// against the 2 GB pak during shader-library/asset loading; an open()+close() pair per read over
// the WSL 9p filesystem dominated the boot (~1.7 s/read live-measured). Containers are immutable
// game data, so a per-path fd held for the process lifetime is safe.
static int apr_cached_fd(const std::string& host) {
    static std::mutex mx;
    static std::map<std::string, int> fds;
    std::lock_guard<std::mutex> lk(mx);
    auto it = fds.find(host);
    if (it != fds.end()) return it->second;
    int fd = ::open(host.c_str(), O_RDONLY);
    if (fd >= 0) fds[host] = fd;
    return fd;
}

// Write `size` bytes into guest VA `dst`, committing lazy-reserved 64K pages the same way the
// SIGSEGV fault handler does. The kernel-side writev CANNOT fault through prosper's lazy-commit
// handler, so a dst inside a guest-RESERVED range whose pages were never touched reports EFAULT
// even though a real guest write would succeed. That EFAULT used to demote the read to "publish
// the prosper-owned staging pointer through the record" — and the engine later FMemory::Free()d
// that HOST pointer: "FMallocBinned3 Attempt to free an unrecognized block" (issue #88's second
// face). Commit the pages exactly like the fault handler and retry once. CONFIDENCE: HIGH (same
// policy as the lazy-commit path in exec_image_linux.cpp). Shared by the record-based
// sceAmprAprCommandBufferReadFile path and the direct vWU-odnS+fU read (issue #115).
static bool apr_write_guest_dst(uint64_t dst, void* buf, uint64_t size) {
    if (!size) return true;
    struct iovec l { buf, (size_t)size }, r { (void*)(uintptr_t)dst, (size_t)size };
    if (process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size) return true;
    bool committed = false;
    for (uint64_t p = dst & ~0xffffull; p < dst + size; p += 0x10000) {
        unsigned char vec;
        if (prosper_mincore((void*)(uintptr_t)p, 1, &vec) != 0 &&
            prosper_reserved_range_state(p) == 1 &&
            mmap((void*)(uintptr_t)p, 0x10000, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == (void*)(uintptr_t)p)
            committed = true;
    }
    if (!committed) return false;
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size;
}
#endif

#ifndef _WIN32
extern "C" uint64_t f_apr_read_submit_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5,
                                        uint64_t entry_rsp) {
#else
// The generated Windows import bridge converts the guest SysV call to Microsoft x64 and forwards
// arguments 7-9 in the host stack slots. Keep the real fixed-arity prototype here: using HleFn's
// six-argument declaration discarded the file offset (arg7), so every Windows APR request read
// from byte zero even when the engine asked for a pak footer or asset range.
extern "C" uint64_t f_apr_read_submit(uint64_t a0, uint64_t a1, uint64_t a2,
                                      uint64_t a3, uint64_t a4, uint64_t a5,
                                      uint64_t a6, uint64_t a7, uint64_t a8) {
    (void)a8;
#endif
    uint8_t* req = (uint8_t*)P(a0);
    if (!req) return 0x80020016ull;
    if (filelog()) {
        fprintf(stderr, "[apr] read-submit req=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx desc=0x%llx descsz=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
#ifndef _WIN32
        fprintf(stderr, "[apr]   stack-args a6=0x%llx a7=0x%llx a8=0x%llx a9=0x%llx (rsp=0x%llx ret=0x%llx)\n",
                (unsigned long long)apr_stack_arg(entry_rsp, 0), (unsigned long long)apr_stack_arg(entry_rsp, 1),
                (unsigned long long)apr_stack_arg(entry_rsp, 2), (unsigned long long)apr_stack_arg(entry_rsp, 3),
                (unsigned long long)entry_rsp,
                (unsigned long long)(entry_rsp ? *(uint64_t*)entry_rsp : 0));
        // Guest call chain: first eboot-range return addresses on the stack (identifies which engine
        // wrapper submitted this read — sync mount path vs async FAPREventQueue path, issue #115).
        if (entry_rsp) {
            const uint64_t* sp = (const uint64_t*)entry_rsp;
            int shown = 0;
            for (int i = 0; i < 160 && shown < 6; i++) {
                uint64_t v = sp[i];
                if (v < 0x400000000ull || v >= 0x4c0000000ull) continue;
                fprintf(stderr, "[apr]   guest-ra[%d] = eboot+0x%llx\n", i,
                        (unsigned long long)(v - 0x400000000ull));
                shown++;
            }
        }
#endif
        for (int o = 0; o < 0x48; o += 8)
            fprintf(stderr, "[apr]   req+0x%02x = 0x%016llx\n", o, (unsigned long long)*(uint64_t*)(req + o));
        // Cursor slots (a1/a2 point at them) and the 0x90-byte descriptor buffer (a4): the desc may
        // carry the REAL read layout (file offset / dest / size) — dump to check whether req+0x20 is
        // truly the engine-chosen dest or a stale value from an unpopulated Ampr begin-cursor.
        if (a1 > 0xffff) fprintf(stderr, "[apr]   *cur1 = 0x%016llx\n", (unsigned long long)*(uint64_t*)a1);
        if (a2 > 0xffff) fprintf(stderr, "[apr]   *cur2 = 0x%016llx\n", (unsigned long long)*(uint64_t*)a2);
        if (a4 > 0xffff) for (int o = 0; o < 0x90; o += 8)
            fprintf(stderr, "[apr]   desc+0x%02x = 0x%016llx\n", o, (unsigned long long)*(uint64_t*)(a4 + o));
        // The real read command lives in the Ampr command buffer registered at init (a3 of the
        // (req, cbSize, 0, cbBuf, poolCtx, 3) init call); "CB offset 40" is decimal 40 = 0x28 into it.
        // g_apr_last_cb is defined in hle_kernel_mem.cpp, which is entirely #ifdef __linux__ (the whole
        // Ampr/APR path is Linux-only), so this diagnostic must be Linux-only too or MinGW fails to
        // link (undefined reference to prosper::g_apr_last_cb).
#ifndef _WIN32
        if (g_apr_last_cb) {
            fprintf(stderr, "[apr]   cb=0x%llx size=0x%llx\n",
                    (unsigned long long)g_apr_last_cb, (unsigned long long)g_apr_last_cb_size);
            uint64_t n = g_apr_last_cb_size > 0x80 ? 0x80 : g_apr_last_cb_size;
            for (uint64_t o = 0; o < n; o += 8)
                fprintf(stderr, "[apr]   cb+0x%02llx = 0x%016llx\n", (unsigned long long)o,
                        (unsigned long long)*(uint64_t*)(g_apr_last_cb + o));
        }
#endif
    }
    // a3 is the APR file id (read1 a3=1 global.utoc, read2 a3=3 pakchunk2-ps5.utoc, read3 a3=4
    // pakchunk1-ps5.pak — read3's stack record is pure frame residue, which is what proved a3 is
    // the id and every record field is an OUTPUT; the begin-populated staging cursor sits in
    // req+0x18/+0x20 until we publish the final data pointer). A begin-cursor "multi-segment"
    // reading of req+0x00=4 / req+0x30=0x22784a4 (36,144,292) is that same residue: id 4 with the
    // whole-file model reads pakchunk1-ps5.pak (339 bytes) and the boot advances through every
    // container.
    uint64_t id   = a3;
    uint64_t dest = *(uint64_t*)(req + 0x20);   // begin staging / residue; diagnostics only
    std::string host = prosper_apr_path_for_id((uint32_t)id);
    // Fallback when the id (a3) doesn't resolve: key by the request's total byte count, but ONLY on
    // an unambiguous match — exactly one resolved container of that size (prosper_apr_match_by_size,
    // the safe-refuse helper from master #76). Ambiguous or no match -> host stays empty and the read
    // fails loudly below, rather than serving the wrong file's bytes.
    if (host.empty()) {
        std::string m;
        if (prosper_apr_match_by_size(*(uint64_t*)(req + 0x30), &m) == 1) host = m;
    }
    if (host.empty()) {
        if (filelog()) fprintf(stderr, "[apr] read-submit: no file for id=%llu\n",
                               (unsigned long long)id);
        return 0x80020016ull;    // record stays incomplete -> the engine reports this read as failed
    }
    // Read range: a5 = byte count, and the FILE OFFSET is the 7th argument (on the guest stack).
    // Verified live with exact footer math (2026-07-08): utoc header reads pass (offset=0,
    // size=0x90=TocHeaderSize); the two FPakInfo footer probes per pak pass
    // (offset=filesize-221, size=221=FPakInfo::Size8a) then (offset=filesize-222, size=222=Size9)
    // — e.g. pakchunk1-ps5.pak (339 bytes): offsets 0x76/0x75; pakchunk0-ps5.pak (2,062,854,722
    // bytes): offsets 0x7af4a965/0x7af4a964. Reads are clamped to EOF like a host pread (empty
    // pakchunk2 placeholders complete with 0 bytes, status 0 — the model the engine already
    // accepted). CONFIDENCE: HIGH (offset+size confirmed on 8 live reads across 3 file kinds).
    uint64_t fsize = 0;
    { struct stat st {}; if (::stat(host.c_str(), &st) == 0) fsize = (uint64_t)st.st_size; }
    uint64_t offset = 0;
#ifndef _WIN32
    offset = apr_stack_arg(entry_rsp, 0);
    uint64_t arg8 = apr_stack_arg(entry_rsp, 1);
    // arg8 discriminates the engine's ASYNC streaming reads from its sync (record-polled) reads —
    // the ONLY ReadFile-level difference found across 90-read boots (identical guest-RA chains):
    // the async wrapper passes a live pointer into its own frame just above the request object
    // (arg8 - req == +0x94, deterministic across runs), sync callsites leave residue in the slot.
    // Async requests must complete via an APREventQueue event (their submitter returns to the
    // thread pool; the blocked consumer's only wake-up is the listener), sync requests must NOT
    // get one (untracked tokens fault the listener's hash-miss path at eboot+0x229df3e). The mark
    // is consumed by k_apr_submit (hle_kernel_mem.cpp). Bounds: same 64 KiB stack page, above the
    // request frame, within 4 KiB — tight enough that observed sync residue (small ints, code
    // addresses, unrelated heap/stack pointers) can't satisfy it. CONFIDENCE: MED (see the
    // g_apr_eventful comment for the full evidence trail; issue #180).
    {
        bool async_notify = arg8 > a0 && arg8 - a0 < 0x1000 &&
                            (arg8 >> 16) == (a0 >> 16);
        prosper_apr_mark_eventful(a0, async_notify);
        if (filelog()) {
            fprintf(stderr, "[apr]   arg8=0x%llx -> %s\n", (unsigned long long)arg8,
                    async_notify ? "ASYNC(eventful)" : "sync");
            if (async_notify)   // dump the notify slot region for offline RE of its real layout
                for (int o = 0; o < 0x20; o += 8)
                    fprintf(stderr, "[apr]   notify+0x%02x = 0x%016llx\n", o,
                            (unsigned long long)*(uint64_t*)(uintptr_t)(arg8 + o));
        }
    }
#else
    offset = a6;
    (void)a7;
#endif
    uint64_t size = a5;
    if (offset > fsize) {
        if (filelog()) fprintf(stderr, "[apr] read-submit: offset 0x%llx past EOF (file %llu) — clamped\n",
                               (unsigned long long)offset, (unsigned long long)fsize);
        offset = fsize;
    }
    if (size > fsize - offset) size = fsize - offset;
#ifndef _WIN32
    // Fault-safe guest-memory read (unmapped guest VA -> false, never SIGSEGV in the HLE).
    auto safe_read = [](uint64_t va, void* out, size_t n) -> bool {
        struct iovec l { out, n }, r { (void*)(uintptr_t)va, n };
        return process_vm_readv(getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
    };
    auto hexdump = [](const char* tag, uint64_t va, const uint8_t* b, size_t n) {
        for (size_t o = 0; o < n; o += 0x10) {
            fprintf(stderr, "[apr]   %s+0x%02zx @0x%llx:", tag, o, (unsigned long long)(va + o));
            for (size_t i = o; i < o + 0x10 && i < n; i++) fprintf(stderr, " %02x", b[i]);
            fprintf(stderr, "\n");
        }
    };
    // DIAGNOSTIC (PROSPER_APR_DIAG): PRE-fill state of every pointer the request carries — is dest a
    // linked freelist node right now? does the seg (req+0x08) hold a scatter-gather (offset,dest,size)
    // descriptor? do the cursors/descBuf carry the real DMA destination?
    static bool aprdiag = getenv("PROSPER_APR_DIAG") != nullptr;
    if (aprdiag) {
        uint8_t b[0x90];
        if (safe_read(dest, b, 0x50)) hexdump("dest-pre", dest, b, 0x50);
        else fprintf(stderr, "[apr]   dest-pre 0x%llx UNMAPPED\n", (unsigned long long)dest);
        uint64_t seg = *(uint64_t*)(req + 0x08);
        if (seg && safe_read(seg, b, 0x60)) hexdump("seg", seg, b, 0x60);
        if (a1 && safe_read(a1, b, 0x20)) hexdump("cur1", a1, b, 0x20);
        if (a2 && safe_read(a2, b, 0x20)) hexdump("cur2", a2, b, 0x20);
        if (a4 && safe_read(a4, b, 0x90)) {
            hexdump("desc-pre", a4, b, 0x90);
            // The pak-read callsite's desc[0] points at a second heap block — follow one level
            // (candidate real parameter block: path/offset/size for the index/footer read).
            uint64_t d0 = *(uint64_t*)b;
            uint8_t b2[0x100];
            if (d0 > 0xffff && safe_read(d0, b2, 0x100)) hexdump("desc0->", d0, b2, 0x100);
        }
    }
    // DIAGNOSTIC (PROSPER_APR_POOLSCAN=0xADDR): the crash-side structure is an array of {ptr,count}
    // entries (0x20 stride). Walk each entry's pointer as a freelist chain BEFORE we fill dest and
    // report whether dest is a LIVE free node right now (the live-overlap hypothesis) or joins the
    // pool only later (a retire/recycle path).
    if (const char* ps = getenv("PROSPER_APR_POOLSCAN")) {
        uint64_t pool = strtoull(ps, nullptr, 0);
        for (int e = 0; e < 8; e++) {
            uint64_t hd[4] = {0, 0, 0, 0};
            if (!safe_read(pool + (uint64_t)e * 0x20, hd, 0x20)) {
                fprintf(stderr, "[apr] pool[%d] @0x%llx UNMAPPED\n", e,
                        (unsigned long long)(pool + (uint64_t)e * 0x20));
                break;
            }
            uint64_t n = hd[0]; int hops = 0; const char* hit = "no";
            while (n && hops < 4096) {
                if (dest >= n && dest < n + 0x50) { hit = (dest == n) ? "YES(head-eq)" : "YES(inside)"; break; }
                uint64_t nx;
                if (!safe_read(n, &nx, 8)) { hit = "walk-fault"; break; }
                n = nx; hops++;
            }
            fprintf(stderr, "[apr] pool[%d] head=0x%llx w1=0x%llx w2=0x%llx w3=0x%llx walked=%d dest-on-list=%s\n",
                    e, (unsigned long long)hd[0], (unsigned long long)hd[1],
                    (unsigned long long)hd[2], (unsigned long long)hd[3], hops, hit);
        }
    }
    // The read: fill a library-owned page-aligned buffer (models the Ampr engine's own DMA pages;
    // one per read, never freed — the engine only ever reads through the published pointer) and
    // COMPLETE the record: +0x20 = data pointer, +0x28 = 0 (success). The completion check is at
    // eboot 0x22738a5 (mov 0x30(%rbx)/0x34(%rbx), rbx = req-8).
    uint64_t rounded = (size + 0xfff) & ~0xfffull; if (!rounded) rounded = 0x1000;
    void* slot = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (slot == MAP_FAILED) return 0x80020016ull;
    ssize_t got = 0;
    if (size) {
        int fd = apr_cached_fd(host);
        if (fd < 0) { munmap(slot, rounded); return 0x80020016ull; }
        got = ::pread(fd, slot, (size_t)size, (off_t)offset);
    }
    bool ok = ((uint64_t)got == size);
    // a4 is the caller's DESTINATION buffer (the `dst` argument of
    // sceAmprAprCommandBufferReadFile) — on real hardware the DMA engine fills it and the
    // completion record's data pointer equals it. The utoc callsite consumes data via the record
    // pointer, but the pak-footer callsite reads its own dst buffer directly (with only the
    // record published to a side buffer, the correct footer bytes were provably never seen: no
    // index read followed and PreInit failed). Copy into dst fault-safely (process_vm_writev
    // refuses unmapped ranges instead of SIGSEGVing in the HLE) and publish record[0] = dst;
    // fall back to the prosper-owned buffer only if dst is absent/unmapped.
    // CONFIDENCE: MED-HIGH (dst role from the recovered real prototype + both callsites' behavior).
    bool in_dst = ok && a4 > 0xffff && apr_write_guest_dst(a4, slot, size);
    if (ok && a2) {
        // Complete the record through the caller-supplied out-pointer (a2 = &record.data; the
        // stack offsets differ per callsite, a2 is the stable handle): [0] data pointer,
        // [+8] status (0 = success; on failure holds {err, CB offset} that the fatal prints),
        // [+0x10] bytes transferred.
        *(uint64_t*)(uintptr_t)(a2 + 0x00) = in_dst ? a4 : (uint64_t)(uintptr_t)slot;
        *(uint64_t*)(uintptr_t)(a2 + 0x08) = 0;
        *(uint64_t*)(uintptr_t)(a2 + 0x10) = size;
    }
    if (!ok || in_dst) {
        munmap(slot, rounded);   // failure: record stays -> guest reports it; success-into-dst: staging no longer needed
    }
    if (filelog()) fprintf(stderr, "[apr] read-submit id=%llu %s -> dst=0x%llx(%s) off=0x%llx size=%llu got=%lld %s\n",
                   (unsigned long long)id, host.c_str(),
                   in_dst ? (unsigned long long)a4 : (unsigned long long)(uintptr_t)slot,
                   in_dst ? "guest" : "staging",
                   (unsigned long long)offset, (unsigned long long)size, (long long)got,
                   ok ? "OK" : "SHORT");
    return ok ? 0 : 0x80020016ull;
#else
    (void)dest;
    // Windows host: same record-completion + DMA-destination model over stdio. Read through a host
    // staging buffer so an invalid guest range cannot make the CRT abort the whole read before EOF.
    void* slot = ::malloc(size ? (size_t)size : 16);
    if (!slot) return 0x80020016ull;
    size_t got = 0;
    if (size) {
        FILE* f = ::fopen(host.c_str(), "rb"); if (!f) { ::free(slot); return 0x80020016ull; }
        if (_fseeki64(f, (__int64)offset, SEEK_SET) == 0)
            got = ::fread(slot, 1, (size_t)size, f);
        ::fclose(f);
    }
    if ((uint64_t)got != size) { ::free(slot); return 0x80020016ull; }
    auto write_guest_dst = [](uint64_t dst, const void* src, uint64_t bytes) -> bool {
        if (!windows_prepare_guest_write(dst, bytes)) return false;
        memcpy((void*)(uintptr_t)dst, src, (size_t)bytes);
        return true;
    };
    const bool in_dst = a4 > 0xffff && write_guest_dst(a4, slot, size);
    if (a2) {
        *(uint64_t*)(uintptr_t)(a2 + 0x00) = in_dst ? a4 : (uint64_t)(uintptr_t)slot;
        *(uint64_t*)(uintptr_t)(a2 + 0x08) = 0;
        *(uint64_t*)(uintptr_t)(a2 + 0x10) = size;
    }
    if (in_dst) ::free(slot);
    if (filelog()) fprintf(stderr,
        "[apr] read-submit id=%llu %s -> dst=0x%llx(%s) off=0x%llx size=%llu got=%llu OK\n",
        (unsigned long long)id, host.c_str(),
        in_dst ? (unsigned long long)a4 : (unsigned long long)(uintptr_t)slot,
        in_dst ? "guest" : "staging", (unsigned long long)offset,
        (unsigned long long)size, (unsigned long long)got);
    return 0;
#endif
}

#ifndef _WIN32
// libSceAmpr vWU-odnS+fU — the APR DIRECT (whole-range) async file read, issue #115. Live capture
// at the CreateGlobalShaderMap hang:
//   vWU-odnS+fU(fileId=8, dst=0x2002860000, size=0x107e3a, off=0x7adec90a, off, 5thArg=4)
// — fileId 8 = pakchunk0-ps5.pak from APR resolve, and (off,size) is EXACTLY the engine's
// GlobalShaderCache-SF_PS5.bin region inside that pak (size matched the reader global's recorded
// file size 0x107e3a). No command buffer, no completion record — and NO completion event (#208):
// the guest consumes these reads without a listener event; only the H896-bound batch channel gets
// events, carrying the guest-chosen binding tag (see hle_kernel_time.cpp for the full contract).
// a3==a4 in the only capture. CONFIDENCE: HIGH on (id,dst,size,off) — verified against the
// resolved file and the reader's recorded size.
HLE(f_apr_read_direct) {
    uint64_t id = a0, dst = a1, size = a2, offset = a3;
    std::string host = prosper_apr_path_for_id((uint32_t)id);
    if (host.empty()) {
        if (filelog()) fprintf(stderr, "[apr] read-direct: no file for id=%llu\n", (unsigned long long)id);
        return 0x80020016ull;   // EINVAL-class: engine sees a synchronous submit failure
    }
    uint64_t fsize = 0;
    { struct stat st {}; if (::stat(host.c_str(), &st) == 0) fsize = (uint64_t)st.st_size; }
    if (offset > fsize) offset = fsize;
    if (size > fsize - offset) size = fsize - offset;
    uint64_t rounded = (size + 0xfff) & ~0xfffull; if (!rounded) rounded = 0x1000;
    void* buf = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) return 0x80020016ull;
    ssize_t got = 0;
    if (size) {
        int fd = apr_cached_fd(host);
        if (fd < 0) { munmap(buf, rounded); return 0x80020016ull; }
        got = ::pread(fd, buf, (size_t)size, (off_t)offset);
    }
    bool ok = (uint64_t)got == size && (size == 0 || apr_write_guest_dst(dst, buf, size));
    munmap(buf, rounded);
    if (filelog()) fprintf(stderr, "[apr] read-direct id=%llu %s -> dst=0x%llx off=0x%llx size=%llu %s\n",
                           (unsigned long long)id, host.c_str(), (unsigned long long)dst,
                           (unsigned long long)offset, (unsigned long long)size, ok ? "OK" : "FAIL");
    if (!ok) return 0x80020016ull;
    // No completion event (issue #208). Direct reads are consumed by the guest WITHOUT a listener
    // event (live-verified: the gdb-unwedged engine streamed the whole remaining load through
    // these with no matching events in flight), and posting an invented counter would REGRESS the
    // listener's ctor-seeded per-ring last-processed (unconditional last:=cnt store at
    // eboot+0x2274143), setting up the fatal +0x229df3e range walk. Only exact H896 binding tags
    // are ever posted (k_apr_submit -> prosper_eq_post_apr_token).
    return 0;
}
// APR completion-token plumbing (hle_kernel_time.cpp) — see the k_apr_submit block in
// hle_kernel_mem.cpp for the recovered contract.
#endif

void register_file_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("fopen", f_fopen);   R("fclose", f_fclose); R("fread", f_fread);   R("fwrite", f_fwrite);
    R("fseek", f_fseek);   R("ftell", f_ftell);   R("fgets", f_fgets);   R("fflush", f_fflush);
    R("feof", f_feof);     R("ferror", f_ferror); R("setvbuf", f_setvbuf); R("rewind", f_rewind);
    R("fgetc", f_fgetc);   R("getc", f_fgetc);
    R("open", f_open);     R("close", f_close);   R("read", f_read);     R("write", f_write);
    R("lseek", f_lseek);   R("stat", f_stat);     R("fstat", f_fstat);   R("access", f_access);
    R("sceKernelOpen", k_open);  R("sceKernelClose", k_close); R("sceKernelRead", k_read);
    R("sceKernelWrite", k_write); R("sceKernelLseek", k_lseek); R("sceKernelStat", k_stat);
    R("sceKernelFtruncate", k_ftruncate);   // real resize (was fake-success -> corrupt saves)
    R("lstat", f_lstat);   R("sceKernelLstat", k_lstat);     // was MISSING -> uninitialized stat buffer
    R("fsync", f_fsync);       R("sceKernelFsync", k_fsync);       // real descriptor durability
    R("fdatasync", f_fdatasync); R("sceKernelFdatasync", k_fdatasync); // data-only durability
    R("sceKernelTruncate", k_truncate);      // path-based sibling (same corruption class)
    R("sceKernelFstat", k_fstat);
    // Low-level POSIX wrappers with the internal leading-underscore names. Real libc.prx implements
    // its stdio/file layer (fopen/fwrite/...) on top of these, so they MUST be real (were stubbed to
    // 0). Same handlers/host-fd space as the unprefixed ones, so libc's fds stay consistent.
    R("_open", f_open);    R("_close", f_close);  R("_read", f_read);    R("_write", f_write);
    R("_lseek", f_lseek);  R("_stat", f_stat);    R("_fstat", f_fstat);  R("_access", f_access);
    R("_pread", f_pread);  R("_pwrite", f_pwrite);
    // directory / unlink (real host ops, /app0-translated)
    R("pread", f_pread);          R("sceKernelPread", k_pread);
    R("pwrite", f_pwrite);        R("sceKernelPwrite", k_pwrite);
    R("mkdir", f_mkdir);          R("sceKernelMkdir", k_mkdir);
    R("rmdir", f_rmdir);          R("sceKernelRmdir", k_rmdir);
    R("unlink", f_unlink);        R("sceKernelUnlink", k_unlink);
    R("rename", f_rename);        R("sceKernelRename", k_rename);   // real move (was fake-success -> lost atomic saves)
    R("dup", f_dup);   R("sceKernelDup", f_dup);   R("dup2", f_dup2);   R("sceKernelDup2", f_dup2);   // were 0 = stdin
    R("fcntl", f_fcntl);          R("sceKernelFcntl", f_fcntl);       // FreeBSD commands/flags need translation
    R("sceKernelGetdents", k_getdents); R("getdents", f_getdents);
    // vectored IO + getdirentries — real host ops (were MISSING -> silent-EOF / empty-dir corruption trap)
    R("sceKernelReadv", k_readv);       R("readv", f_readv);
    R("sceKernelWritev", k_writev);     R("writev", f_writev);
    R("sceKernelPreadv", k_preadv);     R("preadv", f_preadv);
    R("sceKernelPwritev", k_pwritev);   R("pwritev", f_pwritev);
    R("sceKernelGetdirentries", k_getdirentries); R("getdirentries", f_getdirentries);
    // sceKernelAio* (issue #312): NIDs verified identical in shadPS4's PS4 registration table and
    // the PS5 3.20 libkernel stub dump. Raw-NID registration (names not in our NidDb).
    Hle::register_fn("vYU8P9Td2Zo", (HleFn)k_aio_init,         "sceKernelAioInitializeImpl");
    Hle::register_fn("nu4a0-arQis", (HleFn)k_aio_init,         "sceKernelAioInitializeParam");
    Hle::register_fn("9WK-vhNXimw", (HleFn)k_aio_init,         "sceKernelAioSetParam");
    Hle::register_fn("HgX7+AORI58", (HleFn)k_aio_submit_read,  "sceKernelAioSubmitReadCommands");
    Hle::register_fn("XQ8C8y+de+E", (HleFn)k_aio_submit_write, "sceKernelAioSubmitWriteCommands");
    Hle::register_fn("KOF-oJbQVvc", (HleFn)k_aio_wait,         "sceKernelAioWaitRequest");
    Hle::register_fn("2pOuoWoCxdk", (HleFn)k_aio_poll,         "sceKernelAioPollRequest");
    Hle::register_fn("5TgME6AYty4", (HleFn)k_aio_delete,       "sceKernelAioDeleteRequest");
    Hle::register_fn("fR521KIGgb8", (HleFn)k_aio_cancel,       "sceKernelAioCancelRequest");
    R("sceKernelAprResolveFilepathsToIdsAndFileSizes", f_apr_resolve);   // real APR resolve (was EINVAL)
    // libSceAmpr sceAmprAprCommandBufferReadFile (NID name recovered by brute-force). The Linux
    // entry is an asm shim that snapshots rsp so the handler can read the stack args (arg7 = file
    // offset); see f_apr_read_submit_entry above.
#ifndef _WIN32
    Hle::register_fn("mQ16-QdKv7k", (HleFn)f_apr_read_submit_entry, "sceAmprAprCommandBufferReadFile");
    // The direct (whole-range) APR read — completion via the APREventQueue (issue #115).
    Hle::register_fn("vWU-odnS+fU", (HleFn)f_apr_read_direct, "AprReadFileDirect?");
#else
    Hle::register_fn("mQ16-QdKv7k", (HleFn)f_apr_read_submit, "sceAmprAprCommandBufferReadFile");
#endif
    #undef R
}

} // namespace prosper

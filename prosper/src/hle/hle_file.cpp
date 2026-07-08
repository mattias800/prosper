// hle_file.cpp — HLE file I/O. Guest paths like "/app0/..." (the game's own data) are
// translated to the host dump directory; stdio FILE* and POSIX fd calls thunk to the
// host. Set the app0 root via set_app0_root() or the PROSPER_APP0 env var.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE   // pthread_getattr_np (bound the PREADLOG/DEEPTRACE stack walk to the real stack)
#endif
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <mutex>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#else
#include <direct.h>
#include <io.h>
#endif

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))
#define CS(x) ((const char*)(uintptr_t)(x))

namespace {
    std::string g_app0;   // host directory backing guest "/app0"
    bool filelog() { static int v = getenv("PROSPER_FILELOG") ? 1 : 0; return v; }
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
    std::string translate(const char* guest) {
        if (!guest) return {};
        std::string p = guest;
        if (g_app0.empty()) { if (const char* e = getenv("PROSPER_APP0")) g_app0 = e; }
        // Map /app0[/...] -> <root>[/...], /temp0[/...] -> scratch dir; other paths as-is.
        std::string h = (p.rfind("/app0", 0) == 0)  ? g_app0 + p.substr(5)
                      : (p.rfind("/temp0", 0) == 0) ? temp0_root() + p.substr(6)
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
HLE(f_fopen)   { std::string h = translate(CS(a0)); return (uint64_t)(uintptr_t)fopen(h.c_str(), CS(a1)); }
HLE(f_fclose)  { return a0 ? (uint64_t)(int64_t)fclose((FILE*)P(a0)) : 0; }
HLE(f_fread)   { return a3 ? (uint64_t)fread(P(a0), a1, a2, (FILE*)P(a3)) : 0; }
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
HLE(f_open)  { std::string h = translate(CS(a0)); int fd = (int)::open(h.c_str(), (int)a1, (mode_t)a2);
#ifndef _WIN32
               while (fd >= 0 && fd < 3) { int nfd = fcntl(fd, F_DUPFD, 3); ::close(fd); fd = nfd; }
#endif
               if (fd >= 0) preadlog("open", (uint64_t)fd, 0, 0); return (uint64_t)(int64_t)fd; }
HLE(f_close) { if (a0 < 3) { preadlog("close-lo-ignored", a0, 0, 0); return 0; }
               preadlog("close", a0, 0, 0); return (uint64_t)(int64_t)::close((int)a0); }
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
#endif
HLE(f_read)  { if (fdlog_on()) preadlog("read", a0, (uint64_t)::lseek((int)a0, 0, SEEK_CUR), a2);
#ifndef _WIN32
               return (uint64_t)read_full((int)a0, P(a1), (size_t)a2, false, 0);
#else
               return (uint64_t)(int64_t)::read((int)a0, P(a1), (size_t)a2);
#endif
             }
HLE(f_write) { return (uint64_t)(int64_t)::write((int)a0, P(a1), (size_t)a2); }
HLE(f_lseek) { if (fdlog_on() && ((int)a2 != SEEK_CUR || a1 != 0)) preadlog("lseek", a0, a1, (uint64_t)a2);
               return (uint64_t)(int64_t)::lseek((int)a0, (off_t)a1, (int)a2); }
#ifndef _WIN32
HLE(f_pread)  { preadlog("pread", a0, a3, a2); return (uint64_t)read_full((int)a0, P(a1), (size_t)a2, true, (off_t)a3); }
HLE(f_pwrite) { return (uint64_t)(int64_t)::pwrite((int)a0, P(a1), (size_t)a2, (off_t)a3); }
#else
HLE(f_pread)  { return (uint64_t)-1; }
HLE(f_pwrite) { return (uint64_t)-1; }
#endif
#ifndef _WIN32
HLE(f_stat)  { std::string h = translate(CS(a0)); struct stat st; int r = ::stat(h.c_str(), &st); if (r == 0 && a1) to_sce_stat(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
HLE(f_fstat) { struct stat st; int r = ::fstat((int)a0, &st); if (r == 0 && a1) to_sce_stat(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
#else
HLE(f_stat)  { std::string h = translate(CS(a0)); struct _stat64 st; int r = ::_stat64(h.c_str(), &st); if (r == 0 && a1) to_sce_stat64(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
HLE(f_fstat) { struct _stat64 st; int r = ::_fstat64((int)a0, &st); if (r == 0 && a1) to_sce_stat64(st, (uint8_t*)P(a1)); return (uint64_t)(int64_t)r; }
#endif
HLE(f_access){ std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::access(h.c_str(), (int)a1); }
HLE(f_mkdir) { std::string h = translate(CS(a0));   // sceKernelMkdir(path, mode)
#ifdef _WIN32
    return (uint64_t)(int64_t)::_mkdir(h.c_str());
#else
    int r = ::mkdir(h.c_str(), (mode_t)(a1 ? a1 : 0777));
    return (uint64_t)(int64_t)(r == 0 || errno == EEXIST ? 0 : r);   // treat "already exists" as success
#endif
}
HLE(f_rmdir) { std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::rmdir(h.c_str()); }

#ifndef _WIN32
// sceKernelGetdents(int fd, char* buf, size_t nbytes) — fill FreeBSD dirent records
// {u32 fileno; u16 reclen; u8 type; u8 namlen; char name[]} (4-aligned). Backed by the host's
// getdents64 on the SAME fd so the kernel keeps the directory cursor; Linux and BSD DT_* type
// values match. Returns bytes written (0 = end of directory), or an SCE error. UE4 (PPSA17942)
// enumerates its pak directory with this during IO-stack init.
HLE(f_getdents) {
    if (!a1 || a2 < 32) return 0x80020016ull;   // EINVAL
    struct LinuxDirent64 { uint64_t ino; int64_t off; uint16_t reclen; uint8_t type; char name[]; };
    uint8_t tmp[4096];
    size_t want = a2 < sizeof tmp ? (size_t)a2 : sizeof tmp;
    long n = syscall(SYS_getdents64, (int)a0, tmp, (unsigned)want);
    if (n < 0)  return 0x80020000ull | (uint64_t)(errno & 0xff);
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
}
#else
HLE(f_getdents) { return 0; }
#endif
HLE(f_unlink){ std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::
#ifdef _WIN32
    _unlink
#else
    unlink
#endif
    (h.c_str()); }

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
    std::mutex g_apr_mx;
    struct AprFile { std::string path; uint64_t size; };
    std::vector<AprFile> g_apr_files;   // id (1-based index) -> {host path, size}
}
// Exposed to the read path (Ampr page-read) to map an APR id back to its host file.
std::string prosper_apr_path_for_id(uint32_t id) {
    std::lock_guard<std::mutex> lk(g_apr_mx);
    return (id >= 1 && id <= g_apr_files.size()) ? g_apr_files[id - 1].path : std::string();
}
// Find the resolved host path for a given file size (the APR read-request object carries the total
// size at obj+0x30 but no directly-legible id; sizes of the boot-critical containers are distinct).
static std::string apr_path_for_size(uint64_t size) {
    std::lock_guard<std::mutex> lk(g_apr_mx);
    for (auto& f : g_apr_files) if (f.size == size) return f.path;
    return {};
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
        { std::lock_guard<std::mutex> lk(g_apr_mx); g_apr_files.push_back({ host, size }); id = (uint32_t)g_apr_files.size(); }
        if (out_ids)   out_ids[i]   = id;
        if (out_sizes) out_sizes[i] = size;
        if (out_flags) out_flags[i] = 0;
        if (filelog()) fprintf(stderr, "[apr] resolve %s -> id=%u size=%llu\n", gp, id, (unsigned long long)size);
    }
    return 0;
}

// libSceAmpr::mQ16-QdKv7k — the APR read SUBMIT (identified by tracing readFile eboot 0x59b6110 ->
// this import). Live-captured call: mQ16(readReq, &cur1, &cur2, count, outDescBuf, descSize=0x90).
// The read-request object carries the destination buffer at readReq+0x18 and the total byte count
// at readReq+0x30. Real hardware DMAs the resolved file's bytes into that buffer; we do it with a
// synchronous pread (the boot-critical global container is uncompressed — utoc header
// compressionMethodNameCount==0). Route B of docs/UE4_APR_IOSTORE_BRINGUP.md. CONFIDENCE: MED
// (object layout from live capture; file matched by size since the id isn't directly legible in
// the request object — boot container sizes are distinct).
HLE(f_apr_read_submit) {
    uint8_t* req = (uint8_t*)P(a0);
    if (!req) return 0x80020016ull;
    if (filelog()) {
        fprintf(stderr, "[apr] read-submit req=0x%llx a1=0x%llx a2=0x%llx count=0x%llx desc=0x%llx descsz=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
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
        extern uint64_t g_apr_last_cb, g_apr_last_cb_size;
        if (g_apr_last_cb) {
            fprintf(stderr, "[apr]   cb=0x%llx size=0x%llx\n",
                    (unsigned long long)g_apr_last_cb, (unsigned long long)g_apr_last_cb_size);
            uint64_t n = g_apr_last_cb_size > 0x80 ? 0x80 : g_apr_last_cb_size;
            for (uint64_t o = 0; o < n; o += 8)
                fprintf(stderr, "[apr]   cb+0x%02llx = 0x%016llx\n", (unsigned long long)o,
                        (unsigned long long)*(uint64_t*)(g_apr_last_cb + o));
        }
    }
    // req+0x20 is the mapped data buffer (guest direct/flexible memory, 0x10xxxxxxxx range); req+0x18
    // is only a small stack scratch (writing the full file there smashed the stack). req+0x30 is the
    // total byte count. CONFIDENCE: MED (field roles from live req dump).
    uint64_t dest = *(uint64_t*)(req + 0x20);
    uint64_t size = *(uint64_t*)(req + 0x30);
    if (!dest || !size) {
        // A zero-length request completes trivially: clear the pre-seeded failure status at
        // req+0x28 (low32 = error, high32 = CB offset), same as the successful-read path — leaving
        // it made the engine treat the no-op submit as "Apr read failure". CONFIDENCE: MED.
        *(uint64_t*)(req + 0x28) = 0;
        if (filelog()) fprintf(stderr, "[apr] read-submit: empty (dest=0x%llx size=0x%llx)\n",
                               (unsigned long long)dest, (unsigned long long)size);
        return 0;
    }
    std::string host = apr_path_for_size(size);
    if (host.empty()) { if (filelog()) fprintf(stderr, "[apr] read-submit: no file for size=%llu\n",
                        (unsigned long long)size); return 0x80020016ull; }
#ifndef _WIN32
    int fd = ::open(host.c_str(), O_RDONLY);
    if (fd < 0) return 0x80020016ull;
    // DIAGNOSTIC (PROSPER_APR_NOWRITE): read into a HOST scratch instead of the guest dest, to test
    // whether the engine actually reads the TOC from `dest` (req+0x20). If the boot advances the SAME
    // as writing to dest, the engine reads the TOC elsewhere and dest is the wrong buffer.
    static bool nowrite = getenv("PROSPER_APR_NOWRITE") != nullptr;
    // PROSPER_APR_WRITELEN=<n>: write only the first n bytes to the guest dest (read the rest into a
    // host scratch to keep got==size). Tests whether dest is a small header buffer that a full-file
    // write overflows into the adjacent allocator pool.
    static uint64_t wlen = getenv("PROSPER_APR_WRITELEN") ? strtoull(getenv("PROSPER_APR_WRITELEN"), nullptr, 0) : 0;
    ssize_t got;
    if (nowrite) {
        static uint8_t scratch[1 << 20];
        got = ::pread(fd, scratch, (size_t)(size < sizeof scratch ? size : sizeof scratch), 0);
    } else if (wlen && wlen < size) {
        static uint8_t scratch[1 << 20];
        ssize_t g1 = ::pread(fd, (void*)(uintptr_t)dest, (size_t)wlen, 0);
        ssize_t g2 = ::pread(fd, scratch, (size_t)(size - wlen < sizeof scratch ? size - wlen : sizeof scratch), (off_t)wlen);
        got = (g1 == (ssize_t)wlen && g2 >= 0) ? (ssize_t)(g1 + g2) : -1;
    } else {
        got = ::pread(fd, (void*)(uintptr_t)dest, (size_t)size, 0);
    }
    ::close(fd);
#else
    FILE* f = ::fopen(host.c_str(), "rb"); if (!f) return 0x80020016ull;
    size_t got = ::fread((void*)(uintptr_t)dest, 1, (size_t)size, f); ::fclose(f);
#endif
    bool ok = ((uint64_t)got == size);
    if (ok) {
        // Clear the completion-status word at req+0x28 (low32 = error code, high32 = CB offset).
        // The guest's readFile checks this AFTER waitCommandBufferCompletion — a synchronous read
        // that returns 0 but leaves the pre-seeded failure code (0x24 "Apr read failure 24 at CB
        // offset 40") there still fatals. Decoded from the check at eboot 0x22738a5
        // (mov 0x30(%rbx)/0x34(%rbx) with rbx = req-8). CONFIDENCE: MED.
        *(uint64_t*)(req + 0x28) = 0;
    }
    if (filelog()) fprintf(stderr, "[apr] read-submit %s -> dest=0x%llx size=%llu got=%lld %s\n",
                   host.c_str(), (unsigned long long)dest, (unsigned long long)size, (long long)got,
                   ok ? "OK" : "SHORT");
    // DIAGNOSTIC (PROSPER_APR_WATCHDEST): arm a HW write-watch on dest+0 so every subsequent store to
    // the block's first 8 bytes (where the freelist keeps its "next" pointer) is logged — reveals
    // whether the guest overwrites the TOC magic with a valid pointer (an aliasing/visibility bug) or
    // never re-links the block (a recycle-without-reinit bug).
    if (ok && getenv("PROSPER_APR_WATCHDEST") && g_hwwatch_hook) g_hwwatch_hook(dest);
    return ok ? 0 : 0x80020016ull;
}

void register_file_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("fopen", f_fopen);   R("fclose", f_fclose); R("fread", f_fread);   R("fwrite", f_fwrite);
    R("fseek", f_fseek);   R("ftell", f_ftell);   R("fgets", f_fgets);   R("fflush", f_fflush);
    R("feof", f_feof);     R("ferror", f_ferror); R("setvbuf", f_setvbuf); R("rewind", f_rewind);
    R("fgetc", f_fgetc);   R("getc", f_fgetc);
    R("open", f_open);     R("close", f_close);   R("read", f_read);     R("write", f_write);
    R("lseek", f_lseek);   R("stat", f_stat);     R("fstat", f_fstat);   R("access", f_access);
    R("sceKernelOpen", f_open);   R("sceKernelClose", f_close);  R("sceKernelRead", f_read);
    R("sceKernelWrite", f_write); R("sceKernelLseek", f_lseek);  R("sceKernelStat", f_stat);
    R("sceKernelFstat", f_fstat);
    // Low-level POSIX wrappers with the internal leading-underscore names. Real libc.prx implements
    // its stdio/file layer (fopen/fwrite/...) on top of these, so they MUST be real (were stubbed to
    // 0). Same handlers/host-fd space as the unprefixed ones, so libc's fds stay consistent.
    R("_open", f_open);    R("_close", f_close);  R("_read", f_read);    R("_write", f_write);
    R("_lseek", f_lseek);  R("_stat", f_stat);    R("_fstat", f_fstat);  R("_access", f_access);
    R("_pread", f_pread);  R("_pwrite", f_pwrite);
    // directory / unlink (real host ops, /app0-translated)
    R("pread", f_pread);          R("sceKernelPread", f_pread);
    R("pwrite", f_pwrite);        R("sceKernelPwrite", f_pwrite);
    R("mkdir", f_mkdir);          R("sceKernelMkdir", f_mkdir);
    R("rmdir", f_rmdir);          R("sceKernelRmdir", f_rmdir);
    R("unlink", f_unlink);        R("sceKernelUnlink", f_unlink);
    R("sceKernelGetdents", f_getdents); R("getdents", f_getdents);
    R("sceKernelAprResolveFilepathsToIdsAndFileSizes", f_apr_resolve);   // real APR resolve (was EINVAL)
    // libSceAmpr read-submit (raw NID; the APR page-read that fills the request buffer via pread).
    Hle::register_fn("mQ16-QdKv7k", (HleFn)f_apr_read_submit, "sceAmprAprReadSubmit?");
    #undef R
}

} // namespace prosper

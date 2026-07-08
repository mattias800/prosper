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
#include <sys/uio.h>     // process_vm_readv: fault-safe guest-memory reads for the APR diagnostics
#include <sys/mman.h>    // PROSPER_APR_ALTDEST: prosper-owned guest read buffer
#include <pthread.h>
#include <thread>        // PROSPER_APR_DEFER: async (real-hardware-timing) APR fill
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
#ifndef _WIN32
    if (f & 0x0004) h |= O_NONBLOCK;
    if (f & 0x0080) h |= O_SYNC;                // FreeBSD O_FSYNC
    if (f & 0x0100) h |= O_NOFOLLOW;
    if (f & 0x00020000) h |= O_DIRECTORY;
#endif
    return h;
}
HLE(f_open)  { std::string h = translate(CS(a0)); int fd = (int)::open(h.c_str(), host_open_flags(a1), (mode_t)a2);
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
// Register a resolved container and return its stable 1-based id. Re-resolving the same host path
// returns the existing id (updated size) instead of a duplicate entry — a duplicate would make
// every size-keyed read of that file look ambiguous. Exposed (not static) for the unit test.
uint32_t prosper_apr_register(const std::string& path, uint64_t size) {
    std::lock_guard<std::mutex> lk(g_apr_mx);
    for (size_t i = 0; i < g_apr_files.size(); i++)
        if (g_apr_files[i].path == path) { g_apr_files[i].size = size; return (uint32_t)(i + 1); }
    g_apr_files.push_back({ path, size });
    return (uint32_t)g_apr_files.size();
}
// Test hook: drop all registered containers (the registry is process-global).
void prosper_apr_reset_for_test() {
    std::lock_guard<std::mutex> lk(g_apr_mx);
    g_apr_files.clear();
}
// Find resolved host paths whose TOTAL size equals `size`. Returns the match count and sets
// *out_path to the first match. The read path may only act on an unambiguous (count==1) match:
// the APR read-request object carries the total byte count at obj+0x30 but the file id is NOT
// legible in the captured request layout (docs/UE4_APR_IOSTORE_BRINGUP.md field map, +0x00..+0x40),
// so size is the only correlation currently available. Exposed (not static) for the unit test.
int prosper_apr_match_by_size(uint64_t size, std::string* out_path) {
    std::lock_guard<std::mutex> lk(g_apr_mx);
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
// Tracked-mapping state probe (hle_kernel_mem.cpp): 1 = addr is inside a guest-RESERVED range
// whose pages lazy-commit on first touch. Used by the read path's dst-write to replicate the
// fault handler's commit-on-write semantics for kernel-side (process_vm_writev) copies.
extern "C" int prosper_reserved_range_state(uint64_t addr);

#ifndef _WIN32
// MUST NOT be `__thread` (issue #89). An initial-exec thread-local here (`%fs:...@tpoff`) grows the
// HOST binary's static TLS block, and prosper's guest boot aliases the host TCB through %fs (the
// guest runs on the host %fs in the non-GUEST_FS boot; the fault-recovery point was moved off
// `__thread` to gettid-keyed globals for exactly this reason). Adding one static-TLS slot shifted
// the layout enough to crash The Messenger's minimal boot before graphics init ~9 runs in 10 — a
// clean bisect to 653cf6f. A plain global is captured on the SAME thread that immediately reads it
// (the entry shim tail-jumps straight into the handler, which reads apr_stack_arg() first thing),
// so it is correct as long as APR reads don't overlap across threads — they don't in practice: the
// engine builds each sceAmprAprCommandBufferReadFile into its single Ampr command buffer and our
// completion is synchronous. CONFIDENCE: HIGH on the fix (boot restored to 10/10); MED on the
// serialized-APR assumption — if concurrent APR reads ever appear, key this by gettid (still NOT
// __thread) rather than reintroducing static TLS.
extern "C" uint64_t g_apr_entry_rsp;
uint64_t g_apr_entry_rsp = 0;
extern "C" uint64_t f_apr_read_submit_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5);
asm(".text\n"
    ".globl f_apr_read_submit_entry\n"
    ".type f_apr_read_submit_entry,@function\n"
    "f_apr_read_submit_entry:\n"
    "  movq %rsp, g_apr_entry_rsp(%rip)\n"
    "  jmp f_apr_read_submit_c\n");
extern "C" void f_apr_read_submit_entry();
// Fetch the Nth stack argument (0-based: arg7 is n=0) of the in-flight ReadFile call.
static uint64_t apr_stack_arg(int n) {
    uint64_t rsp = g_apr_entry_rsp;
    if (!rsp) return 0;
    // Both paths land the forwarded/natural stack args at [rsp+8] (see the layout note above).
    return *(uint64_t*)(rsp + 0x8 + (uint64_t)n * 8);
}
#endif

#ifndef _WIN32
extern "C" uint64_t f_apr_read_submit_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                        uint64_t a3, uint64_t a4, uint64_t a5) {
#else
HLE(f_apr_read_submit) {
#endif
    uint8_t* req = (uint8_t*)P(a0);
    if (!req) return 0x80020016ull;
    if (filelog()) {
        fprintf(stderr, "[apr] read-submit req=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx desc=0x%llx descsz=0x%llx\n",
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
#ifndef _WIN32
        fprintf(stderr, "[apr]   stack-args a6=0x%llx a7=0x%llx a8=0x%llx a9=0x%llx (rsp=0x%llx ret=0x%llx)\n",
                (unsigned long long)apr_stack_arg(0), (unsigned long long)apr_stack_arg(1),
                (unsigned long long)apr_stack_arg(2), (unsigned long long)apr_stack_arg(3),
                (unsigned long long)g_apr_entry_rsp,
                (unsigned long long)(g_apr_entry_rsp ? *(uint64_t*)g_apr_entry_rsp : 0));
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
    offset = apr_stack_arg(0);
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
        int fd = ::open(host.c_str(), O_RDONLY);
        if (fd < 0) { munmap(slot, rounded); return 0x80020016ull; }
        got = ::pread(fd, slot, (size_t)size, (off_t)offset);
        ::close(fd);
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
    bool in_dst = false;
    if (ok && a4 > 0xffff) {
        if (size) {
            struct iovec l { slot, (size_t)size }, r { (void*)(uintptr_t)a4, (size_t)size };
            in_dst = process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size;
            if (!in_dst) {
                // The kernel-side writev CANNOT fault through prosper's SIGSEGV lazy-commit
                // handler, so a dst inside a guest-RESERVED range whose pages were never touched
                // reports EFAULT even though a real guest write would succeed (the fault handler
                // would back the page). That EFAULT used to demote the read to "publish the
                // prosper-owned staging pointer through the record" — and the engine later
                // FMemory::Free()d that HOST pointer: "FMallocBinned3 Attempt to free an
                // unrecognized block" during config-file loads (issue #88's second face).
                // Replicate the guest-write semantics instead: commit the lazy 64K pages the
                // exact way the fault handler does, then retry once. CONFIDENCE: HIGH (same
                // policy as the lazy-commit path in exec_image_linux.cpp).
                bool committed = false;
                for (uint64_t p = a4 & ~0xffffull; p < a4 + size; p += 0x10000) {
                    unsigned char vec;
                    if (mincore((void*)(uintptr_t)p, 1, &vec) != 0 &&
                        prosper_reserved_range_state(p) == 1 &&
                        mmap((void*)(uintptr_t)p, 0x10000, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == (void*)(uintptr_t)p)
                        committed = true;
                }
                if (committed)
                    in_dst = process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size;
            }
        } else in_dst = true;
    }
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
    // Windows host: same record-completion model over stdio, buffer from the host heap (in-process,
    // guest-readable).
    void* slot = ::malloc(size ? (size_t)size : 16);
    if (!slot) return 0x80020016ull;
    size_t got = 0;
    if (size) {
        FILE* f = ::fopen(host.c_str(), "rb"); if (!f) { ::free(slot); return 0x80020016ull; }
        got = ::fread(slot, 1, (size_t)size, f); ::fclose(f);
    }
    if ((uint64_t)got != size) { ::free(slot); return 0x80020016ull; }
    if (a2) {
        *(uint64_t*)(uintptr_t)(a2 + 0x00) = (uint64_t)(uintptr_t)slot;
        *(uint64_t*)(uintptr_t)(a2 + 0x08) = 0;
        *(uint64_t*)(uintptr_t)(a2 + 0x10) = size;
    }
    return 0;
#endif
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
    // libSceAmpr sceAmprAprCommandBufferReadFile (NID name recovered by brute-force). The Linux
    // entry is an asm shim that snapshots rsp so the handler can read the stack args (arg7 = file
    // offset); see f_apr_read_submit_entry above.
#ifndef _WIN32
    Hle::register_fn("mQ16-QdKv7k", (HleFn)f_apr_read_submit_entry, "sceAmprAprCommandBufferReadFile");
#else
    Hle::register_fn("mQ16-QdKv7k", (HleFn)f_apr_read_submit, "sceAmprAprCommandBufferReadFile");
#endif
    #undef R
}

} // namespace prosper

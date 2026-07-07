// hle_file.cpp — HLE file I/O. Guest paths like "/app0/..." (the game's own data) are
// translated to the host dump directory; stdio FILE* and POSIX fd calls thunk to the
// host. Set the app0 root via set_app0_root() or the PROSPER_APP0 env var.
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/syscall.h>
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
    std::string translate(const char* guest) {
        if (!guest) return {};
        std::string p = guest;
        if (g_app0.empty()) { if (const char* e = getenv("PROSPER_APP0")) g_app0 = e; }
        // Map /app0[/...] -> <root>[/...]; leave other absolute paths as-is.
        std::string h = (p.rfind("/app0", 0) == 0) ? g_app0 + p.substr(5) : p;
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
static void preadlog(const char* fn, uint64_t fd, uint64_t off, uint64_t cnt) {
    if (!fdlog_on()) return;
    // resolve fd -> path
    char path[256] = "?"; char lp[64]; snprintf(lp, sizeof lp, "/proc/self/fd/%lld", (long long)fd);
    ssize_t k = readlink(lp, path, sizeof path - 1); if (k > 0) path[k] = 0; else path[0] = '?', path[1] = 0;
    const char* base = strrchr(path, '/'); base = base ? base + 1 : path;
    // top 3 distinct eboot-range return addresses on the stack (guest call chain)
    uint64_t cc[3] = {0,0,0}; int nc = 0; uint64_t* sp = (uint64_t*)__builtin_frame_address(0);
    for (int i = 0; i < 800 && nc < 3; i++) { uint64_t v = sp[i];
        if (v >= 0x400000000ull && v < 0x420000000ull) { uint64_t o = v - 0x400000000ull;
            if (nc == 0 || cc[nc-1] != o) cc[nc++] = o; } }
    fprintf(stderr, "[preadlog] %-6s %-24s fd=%lld off=0x%llx(blk %lld) cnt=0x%llx tid=%ld callers=eboot+0x%llx,0x%llx,0x%llx\n",
            fn, base, (long long)fd, (unsigned long long)off, (long long)(off / 0x10000),
            (unsigned long long)cnt, (long)syscall(SYS_gettid),
            (unsigned long long)cc[0], (unsigned long long)cc[1], (unsigned long long)cc[2]);
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
HLE(f_unlink){ std::string h = translate(CS(a0)); return (uint64_t)(int64_t)::
#ifdef _WIN32
    _unlink
#else
    unlink
#endif
    (h.c_str()); }

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
    #undef R
}

} // namespace prosper

// Signal-handler-safe raw syscalls (Linux x86-64).
//
// The Linux fault_handler runs on guest threads whose %fs base is GUEST TLS. Any glibc call
// that can fail — including mmap() and even the glibc syscall() wrapper — stores errno through
// %fs on the failure path, which faults *inside* the signal handler and kills the process with
// a nested SIGSEGV instead of reaching the fatal report (issue #1071). Helpers here issue the
// syscall instruction directly and return -errno in-register; they never touch errno or TLS.
#pragma once

// NOTE: exec_image_linux.cpp includes this header above its own platform guard and is globbed
// into prosper_core on every host, so the top level here must stay Windows/MinGW-clean (no
// <sys/mman.h> — see posix_shim.hpp). The errno/TLS-free guarantee below holds for the Linux
// x86-64 asm branch; the POSIX fallback branch is plain libc (safe there, see its comment).
#include <cerrno>
#include <cstdint>

namespace prosper {

// --- snprintf() return -> write() length: the clamp every report site needs ------------------
// snprintf() returns the length it WOULD have written, not the length it did. Handing that value
// straight to a write() length is an out-of-bounds READ of the caller's buffer on every format
// that truncates, and a negative return (encoding error) converts to an enormous size_t. Both
// matter most exactly where this is used: the fault-handler report sites run only once the
// process is already known to be broken, so the diagnostic corrupts the evidence for the crash
// it exists to describe, in the one situation where the output has to be trustworthy. (#2050;
// the class is instrument trap 109, first caught in the [labelhist] fault dumper.)
//
// This is the whole contract: given snprintf's return `n` and the destination array's `cap`,
// return how many bytes actually landed there. On truncation snprintf wrote cap-1 characters
// plus the terminating NUL, so cap-1 is the readable content.
// Async-signal-safe: integer arithmetic only — no allocation, no lock, no errno, no TLS.
// Defined above the platform guard because it is pure arithmetic: Windows/MinGW-clean, and
// unit-testable on any host (see tests/test_raw_syscall.cpp).
inline uint64_t raw_fmt_len(int n, uint64_t cap) {
    if (n < 0 || cap == 0) return 0;              // encoding error, or nowhere to have written
    const uint64_t un = (uint64_t)n;
    return un < cap ? un : cap - 1;               // truncated: cap-1 characters + the NUL
}

} // namespace prosper

#if defined(__linux__) && defined(__x86_64__)

#include <sys/mman.h>
#include <sys/syscall.h>

namespace prosper {

// Fixed, read-only, anonymous, private mapping of [addr, addr+len).
// Returns 0 on success or a negative errno value. Async-signal-safe; no TLS/errno access.
inline long raw_mmap_fixed_ro(uint64_t addr, uint64_t len) {
    long ret;
    register long r10 __asm__("r10") = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
    register long r8  __asm__("r8")  = -1;                       // fd
    register long r9  __asm__("r9")  = 0;                        // offset
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "0"((long)SYS_mmap), "D"(addr), "S"(len), "d"((long)PROT_READ),
                       "r"(r10), "r"(r8), "r"(r9)
                     : "rcx", "r11", "memory");
    if (ret < 0) return ret;                                     // kernel returns -errno directly
    return (uint64_t)ret == addr ? 0 : -EINVAL;                  // MAP_FIXED must land at addr
}

// write(2) without errno/TLS access: returns the byte count or a negative errno value.
// For handler diagnostics; callers may ignore the result (best-effort logging).
inline long raw_write(int fd, const void* buf, uint64_t len) {
    long ret;
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "0"((long)SYS_write), "D"((long)fd), "S"(buf), "d"(len)
                     : "rcx", "r11", "memory");
    return ret;
}

} // namespace prosper

#elif !defined(_WIN32)

// Non-Linux POSIX (macOS): exec_image_linux.cpp is the shared POSIX substrate and compiles
// here too. Host TLS is %gs-based on macOS x86-64, so libc's errno store never goes through
// the guest's %fs — plain libc calls are handler-safe, and these wrappers just adapt them to
// the same 0/-errno in-register contract the Linux asm versions provide.
#include <sys/mman.h>
#include <unistd.h>

namespace prosper {

inline long raw_mmap_fixed_ro(uint64_t addr, uint64_t len) {
    void* p = mmap((void*)addr, (size_t)len, PROT_READ,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (p == MAP_FAILED) return -(long)errno;
    return (uint64_t)p == addr ? 0 : -EINVAL;
}

inline long raw_write(int fd, const void* buf, uint64_t len) {
    long r = (long)write(fd, buf, (size_t)len);
    return r < 0 ? -(long)errno : r;
}

} // namespace prosper

#endif // platform

#if !defined(_WIN32)

namespace prosper {

// A silently shortened fault report is its own trap: the reader cannot tell a complete line from
// one that lost its tail, and the tail is where the interesting operand usually sits. So when the
// clamp bites, say so on the same stream rather than letting the line just end early.
// Async-signal-safe: one raw write of a static literal.
inline void raw_write_trunc_mark(int fd) {
    static const char mark[] = "...[prosper: diagnostic line truncated]\n";
    (void)raw_write(fd, mark, sizeof mark - 1);
}

// Write exactly what snprintf() placed in `buf` (an array of `cap` bytes) when it returned `n`,
// and make any loss visible. One syscall on the normal path, two only when the line truncated.
// Returns the content write's result; report sites ignore it (best-effort logging).
//
// The `cap` argument is spelled `sizeof buf` at every call site on purpose: it sits directly
// under the snprintf() that also says `sizeof buf`, so a reader can check the pair by eye and a
// buffer resize cannot leave the write behind.
inline long raw_write_fmt(int fd, const char* buf, uint64_t cap, int n) {
    const long w = raw_write(fd, buf, raw_fmt_len(n, cap));
    if (n < 0 || (uint64_t)n >= cap) raw_write_trunc_mark(fd);
    return w;
}

} // namespace prosper

#endif // !_WIN32

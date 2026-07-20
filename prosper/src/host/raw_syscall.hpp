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

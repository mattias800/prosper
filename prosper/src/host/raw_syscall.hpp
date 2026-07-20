// Signal-handler-safe raw syscalls (Linux x86-64).
//
// The Linux fault_handler runs on guest threads whose %fs base is GUEST TLS. Any glibc call
// that can fail — including mmap() and even the glibc syscall() wrapper — stores errno through
// %fs on the failure path, which faults *inside* the signal handler and kills the process with
// a nested SIGSEGV instead of reaching the fatal report (issue #1071). Helpers here issue the
// syscall instruction directly and return -errno in-register; they never touch errno or TLS.
#pragma once

#if defined(__linux__) && defined(__x86_64__)

#include <cerrno>
#include <cstdint>
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

#endif // __linux__ && __x86_64__

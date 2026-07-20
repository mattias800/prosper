// raw_mmap_fixed_ro: the %fs-safe fixed read-only anonymous mapping used by the Linux
// fault_handler's PROSPER_NULL_PAGE backing (issue #1071). The handler runs on guest threads
// whose %fs base is guest TLS, so neither glibc mmap() nor glibc syscall() may be used there:
// both store errno through %fs on failure, which faults inside the signal handler and kills
// the process. This helper must return -errno in-register and leave host errno/TLS untouched.
#include "../src/host/raw_syscall.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

#if !defined(__linux__) || !defined(__x86_64__)
int main() { printf("raw_syscall: skipped (Linux x86-64 only)\n"); return 0; }
#else
int main() {
    // --- success: map a fixed read-only zero page at a known-free high address -------------
    // Reserve an address range via the normal allocator, then release it and re-map it raw.
    void* probe = mmap(nullptr, 0x2000, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(probe != MAP_FAILED, "probe reservation failed: %s", strerror(errno));
    uint64_t addr = (uint64_t)probe;
    munmap(probe, 0x2000);

    errno = 999;                                   // sentinel: raw path must not touch errno
    long r = prosper::raw_mmap_fixed_ro(addr, 0x1000);
    CHECK(r == 0, "expected success mapping at 0x%llx, got %ld", (unsigned long long)addr, r);
    CHECK(errno == 999, "errno clobbered on success path: %d", errno);
    if (r == 0) {
        const volatile uint8_t* p = (const uint8_t*)addr;
        uint8_t acc = 0;
        for (int i = 0; i < 0x1000; i++) acc |= p[i];
        CHECK(acc == 0, "mapped page not zero-filled (acc=0x%x)", acc);
        munmap((void*)addr, 0x1000);
    }

    // --- failure: invalid length must report -EINVAL in-register, errno untouched ----------
    errno = 999;
    r = prosper::raw_mmap_fixed_ro(addr, 0);
    CHECK(r == -EINVAL, "expected -EINVAL for len=0, got %ld", r);
    CHECK(errno == 999, "errno clobbered on failure path: %d", errno);

    // --- failure: unaligned fixed address must report -EINVAL, errno untouched -------------
    errno = 999;
    r = prosper::raw_mmap_fixed_ro(addr + 1, 0x1000);
    CHECK(r == -EINVAL, "expected -EINVAL for unaligned addr, got %ld", r);
    CHECK(errno == 999, "errno clobbered on failure path: %d", errno);

    // --- raw_write: success delivers the bytes, returns the count, errno untouched ---------
    int fds[2];
    CHECK(pipe(fds) == 0, "pipe failed: %s", strerror(errno));
    errno = 999;
    r = prosper::raw_write(fds[1], "abc", 3);
    CHECK(errno == 999, "errno clobbered on raw_write success: %d", errno);
    CHECK(r == 3, "expected 3 bytes written, got %ld", r);
    if (r == 3) {                          // only read what was actually written — a blocking
        char rb[4] = {0};                  // read on an empty pipe would hang a regressed build
        CHECK(read(fds[0], rb, 3) == 3 && memcmp(rb, "abc", 3) == 0,
              "pipe did not deliver raw_write bytes (got '%s')", rb);
    }
    close(fds[0]); close(fds[1]);

    // --- raw_write: bad fd reports -EBADF in-register, errno untouched ---------------------
    errno = 999;
    r = prosper::raw_write(-1, "x", 1);
    CHECK(r == -EBADF, "expected -EBADF for bad fd, got %ld", r);
    CHECK(errno == 999, "errno clobbered on raw_write failure: %d", errno);

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("raw_syscall: all checks passed\n");
    return 0;
}
#endif

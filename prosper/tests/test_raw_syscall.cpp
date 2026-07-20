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

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n  ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

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

    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("raw_syscall: all checks passed\n");
    return 0;
}

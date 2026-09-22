// test_guest_memory_copy (#3734) — the one bounds policy for copying between guest and host memory.
//
// The fixture is a real mapping with a hole in it: [RW page][RW page][NO-ACCESS page][RW page], plus
// (POSIX) a read-only page for the write arms. Every arm asserts the documented policy in
// host/memory/guest_memory_copy.hpp: never faults, exact is all-or-nothing, prefix is exact to the
// byte, zero bytes is trivially fine, NULL is never a buffer, a wrapping range is refused, errno is
// preserved. A regression that faulted would take the test process down rather than print [FAIL],
// which ctest reports just the same.
#include "host/memory/guest_memory_copy.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

using namespace prosper::host;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr size_t kPage = 0x1000;

uint8_t* map_pages(size_t n) {
#ifdef _WIN32
    return static_cast<uint8_t*>(VirtualAlloc(nullptr, n * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    void* p = mmap(nullptr, n * kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
#endif
}
bool no_access(uint8_t* page) {
#ifdef _WIN32
    DWORD old = 0;
    return VirtualProtect(page, kPage, PAGE_NOACCESS, &old) != 0;
#else
    return mprotect(page, kPage, PROT_NONE) == 0;
#endif
}
uint64_t addr(const void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); }
}  // namespace

int main() {
    printf("== test_guest_memory_copy ==\n");
    // [0] RW, [1] RW, [2] NO ACCESS, [3] RW, [4] RO (POSIX only)
    uint8_t* base = map_pages(5);
    if (!base) { printf("  [FAIL] fixture mapping\n"); return 1; }
    for (size_t i = 0; i < 2 * kPage; ++i) base[i] = static_cast<uint8_t>(i * 7 + 3);
    CHECK(no_access(base + 2 * kPage), "CONTROL: the hole page is made inaccessible");
#ifndef _WIN32
    CHECK(mprotect(base + 4 * kPage, kPage, PROT_READ) == 0, "CONTROL: the read-only page is made read-only");
#endif

    std::vector<uint8_t> buf(4 * kPage, 0xEE);

    // ---- read_exact
    CHECK(guest_read_exact(addr(base + 100), buf.data(), 2 * kPage - 100) &&
              memcmp(buf.data(), base + 100, 2 * kPage - 100) == 0,
          "read_exact copies a mapped range spanning a page boundary");
    CHECK(!guest_read_exact(addr(base + 100), buf.data(), 2 * kPage),
          "read_exact refuses a range that runs into an inaccessible page (and does not fault)");
    CHECK(!guest_read_exact(addr(base + 2 * kPage), buf.data(), 8), "read_exact refuses a wholly inaccessible range");

    // ---- read_prefix
    std::fill(buf.begin(), buf.end(), 0xEE);
    const size_t got = guest_read_prefix(addr(base + 100), buf.data(), 3 * kPage);
    CHECK(got == 2 * kPage - 100, "read_prefix returns exactly the readable prefix, to the byte");
    CHECK(memcmp(buf.data(), base + 100, got) == 0, "...and those bytes are the guest's");
    CHECK(guest_read_prefix(addr(base + 2 * kPage + 16), buf.data(), 64) == 0,
          "read_prefix of an inaccessible range is 0");
    CHECK(guest_read_prefix(addr(base + 3 * kPage), buf.data(), kPage) == kPage,
          "read_prefix of a fully readable page is the whole page");

    // ---- zero bytes, NULL, wrap
    std::fill(buf.begin(), buf.end(), 0xEE);
    CHECK(guest_read_exact(addr(base), buf.data(), 0) && guest_read_prefix(addr(base), buf.data(), 0) == 0 &&
              guest_write_exact(addr(base), buf.data(), 0) && buf[0] == 0xEE,
          "zero bytes is a trivially successful copy that touches nothing");
    CHECK(!guest_read_exact(0, buf.data(), 8) && guest_read_prefix(0, buf.data(), 8) == 0 &&
              !guest_write_exact(0, buf.data(), 8),
          "a NULL guest address is never a buffer");
    CHECK(!guest_read_exact(0, buf.data(), 0) && !guest_write_exact(0, buf.data(), 0),
          "...not even for zero bytes: a NULL out-pointer is always an error");
    CHECK(guest_read_exact(addr(base), nullptr, 0) && guest_write_exact(addr(base), nullptr, 0),
          "a zero-byte copy with a NULL HOST buffer is the memcpy convention: trivially fine");
    CHECK(!guest_read_exact(addr(base), nullptr, 8) && guest_read_prefix(addr(base), nullptr, 8) == 0 &&
              !guest_write_exact(addr(base), nullptr, 8),
          "a NULL host buffer copies nothing");
    CHECK(!guest_read_exact(UINT64_MAX - 7, buf.data(), 16) && !guest_write_exact(UINT64_MAX - 7, buf.data(), 16),
          "a range that wraps past the top of the address space is refused");
    CHECK(guest_read_prefix(UINT64_MAX - 7, buf.data(), 16) == 0,
          "read_prefix clamps at the top of the address space instead of wrapping (and does not fault)");

    // ---- write_exact
    const uint8_t pattern[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    CHECK(guest_write_exact(addr(base + kPage - 16), pattern, sizeof pattern) &&
              memcmp(base + kPage - 16, pattern, sizeof pattern) == 0,
          "write_exact writes a range spanning a page boundary");
    CHECK(!guest_write_exact(addr(base + 2 * kPage - 16), pattern, sizeof pattern),
          "write_exact refuses a range that runs into an inaccessible page (and does not fault)");
#ifndef _WIN32
    CHECK(!guest_write_exact(addr(base + 4 * kPage), pattern, sizeof pattern) && base[4 * kPage] == 0,
          "write_exact refuses a read-only page and leaves it unchanged");
#endif

    // ---- errno is preserved across a failing copy
    errno = 1234;
    (void)guest_read_exact(addr(base + 2 * kPage), buf.data(), 8);
    (void)guest_read_prefix(addr(base + 2 * kPage), buf.data(), 8);
    (void)guest_write_exact(addr(base + 2 * kPage), pattern, 8);
    CHECK(errno == 1234, "a failing copy leaves errno as the caller had it");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

#include "host/guest_write_watch.hpp"

#include <cstdint>
#include <cstdio>

using prosper::host::GuestWriteWatch;
using prosper::host::GuestWriteWatchQuery;

namespace {
int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; } \
} while (0)
} // namespace

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

int main() {
    constexpr uint64_t kSize = 0x10000;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                        0, static_cast<DWORD>(kSize), nullptr);
    CHECK(section != nullptr, "paging-file section created");
    if (!section) return 1;
    auto* first = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    auto* second = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    CHECK(first && second, "two aliases mapped");
    if (!first || !second) {
        if (first) UnmapViewOfFile(first);
        if (second) UnmapViewOfFile(second);
        CloseHandle(section);
        return 1;
    }

    constexpr uint64_t kPhys = 0x10000000;
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(first), kSize, kPhys, PAGE_READWRITE);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(second), kSize, kPhys, PAGE_READWRITE);

    first[0x180] = 0x11;
    GuestWriteWatch watch = GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(first + 0x100), 0x2100);
    CHECK(!static_cast<bool>(watch),
          "Windows rejects page-fault watches that can corrupt the guest SysV red zone");
    CHECK(watch.query() == GuestWriteWatchQuery::Unknown,
          "unsupported watch selects the caller's exact-comparison fallback");
    CHECK(!watch.rearm(), "unsupported watch cannot arm read-only fault pages");

    second[0x180] = 0x22;
    CHECK(first[0x180] == 0x22, "write through second alias updates first alias");
    prosper::host::guest_write_watch_notify_physical_write(kPhys + 0x1000, 0x1000);
    CHECK(watch.query() == GuestWriteWatchQuery::Unknown,
          "physical writes leave an unsupported watch on the exact fallback");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(second), kSize);
    watch.reset();

    first[0x180] = 0x44;
    CHECK(second[0x180] == 0x44, "invalidation restores writable page protections");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(first), kSize);
    UnmapViewOfFile(second);
    UnmapViewOfFile(first);
    CloseHandle(section);

    const auto stats = prosper::host::guest_write_watch_stats();
    CHECK(stats.create_attempts >= 1, "unsupported watch attempts remain observable");
    CHECK(stats.registrations == 0 && stats.faults == 0 && stats.rearms == 0,
          "safe Windows fallback never arms or handles guest page faults");
    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}

#else   // ---- Linux: exercise the real mprotect + SIGSEGV dirty-tracking (#1144) ------------------

#include <csignal>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace {
// dmem is section-backed (one memfd, MAP_SHARED at multiple VAs) — mirror that so the alias handling
// is exercised, not just a single mapping.
constexpr uint32_t kCpuRw = 0x3;   // SCE CPU_READ|CPU_WRITE (see host_prot in guest_write_watch.cpp)

void seg_handler(int sig, siginfo_t* si, void*) {
    if (sig == SIGSEGV && si->si_addr &&
        prosper::host::guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(si->si_addr)))
        return;                                   // resolved: page restored writable, store re-runs
    // A fault we did not arm: fail loudly rather than loop forever.
    const char m[] = "FAIL: unexpected SIGSEGV not owned by the write-watch\n";
    (void)!write(2, m, sizeof m - 1);
    _exit(2);
}
} // namespace

int main() {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t size = page * 4;

    int fd = memfd_create("prosper-ww-test", 0);
    CHECK(fd >= 0, "memfd created");
    if (fd < 0) return 1;
    CHECK(ftruncate(fd, static_cast<off_t>(size)) == 0, "memfd sized");

    // Two MAP_SHARED aliases of the same physical (memfd offset 0) range.
    auto* a = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    auto* b = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK(a != MAP_FAILED && b != MAP_FAILED, "two aliases mapped");
    if (a == MAP_FAILED || b == MAP_FAILED) return 1;
    constexpr uint64_t kPhys = 0x20000;   // arbitrary distinct phys base for the test

    // Install the SIGSEGV handler on a sigaltstack (SA_ONSTACK) — the red-zone-safe configuration the
    // real path requires; then tell the write-watch that onstack holds so create() will arm.
    static uint8_t altbuf[128 * 1024];   // ample; SIGSTKSZ is not a compile-time constant on glibc
    stack_t ss{}; ss.ss_sp = altbuf; ss.ss_size = sizeof altbuf; ss.ss_flags = 0;
    CHECK(sigaltstack(&ss, nullptr) == 0, "sigaltstack installed");
    struct sigaction sa{}; sa.sa_sigaction = seg_handler; sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    CHECK(sigaction(SIGSEGV, &sa, nullptr) == 0, "SIGSEGV handler installed");

    // Gate check: without onstack, create() must refuse (safe fallback).
    prosper::host::guest_write_watch_set_fault_onstack(false);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(a), size, kPhys, kCpuRw);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(b), size, kPhys, kCpuRw);
    GuestWriteWatch off = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), size);
    CHECK(!static_cast<bool>(off), "create refuses to arm when the handler is not on the sigaltstack");

    prosper::host::guest_write_watch_set_fault_onstack(true);

    // Arm a watch over the whole range and confirm it starts Unchanged.
    GuestWriteWatch watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), size);
    CHECK(static_cast<bool>(watch), "watch armed over a fully-aliased dmem range");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "freshly armed watch reads Unchanged");

    // Write through alias A -> faults -> handler restores write + marks dirty; the byte must land.
    a[page + 0x40] = 0x5a;
    CHECK(a[page + 0x40] == 0x5a, "write through alias A lands (no lost store)");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty, "watch reads Dirty after a CPU write");

    // Re-arm, don't write -> Unchanged again.
    CHECK(watch.rearm(), "rearm succeeds");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "no write since rearm -> Unchanged");

    // ALIAS COVERAGE: write through the OTHER alias B (same phys) must also be caught (both aliases
    // were armed). This is the correctness crux — miss it and the renderer trusts a stale texture.
    b[page * 2 + 0x8] = 0xa5;
    CHECK(b[page * 2 + 0x8] == 0xa5, "write through alias B lands");
    CHECK(a[page * 2 + 0x8] == 0xa5, "aliases share storage");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "write through a SECOND alias of the same phys is caught");

    // A physical-write notification (GPU write) invalidates too.
    CHECK(watch.rearm(), "rearm before physical-write test");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean before physical write");
    prosper::host::guest_write_watch_notify_physical_write(kPhys + page, page);
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty, "physical write marks the watch Dirty");

    // A watch over a range with NO known mapping must return Unknown (exact fallback), not arm.
    GuestWriteWatch none = GuestWriteWatch::create(0x9000000000ULL, page);
    CHECK(!static_cast<bool>(none), "create over an unmapped range yields Unknown");

    watch.reset();
    // After reset the pages are writable and stores fault no more (the handler would _exit(2) if a
    // stale armed page remained). Touch both aliases to prove protections were fully restored.
    a[0x10] = 0x1; b[page + 0x10] = 0x2;
    CHECK(a[0x10] == 0x1 && b[page + 0x10] == 0x2, "reset restored writable protections");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(a), size);
    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(b), size);

    const auto stats = prosper::host::guest_write_watch_stats();
    CHECK(stats.registrations >= 1, "at least one watch armed");
    CHECK(stats.faults >= 2, "both alias-A and alias-B write faults were handled");
    CHECK(stats.rearms >= 2, "rearm path exercised");

    munmap(a, size); munmap(b, size); close(fd);
    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}

#endif

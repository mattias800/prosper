// #3681: the write-watch lookup indexes, and the propagation they replace a walk with.
//
// query()/rearm() no longer re-derive "did anything covered change" by walking every page, and
// notify_gpu_write()/notify_host_write() no longer visit every registration/page in range. Each of
// those is a narrowing of WHICH entries are examined; none of them may change an ANSWER.
//
// The cases below are built to fail against a plausible wrong implementation rather than to
// re-confirm the happy path:
//
//   * a registration on one alias must go Dirty when a DIFFERENT guest VA aliasing the same physical
//     page is written. This is the case a VA-range shortcut gets wrong and a page-keyed inverse index
//     gets right, and it is the reason the propagation hangs off WatchedPage.
//   * a GPU write strictly INSIDE a long registration must dirty it, though that registration's start
//     is in neither the written chunk nor any chunk the write touches.
//   * a registration whose end is exactly the write's start must NOT be dirtied (half-open).
//   * the audit must be shown to FIRE on a hand-built desynchronized state, or its zero on a real
//     title says nothing.

#if !defined(__linux__)
int main() { return 0; }
#else

#include "host/memory/guest_write_watch.hpp"
#include "host/image/exec_image.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/memfd.h>
#include <sys/syscall.h>

namespace {

int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); ++failures; } \
    else { std::fprintf(stderr, "ok: %s\n", (msg)); } \
} while (0)

constexpr uint32_t kCpuRw = 0x3;

int make_memfd(const char* name, size_t bytes) {
    const int fd = static_cast<int>(syscall(SYS_memfd_create, name, 0u));
    if (fd < 0) return -1;
    if (ftruncate(fd, static_cast<off_t>(bytes)) != 0) { close(fd); return -1; }
    return fd;
}

}  // namespace

int main() {
    using namespace prosper::host;

    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    // 2 MiB is kIndexChunk. Span several chunks so the bucketing is actually exercised rather than
    // every registration landing in one bucket, where a broken index would still pass.
    const size_t span = 8u * 1024u * 1024u;

    CHECK(setenv("PROSPER_WRITE_WATCH_MAX_KB", "65536", 1) == 0, "watch size policy raised");
    CHECK(setenv("PROSPER_WATCH_QUERY_AUDIT", "1", 1) == 0, "query audit armed for the whole run");
    CHECK(unsetenv("PROSPER_FAULT_NO_ONSTACK") == 0, "production signal path");
    prosper::install_trap_handler();
    guest_write_watch_set_fault_onstack(true);

    const int fd = make_memfd("prosper-ww-index", span);
    CHECK(fd >= 0, "memfd created");
    if (fd < 0) return 1;

    // Two MAP_SHARED aliases of one physical range: the same bytes at two guest VAs.
    auto* a = static_cast<uint8_t*>(mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    auto* b = static_cast<uint8_t*>(mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    CHECK(a != MAP_FAILED && b != MAP_FAILED, "two aliases mapped");
    if (a == MAP_FAILED || b == MAP_FAILED) return 1;

    constexpr uint64_t kPhys = 0x40000000;
    guest_write_watch_notify_direct_mapping_added(reinterpret_cast<uint64_t>(a), span, kPhys, kCpuRw);
    guest_write_watch_notify_direct_mapping_added(reinterpret_cast<uint64_t>(b), span, kPhys, kCpuRw);

    // ---------------------------------------------------------------- alias propagation
    {
        // Watch alias A only. Then write through alias B. The bytes A watches HAVE changed, because
        // both VAs name one physical page -- so A must read Dirty even though nothing touched A's
        // own address range. An index keyed on the written VA would find no registration here.
        GuestWriteWatch on_a = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), span);
        CHECK(static_cast<bool>(on_a), "watch created over alias A");
        CHECK(on_a.query() == GuestWriteWatchQuery::Unchanged, "fresh watch is clean");

        b[3u * 1024u * 1024u + 64u] = 0x5a;   // a different VA, the same physical page

        CHECK(on_a.query() == GuestWriteWatchQuery::Dirty,
              "a store through the OTHER alias dirties the watch on this one");
        CHECK(b[3u * 1024u * 1024u + 64u] == 0x5a, "the aliased store landed");
        CHECK(on_a.rearm(), "rearm after the aliased store");
        CHECK(on_a.query() == GuestWriteWatchQuery::Unchanged, "rearm restores clean");
        on_a.reset();
    }

    // ------------------------------------------------------- containing interval, hit from inside
    {
        // The registration spans the whole 8 MiB (four 2 MiB chunks). The GPU write is 4 KiB in the
        // FOURTH chunk: the registration's start is nowhere near it. An index ordered by start alone,
        // or one that only records a registration in its first chunk, reports no overlap here.
        GuestWriteWatch wide = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), span);
        CHECK(static_cast<bool>(wide), "watch created over the full span");
        CHECK(wide.query() == GuestWriteWatchQuery::Unchanged, "full-span watch is clean");

        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + 7u * 1024u * 1024u, 4096);
        CHECK(wide.query() == GuestWriteWatchQuery::Dirty,
              "a GPU write inside a containing registration dirties it");
        CHECK(wide.rearm() && wide.query() == GuestWriteWatchQuery::Unchanged,
              "rearm clears a GPU-write dirty");

        // Half-open: a write that ENDS exactly where the registration begins does not overlap it,
        // and one that BEGINS exactly at its end does not either.
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) - 4096, 4096);
        CHECK(wide.query() == GuestWriteWatchQuery::Unchanged,
              "an adjacent write below the registration does not dirty it");
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + span, 4096);
        CHECK(wide.query() == GuestWriteWatchQuery::Unchanged,
              "an adjacent write above the registration does not dirty it");

        // The last byte of the registration is inside it; the first byte after it is not.
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + span - 1, 1);
        CHECK(wide.query() == GuestWriteWatchQuery::Dirty,
              "the final byte of the registration is covered");
        CHECK(wide.rearm(), "rearm after the boundary write");
        wide.reset();
    }

    // --------------------------------------------------- release leaves the index, ids do not leak
    {
        // Create and destroy a registration over the same range, then prove a GPU write to that range
        // neither crashes nor resurrects the dead id, and that a fresh registration still works. A
        // bucket left holding a released id would be a dangling lookup on the next notification.
        {
            GuestWriteWatch transient = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), span);
            CHECK(static_cast<bool>(transient), "transient watch created");
        }
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + 1024, 4096);

        GuestWriteWatch fresh = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), span);
        CHECK(static_cast<bool>(fresh), "a new watch over the released range is created");
        CHECK(fresh.query() == GuestWriteWatchQuery::Unchanged,
              "the new watch is clean: it did not inherit the released registration's state");
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + 1024, 4096);
        CHECK(fresh.query() == GuestWriteWatchQuery::Dirty, "the new watch is still reachable");
        fresh.reset();
    }

    // ------------------------------------------- two registrations over one page, independent state
    {
        // Overlapping watches share their WatchedPages. Dirtying one must not dirty the other, and
        // rearming one must not silently clear the other -- the inverse index carries both ids on the
        // shared page, so an implementation that stored a single owner would fail one of these.
        GuestWriteWatch left = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), 4u * 1024u * 1024u);
        GuestWriteWatch right = GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(a) + 2u * 1024u * 1024u, 4u * 1024u * 1024u);
        CHECK(static_cast<bool>(left) && static_cast<bool>(right), "two overlapping watches created");
        CHECK(left.query() == GuestWriteWatchQuery::Unchanged &&
              right.query() == GuestWriteWatchQuery::Unchanged, "both start clean");

        // A store in the overlap must dirty BOTH.
        a[3u * 1024u * 1024u] = 0x11;
        CHECK(left.query() == GuestWriteWatchQuery::Dirty &&
              right.query() == GuestWriteWatchQuery::Dirty,
              "a store in the shared region dirties both registrations");

        CHECK(left.rearm(), "left rearms");
        CHECK(left.query() == GuestWriteWatchQuery::Unchanged, "left is clean after its own rearm");
        CHECK(right.query() == GuestWriteWatchQuery::Dirty,
              "rearming one registration does not clear its neighbour");
        CHECK(right.rearm() && right.query() == GuestWriteWatchQuery::Unchanged, "right rearms too");

        // A store only in left's exclusive region must leave right clean.
        a[1024] = 0x22;
        CHECK(left.query() == GuestWriteWatchQuery::Dirty, "exclusive store dirties its own watch");
        CHECK(right.query() == GuestWriteWatchQuery::Unchanged,
              "a store outside the neighbour's range leaves it clean");
        left.reset();
        right.reset();
    }

    // -------------------------------------------------------- the audit's own positive control
    {
        // Everything above ran with PROSPER_WATCH_QUERY_AUDIT armed and reported no stale registration.
        // That is only evidence if the audit can fire, so build the defect by hand: bump a covered
        // page's generation WITHOUT propagating, which is precisely a missed propagation site.
        const auto before = guest_write_watch_stats();
        CHECK(before.query_audit_stale == 0,
              "no stale-clean registration was observed by any case above");

        GuestWriteWatch watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(a), span);
        CHECK(static_cast<bool>(watch), "watch created for the audit control");
        CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean before desynchronization");

        CHECK(guest_write_watch_desynchronize_for_test(reinterpret_cast<uint64_t>(a) + 5u * 1024u * 1024u),
              "a covered page was desynchronized by hand");
        // The flag still says clean; the walk the audit performs says dirty. That disagreement is the
        // class the audit exists to report, and this is the arm that proves it reports it.
        (void)watch.query();
        const auto after = guest_write_watch_stats();
        CHECK(after.query_audit_stale > before.query_audit_stale,
              "the audit REPORTS a hand-built stale-clean registration");
        watch.reset();
    }

    guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(a), span);
    guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(b), span);
    munmap(a, span);
    munmap(b, span);
    close(fd);

    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}

#endif

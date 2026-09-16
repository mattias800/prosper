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
//
// PROSPER_WATCH_QUERY_AUDIT is armed by CTest, in one arm of two, rather than by a setenv in main().
// watch_query_audit_enabled() latches into a function-local static on first call, so a setenv here
// would be correct only for as long as nobody moved a query above it -- an ordering argument, which
// is the failure #2214 is about. Arming it outside the process removes the argument, and running the
// same cases with it OFF is what gives the SHIPPED path its own coverage: every assertion below
// except the audit control itself must hold identically in both arms.

#if !defined(__linux__)
int main() { return 0; }
#else

#include "host/memory/guest_write_watch.hpp"
#include "host/image/exec_image.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>
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

    // 256 MiB: large enough to admit the >128 MiB registration the wide-bucket case needs.
    CHECK(setenv("PROSPER_WRITE_WATCH_MAX_KB", "262144", 1) == 0, "watch size policy raised");
    const bool auditing = std::getenv("PROSPER_WATCH_QUERY_AUDIT") != nullptr;
    std::fprintf(stderr, "arm: query audit %s\n", auditing ? "ON" : "off (shipped path)");
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
        // neither crashes nor resurrects the dead id, and that a fresh registration still works.
        //
        // What this does NOT establish, stated because the obvious reading is wrong: ids come from
        // ++w.next_id and are never reused, and consider() resolves through registrations.find(), so
        // a bucket entry left behind by a release can never bind to a different registration. Deleting
        // unindex_registration_locked would still pass this case -- its only consequence is unbounded
        // growth of registration_chunks, which nothing here can observe. This is a lifecycle smoke
        // test, not a discriminator for the unindex call.
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

    // ------------------------------------ a protection change marks, outside set_pages_armed
    {
        // guest_write_watch_notify_direct_mapping_protection sets coverage_incomplete DIRECTLY on the
        // page rather than through set_pages_armed -- a fourth way a registration becomes Dirty, and
        // the one the first version of this change missed. The old per-query walk saw it for free
        // because it read coverage_incomplete itself; the flag sees it only if that site marks.
        //
        // The obvious version of this case does not discriminate, and the first draft of it did not:
        // the same function's SECOND loop clears mapping_valid for every registration whose pages
        // overlap a live AliasRange, and query() tests mapping_valid first, so the missing mark is
        // masked and the case passes with the fix removed. What separates them is that the two loops
        // are keyed differently -- loop 1 on a surviving PageAlias, loop 2 on a live AliasRange -- so
        // the state to build is one where a page alias outlives its AliasRange.
        //
        // notify_direct_mapping_removed does exactly that: purge_va_range_locked drops only the page
        // aliases inside the removed range, but erases every AliasRange that merely OVERLAPS it,
        // whole. So unmapping the first page of a two-page mapping orphans the second page's alias.
        const size_t two_pages = page * 2;
        const int orphan_fd = make_memfd("prosper-ww-orphan", two_pages);
        CHECK(orphan_fd >= 0, "orphan memfd created");
        if (orphan_fd >= 0) {
            auto* m = static_cast<uint8_t*>(
                mmap(nullptr, two_pages, PROT_READ | PROT_WRITE, MAP_SHARED, orphan_fd, 0));
            CHECK(m != MAP_FAILED, "orphan mapping");
            if (m != MAP_FAILED) {
                constexpr uint64_t kOrphanPhys = 0xC0000000;
                const uint64_t base = reinterpret_cast<uint64_t>(m);
                guest_write_watch_notify_direct_mapping_added(base, two_pages, kOrphanPhys, kCpuRw);

                // Watch the SECOND page only, so removing the first does not overlap this range.
                GuestWriteWatch watch = GuestWriteWatch::create(base + page, page);
                CHECK(static_cast<bool>(watch), "watch created over the second page");
                CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean before the unmap");

                // Drops the alias at `base` and the whole AliasRange, leaving the second page's alias
                // with no AliasRange covering it. The watch's own range is untouched, so it stays
                // mapping_valid. The purge does NOT dirty this registration -- it bumps a generation
                // only for pages that actually lost an alias, and this page loses none -- so the
                // rearm below is confirming an already-clean watch rather than recovering it. Stated
                // exactly because these comments are the record of why the case discriminates.
                guest_write_watch_notify_direct_mapping_removed(base, page);
                CHECK(watch.rearm(), "rearm after the partial unmap");
                CHECK(watch.query() == GuestWriteWatchQuery::Unchanged, "clean again after rearm");

                // Loop 1 finds the orphaned PageAlias and sets coverage_incomplete. Loop 2 finds no
                // AliasRange, so nothing clears mapping_valid. Only the mark at that site can make
                // this read Dirty -- which is what makes this the discriminator the simple version
                // was not.
                guest_write_watch_notify_direct_mapping_protection(base + page, page, kCpuRw);
                CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
                      "a protection change on a page whose AliasRange is gone still dirties its watch");
                watch.reset();
                guest_write_watch_notify_direct_mapping_removed(base + page, page);
                munmap(m, two_pages);
            }
            close(orphan_fd);
        }
    }

    // ------------------------------------------------ wide registrations and the query fallback
    {
        // kMaxRegistrationChunks is 64, i.e. 128 MiB, so a registration wider than that is NOT in any
        // chunk bucket and is reachable only through the wide list that notify_gpu_write always
        // scans. Without an arm here, deleting that scan would redden nothing while silently
        // stopping dirty propagation to exactly the largest watches.
        const size_t wide_span = 132u * 1024u * 1024u;
        const int wide_fd = make_memfd("prosper-ww-wide", wide_span);
        CHECK(wide_fd >= 0, "wide memfd created");
        if (wide_fd >= 0) {
            auto* wide_map = static_cast<uint8_t*>(
                mmap(nullptr, wide_span, PROT_READ | PROT_WRITE, MAP_SHARED, wide_fd, 0));
            CHECK(wide_map != MAP_FAILED, "wide mapping");
            if (wide_map != MAP_FAILED) {
                constexpr uint64_t kWidePhys = 0x80000000;
                guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(wide_map), wide_span, kWidePhys, kCpuRw);
                GuestWriteWatch wide = GuestWriteWatch::create(
                    reinterpret_cast<uint64_t>(wide_map), wide_span);
                CHECK(static_cast<bool>(wide), "watch spanning more than 64 chunks created");
                CHECK(wide.query() == GuestWriteWatchQuery::Unchanged, "wide watch starts clean");

                // Inside the registration but far past the 64-chunk bucketing limit.
                guest_write_watch_notify_gpu_write(
                    reinterpret_cast<uint64_t>(wide_map) + 130u * 1024u * 1024u, 4096);
                CHECK(wide.query() == GuestWriteWatchQuery::Dirty,
                      "a GPU write dirties a registration too wide to be bucketed");
                CHECK(wide.rearm() && wide.query() == GuestWriteWatchQuery::Unchanged,
                      "the wide registration rearms");

                // A notification wider than kMaxQueryChunks (8 GiB) abandons the buckets for a full
                // scan. This is a CONFIRMATION that the oversized path still answers correctly, not a
                // guard on the branch: since the fallback routes through the same `consider`, the two
                // branches are answer-identical by construction and deleting the fallback would leave
                // this passing. No mutation arm exists for it, and none can -- the branch is a cost
                // choice, and unifying the predicate is what removed its only way to diverge.
                guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(wide_map),
                                                   9ull * 1024ull * 1024ull * 1024ull);
                CHECK(wide.query() == GuestWriteWatchQuery::Dirty,
                      "the full-scan fallback for an oversized notification still dirties it");
                CHECK(wide.rearm(), "rearm after the fallback notification");
                // No negative arm for the fallback's overlap test, deliberately: the fallback routes
                // through the same `consider` as the bucketed path, so there is one predicate and the
                // adjacency cases above already cover it. A negative here would also be unsound --
                // mmap places these two mappings within a few GiB of each other, so a 9 GiB
                // notification genuinely DOES cover an "unrelated" watch, and asserting otherwise
                // tests the allocator's layout rather than this code. (That is what the first draft
                // of this case did, and it failed for exactly that reason.)
                wide.reset();
                guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(wide_map), wide_span);
                munmap(wide_map, wide_span);
            }
            close(wide_fd);
        }
    }

    // ------------------------------------------- #3681: the index must REDUCE WORK, not just agree
    {
        // Every other case here proves the index returns the same ANSWERS as a full scan. None of
        // them would notice if the index silently stopped narrowing -- a bucketing mistake that fell
        // back to considering every registration would keep all of them green while giving back the
        // whole saving. That is the failure this case exists to catch, and it is why #3681 asks for a
        // "removed-index negative control" rather than more correctness arms.
        //
        // The control is the shipped fallback itself: PROSPER_WATCH_LEGACY_SCAN forces the complete
        // scan, and CMake registers a third arm that sets it. Same assertions about dirty tracking in
        // both arms; opposite assertions about how much was visited.
        const bool legacy = std::getenv("PROSPER_WATCH_LEGACY_SCAN") != nullptr;
        constexpr uint64_t kRegistrations = 64;
        const uint64_t slice = span / kRegistrations;          // 128 KiB each, across four 2 MiB chunks
        std::vector<GuestWriteWatch> watches;
        watches.reserve(kRegistrations);
        for (uint64_t i = 0; i < kRegistrations; ++i)
            watches.push_back(GuestWriteWatch::create(reinterpret_cast<uint64_t>(a) + i * slice, slice));
        bool all_created = true, all_clean = true;
        for (const auto& watch : watches) all_created = all_created && static_cast<bool>(watch);
        CHECK(all_created, "64 registrations created for the work-reduction control");
        for (auto& watch : watches)
            all_clean = all_clean && watch.query() == GuestWriteWatchQuery::Unchanged;
        CHECK(all_clean, "all 64 registrations start clean");

        // One 4 KiB GPU write, entirely inside registration 40 -- deliberately not the first, so an
        // index that only ever buckets the start of the mapping cannot pass by accident.
        constexpr uint64_t kTarget = 40;
        const auto before = guest_write_watch_stats();
        guest_write_watch_notify_gpu_write(reinterpret_cast<uint64_t>(a) + kTarget * slice + 4096, 4096);
        const auto after = guest_write_watch_stats();
        const uint64_t visited =
            after.gpu_write_registrations_visited - before.gpu_write_registrations_visited;
        const uint64_t overlaps = after.gpu_write_overlaps - before.gpu_write_overlaps;

        // Identical in BOTH arms: narrowing may not change which registration the write dirties.
        CHECK(overlaps == 1, "exactly one registration overlaps a write inside one slice");
        CHECK(watches[kTarget].query() == GuestWriteWatchQuery::Dirty,
              "the registration containing the GPU write is dirty");
        bool neighbours_clean = true;
        for (uint64_t i = 0; i < kRegistrations; ++i) {
            if (i == kTarget) continue;
            neighbours_clean = neighbours_clean &&
                               watches[i].query() == GuestWriteWatchQuery::Unchanged;
        }
        CHECK(neighbours_clean, "no other registration was dirtied -- narrowing did not lose coverage");

        // Opposite in the two arms, and this is the whole point of the case.
        if (legacy) {
            CHECK(visited >= kRegistrations,
                  "legacy arm: the complete scan really does visit every registration");
        } else {
            // Four 2 MiB chunks over 64 registrations puts ~16 in the written chunk, so half is a
            // wide margin rather than a tuned threshold -- it discriminates without being brittle if
            // the chunk size changes.
            CHECK(visited <= kRegistrations / 2,
                  "index arm: a narrow write visits far fewer than every registration");
        }
        std::fprintf(stderr, "work-reduction arm: legacy=%d visited=%llu of %llu registrations\n",
                     legacy ? 1 : 0, (unsigned long long)visited,
                     (unsigned long long)kRegistrations);
        for (auto& watch : watches) watch.reset();
    }

    // -------------------------------------------------------- the audit's own positive control
    if (auditing) {
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

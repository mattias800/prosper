// Regression guard for per-draw GpuState snapshot recycling (#2334).
//
// The point of recycling is that a snapshot's `snap->cx = cx` copy-assigns into a map that still
// owns its nodes, so libstdc++ reuses them instead of allocating one per entry. Blue Prince's
// collapsed state takes 900,000 snapshots at ~347 entries each -- 312 million node allocations.
//
// Recycling is deliberately INVISIBLE from outside: same types, same values, only fewer
// allocations. So a pool that never actually recycled would leave every ordinary assertion passing.
// Both arms below are written to fail in that case rather than to confirm what already works.

#include "../src/gpu/command_processor.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

// Global allocation counter. Only counts while `g_counting` is set, so the harness's own
// allocations do not drown the measurement.
static bool g_counting = false;
static long g_allocs = 0;
void* operator new(size_t n) {
    if (g_counting) ++g_allocs;
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

int main() {
    using namespace prosper::gpu;

    // A register file the size of the real `cx` in the collapsed state.
    std::unordered_map<uint32_t, uint32_t> source;
    for (uint32_t i = 0; i < 245; ++i) source[i * 7 + 1] = i;

    // ---- ARM 1: a recycled snapshot must come back RESET in every member but the register files.
    //
    // This is the arm that catches the naive pool. GpuState has 25 top-level data members and a
    // snapshot site assigns seven; the rest are default-constructed on a fresh object. A pool that
    // skipped the reset would hand back the previous snapshot's `draws`, `dispatches`, and
    // `last_snapshot_` -- the last being a shared_ptr<const GpuState> INSIDE GpuState, which chains
    // every retired snapshot into the next one. Wrong data and unbounded retention, both silent.
    GpuState* first_raw = nullptr;
    {
        auto snap = acquire_gpustate_snapshot();
        first_raw = snap.get();
        snap->cx = source;
        snap->index_type = 7;
        snap->num_instances = 99;
        snap->command_order = 1234;
        snap->draws.emplace_back();                       // must not survive recycling
        snap->dispatch_count = 55;                        // must not survive recycling
        // `last_snapshot_` is private (command_processor.hpp:228), so it cannot be set or read
        // here. It is still covered: the reset assigns a default-constructed GpuState over the
        // whole object, and member-wise assignment does not care about access control. Noted
        // because it is the member with the worst failure mode -- a shared_ptr<const GpuState>
        // inside GpuState, which would chain every retired snapshot into the next -- and a reader
        // should know it is handled by construction rather than by an assertion below.
        check(!snap->draws.empty(),
              "arm 1 precondition: the snapshot really carries state to be cleared");
    }
    {
        auto snap = acquire_gpustate_snapshot();
        // Not asserted: that this is the same object. The pool is a stack, so it is in practice,
        // but a future pool that round-robins would still be correct and should not fail here.
        check(snap->draws.empty(),            "recycled snapshot: draws cleared");
        check(snap->dispatches.empty(),       "recycled snapshot: dispatches cleared");
        check(snap->dispatch_count == 0,      "recycled snapshot: dispatch_count reset");
        check(snap->index_type == 0,          "recycled snapshot: index_type reset");
        check(snap->num_instances == 1,       "recycled snapshot: num_instances back to its default");
        check(snap->command_order == 0,       "recycled snapshot: command_order reset");
        // ...and the one thing that must NOT be cleared, because it is the entire point:
        check(snap->cx.bucket_count() > 1,
              "recycled snapshot: cx RETAINED its node storage (the reuse this change exists for)");
    }

    // ---- ARM 2: recycling must actually remove the allocations.
    //
    // Asserted as a bound rather than exactly zero: the first iterations warm the pool, and a source
    // that gains a key can legitimately allocate one node.
    uint64_t acquires_before = 0, hits_before = 0;
    gpustate_snapshot_pool_stats(acquires_before, hits_before);

    constexpr int kIterations = 200;
    g_allocs = 0;
    g_counting = true;
    for (int i = 0; i < kIterations; ++i) {
        auto snap = acquire_gpustate_snapshot();
        snap->cx = source;   // the copy whose per-entry allocations this change removes
    }
    g_counting = false;

    uint64_t acquires_after = 0, hits_after = 0;
    gpustate_snapshot_pool_stats(acquires_after, hits_after);
    const uint64_t hits = hits_after - hits_before;

    std::fprintf(stderr, "  %d snapshots x 245 entries -> %ld allocations (%.2f per snapshot); "
                         "pool hits %llu of %llu acquires\n",
                 kIterations, g_allocs, (double)g_allocs / kIterations,
                 (unsigned long long)hits,
                 (unsigned long long)(acquires_after - acquires_before));

    // Without recycling this is ~246 per snapshot, i.e. ~49,200 total. The bound is deliberately
    // loose: it must fail loudly on a regression to per-entry allocation, not police a small drift.
    check(g_allocs < kIterations * 4L,
          "recycled snapshots allocate far below one node per entry");
    check(hits >= (uint64_t)kIterations - 1,
          "the pool actually recycled -- it is not silently allocating fresh every time");

    std::fprintf(stderr, "%s\n", failures ? "== FAILURES ==" : "== all checks passed ==");
    return failures ? 1 : 0;
}

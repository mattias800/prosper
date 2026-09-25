#pragma once
// How many OS threads the parallel helpers create, and how often.
//
// prosper's parallel helpers -- `parallel_compute_texels`, `parallel_rows` and their siblings --
// each spawn a fresh set of `std::jthread`s per CALL and join them at scope exit. That is correct
// and exception-safe, and it costs a `pthread_create`, a stack `mmap`, a TLS init and a matching
// `munmap` per worker per call. Live profiling of one title put `pthread_create` at 11.4% of the
// busiest thread's non-idle samples with `mmap`/`munmap` alongside, so the lifecycle is a real
// share of the work rather than a rounding error.
//
// What could not be answered from the profile is the VOLUME: a helper that spawns 15 workers ten
// times a second and one that spawns them ten thousand times a second look identical in a leaf
// histogram, and only the second is worth a persistent pool. The profile says the cost is real;
// this says whether it is worth removing, and for which site.
//
// Deliberately counts per SITE. The helpers have different work gates (both skip below 512 KiB,
// but on different units), so a pooled total would average a hot site against a cold one and
// point at the wrong fix -- the same reason the pass-break census reports per reason.
//
// WHAT `ranges` COUNTS, now that these helpers dispatch to a shared pool: the number of parallel
// ranges handed out, NOT the number of OS threads created. Before the pool the two were the same
// number, which is how this census found the problem; afterwards thread creation is O(pool size)
// for the whole process while this keeps measuring dispatch volume, which is the quantity that
// decides whether a further batching change is worth making.
//
// One line at end of run via register_exit_report (NOT std::atexit -- #3353).
// `PROSPER_NO_WORKER_SPAWN_CENSUS=1` silences it.

#include <atomic>
#include <cstdint>

namespace prosper::diagnostics {

// A named spawn site. Construct one as a function-local `static` in the helper; registration is
// one-time and the hot path is two relaxed atomic adds.
class WorkerSpawnSite {
public:
    explicit WorkerSpawnSite(const char* name) noexcept;

    // `threads` is the total worker count for this call INCLUDING the calling thread, which does
    // one chunk itself -- so `threads - 1` OS threads are created. Counting the caller keeps the
    // mean comparable to the helper's own `threads` variable.
    void note(unsigned threads, uint64_t work_bytes) noexcept {
        calls_.fetch_add(1, std::memory_order_relaxed);
        if (threads > 1) spawned_.fetch_add(threads - 1, std::memory_order_relaxed);
        work_bytes_.fetch_add(work_bytes, std::memory_order_relaxed);
    }

    const char* name() const noexcept { return name_; }
    uint64_t calls() const noexcept { return calls_.load(std::memory_order_relaxed); }
    uint64_t spawned() const noexcept { return spawned_.load(std::memory_order_relaxed); }
    uint64_t work_bytes() const noexcept { return work_bytes_.load(std::memory_order_relaxed); }

private:
    const char* name_;
    std::atomic<uint64_t> calls_{0};
    std::atomic<uint64_t> spawned_{0};
    std::atomic<uint64_t> work_bytes_{0};
};

}  // namespace prosper::diagnostics

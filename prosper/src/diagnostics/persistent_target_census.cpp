#include "diagnostics/persistent_target_census.hpp"

#include "diagnostics/exit_census.hpp"
#include "diagnostics/perf/perf_ledger.hpp"   // color-target-count-ceiling alarm (#3891)

#include <atomic>
#include <cstdio>

namespace prosper::diagnostics {
namespace {

std::atomic<uint64_t> g_attempts{0};
std::atomic<uint64_t> g_evicted{0};
std::atomic<uint64_t> g_evicted_bytes{0};
std::atomic<uint64_t> g_entries{0};          // high-water mark
std::atomic<uint64_t> g_entry_limit{0};
std::atomic<uint64_t> g_resident_bytes{0};  // high-water mark
std::atomic<uint64_t> g_limit_bytes{0};

void raise_to(std::atomic<uint64_t>& peak, uint64_t value) {
    uint64_t seen = peak.load(std::memory_order_relaxed);
    while (value > seen &&
           !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
    }
}

bool report() {
    const uint64_t attempts = g_attempts.load(std::memory_order_relaxed);
    const uint64_t entries = g_entries.load(std::memory_order_relaxed);
    if (!attempts && !entries) return false;
    const double MiB = 1024.0 * 1024.0;
    const uint64_t limit = g_limit_bytes.load(std::memory_order_relaxed);
    const uint64_t resident = g_resident_bytes.load(std::memory_order_relaxed);
    const uint64_t entry_limit = g_entry_limit.load(std::memory_order_relaxed);
    const uint64_t evicted = g_evicted.load(std::memory_order_relaxed);
    // The verdict must clear BOTH bounds, because either refuses a target on its own, and it must
    // rest on the peaks rather than on `evicted`: the admission paths skip eviction entirely while
    // a submission batch is pending, so a capacity refusal can leave every eviction counter at 0.
    const bool bytes_clear = limit && resident * 2 < limit;
    const bool entries_clear = entry_limit && entries * 2 < entry_limit;
    std::fprintf(stderr,
                 "[persistent-targets] PEAK residency %llu of %llu entries, %.1f of %.1f MiB "
                 "(%.1f%% of budget)  eviction-attempts=%llu evicted=%llu (%.1f MiB)%s\n",
                 (unsigned long long)entries, (unsigned long long)entry_limit,
                 resident / MiB, limit / MiB,
                 limit ? 100.0 * static_cast<double>(resident) / static_cast<double>(limit) : 0.0,
                 (unsigned long long)attempts, (unsigned long long)evicted,
                 g_evicted_bytes.load(std::memory_order_relaxed) / MiB,
                 (bytes_clear && entries_clear && !evicted)
                     ? "  <- peaked far below BOTH bounds and evicted nothing: a missing "
                       "persistent target here is NOT a capacity problem"
                     : "");
    std::fflush(stderr);
    return true;
}

// Register on FIRST USE, never at namespace-scope static initialisation.
//
// `register_exit_report` installs its `std::atexit` fallback on the first registration in the
// process, and `atexit` runs LIFO against `__cxa_atexit`'s static destructors. Registering during
// static init therefore makes this the earliest atexit entry, which makes the flush the LATEST
// thing to run -- after the function-local statics that OTHER censuses read. It is not this
// census's own report that breaks; it is every report registered after it.
//
// Measured: adding this file with a namespace-scope registration turned a green suite into eleven
// failures (a SIGSEGV in `game_compute_exec` after `[storage-materialize]` printed, plus timeouts),
// none of them in this census and none of them in code this file touches.
void ensure_registered() {
    static const bool once = [] {
        register_census("PROSPER_NO_PERSISTENT_TARGET_CENSUS", report);
        return true;
    }();
    (void)once;
}

}  // namespace

void note_persistent_target_eviction_attempt() {
    ensure_registered();
    g_attempts.fetch_add(1, std::memory_order_relaxed);
}

void note_persistent_target_evicted(uint64_t bytes) {
    g_evicted.fetch_add(1, std::memory_order_relaxed);
    g_evicted_bytes.fetch_add(bytes, std::memory_order_relaxed);
    perf::add(perf::Counter::PersistentTargetEvictions);
    perf::add(perf::Counter::PersistentTargetEvictedBytes, bytes);
}

void note_persistent_target_residency(uint64_t entries, uint64_t entry_limit,
                                      uint64_t bytes, uint64_t limit_bytes) {
    ensure_registered();
    raise_to(g_entries, entries);
    raise_to(g_resident_bytes, bytes);
    g_entry_limit.store(entry_limit, std::memory_order_relaxed);
    g_limit_bytes.store(limit_bytes, std::memory_order_relaxed);
    // The same sample, per alarm window: the run's high-water mark above cannot say WHEN the
    // cache sat at its bound, nor whether it was churning then.
    perf::raise(perf::Peak::PersistentTargetEntries, entries);
    perf::raise(perf::Peak::PersistentTargetBytes, bytes);
    perf::set_if_changed(perf::Gauge::PersistentTargetEntryLimit, entry_limit);
    perf::set_if_changed(perf::Gauge::PersistentTargetByteLimit, limit_bytes);
}

}  // namespace prosper::diagnostics

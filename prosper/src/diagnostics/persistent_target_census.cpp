#include "diagnostics/persistent_target_census.hpp"

#include "diagnostics/exit_census.hpp"

#include <atomic>
#include <cstdio>

namespace prosper::diagnostics {
namespace {

std::atomic<uint64_t> g_attempts{0};
std::atomic<uint64_t> g_evicted{0};
std::atomic<uint64_t> g_evicted_bytes{0};
std::atomic<uint64_t> g_entries{0};
std::atomic<uint64_t> g_resident_bytes{0};
std::atomic<uint64_t> g_limit_bytes{0};

bool report() {
    const uint64_t attempts = g_attempts.load(std::memory_order_relaxed);
    const uint64_t entries = g_entries.load(std::memory_order_relaxed);
    if (!attempts && !entries) return false;
    const double MiB = 1024.0 * 1024.0;
    const uint64_t limit = g_limit_bytes.load(std::memory_order_relaxed);
    const uint64_t resident = g_resident_bytes.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[persistent-targets] RUN TOTAL resident=%llu entries / %.1f MiB of %.1f MiB "
                 "(%.1f%% of budget)  eviction-attempts=%llu evicted=%llu (%.1f MiB)%s\n",
                 (unsigned long long)entries, resident / MiB, limit / MiB,
                 limit ? 100.0 * static_cast<double>(resident) / static_cast<double>(limit) : 0.0,
                 (unsigned long long)attempts,
                 (unsigned long long)g_evicted.load(std::memory_order_relaxed),
                 g_evicted_bytes.load(std::memory_order_relaxed) / MiB,
                 // The one-line verdict, because the whole point is to stop a reader having to
                 // work out which of the two states this is.
                 (!g_evicted.load(std::memory_order_relaxed) && limit &&
                  resident * 2 < limit)
                     ? "  <- far below budget with nothing evicted: a missing persistent target "
                       "here is NOT a capacity problem"
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
}

void note_persistent_target_residency(uint64_t entries, uint64_t bytes, uint64_t limit_bytes) {
    ensure_registered();
    g_entries.store(entries, std::memory_order_relaxed);
    g_resident_bytes.store(bytes, std::memory_order_relaxed);
    g_limit_bytes.store(limit_bytes, std::memory_order_relaxed);
}

}  // namespace prosper::diagnostics

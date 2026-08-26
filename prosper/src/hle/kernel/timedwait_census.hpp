#pragma once
// timedwait_census — which guest timed-wait primitive a title paces on, and by how much each one
// overshoots what the guest asked for. `PROSPER_TIMEDWAIT_CENSUS=1`; off costs two branches.
//
// This exists because #3013 was diagnosed from the primitives' granularity rather than from the
// guest's use of them, and those are different claims. Knowing that winpthreads quantizes every
// timed wait to ~15.6 ms does NOT tell you which call a given title's mixer paces on -- FMOD paces
// on a semaphore timedwait, other middleware sleeps, and a fix aimed at the wrong one is untestable
// and looks like a regression when it changes nothing. So: measure the caller, not just the callee.
//
// What it reports per primitive: call count, mean requested interval, mean ACTUAL interval, and the
// ratio. A ratio near 1.0 is a primitive serving the guest's schedule; the ~2.9x that #3013 measured
// for a 5.33 ms audio grain is what starvation looks like from inside.
//
// Deliberately NOT a rate limiter on the measurement itself: every call is counted, and only the
// REPORT is throttled. A census that sampled would answer "how often does this happen" with a number
// about its own sampling.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace prosper::hle {

enum class WaitKind { Usleep, Nanosleep, SleepSeconds, SemTimedwait, CondTimedwait,
                     MutexTimedlock, Count };

inline const char* wait_kind_name(WaitKind k) {
    switch (k) {
    case WaitKind::Usleep:        return "usleep";
    case WaitKind::Nanosleep:     return "nanosleep";
    case WaitKind::SleepSeconds:  return "sleep(s)";
    case WaitKind::SemTimedwait:  return "sem_timedwait";
    case WaitKind::CondTimedwait: return "cond_timedwait";
    case WaitKind::MutexTimedlock: return "mutex_timedlock";
    default:                      return "?";
    }
}

// inline variables: ONE set of counters across every translation unit that includes this. A
// header-only `static` would give each TU its own, so the sleeps and the timed waits would be
// counted into different tables and each would report the other as never called.
inline std::atomic<uint64_t> g_wait_calls[(size_t)WaitKind::Count];
inline std::atomic<uint64_t> g_wait_requested_ns[(size_t)WaitKind::Count];
inline std::atomic<uint64_t> g_wait_actual_ns[(size_t)WaitKind::Count];
inline std::atomic<uint64_t> g_wait_next_report_ns{0};

inline bool timedwait_census() {
    static const bool on = getenv("PROSPER_TIMEDWAIT_CENSUS") != nullptr;
    return on;
}

inline uint64_t census_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void report_timedwait_census(uint64_t now_ns) {
    // Every 5 s, from whichever thread happens to cross the deadline first. The compare_exchange
    // means concurrent crossers do not each print the same table.
    uint64_t due = g_wait_next_report_ns.load(std::memory_order_relaxed);
    if (due == 0) {   // first call establishes the schedule rather than reporting an empty table
        g_wait_next_report_ns.compare_exchange_strong(due, now_ns + 5000000000ull);
        return;
    }
    if (now_ns < due) return;
    if (!g_wait_next_report_ns.compare_exchange_strong(due, now_ns + 5000000000ull)) return;

    for (size_t i = 0; i < (size_t)WaitKind::Count; i++) {
        const uint64_t n = g_wait_calls[i].load(std::memory_order_relaxed);
        if (!n) continue;
        const uint64_t req_total = g_wait_requested_ns[i].load(std::memory_order_relaxed);
        // A primitive whose deadline is ABSOLUTE in a clock epoch this layer does not resolve
        // (cond_timedwait) records no requested interval rather than a wrong one: subtracting the
        // wrong clock-s now from a guest deadline yields a plausible-looking number that means
        // nothing. Those report duration and call count, and no ratio.
        if (req_total == 0) {
            fprintf(stderr, "[timedwait] %-15s calls=%-9llu requested=      ?     actual=%7.3f ms\n",
                    wait_kind_name((WaitKind)i), (unsigned long long)n,
                    g_wait_actual_ns[i].load(std::memory_order_relaxed) / 1e6 / n);
            continue;
        }
        const double req_ms = req_total / 1e6 / n;

        const double act_ms = g_wait_actual_ns[i].load(std::memory_order_relaxed) / 1e6 / n;
        fprintf(stderr, "[timedwait] %-15s calls=%-9llu requested=%7.3f ms  actual=%7.3f ms  x%.2f\n",
                wait_kind_name((WaitKind)i), (unsigned long long)n, req_ms, act_ms,
                req_ms > 0 ? act_ms / req_ms : 0.0);
    }
}

// RAII: times the wait it encloses and records it. Constructed disabled when the census is off, so
// the cost on a normal run is the one cached getenv branch.
class WaitCensusScope {
public:
    WaitCensusScope(WaitKind kind, uint64_t requested_ns)
        : kind_(kind), requested_ns_(requested_ns), on_(timedwait_census()),
          start_ns_(on_ ? census_now_ns() : 0) {}
    ~WaitCensusScope() {
        if (!on_) return;
        const uint64_t end = census_now_ns();
        const size_t i = (size_t)kind_;
        g_wait_calls[i].fetch_add(1, std::memory_order_relaxed);
        g_wait_requested_ns[i].fetch_add(requested_ns_, std::memory_order_relaxed);
        g_wait_actual_ns[i].fetch_add(end - start_ns_, std::memory_order_relaxed);
        report_timedwait_census(end);
    }
    WaitCensusScope(const WaitCensusScope&) = delete;
    WaitCensusScope& operator=(const WaitCensusScope&) = delete;
private:
    WaitKind kind_;
    uint64_t requested_ns_;
    bool     on_;
    uint64_t start_ns_;
};

}  // namespace prosper::hle

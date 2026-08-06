// sync_retire.cpp — see sync_retire.hpp for why a destroyed guest sync object is quarantined.
#include "sync_retire.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <vector>

namespace prosper {
namespace {
    const char* const kSyncObjectKindNames[(size_t)SyncObjectKind::Count_] = {
        "mutex", "cond", "rwlock", "sem", "barrier", "_Mtx", "_Cnd"
    };

    using Clock = std::chrono::steady_clock;

    struct Retired {
        void* storage = nullptr;
        SyncObjectHostDestroy host_destroy = nullptr;
        Clock::time_point retired_at{};
    };

    std::mutex g_mutex;
    std::deque<Retired> g_quarantine;          // FIFO: retirement time is monotonic, so is the order
    uint64_t g_total = 0;
    uint64_t g_by_kind[(size_t)SyncObjectKind::Count_] = {};
    size_t g_high_water = 0;
    size_t g_reported_high_water = 0;

    bool retire_log() {
        static const bool on = getenv("PROSPER_SYNC_RETIRELOG") != nullptr;
        return on;
    }
    // How long a destroyed object stays unreachable-but-alive. 0 restores the pre-#2042 immediate
    // free (the counter-arm). Negative and unparseable values fall back to the default rather than
    // silently disabling the guard.
    double retire_window_seconds() {
        static const double seconds = [] {
            const char* e = getenv("PROSPER_SYNC_RETIRE_SECONDS");
            if (!e || !*e) return 30.0;
            const double v = atof(e);
            return v >= 0.0 ? v : 30.0;
        }();
        return seconds;
    }
    // Ungated reporting starts here so an ordinary boot stays silent while a title whose churn is
    // far above anything measured cannot. 65,536 objects held at once is a few megabytes.
    constexpr size_t kUngatedReportFloor = 1u << 16;
}

void retire_sync_object(void* storage, SyncObjectKind kind, SyncObjectHostDestroy host_destroy) {
    if (!storage) return;

    std::vector<Retired> reclaim;              // freed outside the lock
    size_t held = 0;
    bool report = false;
    uint64_t by_kind[(size_t)SyncObjectKind::Count_];
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ++g_total;
        if (kind < SyncObjectKind::Count_) ++g_by_kind[(size_t)kind];

        const double window = retire_window_seconds();
        const Clock::time_point now = Clock::now();
        const Clock::time_point cutoff =
            now - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(window));
        while (!g_quarantine.empty() && g_quarantine.front().retired_at <= cutoff) {
            reclaim.push_back(g_quarantine.front());
            g_quarantine.pop_front();
        }

        if (window > 0.0) g_quarantine.push_back(Retired{storage, host_destroy, now});
        else              reclaim.push_back(Retired{storage, host_destroy, now});

        held = g_quarantine.size();
        g_high_water = std::max(g_high_water, held);
        // Report on powers of two of the HIGH WATER, so a steady state reports once rather than
        // oscillating across a boundary forever.
        if (g_high_water > g_reported_high_water && (g_high_water & (g_high_water - 1)) == 0 &&
            (retire_log() || g_high_water >= kUngatedReportFloor)) {
            g_reported_high_water = g_high_water;
            report = true;
            for (size_t k = 0; k < (size_t)SyncObjectKind::Count_; ++k) by_kind[k] = g_by_kind[k];
        }
    }

    for (const Retired& r : reclaim) {
        if (r.host_destroy) r.host_destroy(r.storage);
        free(r.storage);
    }

    if (!report) return;
    char census[224];
    int off = 0;
    for (size_t k = 0; k < (size_t)SyncObjectKind::Count_ && off > -1 && off < (int)sizeof census; ++k)
        off += snprintf(census + off, sizeof census - (size_t)off, " %s=%llu",
                        kSyncObjectKindNames[k], (unsigned long long)by_kind[k]);
    fprintf(stderr, "[sync] %zu destroyed guest sync objects held in quarantine (#2042); "
                    "destroyed so far:%s\n", held, census);
}

uint64_t retired_sync_object_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_total;
}

uint64_t quarantined_sync_object_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return (uint64_t)g_quarantine.size();
}

}  // namespace prosper

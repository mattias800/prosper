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

    struct Registry {
        std::mutex mutex;
        std::deque<Retired> quarantine;   // FIFO: retirement time is monotonic, so is the order
        uint64_t total = 0;
        uint64_t by_kind[(size_t)SyncObjectKind::Count_] = {};
        size_t high_water = 0;
        size_t reported_high_water = 0;
    };
    // DELIBERATELY LEAKED, and never destroyed. A guest thread can reach a *Destroy handler during
    // static teardown (guest TSD destructors and atexit handlers both run there), and a namespace
    // -scope registry would already have been destroyed by then — trading a use-after-free on the
    // guest's object for a use-after-free on the machinery that exists to prevent it.
    Registry& registry() {
        static Registry* const r = new Registry();
        return *r;
    }

    bool retire_log() {
        static const bool on = getenv("PROSPER_SYNC_RETIRELOG") != nullptr;
        return on;
    }
    // How long a destroyed object stays unreachable-but-alive. 0 restores the pre-#2042 immediate
    // free (the counter-arm). Negative and unparseable values fall back to the default rather than
    // silently disabling the guard — and that has to be enforced by `strtod` + an endptr check, not
    // by `atof`, which returns 0.0 on no conversion. Under `atof`, `PROSPER_SYNC_RETIRE_SECONDS=on`
    // or any typo would pass `>= 0.0` and switch the guard OFF, which is the exact use-after-free
    // this file exists to prevent — with the comment above promising the opposite, so a reader
    // checking "does this switch fail safe?" would find the reassurance and stop looking.
    double retire_window_seconds() {
        static const double seconds = [] {
            const char* e = getenv("PROSPER_SYNC_RETIRE_SECONDS");
            if (!e || !*e) return 30.0;
            char* end = nullptr;
            const double v = strtod(e, &end);
            if (end == e || *end != '\0' || !(v >= 0.0)) return 30.0;   // !(v>=0) also rejects NaN
            // Clamp the upper end too (#2176). strtod accepts `inf` and `1e300`, and both
            // overflow the duration_cast to steady_clock::duration below -- undefined
            // behaviour, even though the direction observed happens to be the safe one
            // (nothing is ever reclaimed). An hour is far past any window that could be
            // deliberate: the default is 30 s, and the guard's whole premise is bounding the
            // retained set by RATE, which a 3,600 s window stops doing usefully anyway.
            constexpr double kMaxWindowSeconds = 3600.0;
            return v > kMaxWindowSeconds ? kMaxWindowSeconds : v;
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
    Registry& reg = registry();
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        ++reg.total;
        if (kind < SyncObjectKind::Count_) ++reg.by_kind[(size_t)kind];

        const double window = retire_window_seconds();
        const Clock::time_point now = Clock::now();
        const Clock::time_point cutoff =
            now - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(window));
        while (!reg.quarantine.empty() && reg.quarantine.front().retired_at <= cutoff) {
            reclaim.push_back(reg.quarantine.front());
            reg.quarantine.pop_front();
        }

        if (window > 0.0) reg.quarantine.push_back(Retired{storage, host_destroy, now});
        else              reclaim.push_back(Retired{storage, host_destroy, now});

        held = reg.quarantine.size();
        reg.high_water = std::max(reg.high_water, held);
        // Report on powers of two of the HIGH WATER, so a steady state reports once rather than
        // oscillating across a boundary forever.
        if (reg.high_water > reg.reported_high_water &&
            (reg.high_water & (reg.high_water - 1)) == 0 &&
            (retire_log() || reg.high_water >= kUngatedReportFloor)) {
            reg.reported_high_water = reg.high_water;
            report = true;
            for (size_t k = 0; k < (size_t)SyncObjectKind::Count_; ++k) by_kind[k] = reg.by_kind[k];
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
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return reg.total;
}

uint64_t quarantined_sync_object_count() {
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    return (uint64_t)reg.quarantine.size();
}

}  // namespace prosper

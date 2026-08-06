// sync_retire.cpp — see sync_retire.hpp for why a destroyed guest sync object is retained.
#include "sync_retire.hpp"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace prosper {
namespace {
    std::mutex g_retired_mutex;
    // Owns every retired object for the life of the process. Reachable ON PURPOSE — see the header.
    std::vector<void*> g_retired;

    bool retire_log() {
        static const bool on = getenv("PROSPER_SYNC_RETIRELOG") != nullptr;
        return on;
    }
    // Ungated reporting starts here so an ordinary boot stays silent while a title that churns sync
    // objects cannot. 65,536 retained rwlocks is roughly 5 MB — small enough to be uninteresting,
    // large enough that reaching it means the guest is doing something worth measuring.
    constexpr size_t kUngatedReportFloor = 1u << 16;
}

void retire_sync_object(void* storage) {
    if (!storage) return;
    size_t n;
    {
        std::lock_guard<std::mutex> lock(g_retired_mutex);
        g_retired.push_back(storage);
        n = g_retired.size();
    }
    if ((n & (n - 1)) != 0) return;                              // powers of two only
    if (!retire_log() && n < kUngatedReportFloor) return;
    fprintf(stderr, "[sync] %zu destroyed guest sync objects retained, never freed (#2042)\n", n);
}

uint64_t retired_sync_object_count() {
    std::lock_guard<std::mutex> lock(g_retired_mutex);
    return (uint64_t)g_retired.size();
}

}  // namespace prosper

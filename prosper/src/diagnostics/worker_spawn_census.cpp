#include "diagnostics/worker_spawn_census.hpp"

#include "diagnostics/exit_reports.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace prosper::diagnostics {
namespace {

// A fixed table rather than a growing container: registration happens during static/lazy init on
// arbitrary threads, and a helper that allocates there is a worse problem than a capped census.
constexpr size_t kMaxSites = 16;
std::array<const WorkerSpawnSite*, kMaxSites> g_sites{};
size_t g_site_count = 0;
std::mutex g_mutex;

void report() {
    if (std::getenv("PROSPER_NO_WORKER_SPAWN_CENSUS")) return;
    std::lock_guard lock(g_mutex);
    uint64_t total_calls = 0, total_spawned = 0;
    for (size_t i = 0; i < g_site_count; i++) {
        total_calls += g_sites[i]->calls();
        total_spawned += g_sites[i]->spawned();
    }
    if (!total_spawned) return;
    std::fprintf(stderr, "[worker-spawn] RUN TOTAL calls=%llu ranges=%llu",
                 static_cast<unsigned long long>(total_calls),
                 static_cast<unsigned long long>(total_spawned));
    for (size_t i = 0; i < g_site_count; i++) {
        const auto* s = g_sites[i];
        if (!s->calls()) continue;
        std::fprintf(stderr, "  %s: calls=%llu threads=%llu mean=%.1f MiB/call=%.2f", s->name(),
                     static_cast<unsigned long long>(s->calls()),
                     static_cast<unsigned long long>(s->spawned()),
                     static_cast<double>(s->spawned()) / static_cast<double>(s->calls()),
                     static_cast<double>(s->work_bytes()) /
                         static_cast<double>(s->calls()) / (1024.0 * 1024.0));
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

}  // namespace

WorkerSpawnSite::WorkerSpawnSite(const char* name) noexcept : name_(name) {
    std::lock_guard lock(g_mutex);
    static const bool once = [] {
        register_exit_report([] { report(); });
        return true;
    }();
    (void)once;
    if (g_site_count < kMaxSites) g_sites[g_site_count++] = this;
}

}  // namespace prosper::diagnostics

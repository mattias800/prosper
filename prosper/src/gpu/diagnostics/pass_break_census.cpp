#include "gpu/diagnostics/pass_break_census.hpp"

#include "diagnostics/exit_census.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu {
namespace {

constexpr size_t kCount = static_cast<size_t>(PassBreak::Count);

constexpr std::array<const char*, kCount> kNames{
    "targets-changed", "mrt-resolve-differs", "depth-feedback", "end-of-items",
};
static_assert(kNames.size() == kCount, "every PassBreak needs a stable name");

}  // namespace

const char* pass_break_name(PassBreak reason) {
    const auto i = static_cast<size_t>(reason);
    return i < kCount ? kNames[i] : "unknown";
}

struct PassBreakCensus::State {
    std::array<std::atomic<uint64_t>, kCount> passes{};
    std::array<std::atomic<uint64_t>, kCount> draws{};
    std::array<std::atomic<uint64_t>, kCount> singletons{};
};

PassBreakCensus::State& PassBreakCensus::state() const {
    static State s;
    return s;
}

void PassBreakCensus::note_break(PassBreak reason, uint64_t draws) {
    const auto i = static_cast<size_t>(reason);
    if (i >= kCount) return;
    auto& s = state();
    s.passes[i].fetch_add(1, std::memory_order_relaxed);
    s.draws[i].fetch_add(draws, std::memory_order_relaxed);
    if (draws <= 1) s.singletons[i].fetch_add(1, std::memory_order_relaxed);
}

uint64_t PassBreakCensus::passes(PassBreak reason) const {
    const auto i = static_cast<size_t>(reason);
    return i < kCount ? state().passes[i].load(std::memory_order_relaxed) : 0;
}
uint64_t PassBreakCensus::draws(PassBreak reason) const {
    const auto i = static_cast<size_t>(reason);
    return i < kCount ? state().draws[i].load(std::memory_order_relaxed) : 0;
}

bool PassBreakCensus::report_totals() {
    auto& s = state();
    uint64_t total_passes = 0, total_draws = 0;
    for (size_t i = 0; i < kCount; i++) {
        total_passes += s.passes[i].load(std::memory_order_relaxed);
        total_draws += s.draws[i].load(std::memory_order_relaxed);
    }
    if (total_passes == 0) return false;
    std::fprintf(stderr, "[pass-break] RUN TOTAL passes=%llu draws=%llu mean=%.2f",
                 static_cast<unsigned long long>(total_passes),
                 static_cast<unsigned long long>(total_draws),
                 static_cast<double>(total_draws) / static_cast<double>(total_passes));
    for (size_t i = 0; i < kCount; i++) {
        const uint64_t p = s.passes[i].load(std::memory_order_relaxed);
        if (!p) continue;
        const uint64_t d = s.draws[i].load(std::memory_order_relaxed);
        const uint64_t one = s.singletons[i].load(std::memory_order_relaxed);
        // mean is what decides "lengthen the pass" vs "make a pass cheaper"; the singleton share
        // says whether this reason is the one holding the mean down.
        std::fprintf(stderr, "  %s: passes=%llu mean=%.2f singleton=%.1f%%", kNames[i],
                     static_cast<unsigned long long>(p),
                     static_cast<double>(d) / static_cast<double>(p),
                     100.0 * static_cast<double>(one) / static_cast<double>(p));
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    return true;
}

PassBreakCensus& pass_break_census() {
    static PassBreakCensus census;
    static const bool once = [] {
        prosper::diagnostics::register_census(
            "PROSPER_NO_PASS_BREAK_CENSUS", [] { return pass_break_census().report_totals(); });
        return true;
    }();
    (void)once;
    return census;
}

}  // namespace prosper::gpu

#include "gpu/diagnostics/draw_disposition.hpp"

#include "gpu/diagnostics/diag_ratelimit.hpp"

#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu {
namespace {

constexpr size_t kReasonCount = static_cast<size_t>(DrawDrop::Count);

// Indexed by DrawDrop. Kept adjacent to the enum's declaration order on purpose: a reason added
// to one and not the other is a compile error via the static_assert below.
constexpr std::array<const char*, kReasonCount> kNames{
    "geometry-capability",
    "mesh-shape",
    "subgroup-features",
    "gds-allocation",
    "buffer-resources",
    "shader-rejected",
    "pipeline-creation",
};
static_assert(kNames.size() == kReasonCount,
              "every DrawDrop needs a stable name; logs are grepped by these strings");

bool reporting_enabled() {
    // Default ON. A diagnostic you have to know to enable is one nobody enables; the only switch
    // is an opt-OUT, so a log that is silent about drops really did have none.
    static const bool enabled = std::getenv("PROSPER_NO_DRAW_DISPOSITION") == nullptr;
    return enabled;
}

}  // namespace

const char* draw_drop_name(DrawDrop reason) {
    const auto i = static_cast<size_t>(reason);
    return i < kReasonCount ? kNames[i] : "unknown";
}

struct DrawDispositionCensus::State {
    std::atomic<uint64_t> seen{0};
    std::atomic<uint64_t> recorded{0};
    std::array<std::atomic<uint64_t>, kReasonCount> dropped{};
    // Per-pass figures. The backend serialises passes on one mutex, so plain counters would do;
    // these are atomic anyway so a programmatic reader is never torn.
    std::atomic<uint64_t> pass_seen{0};
    std::atomic<uint64_t> pass_recorded{0};
    std::array<std::atomic<uint64_t>, kReasonCount> pass_dropped{};
    // Per-reason print budget, so a high-volume reason cannot starve a rare one.
    std::array<std::atomic<uint64_t>, kReasonCount> printed{};
    std::atomic<uint64_t> black_passes{0};
    std::atomic<uint64_t> unaccounted_passes{0};
};

DrawDispositionCensus::State& DrawDispositionCensus::state() const {
    static State s;
    return s;
}

void DrawDispositionCensus::note_seen(uint64_t count) {
    auto& s = state();
    s.seen.fetch_add(count, std::memory_order_relaxed);
    s.pass_seen.fetch_add(count, std::memory_order_relaxed);
}

void DrawDispositionCensus::note_recorded(uint64_t count) {
    auto& s = state();
    s.recorded.fetch_add(count, std::memory_order_relaxed);
    s.pass_recorded.fetch_add(count, std::memory_order_relaxed);
}

void DrawDispositionCensus::note_dropped(DrawDrop reason) {
    const auto i = static_cast<size_t>(reason);
    if (i >= kReasonCount) return;
    auto& s = state();
    s.dropped[i].fetch_add(1, std::memory_order_relaxed);
    s.pass_dropped[i].fetch_add(1, std::memory_order_relaxed);
}

uint64_t DrawDispositionCensus::seen() const { return state().seen.load(std::memory_order_relaxed); }
uint64_t DrawDispositionCensus::recorded() const {
    return state().recorded.load(std::memory_order_relaxed);
}
uint64_t DrawDispositionCensus::dropped(DrawDrop reason) const {
    const auto i = static_cast<size_t>(reason);
    return i < kReasonCount ? state().dropped[i].load(std::memory_order_relaxed) : 0;
}
uint64_t DrawDispositionCensus::dropped_total() const {
    uint64_t total = 0;
    for (size_t i = 0; i < kReasonCount; i++)
        total += state().dropped[i].load(std::memory_order_relaxed);
    return total;
}

void DrawDispositionCensus::report_pass() {
    auto& s = state();
    const uint64_t pass_seen = s.pass_seen.exchange(0, std::memory_order_relaxed);
    const uint64_t pass_recorded = s.pass_recorded.exchange(0, std::memory_order_relaxed);
    std::array<uint64_t, kReasonCount> pass_dropped{};
    uint64_t pass_dropped_total = 0;
    for (size_t i = 0; i < kReasonCount; i++) {
        pass_dropped[i] = s.pass_dropped[i].exchange(0, std::memory_order_relaxed);
        pass_dropped_total += pass_dropped[i];
    }
    if (!reporting_enabled() || pass_seen == 0) return;

    // Two independent routes to the same quantity. A disagreement means a drop path exists that
    // does not name a reason -- report the blind spot rather than a tidy total that hides it.
    const bool unaccounted = pass_recorded + pass_dropped_total != pass_seen;
    // A pass that saw draws and recorded none is the observation this instrument exists for, so
    // it is never suppressed by a per-reason budget.
    const bool black_pass = pass_recorded == 0;

    bool print = false;
    if (unaccounted) {
        print = diag_should_print(
            s.unaccounted_passes.fetch_add(1, std::memory_order_relaxed) + 1);
    }
    if (black_pass) {
        print = print || diag_should_print(
            s.black_passes.fetch_add(1, std::memory_order_relaxed) + 1);
    }
    // Per-reason budget: the FIRST occurrence of a reason always prints, then first-N plus powers
    // of two for that reason alone.
    std::array<uint64_t, kReasonCount> ordinal{};
    for (size_t i = 0; i < kReasonCount; i++) {
        if (!pass_dropped[i]) continue;
        ordinal[i] = s.printed[i].fetch_add(1, std::memory_order_relaxed) + 1;
        if (diag_should_print(ordinal[i])) print = true;
    }
    if (!print) return;

    std::fprintf(stderr, "[draw-disposition] seen=%llu recorded=%llu",
                 static_cast<unsigned long long>(pass_seen),
                 static_cast<unsigned long long>(pass_recorded));
    for (size_t i = 0; i < kReasonCount; i++) {
        if (!pass_dropped[i]) continue;
        // ord= is this REASON's 1-based pass-occurrence count, per the diag_ratelimit contract:
        // the last line's ordinal bounds the population from below; the line count never does.
        std::fprintf(stderr, " %s=%llu(ord=%llu)", kNames[i],
                     static_cast<unsigned long long>(pass_dropped[i]),
                     static_cast<unsigned long long>(ordinal[i]));
    }
    if (black_pass) std::fprintf(stderr, "  BLACK-PASS");
    if (unaccounted)
        std::fprintf(stderr, "  UNACCOUNTED=%lld",
                     static_cast<long long>(static_cast<int64_t>(pass_seen) -
                                            static_cast<int64_t>(pass_recorded) -
                                            static_cast<int64_t>(pass_dropped_total)));
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

DrawDispositionCensus& draw_disposition_census() {
    static DrawDispositionCensus census;
    return census;
}

}  // namespace prosper::gpu

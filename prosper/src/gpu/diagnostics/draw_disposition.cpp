#include "gpu/diagnostics/draw_disposition.hpp"

#include "gpu/diagnostics/diag_ratelimit.hpp"
#include "diagnostics/exit_reports.hpp"

#include <atomic>
#include <chrono>
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

// Print EVERY pass, healthy or not. This exists because a silent instrument and an instrument
// that was never reached are indistinguishable from outside -- the failure mode the charter's
// instrument-trap list keeps recording. Verifying this census on a new title starts here: if a
// verbose run prints nothing, the draw path does not go through this function and no conclusion
// may be drawn from the quiet default.
bool verbose_enabled() {
    static const bool enabled = std::getenv("PROSPER_DRAW_DISPOSITION_VERBOSE") != nullptr;
    return enabled;
}

bool reporting_enabled() {
    // Default ON. A diagnostic you have to know to enable is one nobody enables; the only switch
    // is an opt-OUT, so a log that is silent about drops really did have none.
    static const bool enabled = std::getenv("PROSPER_NO_DRAW_DISPOSITION") == nullptr;
    return enabled;
}

}  // namespace

namespace {
constexpr uint64_t kBucketLow[7] = {1, 2, 3, 5, 9, 17, 33};
constexpr const char* kBucketName[7] = {"1", "2", "3-4", "5-8", "9-16", "17-32", "33+"};
size_t bucket_for(uint64_t draws) {
    size_t b = 0;
    for (size_t i = 0; i < 7; i++) if (draws >= kBucketLow[i]) b = i;
    return b;
}
uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
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
    // Pass wall time bucketed by draw count. Bucket i holds passes with kBucketLow[i] draws or
    // more, up to the next bucket's floor.
    static constexpr size_t kBuckets = 7;
    std::array<std::atomic<uint64_t>, kBuckets> bucket_passes{};
    std::array<std::atomic<uint64_t>, kBuckets> bucket_ns{};
    std::array<std::atomic<uint64_t>, kBuckets> bucket_draws{};
    std::atomic<uint64_t> first_pass_ns{0};
    std::atomic<uint64_t> last_pass_end_ns{0};
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

uint64_t DrawDispositionCensus::pass_seen_for_scope() const {
    return state().pass_seen.load(std::memory_order_relaxed);
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
    if (verbose_enabled()) print = true;
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

void DrawDispositionCensus::note_pass_duration(uint64_t draws, uint64_t nanoseconds) {
    if (draws == 0) return;
    auto& s = state();
    const size_t b = bucket_for(draws);
    s.bucket_passes[b].fetch_add(1, std::memory_order_relaxed);
    s.bucket_ns[b].fetch_add(nanoseconds, std::memory_order_relaxed);
    s.bucket_draws[b].fetch_add(draws, std::memory_order_relaxed);
    const uint64_t end = now_ns();
    uint64_t expected = 0;
    s.first_pass_ns.compare_exchange_strong(expected, end - nanoseconds,
                                            std::memory_order_relaxed);
    s.last_pass_end_ns.store(end, std::memory_order_relaxed);
}

DrawDispositionPassScope::DrawDispositionPassScope() : start_ns_(now_ns()) {}

DrawDispositionPassScope::~DrawDispositionPassScope() {
    auto& census = draw_disposition_census();
    // Read the pass's draw count BEFORE report_pass() resets the per-pass counters.
    const uint64_t drawn = census.pass_seen_for_scope();
    if (drawn) census.note_pass_duration(drawn, now_ns() - start_ns_);
    census.report_pass();
}

void DrawDispositionCensus::report_totals() {
    auto& s = state();
    const uint64_t total_seen = s.seen.load(std::memory_order_relaxed);
    if (!reporting_enabled() || total_seen == 0) return;
    const uint64_t total_recorded = s.recorded.load(std::memory_order_relaxed);
    uint64_t total_dropped = 0;
    for (size_t i = 0; i < kReasonCount; i++)
        total_dropped += s.dropped[i].load(std::memory_order_relaxed);
    std::fprintf(stderr, "[draw-disposition] RUN TOTAL seen=%llu recorded=%llu dropped=%llu",
                 static_cast<unsigned long long>(total_seen),
                 static_cast<unsigned long long>(total_recorded),
                 static_cast<unsigned long long>(total_dropped));
    for (size_t i = 0; i < kReasonCount; i++) {
        const uint64_t n = s.dropped[i].load(std::memory_order_relaxed);
        if (n) std::fprintf(stderr, " %s=%llu", kNames[i], static_cast<unsigned long long>(n));
    }
    if (total_recorded + total_dropped != total_seen)
        std::fprintf(stderr, "  UNACCOUNTED=%lld",
                     static_cast<long long>(static_cast<int64_t>(total_seen) -
                                            static_cast<int64_t>(total_recorded) -
                                            static_cast<int64_t>(total_dropped)));
    std::fprintf(stderr, "\n");

    // Pass cost by draw count. The one-draw bucket's mean IS the fixed per-pass cost plus one
    // draw's variable cost, so `passes x that mean` bounds what pass granularity is costing.
    uint64_t all_passes = 0, all_ns = 0;
    for (size_t b = 0; b < State::kBuckets; b++) {
        all_passes += s.bucket_passes[b].load(std::memory_order_relaxed);
        all_ns += s.bucket_ns[b].load(std::memory_order_relaxed);
    }
    if (all_passes) {
        const uint64_t first = s.first_pass_ns.load(std::memory_order_relaxed);
        const uint64_t last = s.last_pass_end_ns.load(std::memory_order_relaxed);
        const double span_ms = last > first ? (last - first) / 1e6 : 0.0;
        std::fprintf(stderr,
                     "[pass-cost] passes=%llu in-pass=%.1fms span=%.1fms (%.1f%% of span)",
                     static_cast<unsigned long long>(all_passes), all_ns / 1e6, span_ms,
                     span_ms > 0 ? 100.0 * (all_ns / 1e6) / span_ms : 0.0);
        for (size_t b = 0; b < State::kBuckets; b++) {
            const uint64_t p = s.bucket_passes[b].load(std::memory_order_relaxed);
            if (!p) continue;
            const uint64_t ns = s.bucket_ns[b].load(std::memory_order_relaxed);
            std::fprintf(stderr, "  d=%s: n=%llu mean=%.1fus", kBucketName[b],
                         static_cast<unsigned long long>(p),
                         static_cast<double>(ns) / static_cast<double>(p) / 1000.0);
        }
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}

DrawDispositionCensus& draw_disposition_census() {
    static DrawDispositionCensus census;
    // Registered on first use so a process that never renders prints nothing at all. NOT
    // std::atexit: every frontend here leaves through _exit()/_Exit(), which skips atexit
    // entirely (#3353), and a missing end-of-run line is indistinguishable from a zero. The
    // census state is atomics in a leaked function-local static, so it satisfies the
    // "no non-trivial destructors" requirement in exit_reports.hpp.
    static const bool once = [] {
        prosper::diagnostics::register_exit_report(
            [] { draw_disposition_census().report_totals(); });
        return true;
    }();
    (void)once;
    return census;
}

}  // namespace prosper::gpu

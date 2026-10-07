#include "gpu/diagnostics/draw_disposition.hpp"

#include "diagnostics/diag_ratelimit.hpp"
#include "diagnostics/exit_census.hpp"
#include "diagnostics/perf/perf_ledger.hpp"   // #3891: dropped-draws alarm

#include <atomic>
#include <chrono>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace prosper::gpu {
namespace {

constexpr size_t kReasonCount = static_cast<size_t>(DrawDrop::Count);

// Indexed by DrawDrop. Kept adjacent to the enum's declaration order on purpose: a reason added
// to one and not the other is a compile error via the static_assert below.
// clang-format off: one grepped name per line, in DrawDrop order
constexpr std::array<const char*, kReasonCount> kNames{
    "geometry-capability",
    "mesh-shape",
    "subgroup-features",
    "gds-allocation",
    "buffer-resources",
    "shader-rejected",
    "pipeline-creation",
    "target-memory",
    "resource-order",
    "resource-contract",
    "unproven-submission",
    "device-unavailable",
    "detile-device",
    "owned-wave",
    "ngg-expansion",
    "volume-view",
    "volume-not-persistent",
    "volume-feedback",
    "command-pool",
    "volume-multi-target",
    "volume-seeded",
    "volume-target-limits",
    "volume-depth-stencil",
    "volume-budget",
    "target-creation",
    "render-pass-creation",
    "framebuffer-creation",
    "pressure-flush",
    "ngg-subgroup",
    "volume-mixed-target",
};
// clang-format on
static_assert(kNames.size() == kReasonCount,
              "every DrawDrop needs a stable name; logs are grepped by these strings");
// The perf-alarm ledger mirrors DrawDrop as its backend/* drop reasons, in this order (#3891).
static_assert(static_cast<size_t>(prosper::diagnostics::perf::kFirstBackendDropReason) +
                      kReasonCount == prosper::diagnostics::perf::kDropReasonCount,
              "perf::DropReason's backend range must mirror DrawDrop one-for-one");
// The count check above cannot see ORDER: two branches each appending a reason can merge into
// tables that compile and mislabel every reason after the insertion point. Compare the names.
constexpr bool backend_names_mirror_draw_drop() {
    constexpr std::string_view prefix = "backend/";
    for (size_t i = 0; i < kReasonCount; i++) {
        const std::string_view ledger = prosper::diagnostics::perf::kDropReasonNames
            [static_cast<size_t>(prosper::diagnostics::perf::kFirstBackendDropReason) + i];
        if (!ledger.starts_with(prefix) || ledger.substr(prefix.size()) != kNames[i]) return false;
    }
    return true;
}
static_assert(backend_names_mirror_draw_drop(),
              "perf::kDropReasonNames' backend/* names must be DrawDrop's names, slot for slot");

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
    // is an opt-OUT, so a log that is silent about drops really did have none. The end-of-run
    // line gates on the same variable through register_census; this one also gates the PER-PASS
    // report, which has no exit hook to gate it.
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

// Per-pass figures are THREAD-LOCAL. A pass runs on one thread from its entry scope to its report
// (every census call in the backend is on the pass's own thread; its memcpy workers count
// nothing), but passes on different threads overlap: `seen` is counted at the pass entry, before
// the backend's persistent-resource lock serialises them. A shared per-pass counter charged one
// thread's entry draws to whichever pass reported first -- `backend_persistent_resource_lock`'s
// concurrent arm read `seen=4 recorded=1 UNACCOUNTED=3` with nothing lost. Process totals stay in
// the shared relaxed atomics above.
struct PassCounters {
    uint64_t seen = 0;
    uint64_t recorded = 0;
    std::array<uint64_t, kReasonCount> dropped{};
};
PassCounters& pass_counters() {
    static thread_local PassCounters counters;
    return counters;
}

DrawDispositionCensus::State& DrawDispositionCensus::state() const {
    static State s;
    return s;
}

// A capture or diagnostic re-realization is not live execution (#3951): its passes count nothing,
// so it can neither raise `dropped-draws` nor `unaccounted-draws` while being used to investigate
// them. One thread-local read per call; the counters stay relaxed atomics (P4).
static bool suppressed() {
    return prosper::diagnostics::perf::thread_draw_drop_suppression() != 0;
}

void DrawDispositionCensus::note_seen(uint64_t count) {
    if (suppressed()) return;
    auto& s = state();
    s.seen.fetch_add(count, std::memory_order_relaxed);
    pass_counters().seen += count;
}

void DrawDispositionCensus::note_rebatched(uint64_t from, uint64_t to) {
    if (suppressed() || from == to) return;
    // Unsigned wrap-around makes the subtraction exact: the pass already holds `from` seen draws.
    auto& s = state();
    s.seen.fetch_add(to - from, std::memory_order_relaxed);
    pass_counters().seen += to - from;
}

void DrawDispositionCensus::note_recorded(uint64_t count) {
    if (suppressed()) return;
    auto& s = state();
    s.recorded.fetch_add(count, std::memory_order_relaxed);
    pass_counters().recorded += count;
}

void DrawDispositionCensus::note_dropped(DrawDrop reason, uint64_t count) {
    const auto i = static_cast<size_t>(reason);
    if (i >= kReasonCount || !count || suppressed()) return;
    auto& s = state();
    s.dropped[i].fetch_add(count, std::memory_order_relaxed);
    pass_counters().dropped[i] += count;
    // #3891: the same reason, in the alarm ledger's backend/* range.
    namespace perf = prosper::diagnostics::perf;
    perf::drop_draw(
        static_cast<perf::DropReason>(static_cast<size_t>(perf::kFirstBackendDropReason) + i),
        count);
}

uint64_t DrawDispositionCensus::pass_seen_for_scope() const {
    return pass_counters().seen;
}

uint64_t DrawDispositionCensus::pass_unaccounted_for_scope() const {
    const PassCounters& pass = pass_counters();
    uint64_t accounted = pass.recorded;
    for (size_t i = 0; i < kReasonCount; i++) accounted += pass.dropped[i];
    return pass.seen > accounted ? pass.seen - accounted : 0;
}

uint64_t DrawDispositionCensus::timed_passes() const {
    uint64_t passes = 0;
    for (const auto& bucket : state().bucket_passes)
        passes += bucket.load(std::memory_order_relaxed);
    return passes;
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
    const PassCounters pass = std::exchange(pass_counters(), PassCounters{});
    const uint64_t pass_seen = pass.seen;
    const uint64_t pass_recorded = pass.recorded;
    const std::array<uint64_t, kReasonCount>& pass_dropped = pass.dropped;
    uint64_t pass_dropped_total = 0;
    for (size_t i = 0; i < kReasonCount; i++) pass_dropped_total += pass_dropped[i];
    // #3891 unaccounted-draws: the census's blind spot as a perf-alarm counter, once per pass and
    // independent of the report switch below.
    if (pass_seen || pass_recorded || pass_dropped_total) {
        const int64_t diff = static_cast<int64_t>(pass_seen) - static_cast<int64_t>(pass_recorded) -
                             static_cast<int64_t>(pass_dropped_total);
        if (diff) prosper::diagnostics::perf::add(prosper::diagnostics::perf::Counter::DrawsUnaccounted,
                                                  static_cast<uint64_t>(diff < 0 ? -diff : diff));
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

DrawDispositionPassScope::DrawDispositionPassScope(uint64_t draws) : start_ns_(now_ns()) {
    draw_disposition_census().note_seen(draws);
}

DrawDispositionPassScope::~DrawDispositionPassScope() {
    auto& census = draw_disposition_census();
    // A named refusal drops whatever the pass had not yet recorded or dropped. Nothing named
    // leaves the gap for report_pass() to call UNACCOUNTED.
    if (refusal_ != DrawDrop::Count)
        census.note_dropped(refusal_, census.pass_unaccounted_for_scope());
    // Read the pass's draw count BEFORE report_pass() resets the per-pass counters. A refused
    // pass returned before building anything, so it is no sample of a pass's FIXED cost: timing
    // it would pull the small-draw bucket means toward zero (every refusal precedes the loop).
    const uint64_t drawn = census.pass_seen_for_scope();
    if (drawn && refusal_ == DrawDrop::Count)
        census.note_pass_duration(drawn, now_ns() - start_ns_);
    census.report_pass();
}

void refuse_draw_pass(uint64_t draws, DrawDrop reason) {
    DrawDispositionPassScope scope(draws);
    scope.refuse(reason);
}

bool DrawDispositionCensus::report_totals() {
    auto& s = state();
    const uint64_t total_seen = s.seen.load(std::memory_order_relaxed);
    if (total_seen == 0) return false;
    const uint64_t total_recorded = s.recorded.load(std::memory_order_relaxed);
    uint64_t total_dropped = 0;
    for (size_t i = 0; i < kReasonCount; i++)
        total_dropped += s.dropped[i].load(std::memory_order_relaxed);
    // No caller proves that CPU pass preparation has stopped. Even balanced independent loads
    // are not a final total; draining the GPU submit gate does not join the guest (#3973).
    std::fprintf(stderr, "[draw-disposition] RUN SNAPSHOT quiescence=unverified "
                         "seen=%llu recorded=%llu dropped=%llu",
                 static_cast<unsigned long long>(total_seen),
                 static_cast<unsigned long long>(total_recorded),
                 static_cast<unsigned long long>(total_dropped));
    for (size_t i = 0; i < kReasonCount; i++) {
        const uint64_t n = s.dropped[i].load(std::memory_order_relaxed);
        if (n) std::fprintf(stderr, " %s=%llu", kNames[i], static_cast<unsigned long long>(n));
    }
    if (total_recorded + total_dropped != total_seen)
        std::fprintf(stderr, "  snapshot-delta=%lld",
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
    return true;
}

DrawDispositionCensus& draw_disposition_census() {
    static DrawDispositionCensus census;
    // Registered on first use so a process that never renders prints nothing at all. NOT
    // std::atexit: every frontend here leaves through _exit()/_Exit(), which skips atexit
    // entirely (#3353), and a missing end-of-run line is indistinguishable from a zero. The
    // census state is atomics in a leaked function-local static, so it satisfies the
    // "no non-trivial destructors" requirement in exit_reports.hpp.
    static const bool once = [] {
        prosper::diagnostics::register_census(
            "PROSPER_NO_DRAW_DISPOSITION",
            [] { return draw_disposition_census().report_totals(); });
        return true;
    }();
    (void)once;
    return census;
}

}  // namespace prosper::gpu

#pragma once
// The frame ledger behind `[perf-alarm]` (#3891): always-on, near-zero-cost accumulation of the
// costs and counters the alarm rules read.
//
// WHY THIS EXISTS. The 2026-09-27 performance pass (#3873) found six problems by hand that prosper
// already had the numbers for -- a texture cache refusing every admission, a CPU readback of a GPU
// surface ~65 times per 5 s, a VideoOut lock blocking a guest thread 85% of the time -- but only
// behind an instrument somebody had to know to switch on. The ledger is the part that is always
// running, so the alarm engine (perf_alarms.hpp) can say so without being asked.
//
// THE COST RULE, and it is the whole design constraint. Every hook here is either
//   * a relaxed atomic add on a COUNTER (per event: a texture refusal, a dropped draw) -- or, for
//     the one per-texture-reference tally, a plain thread-local increment flushed per pass, or
//   * a pair of clock reads around a COARSE event: one readback, one blocking lock wait, one
//     present, one backend pass group. Never a clock read per draw or per texture reference.
// PROSPER_RENDER_TIMING's per-draw clock reads measured ~9% of the render thread, which is why the
// existing timing census is opt-in; this must be cheap enough to leave on. Nothing here allocates,
// takes a lock, or reads the environment after first use.
//
// Header-only on purpose: render_runner.h is compiled straight into dozens of Vulkan tests that
// link only the translation units they name, and every one of them would otherwise need a new
// source added to its CMake target. The one function-local static below is a single object across
// all translation units (an inline function's statics are shared by the ODR), never destroyed.
//
// `PROSPER_NO_PERF_ALARMS=1` turns the whole thing off: hooks skip their clock reads and the
// engine never evaluates. It exists for the overhead A/B and as the global opt-out.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace prosper::diagnostics::perf {

// Time-carrying categories. Each accumulates total nanoseconds, an event count and the largest
// single event of the current window.
enum class Cost : uint8_t {
    // A synchronous CPU readback of a GPU surface (depth or colour target), measured from entry to
    // the bytes being in CPU memory -- so it includes the flush and fence wait that the readback
    // forces, which is the actual cost to the calling thread.
    SurfaceReadback = 0,
    // Time a guest thread spent blocked acquiring an HLE lock held by another thread. Only the
    // contended path is timed (an uncontended try_lock costs no clock read).
    HleBlockingWait,
    // CPU work on the host present thread per GPU-presented frame, outside its waits (fence,
    // acquire, queue lock): the --fps content signature when it is on, slot release, and command
    // recording. Measured on every present in prosper-app's GPU-present path; frontends without
    // that path (tools/screenshot, the CPU present fallback) record nothing, which the exit
    // summary reports as "no data" rather than as quiet.
    PresentCpu,
    // Frontend resource materialisation (build_bds) per backend pass group.
    FrontendBuild,
    // Resolution of ONE sampled texture reference in the frontend (1 in kTextureRefSamplePeriod),
    // excluding any surface readback nested inside it -- that is charged to SurfaceReadback, so one
    // cause does not raise two alarms.
    TextureRefSample,
    Count
};

// Plain event counters.
enum class Counter : uint8_t {
    TextureReferences = 0,     // sampled texture references the frontend resolved
    TextureCacheMisses,        // persistent resident-texture lookups that missed
    TextureCacheRefusals,      // misses refused admission for lack of room (not for size)
    TextureCacheRefusedBytes,  // image bytes of those refusals: re-uploaded, then freed
    TextureCacheEvictions,     // resident images evicted to make room
    DroppedDrawsBackend,       // draws the backend wanted to issue and could not (draw_disposition)
    DroppedDrawsFrontend,      // draws the frontend rejected: a resource did not resolve
    DroppedDrawsContract,      // draws the frontend rejected: descriptor contract validation failed
    DeviceAllocations,         // successful vkAllocateMemory calls
    Count
};

// Last-written values (not summed): the state a State rule reads.
enum class Gauge : uint8_t {
    TextureCacheBytes = 0,
    TextureCacheLimit,
    Count
};

constexpr size_t kCostCount = static_cast<size_t>(Cost::Count);
constexpr size_t kCounterCount = static_cast<size_t>(Counter::Count);
constexpr size_t kGaugeCount = static_cast<size_t>(Gauge::Count);

struct Ledger {
    std::atomic<uint64_t> cost_ns[kCostCount] = {};
    std::atomic<uint64_t> cost_events[kCostCount] = {};
    // Largest single event since the engine last took it (exchange(0) at each window close).
    std::atomic<uint64_t> cost_max_ns[kCostCount] = {};
    std::atomic<uint64_t> counters[kCounterCount] = {};
    std::atomic<uint64_t> gauges[kGaugeCount] = {};
    // The one attribution string a cost may carry: which HLE lock blocked, for instance. A pointer
    // to a string literal, stored without copying.
    std::atomic<const char*> cost_label[kCostCount] = {};
};

inline bool enabled() {
    static const bool on = std::getenv("PROSPER_NO_PERF_ALARMS") == nullptr;
    return on;
}

inline Ledger& ledger() {
    static Ledger* const instance = new Ledger();  // never destroyed: exit reports read it
    return *instance;
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void add_cost(Cost c, uint64_t ns, uint64_t events = 1, const char* label = nullptr) {
    Ledger& l = ledger();
    const size_t i = static_cast<size_t>(c);
    l.cost_ns[i].fetch_add(ns, std::memory_order_relaxed);
    l.cost_events[i].fetch_add(events, std::memory_order_relaxed);
    uint64_t seen = l.cost_max_ns[i].load(std::memory_order_relaxed);
    while (ns > seen &&
           !l.cost_max_ns[i].compare_exchange_weak(seen, ns, std::memory_order_relaxed)) {}
    if (label) l.cost_label[i].store(label, std::memory_order_relaxed);
}

inline void add(Counter c, uint64_t n = 1) {
    if (n) ledger().counters[static_cast<size_t>(c)].fetch_add(n, std::memory_order_relaxed);
}

inline void set(Gauge g, uint64_t v) {
    ledger().gauges[static_cast<size_t>(g)].store(v, std::memory_order_relaxed);
}

// Per-thread tally of texture references, for sites hot enough that even an uncontended atomic add
// per reference is worth avoiding: increment here, flush with flush_thread_texture_references() at
// a coarse boundary (once per backend pass group).
inline uint64_t& thread_texture_references() {
    static thread_local uint64_t n = 0;
    return n;
}
inline void flush_thread_texture_references() {
    uint64_t& n = thread_texture_references();
    add(Counter::TextureReferences, n);
    n = 0;
}

// Nanoseconds this thread has charged to each category through CostScope, so a span can subtract
// the part of its time that a nested, separately-charged event already accounts for.
inline uint64_t& thread_cost_ns(Cost c) {
    static thread_local uint64_t ns[kCostCount] = {};
    return ns[static_cast<size_t>(c)];
}

// On average one in this many texture references is timed. Per-reference clock reads on every
// reference are exactly what PROSPER_RENDER_TIMING pays ~9% for; one in 32 costs a thirty-second
// of that. The choice is a per-thread xorshift, NOT a counter: a counter with period 32 always
// samples the same slot of a draw with 4, 8 or 16 texture references, so an expensive reference in
// another slot would never be timed. Random selection keeps the mean unbiased for any layout.
constexpr uint64_t kTextureRefSamplePeriod = 32;
static_assert((kTextureRefSamplePeriod & (kTextureRefSamplePeriod - 1)) == 0,
              "the sampler masks with period-1");

inline bool sample_texture_reference() {
    static thread_local uint32_t x = 0x9e3779b9u ^
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&x) >> 4);  // distinct per thread
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return (x & (kTextureRefSamplePeriod - 1)) == 0;
}

// Counts one frontend texture reference and times it when it is the sampled one. finish() at the
// end of resolution records it; a reference that leaves early (a dropped draw) is not recorded.
class TextureReferenceSample {
public:
    explicit TextureReferenceSample(bool texture) {
        if (!texture) return;
        ++thread_texture_references();
        sampled_ = enabled() && sample_texture_reference();
        if (sampled_) {
            nested_ns_ = thread_cost_ns(Cost::SurfaceReadback);
            begin_ = now_ns();
        }
    }
    void finish() {
        if (!sampled_) return;
        sampled_ = false;
        const uint64_t elapsed = now_ns() - begin_;
        const uint64_t nested = thread_cost_ns(Cost::SurfaceReadback) - nested_ns_;
        add_cost(Cost::TextureRefSample, elapsed > nested ? elapsed - nested : 0);
    }

private:
    bool sampled_ = false;
    uint64_t begin_ = 0;
    uint64_t nested_ns_ = 0;
};

// Times one coarse event into `c`. Nested scopes of the same category on one thread count only the
// OUTERMOST, so a readback helper that calls another readback helper is one event, not two, and
// its time is not charged twice.
class CostScope {
public:
    // enabled() is evaluated once per scope side; it is a cached constant after first use, so the
    // increment here and the decrement in the destructor always pair.
    explicit CostScope(Cost c, const char* label = nullptr)
        : cost_(c), label_(label), outer_(enabled() && depth(c)++ == 0),
          begin_(outer_ ? now_ns() : 0) {}
    ~CostScope() {
        if (!enabled()) return;
        --depth(cost_);
        if (!outer_) return;
        const uint64_t ns = now_ns() - begin_;
        add_cost(cost_, ns, 1, label_);
        thread_cost_ns(cost_) += ns;
    }
    CostScope(const CostScope&) = delete;
    CostScope& operator=(const CostScope&) = delete;

private:
    static uint32_t& depth(Cost c) {
        static thread_local uint32_t depths[kCostCount] = {};
        return depths[static_cast<size_t>(c)];
    }
    Cost cost_;
    const char* label_;
    bool outer_;
    uint64_t begin_;
};

}  // namespace prosper::diagnostics::perf

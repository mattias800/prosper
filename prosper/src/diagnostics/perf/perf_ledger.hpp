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
    // One RDNA2 -> SPIR-V recompilation (a recompiler cache MISS; hits cost nothing here). Summed
    // over threads, like HleBlockingWait: parallel draw realization can compile on several.
    ShaderCompile,
    // One Vulkan pipeline creation on a pipeline-cache miss: vkCreateGraphicsPipelines /
    // vkCreateComputePipelines plus the wait for the driver-cache lock each takes (shader-module
    // creation is NOT inside it). The driver's own compile lives here. Summed thread time, so
    // threads queued on that lock each count their wait.
    PipelineCreate,
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
    SkippedDispatches,         // compute dispatches prosper wanted to run and did not (DispatchSkip)
    Count
};

// Last-written values (not summed): the state a State rule reads.
enum class Gauge : uint8_t {
    TextureCacheBytes = 0,
    TextureCacheLimit,
    Count
};

// WHY a draw was dropped: one stable code per drop SITE, so `[perf-alarm] rule=dropped-draws` can
// name the site instead of a bare count (#3891 phase 3; #3893 is the case that needed it -- ~2
// frontend drops per flip on Sonic Frontiers' menus, from one of fifteen sites nobody could name).
// Each site pays one relaxed atomic add per DROPPED draw; accepted draws pay nothing.
//
// Frontend codes are the `built.complete = false` sites of the live renderer's resource builder
// (frontends/shared/live/live_renderer.cpp) plus its descriptor-contract check; the render-array
// codes reuse the site names that `[render-array-reject] site=` already prints. Backend codes mirror
// prosper::gpu::DrawDrop in its order -- draw_disposition.cpp static_asserts that, so a reason added
// there without one here fails to compile.
enum class DropReason : uint8_t {
    // complete == false with no code recorded: a drop site added without a reason. Nonzero here is
    // the instrument naming its own blind spot, the same contract as draw_disposition's UNACCOUNTED.
    Unattributed = 0,
    ContractMismatch,          // resources resolved, but the descriptor contract check failed
    VolumeInteriorAlias,       // samples inside an unpublished renderer-owned volume
    VolumeShapeMismatch,       // retained volume, descriptor shape cannot consume its image
    VolumeNoRendererImage,     // retained volume with no renderer image to sample (#3889)
    ArrayFloat32Shape,         // Float32 layered T# with a reflected image shape we cannot bind
    ArraySingleColorRtt,       // Float32 array aliases a single-layer colour RTT
    ArrayShapeBudget,          // Float32 array shape/components/decode budget out of range
    ArrayFormatSupport,        // device cannot sample RGBA32F as needed
    ArrayNoncanonicalDepth,    // unproven retained depth write-base alias
    ArrayDepthView,            // unsupported retained depth view
    ArrayDepthStride,          // unproven retained depth layer stride
    ArrayProducerCompletion,   // retained depth producer submit did not complete
    ArrayDepthUnavailable,     // retained depth array could not be read
    ArrayCompressedNoDepth,    // compressed Float32 array with no retained depth
    ArrayCompressedGuest,      // compressed Float32 array would decode guest backing
    ArrayShortBacking,         // Float32 array guest backing shorter than its slices
    BackendGeometryCapability,
    BackendMeshShape,
    BackendSubgroupFeatures,
    BackendGdsAllocation,
    BackendBufferResources,
    BackendShaderRejected,
    BackendPipelineCreation,
    Count
};
constexpr DropReason kFirstBackendDropReason = DropReason::BackendGeometryCapability;
constexpr size_t kDropReasonCount = static_cast<size_t>(DropReason::Count);
// Stable, grepped: never reword casually.
constexpr const char* kDropReasonNames[kDropReasonCount] = {
    "unattributed",
    "contract-mismatch",
    "volume-interior-alias",
    "volume-shape-mismatch",
    "volume-no-renderer-image",
    "render-array-reject/float32-shape",
    "render-array-reject/single-color-rtt",
    "render-array-reject/shape-budget",
    "render-array-reject/format-support",
    "render-array-reject/noncanonical-depth",
    "render-array-reject/depth-view",
    "render-array-reject/depth-stride",
    "render-array-reject/producer-completion",
    "render-array-reject/depth-unavailable",
    "render-array-reject/compressed-no-depth",
    "render-array-reject/compressed-guest",
    "render-array-reject/short-backing",
    "backend/geometry-capability",
    "backend/mesh-shape",
    "backend/subgroup-features",
    "backend/gds-allocation",
    "backend/buffer-resources",
    "backend/shader-rejected",
    "backend/pipeline-creation",
};

// WHY a compute dispatch prosper wanted to run did not run. A skipped dispatch leaves its output
// stale or zero -- a LUT, an exposure value, a light list -- and, like a dropped draw, can make a
// run look faster while rendering wrong. NOT counted, each pinned by a test arm in
// tests/gpu/execute/test_gpu_execute.cpp: deliberate declines (PROSPER_COMPUTE_SKIP_PROGRAM's
// selector, reported through note_deliberate_dispatch_decline; the parent-walk diagnostic), every
// indirect-dependency skip later in a submit that had a deliberate decline (the broken producer
// epoch carries through parser stalls, so the causes cannot be separated there), and every
// dispatch of a process with no compute backend at all.
enum class DispatchSkip : uint8_t {
    MissingProgram = 0,     // no registered/readable shader at the program address
    ShaderRecompile,        // the recompiler produced no SPIR-V
    DescriptorContract,     // SPIR-V and the realized resource table disagree
    IndirectDependencies,   // an indirect dispatch's producer had not landed for this submit
    IndirectArguments,      // indirect arguments null, misaligned or unreadable
    BackendDeclined,        // realized, but the live compute backend refused it
    Count
};
constexpr size_t kDispatchSkipCount = static_cast<size_t>(DispatchSkip::Count);
constexpr const char* kDispatchSkipNames[kDispatchSkipCount] = {
    "missing-program", "shader-recompile", "descriptor-contract",
    "indirect-dependencies", "indirect-arguments", "backend-declined",
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
    std::atomic<uint64_t> drop_reasons[kDropReasonCount] = {};
    std::atomic<uint64_t> dispatch_skips[kDispatchSkipCount] = {};
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

// One dropped draw, with the site's reason. Also bumps the matching coarse counter (frontend,
// contract or backend), so the totals the rule reads and the breakdown can never disagree.
inline void drop_draw(DropReason reason) {
    const size_t i = static_cast<size_t>(reason);
    if (i >= kDropReasonCount) return;
    add(reason == DropReason::ContractMismatch ? Counter::DroppedDrawsContract
        : i >= static_cast<size_t>(kFirstBackendDropReason) ? Counter::DroppedDrawsBackend
                                                             : Counter::DroppedDrawsFrontend);
    ledger().drop_reasons[i].fetch_add(1, std::memory_order_relaxed);
}

// Re-realizations that are not live execution (an F9 capture re-realizing a submit's dispatches
// for its bundle) must not count as skips a second time: a capture would otherwise raise the
// correctness alarm it is being used to investigate.
inline uint32_t& thread_dispatch_skip_suppression() {
    static thread_local uint32_t depth = 0;
    return depth;
}
class SuppressDispatchSkipCounting {
public:
    SuppressDispatchSkipCounting() { ++thread_dispatch_skip_suppression(); }
    ~SuppressDispatchSkipCounting() { --thread_dispatch_skip_suppression(); }
    SuppressDispatchSkipCounting(const SuppressDispatchSkipCounting&) = delete;
    SuppressDispatchSkipCounting& operator=(const SuppressDispatchSkipCounting&) = delete;
};

// One compute dispatch prosper wanted to run and did not, with its reason.
inline void skip_dispatch(DispatchSkip reason) {
    const size_t i = static_cast<size_t>(reason);
    if (i >= kDispatchSkipCount || thread_dispatch_skip_suppression()) return;
    add(Counter::SkippedDispatches);
    ledger().dispatch_skips[i].fetch_add(1, std::memory_order_relaxed);
}

// A DELIBERATE decline inside the compute backend (PROSPER_COMPUTE_SKIP_PROGRAM's selector) looks,
// from the executor, exactly like a refusal: the backend returns false. The backend calls this on
// the declining thread -- the executor calls the backend synchronously on its own thread -- and
// BackendDispatchOutcome below tells the two apart by whether the count moved during the call.
inline uint64_t& thread_deliberate_dispatch_declines() {
    static thread_local uint64_t n = 0;
    return n;
}
inline void note_deliberate_dispatch_decline() { ++thread_deliberate_dispatch_declines(); }

// Brackets one call into the compute backend. finish(executed) counts a backend-declined skip only
// for a refusal, and returns true when the dispatch was deliberately declined, so the caller can
// also exempt what that decline caused (the later indirect dispatches whose producer epoch it
// broke).
class BackendDispatchOutcome {
public:
    BackendDispatchOutcome() : before_(thread_deliberate_dispatch_declines()) {}
    bool finish(bool executed) const {
        const bool deliberate = thread_deliberate_dispatch_declines() != before_;
        if (!executed && !deliberate) skip_dispatch(DispatchSkip::BackendDeclined);
        return deliberate;
    }

private:
    uint64_t before_;
};

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

// The per-thread sampler seed, from the address of the thread's own state (distinct per thread).
// Never zero: xorshift maps 0 to 0, so a zero seed would select EVERY reference (0 masked by
// period-1 is 0) and time all of them -- the cost this sampler exists to avoid. The address term
// can cancel the constant exactly (address>>4 == 0x9e3779b9), and `| 1` rules that out.
constexpr uint32_t texture_sample_seed(uintptr_t state_address) {
    return (0x9e3779b9u ^ static_cast<uint32_t>(state_address >> 4)) | 1u;
}

inline bool sample_texture_reference() {
    static thread_local uint32_t x = texture_sample_seed(reinterpret_cast<uintptr_t>(&x));
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

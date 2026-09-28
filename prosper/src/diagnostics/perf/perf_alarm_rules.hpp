#pragma once
// The `[perf-alarm]` rules (#3891): a PURE function from one window of ledger deltas to the alarms
// that window raises. No clock, no I/O, no globals -- so every rule is unit-testable by building
// the bad condition by hand (tests/diagnostics/test_perf_alarms.cpp).
//
// Each rule was written against a problem the 2026-09-27 pass (#3873) found by hand, and is
// validated by firing on the commit before that fix and staying quiet after it. The thresholds are
// named constants with the reason for their value beside them; `PROSPER_PERF_ALARM_THRESHOLD_PCT`
// scales every non-correctness threshold (25 = alarm at a quarter of the default, for an
// investigation), and correctness rules stay at "any".
//
// Rule classes (the issue's vocabulary): SHARE of the title's frame budget, which comes from the
// guest's own sceVideoOutSetFlipRate rather than an assumed 60 Hz; RATE per second; STATE (a cache
// that cannot evict); CORRECTNESS (any dropped draw or skipped compute dispatch -- a drop can make a
// run look FASTER, so it is an alarm and not just a counter, and it names the site that dropped --
// and any GPU-only allocation placed off device-local memory, which no pixel test can see).

#include "diagnostics/perf/perf_ledger.hpp"
#include "diagnostics/transfer_pressure.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::diagnostics::perf {

// ---- thresholds ------------------------------------------------------------------------------

// texture-cache-thrash: a refusal is a miss the resident cache would have kept had it been able to
// evict. A working cache refuses ~0 (eviction makes room); Outer Wilds before #3876 refused ~2,400
// per 5 s with 0 evictions. 50/s leaves an order of magnitude on both sides.
constexpr double kTextureRefusalsPerSecond = 50.0;

// surface-readback: synchronous GPU->CPU surface copies (with their forced flush + wait) costing a
// quarter of every frame's budget are worth a GPU path. Sonic Frontiers before #3882 measured
// 60-78% in gameplay; GTA V and Outer Wilds on main sit at 7-15% of a 60 Hz budget from many
// sub-millisecond readbacks, a real but second-order cost that should not read as an alarm.
constexpr double kReadbackBudgetShare = 0.25;
// ...and not a handful of one-off readbacks during a load, which cost once and are not a pattern.
constexpr double kReadbackMinPerSecond = 2.0;

// texture-reference-cost: mean frontend resolution time of one sampled texture reference (1 in
// kTextureRefSamplePeriod, readbacks nested inside excluded). GTA V before #3877 walked the whole
// RTT cache per reference; set between the measured pre- and post-fix means (see the PR for #3891).
constexpr double kTextureReferenceNs = 4000.0;
// ...and only when mean x references is a material share of the frame: a slow mean over a handful
// of references costs nothing.
constexpr double kFrontendBuildBudgetShare = 0.25;
// ...over enough samples for the mean to mean something (20/s is 100 per 5 s window).
constexpr double kTextureReferenceMinPerSecond = 20.0;

// hle-blocking-wait: blocked time on an HLE lock another thread holds, SUMMED over every blocked
// thread, as a share of wall time -- so 20% is a fifth of one thread-equivalent, and several
// threads can together exceed 100%. The Messenger before #3879: 75-90%, almost all of it its main
// thread blocked in sceVideoOutSetFlipRate behind a flipper sleeping with the handle lock held.
constexpr double kHleBlockedThreadShare = 0.20;

// present-cpu-overhead: CPU work per presented frame on the host present thread, outside GPU waits.
// Before #3875 the --fps content sample was read from uncached memory at ~3 ms per present; a
// cached read is well under 0.1 ms. 1 ms is a sixteenth of a 60 Hz frame spent on presentation
// bookkeeping.
constexpr double kPresentCpuMsPerPresent = 1.0;
// ...averaged over enough presents to be a rate rather than one slow frame.
constexpr double kPresentMinPerSecond = 5.0;

// dropped-draws: CORRECTNESS. Any draw prosper wanted to issue and dropped. GTA V between #3842
// and #3889 dropped every draw sampling its colour-grading LUT, silently, for three days.
constexpr uint64_t kDroppedDrawsPerWindow = 1;

// skipped-dispatches: CORRECTNESS, like dropped-draws. A skipped compute dispatch leaves its
// output unwritten (GTA V's missing world was one declined program, #2481).
constexpr uint64_t kSkippedDispatchesPerWindow = 1;

// gpu-memory-off-device: CORRECTNESS class (the frame is right, but on a discrete GPU every access
// to the resource crosses the bus, and no pixel test can see it). Any GPU-only renderer allocation
// placed in a non-device-local type while the device has device-local memory: either
// vkAllocateMemory ran out of device memory and the #3897 fallback retried a host type, or the
// resource's memoryTypeBits allowed no device-local type at all. On this project's APU neither
// happens, so a firing here is news.
constexpr uint64_t kGpuMemoryOffDevicePerWindow = 1;

// host-copy-pressure: bytes this process copies host-side on the guest's behalf, per second -- the
// sum over EVERY default-on [transfer-pressure] category (storage-materialize, buffer-upload,
// buffer-compare, rtt-snapshot, detile), the same total that census's HIGH line uses. Its
// `breakdown` is in whole MiB per category, not a count like the correctness rules' breakdowns, so
// its summary "breakdown over fired windows" is MiB too. 256 MiB/s is that census's own HIGH line, which healthy titles sit
// far below (GTA V on main: ~80 MiB/s over a whole route) and every defect it was written against
// far above (a static splash copying 770 MiB/s; Astro Bot's compute round trip at ~2 GiB/s, #3871).
constexpr double kHostCopyMiBPerSecond = 256.0;

// shader-compile: RDNA2->SPIR-V recompiles plus Vulkan pipeline creations, summed over threads, as a
// share of the title's frame budget per flip. Compiling is expected while new content streams in,
// so the bar is high and must be sustained: a pipeline or recompiler cache that never hits (a key
// that includes a field that changes every frame, a cache evicting what it just built) compiles
// steadily for as long as the scene runs, where a cold-cache load compiles in a decaying burst.
constexpr double kShaderCompileBudgetShare = 0.25;
// ...over a real population of compiles (100 per 5 s window), not a handful of slow ones. This floor
// is also what ends a cold-cache burst: its compile count decays below 100/window within 15-25 s
// (measured on current main with FRESH driver and pipeline caches -- the worst case: Sonic Frontiers
// 4 consecutive windows over it on entering gameplay, GTA V 3, Outer Wilds 5), while a cache that
// never hits stays at hundreds or thousands per window.
constexpr double kShaderCompileMinPerSecond = 20.0;
// Eight windows is 40 s, three windows beyond the longest cold-cache burst measured.
constexpr uint32_t kShaderCompileSustainWindows = 8;

// SUSTAIN: consecutive windows a rule's condition must hold before the engine reports it. A cost
// that lasts one window is usually a load, and a steady-state alarm should not fire on it; a
// correctness alarm fires on the first window.
constexpr uint32_t kSustainWindows = 2;
// texture-reference-cost needs three: GTA V on main spends the first 10-15 s of gameplay resolving
// cold references at 6-7 us each (two consecutive windows at most) before settling at ~1 us, while
// before #3877 it stayed at 5-8 us for the whole run.
constexpr uint32_t kTextureReferenceSustainWindows = 3;
constexpr uint32_t kCorrectnessSustainWindows = 1;

// ---- evaluation ------------------------------------------------------------------------------

// One window of ledger DELTAS (maxima are per window), plus the frame budget in force.
struct WindowSample {
    double seconds = 0;
    uint64_t flips = 0;
    // From the guest's sceVideoOutSetFlipRate (0=60, 1=30, 2=20 Hz); 60 Hz until it asks.
    uint32_t target_hz = 60;
    uint64_t cost_ns[kCostCount] = {};
    uint64_t cost_events[kCostCount] = {};
    uint64_t cost_max_ns[kCostCount] = {};
    const char* cost_label[kCostCount] = {};
    uint64_t counters[kCounterCount] = {};
    uint64_t gauges[kGaugeCount] = {};
    // Per-reason breakdowns of DroppedDraws* and SkippedDispatches (#3891 phase 3).
    uint64_t drop_reasons[kDropReasonCount] = {};
    uint64_t dispatch_skips[kDispatchSkipCount] = {};
    // Per-class breakdown of GpuMemoryOffDevice, and each slot's name (nullptr: never recorded).
    uint64_t gpu_memory_off_device[kGpuMemoryClassSlots] = {};
    const char* gpu_memory_class_names[kGpuMemoryClassSlots] = {};
    // Host-copy bytes per [transfer-pressure] category (diagnostics::Transfer).
    static constexpr size_t kTransferCount = static_cast<size_t>(Transfer::Count);
    uint64_t transfer_bytes[kTransferCount] = {};

    double budget_ms() const { return target_hz ? 1000.0 / target_hz : 1000.0 / 60.0; }
    double ms(Cost c) const { return cost_ns[static_cast<size_t>(c)] / 1e6; }
    uint64_t events(Cost c) const { return cost_events[static_cast<size_t>(c)]; }
    uint64_t count(Counter c) const { return counters[static_cast<size_t>(c)]; }
    uint64_t gauge(Gauge g) const { return gauges[static_cast<size_t>(g)]; }
    // A category's time per guest flip, as a fraction of the title's frame budget.
    double budget_share(Cost c) const {
        return flips ? ms(c) / static_cast<double>(flips) / budget_ms() : 0.0;
    }
};

// The thresholds actually applied. `scaled(p)` multiplies every non-correctness threshold by p/100.
struct RuleThresholds {
    double texture_refusals_per_s = kTextureRefusalsPerSecond;
    double readback_budget_share = kReadbackBudgetShare;
    double readback_min_per_s = kReadbackMinPerSecond;
    double texture_reference_ns = kTextureReferenceNs;
    double frontend_build_budget_share = kFrontendBuildBudgetShare;
    double texture_reference_min_per_s = kTextureReferenceMinPerSecond;
    double hle_blocked_thread_share = kHleBlockedThreadShare;
    double present_cpu_ms = kPresentCpuMsPerPresent;
    double present_min_per_s = kPresentMinPerSecond;
    uint64_t dropped_draws = kDroppedDrawsPerWindow;
    uint64_t skipped_dispatches = kSkippedDispatchesPerWindow;
    uint64_t gpu_memory_off_device = kGpuMemoryOffDevicePerWindow;
    double host_copy_mib_per_s = kHostCopyMiBPerSecond;
    double shader_compile_budget_share = kShaderCompileBudgetShare;
    double shader_compile_min_per_s = kShaderCompileMinPerSecond;

    static RuleThresholds scaled(double percent);
};

struct AlarmFiring {
    const char* rule = "";
    double value = 0;          // the measured quantity the threshold is compared against
    const char* unit = "";
    double threshold = 0;
    double cost_ms = -1;       // estimated time cost in this window, or <0 when not a time
    std::string detail;        // key=value context: counts, state, the budget in force
    const char* hint = "";     // the subsystem and the next instrument to reach for
    // Named sub-counts, largest first (a correctness rule's per-reason breakdown). The JSONL alarm
    // record carries it as an object; the log line carries its top entries in `detail`.
    std::vector<std::pair<const char*, uint64_t>> breakdown;
};

// Non-zero entries of `counts`, largest first (ties in index order), as (name, count) pairs.
std::vector<std::pair<const char*, uint64_t>> ranked(const uint64_t* counts,
                                                     const char* const* names, size_t n);
// "name:count,name:count,name:count(+2 more)" -- the top `top` entries of a ranked breakdown.
std::string top_entries(const std::vector<std::pair<const char*, uint64_t>>& breakdown,
                        size_t top = 3);

// Every rule name, in evaluation order. Stable kebab-case: these are grepped.
const std::vector<const char*>& rule_names();

// Whether `w` carries the input `rule` reads at all. A rule without it cannot fire, and a run whose
// every window lacked it has "no data" for that rule, which the summary must not call quiet
// (present-cpu-overhead in a frontend without GPU present, for instance).
bool rule_has_data(const char* rule, const WindowSample& w);

// Consecutive windows `rule` must hold before it is reported (see kSustainWindows).
uint32_t sustain_windows(const char* rule);

std::vector<AlarmFiring> evaluate_rules(const WindowSample& w, const RuleThresholds& t);

}  // namespace prosper::diagnostics::perf

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
// unsupported-wave64-shaders: correctness, any known Wave64 shader use refused by translation
// or subgroup admission, regardless of whether the host supplies native Wave64 (#3992).
constexpr uint64_t kUnsupportedWave64UsesPerWindow = 1;
// Source-confirmed unverified lowering (#4059/#4062), not measured wrong pixels. Any compiler
// request that reached ordinary guest F32 ADD/MUL deserves a semantics/provenance warning.
constexpr uint64_t kUnverifiedFragmentArithmeticRequestsPerWindow = 1;

// gpu-memory-off-device: CORRECTNESS class (the frame is right, but on a discrete GPU every access
// to the resource crosses the bus, and no pixel test can see it). Any GPU-only renderer allocation
// placed in a non-device-local type while the device has device-local memory: either
// vkAllocateMemory ran out of device memory and the #3897 fallback retried a host type, or the
// resource's memoryTypeBits allowed no device-local type at all. On this project's APU neither
// happens, so a firing here is news.
constexpr uint64_t kGpuMemoryOffDevicePerWindow = 1;

// host-copy-pressure: bytes this process copies host-side on the guest's behalf, per second -- the
// sum over EVERY default-on [transfer-pressure] category (storage-materialize, buffer-upload,
// buffer-compare, rtt-snapshot, detile, guest-scanout), the same total that census's HIGH line uses. Its
// `breakdown` is in whole MiB per category, not a count like the correctness rules' breakdowns, so
// its summary "breakdown over fired windows" is MiB too. 256 MiB/s is that census's own HIGH line, which healthy titles sit
// far below (GTA V on main: ~80 MiB/s over a whole route) and every defect it was written against
// far above (a static splash copying 770 MiB/s; Astro Bot's compute round trip at ~2 GiB/s, #3871).
// Since host-copy-per-flip (below) it is not REPORTED in a window where that rule is reported
// (apply_reporting_deferrals); it is still evaluated, and its streak still counts, every window.
constexpr double kHostCopyMiBPerSecond = 256.0;

// host-copy-per-flip: the same bytes as host-copy-pressure, per GUEST FLIP instead of per second, so
// the verdict does not move with the frame rate. Measured 2026-09-29 on main (cf226ac2c) with this
// rule's JSONL, prosper-app GPU present, routed runs:
//   * GTA V perf-story route, three runs, BOTH regimes caught (the second run was the heavy one):
//     light regime 13.1-13.3 MiB/flip at 7-9 fps (storage-materialize ~1.0-1.2 MiB/call,
//     rtt-snapshot ~1.2-1.8 MiB/call); heavy regime 27.1 MiB/flip at 7 fps with the SAME call
//     counts and twice the bytes per call (2.23 / 3.52 MiB/call). The per-second form never fired
//     in any of the three (max 236 MiB/s in steady gameplay, 264 in one transition window).
//   * Sonic Frontiers gameplay: 63.4 MiB/flip, two 4K RGBA8 rtt-snapshots (31.64 MiB/call) per
//     flip, at 3-7 fps -- the per-second form fired in 19 of 61 windows, this rule in 31.
//   * Outer Wilds first-person route and The Messenger: under 0.25 MiB/flip in every window.
// 20 sits between the regimes (sqrt(13.3 x 27.1) = 19): the issue's proposed 16 fired once on the
// LIGHT regime's gameplay entry, where two consecutive windows read 55.8 then 17.1 MiB/flip before
// settling at 13.3 (run 3), and light-regime windows reach 15.0 during their first 20 s.
constexpr double kHostCopyMiBPerFlip = 20.0;
// ...over enough flips for the mean to be a per-frame cost rather than one load hitch: 10 per
// window, so the two-window sustain spans at least 20 flips. A window with fewer is left to
// host-copy-pressure's rate form. Outer Wilds' load windows (3 and 8 flips at 52-64 MiB/flip) fall
// below it; Sonic's 3-4 fps gameplay tail (14-18 flips per window at 63 MiB/flip) does not.
constexpr uint64_t kHostCopyPerFlipMinFlips = 10;

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

// unaccounted-draws: CORRECTNESS. A pass whose seen != recorded + dropped has a drop (or double
// count) that names no reason: the draw_disposition census's own blind spot. Sonic Frontiers printed
// `UNACCOUNTED=18` at exit on 2026-09-27 and no alarm said so.
constexpr uint64_t kUnaccountedDrawsPerWindow = 1;

// unimplemented-hle-calls: CORRECTNESS class ("news"). The FIRST call of an unregistered NID after
// the first guest flip. The dispatcher answers 0, which reads as SCE_OK, and writes no
// out-parameter -- #2951 (a glyph size read from untouched stack, then divided by) and #2023 (a
// four-session black screen) were both one unregistered NID. Boot-time first calls are before the
// engine's baseline and do not count; the `[prosper] unimplemented:` line names each one.
constexpr uint64_t kUnimplementedFirstCallsPerWindow = 1;

// diagnostic-path-active: STATE. The live renderer's production path (GPU-resident colour targets)
// is off because a diagnostic switch asked for it, so every frame is read back and re-presented and
// no measurement of this run describes what ships. Nothing on screen said so (#3909).

// present-path-fallback: SHARE of presents. With GPU present active, more than half of a window's
// presents showed a CPU-read-back frame instead of the GPU scanout: a full-frame readback plus a
// re-upload per frame, and a sign live targets are off or the publish keeps missing. Startup
// before the first GPU publish takes the fallback legitimately, hence the sustain and a floor.
constexpr double kPresentFallbackShare = 0.5;
constexpr double kPresentFallbackMinPerSecond = 5.0;

// pipeline-cache-thrash: RATE. Evictions from the renderer's graphics-pipeline, pipeline-layout
// and descriptor-set-layout caches. A cache that fits its working set evicts ~0 in a steady scene;
// one that does not rebuilds (a pipeline is a driver compile) what it just threw away.
// 5/s is 25 per window, well above a scene change's one-off churn.
constexpr double kPipelineCacheEvictionsPerSecond = 5.0;

// texture-validation-churn: RATE. Route-specific validation-prefix bytes charged to persistent
// decode-cache validations that FAILED. Direct comparisons count only fully matching chunks;
// the first differing chunk (at most 64 KiB) is omitted, so a first-chunk mismatch can report zero.
// Depth comparisons accumulate matching guest-face prefixes; scratch validation counts copied
// readable bytes. Failure can also mean an incomplete prefix, rather than changed source bytes.
// This is not total comparison traffic. Late differences can still report most of the source
// before a re-decode, which is the cost this rule names. The threshold remains 128 MiB/s.
// The 2026-09-28 GTA V intro calibration reported ~340 failures per 5 s window with at most
// 10.2 MiB charged (~2 MiB/s), not a complete guest-read measurement. The earlier source-size
// estimate was 283 MiB/s for the same windows. See #3891 for the missing-chunk observer proposal.
constexpr double kTextureValidationFailedMiBPerSecond = 128.0;

// present-slot-trouble: SHARE of guest flips. The renderer's GPU-present declines that mean the
// scanout path itself failed rather than that the frame was ineligible: `publish-failed`
// (present_blit_publish declined a publishable front image -- no free slot, a slot that never
// retired) and `compute-scanout-unwatched` (a compute-written front with no write watch, so it can
// never be proven current). Each such span's frame goes through the CPU fallback instead of GPU
// present. The ledger counts declines but not the spans that published, so the denominator is the
// guest flip (every flip has at least one final render span while GPU present is on); a flip whose
// several spans all decline counts more than once, which makes this the stricter reading of the
// proposed "10% of spans". Measured 2026-09-29: ZERO declines of either reason in every window of
// the six runs above (GTA V x3, Sonic Frontiers -- 4,259 compute-scanout publishes -- Outer Wilds,
// The Messenger), so any sustained share is news; 10% keeps an occasional slot miss quiet.
constexpr double kPresentSlotTroubleShare = 0.10;
// The decline names (GpuPresentOutcome, frontends/shared/present/present_blit_policy.hpp) this rule
// counts. Matched by NAME because the ledger stores each slot's name and this layer must not
// include the frontend header; test_present_blit_policy pins that these spellings still exist.
constexpr const char* kPresentSlotTroubleReasons[] = {"publish-failed", "compute-scanout-unwatched"};
constexpr uint64_t kPresentSlotTroubleMinFlips = 20;

// rtt-destination-refused: RATE per guest flip. Staging bytes of compute results whose renderer
// image destination borrow was refused (live_compute's destination_refusal_census() note site), so
// the result went back to the renderer through a CPU snapshot and its consumers re-materialized it
// from guest bytes. GTA V's pre-#3949 heavy host-copy regime (#3873) was this: one 2560x1440 RGBA8
// result refused with destination-creation-refused on every other flip (12.4 GiB / 869 calls in
// one run, ~7 MiB/flip), while the light regime refuses nothing; the reason was printed only at
// exit. The threshold sits under half the heavy regime and far above the light regime's zero.
constexpr double kRttDestinationRefusedMiBPerFlip = 4.0;
// ...over enough flips for a per-frame mean (as host-copy-per-flip).
constexpr uint64_t kRttDestinationRefusedMinFlips = 10;

// color-target-count-ceiling: STATE. The persistent colour-target cache peaked at its ENTRY-count
// bound in the window while its BYTE budget stayed under half used, with observed churn or refusal:
// it kept evicting (valid targets may be read back through a CPU sink and later re-created on
// use), or a compute destination creation was refused after applicable admission/eviction checks.
// Compute creation now tries eligible idle-target eviction (#3949). Creation refusal also covers
// unproven submissions and partial/pinned target state, so this combined alarm suggests count
// pressure without proving that the bound caused the refusal.
// The pre-#3949 GTA V runs (#3873) peaked at 320 of 256 entries at 27-29% of the 4 GiB budget.
// In that historical 2026-09-29 JSONL calibration, gameplay sits at EXACTLY 256 entries with ZERO
// evictions in every window while every flip refuses one 14 MiB
// compute result (destination-creation-refused). So eviction churn alone, as first proposed, would
// have missed the costly state; evictions are the load-time form (up to 392 per window).
constexpr double kColorTargetCeilingByteShare = 0.5;
constexpr double kColorTargetCeilingEvictionsPerSecond = 2.0;   // > 10 per 5 s window
// kDestinationCreationRefused (perf_ledger.hpp) names non-Vulkan creation/admission refusals.

// gpu-present-stalled: a GPU-present frontend presented NOTHING -- neither a GPU scanout nor a CPU
// fallback -- for a whole window while the guest kept flipping, with no recorded unavailable-window
// attempt (minimized, occluded or swapchain recreation). The hint names the last presented frame,
// or black if none was presented; this counter is not a pixel observation. #3951 needed it when a
// recompiler refusal dropped ~99.7% of GTA V's
// draws, the guest flipped at ~30/s, prosper-app printed no `[app] fps` line in 60+ s, and the
// only alarm that fired was a downstream host-copy-per-flip. The flip floor keeps a boot or a
// loading pause (the guest is not flipping) out of it; sustain 2 keeps one long hitch out.
constexpr uint64_t kGpuPresentStalledMinFlips = 20;   // per window (4/s over 5 s)

// gpu-sync-wait: SHARE of the frame budget (#3948 stage 0). The executor thread blocked on the fence
// of work it had just submitted -- each compute dispatch's, each graphics batch's -- summed per guest
// flip, while the GPU had headroom: the device time of prosper's OWN work measured inside those
// waits (a timestamp pair per dispatch/batch; other GPU users are invisible to it) is under half of
// wall time. It fires on every GTA V / Sonic gameplay window by design, a standing signal until
// #3948's later stages land (src/diagnostics/AGENTS.md). Overlapping CPU work with that GPU work is what
// asynchronous submission recovers, so the whole wait is the recoverable cost -- unless the GPU is
// saturated, which is the case this rule must stay quiet on. Measured 2026-09-29 (prosper-app, GPU
// present, routed gameplay, this rule's JSONL):
//   * GTA V perf-story gameplay: waits ~1,060 ms per 5 s window (compute ~510 ms over ~1,520
//     dispatches, graphics ~550 ms over ~2,490 batches) = 120-140% of the budget per flip; device
//     time inside ~860 ms, so the GPU runs prosper's work ~17% of wall time and ~19% of each wait
//     is idle/latency. Fired in 37 of 58 windows, every gameplay window, never before gameplay.
//   * Sonic Frontiers: menus 49-51% of the budget; gameplay 150-163%, GPU ~25% busy, 13-24% idle.
// 75% sits above Sonic's ~50% menus (which would flicker at a 50% bar) and well below both titles'
// gameplay. The idle share is reported, not thresholded: it is what BATCHING submits would save,
// while the device part is what OVERLAP saves.
constexpr double kGpuSyncWaitBudgetShare = 0.75;
constexpr double kGpuSyncWaitMaxGpuBusy = 0.5;
// ...over enough flips for a per-frame mean, and with device time sampled on at least half the waits
// (without samples the rule cannot tell a saturated GPU from an idle one, so it has no data).
constexpr uint64_t kGpuSyncWaitMinFlips = 10;
constexpr double kGpuSyncWaitMinDeviceCoverage = 0.5;

// gpu-device-time-coverage: instrument validity (#3891). Missing batch timestamp envelopes
// hid roughly a third of graphics device time in #3964's first ON arms. Corrected GTA windows
// cover about 88-89% of graphics waits; 80% separates those populations. Check each stream
// independently: complete compute samples must not mask missing graphics samples. A 100-wait
// floor and two-window sustain tolerate a few timestamps crossing a window boundary. This is
// a sampling bound, not a performance sensitivity, so the threshold scale does not change it.
constexpr double kGpuDeviceTimeMinCoverage = 0.8;
constexpr uint64_t kGpuDeviceTimeCoverageMinWaits = 100;

// #3907: publishing a pass's CPU colour bytes without a slot-0 colour writer violates the
// publication contract. One observed violation reports immediately, independent of sensitivity.
constexpr uint64_t kRttColorlessPublicationsPerWindow = 1;
constexpr uint32_t kRttColorlessPublicationSustainWindows = 1;

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
    // Per-reason breakdown of PresentGpuDeclines (#3915), and each slot's name.
    uint64_t present_declines[kPresentDeclineSlots] = {};
    const char* present_decline_names[kPresentDeclineSlots] = {};
    // Host-copy bytes per [transfer-pressure] category (diagnostics::Transfer).
    static constexpr size_t kTransferCount = static_cast<size_t>(Transfer::Count);
    uint64_t transfer_bytes[kTransferCount] = {};
    // ...and the note_transfer calls that carried them, for bytes per call.
    uint64_t transfer_calls[kTransferCount] = {};
    // Per-window high-water marks (Ledger::peaks).
    uint64_t peaks[kPeakCount] = {};
    // Per-reason breakdown of RttDestinationRefusedBytes, and each slot's name.
    uint64_t rtt_destination_refused_bytes[kRttDestinationRefusalSlots] = {};
    const char* rtt_destination_refusal_names[kRttDestinationRefusalSlots] = {};
    uint64_t rtt_destination_refusals[kRttDestinationRefusalSlots] = {};   // results per slot
    // Verdicts of the exact full-overwrite shape test (ExactResultDecline), one per tested result.
    uint64_t exact_result_declines[kExactResultDeclineCount] = {};

    double budget_ms() const { return target_hz ? 1000.0 / target_hz : 1000.0 / 60.0; }
    double ms(Cost c) const { return cost_ns[static_cast<size_t>(c)] / 1e6; }
    uint64_t events(Cost c) const { return cost_events[static_cast<size_t>(c)]; }
    uint64_t count(Counter c) const { return counters[static_cast<size_t>(c)]; }
    uint64_t gauge(Gauge g) const { return gauges[static_cast<size_t>(g)]; }
    uint64_t peak(Peak p) const { return peaks[static_cast<size_t>(p)]; }
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
    double host_copy_mib_per_flip = kHostCopyMiBPerFlip;
    uint64_t host_copy_per_flip_min_flips = kHostCopyPerFlipMinFlips;
    double present_slot_trouble_share = kPresentSlotTroubleShare;
    uint64_t present_slot_trouble_min_flips = kPresentSlotTroubleMinFlips;
    uint64_t gpu_present_stalled_min_flips = kGpuPresentStalledMinFlips;
    double shader_compile_budget_share = kShaderCompileBudgetShare;
    double shader_compile_min_per_s = kShaderCompileMinPerSecond;
    uint64_t unaccounted_draws = kUnaccountedDrawsPerWindow;
    uint64_t unimplemented_first_calls = kUnimplementedFirstCallsPerWindow;
    double present_fallback_share = kPresentFallbackShare;
    double present_fallback_min_per_s = kPresentFallbackMinPerSecond;
    double pipeline_cache_evictions_per_s = kPipelineCacheEvictionsPerSecond;
    double texture_validation_failed_mib_per_s = kTextureValidationFailedMiBPerSecond;
    double rtt_destination_refused_mib_per_flip = kRttDestinationRefusedMiBPerFlip;
    uint64_t rtt_destination_refused_min_flips = kRttDestinationRefusedMinFlips;
    double color_target_ceiling_byte_share = kColorTargetCeilingByteShare;
    double color_target_ceiling_evictions_per_s = kColorTargetCeilingEvictionsPerSecond;
    double gpu_sync_wait_budget_share = kGpuSyncWaitBudgetShare;
    double gpu_sync_wait_max_gpu_busy = kGpuSyncWaitMaxGpuBusy;

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

// host-copy-per-flip's condition.
bool host_copy_per_flip_holds(const WindowSample& w, const RuleThresholds& t);

// REPORTING deferrals, applied by the engine to the rules that passed their sustain in one window
// (never to evaluate_rules' candidates, so every rule's streak keeps counting): host-copy-pressure
// is dropped when host-copy-per-flip is reported in the same window, because it would restate the
// same bytes with the denominator that moves with the frame rate. In a window where the per-flip
// rule is not reported -- too few flips, a small per-frame copy at a high frame rate, or a per-flip
// value that dips under its threshold and breaks that rule's streak -- host-copy-pressure reports
// as it always did.
void apply_reporting_deferrals(std::vector<AlarmFiring>& reported);
// The window's `publish-failed` + `compute-scanout-unwatched` GPU-present declines.
uint64_t present_slot_trouble_declines(const WindowSample& w);
// The window's rtt-destination-refused breakdown in whole MiB per refusal reason, largest first.
std::vector<std::pair<const char*, uint64_t>> rtt_destination_refused_mib(const WindowSample& w);
// gpu-sync-wait's inputs: whether the window's device-time samples cover enough of its waits.
bool gpu_sync_wait_has_data(const WindowSample& w);
// The window's refused bytes whose reason is kDestinationCreationRefused.
uint64_t destination_creation_refused_bytes(const WindowSample& w);
// ...and how many results that was.
uint64_t destination_creation_refused_count(const WindowSample& w);

std::vector<AlarmFiring> evaluate_rules(const WindowSample& w, const RuleThresholds& t);

}  // namespace prosper::diagnostics::perf

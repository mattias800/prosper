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
    // #3948 stage 0: the executor thread blocked on a GPU fence it just submitted -- one compute
    // dispatch's fence, or one graphics submission batch's fence. The GPU device time INSIDE those
    // waits is counted separately (Counter::GpuDevice*Ns), so wait minus device is what the
    // synchronous design costs beyond the GPU's own work: submission latency plus idle GPU.
    GpuWaitCompute,
    GpuWaitGraphics,
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
    // GPU-only renderer allocations (gpu/diagnostics/memory_placement_log.hpp, #3897):
    GpuMemoryFallbacks,        // succeeded only after the preferred type ran out of device memory
    GpuMemoryOffDevice,        // landed on a non-device-local type although the device has one
    GpuMemoryOffDeviceBytes,   // bytes of those placements
    // #3891 queue (2026-09-28). Each is bumped on an EVENT path, never per accepted draw/reference:
    DrawsUnaccounted,          // |seen - recorded - dropped| of a pass (draw_disposition blind spot)
    HleUnimplementedCalls,     // calls into an unregistered NID (the dispatcher answers 0)
    HleUnimplementedFirst,     // ...of which the FIRST call of a distinct import
    PresentCpuFallbacks,       // GPU-present iterations that showed a CPU-read-back frame instead
    PresentGpuDeclines,        // final render spans that declined to publish the front to GPU
                               // present, by reason in Ledger::present_declines (#3915)
    // Host present attempts that could not reach the window at all: a minimized / zero-extent
    // window waiting for its swapchain, an occluded acquire that timed out, or an out-of-date
    // swapchain being recreated. gpu-present-stalled stays silent in a window that has any.
    PresentWindowUnavailable,
    PipelineEvictions,         // graphics pipelines evicted from the full pipeline cache
    PipelineLayoutEvictions,   // pipeline layouts evicted from their full cache
    DescriptorSetLayoutEvictions,  // descriptor-set layouts evicted from their full cache
    TextureValidationFailures,     // persistent decode-cache entries whose exact validation failed
    TextureValidationFailedBytes,  // failed validations' route-specific prefix count: direct/depth
                                   // comparisons count matching chunks, excluding the differing
                                   // chunk; scratch validation counts copied readable bytes.
                                   // Neither total compare traffic nor source mutation is proved.
    // #3891 queue (2026-09-29), all on event paths:
    RttDestinationRefusals,        // compute results whose renderer-image destination borrow was
                                   // refused, so the result went back through a CPU snapshot
    RttDestinationRefusedBytes,    // staging bytes of those results
    PersistentTargetEvictions,     // persistent colour targets evicted (census twin, #3872)
    PersistentTargetEvictedBytes,
    ExactResultCandidates,         // compute storage-writeback results tested for the exact
                                   // full-overwrite shape (live_compute's exact_full_result)
    // #3948 stage 0: GPU device time (timestamp pair) of the work inside the waits above, and how
    // many waits carried a readable pair. A wait without one counts in Cost::GpuWait* only.
    GpuDeviceNsCompute,
    GpuDeviceSamplesCompute,
    GpuDeviceNsGraphics,
    GpuDeviceSamplesGraphics,
    // #3948 stage 2 (PROSPER_GRAPHICS_DEFERRED_WAIT=1): graphics batches submitted without a wait
    // before a dispatch, and how many of their later retirements still found the fence unsignalled
    // (and so blocked, in Cost::GpuWaitGraphics).
    GpuGraphicsDeferred,
    GpuGraphicsDeferredBlocked,
    CpuRttPublicationChecks,     // evaluated slot-0 CPU pass-readback publication candidates
    CpuRttColorlessPublications, // actual publications with no slot-0 colour writer (#3907)
    // Proven guest Wave64 refusals, counted per shader USE, not per distinct shader (#3992).
    Wave64FragmentRecompile,
    Wave64ComputeRecompile,
    Wave64FragmentSubgroup,
    Wave64ComputeSubgroup,
    Wave64NewRefusalIdentities, // newly observed (site, compile identity), bounded inventory
    Wave64UnidentifiedRefusals, // uses with neither a compile identity nor a program address
    Wave64InventoryOverflow,   // uses whose identity could not enter the full inventory
    Wave64ShaderChecks,        // known-Wave64 observations at realization/backend boundaries
    // Actual direct/cached fragment compiler requests, NOT draws, execution or wrong pixels.
    // Families describe emitted guest ADD/MUL sites only; other arithmetic is not inventoried.
    FragmentArithmeticRequests,
    FragmentArithmeticKnownMode,
    FragmentArithmeticUnknownMode,
    FragmentArithmeticAddRequests,
    FragmentArithmeticMulRequests,
    FragmentArithmeticRefusedRequests, // qualifying requests whose final module was empty
    FragmentArithmeticTruncatedRequests,
    FragmentArithmeticInventoryOverflowRequests,
    // Fresh direct-fold WAIT_REG_MEM predicate evaluations only. Ordered-effect acceptance,
    // queued-without-evaluation waits and deferred rechecks do not enter this denominator.
    WaitRegMemDirectEvaluations,
    WaitRegMemDirectCompareFalse,
    WaitRegMemDirectUnreadable,
    WaitRegMemDirectUnsupported,
    WaitRegMemDirectFalseProceed,
    WaitRegMemDirectFalseDefer,
    // Two ordinary direct decode-cache validation routes only; not depth/scratch/watch proofs.
    // Comparison extent includes the differing chunk, but does not measure physical read traffic.
    TextureDirectValidationAttempts,
    TextureDirectValidationMemcmpCalls,
    TextureDirectValidationMemcmpExtentBytes,
    TextureDirectValidationAcceptedPrefix,
    TextureDirectValidationBytesDiffer,
    TextureDirectValidationExpectedMissing,
    TextureDirectValidationIncompletePrefix,
    Count
};

// Last-written values (not summed): the state a State rule reads.
enum class Gauge : uint8_t {
    TextureCacheBytes = 0,
    TextureCacheLimit,
    // Bitmask of DiagnosticPathSwitch: the switches that turned the live renderer's production path
    // (GPU-resident colour targets) off for this run. 0 = production path.
    DiagnosticPathSwitches,
    // The persistent colour-target cache's two admission bounds (persistent_target_census.hpp).
    PersistentTargetEntryLimit,
    PersistentTargetByteLimit,
    // 1 while a GPU-present consumer (prosper-app's swapchain) is attached: set_gpu_present_active.
    // Frontends without one (tools/screenshot, boot_trace) present nothing by design, so a window
    // with no presents there is not a stall (gpu-present-stalled).
    GpuPresentActive,
    Count
};

// Per-window HIGH-WATER marks: raised with raise(), taken and reset by the engine at each window
// close. A last-written gauge cannot say whether a cache touched its bound during a window.
enum class Peak : uint8_t {
    PersistentTargetEntries = 0,
    PersistentTargetBytes,
    Count
};

// Why live GPU colour targets are off (Gauge::DiagnosticPathSwitches bits).
enum class DiagnosticPathSwitch : uint8_t {
    NoLiveTargets = 0,     // PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS
    CaptureReadback,       // PROSPER_GPU_CAPTURE_READBACK
    ReplayExport,          // PROSPER_GPU_REPLAY_EXPORT_RTT / PROSPER_GPU_REPLAY_RTT_SEEDS
    PerPassPixelDump,      // PROSPER_DUMP_* / *_HASH_DIM / PROSPER_RTTLOG per-pass pixel diagnostics
    Other,                 // off for a reason none of the above names
    Count
};
constexpr size_t kDiagnosticPathSwitchCount = static_cast<size_t>(DiagnosticPathSwitch::Count);
constexpr const char* kDiagnosticPathSwitchNames[kDiagnosticPathSwitchCount] = {
    "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS", "PROSPER_GPU_CAPTURE_READBACK",
    "PROSPER_GPU_REPLAY_EXPORT_RTT/RTT_SEEDS", "per-pass-pixel-dump(PROSPER_DUMP_*/HASH_DIM/RTTLOG)",
    "other",
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
    // realize_draw_item (src/gpu/execute/gpu_execute.hpp): a stage produced no SPIR-V, so the draw
    // never reaches the renderer's pass loop at all. Without these a recompiler refusal was the one
    // way to lose nearly every draw with the alarm silent (#3951: GTA V 1.27M -> 4.3k draws/run).
    ShaderRecompileVertex,     // the vertex (or linked ES+VS chain) stage failed to recompile
    ShaderRecompileFragment,   // the pixel stage failed to recompile
    ShaderRecompileGeometry,   // the synthesized interpolation/rect-list geometry stage failed
    BackendGeometryCapability,
    BackendMeshShape,
    BackendSubgroupFeatures,
    BackendGdsAllocation,
    BackendBufferResources,
    BackendShaderRejected,
    BackendPipelineCreation,
    BackendTargetMemory,
    BackendNggSubgroup,
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
    "shader-recompile/vertex",
    "shader-recompile/fragment",
    "shader-recompile/geometry",
    "backend/geometry-capability",
    "backend/mesh-shape",
    "backend/subgroup-features",
    "backend/gds-allocation",
    "backend/buffer-resources",
    "backend/shader-rejected",
    "backend/pipeline-creation",
    "backend/target-memory",
    "backend/ngg-subgroup",
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

// Per-class breakdown of Counter::GpuMemoryOffDevice. The classes are the renderer's
// prosper::gpu::GpuOnlyMemoryClass, which this header must not include (it has no Vulkan types);
// the recorder passes each class's name as a string literal, stored by pointer like cost_label, and
// memory_placement_log.hpp static_asserts that every class fits a slot.
constexpr size_t kGpuMemoryClassSlots = 16;

// Per-reason breakdown of the renderer's GPU-present declines (#3915), counted per final render
// SPAN that did NOT hand the front buffer to GPU present. A flip can end several spans, so these
// counts are not fallback presents (Counter::PresentCpuFallbacks counts those). The reasons are the frontend's prosper::frontend::GpuPresentOutcome, which this header
// must not include; the recorder passes each reason's name as a string literal, stored by pointer.
constexpr size_t kPresentDeclineSlots = 24;

// Per-reason breakdown of Counter::RttDestinationRefusedBytes: the renderer-image destination
// borrow's refusal (prosper::gpu::LiveTargetImageImport::Refusal, which this header must not
// include). The recorder passes the reason's name as a string literal, stored by pointer, and
// static_asserts that the enum fits.
constexpr size_t kRttDestinationRefusalSlots = 24;
// The non-Vulkan creation refusal name (prosper::gpu::live_target_import_refusal_name), covering
// admission checks as well as cache bounds. It does not by itself prove no room.
// live_compute.cpp static_asserts the two spellings agree, so a
// rename on either side fails to compile instead of silently muting color-target-count-ceiling.
constexpr const char* kDestinationCreationRefused = "destination-creation-refused";

// WHY a compute storage-writeback result is not an exact, whole, single-level 2D overwrite -- the
// FIRST failing field of live_compute's exact_full_result, in the order it tests them -- or, for a
// result of that shape, why it still did not reach the renderer-image destination borrow. One
// relaxed add per tested result per dispatch. Sonic Frontiers' one-layer 2D_ARRAY decline (#3929,
// img-dim) and GTA V's R8 snapshots (format-no-seed-path, #3873) were invisible to every census
// until somebody traced a single address.
enum class ExactResultDecline : uint8_t {
    Accepted = 0,          // reached the destination borrow (its refusals: RttDestinationRefused*)
    AliasedBinding,
    InexactStorageBytes,
    PartialWriteMask,
    MirrorToImported,
    PriorOutputConflict,
    FinalOutputConflict,
    ImageDim,
    Depth,
    ArrayLayers,
    TexelDepth,
    MipLevels,
    Samples,
    MipTail,
    MipBaseLevel,
    LayerMipOffset,
    LinearPitch,
    LayerStride,
    ZeroExtent,
    NoStaging,
    // Exact shape, but no renderer-image path for the result:
    FormatUnmapped,        // storage format has no LiveTargetPixelFormat
    FormatNoSeedPath,      // mapped, but not one of the formats with a storage seed path
    FormatNotNative,       // RGBA16F not stored natively, or R11G11B10 not packed
    StagingSizeMismatch,
    Count
};
constexpr size_t kExactResultDeclineCount = static_cast<size_t>(ExactResultDecline::Count);
// Stable, grepped: never reword casually.
constexpr const char* kExactResultDeclineNames[kExactResultDeclineCount] = {
    "accepted", "aliased-binding", "inexact-storage-bytes", "partial-write-mask",
    "mirror-to-imported", "prior-output-conflict", "final-output-conflict", "img-dim", "depth",
    "array-layers", "texel-depth", "mip-levels", "samples", "mip-tail", "mip-base-level",
    "layer-mip-offset", "linear-pitch", "layer-stride", "zero-extent", "no-staging",
    "format-unmapped", "format-no-seed-path", "format-not-native", "staging-size-mismatch",
};

constexpr size_t kCostCount = static_cast<size_t>(Cost::Count);
// Stable, grepped names for the time costs, in Cost order. The frame-breakdown summary prints them;
// never reword casually.
constexpr const char* kCostNames[kCostCount] = {
    "surface-readback", "hle-blocking-wait", "present-cpu", "frontend-build", "texture-ref-sample",
    "shader-compile", "pipeline-create", "gpu-wait-compute", "gpu-wait-graphics",
};
// A Cost added without a name would leave a null slot that the exit summary reads.
constexpr bool all_cost_names_present() {
    for (const char* name : kCostNames)
        if (name == nullptr) return false;
    return true;
}
static_assert(all_cost_names_present(), "every Cost needs an entry in kCostNames, in Cost order");
constexpr size_t kCounterCount = static_cast<size_t>(Counter::Count);
constexpr size_t kGaugeCount = static_cast<size_t>(Gauge::Count);
constexpr size_t kPeakCount = static_cast<size_t>(Peak::Count);

struct Ledger {
    std::atomic<uint64_t> cost_ns[kCostCount] = {};
    std::atomic<uint64_t> cost_events[kCostCount] = {};
    // Largest single event since the engine last took it (exchange(0) at each window close).
    std::atomic<uint64_t> cost_max_ns[kCostCount] = {};
    std::atomic<uint64_t> counters[kCounterCount] = {};
    std::atomic<uint64_t> gauges[kGaugeCount] = {};
    std::atomic<uint64_t> peaks[kPeakCount] = {};
    std::atomic<uint64_t> drop_reasons[kDropReasonCount] = {};
    std::atomic<uint64_t> dispatch_skips[kDispatchSkipCount] = {};
    std::atomic<uint64_t> gpu_memory_off_device[kGpuMemoryClassSlots] = {};
    std::atomic<const char*> gpu_memory_class_names[kGpuMemoryClassSlots] = {};
    std::atomic<uint64_t> present_declines[kPresentDeclineSlots] = {};
    std::atomic<const char*> present_decline_names[kPresentDeclineSlots] = {};
    std::atomic<uint64_t> rtt_destination_refused_bytes[kRttDestinationRefusalSlots] = {};
    std::atomic<const char*> rtt_destination_refusal_names[kRttDestinationRefusalSlots] = {};
    std::atomic<uint64_t> rtt_destination_refusals[kRttDestinationRefusalSlots] = {};
    std::atomic<uint64_t> exact_result_declines[kExactResultDeclineCount] = {};
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

// Called at the actual direct site, before its warning rate limit. An unreadable sample takes
// precedence over an unsupported comparison, matching the predicate's existing early return.
inline void note_wait_regmem_direct_evaluation(bool readable, bool supported, bool satisfied) {
    if (!enabled()) return;
    add(Counter::WaitRegMemDirectEvaluations);
    if (satisfied) return;
    add(!readable ? Counter::WaitRegMemDirectUnreadable :
        !supported ? Counter::WaitRegMemDirectUnsupported : Counter::WaitRegMemDirectCompareFalse);
}

// Called after the existing policy branch has selected an action, without rereading its gate.
inline void note_wait_regmem_direct_false_action(bool deferred) {
    if (!enabled()) return;
    add(deferred ? Counter::WaitRegMemDirectFalseDefer : Counter::WaitRegMemDirectFalseProceed);
}

inline constexpr Counter kWaitRegMemDirectCounters[] = {
    Counter::WaitRegMemDirectEvaluations, Counter::WaitRegMemDirectCompareFalse,
    Counter::WaitRegMemDirectUnreadable, Counter::WaitRegMemDirectUnsupported,
    Counter::WaitRegMemDirectFalseProceed, Counter::WaitRegMemDirectFalseDefer
};
inline constexpr size_t kWaitRegMemDirectCounterCount =
    sizeof(kWaitRegMemDirectCounters) / sizeof(kWaitRegMemDirectCounters[0]);

// Independent relaxed loads can straddle an observation. Positive reason/action counts without
// an evaluation in this snapshot are partial data, never a clean zero or an accounting verdict.
inline const char* wait_regmem_direct_data_status(const uint64_t* counts) {
    if (counts[0]) return "OBSERVED";
    for (size_t i = 1; i < kWaitRegMemDirectCounterCount; ++i)
        if (counts[i]) return "PARTIAL";
    return "NO DATA";
}

// Caller passes the original final helper-result/required-prefix decision. The actual helper
// refusal takes precedence over its caller's later prefix-length check. No sample is reread.
inline void note_texture_direct_validation(bool accepted_prefix, bool bytes_differ,
        bool expected_missing, uint64_t memcmp_calls, uint64_t memcmp_extent_bytes) {
    if (!enabled()) return;
    add(Counter::TextureDirectValidationAttempts);
    add(Counter::TextureDirectValidationMemcmpCalls, memcmp_calls);
    add(Counter::TextureDirectValidationMemcmpExtentBytes, memcmp_extent_bytes);
    add(accepted_prefix ? Counter::TextureDirectValidationAcceptedPrefix :
        expected_missing ? Counter::TextureDirectValidationExpectedMissing :
        bytes_differ ? Counter::TextureDirectValidationBytesDiffer :
        Counter::TextureDirectValidationIncompletePrefix);
}

inline constexpr Counter kTextureDirectValidationCounters[] = {
    Counter::TextureDirectValidationAttempts, Counter::TextureDirectValidationMemcmpCalls,
    Counter::TextureDirectValidationMemcmpExtentBytes, Counter::TextureDirectValidationAcceptedPrefix,
    Counter::TextureDirectValidationBytesDiffer, Counter::TextureDirectValidationExpectedMissing,
    Counter::TextureDirectValidationIncompletePrefix
};
inline constexpr size_t kTextureDirectValidationCounterCount =
    sizeof(kTextureDirectValidationCounters) / sizeof(kTextureDirectValidationCounters[0]);

// Snapshot inputs are independent relaxed loads, not a coherent outcome partition.
inline const char* texture_direct_validation_data_status(const uint64_t* counts) {
    if (counts[0]) return "OBSERVED";
    for (size_t i = 1; i < kTextureDirectValidationCounterCount; ++i)
        if (counts[i]) return "PARTIAL";
    return "NO DATA";
}

inline void set(Gauge g, uint64_t v) {
    ledger().gauges[static_cast<size_t>(g)].store(v, std::memory_order_relaxed);
}

// Raise a per-window high-water mark. A load and a compare when `v` is not a new peak, so it is
// cheap enough for a site sampled on every cache lookup.
inline void raise(Peak p, uint64_t v) {
    std::atomic<uint64_t>& peak = ledger().peaks[static_cast<size_t>(p)];
    uint64_t seen = peak.load(std::memory_order_relaxed);
    while (v > seen && !peak.compare_exchange_weak(seen, v, std::memory_order_relaxed)) {}
}

// A gauge that changes rarely (a configured bound), written only when it changed so a hot caller
// does not dirty the cache line on every call.
inline void set_if_changed(Gauge g, uint64_t v) {
    std::atomic<uint64_t>& gauge = ledger().gauges[static_cast<size_t>(g)];
    if (gauge.load(std::memory_order_relaxed) != v) gauge.store(v, std::memory_order_relaxed);
}

// One compute result whose renderer-image destination borrow was refused, with the refusal's name
// (a string literal) and the result's staging bytes. Bumps both totals, so they and the breakdown
// can never disagree.
inline void note_rtt_destination_refusal(size_t slot, const char* name, uint64_t bytes) {
    if (slot >= kRttDestinationRefusalSlots) return;
    Ledger& l = ledger();
    l.rtt_destination_refusal_names[slot].store(name, std::memory_order_relaxed);
    l.rtt_destination_refused_bytes[slot].fetch_add(bytes, std::memory_order_relaxed);
    l.rtt_destination_refusals[slot].fetch_add(1, std::memory_order_relaxed);
    add(Counter::RttDestinationRefusals);
    add(Counter::RttDestinationRefusedBytes, bytes);
}

// One compute storage-writeback result tested for the exact full-overwrite shape, with its verdict.
inline void note_exact_result(ExactResultDecline verdict) {
    const size_t i = static_cast<size_t>(verdict);
    if (i >= kExactResultDeclineCount) return;
    add(Counter::ExactResultCandidates);
    ledger().exact_result_declines[i].fetch_add(1, std::memory_order_relaxed);
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

// One GPU-only renderer allocation that landed off device-local memory while the device has some
// (#3897): an out-of-memory fallback, or a resource whose memoryTypeBits allowed no device-local
// type. `name` must be a string literal (stored by pointer).
inline void note_gpu_memory_off_device(size_t slot, const char* name, uint64_t bytes) {
    if (slot >= kGpuMemoryClassSlots) return;
    Ledger& l = ledger();
    l.gpu_memory_class_names[slot].store(name, std::memory_order_relaxed);
    l.gpu_memory_off_device[slot].fetch_add(1, std::memory_order_relaxed);
    add(Counter::GpuMemoryOffDevice);
    add(Counter::GpuMemoryOffDeviceBytes, bytes);
}

// One GPU-present decline with its reason (#3915). `name` must be a string literal. Bumps
// Counter::PresentGpuDeclines too, so the total and the breakdown can never disagree.
inline void note_present_decline(size_t slot, const char* name) {
    if (slot >= kPresentDeclineSlots) return;
    Ledger& l = ledger();
    l.present_decline_names[slot].store(name, std::memory_order_relaxed);
    l.present_declines[slot].fetch_add(1, std::memory_order_relaxed);
    add(Counter::PresentGpuDeclines);
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

// The draw half of the same contract (#3951): realize_draw_item counts a draw it cannot realize
// (a stage that failed to recompile, a strict descriptor-contract failure), and an F9 capture or a
// menu capture re-realizes the submit's draws for its bundle. Those re-realizations are not live
// drops and must not raise `dropped-draws` a second time.
inline uint32_t& thread_draw_drop_suppression() {
    static thread_local uint32_t depth = 0;
    return depth;
}
class SuppressDrawDropCounting {
public:
    SuppressDrawDropCounting() { ++thread_draw_drop_suppression(); }
    ~SuppressDrawDropCounting() { --thread_draw_drop_suppression(); }
    SuppressDrawDropCounting(const SuppressDrawDropCounting&) = delete;
    SuppressDrawDropCounting& operator=(const SuppressDrawDropCounting&) = delete;
};
// A draw realize_draw_item could not realize. Counted unless a re-realization suppressed it.
inline void drop_draw_at_realization(DropReason reason) {
    if (thread_draw_drop_suppression()) return;
    drop_draw(reason);
}

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

#include "diagnostics/perf/perf_alarm_rules.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>

namespace prosper::diagnostics::perf {

namespace {

// printf-checked: a mismatched specifier in a rule's detail string is a compile-time warning, not
// a garbled alarm line (or worse, a %s reading an integer) discovered in somebody's run log.
[[gnu::format(printf, 1, 2)]] std::string format(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, args);
    va_end(args);
    return buf;
}

double per_second(uint64_t n, double seconds) {
    return seconds > 0 ? static_cast<double>(n) / seconds : 0.0;
}

constexpr double kMiB = 1024.0 * 1024.0;

}  // namespace

std::vector<std::pair<const char*, uint64_t>> ranked(const uint64_t* counts,
                                                     const char* const* names, size_t n) {
    std::vector<std::pair<const char*, uint64_t>> out;
    for (size_t i = 0; i < n; ++i)
        if (counts[i]) out.emplace_back(names[i], counts[i]);
    std::stable_sort(out.begin(), out.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

std::string top_entries(const std::vector<std::pair<const char*, uint64_t>>& breakdown,
                        size_t top) {
    std::string out;
    for (size_t i = 0; i < breakdown.size() && i < top; ++i) {
        if (!out.empty()) out += ',';
        out += format("%s:%llu", breakdown[i].first, (unsigned long long)breakdown[i].second);
    }
    if (breakdown.size() > top) out += format("(+%llu more)",
                                                      (unsigned long long)(breakdown.size() - top));
    return out.empty() ? "none" : out;
}

RuleThresholds RuleThresholds::scaled(double percent) {
    RuleThresholds t;
    const double k = percent > 0 ? percent / 100.0 : 1.0;
    t.texture_refusals_per_s *= k;
    t.readback_budget_share *= k;
    t.readback_min_per_s *= k;
    t.texture_reference_ns *= k;
    t.frontend_build_budget_share *= k;
    t.texture_reference_min_per_s *= k;
    t.hle_blocked_thread_share *= k;
    t.present_cpu_ms *= k;
    t.present_min_per_s *= k;
    t.host_copy_mib_per_s *= k;
    t.shader_compile_budget_share *= k;
    t.shader_compile_min_per_s *= k;
    t.present_fallback_share *= k;
    t.present_fallback_min_per_s *= k;
    t.pipeline_cache_evictions_per_s *= k;
    t.texture_validation_failed_mib_per_s *= k;
    // dropped_draws, skipped_dispatches, gpu_memory_off_device, unaccounted_draws and
    // unimplemented_first_calls stay at "any": a correctness alarm has no sensitivity to lower.
    // diagnostic-path-active is a state and has no threshold.
    return t;
}

const std::vector<const char*>& rule_names() {
    static const std::vector<const char*> names = {
        "texture-cache-thrash", "surface-readback", "texture-reference-cost",
        "hle-blocking-wait",    "present-cpu-overhead", "dropped-draws",
        "skipped-dispatches",   "host-copy-pressure",   "shader-compile",
        "gpu-memory-off-device", "unaccounted-draws",   "unimplemented-hle-calls",
        "diagnostic-path-active", "present-path-fallback", "pipeline-cache-thrash",
        "texture-validation-churn",
    };
    return names;
}

bool rule_has_data(const char* rule, const WindowSample& w) {
    const std::string r = rule ? rule : "";
    if (r == "present-cpu-overhead") return w.events(Cost::PresentCpu) != 0;
    if (r == "texture-reference-cost") return w.events(Cost::TextureRefSample) != 0;
    // Presents of either kind: a frontend without GPU present has no data for this rule.
    if (r == "present-path-fallback")
        return w.events(Cost::PresentCpu) != 0 || w.count(Counter::PresentCpuFallbacks) != 0;
    // The others measure events whose ABSENCE is the healthy answer: no readback, no contended
    // lock, no refusal, no dropped draw. Zero there is data.
    return true;
}

uint32_t sustain_windows(const char* rule) {
    const std::string r = rule ? rule : "";
    if (r == "dropped-draws" || r == "skipped-dispatches" || r == "gpu-memory-off-device" ||
        r == "unaccounted-draws" || r == "unimplemented-hle-calls" ||
        r == "diagnostic-path-active")
        return kCorrectnessSustainWindows;
    if (r == "shader-compile") return kShaderCompileSustainWindows;
    if (r == "texture-reference-cost") return kTextureReferenceSustainWindows;
    return kSustainWindows;
}

std::vector<AlarmFiring> evaluate_rules(const WindowSample& w, const RuleThresholds& t) {
    std::vector<AlarmFiring> out;
    if (w.seconds <= 0) return out;
    const double budget = w.budget_ms();

    // texture-cache-thrash (RATE + STATE). The refusal count is the signal; evictions and fill are
    // reported beside it because "full, and evicting nothing" is what names the mechanism.
    {
        const uint64_t refusals = w.count(Counter::TextureCacheRefusals);
        const double rate = per_second(refusals, w.seconds);
        if (refusals && rate >= t.texture_refusals_per_s) {
            AlarmFiring a;
            a.rule = "texture-cache-thrash";
            a.value = rate;
            a.unit = "refusals/s";
            a.threshold = t.texture_refusals_per_s;
            a.detail = format(
                "refusals=%llu misses=%llu evictions=%llu reuploaded=%.0fMiB cache=%.0f/%.0fMiB "
                "vk-allocs=%llu",
                (unsigned long long)refusals,
                (unsigned long long)w.count(Counter::TextureCacheMisses),
                (unsigned long long)w.count(Counter::TextureCacheEvictions),
                w.count(Counter::TextureCacheRefusedBytes) / kMiB,
                w.gauge(Gauge::TextureCacheBytes) / kMiB, w.gauge(Gauge::TextureCacheLimit) / kMiB,
                (unsigned long long)w.count(Counter::DeviceAllocations));
            a.hint = "the resident texture cache is full and cannot evict, so every refused texture "
                     "is uploaded into a fresh allocation and freed each frame; next: "
                     "PROSPER_BACKEND_TEXTURE_PATH_CENSUS=1 (refused/retained per pass), F8 "
                     "res_texture_upload_ms; cf. #3876";
            out.push_back(std::move(a));
        }
    }

    // surface-readback (SHARE, with a rate floor).
    {
        const uint64_t n = w.events(Cost::SurfaceReadback);
        const double share = w.budget_share(Cost::SurfaceReadback);
        if (n && share >= t.readback_budget_share &&
            per_second(n, w.seconds) >= t.readback_min_per_s) {
            AlarmFiring a;
            a.rule = "surface-readback";
            a.value = share * 100.0;
            a.unit = "%budget";
            a.threshold = t.readback_budget_share * 100.0;
            a.cost_ms = w.ms(Cost::SurfaceReadback);
            a.detail = format("readbacks=%llu avg=%.2fms max=%.2fms flips=%llu budget=%.1fms(%uHz)",
                              (unsigned long long)n, w.ms(Cost::SurfaceReadback) / n,
                              w.cost_max_ns[static_cast<size_t>(Cost::SurfaceReadback)] / 1e6,
                              (unsigned long long)w.flips, budget, w.target_hz);
            a.hint = "the CPU is waiting for GPU surfaces to be copied back (flush + fence wait + "
                     "copy each time); next: F8 readback_ms, PROSPER_DEPTH_ARRAY_SNAPSHOT_CENSUS=1 "
                     "for depth arrays, PROSPER_READBACK_WHY=1 for colour targets; cf. #3882";
            out.push_back(std::move(a));
        }
    }

    // texture-reference-cost (per-reference MEAN from the 1-in-N sample, times the population, as a
    // SHARE of the budget).
    {
        const uint64_t refs = w.count(Counter::TextureReferences);
        const uint64_t samples = w.events(Cost::TextureRefSample);
        const double mean_ns =
            samples ? w.ms(Cost::TextureRefSample) * 1e6 / static_cast<double>(samples) : 0.0;
        const double est_ms = mean_ns * static_cast<double>(refs) / 1e6;
        const double share = w.flips ? est_ms / static_cast<double>(w.flips) / budget : 0.0;
        if (samples && per_second(samples, w.seconds) >= t.texture_reference_min_per_s &&
            share >= t.frontend_build_budget_share && mean_ns >= t.texture_reference_ns) {
            AlarmFiring a;
            a.rule = "texture-reference-cost";
            a.value = mean_ns / 1000.0;
            a.unit = "us/ref";
            a.threshold = t.texture_reference_ns / 1000.0;
            a.cost_ms = est_ms;
            a.detail = format("refs=%llu sampled=%llu est=%.0fms (%.0f%% of budget per flip) "
                              "frontend-build=%.0fms flips=%llu budget=%.1fms(%uHz)",
                              (unsigned long long)refs, (unsigned long long)samples, est_ms,
                              share * 100.0, w.ms(Cost::FrontendBuild),
                              (unsigned long long)w.flips, budget, w.target_hz);
            a.hint = "resolving each sampled texture reference in the frontend is slow (readbacks "
                     "excluded); next: PROSPER_TEXREF_CENSUS=1 (per-stage cost of the resolution "
                     "chain), F8 frontend_texture_ms; cf. #3877";
            out.push_back(std::move(a));
        }
    }

    // hle-blocking-wait (SHARE of one thread's wall time).
    {
        const uint64_t n = w.events(Cost::HleBlockingWait);
        const double blocked_ms = w.ms(Cost::HleBlockingWait);
        const double thread_share = blocked_ms / (w.seconds * 1000.0);
        if (n && thread_share >= t.hle_blocked_thread_share) {
            const char* label = w.cost_label[static_cast<size_t>(Cost::HleBlockingWait)];
            AlarmFiring a;
            a.rule = "hle-blocking-wait";
            a.value = thread_share * 100.0;
            a.unit = "%wall(all-threads)";
            a.threshold = t.hle_blocked_thread_share * 100.0;
            a.cost_ms = blocked_ms;
            a.detail = format("lock=\"%s\" blocked-calls=%llu avg=%.2fms max=%.2fms "
                              "(blocked time summed over threads; 100%% = one thread-equivalent)",
                              label ? label : "?", (unsigned long long)n, blocked_ms / n,
                              w.cost_max_ns[static_cast<size_t>(Cost::HleBlockingWait)] / 1e6);
            a.hint = "a guest thread is blocked on an HLE lock another thread holds (often across a "
                     "sleep or a wait); next: host gdb 'thread apply all bt' to find the holder, "
                     "PROSPER_EVLOG=1 for the VideoOut call sequence; cf. #3879";
            out.push_back(std::move(a));
        }
    }

    // present-cpu-overhead (per-present average, with a rate floor).
    {
        const uint64_t presents = w.events(Cost::PresentCpu);
        const double cpu_ms = w.ms(Cost::PresentCpu);
        const double per_present = presents ? cpu_ms / static_cast<double>(presents) : 0.0;
        if (presents && per_second(presents, w.seconds) >= t.present_min_per_s &&
            per_present >= t.present_cpu_ms) {
            AlarmFiring a;
            a.rule = "present-cpu-overhead";
            a.value = per_present;
            a.unit = "ms/present";
            a.threshold = t.present_cpu_ms;
            a.cost_ms = cpu_ms;
            a.detail = format("presents=%llu max=%.2fms", (unsigned long long)presents,
                              w.cost_max_ns[static_cast<size_t>(Cost::PresentCpu)] / 1e6);
            a.hint = "the host present thread spends CPU per frame outside its waits (content "
                     "signature under --fps, slot release, command recording); next: perf record -t "
                     "on the present thread, check any CPU-read buffer is HOST_CACHED; cf. #3875";
            out.push_back(std::move(a));
        }
    }

    // dropped-draws (CORRECTNESS).
    {
        const uint64_t backend = w.count(Counter::DroppedDrawsBackend);
        const uint64_t frontend = w.count(Counter::DroppedDrawsFrontend);
        const uint64_t contract = w.count(Counter::DroppedDrawsContract);
        const uint64_t total = backend + frontend + contract;
        if (total && total >= t.dropped_draws) {
            AlarmFiring a;
            a.rule = "dropped-draws";
            a.value = static_cast<double>(total);
            a.unit = "draws";
            a.threshold = static_cast<double>(t.dropped_draws);
            a.breakdown = ranked(w.drop_reasons, kDropReasonNames, kDropReasonCount);
            a.detail = format("frontend-unresolved=%llu frontend-contract=%llu backend-dropped=%llu "
                              "flips=%llu reasons=%s",
                              (unsigned long long)frontend, (unsigned long long)contract,
                              (unsigned long long)backend, (unsigned long long)w.flips,
                              top_entries(a.breakdown).c_str());
            a.hint = "draws prosper wanted to issue were dropped: a run can look faster AND render "
                     "wrong; `reasons` names the drop site (frontends/shared/live/live_renderer.cpp "
                     "for volume-*/render-array-reject/*, perf::DropReason); next: for backend/*, "
                     "[draw-disposition] lines (PROSPER_DRAW_DISPOSITION_VERBOSE=1); for "
                     "render-array-reject/*, [render-array-reject] lines "
                     "(PROSPER_ARRAY_REJECT_LOG_ALL=1); for volume-*, [volume-sample-drop] lines; "
                     "for contract-mismatch, PROSPER_DESCRIPTOR_VALIDATE; cf. #3889";
            out.push_back(std::move(a));
        }
    }

    // skipped-dispatches (CORRECTNESS).
    {
        const uint64_t skipped = w.count(Counter::SkippedDispatches);
        if (skipped && skipped >= t.skipped_dispatches) {
            AlarmFiring a;
            a.rule = "skipped-dispatches";
            a.value = static_cast<double>(skipped);
            a.unit = "dispatches";
            a.threshold = static_cast<double>(t.skipped_dispatches);
            a.breakdown = ranked(w.dispatch_skips, kDispatchSkipNames, kDispatchSkipCount);
            a.detail = format("flips=%llu reasons=%s", (unsigned long long)w.flips,
                              top_entries(a.breakdown).c_str());
            a.hint = "compute dispatches prosper wanted to run did not run, so their outputs are "
                     "stale or zero (a LUT, exposure, a light list): the frame can look faster AND "
                     "render wrong; next: the once-per-program [compute] skip lines name the "
                     "program, PROSPER_COMPUTE_PROGRAM_CENSUS=1 gives executed/skipped per program, "
                     "PROSPER_COMPUTE_DESCRIPTOR_DETAIL=1 for descriptor-contract; cf. #2481";
            out.push_back(std::move(a));
        }
    }

    // gpu-memory-off-device (CORRECTNESS class, by allocation class).
    {
        const uint64_t n = w.count(Counter::GpuMemoryOffDevice);
        if (n && n >= t.gpu_memory_off_device) {
            AlarmFiring a;
            a.rule = "gpu-memory-off-device";
            a.value = static_cast<double>(n);
            a.unit = "allocations";
            a.threshold = static_cast<double>(t.gpu_memory_off_device);
            const char* names[kGpuMemoryClassSlots];
            for (size_t i = 0; i < kGpuMemoryClassSlots; ++i)
                names[i] = w.gpu_memory_class_names[i] ? w.gpu_memory_class_names[i] : "?";
            a.breakdown = ranked(w.gpu_memory_off_device, names, kGpuMemoryClassSlots);
            a.detail = format("off-device=%llu bytes=%.1fMiB oom-fallbacks=%llu classes=%s",
                              (unsigned long long)n,
                              w.count(Counter::GpuMemoryOffDeviceBytes) / kMiB,
                              (unsigned long long)w.count(Counter::GpuMemoryFallbacks),
                              top_entries(a.breakdown).c_str());
            a.hint = "resources only the GPU touches were placed in system memory although the "
                     "device has device-local memory: correct pixels, but on a discrete GPU every "
                     "access crosses the bus. oom-fallbacks>0 means VRAM ran out and the allocation "
                     "was retried on a host type; otherwise the resource allowed no device-local "
                     "type. next: the [mem-placement] lines (class, type, heap), [gpu-mem] for "
                     "prosper's per-heap standing; cf. #3888, #3897";
            out.push_back(std::move(a));
        }
    }

    // host-copy-pressure (RATE).
    {
        uint64_t bytes = 0;
        for (uint64_t b : w.transfer_bytes) bytes += b;
        const double mib_per_s = per_second(bytes, w.seconds) / kMiB;
        if (bytes && mib_per_s >= t.host_copy_mib_per_s) {
            AlarmFiring a;
            a.rule = "host-copy-pressure";
            a.value = mib_per_s;
            a.unit = "MiB/s";
            a.threshold = t.host_copy_mib_per_s;
            const char* names[WindowSample::kTransferCount];
            uint64_t mib[WindowSample::kTransferCount];
            for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
                names[i] = transfer_name(static_cast<Transfer>(i));
                mib[i] = static_cast<uint64_t>(w.transfer_bytes[i] / kMiB + 0.5);
            }
            a.breakdown = ranked(mib, names, WindowSample::kTransferCount);
            a.detail = format("copied=%.0fMiB per-flip=%.1fMiB flips=%llu MiB-by-site=%s",
                              bytes / kMiB, w.flips ? bytes / kMiB / static_cast<double>(w.flips) : 0.0,
                              (unsigned long long)w.flips, top_entries(a.breakdown, 5).c_str());
            a.hint = "the CPU is copying/detiling guest data at a rate that is usually a residency "
                     "or dirty-tracking gap (re-detiling or re-snapshotting what did not change), "
                     "not real work; next: the [transfer-pressure] exit line, [tile-census] for "
                     "detile, [storage-materialize]/[gpu-seed-refused] for storage and snapshots; "
                     "cf. #3407, #3871";
            out.push_back(std::move(a));
        }
    }

    // shader-compile (SHARE, with a rate floor and a long sustain).
    {
        const uint64_t n = w.events(Cost::ShaderCompile) + w.events(Cost::PipelineCreate);
        const double ms = w.ms(Cost::ShaderCompile) + w.ms(Cost::PipelineCreate);
        const double share = w.flips ? ms / static_cast<double>(w.flips) / budget : 0.0;
        if (n && per_second(n, w.seconds) >= t.shader_compile_min_per_s &&
            share >= t.shader_compile_budget_share) {
            AlarmFiring a;
            a.rule = "shader-compile";
            a.value = share * 100.0;
            a.unit = "%budget(thread-time)";
            a.threshold = t.shader_compile_budget_share * 100.0;
            a.cost_ms = ms;
            a.detail = format("recompiles=%llu recompile=%.0fms max=%.1fms pipelines=%llu "
                              "pipeline=%.0fms max=%.1fms flips=%llu budget=%.1fms(%uHz)",
                              (unsigned long long)w.events(Cost::ShaderCompile),
                              w.ms(Cost::ShaderCompile),
                              w.cost_max_ns[static_cast<size_t>(Cost::ShaderCompile)] / 1e6,
                              (unsigned long long)w.events(Cost::PipelineCreate),
                              w.ms(Cost::PipelineCreate),
                              w.cost_max_ns[static_cast<size_t>(Cost::PipelineCreate)] / 1e6,
                              (unsigned long long)w.flips, budget, w.target_hz);
            a.hint = "shaders or pipelines have been compiled steadily for 40 s, which is what a "
                     "cache that never hits looks like (a key that changes every frame, or a cache "
                     "evicting what it just built); next: PROSPER_RENDER_TIMING shader hit/miss "
                     "and pipeline cache stats, F8 setup_pipeline_ms";
            out.push_back(std::move(a));
        }
    }
    // unaccounted-draws (CORRECTNESS).
    {
        const uint64_t n = w.count(Counter::DrawsUnaccounted);
        if (n && n >= t.unaccounted_draws) {
            AlarmFiring a;
            a.rule = "unaccounted-draws";
            a.value = static_cast<double>(n);
            a.unit = "draws";
            a.threshold = static_cast<double>(t.unaccounted_draws);
            a.detail = format("unaccounted=%llu flips=%llu (per pass |seen - recorded - dropped|)",
                              (unsigned long long)n, (unsigned long long)w.flips);
            a.hint = "some draws were neither recorded nor dropped with a reason: a skip site that "
                     "does not call note_dropped (or a draw counted twice), so dropped-draws cannot "
                     "see them; next: the [draw-disposition] ... UNACCOUNTED= lines, "
                     "PROSPER_DRAW_DISPOSITION_VERBOSE=1 for every pass (gpu/diagnostics/"
                     "draw_disposition.hpp); cf. #3891";
            out.push_back(std::move(a));
        }
    }

    // unimplemented-hle-calls (CORRECTNESS class: news).
    {
        const uint64_t first = w.count(Counter::HleUnimplementedFirst);
        if (first && first >= t.unimplemented_first_calls) {
            AlarmFiring a;
            a.rule = "unimplemented-hle-calls";
            a.value = static_cast<double>(first);
            a.unit = "new-nids";
            a.threshold = static_cast<double>(t.unimplemented_first_calls);
            a.detail = format("new-unimplemented=%llu calls=%llu (all unimplemented calls this "
                              "window) flips=%llu",
                              (unsigned long long)first,
                              (unsigned long long)w.count(Counter::HleUnimplementedCalls),
                              (unsigned long long)w.flips);
            a.hint = "the guest called a Sony function prosper has not registered, after boot: the "
                     "dispatcher answered 0 (reads as SCE_OK) and wrote no out-parameter, so the "
                     "guest may act on garbage; next: the [prosper] unimplemented: line names "
                     "library::NID [name]; implement it or answer it honestly; cf. #2951, #2023";
            out.push_back(std::move(a));
        }
    }

    // diagnostic-path-active (STATE).
    {
        const uint64_t mask = w.gauge(Gauge::DiagnosticPathSwitches);
        if (mask) {
            AlarmFiring a;
            a.rule = "diagnostic-path-active";
            a.value = 1;
            a.unit = "state";
            a.threshold = 1;
            std::string names;
            for (size_t i = 0; i < kDiagnosticPathSwitchCount; ++i) {
                if (!(mask & (1ull << i))) continue;
                if (!names.empty()) names += ',';
                names += kDiagnosticPathSwitchNames[i];
            }
            a.detail = format("switches=%s live-gpu-color-targets=off", names.c_str());
            a.hint = "this run is NOT on the production render path: a diagnostic switch turned "
                     "the GPU-resident colour targets off, so every frame is read back to the CPU "
                     "and re-presented; no performance number from this run describes what ships; "
                     "next: unset the named switch; cf. #3909";
            out.push_back(std::move(a));
        }
    }

    // present-path-fallback (SHARE of presents, with a floor).
    {
        const uint64_t fallback = w.count(Counter::PresentCpuFallbacks);
        const uint64_t gpu = w.events(Cost::PresentCpu);
        const double share = fallback + gpu
            ? static_cast<double>(fallback) / static_cast<double>(fallback + gpu) : 0.0;
        if (fallback && per_second(fallback, w.seconds) >= t.present_fallback_min_per_s &&
            share > t.present_fallback_share) {
            AlarmFiring a;
            a.rule = "present-path-fallback";
            a.value = share * 100.0;
            a.unit = "%presents";
            a.threshold = t.present_fallback_share * 100.0;
            a.detail = format("cpu-fallback=%llu gpu-scanout=%llu flips=%llu",
                              (unsigned long long)fallback, (unsigned long long)gpu,
                              (unsigned long long)w.flips);
            a.hint = "GPU present is on, but most frames were shown from a CPU readback (a full "
                     "frame read back and re-uploaded each present): live colour targets are off "
                     "or the GPU publish keeps missing; next: diagnostic-path-active, the [app] "
                     "fps line's gpu-present cpu-fallback count, PROSPER_PRESENT_HANDOFF_TRACE=1; "
                     "cf. #1270, #3895";
            out.push_back(std::move(a));
        }
    }

    // pipeline-cache-thrash (RATE, by cache).
    {
        uint64_t by_cache[3] = {w.count(Counter::PipelineEvictions),
                                w.count(Counter::PipelineLayoutEvictions),
                                w.count(Counter::DescriptorSetLayoutEvictions)};
        static const char* const kCacheNames[3] = {"pipeline", "pipeline-layout",
                                                   "descriptor-set-layout"};
        const uint64_t n = by_cache[0] + by_cache[1] + by_cache[2];
        const double rate = per_second(n, w.seconds);
        if (n && rate >= t.pipeline_cache_evictions_per_s) {
            AlarmFiring a;
            a.rule = "pipeline-cache-thrash";
            a.value = rate;
            a.unit = "evictions/s";
            a.threshold = t.pipeline_cache_evictions_per_s;
            a.breakdown = ranked(by_cache, kCacheNames, 3);
            a.detail = format("evictions=%llu by-cache=%s flips=%llu", (unsigned long long)n,
                              top_entries(a.breakdown).c_str(), (unsigned long long)w.flips);
            a.hint = "a renderer object cache is full and keeps evicting what the scene uses, so "
                     "pipelines (a driver compile each) and layouts are rebuilt every frame; next: "
                     "PROSPER_RENDER_TIMING pipeline/layout cache stats (entries, evictions), "
                     "shader-compile alarm, F8 setup_pipeline_ms";
            out.push_back(std::move(a));
        }
    }

    // texture-validation-churn (RATE).
    {
        const uint64_t bytes = w.count(Counter::TextureValidationFailedBytes);
        const double mib_per_s = per_second(bytes, w.seconds) / kMiB;
        if (bytes && mib_per_s >= t.texture_validation_failed_mib_per_s) {
            AlarmFiring a;
            a.rule = "texture-validation-churn";
            a.value = mib_per_s;
            a.unit = "MiB/s";
            a.threshold = t.texture_validation_failed_mib_per_s;
            a.detail = format("failed-validations=%llu validated=%.0fMiB flips=%llu",
                              (unsigned long long)w.count(Counter::TextureValidationFailures),
                              bytes / kMiB, (unsigned long long)w.flips);
            a.hint = "persistent decode-cache entries are compared byte-for-byte against guest "
                     "memory and found changed every time (a movie plane, a streamed atlas): the "
                     "compare is certain to fail and is paid on top of the re-decode; next: "
                     "PROSPER_TEXREF_CENSUS=1 persist_invalid class, PROSPER_DETILE_STATS; treat "
                     "such an entry as volatile and skip the compare; cf. #3900";
            out.push_back(std::move(a));
        }
    }
    return out;
}

}  // namespace prosper::diagnostics::perf

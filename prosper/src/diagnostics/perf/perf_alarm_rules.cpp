#include "diagnostics/perf/perf_alarm_rules.hpp"

#include <cstdio>

namespace prosper::diagnostics::perf {

namespace {

std::string format(const char* fmt, auto... args) {
    char buf[512];
    std::snprintf(buf, sizeof buf, fmt, args...);
    return buf;
}

double per_second(uint64_t n, double seconds) {
    return seconds > 0 ? static_cast<double>(n) / seconds : 0.0;
}

constexpr double kMiB = 1024.0 * 1024.0;

}  // namespace

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
    // dropped_draws stays at "any": a correctness alarm has no sensitivity to lower.
    return t;
}

const std::vector<const char*>& rule_names() {
    static const std::vector<const char*> names = {
        "texture-cache-thrash", "surface-readback", "texture-reference-cost",
        "hle-blocking-wait",    "present-cpu-overhead", "dropped-draws",
    };
    return names;
}

uint32_t sustain_windows(const char* rule) {
    const std::string r = rule ? rule : "";
    if (r == "dropped-draws") return kCorrectnessSustainWindows;
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
            a.unit = "%thread";
            a.threshold = t.hle_blocked_thread_share * 100.0;
            a.cost_ms = blocked_ms;
            a.detail = format("lock=\"%s\" blocked-calls=%llu avg=%.2fms max=%.2fms",
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
            a.hint = "the host present thread spends CPU per frame outside its GPU waits (the --fps "
                     "content signature reads a mapped sample buffer); next: perf record -t on the "
                     "present thread, check the sample buffer is HOST_CACHED; cf. #3875";
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
            a.detail = format("frontend-unresolved=%llu frontend-contract=%llu backend-dropped=%llu",
                              (unsigned long long)frontend, (unsigned long long)contract,
                              (unsigned long long)backend);
            a.hint = "draws prosper wanted to issue were dropped: a run can look faster AND render "
                     "wrong; next: [draw-disposition] lines (PROSPER_DRAW_DISPOSITION_VERBOSE=1) "
                     "for backend drops; for frontend-unresolved, [volume-sample-drop] lines and "
                     "PROSPER_RENDER_TIMING build_rejected; for frontend-contract, "
                     "PROSPER_DESCRIPTOR_VALIDATE; cf. #3889";
            out.push_back(std::move(a));
        }
    }
    return out;
}

}  // namespace prosper::diagnostics::perf

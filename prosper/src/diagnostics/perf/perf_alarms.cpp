#include "diagnostics/perf/perf_alarms.hpp"
#include "diagnostics/perf/wave64_refusal.hpp"

#include "diagnostics/env_numeric.hpp"
#include "diagnostics/exit_reports.hpp"
#include "gpu/diagnostics/diag_ratelimit.hpp"

#include "diagnostics/transfer_pressure.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace prosper::diagnostics::perf {

namespace {

// First three firings of a rule print, then powers of two (diag_ratelimit's contract).
constexpr uint64_t kPrintFirstN = 3;

// JSON string escaping for the few free-text fields (detail and hint are ASCII we wrote, but the
// lock label is a caller's literal; escape defensively rather than trust every future caller).
void json_string(FILE* f, const char* s) {
    std::fputc('"', f);
    for (; s && *s; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '"' || c == '\\') { std::fputc('\\', f); std::fputc(c, f); }
        else if (c < 0x20) std::fprintf(f, "\\u%04x", c);
        else std::fputc(c, f);
    }
    std::fputc('"', f);
}

void json_counts(FILE* f, const std::vector<std::pair<const char*, uint64_t>>& counts) {
    std::fputc('{', f);
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i) std::fputc(',', f);
        json_string(f, counts[i].first);
        std::fprintf(f, ":%llu", (unsigned long long)counts[i].second);
    }
    std::fputc('}', f);
}

double host_copy_mib(const WindowSample& w) {
    uint64_t bytes = 0;
    for (uint64_t b : w.transfer_bytes) bytes += b;
    return bytes / (1024.0 * 1024.0);
}

}  // namespace

AlarmEngine::AlarmEngine(EngineConfig config) : config_(std::move(config)) {
    for (const char* name : rule_names()) rules_.emplace_back(name, RuleState{});
    if (!config_.jsonl_path.empty()) {
        jsonl_ = std::fopen(config_.jsonl_path.c_str(), "w");
        if (!jsonl_ && config_.log)
            std::fprintf(config_.log, "[perf-alarm] cannot open PROSPER_PERF_ALARM_LOG=%s -- "
                                      "alarms still print here, but no JSONL is written\n",
                         config_.jsonl_path.c_str());
    }
}

AlarmEngine::~AlarmEngine() {
    if (jsonl_) std::fclose(jsonl_);
}

AlarmEngine::RuleState& AlarmEngine::state_for(const char* rule) {
    for (auto& [name, state] : rules_)
        if (std::strcmp(name, rule) == 0) return state;
    rules_.emplace_back(rule, RuleState{});
    return rules_.back().second;
}

std::vector<AlarmFiring> AlarmEngine::on_flip(uint64_t now_ns, Ledger& ledger,
                                              uint32_t target_hz,
                                              const ExternalTotals* external) {
    const ExternalTotals none{};
    const ExternalTotals& ext = external ? *external : none;
    std::unique_lock<std::mutex> lock(mutex_);
    if (!started_) {
        // Baseline: everything accumulated before the first flip (boot, shader warm-up) is not
        // charged to the first window, which would otherwise be a window of unbounded length.
        started_ = true;
        origin_ns_ = window_start_ns_ = now_ns;
        for (size_t i = 0; i < kCostCount; ++i) {
            prev_cost_ns_[i] = ledger.cost_ns[i].load(std::memory_order_relaxed);
            prev_cost_events_[i] = ledger.cost_events[i].load(std::memory_order_relaxed);
            ledger.cost_max_ns[i].store(0, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < kCounterCount; ++i)
            prev_counters_[i] = ledger.counters[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kDropReasonCount; ++i)
            prev_drop_reasons_[i] = ledger.drop_reasons[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kDispatchSkipCount; ++i)
            prev_dispatch_skips_[i] = ledger.dispatch_skips[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kGpuMemoryClassSlots; ++i)
            prev_gpu_memory_off_device_[i] =
                ledger.gpu_memory_off_device[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kPresentDeclineSlots; ++i)
            prev_present_declines_[i] = ledger.present_declines[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kRttDestinationRefusalSlots; ++i)
            prev_rtt_destination_refused_bytes_[i] =
                ledger.rtt_destination_refused_bytes[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kRttDestinationRefusalSlots; ++i)
            prev_rtt_destination_refusals_[i] =
                ledger.rtt_destination_refusals[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kExactResultDeclineCount; ++i)
            prev_exact_result_declines_[i] =
                ledger.exact_result_declines[i].load(std::memory_order_relaxed);
        for (size_t i = 0; i < kPeakCount; ++i) ledger.peaks[i].store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
            prev_transfer_bytes_[i] = ext.transfer_bytes[i];
            prev_transfer_calls_[i] = ext.transfer_calls[i];
        }
        return {};
    }
    ++flips_in_window_;
    if (now_ns - window_start_ns_ < config_.window_ns) return {};

    WindowSample w;
    w.seconds = static_cast<double>(now_ns - window_start_ns_) / 1e9;
    w.flips = flips_in_window_;
    w.target_hz = target_hz;
    for (size_t i = 0; i < kCostCount; ++i) {
        const uint64_t ns = ledger.cost_ns[i].load(std::memory_order_relaxed);
        const uint64_t ev = ledger.cost_events[i].load(std::memory_order_relaxed);
        w.cost_ns[i] = ns - prev_cost_ns_[i];
        w.cost_events[i] = ev - prev_cost_events_[i];
        w.cost_max_ns[i] = ledger.cost_max_ns[i].exchange(0, std::memory_order_relaxed);
        w.cost_label[i] = ledger.cost_label[i].load(std::memory_order_relaxed);
        prev_cost_ns_[i] = ns;
        prev_cost_events_[i] = ev;
    }
    for (size_t i = 0; i < kCounterCount; ++i) {
        const uint64_t v = ledger.counters[i].load(std::memory_order_relaxed);
        w.counters[i] = v - prev_counters_[i];
        prev_counters_[i] = v;
    }
    for (size_t i = 0; i < kGaugeCount; ++i)
        w.gauges[i] = ledger.gauges[i].load(std::memory_order_relaxed);
    for (size_t i = 0; i < kDropReasonCount; ++i) {
        const uint64_t v = ledger.drop_reasons[i].load(std::memory_order_relaxed);
        w.drop_reasons[i] = v - prev_drop_reasons_[i];
        prev_drop_reasons_[i] = v;
    }
    for (size_t i = 0; i < kDispatchSkipCount; ++i) {
        const uint64_t v = ledger.dispatch_skips[i].load(std::memory_order_relaxed);
        w.dispatch_skips[i] = v - prev_dispatch_skips_[i];
        prev_dispatch_skips_[i] = v;
    }
    for (size_t i = 0; i < kGpuMemoryClassSlots; ++i) {
        const uint64_t v = ledger.gpu_memory_off_device[i].load(std::memory_order_relaxed);
        w.gpu_memory_off_device[i] = v - prev_gpu_memory_off_device_[i];
        prev_gpu_memory_off_device_[i] = v;
        w.gpu_memory_class_names[i] = ledger.gpu_memory_class_names[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kPresentDeclineSlots; ++i) {
        const uint64_t v = ledger.present_declines[i].load(std::memory_order_relaxed);
        w.present_declines[i] = v - prev_present_declines_[i];
        prev_present_declines_[i] = v;
        w.present_decline_names[i] = ledger.present_decline_names[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < kPeakCount; ++i)
        w.peaks[i] = ledger.peaks[i].exchange(0, std::memory_order_relaxed);
    for (size_t i = 0; i < kRttDestinationRefusalSlots; ++i) {
        const uint64_t v = ledger.rtt_destination_refused_bytes[i].load(std::memory_order_relaxed);
        w.rtt_destination_refused_bytes[i] = v - prev_rtt_destination_refused_bytes_[i];
        prev_rtt_destination_refused_bytes_[i] = v;
        const uint64_t n = ledger.rtt_destination_refusals[i].load(std::memory_order_relaxed);
        w.rtt_destination_refusals[i] = n - prev_rtt_destination_refusals_[i];
        prev_rtt_destination_refusals_[i] = n;
        w.rtt_destination_refusal_names[i] =
            ledger.rtt_destination_refusal_names[i].load(std::memory_order_relaxed);
        if (w.rtt_destination_refusal_names[i]) rtt_destination_refusal_names_[i] =
            w.rtt_destination_refusal_names[i];
    }
    for (size_t i = 0; i < kExactResultDeclineCount; ++i) {
        const uint64_t v = ledger.exact_result_declines[i].load(std::memory_order_relaxed);
        w.exact_result_declines[i] = v - prev_exact_result_declines_[i];
        prev_exact_result_declines_[i] = v;
    }
    for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
        // A total that went BACKWARDS (a test's fresh source) is a new baseline, not a huge delta.
        const uint64_t v = ext.transfer_bytes[i];
        w.transfer_bytes[i] = v >= prev_transfer_bytes_[i] ? v - prev_transfer_bytes_[i] : 0;
        prev_transfer_bytes_[i] = v;
        const uint64_t c = ext.transfer_calls[i];
        w.transfer_calls[i] = c >= prev_transfer_calls_[i] ? c - prev_transfer_calls_[i] : 0;
        prev_transfer_calls_[i] = c;
    }
    window_start_ns_ = now_ns;
    flips_in_window_ = 0;
    const double t = static_cast<double>(now_ns - origin_ns_) / 1e9;
    lock.unlock();
    return close_window(w, t);
}

std::vector<AlarmFiring> AlarmEngine::close_window(const WindowSample& w, double t_seconds) {
    std::vector<AlarmFiring> candidates = evaluate_rules(w, config_.thresholds);
    std::lock_guard<std::mutex> lock(mutex_);
    ++windows_;
    for (size_t i = 0; i < kWaitRegMemDirectCounterCount; ++i)
        wait_regmem_direct_totals_[i] += w.count(kWaitRegMemDirectCounters[i]);
    for (size_t i = 0; i < kTextureDirectValidationCounterCount; ++i)
        texture_direct_validation_totals_[i] += w.count(kTextureDirectValidationCounters[i]);
    std::vector<AlarmFiring> fired;
    for (auto& [name, state] : rules_) {
        if (rule_has_data(name, w)) ++state.windows_with_data;
        bool held = false;
        for (AlarmFiring& a : candidates)
            if (std::strcmp(a.rule, name) == 0) {
                held = true;
                if (++state.streak >= sustain_windows(name)) fired.push_back(std::move(a));
                break;
            }
        if (!held) state.streak = 0;
    }
    // After the streaks: a deferral changes what is printed, never what is counted.
    apply_reporting_deferrals(fired);
    active_.clear();
    for (const AlarmFiring& a : fired) active_.push_back(a.rule);
    for (const AlarmFiring& a : fired) {
        RuleState& s = state_for(a.rule);
        const uint64_t ordinal = ++s.fired;
        if (ordinal == 1) s.first_t = t_seconds;
        for (const auto& [name, count] : a.breakdown) {
            auto it = std::find_if(s.breakdown_total.begin(), s.breakdown_total.end(),
                                   [&](const auto& e) { return std::strcmp(e.first, name) == 0; });
            if (it == s.breakdown_total.end()) s.breakdown_total.emplace_back(name, count);
            else it->second += count;
        }
        if (ordinal == 1 || a.value > s.worst_value) {
            s.worst_value = a.value;
            s.worst_t = t_seconds;
            s.worst_detail = a.detail;
        }
        if (config_.log && prosper::diag_should_print(ordinal, kPrintFirstN)) {
            char cost[48] = "";
            if (a.cost_ms >= 0) std::snprintf(cost, sizeof cost, " cost~%.0fms", a.cost_ms);
            std::fprintf(config_.log,
                         "[perf-alarm] #%llu rule=%s t=%.0fs window=%.1fs value=%.2f %s "
                         "threshold=%.2f%s %s hint=%s\n",
                         (unsigned long long)ordinal, a.rule, t_seconds, w.seconds, a.value, a.unit,
                         a.threshold, cost, a.detail.c_str(), a.hint);
        }
        if (jsonl_) {
            std::fprintf(jsonl_, "{\"type\":\"alarm\",\"t\":%.3f,\"rule\":", t_seconds);
            json_string(jsonl_, a.rule);
            std::fprintf(jsonl_, ",\"ordinal\":%llu,\"window_s\":%.3f,\"flips\":%llu,"
                                 "\"target_hz\":%u,\"value\":%.4f,\"unit\":",
                         (unsigned long long)ordinal, w.seconds, (unsigned long long)w.flips,
                         w.target_hz, a.value);
            json_string(jsonl_, a.unit);
            std::fprintf(jsonl_, ",\"threshold\":%.4f,\"cost_ms\":", a.threshold);
            if (a.cost_ms >= 0) std::fprintf(jsonl_, "%.3f", a.cost_ms);
            else std::fputs("null", jsonl_);
            std::fputs(",\"detail\":", jsonl_);
            json_string(jsonl_, a.detail.c_str());
            std::fputs(",\"hint\":", jsonl_);
            json_string(jsonl_, a.hint);
            if (!a.breakdown.empty()) {
                std::fputs(",\"breakdown\":", jsonl_);
                json_counts(jsonl_, a.breakdown);
            }
            std::fputs("}\n", jsonl_);
        }
    }
    if (jsonl_) {
        // Every window, fired or not: the raw quantities the rules read, so a tool can see how close
        // a quiet run came to a threshold and calibrate against a known answer.
        const auto ms = [&](Cost c) { return w.ms(c); };
        const auto ev = [&](Cost c) { return (unsigned long long)w.events(c); };
        const auto n = [&](Counter c) { return (unsigned long long)w.count(c); };
        std::fprintf(jsonl_,
                     "{\"type\":\"window\",\"t\":%.3f,\"window_s\":%.3f,\"flips\":%llu,"
                     "\"target_hz\":%u,\"alarms\":%zu,"
                     "\"readback_ms\":%.3f,\"readbacks\":%llu,\"readback_max_ms\":%.3f,"
                     "\"hle_blocked_ms\":%.3f,\"hle_blocked_calls\":%llu,"
                     "\"present_cpu_ms\":%.3f,\"presents\":%llu,"
                     "\"frontend_build_ms\":%.3f,\"frontend_build_groups\":%llu,"
                     "\"texref_sample_ms\":%.3f,\"texref_samples\":%llu,"
                     "\"texture_refs\":%llu,\"texture_misses\":%llu,\"texture_refusals\":%llu,"
                     "\"texture_evictions\":%llu,\"texture_cache_mib\":%.1f,"
                     "\"texture_limit_mib\":%.1f,\"dropped_frontend\":%llu,"
                     "\"dropped_contract\":%llu,\"dropped_backend\":%llu,\"vk_allocs\":%llu,"
                     "\"skipped_dispatches\":%llu,\"shader_compile_ms\":%.3f,"
                     "\"shader_compiles\":%llu,\"shader_compile_max_ms\":%.3f,"
                     "\"pipeline_create_ms\":%.3f,\"pipeline_creates\":%llu,"
                     "\"pipeline_create_max_ms\":%.3f,\"host_copy_mib\":%.1f",
                     t_seconds, w.seconds, (unsigned long long)w.flips, w.target_hz, fired.size(),
                     ms(Cost::SurfaceReadback), ev(Cost::SurfaceReadback),
                     w.cost_max_ns[static_cast<size_t>(Cost::SurfaceReadback)] / 1e6,
                     ms(Cost::HleBlockingWait), ev(Cost::HleBlockingWait),
                     ms(Cost::PresentCpu), ev(Cost::PresentCpu),
                     ms(Cost::FrontendBuild), ev(Cost::FrontendBuild),
                     ms(Cost::TextureRefSample), ev(Cost::TextureRefSample),
                     n(Counter::TextureReferences), n(Counter::TextureCacheMisses),
                     n(Counter::TextureCacheRefusals), n(Counter::TextureCacheEvictions),
                     w.gauge(Gauge::TextureCacheBytes) / (1024.0 * 1024.0),
                     w.gauge(Gauge::TextureCacheLimit) / (1024.0 * 1024.0),
                     n(Counter::DroppedDrawsFrontend), n(Counter::DroppedDrawsContract),
                     n(Counter::DroppedDrawsBackend),
                     n(Counter::DeviceAllocations),
                     n(Counter::SkippedDispatches),
                     ms(Cost::ShaderCompile), ev(Cost::ShaderCompile),
                     w.cost_max_ns[static_cast<size_t>(Cost::ShaderCompile)] / 1e6,
                     ms(Cost::PipelineCreate), ev(Cost::PipelineCreate),
                     w.cost_max_ns[static_cast<size_t>(Cost::PipelineCreate)] / 1e6,
                     host_copy_mib(w));
        // Breakdowns: always present (possibly empty) so a reader need not special-case absence.
        std::fputs(",\"drop_reasons\":", jsonl_);
        json_counts(jsonl_, ranked(w.drop_reasons, kDropReasonNames, kDropReasonCount));
        std::fputs(",\"dispatch_skips\":", jsonl_);
        json_counts(jsonl_, ranked(w.dispatch_skips, kDispatchSkipNames, kDispatchSkipCount));
        {
            uint64_t counts[kWave64RefusalCount];
            for (size_t i = 0; i < kWave64RefusalCount; ++i)
                counts[i] = w.count(kWave64RefusalCounters[i]);
            std::fputs(",\"wave64_refusals\":", jsonl_);
            json_counts(jsonl_, ranked(counts, kWave64RefusalNames, kWave64RefusalCount));
            std::fprintf(jsonl_, ",\"wave64_new_refusal_identities\":%llu,"
                         "\"wave64_unidentified_uses\":%llu,\"wave64_inventory_overflow_uses\":%llu,"
                         "\"wave64_shader_checks\":%llu",
                         n(Counter::Wave64NewRefusalIdentities),
                         n(Counter::Wave64UnidentifiedRefusals), n(Counter::Wave64InventoryOverflow),
                         n(Counter::Wave64ShaderChecks));
        }
        std::fprintf(jsonl_, ",\"fragment_arithmetic_requests\":%llu,"
            "\"fragment_arithmetic_known_mode_requests\":%llu,"
            "\"fragment_arithmetic_unknown_mode_requests\":%llu,"
            "\"fragment_arithmetic_add_requests\":%llu,\"fragment_arithmetic_mul_requests\":%llu,"
            "\"fragment_arithmetic_refused_requests\":%llu,"
            "\"fragment_arithmetic_truncated_requests\":%llu,"
            "\"fragment_arithmetic_announcement_overflow_requests\":%llu",
            n(Counter::FragmentArithmeticRequests), n(Counter::FragmentArithmeticKnownMode),
            n(Counter::FragmentArithmeticUnknownMode), n(Counter::FragmentArithmeticAddRequests),
            n(Counter::FragmentArithmeticMulRequests), n(Counter::FragmentArithmeticRefusedRequests),
            n(Counter::FragmentArithmeticTruncatedRequests),
            n(Counter::FragmentArithmeticInventoryOverflowRequests));
        {
            const char* names[kGpuMemoryClassSlots];
            for (size_t i = 0; i < kGpuMemoryClassSlots; ++i)
                names[i] = w.gpu_memory_class_names[i] ? w.gpu_memory_class_names[i] : "?";
            std::fprintf(jsonl_, ",\"gpu_memory_fallbacks\":%llu,\"gpu_memory_off_device\":",
                         n(Counter::GpuMemoryFallbacks));
            json_counts(jsonl_, ranked(w.gpu_memory_off_device, names, kGpuMemoryClassSlots));
        }
        {
            const char* names[kPresentDeclineSlots];
            for (size_t i = 0; i < kPresentDeclineSlots; ++i)
                names[i] = w.present_decline_names[i] ? w.present_decline_names[i] : "?";
            std::fputs(",\"present_gpu_declines\":", jsonl_);
            json_counts(jsonl_, ranked(w.present_declines, names, kPresentDeclineSlots));
        }
        // #3891 queue rules' raw quantities.
        std::fprintf(jsonl_,
                     ",\"draws_unaccounted\":%llu,\"hle_unimplemented_calls\":%llu,"
                     "\"hle_unimplemented_first\":%llu,\"present_cpu_fallbacks\":%llu,"
                     "\"pipeline_evictions\":%llu,\"pipeline_layout_evictions\":%llu,"
                     "\"descriptor_set_layout_evictions\":%llu,"
                     "\"texture_validation_failures\":%llu,\"texture_validation_failed_mib\":%.2f,"
                     "\"diagnostic_path_switches\":%llu",
                     n(Counter::DrawsUnaccounted), n(Counter::HleUnimplementedCalls),
                     n(Counter::HleUnimplementedFirst), n(Counter::PresentCpuFallbacks),
                     n(Counter::PipelineEvictions), n(Counter::PipelineLayoutEvictions),
                     n(Counter::DescriptorSetLayoutEvictions),
                     n(Counter::TextureValidationFailures),
                     n(Counter::TextureValidationFailedBytes) / (1024.0 * 1024.0),
                     (unsigned long long)w.gauge(Gauge::DiagnosticPathSwitches));
        std::fputs(",\"host_copy_mib_by_site\":{", jsonl_);
        bool first = true;
        for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
            if (!w.transfer_bytes[i]) continue;
            if (!first) std::fputc(',', jsonl_);
            first = false;
            json_string(jsonl_, transfer_name(static_cast<Transfer>(i)));
            std::fprintf(jsonl_, ":%.1f", w.transfer_bytes[i] / (1024.0 * 1024.0));
        }
        // host-copy-per-flip's inputs: the per-flip figure (null below the flip floor, where the
        // rule has no data), the per-second figure host-copy-pressure reads (kept visible even
        // when that rule is not reported in favour of the per-flip one), and calls per site.
        std::fputs("},\"host_copy_mib_per_flip\":", jsonl_);
        if (w.flips >= kHostCopyPerFlipMinFlips)
            std::fprintf(jsonl_, "%.2f", host_copy_mib(w) / static_cast<double>(w.flips));
        else
            std::fputs("null", jsonl_);
        std::fprintf(jsonl_, ",\"host_copy_mib_per_s\":%.1f,\"host_copy_calls_by_site\":{",
                     w.seconds > 0 ? host_copy_mib(w) / w.seconds : 0.0);
        first = true;
        for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
            if (!w.transfer_calls[i]) continue;
            if (!first) std::fputc(',', jsonl_);
            first = false;
            json_string(jsonl_, transfer_name(static_cast<Transfer>(i)));
            std::fprintf(jsonl_, ":%llu", (unsigned long long)w.transfer_calls[i]);
        }
        std::fprintf(jsonl_, "},\"present_slot_trouble_declines\":%llu",
                     (unsigned long long)present_slot_trouble_declines(w));
        std::fprintf(jsonl_, ",\"present_window_unavailable\":%llu",
                     (unsigned long long)w.count(Counter::PresentWindowUnavailable));
        // #3948 stage 0: executor fence waits and the GPU device time inside them.
        std::fprintf(jsonl_, ",\"gpu_wait_compute_ms\":%.3f,\"gpu_wait_compute_n\":%llu,"
                             "\"gpu_device_compute_ms\":%.3f,\"gpu_device_compute_n\":%llu,"
                             "\"gpu_wait_graphics_ms\":%.3f,\"gpu_wait_graphics_n\":%llu,"
                             "\"gpu_device_graphics_ms\":%.3f,\"gpu_device_graphics_n\":%llu",
                     ms(Cost::GpuWaitCompute), ev(Cost::GpuWaitCompute),
                     n(Counter::GpuDeviceNsCompute) / 1e6, n(Counter::GpuDeviceSamplesCompute),
                     ms(Cost::GpuWaitGraphics), ev(Cost::GpuWaitGraphics),
                     n(Counter::GpuDeviceNsGraphics) / 1e6, n(Counter::GpuDeviceSamplesGraphics));
        std::fprintf(jsonl_, ",\"gpu_graphics_deferred\":%llu,\"gpu_graphics_deferred_blocked\":%llu",
                     n(Counter::GpuGraphicsDeferred), n(Counter::GpuGraphicsDeferredBlocked));
        std::fprintf(jsonl_, ",\"cpu_rtt_publication_checks\":%llu,\"cpu_rtt_colorless_publications\":%llu",
                     n(Counter::CpuRttPublicationChecks), n(Counter::CpuRttColorlessPublications));
        {
            uint64_t counts[kWaitRegMemDirectCounterCount];
            for (size_t i = 0; i < kWaitRegMemDirectCounterCount; ++i)
                counts[i] = w.count(kWaitRegMemDirectCounters[i]);
            std::fprintf(jsonl_, ",\"wait_regmem_direct_evaluations\":%llu,"
                         "\"wait_regmem_direct_compare_false\":%llu,"
                         "\"wait_regmem_direct_unreadable\":%llu,"
                         "\"wait_regmem_direct_unsupported\":%llu,"
                         "\"wait_regmem_direct_false_proceed\":%llu,"
                         "\"wait_regmem_direct_false_defer\":%llu,\"wait_regmem_direct_data\":",
                         (unsigned long long)counts[0], (unsigned long long)counts[1],
                         (unsigned long long)counts[2], (unsigned long long)counts[3],
                         (unsigned long long)counts[4], (unsigned long long)counts[5]);
            json_string(jsonl_, wait_regmem_direct_data_status(counts));
        }
        {
            uint64_t counts[kTextureDirectValidationCounterCount];
            for (size_t i = 0; i < kTextureDirectValidationCounterCount; ++i)
                counts[i] = w.count(kTextureDirectValidationCounters[i]);
            std::fprintf(jsonl_, ",\"texture_direct_validation_attempts\":%llu,"
                         "\"texture_direct_validation_memcmp_calls\":%llu,"
                         "\"texture_direct_validation_memcmp_extent_bytes\":%llu,"
                         "\"texture_direct_validation_accepted_prefix\":%llu,"
                         "\"texture_direct_validation_bytes_differ\":%llu,"
                         "\"texture_direct_validation_expected_missing\":%llu,"
                         "\"texture_direct_validation_incomplete_prefix\":%llu,"
                         "\"texture_direct_validation_data\":",
                         (unsigned long long)counts[0], (unsigned long long)counts[1],
                         (unsigned long long)counts[2], (unsigned long long)counts[3],
                         (unsigned long long)counts[4], (unsigned long long)counts[5],
                         (unsigned long long)counts[6]);
            json_string(jsonl_, texture_direct_validation_data_status(counts));
        }
        // 2026-09-29 queue: compute destination refusals (MiB by reason), the colour-target
        // cache's bounds and window peaks, and the exact full-overwrite shape census.
        std::fprintf(jsonl_, ",\"rtt_destination_refusals\":%llu,"
                             "\"rtt_destination_refused_mib\":%.1f,\"rtt_destination_refused_mib_by_reason\":",
                     n(Counter::RttDestinationRefusals),
                     n(Counter::RttDestinationRefusedBytes) / (1024.0 * 1024.0));
        json_counts(jsonl_, rtt_destination_refused_mib(w));
        std::fprintf(jsonl_, ",\"persistent_target_peak_entries\":%llu,"
                             "\"persistent_target_entry_limit\":%llu,"
                             "\"persistent_target_peak_mib\":%.1f,\"persistent_target_limit_mib\":%.1f,"
                             "\"persistent_target_evictions\":%llu,"
                             "\"persistent_target_evicted_mib\":%.1f,"
                             "\"exact_result_candidates\":%llu,\"exact_result_verdicts\":",
                     (unsigned long long)w.peak(Peak::PersistentTargetEntries),
                     (unsigned long long)w.gauge(Gauge::PersistentTargetEntryLimit),
                     w.peak(Peak::PersistentTargetBytes) / (1024.0 * 1024.0),
                     w.gauge(Gauge::PersistentTargetByteLimit) / (1024.0 * 1024.0),
                     n(Counter::PersistentTargetEvictions),
                     n(Counter::PersistentTargetEvictedBytes) / (1024.0 * 1024.0),
                     n(Counter::ExactResultCandidates));
        json_counts(jsonl_, ranked(w.exact_result_declines, kExactResultDeclineNames,
                                   kExactResultDeclineCount));
        std::fputs("}\n", jsonl_);
        std::fflush(jsonl_);
    }
    return fired;
}

bool AlarmEngine::write_summary(FILE* out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!windows_ || !out) return false;
    bool any = false;
    for (const auto& [name, s] : rules_) {
        if (!s.fired) continue;
        any = true;
        std::fprintf(out, "[perf-alarm] summary rule=%s fired in %llu of %llu windows (first t=%.0fs); "
                          "worst value=%.2f at t=%.0fs: %s\n",
                     name, (unsigned long long)s.fired, (unsigned long long)windows_, s.first_t,
                     s.worst_value, s.worst_t, s.worst_detail.c_str());
        if (!s.breakdown_total.empty()) {
            // Over every window the rule fired in: which site/reason it was, for the whole run.
            auto total = s.breakdown_total;
            std::stable_sort(total.begin(), total.end(),
                             [](const auto& a, const auto& b) { return a.second > b.second; });
            std::fprintf(out, "[perf-alarm] summary rule=%s breakdown over fired windows: %s\n",
                         name, top_entries(total, 6).c_str());
        }
    }
    std::string no_data;
    size_t with_data = 0;
    for (const auto& [name, s] : rules_) {
        if (s.windows_with_data) { ++with_data; continue; }
        if (!no_data.empty()) no_data += ",";
        no_data += name;
    }
    if (!any)
        std::fprintf(out, "[perf-alarm] summary: no rule fired in %llu windows of %.1fs "
                          "(%zu of %zu rules had data)\n",
                     (unsigned long long)windows_, config_.window_ns / 1e9, with_data, rules_.size());
    // Observation, not an alarm. This narrow scope neither detects hardware timeouts nor proves
    // completion order; independent relaxed inputs do not certify counter balance/quiescence.
    std::fprintf(out, "[perf-alarm] summary observer=wait-regmem-direct-fold "
                 "completed-windows=%llu snapshot=relaxed data=%s evaluations=%llu "
                 "compare-false=%llu unreadable=%llu unsupported=%llu "
                 "false-proceed=%llu false-defer=%llu\n",
                 (unsigned long long)windows_,
                 wait_regmem_direct_data_status(wait_regmem_direct_totals_),
                 (unsigned long long)wait_regmem_direct_totals_[0],
                 (unsigned long long)wait_regmem_direct_totals_[1],
                 (unsigned long long)wait_regmem_direct_totals_[2],
                 (unsigned long long)wait_regmem_direct_totals_[3],
                 (unsigned long long)wait_regmem_direct_totals_[4],
                 (unsigned long long)wait_regmem_direct_totals_[5]);
    // Direct comparison argument extents are observations, not physical bandwidth or an alarm.
    std::fprintf(out, "[perf-alarm] summary observer=texture-direct-validation "
                 "completed-windows=%llu snapshot=relaxed data=%s attempts=%llu "
                 "memcmp-calls=%llu memcmp-extent-bytes=%llu accepted-prefix=%llu "
                 "bytes-differ=%llu expected-missing=%llu incomplete-prefix=%llu\n",
                 (unsigned long long)windows_,
                 texture_direct_validation_data_status(texture_direct_validation_totals_),
                 (unsigned long long)texture_direct_validation_totals_[0],
                 (unsigned long long)texture_direct_validation_totals_[1],
                 (unsigned long long)texture_direct_validation_totals_[2],
                 (unsigned long long)texture_direct_validation_totals_[3],
                 (unsigned long long)texture_direct_validation_totals_[4],
                 (unsigned long long)texture_direct_validation_totals_[5],
                 (unsigned long long)texture_direct_validation_totals_[6]);
    // Not a rule: the exact full-overwrite shape census over the run (to the last window close),
    // and the destination refusals behind it -- the funnel a compute result takes to reach the
    // renderer's device image, printed whether or not anything fired.
    {
        const auto verdicts =
            ranked(prev_exact_result_declines_, kExactResultDeclineNames, kExactResultDeclineCount);
        const char* names[kRttDestinationRefusalSlots];
        uint64_t mib[kRttDestinationRefusalSlots];
        for (size_t i = 0; i < kRttDestinationRefusalSlots; ++i) {
            names[i] = rtt_destination_refusal_names_[i] ? rtt_destination_refusal_names_[i] : "?";
            mib[i] = prev_rtt_destination_refused_bytes_[i] >> 20;
        }
        const auto refused = ranked(mib, names, kRttDestinationRefusalSlots);
        if (!verdicts.empty() || !refused.empty())
            std::fprintf(out, "[perf-alarm] summary: compute results tested for the exact "
                              "full-overwrite shape, first failing field: %s; destination "
                              "refusals of accepted results (MiB): %s\n",
                         top_entries(verdicts, 8).c_str(), top_entries(refused, 6).c_str());
    }
    if (!no_data.empty())
        std::fprintf(out, "[perf-alarm] summary: NO DATA (not measured in any window, so not "
                          "quiet): %s\n", no_data.c_str());
    return true;
}

uint64_t AlarmEngine::windows_evaluated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return windows_;
}

std::vector<const char*> AlarmEngine::active_rules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_;
}

uint64_t AlarmEngine::times_fired(const char* rule) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [name, s] : rules_)
        if (std::strcmp(name, rule) == 0) return s.fired;
    return 0;
}

// ---- process-wide ------------------------------------------------------------------------------

namespace {

std::atomic<uint32_t> g_target_hz{60};
// Bit i set: rule_names()[i] was reported in the most recently closed window. Published by the
// flip thread after each window, read by the HUD without touching the engine.
std::atomic<uint32_t> g_active_mask{0};

AlarmEngine& process_engine() {
    static AlarmEngine* const engine = [] {
        EngineConfig config;
        const uint64_t window_ms = prosper::diag::env_u64_or_default(
            "PROSPER_PERF_ALARM_WINDOW_MS", std::getenv("PROSPER_PERF_ALARM_WINDOW_MS"), 5000,
            "milliseconds");
        config.window_ns = (window_ms ? window_ms : 5000) * 1'000'000ull;
        const uint64_t pct = prosper::diag::env_u64_or_default(
            "PROSPER_PERF_ALARM_THRESHOLD_PCT", std::getenv("PROSPER_PERF_ALARM_THRESHOLD_PCT"),
            100, "percent");
        config.thresholds = RuleThresholds::scaled(static_cast<double>(pct ? pct : 100));
        if (const char* path = std::getenv("PROSPER_PERF_ALARM_LOG")) config.jsonl_path = path;
        auto* e = new AlarmEngine(std::move(config));  // never destroyed: the exit report reads it
        // Registered on first use, never at static initialisation (exit_census.hpp, trap 289).
        register_exit_report([e] { e->write_summary(stderr); });
        return e;
    }();
    return *engine;
}

}  // namespace

void on_guest_flip() {
    if (!enabled()) return;
    AlarmEngine& engine = process_engine();
    const uint64_t before = engine.windows_evaluated();
    AlarmEngine::ExternalTotals external;
    for (size_t i = 0; i < WindowSample::kTransferCount; ++i) {
        external.transfer_bytes[i] = transfer_bytes(static_cast<Transfer>(i));
        external.transfer_calls[i] = transfer_calls(static_cast<Transfer>(i));
    }
    engine.on_flip(now_ns(), ledger(), g_target_hz.load(std::memory_order_relaxed), &external);
    if (engine.windows_evaluated() == before) return;   // no window closed on this flip
    uint32_t mask = 0;
    const auto& names = rule_names();
    for (const char* active : engine.active_rules())
        for (size_t i = 0; i < names.size() && i < 32; ++i)
            if (std::strcmp(names[i], active) == 0) mask |= 1u << i;
    g_active_mask.store(mask, std::memory_order_relaxed);
}

size_t active_alarm_summary(char* out, size_t cap) {
    if (cap) out[0] = '\0';
    if (!enabled()) return 0;
    const uint32_t mask = g_active_mask.load(std::memory_order_relaxed);
    size_t n = 0, used = 0;
    const auto& names = rule_names();
    for (size_t i = 0; i < names.size() && i < 32; ++i) {
        if (!(mask & (1u << i))) continue;
        ++n;
        if (!cap) continue;
        const int wrote = std::snprintf(out + used, cap - used, "%s%s", used ? "," : "", names[i]);
        if (wrote > 0) used = std::min(cap - 1, used + static_cast<size_t>(wrote));
    }
    return n;
}

void set_guest_flip_rate(int32_t rate) {
    if (rate >= 0 && rate <= 2) g_target_hz.store(60u / static_cast<uint32_t>(rate + 1),
                                                  std::memory_order_relaxed);
}

}  // namespace prosper::diagnostics::perf

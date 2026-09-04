#include "shared/perf/performance_capture.hpp"
#include "build_revision.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#endif

namespace prosper::perf {
namespace {

std::string sanitize_component(std::string_view raw) {
    std::string out;
    out.reserve(std::min<size_t>(raw.size(), 32));
    for (char c : raw) {
        if (out.size() == 32) break;
        const unsigned char u = static_cast<unsigned char>(c);
        const bool keep = (u >= '0' && u <= '9') || (u >= 'A' && u <= 'Z') ||
                          (u >= 'a' && u <= 'z') || c == '.' || c == '_' || c == '-';
        out += keep ? c : '_';
    }
    size_t begin = 0, end = out.size();
    while (begin < end && (out[begin] == '.' || out[begin] == '_' || out[begin] == '-')) ++begin;
    while (end > begin && (out[end - 1] == '.' || out[end - 1] == '_' || out[end - 1] == '-')) --end;
    return out.substr(begin, end - begin);
}

std::string local_stamp(std::chrono::system_clock::time_point when) {
    using namespace std::chrono;
    const auto seconds_floor = floor<seconds>(when);
    const auto milliseconds_part = duration_cast<milliseconds>(when - seconds_floor).count();
    const std::time_t raw = system_clock::to_time_t(seconds_floor);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &raw);
#else
    localtime_r(&raw, &local);
#endif
    char text[40];
    std::snprintf(text, sizeof text, "%04d%02d%02d-%02d%02d%02d-%03d",
                  local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                  local.tm_hour, local.tm_min, local.tm_sec,
                  static_cast<int>(milliseconds_part));
    return text;
}

std::string iso_local_time(std::chrono::system_clock::time_point when) {
    using namespace std::chrono;
    const auto seconds_floor = floor<seconds>(when);
    const auto milliseconds_part = duration_cast<milliseconds>(when - seconds_floor).count();
    const std::time_t raw = system_clock::to_time_t(seconds_floor);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &raw);
#else
    localtime_r(&raw, &local);
#endif
    char text[48];
    std::snprintf(text, sizeof text, "%04d-%02d-%02dT%02d:%02d:%02d.%03d",
                  local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                  local.tm_hour, local.tm_min, local.tm_sec,
                  static_cast<int>(milliseconds_part));
    return text;
}

bool create_exclusive(const std::string& path) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                           _S_IREAD | _S_IWRITE);
    if (fd < 0) return false;
    ::_close(fd);
#else
    const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) return false;
    ::close(fd);
#endif
    return true;
}

std::string json_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    out << '"';
    return out.str();
}

int64_t relative_ns(uint64_t timestamp, uint64_t trigger) {
    if (timestamp >= trigger) {
        const uint64_t delta = timestamp - trigger;
        return delta > static_cast<uint64_t>(INT64_MAX) ? INT64_MAX : static_cast<int64_t>(delta);
    }
    const uint64_t delta = trigger - timestamp;
    return delta > static_cast<uint64_t>(INT64_MAX) ? INT64_MIN : -static_cast<int64_t>(delta);
}

void write_optional(std::ostream& out, const std::optional<uint64_t>& value) {
    if (value) out << *value;
    else out << "null";
}

} // namespace

struct InteractivePerformanceCapture::PendingCapture {
    std::string final_path;
    std::string part_path;
    std::string title_id;
    std::string title_label;
    std::string revision;
    std::string wall_clock;
    uint64_t trigger_ns = 0;
    std::vector<ProcessSample> pre_samples;
    std::vector<ProcessSample> post_samples;
    std::vector<RendererTimingRecord> renderer;
    std::vector<ComputeTimingRecord> compute;
    size_t renderer_dropped = 0;
    size_t compute_dropped = 0;
};

InteractivePerformanceCapture::InteractivePerformanceCapture(CaptureConfig config)
    : config_(config) {
    if (!config_.sample_interval_ns) config_.sample_interval_ns = 1;
    if (!config_.pre_window_ns) config_.pre_window_ns = config_.sample_interval_ns;
    if (!config_.post_window_ns) config_.post_window_ns = config_.sample_interval_ns;
}

InteractivePerformanceCapture::~InteractivePerformanceCapture() = default;

bool InteractivePerformanceCapture::sample_due(uint64_t monotonic_ns) const {
    const uint64_t next = next_sample_ns_.load(std::memory_order_relaxed);
    return next == 0 || monotonic_ns >= next;
}

void InteractivePerformanceCapture::observe_sample(const ProcessSample& sample) {
    std::unique_ptr<PendingCapture> completed;
    {
        std::lock_guard lock(mutex_);
        const uint64_t next = next_sample_ns_.load(std::memory_order_relaxed);
        if (next && sample.monotonic_ns < next) return;
        const uint64_t until_next = UINT64_MAX - sample.monotonic_ns < config_.sample_interval_ns
            ? UINT64_MAX : sample.monotonic_ns + config_.sample_interval_ns;
        next_sample_ns_.store(until_next, std::memory_order_relaxed);

        ring_.push_back(sample);
        while (!ring_.empty() && sample.monotonic_ns >= ring_.front().monotonic_ns &&
               sample.monotonic_ns - ring_.front().monotonic_ns > config_.pre_window_ns)
            ring_.pop_front();

        if (pending_ && sample.monotonic_ns > pending_->trigger_ns)
            pending_->post_samples.push_back(sample);
        completed = finish_if_due_locked(sample.monotonic_ns);
    }
    if (completed) publish_completed(std::move(completed));
}

CaptureArmResult InteractivePerformanceCapture::arm(
    const std::string& directory, const std::string& title_id,
    const std::string& title_label, const std::string& revision,
    uint64_t monotonic_ns, std::chrono::system_clock::time_point wall_clock) {
    CaptureArmResult result;
    std::lock_guard lock(mutex_);
    result.index = ++arm_count_;
    if (pending_) {
        result.error = "a performance capture is already collecting its post-trigger window";
        return result;
    }

    std::string base_dir = directory.empty() ? std::string(".") : directory;
    while (base_dir.size() > 1 && (base_dir.back() == '/' || base_dir.back() == '\\'))
        base_dir.pop_back();
    std::error_code ec;
    std::filesystem::create_directories(base_dir, ec);

    std::string safe_title = sanitize_component(title_id);
    if (safe_title.empty()) safe_title = "notitle";
    const std::string stem = "perf_capture_" + safe_title + "_" + local_stamp(wall_clock);
    std::string final_path, part_path;
    bool reserved = false;
    for (unsigned collision = 0; collision < 1000; ++collision) {
        std::string candidate = stem;
        if (collision) candidate += "-" + std::to_string(collision + 1);
        final_path = base_dir + "/" + candidate + ".prperf";
        part_path = final_path + ".part";
        if (std::filesystem::exists(final_path, ec) && !ec) continue;
        errno = 0;
        if (create_exclusive(part_path)) {
            reserved = true;
            break;
        }
        if (errno == EEXIST) continue;
        result.error = "cannot reserve performance capture: " + std::string(std::strerror(errno));
        return result;
    }
    if (!reserved) {
        result.error = "could not find a free performance-capture name";
        return result;
    }

    pending_ = std::make_unique<PendingCapture>();
    pending_->final_path = std::move(final_path);
    pending_->part_path = std::move(part_path);
    pending_->title_id = title_id;
    pending_->title_label = title_label;
    pending_->revision = revision;
    pending_->wall_clock = iso_local_time(wall_clock);
    pending_->trigger_ns = monotonic_ns;
    const uint64_t cutoff = monotonic_ns > config_.pre_window_ns
        ? monotonic_ns - config_.pre_window_ns : 0;
    for (const ProcessSample& sample : ring_) {
        if (sample.monotonic_ns >= cutoff && sample.monotonic_ns <= monotonic_ns)
            pending_->pre_samples.push_back(sample);
    }
    detailed_active_.store(true, std::memory_order_relaxed);
    result.ok = true;
    result.pre_samples = pending_->pre_samples.size();
    result.post_seconds = static_cast<double>(config_.post_window_ns) / 1e9;
    return result;
}

void InteractivePerformanceCapture::record_renderer(RendererTimingRecord record) {
    if (!detailed_timing_active()) return;
    if (!record.monotonic_ns) record.monotonic_ns = monotonic_now_ns();
    std::lock_guard lock(mutex_);
    if (!pending_) return;
    if (pending_->renderer.size() < config_.max_renderer_records)
        pending_->renderer.push_back(record);
    else
        ++pending_->renderer_dropped;
}

void InteractivePerformanceCapture::record_compute(ComputeTimingRecord record) {
    if (!detailed_timing_active()) return;
    if (!record.monotonic_ns) record.monotonic_ns = monotonic_now_ns();
    std::lock_guard lock(mutex_);
    if (!pending_) return;
    if (pending_->compute.size() < config_.max_compute_records)
        pending_->compute.push_back(record);
    else
        ++pending_->compute_dropped;
}

std::unique_ptr<InteractivePerformanceCapture::PendingCapture>
InteractivePerformanceCapture::finish_if_due_locked(uint64_t monotonic_ns) {
    if (!pending_ || monotonic_ns < pending_->trigger_ns ||
        monotonic_ns - pending_->trigger_ns < config_.post_window_ns)
        return {};
    detailed_active_.store(false, std::memory_order_relaxed);
    return std::move(pending_);
}

void InteractivePerformanceCapture::publish_completed(std::unique_ptr<PendingCapture> capture) {
    CaptureOutcome result;
    result.path = capture->final_path;
    result.pre_samples = capture->pre_samples.size();
    result.post_samples = capture->post_samples.size();
    result.renderer_records = capture->renderer.size();
    result.compute_records = capture->compute.size();
    result.renderer_dropped = capture->renderer_dropped;
    result.compute_dropped = capture->compute_dropped;

    std::ofstream out(capture->part_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        result.error = "could not open the reserved temporary capture for writing";
    } else {
        const bool cpu_available = std::any_of(
            capture->pre_samples.begin(), capture->pre_samples.end(),
            [](const ProcessSample& sample) { return sample.process_cpu_ns.has_value(); }) ||
            std::any_of(capture->post_samples.begin(), capture->post_samples.end(),
            [](const ProcessSample& sample) { return sample.process_cpu_ns.has_value(); });
        const bool rss_available = std::any_of(
            capture->pre_samples.begin(), capture->pre_samples.end(),
            [](const ProcessSample& sample) { return sample.rss_bytes.has_value(); }) ||
            std::any_of(capture->post_samples.begin(), capture->post_samples.end(),
            [](const ProcessSample& sample) { return sample.rss_bytes.has_value(); });
        out << std::setprecision(10);
        out << "{\"type\":\"header\",\"format\":\"prosper-performance-capture\","
               "\"version\":1,\"title_id\":" << json_string(capture->title_id)
            << ",\"title\":" << json_string(capture->title_label)
            << ",\"revision\":" << json_string(capture->revision)
            << ",\"trigger_monotonic_ns\":" << capture->trigger_ns
            << ",\"trigger_local_time\":" << json_string(capture->wall_clock)
            << ",\"pre_window_ns\":" << config_.pre_window_ns
            << ",\"post_window_ns\":" << config_.post_window_ns
            << ",\"sample_interval_ns\":" << config_.sample_interval_ns
            << ",\"logical_cpus\":" << std::thread::hardware_concurrency()
            << ",\"process_cpu_available\":" << (cpu_available ? "true" : "false")
            << ",\"rss_available\":" << (rss_available ? "true" : "false") << "}\n";

        const auto write_sample = [&](const ProcessSample& sample, const char* phase) {
            out << "{\"type\":\"sample\",\"phase\":\"" << phase << "\",\"t_ns\":"
                << relative_ns(sample.monotonic_ns, capture->trigger_ns)
                << ",\"process_cpu_ns\":";
            write_optional(out, sample.process_cpu_ns);
            out << ",\"rss_bytes\":";
            write_optional(out, sample.rss_bytes);
            out << ",\"guest_presents\":" << sample.guest_presents
                << ",\"rendered_frames\":";
            write_optional(out, sample.rendered_frames);
            out << ",\"host_presented_frames\":" << sample.host_presented_frames << "}\n";
        };
        for (const ProcessSample& sample : capture->pre_samples) write_sample(sample, "pre");
        for (const ProcessSample& sample : capture->post_samples) write_sample(sample, "post");

        for (const RendererTimingRecord& record : capture->renderer) {
            out << "{\"type\":\"renderer\",\"t_ns\":"
                << relative_ns(record.monotonic_ns, capture->trigger_ns)
                << ",\"callbacks\":" << record.callbacks << ",\"draws\":" << record.draws
                << ",\"texture_bytes\":" << record.texture_bytes
                << ",\"buffer_bytes\":" << record.buffer_bytes
                << ",\"total_ms\":" << record.total_ms
                << ",\"prelude_ms\":" << record.prelude_ms
                << ",\"pass_ms\":" << record.pass_ms
                << ",\"build_resources_ms\":" << record.build_resources_ms
                << ",\"backend_ms\":" << record.backend_ms
                << ",\"output_copy_ms\":" << record.output_copy_ms
                << ",\"pass_head_ms\":" << record.pass_head_ms
                << ",\"pass_loop_ms\":" << record.pass_loop_ms
                << ",\"pass_pre_ms\":" << record.pass_pre_ms
                << ",\"pass_post_ms\":" << record.pass_post_ms
                << ",\"pass_tail_ms\":" << record.pass_tail_ms
                << ",\"post_stats_ms\":" << record.post_stats_ms
                << ",\"post_slot0_ms\":" << record.post_slot0_ms
                << ",\"post_mrt_ms\":" << record.post_mrt_ms
                << ",\"post_rest_ms\":" << record.post_rest_ms
                << ",\"resolve_stall_ms\":" << record.resolve_stall_ms
                << ",\"resolve_read_ms\":" << record.resolve_read_ms
                << ",\"resolve_copy_stall_ms\":" << record.resolve_copy_stall_ms
                << ",\"resolve_copy_ms\":" << record.resolve_copy_ms
                << ",\"resolve_count\":" << record.resolve_count
                << ",\"resolve_read_count\":" << record.resolve_read_count
                << ",\"resolve_bytes\":" << record.resolve_bytes
                << ",\"gpu_wait_ms\":" << record.gpu_wait_ms
                << ",\"gpu_timestamp_samples\":" << record.gpu_timestamp_samples
                << ",\"gpu_device_ms\":" << record.gpu_device_ms
                << ",\"readback_ms\":" << record.readback_ms
                << ",\"backend_target_ms\":" << record.backend_target_ms
                << ",\"backend_draw_setup_ms\":" << record.backend_draw_setup_ms
                << ",\"backend_record_upload_ms\":" << record.backend_record_upload_ms
                << ",\"backend_cleanup_ms\":" << record.backend_cleanup_ms
                << ",\"backend_setup_shader_ms\":" << record.backend_setup_shader_ms
                << ",\"backend_setup_fixed_ms\":" << record.backend_setup_fixed_ms
                << ",\"setup_resources_ms\":" << record.setup_resources_ms
                << ",\"backend_setup_pipeline_ms\":" << record.backend_setup_pipeline_ms
                << ",\"backend_pipeline_refs\":" << record.backend_pipeline_refs
                << ",\"backend_pipeline_hits\":" << record.backend_pipeline_hits
                << ",\"backend_pipeline_misses\":" << record.backend_pipeline_misses
                << ",\"backend_pipeline_bypasses\":" << record.backend_pipeline_bypasses
                << ",\"backend_pipeline_entries\":" << record.backend_pipeline_entries
                << ",\"backend_pipeline_evictions\":" << record.backend_pipeline_evictions
                << ",\"frontend_texture_ms\":" << record.frontend_texture_ms
                << ",\"frontend_buffer_ms\":" << record.frontend_buffer_ms
                << ",\"frontend_tex_rtt_ms\":" << record.frontend_tex_rtt_ms
                << ",\"frontend_tex_compute_ms\":" << record.frontend_tex_compute_ms
                << ",\"frontend_tex_local_ms\":" << record.frontend_tex_local_ms
                << ",\"frontend_tex_persist_hit_ms\":" << record.frontend_tex_persist_hit_ms
                << ",\"frontend_tex_persist_reuse_ms\":" << record.frontend_tex_persist_reuse_ms
                << ",\"frontend_tex_persist_miss_ms\":" << record.frontend_tex_persist_miss_ms
                << ",\"frontend_tex_persist_invalid_ms\":"
                << record.frontend_tex_persist_invalid_ms
                << ",\"frontend_tex_persist_invalid_n\":"
                << record.frontend_tex_persist_invalid_n
                << ",\"frontend_tex_other_n\":" << record.frontend_tex_other_n
                << ",\"frontend_tex_other_slowest_ms\":"
                << record.frontend_tex_other_slowest_ms
                << ",\"frontend_tex_other_addr\":" << record.frontend_tex_other_addr
                << ",\"frontend_tex_other_source_bytes\":"
                << record.frontend_tex_other_source_bytes
                << ",\"frontend_tex_other_width\":" << record.frontend_tex_other_width
                << ",\"frontend_tex_other_height\":" << record.frontend_tex_other_height
                << ",\"frontend_tex_other_depth\":" << record.frontend_tex_other_depth
                << ",\"frontend_tex_other_format\":" << record.frontend_tex_other_format
                << ",\"frontend_tex_other_components\":"
                << record.frontend_tex_other_components
                << ",\"frontend_tex_other_tile_mode\":"
                << record.frontend_tex_other_tile_mode
                << ",\"frontend_tex_other_img_dim\":" << record.frontend_tex_other_img_dim
                << ",\"frontend_tex_other_class\":" << record.frontend_tex_other_class
                << ",\"frontend_tex_other_compute_candidate\":"
                << (record.frontend_tex_other_compute_candidate ? "true" : "false")
                << ",\"frontend_tex_other_persistent_candidate\":"
                << (record.frontend_tex_other_persistent_candidate ? "true" : "false")
                << ",\"frontend_tex_other_compressed\":"
                << (record.frontend_tex_other_compressed ? "true" : "false")
                << ",\"frontend_tex_other_depth_compare\":"
                << (record.frontend_tex_other_depth_compare ? "true" : "false")
                << ",\"frontend_tex_other_host_backed\":"
                << (record.frontend_tex_other_host_backed ? "true" : "false")
                << ",\"frontend_build_draw_ms\":" << record.frontend_build_draw_ms
                << ",\"frontend_validate_ms\":" << record.frontend_validate_ms
                << ",\"frontend_poison_ms\":" << record.frontend_poison_ms
                << ",\"frontend_indices_ms\":" << record.frontend_indices_ms
                << ",\"frontend_reflect_ms\":" << record.frontend_reflect_ms
                << ",\"res_texture_ms\":" << record.res_texture_ms
                << ",\"res_buffer_ms\":" << record.res_buffer_ms
                << ",\"res_buffer_copy_ms\":" << record.res_buffer_copy_ms
                << ",\"res_buffer_create_ms\":" << record.res_buffer_create_ms
                << ",\"res_buffer_index_find_ms\":" << record.res_buffer_index_find_ms
                << ",\"res_buffer_index_insert_ms\":" << record.res_buffer_index_insert_ms
                << ",\"res_buffer_hash_ms\":" << record.res_buffer_hash_ms
                << ",\"res_descriptor_ms\":" << record.res_descriptor_ms << "}\n";
        }
        for (const ComputeTimingRecord& record : capture->compute) {
            out << "{\"type\":\"compute\",\"t_ns\":"
                << relative_ns(record.monotonic_ns, capture->trigger_ns)
                << ",\"dispatches\":" << record.dispatches
                << ",\"cpu_fast_total\":" << record.cpu_fast_total
                << ",\"program_addr\":";
            write_optional(out, record.program_addr);
            out << ",\"program_hash\":";
            write_optional(out, record.program_hash);
            out << ",\"total_ms\":" << record.total_ms
                << ",\"gpu_timestamp_samples\":" << record.gpu_timestamp_samples
                << ",\"gpu_device_ms\":" << record.gpu_device_ms
                << ",\"gpu_shader_ms\":" << record.gpu_shader_ms
                << ",\"gpu_pre_ms\":" << record.gpu_pre_ms
                << ",\"gpu_storage_copy_ms\":" << record.gpu_storage_copy_ms
                << ",\"gpu_compare_ms\":" << record.gpu_compare_ms
                << ",\"gpu_restore_ms\":" << record.gpu_restore_ms
                << ",\"setup_ms\":" << record.setup_ms
                << ",\"pipeline_ms\":" << record.pipeline_ms
                << ",\"dispatch_wait_ms\":" << record.dispatch_wait_ms
                << ",\"writeback_ms\":" << record.writeback_ms
                << ",\"cleanup_ms\":" << record.cleanup_ms << "}\n";
        }
        out << "{\"type\":\"footer\",\"complete\":true,\"pre_samples\":"
            << capture->pre_samples.size() << ",\"post_samples\":" << capture->post_samples.size()
            << ",\"renderer_records\":" << capture->renderer.size()
            << ",\"compute_records\":" << capture->compute.size()
            << ",\"renderer_dropped\":" << capture->renderer_dropped
            << ",\"compute_dropped\":" << capture->compute_dropped << "}\n";
        out.flush();
        if (!out.good()) result.error = "writing the performance capture failed";
        out.close();
        if (result.error.empty()) {
            std::error_code rename_error;
            std::filesystem::rename(capture->part_path, capture->final_path, rename_error);
            if (rename_error)
                result.error = "could not atomically install the completed capture: " +
                               rename_error.message();
        }
    }
    result.ok = result.error.empty();
    if (!result.ok) {
        std::error_code ignored;
        std::filesystem::remove(capture->part_path, ignored);
        result.path.clear();
    }
    std::lock_guard lock(mutex_);
    outcome_ = std::move(result);
}

bool InteractivePerformanceCapture::take_outcome(CaptureOutcome& outcome) {
    std::lock_guard lock(mutex_);
    if (!outcome_) return false;
    outcome = std::move(*outcome_);
    outcome_.reset();
    return true;
}

void InteractivePerformanceCapture::cancel() {
    std::string part;
    {
        std::lock_guard lock(mutex_);
        detailed_active_.store(false, std::memory_order_relaxed);
        if (pending_) part = pending_->part_path;
        pending_.reset();
    }
    if (!part.empty()) {
        std::error_code ignored;
        std::filesystem::remove(part, ignored);
    }
}

InteractivePerformanceCapture& interactive_performance_capture() {
    static InteractivePerformanceCapture capture;
    return capture;
}

uint64_t monotonic_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

ProcessSample collect_process_sample(uint64_t monotonic_ns, uint64_t guest_presents,
                                     std::optional<uint64_t> rendered_frames,
                                     uint64_t host_presented_frames) {
    ProcessSample sample;
    sample.monotonic_ns = monotonic_ns;
    sample.guest_presents = guest_presents;
    sample.rendered_frames = rendered_frames;
    sample.host_presented_frames = host_presented_frames;

#if defined(_WIN32)
    // MSVC's std::clock measures elapsed wall time, not process CPU time. GetProcessTimes exposes
    // the kernel + user total in 100 ns ticks and therefore preserves the report's "CPU cores"
    // meaning on Windows instead of quietly reporting ~1.0 for an idle process.
    FILETIME created{}, exited{}, kernel{}, user{};
    if (::GetProcessTimes(::GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        ULARGE_INTEGER kernel_ticks{}, user_ticks{};
        kernel_ticks.LowPart = kernel.dwLowDateTime;
        kernel_ticks.HighPart = kernel.dwHighDateTime;
        user_ticks.LowPart = user.dwLowDateTime;
        user_ticks.HighPart = user.dwHighDateTime;
        if (user_ticks.QuadPart <= UINT64_MAX - kernel_ticks.QuadPart) {
            const uint64_t ticks = kernel_ticks.QuadPart + user_ticks.QuadPart;
            if (ticks <= UINT64_MAX / 100) sample.process_cpu_ns = ticks * 100;
        }
    }
#else
    const std::clock_t cpu = std::clock();
    if (cpu != static_cast<std::clock_t>(-1)) {
        const long double ns = static_cast<long double>(cpu) * 1'000'000'000.0L / CLOCKS_PER_SEC;
        if (ns >= 0 && ns <= static_cast<long double>(UINT64_MAX))
            sample.process_cpu_ns = static_cast<uint64_t>(ns);
    }
#endif

#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0, resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        (void)total_pages;
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0 && resident_pages <= UINT64_MAX / static_cast<uint64_t>(page_size))
            sample.rss_bytes = resident_pages * static_cast<uint64_t>(page_size);
    }
#endif
    return sample;
}

const char* build_revision() {
    return ::prosper::embedded_build_revision();
}

} // namespace prosper::perf

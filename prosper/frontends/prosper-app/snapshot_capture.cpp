#include "snapshot_capture.hpp"
#include "build_revision.hpp"
#include "shared/capture/capture_png.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <limits>

namespace prosper::frontend {
namespace {
bool positive_number(const char* text, double& number) {
    char* end = nullptr;
    errno = 0;
    number = std::strtod(text, &end);
    return !errno && end != text && !*end && std::isfinite(number) && number > 0;
}
const char* environment(const char* name) {
    const char* result = std::getenv(name);
    return result ? result : "";
}
} // namespace

bool SnapshotCaptureRequest::argument(const std::string& option, int& index, int argc, char** argv,
                                      std::string& error) {
    if (option == "--snapshot-capture") {
        enabled = true;
        return true;
    }
    if (option == "--snapshot-require-composited-frame") {
        require_composited = true;
        return true;
    }
    if (option != "--snapshot-seconds" && option != "--snapshot-count" &&
        option != "--snapshot-timeout" && option != "--snapshot-out" &&
        option != "--snapshot-manifest") {
        error = "unknown snapshot capture option: " + option;
        return false;
    }
    if (++index >= argc) {
        error = option + " needs a value";
        return false;
    }
    if (option == "--snapshot-out") {
        output = argv[index];
        return true;
    }
    if (option == "--snapshot-manifest") {
        manifest = argv[index];
        return true;
    }
    double value = 0;
    if (!positive_number(argv[index], value)) {
        error = option + " requires a finite positive value";
        return false;
    }
    if (option == "--snapshot-seconds") seconds = value;
    if (option == "--snapshot-timeout") timeout = value;
    if (option == "--snapshot-count") {
        if (value != std::floor(value) || value > std::numeric_limits<int>::max()) {
            error = "--snapshot-count requires a positive integer";
            return false;
        }
        count = static_cast<int>(value);
    }
    return true;
}

bool SnapshotCaptureRequest::valid(std::string& error) const {
    if (!enabled) {
        if (seconds || timeout || count || !output.empty() || !manifest.empty() ||
            require_composited)
            error = "snapshot options require --snapshot-capture";
        return error.empty();
    }
    if (!std::isfinite(seconds) || seconds <= 0 || !std::isfinite(timeout) || timeout <= 0 ||
        count <= 0 || output.empty() || manifest.empty()) {
        error = "snapshot capture requires seconds, count, timeout, output and manifest";
        return false;
    }
    return true;
}

bool PresentedSnapshotFrame::retain(const gpu::PresentFrameLease& frame, bool presented) {
    if (!presented) return false;
    if (!frame.rgba || !frame.width || !frame.height ||
        (frame.source != gpu::PresentSource::Rendered &&
         frame.source != gpu::PresentSource::GuestScanout) ||
        static_cast<uint64_t>(frame.width) * frame.height >
            std::numeric_limits<size_t>::max() / 4 ||
        frame.rgba->size() != static_cast<size_t>(frame.width) * frame.height * 4) {
        frame_ = {};
        return false;
    }
    frame_ = frame;
    return true;
}

bool SnapshotCapture::line(const std::string& text) {
    manifest_ << text << '\n';
    manifest_.flush();
    if (manifest_) return true;
    failed_ = true;
    std::fprintf(stderr, "[snapshot] capture manifest write failed\n");
    return false;
}

bool SnapshotCapture::start(const std::string& title, std::string& error) {
    if (!enabled()) return true;
#ifndef PROSPER_SNAPSHOT_PNG
    error = "this app was built without PNG snapshot capture support";
    return false;
#endif
    std::error_code ec;
    std::filesystem::create_directories(request.output, ec);
    if (ec) {
        error = "snapshot output directory: " + ec.message();
        return false;
    }
    manifest_.open(request.manifest, std::ios::out | std::ios::trunc);
    if (!manifest_) {
        error = "cannot open snapshot capture manifest";
        return false;
    }
    route_ = environment("PROSPER_PAD_SCRIPT");
    screenshot::CaptureRunConfig config;
    config.title = title;
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char timestamp[32]{};
    std::strftime(timestamp, sizeof timestamp, "%Y-%m-%dT%H:%M:%SZ", &utc);
    config.timestamp = timestamp;
    config.output_dir = request.output;
    config.input_route = route_;
    config.time_mode = true;
    config.seconds = request.seconds;
    config.requested = request.count;
    config.require_composited_frame = request.require_composited;
    config.render_every = environment("PROSPER_RENDER_EVERY");
    config.render_every_for_ms = environment("PROSPER_RENDER_EVERY_FOR_MS");
    config.render_scale = environment("PROSPER_RENDER_SCALE");
    config.render_target_dim = environment("PROSPER_RENDER_TARGET_DIM");
    config.render_resource_dim = environment("PROSPER_RENDER_RESOURCE_DIM");
    // Keep the requested wall-clock evidence window even if the primary guest thread ends.
    config.stop_after_guest_fault = false;
    if (!line(screenshot::manifest_run_json(config))) {
        error = "cannot write capture run";
        return false;
    }
    if (!line(std::string("{\"type\":\"frontend\",\"name\":\"prosper-app\",\"present_path\":"
                          "\"cpu\",\"source_identity\":\"") +
              screenshot::json_escape(prosper::embedded_build_source_identity()) +
              "\",\"timeout_seconds\":" + std::to_string(request.timeout) + "}")) {
        error = "cannot write capture producer identity";
        return false;
    }
    conditions_.harness = "prosper-app";
    conditions_.present_path = "cpu_publication";
    conditions_.flip_pace_fps = environment("PROSPER_FLIP_PACE_FPS");
#if defined(_WIN32)
    conditions_.os = "windows";
#elif defined(__APPLE__)
    conditions_.os = "macos";
#else
    conditions_.os = "linux";
#endif
    clock_ = SnapshotCaptureClock(request.seconds, request.timeout, request.count);
    start_ = std::chrono::steady_clock::now();
    started_ = true;
    return true;
}

void SnapshotCapture::retain(const gpu::PresentFrameLease& frame, bool presented) {
    if (started_ && presented && !frame_.retain(frame, true)) {
        failed_ = true;
        std::fprintf(stderr, "[snapshot] presented frame has unavailable capture provenance\n");
    }
}

bool SnapshotCapture::sample(double elapsed) {
#ifdef PROSPER_SNAPSHOT_PNG
    const auto& frame = frame_.get();
    std::vector<uint8_t> pixels(*frame.rgba);
    const auto source = frame.source == gpu::PresentSource::Rendered
                            ? screenshot::CaptureSource::Rendered
                            : screenshot::CaptureSource::GuestScanout;
    screenshot::normalize_capture_rgba(source, pixels);
    const std::string png = (std::filesystem::path(request.output) /
                             ("sample_" + std::to_string(clock_.saved()) + ".png"))
                                .string();
    std::string error;
    if (!screenshot::write_png(png.c_str(), pixels.data(), frame.width, frame.height, error)) {
        std::fprintf(stderr, "[snapshot] PNG write failed: %s\n", error.c_str());
        return false;
    }
    screenshot::CaptureObservation observation;
    observation.source = source;
    observation.source_seq = frame.frame_seq;
    observation.frame_seq = frame.frame_seq;
    observation.present_count = frame.guest_present_count;
    observation.front_index = frame.front_index;
    observation.width = frame.width;
    observation.height = frame.height;
    observation.elapsed_seconds = elapsed;
    observation.pixel_crc32 = screenshot::crc32_bytes(pixels.data(), pixels.size());
    const auto content = screenshot::measure_pixel_content_rgba(pixels);
    observation.distinct_rgb_colors = content.distinct_colors;
    observation.nonblack_rgb_pixels = content.nonblack_pixels;
    const auto perceptual = screenshot::perceptual_hashes_rgba(pixels, frame.width, frame.height);
    observation.average_hash = perceptual.average;
    observation.difference_hash = perceptual.difference;
    observation.luma16x9 = screenshot::perceptual_luma16x9_rgba(pixels, frame.width, frame.height);
    const auto rate = gpu::present_rate_snapshot();
    observation.published_frames = rate.published;
    observation.distinct_frames = rate.distinct;
    const auto classification = tracker_.observe(observation, pixels);
    if (!line(screenshot::manifest_sample_json(clock_.saved(), png, observation, classification,
                                               route_)))
        return false;
    clock_.sampled(elapsed);
    return true;
#else
    (void)elapsed;
    return false;
#endif
}

bool SnapshotCapture::poll(bool visible) {
    if (!started_) return false;
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    return poll_elapsed(elapsed, visible);
}

bool SnapshotCapture::poll_elapsed(double elapsed, bool visible) {
    if (!started_) return false;
    last_elapsed_ = elapsed;
    // A delayed UI iteration cannot turn a publication after the deadline into a satisfied run.
    if (clock_.timed_out(elapsed)) return true;
    if (!visible) {
        std::fprintf(stderr, "[snapshot] normal app window is not visible; capture refused\n");
        failed_ = true;
    }
    if (!failed_ && clock_.due(elapsed) && frame_.available()) failed_ = !sample(elapsed);
    return failed_ || clock_.complete() || clock_.timed_out(elapsed);
}

int SnapshotCapture::finish() {
    if (!enabled()) return 0;
    if (finished_) return verdict_;
    finished_ = true;
    if (!started_) return verdict_ = 1;
    const auto guest = screenshot::read_guest_outcome();
    failed_ |= !clock_.complete() || (request.require_composited && !tracker_.rendered_samples());
    const auto stop = clock_.complete()                 ? screenshot::SamplingStop::RequestSatisfied
                      : clock_.timed_out(last_elapsed_) ? screenshot::SamplingStop::Timeout
                                                        : screenshot::SamplingStop::FrontendStopped;
    const auto rate = gpu::frame_rate_since_first_publication(gpu::present_rate_snapshot());
    // Device identity is deliberately unknown here; this adapter authenticates CPU publications,
    // and does not claim native GPU presentation or authority for performance comparisons.
    failed_ |= !line(screenshot::manifest_conditions_json(conditions_));
    const auto final_verdict = screenshot::decide_run_verdict(failed_, guest, false);
    const bool written = line(screenshot::manifest_summary_json(
        clock_.saved(), request.count, stop, tracker_, final_verdict, guest, false, rate));
    manifest_.close();
    return verdict_ = written && !manifest_.fail() ? final_verdict.exit_code : 1;
}
}   // namespace prosper::frontend

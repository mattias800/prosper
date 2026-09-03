#include "shared/perf/performance_capture.hpp"
#include "shared/perf/performance_timing_gate.hpp"
#include "shared/perf/performance_timing_policy.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {
int failures = 0;
int checks = 0;

void check(bool condition, const char* name) {
    ++checks;
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << name << '\n';
}

size_t count_text(const std::string& text, const std::string& needle) {
    size_t count = 0, at = 0;
    while ((at = text.find(needle, at)) != std::string::npos) {
        ++count;
        at += needle.size();
    }
    return count;
}

prosper::perf::ProcessSample sample(uint64_t at, uint64_t counter) {
    prosper::perf::ProcessSample out;
    out.monotonic_ns = at;
    out.process_cpu_ns = counter * 70;
    out.rss_bytes = 4096 + counter;
    out.guest_presents = counter * 3;
    out.rendered_frames = counter * 2;
    out.host_presented_frames = counter;
    return out;
}
} // namespace

int main() {
    namespace fs = std::filesystem;
    using prosper::perf::CaptureConfig;
    using prosper::perf::InteractivePerformanceCapture;

    const fs::path dir = "performance_capture_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    check(!ec, "scratch directory is available");

    struct SplitTiming { unsigned spans = 0; };
    SplitTiming split{1};
    prosper::frontend::reset_performance_timing_after_span(split, true, false);
    check(split.spans == 1,
          "split-submit timing retains earlier non-final spans until the final callback");
    prosper::frontend::reset_performance_timing_after_span(split, true, true);
    check(split.spans == 0, "split-submit timing resets after its final callback");
    check(!prosper::frontend::interactive_performance_timing(),
          "interactive backend timing starts disabled on the caller thread");
    {
        prosper::frontend::ScopedInteractivePerformanceTiming outer(true);
        check(prosper::frontend::interactive_performance_timing(),
              "interactive backend timing is enabled inside the live callback scope");
        bool other_thread_timing = true;
        std::thread ordinary_caller([&] {
            other_thread_timing = prosper::frontend::interactive_performance_timing();
        });
        ordinary_caller.join();
        check(!other_thread_timing,
              "F8 timing cannot enable an unrelated render_runner caller on another thread");
        {
            prosper::frontend::ScopedInteractivePerformanceTiming nested(false);
            check(!prosper::frontend::interactive_performance_timing(),
                  "a nested timing scope can temporarily restore the ordinary caller policy");
        }
        check(prosper::frontend::interactive_performance_timing(),
              "a nested timing scope restores its prior live-callback state");
    }
    check(!prosper::frontend::interactive_performance_timing(),
          "interactive backend timing cannot leak beyond the live callback scope");
    const auto f8_only = prosper::frontend::performance_timing_mode(false, true);
    check(f8_only.measure && !f8_only.log,
          "F8-only timing measures structured data without enabling periodic stderr logs");
    const auto environment_timing = prosper::frontend::performance_timing_mode(true, false);
    check(environment_timing.measure && environment_timing.log,
          "PROSPER_RENDER_TIMING enables both measurement and its requested logs");

    CaptureConfig config;
    config.pre_window_ns = 500;
    config.post_window_ns = 300;
    config.sample_interval_ns = 100;
    config.max_renderer_records = 2;
    config.max_compute_records = 1;
    InteractivePerformanceCapture capture(config);

    // Six samples fill the exact inclusive 500 ns pre-trigger window. A test that only asserts the
    // file exists would pass if the ring never ran; the exact serialized population is the mechanism.
    for (uint64_t i = 0; i <= 5; ++i) capture.observe_sample(sample(i * 100, i));
    check(!capture.sample_due(550), "fixed-rate ring suppresses an early process sample");
    check(capture.sample_due(600), "fixed-rate ring requests its next process sample");

    const auto wall = std::chrono::system_clock::time_point(std::chrono::milliseconds(1'722'517'353'471LL));
    const auto armed = capture.arm(dir.string(), "PPSA30140", "Syberia \"Remastered\"",
                                   "test-revision", 500, wall);
    check(armed.ok, "F8 arm reserves a temporary capture");
    check(armed.pre_samples == 6, "F8 arm freezes the actual pre-trigger ring");
    check(capture.detailed_timing_active(), "F8 arm enables post-trigger detailed timing");

    size_t part_files = 0, final_files = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        part_files += entry.path().extension() == ".part";
        final_files += entry.path().extension() == ".prperf";
    }
    check(part_files == 1 && final_files == 0,
          "an active capture has only a private .part, never a partial final artifact");

    const auto refused = capture.arm(dir.string(), "PPSA30140", "Syberia", "test-revision",
                                     501, wall);
    check(!refused.ok && refused.error.find("already") != std::string::npos,
          "a second F8 press cannot replace an in-flight capture silently");

    for (uint64_t i = 0; i < 3; ++i) {
        prosper::perf::RendererTimingRecord record;
        record.monotonic_ns = 501 + i;
        record.callbacks = 1;
        record.draws = 100 + i;
        record.total_ms = 20 + i;
        record.pass_loop_ms = 7 + i;
        record.frontend_tex_persist_miss_ms = 2 + i;
        record.frontend_tex_persist_invalid_ms = 4 + i;
        record.frontend_tex_persist_invalid_n = 5 + i;
        record.frontend_tex_other_n = 6 + i;
        record.frontend_tex_other_slowest_ms = 1.5 + i;
        record.frontend_tex_other_addr = 0x2046960000ull + i * 0x1000;
        record.frontend_tex_other_width = 2560;
        record.frontend_tex_other_height = 1440;
        record.frontend_tex_other_depth = 1;
        record.frontend_tex_other_format = 1;
        record.frontend_tex_other_components = 1;
        record.frontend_tex_other_tile_mode = 24;
        record.frontend_tex_other_compute_candidate = true;
        record.resolve_read_count = 3 + i;
        record.setup_resources_ms = 10 + i;
        record.gpu_timestamp_samples = 2 + i;
        capture.record_renderer(record);
    }
    for (uint64_t i = 0; i < 2; ++i) {
        prosper::perf::ComputeTimingRecord record;
        record.monotonic_ns = 504 + i;
        record.dispatches = 7 + i;
        if (i == 0) {
            record.program_addr = 0x1234567800ull;
            record.program_hash = 0xfedcba9876543210ull;
        }
        record.total_ms = 3 + i;
        capture.record_compute(record);
    }

    capture.observe_sample(sample(600, 6));
    auto unavailable_render_count = sample(700, 7);
    unavailable_render_count.rendered_frames.reset();
    capture.observe_sample(unavailable_render_count);
    check(capture.detailed_timing_active(), "detailed timing stays active before the post window ends");
    capture.observe_sample(sample(800, 8));
    check(!capture.detailed_timing_active(), "the bounded post window disables detailed timing");

    prosper::perf::CaptureOutcome outcome;
    check(capture.take_outcome(outcome) && outcome.ok,
          "a complete post window publishes one successful outcome");
    check(outcome.pre_samples == 6 && outcome.post_samples == 3,
          "outcome proves both pre-trigger ring and post-trigger sampler ran");
    check(outcome.renderer_records == 2 && outcome.renderer_dropped == 1,
          "renderer detail cap retains its limit and accounts for every dropped record");
    check(outcome.compute_records == 1 && outcome.compute_dropped == 1,
          "compute detail cap retains its limit and accounts for every dropped record");
    check(fs::exists(outcome.path) && !fs::exists(outcome.path + ".part"),
          "completion atomically replaces the temporary file with .prperf");

    std::ifstream input(outcome.path, std::ios::binary);
    std::ostringstream bytes;
    bytes << input.rdbuf();
    const std::string text = bytes.str();
    check(text.find("\"format\":\"prosper-performance-capture\",\"version\":1") != std::string::npos,
          "capture identifies its format and version");
    check(text.find("Syberia \\\"Remastered\\\"") != std::string::npos,
          "capture metadata is JSON escaped");
    check(count_text(text, "\"type\":\"sample\",\"phase\":\"pre\"") == 6,
          "serialized artifact contains the exact pre-trigger ring population");
    check(count_text(text, "\"type\":\"sample\",\"phase\":\"post\"") == 3,
          "serialized artifact contains the exact post-trigger population");
    check(count_text(text, "\"rendered_frames\":null") == 1,
          "an unavailable rendered-frame population serializes as JSON null");
    check(count_text(text, "\"type\":\"renderer\"") == 2 &&
          count_text(text, "\"type\":\"compute\"") == 1,
          "serialized detail counts match the bounded retained records");
    check(text.find("\"gpu_timestamp_samples\":2") != std::string::npos,
          "renderer records serialize the GPU timestamp availability discriminator");
    check(text.find("\"pass_loop_ms\":7") != std::string::npos &&
          text.find("\"frontend_tex_persist_miss_ms\":2") != std::string::npos &&
          // The invalidation class and the unclassified COUNT. Without both in the artifact the
          // offline report cannot separate "the cache is missing" from "the content really changed",
          // and cannot say whether a millisecond residual has references behind it.
          text.find("\"frontend_tex_persist_invalid_ms\":4") != std::string::npos &&
          text.find("\"frontend_tex_persist_invalid_n\":5") != std::string::npos &&
          text.find("\"frontend_tex_other_n\":6") != std::string::npos &&
          text.find("\"frontend_tex_other_addr\":138623188992") != std::string::npos &&
          text.find("\"frontend_tex_other_compute_candidate\":true") != std::string::npos &&
          text.find("\"resolve_read_count\":3") != std::string::npos,
          "renderer records serialize the frontend phase, cache-class, residual witness, and resolve diagnostics");
    check(text.find("\"program_addr\":78187493376") != std::string::npos &&
          text.find("\"program_hash\":18364758544493064720") != std::string::npos,
          "compute records serialize the bounded program identity");
    check(text.find("\"renderer_dropped\":1,\"compute_dropped\":1") != std::string::npos,
          "footer makes detail truncation fail-visible");
    check(text.find("\"type\":\"footer\",\"complete\":true") != std::string::npos,
          "final artifact carries an explicit completeness marker");

    // The same title and wall-clock millisecond must not overwrite an earlier capture. The live
    // algorithm claims `.part` exclusively, so this is a syscall property rather than check-then-write.
    capture.observe_sample(sample(900, 9));
    const auto collision = capture.arm(dir.string(), "PPSA30140", "Syberia", "test-revision",
                                       900, wall);
    check(collision.ok, "a same-stamp capture advances to a collision suffix");
    capture.observe_sample(sample(1200, 12));
    prosper::perf::CaptureOutcome second;
    check(capture.take_outcome(second) && second.ok && second.path != outcome.path,
          "a same-stamp capture publishes a distinct final path");
    check(fs::exists(outcome.path) && fs::exists(second.path),
          "collision handling preserves both completed captures");

    // Graceful app exit removes only the unfinished private part and never creates a short final.
    capture.observe_sample(sample(1300, 13));
    const auto cancel_arm = capture.arm(dir.string(), "PPSA30140", "Syberia", "test-revision",
                                        1300, wall + std::chrono::milliseconds(1));
    check(cancel_arm.ok, "cancellation control arms");
    capture.cancel();
    check(!capture.detailed_timing_active(), "cancellation disables detailed timing");
    part_files = 0;
    for (const auto& entry : fs::directory_iterator(dir))
        part_files += entry.path().extension() == ".part";
    check(part_files == 0, "graceful cancellation removes the unfinished private part");

    fs::remove_all(dir, ec);
    std::cout << (failures ? "FAIL" : "PASS") << ": " << checks << " checks, "
              << failures << " failures\n";
    return failures ? 1 : 0;
}

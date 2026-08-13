#include "../src/gpu/gpu_timeline.hpp"
#include "../src/gpu/gpu_capture_bundle.hpp"
#include "../src/gpu/pm4_registers.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "test_scratch.h"
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

static bool write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

static void set_test_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

alignas(256) static const uint32_t kSelectorVs[] = {0xBF810000u};
alignas(256) static const uint32_t kSelectorPs[] = {0xBF810000u};
alignas(256) static const uint32_t kWrongSelectorVs[] = {0xBF810000u};
alignas(256) static const uint32_t kWrongSelectorPs[] = {0xBF810000u};
alignas(256) static const uint32_t kSelectorCompute[] = {0xBF810000u};
alignas(256) static const uint32_t kWrongSelectorCompute[] = {0xBF810000u};

static std::string program_address(const uint32_t* program) {
    char value[32];
    std::snprintf(value, sizeof(value), "0x%llx",
                  static_cast<unsigned long long>(reinterpret_cast<uint64_t>(program)));
    return value;
}

static std::shared_ptr<GpuState> selector_draw_state(
    uint32_t width, uint32_t height, const uint32_t* vertex, const uint32_t* fragment) {
    namespace P = prosper::agc::Pm4;
    auto state = std::make_shared<GpuState>();
    auto set_program = [&](uint32_t lo, uint32_t hi, const uint32_t* program) {
        const uint64_t address = reinterpret_cast<uint64_t>(program);
        state->sh[lo] = static_cast<uint32_t>(address >> 8);
        state->sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
    };
    set_program(P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, vertex);
    set_program(P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, fragment);
    state->cx[P::CB_COLOR0_ATTRIB2] = ((width - 1u) << 14) | (height - 1u);
    return state;
}

static GpuState selector_submit(
    std::initializer_list<std::shared_ptr<GpuState>> draw_states, uint64_t first_order,
    const uint32_t* compute = nullptr) {
    namespace P = prosper::agc::Pm4;
    GpuState state;
    uint64_t order = first_order;
    for (const auto& draw_state : draw_states) {
        GpuState::Draw draw{3};
        draw.state = draw_state;
        draw.command_order = order++;
        state.draws.push_back(std::move(draw));
    }
    if (compute) {
        auto dispatch_state = std::make_shared<GpuState>();
        const uint64_t address = reinterpret_cast<uint64_t>(compute);
        dispatch_state->sh[P::COMPUTE_PGM_LO] = static_cast<uint32_t>(address >> 8);
        dispatch_state->sh[P::COMPUTE_PGM_HI] =
            static_cast<uint32_t>((address >> 40) & 0xffu);
        GpuState::Dispatch dispatch;
        dispatch.state = std::move(dispatch_state);
        dispatch.command_order = order++;
        state.dispatches.push_back(std::move(dispatch));
    }
    state.command_order = order;
    return state;
}

static int run_selector_env_child(const std::string& mode) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = prosper_test::test_scratch_dir() /
        ("prosper-gpu-timeline-selector-" + mode + "-" + std::to_string(nonce));
    const std::string timeline_path = base.string() + ".prgtl";
    const std::string capture_path = base.string() + ".prgcap";
    const std::string bundle_path = base.string() + ".prgbundle";
    bool intermediate_ok = true;
    unsigned ds_snapshots = 0;
    GpuCaptureDsSeed expected_ds_seed;
    expected_ds_seed.depth_read_base = expected_ds_seed.depth_write_base = 0x310000;
    expected_ds_seed.htile_data_base = 0x300000;
    expected_ds_seed.width = 2;
    expected_ds_seed.height = 2;
    expected_ds_seed.format = GpuCaptureDsFormat::D32Float;
    expected_ds_seed.depth_valid = true;
    expected_ds_seed.depth.assign(16, 0x5a);
    set_test_env("PROSPER_GPU_TIMELINE", timeline_path);
    set_test_env("PROSPER_GPU_TIMELINE_CAPTURE", capture_path);
    if (mode != "after-work")
        set_test_env("PROSPER_GPU_CAPTURE_METADATA_ONLY", "1");

    if (mode == "invalid") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM", "not-an-address");
        intermediate_ok = !gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();
        begin_gpu_timeline_submit(1);
        record_gpu_timeline_submit(GpuState{}, 1);
    } else if (mode == "after-invalid") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM",
                     "not-an-address");
        intermediate_ok = !gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();
        begin_gpu_timeline_submit(1);
        record_gpu_timeline_submit(GpuState{}, 1);
    } else if (mode == "zero") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM", "0");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM", "0x0");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM", "0");
        intermediate_ok = !gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();
        begin_gpu_timeline_submit(1);
        record_gpu_timeline_submit(GpuState{}, 1);
    } else if (mode == "semantic") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "3");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM",
                     program_address(kSelectorVs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM",
                     program_address(kSelectorPs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_COMPUTE_PROGRAM",
                     program_address(kSelectorCompute));
        intermediate_ok = !gpu_timeline_capture_is_after_compute_gated();

        const auto exact = selector_draw_state(642, 362, kSelectorVs, kSelectorPs);
        const auto wrong_vs = selector_draw_state(642, 362, kWrongSelectorVs, kSelectorPs);
        const auto wrong_ps = selector_draw_state(642, 362, kSelectorVs, kWrongSelectorPs);
        const auto split_vs = selector_draw_state(642, 362, kSelectorVs, kWrongSelectorPs);
        const auto split_ps = selector_draw_state(642, 362, kWrongSelectorVs, kSelectorPs);
        const auto exact_wrong_target = selector_draw_state(1, 1, kSelectorVs, kSelectorPs);
        const auto target_wrong_pair = selector_draw_state(
            642, 362, kWrongSelectorVs, kWrongSelectorPs);
        std::vector<GpuState> submits;
        submits.push_back(selector_submit(                            // below lower bound
            {exact}, 10, kSelectorCompute));
        submits.push_back(selector_submit(                            // below lower bound
            {exact}, 20, kSelectorCompute));
        submits.push_back(selector_submit(                            // stages on different draws
            {split_vs, split_ps}, 30, kSelectorCompute));
        submits.push_back(selector_submit({wrong_vs}, 40, kSelectorCompute));
        submits.push_back(selector_submit({wrong_ps}, 50, kSelectorCompute));
        submits.push_back(selector_submit(                              // submit-wide false positive
            {exact_wrong_target, target_wrong_pair}, 60, kSelectorCompute));
        submits.push_back(selector_submit({exact}, 70));               // compute absent
        submits.push_back(selector_submit({exact}, 80,                 // wrong compute program
                                          kWrongSelectorCompute));
        submits.push_back(selector_submit({exact}, 90,                 // first exact conjunction
                                          kSelectorCompute));
        submits.push_back(selector_submit({exact}, 100,                // must not retarget
                                          kSelectorCompute));
        for (size_t i = 0; i < submits.size(); ++i) {
            const uint64_t submit_no = i + 1;
            begin_gpu_timeline_submit(submit_no);
            record_gpu_timeline_submit(submits[i], submit_no);
        }
    } else if (mode == "after") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM",
                     program_address(kSelectorVs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM",
                     program_address(kSelectorPs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM",
                     program_address(kSelectorCompute));
        intermediate_ok = gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();

        const auto exact = selector_draw_state(642, 362, kSelectorVs, kSelectorPs);
        const auto wrong_vs = selector_draw_state(642, 362, kWrongSelectorVs, kSelectorPs);
        const auto split_vs = selector_draw_state(642, 362, kSelectorVs, kWrongSelectorPs);
        const auto split_ps = selector_draw_state(642, 362, kWrongSelectorVs, kSelectorPs);
        std::vector<GpuState> submits;
        submits.push_back(selector_submit({exact}, 10, kWrongSelectorCompute));
        submits.push_back(selector_submit({exact}, 20));               // wrong gate did not arm
        submits.push_back(selector_submit({exact}, 30,                 // arm, never capture here
                                          kSelectorCompute));
        submits.push_back(selector_submit({wrong_vs}, 40));            // armed, wrong graphics
        submits.push_back(selector_submit({split_vs, split_ps}, 50));  // split graphics pair
        submits.push_back(selector_submit({exact}, 60));               // first valid later submit
        submits.push_back(selector_submit({exact}, 70));               // must not retarget
        for (size_t i = 0; i < submits.size(); ++i) {
            const uint64_t submit_no = i + 1;
            begin_gpu_timeline_submit(submit_no);
            record_gpu_timeline_submit(submits[i], submit_no);
        }
    } else if (mode == "after-bound") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "4");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM",
                     program_address(kSelectorVs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM",
                     program_address(kSelectorPs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM",
                     program_address(kSelectorCompute));
        intermediate_ok = gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();

        const auto exact = selector_draw_state(642, 362, kSelectorVs, kSelectorPs);
        std::vector<GpuState> submits;
        submits.push_back(selector_submit({exact}, 10, kSelectorCompute)); // arm before bound
        submits.push_back(selector_submit({exact}, 20));                   // still below bound
        submits.push_back(selector_submit({exact}, 30));                   // still below bound
        submits.push_back(selector_submit({exact}, 40));                   // first at bound
        for (size_t i = 0; i < submits.size(); ++i) {
            const uint64_t submit_no = i + 1;
            begin_gpu_timeline_submit(submit_no);
            record_gpu_timeline_submit(submits[i], submit_no);
        }
    } else if (mode == "after-work") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM",
                     program_address(kSelectorCompute));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE", bundle_path);
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH", "2");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DRAW_INDEX", "0:0");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY", "1");

        intermediate_ok = gpu_timeline_capture_is_after_compute_gated() &&
            !gpu_timeline_capture_after_compute_gate_armed();
        set_gpu_capture_ds_seed_snapshot_reader(
            [&](std::vector<GpuCaptureDsSeed>& seeds, std::string&) {
                ++ds_snapshots;
                seeds = {expected_ds_seed};
                return true;
            });

        const auto exact = selector_draw_state(642, 362, kSelectorVs, kSelectorPs);
        const auto wrong_target = selector_draw_state(1, 1, kSelectorVs, kSelectorPs);
        begin_gpu_timeline_submit(1);
        record_gpu_timeline_submit(selector_submit({exact}, 10, kWrongSelectorCompute), 1);
        begin_gpu_timeline_submit(2);
        record_gpu_timeline_submit(selector_submit({exact}, 20), 2);
        GpuTimelineCaptureCounters counters = gpu_timeline_capture_counters();
        intermediate_ok = intermediate_ok && counters.phase_observation_submits == 2 &&
            counters.phase_dispatches_scanned == 1 &&
            counters.prearm_history_submits_skipped == 2 &&
            counters.prearm_history_draws_skipped == 2 &&
            counters.prearm_bundle_submits_skipped == 2 &&
            counters.history_submits_recorded == 0 &&
            counters.bundle_submits_captured == 0 &&
            counters.detail_submits_captured == 0 &&
            counters.bundle_provenance_failures == 0 &&
            counters.history_lower_bound_submit_no == 0 &&
            !counters.history_phase_bounded &&
            !gpu_timeline_capture_after_compute_gate_armed() &&
            ds_snapshots == 0 &&
            !std::filesystem::exists(bundle_path) &&
            !std::filesystem::exists(capture_path);

        begin_gpu_timeline_submit(3);
        record_gpu_timeline_submit(selector_submit({exact}, 30, kSelectorCompute), 3);
        counters = gpu_timeline_capture_counters();
        GpuCaptureBundle armed_bundle;
        GpuCaptureFile armed_capture;
        std::string armed_error;
        intermediate_ok = intermediate_ok && counters.phase_observation_submits == 3 &&
            counters.phase_dispatches_scanned == 2 &&
            counters.prearm_history_submits_skipped == 2 &&
            counters.history_submits_recorded == 1 &&
            counters.bundle_submits_captured == 1 &&
            counters.detail_submits_captured == 0 &&
            counters.history_lower_bound_submit_no == 3 &&
            counters.history_phase_bounded &&
            gpu_timeline_capture_after_compute_gate_armed() &&
            ds_snapshots == 1 &&
            read_gpu_capture_bundle(bundle_path, armed_bundle, armed_error) &&
            armed_bundle.submits.size() == 1 &&
            armed_bundle.submits[0].submit_index == 3 &&
            materialize_gpu_capture_bundle_submit(
                armed_bundle, 0, armed_capture, armed_error) &&
            armed_capture.ds_seeds.size() == 1 &&
            armed_capture.ds_seeds[0].depth == expected_ds_seed.depth;

        begin_gpu_timeline_submit(4);
        record_gpu_timeline_submit(selector_submit({wrong_target}, 40), 4);
        begin_gpu_timeline_submit(5);
        record_gpu_timeline_submit(selector_submit({exact}, 50), 5);
    } else if (mode == "after-submit-zero") {
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "1");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_TARGET_DIM", "642x362");
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_VERTEX_PROGRAM",
                     program_address(kSelectorVs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_WHEN_FRAGMENT_PROGRAM",
                     program_address(kSelectorPs));
        set_test_env("PROSPER_GPU_TIMELINE_CAPTURE_AFTER_COMPUTE_PROGRAM",
                     program_address(kSelectorCompute));
        intermediate_ok = gpu_timeline_capture_is_after_compute_gated();

        const auto exact = selector_draw_state(642, 362, kSelectorVs, kSelectorPs);
        begin_gpu_timeline_submit(0);
        record_gpu_timeline_submit(selector_submit({exact}, 10, kSelectorCompute), 0);
        begin_gpu_timeline_submit(1);
        record_gpu_timeline_submit(selector_submit({exact}, 20), 1);
    } else {
        return 90;
    }
    close_gpu_timeline();

    GpuTimelineFile timeline;
    std::string error;
    const bool timeline_ok = read_gpu_timeline(timeline_path, timeline, error);
    bool ok = timeline_ok && intermediate_ok;
    if (mode == "invalid" || mode == "after-invalid") {
        ok = ok && timeline.submits.size() == 1 && timeline.details.empty() &&
             !std::filesystem::exists(capture_path);
    } else if (mode == "zero") {
        GpuCaptureFile capture;
        ok = ok && timeline.details.size() == 1 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 1;
    } else if (mode == "semantic") {
        GpuCaptureFile capture;
        ok = ok && timeline.submits.size() == 10 && timeline.details.size() == 1 &&
             timeline.details[0].submit_no == 9 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 9;
    } else if (mode == "after") {
        GpuCaptureFile capture;
        ok = ok && timeline.submits.size() == 7 && timeline.details.size() == 1 &&
             timeline.details[0].submit_no == 6 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 6;
    } else if (mode == "after-bound") {
        GpuCaptureFile capture;
        ok = ok && timeline.submits.size() == 4 && timeline.details.size() == 1 &&
             timeline.details[0].submit_no == 4 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 4;
    } else if (mode == "after-work") {
        GpuCaptureFile capture;
        GpuCaptureBundle bundle;
        GpuCaptureFile first_bundle_capture, second_bundle_capture;
        const GpuTimelineCaptureCounters counters = gpu_timeline_capture_counters();
        const auto lower_bound = [&](const auto& entry) {
            return entry.first == "PROSPER_CAPTURE_HISTORY_LOWER_BOUND_SUBMIT" &&
                   entry.second == "3";
        };
        ok = ok && intermediate_ok && timeline.submits.size() == 5 &&
             timeline.details.size() == 3 && timeline.details[0].submit_no == 3 &&
             timeline.details[1].submit_no == 4 && timeline.details[2].submit_no == 5 &&
             counters.history_submits_recorded == 3 &&
             counters.bundle_submits_captured == 3 && counters.detail_submits_captured == 1 &&
             counters.bundle_provenance_failures == 0 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 5 &&
             std::any_of(capture.metadata.renderer_env.begin(), capture.metadata.renderer_env.end(),
                         lower_bound) &&
             read_gpu_capture_bundle(bundle_path, bundle, error) && bundle.submits.size() == 2 &&
             bundle.submits[0].submit_index == 4 && bundle.submits[1].submit_index == 5 &&
             materialize_gpu_capture_bundle_submit(
                 bundle, 0, first_bundle_capture, error) &&
             materialize_gpu_capture_bundle_submit(
                 bundle, 1, second_bundle_capture, error) &&
             first_bundle_capture.ds_seeds.size() == 1 &&
             first_bundle_capture.ds_seeds[0].depth == expected_ds_seed.depth &&
             second_bundle_capture.ds_seeds.empty() &&
             // The two phase-boundary snapshots remain mandatory. The standalone endpoint has no
             // realized DS-using draw, so capture_referenced_gpu_ds_seeds correctly omits a third.
             ds_snapshots == 2;
    } else {
        GpuCaptureFile capture;
        const auto zero_lower_bound = [&](const auto& entry) {
            return entry.first == "PROSPER_CAPTURE_HISTORY_LOWER_BOUND_SUBMIT" &&
                   entry.second == "0";
        };
        ok = ok && timeline.submits.size() == 2 && timeline.details.size() == 1 &&
             timeline.details[0].submit_no == 1 &&
             read_gpu_capture(capture_path, capture, error) && capture.metadata.submit_index == 1 &&
             std::any_of(capture.metadata.renderer_env.begin(), capture.metadata.renderer_env.end(),
                         zero_lower_bound);
    }
    set_gpu_capture_ds_seed_snapshot_reader({});
    std::error_code ec;
    std::filesystem::remove(timeline_path, ec);
    std::filesystem::remove(capture_path, ec);
    std::filesystem::remove(bundle_path, ec);
    if (!ok) {
        std::fprintf(stderr, "selector child '%s' failed: %s\n", mode.c_str(), error.c_str());
        return 91;
    }
    return 0;
}

static int run_self(const char* executable, const char* mode) {
    const std::string command = std::string("\"") + executable + "\" --selector-child " + mode;
    const int status = std::system(command.c_str());
#ifdef _WIN32
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--selector-child")
        return run_selector_env_child(argv[2]);
    std::printf("== test_gpu_timeline ==\n");
    CHECK(!gpu_timeline_bundle_provenance_complete(
              GpuTimelineProducerProvenance::PhaseHistoryBounded, 9),
          "a temporal phase-bounded unknown makes requested bundle closure fail");
    CHECK(gpu_timeline_bundle_provenance_complete(
              GpuTimelineProducerProvenance::ExactRttSeed, 9) &&
              gpu_timeline_bundle_provenance_complete(
                  GpuTimelineProducerProvenance::ProducerHistory, 9),
          "an exact live RTT seed or post-arm producer closes a temporal bundle dependency");
    CHECK(gpu_timeline_bundle_provenance_complete(
              GpuTimelineProducerProvenance::PhaseHistoryBounded, UINT32_MAX),
          "a non-temporal guest image input does not require pre-phase producer history");
    GpuTimelineBundleProvenanceState unresolved_arm;
    gpu_timeline_observe_bundle_provenance(
        unresolved_arm, 3, GpuTimelineProducerProvenance::PhaseHistoryBounded, 7);
    gpu_timeline_observe_bundle_provenance(
        unresolved_arm, 4, GpuTimelineProducerProvenance::ExactRttSeed, 8);
    CHECK(!unresolved_arm.complete && unresolved_arm.first_incomplete_submit_no == 3 &&
              unresolved_arm.bounded_unknown_leaf_count == 1,
          "a closed endpoint cannot repair an unresolved phase-arm bundle constituent");
    GpuTimelineBundleProvenanceState closed_constituents;
    gpu_timeline_observe_bundle_provenance(
        closed_constituents, 3, GpuTimelineProducerProvenance::ExactRttSeed, 7);
    gpu_timeline_observe_bundle_provenance(
        closed_constituents, 4, GpuTimelineProducerProvenance::ProducerHistory, 8);
    CHECK(closed_constituents.complete &&
              closed_constituents.bounded_unknown_leaf_count == 0,
          "exactly seeded and post-arm-produced constituents keep the whole bundle closed");
    CHECK(run_self(argv[0], "invalid") == 0,
          "malformed graphics-program selector is rejected without capturing");
    CHECK(run_self(argv[0], "after-invalid") == 0,
          "malformed cross-submit compute gate is rejected without capturing");
    CHECK(run_self(argv[0], "zero") == 0,
          "zero graphics-program selectors and cross-submit compute gate remain disabled");
    CHECK(run_self(argv[0], "semantic") == 0,
          "graphics and compute selectors require one post-bound submit with the same-draw "
          "graphics conjunction");
    CHECK(run_self(argv[0], "after") == 0,
          "cross-submit compute gate arms only on the exact program and captures a later exact "
          "graphics conjunction once");
    CHECK(run_self(argv[0], "after-bound") == 0,
          "cross-submit compute gate may arm before the independent endpoint lower bound");
    CHECK(run_self(argv[0], "after-work") == 0,
          "phase-gated capture performs no history or full bundle work before arming, then retains "
          "the arm submit and a strictly later endpoint");
    CHECK(run_self(argv[0], "after-submit-zero") == 0,
          "cross-submit compute gate retains a submit-zero arm for a later submit-one capture");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = prosper_test::test_scratch_dir() /
        ("prosper-gpu-timeline-" + std::to_string(nonce));
    const auto good = base.string() + ".prgtl";
    const auto truncated = base.string() + "-truncated.prgtl";
    const auto corrupt = base.string() + "-corrupt.prgtl";

    GpuTimelineWriter writer;
    GpuTimelineMetadata metadata{"revision-1", "PPSA15552", "@scripts/dead-cells/route.pad"};
    std::string error;
    CHECK(writer.open(good, metadata, error), "writer creates a versioned timeline");
    GpuTimelineSubmit first;
    first.submit_no = 10; first.process_command_order = 400; first.first_command_order = 390;
    first.last_command_order = 400; first.color0_base = 0x12340000; first.draw_count = 3;
    first.dispatch_count = 1; first.color0_width = 642; first.color0_height = 362;
    first.dma_copy_count = 2;
    first.dma_copies.push_back({0x500000, 0x600000, 64, 0, 395, 0x1000});
    first.dma_copies.push_back({0x500040, 0x600040, 32, 3, 398, 0x1100});
    first.dma_data_count = 3;
    first.dma_data_records.push_back({0x500000, 0x600000, 64, 0, 395, 0x1000});
    first.dma_data_records.push_back({0xc70, 0, 4, 1, 397, 0x1080});
    first.dma_data_records.push_back({0x500040, 0x600040, 32, 3, 398, 0x1100});
    first.capture_incomplete = false;
    first.target_spans.push_back({0, 1, 642, 362});
    first.target_spans.push_back({1, 2, 738, 420});
    GpuTimelineSubmit invalid_spans = first;
    invalid_spans.target_spans[1].first_draw = 2;
    CHECK(!writer.append_submit(invalid_spans, error) &&
          error.find("invalid target spans") != std::string::npos,
          "writer rejects noncontiguous target-span coverage");
    error.clear();
    GpuTimelineSubmit invalid_dma_journal = first;
    invalid_dma_journal.dma_data_count = 4;
    CHECK(!writer.append_submit(invalid_dma_journal, error) &&
          error.find("invalid raw DMA_DATA journal") != std::string::npos,
          "writer rejects an uncapped DMA_DATA count without explicit truncation");
    error.clear();
    GpuTimelineDepthSurface first_depth;
    first_depth.depth_read_base = first_depth.depth_write_base = 0x700000;
    first_depth.stencil_read_base = first_depth.stencil_write_base = 0x830000;
    first_depth.htile_data_base = 0x820000; first_depth.db_depth_view = 0x11;
    first_depth.db_depth_size_xy = 0x01690281; first_depth.db_z_info = 0xa0000003;
    first_depth.db_stencil_info = 0x20000001; first_depth.target_width = 642;
    first_depth.target_height = 362; first_depth.draw_count = 3;
    first_depth.depth_test_count = 3; first_depth.depth_write_count = 2;
    first_depth.compare_mask = 1u << 3;
    first_depth.backing_hash_mask = 7; first_depth.depth_backing_hash = 0x1111;
    first_depth.stencil_backing_hash = 0x2222; first_depth.htile_backing_hash = 0x3333;
    first_depth.backing_writer_kind = 3; first_depth.backing_writer_sequence = 9;
    first_depth.backing_writer_addr = 0x820000; first_depth.backing_writer_size = 0x10000;
    first_depth.backing_writer_order = 380; first_depth.backing_writer_identity = 0xabcdef;
    first.depth_surfaces.push_back(first_depth);
    CHECK(writer.append_submit(first, error), "writer appends a submit record");
    GpuTimelinePresent present;
    present.present_count = 7; present.latest_submit_no = 10; present.buffer_index = 2;
    present.flip_arg = -5; present.width = 1920; present.height = 1080;
    CHECK(writer.append_present(present, error), "writer appends a present record");
    GpuTimelineDetail detail;
    detail.submit_no = 10; detail.capture_path = "/tmp/dead-cells-submit-10.prgcap";
    detail.semantic_draw_count = 3; detail.semantic_dispatch_count = 1;
    detail.realized_draw_count = 2; detail.realized_dispatch_count = 1;
    detail.operation_count = 4; detail.missing_operation_count = 1;
    detail.shader_version_count = 5; detail.resource_version_count = 3; detail.resource_bytes = 4096;
    CHECK(writer.append_detail(detail, error), "writer appends a detailed-capture link");
    GpuTimelineProducer producer;
    producer.consumer_submit_no = 10; producer.consumer_operation = 19;
    producer.future_writer_operation = 28; producer.resource_addr = 0x700000;
    producer.resource_size = 642ull * 362 * 4; producer.resource_width = 642;
    producer.resource_height = 362; producer.resolved = true;
    producer.producer_submit_no = 9; producer.producer_draw_index = 13;
    producer.producer_command_order = 380; producer.producer_target_addr = 0x700000;
    producer.producer_width = 642; producer.producer_height = 362;
    producer.first_writer_kind = GpuTimelineWriterKind::Graphics;
    producer.history_first_submit_no = 2; producer.history_first_draw_index = 7;
    producer.history_first_command_order = 80; producer.history_write_count = 31;
    producer.history_submit_count = 8; producer.history_window_first_submit_no = 1;
    producer.history_lower_bound_submit_no = 2;
    producer.provenance = GpuTimelineProducerProvenance::ProducerHistory;
    producer.lifetime_truncated = true; producer.history_window_truncated = true;
    producer.first_color_has_clear = true; producer.first_color_clear_word0 = 0x11223344;
    producer.first_color_clear_word1 = 0x55667788; producer.first_color_control = 0x60;
    producer.first_color_control_mode = 6; producer.first_target_mask = 0xf;
    producer.first_color_format = 10;
    CHECK(writer.append_producer(producer, error), "writer appends a prior-producer identity");
    GpuTimelineProducer bounded;
    bounded.consumer_submit_no = 10; bounded.consumer_operation = 20;
    bounded.future_writer_operation = 29; bounded.resource_addr = 0x900000;
    bounded.resource_size = 642ull * 362 * 4; bounded.resource_width = 642;
    bounded.resource_height = 362; bounded.history_window_first_submit_no = 7;
    bounded.history_lower_bound_submit_no = 7;
    bounded.provenance = GpuTimelineProducerProvenance::PhaseHistoryBounded;
    CHECK(writer.append_producer(bounded, error),
          "writer appends explicit phase-bounded unknown provenance");
    GpuTimelineSubmit second = first;
    second.submit_no = 11; second.draw_count = 4; second.dispatch_count = 0;
    second.target_spans[1].draw_count = 3;
    second.dma_data_count = 4;
    second.dma_data_records_truncated = true;
    CHECK(writer.append_submit(second, error), "writer appends a second submit record");
    CHECK(writer.flush(error), "writer flushes explicitly");
    writer.close();

    GpuTimelineFile timeline;
    CHECK(read_gpu_timeline(good, timeline, error), "reader accepts a complete timeline");
    CHECK(timeline.metadata.revision == metadata.revision &&
          timeline.metadata.title_id == metadata.title_id &&
          timeline.metadata.input_route == metadata.input_route,
          "metadata round-trips");
    CHECK(timeline.version == 10 && timeline.submits.size() == 2 && timeline.presents.size() == 1 &&
          timeline.details.size() == 1 && timeline.producers.size() == 2,
          "version and record counts round-trip");
    CHECK(timeline.submits[0].sequence == 1 && timeline.presents[0].sequence == 2 &&
          timeline.details[0].sequence == 3 && timeline.producers[0].sequence == 4 &&
          timeline.producers[1].sequence == 5 && timeline.submits[1].sequence == 6,
          "global record ordering round-trips");
    CHECK(timeline.submits[0].color0_base == 0x12340000 &&
          timeline.submits[0].color0_width == 642 && timeline.submits[0].color0_height == 362,
          "target identity and extent round-trip");
    CHECK(timeline.submits[0].dma_copy_count == 2 &&
          !timeline.submits[0].capture_incomplete &&
          timeline.submits[0].dma_copies.size() == 2 &&
          timeline.submits[0].dma_copies[0].src == 0x600000 &&
          timeline.submits[0].dma_copies[1].dst == 0x500040 &&
          timeline.submits[0].dma_copies[1].command_order == 398,
          "ordered-DMA identities and exact command order round-trip");
    CHECK(timeline.submits[0].dma_data_count == 3 &&
          !timeline.submits[0].dma_data_records_truncated &&
          timeline.submits[0].dma_data_records.size() == 3 &&
          timeline.submits[0].dma_data_records[1].src == 0 &&
          timeline.submits[0].dma_data_records[1].dst == 0xc70 &&
          timeline.submits[0].dma_data_records[1].bytes == 4 &&
          timeline.submits[0].dma_data_records[1].sels == 1 &&
          timeline.submits[0].dma_data_records[1].command_order == 397 &&
          timeline.submits[0].dma_data_records[1].packet_addr == 0x1080,
          "raw GDS-immediate DMA_DATA identity and PM4 order round-trip without an execution claim");
    CHECK(timeline.submits[1].dma_data_count == 4 &&
          timeline.submits[1].dma_data_records.size() == 3 &&
          timeline.submits[1].dma_data_records_truncated,
          "raw DMA_DATA original count survives a bounded, explicitly truncated journal");
    CHECK(timeline.submits[0].depth_surfaces.size() == 1 &&
          timeline.submits[0].depth_surfaces[0].htile_data_base == 0x820000 &&
          timeline.submits[0].depth_surfaces[0].db_depth_size_xy == 0x01690281 &&
          timeline.submits[0].depth_surfaces[0].depth_write_count == 2 &&
          timeline.submits[0].depth_surfaces[0].backing_hash_mask == 7 &&
          timeline.submits[0].depth_surfaces[0].htile_backing_hash == 0x3333 &&
          timeline.submits[0].depth_surfaces[0].backing_writer_sequence == 9 &&
          timeline.submits[0].depth_surfaces[0].backing_writer_identity == 0xabcdef,
          "native timeline depth identity, HTILE programming, and use counts round-trip");
    CHECK(timeline.submits[0].target_spans.size() == 2 &&
          timeline.submits[0].target_spans[1].first_draw == 1 &&
          timeline.submits[0].target_spans[1].draw_count == 2 &&
          timeline.submits[0].target_spans[1].width == 738 &&
          !timeline.submits[0].target_spans_truncated,
          "target-extent spans round-trip without run-local addresses");
    GpuTimelineSelector selector;
    selector.min_submit_no = 1; selector.target_width = 738; selector.target_height = 420;
    selector.target_min_draw = 1; selector.target_max_draw = 2;
    selector.min_draws = selector.max_draws = 3;
    selector.min_dispatches = selector.max_dispatches = 1;
    CHECK(gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector matches target position plus draw/dispatch counts");
    selector.target_min_draw = selector.target_max_draw = 0;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector rejects a target outside the requested draw range");
    selector.target_min_draw = 1; selector.target_max_draw = 2;
    selector.min_dispatches = selector.max_dispatches = 2;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector rejects an otherwise identical dispatch-count mismatch");
    selector.min_dispatches = selector.max_dispatches = 1;
    timeline.submits[0].target_spans_truncated = true;
    CHECK(!gpu_timeline_submit_matches(timeline.submits[0], selector),
          "shared selector refuses an incomplete target signature");
    timeline.submits[0].target_spans_truncated = false;
    CHECK(timeline.presents[0].buffer_index == 2 && timeline.presents[0].flip_arg == -5,
          "signed present fields round-trip");
    CHECK(timeline.details[0].submit_no == 10 && timeline.details[0].missing_operation_count == 1 &&
          timeline.details[0].shader_version_count == 5 && timeline.details[0].resource_bytes == 4096,
          "detailed-capture identity and version statistics round-trip");
    CHECK(timeline.producers[0].resolved && timeline.producers[0].consumer_operation == 19 &&
          timeline.producers[0].producer_submit_no == 9 &&
          timeline.producers[0].producer_command_order == 380 &&
          timeline.producers[0].resource_width == 642 &&
          timeline.producers[0].history_first_submit_no == 2 &&
          timeline.producers[0].history_write_count == 31 &&
          timeline.producers[0].history_lower_bound_submit_no == 2 &&
          timeline.producers[0].provenance ==
              GpuTimelineProducerProvenance::ProducerHistory &&
          timeline.producers[0].first_writer_kind == GpuTimelineWriterKind::Graphics &&
          timeline.producers[0].lifetime_truncated &&
          timeline.producers[0].history_window_truncated &&
          timeline.producers[0].first_color_has_clear &&
          timeline.producers[0].first_color_clear_word1 == 0x55667788,
          "prior-producer resource and writer identities round-trip");
    CHECK(!timeline.producers[1].resolved &&
          timeline.producers[1].history_lower_bound_submit_no == 7 &&
          timeline.producers[1].provenance ==
              GpuTimelineProducerProvenance::PhaseHistoryBounded,
          "phase-bounded unknown provenance remains fail-visible after round-trip");

    std::ifstream input(good, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    CHECK(bytes.size() > 16, "test timeline has framed records");
    std::vector<uint8_t> truncated_bytes(bytes.begin(), bytes.end() - 5);
    CHECK(write_bytes(truncated, truncated_bytes), "created a truncated-tail fixture");
    GpuTimelineFile partial;
    CHECK(read_gpu_timeline(truncated, partial, error), "reader recovers complete records from a truncated tail");
    CHECK(partial.truncated_tail && partial.submits.size() == 1 && partial.presents.size() == 1 &&
          partial.details.size() == 1 && partial.producers.size() == 2,
          "truncated final submit is ignored explicitly");

    std::vector<uint8_t> corrupt_bytes = bytes;
    corrupt_bytes[corrupt_bytes.size() - 12] ^= 0x80;
    CHECK(write_bytes(corrupt, corrupt_bytes), "created a checksum-corrupt fixture");
    GpuTimelineFile rejected;
    CHECK(!read_gpu_timeline(corrupt, rejected, error) &&
          error.find("checksum mismatch") != std::string::npos,
          "reader rejects corruption instead of treating it as truncation");

    const auto runtime = base.string() + "-runtime.prgtl";
    const auto runtime_capture = base.string() + "-submit-42.prgcap";
    const auto predecessor_capture = base.string() + "-submit-41.prgcap";
    const auto bundle_capture = base.string() + "-submits-41-42.prgbundle";
#ifdef _WIN32
    _putenv_s("PROSPER_GPU_TIMELINE", runtime.c_str());
    _putenv_s("PROSPER_CAPTURE_TITLE", "runtime-title");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR", predecessor_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE", bundle_capture.c_str());
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH", "2");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM", "642x362");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DRAW_INDEX", "1:1");
    _putenv_s("PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY", "1");
#else
    setenv("PROSPER_GPU_TIMELINE", runtime.c_str(), 1);
    setenv("PROSPER_CAPTURE_TITLE", "runtime-title", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE", runtime_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_SUBMIT", "42", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_PREDECESSOR", predecessor_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_BUNDLE", bundle_capture.c_str(), 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_DEPTH", "2", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DIM", "642x362", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_START_TARGET_DRAW_INDEX", "1:1", 1);
    setenv("PROSPER_GPU_TIMELINE_CAPTURE_CHECKPOINT_EVERY", "1", 1);
#endif
    GpuState prestart_state;
    prestart_state.cx[prosper::agc::Pm4::CB_COLOR0_ATTRIB2] =
        ((642u - 1u) << 14) | (362u - 1u);
    prestart_state.draws.push_back({3});
    prestart_state.draws.back().command_order = 104;
    prestart_state.command_order = 105;
    prestart_state.dma_copies.push_back({0x200000, 0x100000000ull, 16, 0, 104, 0});
    prestart_state.dma_data_record_count = 2;
    prestart_state.dma_data_records.push_back(
        {0x200000, 0x100000000ull, 16, 0, 104, 0});
    prestart_state.dma_data_records.push_back({0xc70, 0, 4, 1, 105, 0x2220});
    begin_gpu_timeline_submit(40);
    record_gpu_timeline_submit(prestart_state, 40);
    GpuState predecessor_state;
    predecessor_state.cx[prosper::agc::Pm4::CB_COLOR0_ATTRIB2] =
        ((642u - 1u) << 14) | (362u - 1u);
    predecessor_state.cx[prosper::agc::Pm4::DB_DEPTH_CONTROL] = 0x36;
    predecessor_state.cx[prosper::agc::Pm4::DB_Z_READ_BASE] = 0x7000;
    predecessor_state.cx[prosper::agc::Pm4::DB_Z_WRITE_BASE] = 0x7000;
    predecessor_state.cx[prosper::agc::Pm4::DB_HTILE_DATA_BASE] = 0x8200;
    predecessor_state.cx[prosper::agc::Pm4::DB_DEPTH_SIZE_XY] = 0x01690281;
    predecessor_state.draws.push_back({3});
    predecessor_state.draws.back().command_order = 119;
    predecessor_state.draws.push_back({3});
    predecessor_state.draws.back().command_order = 120;
    predecessor_state.command_order = 120;
    begin_gpu_timeline_submit(41);
    record_gpu_timeline_submit(predecessor_state, 41);
    GpuCaptureBundle checkpoint_bundle;
    CHECK(read_gpu_capture_bundle(bundle_capture, checkpoint_bundle, error) &&
          checkpoint_bundle.submits.size() == 1 &&
          checkpoint_bundle.submits[0].submit_index == 41,
          "bundle checkpoint preserves predecessor history before the endpoint exists");
    begin_gpu_timeline_submit(42);
    record_gpu_timeline_present(9, 1, 77, 1920, 1080); // a PM4 flip can precede submit completion
    GpuState runtime_state;
    runtime_state.command_order = 123;
    record_gpu_timeline_submit(runtime_state, 42);
    close_gpu_timeline();
    GpuTimelineFile runtime_timeline;
    CHECK(read_gpu_timeline(runtime, runtime_timeline, error), "runtime hooks produce an inspectable timeline");
    CHECK(runtime_timeline.presents.size() == 1 && runtime_timeline.submits.size() == 3 &&
          runtime_timeline.presents[0].latest_submit_no == 42,
          "a flip during folding is associated with its active submit");
    CHECK(runtime_timeline.submits[0].dma_copy_count == 1 &&
          !runtime_timeline.submits[0].capture_incomplete &&
          runtime_timeline.submits[0].dma_copies.size() == 1 &&
          runtime_timeline.submits[0].dma_copies[0].src == 0x100000000ull &&
          runtime_timeline.submits[0].dma_copies[0].dst == 0x200000 &&
          runtime_timeline.submits[0].first_command_order == 104 &&
          runtime_timeline.submits[0].last_command_order == 105,
          "runtime compact timeline exposes replayable ordered DMA identities");
    CHECK(runtime_timeline.submits[0].dma_data_count == 2 &&
          runtime_timeline.submits[0].dma_data_records.size() == 2 &&
          runtime_timeline.submits[0].dma_data_records[1].dst == 0xc70 &&
          runtime_timeline.submits[0].dma_data_records[1].src == 0 &&
          runtime_timeline.submits[0].dma_data_records[1].sels == 1 &&
          runtime_timeline.submits[0].dma_data_records[1].command_order == 105,
          "runtime compact timeline retains a raw GDS reset beside address-backed DMA");
    CHECK(runtime_timeline.submits[1].depth_surfaces.size() == 1 &&
          runtime_timeline.submits[1].depth_surfaces[0].depth_read_base == 0x700000 &&
          runtime_timeline.submits[1].depth_surfaces[0].htile_data_base == 0x820000 &&
          runtime_timeline.submits[1].depth_surfaces[0].depth_test_count == 2,
          "runtime timeline extracts compact DS programming without realizing resources");
    CHECK(runtime_timeline.submits[1].target_spans.size() == 1 &&
          runtime_timeline.submits[1].target_spans[0].first_draw == 0 &&
          runtime_timeline.submits[1].target_spans[0].draw_count == 2 &&
          runtime_timeline.submits[1].target_spans[0].width == 642 &&
          runtime_timeline.submits[1].target_spans[0].height == 362,
          "runtime timeline records compact semantic target spans");
    CHECK(runtime_timeline.details.size() == 3 && runtime_timeline.details[0].submit_no == 41 &&
          runtime_timeline.details[0].capture_path.find(bundle_capture + "#submit=41") == 0 &&
          runtime_timeline.details[1].capture_path == predecessor_capture &&
          runtime_timeline.details[2].submit_no == 42 &&
          runtime_timeline.details[2].capture_path == runtime_capture &&
          std::filesystem::exists(predecessor_capture) && std::filesystem::exists(runtime_capture),
          "exact submit selection writes and links producer-time predecessor and consumer capsules");
    GpuCaptureBundle runtime_bundle;
    CHECK(read_gpu_capture_bundle(bundle_capture, runtime_bundle, error) &&
          runtime_bundle.submits.size() == 2 && runtime_bundle.submits[0].submit_index == 41 &&
          runtime_bundle.submits[1].submit_index == 42,
          "exact submit selection writes an ordered same-run capture bundle");
    CHECK(runtime_bundle.submits.front().submit_index == 41,
          "bundle start target draw window excludes an earlier matching extent and includes the first writer");

    const auto compat_v2 = base.string() + "-compat-v2.prgtl";
    GpuTimelineWriter compat_writer;
    GpuTimelineSubmit legacy_first = first;
    legacy_first.depth_surfaces.clear(); legacy_first.target_spans.clear();
    legacy_first.dma_copies.clear();
    legacy_first.dma_copy_count = 0; legacy_first.capture_incomplete = false;
    legacy_first.dma_data_records.clear(); legacy_first.dma_data_count = 0;
    legacy_first.dma_data_records_truncated = false;
    CHECK(compat_writer.open(compat_v2, metadata, error) && compat_writer.append_submit(legacy_first, error),
          "created a detail-free compatibility timeline");
    compat_writer.close();
    std::ifstream compat_input(compat_v2, std::ios::binary);
    std::vector<uint8_t> version1_bytes((std::istreambuf_iterator<char>(compat_input)), {});
    version1_bytes[8] = 1; version1_bytes[9] = version1_bytes[10] = version1_bytes[11] = 0;
    const auto version1 = base.string() + "-version1.prgtl";
    CHECK(write_bytes(version1, version1_bytes), "created a version-1 compatibility fixture");
    GpuTimelineFile version1_timeline;
    CHECK(read_gpu_timeline(version1, version1_timeline, error) && version1_timeline.version == 1 &&
          version1_timeline.submits.size() == 1,
          "reader remains backward-compatible with version-1 semantic timelines");

    std::error_code ec;
    std::filesystem::remove(good, ec);
    std::filesystem::remove(truncated, ec);
    std::filesystem::remove(corrupt, ec);
    std::filesystem::remove(runtime, ec);
    std::filesystem::remove(runtime_capture, ec);
    std::filesystem::remove(predecessor_capture, ec);
    std::filesystem::remove(bundle_capture, ec);
    std::filesystem::remove(compat_v2, ec);
    std::filesystem::remove(version1, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

// Externally triggered headless whole-frame capture. Pure/offline: no guest or Vulkan device.
#include "../src/gpu/gpu_capture_bundle.hpp"
#include "../src/gpu/gpu_timeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include "test_scratch.h"
#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void clear_test_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static int run_conflict_child(const std::string& mode) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto trigger_path = prosper_test::test_scratch_dir() /
        ("prosper-capture-trigger-conflict-" + mode + "-" + std::to_string(nonce) + ".ready");
    const auto bundle_path = prosper_test::test_scratch_dir() /
        ("prosper-capture-trigger-conflict-" + mode + "-" + std::to_string(nonce) +
         ".prgbundle");

    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    clear_test_env("PROSPER_GPU_TIMELINE");
    set_test_env("PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE", trigger_path.string());
    set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
    { std::ofstream trigger(trigger_path, std::ios::binary); }

    if (mode == "scheduled") {
        set_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT", "1");
        CHECK(!capture_bundle_trigger_file_enabled(),
              "trigger-file gate fails closed beside a fixed-present gate");
    } else if (mode == "guest-log") {
        set_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG", "phase-ready");
        CHECK(!guest_log_capture_bundle_enabled() && !capture_bundle_trigger_file_enabled(),
              "guest-log and trigger-file gates both fail closed when combined");
        observe_guest_log_for_capture("phase-ready\n", 12);
    } else {
        return 2;
    }

    record_gpu_timeline_present(1, 0, 0, 4, 4);
    CHECK(!interactive_capture_bundle_active() && !std::filesystem::exists(bundle_path),
          "an ambiguous automatic-gate configuration cannot arm or write a bundle");

    std::error_code ec;
    std::filesystem::remove(trigger_path, ec);
    std::filesystem::remove(bundle_path, ec);
    return fails ? 1 : 0;
}

static int run_self(const char* executable, const char* mode) {
    const std::string command = std::string("\"") + executable +
        "\" --conflict-child " + mode;
    const int status = std::system(command.c_str());
#ifdef _WIN32
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--conflict-child")
        return run_conflict_child(argv[2]);
    std::printf("== test_capture_trigger_file ==\n");
    CHECK(run_self(argv[0], "scheduled") == 0,
          "fixed-present and trigger-file interaction fails closed");
    CHECK(run_self(argv[0], "guest-log") == 0,
          "guest-log and trigger-file interaction fails closed");

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto trigger_path = prosper_test::test_scratch_dir() /
        ("prosper-capture-trigger-" + std::to_string(nonce) + ".ready");
    const auto bundle_path = prosper_test::test_scratch_dir() /
        ("prosper-capture-trigger-" + std::to_string(nonce) + ".prgbundle");

    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG");
    clear_test_env("PROSPER_GPU_TIMELINE");
    set_test_env("PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE", trigger_path.string());
    set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
    set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "64");
    set_test_env("PROSPER_CAPTURE_FRAMES", "1");

    CHECK(capture_bundle_trigger_file_enabled(),
          "trigger-file gate enables only with trigger and bundle paths");
    record_gpu_timeline_present(1, 0, 0, 4, 4);
    CHECK(!interactive_capture_bundle_active(),
          "an absent trigger file leaves the whole-frame capture inert");

    std::error_code ec;
    CHECK(std::filesystem::create_directory(trigger_path, ec) && !ec,
          "create an existing non-regular trigger candidate");
    record_gpu_timeline_present(2, 0, 0, 4, 4);
    CHECK(!interactive_capture_bundle_active(),
          "an existing directory cannot satisfy the regular-file trigger contract");
    std::filesystem::remove(trigger_path, ec);

    { std::ofstream trigger(trigger_path, std::ios::binary); }
    CHECK(std::filesystem::is_regular_file(trigger_path),
          "external controller creates a durable regular-file witness");
    record_gpu_timeline_present(3, 0, 0, 4, 4);
    CHECK(interactive_capture_bundle_active() && !capture_bundle_trigger_file_enabled(),
          "the first present observing the file arms exactly once");

    GpuState state;
    state.command_order = 9;
    record_gpu_timeline_submit(state, 91);
    record_gpu_timeline_present(4, 0, 0, 4, 4);

    GpuCaptureBundle bundle;
    std::string error;
    CHECK(read_gpu_capture_bundle(bundle_path.string(), bundle, error),
          "the trigger-file request hands off to the existing whole-frame writer");
    CHECK(bundle.submits.size() == 1 && bundle.submits[0].submit_index == 91,
          "the bundle contains the complete frame after the observed trigger");
    CHECK(std::filesystem::is_regular_file(trigger_path),
          "capture leaves the trigger witness intact");
    CHECK(!interactive_capture_bundle_active(), "completed trigger-file capture disarms");

    record_gpu_timeline_present(5, 0, 0, 4, 4);
    CHECK(!interactive_capture_bundle_active(),
          "a persistent trigger file cannot re-arm the one-shot capture");

    std::filesystem::remove(trigger_path, ec);
    std::filesystem::remove(bundle_path, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

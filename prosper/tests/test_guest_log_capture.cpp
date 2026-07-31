// Exact guest-log phase gate for whole-frame captures. Pure/offline: no guest or Vulkan device.
#include "../src/gpu/gpu_capture_bundle.hpp"
#include "../src/gpu/gpu_timeline.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

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

int main() {
    std::printf("== test_guest_log_capture ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto bundle_path = std::filesystem::temp_directory_path() /
        ("prosper-guest-log-capture-" + std::to_string(nonce) + ".prgbundle");

    clear_test_env("PROSPER_CAPTURE_BUNDLE_AT_PRESENT");
    clear_test_env("PROSPER_GPU_TIMELINE");
    set_test_env("PROSPER_CAPTURE_BUNDLE_AFTER_GUEST_LOG", "phase-ready");
    set_test_env("PROSPER_CAPTURE_BUNDLE", bundle_path.string());
    set_test_env("PROSPER_CAPTURE_BUNDLE_MAX_MB", "64");
    set_test_env("PROSPER_CAPTURE_FRAMES", "1");

    CHECK(guest_log_capture_bundle_enabled(), "exact-line gate enables only with marker and bundle path");

    prosper::register_builtin_hle();
    using PrintfFn = int (*)(const char*, ...);
    using FputsHleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
    auto guest_printf = reinterpret_cast<PrintfFn>(
        reinterpret_cast<void*>(prosper::Hle::lookup(prosper::nid_hash("printf"))));
    auto guest_fputs = reinterpret_cast<FputsHleFn>(
        reinterpret_cast<void*>(prosper::Hle::lookup(prosper::nid_hash("fputs"))));
    CHECK(guest_printf && guest_fputs, "guest stdout adapters are registered for the integration probe");

    guest_printf("prefix %s suffix\n", "phase-ready");
    observe_guest_log_for_capture("phase-ready-suffix\r\n", 20);
    observe_guest_log_for_capture("phase-ready", 11);
    CHECK(!interactive_capture_bundle_active(), "substring, extended, and incomplete lines do not match");
    observe_guest_log_for_capture("x\n", 2);
    CHECK(!interactive_capture_bundle_active(), "a later terminator cannot turn a non-exact line into a match");

    std::string overlong(kGuestLogCaptureMaxLineBytes + 1, 'x');
    observe_guest_log_for_capture(overlong.data(), overlong.size());
    observe_guest_log_for_capture("\r", 1);
    CHECK(!interactive_capture_bundle_active(), "an overlong line is discarded through its terminator");

    const char fragment_a[] = "phase-";
    const char fragment_b[] = "ready";
    const char fragment_crlf[] = "\r\n";
    guest_fputs(reinterpret_cast<uint64_t>(fragment_a), 0, 0, 0, 0, 0);
    guest_fputs(reinterpret_cast<uint64_t>(fragment_b), 0, 0, 0, 0, 0);
    CHECK(!interactive_capture_bundle_active(), "a fragmented exact line waits for completion");
    guest_fputs(reinterpret_cast<uint64_t>(fragment_crlf), 0, 0, 0, 0, 0);
    CHECK(interactive_capture_bundle_active() && !guest_log_capture_bundle_enabled(),
          "fragmented CRLF exact match arms once and suppresses the LF half");

    // The first completed present is deliberately skipped. Repeating the exact marker afterward must
    // not re-arm/reset that delay, including concurrent duplicate observers.
    record_gpu_timeline_present(1, 0, 0, 4, 4);
    std::vector<std::thread> duplicates;
    for (unsigned i = 0; i < 4; ++i) {
        duplicates.emplace_back([] {
            observe_guest_log_for_capture("phase-ready\n", 12);
        });
    }
    for (auto& thread : duplicates) thread.join();

    record_gpu_timeline_present(2, 0, 0, 4, 4);
    GpuState state;
    state.command_order = 7;
    record_gpu_timeline_submit(state, 77);
    CHECK(!std::filesystem::exists(bundle_path),
          "capture starts only after the one-present delay and remains open until the next present");
    record_gpu_timeline_present(3, 0, 0, 4, 4);

    GpuCaptureBundle bundle;
    std::string error;
    CHECK(read_gpu_capture_bundle(bundle_path.string(), bundle, error),
          "the marker request hands off to the existing whole-frame bundle writer");
    CHECK(bundle.submits.size() == 1 && bundle.submits[0].submit_index == 77,
          "one-present delay and duplicate one-shot preserve the intended captured frame");
    CHECK(!interactive_capture_bundle_active(), "one-shot capture disarms after the frame is written");

    std::error_code ec;
    std::filesystem::remove(bundle_path, ec);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

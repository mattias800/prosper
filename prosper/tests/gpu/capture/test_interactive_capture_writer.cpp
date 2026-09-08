#include "fixtures/interactive_capture_wait.h"
#include "fixtures/test_scratch.h"
#include "gpu/capture/gpu_capture_bundle.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>

using namespace prosper::gpu;
using namespace std::chrono_literals;
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++fails; } } while (0)

int main(int argc, char**) {
    const auto directory = prosper_test::test_scratch_dir();
    const auto path = (directory / "owned.prgbundle").string();
    GpuCaptureDsSeed seed;
    seed.depth_read_base = seed.depth_write_base = 0x310000;
    seed.width = seed.height = 64;
    seed.depth_valid = true;
    seed.depth.resize(64 * 64 * 4);
    for (size_t i = 0; i < seed.depth.size(); ++i) seed.depth[i] = uint8_t(i * 37 + i / 251);
    const auto expected = seed.depth;
    set_gpu_capture_ds_seed_snapshot_reader(
        [&](std::vector<GpuCaptureDsSeed>& seeds, std::string&) { seeds = {seed}; return true; });
    GpuState state;
    auto collect = [&](const std::string& target) {
        CHECK(request_interactive_capture_bundle(target, 64).accepted, "capture admitted");
        record_gpu_timeline_present(1, 0, 0, 64, 64);
        record_gpu_timeline_submit(state, 77);
    };

    if (argc > 1) {
        collect(path);
        shutdown_interactive_capture_bundle();
        InteractiveGrabOutcome cancelled;
        CHECK(take_interactive_grab_outcome(cancelled) && !cancelled.ok &&
              cancelled.bundle_path == path && cancelled.max_unique_bytes == (64ull << 20) &&
              cancelled.error.find("cancelled at shutdown") != std::string::npos,
              "shutdown reports the incomplete window with its own identity and budget");
        CHECK(!std::filesystem::exists(path), "incomplete window is never installed as complete");
        CHECK(!interactive_capture_bundle_active(), "shutdown disables guest capture hooks");
        CHECK(!request_interactive_capture_bundle(path).accepted, "shutdown stops admission");
        set_gpu_capture_ds_seed_snapshot_reader({});
        return fails ? 1 : 0;
    }

    std::promise<void> entered, release;
    auto entered_future = entered.get_future();
    auto released = release.get_future().share();
    CHECK(set_interactive_bundle_writer_hook_for_test([&] {
        entered.set_value(); released.wait();
    }), "install latch on the actual writer");
    const auto replaced = request_interactive_capture_bundle("unused.prgbundle", 128);
    CHECK(replaced.accepted, "unstarted arm admitted");
    const auto replacement = request_interactive_capture_bundle(path, 64);
    CHECK(replacement.accepted && replacement.replaced_path == "unused.prgbundle",
          "unstarted arm replacement remains explicit");
    record_gpu_timeline_present(1, 0, 0, 64, 64);
    CHECK(!request_interactive_capture_bundle("busy-collecting", 3072).accepted,
          "collection rejects a second job before it can change the current budget");
    record_gpu_timeline_submit(state, 77);
    auto closing = std::async(std::launch::async, [&] {
        record_gpu_timeline_present(2, 0, 0, 64, 64);
    });
    CHECK(entered_future.wait_for(5s) == std::future_status::ready, "writer reaches blocked boundary");
    const bool returned = closing.wait_for(1s) == std::future_status::ready;
    CHECK(returned, "closing present returns while writer is still blocked");
    if (returned) {
        CHECK(!interactive_capture_bundle_active(), "writing does not enable guest capture hooks");
        InteractiveGrabOutcome premature;
        CHECK(!take_interactive_grab_outcome(premature), "no success before installation");
        CHECK(!request_interactive_capture_bundle("busy-writing", 3072).accepted,
              "blocked writer rejects another large capture without replacing its result");
        CHECK(!std::filesystem::exists(path), "blocked writer has installed no file");
        seed.depth.assign(seed.depth.size(), 0xee); // producer storage no longer has captured bytes
    }
    // Always release, including the synchronous-writer negative control. Never leave a failed
    // assertion holding a thread or static teardown forever.
    release.set_value();
    closing.get();
    if (!returned) { shutdown_interactive_capture_bundle(); return 1; }

    // Clearing the hook succeeds only after the actual writer releases its job. Leave the result
    // unread to test admission separately from the writer-busy case above.
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    bool idle = false;
    do {
        idle = set_interactive_bundle_writer_hook_for_test({});
        if (!idle) std::this_thread::sleep_for(1ms);
    } while (!idle && std::chrono::steady_clock::now() < deadline);
    CHECK(idle, "writer releases its owned work");
    CHECK(!request_interactive_capture_bundle("unread-result", 3072).accepted,
          "unread outcome cannot be overwritten by another request");
    InteractiveGrabOutcome outcome;
    CHECK(wait_for_interactive_grab(outcome) && outcome.ok && outcome.bundle_path == path &&
          outcome.max_unique_bytes == (64ull << 20), "original identity and budget survive busy requests");
    GpuCaptureBundle bundle;
    GpuCaptureFile capture;
    std::string error;
    CHECK(read_gpu_capture_bundle(path, bundle, error) && bundle.submits.size() == 1 &&
          materialize_gpu_capture_bundle_submit(bundle, 0, capture, error) &&
          capture.ds_seeds.size() == 1 && capture.ds_seeds[0].depth == expected,
          "real installed bundle retains original owned bytes after producer storage changes");
    CHECK(!take_interactive_grab_outcome(outcome), "completion is reported exactly once");

    const auto bad_path = directory / "directory-not-a-file";
    std::filesystem::create_directory(bad_path);
    { std::ofstream sentinel(bad_path / "keep"); sentinel << "existing content"; }
    collect(bad_path.string());
    record_gpu_timeline_present(2, 0, 0, 64, 64);
    CHECK(wait_for_interactive_grab(outcome) && !outcome.ok && !outcome.error.empty() &&
          outcome.bundle_path == bad_path.string(), "real installation failure reports its own outcome");
    CHECK(std::filesystem::is_regular_file(bad_path / "keep"), "failed install preserves existing content");
    CHECK(set_interactive_bundle_writer_hook_for_test([] { throw std::runtime_error("writer failure"); }),
          "install worker exception control");
    collect((directory / "exception.prgbundle").string());
    record_gpu_timeline_present(2, 0, 0, 64, 64);
    CHECK(wait_for_interactive_grab(outcome) && !outcome.ok && outcome.error == "writer failure",
          "worker exception becomes a failure and leaves the worker usable");

    std::promise<void> shutdown_entered, shutdown_release;
    auto shutdown_started = shutdown_entered.get_future();
    auto shutdown_released = shutdown_release.get_future().share();
    CHECK(set_interactive_bundle_writer_hook_for_test([&] {
        shutdown_entered.set_value(); shutdown_released.wait();
    }), "install shutdown writer latch");
    const auto final_path = (directory / "shutdown.prgbundle").string();
    collect(final_path);
    record_gpu_timeline_present(2, 0, 0, 64, 64);
    CHECK(shutdown_started.wait_for(5s) == std::future_status::ready, "final writer is blocked");
    auto shutdown = std::async(std::launch::async, shutdown_interactive_capture_bundle);
    const auto stop_deadline = std::chrono::steady_clock::now() + 5s;
    bool stopped = false;
    do {
        stopped = request_interactive_capture_bundle("after-shutdown").error ==
                  "capture writer is shutting down";
        if (!stopped) std::this_thread::sleep_for(1ms);
    } while (!stopped && std::chrono::steady_clock::now() < stop_deadline);
    CHECK(stopped, "shutdown has actually stopped admission before the drain assertion");
    CHECK(shutdown.wait_for(0ms) == std::future_status::timeout, "shutdown drains pending owned writing");
    shutdown_release.set_value();
    shutdown.get();
    CHECK(take_interactive_grab_outcome(outcome) && outcome.ok && outcome.bundle_path == final_path,
          "shutdown preserves the completed result for frontend reporting");
    CHECK(!request_interactive_capture_bundle(path).accepted, "drained writer cannot admit new work");
    shutdown_interactive_capture_bundle(); // ordinary teardown can safely repeat this
    set_gpu_capture_ds_seed_snapshot_reader({});
    return fails ? 1 : 0;
}

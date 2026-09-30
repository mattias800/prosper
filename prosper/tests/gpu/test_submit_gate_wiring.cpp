// test_submit_gate_wiring — the offscreen backend's queue wrappers really consult the shutdown gate
// (#3225).
//
// test_gpu_submit_gate asserts the gate's own logic. This asserts the WIRING, which is the half a
// pure test cannot see: render_locked_queue_submit / render_locked_queue_wait_idle
// (tests/fixtures/render_runner.h — the live offscreen backend, not a test harness; see
// tests/fixtures/AGENTS.md) must refuse once prosper-app has begun shutting down, and must refuse
// BEFORE entering the driver. Passing VK_NULL_HANDLE is how that second half is asserted: the call
// can only return without faulting if it never reached vkQueueSubmit.
//
// The positive arm runs first and on a REAL queue, so "the wrapper always returns DEVICE_LOST"
// cannot pass this test. It is skipped, loudly, when the host has no usable render device.
#include "host/platform/gpu_submit_gate.hpp"
#include <vulkan/vulkan.h>

#include <cstdio>
#include <string>
#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#else
#include <unistd.h>
#endif

// Fixture-local observers forward real calls. Production has no injection switch or duplicate
// queue/gate implementation. These definitions precede the scoped substitutions below.
static unsigned submit_calls = 0, idle_calls = 0;
static bool close_after_admission = false, substitute_completed_fence_timeout = false;
static bool inspect_origin_at_entry = false, origins_visible_at_entry = true;
static bool origin_is_driver_called();
static VkResult actual_fence_result = VK_NOT_READY;
static VkResult observed_queue_submit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
    ++submit_calls;
    if (inspect_origin_at_entry) origins_visible_at_entry &= origin_is_driver_called();
    if (close_after_admission) prosper::gpu_submit_gate_begin_shutdown();
    return vkQueueSubmit(q, n, s, f);
}
static VkResult observed_queue_wait_idle(VkQueue q) {
    ++idle_calls;
    if (inspect_origin_at_entry) origins_visible_at_entry &= origin_is_driver_called();
    return vkQueueWaitIdle(q);
}
static VkResult observed_wait_for_fences(VkDevice d, uint32_t n, const VkFence* f,
                                       VkBool32 all, uint64_t timeout) {
    actual_fence_result = vkWaitForFences(d, n, f, all, timeout);
    if (substitute_completed_fence_timeout && actual_fence_result == VK_SUCCESS) {
        prosper::gpu_submit_gate_begin_shutdown();
        return VK_TIMEOUT;
    }
    return actual_fence_result;
}
#define vkQueueSubmit observed_queue_submit
#define vkQueueWaitIdle observed_queue_wait_idle
#define vkWaitForFences observed_wait_for_fences
#include "fixtures/render_runner.h"
#undef vkQueueSubmit
#undef vkQueueWaitIdle
#undef vkWaitForFences

static prosper::test::BackendQueueCallOrigin* origin_at_entry = nullptr;
static bool origin_is_driver_called() {
    return origin_at_entry && *origin_at_entry == prosper::test::BackendQueueCallOrigin::DriverCalled;
}

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Capture the shipping failure line rather than testing a second formatter.
class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        file_ = std::tmpfile();
        if (file_) saved_ = dup(fileno(stderr));
        valid_ = file_ && saved_ >= 0 && dup2(fileno(file_), fileno(stderr)) >= 0;
    }
    bool valid() const { return valid_; }
    std::string finish() {
        restore_stderr();
        std::string text;
        if (file_) {
            std::rewind(file_);
            char buffer[512];
            while (const size_t n = std::fread(buffer, 1, sizeof(buffer), file_)) text.append(buffer, n);
            std::fclose(file_); file_ = nullptr;
        }
        std::fputs(text.c_str(), stderr);
        return text;
    }
    ~StderrCapture() {
        // Exceptional/unfinished capture cleanup must not allocate or throw.
        restore_stderr();
        if (file_) std::fclose(file_);
    }
private:
    void restore_stderr() noexcept {
        std::fflush(stderr);
        if (saved_ >= 0) { dup2(saved_, fileno(stderr)); close(saved_); saved_ = -1; }
    }
    FILE* file_ = nullptr;
    int saved_ = -1;
    bool valid_ = false;
};

static prosper::test::RenderCommandPoolLease empty_command(const prosper::test::RenderVkCtx& ctx) {
    using namespace prosper::test;
    const auto lease = acquire_render_command_pool(ctx.dev, ctx.qfi);
    CHECK(lease, "fixture acquires a real command buffer");
    if (!lease) return {};
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    const bool recorded = vkBeginCommandBuffer(lease.command, &begin) == VK_SUCCESS &&
                          vkEndCommandBuffer(lease.command) == VK_SUCCESS;
    CHECK(recorded, "fixture records a legal empty command buffer");
    if (!recorded) { release_render_command_pool(ctx.dev, ctx.qfi, lease); return {}; }
    return lease;
}

static std::string failure_line(const std::string& log) {
    constexpr const char* prefix = "[backend] GRAPHICS submission failed:";
    const size_t at = log.find(prefix);
    if (at == std::string::npos || log.find(prefix, at + 1) != std::string::npos) return {};
    return log.substr(at, log.find('\n', at) - at);
}

int main() {
    using namespace prosper::test;
    printf("== test_submit_gate_wiring ==\n");
    gpu_submit_gate_reset();
    CHECK(!gpu_submit_gate_shutting_down(), "the gate starts open");

    // Positive arm: with the gate open the wrapper reaches the driver and succeeds on an idle
    // queue. Without this, a wrapper hard-coded to fail would satisfy the refusal arm below.
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    submit_calls = idle_calls = 0;
    BackendQueueCallOrigin origin = BackendQueueCallOrigin::ShutdownGateRefused;
    if (ctx.ok && ctx.queue != VK_NULL_HANDLE) {
        origin_at_entry = &origin;
        inspect_origin_at_entry = true;
        CHECK(render_locked_queue_wait_idle(ctx.queue, &origin) == VK_SUCCESS,
              "with the gate OPEN render_locked_queue_wait_idle reaches the driver and succeeds");
        CHECK(origin == BackendQueueCallOrigin::DriverCalled && idle_calls == 1,
              "open idle origin agrees with an actual driver entry");
        // Both wrappers need an open-gate arm, or one of them could be hard-coded to fail and still
        // pass the refusal arm below. An empty submit (submitCount 0, pSubmits ignored, no fence) is
        // a legal no-op that reaches the driver and returns VK_SUCCESS.
        origin = BackendQueueCallOrigin::ShutdownGateRefused;
        CHECK(render_locked_queue_submit(ctx.queue, 0, nullptr, VK_NULL_HANDLE, &origin) ==
                  VK_SUCCESS,
              "with the gate OPEN render_locked_queue_submit reaches the driver and succeeds");
        CHECK(origin == BackendQueueCallOrigin::DriverCalled && submit_calls == 1,
              "open submit origin agrees with an actual driver entry");

        prosper::gpu::set_shared_present_active(true);
        origin = BackendQueueCallOrigin::ShutdownGateRefused;
        CHECK(render_locked_queue_wait_idle(ctx.queue, &origin) == VK_SUCCESS &&
                  origin == BackendQueueCallOrigin::DriverCalled && idle_calls == 2,
              "shared-present idle reports the actual call under the queue mutex");
        origin = BackendQueueCallOrigin::ShutdownGateRefused;
        CHECK(render_locked_queue_submit(ctx.queue, 0, nullptr, VK_NULL_HANDLE, &origin) == VK_SUCCESS &&
                  origin == BackendQueueCallOrigin::DriverCalled && submit_calls == 2,
              "shared-present submit reports the actual call under the queue mutex");
        prosper::gpu::set_shared_present_active(false);

        // The observer closes the gate AFTER admission. Its later state cannot classify this call.
        close_after_admission = true;
        origin = BackendQueueCallOrigin::ShutdownGateRefused;
        CHECK(render_locked_queue_submit(ctx.queue, 0, nullptr, VK_NULL_HANDLE, &origin) == VK_SUCCESS,
              "an admitted call still succeeds when shutdown starts at driver entry");
        CHECK(origin == BackendQueueCallOrigin::DriverCalled && submit_calls == 3 &&
                  gpu_submit_gate_shutting_down() && gpu_submit_gate_in_flight() == 0,
              "origin describes admission rather than the later shutdown state");
        close_after_admission = false;
        inspect_origin_at_entry = false;
        CHECK(origins_visible_at_entry, "origin is visible before each actual wrapper driver entry");
        gpu_submit_gate_reset();

        for (const bool deferred : {false, true}) {
            const auto lease = empty_command(ctx);
            if (!lease) continue;
            BackendSubmissionBatch batch;
            batch.enqueue(lease.command);
            const unsigned before = submit_calls;
            BackendSubmissionBatchResult result;
            if (deferred) {
                CHECK(batch.submit_deferred(ctx.dev, ctx.queue), "real deferred batch submits");
                CHECK(submit_calls == before + 1, "deferred batch enters the actual driver once");
                result = batch.retire_deferred();
            } else result = batch.submit_and_wait(ctx.dev, ctx.queue, false);
            CHECK(result.submit_result == VK_SUCCESS && result.wait_result == VK_SUCCESS &&
                      result.fence_waits == 1, "real batch retains successful effective results");
            CHECK(result.submit_origin == BackendQueueCallOrigin::DriverCalled &&
                      result.idle_origin == BackendQueueCallOrigin::NotAttempted &&
                      result.fence_wait_result == VK_SUCCESS && submit_calls == before + 1,
                  "synchronous and deferred batch results retain the actual call origin");
            if (result.submit_result == VK_SUCCESS && result.wait_result == VK_SUCCESS)
                release_render_command_pool(ctx.dev, ctx.qfi, lease);
        }
    } else {
        printf("  [note] no render device on this host; the open-gate arms did not run\n");
    }

    gpu_submit_gate_begin_shutdown();
    const unsigned submits_before = submit_calls, idles_before = idle_calls;

    // Refusal arm. Null handles are deliberate: reaching Vulkan with them would fault, so a clean
    // VK_ERROR_DEVICE_LOST is proof the gate short-circuited the call.
    origin = BackendQueueCallOrigin::DriverCalled;
    CHECK(render_locked_queue_submit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE, &origin) ==
              VK_ERROR_DEVICE_LOST,
          "render_locked_queue_submit refuses after begin_shutdown, without entering the driver");
    CHECK(origin == BackendQueueCallOrigin::ShutdownGateRefused && submit_calls == submits_before,
          "closed submit origin agrees with zero actual driver entries");
    origin = BackendQueueCallOrigin::DriverCalled;
    CHECK(render_locked_queue_wait_idle(VK_NULL_HANDLE, &origin) == VK_ERROR_DEVICE_LOST,
          "render_locked_queue_wait_idle refuses after begin_shutdown, without entering the driver");
    CHECK(origin == BackendQueueCallOrigin::ShutdownGateRefused && idle_calls == idles_before,
          "closed idle origin agrees with zero actual driver entries");
    CHECK(gpu_submit_gate_in_flight() == 0, "a refused wrapper call leaves no region behind");

    BackendSubmissionBatch empty;
    const auto empty_result = empty.submit_and_wait(VK_NULL_HANDLE, VK_NULL_HANDLE, false);
    CHECK(empty_result.submit_origin == BackendQueueCallOrigin::NotAttempted &&
              empty_result.idle_origin == BackendQueueCallOrigin::NotAttempted &&
              empty_result.queue_submits == 0 && empty_result.fence_waits == 0,
          "empty batches distinguish no queue call from gate refusal");

    if (ctx.ok && ctx.queue != VK_NULL_HANDLE) {
        BackendSubmissionBatch refused;
        refused.enqueue(VK_NULL_HANDLE); // Gate must refuse before this reaches Vulkan.
        StderrCapture capture;
        CHECK(capture.valid(), "fixture captures the actual closed-batch failure line");
        const auto result = refused.submit_and_wait(ctx.dev, VK_NULL_HANDLE, false);
        const std::string log = failure_line(capture.finish());
        CHECK(result.submit_result == VK_ERROR_DEVICE_LOST && result.wait_result == VK_SUCCESS &&
                  result.queue_submits == 1 && result.fence_waits == 0,
              "closed batch preserves legacy refusal results and wrapper-attempt count");
        CHECK(result.submit_origin == BackendQueueCallOrigin::ShutdownGateRefused &&
                  result.idle_origin == BackendQueueCallOrigin::NotAttempted &&
                  submit_calls == submits_before && idle_calls == idles_before,
              "closed batch propagates refusal without claiming a driver call");
        CHECK(log.find("submit=-4 wait=0") != std::string::npos &&
                  log.find("submit-origin=shutdown-gate-refused fence-wait=0 fence-waits=0 "
                           "idle-origin=not-attempted") != std::string::npos,
              "actual closed-batch diagnostic retains old tokens and names the origin");
    }

    gpu_submit_gate_reset();
    if (ctx.ok && ctx.queue != VK_NULL_HANDLE) {
        // First complete a REAL fence wait successfully, then substitute timeout and close the
        // gate. This calibrates reporting/control flow, not a driver timeout/loss witness.
        // Keep last: the unchanged unproven-completion retention policy is deliberately sticky.
        const auto lease = empty_command(ctx);
        if (lease) {
            BackendSubmissionBatch batch;
            batch.enqueue(lease.command);
            const unsigned idle_before = idle_calls;
            substitute_completed_fence_timeout = true;
            StderrCapture capture;
            CHECK(capture.valid(), "fixture captures the actual fallback failure line");
            const auto result = batch.submit_and_wait(ctx.dev, ctx.queue, false);
            substitute_completed_fence_timeout = false;
            const std::string log = failure_line(capture.finish());
            CHECK(actual_fence_result == VK_SUCCESS && result.submit_result == VK_SUCCESS &&
                      result.wait_result == VK_ERROR_DEVICE_LOST && result.fence_waits == 1,
                  "controlled fallback preserves the existing effective failure result");
            CHECK(result.fence_wait_result == VK_TIMEOUT &&
                      result.submit_origin == BackendQueueCallOrigin::DriverCalled &&
                      result.idle_origin == BackendQueueCallOrigin::ShutdownGateRefused &&
                      idle_calls == idle_before,
                  "fallback retains initial fence outcome and actual idle-call origin");
            CHECK(log.find("submit=0 wait=-4") != std::string::npos &&
                      log.find("submit-origin=driver-called fence-wait=2 fence-waits=1 "
                               "idle-origin=shutdown-gate-refused") != std::string::npos,
                  "actual fallback diagnostic distinguishes initial and effective results");
            CHECK(backend_has_unproven_submission(),
                  "reporting does not change indeterminate-submission retention policy");
            // Retain the command pool, as any indeterminate-completion caller must.
        }
    }
    gpu_submit_gate_reset();
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

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
#include "fixtures/render_runner.h"
#include "host/platform/gpu_submit_gate.hpp"

#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_submit_gate_wiring ==\n");
    gpu_submit_gate_reset();
    CHECK(!gpu_submit_gate_shutting_down(), "the gate starts open");

    // Positive arm: with the gate open the wrapper reaches the driver and succeeds on an idle
    // queue. Without this, a wrapper hard-coded to fail would satisfy the refusal arm below.
    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    if (ctx.ok && ctx.queue != VK_NULL_HANDLE) {
        CHECK(prosper::test::render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS,
              "with the gate OPEN the wrapper reaches the driver and succeeds");
    } else {
        printf("  [note] no render device on this host; the open-gate arm did not run\n");
    }

    gpu_submit_gate_begin_shutdown();

    // Refusal arm. Null handles are deliberate: reaching Vulkan with them would fault, so a clean
    // VK_ERROR_DEVICE_LOST is proof the gate short-circuited the call.
    CHECK(prosper::test::render_locked_queue_submit(VK_NULL_HANDLE, 0, nullptr, VK_NULL_HANDLE) ==
              VK_ERROR_DEVICE_LOST,
          "render_locked_queue_submit refuses after begin_shutdown, without entering the driver");
    CHECK(prosper::test::render_locked_queue_wait_idle(VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST,
          "render_locked_queue_wait_idle refuses after begin_shutdown, without entering the driver");
    CHECK(gpu_submit_gate_in_flight() == 0, "a refused wrapper call leaves no region behind");

    gpu_submit_gate_reset();
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

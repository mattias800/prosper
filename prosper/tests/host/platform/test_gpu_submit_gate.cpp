// test_gpu_submit_gate — the shutdown gate (src/host/platform/gpu_submit_gate.hpp) that keeps
// prosper-app from calling std::_Exit while the detached guest thread is inside a GPU submission
// (#3225: an unreapable zombie parked in __drm_exec_lock_obj froze the host compositor).
//
// Everything the frontend relies on is asserted here, because the defect it prevents cannot be
// reproduced in a test: admission before shutdown, refusal after, a drain that returns when the
// last region leaves, a drain that TIMES OUT rather than blocking forever when one does not, and a
// count that survives concurrent enters and leaves. Pure — no Vulkan, no dump, no GPU.
#include "host/platform/gpu_submit_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Clock = std::chrono::steady_clock;

static int64_t elapsed_ms(Clock::time_point since) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - since).count();
}

int main() {
    printf("== test_gpu_submit_gate ==\n");

    gpu_submit_gate_reset();
    CHECK(!gpu_submit_gate_shutting_down(), "open after reset");
    CHECK(gpu_submit_gate_in_flight() == 0, "no regions after reset");

    // --- admitted while the gate is open -----------------------------------------------------
    {
        GpuSubmitRegion region;
        CHECK(region.admitted(), "a region is admitted while the gate is open");
        CHECK(gpu_submit_gate_in_flight() == 1, "an admitted region is counted");
        {
            GpuSubmitRegion nested;
            CHECK(nested.admitted(), "regions nest");
            CHECK(gpu_submit_gate_in_flight() == 2, "the count is a depth, not a flag");
        }
        CHECK(gpu_submit_gate_in_flight() == 1, "leaving the inner region decrements once");
    }
    CHECK(gpu_submit_gate_in_flight() == 0, "leaving the outer region returns the count to zero");

    // A drain with nothing in flight is immediate, even with a zero timeout.
    CHECK(gpu_submit_gate_drain(0), "drain succeeds immediately when nothing is in flight");

    // --- refused once shutdown begins --------------------------------------------------------
    gpu_submit_gate_begin_shutdown();
    CHECK(gpu_submit_gate_shutting_down(), "shutting_down is true after begin_shutdown");
    {
        GpuSubmitRegion refused;
        CHECK(!refused.admitted(), "a region is REFUSED once shutdown has begun");
        CHECK(gpu_submit_gate_in_flight() == 0, "a refused region is not counted");
    }
    gpu_submit_gate_begin_shutdown();   // idempotent
    CHECK(gpu_submit_gate_shutting_down(), "begin_shutdown is idempotent");
    {
        GpuSubmitRegion still_refused;
        CHECK(!still_refused.admitted(), "still refused after a second begin_shutdown");
    }

    gpu_submit_gate_reset();
    CHECK(!gpu_submit_gate_shutting_down() && gpu_submit_gate_in_flight() == 0,
          "reset reopens the gate and clears the count");
    {
        GpuSubmitRegion reopened;
        CHECK(reopened.admitted(), "a region is admitted again after reset");
    }

    // --- a region entered BEFORE shutdown stays admitted, and the drain waits for it ----------
    {
        std::atomic<bool> inside{false};
        std::atomic<bool> release{false};
        std::atomic<bool> was_admitted{false};
        std::thread worker([&] {
            GpuSubmitRegion region;                    // entered while the gate is still open
            was_admitted.store(region.admitted());
            inside.store(true);
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
        while (!inside.load()) std::this_thread::yield();
        CHECK(was_admitted.load(), "a region entered before shutdown is admitted");

        gpu_submit_gate_begin_shutdown();
        CHECK(gpu_submit_gate_in_flight() == 1,
              "shutdown does not evict a region that is already inside");
        {
            GpuSubmitRegion after;
            CHECK(!after.admitted(), "new regions are refused while an old one is still inside");
        }

        // The drain must TIME OUT rather than block forever while that region is held. This is the
        // assertion the whole design turns on: an unbounded wait here would move the freeze earlier
        // rather than remove it.
        const auto t0 = Clock::now();
        const bool drained_while_held = gpu_submit_gate_drain(120);
        const int64_t waited = elapsed_ms(t0);
        CHECK(!drained_while_held, "drain reports failure while a region is still held");
        CHECK(waited >= 100, "drain actually waited for its timeout (>=100 ms of a 120 ms budget)");
        CHECK(waited < 5000, "drain returned rather than blocking indefinitely");

        // Now let it out: the drain must observe the region leaving and return true well inside its
        // budget, i.e. it is woken rather than polling to the deadline.
        const auto t1 = Clock::now();
        std::thread releaser([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            release.store(true);
        });
        const bool drained = gpu_submit_gate_drain(5000);
        const int64_t drain_ms = elapsed_ms(t1);
        CHECK(drained, "drain returns true once the last region leaves");
        CHECK(drain_ms < 4000, "drain returns when woken, not at its deadline");
        CHECK(gpu_submit_gate_in_flight() == 0, "no regions remain after the drain");
        releaser.join();
        worker.join();
    }

    // --- concurrency: the count is exact under many threads -----------------------------------
    {
        gpu_submit_gate_reset();
        constexpr int kThreads = 8;
        constexpr int kIterations = 5000;
        std::atomic<int> admitted{0};
        std::atomic<int> refused{0};
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; i++)
            threads.emplace_back([&] {
                for (int n = 0; n < kIterations; n++) {
                    GpuSubmitRegion region;
                    if (region.admitted()) admitted.fetch_add(1, std::memory_order_relaxed);
                    else                   refused.fetch_add(1, std::memory_order_relaxed);
                }
            });
        for (auto& t : threads) t.join();
        CHECK(admitted.load() == kThreads * kIterations,
              "every region is admitted while the gate stays open");
        CHECK(refused.load() == 0, "no region is refused while the gate stays open");
        CHECK(gpu_submit_gate_in_flight() == 0,
              "the count returns to exactly zero after concurrent enter/leave");
    }

    // --- the frontend's own sequence: close, then drain to zero with live traffic --------------
    //
    // This is the shape prosper-app uses. Threads are submitting when shutdown begins and the drain
    // must still reach zero, which in a build that did not refuse would be a race rather than a
    // guarantee. Note honestly which assertion carries that: the drain check below is
    // PROBABILISTIC — an un-refusing build could in principle be sampled at a transient all-out
    // instant — while `refusals > 0` is the deterministic one, and it is what goes red when
    // begin_shutdown() is mutated to a no-op.
    {
        gpu_submit_gate_reset();
        constexpr int kThreads = 6;
        std::atomic<bool> stop{false};
        std::atomic<int> refusals{0};
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; i++)
            threads.emplace_back([&] {
                while (!stop.load(std::memory_order_relaxed)) {
                    GpuSubmitRegion region;
                    if (!region.admitted()) {
                        refusals.fetch_add(1, std::memory_order_relaxed);
                        continue;   // a real caller returns VK_ERROR_DEVICE_LOST here
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(50));   // "in the driver"
                }
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // let traffic build up
        gpu_submit_gate_begin_shutdown();
        const bool drained = gpu_submit_gate_drain(2000);
        CHECK(drained, "drain reaches zero against live submitting threads once the gate is closed");
        CHECK(gpu_submit_gate_in_flight() == 0, "nothing is in flight after that drain");
        stop.store(true);
        for (auto& t : threads) t.join();
        CHECK(refusals.load() > 0, "the closed gate really refused work the threads then skipped");
        CHECK(gpu_submit_gate_in_flight() == 0,
              "refused regions after the drain never re-enter the count");
    }

    gpu_submit_gate_reset();
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

// Guest clocks advance in real time; only the internal GPU dependency watchdog discounts stalls.
#include "hle/dispatch/dispatch.hpp"
#include "hle/kernel/hle_kernel_time.hpp"
#include "hle/dispatch/nid.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_gpu_time_compensation ==\n");
    register_builtin_hle();
    auto ptc = Hle::lookup(nid_hash("sceKernelGetProcessTimeCounter"));
    auto clock_gettime_fn = Hle::lookup(nid_hash("sceKernelClockGettime"));
    auto tsc = Hle::lookup(nid_hash("sceKernelReadTsc"));
    CHECK(ptc && tsc && clock_gettime_fn, "monotonic, TSC and realtime entry points registered");
    if (fails) return 1;

    constexpr uint64_t kBudgetNs = 8'000'000;
    const uint64_t guest_before = ptc(0, 0, 0, 0, 0, 0);
    uint64_t before = host_gpu_progress_ns();
    int64_t rt0[2] = {};
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt0, 0, 0, 0, 0);
    {
        HostGpuClockScope scope(kBudgetNs);
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
        uint64_t held0 = host_gpu_progress_ns();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        uint64_t held1 = host_gpu_progress_ns();
        CHECK(held0 >= before + 4'000'000 && held0 <= before + 25'000'000,
              "host-GPU interval retains its bounded internal progress budget");
        CHECK(held1 == held0, "internal progress time holds after the GPU budget is exhausted");

        const uint64_t guest_now = ptc(0, 0, 0, 0, 0, 0);
        CHECK(guest_now >= guest_before + 35'000'000,
              "guest process time advances while the internal GPU budget is exhausted");
        std::atomic<bool> guest_clocks_consistent{true};
        std::vector<std::thread> guest_readers;
        for (unsigned i = 0; i < 8; ++i) {
            guest_readers.emplace_back([&] {
                const uint64_t first = ptc(0, 0, 0, 0, 0, 0);
                const uint64_t cycles = tsc(0, 0, 0, 0, 0, 0);
                int64_t mono[2]{};
                clock_gettime_fn(4, (uint64_t)(uintptr_t)mono, 0, 0, 0, 0);
                const uint64_t nanos = uint64_t(mono[0]) * 1'000'000'000 + mono[1];
                const uint64_t last = ptc(0, 0, 0, 0, 0, 0);
                if (first < guest_now || cycles < first || nanos < cycles || last < nanos)
                    guest_clocks_consistent.store(false);
            });
        }
        for (auto& reader : guest_readers) reader.join();
        CHECK(guest_clocks_consistent.load(),
              "concurrent guest process-time, TSC and monotonic reads share one advancing epoch");

        std::vector<uint64_t> concurrent(8);
        std::vector<std::thread> readers;
        for (size_t i = 0; i < concurrent.size(); ++i)
            readers.emplace_back([&, i] { concurrent[i] = host_gpu_progress_ns(); });
        for (auto& reader : readers) reader.join();
        CHECK(*std::min_element(concurrent.begin(), concurrent.end()) == held0 &&
                  *std::max_element(concurrent.begin(), concurrent.end()) == held0,
              "internal readers share one held progress value");
    }
    uint64_t after_scope = host_gpu_progress_ns();
    CHECK(after_scope - before <= 25'000'000,
          "scope exit permanently removes excess synchronous GPU time");

    int64_t rt1[2] = {};
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt1, 0, 0, 0, 0);
    int64_t wall_delta = (rt1[0] - rt0[0]) * 1'000'000'000ll + (rt1[1] - rt0[1]);
    CHECK(wall_delta >= 35'000'000,
          "CLOCK_REALTIME continues through a compensated host-GPU interval");

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    uint64_t resumed = host_gpu_progress_ns();
    CHECK(resumed >= after_scope + 8'000'000,
          "internal progress time resumes after the host-GPU scope");

    // A rejected overlapping begin returns zero, and ending that token must not truncate the
    // active outer scope. This pins the fail-open behavior used if serialization ever regresses.
    uint64_t outer = guest_clock_host_gpu_begin(2'000'000);
    uint64_t inner = guest_clock_host_gpu_begin(2'000'000);
    CHECK(outer != 0 && inner == 0, "overlapping host-GPU scopes fail open without nesting");
    guest_clock_host_gpu_end(inner);
    guest_clock_host_gpu_end(outer + 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    uint64_t overlap_held = host_gpu_progress_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(host_gpu_progress_ns() == overlap_held,
          "rejected and mismatched ends cannot release the outer clock hold");
    guest_clock_host_gpu_end(outer);

    // Exercise the lock-free reader snapshot across rapid begin/end publications. Every reader
    // must retain its local monotonic ordering, and the one writer must never lose its scope token.
    std::atomic<bool> stop_readers{false};
    std::atomic<bool> readers_monotonic{true};
    std::vector<std::thread> racing_readers;
    for (unsigned i = 0; i < 4; ++i) {
        racing_readers.emplace_back([&] {
            uint64_t previous = host_gpu_progress_ns();
            while (!stop_readers.load(std::memory_order_relaxed)) {
                const uint64_t current = host_gpu_progress_ns();
                if (current < previous)
                    readers_monotonic.store(false, std::memory_order_relaxed);
                previous = current;
            }
        });
    }
    bool writer_tokens_valid = true;
    for (unsigned i = 0; i < 2000; ++i) {
        const uint64_t token = guest_clock_host_gpu_begin(0);
        writer_tokens_valid &= token != 0;
        guest_clock_host_gpu_end(token);
    }
    stop_readers.store(true, std::memory_order_relaxed);
    for (auto& reader : racing_readers) reader.join();
    CHECK(writer_tokens_valid && readers_monotonic.load(std::memory_order_relaxed),
          "lock-free readers stay monotonic across scope publication races");
    const uint64_t stress_end = host_gpu_progress_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(host_gpu_progress_ns() >= stress_end + 2'000'000,
          "internal progress time resumes after rapid scope transitions");

    if (fails) std::printf("== FAIL (%d) ==\n", fails);
    else       std::printf("== PASS ==\n");
    return fails ? 1 : 0;
}

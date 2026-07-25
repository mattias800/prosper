// test_gpu_time_compensation - discount synchronous host-GPU overhead from guest monotonic time.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/hle_kernel_time.hpp"
#include "../src/hle/nid.hpp"
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
    CHECK(ptc && clock_gettime_fn, "monotonic and realtime entry points registered");
    if (fails) return 1;

    constexpr uint64_t kBudgetNs = 8'000'000;
    uint64_t before = ptc(0, 0, 0, 0, 0, 0);
    int64_t rt0[2] = {};
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt0, 0, 0, 0, 0);
    {
        HostGpuClockScope scope(kBudgetNs);
        std::this_thread::sleep_for(std::chrono::milliseconds(35));
        uint64_t held0 = ptc(0, 0, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        uint64_t held1 = ptc(0, 0, 0, 0, 0, 0);
        CHECK(held0 >= before + 4'000'000 && held0 <= before + 25'000'000,
              "host-GPU interval retains its bounded guest frame budget");
        CHECK(held1 == held0, "guest monotonic time holds after the GPU budget is exhausted");

        std::vector<uint64_t> concurrent(8);
        std::vector<std::thread> readers;
        for (size_t i = 0; i < concurrent.size(); ++i)
            readers.emplace_back([&, i] { concurrent[i] = ptc(0, 0, 0, 0, 0, 0); });
        for (auto& reader : readers) reader.join();
        CHECK(*std::min_element(concurrent.begin(), concurrent.end()) == held0 &&
                  *std::max_element(concurrent.begin(), concurrent.end()) == held0,
              "concurrent readers share one held monotonic value");
    }
    uint64_t after_scope = ptc(0, 0, 0, 0, 0, 0);
    CHECK(after_scope - before <= 25'000'000,
          "scope exit permanently removes excess synchronous GPU time");

    int64_t rt1[2] = {};
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt1, 0, 0, 0, 0);
    int64_t wall_delta = (rt1[0] - rt0[0]) * 1'000'000'000ll + (rt1[1] - rt0[1]);
    CHECK(wall_delta >= 35'000'000,
          "CLOCK_REALTIME continues through a compensated host-GPU interval");

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    uint64_t resumed = ptc(0, 0, 0, 0, 0, 0);
    CHECK(resumed >= after_scope + 8'000'000,
          "guest monotonic time resumes after the host-GPU scope");

    // A rejected overlapping begin returns zero, and ending that token must not truncate the
    // active outer scope. This pins the fail-open behavior used if serialization ever regresses.
    uint64_t outer = guest_clock_host_gpu_begin(2'000'000);
    uint64_t inner = guest_clock_host_gpu_begin(2'000'000);
    CHECK(outer != 0 && inner == 0, "overlapping host-GPU scopes fail open without nesting");
    guest_clock_host_gpu_end(inner);
    guest_clock_host_gpu_end(outer + 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    uint64_t overlap_held = ptc(0, 0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(ptc(0, 0, 0, 0, 0, 0) == overlap_held,
          "rejected and mismatched ends cannot release the outer clock hold");
    guest_clock_host_gpu_end(outer);

    // Exercise the lock-free reader snapshot across rapid begin/end publications. Every reader
    // must retain its local monotonic ordering, and the one writer must never lose its scope token.
    std::atomic<bool> stop_readers{false};
    std::atomic<bool> readers_monotonic{true};
    std::vector<std::thread> racing_readers;
    for (unsigned i = 0; i < 4; ++i) {
        racing_readers.emplace_back([&] {
            uint64_t previous = ptc(0, 0, 0, 0, 0, 0);
            while (!stop_readers.load(std::memory_order_relaxed)) {
                const uint64_t current = ptc(0, 0, 0, 0, 0, 0);
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
    const uint64_t stress_end = ptc(0, 0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(ptc(0, 0, 0, 0, 0, 0) >= stress_end + 2'000'000,
          "guest monotonic time resumes after rapid scope transitions");

    if (fails) std::printf("== FAIL (%d) ==\n", fails);
    else       std::printf("== PASS ==\n");
    return fails ? 1 : 0;
}

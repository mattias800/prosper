// test_det_clock - opt-in flip-paced guest monotonic time (#240).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace prosper;

extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx,
                                           uint32_t flip_mode, int64_t flip_arg);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_det_clock ==\n");
#ifdef _WIN32
    _putenv_s("PROSPER_DET_CLOCK", "1");
    _putenv_s("PROSPER_DET_FPS", "60");
#else
    setenv("PROSPER_DET_CLOCK", "1", 1);
    setenv("PROSPER_DET_FPS", "60", 1);
#endif
    register_builtin_hle();
    auto ptc = Hle::lookup(nid_hash("sceKernelGetProcessTimeCounter"));
    auto clock_gettime_fn = Hle::lookup(nid_hash("sceKernelClockGettime"));
    CHECK(ptc && clock_gettime_fn, "monotonic and realtime entry points registered");
    if (fails) return 1;

    uint64_t pre0 = ptc(0, 0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    uint64_t pre1 = ptc(0, 0, 0, 0, 0, 0);
    CHECK(pre1 > pre0, "real monotonic time advances before the first flip");

    prosper_vo_flip_from_gpu(0, 0, 0, 1);
    constexpr size_t kThreads = 16;
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<uint64_t> first(kThreads);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            first[i] = ptc(0, 0, 0, 0, 0, 0);
        });
    }
    while (ready.load() != kThreads) std::this_thread::yield();
    go.store(true);
    for (auto& thread : threads) thread.join();
    uint64_t first_min = first[0], first_max = first[0];
    for (uint64_t value : first) {
        first_min = std::min(first_min, value);
        first_max = std::max(first_max, value);
    }
    CHECK(first_min >= pre1 && first_max == first_min,
          "concurrent first-flip reads share one initialized monotonic anchor");

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    uint64_t held = ptc(0, 0, 0, 0, 0, 0);
    CHECK(held == first_min, "deterministic monotonic time pauses between flips");

    int64_t rt0[2] = {}, rt1[2] = {};
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt0, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    clock_gettime_fn(0, (uint64_t)(uintptr_t)rt1, 0, 0, 0, 0);
    int64_t wall_delta = (rt1[0] - rt0[0]) * 1000000000ll + (rt1[1] - rt0[1]);
    CHECK(wall_delta >= 10000000ll, "CLOCK_REALTIME continues advancing while monotonic time is paused");

    prosper_vo_flip_from_gpu(0, 0, 0, 2);
    uint64_t next = ptc(0, 0, 0, 0, 0, 0);
    CHECK(next - held == 1000000000ull / 60ull, "one flip advances monotonic time by exactly 1/60 second");

    if (fails) std::printf("== FAIL (%d) ==\n", fails);
    else       std::printf("== PASS ==\n");
    return fails ? 1 : 0;
}

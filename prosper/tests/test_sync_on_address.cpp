// test_sync_on_address -- basic behavior for the raw wait/wake-by-address HLE NIDs.
#include "../src/hle/dispatch.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_sync_on_address ==\n");
    register_builtin_hle();

    HleFn wait = Hle::lookup("Hc4CaR6JBL0");
    HleFn wake = Hle::lookup("q2y-wDIVWZA");
    CHECK(wait && wake, "wait/wake-by-address NIDs registered");
    if (!wait || !wake) return 1;

    alignas(4) uint32_t word = 2;
    std::atomic_ref<uint32_t> word_atomic(word);
    wait((uint64_t)(uintptr_t)&word, 1, 0, 0, 0, 0);
    CHECK(word_atomic.load(std::memory_order_acquire) == 2, "wait returns immediately when value does not match expected");

    word_atomic.store(1, std::memory_order_release);
    std::atomic<bool> started{false};
    std::atomic<bool> done{false};
    std::thread waiter([&] {
        started.store(true, std::memory_order_release);
        wait((uint64_t)(uintptr_t)&word, 1, 0, 0, 0, 0);
        done.store(true, std::memory_order_release);
    });

    while (!started.load(std::memory_order_acquire))
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!done.load(std::memory_order_acquire), "wait blocks while value matches expected");

    word_atomic.store(2, std::memory_order_release);
    wake((uint64_t)(uintptr_t)&word, 1, 0, 0, 0, 0);
    for (int i = 0; i < 100 && !done.load(std::memory_order_acquire); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(done.load(std::memory_order_acquire), "wake releases a waiter after the value changes");
    waiter.join();

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

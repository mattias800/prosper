// test_sync_on_address -- basic behavior for the raw wait/wake-by-address HLE NIDs.
#include "../src/hle/dispatch.hpp"
#include <atomic>
#include <climits>
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

    // Timed wait honors the timeout by DEFAULT (#139): a wait on a still-matching value with a short
    // timeout must return the Sony ETIMEDOUT (0x8002003c) after ~the timeout — not block forever and
    // return 0 (=signaled). Previously this was gated off (PROSPER_WAIT_TIMEOUT) so it blocked forever.
    {
        alignas(4) uint32_t tw = 5;
        uint32_t timeout_us = 30000;   // 30 ms
        auto t0 = std::chrono::steady_clock::now();
        uint64_t rc = wait((uint64_t)(uintptr_t)&tw, 5 /*expected == value -> blocks*/,
                           (uint64_t)(uintptr_t)&timeout_us, 0, 0, 0);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0).count();
        // The expected value changed in #1612. It was 0x80020060, which is the decimal 60 written as
        // if it were hex: the libkernel encoding is 0x80020000|errno with a FreeBSD errno, and 0x60
        // is 96 = EOWNERDEAD, not a timeout. FreeBSD ETIMEDOUT is 60 = 0x3c, which is what every
        // other timeout in the emulator already encodes (event flags, semaphores, cond timedwait) —
        // so this one site was telling a guest that a mutex owner had died. The old assertion pinned
        // the emulator's own output rather than the contract; it was introduced alongside an explicit
        // "CONFIDENCE: MED on the exact code value".
        CHECK(rc == 0x8002003cull, "timed wait returns SCE ETIMEDOUT (0x8002003c), not 0/blocked-forever");
        CHECK(elapsed >= 20 && elapsed < 2000, "timed wait actually waited ~the timeout (not the old 5 s cap / forever)");
    }

    // Windows rounds microseconds to milliseconds. The addition used for that rounding must happen
    // after widening: UINT32_MAX otherwise wraps to 998 and times out in 1 ms instead of remaining
    // blocked until this explicit wake.
    {
        alignas(4) uint32_t tw = 9;
        uint32_t timeout_us = UINT32_MAX;
        std::atomic<uint64_t> result{~uint64_t{0}};
        std::thread waiter([&] {
            result.store(wait((uint64_t)(uintptr_t)&tw, 9,
                              (uint64_t)(uintptr_t)&timeout_us, 0, 0, 0),
                         std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        std::atomic_ref<uint32_t> tw_atomic(tw);
        tw_atomic.store(10, std::memory_order_release);
        wake((uint64_t)(uintptr_t)&tw, 1, 0, 0, 0, 0);
        waiter.join();
        CHECK(result.load(std::memory_order_acquire) == 0,
              "maximum uint32 timeout remains blocked until an explicit wake");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

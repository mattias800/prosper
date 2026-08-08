#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;

static int failures;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++failures; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    register_builtin_hle();
    auto init = Hle::lookup(nid_hash("sem_init"));
    auto wait = Hle::lookup(nid_hash("sem_wait"));
    auto post = Hle::lookup(nid_hash("sem_post"));
    auto destroy = Hle::lookup(nid_hash("sem_destroy"));
    CHECK(init && wait && post && destroy, "plain libScePosix semaphore imports are registered");

    alignas(16) uint8_t sem[32]{};
    CHECK(init((uint64_t)sem, 0, 0, 0, 0, 0) == 0, "zero-count semaphore initializes");

    std::atomic<bool> entered{false}, released{false};
    std::thread consumer([&] {
        entered.store(true, std::memory_order_release);
        const uint64_t rc = wait((uint64_t)sem, 0, 0, 0, 0, 0);
        released.store(rc == 0, std::memory_order_release);
    });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!released.load(std::memory_order_acquire), "sem_wait blocks while the count is zero");
    CHECK(post((uint64_t)sem, 0, 0, 0, 0, 0) == 0, "sem_post wakes one waiter");
    consumer.join();
    CHECK(released.load(std::memory_order_acquire), "blocked waiter resumes successfully");
    // The PROPERTY, not the literal: #2170 gave destroyed slots their own sentinel and #2176 routed
    // every destroy through one atomic claim, so this now holds kPtDestroyed rather than NULL. The
    // guest cannot tell the two apart -- ensure_sem refuses any sentinel with EINVAL, and sem_init
    // overwrites the slot unconditionally either way -- but the old assertion pinned the value.
    void* const sem_before = *(void**)sem;
    CHECK(destroy((uint64_t)sem, 0, 0, 0, 0, 0) == 0,
          "semaphore destroys without reporting failure");
    CHECK(*(void**)sem != sem_before && (uintptr_t)*(void**)sem < 0x1000,
          "semaphore destroy detaches the guest handle from the object (a sentinel, not a pointer)");

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}

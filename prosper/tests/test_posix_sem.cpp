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
    CHECK(destroy((uint64_t)sem, 0, 0, 0, 0, 0) == 0 && *(void**)sem == nullptr,
          "semaphore destroys and clears its guest handle");

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}

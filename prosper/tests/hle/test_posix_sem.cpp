// test_posix_sem — the plain libScePosix semaphore surface: registration, a zero-count
// initialisation, a wait that genuinely blocks until a post, and a destroy that detaches the guest's
// handle from the object instead of leaving a dangling pointer behind.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

using namespace prosper;

namespace {
// The wait/post/destroy trio for one 32-byte guest semaphore, resolved through the built-in registry
// rather than called directly: the registration itself is part of the contract (#3022's shadowing
// class of defect replaced a working handler with a no-op that returned OK and wrote nothing).
struct PosixSem {
    HleFn init = nullptr, wait = nullptr, post = nullptr, destroy = nullptr;
    bool registered() const { return init && wait && post && destroy; }
};
PosixSem lookup_posix_sem() {
    register_builtin_hle();
    PosixSem sem;
    sem.init = Hle::lookup(nid_hash("sem_init"));
    sem.wait = Hle::lookup(nid_hash("sem_wait"));
    sem.post = Hle::lookup(nid_hash("sem_post"));
    sem.destroy = Hle::lookup(nid_hash("sem_destroy"));
    return sem;
}
}   // namespace

TEST(PosixSem, PlainLibScePosixSemaphoreImportsAreRegistered) {
    EXPECT_TRUE(lookup_posix_sem().registered())
        << "plain libScePosix semaphore imports are registered";
}

TEST(PosixSem, ZeroCountSemaphoreInitializes) {
    const PosixSem sem = lookup_posix_sem();
    ASSERT_TRUE(sem.registered());
    alignas(16) uint8_t storage[32]{};
    EXPECT_EQ(sem.init(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0), 0u)
        << "zero-count semaphore initializes";
}

TEST(PosixSem, WaitBlocksUntilAPostWakesIt) {
    const PosixSem sem = lookup_posix_sem();
    ASSERT_TRUE(sem.registered());
    alignas(16) uint8_t storage[32]{};
    ASSERT_EQ(sem.init(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0), 0u);

    std::atomic<bool> entered{false}, released{false};
    std::thread consumer([&] {
        entered.store(true, std::memory_order_release);
        const uint64_t rc = sem.wait(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0);
        released.store(rc == 0, std::memory_order_release);
    });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(released.load(std::memory_order_acquire))
        << "sem_wait blocks while the count is zero";
    EXPECT_EQ(sem.post(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0), 0u)
        << "sem_post wakes one waiter";
    consumer.join();
    EXPECT_TRUE(released.load(std::memory_order_acquire)) << "blocked waiter resumes successfully";
}

TEST(PosixSem, DestroyDetachesTheGuestHandleFromTheObject) {
    // The PROPERTY, not the literal: #2170 gave destroyed slots their own sentinel and #2176 routed
    // every destroy through one atomic claim, so this now holds kPtDestroyed rather than NULL. The
    // guest cannot tell the two apart -- ensure_sem refuses any sentinel with EINVAL, and sem_init
    // overwrites the slot unconditionally either way -- but the old assertion pinned the value.
    const PosixSem sem = lookup_posix_sem();
    ASSERT_TRUE(sem.registered());
    alignas(16) uint8_t storage[32]{};
    ASSERT_EQ(sem.init(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0), 0u);

    void* const before = *reinterpret_cast<void**>(storage);
    EXPECT_EQ(sem.destroy(reinterpret_cast<uint64_t>(storage), 0, 0, 0, 0, 0), 0u)
        << "semaphore destroys without reporting failure";
    void* const after = *reinterpret_cast<void**>(storage);
    EXPECT_NE(after, before) << "semaphore destroy changes the guest handle";
    EXPECT_LT(reinterpret_cast<uintptr_t>(after), 0x1000u)
        << "the destroyed slot holds a sentinel, not a pointer into a freed object";
}
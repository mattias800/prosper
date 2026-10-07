// sceKernelRaiseException under a stop-the-world BURST on POSIX hosts (#4324).
//
// IL2CPP stops the world by raising every attached thread back to back and only then waiting for
// one acknowledgement per raise. On macOS the request type travels through a pending table keyed
// by the target thread (Darwin has no pthread_sigqueue payload). That table held 16 entries; under
// Rosetta the raiser outran delivery, the 17th raise returned EAGAIN, the guest still counted that
// thread, and il2cpp_stop_gc_world waited forever (Silksong, ~1 boot in 8 unattended).
//
// The targets here block every asynchronous signal before the raises, so every request is still
// pending when the next one is made -- the overflow is forced deterministically rather than left
// to scheduling. Then they unblock and every handler must run exactly once on its own thread.
// A second case raises one blocked thread twice: plain signals coalesce into one delivery, and
// both requests must still run. On Linux RT signals queue, so both cases also pin that platform's
// contract; they fail only on the pre-fix macOS table.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/exc_pending.hpp"

#include <gtest/gtest.h>
#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

using namespace prosper;

namespace {

constexpr uint64_t kGcSuspendType = 0x1e;   // the type IL2CPP installs and raises

struct Target {
    pthread_t thread{};
    std::atomic<bool> ready{false};
    std::atomic<int> handled{0};
    std::atomic<bool> handled_on_self{true};
};

std::atomic<bool> g_release{false};
std::atomic<bool> g_exit{false};
thread_local Target* t_self = nullptr;

// The "guest" handler: handler(type, mcontext*). Counts on the thread it interrupted.
void suspend_handler(uint64_t type, void*) {
    if (!t_self || type != kGcSuspendType) return;
    t_self->handled.fetch_add(1);
    if (!pthread_equal(pthread_self(), t_self->thread)) t_self->handled_on_self = false;
}

sigset_t async_signals() {
    sigset_t set;
    sigfillset(&set);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT}) sigdelset(&set, sig);
    return set;
}

void* target_main(void* arg) {
    auto* self = static_cast<Target*>(arg);
    t_self = self;
    const sigset_t set = async_signals();
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
    self->ready = true;
    while (!g_release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    pthread_sigmask(SIG_UNBLOCK, &set, nullptr);   // every pending request is delivered here
    while (!g_exit) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    t_self = nullptr;
    return nullptr;
}

bool wait_until(const std::function<bool()>& done, std::chrono::seconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

class ExcRaiseBurst : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        install_ = Hle::lookup(nid_hash("sceKernelInstallExceptionHandler"));
        raise_ = Hle::lookup(nid_hash("sceKernelRaiseException"));
        ASSERT_NE(install_, nullptr);
        ASSERT_NE(raise_, nullptr);
        ASSERT_EQ(
            install_(kGcSuspendType, reinterpret_cast<uint64_t>(&suspend_handler), 0, 0, 0, 0), 0u);
        g_release = false;
        g_exit = false;
    }
    void start(std::vector<Target>& targets) {
        for (auto& t : targets) ASSERT_EQ(pthread_create(&t.thread, nullptr, target_main, &t), 0);
        ASSERT_TRUE(wait_until(
            [&] {
                for (auto& t : targets)
                    if (!t.ready) return false;
                return true;
            },
            std::chrono::seconds(10)));
    }
    void finish(std::vector<Target>& targets) {
        g_exit = true;
        for (auto& t : targets) pthread_join(t.thread, nullptr);
    }
    uint64_t raise(const Target& t) {
        return raise_(reinterpret_cast<uint64_t>(t.thread), kGcSuspendType, 0, 0, 0, 0);
    }
    HleFn install_ = nullptr;
    HleFn raise_ = nullptr;
};

TEST_F(ExcRaiseBurst, EveryThreadOfABurstLargerThanTheOldTableIsRaisedAndDelivered) {
    // Comfortably past the old 16-slot table, and past Silksong's observed ~24-thread burst.
    std::vector<Target> targets(64);
    start(targets);
    for (size_t i = 0; i < targets.size(); ++i)
        EXPECT_EQ(raise(targets[i]), 0u) << "raise " << i << " of a pending burst must succeed";
    g_release = true;
    EXPECT_TRUE(wait_until(
        [&] {
            for (auto& t : targets)
                if (t.handled.load() < 1) return false;
            return true;
        },
        std::chrono::seconds(20)))
        << "every raised thread must run its handler";
    for (size_t i = 0; i < targets.size(); ++i) {
        EXPECT_EQ(targets[i].handled.load(), 1) << "thread " << i << " handled its request once";
        EXPECT_TRUE(targets[i].handled_on_self.load()) << "on its own thread";
    }
    finish(targets);
}

TEST_F(ExcRaiseBurst, TwoRequestsThatCoalesceIntoOneSignalBothRun) {
    std::vector<Target> targets(1);
    start(targets);
    EXPECT_EQ(raise(targets[0]), 0u);
    EXPECT_EQ(raise(targets[0]), 0u);
    g_release = true;
    EXPECT_TRUE(
        wait_until([&] { return targets[0].handled.load() >= 2; }, std::chrono::seconds(10)))
        << "a plain signal delivers once for both raises; both requests must still run";
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(targets[0].handled.load(), 2);
    finish(targets);
}

} // namespace

namespace {
// The table itself: capacity, release, and at-most-once hand-out. Distinct fake thread ids well away
// from any live pthread_t (they are only keys here).
TEST(ExcPendingTable, HoldsAFullBurstReleasesUnsentRequestsAndHandsEachOutOnce) {
    constexpr uint64_t kBase = 0x7f0000000000ull;
    std::vector<size_t> handles;
    for (size_t i = 0; i < prosper::kExcPendingSlots; ++i) {
        const size_t h = prosper::exc_pending_put(kBase + i * 0x1000, 0x1e);
        ASSERT_NE(h, 0u) << "slot " << i << " of a full burst";
        handles.push_back(h);
    }
    EXPECT_EQ(prosper::exc_pending_put(kBase + 0xfff000, 0x1e), 0u) << "a full table refuses";
    // A raise whose signal was never sent gives its slot back.
    prosper::exc_pending_release(handles[5], kBase + uint64_t{5} * 0x1000);
    EXPECT_EQ(prosper::exc_pending_take(kBase + uint64_t{5} * 0x1000), -1)
        << "a released request is gone";
    const size_t reused = prosper::exc_pending_put(kBase + 0xfff000, 0x1f);
    EXPECT_NE(reused, 0u) << "and its slot is reusable";
    EXPECT_EQ(prosper::exc_pending_take(kBase + 0xfff000), 0x1f);
    // Each request is handed out exactly once.
    for (size_t i = 0; i < prosper::kExcPendingSlots; ++i) {
        if (i == 5) continue;
        EXPECT_EQ(prosper::exc_pending_take(kBase + i * 0x1000), 0x1e);
        EXPECT_EQ(prosper::exc_pending_take(kBase + i * 0x1000), -1);
    }
    // Releasing after a take is a no-op, not a corruption of someone else's slot.
    const size_t h = prosper::exc_pending_put(kBase, 0x1e);
    EXPECT_EQ(prosper::exc_pending_take(kBase), 0x1e);
    const size_t other = prosper::exc_pending_put(kBase + 0x1000, 0x1e);
    prosper::exc_pending_release(h, kBase);
    EXPECT_EQ(prosper::exc_pending_take(kBase + 0x1000), 0x1e)
        << "the stale release left this alone";
    (void)other;
    // Leave the process-global table empty even if an assertion above failed part-way, so this
    // case cannot poison a --gtest_repeat run or a later case in the same process.
    for (size_t i = 0; i <= prosper::kExcPendingSlots; ++i)
        while (prosper::exc_pending_take(kBase + i * 0x1000) >= 0) {}
    while (prosper::exc_pending_take(kBase + 0xfff000) >= 0) {}
}
TEST(ExcPendingTable, DrainDeliversEveryPendingRequestOnceAndReportsAnEmptyDrainOnce) {
    std::vector<int> seen;
    auto record = [](int type, void* ctx) { static_cast<std::vector<int>*>(ctx)->push_back(type); };
    prosper::exc_pending_drain(0x7001, record, &seen);   // nothing pending: one -1 report
    ASSERT_EQ(seen, std::vector<int>({-1}));
    seen.clear();
    ASSERT_NE(prosper::exc_pending_put(0x7001, 5), 0u);
    ASSERT_NE(prosper::exc_pending_put(0x7001, 9), 0u);
    ASSERT_NE(prosper::exc_pending_put(0x7002, 3), 0u);   // another thread's request is untouched
    prosper::exc_pending_drain(0x7001, record, &seen);
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0] + seen[1], 14);
    EXPECT_EQ(prosper::exc_pending_take(0x7002), 3);
    EXPECT_EQ(prosper::exc_pending_take(0x7001), -1);
}
}   // namespace

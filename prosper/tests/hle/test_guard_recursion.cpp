// test_guard_recursion — #979: __cxa_guard_acquire / std::call_once recursion must not self-deadlock.
//
// The Itanium C++ ABI static-init guard (h_guard_acquire) and std::call_once core (h_execute_once)
// let exactly one thread run the initializer and block the rest. Before #979 the winner tracked no
// owner, so if the initializer re-entered the SAME guard/flag on the SAME thread (a lazy singleton
// whose constructor touches itself), the re-entry saw the busy/running state it had set itself and
// blocked on the shared condvar — a self-deadlock (waiting for itself to release). These tests prove
// (1) same-thread recursion now returns without blocking and without re-running the initializer, and
// (2) the multi-thread one-winner contract (#490) is preserved unchanged.
//
// P() in hle_libc.cpp is the identity map (guest addr == host addr), so a test can hand these HLEs
// real host pointers via the registry (Hle::lookup) and call them directly.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <thread>

using namespace prosper;

namespace {
// Run fn on a fresh detached thread; return true if it finished within `ms`, false if it appears to
// have deadlocked. A detached thread + shared promise means a genuine deadlock leaves a blocked
// thread rather than hanging the whole test (the test reports the failure and moves on); with the
// fix in place nothing ever deadlocks, so no thread is leaked on a passing run.
template <class F>
bool run_bounded(F fn, int ms) {
    auto p = std::make_shared<std::promise<void>>();
    auto fut = p->get_future();
    std::thread([fn, p]() mutable {
        fn();
        p->set_value();
    }).detach();
    return fut.wait_for(std::chrono::milliseconds(ms)) == std::future_status::ready;
}

// A regression here leaves a worker blocked on the shared condvar, and returning from main would
// then run static destructors that observe it -- the hang the pre-GTest version's std::_Exit(1)
// avoided. Report the failure and leave immediately rather than let gtest print a summary the
// process may never finish printing. On a passing run nothing is leaked, so this is never reached.
[[noreturn]] void exit_with_blocked_worker(const char* what) {
    std::fprintf(stderr,
                 "[guard-recursion] %s: a worker is still blocked, so this process cannot run its "
                 "static destructors safely; exiting with the verdict\n",
                 what);
    std::fflush(nullptr);
    std::_Exit(1);
}

uint64_t U(void* p) {
    return (uint64_t)(uintptr_t)p;
}

HleFn g_acquire = nullptr, g_release = nullptr, g_once = nullptr;

// --- std::call_once callback that recursively re-enters call_once on its own flag ---------------
uint64_t g_once_flag_addr = 0;
uint64_t g_once_cb_addr = 0;
std::atomic<int> g_once_cb_calls{0};
std::atomic<uint64_t> g_once_recursive_ret{~0ull};
int once_recursive_cb(void* /*flag*/, void* /*arg*/, void** /*ctx*/) {
    g_once_cb_calls.fetch_add(1);
    // Re-enter call_once on the SAME flag from within its own initializer (guest UB / recursion).
    g_once_recursive_ret.store(g_once(g_once_flag_addr, g_once_cb_addr, 0, 0, 0, 0));
    return 1;  // success
}
// A plain callback that just counts, for the concurrent one-run test.
std::atomic<int> g_once_plain_calls{0};
int once_plain_cb(void*, void*, void**) {
    g_once_plain_calls.fetch_add(1);
    return 1;
}

// The three handlers under test, resolved once through the registry.
void resolve_handlers() {
    register_builtin_hle();
    g_acquire = Hle::lookup(nid_hash("__cxa_guard_acquire"));
    g_release = Hle::lookup(nid_hash("__cxa_guard_release"));
    g_once    = Hle::lookup(nid_hash("_ZSt13_Execute_onceRSt9once_flagPFiPvS1_PS1_ES1_"));
}
}   // namespace

TEST(GuardRecursion, GuardAndCallOnceHandlersAreRegistered) {
    resolve_handlers();
    EXPECT_NE(g_acquire, nullptr) << "__cxa_guard_acquire registered";
    EXPECT_NE(g_release, nullptr) << "__cxa_guard_release registered";
    EXPECT_NE(g_once, nullptr) << "std::_Execute_once (call_once core) registered";
}

TEST(GuardRecursion, SameThreadGuardRecursionReturnsWithoutReRunningTheInitializer) {
    resolve_handlers();
    ASSERT_NE(g_acquire, nullptr);
    ASSERT_NE(g_release, nullptr);
    uint64_t guard = 0;   // byte0=initialized, byte1=busy
    std::atomic<uint64_t> r1{~0ull}, r2{~0ull};
    const bool done = run_bounded(
        [&] {
            r1.store(g_acquire(U(&guard), 0, 0, 0, 0, 0));   // win -> 1, runs initializer
            r2.store(g_acquire(U(&guard), 0, 0, 0, 0, 0));   // recursion -> pre-#979 deadlock
        },
        3000);
    EXPECT_TRUE(done) << "recursive __cxa_guard_acquire returns (no self-deadlock)";
    if (!done) exit_with_blocked_worker("recursive __cxa_guard_acquire");
    EXPECT_EQ(r1.load(), 1u) << "first acquire wins (returns 1)";
    EXPECT_EQ(r2.load(), 0u) << "recursive acquire returns 0 (do not re-run initializer)";
    // Guard was never released, so it must still read uninitialized (byte0 == 0): the recursive
    // return-0 is 'skip re-init', NOT a spurious 'already initialized' completion.
    EXPECT_EQ(guard & 0xff, 0u) << "recursion did not mark the guard initialized";
    if (r1.load() == 1) g_release(U(&guard), 0, 0, 0, 0, 0);   // clean up the in-flight guard
}

TEST(GuardRecursion, ConcurrentGuardRaceStillHasExactlyOneWinner) {
    resolve_handlers();
    ASSERT_NE(g_acquire, nullptr);
    ASSERT_NE(g_release, nullptr);
    // Two threads racing the SAME fresh guard: exactly one wins (1), the other blocks then gets 0.
    // Proves the #490 one-winner contract is unchanged by the owner tracking.
    uint64_t guard = 0;
    std::atomic<int> winners{0}, zeros{0};
    const bool done = run_bounded(
        [&] {
            auto worker = [&] {
                uint64_t r = g_acquire(U(&guard), 0, 0, 0, 0, 0);
                if (r == 1) {
                    winners.fetch_add(1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // hold the init
                    g_release(U(&guard), 0, 0, 0, 0, 0);
                } else if (r == 0) {
                    zeros.fetch_add(1);
                }
            };
            std::thread a(worker), b(worker);
            a.join();
            b.join();
        },
        3000);
    EXPECT_TRUE(done) << "concurrent guard race completes (no deadlock)";
    if (!done) exit_with_blocked_worker("concurrent guard race");
    EXPECT_EQ(winners.load(), 1) << "exactly one thread wins the guard";
    EXPECT_EQ(zeros.load(), 1) << "the other thread returns 0 after release";
}

TEST(GuardRecursion, SameThreadCallOnceRecursionRunsTheCallbackOnceAndReportsDone) {
    resolve_handlers();
    ASSERT_NE(g_once, nullptr);
    uint32_t flag = 0;
    g_once_flag_addr = U(&flag);
    g_once_cb_addr = U((void*)&once_recursive_cb);
    g_once_cb_calls.store(0);
    g_once_recursive_ret.store(~0ull);
    std::atomic<uint64_t> outer{~0ull};
    const bool done = run_bounded(
        [&] { outer.store(g_once(U(&flag), U((void*)&once_recursive_cb), 0, 0, 0, 0)); }, 3000);
    EXPECT_TRUE(done) << "recursive std::call_once returns (no self-deadlock)";
    if (!done) exit_with_blocked_worker("recursive std::call_once");
    EXPECT_EQ(g_once_cb_calls.load(), 1) << "call_once callback runs exactly once under recursion";
    EXPECT_EQ(g_once_recursive_ret.load(), 1u)
        << "recursive call_once reports done (1) without re-running";
    EXPECT_EQ(outer.load(), 1u) << "outer call_once succeeds";
}

TEST(GuardRecursion, ConcurrentCallOnceRaceRunsTheCallbackExactlyOnceForBothThreads) {
    resolve_handlers();
    ASSERT_NE(g_once, nullptr);
    uint32_t flag = 0;
    g_once_plain_calls.store(0);
    std::atomic<int> ones{0};
    const bool done = run_bounded(
        [&] {
            auto worker = [&] {
                uint64_t r = g_once(U(&flag), U((void*)&once_plain_cb), 0, 0, 0, 0);
                if (r == 1) ones.fetch_add(1);
            };
            std::thread a(worker), b(worker);
            a.join();
            b.join();
        },
        3000);
    EXPECT_TRUE(done) << "concurrent call_once race completes (no deadlock)";
    if (!done) exit_with_blocked_worker("concurrent call_once race");
    EXPECT_EQ(g_once_plain_calls.load(), 1)
        << "call_once callback runs exactly once across threads";
    EXPECT_EQ(ones.load(), 2) << "both threads see call_once as done (return 1)";
}
// test_pthread_key_teardown — deleting a pthread key while a thread holds a value must
// drop the value SILENTLY (POSIX; FreeBSD lib/libthr/thread/thr_spec.c _thread_cleanupspecific).
//
// #4319 reported that Windows MinGW winpthreads fired the deleted key's destructor once
// more at thread exit, giving a guest doing TLS teardown a use-after-delete callback on
// Windows but silence everywhere else. Follow-up measurement on a second box (same MinGW
// GCC 15.2 toolchain) shows the delete-vs-thread-exit path is NONDETERMINISTIC there, not
// silently deterministic: one run fired exactly once, two reruns wedged the join entirely
// (no output, processes killed). On this box (Brecht Sanders build, winpthreads 0.5.0) the
// same shape is silent across ~1400 sampled thread exits in every interleaving probed
// (cross-thread delete, same-thread delete, delete-plus-numeric-reuse, multi-worker
// teardown overlap, single-core affinity) — so the race lives in winpthreads' delete/exit
// path and differs by winpthreads build, not by prosper code.
//
// What the disarm in k_key_delete covers, exactly: the FIRING case only. A deleted key's
// thunk is patched to ret, so a destructor a firing build still invokes is a silent no-op.
// It cannot cover the WEDGE case: if the host's key-table mutation deadlocks against its
// own exit-time destructor scan, no prosper-side patch changes that ordering. Fixing the
// wedge needs the host lock ordering fixed (winpthreads upgrade or patch), not more HLE.
//
// The DeletedKey arm therefore does two jobs: it asserts the silence contract (which the
// disarm guarantees on firing builds and the host provides on silent ones), and it treats
// a wedged join as a NAMED FAILURE ("join wedged: probable delete/exit lock race") instead
// of hanging until the ctest TIMEOUT. Ten rounds on fresh keys widen the race window; the
// join of each round runs on a helper host thread with a 10 s deadline, because HLE exposes
// no non-blocking join and no guest-visible API may be added for a test harness.
// Counters are per-arm so a late destructor from a wedged DeletedKey worker cannot leak
// into the LiveKey control.
//
// WHAT EACH ARM KILLS:
//   DeletedKeyDropsValueSilently: pins the silence contract on every host. On a winpthreads
//                                 that still fires, reverting the thunk disarm turns it red
//                                 (calls != 0). On a wedging build it fails named instead of
//                                 hanging. Here it is green with and without the fix; see the
//                                 PR body for the probe evidence.
//   LiveKeyFiresDestructorOnce:   hand-built POSITIVE CONTROL (test rules / CLAUDE.md). It
//                                 proves the counting machinery observes destructor calls at
//                                 all -- without it a silently-broken counter would bless any
//                                 fix with a meaningless zero. A broken hookup (wrong key,
//                                 destructor never registered) fails here, not there.
#include <gtest/gtest.h>
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <pthread.h>
#include <thread>

using namespace prosper;

namespace {

constexpr uintptr_t kHeldValue = 0x1122334455667788ull;
constexpr int kDeleteRounds = 10;
constexpr auto kJoinDeadline = std::chrono::seconds(0);  // TEMP wedge-path check

pthread_key_t g_key{};
std::atomic<unsigned> g_del_calls{0};
std::atomic<unsigned> g_live_calls{0};
std::atomic<uintptr_t> g_live_value{0};
std::atomic<bool> g_ready{false};
std::atomic<bool> g_release{false};

// Trivial on purpose: no setspecific inside, so a live key fires exactly once (no POSIX
// reinstall retry) and the counted value is the one the worker installed.
// Bodies stay noinline outside the sysv_abi shims: inlined unwind code inside a sysv_abi
// frame makes the MinGW assembler reject the object with
// ".seh_handlerdata used outside of .seh_proc block" (#2142).
__attribute__((noinline)) void record_deleted_body(void* value) {
    (void)value;
    g_del_calls.fetch_add(1, std::memory_order_relaxed);
}

__attribute__((noinline)) void record_live_body(void* value) {
    g_live_calls.fetch_add(1, std::memory_order_relaxed);
    g_live_value.store((uintptr_t)value, std::memory_order_relaxed);
}

__attribute__((noinline)) void* hold_value_body() {
    pthread_setspecific(g_key, (void*)kHeldValue);
    g_ready.store(true, std::memory_order_release);
    while (!g_release.load(std::memory_order_acquire)) std::this_thread::yield();
    return nullptr;
}

__attribute__((noinline)) void* set_and_exit_body() {
    pthread_setspecific(g_key, (void*)kHeldValue);
    return nullptr;
}

#ifdef _WIN32
extern "C" __attribute__((sysv_abi)) void teardown_destructor_deleted(void* value) {
    record_deleted_body(value);
}
extern "C" __attribute__((sysv_abi)) void teardown_destructor_live(void* value) {
    record_live_body(value);
}
extern "C" __attribute__((sysv_abi)) void* hold_value_worker(void*) { return hold_value_body(); }
extern "C" __attribute__((sysv_abi)) void* set_and_exit_worker(void*) {
    return set_and_exit_body();
}
#else
extern "C" void teardown_destructor_deleted(void* value) { record_deleted_body(value); }
extern "C" void teardown_destructor_live(void* value) { record_live_body(value); }
extern "C" void* hold_value_worker(void*) { return hold_value_body(); }
extern "C" void* set_and_exit_worker(void*) { return set_and_exit_body(); }
#endif

// Bounded wait: a stuck worker reports a test failure, never a hung suite.
bool wait_for(const std::atomic<bool>& flag) {
    for (int i = 0; i < 5000 && !flag.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return flag.load(std::memory_order_acquire);
}

struct KeyHle {
    HleFn key_create = nullptr;
    HleFn key_delete = nullptr;
    HleFn thread_create = nullptr;
    HleFn thread_join = nullptr;
};

KeyHle resolve_hle() {
    register_builtin_hle();
    KeyHle hle;
    hle.key_create = Hle::lookup(nid_hash("scePthreadKeyCreate"));
    hle.key_delete = Hle::lookup(nid_hash("scePthreadKeyDelete"));
    hle.thread_create = Hle::lookup(nid_hash("scePthreadCreate"));
    hle.thread_join = Hle::lookup(nid_hash("scePthreadJoin"));
    return hle;
}

// HLE exposes no non-blocking join, so the bound lives OUTSIDE the guest path: a helper
// host thread performs the blocking scePthreadJoin while the test thread enforces the
// deadline. Returns false (leaving the joiner detached) when the join wedges, so a host
// delete/exit lock race fails named instead of hanging until the ctest TIMEOUT.
bool bounded_join(const KeyHle& hle, uint64_t thread, uint64_t& join_rc) {
    std::atomic<bool> done{false};
    std::atomic<uint64_t> rc{UINT64_MAX};
    std::thread joiner([&] {
        rc.store(hle.thread_join(thread, 0, 0, 0, 0, 0), std::memory_order_release);
        done.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + kJoinDeadline;
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (!done.load(std::memory_order_acquire)) {
        joiner.detach();
        return false;
    }
    joiner.join();
    join_rc = rc.load(std::memory_order_acquire);
    return true;
}

}   // namespace

TEST(PthreadKeyTeardown, DeletedKeyDropsValueSilently) {
    const KeyHle hle = resolve_hle();
    ASSERT_TRUE(hle.key_create && hle.key_delete && hle.thread_create && hle.thread_join)
        << "key/thread entry points are registered";
    g_del_calls.store(0, std::memory_order_relaxed);

    // Ten rounds on fresh keys: one pass only samples the race once, and the delete/exit
    // window is timing-dependent. Calls accumulate across rounds, so a firing in ANY round
    // fails the final count.
    for (int round = 0; round < kDeleteRounds; ++round) {
        g_ready.store(false, std::memory_order_relaxed);
        g_release.store(false, std::memory_order_relaxed);
        uint32_t key = 0;
        ASSERT_EQ(hle.key_create((uint64_t)(uintptr_t)&key,
                                 (uint64_t)(uintptr_t)&teardown_destructor_deleted, 0, 0, 0,
                                 0),
                  0u)
            << "round " << round << ": scePthreadKeyCreate succeeds";
        g_key = (pthread_key_t)key;

        uint64_t thread = 0;
        ASSERT_EQ(hle.thread_create((uint64_t)(uintptr_t)&thread, 0,
                                    (uint64_t)(uintptr_t)&hold_value_worker, 0, 0, 0),
                  0u)
            << "round " << round << ": scePthreadCreate succeeds";
        EXPECT_TRUE(wait_for(g_ready))
            << "round " << round << ": worker installed its key value within the bound";

        // Delete while the worker still holds a live value: POSIX/FreeBSD drop it silently.
        EXPECT_EQ(hle.key_delete(key, 0, 0, 0, 0, 0), 0u)
            << "round " << round << ": scePthreadKeyDelete succeeds";
        g_release.store(true, std::memory_order_release);
        uint64_t join_rc = UINT64_MAX;
        if (!bounded_join(hle, thread, join_rc)) {
            FAIL() << "round " << round
                   << ": join wedged: probable delete/exit lock race (#4319)";
            return;
        }
        EXPECT_EQ(join_rc, 0u) << "round " << round << ": worker joins";
    }
    EXPECT_EQ(g_del_calls.load(std::memory_order_relaxed), 0u)
        << "a deleted key's destructor must not fire across " << kDeleteRounds
        << " rounds (#4319)";
}

TEST(PthreadKeyTeardown, LiveKeyFiresDestructorOnce) {
    const KeyHle hle = resolve_hle();
    ASSERT_TRUE(hle.key_create && hle.key_delete && hle.thread_create && hle.thread_join)
        << "key/thread entry points are registered";
    g_live_calls.store(0, std::memory_order_relaxed);
    g_live_value.store(0, std::memory_order_relaxed);

    uint32_t key = 0;
    ASSERT_EQ(hle.key_create((uint64_t)(uintptr_t)&key,
                             (uint64_t)(uintptr_t)&teardown_destructor_live, 0, 0, 0, 0),
              0u)
        << "scePthreadKeyCreate succeeds";
    g_key = (pthread_key_t)key;

    uint64_t thread = 0;
    ASSERT_EQ(hle.thread_create((uint64_t)(uintptr_t)&thread, 0,
                                (uint64_t)(uintptr_t)&set_and_exit_worker, 0, 0, 0),
              0u)
        << "scePthreadCreate succeeds";
    EXPECT_EQ(hle.thread_join(thread, 0, 0, 0, 0, 0), 0u) << "worker joins";
    EXPECT_EQ(hle.key_delete(key, 0, 0, 0, 0, 0), 0u) << "scePthreadKeyDelete succeeds";

    // Positive control: the harness observes a live destructor call with its argument, so a
    // zero in DeletedKeyDropsValueSilently is a measured silence, not a deaf counter.
    EXPECT_EQ(g_live_calls.load(std::memory_order_relaxed), 1u)
        << "a live key's destructor fires exactly once";
    EXPECT_EQ(g_live_value.load(std::memory_order_relaxed), (uintptr_t)kHeldValue)
        << "the destructor receives the installed value";
}

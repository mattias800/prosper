// test_pthread_key_teardown — deleting a pthread key while a thread holds a value must
// drop the value SILENTLY (POSIX; FreeBSD lib/libthr/thread/thr_spec.c _thread_cleanupspecific).
//
// #4319 reported that Windows MinGW winpthreads fired the deleted key's destructor once
// more at thread exit, giving a guest doing TLS teardown a use-after-delete callback on
// Windows but silence everywhere else. Measured on the current toolchain (winpthreads 0.5.0,
// GCC 15.2): the host is ALREADY silent in every interleaving probed -- cross-thread delete,
// same-thread delete, and delete-plus-numeric-reuse all report zero calls at raw-host level,
// and this arm reports zero through the full HLE path with the disarm reverted. The upstream
// firing behavior is therefore not reproducible here; k_key_delete still disarms the key's
// destructor thunk (patches its entry to ret) as host-independent insurance, because
// pthread_setspecific only affects the calling thread and pre-clearing other threads' values
// at delete time is impossible -- the thunk is the only lever prosper owns.
//
// WHAT EACH ARM KILLS:
//   DeletedKeyDropsValueSilently: pins the silence contract on every host. On a winpthreads
//                                 that still fires, reverting the thunk disarm turns it red
//                                 (calls == 1). Here it is green with and without the fix;
//                                 see the PR body for the probe evidence.
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

pthread_key_t g_key{};
std::atomic<unsigned> g_calls{0};
std::atomic<uintptr_t> g_last_value{0};
std::atomic<bool> g_ready{false};
std::atomic<bool> g_release{false};

void reset_state() {
    g_calls.store(0, std::memory_order_relaxed);
    g_last_value.store(0, std::memory_order_relaxed);
    g_ready.store(false, std::memory_order_relaxed);
    g_release.store(false, std::memory_order_relaxed);
}

// Trivial on purpose: no setspecific inside, so a live key fires exactly once (no POSIX
// reinstall retry) and the counted value is the one the worker installed.
// Bodies stay noinline outside the sysv_abi shims: inlined unwind code inside a sysv_abi
// frame makes the MinGW assembler reject the object with
// ".seh_handlerdata used outside of .seh_proc block" (#2142).
__attribute__((noinline)) void record_only_body(void* value) {
    g_calls.fetch_add(1, std::memory_order_relaxed);
    g_last_value.store((uintptr_t)value, std::memory_order_relaxed);
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
extern "C" __attribute__((sysv_abi)) void teardown_destructor(void* value) {
    record_only_body(value);
}
extern "C" __attribute__((sysv_abi)) void* hold_value_worker(void*) { return hold_value_body(); }
extern "C" __attribute__((sysv_abi)) void* set_and_exit_worker(void*) {
    return set_and_exit_body();
}
#else
extern "C" void teardown_destructor(void* value) { record_only_body(value); }
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

}   // namespace

TEST(PthreadKeyTeardown, DeletedKeyDropsValueSilently) {
    const KeyHle hle = resolve_hle();
    ASSERT_TRUE(hle.key_create && hle.key_delete && hle.thread_create && hle.thread_join)
        << "key/thread entry points are registered";
    reset_state();

    uint32_t key = 0;
    ASSERT_EQ(hle.key_create((uint64_t)(uintptr_t)&key,
                             (uint64_t)(uintptr_t)&teardown_destructor, 0, 0, 0, 0),
              0u)
        << "scePthreadKeyCreate succeeds";
    g_key = (pthread_key_t)key;

    uint64_t thread = 0;
    ASSERT_EQ(hle.thread_create((uint64_t)(uintptr_t)&thread, 0,
                                (uint64_t)(uintptr_t)&hold_value_worker, 0, 0, 0),
              0u)
        << "scePthreadCreate succeeds";
    EXPECT_TRUE(wait_for(g_ready)) << "worker installed its key value within the bound";

    // Delete while the worker still holds a live value: POSIX/FreeBSD drop it silently.
    EXPECT_EQ(hle.key_delete(key, 0, 0, 0, 0, 0), 0u) << "scePthreadKeyDelete succeeds";
    g_release.store(true, std::memory_order_release);
    EXPECT_EQ(hle.thread_join(thread, 0, 0, 0, 0, 0), 0u) << "worker joins";
    EXPECT_EQ(g_calls.load(std::memory_order_relaxed), 0u)
        << "a deleted key's destructor must not fire (#4319)";
}

TEST(PthreadKeyTeardown, LiveKeyFiresDestructorOnce) {
    const KeyHle hle = resolve_hle();
    ASSERT_TRUE(hle.key_create && hle.key_delete && hle.thread_create && hle.thread_join)
        << "key/thread entry points are registered";
    reset_state();

    uint32_t key = 0;
    ASSERT_EQ(hle.key_create((uint64_t)(uintptr_t)&key,
                             (uint64_t)(uintptr_t)&teardown_destructor, 0, 0, 0, 0),
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
    EXPECT_EQ(g_calls.load(std::memory_order_relaxed), 1u)
        << "a live key's destructor fires exactly once";
    EXPECT_EQ(g_last_value.load(std::memory_order_relaxed), (uintptr_t)kHeldValue)
        << "the destructor receives the installed value";
}

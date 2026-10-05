// Windows winpthreads owns pthread-key teardown, but guest destructors use the PS5 SysV ABI.
// Drive the real HLE key and thread lifecycle and verify both the callback argument and POSIX
// destructor retry when the callback installs another non-null value. Also cover clearing a value:
// winpthreads calls its destructor with nullptr in that case, while POSIX requires no callback.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace prosper;

namespace {
constexpr uintptr_t kFirstValue = 0x1122334455667788ull;
constexpr uintptr_t kSecondValue = 0x8877665544332211ull;

pthread_key_t g_key{};
std::atomic<unsigned> g_calls{0};
std::atomic<uintptr_t> g_values[2]{};
std::atomic<bool> g_delete_host_done{false};
std::atomic<bool> g_release_delete{false};

// #1020: a guest key destructor that calls scePthreadExit. winpthreads runs destructors AFTER
// win_thread_trampoline has returned and cleared its longjmp arm, so before the fix this fell
// through to host pthread_exit and killed the whole process via RtlRaiseStatus -- the same
// death #997 removed for the normal worker path.
pthread_key_t g_exit_key{};
std::atomic<unsigned> g_exit_dtor_calls{0};
std::atomic<bool> g_exit_worker_ran{false};
// Resolved ONCE from the case body, not inside the destructor. A sysv_abi function that also
// contains a C++ call with an exception region makes MinGW's assembler reject the object outright
// with ".seh_handlerdata used outside of .seh_proc block" -- so the guest-ABI callback below must
// stay free of anything needing unwind data. Hle::lookup is not safe to call from there.
HleFn g_exit_fn = nullptr;

void delete_after_host_hook(uint64_t) {
    g_delete_host_done.store(true, std::memory_order_release);
    while (!g_release_delete.load(std::memory_order_acquire)) std::this_thread::yield();
}

extern "C" __attribute__((sysv_abi)) void guest_destructor(void* value) {
    const unsigned call = g_calls.fetch_add(1, std::memory_order_relaxed);
    if (call < 2) g_values[call].store((uintptr_t)value, std::memory_order_relaxed);
    if (call == 0)
        pthread_setspecific(g_key, (void*)kSecondValue);
}

extern "C" __attribute__((sysv_abi)) void* guest_worker(void*) {
    const int result = pthread_setspecific(g_key, (void*)kFirstValue);
    return (void*)(uintptr_t)result;
}

extern "C" __attribute__((sysv_abi)) void* guest_clear_worker(void*) {
    int result = pthread_setspecific(g_key, (void*)kFirstValue);
    if (!result) result = pthread_setspecific(g_key, nullptr);
    return (void*)(uintptr_t)result;
}
extern "C" __attribute__((sysv_abi)) void guest_exiting_destructor(void* value) {
    g_exit_dtor_calls.fetch_add(1, std::memory_order_relaxed);
    (void)value;
    // The call under test: it must not return, and must not take the process with it.
    if (g_exit_fn) g_exit_fn(0, 0, 0, 0, 0, 0);
}

extern "C" __attribute__((sysv_abi)) void* guest_exiting_worker(void*) {
    g_exit_worker_ran.store(true, std::memory_order_release);
    pthread_setspecific(g_exit_key, (void*)kFirstValue);
    return nullptr;
}

// The four entry points under test. ctest runs each case in its own process, but a direct run or
// --gtest_repeat shares one, so every destructor-counting case resets the counters it reads.
struct KeyFamily {
    HleFn key_create = nullptr, key_delete = nullptr, thread_create = nullptr,
          thread_join = nullptr, thread_exit = nullptr;
};
KeyFamily key_family() {
    register_builtin_hle();
    KeyFamily family;
    family.key_create = Hle::lookup(nid_hash("scePthreadKeyCreate"));
    family.key_delete = Hle::lookup(nid_hash("scePthreadKeyDelete"));
    family.thread_create = Hle::lookup(nid_hash("scePthreadCreate"));
    family.thread_join = Hle::lookup(nid_hash("scePthreadJoin"));
    family.thread_exit = Hle::lookup(nid_hash("scePthreadExit"));
    return family;
}
uint64_t addr(void* p) {
    return reinterpret_cast<uint64_t>(p);
}
// The guest-ABI callbacks below carry the SysV attribute, so their address is not convertible to
// void* without the cast. One template serves both the void destructors and the void* workers.
template <class Fn>
uint64_t fn_addr(Fn fn) {
    return reinterpret_cast<uint64_t>(reinterpret_cast<void*>(fn));
}
void reset_destructor_counters() {
    g_calls.store(0, std::memory_order_relaxed);
    g_values[0].store(0, std::memory_order_relaxed);
    g_values[1].store(0, std::memory_order_relaxed);
    g_exit_dtor_calls.store(0, std::memory_order_relaxed);
}
}  // namespace

TEST(WinPthreadKeyDestructor, TheKeyAndThreadEntryPointsAreRegistered) {
    const KeyFamily family = key_family();
    EXPECT_NE(family.key_create, nullptr) << "scePthreadKeyCreate is registered";
    EXPECT_NE(family.key_delete, nullptr) << "scePthreadKeyDelete is registered";
    EXPECT_NE(family.thread_create, nullptr) << "scePthreadCreate is registered";
    EXPECT_NE(family.thread_join, nullptr) << "scePthreadJoin is registered";
}

TEST(WinPthreadKeyDestructor, TheGuestDestructorRunsWithThePs5AbiAndIsRetriedForItsOwnNewValue) {
    const KeyFamily family = key_family();
    reset_destructor_counters();
    ASSERT_NE(family.key_create, nullptr);
    ASSERT_NE(family.key_delete, nullptr);
    ASSERT_NE(family.thread_create, nullptr);
    ASSERT_NE(family.thread_join, nullptr);
    uint32_t key = 0;
    ASSERT_EQ(family.key_create(addr(&key), fn_addr(&guest_destructor), 0, 0, 0, 0), 0u);
    g_key = (pthread_key_t)key;

    uint64_t thread = 0;
    ASSERT_EQ(family.thread_create(addr(&thread), 0, fn_addr(&guest_worker), 0, 0, 0), 0u);
    void* worker_result = (void*)1;
    const uint64_t join_result = family.thread_join(thread, addr(&worker_result), 0, 0, 0, 0);
    const uint64_t delete_result = family.key_delete(key, 0, 0, 0, 0, 0);

    EXPECT_EQ(join_result, 0u) << "scePthreadJoin reports success";
    EXPECT_EQ(worker_result, nullptr) << "the guest worker's pthread_setspecific returned 0";
    EXPECT_EQ(delete_result, 0u) << "scePthreadKeyDelete reports success";
    // POSIX requires the destructor to be called again for the value the first call installed, which
    // is why the count is 2 and the second argument is the callback's OWN new value rather than the
    // original one. A shim that ran the destructor once, or passed the stale value both times, lands
    // here.
    EXPECT_EQ(g_calls.load(std::memory_order_relaxed), 2u)
        << "the destructor ran once per non-null value it was handed";
    EXPECT_EQ(g_values[0].load(std::memory_order_relaxed), kFirstValue)
        << "first destructor call received the worker's value";
    EXPECT_EQ(g_values[1].load(std::memory_order_relaxed), kSecondValue)
        << "the retry received the value the first call installed";
    EXPECT_EQ(win_key_destructor_thunk_count_for_test(), 0u)
        << "every deleted key released its destructor thunk";
}

TEST(WinPthreadKeyDestructor, ADestructorCallingScePthreadExitDoesNotTakeTheProcessDown) {
    // Reaching the assertions below IS the arm. A regression does not fail a check here -- it
    // terminates the process (exit ~0xC00000FF, no output), which is the death #997 removed for the
    // normal worker path and #1020 removed for the destructor path.
    const KeyFamily family = key_family();
    reset_destructor_counters();
    ASSERT_NE(family.key_create, nullptr);
    ASSERT_NE(family.key_delete, nullptr);
    ASSERT_NE(family.thread_create, nullptr);
    ASSERT_NE(family.thread_join, nullptr);
    ASSERT_NE(family.thread_exit, nullptr)
        << "scePthreadExit is registered, or the #1020 arm is vacuous";
    g_exit_fn = family.thread_exit;
    uint32_t exit_key = 0;
    ASSERT_EQ(family.key_create(addr(&exit_key), fn_addr(&guest_exiting_destructor), 0, 0, 0, 0),
              0u);
    g_exit_key = (pthread_key_t)exit_key;
    uint64_t exit_thread = 0;
    ASSERT_EQ(family.thread_create(addr(&exit_thread), 0, fn_addr(&guest_exiting_worker), 0, 0, 0),
              0u);
    EXPECT_EQ(family.thread_join(exit_thread, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(family.key_delete(exit_key, 0, 0, 0, 0, 0), 0u);
    EXPECT_TRUE(g_exit_worker_ran.load(std::memory_order_acquire))
        << "#1020: the worker that installed the exiting key ran";
    EXPECT_GE(g_exit_dtor_calls.load(std::memory_order_relaxed), 1u)
        << "#1020: the destructor that calls scePthreadExit was reached";
    EXPECT_EQ(win_key_destructor_thunk_count_for_test(), 0u)
        << "every deleted key released its destructor thunk";
}

TEST(WinPthreadKeyDestructor, AClearedValueInvokesNoDestructor) {
    // winpthreads calls the destructor with nullptr when a value is cleared, while POSIX requires no
    // callback at all. The HLE has to filter the nullptr case, or every guest that clears a key pays
    // a destructor call it never asked for.
    const KeyFamily family = key_family();
    reset_destructor_counters();
    ASSERT_NE(family.key_create, nullptr);
    ASSERT_NE(family.key_delete, nullptr);
    ASSERT_NE(family.thread_create, nullptr);
    ASSERT_NE(family.thread_join, nullptr);
    uint32_t key = 0;
    ASSERT_EQ(family.key_create(addr(&key), fn_addr(&guest_destructor), 0, 0, 0, 0), 0u);
    g_key = (pthread_key_t)key;

    uint64_t thread = 0;
    ASSERT_EQ(family.thread_create(addr(&thread), 0, fn_addr(&guest_clear_worker), 0, 0, 0), 0u);
    void* worker_result = (void*)1;
    const uint64_t join_result = family.thread_join(thread, addr(&worker_result), 0, 0, 0, 0);
    const uint64_t delete_result = family.key_delete(key, 0, 0, 0, 0, 0);
    EXPECT_EQ(join_result, 0u) << "scePthreadJoin reports success";
    EXPECT_EQ(worker_result, nullptr) << "the guest worker cleared its value and reported 0";
    EXPECT_EQ(delete_result, 0u) << "scePthreadKeyDelete reports success";
    EXPECT_EQ(g_calls.load(std::memory_order_relaxed), 0u)
        << "a cleared pthread key invoked no destructor";
    EXPECT_EQ(win_key_destructor_thunk_count_for_test(), 0u)
        << "every deleted key released its destructor thunk";
}

TEST(WinPthreadKeyDestructor, KeyDeleteAndTheReuseOfItsNumberCannotOverlap) {
    const KeyFamily family = key_family();
    ASSERT_NE(family.key_create, nullptr);
    ASSERT_NE(family.key_delete, nullptr);
    EXPECT_EQ(win_key_destructor_thunk_count_for_test(), 0u)
        << "no key destructor thunk is installed before this case arms one";
    uint32_t old_key = 0;
    ASSERT_EQ(family.key_create(addr(&old_key), fn_addr(&guest_destructor), 0, 0, 0, 0), 0u);

    // Park the delete inside the HLE, after its host-side pthread_key_delete and before it installs
    // the destructor thunk. A create that lands in that window reuses a key number whose destructor
    // state is already gone -- or installs a thunk onto a key another thread is about to destroy.
    g_delete_host_done.store(false, std::memory_order_relaxed);
    g_release_delete.store(false, std::memory_order_relaxed);
    win_set_key_delete_after_host_hook_for_test(&delete_after_host_hook);

    std::atomic<uint64_t> raced_delete_result{UINT64_MAX};
    std::thread deleter([&] {
        raced_delete_result.store(family.key_delete(old_key, 0, 0, 0, 0, 0),
                                  std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!g_delete_host_done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    if (!g_delete_host_done.load(std::memory_order_acquire)) {
        g_release_delete.store(true, std::memory_order_release);
        deleter.join();
        win_set_key_delete_after_host_hook_for_test(nullptr);
        FAIL() << "the delete never reached the post-host hook, so this case proved nothing";
    }

    uint32_t reused_key = UINT32_MAX;
    std::atomic<uint64_t> raced_create_result{UINT64_MAX};
    std::atomic<bool> create_done{false};
    std::thread creator([&] {
        raced_create_result.store(
            family.key_create(addr(&reused_key), fn_addr(&guest_destructor), 0, 0, 0, 0),
            std::memory_order_release);
        create_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool create_escaped_delete_transition = create_done.load(std::memory_order_acquire);
    g_release_delete.store(true, std::memory_order_release);
    deleter.join();
    creator.join();
    win_set_key_delete_after_host_hook_for_test(nullptr);

    const uint64_t create_result = raced_create_result.load(std::memory_order_acquire);
    const uint64_t delete_race_result = raced_delete_result.load(std::memory_order_acquire);
    const uint64_t reused_delete_result =
        create_result == 0 ? family.key_delete(reused_key, 0, 0, 0, 0, 0) : UINT64_MAX;
    EXPECT_FALSE(create_escaped_delete_transition)
        << "the create escaped the parked delete's transition window";
    EXPECT_EQ(delete_race_result, 0u) << "the racing delete reports success";
    EXPECT_EQ(create_result, 0u) << "the racing create reports success";
    EXPECT_EQ(reused_key, old_key) << "the reused key number is the one just deleted";
    EXPECT_EQ(reused_delete_result, 0u) << "the reused key is deletable again";
    EXPECT_EQ(win_key_destructor_thunk_count_for_test(), 0u)
        << "the destructor thunk was uninstalled (tracked="
        << win_key_destructor_thunk_count_for_test() << ")";
}
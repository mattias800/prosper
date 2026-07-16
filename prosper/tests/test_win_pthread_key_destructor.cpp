// Windows winpthreads owns pthread-key teardown, but guest destructors use the PS5 SysV ABI.
// Drive the real HLE key and thread lifecycle and verify both the callback argument and POSIX
// destructor retry when the callback installs another non-null value. Also cover clearing a value:
// winpthreads calls its destructor with nullptr in that case, while POSIX requires no callback.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
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
}

int main() {
    register_builtin_hle();
    HleFn key_create = Hle::lookup(nid_hash("scePthreadKeyCreate"));
    HleFn key_delete = Hle::lookup(nid_hash("scePthreadKeyDelete"));
    HleFn thread_create = Hle::lookup(nid_hash("scePthreadCreate"));
    HleFn thread_join = Hle::lookup(nid_hash("scePthreadJoin"));
    if (!key_create || !key_delete || !thread_create || !thread_join) return 1;

    uint32_t key = 0;
    if (key_create((uint64_t)(uintptr_t)&key,
                   (uint64_t)(uintptr_t)&guest_destructor, 0, 0, 0, 0) != 0)
        return 1;
    g_key = (pthread_key_t)key;

    uint64_t thread = 0;
    if (thread_create((uint64_t)(uintptr_t)&thread, 0,
                      (uint64_t)(uintptr_t)&guest_worker, 0, 0, 0) != 0)
        return 1;
    void* worker_result = (void*)1;
    const uint64_t join_result = thread_join(
        thread, (uint64_t)(uintptr_t)&worker_result, 0, 0, 0, 0);
    const uint64_t delete_result = key_delete(key, 0, 0, 0, 0, 0);

    const bool ok = join_result == 0 && worker_result == nullptr && delete_result == 0 &&
                    g_calls.load(std::memory_order_relaxed) == 2 &&
                    g_values[0].load(std::memory_order_relaxed) == kFirstValue &&
                    g_values[1].load(std::memory_order_relaxed) == kSecondValue;
    if (!ok) {
        std::fprintf(stderr,
                     "pthread key destructor mismatch: join=%llu worker=%p delete=%llu "
                     "calls=%u values=%llx,%llx\n",
                     (unsigned long long)join_result, worker_result,
                     (unsigned long long)delete_result,
                     g_calls.load(std::memory_order_relaxed),
                     (unsigned long long)g_values[0].load(std::memory_order_relaxed),
                     (unsigned long long)g_values[1].load(std::memory_order_relaxed));
        return 1;
    }

    g_calls.store(0, std::memory_order_relaxed);
    g_values[0].store(0, std::memory_order_relaxed);
    g_values[1].store(0, std::memory_order_relaxed);
    if (key_create((uint64_t)(uintptr_t)&key,
                   (uint64_t)(uintptr_t)&guest_destructor, 0, 0, 0, 0) != 0)
        return 1;
    g_key = (pthread_key_t)key;

    thread = 0;
    if (thread_create((uint64_t)(uintptr_t)&thread, 0,
                      (uint64_t)(uintptr_t)&guest_clear_worker, 0, 0, 0) != 0)
        return 1;
    worker_result = (void*)1;
    const uint64_t clear_join_result = thread_join(
        thread, (uint64_t)(uintptr_t)&worker_result, 0, 0, 0, 0);
    const uint64_t clear_delete_result = key_delete(key, 0, 0, 0, 0, 0);
    if (clear_join_result != 0 || worker_result != nullptr || clear_delete_result != 0 ||
        g_calls.load(std::memory_order_relaxed) != 0) {
        std::fprintf(stderr,
                     "cleared pthread key invoked destructor: join=%llu worker=%p delete=%llu "
                     "calls=%u\n",
                     (unsigned long long)clear_join_result, worker_result,
                     (unsigned long long)clear_delete_result,
                     g_calls.load(std::memory_order_relaxed));
        return 1;
    }

    if (win_key_destructor_thunk_count_for_test() != 0) return 1;

    uint32_t old_key = 0;
    if (key_create((uint64_t)(uintptr_t)&old_key,
                   (uint64_t)(uintptr_t)&guest_destructor, 0, 0, 0, 0) != 0)
        return 1;

    g_delete_host_done.store(false, std::memory_order_relaxed);
    g_release_delete.store(false, std::memory_order_relaxed);
    win_set_key_delete_after_host_hook_for_test(&delete_after_host_hook);

    std::atomic<uint64_t> raced_delete_result{UINT64_MAX};
    std::thread deleter([&] {
        raced_delete_result.store(key_delete(old_key, 0, 0, 0, 0, 0),
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
        return 1;
    }

    uint32_t reused_key = UINT32_MAX;
    std::atomic<uint64_t> raced_create_result{UINT64_MAX};
    std::atomic<bool> create_done{false};
    std::thread creator([&] {
        raced_create_result.store(
            key_create((uint64_t)(uintptr_t)&reused_key,
                       (uint64_t)(uintptr_t)&guest_destructor, 0, 0, 0, 0),
            std::memory_order_release);
        create_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool create_escaped_delete_transition =
        create_done.load(std::memory_order_acquire);
    g_release_delete.store(true, std::memory_order_release);
    deleter.join();
    creator.join();
    win_set_key_delete_after_host_hook_for_test(nullptr);

    const uint64_t create_result = raced_create_result.load(std::memory_order_acquire);
    const uint64_t delete_race_result = raced_delete_result.load(std::memory_order_acquire);
    const uint64_t reused_delete_result =
        create_result == 0 ? key_delete(reused_key, 0, 0, 0, 0, 0) : UINT64_MAX;
    if (create_escaped_delete_transition || delete_race_result != 0 || create_result != 0 ||
        reused_key != old_key || reused_delete_result != 0 ||
        win_key_destructor_thunk_count_for_test() != 0) {
        std::fprintf(stderr,
                     "pthread key reuse race: escaped=%d old=%u new=%u delete=%llu "
                     "create=%llu cleanup=%llu tracked=%zu\n",
                     create_escaped_delete_transition, old_key, reused_key,
                     (unsigned long long)delete_race_result,
                     (unsigned long long)create_result,
                     (unsigned long long)reused_delete_result,
                     win_key_destructor_thunk_count_for_test());
        return 1;
    }
    return 0;
}

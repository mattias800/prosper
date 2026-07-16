// Windows winpthreads owns pthread-key teardown, but guest destructors use the PS5 SysV ABI.
// Drive the real HLE key and thread lifecycle and verify both the callback argument and POSIX
// destructor retry when the callback installs another non-null value. Also cover clearing a value:
// winpthreads calls its destructor with nullptr in that case, while POSIX requires no callback.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <pthread.h>
#include <atomic>
#include <cstdint>
#include <cstdio>

using namespace prosper;

namespace {
constexpr uintptr_t kFirstValue = 0x1122334455667788ull;
constexpr uintptr_t kSecondValue = 0x8877665544332211ull;

pthread_key_t g_key{};
std::atomic<unsigned> g_calls{0};
std::atomic<uintptr_t> g_values[2]{};

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
    return 0;
}

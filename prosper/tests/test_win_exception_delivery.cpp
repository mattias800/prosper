// Windows must run sceKernelRaiseException's guest handler on the requested target thread,
// then restore the exact interrupted host context so that thread continues normally.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <pthread.h>
#include <windows.h>
#include <cstdint>
#include <cstdio>

using namespace prosper;

static volatile LONG delivered;
static volatile LONG resumed;
static volatile DWORD worker_tid;
static volatile DWORD handler_tid;
static volatile uint64_t context_rip;
static volatile uint64_t context_rsp;
static volatile uint32_t wait_word;
static volatile uint64_t raise_result;
static volatile LONG delivered_before_fallback;
static volatile LONG block_handler;
static volatile LONG handler_timed_out;
static volatile LONG real_cond_signal_sent;
static volatile LONG multi_wait_mode;
static volatile LONG multi_delivered;
static volatile LONG multi_returned;
static volatile LONG mutex_wait_started;
static volatile LONG mutex_wait_returned;
static HANDLE handler_ack;
static HANDLE handler_resume;
static HANDLE multi_resume;
static pthread_t main_thread;
static void* blocked_cond;
static void* shared_event_flag;
static HleFn wait_on_address;
static HleFn wait_event_flag;
static HleFn raise_exception;
static HleFn cond_broadcast;
static HleFn guest_mutex_lock;
static HleFn guest_mutex_unlock;
static void* blocked_guest_mutex;
static int fails;

#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

extern "C" __attribute__((sysv_abi)) void guest_exception_handler(uint64_t type, void* raw_ctx) {
    auto* ctx = (const uint8_t*)raw_ctx;
    handler_tid = GetCurrentThreadId();
    context_rip = *(const uint64_t*)(ctx + 0xa0);
    context_rsp = *(const uint64_t*)(ctx + 0xf8);
    if (type == 0x1e) {
        if (multi_wait_mode) {
            InterlockedIncrement(&multi_delivered);
            WaitForSingleObject(multi_resume, 2000);
            return;
        }
        wait_word = 1;
        InterlockedExchange(&delivered, 1);
        if (block_handler) {
            SetEvent(handler_ack);
            if (WaitForSingleObject(handler_resume, 500) == WAIT_TIMEOUT)
                InterlockedExchange(&handler_timed_out, 1);
        }
    }
}

static void* worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* shared_event_waiter(void*) {
    wait_event_flag((uint64_t)(uintptr_t)shared_event_flag, 1, 0, 0, 0, 0);
    InterlockedIncrement(&multi_returned);
    return nullptr;
}

static void* guest_mutex_waiter(void*) {
    worker_tid = GetCurrentThreadId();
    InterlockedExchange(&mutex_wait_started, 1);
    guest_mutex_lock((uint64_t)(uintptr_t)&blocked_guest_mutex, 0, 0, 0, 0, 0);
    InterlockedExchange(&mutex_wait_returned, 1);
    guest_mutex_unlock((uint64_t)(uintptr_t)&blocked_guest_mutex, 0, 0, 0, 0, 0);
    return nullptr;
}

extern "C" __attribute__((sysv_abi)) void* detached_guest_worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* raise_to_initial_thread(void*) {
    Sleep(20); // let the process's initial thread enter the blocking HLE wait
    raise_result = raise_exception((uint64_t)main_thread, 0x1e, 0, 0, 0, 0);
    if (raise_result != 0) {
        wait_word = 1;
        WakeByAddressAll((void*)&wait_word);
    }
    return nullptr;
}

static void* raise_to_cond_waiter(void*) {
    Sleep(20);
    raise_result = raise_exception((uint64_t)main_thread, 0x1e, 0, 0, 0, 0);
    WaitForSingleObject(handler_ack, 500);
    SetEvent(handler_resume);
    for (int i = 0; i < 200 && !delivered; ++i) Sleep(1);
    delivered_before_fallback = delivered != 0;
    Sleep(50);
    InterlockedExchange(&real_cond_signal_sent, 1);
    cond_broadcast((uint64_t)(uintptr_t)&blocked_cond, 0, 0, 0, 0, 0);
    return nullptr;
}

int main() {
    std::printf("== test_win_exception_delivery ==\n");
    register_builtin_hle();
    HleFn install = Hle::lookup(nid_hash("sceKernelInstallExceptionHandler"));
    HleFn raise = Hle::lookup(nid_hash("sceKernelRaiseException"));
    raise_exception = raise;
    wait_on_address = Hle::lookup("Hc4CaR6JBL0");
    cond_broadcast = Hle::lookup(nid_hash("scePthreadCondBroadcast"));
    CHECK(install && raise && wait_on_address, "exception and futex HLE functions registered");
    if (!install || !raise || !wait_on_address) return 1;

    CHECK(install(0x1e, (uint64_t)(uintptr_t)&guest_exception_handler, 0, 0, 0, 0) == 0,
          "install guest exception handler");
    pthread_t thread = 0;
    CHECK(pthread_create(&thread, nullptr, worker, nullptr) == 0, "create target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    CHECK(worker_tid != 0, "target thread entered blocking HLE futex");

    uint64_t result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "raise exception to target thread");
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) {
        InterlockedExchange((volatile LONG*)&wait_word, 1);
        WakeByAddressAll((void*)&wait_word);
    }
    pthread_join(thread, nullptr);

    CHECK(delivered != 0, "guest handler executed");
    CHECK(handler_tid == worker_tid, "guest handler executed on requested target");
    CHECK(context_rip != 0 && context_rsp != 0, "interrupted RIP and GC stack pointer captured");
    CHECK(resumed != 0, "target resumed its interrupted context after handler return");

    // The emulator runs the guest entry point on the process's initial thread. Winpthreads assigns
    // that thread an opaque pthread_t too; Unity's GC suspends it on its first collection. Exercise
    // that distinct path in addition to the pthread_create target above.
    delivered = 0; resumed = 0;
    handler_tid = 0; worker_tid = 0;
    context_rip = 0; context_rsp = 0;
    wait_word = 0;
    raise_result = ~0ull;
    main_thread = pthread_self();
    DWORD initial_tid = GetCurrentThreadId();
    pthread_t raiser = 0;
    CHECK(pthread_create(&raiser, nullptr, raise_to_initial_thread, nullptr) == 0,
          "create exception raiser for process initial thread");
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    pthread_join(raiser, nullptr);
    CHECK(raise_result == 0, "raise exception to process initial thread");
    CHECK(delivered != 0, "guest handler executed on process initial thread");
    CHECK(handler_tid == initial_tid, "guest handler used process initial thread identity");
    CHECK(context_rip != 0 && context_rsp != 0, "initial-thread RIP and GC stack pointer captured");

    HleFn mutex_init = Hle::lookup(nid_hash("scePthreadMutexInit"));
    HleFn mutex_lock = Hle::lookup(nid_hash("scePthreadMutexLock"));
    HleFn mutex_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
    guest_mutex_lock = mutex_lock;
    guest_mutex_unlock = mutex_unlock;
    HleFn cond_init = Hle::lookup(nid_hash("scePthreadCondInit"));
    HleFn cond_wait = Hle::lookup(nid_hash("scePthreadCondWait"));
    CHECK(mutex_init && mutex_lock && mutex_unlock && cond_init && cond_wait && cond_broadcast,
          "guest mutex/condition HLE functions registered");
    void* blocked_mutex = nullptr;
    blocked_cond = nullptr;
    mutex_init((uint64_t)(uintptr_t)&blocked_mutex, 0, 0, 0, 0, 0);
    cond_init((uint64_t)(uintptr_t)&blocked_cond, 0, 0, 0, 0, 0);
    delivered = 0;
    delivered_before_fallback = 0;
    block_handler = 1;
    handler_timed_out = 0;
    real_cond_signal_sent = 0;
    handler_ack = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    handler_resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    handler_tid = 0;
    context_rip = 0; context_rsp = 0;
    raise_result = ~0ull;
    main_thread = pthread_self();
    pthread_t cond_raiser = 0;
    mutex_lock((uint64_t)(uintptr_t)&blocked_mutex, 0, 0, 0, 0, 0);
    CHECK(pthread_create(&cond_raiser, nullptr, raise_to_cond_waiter, nullptr) == 0,
          "create exception raiser for guest condition waiter");
    cond_wait((uint64_t)(uintptr_t)&blocked_cond, (uint64_t)(uintptr_t)&blocked_mutex, 0, 0, 0, 0);
    mutex_unlock((uint64_t)(uintptr_t)&blocked_mutex, 0, 0, 0, 0, 0);
    pthread_join(cond_raiser, nullptr);
    CHECK(raise_result == 0, "raise exception while target is in guest condition wait");
    CHECK(delivered_before_fallback != 0, "condition wait interrupted promptly for exception delivery");
    CHECK(handler_tid == initial_tid, "condition-wait exception ran on requested target");
    CHECK(handler_timed_out == 0, "raiser returns while injected handler awaits its acknowledgement");
    CHECK(real_cond_signal_sent != 0, "exception-only wake preserves wait until a guest condition signal");
    block_handler = 0;
    CloseHandle(handler_ack);
    CloseHandle(handler_resume);

    // A GC target can be parked acquiring an ordinary guest mutex, not just a condition/event
    // predicate. Windows must deliver cooperatively without redirecting the native Winpthreads
    // wait and must then resume the original acquisition once the mutex becomes available.
    blocked_guest_mutex = nullptr;
    mutex_init((uint64_t)(uintptr_t)&blocked_guest_mutex, 0, 0, 0, 0, 0);
    mutex_wait_started = 0;
    mutex_wait_returned = 0;
    delivered = 0;
    worker_tid = 0;
    handler_tid = 0;
    mutex_lock((uint64_t)(uintptr_t)&blocked_guest_mutex, 0, 0, 0, 0, 0);
    pthread_t mutex_waiter = 0;
    CHECK(pthread_create(&mutex_waiter, nullptr, guest_mutex_waiter, nullptr) == 0,
          "create blocked guest-mutex waiter");
    for (int i = 0; i < 1000 && !mutex_wait_started; ++i) Sleep(1);
    Sleep(20);
    CHECK(raise((uint64_t)mutex_waiter, 0x1e, 0, 0, 0, 0) == 0,
          "raise exception while target is acquiring guest mutex");
    for (int i = 0; i < 1000 && !delivered; ++i) Sleep(1);
    CHECK(delivered != 0 && handler_tid == worker_tid,
          "guest-mutex wait delivered on requested target thread");
    CHECK(mutex_wait_returned == 0, "target resumes the original mutex wait after handler");
    mutex_unlock((uint64_t)(uintptr_t)&blocked_guest_mutex, 0, 0, 0, 0, 0);
    pthread_join(mutex_waiter, nullptr);
    CHECK(mutex_wait_returned != 0, "guest-mutex waiter returns after real unlock");

    // Unity parks a large worker pool on one event flag, so every waiter shares the same predicate
    // mutex. Cooperative delivery must release that mutex while each handler is parked or only the
    // first worker can acknowledge a stop-the-world cycle.
    HleFn ef_create = Hle::lookup(nid_hash("sceKernelCreateEventFlag"));
    wait_event_flag = Hle::lookup(nid_hash("sceKernelWaitEventFlag"));
    HleFn ef_set = Hle::lookup(nid_hash("sceKernelSetEventFlag"));
    HleFn ef_delete = Hle::lookup(nid_hash("sceKernelDeleteEventFlag"));
    CHECK(ef_create && wait_event_flag && ef_set && ef_delete,
          "guest event-flag HLE functions registered");
    shared_event_flag = nullptr;
    ef_create((uint64_t)(uintptr_t)&shared_event_flag, 0, 0, 0, 0, 0);
    constexpr int kSharedWaiters = 4;
    pthread_t shared_waiters[kSharedWaiters]{};
    multi_delivered = 0;
    multi_returned = 0;
    multi_wait_mode = 1;
    multi_resume = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    for (auto& waiter : shared_waiters)
        CHECK(pthread_create(&waiter, nullptr, shared_event_waiter, nullptr) == 0,
              "create shared event-flag waiter");
    Sleep(40);
    for (auto waiter : shared_waiters)
        CHECK(raise((uint64_t)waiter, 0x1e, 0, 0, 0, 0) == 0,
              "raise exception to shared event-flag waiter");
    for (int i = 0; i < 1000 && multi_delivered != kSharedWaiters; ++i) Sleep(1);
    CHECK(multi_delivered == kSharedWaiters,
          "all shared event-flag waiters enter their handlers concurrently");
    SetEvent(multi_resume);
    multi_wait_mode = 0;
    ef_set((uint64_t)(uintptr_t)shared_event_flag, 1, 0, 0, 0, 0);
    for (auto waiter : shared_waiters) pthread_join(waiter, nullptr);
    CHECK(multi_returned == kSharedWaiters,
          "all event-flag waiters resume their original wait and return");
    ef_delete((uint64_t)(uintptr_t)shared_event_flag, 0, 0, 0, 0, 0);
    CloseHandle(multi_resume);

    // Winpthreads may discard its lookup record for a detached pthread_t even while that worker is
    // still alive. Guest runtimes retain the opaque handle and later suspend that worker for GC, so
    // the HLE trampoline must retain its own duplicated native handle until the worker actually exits.
    HleFn attr_init = Hle::lookup(nid_hash("scePthreadAttrInit"));
    HleFn attr_setdetach = Hle::lookup(nid_hash("scePthreadAttrSetdetachstate"));
    HleFn attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
    HleFn guest_create = Hle::lookup(nid_hash("scePthreadCreate"));
    void* detached_attr = nullptr;
    uint64_t detached_thread = 0;
    delivered = 0; resumed = 0; worker_tid = 0; handler_tid = 0; wait_word = 0;
    attr_init((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
    attr_setdetach((uint64_t)(uintptr_t)&detached_attr, 1, 0, 0, 0, 0);
    CHECK(guest_create((uint64_t)(uintptr_t)&detached_thread,
                       (uint64_t)(uintptr_t)&detached_attr,
                       (uint64_t)(uintptr_t)&detached_guest_worker, 0, 0, 0) == 0,
          "create detached guest worker through HLE trampoline");
    attr_destroy((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    CHECK(worker_tid != 0, "detached guest worker entered blocking HLE futex");
    uint64_t detached_result = raise(detached_thread, 0x1e, 0, 0, 0, 0);
    CHECK(detached_result == 0, "raise exception to live detached guest worker");
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    CHECK(delivered != 0 && handler_tid == worker_tid,
          "detached worker received exception on its native target thread");
    CHECK(resumed != 0, "detached worker resumed and exited normally");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

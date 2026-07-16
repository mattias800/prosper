// Windows must run sceKernelRaiseException's guest handler on the requested target thread,
// then restore the exact interrupted host context so that thread continues normally.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sync_futex.hpp"
#include <pthread.h>
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static volatile LONG delivered;
static volatile LONG resumed;
static volatile DWORD worker_tid;
static volatile DWORD handler_tid;
static volatile uint64_t context_rip;
static volatile uint64_t context_rsp;
static volatile uint64_t handler_rsp;
static volatile uint32_t wait_word;
static volatile LONG cond_ready;
static volatile LONG mutex_ready;
static volatile LONG hold_handler;
static volatile LONG release_handler;
static volatile LONG repeat_stage;
static volatile LONG avx_stop;
static volatile LONG avx_test_active;
static volatile LONG gpr_ready;
static volatile LONG gpr_stop;
static volatile uint64_t gpr_observed;
static volatile LONG prewait_ready;
static volatile LONG prewait_release;
static volatile LONG nested_wait_enabled;
static volatile LONG nested_wait_ready;
static volatile LONG nested_wait_release;
static volatile LONG nested_wait_returned;
static volatile LONG nested_wait_finish;
static constexpr uint64_t kGprExpected = 0xd3a5f17c2468be90ull;
alignas(32) static uint8_t avx_expected[32];
alignas(32) static uint8_t avx_observed[32];
static HleFn wait_on_address;
static pthread_mutex_t wait_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wait_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t blocked_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t nested_wait_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t nested_wait_cond = PTHREAD_COND_INITIALIZER;
static int fails;

#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

extern "C" __attribute__((sysv_abi)) void guest_exception_handler(uint64_t type, void* raw_ctx) {
    uint64_t stack_marker = 0;
    auto* ctx = (const uint8_t*)raw_ctx;
    handler_tid = GetCurrentThreadId();
    context_rip = *(const uint64_t*)(ctx + 0xa0);
    context_rsp = *(const uint64_t*)(ctx + 0xf8);
    handler_rsp = (uint64_t)(uintptr_t)&stack_marker;
    if (type == 0x1e) {
        wait_word = 1;
        InterlockedIncrement(&delivered);
        if (avx_test_active)
            __asm__ volatile("vpxor %%ymm0, %%ymm0, %%ymm0" ::: "ymm0");
        while (hold_handler && !release_handler) Sleep(1);
        if (nested_wait_enabled) {
            pthread_mutex_lock(&nested_wait_mutex);
            InterlockedExchange(&nested_wait_ready, 1);
            while (!nested_wait_release)
                interruptible_cond_wait(&nested_wait_cond, &nested_wait_mutex);
            pthread_mutex_unlock(&nested_wait_mutex);
            InterlockedExchange(&nested_wait_returned, 1);
            while (!nested_wait_finish) Sleep(1);
        }
    }
}

static void* worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* cond_worker(void*) {
    worker_tid = GetCurrentThreadId();
    pthread_mutex_lock(&wait_mutex);
    InterlockedExchange(&cond_ready, 1);
    interruptible_cond_wait(&wait_cond, &wait_mutex);
    pthread_mutex_unlock(&wait_mutex);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* mutex_worker(void*) {
    worker_tid = GetCurrentThreadId();
    InterlockedExchange(&mutex_ready, 1);
    const int result = interruptible_mutex_lock(&blocked_mutex);
    if (result == 0) pthread_mutex_unlock(&blocked_mutex);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* prewait_worker(void*) {
    worker_tid = GetCurrentThreadId();
    InterlockedExchange(&prewait_ready, 1);
    while (!prewait_release) Sleep(1);
    pthread_mutex_lock(&wait_mutex);
    InterlockedExchange(&cond_ready, 1);
    interruptible_cond_wait(&wait_cond, &wait_mutex);
    pthread_mutex_unlock(&wait_mutex);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* repeat_worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    wait_word = 0;
    InterlockedExchange(&repeat_stage, 1);
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

extern "C" __attribute__((sysv_abi)) void* detached_guest_worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* avx_worker(void*) {
    worker_tid = GetCurrentThreadId();
    __asm__ volatile(
        "vmovdqu (%0), %%ymm0\n"
        "1:\n"
        "cmpl $0, (%1)\n"
        "je 1b\n"
        "vmovdqu %%ymm0, (%2)\n"
        "vzeroupper\n"
        :
        : "r"(avx_expected), "r"(&avx_stop), "r"(avx_observed)
        : "ymm0", "memory", "cc");
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* gpr_worker(void*) {
    worker_tid = GetCurrentThreadId();
    __asm__ volatile(
        "movq %[expected], %%r12\n"
        "movl $1, (%[ready])\n"
        "1:\n"
        "cmpl $0, (%[stop])\n"
        "je 1b\n"
        "movq %%r12, (%[observed])\n"
        :
        : [expected] "r"(kGprExpected), [ready] "r"(&gpr_ready),
          [stop] "r"(&gpr_stop), [observed] "r"(&gpr_observed)
        : "r12", "memory", "cc");
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void reset_delivery_state() {
    delivered = 0;
    resumed = 0;
    worker_tid = 0;
    handler_tid = 0;
    context_rip = 0;
    context_rsp = 0;
    handler_rsp = 0;
    wait_word = 0;
    cond_ready = 0;
    mutex_ready = 0;
    hold_handler = 0;
    release_handler = 0;
    repeat_stage = 0;
    avx_stop = 0;
    avx_test_active = 0;
    gpr_ready = 0;
    gpr_stop = 0;
    gpr_observed = 0;
    prewait_ready = 0;
    prewait_release = 0;
    nested_wait_enabled = 0;
    nested_wait_ready = 0;
    nested_wait_release = 0;
    nested_wait_returned = 0;
    nested_wait_finish = 0;
    memset(avx_observed, 0, sizeof avx_observed);
}

int main() {
    std::printf("== test_win_exception_delivery ==\n");
    // Exercise the forced-CONTEXT compatibility path first; production Windows runs use cooperative
    // safe points by default. The final cases below remove this override.
    _putenv_s("PROSPER_WIN_LEGACY_EXC", "1");
    register_builtin_hle();
    HleFn install = Hle::lookup(nid_hash("sceKernelInstallExceptionHandler"));
    HleFn raise = Hle::lookup(nid_hash("sceKernelRaiseException"));
    wait_on_address = Hle::lookup("Hc4CaR6JBL0");
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
    const uint64_t stack_gap = handler_rsp > context_rsp ? handler_rsp - context_rsp
                                                         : context_rsp - handler_rsp;
    CHECK(stack_gap > 64 * 1024, "guest handler ran on a dedicated exception stack");
    CHECK(resumed != 0, "target resumed its interrupted context after handler return");

    reset_delivery_state();
    pthread_mutex_lock(&wait_mutex);
    CHECK(pthread_create(&thread, nullptr, cond_worker, nullptr) == 0,
          "create pthread-condition target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    pthread_mutex_unlock(&wait_mutex);
    for (int i = 0; i < 1000 && !cond_ready; ++i) Sleep(1);
    // The worker holds wait_mutex until interruptible_cond_wait has registered its pthread handle
    // and atomically released the mutex inside pthread_cond_wait. Acquiring it here closes the race
    // between the ready marker and the actual blocking wait.
    pthread_mutex_lock(&wait_mutex);
    pthread_mutex_unlock(&wait_mutex);
    CHECK(worker_tid != 0 && cond_ready != 0, "target entered registered pthread-condition wait");

    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "raise exception to pthread-condition target");
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) interruptible_cond_broadcast(&wait_cond);
    pthread_join(thread, nullptr);

    CHECK(delivered != 0, "guest handler executed from pthread-condition wait");
    CHECK(handler_tid == worker_tid, "pthread-condition handler ran on requested target");
    CHECK(context_rip != 0 && context_rsp != 0,
          "pthread-condition interruption captured RIP and GC stack pointer");
    CHECK(resumed != 0, "pthread-condition target resumed after handler return");

    reset_delivery_state();
    pthread_mutex_lock(&blocked_mutex);
    CHECK(pthread_create(&thread, nullptr, mutex_worker, nullptr) == 0,
          "create pthread-mutex target thread");
    for (int i = 0; i < 1000 && !mutex_ready; ++i) Sleep(1);
    CHECK(worker_tid != 0 && mutex_ready != 0, "target entered contended pthread-mutex lock");
    Sleep(10);

    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "raise exception to pthread-mutex target");
    for (int i = 0; i < 2000 && !delivered; ++i) Sleep(1);
    CHECK(delivered != 0, "guest handler interrupted contended pthread-mutex lock");
    CHECK(handler_tid == worker_tid, "pthread-mutex handler ran on requested target");
    pthread_mutex_unlock(&blocked_mutex);
    pthread_join(thread, nullptr);
    CHECK(resumed != 0, "pthread-mutex target resumed and acquired lock after handler return");

    reset_delivery_state();
    hold_handler = 1;
    CHECK(pthread_create(&thread, nullptr, worker, nullptr) == 0,
          "create nested-delivery target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    CHECK(worker_tid != 0, "nested-delivery target entered blocking wait");
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "start first exception delivery");
    for (int i = 0; i < 2000 && !delivered; ++i) Sleep(1);
    CHECK(delivered != 0, "first handler is active on its exception stack");
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0x80020010ull, "overlapping delivery to one target is rejected as EBUSY");
    InterlockedExchange(&release_handler, 1);
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) WakeByAddressAll((void*)&wait_word);
    pthread_join(thread, nullptr);
    CHECK(resumed != 0, "target resumes cleanly after serialized delivery");

    reset_delivery_state();
    CHECK(pthread_create(&thread, nullptr, repeat_worker, nullptr) == 0,
          "create repeated-delivery target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    CHECK(worker_tid != 0, "repeated-delivery target entered first wait");
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "start first serialized delivery");
    for (int i = 0; i < 2000 && !repeat_stage; ++i) Sleep(1);
    CHECK(repeat_stage == 1, "first delivery restored the target's original context");
    Sleep(10);
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "reuse exception stack after restored context is observed");
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) WakeByAddressAll((void*)&wait_word);
    pthread_join(thread, nullptr);
    CHECK(delivered == 2, "both serialized exception handlers executed");
    CHECK(resumed != 0, "target resumes after repeated exception delivery");

    reset_delivery_state();
    if (GetEnabledXStateFeatures() & XSTATE_MASK_AVX) {
        for (size_t i = 0; i < sizeof avx_expected; i++)
            avx_expected[i] = static_cast<uint8_t>(0x31u + i * 7u);
        avx_test_active = 1;
        CHECK(pthread_create(&thread, nullptr, avx_worker, nullptr) == 0,
              "create AVX-state target thread");
        for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
        CHECK(worker_tid != 0, "AVX-state target entered register-live loop");
        result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
        CHECK(result == 0, "deliver exception while YMM register is live");
        for (int i = 0; i < 2000 && !delivered; ++i) Sleep(1);
        InterlockedExchange(&avx_stop, 1);
        pthread_join(thread, nullptr);
        CHECK(memcmp(avx_expected, avx_observed, sizeof avx_expected) == 0,
              "extended AVX state survives injected exception handler");
        CHECK(resumed != 0, "AVX-state target resumes after delivery");
    } else {
        std::printf("  [skip] AVX XSTATE is not enabled on this host\n");
    }

    reset_delivery_state();
    CHECK(pthread_create(&thread, nullptr, gpr_worker, nullptr) == 0,
          "create nonvolatile-register target thread");
    for (int i = 0; i < 1000 && !gpr_ready; ++i) Sleep(1);
    CHECK(worker_tid != 0 && gpr_ready != 0, "nonvolatile R12 pattern is live");
    bool all_deliveries = true;
    for (LONG i = 1; i <= 100; ++i) {
        result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
        if (result != 0) { all_deliveries = false; break; }
        for (int spin = 0; spin < 2000 && delivered < i; ++spin) Sleep(1);
        if (delivered < i) { all_deliveries = false; break; }
    }
    InterlockedExchange(&gpr_stop, 1);
    pthread_join(thread, nullptr);
    CHECK(all_deliveries && delivered == 100,
          "100 serialized deliveries complete while R12 is live");
    CHECK(gpr_observed == kGprExpected,
          "nonvolatile R12 survives repeated context redirection");
    CHECK(resumed != 0, "nonvolatile-register target resumes after stress delivery");

    reset_delivery_state();
    _putenv_s("PROSPER_WIN_LEGACY_EXC", "");
    _putenv_s("PROSPER_WIN_COOPERATIVE_EXC", "1");
    pthread_mutex_lock(&wait_mutex);
    CHECK(pthread_create(&thread, nullptr, cond_worker, nullptr) == 0,
          "create cooperative-delivery target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    pthread_mutex_unlock(&wait_mutex);
    for (int i = 0; i < 1000 && !cond_ready; ++i) Sleep(1);
    pthread_mutex_lock(&wait_mutex);
    pthread_mutex_unlock(&wait_mutex);
    CHECK(worker_tid != 0 && cond_ready != 0,
          "cooperative target entered registered wait");

    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "queue cooperative exception delivery");
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) interruptible_cond_broadcast(&wait_cond);
    pthread_join(thread, nullptr);
    _putenv_s("PROSPER_WIN_COOPERATIVE_EXC", "");

    const uintptr_t cooperative_checkpoint =
        (uintptr_t)&dispatch_pending_guest_exception;
    const uint64_t cooperative_rip_gap = context_rip > cooperative_checkpoint
        ? context_rip - cooperative_checkpoint : cooperative_checkpoint - context_rip;
    CHECK(delivered == 1, "cooperative guest handler executed exactly once");
    CHECK(handler_tid == worker_tid, "cooperative handler ran on requested target");
    CHECK(cooperative_rip_gap < 64 * 1024,
          "context was captured at the cooperative wait checkpoint");
    const uint64_t cooperative_stack_gap = handler_rsp > context_rsp
        ? handler_rsp - context_rsp : context_rsp - handler_rsp;
    CHECK(cooperative_stack_gap > 64 * 1024,
          "cooperative handler ran on a dedicated exception stack");
    CHECK(resumed != 0, "cooperative target returned normally from its wait wrapper");

    reset_delivery_state();
    nested_wait_enabled = 1;
    pthread_mutex_lock(&wait_mutex);
    CHECK(pthread_create(&thread, nullptr, cond_worker, nullptr) == 0,
          "create nested-wait cooperative target thread");
    for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
    pthread_mutex_unlock(&wait_mutex);
    for (int i = 0; i < 1000 && !cond_ready; ++i) Sleep(1);
    pthread_mutex_lock(&wait_mutex);
    pthread_mutex_unlock(&wait_mutex);
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "deliver cooperative exception whose handler performs a nested wait");
    for (int i = 0; i < 2000 && !nested_wait_ready; ++i) Sleep(1);
    CHECK(nested_wait_ready != 0, "guest handler entered its nested semaphore-style wait");
    InterlockedExchange(&nested_wait_release, 1);
    interruptible_cond_broadcast(&nested_wait_cond);
    for (int i = 0; i < 2000 && !nested_wait_returned; ++i) Sleep(1);
    CHECK(nested_wait_returned != 0, "nested wait returned while handler remains active");
    CHECK(interrupt_guest_wait((uint64_t)thread),
          "outer wait remains registered after nested wait unregisters");
    InterlockedExchange(&nested_wait_finish, 1);
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) interruptible_cond_broadcast(&wait_cond);
    pthread_join(thread, nullptr);
    CHECK(delivered == 1, "nested-wait cooperative handler executed exactly once");
    CHECK(resumed != 0, "target resumed after nested-wait handler returned");

    reset_delivery_state();
    CHECK(pthread_create(&thread, nullptr, prewait_worker, nullptr) == 0,
          "create queue-before-wait target thread");
    for (int i = 0; i < 1000 && !prewait_ready; ++i) Sleep(1);
    CHECK(prewait_ready != 0, "target paused before registering its wait");
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "queue cooperative delivery before wait registration");
    InterlockedExchange(&prewait_release, 1);
    for (int i = 0; i < 2000 && !delivered; ++i) Sleep(1);
    interruptible_cond_broadcast(&wait_cond);
    for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
    if (!resumed) interruptible_cond_broadcast(&wait_cond);
    pthread_join(thread, nullptr);

    CHECK(delivered == 1, "pre-sleep checkpoint accepted the queued delivery once");
    CHECK(handler_tid == worker_tid, "pre-sleep handler ran on requested target");
    CHECK(resumed != 0, "target continued through its wait after pre-sleep delivery");
    _putenv_s("PROSPER_WIN_COOPERATIVE_EXC", "");

    // Winpthreads may discard its pthread_t lookup for a detached worker before that worker exits.
    // Guest runtimes retain the opaque id and may still stop the worker for GC, so both cooperative
    // wait lookup and the legacy/fallback HANDLE path must follow the worker's actual lifetime.
    reset_delivery_state();
    HleFn attr_init = Hle::lookup(nid_hash("scePthreadAttrInit"));
    HleFn attr_setdetach = Hle::lookup(nid_hash("scePthreadAttrSetdetachstate"));
    HleFn attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
    HleFn guest_create = Hle::lookup(nid_hash("scePthreadCreate"));
    CHECK(attr_init && attr_setdetach && attr_destroy && guest_create,
          "guest detached-thread HLE functions registered");
    void* detached_attr = nullptr;
    uint64_t detached_thread = 0;
    if (attr_init && attr_setdetach && attr_destroy && guest_create) {
        attr_init((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
        attr_setdetach((uint64_t)(uintptr_t)&detached_attr, 1, 0, 0, 0, 0);
        CHECK(guest_create((uint64_t)(uintptr_t)&detached_thread,
                           (uint64_t)(uintptr_t)&detached_attr,
                           (uint64_t)(uintptr_t)&detached_guest_worker, 0, 0, 0) == 0,
              "create detached guest worker through HLE trampoline");
        attr_destroy((uint64_t)(uintptr_t)&detached_attr, 0, 0, 0, 0, 0);
        for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
        CHECK(worker_tid != 0, "detached guest worker entered blocking HLE futex");
        result = raise(detached_thread, 0x1e, 0, 0, 0, 0);
        CHECK(result == 0, "raise exception to live detached guest worker");
        for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
        CHECK(delivered == 1 && handler_tid == worker_tid,
              "detached worker received one exception on its native target thread");
        CHECK(resumed != 0, "detached worker resumed and exited normally");
        Sleep(20); // allow the detached trampoline to finish its lifetime-bound cleanup
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

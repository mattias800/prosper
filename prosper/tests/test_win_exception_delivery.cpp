// Windows must run sceKernelRaiseException's guest handler on the requested target thread,
// then restore the exact interrupted host context so that thread continues normally.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sync_futex.hpp"
#include "../src/host/exec_image.hpp"
#include "../src/host/sse4a.hpp"
#include <pthread.h>
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
static volatile DWORD registry_tid;
static volatile LONG registry_done;
static volatile LONG registry_torn;
static volatile LONG registry_observations;
static volatile DWORD pin_boundary_tid;
static volatile LONG pin_boundary_ready;
static volatile LONG pin_boundary_transition;
static volatile LONG pin_boundary_republished;
static volatile LONG pin_boundary_finish;
static GuestWaitKind classified_wait_kind = GuestWaitKind::ConditionSequence;
static uintptr_t classified_wait_source;
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

struct CondDestroyRace {
    pthread_mutex_t mutex{};
    pthread_cond_t cond{};
    volatile LONG ready = 0;
    volatile LONG returned = 0;
};

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

static void* classified_cond_worker(void*) {
    worker_tid = GetCurrentThreadId();
    pthread_mutex_lock(&wait_mutex);
    InterlockedExchange(&cond_ready, 1);
    interruptible_cond_wait(&wait_cond, &wait_mutex,
                            classified_wait_kind, classified_wait_source);
    pthread_mutex_unlock(&wait_mutex);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

static void* registry_worker(void*) {
    const DWORD native_id = GetCurrentThreadId();
    InterlockedExchange((volatile LONG*)&registry_tid, (LONG)native_id);
    constexpr uint64_t pthread_a = 0x1111222233334444ull;
    constexpr uint64_t pthread_b = 0xaaaabbbbccccddddull;
    constexpr uintptr_t stack_a = 0x11110000u;
    constexpr uintptr_t stack_b = 0x22220000u;
    constexpr size_t size_a = 0x1111u;
    constexpr size_t size_b = 0x2222u;
    for (unsigned i = 0; i < 50000; ++i) {
        trace_guest_thread_lifecycle(true, pthread_a, native_id, (void*)stack_a, size_a);
        if (i == 0) Sleep(10); // guarantee the observer sees at least one valid published interval
        if ((i & 31u) == 0) SwitchToThread();
        trace_guest_thread_lifecycle(false, pthread_a, native_id, (void*)stack_a, size_a);
        trace_guest_thread_lifecycle(true, pthread_b, native_id, (void*)stack_b, size_b);
        if ((i & 31u) == 0) SwitchToThread();
        trace_guest_thread_lifecycle(false, pthread_b, native_id, (void*)stack_b, size_b);
    }
    InterlockedExchange(&registry_done, 1);
    return nullptr;
}

static void* pin_boundary_worker(void*) {
    const DWORD native_id = GetCurrentThreadId();
    InterlockedExchange((volatile LONG*)&pin_boundary_tid, (LONG)native_id);
    trace_guest_thread_lifecycle(true, 0x1111222233334444ull, native_id,
                                 (void*)0x11110000u, 0x1111u);
    InterlockedExchange(&pin_boundary_ready, 1);
    while (!pin_boundary_transition) Sleep(1);
    trace_guest_thread_lifecycle(false, 0x1111222233334444ull, native_id,
                                 (void*)0x11110000u, 0x1111u);
    trace_guest_thread_lifecycle(true, 0xaaaabbbbccccddddull, native_id,
                                 (void*)0x22220000u, 0x2222u);
    InterlockedExchange(&pin_boundary_republished, 1);
    while (!pin_boundary_finish) Sleep(1);
    trace_guest_thread_lifecycle(false, 0xaaaabbbbccccddddull, native_id,
                                 (void*)0x22220000u, 0x2222u);
    return nullptr;
}

static void pin_boundary_hook(uint32_t native_id, void*) {
    if (native_id != pin_boundary_tid) return;
    InterlockedExchange(&pin_boundary_transition, 1);
    for (int i = 0; i < 2000 && !pin_boundary_republished; ++i) Sleep(1);
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

static void* cond_destroy_race_worker(void* raw) {
    auto* race = static_cast<CondDestroyRace*>(raw);
    pthread_mutex_lock(&race->mutex);
    InterlockedExchange(&race->ready, 1);
    interruptible_cond_wait(&race->cond, &race->mutex);
    pthread_mutex_unlock(&race->mutex);
    InterlockedExchange(&race->returned, 1);
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

static void* exited_worker(void*) {
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

static void test_sse4a_fastpath() {
    constexpr uint64_t base = 0x410000000ull;
    constexpr size_t code_size = 85;
    constexpr size_t sse4a_offset = 45;
    constexpr size_t short_code_offset = 0x200;
    constexpr size_t short_sse4a_offset = short_code_offset + 34;
    constexpr size_t short_successor_wrapper_offset = 0x3c0;
    constexpr size_t chain_code_offset = 0x300;
    constexpr size_t chain_sse4a_offset = chain_code_offset + 28;
    constexpr size_t chain_second_wrapper_offset = 0x380;
    constexpr size_t data_offset = 0x100;
    const uint8_t code[code_size] = {
        0x48,0x83,0xec,0x18,                         // sub rsp,0x18
        0xf3,0x44,0x0f,0x7f,0x04,0x24,               // save nonvolatile xmm8
        0x48,0xba, 0xef,0xcd,0xab,0x89,0x67,0x45,0x23,0x01, // movabs rdx,red-zone sentinel
        0x48,0x89,0x54,0x24,0xf8,                    // mov [rsp-8],rdx
        0x48,0xb8, 0,0,0,0,0,0,0,0,                 // movabs rax,data
        0xf3,0x44,0x0f,0x6f,0x00,                    // movdqu xmm8,[rax] (control)
        0xf3,0x0f,0x6f,0x48,0x10,                    // movdqu xmm1,[rax+0x10] (value)
        0x66,0x41,0x0f,0x79,0xc8,                    // extrq xmm1,xmm8 (five-byte form)
        0xf3,0x0f,0x7f,0x48,0x30,                    // movdqu [rax+0x30],xmm1
        0x66,0x48,0x0f,0x7e,0xc8,                    // movq rax,xmm1
        0x48,0x39,0x54,0x24,0xf8,                    // cmp [rsp-8],rdx
        0x74,0x07,                                    // je .red_zone_ok
        0x48,0xc7,0xc0,0xff,0xff,0xff,0xff,          // mov rax,-1
        0xf3,0x44,0x0f,0x6f,0x04,0x24,               // restore xmm8
        0x48,0x83,0xc4,0x18,                         // add rsp,0x18
        0xc3                                           // ret
    };
    const uint8_t short_code[] = {
        0x48,0xb8, 0,0,0,0,0,0,0,0,                 // movabs rax,data
        0x48,0xba, 0xef,0xcd,0xab,0x89,0x67,0x45,0x23,0x01, // movabs rdx,red-zone sentinel
        0x48,0x89,0x54,0x24,0xf8,                    // mov [rsp-8],rdx
        0xf3,0x0f,0x6f,0x00,                          // movdqu xmm0,[rax] (control)
        0xf3,0x0f,0x6f,0x48,0x10,                     // movdqu xmm1,[rax+0x10] (value)
        0x66,0x0f,0x79,0xc8,                           // extrq xmm1,xmm0 (four-byte form)
        0xc5,0xf9,0x6f,0xd1,                           // vmovdqa xmm2,xmm1 (four-byte successor)
        0xc5,0xf9,0x6f,0xda,                           // vmovdqa xmm3,xmm2 (four-byte successor)
        0xc5,0xf9,0x6f,0xe3,                           // vmovdqa xmm4,xmm3 (four-byte successor)
        0xc4,0xe1,0x79,0x6f,0xec,                      // vmovdqa xmm5,xmm4 (five-byte terminator)
        0x66,0x48,0x0f,0x7e,0xe8,                     // movq rax,xmm5
        0x48,0x39,0x54,0x24,0xf8,                     // cmp [rsp-8],rdx
        0x74,0x07,                                     // je .red_zone_ok
        0x48,0xc7,0xc0,0xff,0xff,0xff,0xff,           // mov rax,-1
        0xc3
    };
    const uint8_t short_successor_wrapper[] = {
        0x48,0xb8, 0,0,0,0,0,0,0,0,                 // movabs rax,data
        0x48,0xba, 0xef,0xcd,0xab,0x89,0x67,0x45,0x23,0x01, // movabs rdx,red-zone sentinel
        0x48,0x89,0x54,0x24,0xf8,                    // mov [rsp-8],rdx
        0xf3,0x0f,0x6f,0x48,0x10,                    // movdqu xmm1,[rax+0x10]
        0xe9, 0,0,0,0                                // jmp consumed VEX successor
    };
    const uint8_t chain_code[] = {
        0x48,0xb8, 0,0,0,0,0,0,0,0,                 // movabs rax,data
        0xf3,0x0f,0x6f,0x00,                          // movdqu xmm0,[rax] (control)
        0xf3,0x0f,0x6f,0x18,                          // movdqu xmm3,[rax] (control)
        0xf3,0x0f,0x6f,0x48,0x10,                    // movdqu xmm1,[rax+0x10]
        0xf3,0x0f,0x6f,0x50,0x20,                    // movdqu xmm2,[rax+0x20]
        0x66,0x0f,0x79,0xc8,                          // extrq xmm1,xmm0
        0x66,0x48,0x0f,0x79,0xd3,                    // extrq xmm2,xmm3 (five-byte form)
        0x66,0x48,0x0f,0x7e,0xd0,                    // movq rax,xmm2
        0xc3
    };
    const uint8_t chain_second_wrapper[] = {
        0x48,0xb8, 0,0,0,0,0,0,0,0,                 // movabs rax,data
        0xf3,0x0f,0x6f,0x18,                          // movdqu xmm3,[rax] (control)
        0xf3,0x0f,0x6f,0x50,0x20,                    // movdqu xmm2,[rax+0x20]
        0xe9, 0,0,0,0                                // jmp chained second EXTRQ
    };
    LoadedImage image;
    image.base = base;
    image.min_vaddr = 0;
    image.max_vaddr = 0x1000;
    image.mem.assign(0x1000, 0xcc);
    image.prot.push_back({0, 0x1000, true, false, true});
    memcpy(image.mem.data(), code, sizeof(code));
    memcpy(image.mem.data() + short_code_offset, short_code, sizeof(short_code));
    memcpy(image.mem.data() + short_successor_wrapper_offset, short_successor_wrapper,
           sizeof(short_successor_wrapper));
    memcpy(image.mem.data() + chain_code_offset, chain_code, sizeof(chain_code));
    memcpy(image.mem.data() + chain_second_wrapper_offset, chain_second_wrapper,
           sizeof(chain_second_wrapper));
    const uint64_t data_address = base + data_offset;
    memcpy(image.mem.data() + 27, &data_address, sizeof(data_address));
    memcpy(image.mem.data() + short_code_offset + 2, &data_address, sizeof(data_address));
    memcpy(image.mem.data() + short_successor_wrapper_offset + 2, &data_address,
           sizeof(data_address));
    memcpy(image.mem.data() + chain_code_offset + 2, &data_address, sizeof(data_address));
    memcpy(image.mem.data() + chain_second_wrapper_offset + 2, &data_address,
           sizeof(data_address));
    const int32_t short_successor_delta = static_cast<int32_t>(
        short_sse4a_offset + 4 -
        (short_successor_wrapper_offset + sizeof(short_successor_wrapper)));
    memcpy(image.mem.data() + short_successor_wrapper_offset + 31, &short_successor_delta,
           sizeof(short_successor_delta));
    const int32_t chain_second_delta = static_cast<int32_t>(
        chain_sse4a_offset + 4 -
        (chain_second_wrapper_offset + sizeof(chain_second_wrapper)));
    memcpy(image.mem.data() + chain_second_wrapper_offset + 20, &chain_second_delta,
           sizeof(chain_second_delta));
    const uint64_t control = 0x0808;  // length=8, index=8
    const uint64_t value = 0xab00;
    const uint64_t upper_value = 0x0123456789abcdefull;
    const uint64_t chained_value = 0xcd00;
    memcpy(image.mem.data() + data_offset, &control, sizeof(control));
    memcpy(image.mem.data() + data_offset + 16, &value, sizeof(value));
    memcpy(image.mem.data() + data_offset + 24, &upper_value, sizeof(upper_value));
    memcpy(image.mem.data() + data_offset + 32, &chained_value, sizeof(chained_value));
    const uint64_t before_map = sse4a_fastpath_patch_count();
    std::string error;
    CHECK(map_image(image, &error), "map synthetic SSE4a guest image");
    if (!error.empty()) std::printf("  map detail: %s\n", error.c_str());
    const uint64_t after_map = sse4a_fastpath_patch_count();
    CHECK(after_map >= before_map + 8,
          "EXTRQ and relocated-successor entry points exist before guest entry");
    install_trap_handler();
    using GuestFn = uint64_t (*)();
    const GuestFn function = (GuestFn)(uintptr_t)base;
    const uint64_t first = function();
    const uint64_t second = function();
    const uint64_t after_second = sse4a_fastpath_patch_count();
    CHECK(first == 0xab && second == 0xab,
          "EXTRQ returns the AMD-defined field without overwriting the guest red zone");
    uint64_t staged_upper = 0;
    memcpy(&staged_upper, (const void*)(uintptr_t)(base + data_offset + 56),
           sizeof(staged_upper));
    CHECK(staged_upper == upper_value,
          "EXTRQ fast and trapped paths preserve the neighboring qword like the AMD CPU model");
    CHECK(after_second == after_map,
          "executing EXTRQ does not enter the live exception patcher");
    CHECK(*(const uint8_t*)(uintptr_t)(base + sse4a_offset) == 0xe9,
          "five-byte EXTRQ is a near-jump before it can fault on an Intel host");

    const GuestFn short_function = (GuestFn)(uintptr_t)(base + short_code_offset);
    const uint64_t short_first = short_function();
    const uint64_t short_second = short_function();
    const uint64_t short_after_second = sse4a_fastpath_patch_count();
    CHECK(short_first == 0xab && short_second == 0xab,
          "four-byte EXTRQ and its copied VEX successor preserve their result");
    CHECK(short_after_second == after_map,
          "four-byte EXTRQ does not enter the live exception patcher");
    CHECK(*(const uint8_t*)(uintptr_t)(base + short_sse4a_offset) == 0xe9,
          "four-byte EXTRQ and validated VEX successor are translated before entry");
    CHECK(*(const uint8_t*)(uintptr_t)(base + short_sse4a_offset + 4) == 0xe9 &&
              *(const uint8_t*)(uintptr_t)(base + short_sse4a_offset + 8) == 0xe9 &&
              *(const uint8_t*)(uintptr_t)(base + short_sse4a_offset + 12) == 0xe9 &&
              *(const uint8_t*)(uintptr_t)(base + short_sse4a_offset + 16) == 0xe9,
          "every consumed VEX boundary starts an exception-free near jump");
    const GuestFn short_successor_function =
        (GuestFn)(uintptr_t)(base + short_successor_wrapper_offset);
    const uint64_t short_successor_result = short_successor_function();
    CHECK(short_successor_result == 0xab00,
          "direct consumed-VEX entry preserves its result and the guest red zone");

    const GuestFn chain_function = (GuestFn)(uintptr_t)(base + chain_code_offset);
    const uint64_t chain_first = chain_function();
    const uint64_t chain_second = chain_function();
    const uint64_t chain_after_second = sse4a_fastpath_patch_count();
    CHECK(chain_first == 0xcd && chain_second == 0xcd,
          "adjacent EXTRQs run through separate chained expansions");
    CHECK(chain_after_second == after_map,
          "chained EXTRQs do not enter the live exception patcher");
    CHECK(*(const uint8_t*)(uintptr_t)(base + chain_sse4a_offset) == 0xe9,
          "adjacent EXTRQs are translated before either instruction can fault");
    const GuestFn chain_second_function =
        (GuestFn)(uintptr_t)(base + chain_second_wrapper_offset);
    CHECK(chain_second_function() == 0xcd,
          "direct entry to the chained EXTRQ uses its overlapping near jump");

    bool randomized_exact = true, randomized_short = true, randomized_chain = true;
    uint64_t random = 0x9e3779b97f4a7c15ull;
    auto next_random = [&] {
        random ^= random << 13; random ^= random >> 7; random ^= random << 17;
        return random;
    };
    for (unsigned i = 0; i < 4096; ++i) {
        const uint32_t length = (uint32_t)(next_random() & 63);
        const uint32_t index = (uint32_t)(next_random() & 63);
        const uint64_t randomized_control = length | ((uint64_t)index << 8);
        const uint64_t randomized_value = next_random();
        const uint64_t randomized_second = next_random();
        memcpy((void*)(uintptr_t)(base + data_offset), &randomized_control,
               sizeof(randomized_control));
        memcpy((void*)(uintptr_t)(base + data_offset + 16), &randomized_value,
               sizeof(randomized_value));
        memcpy((void*)(uintptr_t)(base + data_offset + 32), &randomized_second,
               sizeof(randomized_second));
        const uint64_t expected_value =
            sse4a_extrq(randomized_value, length, index);
        const uint64_t expected_second =
            sse4a_extrq(randomized_second, length, index);
        randomized_exact &= function() == expected_value;
        randomized_short &= short_function() == expected_value;
        randomized_chain &= chain_function() == expected_second;
    }
    CHECK(randomized_exact, "five-byte EXTRQ fast path matches randomized AMD fields");
    CHECK(randomized_short,
          "four-byte EXTRQ plus copied VEX successor matches randomized AMD fields");
    CHECK(randomized_chain,
          "adjacent EXTRQ expansions match randomized AMD fields");

}

static void test_condition_slot_lifecycle() {
    size_t used_before = 0;
    size_t capacity = 0;
    snapshot_guest_wait_registry(used_before, capacity);
    CHECK(capacity > used_before, "condition wait registry reports reusable capacity");

    std::vector<pthread_cond_t> conditions(capacity - used_before);
    size_t initialized = 0;
    bool filled = true;
    for (pthread_cond_t& condition : conditions) {
        if (pthread_cond_init(&condition, nullptr) != 0) {
            filled = false;
            break;
        }
        ++initialized;
        if (interruptible_cond_signal(&condition) != 0) {
            filled = false;
            break;
        }
    }

    size_t used_full = 0;
    size_t capacity_full = 0;
    snapshot_guest_wait_registry(used_full, capacity_full);
    CHECK(filled && initialized == conditions.size() && used_full == capacity,
          "condition wait registry can use every advertised slot");

    bool destroyed = true;
    for (size_t i = 0; i < initialized; ++i) {
        destroyed &= pthread_cond_destroy(&conditions[i]) == 0;
        interruptible_cond_forget(&conditions[i]);
    }
    size_t used_after = 0;
    size_t capacity_after = 0;
    snapshot_guest_wait_registry(used_after, capacity_after);
    CHECK(destroyed && capacity_after == capacity && used_after == used_before,
          "destroyed condition variables release their wait registry slots");

    pthread_cond_t replacement{};
    const bool replacement_initialized = pthread_cond_init(&replacement, nullptr) == 0;
    const bool replacement_registered = replacement_initialized &&
                                        interruptible_cond_signal(&replacement) == 0;
    bool replacement_destroyed = false;
    if (replacement_initialized) {
        replacement_destroyed = pthread_cond_destroy(&replacement) == 0;
        interruptible_cond_forget(&replacement);
    }
    CHECK(replacement_registered && replacement_destroyed,
          "a fresh condition variable reuses a retired registry slot");
}

int main() {
    std::printf("== test_win_exception_delivery ==\n");

    test_sse4a_fastpath();
    test_condition_slot_lifecycle();
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

    registry_tid = 0;
    registry_done = 0;
    registry_torn = 0;
    registry_observations = 0;
    CHECK(pthread_create(&thread, nullptr, registry_worker, nullptr) == 0,
          "create lifecycle-registry publication stress thread");
    for (int i = 0; i < 1000 && !registry_tid; ++i) Sleep(1);
    while (!registry_done) {
        GuestThreadSnapshot snapshot{};
        if (!snapshot_guest_thread_registration(registry_tid, snapshot)) continue;
        InterlockedIncrement(&registry_observations);
        const bool generation_a = snapshot.pthread_id == 0x1111222233334444ull &&
                                  snapshot.stack_base == 0x11110000u &&
                                  snapshot.stack_size == 0x1111u;
        const bool generation_b = snapshot.pthread_id == 0xaaaabbbbccccddddull &&
                                  snapshot.stack_base == 0x22220000u &&
                                  snapshot.stack_size == 0x2222u;
        if (!generation_a && !generation_b) InterlockedExchange(&registry_torn, 1);
    }
    pthread_join(thread, nullptr);
    GuestThreadSnapshot retired_snapshot{};
    CHECK(registry_observations != 0, "lifecycle-registry stress observed published slots");
    CHECK(registry_torn == 0, "lifecycle-registry snapshots never mix slot generations");
    CHECK(!snapshot_guest_thread_registration(registry_tid, retired_snapshot),
          "retired lifecycle slot is no longer visible");

    pin_boundary_tid = 0;
    pin_boundary_ready = 0;
    pin_boundary_transition = 0;
    pin_boundary_republished = 0;
    pin_boundary_finish = 0;
    const char* pin_trace_path = "test_win_exception_delivery_pin_boundary.tmp";
    DeleteFileA(pin_trace_path);
    pthread_t pin_thread{};
    const int pin_create = pthread_create(&pin_thread, nullptr, pin_boundary_worker, nullptr);
    CHECK(pin_create == 0,
          "create deterministic sampler generation-boundary target");
    if (pin_create == 0) {
        for (int i = 0; i < 2000 && !pin_boundary_ready; ++i) Sleep(1);
        set_guest_thread_trace_test_hook(&pin_boundary_hook);
        dump_guest_thread_trace(pin_trace_path);
        set_guest_thread_trace_test_hook(nullptr);
        CHECK(pin_boundary_republished != 0,
              "sampler boundary hook retires A and republishes B before suspension");
        char pin_trace[4096]{};
        if (FILE* file = std::fopen(pin_trace_path, "rb")) {
            std::fread(pin_trace, 1, sizeof(pin_trace) - 1, file);
            std::fclose(file);
        }
        CHECK(std::strstr(pin_trace, "unavailable error=1237") != nullptr,
              "sampler rejects the retired generation after pinning its handle");
        CHECK(std::strstr(pin_trace, " rip=") == nullptr,
              "sampler never captures the replacement generation through a stale snapshot");
        InterlockedExchange(&pin_boundary_finish, 1);
        pthread_join(pin_thread, nullptr);
    }
    DeleteFileA(pin_trace_path);

    void* readwrite_page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    void* executable_page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READ);
    CHECK(readwrite_page && !guest_trace_page_executable((uintptr_t)readwrite_page),
          "thread trace rejects committed non-executable pages");
    CHECK(executable_page && guest_trace_page_executable((uintptr_t)executable_page),
          "thread trace accepts committed executable pages");
    if (readwrite_page) VirtualFree(readwrite_page, 0, MEM_RELEASE);
    if (executable_page) VirtualFree(executable_page, 0, MEM_RELEASE);

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

    // Default delivery first offers a cooperative checkpoint, but a target running guest CPU code
    // may not reach one. The bounded queue must be withdrawn and forced-CONTEXT delivery must run the
    // handler before success is returned. Repeating on one target also proves its slot was returned to
    // Idle and the global queued counter did not leak.
    CHECK(pthread_create(&thread, nullptr, gpr_worker, nullptr) == 0,
          "create CPU-bound default-delivery target thread");
    for (int i = 0; i < 1000 && !gpr_ready; ++i) Sleep(1);
    bool cpu_fallback_ok = worker_tid != 0 && gpr_ready != 0;
    for (LONG expected_deliveries = 1; expected_deliveries <= 2 && cpu_fallback_ok;
         ++expected_deliveries) {
        result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
        for (int spin = 0; spin < 2000 && delivered < expected_deliveries; ++spin) Sleep(1);
        cpu_fallback_ok = result == 0 && delivered == expected_deliveries &&
                          pending_guest_exception_count() == 0;
    }
    InterlockedExchange(&gpr_stop, 1);
    pthread_join(thread, nullptr);
    CHECK(cpu_fallback_ok,
          "CPU-bound target receives two fallback deliveries with no queued state leak");

    pthread_t exited_thread{};
    CHECK(pthread_create(&exited_thread, nullptr, exited_worker, nullptr) == 0,
          "create target that exits before exception delivery");
    pthread_join(exited_thread, nullptr);
    result = raise((uint64_t)exited_thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0x80020003ull && pending_guest_exception_count() == 0,
          "exited target returns ESRCH without publishing queued state");

    reset_delivery_state();
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
    GuestWaitSnapshot cooperative_wait{};
    CHECK(snapshot_guest_wait(worker_tid, cooperative_wait),
          "cooperative condition wait has a stable registry snapshot");
    CHECK(cooperative_wait.kind == GuestWaitKind::ConditionSequence &&
          cooperative_wait.source == (uintptr_t)&wait_cond && cooperative_wait.object != 0,
          "condition wait snapshot keeps kind, object, and source in one generation");

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
    GuestWaitSnapshot nested_outer_wait{};
    CHECK(snapshot_guest_wait(worker_tid, nested_outer_wait) &&
          nested_outer_wait.kind == GuestWaitKind::ConditionSequence &&
          nested_outer_wait.object != 0,
          "nested-delivery target begins with one stable outer wait");
    result = raise((uint64_t)thread, 0x1e, 0, 0, 0, 0);
    CHECK(result == 0, "deliver cooperative exception whose handler performs a nested wait");
    for (int i = 0; i < 2000 && !nested_wait_ready; ++i) Sleep(1);
    CHECK(nested_wait_ready != 0, "guest handler entered its nested semaphore-style wait");
    GuestWaitSnapshot nested_waits[4]{};
    const size_t nested_wait_count =
        snapshot_guest_waits(worker_tid, nested_waits, sizeof(nested_waits) / sizeof(nested_waits[0]));
    bool nested_has_outer = false;
    bool nested_has_distinct_inner = false;
    for (size_t i = 0; i < nested_wait_count && i < 4; ++i) {
        nested_has_outer |= nested_waits[i].object == nested_outer_wait.object;
        nested_has_distinct_inner |= nested_waits[i].object != 0 &&
                                     nested_waits[i].object != nested_outer_wait.object;
    }
    CHECK(nested_wait_count == 2 && nested_has_outer && nested_has_distinct_inner,
          "nested snapshot reports both active waits without choosing an arbitrary current slot");
    GuestWaitSnapshot ambiguous_wait{};
    CHECK(!snapshot_guest_wait(worker_tid, ambiguous_wait),
          "singular wait snapshot rejects nested ambiguity");
    InterlockedExchange(&nested_wait_release, 1);
    interruptible_cond_broadcast(&nested_wait_cond);
    for (int i = 0; i < 2000 && !nested_wait_returned; ++i) Sleep(1);
    CHECK(nested_wait_returned != 0, "nested wait returned while handler remains active");
    GuestWaitSnapshot remaining_outer_wait{};
    CHECK(snapshot_guest_wait(worker_tid, remaining_outer_wait) &&
          remaining_outer_wait.object == nested_outer_wait.object,
          "outer wait remains the sole snapshot after inner wait unregisters");
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

    // A natural condition signal can let the target return and destroy its guest synchronization
    // objects immediately after the GC wake is queued. Repeating the same stack addresses stresses
    // both delayed wake work and address reuse: exception delivery must touch only the lifetime-stable
    // sequence slot, never a detached helper's raw pthread_cond_t/pthread_mutex_t pointers.
    bool cond_destroy_race_ok = true;
    for (int iteration = 0; iteration < 32 && cond_destroy_race_ok; ++iteration) {
        reset_delivery_state();
        CondDestroyRace race{};
        const int mutex_init = pthread_mutex_init(&race.mutex, nullptr);
        const int cond_init = mutex_init == 0 ? pthread_cond_init(&race.cond, nullptr) : -1;
        if (mutex_init != 0 || cond_init != 0) {
            if (cond_init == 0) pthread_cond_destroy(&race.cond);
            if (mutex_init == 0) pthread_mutex_destroy(&race.mutex);
            cond_destroy_race_ok = false;
            break;
        }
        pthread_t race_thread{};
        if (pthread_create(&race_thread, nullptr, cond_destroy_race_worker, &race) != 0) {
            pthread_cond_destroy(&race.cond);
            pthread_mutex_destroy(&race.mutex);
            cond_destroy_race_ok = false;
            break;
        }
        for (int spin = 0; spin < 1000 && !race.ready; ++spin) Sleep(1);
        // The worker registers its wait before unlocking this mutex. Acquiring it proves publication
        // completed before the exception and natural signal are raced.
        pthread_mutex_lock(&race.mutex);
        pthread_mutex_unlock(&race.mutex);
        const uint64_t race_result = raise((uint64_t)race_thread, 0x1e, 0, 0, 0, 0);
        interruptible_cond_signal(&race.cond);
        pthread_join(race_thread, nullptr);
        const int cond_destroyed = pthread_cond_destroy(&race.cond);
        const int mutex_destroyed = pthread_mutex_destroy(&race.mutex);
        cond_destroy_race_ok = race_result == 0 && delivered == 1 && race.returned != 0 &&
                               cond_destroyed == 0 && mutex_destroyed == 0;
        SwitchToThread();
    }
    CHECK(cond_destroy_race_ok,
          "natural condition wake plus immediate destroy is safe across 32 queued GC races");
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
    auto check_classified_wait = [&](GuestWaitKind kind, uintptr_t source,
                                     const char* stable_message, const char* wake_message) {
        reset_delivery_state();
        classified_wait_kind = kind;
        classified_wait_source = source;
        pthread_mutex_lock(&wait_mutex);
        CHECK(pthread_create(&thread, nullptr, classified_cond_worker, nullptr) == 0,
              "create classified wait target thread");
        for (int i = 0; i < 1000 && !worker_tid; ++i) Sleep(1);
        pthread_mutex_unlock(&wait_mutex);
        for (int i = 0; i < 1000 && !cond_ready; ++i) Sleep(1);
        pthread_mutex_lock(&wait_mutex);
        pthread_mutex_unlock(&wait_mutex);
        GuestWaitSnapshot wait{};
        CHECK(snapshot_guest_wait(worker_tid, wait) && wait.kind == kind &&
              wait.source == source && wait.object != 0, stable_message);
        CHECK(interrupt_guest_wait((uint64_t)thread), wake_message);
        for (int i = 0; i < 2000 && !resumed; ++i) Sleep(1);
        if (!resumed) interruptible_cond_broadcast(&wait_cond);
        pthread_join(thread, nullptr);
        CHECK(resumed != 0, "classified wait target resumed after registry wake");
    };
    static uint64_t event_source_token;
    static uint64_t semaphore_source_token;
    check_classified_wait(GuestWaitKind::EventFlag, (uintptr_t)&event_source_token,
                          "event-flag wait retains its source classification",
                          "event-flag wait is interruptible through its sequence");
    check_classified_wait(GuestWaitKind::Semaphore, (uintptr_t)&semaphore_source_token,
                          "semaphore wait retains its source classification",
                          "semaphore wait is interruptible through its sequence");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

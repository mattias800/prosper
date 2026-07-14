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
static HleFn wait_on_address;
static int fails;

#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

extern "C" __attribute__((sysv_abi)) void guest_exception_handler(uint64_t type, void* raw_ctx) {
    auto* ctx = (const uint8_t*)raw_ctx;
    handler_tid = GetCurrentThreadId();
    context_rip = *(const uint64_t*)(ctx + 0xa0);
    context_rsp = *(const uint64_t*)(ctx + 0xf8);
    if (type == 0x1e) {
        wait_word = 1;
        InterlockedExchange(&delivered, 1);
    }
}

static void* worker(void*) {
    worker_tid = GetCurrentThreadId();
    wait_on_address((uint64_t)(uintptr_t)&wait_word, 0, 0, 0, 0, 0);
    InterlockedExchange(&resumed, 1);
    return nullptr;
}

int main() {
    std::printf("== test_win_exception_delivery ==\n");
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
    CHECK(resumed != 0, "target resumed its interrupted context after handler return");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}

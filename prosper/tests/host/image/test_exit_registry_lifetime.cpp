// test_exit_registry_lifetime — regression guard for #2613.
//
// prosper never joins guest threads: k_pthread_create spawns them and nothing quiesces them at
// shutdown, so every process reaches exit() with live guest threads. thread_trampoline's tail —
// tls_dtv_purge_current_thread(), unregister_thread_stack(), retire_guest_thread_name() — then runs
// while __run_exit_handlers is already tearing the process down, and it touches namespace-scope
// registries whose static destructors that teardown has already executed. #2613 has the core dump:
// main inside _dl_fini, a guest thread inside an unordered_map lookup on a destroyed map.
//
// THE HARD PART OF TESTING THIS IS DETERMINISM. Reproducing the original crash needs a thread to be
// in the tail during teardown — 1 fault in 300 runs pinned to one CPU. A test that "passes N times"
// would be evidence of nothing, and #2617 removed the accidental trigger precisely because a random
// SEGFAULT with no diagnosis is worse than an issue. So this test does not race: it PARKS a guest
// worker inside its entry, lets main return, and drives the whole thing from an exit-time probe that
// is guaranteed to run after the library's static destruction. Post-destruction registry access is
// then a certainty on every run, not a 1-in-300 accident.
//
// The ordering that makes that true, and how the test proves it rather than assuming it:
//   * ExitProbe is declared FIRST in this translation unit, and this TU's object file precedes
//     libprosper_core.a on the link line. ELF runs .init_array in link order, so this TU's dynamic
//     initialization happens before the library's; __cxa_atexit handlers run in reverse registration
//     order, so this probe's destructor runs after every static destructor in the library.
//   * That is a property of this toolchain and link, not of the standard — so the probe DOES NOT
//     TRUST IT. hle_kernel.cpp and exec_image_linux.cpp each carry a canary object declared before
//     their registries, which flips a flag when destroyed. If either flag is still false when the
//     probe runs, the probe fails the test loudly instead of passing vacuously: "the ordering
//     assumption broke" and "the fix works" must never look alike.
//   * main() asserts the same two flags read FALSE while it runs, so the flags are shown to move.
//
// What each phase covers: the read side of all three registries after destruction (thread names,
// thread stacks, the __tls_get_addr DTV), then the real trampoline tail itself — the parked worker
// is released and joined from inside the probe, so retire_guest_thread_name / unregister_thread_stack
// / tls_dtv_purge_current_thread all execute post-destruction and their effects are asserted.
//
// Without the fix the map/vector destructors have run by then, so every one of those touches is a
// use-after-destruction: the observed failures are a SIGSEGV or a registry that reports its live
// rows as missing. Either way the test fails; it cannot pass.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/image/exec_image.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace prosper;

namespace {

// ---------------------------------------------------------------------------------------------
// The exit-time probe. MUST stay the first object with a non-trivial constructor/destructor in this
// file: everything else here is constant-initialized (atomics, plain function pointers, PODs), so
// this is unambiguously the first registration and therefore the last destructor in this TU.
// ---------------------------------------------------------------------------------------------
struct ExitProbe {
    ExitProbe() noexcept;
    ~ExitProbe();
};
ExitProbe g_exit_probe;

// Everything below is constant-initialized: no dynamic initialization, no destructors.
HleFn g_create = nullptr, g_join = nullptr, g_getname = nullptr, g_tlsget = nullptr;

std::atomic<uint64_t> g_parked_thread{0};
std::atomic<bool> g_parked_running{false};
std::atomic<bool> g_release_parked{false};
std::atomic<bool> g_main_phase_ok{false};
std::atomic<bool> g_probe_constructed{false};

constexpr uint64_t kSyntheticTid  = 0x2613cafe;      // a stack row no thread exit ever removes
constexpr uint64_t kSyntheticBase = 0x7f2613000000ull;
constexpr uint64_t kSyntheticSize = 0x40000;
constexpr uint64_t kWorkerResult  = 0x2613;

const char kParkedName[] = "exit-probe-worker";

// One TLS module template so the parked worker can take a real DTV entry through __tls_get_addr.
unsigned char g_tdata[8] = {0x26, 0x13, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
enum { kMemsz = 32, kFilesz = 8 };

int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); g_fails++; } \
                         else      { std::printf("  [ok]   %s\n", m); } } while (0)

// Probe-side reporting: stdio is still alive inside __run_exit_handlers (_IO_cleanup runs after
// them), but flush every line so nothing is lost if a later phase faults.
int g_probe_fails = 0;
void probe_check(bool ok, const char* what) {
    std::fprintf(stderr, "  %s %s\n", ok ? "[ok]  " : "[FAIL]", what);
    std::fflush(stderr);
    if (!ok) g_probe_fails++;
}

// The parked guest worker. Takes a DTV entry, announces itself, then blocks inside its guest entry
// so its name/stack/DTV rows are all live when the process starts exiting. Body kept out of the
// sysv_abi shim for the MinGW assembler reason documented in test_pthread_names.cpp.
__attribute__((noinline)) void* parked_worker_body() {
    uint64_t ti[2] = {0, 0};                       // module 0, offset 0
    (void)g_tlsget((uint64_t)(uintptr_t)ti, 0, 0, 0, 0, 0);
    g_parked_running.store(true, std::memory_order_release);
    while (!g_release_parked.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return (void*)(uintptr_t)kWorkerResult;
}
#ifdef _WIN32
extern "C" __attribute__((sysv_abi)) void* parked_worker(void*) {
#else
void* parked_worker(void*) {
#endif
    return parked_worker_body();
}

ExitProbe::ExitProbe() noexcept { g_probe_constructed.store(true, std::memory_order_release); }

ExitProbe::~ExitProbe() {
    std::fprintf(stderr, "== exit probe (static destruction is under way) ==\n");
    std::fflush(stderr);

    // ---- validity gate. Everything below is only meaningful AFTER the library's static
    // destruction; if the link order that guarantees it ever changes, fail rather than pass.
    const bool hle_done  = hle_kernel_statics_destroyed();
    const bool exec_done = exec_image_statics_destroyed();
    if (!hle_done || !exec_done) {
        std::fprintf(stderr,
                     "  [FAIL] probe ran BEFORE the library's static destruction "
                     "(hle_kernel=%d exec_image=%d). This test's ordering assumption is broken -- "
                     "the result says nothing about #2613 either way.\n",
                     (int)hle_done, (int)exec_done);
        std::fflush(stderr);
        std::_Exit(1);
    }
    probe_check(true, "probe runs after the library's static destructors (both canaries flipped)");

    if (!g_main_phase_ok.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "  [FAIL] main phase did not complete; probe has nothing to check\n");
        std::fflush(stderr);
        std::_Exit(1);
    }

    const uint64_t parked = g_parked_thread.load(std::memory_order_acquire);

    // ---- (1) read side, post-destruction: the thread-stack registry (exec_image_linux.cpp).
    void* base = nullptr; size_t size = 0;
    const bool stack_found = guest_stack_for_thread(kSyntheticTid, &base, &size);
    probe_check(stack_found && (uint64_t)(uintptr_t)base == kSyntheticBase && size == kSyntheticSize,
                "thread-stack registry still serves its row after static destruction");

    // ---- (2) read side, post-destruction: the guest thread-name registry (hle_kernel.cpp).
    unsigned char name_out[40];
    std::memset(name_out, 0xa5, sizeof name_out);
    const uint64_t name_rc = g_getname(parked, (uint64_t)(uintptr_t)name_out, 0, 0, 0, 0);
    probe_check(name_rc == 0 && std::strcmp((const char*)name_out, kParkedName) == 0,
                "thread-name registry still serves the parked worker's name after static destruction");

    // ---- (3) read side, post-destruction: the __tls_get_addr DTV (hle_kernel.cpp).
    probe_check(tls_dtv_thread_count() == 1,
                "DTV still reports the parked worker's entry after static destruction");

    // ---- (4) the real thing: release the parked worker so thread_trampoline's tail runs NOW,
    // with the process already inside its exit handlers. This is the exact interleaving #2613's
    // core dump caught, made deterministic.
    g_release_parked.store(true, std::memory_order_release);
    uint64_t worker_rv = 0;
    const uint64_t join_rc = g_join(parked, (uint64_t)(uintptr_t)&worker_rv, 0, 0, 0, 0);
    probe_check(join_rc == 0 && worker_rv == kWorkerResult,
                "parked guest worker ran its trampoline tail during exit and joined cleanly");

    // ---- (5) and the tail's effects landed in the still-live registries.
    std::memset(name_out, 0xa5, sizeof name_out);
    probe_check(g_getname(parked, (uint64_t)(uintptr_t)name_out, 0, 0, 0, 0) == 3,
                "retire_guest_thread_name erased the row from the post-destruction registry");
    probe_check(!guest_stack_for_thread(parked, &base, &size),
                "unregister_thread_stack erased the worker's row from the post-destruction registry");
    probe_check(tls_dtv_thread_count() == 0,
                "tls_dtv_purge_current_thread purged the DTV from the post-destruction registry");

    // ---- (6) write side, post-destruction: mutate the stack registry and read the mutation back.
    unregister_thread_stack(kSyntheticTid);
    probe_check(!guest_stack_for_thread(kSyntheticTid, &base, &size),
                "thread-stack registry accepts a write after static destruction");

    if (g_probe_fails) {
        std::fprintf(stderr, "== FAIL: %d exit-probe check(s) ==\n", g_probe_fails);
        std::fflush(stderr);
        std::_Exit(1);
    }
    std::fprintf(stderr, "== PASS (exit probe) ==\n");
    std::fflush(stderr);
}

}   // namespace

int main() {
    std::printf("== test_exit_registry_lifetime ==\n");
    register_builtin_hle();

    g_create  = Hle::lookup(nid_hash("scePthreadCreate"));
    g_join    = Hle::lookup(nid_hash("scePthreadJoin"));
    g_getname = Hle::lookup(nid_hash("scePthreadGetname"));
    g_tlsget  = Hle::lookup(nid_hash("__tls_get_addr"));
    CHECK(g_create && g_join && g_getname && g_tlsget,
          "scePthreadCreate/Join/Getname and __tls_get_addr are registered");
    if (!(g_create && g_join && g_getname && g_tlsget)) { std::printf("== FAIL ==\n"); return 1; }

    // The canaries must read LIVE here. Without this arm the probe's validity gate could be passing
    // on a flag that is simply always true.
    CHECK(!hle_kernel_statics_destroyed() && !exec_image_statics_destroyed(),
          "both static-destruction canaries read LIVE while main() runs");
    CHECK(g_probe_constructed.load(std::memory_order_acquire),
          "the exit probe was constructed during static initialization");

    TlsModuleDesc desc{};
    desc.init_va = (uint64_t)(uintptr_t)g_tdata;
    desc.filesz  = kFilesz;
    desc.memsz   = kMemsz;
    set_tls_modules(&desc, 1);
    CHECK(tls_dtv_thread_count() == 0, "no thread holds a DTV entry yet");

    // A stack row keyed by an id no thread owns, so nothing but the probe ever removes it.
    register_thread_stack(kSyntheticTid, (void*)(uintptr_t)kSyntheticBase, kSyntheticSize);
    void* base = nullptr; size_t size = 0;
    CHECK(guest_stack_for_thread(kSyntheticTid, &base, &size) &&
          (uint64_t)(uintptr_t)base == kSyntheticBase && size == kSyntheticSize,
          "synthetic stack row is registered before exit");

    uint64_t tid = 0;
    const uint64_t rc = g_create((uint64_t)(uintptr_t)&tid, 0,
                                 (uint64_t)(uintptr_t)&parked_worker, 0,
                                 (uint64_t)(uintptr_t)kParkedName, 0);
    CHECK(rc == 0 && tid != 0, "parked guest worker created through scePthreadCreate");
    if (rc != 0 || tid == 0) { std::printf("== FAIL ==\n"); return 1; }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!g_parked_running.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(g_parked_running.load(std::memory_order_acquire),
          "parked worker reached its guest entry and is blocked there");

    unsigned char name_out[40];
    std::memset(name_out, 0xa5, sizeof name_out);
    CHECK(g_getname(tid, (uint64_t)(uintptr_t)name_out, 0, 0, 0, 0) == 0 &&
          std::strcmp((const char*)name_out, kParkedName) == 0,
          "parked worker's name is registered before exit");
    CHECK(tls_dtv_thread_count() == 1, "parked worker holds exactly one DTV entry before exit");

    g_parked_thread.store(tid, std::memory_order_release);

    if (g_fails) {
        std::printf("== FAIL: %d ==\n", g_fails);
        return 1;   // the probe still runs and will report its own gate failure
    }
    g_main_phase_ok.store(true, std::memory_order_release);
    std::printf("== main phase ok -- the verdict is the exit probe's ==\n");
    return 0;   // return, do NOT _Exit: static destruction is the thing under test
}

// test_stack_registry — guards the per-thread stack registry that scePthreadAttrGet /
// GC_get_stack_base rely on (and that the k_pthread_create trampoline populates). A regression
// here reintroduces "Bad stack base in GC_register_my_thread" during the boot.
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

namespace {
struct LateGuestEntryProbe {
    std::atomic<int> sequence{0};
    std::atomic<int> boundary_order{0};
    std::atomic<int> guest_order{0};
    std::atomic<int> entry_boundary_order{0};
    std::atomic<int> entry_guest_order{0};
    std::atomic<int> boundary_calls{0};
    std::atomic<int> primary_calls{0};
    std::atomic<bool> release_init{false};
    std::thread::id init_boundary_thread{};
    std::thread::id init_guest_thread{};
    std::thread::id entry_boundary_thread{};
    std::thread::id entry_guest_thread{};
};

LateGuestEntryProbe* g_late_guest_entry_probe = nullptr;

void observe_guest_entry(bool primary, void* opaque) {
    auto* probe = static_cast<LateGuestEntryProbe*>(opaque);
    const int call = probe->boundary_calls.fetch_add(1) + 1;
    if (primary) probe->primary_calls.fetch_add(1);
    const int order = probe->sequence.fetch_add(1) + 1;
    if (call == 1) {
        probe->init_boundary_thread = std::this_thread::get_id();
        probe->boundary_order.store(order, std::memory_order_release);
    } else if (call == 2) {
        probe->entry_boundary_thread = std::this_thread::get_id();
        probe->entry_boundary_order.store(order, std::memory_order_release);
    }
}

void late_guest_init(uint64_t, uint64_t) {
    // Filled in by main through this single-test-process pointer; run_guest_inits requires a plain
    // guest function address, so a capturing lambda is not an option.
    g_late_guest_entry_probe->init_guest_thread = std::this_thread::get_id();
    g_late_guest_entry_probe->guest_order.store(
        g_late_guest_entry_probe->sequence.fetch_add(1) + 1, std::memory_order_release);
    while (!g_late_guest_entry_probe->release_init.load(std::memory_order_acquire))
        std::this_thread::yield();
}
}

int main() {
    printf("== test_stack_registry ==\n");

    register_thread_stack(0x1111, (void*)0x40000000ull, 0x8000);
    register_thread_stack(0x2222, (void*)0x50000000ull, 0x100000);

    void* base = nullptr; size_t sz = 0;
    CHECK(guest_stack_for_thread(0x1111, &base, &sz) && base == (void*)0x40000000ull && sz == 0x8000,
          "lookup returns the registered base+size for tid 0x1111");
    base = nullptr; sz = 0;
    CHECK(guest_stack_for_thread(0x2222, &base, &sz) && base == (void*)0x50000000ull && sz == 0x100000,
          "distinct entry for tid 0x2222");
    CHECK(!guest_stack_for_thread(0x9999, &base, &sz),
          "unregistered tid misses (so k_attr_get leaves the attr untouched rather than lying)");

    // Re-registration (a tid value can be reused after a thread exits) overwrites cleanly.
    register_thread_stack(0x1111, (void*)0x60000000ull, 0x4000);
    CHECK(guest_stack_for_thread(0x1111, &base, &sz) && base == (void*)0x60000000ull && sz == 0x4000,
          "re-registration updates the entry");

    // Unregister erases (#138): pthread ids recycle, so a dead thread's entry must not survive to
    // serve the NEXT thread on the same id the old bounds.
    unregister_thread_stack(0x1111);
    CHECK(!guest_stack_for_thread(0x1111, &base, &sz), "unregister erases the entry");

    // End-to-end HLE thread lifecycle (#138): the guest attr's stacksize is honored (was: fixed
    // 8 MiB), the running thread sees its own registered bounds, and after join the entry is GONE
    // (the trampoline unregisters on exit; the glibc-owned stack is reclaimed with the thread).
#ifdef __linux__
    {
        register_builtin_hle();
        auto attr_init = Hle::lookup(nid_hash("scePthreadAttrInit"));
        auto attr_ssz  = Hle::lookup(nid_hash("scePthreadAttrSetstacksize"));
        auto attr_stack = Hle::lookup(nid_hash("scePthreadAttrSetstack"));
        auto attr_destroy = Hle::lookup(nid_hash("scePthreadAttrDestroy"));
        auto t_create  = Hle::lookup(nid_hash("scePthreadCreate"));
        auto t_join    = Hle::lookup(nid_hash("scePthreadJoin"));
        CHECK(attr_init && attr_ssz && attr_stack && attr_destroy && t_create && t_join,
              "thread HLE functions registered");
        struct Probe {
            size_t sz = 0;
            bool found = false;
            const void* expected_base = nullptr;
            size_t expected_size = 0;
            bool local_on_expected_stack = false;
        };
        auto entry = +[](void* p) -> void* {
            auto* pr = (Probe*)p;
            uint8_t local = 0;
            void* b = nullptr; size_t s = 0;
            pr->found = guest_stack_for_current_thread(&b, &s);
            pr->sz = s;
            const uintptr_t address = (uintptr_t)&local;
            const uintptr_t first = (uintptr_t)pr->expected_base;
            pr->local_on_expected_stack = !pr->expected_base ||
                (address >= first && address < first + pr->expected_size);
            return nullptr;
        };
        auto U = [](const void* p) { return (uint64_t)(uintptr_t)p; };

        // 2 MiB request: registered size ~= the request (glibc may round up by pages).
        void* at = nullptr; attr_init(U(&at), 0, 0, 0, 0, 0);
        attr_ssz(U(&at), 2 * 1024 * 1024, 0, 0, 0, 0);
        uint64_t tid = 0; Probe pr;
        CHECK(t_create(U(&tid), U(&at), U((void*)entry), U(&pr), 0, 0) == 0, "create with 2 MiB attr");
        t_join(tid, 0, 0, 0, 0, 0);
        CHECK(pr.found, "thread saw its own registered stack bounds");
        CHECK(pr.sz >= 2 * 1024 * 1024 && pr.sz < 3 * 1024 * 1024,
              "attr stacksize honored (~2 MiB, not the old fixed 8 MiB)");
        CHECK(!guest_stack_for_thread(tid, &base, &sz), "entry unregistered after the thread exited");
        attr_destroy(U(&at), 0, 0, 0, 0, 0);

        // Tiny request: floored (our HLE + host libc need headroom the Sony runtime doesn't).
        void* at2 = nullptr; attr_init(U(&at2), 0, 0, 0, 0, 0);
        attr_ssz(U(&at2), 64 * 1024, 0, 0, 0, 0);
        uint64_t tid2 = 0; Probe pr2;
        CHECK(t_create(U(&tid2), U(&at2), U((void*)entry), U(&pr2), 0, 0) == 0, "create with 64 KiB attr");
        t_join(tid2, 0, 0, 0, 0, 0);
        CHECK(pr2.found && pr2.sz >= 1 * 1024 * 1024, "tiny request floored to >= 1 MiB");
        attr_destroy(U(&at2), 0, 0, 0, 0, 0);

        // An exact caller-owned stack must survive the guest-attr -> local-attr copy. Checking a
        // local variable proves pthread actually ran on that address range, not merely its size.
        const size_t exact_size = 2 * 1024 * 1024;
        void* exact_stack = mmap(nullptr, exact_size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(exact_stack != MAP_FAILED, "allocated caller-owned exact stack");
        if (exact_stack != MAP_FAILED) {
            void* at3 = nullptr; attr_init(U(&at3), 0, 0, 0, 0, 0);
            CHECK(attr_stack(U(&at3), U(exact_stack), exact_size, 0, 0, 0) == 0,
                  "scePthreadAttrSetstack accepts an exact POSIX stack range");
            uint64_t tid3 = 0; Probe pr3;
            pr3.expected_base = exact_stack; pr3.expected_size = exact_size;
            CHECK(t_create(U(&tid3), U(&at3), U((void*)entry), U(&pr3), 0, 0) == 0,
                  "create with caller-owned exact stack");
            t_join(tid3, 0, 0, 0, 0, 0);
            CHECK(pr3.found && pr3.local_on_expected_stack,
                  "created thread executes inside the exact requested stack range");
            attr_destroy(U(&at3), 0, 0, 0, 0, 0);
            munmap(exact_stack, exact_size);
        }
    }
#endif

    // boot_program completes setup on one thread, then frontends can enter guest code from a later
    // std::thread. Drive the real pre-run_entry guest path: module init is itself guest execution and
    // must pass the HWBP boundary before its first instruction. The guest-order assertion executes
    // independently, so deleting the boundary call makes the exact named check red instead of making
    // the rest of this experiment skip. Keep this after the stack-size cases: glibc may reuse a prior
    // std::thread's default-size cached stack for a later pthread and obscure what those cases measure.
    LateGuestEntryProbe late_probe;
    g_late_guest_entry_probe = &late_probe;
    set_guest_execution_thread_enter_test_hook(&observe_guest_entry, &late_probe);
    std::atomic<size_t> late_init_count{0};
    const std::thread::id test_main_thread = std::this_thread::get_id();
    std::thread init_execution_thread([&] {
        late_init_count.store(run_guest_inits(
            {reinterpret_cast<uint64_t>(&late_guest_init)}), std::memory_order_release);
    });
    const auto init_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (late_probe.guest_order.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < init_deadline)
        std::this_thread::yield();
    // prosper-app/screenshot use this exact topology: boot_program's init executes on their caller,
    // then run_entry crosses the primary boundary from a different, simultaneously-live std::thread.
    const bool init_started = late_probe.guest_order.load(std::memory_order_acquire) != 0;
    if (init_started) {
        std::thread entry_execution_thread([&] {
            guest_execution_thread_enter(/*primary=*/true);
            late_probe.entry_guest_thread = std::this_thread::get_id();
            late_probe.entry_guest_order.store(
                late_probe.sequence.fetch_add(1) + 1, std::memory_order_release);
        });
        entry_execution_thread.join();
    }
    late_probe.release_init.store(true, std::memory_order_release);
    init_execution_thread.join();
    set_guest_execution_thread_enter_test_hook(nullptr);
    CHECK(late_probe.guest_order.load(std::memory_order_acquire) != 0 &&
          late_init_count.load(std::memory_order_acquire) == 1,
          "late module-init guest function still executes");
    CHECK(late_probe.boundary_calls.load(std::memory_order_acquire) == 2 &&
          late_probe.primary_calls.load(std::memory_order_acquire) == 2 &&
          late_probe.boundary_order.load(std::memory_order_acquire) == 1 &&
          late_probe.guest_order.load(std::memory_order_acquire) == 2 &&
          late_probe.entry_boundary_order.load(std::memory_order_acquire) == 3 &&
          late_probe.entry_guest_order.load(std::memory_order_acquire) == 4 &&
          late_probe.init_boundary_thread == late_probe.init_guest_thread &&
          late_probe.entry_boundary_thread == late_probe.entry_guest_thread &&
          late_probe.init_boundary_thread != late_probe.entry_boundary_thread &&
          late_probe.init_boundary_thread != test_main_thread &&
          late_probe.entry_boundary_thread != test_main_thread,
          "distinct init and frontend threads each reach the primary HWBP boundary before guest code");
    g_late_guest_entry_probe = nullptr;

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

// test_stack_registry — guards the per-thread stack registry that scePthreadAttrGet /
// GC_get_stack_base rely on (and that the k_pthread_create trampoline populates). A regression
// here reintroduces "Bad stack base in GC_register_my_thread" during the boot.
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstddef>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

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
        auto t_create  = Hle::lookup(nid_hash("scePthreadCreate"));
        auto t_join    = Hle::lookup(nid_hash("scePthreadJoin"));
        CHECK(attr_init && attr_ssz && t_create && t_join, "thread HLE functions registered");
        struct Probe { size_t sz = 0; bool found = false; };
        auto entry = +[](void* p) -> void* {
            auto* pr = (Probe*)p;
            void* b = nullptr; size_t s = 0;
            pr->found = guest_stack_for_current_thread(&b, &s);
            pr->sz = s;
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

        // Tiny request: floored (our HLE + host libc need headroom the Sony runtime doesn't).
        void* at2 = nullptr; attr_init(U(&at2), 0, 0, 0, 0, 0);
        attr_ssz(U(&at2), 64 * 1024, 0, 0, 0, 0);
        uint64_t tid2 = 0; Probe pr2;
        CHECK(t_create(U(&tid2), U(&at2), U((void*)entry), U(&pr2), 0, 0) == 0, "create with 64 KiB attr");
        t_join(tid2, 0, 0, 0, 0, 0);
        CHECK(pr2.found && pr2.sz >= 1 * 1024 * 1024, "tiny request floored to >= 1 MiB");
    }
#endif

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

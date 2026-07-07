// test_stack_registry — guards the per-thread stack registry that scePthreadAttrGet /
// GC_get_stack_base rely on (and that the k_pthread_create trampoline populates). A regression
// here reintroduces "Bad stack base in GC_register_my_thread" during the boot.
#include "../src/host/exec_image.hpp"
#include <cstdio>
#include <cstddef>

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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

// Guest libc allocation entry points must use one mutually compatible allocator family. On
// Windows, _aligned_malloc pointers cannot be passed to plain free; Astro exercises that path when
// aligned mesh buffers are destroyed during its world-map load.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t U(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_libc_alloc ==\n");
    register_builtin_hle();

    auto malloc_fn = Hle::lookup(nid_hash("malloc"));
    auto calloc_fn = Hle::lookup(nid_hash("calloc"));
    auto realloc_fn = Hle::lookup(nid_hash("realloc"));
    auto free_fn = Hle::lookup(nid_hash("free"));
    auto memalign_fn = Hle::lookup(nid_hash("memalign"));
    auto posix_memalign_fn = Hle::lookup(nid_hash("posix_memalign"));
    auto aligned_new_fn = Hle::lookup(nid_hash("_ZnwmSt11align_val_t"));
    auto aligned_delete_fn = Hle::lookup(nid_hash("_ZdlPvSt11align_val_t"));
    CHECK(malloc_fn && calloc_fn && realloc_fn && free_fn && memalign_fn &&
              posix_memalign_fn && aligned_new_fn && aligned_delete_fn,
          "allocation HLE functions registered");
    if (fails) return 1;

    auto* ordinary = (uint8_t*)(uintptr_t)malloc_fn(37, 0, 0, 0, 0, 0);
    CHECK(ordinary != nullptr &&
              ((uintptr_t)ordinary % alignof(std::max_align_t)) == 0,
          "malloc returns max-aligned storage");
    if (ordinary) memset(ordinary, 0x5a, 37);
    auto* grown = (uint8_t*)(uintptr_t)realloc_fn(U(ordinary), 211, 0, 0, 0, 0);
    bool preserved = grown != nullptr;
    for (size_t i = 0; preserved && i < 37; ++i) preserved = grown[i] == 0x5a;
    CHECK(preserved, "realloc preserves ordinary allocation contents");
    free_fn(U(grown), 0, 0, 0, 0, 0);

    auto* zeroed = (uint8_t*)(uintptr_t)calloc_fn(23, 7, 0, 0, 0, 0);
    bool all_zero = zeroed != nullptr;
    for (size_t i = 0; all_zero && i < 23 * 7; ++i) all_zero = zeroed[i] == 0;
    CHECK(all_zero, "calloc zeroes the complete allocation");
    free_fn(U(zeroed), 0, 0, 0, 0, 0);
    CHECK(calloc_fn(std::numeric_limits<uint64_t>::max(), 2, 0, 0, 0, 0) == 0,
          "calloc rejects multiplication overflow");

    auto* aligned = (uint8_t*)(uintptr_t)memalign_fn(256, 513, 0, 0, 0, 0);
    CHECK(aligned != nullptr && ((uintptr_t)aligned & 255) == 0,
          "memalign honors a 256-byte alignment");
    if (aligned) memset(aligned, 0xa5, 513);
    auto* aligned_grown =
        (uint8_t*)(uintptr_t)realloc_fn(U(aligned), 1027, 0, 0, 0, 0);
    preserved = aligned_grown != nullptr;
    for (size_t i = 0; preserved && i < 513; ++i) preserved = aligned_grown[i] == 0xa5;
    CHECK(preserved, "realloc accepts and preserves a memalign allocation");
    free_fn(U(aligned_grown), 0, 0, 0, 0, 0);

    void* posix = nullptr;
    CHECK(posix_memalign_fn(U(&posix), 512, 777, 0, 0, 0) == 0 && posix &&
              ((uintptr_t)posix & 511) == 0,
          "posix_memalign storage is compatible with free");
    free_fn(U(posix), 0, 0, 0, 0, 0);

    void* cpp = (void*)(uintptr_t)aligned_new_fn(901, 1024, 0, 0, 0, 0);
    CHECK(cpp && ((uintptr_t)cpp & 1023) == 0,
          "aligned operator new honors its alignment");
    aligned_delete_fn(U(cpp), 1024, 0, 0, 0, 0);

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

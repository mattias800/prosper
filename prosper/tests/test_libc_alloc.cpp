// Guest libc allocation entry points must use one mutually compatible allocator family. On
// Windows, _aligned_malloc pointers cannot be passed to plain free; Astro exercises that path when
// aligned mesh buffers are destroyed during its world-map load.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t U(const void* p) { return (uint64_t)(uintptr_t)p; }

static int run_self(const char* executable, const char* mode) {
    std::string command = "\"";
    command += executable;
    command += "\" ";
    command += mode;
    return std::system(command.c_str());
}

int main(int argc, char** argv) {
    printf("== test_libc_alloc ==\n");
    register_builtin_hle();

    // Child modes return normally only if a throwing allocation handler violates its contract.
    // The parent first proves that the quoted self-launch works, then requires these modes to
    // terminate before returning from the HLE call.
    if (argc == 2) {
        if (!strcmp(argv[1], "probe")) return 0;
        if (!strcmp(argv[1], "throwing-new")) {
            Hle::lookup(nid_hash("_Znwm"))(std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0, 0);
            return 0;
        }
        if (!strcmp(argv[1], "throwing-new-array")) {
            Hle::lookup(nid_hash("_Znam"))(std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0, 0);
            return 0;
        }
        if (!strcmp(argv[1], "throwing-aligned-new")) {
            Hle::lookup(nid_hash("_ZnwmSt11align_val_t"))(
                std::numeric_limits<uint64_t>::max(), 64, 0, 0, 0, 0);
            return 0;
        }
        if (!strcmp(argv[1], "throwing-aligned-new-array")) {
            Hle::lookup(nid_hash("_ZnamSt11align_val_t"))(
                std::numeric_limits<uint64_t>::max(), 64, 0, 0, 0, 0);
            return 0;
        }
        return 2;
    }

    auto malloc_fn = Hle::lookup(nid_hash("malloc"));
    auto calloc_fn = Hle::lookup(nid_hash("calloc"));
    auto realloc_fn = Hle::lookup(nid_hash("realloc"));
    auto free_fn = Hle::lookup(nid_hash("free"));
    auto memalign_fn = Hle::lookup(nid_hash("memalign"));
    auto posix_memalign_fn = Hle::lookup(nid_hash("posix_memalign"));
    auto new_fn = Hle::lookup(nid_hash("_Znwm"));
    auto new_nothrow_fn = Hle::lookup(nid_hash("_ZnwmRKSt9nothrow_t"));
    auto new_array_fn = Hle::lookup(nid_hash("_Znam"));
    auto new_array_nothrow_fn = Hle::lookup(nid_hash("_ZnamRKSt9nothrow_t"));
    auto aligned_new_fn = Hle::lookup(nid_hash("_ZnwmSt11align_val_t"));
    auto aligned_new_nothrow_fn = Hle::lookup(nid_hash("_ZnwmSt11align_val_tRKSt9nothrow_t"));
    auto aligned_new_array_fn = Hle::lookup(nid_hash("_ZnamSt11align_val_t"));
    auto aligned_new_array_nothrow_fn = Hle::lookup(nid_hash("_ZnamSt11align_val_tRKSt9nothrow_t"));
    auto aligned_delete_fn = Hle::lookup(nid_hash("_ZdlPvSt11align_val_t"));
    CHECK(malloc_fn && calloc_fn && realloc_fn && free_fn && memalign_fn &&
              posix_memalign_fn && new_fn && new_nothrow_fn && new_array_fn &&
              new_array_nothrow_fn && aligned_new_fn && aligned_new_nothrow_fn &&
              aligned_new_array_fn && aligned_new_array_nothrow_fn && aligned_delete_fn,
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
    CHECK(memalign_fn(std::numeric_limits<uint64_t>::max(), 1, 0, 0, 0, 0) == 0,
          "memalign rejects an alignment that cannot be rounded to a power of two");

    CHECK(new_nothrow_fn(std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0, 0) == 0,
          "nothrow operator new returns null on allocation failure");
    CHECK(new_array_nothrow_fn(std::numeric_limits<uint64_t>::max(), 0, 0, 0, 0, 0) == 0,
          "nothrow operator new[] returns null on allocation failure");
    CHECK(aligned_new_nothrow_fn(std::numeric_limits<uint64_t>::max(), 64, 0, 0, 0, 0) == 0,
          "aligned nothrow operator new returns null on allocation failure");
    CHECK(aligned_new_array_nothrow_fn(
              std::numeric_limits<uint64_t>::max(), 64, 0, 0, 0, 0) == 0,
          "aligned nothrow operator new[] returns null on allocation failure");
    CHECK(run_self(argv[0], "probe") == 0, "allocation death-test child launches successfully");
    CHECK(run_self(argv[0], "throwing-new") != 0,
          "throwing operator new never returns null on allocation failure");
    CHECK(run_self(argv[0], "throwing-new-array") != 0,
          "throwing operator new[] never returns null on allocation failure");
    CHECK(run_self(argv[0], "throwing-aligned-new") != 0,
          "throwing aligned operator new never returns null on allocation failure");
    CHECK(run_self(argv[0], "throwing-aligned-new-array") != 0,
          "throwing aligned operator new[] never returns null on allocation failure");

    void* cpp_plain = (void*)(uintptr_t)new_fn(333, 0, 0, 0, 0, 0);
    CHECK(cpp_plain != nullptr, "throwing operator new still returns storage on success");
    free_fn(U(cpp_plain), 0, 0, 0, 0, 0);
    void* cpp_array = (void*)(uintptr_t)new_array_fn(444, 0, 0, 0, 0, 0);
    CHECK(cpp_array != nullptr, "throwing operator new[] still returns storage on success");
    free_fn(U(cpp_array), 0, 0, 0, 0, 0);

    auto* aligned = (uint8_t*)(uintptr_t)memalign_fn(256, 513, 0, 0, 0, 0);
    CHECK(aligned != nullptr && ((uintptr_t)aligned & 255) == 0,
          "memalign honors a 256-byte alignment");
    if (aligned) memset(aligned, 0xa5, 513);
    auto* aligned_grown =
        (uint8_t*)(uintptr_t)realloc_fn(U(aligned), 1027, 0, 0, 0, 0);
    preserved = aligned_grown != nullptr;
    for (size_t i = 0; preserved && i < 513; ++i) preserved = aligned_grown[i] == 0xa5;
    // realloc guarantees the CONTENTS and *fundamental* alignment, and nothing beyond that. C17
    // 7.22.3p1: the pointer it returns "is suitably aligned so that it may be assigned to a pointer
    // to any type of object with a fundamental alignment requirement"; 6.2.8p2 fixes that at
    // alignof(max_align_t), and 6.2.8p3 makes 256 an *extended* alignment, which realloc never
    // promises to carry. The platform agrees rather than merely permitting this: libSceLibcInternal
    // exports `reallocalign` (OGybVuPAhAY) as a function SEPARATE from `realloc` (Y7aJ1uydPMo),
    // because a two-argument resize cannot know an alignment. Registering it is #2185.
    //
    // So do NOT re-add an `& 255` assertion here. One stood until #2165 and passed on heap-layout
    // luck rather than on behavior: on POSIX `guest_realloc_portable` is `return realloc(p, size)`
    // (hle_libc.cpp:183) and holds no lever that could make such an assertion pass or fail. It was
    // passing only because glibc happened to grow this block IN PLACE, which trivially keeps the
    // original alignment; once the block relocates, the new address has no reason to be 256-aligned.
    // Allocation performed before the test body decides which of those two happens, so four inert,
    // never-imported NID registrations were enough to turn it red on unmodified master -- meaning
    // any PR that registered a NID could fail this file without touching it.
    //
    // The Windows branch does preserve the alignment, as a side effect of the header it needs so
    // _aligned_malloc storage can reach plain free (hle_libc.cpp:78-88). That is an implementation
    // superset, deliberately not asserted here: the guest-visible contract stays identical on both
    // platforms, and the console's own realloc does not preserve an extended alignment either.
    CHECK(preserved, "realloc carries a memalign block's contents into the resized block");
    CHECK(aligned_grown != nullptr &&
              ((uintptr_t)aligned_grown % alignof(std::max_align_t)) == 0,
          "realloc returns fundamentally-aligned storage when resizing an over-aligned block");
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

    void* cpp_aligned_array =
        (void*)(uintptr_t)aligned_new_array_fn(555, 128, 0, 0, 0, 0);
    CHECK(cpp_aligned_array && ((uintptr_t)cpp_aligned_array & 127) == 0,
          "aligned operator new[] honors its alignment");
    aligned_delete_fn(U(cpp_aligned_array), 128, 0, 0, 0, 0);

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

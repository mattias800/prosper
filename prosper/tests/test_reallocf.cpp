// reallocf (YMZO9ChZb0E) is realloc except that the ORIGINAL block is freed when the resize fails
// (#2203). BSD ships it because `p = realloc(p, n)` leaks `p` on failure, so a caller of reallocf
// has delegated the free-on-failure to libc and will not do it itself.
//
// Unregistered, the NID fell to the dispatcher's default 0 — a NULL return the caller reads as OOM.
// That is a false FAILURE on every call, not only on the failure path: a resize that should have
// succeeded reported allocation failure. So the registration arm below is the whole defect, and it
// is structurally red without the fix (Hle::lookup returns nullptr).
//
// THE DISCRIMINATOR, and it is the reason this file is longer than the change it covers.
//
// #2203 flagged the trap when it filed the bug: asserting the return is NULL on failure is
// necessary and NOT sufficient, because a plain `realloc` alias returns NULL there too. Such an arm
// passes whether or not the original was freed, so it cannot fail for the reason it claims — the
// exact "true assertion, mechanism never ran" shape the charter warns about.
//
// What actually separates the two is observable without reading freed memory: **a live block's
// address can never be handed out again.** So after a failed reallocf, allocate a batch of blocks
// in the same size class and look for the original address among them.
//
//   - If the handler is reallocf (correct): the block was freed, and glibc's tcache returns it to
//     the very next same-size request. The address reappears.
//   - If the handler were realloc (the defect): the block is STILL LIVE, and no allocator may
//     return its address for a new allocation. The address CANNOT reappear.
//
// The failing direction is therefore guaranteed by allocator correctness rather than by allocator
// policy, which is what makes this a real discriminator: mutate `h_reallocf` into `h_realloc` and
// this arm goes red deterministically. The passing direction relies on same-size-class reuse, so it
// is given several attempts rather than one, and the block is small enough to be a tcache candidate.
//
// The mutation was run: replacing the body with plain `guest_realloc_portable` turns
// "the failed resize RELEASED the original block" red while every other arm stays green.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t U(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_reallocf ==\n");
    register_builtin_hle();

    HleFn reallocf_fn = Hle::lookup(nid_hash("reallocf"));
    HleFn malloc_fn   = Hle::lookup(nid_hash("malloc"));
    HleFn free_fn     = Hle::lookup(nid_hash("free"));

    CHECK(reallocf_fn != nullptr,
          "reallocf is registered (unregistered -> dispatcher 0 -> every call reads as OOM)");
    CHECK(malloc_fn != nullptr && free_fn != nullptr, "malloc/free are registered");
    if (!reallocf_fn || !malloc_fn || !free_fn) {
        printf("== %d failure(s) ==\n", fails);
        return 1;
    }

    // The name the guest imports must hash to the NID libSceLibcInternal actually exports. A
    // registration under a mistyped name would satisfy every behavioural arm below while leaving
    // the guest's call unresolved, so this is checked against the 3.20 dump's value directly.
    CHECK(nid_hash("reallocf") == "YMZO9ChZb0E",
          "nid_hash(\"reallocf\") == YMZO9ChZb0E, the NID libSceLibcInternal exports");

    // --- the success path is ordinary realloc: resize, preserving contents ----------------------
    {
        const size_t kOld = 64, kNew = 4096;
        auto* p = (uint8_t*)(uintptr_t)malloc_fn(kOld, 0, 0, 0, 0, 0);
        CHECK(p != nullptr, "malloc returns a block");
        if (p) {
            for (size_t i = 0; i < kOld; ++i) p[i] = (uint8_t)(i * 17u + 3u);
            auto* grown = (uint8_t*)(uintptr_t)reallocf_fn(U(p), kNew, 0, 0, 0, 0);
            CHECK(grown != nullptr, "a resize that CAN succeed returns storage, not a false OOM");
            if (grown) {
                bool kept = true;
                for (size_t i = 0; i < kOld && kept; ++i) kept = grown[i] == (uint8_t)(i * 17u + 3u);
                CHECK(kept, "contents survive the resize, all 64 original bytes");
                free_fn(U(grown), 0, 0, 0, 0, 0);
            }
        }
    }

    // --- reallocf(NULL, n) is malloc(n); there is nothing to release ----------------------------
    {
        auto* fresh = (void*)(uintptr_t)reallocf_fn(0, 128, 0, 0, 0, 0);
        CHECK(fresh != nullptr, "reallocf(NULL, n) allocates, exactly as realloc(NULL, n) does");
        if (fresh) free_fn(U(fresh), 0, 0, 0, 0, 0);
    }

    // --- THE ARM THAT DISTINGUISHES reallocf FROM realloc ---------------------------------------
    // A resize the allocator must refuse, then look for the original address in a fresh batch.
    {
        const size_t kSize = 96;                 // small: a tcache size class
        const size_t kImpossible = (size_t)-1;   // SIZE_MAX: no allocator can satisfy this
        auto* victim = (uint8_t*)(uintptr_t)malloc_fn(kSize, 0, 0, 0, 0, 0);
        CHECK(victim != nullptr, "malloc returns the block that the failed resize must release");
        if (victim) {
            const uintptr_t victim_addr = (uintptr_t)victim;

            auto* r = (void*)(uintptr_t)reallocf_fn(U(victim), kImpossible, 0, 0, 0, 0);
            // Necessary but NOT sufficient on its own -- a realloc alias also returns NULL here.
            // Recorded as such so nobody later reads this line as the coverage.
            CHECK(r == nullptr, "a resize to SIZE_MAX fails and returns NULL (true for realloc too)");

            // Sufficient. A live block's address cannot be reissued, so seeing it again proves the
            // failed resize released it.
            const int kTries = 8;
            void* probe[kTries] = {};
            bool reissued = false;
            for (int i = 0; i < kTries; ++i) {
                probe[i] = (void*)(uintptr_t)malloc_fn(kSize, 0, 0, 0, 0, 0);
                if ((uintptr_t)probe[i] == victim_addr) reissued = true;
            }
            CHECK(reissued,
                  "the failed resize RELEASED the original block -- its address is reissued to a "
                  "later same-size allocation, which a still-live block's address never could be");
            for (int i = 0; i < kTries; ++i)
                if (probe[i]) free_fn(U(probe[i]), 0, 0, 0, 0, 0);
        }
    }

    // --- reallocf(p, 0) must not double-free -----------------------------------------------------
    // The zero case is implementation-defined (C17 7.22.3.5) and the platforms genuinely differ, so
    // this arm asserts the contract that holds everywhere rather than one platform's spelling of it.
    // An earlier revision asserted `r == nullptr` and was red on macOS -- correctly, and the code was
    // never wrong; the ASSERTION was:
    //
    //   glibc    realloc(p, 0) frees and returns NULL
    //   Windows  guest_realloc_portable's `if (!size)` frees and returns NULL
    //   macOS    realloc(p, 0) returns a VALID minimum-size block and frees nothing
    //
    // What must be true on all three is that the free-on-failure rule does NOT fire on top of a
    // release that already happened. Both outcomes are checked for their own follow-through: a NULL
    // means the block is gone and nothing more is owed, a non-NULL means the caller now owns a real
    // block and must free it. The heap-intact probe afterwards is the double-free detector -- glibc
    // aborts outright on a double free, so reaching the probe is most of the evidence and the
    // successful allocation confirms the heap is usable rather than merely un-aborted.
    {
        auto* p = (void*)(uintptr_t)malloc_fn(64, 0, 0, 0, 0, 0);
        CHECK(p != nullptr, "malloc returns the block for the zero-size arm");
        if (p) {
            auto* r = (void*)(uintptr_t)reallocf_fn(U(p), 0, 0, 0, 0, 0);
            // Which branch runs is a platform fact, not a pass/fail condition, so it is REPORTED
            // rather than asserted -- an assertion that cannot fail is not evidence.
            if (r) {
                printf("  [note] reallocf(p, 0) returned a block (macOS-style, C17-conforming)\n");
                *(volatile unsigned char*)r = 0xA5;   // real storage, not a non-null sentinel
                free_fn(U(r), 0, 0, 0, 0, 0);
            } else {
                printf("  [note] reallocf(p, 0) returned NULL, block released (glibc/Windows)\n");
            }
            auto* after = (void*)(uintptr_t)malloc_fn(64, 0, 0, 0, 0, 0);
            CHECK(after != nullptr, "the heap is intact afterwards -- no double free on the 0 path");
            if (after) free_fn(U(after), 0, 0, 0, 0, 0);
        }
    }

    printf("== %d failure(s) ==\n", fails);
    return fails ? 1 : 0;
}

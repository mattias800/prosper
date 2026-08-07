// reallocalign (OGybVuPAhAY) must resize a block AND give the result the requested extended
// alignment (#2185).
//
// KNOWN GAP, so nobody reads these arms as covering more than they do: the checks below detect an
// over-WRITE (the copy bound too large in the destination direction) by two mechanisms — the slack
// guard for a sub-fatal overrun, and glibc's own heap check for a gross one. **Neither detects an
// over-READ.** Dropping the cap in the source direction (`memcpy(dst, src, size)`) over-reads past
// the source block on the grow arm, and every assertion here still passes: no out-of-bounds write
// occurs, the copied prefix is correct, and the shrink arm is unaffected. That direction is the more
// on-point half, since bounding a read out of a block whose request size is unrecoverable is the
// whole reason malloc_usable_size appears in the implementation. Detecting it needs a guard page,
// i.e. abandoning posix_memalign and the real allocation path — deliberately not done. **ASan catches
// it cleanly; run this suite under a sanitizer if you change the copy.**
//
// Unregistered, this NID fell to the dispatcher's default 0. For an allocator that is a NULL
// return, which a caller reads as OOM — a false FAILURE, the mirror of #2081's false successes:
// a title treating allocation failure as fatal aborts, and one that retries loops. So the arm that
// matters most is simply "does it answer at all", and it is structurally red without the fix:
// Hle::lookup returns nullptr and the registration CHECK fails before anything else runs.
//
// Why the function has to exist separately from realloc, which is the thing worth not
// re-deriving: a two-argument resize is never told an alignment. C17 7.22.3p1 guarantees only
// FUNDAMENTAL alignment (<= alignof(max_align_t), 6.2.8p2); an extended alignment (6.2.8p3) such
// as 256 may be dropped the moment realloc relocates the block. #2165 measured exactly that —
// unperturbed, glibc grew the block in place and the alignment survived by luck; perturbed, it
// relocated to 16- but not 256-aligned. Sony ships `reallocalign` beside `realloc` for this
// reason, and pairs sceLibcMspaceRealloc with sceLibcMspaceReallocalign.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#if !defined(_WIN32)
#  if defined(__APPLE__)
#    include <malloc/malloc.h>
#    define USABLE(p) malloc_size(p)
#  else
#    include <malloc.h>
#    define USABLE(p) malloc_usable_size(p)
#  endif
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t U(const void* p) { return (uint64_t)(uintptr_t)p; }

int main() {
    printf("== test_reallocalign ==\n");
    register_builtin_hle();

    HleFn reallocalign_fn = Hle::lookup(nid_hash("reallocalign"));
    HleFn memalign_fn     = Hle::lookup(nid_hash("memalign"));
    HleFn free_fn         = Hle::lookup(nid_hash("free"));

    // The whole defect in one assertion: unregistered, the dispatcher answers 0 and the guest
    // reads OOM. Everything below depends on this, so a regression stops here rather than
    // producing a confusing cascade.
    CHECK(reallocalign_fn != nullptr, "reallocalign is registered (unregistered -> dispatcher 0 -> reads as OOM)");
    CHECK(memalign_fn != nullptr && free_fn != nullptr, "memalign/free are registered");
    if (!reallocalign_fn || !memalign_fn || !free_fn) {
        printf("== %d failure(s) ==\n", fails);
        return 1;
    }

    // --- grow, preserving BOTH contents and the extended alignment ------------------------------
    {
        const size_t kAlign = 256, kOld = 513, kNew = 4099;
        auto* p = (uint8_t*)(uintptr_t)memalign_fn(kAlign, kOld, 0, 0, 0, 0);
        CHECK(p != nullptr && ((uintptr_t)p % kAlign) == 0, "memalign returns a 256-aligned block");
        if (!p) { printf("== %d failure(s) ==\n", fails); return 1; }
        for (size_t i = 0; i < kOld; ++i) p[i] = (uint8_t)(i * 31u + 7u);

        auto* grown = (uint8_t*)(uintptr_t)reallocalign_fn(U(p), kNew, kAlign, 0, 0, 0);
        CHECK(grown != nullptr, "reallocalign returns storage rather than null (a null here IS the OOM report)");
        if (grown) {
            CHECK(((uintptr_t)grown % kAlign) == 0,
                  "the RESIZED block carries the requested 256-byte alignment");
            bool kept = true;
            for (size_t i = 0; i < kOld && kept; ++i) kept = grown[i] == (uint8_t)(i * 31u + 7u);
            CHECK(kept, "contents survive the resize, all 513 original bytes");
            free_fn(U(grown), 0, 0, 0, 0, 0);
        }
    }

    // --- shrink: the copy must be capped by the NEW size, not the old readable extent -----------
    // This is the arm that catches a bounds error in the copy. If the implementation copied the
    // old block's usable size unconditionally it would overrun a smaller destination, which under
    // a hardened allocator aborts and otherwise corrupts the heap silently.
    {
        const size_t kAlign = 128, kOld = 8192, kNew = 64;
        auto* p = (uint8_t*)(uintptr_t)memalign_fn(kAlign, kOld, 0, 0, 0, 0);
        if (p) {
            memset(p, 0xC3, kOld);
            auto* small = (uint8_t*)(uintptr_t)reallocalign_fn(U(p), kNew, kAlign, 0, 0, 0);
            CHECK(small != nullptr, "reallocalign shrinks without reporting failure");
            if (small) {
                CHECK(((uintptr_t)small % kAlign) == 0, "a shrunk block keeps the requested alignment");
                bool kept = true;
                for (size_t i = 0; i < kNew && kept; ++i) kept = small[i] == 0xC3;
                CHECK(kept, "the surviving prefix is intact after a shrink");
#if !defined(_WIN32)
                // The arm that makes an over-copy DETECTABLE rather than merely fatal-sometimes.
                // Without it, an implementation copying the source's full readable extent instead of
                // min(extent, new_size) is caught only by crashing — and a destination inside a large
                // free region corrupts silently and the suite exits 0. Here the source is 8192 bytes
                // of 0xC3 and the request is 64, so an unbounded copy necessarily writes 0xC3 into
                // the destination's own slack. That slack is readable by the same argument the
                // implementation relies on, so this checks a defined region.
                const size_t usable = USABLE(small);
                // Make the guard's own vacuity VISIBLE. The window is non-empty only because of the
                // constant chosen: glibc's size ladder is 24, 40, 56, 72, ... so usable(64) = 72 and
                // eight bytes are checkable. Pick kNew = 72, 88 or 104 and the loop body never runs
                // while still printing [ok]. It is genuinely empty on macOS (malloc_size(64) == 64),
                // on musl, and under ASan (which intercepts and returns the REQUESTED size) — so on
                // those configurations this arm reports nothing and must say so rather than pass.
                // An empty window is a property of the host allocator, NOT a defect in
                // reallocalign, so it is reported and skipped rather than failed — macOS, musl and
                // ASan are all legitimate configurations where no slack is inspectable, and failing
                // there makes the suite red on a correct implementation. It must still be LOUD:
                // running the loop anyway would print [ok] for an arm whose body never executed,
                // which is precisely the vacuous pass this guard exists to prevent.
                if (usable > kNew) {
                    bool slack_clean = true;
                    for (size_t i = kNew; i < usable && slack_clean; ++i) slack_clean = small[i] != 0xC3;
                    CHECK(slack_clean,
                          "the copy is capped by the NEW size — no source bytes past it (over-copy guard)");
                } else {
                    printf("  [skip] over-copy guard did NOT run: this allocator reports usable=%zu "
                           "for a %zu-byte request, so there is no readable slack to inspect\n",
                           usable, kNew);
                }
#endif
                free_fn(U(small), 0, 0, 0, 0, 0);
            }
        }
    }

    // --- a null pointer behaves as an aligned allocation ----------------------------------------
    {
        auto* fresh = (uint8_t*)(uintptr_t)reallocalign_fn(0, 300, 512, 0, 0, 0);
        CHECK(fresh != nullptr && ((uintptr_t)fresh % 512) == 0,
              "reallocalign(nullptr, n, align) allocates, like realloc(nullptr, n)");
        if (fresh) free_fn(U(fresh), 0, 0, 0, 0, 0);
    }

    // --- a non-power-of-two alignment is a parameter error, not a silent success ----------------
    // Direction matters: answering with unaligned storage would be worse than failing, because the
    // caller asked for the alignment precisely because it cannot use storage without it.
    {
        auto* p = (uint8_t*)(uintptr_t)memalign_fn(64, 128, 0, 0, 0, 0);
        if (p) {
            auto* bad = (uint8_t*)(uintptr_t)reallocalign_fn(U(p), 256, 300, 0, 0, 0);
            CHECK(bad == nullptr, "a non-power-of-two alignment fails rather than returning misaligned storage");
            if (bad) free_fn(U(bad), 0, 0, 0, 0, 0);
            else     free_fn(U(p), 0, 0, 0, 0, 0);
        }
    }

    printf(fails ? "== %d failure(s) ==\n" : "== all checks passed ==\n", fails);
    return fails ? 1 : 0;
}

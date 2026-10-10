// reallocf (YMZO9ChZb0E) is realloc except that the ORIGINAL block is freed when the resize fails
// (#2203). BSD ships it because `p = realloc(p, n)` leaks `p` on failure, so a caller of reallocf
// has delegated the free-on-failure to libc and will not do it itself.
//
// Unregistered, the NID fell to the dispatcher's default 0 — a NULL return the caller reads as OOM.
// That is a false FAILURE on every call, not only on the failure path: a resize that should have
// succeeded reported allocation failure. So the registration arm below is the whole defect, and it
// is structurally red without the fix (Hle::lookup returns nullptr).
//
// A null result alone cannot distinguish reallocf from realloc. The ownership tests exercise the
// registered handler's shared policy with an allocator that records each release. Omitting the
// release fails deterministically, without assuming the host will reuse a freed address. The old
// eight-allocation reuse probe intermittently failed on macOS even when the block was released.
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include "hle/dispatch/nid.hpp"
#include "hle/libc/reallocf.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) EXPECT_TRUE(c) << (m)

static uint64_t U(const void* p) { return (uint64_t)(uintptr_t)p; }

TEST(Reallocf, Contract) {
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
        FAIL() << "legacy early exit";
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

    // The real host allocator refuses an impossible resize. Release ownership is checked below
    // against the same policy used by this registered handler, rather than host reuse timing.
    {
        const size_t kSize = 96;
        const size_t kImpossible = (size_t)-1;   // SIZE_MAX: no allocator can satisfy this
        auto* victim = (uint8_t*)(uintptr_t)malloc_fn(kSize, 0, 0, 0, 0, 0);
        CHECK(victim != nullptr, "malloc returns the block that the failed resize must release");
        if (victim) {
            auto* r = (void*)(uintptr_t)reallocf_fn(U(victim), kImpossible, 0, 0, 0, 0);
            // Necessary but NOT sufficient on its own -- a realloc alias also returns NULL here.
            // Recorded as such so nobody later reads this line as the coverage.
            CHECK(r == nullptr, "a resize to SIZE_MAX fails and returns NULL (true for realloc too)");
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

    EXPECT_EQ(fails, 0);
}

TEST(Reallocf, FailedResizeReleasesOriginalExactlyOnce) {
    int original = 0;
    void* released = nullptr;
    int releases = 0;
    auto resize = [&](void* p, size_t size) -> void* {
        EXPECT_EQ(p, &original) << "the allocator receives the original block";
        EXPECT_EQ(size, SIZE_MAX) << "the requested failing size reaches the allocator";
        return nullptr;
    };
    auto release = [&](void* p) {
        released = p;
        ++releases;
    };
    EXPECT_EQ(libc::realloc_free_on_failure(&original, SIZE_MAX, resize, release), nullptr)
        << "the allocator's failure is returned unchanged";
    EXPECT_EQ(released, &original) << "allocation failure transfers the original to release";
    EXPECT_EQ(releases, 1) << "the original is released exactly once";
}

TEST(Reallocf, SuccessfulResizeDoesNotReleaseReturnedStorage) {
    int original = 0;
    int replacement = 0;
    int releases = 0;
    auto resize = [&](void* p, size_t size) -> void* {
        EXPECT_EQ(p, &original) << "the allocator receives the original block";
        EXPECT_EQ(size, 4096u) << "the successful requested size reaches the allocator";
        return &replacement;
    };
    auto release = [&](void*) { ++releases; };
    EXPECT_EQ(libc::realloc_free_on_failure(&original, 4096, resize, release), &replacement)
        << "the allocator's replacement is returned unchanged";
    EXPECT_EQ(releases, 0) << "a successful resize owes no extra release";
}

TEST(Reallocf, NullOriginalHasNothingToReleaseOnFailure) {
    int releases = 0;
    auto resize = [](void* p, size_t size) -> void* {
        EXPECT_EQ(p, nullptr) << "a null original is passed through";
        EXPECT_EQ(size, SIZE_MAX) << "a fresh allocation can also fail";
        return nullptr;
    };
    auto release = [&](void*) { ++releases; };
    EXPECT_EQ(libc::realloc_free_on_failure(nullptr, SIZE_MAX, resize, release), nullptr)
        << "fresh allocation failure also returns null";
    EXPECT_EQ(releases, 0) << "there is no original allocation to release";
}

TEST(Reallocf, ZeroSizeRespectsBothAllocatorPolicies) {
    int original = 0;
    int minimum_block = 0;
    int releases = 0;
    auto release = [&](void* p) {
        EXPECT_EQ(p, &original) << "only the original allocation is eligible for release";
        ++releases;
    };
    auto resize_and_release = [&](void* p, size_t size) -> void* {
        EXPECT_EQ(size, 0u) << "the allocator decides the zero-size policy";
        release(p);
        return nullptr;
    };
    EXPECT_EQ(libc::realloc_free_on_failure(&original, 0, resize_and_release, release), nullptr)
        << "the null zero-size policy is preserved";
    EXPECT_EQ(releases, 1) << "a null zero-size result must not release the block twice";
    auto resize_to_minimum = [&](void* p, size_t size) -> void* {
        EXPECT_EQ(p, &original) << "the other policy receives the same original";
        EXPECT_EQ(size, 0u) << "a zero-size resize may return valid minimum storage";
        return &minimum_block;
    };
    EXPECT_EQ(libc::realloc_free_on_failure(&original, 0, resize_to_minimum, release),
              &minimum_block)
        << "the minimum-allocation zero-size policy is also preserved";
    EXPECT_EQ(releases, 1) << "a non-null zero-size result also owes no extra release";
}

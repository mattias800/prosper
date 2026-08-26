// test_kernel_nanosleep (#3013) — the guest timespec contract of sceKernelNanosleep, which is a
// GUEST-memory layout question and not a timing one.
//
// Two properties, and both were broken at some point in #3022's history, which is why they are pinned
// here rather than argued in a comment:
//
//  1. The remainder out-param is filled with SIXTEEN zero bytes. The guest timespec is FreeBSD
//     x86-64's {int64 tv_sec, int64 tv_nsec}; MinGW-w64's HOST struct declares `long tv_nsec`, which
//     is 32-bit on Windows x64. Writing the remainder through a host struct therefore covered 12 of
//     the 16 bytes and left the high half of the guest's tv_nsec holding whatever was there, so a
//     guest reading it back saw a non-zero remainder and could resume a wait already served. This
//     test seeds BOTH slots with a sentinel, so it fails on any partial write.
//
//  2. A malformed request returns PROMPTLY. POSIX makes tv_nsec outside [0, 1e9) EINVAL, and the
//     original body -- which handed the guest struct to the host nanosleep -- was refused instantly
//     and returned 0 having slept nothing. An intermediate version of the fix carried the
//     out-of-range value into the total instead, turning a garbage tv_nsec into a near-infinite
//     sleep. Asserting "promptly" rather than an exact duration keeps this off the host scheduler:
//     the failure mode it guards is seconds-to-forever, so a generous ceiling still catches it.
//
// Deliberately NOT a test of sleep accuracy. That is a property of host::sleep_until_steady_ns and a
// wall-clock assertion on it would be a flake on a loaded machine; precise_sleep exposes
// sleep_backend() so the MECHANISM can be asserted instead.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_kernel_nanosleep ==\n");
    register_builtin_hle();

    const std::string nid = nid_hash("sceKernelNanosleep");
    HleFn ns = Hle::lookup(nid.c_str());
    CHECK(ns != nullptr, "sceKernelNanosleep is registered (resolves to a real impl, not the stub)");
    if (!ns) { printf("== FAIL (unresolved) ==\n"); return 1; }

    constexpr int64_t kSentinel = (int64_t)0x0BADF00DDEADBEEFll;

    // 1. A 1 ms request: both remainder slots must come back zero.
    {
        int64_t req[2] = { 0, 1000000 };                 // 1 ms
        int64_t rem[2] = { kSentinel, kSentinel };
        const uint64_t rc = ns((uint64_t)(uintptr_t)req, (uint64_t)(uintptr_t)rem, 0, 0, 0, 0);
        CHECK(rc == 0, "a valid request returns 0");
        CHECK(rem[0] == 0, "remainder tv_sec is zeroed");
        CHECK(rem[1] == 0, "remainder tv_nsec is zeroed IN FULL (all 8 bytes, not just the low 4)");
    }

    // 2. tv_nsec >= 1e9 is malformed: return promptly, sleep nothing, and still define the remainder.
    {
        int64_t req[2] = { 0, 2000000000ll };            // 2e9 ns: out of range
        int64_t rem[2] = { kSentinel, kSentinel };
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t rc = ns((uint64_t)(uintptr_t)req, (uint64_t)(uintptr_t)rem, 0, 0, 0, 0);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        CHECK(rc == 0, "a malformed request still returns 0 (the contract this entry point always had)");
        CHECK(ms < 500.0, "a malformed tv_nsec does NOT become a long sleep (carried, it was ~2 s)");
        CHECK(rem[0] == 0 && rem[1] == 0, "the remainder is defined even on a refused request");
    }

    // 3. A garbage tv_nsec must not sleep for years. Same guard, at the value that motivated it.
    {
        int64_t req[2] = { 0, (int64_t)0x7fffffffffffffffll };
        const auto t0 = std::chrono::steady_clock::now();
        ns((uint64_t)(uintptr_t)req, 0, 0, 0, 0, 0);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        CHECK(ms < 500.0, "a garbage tv_nsec returns promptly rather than sleeping ~292 years");
    }

    // 4. Contract documentation, NOT a regression arm, and labelled so rather than left to look like
    //    one: a negative tv_sec and a NULL request both returned 0 promptly in every version of this
    //    body -- the original (host nanosleep EINVALs), the intermediate one (negatives clamped to 0),
    //    and the current guard. So these cannot fail for the fix's sake and pin only the stable part
    //    of the contract. Kept because that part is worth stating; counted honestly because an arm
    //    that cannot distinguish the versions is not evidence about them.
    {
        int64_t req[2] = { -1, 0 };
        CHECK(ns((uint64_t)(uintptr_t)req, 0, 0, 0, 0, 0) == 0, "a negative tv_sec returns 0");
        CHECK(ns(0, 0, 0, 0, 0, 0) == 0, "a NULL request is a no-op returning 0");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}

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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

using namespace prosper;

namespace {
constexpr int64_t kSentinel = static_cast<int64_t>(0x0BADF00DDEADBEEFll);
// Generous on purpose: the defect this bounds is seconds-to-forever, so a ceiling this loose still
// catches it without putting a wall-clock assertion on a loaded host scheduler.
constexpr double kPromptMs = 500.0;

HleFn nanosleep() {
    register_builtin_hle();
    return Hle::lookup(nid_hash("sceKernelNanosleep").c_str());
}
double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
}   // namespace

TEST(KernelNanosleep, ResolvesToARealImplementation) {
    EXPECT_NE(nanosleep(), nullptr)
        << "sceKernelNanosleep is registered (resolves to a real impl, not the stub)";
}

TEST(KernelNanosleep, ValidRequestZeroesBothRemainderSlots) {
    const HleFn ns = nanosleep();
    ASSERT_NE(ns, nullptr);
    int64_t req[2] = {0, 1000000};   // 1 ms
    int64_t rem[2] = {kSentinel, kSentinel};
    EXPECT_EQ(ns(reinterpret_cast<uint64_t>(req), reinterpret_cast<uint64_t>(rem), 0, 0, 0, 0), 0u)
        << "a valid request returns 0";
    EXPECT_EQ(rem[0], 0) << "remainder tv_sec is zeroed";
    EXPECT_EQ(rem[1], 0) << "remainder tv_nsec is zeroed IN FULL (all 8 bytes, not just the low 4)";
}

TEST(KernelNanosleep, MalformedTvNsecReturnsPromptlyAndStillDefinesTheRemainder) {
    const HleFn ns = nanosleep();
    ASSERT_NE(ns, nullptr);
    int64_t req[2] = {0, 2000000000ll};   // 2e9 ns: out of range
    int64_t rem[2] = {kSentinel, kSentinel};
    const auto start = std::chrono::steady_clock::now();
    const uint64_t rc =
        ns(reinterpret_cast<uint64_t>(req), reinterpret_cast<uint64_t>(rem), 0, 0, 0, 0);
    const double ms = elapsed_ms_since(start);
    EXPECT_EQ(rc, 0u)
        << "a malformed request still returns 0 (the contract this entry point always had)";
    EXPECT_LT(ms, kPromptMs)
        << "a malformed tv_nsec does NOT become a long sleep (carried, it was ~2 s)";
    EXPECT_EQ(rem[0], 0) << "the remainder is defined even on a refused request";
    EXPECT_EQ(rem[1], 0) << "the remainder is defined even on a refused request";
}

TEST(KernelNanosleep, GarbageTvNsecDoesNotSleepForYears) {
    // Same guard, at the value that motivated it: 2^63-1 nanoseconds is ~292 years.
    const HleFn ns = nanosleep();
    ASSERT_NE(ns, nullptr);
    int64_t req[2] = {0, static_cast<int64_t>(0x7fffffffffffffffll)};
    const auto start = std::chrono::steady_clock::now();
    ns(reinterpret_cast<uint64_t>(req), 0, 0, 0, 0, 0);
    EXPECT_LT(elapsed_ms_since(start), kPromptMs)
        << "a garbage tv_nsec returns promptly rather than sleeping ~292 years";
}

TEST(KernelNanosleep, StableContractArmsAreNotRegressionsForTheFix) {
    // Contract documentation, NOT a regression arm, and named as such rather than left to look like
    // one: a negative tv_sec and a NULL request both returned 0 promptly in every version of this
    // body -- the original (host nanosleep EINVALs), the intermediate one (negatives clamped to 0),
    // and the current guard. So these cannot fail for the fix's sake and pin only the stable part
    // of the contract. Kept because that part is worth stating; counted honestly because an arm
    // that cannot distinguish the versions is not evidence about them.
    const HleFn ns = nanosleep();
    ASSERT_NE(ns, nullptr);
    int64_t req[2] = {-1, 0};
    EXPECT_EQ(ns(reinterpret_cast<uint64_t>(req), 0, 0, 0, 0, 0), 0u)
        << "a negative tv_sec returns 0";
    EXPECT_EQ(ns(0, 0, 0, 0, 0, 0), 0u) << "a NULL request is a no-op returning 0";
}
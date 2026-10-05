// test_select_sleep - select/pselect honour the pure-sleep shape and refuse everything else (#1660).
//
// `select(0, NULL, NULL, NULL, &tv)` is the portable sleep idiom: nfds <= 0 with all three
// descriptor sets NULL, so no descriptor is ever examined. Before this was implemented the call fell
// through to the generic unimplemented stub, which returns 0 in ~111 ns - collapsing a 3-second
// sleep into a no-op. On PPSA19244 that made one service thread spin **9.0 million times per
// second** (1,300,185,430 calls in 144 s), saturating a core through the dispatch path.
//
// The contract under test, and the reason each half matters:
//
//   1. The pure-sleep shape must WAIT. `0` is the correct return ("timed out, nothing ready") - the
//      defect was returning it immediately. So the assertion is on ELAPSED TIME, not the value: a
//      test that only checked the return value passes against the very stub this replaces.
//   2. A descriptor-set query must stay FAIL-VISIBLE. prosper has no socket backing, so answering
//      "0 = nothing ready" would be the same success-shaped lie in a new place. It must return -1.
//
// Both halves fail without the fix: (1) elapses ~0 ms against the unimplemented stub, and (2)
// returns 0 rather than -1.
//
// Section 1b (#3038) adds a third: an out-of-range timeout must be REFUSED, and refusing must mean
// returning promptly rather than sleeping. Its own comment says what it can and cannot discriminate
// on a 64-bit-`long` host, which is not the same question as what it is worth pinning.
//
// Split into four cases: the NID identity, the sleep shape, the refused timeouts, and the
// fail-visible descriptor queries. The last two share one property -- what the call reports when it
// will not do the thing -- and are separated so a red case says which half moved.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>

using namespace prosper;

namespace {
int64_t elapsed_ms(const std::function<void()>& body) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 t0)
        .count();
}
uint64_t U(const void* p) {
    return reinterpret_cast<uint64_t>(p);
}

struct SelectFamily {
    HleFn select = nullptr, pselect = nullptr;
};
SelectFamily select_family() {
    register_builtin_hle();
    SelectFamily family;
    family.select = Hle::lookup(nid_hash("select"));
    family.pselect = Hle::lookup(nid_hash("pselect"));
    return family;
}
}   // namespace

TEST(SelectSleep, TheTwoEntryPointsBindTheirRealPsaExportNids) {
    // The guest binds these by NID; assert the hash matches the real PS5 3.20 export so the
    // registration cannot silently stop binding (nid_hash is the same function the stub emitter
    // uses, and these are the NIDs PPSA19244's import table actually carries).
    EXPECT_EQ(nid_hash("select"), "T8fER+tIGgk")
        << "nid_hash(\"select\") matches the PS5 3.20 export";
    EXPECT_EQ(nid_hash("pselect"), "ZO2nWoTAv60")
        << "nid_hash(\"pselect\") matches the PS5 3.20 export";
    const SelectFamily family = select_family();
    EXPECT_NE(family.select, nullptr)
        << "select is registered (not left to the unimplemented stub)";
    EXPECT_NE(family.pselect, nullptr) << "pselect is registered";
}

TEST(SelectSleep, ThePureSleepShapeWaitsAndThenReportsNothingReady) {
    const SelectFamily family = select_family();
    ASSERT_NE(family.select, nullptr);
    ASSERT_NE(family.pselect, nullptr);
    // 150 ms is long enough to be unambiguous against scheduler noise and short enough for ctest.
    // Assert a small tolerance below the request rather than exact equality: nanosleep may round.
    {
        int64_t tv[2] = {0, 150000};   // struct timeval { sec, usec }
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.select(0, 0, 0, 0, U(tv), 0); });
        std::printf("  select(0,NULL,NULL,NULL,{0,150000}) -> %llu in %lld ms\n",
                    (unsigned long long)rc, (long long)ms);
        EXPECT_GE(ms, 140) << "select honours a 150 ms timeout (the stub returned in ~0 ms)";
        EXPECT_EQ(rc, 0u) << "select returns 0 (timed out, nothing ready) after waiting";
    }
    {
        int64_t ts[2] = {0, 150000000};   // struct timespec { sec, nsec }
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.pselect(0, 0, 0, 0, U(ts), 0); });
        std::printf("  pselect(0,NULL,NULL,NULL,{0,150000000}) -> %llu in %lld ms\n",
                    (unsigned long long)rc, (long long)ms);
        EXPECT_GE(ms, 140) << "pselect honours a 150 ms timespec timeout";
        EXPECT_EQ(rc, 0u) << "pselect returns 0 after waiting";
    }
    {
        // A zero timeout is a legal poll of an empty set: it must return promptly, and must not be
        // confused with the unbounded-wait (NULL timeout) case.
        int64_t tv[2] = {0, 0};
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.select(0, 0, 0, 0, U(tv), 0); });
        EXPECT_LT(ms, 100) << "select with a zero timeout returns promptly";
        EXPECT_EQ(rc, 0u) << "select with a zero timeout returns 0";
    }
}

TEST(SelectSleep, AnOutOfRangeTimeoutIsRefusedPromptlyAndStillReportsNothingReady) {
    // POSIX puts tv_usec in [0, 1e6) and tv_nsec in [0, 1e9). Before #3038 these call sites applied
    // no range rule of their own: they built a host `struct timespec` and let the host nanosleep
    // notice, which returned EINVAL instantly. That is the behaviour pinned here, because the
    // tempting "fix" is to carry the out-of-range field into the total instead -- which turns a
    // garbage 0x7fff... into a near-infinite sleep, i.e. a HANG where there used to be an immediate
    // return. k_nanosleep's own guard comment records that being got wrong once already; these arms
    // make the same mistake in select()/pselect() a red test rather than a wedged title.
    //
    // Deliberately stated: on a 64-bit-`long` host these arms pass both before and after #3038, so
    // they are a REGRESSION pin, not a discriminator for that fix. The half they can discriminate
    // is the one above -- a value carried into the total -- and the half they cannot is the 32-bit
    // narrowing, which this host cannot express at all. That one is tested as pure arithmetic in
    // tests/host/platform/test_precise_sleep.cpp arms 18-21.
    const SelectFamily family = select_family();
    ASSERT_NE(family.select, nullptr);
    ASSERT_NE(family.pselect, nullptr);
    {
        int64_t tv[2] = {0, 1500000};   // 1.5e6 us: out of range, and > 1 s
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.select(0, 0, 0, 0, U(tv), 0); });
        std::printf("  select(0,NULL,NULL,NULL,{0,1500000}) -> %llu in %lld ms\n",
                    (unsigned long long)rc, (long long)ms);
        EXPECT_LT(ms, 100) << "an out-of-range tv_usec returns promptly, never as a 1.5 s sleep";
        EXPECT_EQ(rc, 0u) << "...and still returns 0, the answer this call already gave";
    }
    {
        // The value that the pre-#3038 32-bit narrowing mapped back INTO range as a 704 ns sleep on
        // Windows: 4,294,968 us is 4.29 s. Out of range by any reading, so the answer is a refusal.
        int64_t tv[2] = {0, 4294968};
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.select(0, 0, 0, 0, U(tv), 0); });
        EXPECT_LT(ms, 100)
            << "the 4.29 s out-of-range tv_usec is refused, not slept and not re-mapped";
        EXPECT_EQ(rc, 0u) << "...returning 0";
    }
    {
        // A tv_usec whose x1000 product would overflow a signed 64-bit multiply. Before #3038 the
        // multiply happened BEFORE any range check, on a value read straight from guest memory --
        // undefined behaviour on every platform, not merely a narrowing. Now it cannot be reached.
        int64_t tv[2] = {0, INT64_MAX};
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.select(0, 0, 0, 0, U(tv), 0); });
        EXPECT_LT(ms, 100)
            << "a tv_usec that would overflow the x1000 multiply is refused before it";
        EXPECT_EQ(rc, 0u) << "...returning 0";
    }
    {
        int64_t ts[2] = {0, 1000000000};   // exactly 1e9 ns: out of range
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.pselect(0, 0, 0, 0, U(ts), 0); });
        EXPECT_LT(ms, 100)
            << "pselect refuses a tv_nsec of exactly 1e9 rather than sleeping a second";
        EXPECT_EQ(rc, 0u) << "...returning 0";
    }
    {
        int64_t ts[2] = {0, INT64_MAX};   // the garbage-value hang direction
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.pselect(0, 0, 0, 0, U(ts), 0); });
        EXPECT_LT(ms, 100)
            << "a 0x7fff... tv_nsec returns at once; carrying it would be a 292-year wait";
        EXPECT_EQ(rc, 0u) << "...returning 0";
    }
    {
        int64_t ts[2] = {-5, 0};   // a negative tv_sec
        uint64_t rc = 0;
        const int64_t ms = elapsed_ms([&] { rc = family.pselect(0, 0, 0, 0, U(ts), 0); });
        EXPECT_LT(ms, 100) << "a negative tv_sec returns promptly";
        EXPECT_EQ(rc, 0u) << "...returning 0";
    }
}

TEST(SelectSleep, ADescriptorSetQueryStaysFailVisible) {
    // This is the half that keeps the fix honest: the sleep path must NOT be widened to cover a
    // real readiness query, because prosper has no socket backing to answer one with.
    const SelectFamily family = select_family();
    ASSERT_NE(family.select, nullptr);
    ASSERT_NE(family.pselect, nullptr);
    {
        uint64_t fdset[16] = {0};   // contents irrelevant; non-NULL is
        int64_t tv[2] = {0, 0};   // what makes this a real query
        errno = 0;
        const uint64_t rc = family.select(4, U(fdset), 0, 0, U(tv), 0);
        std::printf("  select(4, &readfds, ...) -> %lld (errno=%d)\n", (long long)(int64_t)rc,
                    errno);
        EXPECT_EQ(static_cast<int64_t>(rc), -1)
            << "select with a readfds set fails visibly (-1), never a false 0";
        // FreeBSD ENOSYS (78), NOT the host constant: the guest reads this slot through
        // __error() and compares against its own numbering (host ENOSYS is 38 on Linux, 40 on
        // MinGW). Writing `ENOSYS` here would assert the host value and pass on exactly the
        // code #2296 is about. On a host whose ENOSYS is already 78 (Darwin) this arm cannot
        // distinguish -- the printf above shows which case a given run is.
        EXPECT_EQ(errno, static_cast<int>(prosper::hle::FreeBsdErrno::ENoSys))
            << "select publishes FreeBSD ENOSYS (78) for the unimplemented descriptor query";
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = family.select(4, 0, U(fdset), 0, 0, 0);
        EXPECT_EQ(static_cast<int64_t>(rc), -1) << "select with a writefds set fails visibly (-1)";
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = family.select(4, 0, 0, U(fdset), 0, 0);
        EXPECT_EQ(static_cast<int64_t>(rc), -1)
            << "select with an exceptfds set fails visibly (-1)";
    }
    {
        uint64_t fdset[16] = {0};
        const uint64_t rc = family.pselect(4, U(fdset), 0, 0, 0, 0);
        EXPECT_EQ(static_cast<int64_t>(rc), -1) << "pselect with a readfds set fails visibly (-1)";
    }
    {
        // The other half of the conjunction the implemented shape is built from. `nfds <= 0` with a
        // non-NULL set examines no descriptor either, so it is a sleep by the same argument as
        // `nfds > 0` with all sets NULL above -- but the evidence only ever supported the narrower
        // shape, and this arm is what keeps `select_is_pure_sleep`'s set terms load-bearing rather
        // than decorative. Without it, dropping those three terms from the predicate still passes
        // every arm above, because they all pass nfds > 0 and are refused by the nfds test alone
        // (measured: `nfds <= 0` alone leaves this file fully green).
        uint64_t fdset[16] = {0};
        int64_t tv[2] = {0, 0};
        const uint64_t rc = family.select(0, U(fdset), 0, 0, U(tv), 0);
        EXPECT_EQ(static_cast<int64_t>(rc), -1) << "nfds<=0 with a non-NULL set is outside the "
                                                   "observed shape too: refused, not assumed";
    }
    {
        // Deliberate scope boundary. `nfds > 0` with every set NULL examines no descriptor either, so
        // by POSIX semantics it is *also* a sleep -- but no title has been observed making that call,
        // so the implemented shape stays exactly the one the evidence supports (`nfds <= 0` AND all
        // sets NULL) and this form is refused loudly rather than assumed. Pinning it here means
        // widening the contract later has to be a deliberate edit with its own evidence, not an
        // accident.
        int64_t tv[2] = {0, 0};
        const uint64_t rc = family.select(8, 0, 0, 0, U(tv), 0);
        EXPECT_EQ(static_cast<int64_t>(rc), -1)
            << "nfds>0 with all sets NULL is outside the observed shape: refused, not assumed";
    }
}
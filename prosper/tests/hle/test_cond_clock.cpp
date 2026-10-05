// Condition attributes retain the PS5 clock identity and absolute waits use that selected clock.
// Before #386/F8, CondInit discarded the attribute, so a MONOTONIC/CPU-clock deadline was treated
// as CLOCK_REALTIME and normally appeared decades in the past.
//
// Split into one TEST per behaviour, which is also one TEST per object: each case builds its own
// condattr/cond/mutex, so no case depends on state a sibling left behind (each ctest case is its own
// process, and the handlers it drives are per-object state, not per-process).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/sce_errno.hpp"   // #2178: the Sony spellings report the libkernel encoding
#include "hle/sync/sync_futex.hpp"

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <pthread.h>
#include <thread>

using namespace prosper;

namespace {
struct GuestTimespec {
    int64_t sec;
    int64_t nsec;
};

uint64_t U(const void* value) {
    return reinterpret_cast<uint64_t>(value);
}

GuestTimespec deadline_after(clockid_t clock, int64_t milliseconds) {
    timespec now{};
    clock_gettime(clock, &now);
    GuestTimespec result{(int64_t)now.tv_sec, (int64_t)now.tv_nsec + milliseconds * 1'000'000};
    result.sec += result.nsec / 1'000'000'000;
    result.nsec %= 1'000'000'000;
    return result;
}

// The entry points under test, resolved once per case. `registration` is separate because "is the
// NID bound at all" and "does the bound handler behave" are different questions, and the second
// CHECK in FamilyIsRegistered pins that the four clock NIDs resolve to the EXPECTED handlers rather
// than merely to something.
struct CondFamily {
    HleFn attr_init = nullptr, attr_destroy = nullptr, attr_setclock = nullptr,
          attr_getclock = nullptr, attr_setpshared = nullptr, attr_getpshared = nullptr,
          posix_setclock = nullptr, posix_getclock = nullptr, cond_init = nullptr,
          cond_destroy = nullptr, cond_signal = nullptr, cond_timedwait = nullptr,
          sce_cond_timedwait = nullptr, mutex_init = nullptr, mutex_destroy = nullptr,
          mutex_lock = nullptr, mutex_unlock = nullptr;
    bool complete() const {
        return attr_init && attr_destroy && attr_setclock && attr_getclock && attr_setpshared &&
               attr_getpshared && posix_setclock && posix_getclock && cond_init && cond_destroy &&
               cond_signal && cond_timedwait && sce_cond_timedwait && mutex_init && mutex_destroy &&
               mutex_lock && mutex_unlock;
    }
};
CondFamily cond_family() {
    register_builtin_hle();
    CondFamily f;
    f.attr_init = Hle::lookup(nid_hash("scePthreadCondattrInit"));
    f.attr_destroy = Hle::lookup(nid_hash("scePthreadCondattrDestroy"));
    f.attr_setclock = Hle::lookup(nid_hash("scePthreadCondattrSetclock"));
    f.attr_getclock = Hle::lookup(nid_hash("scePthreadCondattrGetclock"));
    f.attr_setpshared = Hle::lookup(nid_hash("scePthreadCondattrSetpshared"));
    f.attr_getpshared = Hle::lookup(nid_hash("scePthreadCondattrGetpshared"));
    f.posix_setclock = Hle::lookup(nid_hash("pthread_condattr_setclock"));
    f.posix_getclock = Hle::lookup(nid_hash("pthread_condattr_getclock"));
    f.cond_init = Hle::lookup(nid_hash("scePthreadCondInit"));
    f.cond_destroy = Hle::lookup(nid_hash("scePthreadCondDestroy"));
    f.cond_signal = Hle::lookup(nid_hash("scePthreadCondSignal"));
    f.cond_timedwait = Hle::lookup(nid_hash("pthread_cond_timedwait"));
    f.sce_cond_timedwait = Hle::lookup(nid_hash("scePthreadCondTimedwait"));
    f.mutex_init = Hle::lookup(nid_hash("scePthreadMutexInit"));
    f.mutex_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
    f.mutex_lock = Hle::lookup(nid_hash("scePthreadMutexLock"));
    f.mutex_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));
    return f;
}

// A guest condattr, created and left for the case to destroy.
struct GuestCondattr {
    HleFn init = nullptr, destroy = nullptr;
    void* handle = nullptr;
    bool create() {
        init = Hle::lookup(nid_hash("scePthreadCondattrInit"));
        destroy = Hle::lookup(nid_hash("scePthreadCondattrDestroy"));
        if (!init || !destroy) return false;
        return init(U(&handle), 0, 0, 0, 0, 0) == 0 && handle != nullptr;
    }
    ~GuestCondattr() {
        if (destroy && handle) destroy(U(&handle), 0, 0, 0, 0, 0);
    }
    // After a case has destroyed the attribute itself, so the destructor does not destroy it twice
    // (the handler frees the guest object; a second destroy is a double free, not a no-op).
    void forget() { handle = nullptr; }
};
}   // namespace

TEST(CondClock, ConditionClockAndMutexHandlersAreRegistered) {
    const CondFamily f = cond_family();
    EXPECT_TRUE(f.complete()) << "condition-clock and mutex HLE functions are registered";
    EXPECT_TRUE(Hle::lookup("c-bxj027czs") == f.attr_setclock &&
                Hle::lookup("6qM3kO5S3Oo") == f.attr_getclock &&
                Hle::lookup("EjllaAqAPZo") == f.posix_setclock &&
                Hle::lookup("cTDYxTUNPhM") == f.posix_getclock)
        << "Sony and libScePosix clock NIDs resolve to the expected handlers";
}

TEST(CondClock, AFreshAttributeDefaultsToSonyRealtime) {
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create()) << "condition attribute initializes";
    int32_t value = -1;
    EXPECT_EQ(f.attr_getclock(U(&attr.handle), U(&value), 0, 0, 0, 0), 0u);
    EXPECT_EQ(value, 0) << "fresh attribute defaults to Sony CLOCK_REALTIME(0)";
}

TEST(CondClock, SupportedSonyClocksRoundTrip) {
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    for (int32_t clock : {0, 1, 2, 4}) {
        int32_t value = -1;
        EXPECT_EQ(f.attr_setclock(U(&attr.handle), static_cast<uint64_t>(clock), 0, 0, 0, 0), 0u)
            << "setclock accepts Sony clock " << clock;
        EXPECT_EQ(f.attr_getclock(U(&attr.handle), U(&value), 0, 0, 0, 0), 0u);
        EXPECT_EQ(value, clock) << "supported Sony condition clock round-trips";
    }
}

TEST(CondClock, ARejectedClockReportsEncodedEinvalAndKeepsThePreviousOne) {
    // #2178: these are the SONY spellings, so a rejection reports the libkernel-encoded form. The
    // POSIX spellings registered on the same bodies keep the bare 22, and test_pthread_error_encoding
    // asserts that half -- a single-spelling assertion here cannot tell the two wirings apart.
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    ASSERT_EQ(f.attr_setclock(U(&attr.handle), 4, 0, 0, 0, 0), 0u);
    EXPECT_EQ(f.attr_setclock(U(&attr.handle), 3, 0, 0, 0, 0), 0x80020016ull)
        << "unsupported condition clock is rejected with encoded EINVAL (0x80020016)";
    int32_t value = -1;
    EXPECT_EQ(f.attr_getclock(U(&attr.handle), U(&value), 0, 0, 0, 0), 0u);
    EXPECT_EQ(value, 4) << "rejected setclock preserves the previous clock";
    EXPECT_EQ(f.attr_getclock(U(&attr.handle), 0, 0, 0, 0, 0), 0x80020016ull)
        << "getclock rejects a null output pointer with encoded EINVAL";
}

TEST(CondClock, ProcessPrivateIsTheOnlySupportedPsharedSetting) {
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    int32_t value = -1;
    EXPECT_EQ(f.attr_getpshared(U(&attr.handle), U(&value), 0, 0, 0, 0), 0u);
    EXPECT_EQ(value, 0) << "fresh attribute is process-private";
    EXPECT_EQ(f.attr_setpshared(U(&attr.handle), 0, 0, 0, 0, 0), 0u)
        << "process-private condition attribute is accepted";
    EXPECT_EQ(f.attr_setpshared(U(&attr.handle), 1, 0, 0, 0, 0), 0x80020016ull)
        << "unsupported process-shared condition attribute is rejected with encoded EINVAL";
}

TEST(CondClock, ANonRealtimeAbsoluteWaitHonoursTheCopiedClockAndReceivesTheSignal) {
    // CondInit must copy the attribute. Change the reusable attr back to realtime after init, then
    // use a deadline that is far in the future for every non-realtime clock but in the past for
    // realtime. A discarded/live-linked attr returns ETIMEDOUT; a copied attr waits for the signal.
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    for (int32_t clock : {1, 2, 4}) {
        void* cond = nullptr;
        void* mutex = nullptr;
        ASSERT_EQ(f.attr_setclock(U(&attr.handle), static_cast<uint64_t>(clock), 0, 0, 0, 0), 0u);
        ASSERT_EQ(f.cond_init(U(&cond), U(&attr.handle), 0, 0, 0, 0), 0u)
            << "condition initializes";
        ASSERT_NE(cond, nullptr);
        ASSERT_EQ(f.attr_setclock(U(&attr.handle), 0, 0, 0, 0, 0), 0u);
        ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u) << "mutex initializes for clock wait";
        ASSERT_NE(mutex, nullptr);
        ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u) << "mutex locks before clock wait";

        uint64_t worker_result = 0;
        std::thread signaler([&] {
            worker_result = f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0);
            if (worker_result == 0) {
                worker_result = f.cond_signal(U(&cond), 0, 0, 0, 0, 0);
                f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
            }
        });
        GuestTimespec far_selected_deadline{1'000'000'000ll, 0};
        const uint64_t wait_result =
            f.cond_timedwait(U(&cond), U(&mutex), U(&far_selected_deadline), 0, 0, 0);
        f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
        signaler.join();
        EXPECT_EQ(wait_result, 0u) << "non-realtime absolute wait (clock " << clock
                                   << ") honours the copied clock and receives the signal";
        EXPECT_EQ(worker_result, 0u) << "the signalling thread took and released the mutex";
        f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
        f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
    }
}

TEST(CondClock, AMonotonicDeadlineWaitsAndReturnsFreeBsdEtimedout) {
    // A MONOTONIC deadline must be interpreted against monotonic time, not rejected immediately as
    // an epoch-time deadline. It should actually wait and return FreeBSD ETIMEDOUT(60).
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.attr_setclock(U(&attr.handle), 4, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.cond_init(U(&cond), U(&attr.handle), 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u) << "mutex locks before monotonic wait";
    GuestTimespec monotonic_deadline = deadline_after(CLOCK_MONOTONIC, 25);
    const auto started = std::chrono::steady_clock::now();
    const uint64_t timeout_result =
        f.cond_timedwait(U(&cond), U(&mutex), U(&monotonic_deadline), 0, 0, 0);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
    EXPECT_EQ(timeout_result, 60u) << "monotonic wait returns FreeBSD ETIMEDOUT(60)";
    EXPECT_GE(elapsed_ms, 10) << "monotonic deadline waits at least part of the requested interval";
    EXPECT_LT(elapsed_ms, 1000) << "monotonic deadline does not run away";
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}

TEST(CondClock, AnInvalidAbsoluteTimespecIsRejectedWithEInval) {
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.attr_setclock(U(&attr.handle), 4, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.cond_init(U(&cond), U(&attr.handle), 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex locks before invalid-deadline check";
    GuestTimespec invalid_deadline{0, 1'000'000'000ll};
    EXPECT_EQ(f.cond_timedwait(U(&cond), U(&mutex), U(&invalid_deadline), 0, 0, 0), 22u)
        << "invalid absolute timespec is rejected with EINVAL(22)";
    f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}

TEST(CondClock, TheSonyRelativeWaitReturnsEncodedEtimedoutAndReacquiresTheMutex) {
    // The Sony API takes a relative microsecond count and returns an encoded SCE kernel error.
    // This contract intentionally differs from the POSIX entry point above, which takes an absolute
    // timespec and returns positive FreeBSD ETIMEDOUT(60).
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.cond_init(U(&cond), 0, 0, 0, 0, 0), 0u) << "Sony timed-wait condition initializes";
    ASSERT_NE(cond, nullptr);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u) << "Sony timed-wait mutex initializes";
    ASSERT_NE(mutex, nullptr);
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u) << "Sony timed-wait mutex locks";
    const auto started = std::chrono::steady_clock::now();
    const uint64_t timeout_result = f.sce_cond_timedwait(U(&cond), U(&mutex), 5'000, 0, 0, 0);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    EXPECT_EQ(timeout_result, 0x8002003cu)
        << "Sony relative wait returns SCE_KERNEL_ERROR_ETIMEDOUT";
    EXPECT_GE(elapsed_ms, 2) << "Sony relative wait honours its microsecond interval";
    EXPECT_LT(elapsed_ms, 1000) << "Sony relative wait does not run away";
    EXPECT_EQ(f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "Sony timed wait reacquires the mutex before returning";
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}

TEST(CondClock, AttributeDestroyClearsTheGuestHandle) {
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    ASSERT_EQ(f.attr_destroy(U(&attr.handle), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(attr.handle, nullptr) << "condition attribute destroy clears the guest handle";
    attr.forget();
}

#ifdef _WIN32
TEST(CondClock, AnInjectedPreUnlockFailureIsReturnedAndLeavesTheGuestStillOwningTheMutex) {
    // A Windows helper error before pthread_mutex_unlock leaves the host mutex owned. The HLE had
    // already cleared its ERRORCHECK owner map before entering the helper, so it must restore that
    // map from the reported unlock=false fact rather than keying restoration on rc==0/ETIMEDOUT.
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    GuestCondattr attr;
    ASSERT_TRUE(attr.create());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.attr_setclock(U(&attr.handle), 4, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.cond_init(U(&cond), U(&attr.handle), 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex locks before injected pre-unlock failure";
    win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest::BeforeUnlock, ENOMEM);
    GuestTimespec deadline = deadline_after(CLOCK_MONOTONIC, 25);
    EXPECT_EQ(f.cond_timedwait(U(&cond), U(&mutex), U(&deadline), 0, 0, 0),
              static_cast<uint64_t>(ENOMEM))
        << "injected pre-unlock wait failure is returned";
    EXPECT_TRUE(win_guest_mutex_owned_by_current_thread_for_test(mutex))
        << "pre-unlock failure preserves guest mutex ownership bookkeeping";
    EXPECT_EQ(f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex remains unlockable after pre-unlock wait failure";
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}

TEST(CondClock, ANonOwnerWaitReturnsEncodedEpermWithoutInventingOwnership) {
    // Calling a condition wait without owning its ERRORCHECK mutex makes the host unlock fail with
    // EPERM. The bookkeeping transaction must not publish this non-owner as the mutex owner.
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.cond_init(U(&cond), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u);
    EXPECT_FALSE(win_guest_mutex_owned_by_current_thread_for_test(mutex))
        << "unlocked mutex has no current-thread owner before invalid wait";
    // #2178: this entry point encoded its TIMEOUT (0x8002003c, asserted in the case above) and left
    // every other failure bare -- the same question answered two ways eleven lines apart. Both are
    // encoded now.
    EXPECT_EQ(f.sce_cond_timedwait(U(&cond), U(&mutex), 1'000, 0, 0, 0),
              prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EPerm))
        << "condition wait by a mutex non-owner returns encoded EPERM (0x80020001)";
    EXPECT_FALSE(win_guest_mutex_owned_by_current_thread_for_test(mutex))
        << "failed non-owner unlock does not invent guest mutex ownership";
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex remains acquirable after non-owner condition wait";
    EXPECT_EQ(f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex unlocks after non-owner wait recovery";
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}

TEST(CondClock, AnInjectedRelockFailureLeavesOwnershipReleasedAndRecoverable) {
    // Conversely, an error at the relock boundary follows a successful host unlock. Bookkeeping
    // must remain released so a later HLE lock can acquire the genuinely unowned mutex.
    const CondFamily f = cond_family();
    ASSERT_TRUE(f.complete());
    void* cond = nullptr;
    void* mutex = nullptr;
    ASSERT_EQ(f.cond_init(U(&cond), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_init(U(&mutex), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex locks before injected relock failure";
    win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest::BeforeRelock, EIO);
    EXPECT_EQ(f.sce_cond_timedwait(U(&cond), U(&mutex), 1'000, 0, 0, 0),
              prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EIo))
        << "injected relock failure is returned, encoded (0x80020005)";
    EXPECT_FALSE(win_guest_mutex_owned_by_current_thread_for_test(mutex))
        << "relock failure leaves guest mutex ownership released";
    ASSERT_EQ(f.mutex_lock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex can be acquired normally after relock failure";
    EXPECT_TRUE(win_guest_mutex_owned_by_current_thread_for_test(mutex))
        << "subsequent lock republishes guest mutex ownership";
    EXPECT_EQ(f.mutex_unlock(U(&mutex), 0, 0, 0, 0, 0), 0u)
        << "mutex unlocks after relock-failure recovery";
    f.cond_destroy(U(&cond), 0, 0, 0, 0, 0);
    f.mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
}
#endif   // _WIN32
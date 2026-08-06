// Condition attributes retain the PS5 clock identity and absolute waits use that selected clock.
// Before #386/F8, CondInit discarded the attribute, so a MONOTONIC/CPU-clock deadline was treated
// as CLOCK_REALTIME and normally appeared decades in the past.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/sce_errno.hpp"   // #2178: the Sony spellings report the libkernel encoding
#include "../src/hle/sync_futex.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <pthread.h>
#include <thread>

using namespace prosper;

static int failures;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++failures; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

struct GuestTimespec {
    int64_t sec;
    int64_t nsec;
};

static uint64_t U(const void* value) {
    return (uint64_t)(uintptr_t)value;
}

static GuestTimespec deadline_after(clockid_t clock, int64_t milliseconds) {
    timespec now{};
    clock_gettime(clock, &now);
    GuestTimespec result{(int64_t)now.tv_sec,
                         (int64_t)now.tv_nsec + milliseconds * 1'000'000};
    result.sec += result.nsec / 1'000'000'000;
    result.nsec %= 1'000'000'000;
    return result;
}

int main() {
    std::printf("== test_cond_clock ==\n");
    register_builtin_hle();

    HleFn attr_init = Hle::lookup(nid_hash("scePthreadCondattrInit"));
    HleFn attr_destroy = Hle::lookup(nid_hash("scePthreadCondattrDestroy"));
    HleFn attr_setclock = Hle::lookup(nid_hash("scePthreadCondattrSetclock"));
    HleFn attr_getclock = Hle::lookup(nid_hash("scePthreadCondattrGetclock"));
    HleFn attr_setpshared = Hle::lookup(nid_hash("scePthreadCondattrSetpshared"));
    HleFn attr_getpshared = Hle::lookup(nid_hash("scePthreadCondattrGetpshared"));
    HleFn posix_setclock = Hle::lookup(nid_hash("pthread_condattr_setclock"));
    HleFn posix_getclock = Hle::lookup(nid_hash("pthread_condattr_getclock"));
    HleFn cond_init = Hle::lookup(nid_hash("scePthreadCondInit"));
    HleFn cond_destroy = Hle::lookup(nid_hash("scePthreadCondDestroy"));
    HleFn cond_signal = Hle::lookup(nid_hash("scePthreadCondSignal"));
    HleFn cond_timedwait = Hle::lookup(nid_hash("pthread_cond_timedwait"));
    HleFn sce_cond_timedwait = Hle::lookup(nid_hash("scePthreadCondTimedwait"));
    HleFn mutex_init = Hle::lookup(nid_hash("scePthreadMutexInit"));
    HleFn mutex_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
    HleFn mutex_lock = Hle::lookup(nid_hash("scePthreadMutexLock"));
    HleFn mutex_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));

    CHECK(attr_init && attr_destroy && attr_setclock && attr_getclock &&
              attr_setpshared && attr_getpshared && posix_setclock && posix_getclock &&
              cond_init && cond_destroy && cond_signal && cond_timedwait && sce_cond_timedwait && mutex_init &&
              mutex_destroy && mutex_lock && mutex_unlock,
          "condition-clock and mutex HLE functions are registered");
    CHECK(Hle::lookup("c-bxj027czs") == attr_setclock &&
              Hle::lookup("6qM3kO5S3Oo") == attr_getclock &&
              Hle::lookup("EjllaAqAPZo") == posix_setclock &&
              Hle::lookup("cTDYxTUNPhM") == posix_getclock,
          "Sony and libScePosix clock NIDs resolve to the expected handlers");
    if (failures) return 1;

    void* attr = nullptr;
    CHECK(attr_init(U(&attr), 0, 0, 0, 0, 0) == 0 && attr,
          "condition attribute initializes");
    int32_t value = -1;
    CHECK(attr_getclock(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == 0,
          "fresh attribute defaults to Sony CLOCK_REALTIME(0)");
    for (int32_t clock : {0, 1, 2, 4}) {
        value = -1;
        CHECK(attr_setclock(U(&attr), (uint64_t)clock, 0, 0, 0, 0) == 0 &&
                  attr_getclock(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == clock,
              "supported Sony condition clock round-trips");
    }
    // #2178: these are the SONY spellings, so a rejection reports the libkernel-encoded form. The
    // POSIX spellings registered on the same bodies keep the bare 22, and test_pthread_error_encoding
    // asserts that half — a single-spelling assertion here cannot tell the two wirings apart.
    CHECK(attr_setclock(U(&attr), 3, 0, 0, 0, 0) == 0x80020016ull,
          "unsupported condition clock is rejected with encoded EINVAL (0x80020016)");
    value = -1;
    CHECK(attr_getclock(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == 4,
          "rejected setclock preserves the previous clock");
    CHECK(attr_getclock(U(&attr), 0, 0, 0, 0, 0) == 0x80020016ull,
          "getclock rejects a null output pointer with encoded EINVAL");

    value = -1;
    CHECK(attr_getpshared(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == 0,
          "fresh attribute is process-private");
    CHECK(attr_setpshared(U(&attr), 0, 0, 0, 0, 0) == 0,
          "process-private condition attribute is accepted");
    CHECK(attr_setpshared(U(&attr), 1, 0, 0, 0, 0) == 0x80020016ull,
          "unsupported process-shared condition attribute is rejected with encoded EINVAL");

    // CondInit must copy the attribute. Change the reusable attr back to realtime after init, then
    // use a deadline that is far in the future for every non-realtime clock but in the past for
    // realtime. A discarded/live-linked attr returns ETIMEDOUT; a copied attr waits for the signal.
    for (int32_t clock : {1, 2, 4}) {
        void* cond = nullptr;
        void* mutex = nullptr;
        attr_setclock(U(&attr), (uint64_t)clock, 0, 0, 0, 0);
        CHECK(cond_init(U(&cond), U(&attr), 0, 0, 0, 0) == 0 && cond,
              "condition initializes with selected clock");
        attr_setclock(U(&attr), 0, 0, 0, 0, 0);
        CHECK(mutex_init(U(&mutex), 0, 0, 0, 0, 0) == 0 && mutex,
              "mutex initializes for clock wait");
        CHECK(mutex_lock(U(&mutex), 0, 0, 0, 0, 0) == 0,
              "mutex locks before clock wait");

        uint64_t worker_result = 0;
        std::thread signaler([&] {
            worker_result = mutex_lock(U(&mutex), 0, 0, 0, 0, 0);
            if (worker_result == 0) {
                worker_result = cond_signal(U(&cond), 0, 0, 0, 0, 0);
                mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
            }
        });
        GuestTimespec far_selected_deadline{1'000'000'000ll, 0};
        const uint64_t wait_result = cond_timedwait(U(&cond), U(&mutex),
                                                    U(&far_selected_deadline), 0, 0, 0);
        mutex_unlock(U(&mutex), 0, 0, 0, 0, 0);
        signaler.join();
        CHECK(wait_result == 0 && worker_result == 0,
              "non-realtime absolute wait honors copied clock and receives signal");
        cond_destroy(U(&cond), 0, 0, 0, 0, 0);
        mutex_destroy(U(&mutex), 0, 0, 0, 0, 0);
    }

    // A MONOTONIC deadline must be interpreted against monotonic time, not rejected immediately as
    // an epoch-time deadline. It should actually wait and return FreeBSD ETIMEDOUT(60).
    void* monotonic_cond = nullptr;
    void* monotonic_mutex = nullptr;
    attr_setclock(U(&attr), 4, 0, 0, 0, 0);
    cond_init(U(&monotonic_cond), U(&attr), 0, 0, 0, 0);
    mutex_init(U(&monotonic_mutex), 0, 0, 0, 0, 0);
    CHECK(mutex_lock(U(&monotonic_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex locks before monotonic wait");
    GuestTimespec monotonic_deadline = deadline_after(CLOCK_MONOTONIC, 25);
    const auto started = std::chrono::steady_clock::now();
    const uint64_t timeout_result = cond_timedwait(U(&monotonic_cond), U(&monotonic_mutex),
                                                   U(&monotonic_deadline), 0, 0, 0);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    mutex_unlock(U(&monotonic_mutex), 0, 0, 0, 0, 0);
    CHECK(timeout_result == 60, "monotonic wait returns FreeBSD ETIMEDOUT(60)");
    CHECK(elapsed_ms >= 10 && elapsed_ms < 1000,
          "monotonic deadline waits for the requested interval");

    CHECK(mutex_lock(U(&monotonic_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex locks before invalid-deadline check");
    GuestTimespec invalid_deadline{0, 1'000'000'000ll};
    CHECK(cond_timedwait(U(&monotonic_cond), U(&monotonic_mutex), U(&invalid_deadline),
                         0, 0, 0) == 22,
          "invalid absolute timespec is rejected with EINVAL(22)");
    mutex_unlock(U(&monotonic_mutex), 0, 0, 0, 0, 0);
#ifdef _WIN32
    // A Windows helper error before pthread_mutex_unlock leaves the host mutex owned. The HLE had
    // already cleared its ERRORCHECK owner map before entering the helper, so it must restore that
    // map from the reported unlock=false fact rather than keying restoration on rc==0/ETIMEDOUT.
    CHECK(mutex_lock(U(&monotonic_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex locks before injected pre-unlock failure");
    win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest::BeforeUnlock, ENOMEM);
    monotonic_deadline = deadline_after(CLOCK_MONOTONIC, 25);
    CHECK(cond_timedwait(U(&monotonic_cond), U(&monotonic_mutex), U(&monotonic_deadline),
                         0, 0, 0) == ENOMEM,
          "injected pre-unlock wait failure is returned");
    CHECK(win_guest_mutex_owned_by_current_thread_for_test(monotonic_mutex),
          "pre-unlock failure preserves guest mutex ownership bookkeeping");
    CHECK(mutex_unlock(U(&monotonic_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex remains unlockable after pre-unlock wait failure");
#endif
    cond_destroy(U(&monotonic_cond), 0, 0, 0, 0, 0);
    mutex_destroy(U(&monotonic_mutex), 0, 0, 0, 0, 0);

    // The Sony API takes a relative microsecond count and returns an encoded SCE kernel error.
    // This contract intentionally differs from the POSIX entry point above, which takes an absolute
    // timespec and returns positive FreeBSD ETIMEDOUT(60).
    void* sce_cond = nullptr;
    void* sce_mutex = nullptr;
    CHECK(cond_init(U(&sce_cond), 0, 0, 0, 0, 0) == 0 && sce_cond,
          "Sony timed-wait condition initializes");
    CHECK(mutex_init(U(&sce_mutex), 0, 0, 0, 0, 0) == 0 && sce_mutex,
          "Sony timed-wait mutex initializes");
    CHECK(mutex_lock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "Sony timed-wait mutex locks");
    const auto sce_started = std::chrono::steady_clock::now();
    const uint64_t sce_timeout_result = sce_cond_timedwait(
        U(&sce_cond), U(&sce_mutex), 5'000, 0, 0, 0);
    const auto sce_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - sce_started).count();
    CHECK(sce_timeout_result == 0x8002003cu,
          "Sony relative wait returns SCE_KERNEL_ERROR_ETIMEDOUT");
    CHECK(sce_elapsed_ms >= 2 && sce_elapsed_ms < 1000,
          "Sony relative wait honors its microsecond interval");
    CHECK(mutex_unlock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "Sony timed wait reacquires the mutex before returning");
#ifdef _WIN32
    // Calling a condition wait without owning its ERRORCHECK mutex makes the host unlock fail with
    // EPERM. The bookkeeping transaction must not publish this non-owner as the mutex owner.
    CHECK(!win_guest_mutex_owned_by_current_thread_for_test(sce_mutex),
          "unlocked mutex has no current-thread owner before invalid wait");
    // #2178: this entry point encoded its TIMEOUT (0x8002003c, asserted above) and left every other
    // failure bare — the same question answered two ways eleven lines apart. Both are encoded now.
    CHECK(sce_cond_timedwait(U(&sce_cond), U(&sce_mutex), 1'000, 0, 0, 0)
              == prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EPerm),
          "condition wait by a mutex non-owner returns encoded EPERM (0x80020001)");
    CHECK(!win_guest_mutex_owned_by_current_thread_for_test(sce_mutex),
          "failed non-owner unlock does not invent guest mutex ownership");
    CHECK(mutex_lock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex remains acquirable after non-owner condition wait");
    CHECK(mutex_unlock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex unlocks after non-owner wait recovery");

    // Conversely, an error at the relock boundary follows a successful host unlock. Bookkeeping
    // must remain released so a later HLE lock can acquire the genuinely unowned mutex.
    CHECK(mutex_lock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex locks before injected relock failure");
    win_set_cond_wait_failure_for_test(CondWaitFailurePointForTest::BeforeRelock, EIO);
    CHECK(sce_cond_timedwait(U(&sce_cond), U(&sce_mutex), 1'000, 0, 0, 0)
              == prosper::hle::sce_kernel_error(prosper::hle::FreeBsdErrno::EIo),
          "injected relock failure is returned, encoded (0x80020005)");
    CHECK(!win_guest_mutex_owned_by_current_thread_for_test(sce_mutex),
          "relock failure leaves guest mutex ownership released");
    CHECK(mutex_lock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex can be acquired normally after relock failure");
    CHECK(win_guest_mutex_owned_by_current_thread_for_test(sce_mutex),
          "subsequent lock republishes guest mutex ownership");
    CHECK(mutex_unlock(U(&sce_mutex), 0, 0, 0, 0, 0) == 0,
          "mutex unlocks after relock-failure recovery");
#endif
    cond_destroy(U(&sce_cond), 0, 0, 0, 0, 0);
    mutex_destroy(U(&sce_mutex), 0, 0, 0, 0, 0);

    attr_destroy(U(&attr), 0, 0, 0, 0, 0);
    CHECK(attr == nullptr, "condition attribute destroy clears the guest handle");

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}

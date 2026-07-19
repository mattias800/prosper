// Condition attributes retain the PS5 clock identity and absolute waits use that selected clock.
// Before #386/F8, CondInit discarded the attribute, so a MONOTONIC/CPU-clock deadline was treated
// as CLOCK_REALTIME and normally appeared decades in the past.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"

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
    HleFn mutex_init = Hle::lookup(nid_hash("scePthreadMutexInit"));
    HleFn mutex_destroy = Hle::lookup(nid_hash("scePthreadMutexDestroy"));
    HleFn mutex_lock = Hle::lookup(nid_hash("scePthreadMutexLock"));
    HleFn mutex_unlock = Hle::lookup(nid_hash("scePthreadMutexUnlock"));

    CHECK(attr_init && attr_destroy && attr_setclock && attr_getclock &&
              attr_setpshared && attr_getpshared && posix_setclock && posix_getclock &&
              cond_init && cond_destroy && cond_signal && cond_timedwait && mutex_init &&
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
    CHECK(attr_setclock(U(&attr), 3, 0, 0, 0, 0) == 22,
          "unsupported condition clock is rejected with EINVAL(22)");
    value = -1;
    CHECK(attr_getclock(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == 4,
          "rejected setclock preserves the previous clock");
    CHECK(attr_getclock(U(&attr), 0, 0, 0, 0, 0) == 22,
          "getclock rejects a null output pointer");

    value = -1;
    CHECK(attr_getpshared(U(&attr), U(&value), 0, 0, 0, 0) == 0 && value == 0,
          "fresh attribute is process-private");
    CHECK(attr_setpshared(U(&attr), 0, 0, 0, 0, 0) == 0,
          "process-private condition attribute is accepted");
    CHECK(attr_setpshared(U(&attr), 1, 0, 0, 0, 0) == 22,
          "unsupported process-shared condition attribute is rejected");

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
    mutex_lock(U(&monotonic_mutex), 0, 0, 0, 0, 0);
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

    mutex_lock(U(&monotonic_mutex), 0, 0, 0, 0, 0);
    GuestTimespec invalid_deadline{0, 1'000'000'000ll};
    CHECK(cond_timedwait(U(&monotonic_cond), U(&monotonic_mutex), U(&invalid_deadline),
                         0, 0, 0) == 22,
          "invalid absolute timespec is rejected with EINVAL(22)");
    mutex_unlock(U(&monotonic_mutex), 0, 0, 0, 0, 0);
    cond_destroy(U(&monotonic_cond), 0, 0, 0, 0, 0);
    mutex_destroy(U(&monotonic_mutex), 0, 0, 0, 0, 0);

    attr_destroy(U(&attr), 0, 0, 0, 0, 0);
    CHECK(attr == nullptr, "condition attribute destroy clears the guest handle");

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}

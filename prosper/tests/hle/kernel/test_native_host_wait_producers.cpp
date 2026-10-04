// #4330: final built-in HLE handlers must produce real native-wait provenance, then withdraw it.
// The observation is not a wakeup primitive. Normal user events, deletion and once completion
// release these callers even when the diagnostic hook is removed by a production-only control.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include "diagnostics/native_host_wait.hpp"
#include "fixtures/test_scratch.h"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>
#include <windows.h>
#include <pthread.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iterator>
#include <thread>

using namespace prosper;
using namespace prosper::diagnostics;
namespace {
struct KEvent {
    int64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    uint64_t udata;
};
static_assert(sizeof(KEvent) == 32);
struct WorkerRegistration {
    const uint32_t native_id = GetCurrentThreadId();
    const uint64_t pthread_id = static_cast<uint64_t>(pthread_self());
    WorkerRegistration() {
        ULONG_PTR low = 0, high = 0;
        GetCurrentThreadStackLimits(&low, &high);
        trace_guest_thread_lifecycle(true, pthread_id, native_id, (void*)low, high - low);
    }
    ~WorkerRegistration() {
        trace_guest_thread_lifecycle(false, pthread_id, native_id, nullptr, 0);
    }
};
template <class Predicate>
bool await(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
bool observe(uint32_t native_id, NativeHostWaitSite site, uintptr_t object,
             NativeHostWaitRecord& observed) {
    if (!native_id) return false;
    std::array<NativeHostWaitRecord, 4> records{};
    const auto snapshot = native_host_wait_registry().snapshot(native_id, records);
    for (size_t i = 0; i < std::min(snapshot.found, records.size()); ++i)
        if (records[i].site == site && records[i].object == object) {
            observed = records[i];
            return true;
        }
    return false;
}
std::atomic<bool> once_entered{false}, once_release{false};
std::atomic<unsigned> once_calls{0}, independent_calls{0};
void held_once_init() {
    once_calls.fetch_add(1);
    once_entered.store(true, std::memory_order_release);
    while (!once_release.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
void independent_once_init() {
    independent_calls.fetch_add(1);
}

class NativeHostWaitProducers : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(native_host_wait_observation_enabled())
            << "CTest must arm the process-start lever";
        register_builtin_hle();
        create = Hle::lookup(nid_hash("sceKernelCreateEqueue"));
        destroy = Hle::lookup(nid_hash("sceKernelDeleteEqueue"));
        add = Hle::lookup(nid_hash("sceKernelAddUserEvent"));
        trigger = Hle::lookup(nid_hash("sceKernelTriggerUserEvent"));
        wait = Hle::lookup(nid_hash("sceKernelWaitEqueue"));
        once = Hle::lookup(nid_hash("scePthreadOnce"));
        ASSERT_TRUE(create && destroy && add && trigger && wait && once);
        ASSERT_EQ(create((uint64_t)&queue, 0, 0, 0, 0, 0), 0u);
        ASSERT_NE(queue, 0u);
        ASSERT_EQ(add(queue, 91, 0x1234, 0, 0, 0), 0u);
    }
    void TearDown() override {
        if (queue && destroy) destroy(queue, 0, 0, 0, 0, 0);
    }
    HleFn create = nullptr, destroy = nullptr, add = nullptr, trigger = nullptr, wait = nullptr,
          once = nullptr;
    uint64_t queue = 0;
};
}   // namespace

TEST_F(NativeHostWaitProducers, InfiniteEqueueIsObservedAndActualDumpNamesItsSource) {
    std::atomic<uint32_t> native_id{0};
    std::atomic<uint64_t> pthread_id{0};
    std::atomic<bool> returned{false};
    uint64_t result = UINT64_MAX;
    int32_t out = -1;
    KEvent event{};
    std::thread worker([&] {
        WorkerRegistration registration;
        pthread_id.store(registration.pthread_id, std::memory_order_release);
        native_id.store(registration.native_id, std::memory_order_release);
        result = wait(queue, (uint64_t)&event, 1, (uint64_t)&out, 0, 0);
        returned.store(true, std::memory_order_release);
    });
    NativeHostWaitRecord record{};
    const bool observed = await([&] {
        return observe(native_id.load(std::memory_order_acquire), NativeHostWaitSite::Equeue, queue,
                       record);
    });
    EXPECT_TRUE(observed) << "removing the actual equeue producer hook must fail, not time out";
    EXPECT_FALSE(returned.load())
        << "the observer must not invent an event or cap the infinite wait";
    EXPECT_EQ(record.mode, NativeHostWaitMode::Infinite);
    EXPECT_EQ(record.timeout_us, 0u);
    EXPECT_GT(record.entered_us, 0u);
    const std::string path = prosper_test::test_scratch_file("native-host-wait.log");
    if (observed) dump_guest_thread_trace(path.c_str(), pthread_id.load(std::memory_order_acquire));
    // Always release the real handler, including after a missing-observation assertion.
    EXPECT_EQ(trigger(queue, 91, 0, 0, 0, 0), 0u);
    worker.join();
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(out, 1);
    EXPECT_EQ(event.ident, 91);
    EXPECT_EQ(event.filter, -11);
    EXPECT_EQ(event.udata, 0x1234u);
    EXPECT_EQ(native_host_wait_registry().snapshot(native_id.load(), {}).found, 0u);
    std::ifstream input(path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    EXPECT_NE(text.find("site=sceKernelWaitEqueue"), std::string::npos)
        << "the actual sampler/writer route must expose the source, not only the registry API";
    EXPECT_NE(text.find("scope=copied-window-only raw-candidates-not-cfi"), std::string::npos);
    EXPECT_NE(text.find("absent=unobserved-not-no-wait"), std::string::npos);
}

TEST_F(NativeHostWaitProducers, TimedEqueueRetainsTheActualRequestedTimeoutAndWithdraws) {
    std::atomic<uint32_t> native_id{0};
    const uint32_t timeout_us = 2000000;
    uint64_t result = UINT64_MAX;
    int32_t out = -1;
    KEvent event{};
    std::thread worker([&] {
        WorkerRegistration registration;
        native_id.store(registration.native_id, std::memory_order_release);
        result = wait(queue, (uint64_t)&event, 1, (uint64_t)&out, (uint64_t)&timeout_us, 0);
    });
    NativeHostWaitRecord record{};
    const bool observed =
        await([&] { return observe(native_id.load(), NativeHostWaitSite::Equeue, queue, record); });
    EXPECT_TRUE(observed);
    EXPECT_EQ(record.mode, NativeHostWaitMode::RelativeMicroseconds);
    EXPECT_EQ(record.timeout_us, 2000000u) << "no diagnostic timeout is substituted";
    EXPECT_EQ(trigger(queue, 91, 0, 0, 0, 0), 0u);
    worker.join();
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(out, 1);
    EXPECT_EQ(native_host_wait_registry().snapshot(native_id.load(), {}).found, 0u);
    // A real empty timed wait retains the pre-existing timeout error and clears the output.
    uint32_t short_timeout = 1000;
    out = -1;
    EXPECT_EQ(wait(queue, (uint64_t)&event, 1, (uint64_t)&out, (uint64_t)&short_timeout, 0),
              0x8002003Cull);
    EXPECT_EQ(out, 0);
    EXPECT_EQ(native_host_wait_registry().snapshot(GetCurrentThreadId(), {}).found, 0u);
}

TEST_F(NativeHostWaitProducers, EqueueDeletionStillCancelsWithoutDiagnosticWakeAuthority) {
    std::atomic<uint32_t> native_id{0};
    uint64_t result = UINT64_MAX;
    int32_t out = -1;
    KEvent event{};
    const uint64_t queue_handle = queue;
    std::thread worker([&] {
        WorkerRegistration registration;
        native_id.store(registration.native_id, std::memory_order_release);
        result = wait(queue_handle, (uint64_t)&event, 1, (uint64_t)&out, 0, 0);
    });
    NativeHostWaitRecord record{};
    EXPECT_TRUE(await([&] {
        return observe(native_id.load(), NativeHostWaitSite::Equeue, queue_handle, record);
    }));
    EXPECT_EQ(destroy(queue, 0, 0, 0, 0, 0), 0u);
    queue = 0;
    worker.join();
    EXPECT_EQ(result, 0x80020009ull);
    EXPECT_EQ(out, 0);
    EXPECT_EQ(native_host_wait_registry().snapshot(native_id.load(), {}).found, 0u);
}

TEST_F(NativeHostWaitProducers, ReadyAndInvalidEqueuesDoNotClaimAWaitWasEntered) {
    const auto before = native_host_wait_registry().snapshot(GetCurrentThreadId(), {});
    EXPECT_EQ(trigger(queue, 91, 0, 0, 0, 0), 0u);
    KEvent event{};
    int32_t out = -1;
    EXPECT_EQ(wait(queue, (uint64_t)&event, 1, (uint64_t)&out, 0, 0), 0u);
    EXPECT_EQ(out, 1);
    EXPECT_EQ(event.udata, 0x1234u);
    out = 17;
    EXPECT_EQ(wait(queue, 0, 1, (uint64_t)&out, 0, 0), 0x8002000eull);
    EXPECT_EQ(out, 17);
    EXPECT_EQ(wait(queue, (uint64_t)&event, 0, (uint64_t)&out, 0, 0), 0x80020016ull);
    EXPECT_EQ(out, 0);
    out = 17;
    EXPECT_EQ(wait(UINT64_MAX, (uint64_t)&event, 1, (uint64_t)&out, 0, 0), 0x80020009ull);
    EXPECT_EQ(out, 0);
    const auto after = native_host_wait_registry().snapshot(GetCurrentThreadId(), {});
    EXPECT_EQ(after.entered_run, before.entered_run) << "even a transient false scope is forbidden";
    EXPECT_EQ(after.found, 0u);
}

TEST_F(NativeHostWaitProducers, ContendedOnceNamesOnlyTheWaitingControlAndPreservesCompletion) {
    std::atomic<int> control{0}, independent{0};
    once_entered.store(false);
    once_release.store(false);
    once_calls.store(0);
    independent_calls.store(0);
    std::atomic<uint32_t> owner_id{0}, waiter_id{0};
    uint64_t first = UINT64_MAX, second = UINT64_MAX;
    std::thread owner([&] {
        WorkerRegistration registration;
        owner_id.store(registration.native_id, std::memory_order_release);
        first = once((uint64_t)&control, (uint64_t)&held_once_init, 0, 0, 0, 0);
    });
    EXPECT_TRUE(await([&] { return once_entered.load(std::memory_order_acquire); }));
    std::thread waiter([&] {
        WorkerRegistration registration;
        waiter_id.store(registration.native_id, std::memory_order_release);
        second = once((uint64_t)&control, (uint64_t)&held_once_init, 0, 0, 0, 0);
    });
    NativeHostWaitRecord record{};
    EXPECT_TRUE(await([&] {
        return observe(waiter_id.load(), NativeHostWaitSite::PthreadOnce, (uintptr_t)&control,
                       record);
    })) << "removing the actual once producer hook must fail while the original once still "
           "completes";
    EXPECT_EQ(record.mode, NativeHostWaitMode::Infinite);
    EXPECT_EQ(record.timeout_us, 0u);
    EXPECT_EQ(native_host_wait_registry().snapshot(owner_id.load(), {}).found, 0u)
        << "running the initializer is not the contended wait site";
    EXPECT_EQ(once((uint64_t)&independent, (uint64_t)&independent_once_init, 0, 0, 0, 0), 0u);
    EXPECT_EQ(independent_calls.load(), 1u)
        << "diagnostics must not serialize independent controls";
    once_release.store(true, std::memory_order_release);
    owner.join();
    waiter.join();
    EXPECT_EQ(first, 0u);
    EXPECT_EQ(second, 0u);
    EXPECT_EQ(control.load(), 1);
    EXPECT_EQ(once_calls.load(), 1u);
    EXPECT_EQ(native_host_wait_registry().snapshot(waiter_id.load(), {}).found, 0u);
    const auto before = native_host_wait_registry().snapshot(GetCurrentThreadId(), {});
    EXPECT_EQ(once((uint64_t)&control, (uint64_t)&held_once_init, 0, 0, 0, 0), 0u);
    EXPECT_EQ(native_host_wait_registry().snapshot(GetCurrentThreadId(), {}).entered_run,
              before.entered_run);
    EXPECT_EQ(once(0, (uint64_t)&held_once_init, 0, 0, 0, 0), 0x80020016ull);
}

// #4330: actual observer publication, withdrawal, bounded loss and raw-scan/report contracts.
// Registry controls do NOT certify a HLE producer; separate registered-handler tests do that.
#include "diagnostics/native_host_wait.hpp"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

using namespace prosper::diagnostics;
namespace {
bool executable(uintptr_t address) {
    return address != 0x10007;
}
std::string describe(uint64_t address) {
    return address == 0x10008 ? "ntdll+8" : "prosper+" + std::to_string(address);
}
template <class Predicate>
bool await(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        if (predicate()) return true;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
}   // namespace

TEST(NativeHostWaitRegistry, NestedScopesAndStaleTokensPreserveIdentity) {
    NativeHostWaitRegistry registry;
    const auto outer = registry.enter(
        {31, NativeHostWaitSite::Equeue, 0xa1, NativeHostWaitMode::RelativeMicroseconds, 991, 101});
    const auto inner = registry.enter(
        {31, NativeHostWaitSite::PthreadOnce, 0xb2, NativeHostWaitMode::Infinite, 0, 102});
    std::array<NativeHostWaitRecord, 2> records{};
    EXPECT_EQ(registry.snapshot(31, records).found, 2u);
    EXPECT_EQ(records[0].object, 0xa1u);
    EXPECT_EQ(records[0].timeout_us, 991u);
    EXPECT_EQ(records[1].site, NativeHostWaitSite::PthreadOnce);
    registry.leave(inner);
    const auto recycled = registry.enter(
        {31, NativeHostWaitSite::PthreadOnce, 0xc3, NativeHostWaitMode::Infinite, 0, 103});
    EXPECT_NE(recycled.generation, inner.generation);
    registry.leave(inner);   // stale destructor must not remove the replacement or the outer scope
    EXPECT_EQ(registry.snapshot(31, records).found, 2u);
    EXPECT_EQ(records[1].object, 0xc3u);
    const auto other = registry.enter({80, NativeHostWaitSite::Equeue, 0xd4});
    registry.retire_thread(31);
    EXPECT_EQ(registry.snapshot(31, records).found, 0u);
    EXPECT_EQ(registry.snapshot(80, records).found, 1u);
    registry.leave(outer);
    registry.leave(recycled);
    registry.leave(other);
}

TEST(NativeHostWaitRegistry, RealCapacityAndOutputOmissionsAreCounted) {
    NativeHostWaitRegistry registry;
    std::array<NativeHostWaitToken, NativeHostWaitRegistry::capacity> tokens{};
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = registry.enter({19, NativeHostWaitSite::Equeue, i + 1});
        ASSERT_NE(tokens[i].generation, 0u) << "the real bounded registry must fill, not a model";
    }
    const auto overflow = registry.enter({77, NativeHostWaitSite::PthreadOnce, 9});
    EXPECT_EQ(overflow.generation, 0u);
    std::array<NativeHostWaitRecord, 2> records{};
    const auto snapshot = registry.snapshot(19, records);
    EXPECT_EQ(snapshot.found, 512u);
    EXPECT_EQ(snapshot.entered_run, 512u);
    EXPECT_EQ(snapshot.dropped_run, 1u);
    EXPECT_EQ(records[0].object, 1u);
    EXPECT_EQ(records[1].object, 2u);
    EXPECT_EQ(registry.snapshot(77, records).found, 0u) << "global loss is not evidence of no wait";
    for (const auto token : tokens) registry.leave(token);
    const auto fresh = registry.enter({77, NativeHostWaitSite::Equeue, 91});
    EXPECT_NE(fresh.generation, 0u);
    EXPECT_EQ(registry.snapshot(77, records).found, 1u);
    registry.leave(fresh);
}

TEST(NativeHostWaitRegistry, PausedActualPublisherReportsUnattributedIncompleteSlot) {
    NativeHostWaitRegistry registry;
    struct Pause {
        std::atomic<bool> reserved{false}, release{false};
    } pause;
    NativeHostWaitToken token;
    std::thread writer([&] {
        token = registry.enter(
            {41, NativeHostWaitSite::Equeue, 0xa11},
            +[](void* opaque) noexcept {
                auto& p = *static_cast<Pause*>(opaque);
                p.reserved.store(true, std::memory_order_release);
                while (!p.release.load(std::memory_order_acquire)) std::this_thread::yield();
            },
            &pause);
    });
    const bool reserved = await([&] { return pause.reserved.load(std::memory_order_acquire); });
    EXPECT_TRUE(reserved);
    std::array<NativeHostWaitRecord, 1> records{};
    const auto incomplete = registry.snapshot(41, records);
    EXPECT_EQ(incomplete.found, 0u);
    EXPECT_EQ(incomplete.unstable_slots, 1u)
        << "the real publication must not read as a clean zero";
    pause.release.store(true, std::memory_order_release);
    writer.join();
    EXPECT_EQ(registry.snapshot(41, records).found, 1u);
    EXPECT_EQ(records[0].object, 0xa11u);
    registry.leave(token);
}

TEST(NativeHostWaitRegistry, ConcurrentReuseNeverPublishesMixedPayloads) {
    NativeHostWaitRegistry registry;
    std::atomic<unsigned> ready{0}, done{0}, lost{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    for (uint32_t tid = 1; tid <= 4; ++tid)
        workers.emplace_back([&, tid] {
            const NativeHostWaitRecord record{
                tid,           NativeHostWaitSite::Equeue,
                0x1000u + tid, NativeHostWaitMode::RelativeMicroseconds,
                tid * 7u,      tid * 11u};
            auto token = registry.enter(record);
            ready.fetch_add(1, std::memory_order_release);
            while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
            registry.leave(token);
            for (unsigned i = 0; i < 4000; ++i) {
                token = registry.enter(record);
                if (!token.generation) lost.fetch_add(1);
                registry.leave(token);
            }
            done.fetch_add(1, std::memory_order_release);
        });
    EXPECT_TRUE(await([&] { return ready.load(std::memory_order_acquire) == 4; }));
    std::array<NativeHostWaitRecord, 1> records{};
    for (uint32_t tid = 1; tid <= 4; ++tid) EXPECT_EQ(registry.snapshot(tid, records).found, 1u);
    release.store(true, std::memory_order_release);
    do {
        for (uint32_t tid = 1; tid <= 4; ++tid) {
            const auto sample = registry.snapshot(tid, records);
            EXPECT_LE(sample.found, 1u);
            if (!sample.found) continue;
            EXPECT_EQ(records[0].native_id, tid);
            EXPECT_EQ(records[0].site, NativeHostWaitSite::Equeue);
            EXPECT_EQ(records[0].object, 0x1000u + tid);
            EXPECT_EQ(records[0].timeout_us, tid * 7u);
            EXPECT_EQ(records[0].entered_us, tid * 11u);
        }
    } while (done.load(std::memory_order_acquire) != 4);
    for (auto& worker : workers) worker.join();
    EXPECT_EQ(lost.load(), 0u);
    EXPECT_EQ(registry.snapshot(1, records).found, 0u);
    EXPECT_EQ(registry.snapshot(1, records).entered_run, 16004u);
}

TEST(NativeHostWaitRegistry, ActualRecyclingBetweenCopyAndValidationRejectsTheOldIncarnation) {
    NativeHostWaitRegistry registry;
    const auto old = registry.enter(
        {41, NativeHostWaitSite::Equeue, 0xa11, NativeHostWaitMode::Infinite, 0, 91});
    ASSERT_NE(old.generation, 0u);
    struct Recycle {
        NativeHostWaitToken replacement;
        std::atomic<bool> copied{false}, changed{false}, missed{false};
    } recycle;
    std::thread writer([&] {
        if (await([&] { return recycle.copied.load(std::memory_order_acquire); })) {
            registry.leave(old);
            recycle.replacement =
                registry.enter({42, NativeHostWaitSite::PthreadOnce, 0xb22,
                                NativeHostWaitMode::RelativeMicroseconds, 1234, 92});
        } else {
            recycle.missed.store(true);
        }
        recycle.changed.store(true, std::memory_order_release);
    });
    std::array<NativeHostWaitRecord, 1> output{{{99, NativeHostWaitSite::Equeue, 0xc33}}};
    const auto sampled = registry.snapshot(
        41, output,
        +[](void* opaque) noexcept {
            auto& state = *static_cast<Recycle*>(opaque);
            state.copied.store(true, std::memory_order_release);
            if (!await([&] { return state.changed.load(std::memory_order_acquire); }))
                state.missed.store(true);
        },
        &recycle);
    writer.join();
    ASSERT_FALSE(recycle.missed.load()) << "the bounded real interleaving must have occurred";
    ASSERT_EQ(recycle.replacement.slot, old.slot) << "the actual slot must have been recycled";
    ASSERT_NE(recycle.replacement.generation, old.generation);
    EXPECT_EQ(sampled.found, 0u) << "copied old payload must not survive a changed incarnation";
    EXPECT_EQ(sampled.unstable_slots, 1u);
    EXPECT_EQ(output[0].native_id, 99u) << "rejected payload must not write the output";
    EXPECT_EQ(output[0].object, 0xc33u);
    registry.leave(old);
    EXPECT_EQ(registry.snapshot(42, output).found, 1u);
    EXPECT_EQ(output[0].site, NativeHostWaitSite::PthreadOnce);
    EXPECT_EQ(output[0].object, 0xb22u);
    EXPECT_EQ(output[0].mode, NativeHostWaitMode::RelativeMicroseconds);
    EXPECT_EQ(output[0].timeout_us, 1234u);
    EXPECT_EQ(output[0].entered_us, 92u);
    registry.leave(recycle.replacement);
    EXPECT_EQ(registry.snapshot(42, output).found, 0u);
    // This real interleaving kills missing final validation, NOT a relaxed-payload mutation on
    // an x64 host. The cross-object C++ ordering argument is a separate source-review obligation.
}

TEST(NativeHostWaitReport, RawCandidatePrefixAndCopiedWindowAreNotABacktrace) {
    const std::array<uint64_t, 11> words{0,       0x10001, 0x10007, 0x10008, 0x10001, 0x10002,
                                         0x10003, 0x10004, 0x10005, 0x10006, 0x10009};
    const auto limited = scan_host_stack_candidates(words, executable, describe);
    EXPECT_EQ(limited.retained, 6u);
    EXPECT_EQ(limited.examined_words, 10u);
    EXPECT_TRUE(limited.prefix_limited) << "a seventh in-window candidate can be omitted";
    EXPECT_EQ(std::string(limited.text.data()).find("65545"), std::string::npos);
    const auto entire =
        scan_host_stack_candidates(std::span(words).first(10), executable, describe);
    EXPECT_EQ(entire.retained, 6u);
    EXPECT_FALSE(entire.prefix_limited) << "six at the last copied word is not an omitted prefix";
    const auto none = scan_host_stack_candidates(std::span(words).first(1), executable, describe);
    EXPECT_STREQ(none.text.data(), "-");
    EXPECT_FALSE(none.prefix_limited);
    const auto text = format_native_host_wait_trace(23, limited, words.size(), false, {}, {});
    EXPECT_NE(std::string(text.lines[0].data()).find("prefix-limited=1"), std::string::npos);
    EXPECT_NE(std::string(text.lines[0].data()).find("copied-window-only raw-candidates-not-cfi"),
              std::string::npos);
    EXPECT_NE(std::string(text.lines[1].data()).find(limited.text.data()), std::string::npos);
    EXPECT_NE(std::string(text.lines[2].data()).find("armed=0 found=n/a stored=n/a"),
              std::string::npos);
    EXPECT_NE(std::string(text.lines[2].data()).find("unobserved-not-no-wait"), std::string::npos);
}

TEST(NativeHostWaitReport, LossAndOutputBoundsSurviveTheRealLineFormatter) {
    std::array<NativeHostWaitRecord, 8> records{};
    for (auto& record : records)
        record = {23,          NativeHostWaitSite::PthreadOnce,
                  UINTPTR_MAX, NativeHostWaitMode::RelativeMicroseconds,
                  UINT64_MAX,  UINT64_MAX};
    const NativeHostWaitSnapshot sample{512, 3, UINT64_MAX, 19};
    const auto text = format_native_host_wait_trace(23, {}, 2048, true, sample, records);
    EXPECT_NE(std::string(text.lines[2].data()).find("found=512 stored=8 unstable-global=3"),
              std::string::npos);
    EXPECT_NE(std::string(text.lines[2].data()).find("dropped-run=19"), std::string::npos);
    for (size_t i = 0; i < text.lines.size(); ++i) {
        const std::string line(text.lines[i].data());
        EXPECT_FALSE(line.empty()) << i;
        EXPECT_LT(line.size(), 384u) << "do not silently truncate at the actual 512-byte dump sink";
        EXPECT_EQ(line.back(), '\n');
    }
    EXPECT_NE(std::string(text.lines[10].data()).find("site=pthread_once/contended"),
              std::string::npos);
}

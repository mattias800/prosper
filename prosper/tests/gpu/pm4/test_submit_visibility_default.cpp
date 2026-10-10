// #2220: completion ownership belongs to the submit import, including before an SDK table query.
// Exercise real packets and the registered return hook; no Vulkan device or game data is needed.
#include "gpu/pm4/command_processor.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pending_write_snapshot.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <thread>
#include <vector>

extern "C" void* prosper_agc_reg_defaults(unsigned int version);
extern "C" void* prosper_agc_reg_defaults_internal(unsigned int version);

namespace {
using namespace prosper;
using namespace prosper::gpu;

uint32_t header(uint32_t words, uint32_t subop) {
    return 0xc0000000u | ((words - 2u) << 16u) | (IT_NOP << 8u) | (subop << 2u);
}

void append_write(std::vector<uint32_t>& stream, void* destination, uint32_t value) {
    const auto address = reinterpret_cast<uintptr_t>(destination);
    const uint32_t packet[] = {header(6, R_WRITE_DATA),
                               0,
                               static_cast<uint32_t>(address),
                               static_cast<uint32_t>(address >> 32u),
                               1,
                               value};
    stream.insert(stream.end(), std::begin(packet), std::end(packet));
}

void append_dma(std::vector<uint32_t>& stream, uint64_t destination, uint64_t source,
                uint32_t bytes, uint32_t selectors) {
    const uint32_t packet[] = {header(7, R_DMA_DATA),
                               static_cast<uint32_t>(destination),
                               static_cast<uint32_t>(destination >> 32u),
                               static_cast<uint32_t>(source),
                               static_cast<uint32_t>(source >> 32u),
                               bytes,
                               selectors};
    stream.insert(stream.end(), std::begin(packet), std::end(packet));
}

struct SubmitReturn {
    void (*hook)() = nullptr;
    ~SubmitReturn() {
        if (hook) hook();
        prosper_gpu_drain_completion_writes();
    }
};

std::optional<PendingWriteSnapshot> observe_queue() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        if (auto snapshot = try_pending_write_snapshot()) return snapshot;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return std::nullopt;
}

void check_import_visibility() {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_EQ(hook, &prosper_gpu_submit_scope_end);
    prosper_gpu_drain_completion_writes();

    alignas(8) uint64_t label = 0xaaaaaaaa55555555ull;
    uint32_t upload = 0;
    std::vector<uint32_t> stream;
    // An initialization of the completion's own bytes must stay private too. An unrelated
    // resource upload must be available to the synchronous renderer during this same import.
    append_write(stream, &label, 0);
    append_write(stream, &upload, 0x12345678u);
    const auto address = reinterpret_cast<uintptr_t>(&label);
    const uint32_t release[] = {header(7, R_RELEASE_MEM),
                                static_cast<uint32_t>(address),
                                static_cast<uint32_t>(address >> 32u),
                                2,
                                0x89abcdefu,
                                0x01234567u,
                                0x04};
    stream.insert(stream.end(), std::begin(release), std::end(release));
    struct Packet {
        uint32_t* address;
        uint32_t words;
        uint32_t padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};

    const uint64_t result = submit(reinterpret_cast<uintptr_t>(&packet), 0, 0, 0, 0, 0);
    SubmitReturn retirement{
        hook}; // Also drains before the labels and packet storage go out of scope.
    EXPECT_EQ(result, 0u);
    ASSERT_TRUE(prosper_gpu_submit_scope_active())
        << "SDK query must not control completion ownership";
    prosper_gpu_drain_renderer_writes();
    EXPECT_EQ(upload, 0x12345678u);
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull);
    const auto private_queue = observe_queue();
    ASSERT_TRUE(private_queue.has_value());
    EXPECT_EQ(private_queue->active_submits, 1u);
    EXPECT_EQ(private_queue->queued, 2u);

    hook();
    retirement.hook = nullptr;
    EXPECT_FALSE(prosper_gpu_submit_scope_active());
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, 0x0123456789abcdefull);
    const auto retired = observe_queue();
    ASSERT_TRUE(retired.has_value());
    EXPECT_EQ(retired->queued, 0u);
    EXPECT_EQ(retired->inflight_batches, 0u);
}

TEST(SubmitVisibilityDefault, BeforeAnySdkQuery) {
    check_import_visibility();
}

class SubmitVisibilitySdk : public testing::TestWithParam<unsigned int> {};

TEST_P(SubmitVisibilitySdk, PublicDefaults) {
    ASSERT_NE(prosper_agc_reg_defaults(GetParam()), nullptr);
    check_import_visibility();
}

TEST_P(SubmitVisibilitySdk, InternalDefaults) {
    ASSERT_NE(prosper_agc_reg_defaults_internal(GetParam()), nullptr);
    check_import_visibility();
}

INSTANTIATE_TEST_SUITE_P(AllSdkVersions, SubmitVisibilitySdk, testing::Values(8u, 10u, 12u, 13u));

class SubmitVisibilityOrdered : public testing::TestWithParam<bool> {};

TEST_P(SubmitVisibilityOrdered, CompletionAndAliasingSuffixStayPrivate) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_EQ(hook, &prosper_gpu_submit_scope_end);
    prosper_gpu_drain_completion_writes();

    alignas(8) uint64_t label = 0xaaaaaaaa55555555ull;
    uint32_t source = 0x12345678u, copied = 0, upload = 0;
    std::vector<uint32_t> stream;
    // A real address copy selects the shipping ordered executor, whose suffix once bypassed
    // submit ownership. Preserve a later same-label write as well as the completion itself.
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), reinterpret_cast<uint64_t>(&source),
               sizeof(source), kDmaDataAddressSource);
    const auto address = reinterpret_cast<uint64_t>(&label);
    if (GetParam()) {
        const uint32_t event[] = {0xc0020000u | (IT_EVENT_WRITE << 8u), 0x14u,
                                  static_cast<uint32_t>(address),
                                  static_cast<uint32_t>(address >> 32u)};
        stream.insert(stream.end(), std::begin(event), std::end(event));
    } else {
        const uint32_t release[] = {header(7, R_RELEASE_MEM),
                                    static_cast<uint32_t>(address),
                                    static_cast<uint32_t>(address >> 32u),
                                    2u,
                                    0x89abcdefu,
                                    0x01234567u,
                                    0x04u};
        stream.insert(stream.end(), std::begin(release), std::end(release));
    }
    append_write(stream, &label, 0xdeadbeefu);
    append_write(stream, &upload, 0x87654321u);
    struct Packet {
        uint32_t* address;
        uint32_t words;
        uint32_t padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    const auto result = submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0);
    SubmitReturn retirement{hook};
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(copied, source) << "the ordered DMA must actually execute";
    EXPECT_EQ(upload, 0x87654321u) << "unrelated ordered resources remain available";
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull);
    const auto pending = observe_queue();
    ASSERT_TRUE(pending);
    EXPECT_EQ(pending->queued, 2u);
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(static_cast<uint32_t>(label), 0xdeadbeefu)
        << "the same-label suffix must land after its completion";
    if (!GetParam()) EXPECT_EQ(label, 0x01234567deadbeefull);
}

INSTANTIATE_TEST_SUITE_P(CompletionKinds, SubmitVisibilityOrdered, testing::Bool());

TEST(SubmitVisibilityDefault, OrderedGdsOffsetZeroRemainsAvailable) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_EQ(hook, &prosper_gpu_submit_scope_end);
    prosper_gpu_drain_completion_writes();
    uint32_t source = 0x12345678u, copied = 0;
    uint8_t* gds = compute_gds_backing();
    ASSERT_NE(gds, nullptr);
    ASSERT_GE(compute_gds_size(), sizeof(source));
    const uint32_t previous = [&] {
        uint32_t value;
        std::memcpy(&value, gds, sizeof(value));
        return value;
    }();
    std::vector<uint32_t> stream;
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), reinterpret_cast<uint64_t>(&source),
               sizeof(source), kDmaDataAddressSource);
    append_dma(stream, 0, 0x87654321u, sizeof(source), 1u | (3u << 8u));
    struct Packet {
        uint32_t* address;
        uint32_t words;
        uint32_t padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    const auto result = submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0);
    SubmitReturn retirement{hook};
    uint32_t observed = 0;
    std::memcpy(&observed, gds, sizeof(observed));
    std::memcpy(gds, &previous, sizeof(previous));
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(copied, source);
    EXPECT_TRUE(prosper_gpu_submit_scope_active());
    EXPECT_EQ(observed, 0x87654321u) << "GDS offset zero is not a null guest address";
}

TEST(SubmitVisibilityDefault, ReadyDeferredTailRetainsCompletionOwnership) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_EQ(hook, &prosper_gpu_submit_scope_end);
    prosper_gpu_drain_completion_writes();
    alignas(8) uint64_t barrier = 0, label = 0xaaaaaaaa55555555ull;
    uint32_t upload = 0;
    const auto barrier_address = reinterpret_cast<uint64_t>(&barrier);
    const auto label_address = reinterpret_cast<uint64_t>(&label);
    std::vector<uint32_t> stream{header(8, R_WAIT_MEM_64),
                                 static_cast<uint32_t>(barrier_address),
                                 static_cast<uint32_t>(barrier_address >> 32u),
                                 0xffffffffu,
                                 0u,
                                 1u,
                                 0u,
                                 3u,
                                 header(7, R_RELEASE_MEM),
                                 static_cast<uint32_t>(label_address),
                                 static_cast<uint32_t>(label_address >> 32u),
                                 2u,
                                 0x89abcdefu,
                                 0x01234567u,
                                 0x04u};
    append_write(stream, &label, 0xdeadbeefu);
    struct Packet {
        uint32_t* address;
        uint32_t words;
        uint32_t padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_TRUE(deferred_pending()) << "the first submit must really stop at its wait";
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull);
    hook();
    retirement.hook = nullptr;

    // Keep the publishing checkpoint active before making the barrier ready: the watchdog may
    // otherwise release the tail between the host store and the next submit's admission.
    prosper_gpu_submit_scope_begin();
    SubmitReturn publishing_scope{&prosper_gpu_submit_scope_end};
    barrier = 1;
    std::vector<uint32_t> next;
    append_write(next, &upload, 0x12345678u);
    packet = {next.data(), static_cast<uint32_t>(next.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    retirement.hook = hook;
    EXPECT_FALSE(deferred_pending());
    prosper_gpu_drain_renderer_writes();
    EXPECT_EQ(upload, 0x12345678u);
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull)
        << "a ready deferred tail must not bypass the active import boundary";
    hook();
    retirement.hook = nullptr;
    publishing_scope.hook();
    publishing_scope.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, 0x01234567deadbeefull);
}

TEST(SubmitVisibilityDefault, RejectedImportRetiresItsScope) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_EQ(hook, &prosper_gpu_submit_scope_end);
    EXPECT_NE(submit(0, 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_TRUE(prosper_gpu_submit_scope_active());
    hook();
    retirement.hook = nullptr;
    EXPECT_FALSE(prosper_gpu_submit_scope_active());
}
} // namespace

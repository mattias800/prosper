// #2220: completion ownership belongs to the submit import, including before an SDK table query.
// Exercise real packets and the registered return hook; no Vulkan device or game data is needed.
#include "gpu/pm4/command_processor.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pending_write_snapshot.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "host/memory/guest_memory_map.hpp"
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <thread>
#include <tuple>
#include <vector>
#ifdef PROSPER_TEST_NATIVE_UNTRACKED_TAIL
#include <sys/mman.h>
#endif

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

void append_release(std::vector<uint32_t>& stream, uint64_t* destination, uint64_t value) {
    const auto address = reinterpret_cast<uint64_t>(destination);
    const uint32_t packet[] = {header(7, R_RELEASE_MEM),
                               static_cast<uint32_t>(address),
                               static_cast<uint32_t>(address >> 32u),
                               2u,
                               static_cast<uint32_t>(value),
                               static_cast<uint32_t>(value >> 32u),
                               0x04u};
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
    if (!private_queue) {
        ADD_FAILURE() << "private queue observation timed out";
        return;
    }
    EXPECT_EQ(private_queue.value().active_submits, 1u);
    EXPECT_EQ(private_queue.value().queued, 2u);

    hook();
    retirement.hook = nullptr;
    EXPECT_FALSE(prosper_gpu_submit_scope_active());
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, 0x0123456789abcdefull);
    const auto retired = observe_queue();
    if (!retired) {
        ADD_FAILURE() << "retired queue observation timed out";
        return;
    }
    EXPECT_EQ(retired.value().queued, 0u);
    EXPECT_EQ(retired.value().inflight_batches, 0u);
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
    if (!pending) {
        ADD_FAILURE() << "pending queue observation timed out";
        return;
    }
    EXPECT_EQ(pending.value().queued, 2u);
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

class SubmitVisibilityGdsPrefix : public testing::TestWithParam<uint32_t> {};

TEST_P(SubmitVisibilityGdsPrefix, UploadPrecedesFirstAddressCopy) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    uint32_t source = 0x12345678u, copied = 0;
    auto* gds = compute_gds_backing();
    ASSERT_NE(gds, nullptr);
    ASSERT_GE(compute_gds_size(), GetParam() + sizeof(source));
    uint32_t previous = 0, observed = 0;
    std::memcpy(&previous, gds + GetParam(), sizeof(previous));
    std::vector<uint32_t> stream;
    append_dma(stream, GetParam(), previous ^ 0xffffffffu, sizeof(source), 1u | (3u << 8u));
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), reinterpret_cast<uint64_t>(&source),
               sizeof(source), kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    std::memcpy(&observed, gds + GetParam(), sizeof(observed));
    EXPECT_EQ(copied, source) << "the address copy selects the ordered executor";
    EXPECT_TRUE(prosper_gpu_submit_scope_active());
    EXPECT_EQ(observed, previous ^ 0xffffffffu)
        << "prefix GDS uploads use their own domain and execute before the copy";
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    std::memcpy(gds + GetParam(), &previous, sizeof(previous));
}

INSTANTIATE_TEST_SUITE_P(ZeroAndNonzeroOffsets, SubmitVisibilityGdsPrefix,
                         testing::Values(0u, 0x24u));

TEST(SubmitVisibilityDefault, AddresslessPipelineEventDoesNotBlockResources) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    uint32_t upload = 0, source = 0x12345678u, copied = 0;
    std::vector<uint32_t> stream{0xc0000000u | (IT_EVENT_WRITE << 8u), 0x16u};
    append_write(stream, &upload, 0x87654321u);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), reinterpret_cast<uint64_t>(&source),
               sizeof(source), kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_TRUE(prosper_gpu_submit_scope_active());
    EXPECT_EQ(upload, 0x87654321u);
    EXPECT_EQ(copied, source) << "an address-less pipeline event has no private memory footprint";
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

class SubmitVisibilityDma : public testing::TestWithParam<bool> {};

TEST_P(SubmitVisibilityDma, ReadsPrivateCompletionAtItsOrderedPosition) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0xaaaaaaaa55555555ull, copied = 0;
    constexpr uint64_t completed = 0x0123456789abcdefull;
    const uint32_t offset = GetParam() ? 2u : 0u, bytes = GetParam() ? 4u : 8u;
    uint64_t expected = 0;
    std::memcpy(&expected, reinterpret_cast<const uint8_t*>(&completed) + offset, bytes);
    std::vector<uint32_t> stream;
    append_release(stream, &label, completed);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied),
               reinterpret_cast<uint64_t>(&label) + offset, bytes, kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_EQ(copied, expected) << "GPU consumers read private bytes in command order";
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull) << "the CPU cannot recycle this completion yet";
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, completed);
}

TEST_P(SubmitVisibilityDma, AliasingCopyOwnsItsSourceAndFeedsALaterCopy) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0xaaaaaaaa55555555ull, source = 0x1020304050607080ull, copied = 0;
    uint64_t expected = 0x0123456789abcdefull;
    const uint32_t offset = GetParam() ? 2u : 0u, bytes = GetParam() ? 3u : 8u;
    std::memcpy(reinterpret_cast<uint8_t*>(&expected) + offset, &source, bytes);
    std::vector<uint32_t> stream;
    append_release(stream, &label, 0x0123456789abcdefull);
    append_dma(stream, reinterpret_cast<uint64_t>(&label) + offset,
               reinterpret_cast<uint64_t>(&source), bytes, kDmaDataAddressSource);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), reinterpret_cast<uint64_t>(&label),
               sizeof(label), kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull) << "address DMA must retain aliasing writes too";
    EXPECT_EQ(copied, expected) << "the later copy consumes the first copy's owned private payload";
    source = ~source; // The guest can recycle the input after the DMA operation consumed it.
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(label, expected) << "retirement must not reread a later generation of the source";
}

INSTANTIATE_TEST_SUITE_P(ExactAndPartialOverlap, SubmitVisibilityDma, testing::Bool());

// Map a second virtual window onto the second physical page, then split the first tracker
// record. Neither a virtual overlap nor the first mapping record can identify this alias.
struct DirectAliases {
    static constexpr uint64_t page = 0x10000;
    using Function = decltype(Hle::lookup(""));
    Function unmap = nullptr, release = nullptr;
    uint64_t physical = 0, first = 0, second = 0, first_bytes = 2 * page;
    bool initialize(bool first_page_alias = false) {
        register_builtin_hle();
        const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        const auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
        unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
        release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        return allocate && map && protect && unmap && release &&
               allocate(0, 16ull << 30, 2 * page, page, 0, reinterpret_cast<uint64_t>(&physical)) ==
                   0 &&
               physical &&
               map(reinterpret_cast<uint64_t>(&first), 2 * page, 2, 0, physical, page) == 0 &&
               first &&
               map(reinterpret_cast<uint64_t>(&second), page, 2, 0,
                   physical + (first_page_alias ? 0 : page), page) == 0 &&
               second && second != first && protect(first + page, 0x4000, 2, 0, 0, 0) == 0;
    }
    ~DirectAliases() {
        if (first && unmap) EXPECT_EQ(unmap(first, first_bytes, 0, 0, 0, 0), 0u);
        if (second && unmap) EXPECT_EQ(unmap(second, page, 0, 0, 0, 0), 0u);
        if (physical && release) EXPECT_EQ(release(physical, 2 * page, 0, 0, 0, 0), 0u);
    }
    uint64_t* label() const { return reinterpret_cast<uint64_t*>(first + page + 0x128); }
    uint64_t alias() const { return second + 0x128; }
};

// A completion at the head of a resource must not hide its unrelated argument tail from an
// ordered consumer. A backend write to that tail must also survive later completion retirement.
class SubmitVisibilityResourceTail : public testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(SubmitVisibilityResourceTail, IndirectConsumerSeesTailAndRetirementKeepsLaterProducer) {
    register_builtin_hle();
    prosper_gpu_drain_completion_writes();
    const auto create_shader = Hle::lookup("f3dg2CSgRKY");
    ASSERT_NE(create_shader, nullptr);
    namespace P = prosper::agc::Pm4;
    alignas(256) static const uint32_t code[] = {0xbf810000u};
    ShaderReg registers[] = {{P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0}};
    AgcShaderHeader shader{};
    shader.file_header = 0x34333231u;
    shader.version = 0x18;
    shader.sh_registers = registers;
    shader.shader_size = sizeof(code);
    shader.num_sh_registers = 2;
    void* registered = nullptr;
    ASSERT_EQ(create_shader(reinterpret_cast<uint64_t>(&registered),
                            reinterpret_cast<uint64_t>(&shader), reinterpret_cast<uint64_t>(code),
                            0, 0, 0),
              0u);
    ASSERT_EQ(registered, &shader);
    const auto [physical_alias, device_indirect] = GetParam();
    DirectAliases aliases;
    if (physical_alias) ASSERT_TRUE(aliases.initialize());
    std::array<uint64_t, 4> storage{};
    auto* label = physical_alias ? aliases.label() : storage.data();
    const uint64_t destination =
        physical_alias ? aliases.alias() : reinterpret_cast<uint64_t>(label);
    std::memset(label, 0, sizeof(storage));
    const uint64_t arguments = reinterpret_cast<uint64_t>(label) + 16;
    std::array<uint32_t, 8> source{0x1234u, 0u, 0u, 0u, 2u, 3u, 4u, 0u};
    uint64_t scratch_source = 0x42, scratch_destination = 0;
    GpuState state;
    state.sh[P::COMPUTE_PGM_LO] = registers[0].value;
    state.sh[P::COMPUTE_PGM_HI] = registers[1].value;
    state.dma_copies.push_back({reinterpret_cast<uint64_t>(&scratch_destination),
                                reinterpret_cast<uint64_t>(&scratch_source), 8,
                                kDmaDataAddressSource, 1});
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(label);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    state.ordered_memory_effects.emplace_back(completion, 2);
    state.dma_copies.push_back({destination, reinterpret_cast<uint64_t>(source.data()),
                                sizeof(source), kDmaDataAddressSource, 3});
    state.parser_stalls.push_back({4});
    GpuState::Dispatch dispatch;
    dispatch.indirect = true;
    dispatch.indirect_args_addr = arguments;
    dispatch.command_order = 5;
    state.dispatches.push_back(dispatch);
    uint32_t calls = 0;
    struct ResetBackend {
        ~ResetBackend() {
            set_submit_compute({});
            set_submit_compute_indirect_dispatch(false);
        }
    } reset_backend;
    set_submit_compute([&](const std::vector<ComputeItem>& items) {
        for (const auto& item : items) {
            ++calls;
            EXPECT_TRUE(item.indirect_dispatch) << "the item retains the source packet form";
            EXPECT_EQ(item.indirect_args_addr, device_indirect ? arguments : 0u)
                << "only the device route retains an argument address";
            std::array<uint32_t, 3> observed{};
            if (device_indirect && item.indirect_args_addr)
                std::memcpy(observed.data(), reinterpret_cast<const void*>(item.indirect_args_addr),
                            sizeof(observed));
            else
                observed = {item.launch.groups_x, item.launch.groups_y, item.launch.groups_z};
            EXPECT_EQ(observed, (std::array<uint32_t, 3>{2, 3, 4}));
            EXPECT_EQ(*label, 0u) << "GPU resource admission must not publish the completion";
            const std::array<uint32_t, 3> produced{5, 6, 7};
            std::memcpy(reinterpret_cast<void*>(arguments), produced.data(), sizeof(produced));
        }
        return !items.empty();
    });
    set_submit_compute_indirect_dispatch(device_indirect);
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    EXPECT_TRUE(execute_nonrender_submit_work(state, 1));
    EXPECT_EQ(calls, 1u) << "private labels cannot turn a nonzero indirect dispatch into a no-op";
    source.fill(0x79);
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*label, 0x1234u) << "the completion-overlapping fragment owns the original source";
    std::array<uint32_t, 3> tail{};
    std::memcpy(tail.data(), reinterpret_cast<const void*>(arguments), sizeof(tail));
    EXPECT_EQ(tail, (std::array<uint32_t, 3>{5, 6, 7}))
        << "retirement cannot replay an already consumed tail over its later producer";
}

INSTANTIATE_TEST_SUITE_P(VirtualAndPhysicalCpuAndDevice, SubmitVisibilityResourceTail,
                         testing::Combine(testing::Bool(), testing::Bool()));

class SubmitVisibilityIndex : public testing::TestWithParam<unsigned> {};

TEST_P(SubmitVisibilityIndex, OrderedUploadsAndDisjointCopiesVisitALinearNumberOfRecords) {
    prosper_gpu_drain_completion_writes();
    constexpr size_t count = 128;
    std::array<uint64_t, count> labels{}, uploads{}, copied{};
    std::array<uint64_t, count> sources;
    sources.fill(0x12345678u);
    std::array<std::array<uint32_t, 2>, count> words;
    words.fill({0x12345678u, 0u});
    uint64_t scratch_source = 1, scratch_destination = 0;
    GpuState state;
    uint64_t order = 1;
    state.dma_copies.push_back({reinterpret_cast<uint64_t>(&scratch_destination),
                                reinterpret_cast<uint64_t>(&scratch_source), 8,
                                kDmaDataAddressSource, order++});
    const auto completion = [&](size_t i) {
        Pm4Command command{};
        command.kind = Pm4Command::Kind::ReleaseMem;
        command.rel_addr = reinterpret_cast<uint64_t>(&labels[i]);
        command.rel_data_sel = 2;
        command.rel_value_valid = true;
        command.rel_value = 0x42;
        state.ordered_memory_effects.emplace_back(command, order++);
    };
    const auto upload = [&](size_t i) {
        Pm4Command command{};
        command.kind = Pm4Command::Kind::WriteData;
        command.wd_addr = reinterpret_cast<uint64_t>(&uploads[i]);
        command.wd_data = words[i].data();
        command.wd_num = 2;
        command.wd_valid = true;
        state.ordered_memory_effects.emplace_back(command, order++);
    };
    const auto copy = [&](size_t i) {
        state.dma_copies.push_back({reinterpret_cast<uint64_t>(&copied[i]),
                                    reinterpret_cast<uint64_t>(&sources[i]), 8,
                                    kDmaDataAddressSource, order++});
    };
    if (GetParam() == 0) {
        for (size_t i = 0; i < count; ++i) completion(i);
        for (size_t i = 0; i < count; ++i) upload(i);
        for (size_t i = 0; i < count; ++i) copy(i);
    } else {
        for (size_t i = 0; i < count; ++i) {
            completion(i);
            if (GetParam() == 1)
                upload(i);
            else
                copy(i);
        }
        for (size_t i = 0; i < count; ++i)
            if (GetParam() == 1)
                copy(i);
            else
                upload(i);
    }
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    const auto before = observe_queue();
    ASSERT_TRUE(before);
    EXPECT_TRUE(execute_nonrender_submit_work(state, 1));
    const auto after = observe_queue();
    ASSERT_TRUE(after);
    if (!before || !after)
        return; // ASSERT_TRUE reports the failure; make optional access explicit.
    EXPECT_EQ(uploads, sources);
    EXPECT_EQ(copied, sources);
    EXPECT_EQ(labels, (std::array<uint64_t, count>{}));
    EXPECT_EQ(after->queued, count);
    // Count actual record visits, including cached visits. A full scan per ordered upload/copy
    // violates this bound even when all mapping queries themselves were cached.
    EXPECT_LE(after->pending_records_examined - before->pending_records_examined,
              8u * (3u * count + 1u));
    EXPECT_LE(after->pending_geometry_queries - before->pending_geometry_queries,
              8u * (3u * count + 1u));
}

INSTANTIATE_TEST_SUITE_P(BurstAndAlternatingWriteAndDma, SubmitVisibilityIndex,
                         testing::Values(0u, 1u, 2u));

TEST_P(SubmitVisibilityIndex, SkippedUnalignedEffectCannotSupplyAPrivateOverlay) {
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0, copied = 0;
    uint32_t invalid_payload = 0x12345678u;
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(&label);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    Pm4Command invalid{};
    invalid.kind = GetParam() == 0   ? Pm4Command::Kind::WriteData
                   : GetParam() == 1 ? Pm4Command::Kind::ReleaseMem
                                     : Pm4Command::Kind::EventWrite;
    invalid.wd_addr = invalid.rel_addr = invalid.event_addr =
        reinterpret_cast<uint64_t>(&label) + 1;
    invalid.wd_data = &invalid_payload;
    invalid.wd_num = 1;
    invalid.wd_valid = true;
    invalid.rel_data_sel = 2;
    invalid.rel_value_valid = true;
    invalid.rel_value = 2;
    execute_ordered_memory_effect(GpuState::MemoryEffect(invalid, 2));
    EXPECT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied),
                                          reinterpret_cast<uint64_t>(&label), sizeof(copied),
                                          kDmaDataAddressSource, 3}));
    EXPECT_EQ(copied, 1u) << "a skipped packet is not an ordered byte producer";
    EXPECT_EQ(label, 0u);
}

TEST_P(SubmitVisibilityIndex, RejectedImmediateDmaCannotSupplyAPrivateOverlay) {
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0, copied = 0;
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(&label);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    Pm4Command invalid{};
    invalid.kind = Pm4Command::Kind::DmaData;
    invalid.dd_valid = true;
    invalid.dd_dst = reinterpret_cast<uint64_t>(&label) + (GetParam() == 1 ? 1u : 0u);
    invalid.dd_src = 0x1234;
    invalid.dd_sels = GetParam() == 0 ? 0x103u : 3u;
    invalid.dd_bytes = GetParam() == 2 ? 0x1000001u : 8u;
    execute_ordered_memory_effect(GpuState::MemoryEffect(invalid, 2));
    EXPECT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied),
                                          reinterpret_cast<uint64_t>(&label), sizeof(copied),
                                          kDmaDataAddressSource, 3}));
    EXPECT_EQ(copied, 1u) << "a rejected DMA form is not an ordered byte producer";
    EXPECT_EQ(label, 0u);
}

TEST(SubmitVisibilityDefault, RetiredIntervalsCannotBlockANewSubmit) {
    prosper_gpu_drain_completion_writes();
    uint64_t destination = 0, source = 0x12345678u;
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(&destination);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    prosper_gpu_drain_renderer_writes();
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    ASSERT_EQ(destination, 1u);
    prosper_gpu_submit_scope_begin();
    retirement.hook = &prosper_gpu_submit_scope_end;
    ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&destination),
                                          reinterpret_cast<uint64_t>(&source), 8,
                                          kDmaDataAddressSource, 1}));
    EXPECT_EQ(destination, source) << "a drained completion has no private footprint";
}

class SubmitVisibilityFragments : public testing::TestWithParam<unsigned> {};

TEST_P(SubmitVisibilityFragments, InteriorLabelsKeepOnlyTheirBytesAndPublishedProducerSurvives) {
    prosper_gpu_drain_completion_writes();
    alignas(8) std::array<uint32_t, 16> destination{};
    std::array<uint32_t, 16> source;
    source.fill(0x12345678u);
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    for (size_t word : {2u, 8u}) {
        Pm4Command completion{};
        completion.kind = Pm4Command::Kind::ReleaseMem;
        completion.rel_addr = reinterpret_cast<uint64_t>(&destination[word]);
        completion.rel_data_sel = 2;
        completion.rel_value_valid = true;
        completion.rel_value = 1;
        execute_ordered_memory_effect(GpuState::MemoryEffect(completion, word));
    }
    if (GetParam() == 2) {
        ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(destination.data()),
                                              reinterpret_cast<uint64_t>(source.data()),
                                              sizeof(source), kDmaDataAddressSource, 10}));
    } else {
        Pm4Command command{};
        command.kind = GetParam() == 0 ? Pm4Command::Kind::WriteData : Pm4Command::Kind::DmaData;
        command.wd_addr = command.dd_dst = reinterpret_cast<uint64_t>(destination.data());
        command.wd_data = source.data();
        command.wd_num = source.size();
        command.wd_valid = command.dd_valid = true;
        command.dd_src = source[0];
        command.dd_bytes = sizeof(source);
        execute_ordered_memory_effect(GpuState::MemoryEffect(command, 10));
    }
    for (size_t i = 0; i < destination.size(); ++i)
        EXPECT_EQ(destination[i], (i == 2 || i == 3 || i == 8 || i == 9) ? 0u : source[i]);
    destination[12] = 0xabcdefu; // later GPU producer in an already consumed public suffix
    uint32_t copied = 0;
    ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied),
                                          reinterpret_cast<uint64_t>(&destination[12]),
                                          sizeof(copied), kDmaDataAddressSource, 11}));
    EXPECT_EQ(copied, 0xabcdefu) << "private overlays cannot replay a published suffix";
    source.fill(0x79);
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    for (size_t i = 0; i < destination.size(); ++i)
        EXPECT_EQ(destination[i], i == 12 ? 0xabcdefu : 0x12345678u);
}

INSTANTIATE_TEST_SUITE_P(WriteImmediateAndAddressDma, SubmitVisibilityFragments,
                         testing::Values(0u, 1u, 2u));

class SubmitVisibilityUnknownSource : public testing::TestWithParam<bool> {};

TEST_P(SubmitVisibilityUnknownSource, LaterFixedWriteResolvesOnlyTheBytesItOverwrites) {
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0, copied = 0xaaaaaaaa55555555ull;
    std::array<uint32_t, 2> fixed{0x12345678u, 0xabcdefu};
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command timestamp{};
    timestamp.kind = Pm4Command::Kind::EventWrite;
    timestamp.event_addr = reinterpret_cast<uint64_t>(&label);
    execute_ordered_memory_effect(GpuState::MemoryEffect(timestamp, 1));
    Pm4Command write{};
    write.kind = Pm4Command::Kind::WriteData;
    write.wd_addr = reinterpret_cast<uint64_t>(&label);
    write.wd_data = fixed.data();
    write.wd_num = GetParam() ? 1 : 2;
    write.wd_valid = true;
    execute_ordered_memory_effect(GpuState::MemoryEffect(write, 2));
    const bool copied_ok = execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied),
                                                     reinterpret_cast<uint64_t>(&label),
                                                     sizeof(copied), kDmaDataAddressSource, 3});
    EXPECT_EQ(copied_ok, !GetParam());
    if (GetParam())
        EXPECT_EQ(copied, 0xaaaaaaaa55555555ull);
    else
        EXPECT_EQ(std::memcmp(&copied, fixed.data(), sizeof(copied)), 0);
    EXPECT_EQ(label, 0u) << "neither a timestamp nor its fixed suffix becomes CPU-visible";
}

INSTANTIATE_TEST_SUITE_P(FullAndPartialOverwrite, SubmitVisibilityUnknownSource, testing::Bool());

TEST(SubmitVisibilityDefault, PartialByteFragmentsKeepOriginalPayloadOffsets) {
    prosper_gpu_drain_completion_writes();
    alignas(8) std::array<uint8_t, 24> destination{};
    std::array<uint8_t, 17> source;
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i + 1);
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(destination.data() + 8);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(destination.data() + 3),
                                          reinterpret_cast<uint64_t>(source.data()), source.size(),
                                          kDmaDataAddressSource, 2}));
    for (size_t i = 0; i < source.size(); ++i)
        EXPECT_EQ(destination[i + 3], i >= 5 && i < 13 ? 0u : source[i]);
    uint64_t copied = 0;
    ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied),
                                          reinterpret_cast<uint64_t>(destination.data() + 8),
                                          sizeof(copied), kDmaDataAddressSource, 3}));
    EXPECT_EQ(std::memcmp(&copied, source.data() + 5, sizeof(copied)), 0);
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(std::memcmp(destination.data() + 3, source.data(), source.size()), 0);
    EXPECT_EQ(destination[2], 0u);
    EXPECT_EQ(destination[20], 0u);
}

TEST_P(SubmitVisibilityFragments, RetirementDoesNotRequireAnAlreadyPublishedUnmappedTail) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    prosper_gpu_drain_completion_writes();
    const uint64_t address = aliases.first + 0x128;
    auto* label = reinterpret_cast<uint64_t*>(address);
    *label = 0;
    std::vector<uint32_t> source((DirectAliases::page + 32) / 4, 0x12345678u);
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = address;
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    if (GetParam() == 2) {
        ASSERT_TRUE(execute_ordered_dma_copy({address, reinterpret_cast<uint64_t>(source.data()),
                                              static_cast<uint32_t>(source.size() * 4),
                                              kDmaDataAddressSource, 2}));
    } else {
        Pm4Command resource{};
        resource.kind = GetParam() == 0 ? Pm4Command::Kind::WriteData : Pm4Command::Kind::DmaData;
        resource.wd_addr = resource.dd_dst = address;
        resource.wd_data = source.data();
        resource.wd_num = source.size();
        resource.wd_valid = resource.dd_valid = true;
        resource.dd_src = source[0];
        resource.dd_bytes = source.size() * 4;
        execute_ordered_memory_effect(GpuState::MemoryEffect(resource, 2));
    }
    EXPECT_EQ(*label, 0u);
    EXPECT_EQ(*reinterpret_cast<const uint32_t*>(aliases.first + DirectAliases::page + 0x128),
              source[0]);
    ASSERT_EQ(aliases.unmap(aliases.first + DirectAliases::page, DirectAliases::page, 0, 0, 0, 0),
              0u);
    aliases.first_bytes = DirectAliases::page;
    uint64_t copied = 0;
    EXPECT_TRUE(execute_ordered_dma_copy(
        {reinterpret_cast<uint64_t>(&copied), address, sizeof(copied), kDmaDataAddressSource, 3}))
        << "a consumed tail has no lifetime dependency for a retained-head consumer";
    EXPECT_EQ(copied, 0x1234567812345678ull);
    source.assign(source.size(), 0x79);
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*label, 0x1234567812345678ull)
        << "the retained head owns its payload and has no dependency on the consumed tail";
}

TEST(SubmitVisibilityDefault, MappingChangeRetainsTheEarlierResourceFragmentAsAFifoBlocker) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    ASSERT_NE(map, nullptr);
    prosper_gpu_drain_completion_writes();
    auto* label = aliases.label();
    auto* alias = reinterpret_cast<uint64_t*>(aliases.alias());
    *label = 0;
    uint64_t source = 2, copied = 0;
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(label);
    completion.rel_data_sel = 2;
    completion.rel_value_valid = true;
    completion.rel_value = 1;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    ASSERT_TRUE(execute_ordered_dma_copy({aliases.alias(), reinterpret_cast<uint64_t>(&source),
                                          sizeof(source), kDmaDataAddressSource, 2}));
    EXPECT_EQ(*alias, 0u);
    // Move the completion VA to the OTHER physical page while its old resource alias stays put.
    uint64_t remapped = aliases.first + DirectAliases::page;
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&remapped), DirectAliases::page, 2, 0x10,
                  aliases.physical, DirectAliases::page),
              0u);
    ASSERT_EQ(remapped, aliases.first + DirectAliases::page);
    source = 3;
    ASSERT_TRUE(execute_ordered_dma_copy({aliases.alias(), reinterpret_cast<uint64_t>(&source),
                                          sizeof(source), kDmaDataAddressSource, 3}));
    EXPECT_EQ(*alias, 0u) << "a retained residual still blocks later writes after remapping";
    ASSERT_TRUE(execute_ordered_dma_copy({reinterpret_cast<uint64_t>(&copied), aliases.alias(),
                                          sizeof(copied), kDmaDataAddressSource, 4}));
    EXPECT_EQ(copied, 3u) << "the private consumer observes both resource writes in FIFO order";
    prosper_gpu_submit_scope_end();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*label, 1u);
    EXPECT_EQ(*alias, 3u) << "retirement cannot replay the older residual over a public suffix";
}

TEST_P(SubmitVisibilityDma, ReadsPrivateCompletionViaPhysicalAlias) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    ASSERT_EQ(guest_memory_topology_relation(reinterpret_cast<uint64_t>(aliases.label()), 8,
                                             aliases.alias(), 8),
              GuestMemoryTopologyRelation::Overlap);
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    *aliases.label() = 0xaaaaaaaa55555555ull;
    constexpr uint64_t completed = 0x0123456789abcdefull;
    const uint32_t offset = GetParam() ? 2u : 0u, bytes = GetParam() ? 4u : 8u;
    uint64_t copied = 0, expected = 0;
    std::memcpy(&expected, reinterpret_cast<const uint8_t*>(&completed) + offset, bytes);
    std::vector<uint32_t> stream;
    append_release(stream, aliases.label(), completed);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied), aliases.alias() + offset, bytes,
               kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_EQ(copied, expected) << "a second VA names the same private physical bytes";
    EXPECT_EQ(*aliases.label(), 0xaaaaaaaa55555555ull);
}

TEST_P(SubmitVisibilityDma, CopyToPhysicalAliasRemainsPrivate) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    *aliases.label() = 0xaaaaaaaa55555555ull;
    uint64_t source = 0x1020304050607080ull, copied = 0;
    uint64_t expected = 0x0123456789abcdefull;
    const uint32_t offset = GetParam() ? 2u : 0u, bytes = GetParam() ? 3u : 8u;
    std::memcpy(reinterpret_cast<uint8_t*>(&expected) + offset, &source, bytes);
    std::vector<uint32_t> stream;
    append_release(stream, aliases.label(), 0x0123456789abcdefull);
    append_dma(stream, aliases.alias() + offset, reinterpret_cast<uint64_t>(&source), bytes,
               kDmaDataAddressSource);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied),
               reinterpret_cast<uint64_t>(aliases.label()), 8, kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_EQ(*aliases.label(), 0xaaaaaaaa55555555ull)
        << "a physically aliased destination must not release the CPU-visible label";
    EXPECT_EQ(copied, expected) << "a later consumer sees the alias write after the completion";
    source = ~source;
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*aliases.label(), expected);
}

TEST_P(SubmitVisibilityDma, ResourceWriteToPhysicalAliasRemainsPrivate) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    *aliases.label() = 0xaaaaaaaa55555555ull;
    uint64_t expected = 0x0123456789abcdefull;
    const uint32_t offset = GetParam() ? 4u : 0u, value = 0x10203040u;
    uint64_t copied = 0;
    std::memcpy(reinterpret_cast<uint8_t*>(&expected) + offset, &value, sizeof(value));
    std::vector<uint32_t> stream;
    append_release(stream, aliases.label(), 0x0123456789abcdefull);
    append_write(stream, reinterpret_cast<void*>(aliases.alias() + offset), value);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied),
               reinterpret_cast<uint64_t>(aliases.label()), 8, kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    EXPECT_EQ(*aliases.label(), 0xaaaaaaaa55555555ull);
    EXPECT_EQ(copied, expected) << "renderer drain cannot expose a physically aliased label write";
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    EXPECT_EQ(*aliases.label(), expected);
}

TEST(SubmitVisibilityDefault, MappingSlicesAreExactAndFailWithoutPartialProof) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize());
    {
        const GuestMappingLease lease;
        std::vector<GuestMemoryMappingSlice> slices;
        GuestMemoryMappingCoverage coverage = GuestMemoryMappingCoverage::Untracked;
        ASSERT_TRUE(guest_memory_mapping_slices(lease, aliases.first, 2 * DirectAliases::page,
                                                slices, &coverage));
        EXPECT_EQ(coverage, GuestMemoryMappingCoverage::Complete);
        ASSERT_EQ(slices.size(), 1u) << "a protection split preserves contiguous physical bytes";
        EXPECT_EQ(slices[0].offset, 0u);
        EXPECT_EQ(slices[0].physical, aliases.physical);
        EXPECT_EQ(slices[0].bytes, 2 * DirectAliases::page);
        EXPECT_TRUE(slices[0].direct);
        ASSERT_TRUE(guest_memory_mapping_slices(lease, aliases.alias(), 8, slices));
        ASSERT_EQ(slices.size(), 1u);
        EXPECT_EQ(slices[0].physical, aliases.physical + DirectAliases::page + 0x128);
        for (const auto& range : std::array<std::array<uint64_t, 2>, 3>{
                 {{0, 8}, {aliases.first, 0}, {UINT64_MAX - 3, 8}}}) {
            EXPECT_FALSE(guest_memory_mapping_slices(lease, range[0], range[1], slices, &coverage));
            EXPECT_EQ(coverage, GuestMemoryMappingCoverage::Incomplete);
            EXPECT_TRUE(slices.empty()) << "unknown topology cannot leave an earlier proof";
        }
    }
    ASSERT_EQ(aliases.unmap(aliases.first + DirectAliases::page, DirectAliases::page, 0, 0, 0, 0),
              0u);
    aliases.first_bytes = DirectAliases::page;
    {
        const GuestMappingLease lease;
        std::vector<GuestMemoryMappingSlice> slices{{0, 8, aliases.physical, true}};
        GuestMemoryMappingCoverage coverage = GuestMemoryMappingCoverage::Complete;
        EXPECT_FALSE(guest_memory_mapping_slices(lease, aliases.first, 2 * DirectAliases::page,
                                                 slices, &coverage));
        EXPECT_EQ(coverage, GuestMemoryMappingCoverage::Incomplete);
        EXPECT_TRUE(slices.empty()) << "a mapped prefix cannot prove the missing suffix";
    }
}

#ifdef PROSPER_TEST_NATIVE_UNTRACKED_TAIL
// Replace only our unmapped tail, without overwriting another allocation. The resulting source
// is readable across a direct prefix and an untracked native suffix: readability is not topology.
struct UntrackedTail {
    uint64_t address = 0;
    bool initialize(uint64_t requested) {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
        flags |= MAP_FIXED_NOREPLACE;
#endif
        void* mapped = mmap(reinterpret_cast<void*>(requested), DirectAliases::page,
                            PROT_READ | PROT_WRITE, flags, -1, 0);
        if (mapped == MAP_FAILED) return false;
        if (reinterpret_cast<uint64_t>(mapped) != requested) {
            munmap(mapped, DirectAliases::page);
            return false;
        }
        address = requested;
        host::notify_guest_mapping_added(address, DirectAliases::page, true);
        return true;
    }
    ~UntrackedTail() {
        if (!address) return;
        host::notify_guest_mapping_removed(address, DirectAliases::page);
        EXPECT_EQ(munmap(reinterpret_cast<void*>(address), DirectAliases::page), 0);
    }
};

class SubmitVisibilityIncomplete : public testing::TestWithParam<bool> {};

TEST_P(SubmitVisibilityIncomplete, ReadableMixedRangeCannotBypassPrivateAlias) {
    DirectAliases aliases;
    ASSERT_TRUE(aliases.initialize(true));
    ASSERT_EQ(aliases.unmap(aliases.first + DirectAliases::page, DirectAliases::page, 0, 0, 0, 0),
              0u);
    aliases.first_bytes = DirectAliases::page;
    UntrackedTail tail;
    ASSERT_TRUE(tail.initialize(aliases.first + DirectAliases::page));
    auto* label = reinterpret_cast<uint64_t*>(aliases.second + DirectAliases::page - 8);
    const uint64_t mixed = aliases.first + DirectAliases::page - 8;
    {
        const GuestMappingLease lease;
        GuestMemoryMappingCoverage coverage = GuestMemoryMappingCoverage::Complete;
        std::vector<GuestMemoryMappingSlice> slices;
        EXPECT_FALSE(guest_memory_mapping_slices(lease, mixed, 16, slices, &coverage));
        EXPECT_EQ(coverage, GuestMemoryMappingCoverage::Incomplete);
        EXPECT_TRUE(slices.empty());
        EXPECT_FALSE(guest_memory_mapping_slices(lease, tail.address, 8, slices, &coverage));
        EXPECT_EQ(coverage, GuestMemoryMappingCoverage::Untracked);
        EXPECT_TRUE(slices.empty());
    }
    *label = 0xaaaaaaaa55555555ull;
    std::memset(reinterpret_cast<void*>(tail.address), 0x53, 8);
    std::array<uint8_t, 16> copied{};
    GpuState::DmaCopy copy{};
    copy.src = mixed;
    copy.dst = reinterpret_cast<uint64_t>(copied.data());
    copy.bytes = copied.size();
    copy.sels = kDmaDataAddressSource;
    prosper_gpu_drain_completion_writes();
    prosper_gpu_submit_scope_begin();
    SubmitReturn noop_retirement{&prosper_gpu_submit_scope_end};
    Pm4Command pipeline_event{};
    pipeline_event.kind = Pm4Command::Kind::EventWrite;
    execute_ordered_memory_effect(GpuState::MemoryEffect(pipeline_event, 1));
    ASSERT_TRUE(execute_ordered_dma_copy(copy))
        << "both halves are readable; an address-less event adds no private dependency";
    ASSERT_EQ(std::memcmp(copied.data(), reinterpret_cast<void*>(mixed), copied.size()), 0);
    prosper_gpu_submit_scope_end();
    noop_retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    copied.fill(0x79);
    if (GetParam()) std::swap(copy.src, copy.dst);
    std::array<uint8_t, 16> before;
    std::memcpy(before.data(), reinterpret_cast<void*>(copy.dst), before.size());
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    Pm4Command completion{};
    completion.kind = Pm4Command::Kind::ReleaseMem;
    completion.rel_addr = reinterpret_cast<uint64_t>(label);
    completion.rel_value = 0x0123456789abcdefull;
    completion.rel_value_valid = true;
    completion.rel_data_sel = 2;
    execute_ordered_memory_effect(GpuState::MemoryEffect(completion, 1));
    EXPECT_FALSE(execute_ordered_dma_copy(copy))
        << "partial mapping identity cannot establish independence from a private physical alias";
    EXPECT_EQ(std::memcmp(before.data(), reinterpret_cast<void*>(copy.dst), before.size()), 0);
    EXPECT_EQ(*label, 0xaaaaaaaa55555555ull);
}

INSTANTIATE_TEST_SUITE_P(SourceAndDestination, SubmitVisibilityIncomplete, testing::Bool());
#endif

class SubmitVisibilitySuffix : public testing::TestWithParam<bool> {};

TEST_P(SubmitVisibilitySuffix, OverlappingTailPreservesPrivateWriteOrder) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    alignas(8) std::array<uint8_t, 32> destination{};
    std::array<uint8_t, 32> source;
    source.fill(0x53);
    uint32_t suffix = 0x12345678u, copied = 0;
    std::vector<uint32_t> stream;
    append_release(stream, reinterpret_cast<uint64_t*>(destination.data()), 1);
    append_dma(stream, reinterpret_cast<uint64_t>(destination.data()),
               reinterpret_cast<uint64_t>(source.data()), source.size(), kDmaDataAddressSource);
    if (GetParam())
        append_dma(stream, reinterpret_cast<uint64_t>(destination.data() + 16),
                   reinterpret_cast<uint64_t>(&suffix), sizeof(suffix), kDmaDataAddressSource);
    else
        append_write(stream, destination.data() + 16, suffix);
    append_dma(stream, reinterpret_cast<uint64_t>(&copied),
               reinterpret_cast<uint64_t>(destination.data() + 16), sizeof(copied),
               kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    uint32_t public_tail = 0;
    std::memcpy(&public_tail, destination.data() + 16, sizeof(public_tail));
    EXPECT_EQ(public_tail, suffix) << "bytes outside a private completion remain available";
    EXPECT_EQ(destination[0], 0u) << "the completion-overlapping fragment stays private";
    EXPECT_EQ(copied, suffix) << "the final consumer must see the suffix in FIFO order";
    source.fill(0x79);
    suffix = ~suffix;
    hook();
    retirement.hook = nullptr;
    prosper_gpu_drain_completion_writes();
    uint32_t retired_tail = 0;
    std::memcpy(&retired_tail, destination.data() + 16, sizeof(retired_tail));
    EXPECT_EQ(retired_tail, 0x12345678u) << "the older large DMA cannot overwrite a later write";
    EXPECT_EQ(destination[0], 0x53u) << "the private DMA owns its already consumed source";
}

INSTANTIATE_TEST_SUITE_P(WriteAndDmaSuffix, SubmitVisibilitySuffix, testing::Bool());

TEST(SubmitVisibilityDefault, MemoryToGdsConsumesPrivateCompletionAtOffsetZero) {
    register_builtin_hle();
    const auto submit = Hle::lookup("UglJIZjGssM");
    const auto hook = Hle::return_hook_of("UglJIZjGssM");
    ASSERT_NE(submit, nullptr);
    ASSERT_NE(hook, nullptr);
    prosper_gpu_drain_completion_writes();
    auto* gds = compute_gds_backing();
    ASSERT_NE(gds, nullptr);
    ASSERT_GE(compute_gds_size(), sizeof(uint32_t));
    uint32_t previous = 0;
    std::memcpy(&previous, gds, sizeof(previous));
    uint64_t label = 0xaaaaaaaa55555555ull;
    std::vector<uint32_t> stream;
    append_release(stream, &label, 0x0123456789abcdefull);
    append_dma(stream, 0, reinterpret_cast<uint64_t>(&label), sizeof(uint32_t),
               1u | (3u << 8u) | kDmaDataAddressSource);
    struct Packet {
        uint32_t* address;
        uint32_t words, padding;
    } packet{stream.data(), static_cast<uint32_t>(stream.size()), 0};
    EXPECT_EQ(submit(reinterpret_cast<uint64_t>(&packet), 0, 0, 0, 0, 0), 0u);
    SubmitReturn retirement{hook};
    uint32_t observed = 0;
    std::memcpy(&observed, gds, sizeof(observed));
    std::memcpy(gds, &previous, sizeof(previous));
    EXPECT_EQ(observed, 0x89abcdefu)
        << "GDS has its own domain but reads the ordered private source";
    EXPECT_EQ(label, 0xaaaaaaaa55555555ull);
}

TEST(SubmitVisibilityDefault, UncapturedTimestampDependencyIsRefused) {
    prosper_gpu_drain_completion_writes();
    uint64_t label = 0, copied = 0x1122334455667788ull;
    Pm4Command event{};
    event.kind = Pm4Command::Kind::EventWrite;
    event.event_addr = reinterpret_cast<uint64_t>(&label);
    prosper_gpu_submit_scope_begin();
    SubmitReturn retirement{&prosper_gpu_submit_scope_end};
    execute_ordered_memory_effect(GpuState::MemoryEffect(event, 1));
    GpuState::DmaCopy copy{};
    copy.dst = reinterpret_cast<uint64_t>(&copied);
    copy.src = reinterpret_cast<uint64_t>(&label);
    copy.bytes = sizeof(label);
    copy.sels = kDmaDataAddressSource;
    copy.command_order = 2;
    EXPECT_FALSE(execute_ordered_dma_copy(copy))
        << "an early invented timestamp is not ordered data";
    EXPECT_EQ(copied, 0x1122334455667788ull);
    EXPECT_EQ(label, 0u);
}
}   // namespace

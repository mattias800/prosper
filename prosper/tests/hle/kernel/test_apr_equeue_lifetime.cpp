// #4103: APR userdata belongs to a queue lifetime, even when a real CreateEqueue allocation
// reuses the previous handle. Force only that allocation to reuse its retained storage; every
// handle, registration and completion still comes from the registered production callers.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

namespace allocation_probe {
struct Record {
    void* base;
    size_t size;
    bool retired;
};
thread_local bool capture = false, reuse = false;
thread_local std::array<Record, 32> records{};
thread_local size_t count = 0, attempts = 0, hit_ordinal = 0;
thread_local bool overflow = false;
std::atomic<void*> retained{nullptr};
std::atomic<size_t> retained_size{0}, deletions{0};
std::atomic<bool> retired{false}, duplicate_delete{false};

void begin_capture() {
    count = 0;
    overflow = false;
    capture = true;
}

Record containing(uint64_t handle) {
    Record match{};
    unsigned matches = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& record = records[i];
        const uint64_t base = reinterpret_cast<uintptr_t>(record.base);
        if (!record.retired && handle >= base && handle - base < record.size) {
            match = record;
            ++matches;
        }
    }
    return matches == 1 ? match : Record{};
}

void retain(const Record& record) {
    retained_size = record.size;
    deletions = 0;
    retired = false;
    duplicate_delete = false;
    retained = record.base;
}

void arm_reuse() {
    attempts = 0;
    hit_ordinal = 0;
    reuse = true;
}

void release_retired_storage() {
    // Called only after the real last-reference deallocation has been observed. Completion
    // receipt alone does not prove that a producer has released its shared queue reference.
    void* block = retained.exchange(nullptr);
    std::free(block);
}
} // namespace allocation_probe

void* operator new(size_t size) {
    using namespace allocation_probe;
    void* block = nullptr;
    if (reuse) {
        ++attempts;
        if (retired && size == retained_size) {
            block = retained.load();
            retired = false;
            reuse = false;
            hit_ordinal = attempts;
        }
    }
    if (!block) block = std::malloc(size ? size : 1);
    if (!block) throw std::bad_alloc();
    if (capture) {
        if (count < records.size())
            records[count++] = {block, size, false};
        else
            overflow = true;
    }
    return block;
}

void operator delete(void* block) noexcept {
    using namespace allocation_probe;
    if (!block) return;
    if (capture)
        for (size_t i = 0; i < count; ++i)
            if (records[i].base == block && !records[i].retired) records[i].retired = true;
    if (block == retained.load()) {
        ++deletions;
        if (retired.exchange(true)) duplicate_delete = true;
    } else
        std::free(block);
}

void operator delete(void* block, size_t) noexcept {
    ::operator delete(block);
}

namespace prosper {
uint64_t prosper_eq_identity(uint64_t eq);
uint64_t prosper_eq_apr_udata(uint64_t eq, int64_t id);
}   // namespace prosper
using namespace prosper;

namespace {
struct UnfinishedStorage {
    bool drained = false;
    ~UnfinishedStorage() {
        if (drained) return;
        // A fatal assertion may leave retained allocation storage or an accepted Submit live.
        // Preserve its diagnostic, then bypass static registry teardown and unsafe reclamation.
        // Such a setup/protocol exit is not a targeted userdata negative control.
        std::fflush(nullptr);
        std::_Exit(1);
    }
};
uint64_t address(const void* value) {
    return reinterpret_cast<uintptr_t>(value);
}

struct Event {
    int64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    uint64_t udata;
};
static_assert(sizeof(Event) == 32 && offsetof(Event, data) == 16 && offsetof(Event, udata) == 24);
struct Api {
    HleFn create, destroy, add, bind, submit, wait, count, userdata, destroy_cb;
};
struct Channel {
    uint64_t eq;
    int64_t id;
    alignas(16) std::array<uint8_t, 256> cb{};
};

TEST(AprEqueueLifetime, AllocatorInterposerControl) {
    UnfinishedStorage storage;
    using namespace allocation_probe;
    // Observe replacement-allocator side effects through ordinary indirect calls: GCC can fold
    // bookkeeping checks across direct allocation calls using its builtin allocation assumptions.
    void* (*volatile allocate)(size_t) = ::operator new;
    void (*volatile deallocate)(void*) noexcept = ::operator delete;
    begin_capture();
    void* original = allocate(73);
    capture = false;
    ASSERT_TRUE(!overflow && count == 1 && records[0].base == original && records[0].size == 73)
        << "allocator control records one real scalar allocation";
    retain(records[0]);
    deallocate(original);
    ASSERT_TRUE(retired && deletions == 1 && !duplicate_delete)
        << "allocator control observes real deletion while retaining raw storage";
    void* ordinary = allocate(73);
    ASSERT_NE(ordinary, original)
        << "unarmed same-size allocations do not consume retained storage";
    deallocate(ordinary);
    ASSERT_EQ(deletions.load(), 1U)
        << "unrelated deletion does not retire the retained block twice";
    arm_reuse();
    void* replacement = allocate(73);
    const bool consumed = !reuse && hit_ordinal == 1;
    reuse = false;
    ASSERT_TRUE(replacement == original && consumed && !retired)
        << "one armed allocation reuses the exact retained block";
    auto* byte = static_cast<unsigned char*>(replacement);
    *byte = 0x5A;
    ASSERT_EQ(*byte, 0x5A) << "reused allocator-control storage is writable";
    deallocate(replacement);
    ASSERT_TRUE(retired && deletions == 2 && !duplicate_delete)
        << "allocator control observes the replacement deletion exactly once";
    release_retired_storage();
    storage.drained = true;
}

void deliver(const Api& api, const Channel& channel, uint64_t tag, uint64_t expected,
             const char* userdata_label) {
    ASSERT_EQ(api.bind(address(channel.cb.data()), channel.eq, static_cast<uint64_t>(channel.id),
                       tag, 0, 0),
              0ULL)
        << "registered legacy binding accepts the live queue";
    ASSERT_EQ(api.submit(address(channel.cb.data()), 1, 0, 0, 0, 0), 0ULL)
        << "registered plain Submit accepts the owned command buffer";
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    uint32_t timeout = 0;
    Event event{};
    bool received = false, protocol = true;
    for (unsigned polls = 0; polls < 2000 && std::chrono::steady_clock::now() < deadline; ++polls) {
        int32_t count = -1;
        const auto rc =
            api.wait(channel.eq, address(&event), 1, address(&count), address(&timeout), 0);
        if (rc == 0 && count == 1) {
            received = true;
            break;
        }
        if (rc != 0x8002003CULL || count != 0) {
            protocol = false;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(received && protocol)
        << "bounded timeout-zero polling receives the accepted completion";
    ASSERT_TRUE(event.ident == channel.id && event.filter == -25 &&
                static_cast<uint64_t>(event.data) == tag)
        << "the real completion retains its APR id, filter and tag";
    ASSERT_EQ(api.count(channel.eq, 0, 0, 0, 0, 0), 0ULL)
        << "the consumed completion leaves the live queue empty";
    const auto accessed = api.userdata(address(&event), 0, 0, 0, 0, 0);
    EXPECT_TRUE(event.udata == expected && accessed == expected) << userdata_label;
}

void await_retirement() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (unsigned polls = 0;
         polls < 2000 && !(allocation_probe::retired && allocation_probe::deletions == 2) &&
         std::chrono::steady_clock::now() < deadline;
         ++polls)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_TRUE(allocation_probe::retired && allocation_probe::deletions == 2 &&
                !allocation_probe::duplicate_delete)
        << "the replacement's actual last-reference deletion retires its storage exactly once";
    allocation_probe::release_retired_storage();
}
}   // namespace

TEST(AprEqueueLifetime, DeletedHandleRejectsRegistrationAndReuseStartsFresh) {
    UnfinishedStorage storage;
    constexpr const char* diagnostics[] = {"PROSPER_DUMPCODE", "PROSPER_WAITCALLER",
                                           "PROSPER_WAITCAP", "PROSPER_AMPRLOG", "PROSPER_EVLOG"};
    bool quiet = true;
    for (const char* key : diagnostics) {
#ifdef _WIN32
        quiet = (_putenv_s(key, "") == 0) && quiet;
#else
        quiet = (::unsetenv(key) == 0) && quiet;
#endif
    }
    ASSERT_TRUE(quiet) << "guest-memory diagnostics are disabled before registered calls";
    register_builtin_hle();
    const Api api{Hle::lookup(nid_hash("sceKernelCreateEqueue")),
                  Hle::lookup(nid_hash("sceKernelDeleteEqueue")),
                  Hle::lookup("bBfz7kMF2Ho"),
                  Hle::lookup("H896Pt-yB4I"),
                  Hle::lookup("eE4Szl8sil8"),
                  Hle::lookup(nid_hash("sceKernelWaitEqueue")),
                  Hle::lookup(nid_hash("sceKernelGetEventCount")),
                  Hle::lookup(nid_hash("sceKernelGetEventUserData")),
                  Hle::lookup("GuchCTefuZw")};
    ASSERT_TRUE(api.create && api.destroy && api.add && api.bind && api.submit && api.wait &&
                api.count && api.userdata && api.destroy_cb)
        << "all queue, APR and completion APIs are registered";
    ASSERT_EQ(api.add, Hle::lookup(nid_hash("sceKernelAddAmprEvent")))
        << "the actual AddAmprEvent NID resolves to its named caller";

    using namespace allocation_probe;
    uint64_t original = 0;
    begin_capture();
    const auto created = api.create(address(&original), 0, 0, 0, 0, 0);
    capture = false;
    ASSERT_TRUE(created == 0 && original && !overflow)
        << "real first CreateEqueue succeeds within the fixed trace";
    const Record first = containing(original);
    ASSERT_TRUE(first.base && first.size)
        << "a unique live Create allocation contains the actual returned handle";
    retain(first);
    const uint64_t old_identity = prosper_eq_identity(original);
    ASSERT_NE(old_identity, 0ULL) << "the first real queue has a live lifetime identity";

    std::array<uint64_t, 8> contexts{};
    const uint64_t a0 = address(&contexts[0]), a1 = address(&contexts[1]);
    const uint64_t z0 = address(&contexts[2]), z1 = address(&contexts[3]);
    const uint64_t b0 = address(&contexts[4]), b1 = address(&contexts[5]);
    const uint64_t f0 = address(&contexts[6]), f1 = address(&contexts[7]);
    uint64_t completion_tag = 0;
    ASSERT_TRUE(address(&completion_tag) != b0 && address(&completion_tag) != a0 &&
                address(&completion_tag) != z0)
        << "the pointer completion tag differs from opaque userdata";
    ASSERT_EQ(api.add(original, 0, a0, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.add(original, 1, a1, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_TRUE(prosper_eq_apr_udata(original, 0) == a0 && prosper_eq_apr_udata(original, 1) == a1)
        << "the first lifetime stores both original nonzero contexts";
    // No submission references the first queue before its real allocation is deleted.
    uint64_t neighbor = 0;
    ASSERT_TRUE(api.create(address(&neighbor), 0, 0, 0, 0, 0) == 0 && neighbor &&
                neighbor != original)
        << "an unarmed ordinary queue receives independent live storage";
    ASSERT_EQ(api.add(neighbor, 0, f0, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.add(neighbor, 1, f1, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.destroy(original, 0, 0, 0, 0, 0), 0ULL)
        << "registered DeleteEqueue accepts the first lifetime";
    ASSERT_TRUE(retired && deletions == 1 && !duplicate_delete)
        << "real DeleteEqueue releases the captured allocation before forced reuse";
    ASSERT_EQ(prosper_eq_identity(original), 0ULL)
        << "the deleted numeric handle has no live identity";
    EXPECT_EQ(prosper_eq_apr_udata(original, 0), 0ULL)
        << "DeleteEqueue purges the old id0 APR registration";
    EXPECT_EQ(prosper_eq_apr_udata(original, 1), 0ULL)
        << "DeleteEqueue purges the old id1 APR registration";
    ASSERT_EQ(api.add(original, 0, z0, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.add(original, 1, z1, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    EXPECT_EQ(prosper_eq_apr_udata(original, 0), 0ULL)
        << "Add on the deleted handle cannot resurrect id0 userdata";
    EXPECT_EQ(prosper_eq_apr_udata(original, 1), 0ULL)
        << "Add on the deleted handle cannot resurrect id1 userdata";
    EXPECT_TRUE(prosper_eq_apr_udata(neighbor, 0) == f0 && prosper_eq_apr_udata(neighbor, 1) == f1)
        << "purging one lifetime preserves both neighboring live registrations";

    uint64_t replacement = 0;
    arm_reuse();
    begin_capture();
    const auto recreated = api.create(address(&replacement), 0, 0, 0, 0, 0);
    capture = false;
    const bool consumed = !reuse && hit_ordinal == 1;
    reuse = false;
    ASSERT_TRUE(recreated == 0 && consumed && !overflow && replacement == original &&
                containing(replacement).base == first.base && !retired)
        << "replacement Create consumes the retained first allocation and returns the exact same "
           "handle";
    ASSERT_TRUE(prosper_eq_identity(replacement) != 0 &&
                prosper_eq_identity(replacement) != old_identity)
        << "the reused handle belongs to a new real queue identity";
    ASSERT_EQ(api.add(replacement, 0, b0, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.add(replacement, 1, b1, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    EXPECT_EQ(prosper_eq_apr_udata(replacement, 0), b0)
        << "the new id0 lifetime stores its own context";
    EXPECT_EQ(prosper_eq_apr_udata(replacement, 1), b1)
        << "the new id1 lifetime stores its own context";
    ASSERT_EQ(api.add(replacement, 0, 0, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    ASSERT_EQ(api.add(replacement, 1, a1, 0, 0, 0), 0ULL)
        << "registered AddAmprEvent retains its success return";
    EXPECT_EQ(prosper_eq_apr_udata(replacement, 0), b0)
        << "same-lifetime zero registration preserves the new id0 context";
    EXPECT_EQ(prosper_eq_apr_udata(replacement, 1), b1)
        << "same-lifetime nonzero registration preserves the new id1 context";

    std::array<Channel, 3> channels{{{replacement, 0}, {replacement, 1}, {neighbor, 1}}};
    ASSERT_NO_FATAL_FAILURE(
        deliver(api, channels[0], address(&completion_tag), b0,
                "reused-handle pointer completion and accessor return only the new id0 context"));
    ASSERT_NO_FATAL_FAILURE(
        deliver(api, channels[1], 0, b1,
                "reused-handle id1 completion and accessor return only the new id1 context"));
    ASSERT_NO_FATAL_FAILURE(
        deliver(api, channels[2], 0, f1,
                "ordinary neighboring completion and accessor retain their independent context"));
    for (const auto& channel : channels)
        ASSERT_EQ(api.destroy_cb(address(channel.cb.data()), 0, 0, 0, 0, 0), 0ULL)
            << "registered command-buffer cleanup succeeds after completion receipt";
    ASSERT_EQ(api.destroy(neighbor, 0, 0, 0, 0, 0), 0ULL)
        << "registered deletion releases the drained ordinary queue";
    ASSERT_EQ(api.destroy(replacement, 0, 0, 0, 0, 0), 0ULL)
        << "registered deletion releases the drained replacement queue";
    ASSERT_NO_FATAL_FAILURE(await_retirement());
    storage.drained = true;
}

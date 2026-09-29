#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>

using namespace prosper;
using namespace std::chrono_literals;
namespace P = prosper::agc::Pm4;

alignas(256) static const uint32_t kNoopCompute[] = {0xbf810000u};
static std::atomic<std::promise<void>*> g_contention_promise{nullptr};
static void on_mapping_contention() {
    if (auto* promise = g_contention_promise.exchange(nullptr)) promise->set_value();
}

int main() {
    register_builtin_hle();
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    const auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    const auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    const auto batch = Hle::lookup(nid_hash("sceKernelBatchMap"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    if (!allocate || !map || !reserve || !flexible || !protect || !batch ||
        !unmap || !release) return 2;

    constexpr uint64_t page = 0x10000;
    uint64_t physical = 0, source = 0, destination = 0, reservation = 0;
    if (allocate(0, 0x200000000ull, 2 * page, page, 0,
                 reinterpret_cast<uint64_t>(&physical)) != 0 || !physical ||
        map(reinterpret_cast<uint64_t>(&source), page, 3, 0, physical, page) != 0 ||
        map(reinterpret_cast<uint64_t>(&destination), page, 3, 0,
            physical + page, page) != 0 ||
        reserve(reinterpret_cast<uint64_t>(&reservation), page, 0, page, 0, 0) != 0)
        return 2;

    int failures = 0;
    const auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "guest mapping lease: %s\n", message);
            ++failures;
        }
    };
    const auto relation = [&](uint64_t a, uint64_t b) {
        return guest_memory_topology_relation(a, page, b, page);
    };
    expect(relation(source, destination) == GuestMemoryTopologyRelation::Disjoint,
           "initial physical ranges are disjoint");
    *reinterpret_cast<uint8_t*>(source) = 0x5a;
#if defined(__linux__)
    {
        GuestMappingLease query_lease;
        expect(guest_memory_direct_range_fault_safe(query_lease, source, page),
               "fully committed direct source covers its fault granule");
        expect(!guest_memory_direct_range_fault_safe(query_lease, reservation, page),
               "uncommitted reservation cannot certify a fault granule");
    }
    uint64_t partial = 0;
    expect(reserve(reinterpret_cast<uint64_t>(&partial), page, 0, page, 0, 0) == 0 &&
               map(reinterpret_cast<uint64_t>(&partial), 0x4000, 3, 0x10,
                   physical, 0x4000) == 0,
           "map a direct 16 KiB slice into a 64 KiB reservation");
    {
        GuestMappingLease query_lease;
        expect(!guest_memory_direct_range_fault_safe(query_lease, partial, 0x4000),
               "a direct 16 KiB slice cannot certify its surrounding fault granule");
        expect(!guest_memory_direct_range_fault_safe(query_lease, source, UINT64_MAX),
               "overflow cannot certify a fault granule");
    }
#else
    {
        GuestMappingLease query_lease;
        expect(!guest_memory_direct_range_fault_safe(query_lease, source, page),
               "unsupported platform keeps fault safety unproved");
    }
#endif

    // The worker announces that it is about to enter the real HLE operation. A held lease must
    // keep the OS mapping and the tracker unchanged until that entire operation can finish.
    const auto blocked_mutation = [&](auto operation, auto while_leased,
                                      const char* message, bool changes_mapping = true) {
        auto lease = std::make_unique<GuestMappingLease>();
        const uint64_t generation_before = host::guest_mapping_generation();
        std::promise<void> contended;
        auto attempted = contended.get_future();
        g_contention_promise.store(&contended);
        set_guest_mapping_mutation_contention_observer_for_test(on_mapping_contention);
        auto completed = std::async(std::launch::async, operation);
        expect(attempted.wait_for(5s) == std::future_status::ready,
               "mutation reached the held lease rather than merely starting a worker");
        set_guest_mapping_mutation_contention_observer_for_test(nullptr);
        g_contention_promise.store(nullptr);
        expect(completed.wait_for(100ms) == std::future_status::timeout, message);
        expect(host::guest_mapping_generation() == generation_before,
               "mapping generation stays fixed until lease release");
        while_leased();
        lease.reset();
        expect(completed.get() == 0, "mutation succeeds after lease release");
        if (changes_mapping)
            expect(host::guest_mapping_generation() != generation_before,
                   "completed mutation advances mapping generation");
    };

    blocked_mutation(
        [&] { return protect(source, page, 1, 0, 0, 0); },
        [&] {
            expect(relation(source, destination) == GuestMemoryTopologyRelation::Disjoint,
                   "protection cannot change source topology during lease");
            expect(*reinterpret_cast<const uint8_t*>(source) == 0x5a,
                   "source bytes remain mapped during protection attempt");
        },
        "protect cannot finish under shared lease");

    // BatchMap is a separate mutation entry point, commonly used by guest allocators. Its whole
    // transaction must wait even for a one-entry protection update.
    alignas(8) uint8_t batch_entry[0x20]{};
    std::memcpy(batch_entry, &destination, sizeof(destination));
    std::memcpy(batch_entry + 0x10, &page, sizeof(page));
    batch_entry[0x18] = 1;
    const int32_t protect_operation = 2;
    std::memcpy(batch_entry + 0x1c, &protect_operation, sizeof(protect_operation));
    int32_t batch_done = -1;
    blocked_mutation(
        [&] { return batch(reinterpret_cast<uint64_t>(batch_entry), 1,
                           reinterpret_cast<uint64_t>(&batch_done), 0, 0, 0); },
        [&] {
            expect(relation(source, destination) == GuestMemoryTopologyRelation::Disjoint,
                   "BatchMap cannot alter leased topology");
        },
        "BatchMap cannot finish under shared lease");
    expect(batch_done == 1, "BatchMap completes its entry after lease release");

    // A descriptor-free production compute has no nested candidate and must not hold a mapping
    // lease through an unrelated backend wait. The admission tests exercise the positive gate;
    // the held-lease HLE tests above and below exercise the mutation boundary itself.
    const auto create_shader = Hle::lookup("f3dg2CSgRKY");
    gpu::ShaderReg registers[2] = {{P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0}};
    gpu::AgcShaderHeader shader_header{};
    shader_header.file_header = 0x34333231u;
    shader_header.version = 0x18;
    shader_header.sh_registers = registers;
    shader_header.shader_size = sizeof(kNoopCompute);
    shader_header.num_sh_registers = 2;
    void* registered_shader = nullptr;
    expect(create_shader &&
               create_shader(reinterpret_cast<uint64_t>(&registered_shader),
                             reinterpret_cast<uint64_t>(&shader_header),
                             reinterpret_cast<uint64_t>(kNoopCompute), 0, 0, 0) == 0 &&
               registered_shader == &shader_header,
           "register descriptor-free compute shader");
    uint64_t ephemeral = 0;
    expect(flexible(reinterpret_cast<uint64_t>(&ephemeral), page, 3, 0,
                    reinterpret_cast<uint64_t>("lease-submit-test"), 0) == 0 && ephemeral,
           "map guest memory for production submit lifetime test");
    if (registered_shader == &shader_header && ephemeral) {
        gpu::GpuState submit_state;
        submit_state.sh[P::COMPUTE_PGM_LO] = registers[0].value;
        submit_state.sh[P::COMPUTE_PGM_HI] = registers[1].value;
        gpu::GpuState::Dispatch dispatch;
        dispatch.threads_x = dispatch.threads_y = dispatch.threads_z = 1;
        submit_state.dispatches.push_back(dispatch);
        std::promise<void> backend_entered;
        auto entered = backend_entered.get_future();
        std::promise<void> finish_backend;
        const auto finish = finish_backend.get_future().share();
        gpu::set_submit_compute([&](const std::vector<gpu::ComputeItem>& items) {
            backend_entered.set_value();
            finish.wait();
            return !items.empty();
        });
        auto submit = std::async(std::launch::async, [&] {
            return gpu::execute_nonrender_submit_work(submit_state, 1);
        });
        const bool reached_backend = entered.wait_for(5s) == std::future_status::ready;
        expect(reached_backend, "descriptor-free production compute reaches backend");
        if (reached_backend) {
            std::promise<void> contended;
            auto attempted = contended.get_future();
            g_contention_promise.store(&contended);
            set_guest_mapping_mutation_contention_observer_for_test(on_mapping_contention);
            auto deferred_unmap = std::async(std::launch::async, [&] {
                return unmap(ephemeral, page, 0, 0, 0, 0);
            });
            expect(deferred_unmap.wait_for(5s) == std::future_status::ready,
                   "noncandidate submit does not block an unrelated guest unmap");
            set_guest_mapping_mutation_contention_observer_for_test(nullptr);
            g_contention_promise.store(nullptr);
            expect(attempted.wait_for(0ms) == std::future_status::timeout,
                   "noncandidate submit never acquires the admission lease");
            finish_backend.set_value();
            expect(submit.get(), "compute submit finishes after backend release");
            expect(deferred_unmap.get() == 0,
                   "unrelated guest unmap completes during noncandidate compute");
        } else {
            finish_backend.set_value();
            (void)submit.get();
            unmap(ephemeral, page, 0, 0, 0, 0);
        }
        gpu::set_submit_compute({});
    }

    blocked_mutation(
        [&] { return map(reinterpret_cast<uint64_t>(&reservation), page, 3, 0x10,
                         physical + page, page); },
        [&] {
            expect(relation(reservation, destination) == GuestMemoryTopologyRelation::Unknown,
                   "reservation is not published as an alias before lease release");
        },
        "remap-to-alias cannot finish under shared lease");
    expect(relation(reservation, destination) == GuestMemoryTopologyRelation::Overlap,
           "new mapping is observed as a physical alias after lease release");

    blocked_mutation(
        [&] { return unmap(source, page, 0, 0, 0, 0); },
        [&] {
            expect(relation(source, destination) == GuestMemoryTopologyRelation::Disjoint,
                   "unmap cannot remove source during lease");
        },
        "unmap cannot finish under shared lease");
    expect(relation(source, destination) == GuestMemoryTopologyRelation::Unknown,
           "unmap removes source after lease release");
#if defined(__linux__)
    {
        GuestMappingLease query_lease;
        expect(!guest_memory_direct_range_fault_safe(query_lease, source, page),
               "unmap revokes fault-granule safety");
    }
#endif

    blocked_mutation(
        [&] { return release(physical, 2 * page, 0, 0, 0, 0); },
        [&] {
            expect(relation(reservation, destination) == GuestMemoryTopologyRelation::Overlap,
                   "physical release cannot race leased aliases");
        },
        "direct-memory release cannot finish under shared lease", false);
    unmap(reservation, page, 0, 0, 0, 0);
    unmap(destination, page, 0, 0, 0, 0);
#if defined(__linux__)
    if (partial) unmap(partial, page, 0, 0, 0, 0);
#endif
    return failures ? 1 : 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <shared_mutex>

namespace prosper {

// Relationship between two guest virtual ranges after consulting the kernel-memory mapping table.
// `Unknown` is intentionally distinct from `Disjoint`: consumers that need an ordering proof must
// keep fail-closed behavior when either range is untracked, uncommitted, malformed, or has a
// mapping gap. Contiguous committed segments are resolved using each segment's own backing.
enum class GuestMemoryTopologyRelation : uint8_t {
    Unknown,
    Disjoint,
    Overlap,
};

// Keeps HLE mapping transactions from replacing or protecting guest backing while synchronous
// compute work reads its sources, executes, and publishes its completed result. This does not
// exclude fault-handler/VEH lazy commits or guest byte writes. Callers admitting a borrowed source
// must separately prove fault safety, physical alias isolation and current-byte visibility.
// Acquire before inspecting topology or guest bytes; release only after final writeback.
class GuestMappingLease {
public:
    GuestMappingLease();
    GuestMappingLease(const GuestMappingLease&) = delete;
    GuestMappingLease& operator=(const GuestMappingLease&) = delete;

private:
    std::shared_lock<std::shared_mutex> lock_;
};

// Test-only observation after an exclusive mapping attempt has actually found a held reader.
// Null in production; the callback must not call HLE mapping functions.
void set_guest_mapping_mutation_contention_observer_for_test(void (*observer)());

GuestMemoryTopologyRelation guest_memory_topology_relation(
    uint64_t first_address, uint64_t first_size,
    uint64_t second_address, uint64_t second_size);

// Linux's guest fault handler may replace a whole 64 KiB reservation granule on first touch.
// Admission under a GuestMappingLease must require each granule intersecting a source or writable
// destination to be fully backed by committed direct mappings, or a fault in an adjacent reserved
// slice could replace already leased bytes. Other platforms return false until their lazy/sparse
// commitment paths have an equivalent proof. The caller still checks access and physical aliases.
bool guest_memory_direct_range_fault_safe(const GuestMappingLease& lease,
                                          uint64_t address, uint64_t size);

// Compare an exact committed-direct source with the WHOLE allocated physical backing that
// contains producer_address. Retained tiled/layered images must not use their linear pixel size
// as an alias bound. Unknown/unallocated/flexible mappings refuse. minimum_producer_bytes checks
// the producer's complete known physical layout extent; zero is unknown. The returned isolation
// is over its complete original allocation. Retype ledger slices retain original ownership;
// release/reuse never makes a new allocation birth physically disjoint from a retained origin.
struct GuestDirectAllocation {
    uint64_t address = 0, minimum_bytes = 0;
    uint64_t physical_begin = 0, physical_end = 0, identity = 0;
};
// Capture at producer realization/publication while the backing is leased. Retained renderer
// owners keep this observation even after the old VA is unmapped; no capture file supplies it.
GuestDirectAllocation guest_memory_direct_allocation(const GuestMappingLease& lease,
                                                     uint64_t address, uint64_t minimum_bytes);

// A contiguous, readable, already backed virtual window belonging to ONE live original direct
// allocation. Physical bytes progress with virtual bytes; adjacent aliases or allocations cannot
// extend it. Protection/retype splits may extend it only when those facts remain true. The window
// is intersected with native fault/commit safety, never inferred from a VMA or allocation size.
// This observation is valid only under the supplied lease. It does not grant current-byte or
// producer authority: an observer must check the entire selected domain against retained origins
// and attachments, then own its bytes under the submitted-input stability contract.
struct GuestDirectReadableWindow {
    uint64_t virtual_begin = 0, virtual_end = 0;
    uint64_t physical_begin = 0, physical_end = 0;
    GuestDirectAllocation allocation;
    explicit operator bool() const {
        return allocation.identity && virtual_begin < virtual_end && physical_begin < physical_end;
    }
};
GuestDirectReadableWindow guest_memory_direct_readable_window(const GuestMappingLease& lease,
                                                              uint64_t address);

GuestMemoryTopologyRelation guest_memory_retained_allocation_relation(
    const GuestMappingLease& lease, uint64_t source_address, uint64_t source_bytes,
    const GuestDirectAllocation& producer);
GuestMemoryTopologyRelation guest_memory_direct_allocation_relation(
    const GuestMappingLease& lease, uint64_t source_address, uint64_t source_bytes,
    uint64_t producer_address, uint64_t minimum_producer_bytes);

// Copy host bytes into a fully committed direct-memory mapping through prosper's authoritative
// physical backing. This models a device write without weakening the guest VA's CPU protection.
// Private, untracked, malformed, and cross-mapping destinations fail closed.
bool guest_memory_gpu_write_supported(uint64_t destination, size_t bytes);
bool guest_memory_gpu_write(uint64_t destination, const void* source, size_t bytes);

// Linux direct-memory copy through an emulator-private writable shared alias. The callback is
// called only after the destination's topology is proved and its watched guest aliases are marked
// GPU-dirty; the guest VA protection stays armed. False means no complete copy was published and
// the caller must take its ordinary writeback path. The callback must only copy the supplied
// disjoint host-staging bytes; it runs under the topology lock and must not enter mapping/HLE APIs.
// Other hosts conservatively decline.
using GuestMemoryByteCopy = void (*)(void* destination, const void* source, size_t bytes);
bool guest_memory_gpu_write_alias(uint64_t destination, const void* source, size_t bytes,
                                  GuestMemoryByteCopy copy);

// Unit-test witness that the protected/backing-aware device-write path actually ran. Output bytes
// alone cannot prove that lever moved: a host memmove can coincidentally choose the correct direction
// for one pair of physically aliased but VA-disjoint views.
uint64_t guest_memory_gpu_write_successes_for_test();
uint64_t guest_memory_gpu_alias_write_successes_for_test();

// #2384: does an AMPR constructor's `a2` say the command buffer keeps a byte cursor? Exposed for
// test because the answer is a PREDICATE over guest-supplied bit patterns, and the only way to show
// the widening is strictly additive is to assert it over the shapes the working titles pass -- which
// a boot test cannot do without those titles' dumps. The implementation lives in hle_kernel_mem.cpp.
bool ampr_cb_tracks_offset_arg_for_test(uint64_t a2);

} // namespace prosper

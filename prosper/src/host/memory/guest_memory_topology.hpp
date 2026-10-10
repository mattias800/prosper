#pragma once

#include "host/memory/guest_direct_allocation.hpp"
#include <cstdint>
#include <shared_mutex>
#include <vector>

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

GuestMemoryTopologyRelation guest_memory_topology_relation(uint64_t first_address,
                                                           uint64_t first_size,
                                                           uint64_t second_address,
                                                           uint64_t second_size);

// Exact committed mapping slices, including protection splits and GPU-only direct views.
// The lease keeps the mapping identity stable; this observation does not establish current bytes.
// False means incomplete/unknown topology, never physical isolation. Non-direct slices carry no
// pool offset. The output is empty on failure, so no partial mapping proof escapes.
// Optional coverage distinguishes a wholly untracked span from one with tracked but incomplete or
// uncommitted backing. Neither establishes physical isolation; consumers must preserve that status.
struct GuestMemoryMappingSlice {
    uint64_t offset = 0, bytes = 0, physical = 0;
    bool direct = false;
};
enum class GuestMemoryMappingCoverage : uint8_t { Complete, Untracked, Incomplete };
bool guest_memory_mapping_slices(const GuestMappingLease& lease, uint64_t address, uint64_t bytes,
                                 std::vector<GuestMemoryMappingSlice>& slices,
                                 GuestMemoryMappingCoverage* coverage = nullptr);

// True when the kernel-memory HLE tracks a guest mapping that contains `address`, whatever its CPU
// protection and whether or not it is committed yet. This is the guest's own view, so GPU-only
// direct memory (no CPU bits, host PROT_NONE) and lazily committed ranges both answer true, where a
// host readability probe would answer false. It is what separates a descriptor naming no guest
// memory at all (stale bytes) from one at memory the CPU simply cannot read (#4796). Takes the
// mapping-table lock: keep it off per-draw paths that are not already declining something.
bool guest_virtual_address_tracked(uint64_t address);

// Linux's guest fault handler may replace a whole 64 KiB reservation granule on first touch.
// Admission under a GuestMappingLease must require each granule intersecting a source or writable
// destination to be fully backed by committed direct mappings, or a fault in an adjacent reserved
// slice could replace already leased bytes. Windows authenticates one already committed, non-lazy
// direct section view against its live original allocation, tracking and native commitment; private
// fallback, legacy sparse views and COW/guarded access refuse. Other platforms still return false.
// Windows backing identity assumes HLE-managed topology: arbitrary native remap/protection changes
// are not authenticated by this observation (VirtualQuery is not a COW-history detector).
// Neither proof excludes guest byte writers or grants ordering/currentness/physical alias isolation.
// The caller still checks its required access and physical aliases.
bool guest_memory_direct_range_fault_safe(const GuestMappingLease& lease, uint64_t address,
                                          uint64_t size);

// Compare an exact committed-direct source with the WHOLE allocated physical backing that
// contains producer_address. Retained tiled/layered images must not use their linear pixel size
// as an alias bound. Unknown/unallocated/flexible mappings refuse. minimum_producer_bytes checks
// the producer's complete known physical layout extent; zero is unknown. The returned isolation
// is over its complete original allocation. Retype ledger slices retain original ownership;
// release/reuse never makes a new allocation birth physically disjoint from a retained origin.
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

GuestMemoryTopologyRelation
guest_memory_retained_allocation_relation(const GuestMappingLease& lease, uint64_t source_address,
                                          uint64_t source_bytes,
                                          const GuestDirectAllocation& producer);
GuestMemoryTopologyRelation
guest_memory_direct_allocation_relation(const GuestMappingLease& lease, uint64_t source_address,
                                        uint64_t source_bytes, uint64_t producer_address,
                                        uint64_t minimum_producer_bytes);

}   // namespace prosper

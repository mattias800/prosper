#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace prosper::gpu {
// Observations only: none of these fields grant source, compiler or dispatch authority.
enum class RawSnapshotGate : uint8_t {
    None,
    LeaseUnavailable,
    EntryUnavailable,
    ContextUnavailable,
    ProducerIncomplete,
    WidthConflict,
    ProbeShape,
    ObservationChanged,
    ParentMissing,
    PointerMismatch,
    FaultUnsafe,
    AllocationUnknown,
    Unreadable,
    CurrentSourceUnproved,
    OutputNotDisjoint,
    MissingObservation,
    WritePlanIncomplete,
    DescriptorMissingOrAmbiguous,
    DescriptorSourceInvalid,
    DescriptorFaultUnsafe,
    DescriptorUnreadable,
    DescriptorCurrentUnproved,
    DescriptorChanged,
    OutputMissing,
    OutputExtentUnproved,
    OutputFaultUnsafe,
    PhysicalNotDisjoint,
    ResourceConflict,
};
struct RawSnapshotProducerObservation {
    bool available = false;
    bool deferred = false, producer_epoch_ok = false, graphics_epoch_ok = false;
    bool indirect_dependencies_ok = false, initial_known = false, completed_known = false;
    bool completed_pending = false;
    uint64_t initial_failures = 0, completed_failures = 0;
};
struct RawSnapshotLoadObservation {
    uint32_t pc = UINT32_MAX, bytes = 0;
    uint64_t address = 0;
    std::array<uint32_t, 2> words{};
};
struct RawSnapshotDiagnostic {
    RawSnapshotGate first_refusal = RawSnapshotGate::None;
    uint32_t refusal_pc = UINT32_MAX, refusal_bytes = 0;
    uint64_t refusal_address = 0, other_address = 0;
    uint32_t required_user_words = 0, actual_user_words = 0, chain_count = 0;
    uint32_t first_parent_pc = UINT32_MAX, first_child_pc = UINT32_MAX; // code facts, not reads
    bool ordered_lease_available = false, context_supplied = false, published = false;
    RawSnapshotProducerObservation producer;
    std::array<RawSnapshotLoadObservation, 8> loads{};
    uint32_t load_count = 0;
    bool loads_truncated = false;
    void refuse(RawSnapshotGate, uint32_t pc = UINT32_MAX, uint64_t address = 0, uint32_t bytes = 0,
                uint64_t other = 0);
    // The caller supplies existing owner bytes, never a guest pointer to reread.
    void observed(uint32_t pc, uint64_t address, std::span<const uint8_t> owner);
};
bool raw_snapshot_diagnostic_enabled();
std::string format_raw_snapshot_diagnostic(const RawSnapshotDiagnostic&);
// Hard 64-record process cap and per-thread duplicate suppression. No global lock, guest
// reads or admission changes; source words are the executor's already-owned decode-cache copy.
void log_raw_snapshot_diagnostic(uint64_t program, uint64_t submit, uint64_t order,
                                 std::span<const uint32_t> original,
                                 const RawSnapshotDiagnostic&) noexcept;
} // namespace prosper::gpu

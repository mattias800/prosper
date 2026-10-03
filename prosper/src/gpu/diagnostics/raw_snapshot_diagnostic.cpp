// Name the first real owned-snapshot refusal without retrying a fold or reading guest memory.
#include "gpu/diagnostics/raw_snapshot_diagnostic.hpp"
#include "diagnostics/env_submit.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>

namespace prosper::gpu {
void RawSnapshotDiagnostic::refuse(RawSnapshotGate gate, uint32_t pc, uint64_t address,
                                   uint32_t bytes, uint64_t other) {
    if (first_refusal != RawSnapshotGate::None) return;
    first_refusal = gate;
    refusal_pc = pc;
    refusal_address = address;
    refusal_bytes = bytes;
    other_address = other;
}
void RawSnapshotDiagnostic::observed(uint32_t pc, uint64_t address,
                                     std::span<const uint8_t> owner) {
    if (load_count >= loads.size()) {
        loads_truncated = true;
        return;
    }
    auto& load = loads[load_count++];
    load.pc = pc;
    load.address = address;
    load.bytes = static_cast<uint32_t>(owner.size());
    std::memcpy(load.words.data(), owner.data(), std::min(owner.size(), sizeof(load.words)));
}
bool raw_snapshot_diagnostic_enabled() {
    const bool enabled = PROSPER_ENV_ON_PER_SUBMIT("PROSPER_RAW_SNAPSHOT_DIAG");
    if (enabled) {
        static std::atomic<bool> announced{false};
        if (!announced.exchange(true, std::memory_order_relaxed))
            std::fprintf(stderr, "[raw-snapshot-diag] armed; limit=64 records, owned loads<=8; "
                                 "absence is not admission evidence\n");
    }
    return enabled;
}
namespace {
const char* gate_name(RawSnapshotGate gate) {
#define GATE(name)                                                                                 \
    case RawSnapshotGate::name: return #name
    switch (gate) {
        GATE(None);
        GATE(LeaseUnavailable);
        GATE(EntryUnavailable);
        GATE(ContextUnavailable);
        GATE(ProducerIncomplete);
        GATE(WidthConflict);
        GATE(ProbeShape);
        GATE(ObservationChanged);
        GATE(ParentMissing);
        GATE(PointerMismatch);
        GATE(FaultUnsafe);
        GATE(AllocationUnknown);
        GATE(Unreadable);
        GATE(CurrentSourceUnproved);
        GATE(OutputNotDisjoint);
        GATE(MissingObservation);
        GATE(WritePlanIncomplete);
        GATE(DescriptorMissingOrAmbiguous);
        GATE(DescriptorSourceInvalid);
        GATE(DescriptorFaultUnsafe);
        GATE(DescriptorUnreadable);
        GATE(DescriptorCurrentUnproved);
        GATE(DescriptorChanged);
        GATE(OutputMissing);
        GATE(OutputExtentUnproved);
        GATE(OutputFaultUnsafe);
        GATE(PhysicalNotDisjoint);
        GATE(ResourceConflict);
    }
#undef GATE
    return "Invalid";
}
} // namespace
std::string format_raw_snapshot_diagnostic(const RawSnapshotDiagnostic& d) {
    std::ostringstream out;
    out << "first=" << gate_name(d.first_refusal) << " pc=";
    if (d.refusal_pc == UINT32_MAX)
        out << "unavailable";
    else
        out << d.refusal_pc;
    out << " address=0x" << std::hex << d.refusal_address << " other=0x" << d.other_address
        << std::dec << " bytes=" << d.refusal_bytes << " snapshot-published=" << d.published
        << " entry=" << d.actual_user_words << '/' << d.required_user_words
        << " lease=" << d.ordered_lease_available << " context=" << d.context_supplied
        << " chains=" << d.chain_count << " first-code-chain=" << d.first_parent_pc << '/'
        << d.first_child_pc << " producer-observed=" << d.producer.available;
    if (d.producer.available)
        out << " deferred=" << d.producer.deferred << " producer=" << d.producer.producer_epoch_ok
            << " graphics=" << d.producer.graphics_epoch_ok
            << " indirect=" << d.producer.indirect_dependencies_ok
            << " known=" << d.producer.initial_known << '/' << d.producer.completed_known
            << " pending=" << d.producer.completed_pending
            << " generations=" << d.producer.initial_failures << '/'
            << d.producer.completed_failures;
    out << " loads=" << d.load_count << " truncated=" << d.loads_truncated;
    for (uint32_t n = 0; n < std::min<size_t>(d.load_count, d.loads.size()); ++n) {
        const auto& load = d.loads[n];
        out << " [pc=" << load.pc << " address=0x" << std::hex << load.address << std::dec
            << " bytes=" << load.bytes << " owned-words=" << std::hex << load.words[0] << ','
            << load.words[1] << std::dec << ']';
    }
    return out.str();
}
void log_raw_snapshot_diagnostic(uint64_t program, uint64_t submit, uint64_t order,
                                 std::span<const uint32_t> original,
                                 const RawSnapshotDiagnostic& d) noexcept {
    try {
        if (!raw_snapshot_diagnostic_enabled()) return;
        static std::atomic<uint32_t> emitted{0};
        if (emitted.load(std::memory_order_relaxed) >= 64u) return;
        uint64_t hash = 14695981039346656037ull;
        for (uint32_t word : original)
            for (uint32_t byte = 0; byte < 4u; ++byte)
                hash = (hash ^ ((word >> (byte * 8u)) & 0xffu)) * 1099511628211ull;
        struct Key {
            uint64_t hash;
            RawSnapshotGate gate;
            uint32_t pc;
            bool published;
        };
        static thread_local std::array<Key, 64> seen{};
        static thread_local uint32_t count = 0;
        for (uint32_t n = 0; n < count; ++n)
            if (seen[n].hash == hash && seen[n].gate == d.first_refusal &&
                seen[n].pc == d.refusal_pc && seen[n].published == d.published)
                return;
        uint32_t ticket = emitted.load(std::memory_order_relaxed);
        while (ticket < 64u &&
               !emitted.compare_exchange_weak(ticket, ticket + 1u, std::memory_order_relaxed)) {}
        if (ticket >= 64u) return;
        seen[count++] = {hash, d.first_refusal, d.refusal_pc, d.published};
        const auto text = format_raw_snapshot_diagnostic(d);
        std::fprintf(
            stderr,
            "[raw-snapshot-diag] program=0x%llx submit=%llu order=%llu decoded-source-fnv64=%llx "
            "dwords=%zu %s\n",
            static_cast<unsigned long long>(program), static_cast<unsigned long long>(submit),
            static_cast<unsigned long long>(order), static_cast<unsigned long long>(hash),
            original.size(), text.c_str());
    } catch (const std::exception&) {
        std::fprintf(stderr, "[raw-snapshot-diag] record unavailable; admission unchanged\n");
    }
}
} // namespace prosper::gpu

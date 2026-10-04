#pragma once

#include "gpu/recompiler/fragment_packet_scalar_reads.hpp"
#include "gpu/state/fragment_entry_facts.hpp"
#include "host/memory/guest_direct_allocation.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper {
class GuestMappingLease;
}
namespace prosper::gpu {
struct FragmentPacketVgprRequirements;
class OrderedScalarBankReadPoint;

// The original V# domain, the actual allocation domain, and copied bytes are deliberately
// different. A short demanded interval never becomes a rebased/truncated descriptor.
struct FragmentScalarBankSource {
    std::array<uint32_t, 4> descriptor{};
    uint64_t guest_base = 0, declared_bytes = 0;
    prosper::GuestDirectAllocation
        allocation; // full original physical extent and allocation birth
};
struct FragmentScalarBankInterval {
    uint32_t source_index = 0, byte_offset = 0;
    uint64_t guest_begin = 0, physical_begin = 0, physical_end = 0;
    std::shared_ptr<const std::vector<uint8_t>>
        bytes; // exact checked interval, not a padded tail
};
struct FragmentScalarBankSite {
    uint32_t pc = 0, opcode = 0, words = 0, byte_offset = 0;
    uint32_t source_index = 0, interval_index = 0, interval_byte_offset = 0;
};

// Only the real ordered producer can seal this owner. Permission is checked while copying and
// publishing it, not again at later GPU upload: an expired issuer cannot revoke already-owned
// immutable submitted bytes. This does NOT exclude arbitrary concurrent guest CPU byte writers.
// Existing submitted-input lifetime/stability remains a separate required contract.
class FragmentScalarBank {
public:
    const auto& sources() const { return sources_; }
    const auto& intervals() const { return intervals_; }
    const auto& sites() const { return sites_; }
    const auto& original_words() const { return original_words_; }
    const auto& requirements() const { return requirements_; }
    const auto& packet_requirements() const { return packet_requirements_; }
    const FragmentEntryFacts& entry() const { return entry_; }
    uint64_t read_point_identity() const { return read_point_identity_; }
    uint64_t payload_bytes() const { return payload_bytes_; }
    // Header/site metadata only. Normal upload copies each immutable interval directly into
    // its absolute shared-plane offset once, without concatenating/copying payload per wave.
    const auto& wire_metadata() const { return wire_metadata_; }
    const auto& wire_interval_words() const { return wire_interval_words_; }
    uint32_t wire_words() const { return wire_words_; }
    bool matches(const std::shared_ptr<const std::vector<uint32_t>>&,
                 const std::shared_ptr<const FragmentPacketVgprRequirements>&,
                 const FragmentEntryFacts&) const;

private:
    friend std::shared_ptr<const FragmentScalarBank>
    seal_fragment_scalar_bank(const OrderedScalarBankReadPoint&, uint64_t, uint64_t,
                              const prosper::GuestMappingLease&, std::string&);
    FragmentScalarBank() = default;
    std::shared_ptr<const std::vector<uint32_t>> original_words_;
    std::shared_ptr<const FragmentPacketScalarReadRequirements> requirements_;
    std::shared_ptr<const FragmentPacketVgprRequirements> packet_requirements_;
    FragmentEntryFacts entry_;
    uint64_t source_address_ = 0, source_submit_ = 0, command_order_ = 0;
    uint64_t read_point_identity_ = 0, payload_bytes_ = 0;
    std::vector<FragmentScalarBankSource> sources_;
    std::vector<FragmentScalarBankInterval> intervals_;
    std::vector<FragmentScalarBankSite> sites_;
    std::vector<uint32_t> wire_metadata_, wire_interval_words_;
    uint32_t wire_words_ = 0;
};

// Checked raw S_BUFFER x1/x2/x4, constant aligned byte offsets, canonical genuine user-prefix
// descriptor origins. Other memory/image/write forms, partial OOB and unbound inputs refuse.
// The mapping/fault/readability proof covers copied intervals; retained-producer/output alias
// exclusion covers the FULL allocation, including unused tails. No descriptor-size ISA cap.
std::shared_ptr<const FragmentScalarBank>
seal_fragment_scalar_bank(const OrderedScalarBankReadPoint&, uint64_t source_address,
                          uint64_t command_order, const prosper::GuestMappingLease&,
                          std::string& refusal);
} // namespace prosper::gpu

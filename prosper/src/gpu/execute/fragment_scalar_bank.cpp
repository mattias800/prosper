#include "gpu/execute/fragment_scalar_bank.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/execute/scalar_bank_read_point.hpp"
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"
#include "gpu/recompiler/fragment_scalar_bank_wire.hpp"
#include "gpu/resources/buffer_source_read.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "host/memory/guest_memory_topology.hpp"
#include <algorithm>
#include <utility>

namespace prosper::gpu {
namespace {
GraphicsRawAllocationAuthorityFn allocation_authority;
bool same_allocation(const prosper::GuestDirectAllocation& a,
                     const prosper::GuestDirectAllocation& b) {
    return a.identity && a.identity == b.identity && a.physical_begin == b.physical_begin &&
           a.physical_end == b.physical_end && a.physical_begin < a.physical_end;
}
bool disjoint(const prosper::GuestDirectAllocation& a, const prosper::GuestDirectAllocation& b) {
    return a.identity && b.identity && a.physical_begin < a.physical_end &&
           b.physical_begin < b.physical_end &&
           (a.physical_end <= b.physical_begin || b.physical_end <= a.physical_begin);
}
} // namespace
void set_graphics_raw_allocation_authority(GraphicsRawAllocationAuthorityFn fn) {
    allocation_authority = std::move(fn);
}
bool graphics_raw_allocation_is_guest_current(const prosper::GuestMappingLease& lease,
                                              const prosper::GuestDirectAllocation& source) {
    return allocation_authority && allocation_authority(lease, source);
}
bool FragmentScalarBank::matches(
    const std::shared_ptr<const std::vector<uint32_t>>& words,
    const std::shared_ptr<const FragmentPacketVgprRequirements>& requirements,
    const FragmentEntryFacts& entry) const {
    return words && requirements && words.get() == original_words_.get() &&
           !words.owner_before(original_words_) && !original_words_.owner_before(words) &&
           requirements.get() == packet_requirements_.get() &&
           !requirements.owner_before(packet_requirements_) &&
           !packet_requirements_.owner_before(requirements) && entry == entry_;
}

std::shared_ptr<const FragmentScalarBank>
seal_fragment_scalar_bank(const OrderedScalarBankReadPoint& point, uint64_t address, uint64_t order,
                          const prosper::GuestMappingLease& lease, std::string& refusal) {
    refusal.clear();
    const auto reject = [&](const char* why) -> std::shared_ptr<const FragmentScalarBank> {
        refusal = why;
        return {};
    };
    const auto words = point.packet_source(address);
    const auto requirements = point.packet_requirements(address);
    const auto& entry =
        point.entry(); // bound by the actual issuer, not a caller's canonical facts
    const auto permission = [&] {
        return point.valid_for_packet(point.source_submit(), order, address, words, requirements);
    };
    if (!permission()) return reject("scalar-bank-original-permission-unavailable");
    if (!requirements->rejection.empty() || !requirements->masks.rejection.empty() ||
        !requirements->scalar_reads.rejection.empty() || !requirements->scalar_reads.has_smem ||
        requirements->scalar_reads.sites.empty())
        return reject("scalar-bank-original-program-requirements-unproved");
    if (!entry.observed || !entry.canonical() || !entry.rsrc2_available)
        return reject("scalar-bank-canonical-user-prefix-unavailable");
    namespace P = prosper::agc::Pm4;
    const uint32_t user_count = ((entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
                                 P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
                                (((entry.rsrc2 >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
                                  P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK)
                                 << 5);
    if (user_count > entry.user_data.size()) return reject("scalar-bank-user-prefix-count-invalid");
    // Site/payload budgets are bank allocation policy, not a descriptor-size or scalar ISA cap.
    constexpr uint32_t max_sites = 64;
    if (requirements->scalar_reads.sites.size() > max_sites)
        return reject("scalar-bank-site-allocation-budget");
    auto result = std::shared_ptr<FragmentScalarBank>(new FragmentScalarBank);
    result->original_words_ = words;
    result->packet_requirements_ = requirements;
    result->entry_ = entry;
    result->requirements_ = std::shared_ptr<const FragmentPacketScalarReadRequirements>(
        requirements, &requirements->scalar_reads);
    result->source_address_ = address;
    result->source_submit_ = point.source_submit();
    result->command_order_ = order;
    result->read_point_identity_ = point.identity();
    struct Demand {
        uint32_t source;
        uint64_t begin, end;
    };
    std::vector<Demand> demands;
    for (const auto& site : requirements->scalar_reads.sites) {
        std::array<uint32_t, 4> descriptor;
        for (uint32_t word = 0; word < descriptor.size(); ++word) {
            const auto reg = site.entry_words[word];
            if (reg >= user_count || !(entry.user_data_available & (uint32_t(1) << reg)))
                return reject("scalar-bank-original-descriptor-word-unavailable");
            descriptor[word] = entry.user_data[reg];
        }
        // RDNA2 scalar V# uses the 48-bit base, 14-bit STRIDE and NUM_RECORDS, not formatted
        // vector fields. Full raw four-word identity still matters at the actual original PC.
        // STRIDE affects byte bounds only; #2528 / RDNA3 8.1.2 corrects RDNA2's missing product.
        const uint64_t base =
            (uint64_t(descriptor[0]) | (uint64_t(descriptor[1] & 0xffffu) << 32)) & ~uint64_t(3);
        const uint64_t stride = (descriptor[1] >> 16) & 0x3fffu;
        const uint64_t declared = (stride ? stride : 1u) * uint64_t(descriptor[2]);
        if (!base || (descriptor[1] & 0x80000000u))
            return reject("scalar-bank-null-or-swizzled-domain-unimplemented");
        if (site.words != 1u && site.words != 2u && site.words != 4u)
            return reject("scalar-bank-load-width-unimplemented");
        const uint32_t bytes = site.words * sizeof(uint32_t);
        if (!declared || site.byte_offset > declared || bytes > declared - site.byte_offset)
            return reject("scalar-bank-partial-or-complete-oob-unimplemented");
        if (base > UINT64_MAX - site.byte_offset || base + site.byte_offset > UINT64_MAX - bytes)
            return reject("scalar-bank-source-address-overflow");
        const uint64_t begin = base + site.byte_offset;
        const auto window = prosper::guest_memory_direct_readable_window(lease, begin);
        if (!window || begin < window.virtual_begin || bytes > window.virtual_end - begin ||
            !prosper::guest_memory_direct_range_fault_safe(lease, begin, bytes))
            return reject("scalar-bank-demand-fault-or-mapping-unproved");
        auto selected =
            std::find_if(result->sources_.begin(), result->sources_.end(),
                         [&](const auto& source) { return source.descriptor == descriptor; });
        if (selected == result->sources_.end()) {
            result->sources_.push_back({descriptor, base, declared, window.allocation});
            selected = std::prev(result->sources_.end());
        } else if (!same_allocation(selected->allocation, window.allocation)) {
            return reject("scalar-bank-one-descriptor-crosses-original-allocations");
        }
        const uint32_t index = static_cast<uint32_t>(selected - result->sources_.begin());
        demands.push_back({index, site.byte_offset, uint64_t(site.byte_offset) + bytes});
        result->sites_.push_back({site.pc, site.opcode, site.words, site.byte_offset, index});
    }
    std::sort(demands.begin(), demands.end(), [](const auto& a, const auto& b) {
        return std::pair{a.source, a.begin} < std::pair{b.source, b.begin};
    });
    std::vector<Demand> merged;
    for (const auto& demand : demands) {
        if (!merged.empty() && merged.back().source == demand.source &&
            demand.begin <= merged.back().end)
            merged.back().end = std::max(merged.back().end, demand.end);
        else
            merged.push_back(demand);
    }
    const auto source_current = [&] {
        if (!permission()) return false;
        for (const auto& source : result->sources_) {
            const auto current = prosper::guest_memory_direct_allocation(
                lease, source.allocation.address, source.allocation.minimum_bytes);
            if (!same_allocation(source.allocation, current) ||
                !graphics_raw_allocation_is_guest_current(lease, source.allocation))
                return false;
            for (const auto& target : point.output_allocations())
                if (!disjoint(source.allocation, target)) return false;
        }
        return permission();
    };
    if (!source_current()) return reject("scalar-bank-full-allocation-authority-unproved");
    // Check ALL source scopes before the first read. Only demanded intervals are read; unused
    // declared tails are neither probed/copied nor silently converted into zero backing.
    for (const auto& demand : merged) {
        const auto& source = result->sources_[demand.source];
        const auto begin = source.guest_base + demand.begin;
        const uint32_t bytes = static_cast<uint32_t>(demand.end - demand.begin);
        const auto window = prosper::guest_memory_direct_readable_window(lease, begin);
        if (!window || !same_allocation(source.allocation, window.allocation) ||
            begin < window.virtual_begin || bytes > window.virtual_end - begin ||
            !prosper::guest_memory_direct_range_fault_safe(lease, begin, bytes) ||
            !guest_readable(begin, bytes))
            return reject("scalar-bank-merged-demand-readability-unproved");
    }
    for (const auto& demand : merged) {
        if (!permission()) return reject("scalar-bank-copy-permission-expired");
        const auto& source = result->sources_[demand.source];
        const uint64_t begin = source.guest_base + demand.begin;
        const uint32_t bytes = static_cast<uint32_t>(demand.end - demand.begin);
        const auto window = prosper::guest_memory_direct_readable_window(lease, begin);
        if (!window || !same_allocation(source.allocation, window.allocation) ||
            begin < window.virtual_begin || bytes > window.virtual_end - begin ||
            !prosper::guest_memory_direct_range_fault_safe(lease, begin, bytes))
            return reject("scalar-bank-copy-window-permission-unproved");
        auto owner = std::make_shared<std::vector<uint8_t>>(bytes);
        const auto copied =
            read_buffer_source(owner->data(), begin, bytes, [&](uint64_t current, size_t length) {
                return length <= UINT32_MAX && guest_readable(current, uint32_t(length))
                           ? length
                           : size_t(0);
            });
        if (!copied.complete() || !permission())
            return reject("scalar-bank-demand-copy-incomplete-or-expired");
        const uint32_t index = static_cast<uint32_t>(result->intervals_.size());
        for (auto& site : result->sites_)
            if (site.source_index == demand.source && site.byte_offset >= demand.begin &&
                uint64_t(site.byte_offset) + site.words * 4u <= demand.end) {
                site.interval_index = index;
                site.interval_byte_offset = site.byte_offset - demand.begin;
            }
        const uint64_t physical = window.physical_begin + begin - window.virtual_begin;
        result->intervals_.push_back({demand.source, static_cast<uint32_t>(demand.begin), begin,
                                      physical, physical + bytes, std::move(owner)});
        result->payload_bytes_ += bytes;
    }
    const uint32_t payload_begin = kFragmentScalarBankHeaderWords +
                                   uint32_t(result->sites_.size()) * kFragmentScalarBankSiteWords;
    result->wire_metadata_.resize(payload_begin, 0);
    uint64_t total_words = payload_begin;
    for (const auto& interval : result->intervals_) {
        if (!interval.bytes || interval.bytes->empty() || interval.bytes->size() % 4 ||
            total_words > UINT32_MAX || interval.bytes->size() / 4 > UINT32_MAX - total_words)
            return reject("scalar-bank-upload-layout-invalid");
        result->wire_interval_words_.push_back(uint32_t(total_words));
        total_words += interval.bytes->size() / 4;
    }
    result->wire_words_ = uint32_t(total_words);
    auto& metadata = result->wire_metadata_;
    metadata[ScalarBankMagic] = kFragmentScalarBankMagic;
    metadata[ScalarBankVersion] = kFragmentScalarBankVersion;
    metadata[ScalarBankTotalWords] = result->wire_words_;
    metadata[ScalarBankSiteCount] = uint32_t(result->sites_.size());
    metadata[ScalarBankPayloadBegin] = payload_begin;
    metadata[ScalarBankPayloadWords] = result->wire_words_ - payload_begin;
    for (uint32_t index = 0; index < result->sites_.size(); ++index) {
        const auto& site = result->sites_[index];
        if (site.source_index >= result->sources_.size() ||
            site.interval_index >= result->intervals_.size())
            return reject("scalar-bank-upload-site-association-invalid");
        const auto& source = result->sources_[site.source_index];
        const auto& interval = result->intervals_[site.interval_index];
        if (interval.source_index != site.source_index ||
            site.byte_offset != uint64_t(interval.byte_offset) + site.interval_byte_offset ||
            site.interval_byte_offset > interval.bytes->size() ||
            uint64_t(site.words) * 4 > interval.bytes->size() - site.interval_byte_offset)
            return reject("scalar-bank-upload-site-extent-invalid");
        const uint32_t row = kFragmentScalarBankHeaderWords + index * kFragmentScalarBankSiteWords;
        metadata[row + ScalarBankPc] = site.pc;
        metadata[row + ScalarBankOpcode] = site.opcode;
        metadata[row + ScalarBankLoadWords] = site.words;
        metadata[row + ScalarBankByteOffset] = site.byte_offset;
        metadata[row + ScalarBankDeclaredLo] = uint32_t(source.declared_bytes);
        metadata[row + ScalarBankDeclaredHi] = uint32_t(source.declared_bytes >> 32);
        for (uint32_t word = 0; word < source.descriptor.size(); ++word)
            metadata[row + ScalarBankDescriptor0 + word] = source.descriptor[word];
        metadata[row + ScalarBankIntervalBase] = result->wire_interval_words_[site.interval_index];
        metadata[row + ScalarBankIntervalWords] = uint32_t(interval.bytes->size() / 4);
        metadata[row + ScalarBankIntervalByteOffset] = site.interval_byte_offset;
    }
    if (!source_current()) return reject("scalar-bank-final-seal-permission-expired");
    return result;
}
}   // namespace prosper::gpu

#pragma once
// Read-only queries over a sorted guest mapping table (readable prefix, physical-alias topology).
// Templates over the mapping record so the Linux and Windows tables in hle_kernel_mem.cpp share one
// definition. The caller of each owns the table; these only lock the mutex they are handed.

#include "host/memory/guest_memory_topology.hpp"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <vector>

namespace prosper {

template <typename Mapping>
bool mapped_memory_slices(const std::vector<Mapping>& maps, std::mutex& mapping_mutex,
                          uint32_t direct_flag, uint64_t address, uint64_t bytes,
                          std::vector<GuestMemoryMappingSlice>& output,
                          GuestMemoryMappingCoverage* coverage = nullptr) {
    output.clear();
    if (coverage) *coverage = GuestMemoryMappingCoverage::Incomplete;
    if (!address || !bytes || address > UINT64_MAX - bytes) return false;
    std::lock_guard<std::mutex> lock(mapping_mutex);
    const auto after = std::upper_bound(
        maps.begin(), maps.end(), address,
        [](uint64_t value, const Mapping& mapping) { return value < mapping.base; });
    auto mapping = after;
    if (after != maps.begin()) {
        const auto previous = std::prev(after);
        if (previous->size > UINT64_MAX - previous->base ||
            address - previous->base < previous->size)
            mapping = previous;
    }
    if (mapping == maps.end() || mapping->base >= address + bytes) {
        if (coverage) *coverage = GuestMemoryMappingCoverage::Untracked;
        return false;
    }
    std::vector<GuestMemoryMappingSlice> slices;
    uint64_t cursor = address;
    while (cursor < address + bytes) {
        if (mapping == maps.end() || !mapping->committed || mapping->base > cursor ||
            mapping->size > UINT64_MAX - mapping->base || cursor - mapping->base >= mapping->size)
            return false;
        const uint64_t count = std::min(address + bytes, mapping->base + mapping->size) - cursor;
        const bool direct = (mapping->query_flags & direct_flag) != 0;
        const uint64_t delta = cursor - mapping->base;
        if (direct &&
            (mapping->offset > UINT64_MAX - delta || mapping->offset + delta > UINT64_MAX - count))
            return false;
        const uint64_t physical = direct ? mapping->offset + delta : 0;
        if (!slices.empty() && slices.back().direct == direct &&
            slices.back().offset + slices.back().bytes == cursor - address &&
            (!direct || slices.back().physical + slices.back().bytes == physical))
            slices.back().bytes += count;
        else
            slices.push_back({cursor - address, count, physical, direct});
        cursor += count;
        ++mapping;
    }
    output = std::move(slices);
    if (coverage) *coverage = GuestMemoryMappingCoverage::Complete;
    return true;
}

// Renderer reads must stop at the first reserved or unreadable guest byte. Mapping records can
// split at a 16 KiB guest page, so sampling one address per 64 KiB can silently cross a hole.
// Adjacent committed records remain one readable prefix even when protection split them.
template <typename Mapping>
uint64_t mapped_readable_prefix(const std::vector<Mapping>& maps, std::mutex& mapping_mutex,
                                uint64_t address, uint64_t bytes, int read_protection) {
    if (address < 0x1000 || !bytes || address > UINT64_MAX - bytes) return 0;
    std::lock_guard<std::mutex> lock(mapping_mutex);
    const auto after = std::upper_bound(
        maps.begin(), maps.end(), address,
        [](uint64_t value, const Mapping& mapping) { return value < mapping.base; });
    auto mapping = after == maps.begin() ? maps.end() : std::prev(after);
    uint64_t cursor = address;
    const uint64_t end = address + bytes;
    while (cursor < end && mapping != maps.end()) {
        if (mapping->base > cursor || cursor - mapping->base >= mapping->size ||
            mapping->size > UINT64_MAX - mapping->base || !mapping->committed ||
            !(mapping->prot & read_protection)) break;
        cursor = std::min(end, mapping->base + mapping->size);
        ++mapping;
    }
    return cursor - address;
}

// A protection change or a partial remap splits one guest allocation into several mapping
// records. Physical-alias queries must examine every covered segment: the first record's extent
// and physical offset say nothing about its neighbors. The caller holds the mapping-table lock.
template <typename Mapping>
GuestMemoryTopologyRelation mapped_topology_relation(
        const std::vector<Mapping>& maps, uint32_t direct_flag,
        uint64_t first_address, uint64_t first_end,
        uint64_t second_address, uint64_t second_end) {
    const auto at_or_before = [&](uint64_t address) {
        auto after = std::upper_bound(
            maps.begin(), maps.end(), address,
            [](uint64_t value, const Mapping& mapping) { return value < mapping.base; });
        return after == maps.begin() ? maps.end() : std::prev(after);
    };
    const auto covered = [&](auto it, uint64_t address, uint64_t end) {
        while (address < end) {
            if (it == maps.end() || !it->committed || it->base > address ||
                it->size > UINT64_MAX - it->base ||
                address - it->base >= it->size) return false;
            address = std::min(end, it->base + it->size);
            ++it;
        }
        return true;
    };
    const auto first_begin = at_or_before(first_address);
    const auto second_begin = at_or_before(second_address);
    if (!covered(first_begin, first_address, first_end) ||
        !covered(second_begin, second_address, second_end))
        return GuestMemoryTopologyRelation::Unknown;

    for (auto first = first_begin;
         first != maps.end() && first->base < first_end; ++first) {
        if (!(first->query_flags & direct_flag)) continue;
        const uint64_t first_start = std::max(first_address, first->base);
        const uint64_t first_stop = std::min(first_end, first->base + first->size);
        const uint64_t first_delta = first_start - first->base;
        const uint64_t first_bytes = first_stop - first_start;
        if (first->offset > UINT64_MAX - first_delta)
            return GuestMemoryTopologyRelation::Unknown;
        const uint64_t first_physical = first->offset + first_delta;
        if (first_physical > UINT64_MAX - first_bytes)
            return GuestMemoryTopologyRelation::Unknown;
        for (auto second = second_begin;
             second != maps.end() && second->base < second_end; ++second) {
            if (!(second->query_flags & direct_flag)) continue;
            const uint64_t second_start = std::max(second_address, second->base);
            const uint64_t second_stop = std::min(second_end, second->base + second->size);
            const uint64_t second_delta = second_start - second->base;
            const uint64_t second_bytes = second_stop - second_start;
            if (second->offset > UINT64_MAX - second_delta)
                return GuestMemoryTopologyRelation::Unknown;
            const uint64_t second_physical = second->offset + second_delta;
            if (second_physical > UINT64_MAX - second_bytes)
                return GuestMemoryTopologyRelation::Unknown;
            if (first_physical < second_physical + second_bytes &&
                second_physical < first_physical + first_bytes)
                return GuestMemoryTopologyRelation::Overlap;
        }
    }
    return GuestMemoryTopologyRelation::Disjoint;
}

template <typename Mapping>
GuestMemoryTopologyRelation checked_topology_relation(
        const std::vector<Mapping>& maps, std::mutex& mapping_mutex, uint32_t direct_flag,
        uint64_t first_address, uint64_t first_size,
        uint64_t second_address, uint64_t second_size) {
    if (!first_address || !first_size || first_address > UINT64_MAX - first_size ||
        !second_address || !second_size || second_address > UINT64_MAX - second_size)
        return GuestMemoryTopologyRelation::Unknown;
    const uint64_t first_end = first_address + first_size;
    const uint64_t second_end = second_address + second_size;
    if (first_address < second_end && second_address < first_end)
        return GuestMemoryTopologyRelation::Overlap;

    std::lock_guard<std::mutex> lock(mapping_mutex);
    return mapped_topology_relation(maps, direct_flag, first_address, first_end,
                                    second_address, second_end);
}

} // namespace prosper

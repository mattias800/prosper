#include "rdna2_indirect_buffer_shadow.hpp"

#include "shader_resources.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace prosper::gpu {

bool guest_readable(uint64_t address, uint32_t bytes);

namespace {

uint32_t load_u32(const uint8_t* bytes, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

uint64_t load_u64(const uint8_t* bytes, size_t offset) {
    uint64_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

void store_u32(uint8_t* bytes, size_t offset, uint32_t value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}

void store_u64(uint8_t* bytes, size_t offset, uint64_t value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}

constexpr size_t kRelocationHeaderBytes = 40u;
constexpr size_t kRelocationRecordBytes = 24u;
constexpr size_t kRelocationSegmentBytes = 24u;

bool relocation_layout_valid(const ShaderResource& source,
                             const IndirectBufferRelocationLayout& layout,
                             size_t record_count) {
    return layout.tag && layout.version == 2u && layout.max_records &&
           layout.max_segments && layout.max_binding_bytes &&
           record_count && record_count <= layout.max_records &&
           source.size && source.size <= UINT32_MAX;
}

bool interval_end(uint64_t address, uint32_t bytes, uint64_t& end) {
    if (!address || !bytes || bytes > UINT64_MAX - address) return false;
    end = address + bytes;
    return true;
}

struct RelocationInterval { uint64_t begin, end; };

bool canonical_relocation_intervals(
        std::span<const IndirectBufferRelocationRecord> records,
        std::vector<RelocationInterval>& merged) {
    std::vector<RelocationInterval> intervals;
    intervals.reserve(records.size());
    for (const auto& record : records) {
        if (!record.guest_address || !record.byte_count) {
            if (record.guest_address || record.byte_count) return false;
            continue;
        }
        uint64_t end = 0;
        if (!interval_end(record.guest_address, record.byte_count, end)) return false;
        intervals.push_back({record.guest_address, end});
    }
    std::sort(intervals.begin(), intervals.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.begin < rhs.begin || (lhs.begin == rhs.begin && lhs.end < rhs.end);
    });
    merged.clear();
    for (const auto& interval : intervals) {
        if (merged.empty() || interval.begin > merged.back().end)
            merged.push_back(interval);
        else
            merged.back().end = std::max(merged.back().end, interval.end);
    }
    return true;
}

bool record_offsets_fit(const ShaderResource& source,
                        const IndirectBufferShadowLayout& layout,
                        std::span<const uint32_t> pointer_records) {
    if (!layout.tag || !layout.version || !layout.source_stride || !layout.slot_bytes ||
        pointer_records.empty() || pointer_records.size() > layout.max_pointer_records ||
        source.size > UINT32_MAX)
        return false;
    uint64_t prior_end = 0;
    for (const uint32_t record : pointer_records) {
        const uint64_t offset = static_cast<uint64_t>(record) * layout.source_stride;
        if (offset < prior_end || offset > source.size ||
            sizeof(uint64_t) > source.size - offset)
            return false;
        prior_end = offset + sizeof(uint64_t);
    }
    return true;
}

size_t record_offset(const IndirectBufferShadowLayout& layout, uint32_t record) {
    return static_cast<size_t>(record) * layout.source_stride;
}

bool table_owns(const ShaderResourceTable& table, const uint8_t* data) {
    return std::any_of(table.owned_host_data.begin(), table.owned_host_data.end(),
                       [&](const auto& owner) { return owner && owner->data() == data; });
}

} // namespace

size_t indirect_buffer_shadow_header_bytes(const IndirectBufferShadowLayout& layout) {
    constexpr size_t fixed_bytes = 16u;
    const size_t pointer_bytes =
        static_cast<size_t>(layout.max_pointer_records) * sizeof(uint64_t);
    const size_t witness_bytes =
        static_cast<size_t>(layout.witness_word_count) * sizeof(uint32_t);
    if (pointer_bytes > std::numeric_limits<size_t>::max() - fixed_bytes ||
        witness_bytes > std::numeric_limits<size_t>::max() - fixed_bytes - pointer_bytes)
        return 0u;
    return fixed_bytes + pointer_bytes + witness_bytes;
}

bool parse_indirect_buffer_shadow(
        const ShaderResource& source, const uint8_t* bytes, size_t byte_count,
        const IndirectBufferShadowLayout& layout, std::span<const uint32_t> pointer_records,
        IndirectBufferShadowInfo& info) {
    info = {};
    if (!bytes || !record_offsets_fit(source, layout, pointer_records)) return false;
    const size_t source_bytes = static_cast<size_t>(source.size);
    const size_t header_bytes = indirect_buffer_shadow_header_bytes(layout);
    if (!header_bytes || source_bytes > std::numeric_limits<size_t>::max() - header_bytes)
        return false;
    const size_t slot_base = source_bytes + header_bytes;
    if (byte_count < slot_base || load_u32(bytes, source_bytes) != layout.tag ||
        load_u32(bytes, source_bytes + 4u) != layout.version ||
        load_u32(bytes, source_bytes + 8u) != layout.tag)
        return false;
    info.slot_count = load_u32(bytes, source_bytes + 12u);
    const size_t slot_bytes = static_cast<size_t>(info.slot_count) * layout.slot_bytes;
    if (!info.slot_count || info.slot_count > pointer_records.size() ||
        slot_base > UINT32_MAX || slot_bytes > std::numeric_limits<size_t>::max() - slot_base ||
        byte_count != slot_base + slot_bytes)
        return false;

    std::vector<uint32_t> record_slots(pointer_records.size(), UINT32_MAX);
    for (size_t index = 0; index < pointer_records.size(); ++index) {
        const uint64_t original = load_u64(bytes, source_bytes + 16u + index * 8u);
        const size_t offset = record_offset(layout, pointer_records[index]);
        const uint32_t low = load_u32(bytes, offset);
        const uint32_t high = load_u32(bytes, offset + 4u);
        if (!original || high != layout.tag || low < slot_base ||
            (low - slot_base) % layout.slot_bytes != 0u)
            return false;
        record_slots[index] = static_cast<uint32_t>((low - slot_base) / layout.slot_bytes);
        if (record_slots[index] >= info.slot_count) return false;
        for (size_t prior = 0; prior < index; ++prior) {
            const uint64_t prior_original =
                load_u64(bytes, source_bytes + 16u + prior * 8u);
            if ((prior_original == original) != (record_slots[prior] == record_slots[index]))
                return false;
        }
    }
    for (size_t index = pointer_records.size(); index < layout.max_pointer_records; ++index)
        if (load_u64(bytes, source_bytes + 16u + index * 8u) != 0u) return false;
    for (uint32_t slot = 0; slot < info.slot_count; ++slot)
        if (std::find(record_slots.begin(), record_slots.end(), slot) == record_slots.end())
            return false;

    const size_t witness_base = source_bytes + 16u +
        static_cast<size_t>(layout.max_pointer_records) * sizeof(uint64_t);
    info.witness_words.resize(layout.witness_word_count);
    for (size_t index = 0; index < info.witness_words.size(); ++index)
        info.witness_words[index] = load_u32(bytes, witness_base + index * sizeof(uint32_t));
    return true;
}

bool build_indirect_buffer_shadow(
        const ShaderResource& source, const uint8_t* source_bytes,
        const IndirectBufferShadowLayout& layout, std::span<const uint32_t> pointer_records,
        std::span<const uint32_t> witness_words,
        std::shared_ptr<std::vector<uint8_t>>& owner, uint32_t& slot_count) {
    owner.reset();
    slot_count = 0;
    if (!source_bytes || witness_words.size() != layout.witness_word_count ||
        !record_offsets_fit(source, layout, pointer_records))
        return false;

    std::vector<uint64_t> pointers(pointer_records.size());
    std::vector<uint32_t> slots(pointer_records.size());
    for (size_t index = 0; index < pointer_records.size(); ++index) {
        pointers[index] = load_u64(
            source_bytes, record_offset(layout, pointer_records[index]));
        if (!pointers[index] || !guest_readable(pointers[index], layout.slot_bytes)) return false;
        size_t prior = 0;
        for (; prior < index; ++prior)
            if (pointers[prior] == pointers[index]) break;
        slots[index] = prior < index ? slots[prior] : slot_count++;
    }

    const size_t source_size = static_cast<size_t>(source.size);
    const size_t header_bytes = indirect_buffer_shadow_header_bytes(layout);
    if (!header_bytes || source_size > std::numeric_limits<size_t>::max() - header_bytes)
        return false;
    const size_t slot_base = source_size + header_bytes;
    const size_t slot_bytes = static_cast<size_t>(slot_count) * layout.slot_bytes;
    if (slot_base > UINT32_MAX || slot_bytes > UINT32_MAX - slot_base) return false;
    const size_t binding_bytes = slot_base + slot_bytes;
    owner = std::make_shared<std::vector<uint8_t>>(binding_bytes);
    std::memcpy(owner->data(), source_bytes, source_size);
    store_u32(owner->data(), source_size, layout.tag);
    store_u32(owner->data(), source_size + 4u, layout.version);
    store_u32(owner->data(), source_size + 8u, layout.tag);
    store_u32(owner->data(), source_size + 12u, slot_count);
    for (size_t index = 0; index < pointer_records.size(); ++index) {
        store_u64(owner->data(), source_size + 16u + index * 8u, pointers[index]);
        const uint32_t packed_offset = static_cast<uint32_t>(
            slot_base + static_cast<size_t>(slots[index]) * layout.slot_bytes);
        const size_t offset = record_offset(layout, pointer_records[index]);
        store_u32(owner->data(), offset, packed_offset);
        store_u32(owner->data(), offset + 4u, layout.tag);
        if (std::find(slots.begin(), slots.begin() + index, slots[index]) ==
            slots.begin() + index)
            std::memcpy(owner->data() + packed_offset,
                        reinterpret_cast<const void*>(static_cast<uintptr_t>(pointers[index])),
                        layout.slot_bytes);
    }
    const size_t witness_base = source_size + 16u +
        static_cast<size_t>(layout.max_pointer_records) * sizeof(uint64_t);
    for (size_t index = 0; index < witness_words.size(); ++index)
        store_u32(owner->data(), witness_base + index * sizeof(uint32_t), witness_words[index]);
    return true;
}

bool current_indirect_buffer_shadow_matches(
        const ShaderResourceTable& table, const ShaderResource& source,
        const IndirectBufferShadowLayout& layout,
        std::span<const uint32_t> pointer_records) {
    IndirectBufferShadowInfo info;
    if (!parse_indirect_buffer_shadow(
            source, source.host_data, source.host_data_size, layout, pointer_records, info))
        return false;
    if (!source.host_data || !table_owns(table, source.host_data))
        return source.host_data != nullptr; // replay-owned, self-contained snapshot
    if (!guest_readable(source.gpu_addr, static_cast<uint32_t>(source.size)))
        return false;
    const auto* guest = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(source.gpu_addr));
    const size_t source_bytes = static_cast<size_t>(source.size);
    size_t cursor = 0;
    for (size_t index = 0; index < pointer_records.size(); ++index) {
        const size_t offset = record_offset(layout, pointer_records[index]);
        if (std::memcmp(source.host_data + cursor, guest + cursor, offset - cursor) != 0)
            return false;
        const uint64_t pointer = load_u64(guest, offset);
        if (pointer != load_u64(source.host_data, source_bytes + 16u + index * 8u) ||
            !guest_readable(pointer, layout.slot_bytes))
            return false;
        const uint32_t packed_offset = load_u32(source.host_data, offset);
        if (std::memcmp(source.host_data + packed_offset,
                        reinterpret_cast<const void*>(static_cast<uintptr_t>(pointer)),
                        layout.slot_bytes) != 0)
            return false;
        cursor = offset + sizeof(uint64_t);
    }
    return std::memcmp(source.host_data + cursor, guest + cursor, source_bytes - cursor) == 0;
}

bool parse_indirect_buffer_relocation(
        const ShaderResource& source, const uint8_t* bytes, size_t byte_count,
        const IndirectBufferRelocationLayout& layout,
        std::span<const IndirectBufferRelocationRecord> expected_records,
        IndirectBufferRelocationInfo& info) {
    info = {};
    if (!bytes || !relocation_layout_valid(source, layout, expected_records.size())) return false;
    const size_t source_bytes = static_cast<size_t>(source.size);
    if (source_bytes > byte_count || kRelocationHeaderBytes > byte_count - source_bytes)
        return false;
    const size_t header = source_bytes;
    const uint32_t record_count = load_u32(bytes, header + 16u);
    const uint32_t segment_count = load_u32(bytes, header + 20u);
    const uint32_t witness_count = load_u32(bytes, header + 24u);
    const uint32_t payload_offset = load_u32(bytes, header + 28u);
    const uint32_t payload_bytes = load_u32(bytes, header + 32u);
    if (load_u32(bytes, header) != layout.tag ||
        load_u32(bytes, header + 4u) != layout.version ||
        load_u32(bytes, header + 8u) != layout.tag ||
        load_u32(bytes, header + 12u) != source.size ||
        load_u32(bytes, header + 36u) != (layout.tag ^ source.size ^ record_count ^
                                         segment_count ^ payload_offset ^ payload_bytes) ||
        record_count != expected_records.size() || segment_count > layout.max_segments ||
        witness_count != layout.witness_word_count)
        return false;

    const uint64_t record_bytes = static_cast<uint64_t>(record_count) * kRelocationRecordBytes;
    const uint64_t segment_bytes = static_cast<uint64_t>(segment_count) * kRelocationSegmentBytes;
    const uint64_t witness_bytes = static_cast<uint64_t>(witness_count) * sizeof(uint32_t);
    const uint64_t expected_payload = static_cast<uint64_t>(source_bytes) +
        kRelocationHeaderBytes + record_bytes + segment_bytes + witness_bytes;
    if (expected_payload > UINT32_MAX || payload_offset != expected_payload ||
        payload_bytes > layout.max_binding_bytes ||
        payload_offset > layout.max_binding_bytes - payload_bytes ||
        byte_count != static_cast<size_t>(payload_offset) + payload_bytes)
        return false;

    info.source_bytes = static_cast<uint32_t>(source_bytes);
    info.payload_byte_offset = payload_offset;
    info.payload_bytes = payload_bytes;
    info.records.resize(record_count);
    const size_t records_base = header + kRelocationHeaderBytes;
    uint64_t prior_source_end = 0;
    for (size_t index = 0; index < record_count; ++index) {
        const size_t offset = records_base + index * kRelocationRecordBytes;
        auto& record = info.records[index];
        record.source_byte_offset = load_u32(bytes, offset);
        const uint32_t segment = load_u32(bytes, offset + 4u);
        record.guest_address = load_u64(bytes, offset + 8u);
        record.byte_count = load_u32(bytes, offset + 16u);
        if (load_u32(bytes, offset + 20u) != 0u ||
            record.source_byte_offset != expected_records[index].source_byte_offset ||
            record.guest_address != expected_records[index].guest_address ||
            record.byte_count != expected_records[index].byte_count ||
            record.source_byte_offset > source.size ||
            sizeof(uint64_t) > source.size - record.source_byte_offset ||
            record.source_byte_offset < prior_source_end ||
            load_u64(bytes, record.source_byte_offset) != record.guest_address)
            return false;
        prior_source_end = static_cast<uint64_t>(record.source_byte_offset) + sizeof(uint64_t);
        if (!record.guest_address || !record.byte_count) {
            if (record.guest_address || record.byte_count || segment != UINT32_MAX) return false;
        } else if (segment >= segment_count) {
            return false;
        }
    }

    const size_t segments_base = records_base + static_cast<size_t>(record_bytes);
    std::vector<RelocationInterval> canonical_segments;
    if (!canonical_relocation_intervals(expected_records, canonical_segments) ||
        canonical_segments.size() != segment_count)
        return false;
    info.segments.resize(segment_count);
    uint64_t prior_end = 0;
    uint32_t prior_packed_end = payload_offset;
    for (size_t index = 0; index < segment_count; ++index) {
        const size_t offset = segments_base + index * kRelocationSegmentBytes;
        auto& segment = info.segments[index];
        segment.guest_address = load_u64(bytes, offset);
        segment.byte_count = load_u32(bytes, offset + 8u);
        segment.packed_byte_offset = load_u32(bytes, offset + 12u);
        uint64_t end = 0;
        if (load_u64(bytes, offset + 16u) != 0u ||
            !interval_end(segment.guest_address, segment.byte_count, end) ||
            segment.guest_address != canonical_segments[index].begin ||
            end != canonical_segments[index].end ||
            (index && segment.guest_address < prior_end) ||
            segment.packed_byte_offset != prior_packed_end ||
            segment.packed_byte_offset > byte_count ||
            segment.byte_count > byte_count - segment.packed_byte_offset)
            return false;
        prior_end = end;
        if (segment.byte_count > UINT32_MAX - prior_packed_end) return false;
        prior_packed_end += segment.byte_count;
    }
    if (prior_packed_end != byte_count) return false;

    std::vector<bool> referenced_segments(segment_count, false);
    for (size_t index = 0; index < record_count; ++index) {
        if (!info.records[index].guest_address) continue;
        const size_t offset = records_base + index * kRelocationRecordBytes;
        const uint32_t segment_index = load_u32(bytes, offset + 4u);
        referenced_segments[segment_index] = true;
        const auto& segment = info.segments[segment_index];
        uint64_t record_end = 0, segment_end = 0;
        if (!interval_end(info.records[index].guest_address,
                          info.records[index].byte_count, record_end) ||
            !interval_end(segment.guest_address, segment.byte_count, segment_end) ||
            info.records[index].guest_address < segment.guest_address ||
            record_end > segment_end)
            return false;
    }
    if (std::find(referenced_segments.begin(), referenced_segments.end(), false) !=
        referenced_segments.end())
        return false;

    const size_t witness_base = segments_base + static_cast<size_t>(segment_bytes);
    info.witness_words.resize(witness_count);
    for (size_t index = 0; index < witness_count; ++index)
        info.witness_words[index] = load_u32(bytes, witness_base + index * sizeof(uint32_t));
    return true;
}

bool build_indirect_buffer_relocation(
        const ShaderResource& source, const uint8_t* source_bytes,
        const IndirectBufferRelocationLayout& layout,
        std::span<const IndirectBufferRelocationRecord> records,
        std::span<const uint32_t> witness_words,
        std::shared_ptr<std::vector<uint8_t>>& owner,
        IndirectBufferRelocationInfo& info) {
    owner.reset();
    info = {};
    if (!source_bytes || !relocation_layout_valid(source, layout, records.size()) ||
        witness_words.size() != layout.witness_word_count)
        return false;

    uint64_t prior_source_end = 0;
    for (const auto& record : records) {
        if (record.source_byte_offset > source.size ||
            sizeof(uint64_t) > source.size - record.source_byte_offset ||
            record.source_byte_offset < prior_source_end)
            return false;
        prior_source_end = static_cast<uint64_t>(record.source_byte_offset) + sizeof(uint64_t);
        uint64_t source_pointer = 0;
        std::memcpy(&source_pointer, source_bytes + record.source_byte_offset,
                    sizeof(source_pointer));
        if (source_pointer != record.guest_address) return false;
        if (!record.guest_address || !record.byte_count) {
            if (record.guest_address || record.byte_count) return false;
            continue;
        }
        uint64_t end = 0;
        if (!interval_end(record.guest_address, record.byte_count, end) ||
            !guest_readable(record.guest_address, record.byte_count))
            return false;
    }
    std::vector<RelocationInterval> merged;
    if (!canonical_relocation_intervals(records, merged)) return false;
    if (merged.size() > layout.max_segments) return false;

    const uint64_t record_bytes = static_cast<uint64_t>(records.size()) * kRelocationRecordBytes;
    const uint64_t segment_bytes = static_cast<uint64_t>(merged.size()) * kRelocationSegmentBytes;
    const uint64_t witness_bytes = static_cast<uint64_t>(witness_words.size()) * sizeof(uint32_t);
    const uint64_t payload_offset64 = source.size + kRelocationHeaderBytes + record_bytes +
        segment_bytes + witness_bytes;
    uint64_t payload_bytes64 = 0;
    for (const auto& interval : merged) {
        const uint64_t bytes = interval.end - interval.begin;
        if (bytes > UINT64_MAX - payload_bytes64) return false;
        payload_bytes64 += bytes;
    }
    if (payload_offset64 > UINT32_MAX || payload_bytes64 > UINT32_MAX ||
        payload_offset64 > layout.max_binding_bytes ||
        payload_bytes64 > layout.max_binding_bytes - payload_offset64)
        return false;
    const uint32_t payload_offset = static_cast<uint32_t>(payload_offset64);
    const uint32_t payload_bytes = static_cast<uint32_t>(payload_bytes64);
    owner = std::make_shared<std::vector<uint8_t>>(
        static_cast<size_t>(payload_offset) + payload_bytes);
    std::memcpy(owner->data(), source_bytes, static_cast<size_t>(source.size));
    const size_t header = static_cast<size_t>(source.size);
    store_u32(owner->data(), header, layout.tag);
    store_u32(owner->data(), header + 4u, layout.version);
    store_u32(owner->data(), header + 8u, layout.tag);
    store_u32(owner->data(), header + 12u, static_cast<uint32_t>(source.size));
    store_u32(owner->data(), header + 16u, static_cast<uint32_t>(records.size()));
    store_u32(owner->data(), header + 20u, static_cast<uint32_t>(merged.size()));
    store_u32(owner->data(), header + 24u, static_cast<uint32_t>(witness_words.size()));
    store_u32(owner->data(), header + 28u, payload_offset);
    store_u32(owner->data(), header + 32u, payload_bytes);
    store_u32(owner->data(), header + 36u,
              layout.tag ^ static_cast<uint32_t>(source.size) ^
                  static_cast<uint32_t>(records.size()) ^
                  static_cast<uint32_t>(merged.size()) ^ payload_offset ^ payload_bytes);

    const size_t records_base = header + kRelocationHeaderBytes;
    for (size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        const size_t offset = records_base + index * kRelocationRecordBytes;
        uint32_t segment_index = UINT32_MAX;
        if (record.guest_address) {
            auto segment = std::find_if(merged.begin(), merged.end(), [&](const auto& candidate) {
                return record.guest_address >= candidate.begin &&
                       record.guest_address <= candidate.end &&
                       record.byte_count <= candidate.end - record.guest_address;
            });
            if (segment == merged.end()) return false;
            segment_index = static_cast<uint32_t>(segment - merged.begin());
        }
        store_u32(owner->data(), offset, record.source_byte_offset);
        store_u32(owner->data(), offset + 4u, segment_index);
        store_u64(owner->data(), offset + 8u, record.guest_address);
        store_u32(owner->data(), offset + 16u, record.byte_count);
        store_u32(owner->data(), offset + 20u, 0u);
    }

    const size_t segments_base = records_base + static_cast<size_t>(record_bytes);
    uint32_t packed_offset = payload_offset;
    for (size_t index = 0; index < merged.size(); ++index) {
        const size_t offset = segments_base + index * kRelocationSegmentBytes;
        const uint32_t bytes = static_cast<uint32_t>(merged[index].end - merged[index].begin);
        store_u64(owner->data(), offset, merged[index].begin);
        store_u32(owner->data(), offset + 8u, bytes);
        store_u32(owner->data(), offset + 12u, packed_offset);
        store_u64(owner->data(), offset + 16u, 0u);
        std::memcpy(owner->data() + packed_offset,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(merged[index].begin)),
                    bytes);
        packed_offset += bytes;
    }
    const size_t witness_base = segments_base + static_cast<size_t>(segment_bytes);
    for (size_t index = 0; index < witness_words.size(); ++index)
        store_u32(owner->data(), witness_base + index * sizeof(uint32_t), witness_words[index]);

    return parse_indirect_buffer_relocation(
        source, owner->data(), owner->size(), layout, records, info);
}

bool current_indirect_buffer_relocation_matches(
        const ShaderResourceTable& table, const ShaderResource& source,
        const IndirectBufferRelocationLayout& layout,
        std::span<const IndirectBufferRelocationRecord> expected_records) {
    IndirectBufferRelocationInfo info;
    if (!parse_indirect_buffer_relocation(
            source, source.host_data, source.host_data_size, layout, expected_records, info))
        return false;
    if (!source.host_data || !table_owns(table, source.host_data))
        return source.host_data != nullptr;
    if (!guest_readable(source.gpu_addr, static_cast<uint32_t>(source.size))) return false;
    const auto* live_source = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(source.gpu_addr));
    if (std::memcmp(source.host_data, live_source, static_cast<size_t>(source.size)) != 0)
        return false;
    for (const auto& segment : info.segments) {
        if (!guest_readable(segment.guest_address, segment.byte_count) ||
            std::memcmp(source.host_data + segment.packed_byte_offset,
                        reinterpret_cast<const void*>(
                            static_cast<uintptr_t>(segment.guest_address)),
                        segment.byte_count) != 0)
            return false;
    }
    return true;
}

} // namespace prosper::gpu

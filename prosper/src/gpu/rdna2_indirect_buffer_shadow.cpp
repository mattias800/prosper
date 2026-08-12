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

} // namespace prosper::gpu

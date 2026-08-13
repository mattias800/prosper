#include "gpu/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/shader_resources.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

void store_u64(uint8_t* bytes, size_t offset, uint64_t value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}

uint32_t load_u32(const uint8_t* bytes, size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

void store_u32(uint8_t* bytes, size_t offset, uint32_t value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}

} // namespace

int main() {
    std::vector<uint8_t> source_bytes(64u, 0x5au);
    std::array<uint8_t, 80> pointee{};
    for (size_t index = 0; index < pointee.size(); ++index)
        pointee[index] = static_cast<uint8_t>(index * 3u + 1u);
    const uint64_t pointer0 = reinterpret_cast<uint64_t>(pointee.data());
    const uint64_t pointer1 = pointer0 + 16u;
    store_u64(source_bytes.data(), 0u, pointer0);
    store_u64(source_bytes.data(), 16u, pointer1);
    store_u64(source_bytes.data(), 32u, 0u);

    ShaderResource source;
    source.cls = ResourceClass::ConstantBuffer;
    source.format = DataFormat::Uint32;
    source.num_components = 1u;
    source.gpu_addr = reinterpret_cast<uint64_t>(source_bytes.data());
    source.size = source_bytes.size();
    source.stride = 16u;
    source.fetch_pc = 6u;
    source.host_data = source_bytes.data();
    source.host_data_size = source_bytes.size();

    const IndirectBufferRelocationLayout layout{
        0x1d1e2481u, 2u, 4u, 4u, 4096u, 2u};
    const std::array<IndirectBufferRelocationRecord, 3> records{{
        {0u, pointer0, 32u},
        {16u, pointer1, 32u},
        {32u, 0u, 0u},
    }};
    const std::array<uint32_t, 2> witnesses{0x12345678u, 0x9abcdef0u};

    std::shared_ptr<std::vector<uint8_t>> owner;
    IndirectBufferRelocationInfo info;
    CHECK(build_indirect_buffer_relocation(
              source, source_bytes.data(), layout, records, witnesses, owner, info),
          "overlapping and null records build one preserved-source relocation shadow");
    if (!owner) {
        std::fputs("FAIL: relocation shadow owner was not produced\n", stderr);
        return 1;
    }
    CHECK(owner && info.records.size() == 3u && info.segments.size() == 1u &&
              info.segments[0].guest_address == pointer0 &&
              info.segments[0].byte_count == 48u && info.payload_bytes == 48u,
          "overlapping pointee intervals normalize to one disjoint segment");
    CHECK(owner && std::memcmp(owner->data(), source_bytes.data(), source_bytes.size()) == 0,
          "version-2 shadow preserves every source byte including pointer qwords");
    CHECK(owner && std::memcmp(owner->data() + info.payload_byte_offset,
                               pointee.data(), 48u) == 0,
          "normalized segment copies the exact bounded pointee bytes");
    CHECK(info.witness_words == std::vector<uint32_t>(witnesses.begin(), witnesses.end()),
          "proof witnesses survive the serialized carrier");

    IndirectBufferRelocationInfo reparsed;
    CHECK(owner && parse_indirect_buffer_relocation(
              source, owner->data(), owner->size(), layout, records, reparsed) &&
              reparsed.payload_byte_offset == info.payload_byte_offset,
          "serialized relocation directory reparses against its exact record proof");

    ShaderResourceTable table;
    table.owned_host_data.push_back(owner);
    source.host_data = owner->data();
    source.host_data_size = owner->size();
    table.resources.push_back(source);
    CHECK(current_indirect_buffer_relocation_matches(table, table.resources[0], layout, records),
          "owned shadow revalidates the live source and pointee bytes");

    source_bytes[40] ^= 1u;
    CHECK(!current_indirect_buffer_relocation_matches(table, table.resources[0], layout, records),
          "same live source snapshot mutation invalidates authority");
    source_bytes[40] ^= 1u;
    source_bytes[0] ^= 1u;
    CHECK(!current_indirect_buffer_relocation_matches(table, table.resources[0], layout, records),
          "same source pointer-qword mutation invalidates record authority");
    source_bytes[0] ^= 1u;
    pointee[47] ^= 1u;
    CHECK(!current_indirect_buffer_relocation_matches(table, table.resources[0], layout, records),
          "same copied pointee endpoint mutation invalidates authority");
    pointee[47] ^= 1u;

    auto corrupt = *owner;
    const size_t segment_base = source_bytes.size() + 40u + records.size() * 24u;
    if (segment_base + 16u > corrupt.size()) {
        std::fputs("FAIL: relocation segment directory is truncated\n", stderr);
        return 1;
    }
    store_u32(corrupt.data(), segment_base + 12u,
              load_u32(corrupt.data(), segment_base + 12u) + 4u);
    CHECK(!parse_indirect_buffer_relocation(
              source, corrupt.data(), corrupt.size(), layout, records, reparsed),
          "directory payload-offset mutation fails closed");

    corrupt = *owner;
    store_u64(corrupt.data(), segment_base, pointer0 - 4u);
    CHECK(!parse_indirect_buffer_relocation(
              source, corrupt.data(), corrupt.size(), layout, records, reparsed),
          "segment range widened before the canonical proof union fails closed");

    corrupt = *owner;
    store_u32(corrupt.data(), segment_base + 8u,
              load_u32(corrupt.data(), segment_base + 8u) + 4u);
    CHECK(!parse_indirect_buffer_relocation(
              source, corrupt.data(), corrupt.size(), layout, records, reparsed),
          "segment byte count widened beyond the canonical proof union fails closed");

    auto wrong_records = records;
    ++wrong_records[1].byte_count;
    CHECK(!parse_indirect_buffer_relocation(
              source, owner->data(), owner->size(), layout, wrong_records, reparsed),
          "changed static bound proof cannot reuse a serialized shadow");

    const IndirectBufferRelocationLayout too_small{
        layout.tag, layout.version, layout.max_records, layout.max_segments,
        static_cast<uint32_t>(info.payload_byte_offset + info.payload_bytes - 1u),
        layout.witness_word_count};
    std::shared_ptr<std::vector<uint8_t>> rejected_owner;
    CHECK(!build_indirect_buffer_relocation(
              source, source_bytes.data(), too_small, records, witnesses,
              rejected_owner, reparsed),
          "binding cap rejects the same production payload before allocation");

    if (failures) {
        std::fprintf(stderr, "%d indirect-buffer shadow assertion(s) failed\n", failures);
        return 1;
    }
    std::puts("indirect-buffer relocation shadow tests passed");
    return 0;
}

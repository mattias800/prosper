#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/resources/shader_resources.hpp"

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
    using SourceAddressKind = IndirectBufferRelocationRecord::SourceAddressKind;

    std::vector<uint8_t> source_bytes(64u, 0x5au);
    std::array<uint8_t, 80> pointee{};
    for (size_t index = 0; index < pointee.size(); ++index)
        pointee[index] = static_cast<uint8_t>(index * 3u + 1u);
    const uint64_t pointer0 = reinterpret_cast<uint64_t>(pointee.data());
    const uint64_t pointer1 = pointer0 + 16u;
    store_u64(source_bytes.data(), 0u, pointer0);
    store_u64(source_bytes.data(), 16u, pointer1);
    store_u64(source_bytes.data(), 32u, 0u);
    store_u64(source_bytes.data(), 48u, pointer0 + 64u);

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
    const std::array<IndirectBufferRelocationRecord, 4> records{{
        {0u, pointer0, 32u},
        {16u, pointer1, 32u},
        {32u, 0u, 0u},
        {48u, pointer0 + 64u, 0u},
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
    CHECK(owner && info.records.size() == 4u && info.segments.size() == 1u &&
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
    const uint8_t* packed_middle = indirect_buffer_relocation_payload_bytes(
        owner->data(), owner->size(), info, pointer0 + 12u, 8u);
    CHECK(packed_middle && std::memcmp(packed_middle, pointee.data() + 12u, 8u) == 0,
          "parsed relocation directory resolves an exact interior payload interval");
    CHECK(!indirect_buffer_relocation_payload_bytes(
              owner->data(), owner->size(), info, pointer0 + 44u, 8u) &&
              !indirect_buffer_relocation_payload_bytes(
                  owner->data(), info.payload_byte_offset + 4u,
                  info, pointer0, 8u),
          "payload resolver rejects guest and carrier boundary crossings");
    const size_t records_base = source_bytes.size() + kIndirectBufferRelocationHeaderBytes;
    CHECK(info.records[0].source_address_kind == SourceAddressKind::RawU64 &&
              load_u32(owner->data(), records_base + 20u) == 0u,
          "version-2 records retain RawU64 semantics and a zero reserved word");

    IndirectBufferRelocationInfo reparsed;
    CHECK(owner && parse_indirect_buffer_relocation(
              source, owner->data(), owner->size(), layout, records, reparsed) &&
              reparsed.payload_byte_offset == info.payload_byte_offset,
          "serialized relocation directory reparses against its exact record proof");
    CHECK(owner && inspect_indirect_buffer_relocation(
              source, owner->data(), owner->size(), layout, reparsed) &&
              reparsed.records == info.records,
          "syntax-only inspection retains null and nonzero inactive record witnesses");
    auto inactive_pointer_corrupt = *owner;
    inactive_pointer_corrupt[48u] ^= 1u;
    CHECK(!inspect_indirect_buffer_relocation(
              source, inactive_pointer_corrupt.data(), inactive_pointer_corrupt.size(),
              layout, reparsed),
          "syntax-only inspection rejects a changed inactive nonzero pointer witness");

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

    corrupt = *owner;
    store_u32(corrupt.data(), records_base + 20u,
              static_cast<uint32_t>(SourceAddressKind::BufferDescriptorBase48));
    CHECK(!parse_indirect_buffer_relocation(
              source, corrupt.data(), corrupt.size(), layout, records, reparsed) &&
              !inspect_indirect_buffer_relocation(
                  source, corrupt.data(), corrupt.size(), layout, reparsed),
          "version-2 parser rejects a typed record in its reserved word");

    std::shared_ptr<std::vector<uint8_t>> rejected_owner;
    auto illegal_v2_records = records;
    illegal_v2_records[0].source_address_kind = SourceAddressKind::BufferDescriptorBase48;
    CHECK(!build_indirect_buffer_relocation(
              source, source_bytes.data(), layout, illegal_v2_records, witnesses,
              rejected_owner, reparsed),
          "version-2 builder rejects a typed source-address record");

    auto wrong_records = records;
    ++wrong_records[1].byte_count;
    CHECK(!parse_indirect_buffer_relocation(
              source, owner->data(), owner->size(), layout, wrong_records, reparsed),
          "changed static bound proof cannot reuse a serialized shadow");

    const IndirectBufferRelocationLayout too_small{
        layout.tag, layout.version, layout.max_records, layout.max_segments,
        static_cast<uint32_t>(info.payload_byte_offset + info.payload_bytes - 1u),
        layout.witness_word_count};
    CHECK(!build_indirect_buffer_relocation(
              source, source_bytes.data(), too_small, records, witnesses,
              rejected_owner, reparsed),
          "binding cap rejects the same production payload before allocation");

    // Two valid intervals can overlap at a non-dword displacement. Keep the guest interval exact,
    // but align the physical segment tail so an unaligned final dword can read its second u32
    // without relying on a partially out-of-range Vulkan load.
    std::vector<uint8_t> unaligned_source_bytes(32u, 0u);
    std::array<uint8_t, 96> unaligned_pointee{};
    for (size_t index = 0; index < unaligned_pointee.size(); ++index)
        unaligned_pointee[index] = static_cast<uint8_t>(index + 7u);
    const uint64_t unaligned_pointer0 =
        reinterpret_cast<uint64_t>(unaligned_pointee.data());
    const uint64_t unaligned_pointer1 = unaligned_pointer0 + 17u;
    store_u64(unaligned_source_bytes.data(), 0u, unaligned_pointer0);
    store_u64(unaligned_source_bytes.data(), 16u, unaligned_pointer1);
    ShaderResource unaligned_source = source;
    unaligned_source.gpu_addr =
        reinterpret_cast<uint64_t>(unaligned_source_bytes.data());
    unaligned_source.size = unaligned_source_bytes.size();
    unaligned_source.host_data = unaligned_source_bytes.data();
    unaligned_source.host_data_size = unaligned_source_bytes.size();
    const IndirectBufferRelocationLayout unaligned_layout{
        0x1d1e2482u, 2u, 2u, 2u, 4096u, 0u};
    const std::array<IndirectBufferRelocationRecord, 2> unaligned_records{{
        {0u, unaligned_pointer0, 32u},
        {16u, unaligned_pointer1, 32u},
    }};
    std::shared_ptr<std::vector<uint8_t>> unaligned_owner;
    IndirectBufferRelocationInfo unaligned_info;
    CHECK(build_indirect_buffer_relocation(
              unaligned_source, unaligned_source_bytes.data(), unaligned_layout,
              unaligned_records, {}, unaligned_owner, unaligned_info) &&
              unaligned_owner && unaligned_info.segments.size() == 1u &&
              unaligned_info.segments[0].byte_count == 49u &&
              unaligned_info.payload_bytes == 52u &&
              unaligned_owner->size() % sizeof(uint32_t) == 0u,
          "non-dword canonical interval keeps exact authority with a zero-padded physical tail");
    if (unaligned_owner) {
        CHECK((*unaligned_owner)[unaligned_owner->size() - 1u] == 0u &&
                  (*unaligned_owner)[unaligned_owner->size() - 2u] == 0u &&
                  (*unaligned_owner)[unaligned_owner->size() - 3u] == 0u,
              "physical tail padding is deterministic zero data");
        auto nonzero_padding = *unaligned_owner;
        nonzero_padding.back() = 1u;
        CHECK(!parse_indirect_buffer_relocation(
                  unaligned_source, nonzero_padding.data(), nonzero_padding.size(),
                  unaligned_layout, unaligned_records, unaligned_info),
              "same physical padding mutation fails closed");
    }

    // Version 3 independently derives the guest address from a preserved RDNA2 V# descriptor. Its
    // Base48 is word0 plus word1[15:0]; word1[31:16] remains descriptor control, not pointer bits.
    std::vector<uint8_t> descriptor_source_bytes(16u, 0x3cu);
    store_u32(descriptor_source_bytes.data(), 0u, static_cast<uint32_t>(pointer0));
    store_u32(descriptor_source_bytes.data(), 4u,
              static_cast<uint32_t>((pointer0 >> 32u) & 0xffffu) | (16u << 16u));
    ShaderResource descriptor_source = source;
    descriptor_source.gpu_addr = reinterpret_cast<uint64_t>(descriptor_source_bytes.data());
    descriptor_source.size = descriptor_source_bytes.size();
    descriptor_source.stride = 16u;
    descriptor_source.host_data = descriptor_source_bytes.data();
    descriptor_source.host_data_size = descriptor_source_bytes.size();
    const IndirectBufferRelocationLayout descriptor_layout{
        0x1d1e2483u, 3u, 1u, 1u, 4096u, 0u};
    const std::array<IndirectBufferRelocationRecord, 1> descriptor_records{{
        {0u, pointer0, 24u, SourceAddressKind::BufferDescriptorBase48},
    }};
    std::shared_ptr<std::vector<uint8_t>> descriptor_owner;
    IndirectBufferRelocationInfo descriptor_info;
    CHECK(build_indirect_buffer_relocation(
              descriptor_source, descriptor_source_bytes.data(), descriptor_layout,
              descriptor_records, {}, descriptor_owner, descriptor_info) &&
              descriptor_owner && descriptor_info.records ==
                  std::vector<IndirectBufferRelocationRecord>(
                      descriptor_records.begin(), descriptor_records.end()) &&
              std::memcmp(descriptor_owner->data(), descriptor_source_bytes.data(),
                          descriptor_source_bytes.size()) == 0,
          "version-3 descriptor-base record builds without rewriting source V# bytes");
    const size_t descriptor_record_base =
        descriptor_source_bytes.size() + kIndirectBufferRelocationHeaderBytes;
    if (!descriptor_owner ||
        descriptor_owner->size() < descriptor_record_base +
            kIndirectBufferRelocationRecordBytes) {
        std::fputs("FAIL: version-3 relocation record directory is truncated\n", stderr);
        return 1;
    }
    CHECK(load_u32(descriptor_owner->data(), descriptor_record_base + 20u) ==
                  static_cast<uint32_t>(SourceAddressKind::BufferDescriptorBase48) &&
              inspect_indirect_buffer_relocation(
                  descriptor_source, descriptor_owner->data(), descriptor_owner->size(),
                  descriptor_layout, descriptor_info),
          "version-3 syntax inspection derives a typed Base48 record from its source bytes");

    {
        auto descriptor_source_mutation = *descriptor_owner;
        descriptor_source_mutation[0] ^= 1u;
        CHECK(!parse_indirect_buffer_relocation(
                  descriptor_source, descriptor_source_mutation.data(),
                  descriptor_source_mutation.size(), descriptor_layout,
                  descriptor_records, descriptor_info),
              "version-3 source Base48 mutation invalidates relocation authority");

        auto descriptor_kind_mutation = *descriptor_owner;
        descriptor_kind_mutation.at(descriptor_record_base + 20u) =
            static_cast<uint8_t>(SourceAddressKind::RawU64);
        CHECK(!parse_indirect_buffer_relocation(
                  descriptor_source, descriptor_kind_mutation.data(),
                  descriptor_kind_mutation.size(), descriptor_layout,
                  descriptor_records, descriptor_info),
              "same-record source-address-kind mutation invalidates relocation authority");

        auto descriptor_address_mutation = *descriptor_owner;
        descriptor_address_mutation.at(descriptor_record_base + 8u) ^= 1u;
        CHECK(!parse_indirect_buffer_relocation(
                  descriptor_source, descriptor_address_mutation.data(),
                  descriptor_address_mutation.size(), descriptor_layout,
                  descriptor_records, descriptor_info),
              "serialized canonical address must equal the address derived from the source V#");
    }

    auto mismatched_descriptor_records = descriptor_records;
    mismatched_descriptor_records[0].guest_address += 4u;
    CHECK(!build_indirect_buffer_relocation(
              descriptor_source, descriptor_source_bytes.data(), descriptor_layout,
              mismatched_descriptor_records, {}, descriptor_owner, descriptor_info),
          "builder rejects a proof address that disagrees with the source V# Base48");

    auto canonical_high_source_bytes = descriptor_source_bytes;
    store_u32(canonical_high_source_bytes.data(), 0u, 0u);
    store_u32(canonical_high_source_bytes.data(), 4u, 0x00008000u);
    const std::array<IndirectBufferRelocationRecord, 1> canonical_high_records{{
        {0u, UINT64_C(1) << 47u, 0u, SourceAddressKind::BufferDescriptorBase48},
    }};
    CHECK(!build_indirect_buffer_relocation(
              descriptor_source, canonical_high_source_bytes.data(), descriptor_layout,
              canonical_high_records, {}, descriptor_owner, descriptor_info),
          "unsupported canonical-high Base48 descriptor fails closed");

    if (failures) {
        std::fprintf(stderr, "%d indirect-buffer shadow assertion(s) failed\n", failures);
        return 1;
    }
    std::puts("indirect-buffer relocation shadow tests passed");
    return 0;
}

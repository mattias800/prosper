// Generic dispatch-owned representation for shaders that load bounded guest pointers from a source
// buffer. A title/program contract supplies the exact pointer-record domain and witness words; this
// layer owns only snapshot layout, deduplication, bounds, lifetime, and live-state revalidation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace prosper::gpu {

struct ShaderResource;
struct ShaderResourceTable;

struct IndirectBufferShadowLayout {
    uint32_t tag = 0;
    uint32_t version = 0;
    uint32_t source_stride = 0;
    uint32_t slot_bytes = 0;
    uint32_t max_pointer_records = 0;
    uint32_t witness_word_count = 0;
};

struct IndirectBufferShadowInfo {
    uint32_t slot_count = 0;
    std::vector<uint32_t> witness_words;
};

// Program contracts identify their exact indirect instruction sites and return only this bounded
// access description. The recompiler's address validation and buffer lowering remain generic.
struct IndirectBufferShadowAccess {
    uint32_t byte_offset = 0;
    uint32_t components = 0;
};

// Version-2 relocation shadows preserve the source bytes verbatim. Each proven source record names
// one bounded guest interval; overlapping intervals are normalized into disjoint packed segments.
// The translated shader keeps computing the original guest address and relocates only at a proven
// GLOBAL consumer, so guest pointer arithmetic (including high-word canonicalization) stays intact.
struct IndirectBufferRelocationLayout {
    uint32_t tag = 0;
    uint32_t version = 2;
    uint32_t max_records = 0;
    uint32_t max_segments = 0;
    uint32_t max_binding_bytes = 0;
    uint32_t witness_word_count = 0;
};

struct IndirectBufferRelocationRecord {
    uint32_t source_byte_offset = 0;
    uint64_t guest_address = 0;
    uint32_t byte_count = 0;

    bool operator==(const IndirectBufferRelocationRecord&) const = default;
};

struct IndirectBufferRelocationSegment {
    uint64_t guest_address = 0;
    uint32_t byte_count = 0;
    uint32_t packed_byte_offset = 0;
};

struct IndirectBufferRelocationInfo {
    uint32_t source_bytes = 0;
    uint32_t payload_byte_offset = 0;
    uint32_t payload_bytes = 0;
    std::vector<IndirectBufferRelocationRecord> records;
    std::vector<IndirectBufferRelocationSegment> segments;
    std::vector<uint32_t> witness_words;
};

size_t indirect_buffer_shadow_header_bytes(const IndirectBufferShadowLayout& layout);

bool parse_indirect_buffer_shadow(
    const ShaderResource& source, const uint8_t* bytes, size_t byte_count,
    const IndirectBufferShadowLayout& layout, std::span<const uint32_t> pointer_records,
    IndirectBufferShadowInfo& info);

bool build_indirect_buffer_shadow(
    const ShaderResource& source, const uint8_t* source_bytes,
    const IndirectBufferShadowLayout& layout, std::span<const uint32_t> pointer_records,
    std::span<const uint32_t> witness_words,
    std::shared_ptr<std::vector<uint8_t>>& owner, uint32_t& slot_count);

bool current_indirect_buffer_shadow_matches(
    const ShaderResourceTable& table, const ShaderResource& source,
    const IndirectBufferShadowLayout& layout, std::span<const uint32_t> pointer_records);

bool parse_indirect_buffer_relocation(
    const ShaderResource& source, const uint8_t* bytes, size_t byte_count,
    const IndirectBufferRelocationLayout& layout,
    std::span<const IndirectBufferRelocationRecord> expected_records,
    IndirectBufferRelocationInfo& info);

// Syntax-only inspection for backend/capture boundaries that do not own the shader proof. It
// derives the serialized record directory and still enforces source-qword identity plus the exact
// canonical segment union. A compiler/discovery boundary must additionally call the overload above
// with independently derived expected records before granting relocation authority.
bool inspect_indirect_buffer_relocation(
    const ShaderResource& source, const uint8_t* bytes, size_t byte_count,
    const IndirectBufferRelocationLayout& layout,
    IndirectBufferRelocationInfo& info);

bool build_indirect_buffer_relocation(
    const ShaderResource& source, const uint8_t* source_bytes,
    const IndirectBufferRelocationLayout& layout,
    std::span<const IndirectBufferRelocationRecord> records,
    std::span<const uint32_t> witness_words,
    std::shared_ptr<std::vector<uint8_t>>& owner,
    IndirectBufferRelocationInfo& info);

bool current_indirect_buffer_relocation_matches(
    const ShaderResourceTable& table, const ShaderResource& source,
    const IndirectBufferRelocationLayout& layout,
    std::span<const IndirectBufferRelocationRecord> expected_records);

} // namespace prosper::gpu

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

} // namespace prosper::gpu

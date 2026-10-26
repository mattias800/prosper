#include "gpu/execute/draw_index_source.hpp"
#include "gpu/execute/draw_index_width.hpp"
#include "gpu/execute/ngg_draw_indices.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <stdint.h>
#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu {
// clang-format off: preserve the byte-identical moved bodies.

// The bound vertex buffers' UNCLAMPED record count (size/stride, the largest) -- the bound the #304
// part-two detector needs. 0 when the table has none.
uint32_t vertex_buffer_records_unclamped(const ShaderResourceTable* table) {
    uint32_t records = 0;
    if (table)
        for (const auto& r : table->resources)
            if (r.cls == ResourceClass::VertexBuffer && r.stride)
                records = std::max(records, r.size / r.stride);
    return records;
}
DrawIndexSource resolve_draw_index_source(const GpuState& ds, const GpuState::Draw& draw,
                                                 uint32_t n, uint32_t vb_records_unclamped) {
    DrawIndexSource source;
    source.element_bytes = index_elem_bytes(ds.index_type);
    source.addr = draw.index_addr;
    source.addr32 =
        draw.from_offset ? (draw.index_base + (uint64_t)draw.index_offset * 4u) : draw.index_addr;
    if (!index_size_detection_permitted(ds.index_type, ds.index_type_announced) || n < 2)
        return source;
    if (!guest_readable(draw.index_addr, n * 2u) || !guest_readable(source.addr32, n * 4u))
        return source;
    // The fingerprint is read from the CANDIDATE's own bytes: its odd 16-bit words are the high
    // halves of the 32-bit indices. For a DrawIndexOffset the CP's 16-bit address is index_base +
    // offset*2, half way to the indices, so words read there belong to other indices -- often in
    // another 64 KiB window, or at the other parity -- and Tomb Raider I's world draws were
    // declined on them (#4637). Without an offset the two addresses are the same bytes.
    const uint16_t* p16 = (const uint16_t*)(uintptr_t)source.addr32;
    const uint32_t* p32 = (const uint32_t*)(uintptr_t)source.addr32;
    // Zero high halves first, so every buffer that detector already classifies keeps its existing
    // verdict; the constant-non-zero form (#304 part two) only ever sees what it rejected.
    if (index_buffer_is_unannounced_32bit(p16, p32, n))
        source.detected = "zero-high-half";
    else if (index_buffer_is_unannounced_32bit_high(p16, p32, n, vb_records_unclamped))
        source.detected = "constant-high-half";
    else if (index_buffer_is_unannounced_32bit_in_pool(p32, n, vb_records_unclamped))
        source.detected = "in-pool";
    if (source.detected) {
        source.element_bytes = 4;
        source.addr = source.addr32;
    }
    return source;
}
NggDrawIndexFetch fetch_ngg_draw_indices(const GpuState& ds, const GpuState::Draw& draw,
                                                uint32_t vb_records_unclamped) {
    NggDrawIndexFetch out;
    if (draw.index_count > kNggMaxIndices) {
        out.refusal = "ngg-index-count";
        return out;
    }
    out.refusal = "ngg-index-unavailable";
    if (!draw.index_addr || !draw.index_count) return out;
    const DrawIndexSource source =
        resolve_draw_index_source(ds, draw, draw.index_count, vb_records_unclamped);
    if (!source.element_bytes ||
        !guest_readable(source.addr, draw.index_count * source.element_bytes))
        return out;
    const NggDrawIndices fetched =
        decode_ngg_draw_indices(reinterpret_cast<const void*>(static_cast<uintptr_t>(source.addr)),
                                source.element_bytes, draw.index_count, read_ngg_index_restart(ds));
    out.indices = fetched.indices;
    out.refusal = fetched.refusal;
    if (fetched.indices) out.vertex_range = fetched.max_index + 1u;
    return out;
}
// clang-format on

} // namespace prosper::gpu

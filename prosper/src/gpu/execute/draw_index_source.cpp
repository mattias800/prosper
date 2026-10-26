#include "gpu/execute/draw_index_source.hpp"
#include "gpu/execute/draw_index_width.hpp"
#include "gpu/execute/ngg_draw_indices.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <stdint.h>
#include "diagnostics/env_cache.hpp"
#include "gpu/execute/gpu_execute.hpp"

namespace prosper::gpu {

namespace {
// PROSPER_INDEX_DETECT_CENSUS=1 -- diagnostic only. For an unannounced indexed draw that all 32-bit
// detectors left as 16-bit, print once per index buffer what each detector's clauses saw: the count,
// the vertex bound, the first readings at 32 bits, their spread, and how many distinct high halves
// each parity holds. A level that still renders as stretched triangles (#4637) is then either a 16-bit
// buffer read correctly, or a 32-bit one whose shape no clause admits, and this reports the sampled clauses; it does not prove the width.
void note_index_left_16bit(uint64_t addr, const uint16_t* p16, const uint32_t* p32, uint32_t n,
                           uint32_t vertex_bound) {
    static std::mutex mutex;
    static std::set<uint64_t> seen;
    static std::atomic<uint32_t> printed{0};
    constexpr uint32_t kMaxLines = 256;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (printed.load() >= kMaxLines || !seen.insert(addr).second) return;
        ++printed;
    }
    const uint32_t m = std::min(n, 64u);
    uint32_t lo = UINT32_MAX, hi = 0, above_bound = 0;
    std::set<uint32_t> distinct32;
    std::set<uint16_t> even_halves, odd_halves;
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t d = 0;
        std::memcpy(&d, reinterpret_cast<const char*>(p32) + size_t{4} * i, 4);
        lo = std::min(lo, d);
        hi = std::max(hi, d);
        if (d >= vertex_bound) ++above_bound;
        distinct32.insert(d);
        uint16_t w = 0;
        std::memcpy(&w, reinterpret_cast<const char*>(p16) + size_t{2} * i, 2);
        (i & 1 ? odd_halves : even_halves).insert(w);
    }
    char first[160] = {};
    int at = 0;
    for (uint32_t i = 0; i < std::min(m, 8u) && at < (int)sizeof(first) - 12; ++i) {
        uint32_t d = 0;
        std::memcpy(&d, reinterpret_cast<const char*>(p32) + size_t{4} * i, 4);
        at += std::snprintf(first + at, sizeof(first) - at, "%s%u", i ? "," : "", d);
    }
    std::fprintf(stderr,
                 "[index-census] left 16-bit addr=0x%llx n=%u vertex_bound=%u as32=[%s] "
                 "lo=%u hi=%u span=%u distinct32=%zu above_bound=%u/%u even_halves=%zu "
                 "odd_halves=%zu (#4637)\n",
                 static_cast<unsigned long long>(addr), n, vertex_bound, first, lo, hi, hi - lo,
                 distinct32.size(), above_bound, m, even_halves.size(), odd_halves.size());
    if (printed.load() == kMaxLines)
        std::fprintf(stderr, "[index-census] line cap %u reached (a CAP, not a count)\n",
                     kMaxLines);
}
}   // namespace

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
// clang-format on

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
    const uint16_t* p16 = (const uint16_t*)(uintptr_t)draw.index_addr;
    const uint32_t* p32 = (const uint32_t*)(uintptr_t)source.addr32;
    // Retain both original checks, including the draw's own 16-bit region on offset draws.
    // A readable 32-bit candidate can be another resource after a genuine 16-bit index buffer.
    if (index_buffer_is_unannounced_32bit(p16, p32, n))
        source.detected = "zero-high-half";
    else if (index_buffer_is_unannounced_32bit_high(p16, p32, n, vb_records_unclamped))
        source.detected = "constant-high-half";
    if (!source.detected) {
        const auto detect_same_bytes = [vb_records_unclamped](uint64_t addr,
                                                              uint32_t count) -> const char* {
            const auto* words = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(addr));
            const auto* dwords = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(addr));
            if (index_buffer_is_unannounced_32bit(words, dwords, count)) return "zero-high-half";
            if (index_buffer_is_unannounced_32bit_high(words, dwords, count, vb_records_unclamped))
                return "constant-high-half";
            if (index_buffer_is_unannounced_32bit_in_pool(dwords, count, vb_records_unclamped))
                return "in-pool";
            return nullptr;
        };
        // New offset recovery also needs a 32-bit shape WITHIN the already validated n*2 bytes
        // of the draw's own region. At an odd offset its first halfword is a high half: skip it
        // and inspect only complete dwords aligned relative to index_base. No additional read
        // beyond the 16-bit draw's extent is allowed to supply this corroboration.
        const uint32_t skip = draw.index_offset & 1u;
        if (source.addr32 == draw.index_addr ||
            detect_same_bytes(draw.index_addr + uint64_t{2} * skip, (n - skip) / 2u))
            source.detected = detect_same_bytes(source.addr32, n);
    }
    if (source.detected) {
        source.element_bytes = 4;
        source.addr = source.addr32;
        // NOLINTNEXTLINE(concurrency-mt-unsafe): cached environment read
    } else if (PROSPER_ENV_ON("PROSPER_INDEX_DETECT_CENSUS")) {
        note_index_left_16bit(draw.index_addr, p16, p32, n, vb_records_unclamped);
    }
    return source;
}
// clang-format off: preserve the unchanged fetch body.
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

#pragma once
#include "gpu/pm4/command_processor.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {

struct ShaderResourceTable;

// clang-format off: preserve the byte-identical moved result types.

// Where an indexed draw's `n` indices live and at what element size: the announced size, or -- for a
// title that announced none (#3009) -- the size the #304 detectors recover from the bytes, with a
// DrawIndexOffset's address recomputed at that stride. ONE rule for every consumer: the ordinary
// path binds what this names for vkCmdDrawIndexed, and the merged-NGG path (#3135 P6) reads the
// same bytes into its subgroup plan, so the two can never read one buffer two ways.
// `element_bytes` is 0 for an unknown announced size. `detected` names the detector that fired.
struct DrawIndexSource {
    uint64_t addr = 0;
    uint64_t addr32 = 0;   // the same buffer at a 4-byte stride (differs only for an offset draw)
    uint32_t element_bytes = 0;
    const char* detected = nullptr;
};

// #3135 P6: an indexed merged-NGG draw's indices, read where and at the size the ordinary path would
// bind them (resolve_draw_index_source), then decoded by ngg_draw_indices. `vertex_range` (max index
// + 1) is what sizes the linked fold and the vertex buffers, never the index count. `refusal` names
// the rule when no indices were read: ngg-index-count, ngg-index-unavailable (no address or count, an
// unknown element size, or unreadable bytes), or the decode's own refusal.
struct NggDrawIndexFetch {
    std::shared_ptr<const std::vector<uint32_t>> indices;
    const char* refusal = nullptr;
    uint32_t vertex_range = 0;
};
// clang-format on

uint32_t vertex_buffer_records_unclamped(const ShaderResourceTable* table);
DrawIndexSource resolve_draw_index_source(const GpuState& ds, const GpuState::Draw& draw,
                                          uint32_t n, uint32_t vb_records_unclamped);
NggDrawIndexFetch fetch_ngg_draw_indices(const GpuState& ds, const GpuState::Draw& draw,
                                         uint32_t vb_records_unclamped);

} // namespace prosper::gpu

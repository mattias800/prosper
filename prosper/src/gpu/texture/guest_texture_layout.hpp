// guest_texture_layout.hpp — exact host-produced layouts for guest-visible sampled textures.
#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

// Register a guest-visible allocation whose sampled-linear rows have an exact HLE-owned pitch.
// The ordinary GFX10 fallback remains 256-byte aligned; this provenance is for producers such as
// AvPlayer that return CPU-staged pixels and explicitly publish their physical pitch to the title.
void register_guest_linear_texture_layout(uint64_t base, size_t bytes,
                                          uint32_t row_pitch_bytes);
void unregister_guest_linear_texture_layout(uint64_t base);

// Return the registered pitch only when `address` belongs to a registered allocation and the
// descriptor's visible row fits within that physical row. Zero means no exact host provenance.
uint32_t guest_linear_texture_row_pitch(uint64_t address, uint32_t visible_row_bytes);

} // namespace prosper::gpu

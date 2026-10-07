// The guest-memory claim one volume colour slot's producer pass makes, from the native layout its
// CB_COLORn registers prove (#3842, #4625). One rule for every slot of a layered pass (#4643):
// Kena's translucency-lighting injection writes two 64^3 volumes per pass, slot 0 and slot 1, and
// both must be claimed, tracked and published the same way.
#pragma once

#include "gpu/execute/renderer_volume_publication.hpp"   // VolumeGuestLayout
#include "gpu/texture/tile.hpp"                          // tile_mode_supports_volume
#include "shared/rtt/rtt_authority.hpp"                  // live_rtt_color_footprint_bytes

#include <cstdint>

namespace prosper::frontend {

struct VolumeProducerShape {
    uint32_t depth = 0;            // zero: the slot is a 2D target
    uint64_t physical_bytes = 0;   // native tiled footprint; zero when the layout is unproven
    uint64_t guard_bytes = 0;      // what the claim guards: physical bytes, else a linear bound
    prosper::gpu::VolumeGuestLayout layout{};   // set only when the footprint is proven
    bool proven() const { return physical_bytes != 0; }
};

// `Binding` is a DrawItem::ColorTargetBinding; the tests use a stand-in with the same fields.
template <class Binding>
VolumeProducerShape volume_producer_shape(const Binding& view, uint32_t native_w, uint32_t native_h,
                                          uint32_t bytes_per_texel) {
    VolumeProducerShape shape;
    shape.depth = view.selected_mip_depth;
    if (!shape.depth) return shape;
    const bool layout_supported = view.native_layout_known && !view.mip_level &&
                                  !view.in_mip_tail &&
                                  prosper::gpu::tile_mode_supports_volume(view.tile_mode);
    shape.physical_bytes = layout_supported
                               ? prosper::gpu::tiled_volume_bytes(native_w, native_h, shape.depth,
                                                                  view.tile_mode, bytes_per_texel)
                               : 0u;
    shape.guard_bytes =
        shape.physical_bytes
            ? shape.physical_bytes
            : live_rtt_color_footprint_bytes(native_w, native_h, shape.depth, bytes_per_texel);
    if (shape.physical_bytes)
        shape.layout = {native_w, native_h, shape.depth, view.tile_mode, bytes_per_texel};
    return shape;
}

}   // namespace prosper::frontend

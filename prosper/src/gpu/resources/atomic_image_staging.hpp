#pragma once
// Staging EXTENTS for an R32_UINT StorageImage lowered to a linear atomic SSBO (#2265's transform,
// extracted by #3195 so the extent contract can be stated and tested).
//
// Two byte counts live here, and confusing them is the failure this header exists to prevent:
//
//   * `linear_bytes` -- LOGICAL. Tightly packed w*h*layers*4, the shape the shader indexes and the
//     size of the host staging buffer the dispatch reads and writes.
//   * `slice_bytes` / `guest_bytes` -- PHYSICAL. What the guest surface actually occupies. A tiled
//     slice is padded up to whole micro-tiles, so it is strictly LARGER than the logical w*h*4:
//     measured for Sonic Racing: CrossWorlds' (3840, 2160, bpe 4, tile 27), physical 33,423,360
//     against logical 33,177,600, a 245,760-byte difference. `tiled_mip_chain_bytes` -- documented
//     as the array slice stride -- returns the same 33,423,360, so the two candidate strides agree
//     and the slice needs no further alignment.
//
// `guest_bytes` is the extent every guest-side question must be asked with: the readability probe
// before staging, the bound handed to `resource_bytes_for`, and the write-back notification. Using
// the logical extent there would under-bound the probe by exactly that padding, and the per-layer
// detile -- which steps `slice_bytes` per layer -- would then walk past the region proven readable.
//
// #3195 was filed against a ternary that appeared to make a distinction here and did not
// (`atomic_image ? resource_bytes_for(r, guest_bytes) : resource_bytes_for(r, guest_bytes)`). The
// history says the arms converged when the NON-atomic arm adopted the bounded call in
// "gpu: honor scalar buffer descriptor bounds"; the atomic arm has never changed. Nothing was
// dropped -- but the invariant that made the ternary redundant, that `guest_bytes` already carries
// the PHYSICAL extent by the time the source pointer is taken, was an unnamed local expression no
// test could reach. It is named and tested here so a future "restoration" of that distinction
// reddens instead of silently under-bounding the probe.
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"

#include <cstddef>
#include <cstdint>

namespace prosper::gpu {

struct AtomicImageStagingExtents {
    uint32_t layers      = 1;   // array layers staged (1 for a plain 2D view)
    size_t   slice_bytes = 0;   // PHYSICAL guest bytes per layer, padding included
    size_t   guest_bytes = 0;   // slice_bytes * layers -- the guest footprint to prove readable
    uint64_t linear_bytes = 0;  // LOGICAL w*h*layers*4 -- the tightly packed staging buffer
    bool     valid       = false;
};

// Extents for a resource that has already passed `shader_resource_supports_atomic_image_buffer`.
// `valid` is false for a layout the staging transform cannot express: a logical or physical extent
// that does not fit the 32-bit guest-readability probe, an empty footprint, or a linear surface
// whose declared row pitch is narrower than one tight row (which would make the row-copy source
// stride shorter than the destination and read the surface skewed).
inline AtomicImageStagingExtents atomic_image_staging_extents(const ShaderResource& resource) {
    AtomicImageStagingExtents e;
    e.layers = shader_resource_atomic_image_layers(resource);
    e.linear_bytes = static_cast<uint64_t>(resource.width) * resource.height * e.layers *
                     sizeof(uint32_t);
    const size_t tight_pitch = static_cast<size_t>(resource.width) * sizeof(uint32_t);
    e.slice_bytes = resource.tile_mode
        ? tiled_surface_bytes(resource.width, resource.height, resource.tile_mode, 0,
                              sizeof(uint32_t))
        : (resource.linear_row_pitch_bytes
               ? static_cast<size_t>(resource.linear_row_pitch_bytes)
               : tight_pitch) * resource.height;
    e.guest_bytes = e.slice_bytes * e.layers;
    e.valid = !(e.linear_bytes > UINT32_MAX ||
                (!resource.tile_mode && resource.linear_row_pitch_bytes &&
                 resource.linear_row_pitch_bytes < tight_pitch) ||
                !e.guest_bytes || e.guest_bytes > UINT32_MAX);
    return e;
}

} // namespace prosper::gpu

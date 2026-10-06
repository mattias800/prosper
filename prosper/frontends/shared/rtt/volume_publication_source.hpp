// What the live renderer hands to publish_volume_to_guest for one claimed volume (#4625), kept
// separate from the renderer so the claim, alias and release rules are testable without a device.
// `Surface` is the renderer's RTT cache entry; the tests use a stand-in with the same fields.
#pragma once

#include "gpu/execute/renderer_volume_publication.hpp"
#include "shared/rtt/rtt_authority.hpp"

#include <cstdint>
#include <utility>

namespace prosper::frontend {

template <class Surface>
prosper::gpu::VolumePublicationSource volume_publication_source(uint64_t base,
                                                                const Surface& surface) {
    prosper::gpu::VolumePublicationSource source;
    source.base = base;
    source.claimed_bytes = surface.volume_guest_bytes;
    source.footprint_proven = surface.volume_footprint_proven;
    // A tombstone keeps the claim with no image, and a later 2D pass at the base leaves the claim
    // with a 2D image: neither can supply the volume's slices.
    source.renderer_image_valid = surface.gpu_valid && surface.volume_depth != 0;
    source.image_width = surface.w;
    source.image_height = surface.h;
    source.image_depth = surface.volume_depth;
    source.layout = surface.volume_layout;
    source.exact_representation = surface.guest_format == surface.format;
    return source;
}

// The first other entry that may overlap the claimed footprint, as {base, bytes}, or {0, 0}. The
// rule is the guest-write drain's own: an entry holding a volume claim by physical topology as well
// as address (`volume_may_overlap`), any other surface by address alone. Asking topology of a plain
// 2D surface is wrong in the refusing direction: an unproven topology answers "may overlap", which
// refused Kena's volume over an unrelated 3200x1800 target in the first live run.
template <class Cache, class SurfaceBytes, class VolumeMayOverlap>
std::pair<uint64_t, uint64_t>
volume_publication_alias(const Cache& cache, uint64_t base, uint64_t claimed_bytes,
                         SurfaceBytes&& surface_bytes, VolumeMayOverlap&& volume_may_overlap) {
    for (const auto& [other, other_surface] : cache) {
        if (other == base) continue;
        const uint64_t other_bytes = surface_bytes(other_surface);
        const bool overlaps =
            other_surface.volume_guest_bytes
                ? volume_may_overlap(other, other_surface.volume_guest_bytes, base, claimed_bytes)
                : live_rtt_ranges_overlap(other, other_bytes, base, claimed_bytes);
        if (overlaps) return {other, other_bytes};
    }
    return {0, 0};
}

// After a publication guest memory holds every slice; the entry no longer claims its footprint.
template <class Surface>
void release_volume_claim(Surface& surface) {
    surface.volume_guest_bytes = 0;
    surface.volume_footprint_proven = false;
    surface.volume_layout = {};
}

}   // namespace prosper::frontend

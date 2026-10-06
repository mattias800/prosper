// renderer_volume_publication.cpp -- see renderer_volume_publication.hpp (#4625).
#include "gpu/execute/renderer_volume_publication.hpp"

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/texture/tile.hpp"
#include "host/memory/guest_write_watch.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace prosper::gpu {

const char* volume_publication_name(VolumePublication publication) {
    switch (publication) {
        case VolumePublication::Published: return "published";
        case VolumePublication::NothingToPublish: return "nothing-to-publish";
        case VolumePublication::NoPublisher: return "no-publisher";
        case VolumePublication::NoRendererImage: return "no-renderer-image";
        case VolumePublication::UnprovenFootprint: return "unproven-footprint";
        case VolumePublication::UnknownLayout: return "unknown-layout";
        case VolumePublication::ExtentMismatch: return "extent-mismatch";
        case VolumePublication::FormatConversion: return "format-conversion";
        case VolumePublication::OverlappingAlias: return "overlapping-alias";
        case VolumePublication::GuestUnmapped: return "guest-unmapped";
        case VolumePublication::ReadbackFailed: return "readback-failed";
        case VolumePublication::Count: break;
    }
    return "unknown";
}

VolumePublication plan_volume_publication(const VolumePublicationSource& source) {
    const VolumeGuestLayout& layout = source.layout;
    if (!source.base || !source.claimed_bytes) return VolumePublication::NothingToPublish;
    if (!source.renderer_image_valid || !source.image_depth)
        return VolumePublication::NoRendererImage;
    if (!source.footprint_proven) return VolumePublication::UnprovenFootprint;
    if (!layout.depth || !layout.bytes_per_texel || !tile_mode_supports_volume(layout.tile_mode))
        return VolumePublication::UnknownLayout;
    // The retained image must be the native extent: a scaled or reallocated image would tile its
    // texels at the wrong addresses even when the byte counts happened to agree.
    if (source.image_width != layout.width || source.image_height != layout.height ||
        source.image_depth != layout.depth ||
        tiled_volume_bytes(layout.width, layout.height, layout.depth, layout.tile_mode,
                           layout.bytes_per_texel) != source.claimed_bytes)
        return VolumePublication::ExtentMismatch;
    if (!source.exact_representation) return VolumePublication::FormatConversion;
    if (source.overlapping_alias) return VolumePublication::OverlappingAlias;
    if (source.claimed_bytes > std::numeric_limits<uint32_t>::max())
        return VolumePublication::GuestUnmapped;
    return VolumePublication::Published;
}

VolumePublication publish_volume_to_guest(const VolumePublicationSource& source,
                                          const VolumeReadbackFn& readback, uint8_t* guest) {
    const VolumePublication plan = plan_volume_publication(source);
    if (plan != VolumePublication::Published) return plan;
    // Mapped, as the compute writeback this stands in for checks: a write-watched page is
    // read-only to the OS until the fault handler admits the write, so guest_writable() would
    // refuse exactly the GPU allocations a volume lives in.
    if (!guest || !guest_readable(source.base, static_cast<uint32_t>(source.claimed_bytes)))
        return VolumePublication::GuestUnmapped;
    const VolumeGuestLayout& layout = source.layout;
    const uint64_t linear_bytes =
        static_cast<uint64_t>(layout.width) * layout.height * layout.depth * layout.bytes_per_texel;
    std::vector<uint8_t> linear;
    if (!readback || !readback(linear) || linear.size() != linear_bytes)
        return VolumePublication::ReadbackFailed;
    // A host store into guest memory: open any write-watched page first, as compute's own storage
    // writeback does, so the tile pass does not fault once per page (guest_write_watch.hpp).
    struct PreparedHostWrite {
        uint64_t address, bytes;
        ~PreparedHostWrite() { host::guest_write_watch_notify_host_write_done(address, bytes); }
    } const prepared{source.base, source.claimed_bytes};
    host::guest_write_watch_notify_host_write(prepared.address, prepared.bytes);
    if (!tile_volume(guest, static_cast<size_t>(source.claimed_bytes), linear.data(), layout.width,
                     layout.height, layout.depth, layout.tile_mode, layout.bytes_per_texel))
        return VolumePublication::UnknownLayout;
    // A GPU write as far as every other cache is concerned: the journal and page watches stop
    // trusting their old view of these bytes, and renderer aliases of the range are invalidated.
    set_guest_gpu_write_origin("renderer-volume-publication");
    notify_guest_gpu_write(source.base, source.claimed_bytes);
    set_guest_gpu_write_origin(nullptr);
    return VolumePublication::Published;
}

namespace {
RendererVolumePublisherFn g_publisher;
}   // namespace

void set_renderer_volume_publisher(RendererVolumePublisherFn fn) {
    g_publisher = std::move(fn);
}

VolumePublication publish_renderer_volumes(uint64_t gpu_addr, uint64_t bytes) {
    return g_publisher ? g_publisher(gpu_addr, bytes) : VolumePublication::NoPublisher;
}

const char* compute_renderer_volume_refusal(uint64_t gpu_addr, uint64_t bytes,
                                            bool guest_bytes_needed) {
    if (!guest_bytes_needed || !overlaps_unpublished_renderer_volume(gpu_addr, bytes))
        return nullptr;
    const VolumePublication result = publish_renderer_volumes(gpu_addr, bytes);
    // Re-ask rather than trusting the answer: only a claim actually released admits guest bytes.
    if (result == VolumePublication::Published &&
        !overlaps_unpublished_renderer_volume(gpu_addr, bytes))
        return nullptr;
    switch (result) {
        case VolumePublication::NoPublisher:
            return "renderer volume has no complete guest publication (no publisher)";
        case VolumePublication::NoRendererImage:
            return "renderer volume has no complete guest publication (no renderer image)";
        case VolumePublication::UnprovenFootprint:
            return "renderer volume has no complete guest publication (unproven footprint)";
        case VolumePublication::UnknownLayout:
            return "renderer volume has no complete guest publication (unknown layout)";
        case VolumePublication::ExtentMismatch:
            return "renderer volume has no complete guest publication (extent mismatch)";
        case VolumePublication::FormatConversion:
            return "renderer volume has no complete guest publication (format conversion)";
        case VolumePublication::OverlappingAlias:
            return "renderer volume has no complete guest publication (overlapping alias)";
        case VolumePublication::GuestUnmapped:
            return "renderer volume has no complete guest publication (guest unmapped)";
        case VolumePublication::ReadbackFailed:
            return "renderer volume has no complete guest publication (readback failed)";
        default: return "renderer volume has no complete guest publication (claim not released)";
    }
}

}   // namespace prosper::gpu

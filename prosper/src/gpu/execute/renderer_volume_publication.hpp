// renderer_volume_publication.hpp -- writing a renderer-produced volume back to guest memory (#4625).
//
// A volume (3D colour target) that the live renderer drew stays in a Vulkan image, and the renderer
// claims its guest footprint: guest memory is stale there, so a consumer that would read guest bytes
// is refused instead. Graphics can sample the retained image directly. Compute has no such path for
// a layered binding, so before this file every compute dispatch that bound a claimed volume was
// skipped -- Kena's translucency-lighting clear (`0x5007bc0000`, four 64x64x64 RGBA16F storage
// volumes) once per frame, which left the volume accumulating across frames.
//
// Publication is the fix: read the retained image back, tile it into the producer's own native
// layout, write it to the claimed guest footprint and release the claim. Guest memory is then the
// only authority, and the dispatch runs through the ordinary guest-byte path. It costs a readback
// the guest observes whenever the dispatch reads the volume or leaves any texel unwritten; nothing
// here can prove a dispatch overwrites every texel, so it is paid for write-only bindings too.
//
// Every refusal is named: a claim that cannot be published exactly keeps the dispatch skipped.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {

// The guest layout a volume producer pass wrote, recorded when the renderer takes its claim.
struct VolumeGuestLayout {
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t tile_mode = 0;
    uint32_t bytes_per_texel = 0;
    bool operator==(const VolumeGuestLayout&) const = default;
};

enum class VolumePublication : uint8_t {
    Published,   // guest memory holds the renderer's volume; the caller releases the claim
    NothingToPublish,   // no claimed volume overlaps the range
    NoPublisher,   // no live renderer registered a publisher
    NoRendererImage,   // the claim survives without a complete retained 3D image (a tombstone)
    UnprovenFootprint,   // the claimed extent was not derived from the native tiled layout
    UnknownLayout,   // no producer layout was recorded, or its tile mode has no volume pattern
    ExtentMismatch,   // the retained image or the native tiled size disagrees with the claim
    FormatConversion,   // the retained image's format is not the guest's byte representation
    OverlappingAlias,   // another renderer surface may overlap the footprint
    GuestUnmapped,
    ReadbackFailed,
    Count
};
const char* volume_publication_name(VolumePublication publication);

// What the renderer knows about one claimed volume.
struct VolumePublicationSource {
    uint64_t base = 0;
    uint64_t claimed_bytes = 0;
    bool footprint_proven = false;
    bool renderer_image_valid = false;
    uint32_t image_width = 0, image_height = 0, image_depth = 0;
    VolumeGuestLayout layout;
    bool exact_representation = false;   // retained image format == raw guest format
    bool overlapping_alias = false;
};

// The decision alone, before any readback.
VolumePublication plan_volume_publication(const VolumePublicationSource& source);

// Fills `linear` with the retained image, slice by slice (x fastest, then y, then z).
using VolumeReadbackFn = std::function<bool(std::vector<uint8_t>& linear)>;

// Plans, reads back, tiles into `guest` (the footprint's host view, `claimed_bytes` long) and
// notifies the write as a GPU write so guest-memory caches, watches and renderer aliases all see it.
// On Published the caller must release its claim before its next guest-write drain.
VolumePublication publish_volume_to_guest(const VolumePublicationSource& source,
                                          const VolumeReadbackFn& readback, uint8_t* guest);

// The live renderer: publish every claimed volume overlapping [gpu_addr, gpu_addr + bytes).
using RendererVolumePublisherFn =
    std::function<VolumePublication(uint64_t gpu_addr, uint64_t bytes)>;
void set_renderer_volume_publisher(RendererVolumePublisherFn fn);
VolumePublication publish_renderer_volumes(uint64_t gpu_addr, uint64_t bytes);

// The compute binding gate. `guest_bytes_needed` is false only for a binding the renderer can serve
// from its current 2D representation. Returns nullptr to admit the binding, else the skip reason.
const char* compute_renderer_volume_refusal(uint64_t gpu_addr, uint64_t bytes,
                                            bool guest_bytes_needed);

}   // namespace prosper::gpu

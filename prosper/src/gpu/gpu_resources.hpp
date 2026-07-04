// gpu_resources.hpp — the AGC→Vulkan resource layer's front-half↔back-half CONTRACT.
//
// This is the single seam where the two halves of the graphics pipeline meet, defined so neither
// agent edits the other's files:
//   * FRONT-HALF (hle_agc.cpp / hle_graphics.cpp): reverse-engineers the AGC/Gnm resource-creation
//     call (which NID, the guest resource-descriptor layout), fills a ResourceDesc from the guest's
//     arguments, calls resource_create(), and stores the returned handle where the guest expects its
//     GPU-backing pointer (the null `[obj+0x140]` that currently faults at eboot+0xba6e08).
//   * BACK-HALF (this module + vk_translate): owns the descriptor registry and (next increment) the
//     real Vulkan VkImage/VkBuffer + memory + view creation; the draw path binds resources by handle.
//
// The registry here is real (not a fake object graph to trick the game): it tracks genuine resource
// descriptors keyed by a stable handle. Vulkan backing is layered on next — correctness-first,
// built up in verifiable increments rather than fabricated.
#pragma once
#include <cstdint>
#include <cstddef>

namespace prosper::gpu {

enum class ResourceKind : uint32_t {
    Buffer       = 0,   // vertex / index / constant buffer
    Texture2D    = 1,   // sampled image
    RenderTarget = 2,   // color attachment
    DepthTarget  = 3,   // depth/stencil attachment
};

// An AGC resource can serve several roles at once; usage is a bitmask.
enum ResourceUsage : uint32_t {
    Usage_Sampled         = 1u << 0,
    Usage_ColorAttachment = 1u << 1,
    Usage_DepthAttachment = 1u << 2,
    Usage_Storage         = 1u << 3,
    Usage_Vertex          = 1u << 4,
    Usage_Index           = 1u << 5,
};

// Filled by the front-half from the guest's AGC/Gnm resource descriptor. The union of fields needed
// across kinds; unused fields stay 0. `format` is the RAW RDNA2/AGC surface-format enum — the
// back-half maps it to a VkFormat (via vk_translate), so no format guessing crosses the seam.
struct ResourceDesc {
    ResourceKind kind       = ResourceKind::Buffer;
    uint64_t     gpu_addr   = 0;   // guest GPU virtual address of the backing memory (unified CPU/GPU)
    uint64_t     size       = 0;   // byte size (buffers; or total surface bytes)
    uint32_t     width      = 0;
    uint32_t     height     = 0;
    uint32_t     depth      = 1;
    uint32_t     mip_levels = 1;
    uint32_t     array_layers = 1;
    uint32_t     format     = 0;   // raw surface-format enum
    uint32_t     usage      = 0;   // ResourceUsage bitmask
};

using ResourceHandle = uint64_t;   // 0 == invalid

// Create + track a GPU resource. Returns a non-zero, stable handle the front-half stores where the
// guest expects its GPU-backing pointer. Thread-safe (AGC calls may arrive on the game's render
// thread). The Vulkan object is created by the back-half (lazily at first bind, initially deferred).
ResourceHandle resource_create(const ResourceDesc& desc);

// Look up a tracked resource by handle; the back-half draw path binds through this. nullptr if invalid.
const ResourceDesc* resource_get(ResourceHandle h);

// True iff `h` refers to a live resource.
bool resource_valid(ResourceHandle h);

// Number of live resources (stats / tests).
size_t resource_count();

// Test helper: drop all resources and reset handle numbering.
void resource_reset();

} // namespace prosper::gpu

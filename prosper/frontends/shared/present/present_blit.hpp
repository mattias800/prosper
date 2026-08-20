// present_blit.hpp — GPU-side scanout handoff for the unified-device present path (#1270).
//
// Today prosper reads the rendered front-buffer image back to CPU every present and the frontend
// re-uploads it (a 4K GPU->CPU->GPU round-trip). Once prosper-app presents on the SAME VkDevice as the
// renderer (see render_vk_ctx present-capability, #1270), this module removes the round-trip: on the
// render device the renderer blits the front-buffer image into a small pool of renderer-owned "scanout"
// images, and the app blits the chosen scanout image straight to its swapchain.
//
// Ownership / lifetime model (a tiny triple-buffered producer->consumer):
//   - The renderer (guest thread) calls present_blit_publish(front image): it takes a FREE slot, records a
//     blit front->scanout[slot] on the render queue (serialized by the shared present mutex), submits it
//     with a fence and WAITS that fence, then marks the slot PUBLISHED as the newest frame. A previously
//     published-but-untaken slot returns to FREE.
//   - The consumer (app main thread) calls present_blit_acquire(): it takes the PUBLISHED slot (-> IN_FLIGHT)
//     and blits it straight to its swapchain. After its present fence signals (its GPU read of the slot is
//     complete) it calls present_blit_release(slot), returning it to FREE for reuse.
//
// Synchronization needs no semaphore between the two threads. A slot only becomes acquirable AFTER its
// front->scanout blit has fully completed (publish fence-waits it), so the consumer never reads a
// half-written image. A slot is only reused by the producer AFTER the consumer released it, i.e. after the
// consumer's present fence proved its GPU read finished. The per-side fences plus the slot state machine
// therefore fully order producer-write vs consumer-read; concurrent host CALLS to the shared VkQueue are
// serialized by shared_present_submit_mutex(). Everything is on ONE VkDevice (render_vk_ctx) -- no
// queue-family ownership transfers, no external memory. The blit fence-wait is a fast GPU->GPU sync that
// replaces (and is strictly cheaper than) today's GPU->CPU readback + CPU->GPU re-upload.
//
// All functions are no-ops that return false / do nothing when GPU present is unavailable (no render
// device, allocation failure), so the caller always retains the CPU present path as a fallback and the
// headless test/screenshot path is never affected.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

namespace prosper::frontend {

// A published GPU scanout frame handed to the consumer. The image is fully written and stable (in
// TRANSFER_SRC_OPTIMAL) until the consumer releases the slot.
struct GpuScanoutFrame {
    VkImage     image = VK_NULL_HANDLE;      // RGBA8_UNORM, in TRANSFER_SRC_OPTIMAL, ready to read
    uint32_t    width = 0, height = 0;
    uint64_t    frame_seq = 0;               // caller-supplied source identity (guest flip in production)
    int         slot = -1;                   // opaque handle to pass back to present_blit_release
    bool valid() const { return image && width && height && slot >= 0; }
};

// Renderer side (guest thread). Blit `src` (the front-buffer image; it is transitioned to TRANSFER_SRC and
// restored to `src_layout` afterward) into a free scanout slot sized w*h and publish it. Returns false and
// changes nothing if GPU present cannot run this frame (uninitialized device, no free slot, Vulkan error) --
// the caller keeps the CPU present path. `frame_seq` is echoed back in the published frame for identity.
bool present_blit_publish(VkImage src, VkImageLayout src_layout, VkFormat src_format,
                          uint32_t w, uint32_t h, uint64_t frame_seq);

// Consumer side. Fetch the newest published frame not already taken; false if nothing new is available.
// The caller owns the returned slot until it calls present_blit_release(out.slot).
bool present_blit_acquire(GpuScanoutFrame& out);

// Consumer side. Return a slot after the consumer's GPU read of it has completed (its present fence
// signaled). Safe to call with a stale/-1 slot (ignored).
void present_blit_release(int slot);

// Destroy all owned Vulkan resources. The render device must be idle. Used by tests and teardown.
void present_blit_reset();

// True once at least one publish has initialized the pool on a present-capable render device.
bool present_blit_ready();

} // namespace prosper::frontend

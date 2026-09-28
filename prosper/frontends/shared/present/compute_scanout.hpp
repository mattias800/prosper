// compute_scanout.hpp — GPU present for a display buffer that a COMPUTE dispatch writes (#3915).
//
// GPU present (#1270) copies the flipped front buffer into a scanout slot on the GPU. It finds that
// buffer in the renderer's render-target map, which holds only what graphics passes rendered. A
// title whose final composite is a compute dispatch writing the display buffer as a storage image
// has no entry there. Sonic Frontiers (PPSA03831) is one: program 0x20002fe800 writes both 3840x2160
// display buffers as tile-27 RGBA8_UINT storage images. So every one of its flips went through the
// CPU fallback: the compute backend wrote the result back to guest memory, the renderer read the
// 33 MB buffer back out of guest memory and de-swizzled it, and prosper-app uploaded it again.
//
// This module keeps a GPU copy of that compute result so the flip can take the GPU path instead:
//   * The compute backend asks for a mirror image when a storage-image binding is the complete,
//     exact result for a registered VideoOut buffer (compute_scanout_eligible). In the same command
//     buffer as the dispatch it copies the canonical row-major staging result, the very bytes it then
//     retiles into guest memory, into that image.
//   * After the guest writeback has completed, it commits the mirror and arms a guest write watch
//     over the buffer's whole guest footprint.
//   * The renderer's final span, finding no render target for the flipped buffer, publishes the
//     mirror through present_blit_publish when compute_scanout_present_decision allows it.
//
// WHY IT IS PIXEL-IDENTICAL TO THE CPU PATH. The CPU fallback presents the guest buffer's bytes
// de-swizzled with the VideoOut tiling mode, reinterpreted as RGBA8 with no channel conversion. The
// mirror holds the pre-retile linear bytes, byte for byte: a raw buffer-to-image copy into an
// RGBA8_UNORM image, and present_blit copies UNORM to UNORM. So the two agree whenever
// detile(retile(x)) == x. That holds when the compute tile mode is the one VideoOut de-swizzles with,
// and the present decision checks exactly that.
//
// WHY THE WATCH. After the commit, anything that changes the guest bytes makes the mirror stale:
// a guest CPU store (page protection), a GPU or DMA write, a later partial compute write, or a host
// write into dmem. The guest write watch reports all of these. Only `Unchanged` publishes. Windows
// has no page-protection watches, so there the mirror could never be proven current and is not made
// at all (compute_scanout_enabled() is false): those flips keep the CPU path, named no-render-target.
//
// WHY ONLY WHEN THE RENDERER HOLDS NOTHING AT THE FRONT ADDRESS. For ANY renderer entry at the
// flipped address -- including one with no pixels anywhere (a tombstone left by an invalidation) --
// the CPU path refuses to show the guest bytes (guest_scanout_present.hpp: SkipRendererOwnsTarget)
// and keeps the previous frame. Tombstones exist precisely so guest bytes do not stand in, so the
// mirror, which IS those guest bytes, must decline there too (RendererOwnsTarget).
//
// Nothing here runs unless GPU present is active, so headless and screenshot frontends are
// unaffected.
#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>

#include "shared/present/producer_lineage.hpp"

namespace prosper { struct VideoOutBufferSnapshot; }

namespace prosper::frontend {

// ---- Recording-time eligibility (compute thread) -------------------------------------------------

enum class ComputeScanoutEligibility : uint8_t {
    Eligible,
    Inactive,        // no GPU-present consumer
    NotScanout,      // the address is not a registered VideoOut buffer
    NotExactResult,  // not the whole, exact, single-level 2D result of this dispatch
    WrongTexelSize,  // the linear result is not width*height*4 bytes
    DeviceMismatch,  // compute runs on its own device, which cannot share an image with present
};

constexpr const char* compute_scanout_eligibility_name(ComputeScanoutEligibility e) {
    switch (e) {
    case ComputeScanoutEligibility::Eligible: return "eligible";
    case ComputeScanoutEligibility::Inactive: return "inactive";
    case ComputeScanoutEligibility::NotScanout: return "not-scanout";
    case ComputeScanoutEligibility::NotExactResult: return "not-exact-result";
    case ComputeScanoutEligibility::WrongTexelSize: return "wrong-texel-size";
    case ComputeScanoutEligibility::DeviceMismatch: return "device-mismatch";
    }
    return "?";
}

struct ComputeScanoutCandidate {
    bool gpu_present_active = false;
    bool registered_scanout = false;   // gpu_addr equals a registered VideoOut buffer's base
    bool exact_full_result = false;    // the compute backend's exact full-overwrite shape proof
    uint32_t width = 0, height = 0;
    uint64_t linear_bytes = 0;         // canonical row-major staging bytes of the result
    bool shared_device = false;        // compute adopted the renderer's device
};

constexpr ComputeScanoutEligibility compute_scanout_eligible(const ComputeScanoutCandidate& c) {
    if (!c.gpu_present_active) return ComputeScanoutEligibility::Inactive;
    if (!c.registered_scanout) return ComputeScanoutEligibility::NotScanout;
    if (!c.exact_full_result || !c.width || !c.height)
        return ComputeScanoutEligibility::NotExactResult;
    if (c.linear_bytes != static_cast<uint64_t>(c.width) * c.height * 4u)
        return ComputeScanoutEligibility::WrongTexelSize;
    if (!c.shared_device) return ComputeScanoutEligibility::DeviceMismatch;
    return ComputeScanoutEligibility::Eligible;
}

// ---- Present-time decision (renderer, final span) -------------------------------------------------

enum class ComputeScanoutPresent : uint8_t {
    Publish,
    Absent,             // no committed compute result for this address
    Stale,              // the guest bytes may have changed since the commit (watch Dirty)
    Unwatched,          // no watch could be armed (or the platform has none): cannot prove current
    ExtentMismatch,     // the flipped buffer's geometry is not the mirror's
    TileMismatch,       // VideoOut de-swizzles with a different tile mode than compute retiled with
    RendererSource,     // the CPU path would present something prosper rendered instead
    RendererOwnsTarget, // the renderer holds an entry at the front address (even a pixel-less
                        // tombstone): the CPU path would keep the previous frame, not guest bytes
    ScaledPresent,      // PROSPER_RENDER_SCALE: the present extent is not the display extent
};

static_assert(static_cast<int>(ComputeScanoutPresent::RendererOwnsTarget) < 16,
              "compute_scanout.cpp's census keeps sixteen decision slots");

constexpr const char* compute_scanout_present_name(ComputeScanoutPresent p) {
    switch (p) {
    case ComputeScanoutPresent::Publish: return "publish";
    case ComputeScanoutPresent::Absent: return "absent";
    case ComputeScanoutPresent::Stale: return "stale";
    case ComputeScanoutPresent::Unwatched: return "unwatched";
    case ComputeScanoutPresent::ExtentMismatch: return "extent-mismatch";
    case ComputeScanoutPresent::TileMismatch: return "tile-mismatch";
    case ComputeScanoutPresent::RendererSource: return "renderer-source";
    case ComputeScanoutPresent::RendererOwnsTarget: return "renderer-owns-target";
    case ComputeScanoutPresent::ScaledPresent: return "scaled-present";
    }
    return "?";
}

struct ComputeScanoutPresentInputs {
    // The CPU fallback's own priority: prosper's rendered pixels (a selected present source, or a
    // render target cached at any VideoOut buffer) beat the guest buffer. Presenting the compute
    // mirror in those cases would show a different frame than the CPU path would.
    bool have_selected_pixels = false;
    bool renderer_scanout = false;
    bool renderer_owns_front = false;   // any renderer entry at the flipped address, tombstones too
    uint64_t present_extent_bytes = 0;
    uint64_t display_bytes = 0;
    // The committed mirror, if any.
    bool committed = false;
    // The watch's answer: 0 Unchanged, 1 Dirty, 2 Unknown (GuestWriteWatchQuery's order).
    uint8_t watch_state = 2;
    uint32_t mirror_width = 0, mirror_height = 0;
    uint32_t mirror_tile_mode = 0;
    // The flipped buffer.
    uint32_t front_width = 0, front_height = 0;
    uint32_t front_scanout_tile_mode = 0;  // videoout_scanout_tile_mode(front.tiling_mode, 4)
};

constexpr ComputeScanoutPresent compute_scanout_present_decision(
    const ComputeScanoutPresentInputs& in) {
    if (in.renderer_owns_front) return ComputeScanoutPresent::RendererOwnsTarget;
    if (in.have_selected_pixels || in.renderer_scanout) return ComputeScanoutPresent::RendererSource;
    if (!in.present_extent_bytes || in.present_extent_bytes != in.display_bytes)
        return ComputeScanoutPresent::ScaledPresent;
    if (!in.committed) return ComputeScanoutPresent::Absent;
    if (in.mirror_width != in.front_width || in.mirror_height != in.front_height ||
        static_cast<uint64_t>(in.mirror_width) * in.mirror_height * 4u != in.display_bytes)
        return ComputeScanoutPresent::ExtentMismatch;
    if (in.mirror_tile_mode != in.front_scanout_tile_mode) return ComputeScanoutPresent::TileMismatch;
    if (in.watch_state == 1) return ComputeScanoutPresent::Stale;
    if (in.watch_state != 0) return ComputeScanoutPresent::Unwatched;
    return ComputeScanoutPresent::Publish;
}

// ---- Runtime (Vulkan) -----------------------------------------------------------------------------

// A mirror image reserved for one dispatch. Invalid when the reservation failed.
struct ComputeScanoutTarget {
    uint64_t address = 0;
    uint64_t reservation = 0;       // matches begin/commit/abort; a later begin supersedes it
    VkImage image = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    bool valid() const { return address && reservation && image; }
};

// False under PROSPER_NO_COMPUTE_SCANOUT_PRESENT=1, the same-binary A/B control that restores the
// CPU fallback for compute-written display buffers, and always false on Windows (no watches).
bool compute_scanout_enabled();

// Compute thread, once per dispatch that may carry a mirror: destroys the mirrors (and watches) of
// addresses that are no longer registered VideoOut buffers. `registered` holds the current bases.
void compute_scanout_retain_registered(const uint64_t* registered, size_t count);

// Whether `device` is the renderer's (and so the present path's) device.
bool compute_scanout_device_shared(VkDevice device);

// Compute thread, while recording a dispatch. Drops any committed mirror at `address` (the dispatch
// is about to replace those bytes; a null device ONLY does that) and returns an image of `width` x `height` to copy into. Returns
// an invalid target if the renderer device is missing or `device` is not it, or allocation fails.
ComputeScanoutTarget compute_scanout_begin(VkDevice device, uint64_t address, uint32_t width,
                                           uint32_t height);

// Records, into the dispatch's command buffer: staging (written by the shader or a transfer) ->
// mirror image, leaving the image in TRANSFER_SRC_OPTIMAL for present_blit_publish.
void compute_scanout_record_copy(VkCommandBuffer command, VkBuffer staging,
                                 const ComputeScanoutTarget& target);

// After the dispatch's fence AND its guest writeback have completed. Arms a watch over the guest
// footprint [address, address + guest_bytes) and makes the mirror publishable. `tile_mode` is the
// compute resource's tile mode, the one its writeback retiled with.
void compute_scanout_commit(const ComputeScanoutTarget& target, uint64_t guest_bytes,
                            uint32_t tile_mode, uint64_t source_submit);

// The dispatch failed or its result was not published. The mirror stays unpublishable.
void compute_scanout_abort(const ComputeScanoutTarget& target);

struct ComputeScanoutPublishResult {
    ComputeScanoutPresent decision = ComputeScanoutPresent::Absent;
    bool published = false;         // present_blit_publish accepted the frame
};

// Renderer, final render span of a flip whose front buffer GPU present could not publish from a
// render target (none, or one not GPU-resident). Evaluates
// compute_scanout_present_decision (the caller supplies the renderer-priority inputs) and, on
// Publish, hands the mirror to present_blit_publish.
ComputeScanoutPublishResult compute_scanout_publish(const VideoOutBufferSnapshot& front,
                                                    uint64_t front_flip,
                                                    const ComputeScanoutPresentInputs& renderer);

// The run totals the exit census prints (`[compute-scanout] RUN TOTAL`).
struct ComputeScanoutTotals { uint64_t reserved = 0, committed = 0, retired = 0, published = 0; };
ComputeScanoutTotals compute_scanout_totals();

// Whether a committed mirror exists for `address` (diagnostics and tests).
bool compute_scanout_committed(uint64_t address);

// Test-only: destroy every mirror (device idle is the caller's responsibility).
void compute_scanout_reset_for_test();

} // namespace prosper::frontend

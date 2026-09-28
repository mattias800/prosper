#pragma once

#include <cstdint>
#include <limits>
#include <vulkan/vulkan.h>

namespace prosper::frontend {

// A scanout slot is publishable only after its copy fence is known to be signaled. Timeouts and
// device errors leave the submitted command buffer potentially in flight.
constexpr bool present_blit_wait_completed(VkResult result) {
    return result == VK_SUCCESS;
}

// Render submissions build the next display image in several ordered pieces, but VideoOut exposes
// it only when the guest reaches SetFlip. Publishing every intermediate submission both presents
// partially assembled frames and needlessly synchronizes the renderer with the window system.
// UINT64_MAX is the initial sentinel so a title that renders before its first flip still gets one
// recoverable scanout.
constexpr bool present_blit_has_new_flip(uint64_t last_published_flip,
                                         uint64_t current_flip) {
    return last_published_flip == std::numeric_limits<uint64_t>::max() ||
           last_published_flip != current_flip;
}

// GPU scanout images and CPU readback fallbacks are produced by different paths, but both carry
// the guest flip that they represent. Never let an older fallback replace a frame that has already
// reached the window; equal identities are duplicate representations of the same guest frame.
constexpr bool present_source_is_newer(bool have_presented_source,
                                       uint64_t last_presented_flip,
                                       uint64_t candidate_flip) {
    return !have_presented_source || candidate_flip > last_presented_flip;
}

// GPU scanout frames acquired from the ring buffer represent newly completed GPU blits.
// A GPU publication is displayable if it is newer than or updates the currently displayed
// guest flip. Only strictly older flips (reordered/stale) are rejected.
constexpr bool gpu_present_frame_is_newer(bool have_presented_source,
                                          uint64_t last_presented_flip,
                                          uint64_t candidate_flip) {
    return !have_presented_source || candidate_flip >= last_presented_flip;
}

// Why a final render span did NOT hand the flipped front buffer to GPU present (#3915). Every
// decline sends that frame through the CPU fallback (a full-frame readback plus re-upload), so the
// reason is worth naming: "gpu-scanout=0" alone cannot say which of these checks refused it.
enum class GpuPresentOutcome : uint8_t {
    Published,            // present_blit_publish copied the front image into a scanout slot
    SameFlip,             // this flip was already published; not a decline
    Inactive,             // no GPU-present consumer (headless/test/screenshot); not a decline
    CaptureNeedsCpu,      // a pending one-shot capture needs the CPU frame (#3895)
    NoFront,              // VideoOut has no selected front buffer
    NoFlipIdentity,       // the front has no originating flip
    NoRenderTarget,       // the renderer holds no target at the front buffer's address, and no
                          // compute dispatch left a mirror of it either
    VolumeTarget,         // the target there is a 3D volume
    NotGpuResident,       // the target is known, but its current pixels are not on the GPU
    NoPersistentImage,    // no persistent colour image matches the target's extent and format
    UndefinedLayout,      // the persistent image has never been written
    PublishFailed,        // present_blit_publish itself declined (see its handoff trace events)
    // No render target, and the front buffer's compute-written mirror (compute_scanout.hpp) could
    // not stand in for it either. One per ComputeScanoutPresent decline reason.
    ComputeScanoutStale,          // the guest bytes changed after the compute result was mirrored
    ComputeScanoutUnwatched,      // no guest write watch could be armed over the buffer
    ComputeScanoutExtentMismatch, // the flipped buffer's geometry is not the mirror's
    ComputeScanoutTileMismatch,   // VideoOut de-swizzles with another tile mode than compute wrote
    ComputeScanoutRendererSource, // the CPU path would present prosper's rendered pixels instead
    ComputeScanoutScaled,         // PROSPER_RENDER_SCALE: present extent is not the display extent
    Count,
};

constexpr const char* gpu_present_outcome_name(GpuPresentOutcome o) {
    switch (o) {
    case GpuPresentOutcome::Published: return "published";
    case GpuPresentOutcome::SameFlip: return "same-flip";
    case GpuPresentOutcome::Inactive: return "inactive";
    case GpuPresentOutcome::CaptureNeedsCpu: return "capture-needs-cpu";
    case GpuPresentOutcome::NoFront: return "no-front";
    case GpuPresentOutcome::NoFlipIdentity: return "no-flip-identity";
    case GpuPresentOutcome::NoRenderTarget: return "no-render-target";
    case GpuPresentOutcome::VolumeTarget: return "volume-target";
    case GpuPresentOutcome::NotGpuResident: return "not-gpu-resident";
    case GpuPresentOutcome::NoPersistentImage: return "no-persistent-image";
    case GpuPresentOutcome::UndefinedLayout: return "undefined-layout";
    case GpuPresentOutcome::PublishFailed: return "publish-failed";
    case GpuPresentOutcome::ComputeScanoutStale: return "compute-scanout-stale";
    case GpuPresentOutcome::ComputeScanoutUnwatched: return "compute-scanout-unwatched";
    case GpuPresentOutcome::ComputeScanoutExtentMismatch: return "compute-scanout-extent-mismatch";
    case GpuPresentOutcome::ComputeScanoutTileMismatch: return "compute-scanout-tile-mismatch";
    case GpuPresentOutcome::ComputeScanoutRendererSource: return "compute-scanout-renderer-source";
    case GpuPresentOutcome::ComputeScanoutScaled: return "compute-scanout-scaled";
    case GpuPresentOutcome::Count: break;
    }
    return "?";
}

// A decline is an outcome that sends a frame the GPU-present consumer is waiting for through the
// CPU fallback. Published/SameFlip succeeded, and Inactive has no consumer to fall back for.
constexpr bool gpu_present_outcome_is_decline(GpuPresentOutcome o) {
    return o != GpuPresentOutcome::Published && o != GpuPresentOutcome::SameFlip &&
           o != GpuPresentOutcome::Inactive && o != GpuPresentOutcome::Count;
}

} // namespace prosper::frontend

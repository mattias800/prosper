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

} // namespace prosper::frontend

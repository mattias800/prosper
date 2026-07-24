#pragma once

#include <vulkan/vulkan.h>

namespace prosper::frontend {

// A scanout slot is publishable only after its copy fence is known to be signaled. Timeouts and
// device errors leave the submitted command buffer potentially in flight.
constexpr bool present_blit_wait_completed(VkResult result) {
    return result == VK_SUCCESS;
}

} // namespace prosper::frontend

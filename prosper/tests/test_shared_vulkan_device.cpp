// test_shared_vulkan_device — the live renderer must PUBLISH its Vulkan device so the compute
// backend can adopt it (#1091 phase 1).
//
// This guards a failure mode that is otherwise invisible: without it, deleting the publish call
// leaves every other test green while the optimization silently reverts to two devices and the
// host round trip it exists to remove. The published handles must also be the renderer's own, since
// adopting a different device would be worse than not adopting at all.
#include "../src/gpu/gpu_execute.hpp"
#include "render_runner.h"
#include "../frontends/shared/live_renderer.hpp"
#include <string>
#include <cstdio>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else         { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    // Nothing published before the renderer initializes: headless compute-only use must keep
    // creating its own device (tests/test_game_compute.cpp depends on this).
    CHECK(!prosper::gpu::shared_vulkan_context().valid(),
          "no shared context is published before the renderer initializes");

    // THE load-bearing assertion: registering the live renderer must publish, WITHOUT anyone having
    // called render_vk_ctx() first. Compute initializes lazily on its first dispatch and titles
    // routinely dispatch before their first draw, so if publication waited for the first draw the
    // compute device would win the race and the two would never share. Calling render_vk_ctx()
    // directly here would hide that: it publishes internally, so the check must come first.
    prosper::frontend::register_live_renderer(std::string(), false);
    CHECK(prosper::gpu::shared_vulkan_context().valid() ||
              !prosper::test::render_vk_ctx().ok,
          "register_live_renderer() publishes the shared context (or no device exists at all)");

    const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
    if (!ctx.ok) {
        printf("test_shared_vulkan_device: no Vulkan device available, skipping\n");
        return 0;                       // CI images without a usable ICD skip rather than fail
    }

    const prosper::gpu::SharedVulkanContext shared = prosper::gpu::shared_vulkan_context();
    CHECK(shared.valid(), "renderer publishes a valid shared context once initialized");
    CHECK(shared.device == static_cast<void*>(ctx.dev), "published device is the renderer's device");
    CHECK(shared.queue == static_cast<void*>(ctx.queue), "published queue is the renderer's queue");
    CHECK(shared.physical == static_cast<void*>(ctx.phys), "published physical device matches");
    CHECK(shared.instance == static_cast<void*>(ctx.inst), "published instance matches");
    CHECK(shared.queue_family == ctx.qfi, "published queue family matches");

    // The compute backend only adopts when the storage-image features are enabled; a device without
    // them must publish them as false so compute declines rather than emitting illegal capabilities.
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(ctx.phys, &supported);
    CHECK(shared.storage_image_read_without_format ==
              (supported.shaderStorageImageReadWithoutFormat == VK_TRUE),
          "published storage-image READ flag reflects the device");
    CHECK(shared.storage_image_write_without_format ==
              (supported.shaderStorageImageWriteWithoutFormat == VK_TRUE),
          "published storage-image WRITE flag reflects the device");

    // Adopting a queue family without COMPUTE would make every dispatch invalid usage.
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &count, props.data());
    CHECK(shared.queue_family < count &&
          (props[shared.queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0,
          "published queue family supports COMPUTE (dispatches are legal on it)");

    printf(fails ? "test_shared_vulkan_device: %d FAILURE(S)\n"
                 : "test_shared_vulkan_device: all ok\n", fails);
    return fails ? 1 : 0;
}

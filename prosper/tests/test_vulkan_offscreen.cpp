// test_vulkan_offscreen — foundation for the graphics phase + its agentic-first verification harness.
// Proves the build can drive Vulkan headlessly: create an instance+device (llvmpipe/RADV/any),
// allocate an offscreen image, clear it to a known color on the GPU, copy it back to host memory,
// and assert the pixels + a CRC. This is the "render offscreen, hash the framebuffer, compare to a
// golden value" pattern every later graphics milestone (AGC->PM4->Vulkan) will verify against — no
// window, no eyeballing. It intentionally uses vkCmdClearColorImage (no pipeline/shaders yet) so it
// stays a pure environment/harness smoke test.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)
#define VKCHECK(e, m) CHECK((e) == VK_SUCCESS, m)

static uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) { c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (~(c & 1) + 1)); }
    return ~c;
}

int main() {
    printf("== test_vulkan_offscreen ==\n");
    const uint32_t W = 64, H = 64;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "prosper-gfx-harness"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    VKCHECK(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
    if (!inst) { printf("== FAIL: no instance ==\n"); return 1; }

    uint32_t ndev = 0; vkEnumeratePhysicalDevices(inst, &ndev, nullptr);
    CHECK(ndev > 0, "at least one physical device");
    if (!ndev) { printf("== FAIL: no device ==\n"); return 1; }
    std::vector<VkPhysicalDevice> devs(ndev);
    vkEnumeratePhysicalDevices(inst, &ndev, devs.data());
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    printf("  device: %s (Vulkan %u.%u)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));

    // A queue family that supports graphics (implies transfer/clear).
    uint32_t nqf = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nqf, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfi = i; break; }
    CHECK(qfi != UINT32_MAX, "found a graphics queue family");
    if (qfi == UINT32_MAX) { printf("== FAIL ==\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice dev = VK_NULL_HANDLE;
    VKCHECK(vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice");
    if (!dev) { printf("== FAIL ==\n"); return 1; }
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    auto pickMem = [&](uint32_t typeBits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((typeBits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    };

    // Offscreen image (RGBA8), used as clear target + transfer source.
    VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgci.extent = {W, H, 1}; imgci.mipLevels = 1; imgci.arrayLayers = 1;
    imgci.samples = VK_SAMPLE_COUNT_1_BIT; imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage img; VKCHECK(vkCreateImage(dev, &imgci, nullptr, &img), "vkCreateImage");
    VkMemoryRequirements ireq; vkGetImageMemoryRequirements(dev, img, &ireq);
    VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    iai.allocationSize = ireq.size; iai.memoryTypeIndex = pickMem(ireq.memoryTypeBits, 0);
    VkDeviceMemory imem; VKCHECK(vkAllocateMemory(dev, &iai, nullptr, &imem), "alloc image memory");
    vkBindImageMemory(dev, img, imem, 0);

    // Host-visible readback buffer (W*H*4 bytes).
    VkDeviceSize bytes = (VkDeviceSize)W * H * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf; VKCHECK(vkCreateBuffer(dev, &bci, nullptr, &buf), "vkCreateBuffer");
    VkMemoryRequirements breq; vkGetBufferMemoryRequirements(dev, buf, &breq);
    VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    bai.allocationSize = breq.size;
    bai.memoryTypeIndex = pickMem(breq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem; VKCHECK(vkAllocateMemory(dev, &bai, nullptr, &bmem), "alloc buffer memory");
    vkBindBufferMemory(dev, buf, bmem, 0);

    // Command buffer: transition->clear->transition->copy-to-buffer.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &cbai, &cb);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &cbbi);
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    auto barrier = [&](VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from; b.newLayout = to; b.srcAccessMask = sa; b.dstAccessMask = da;
        b.image = img; b.subresourceRange = range;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    // Clear to (0x40, 0x80, 0xC0, 0xFF) — a known, distinctive color.
    VkClearColorValue clear; clear.float32[0] = 0x40/255.0f; clear.float32[1] = 0x80/255.0f;
    clear.float32[2] = 0xC0/255.0f; clear.float32[3] = 1.0f;
    vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &copy);
    vkEndCommandBuffer(cb);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; vkCreateFence(dev, &fci, nullptr, &fence);
    VKCHECK(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");
    VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000), "vkWaitForFences");

    void* mapped = nullptr; vkMapMemory(dev, bmem, 0, bytes, 0, &mapped);
    const uint8_t* px = (const uint8_t*)mapped;
    CHECK(px[0] == 0x40 && px[1] == 0x80 && px[2] == 0xC0 && px[3] == 0xFF,
          "pixel[0] == clear color (GPU cleared + read back correctly)");
    // Every pixel identical -> a stable CRC we can use as a golden value for future render tests.
    bool uniform = true;
    for (uint32_t i = 1; i < W * H; i++)
        if (memcmp(px, px + i * 4, 4) != 0) { uniform = false; break; }
    CHECK(uniform, "all pixels equal the clear color");
    printf("  framebuffer CRC32 = 0x%08x\n", crc32(px, (size_t)bytes));
    vkUnmapMemory(dev, bmem);

    vkDestroyFence(dev, fence, nullptr); vkDestroyCommandPool(dev, pool, nullptr);
    vkDestroyBuffer(dev, buf, nullptr); vkFreeMemory(dev, bmem, nullptr);
    vkDestroyImage(dev, img, nullptr); vkFreeMemory(dev, imem, nullptr);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}

// vkval sync_probe — the positive control for `vk_validation_scan.py --sync`.
//
// WHY THIS EXISTS
//
// Synchronization validation is a SEPARATE check set from the layer's default (core) checks, and it
// is switched on through a layer setting rather than by loading anything extra. So a scan that asks
// for it and does not get it — an older layer that does not know the setting's name, a spelling
// change, a loader that drops the environment — produces exactly the output of a clean sync run:
// nothing. That is the same silent-green this whole tool was built around (see the scanner's header
// on `report_flags` and on the parser that read 0 of 187 messages), one layer setting further in.
//
// Proving the LAYER loaded, which the scanner already does, does not prove syncval is armed: the
// layer loads either way. So this program deliberately commits ONE synchronization hazard — two
// transfer writes to the same buffer range with no memory dependency between them — and the scanner
// requires SYNC-HAZARD-WRITE-AFTER-WRITE to come back before it will report a sync verdict.
//
// It is NOT a ctest and must never become one: it exists to be invalid, and the gate it feeds fails
// on any hazard the ledger does not carry. It is run directly by the scanner, out of band, exactly
// like the layer-load probe.
//
// Exit status: 0 when the hazard was recorded and submitted, 2 when Vulkan could not be brought up.
// A setup failure is never reported as a successful provocation — a control that could not run must
// not read as a control that ran.
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>

namespace {

int setup_failure(const char* what) {
    std::printf("[sync_probe] SETUP FAILED: %s\n", what);
    return 2;
}

}  // namespace

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vkval_sync_probe";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
        return setup_failure("vkCreateInstance");

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (!device_count) return setup_failure("no Vulkan physical device");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    device_count = 1;
    vkEnumeratePhysicalDevices(instance, &device_count, &physical);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    VkQueueFamilyProperties families[16]{};
    if (family_count > 16) family_count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) { family = i; break; }
    if (family == UINT32_MAX) return setup_failure("no transfer-capable queue family");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS)
        return setup_failure("vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    constexpr VkDeviceSize kBytes = 4096;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = kBytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS)
        return setup_failure("vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
        if (requirements.memoryTypeBits & (1u << i)) { memory_type = i; break; }
    if (memory_type == UINT32_MAX) return setup_failure("no usable memory type");
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = requirements.size; mai.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &mai, nullptr, &memory) != VK_SUCCESS)
        return setup_failure("vkAllocateMemory");
    vkBindBufferMemory(device, buffer, memory, 0);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &pci, nullptr, &pool) != VK_SUCCESS)
        return setup_failure("vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &cbai, &command) != VK_SUCCESS)
        return setup_failure("vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command, &cbbi);
    // THE PROVOCATION, and the only reason this file exists. Two transfer writes over the same
    // range, back to back, with no memory dependency between them. Core validation is silent on it;
    // synchronization validation reports SYNC-HAZARD-WRITE-AFTER-WRITE.
    vkCmdFillBuffer(command, buffer, 0, kBytes, 0x11111111u);
    vkCmdFillBuffer(command, buffer, 0, kBytes, 0x22222222u);
    vkEndCommandBuffer(command);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &command;
    if (vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
        return setup_failure("vkQueueSubmit");
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("[sync_probe] recorded and submitted one deliberate write-after-write\n");
    return 0;
}

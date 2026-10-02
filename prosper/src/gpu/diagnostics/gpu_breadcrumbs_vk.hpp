// gpu_breadcrumbs_vk.hpp -- writes the markers gpu_breadcrumbs.hpp resolves, and reads them back.
//
// OFF BY DEFAULT, AND FREE WHEN OFF. Nothing here runs unless `PROSPER_GPU_BREADCRUMBS` is set: the
// device extensions are not requested, no marker buffer exists, and `begin`/`end` return after one
// relaxed atomic load. The markers cost GPU time (a pipeline-stage write per draw), which is why this is
// a switch and not an always-on instrument like the memory budget.
//
// TWO HARDWARE PATHS, ONE VERDICT.
//   * VK_AMD_buffer_marker: a 32-bit write into a host-visible buffer at TOP_OF_PIPE before the site and
//     BOTTOM_OF_PIPE after it, to two fixed slots. After a loss the slots hold the last marker that
//     executed. Needs a real marker buffer.
//   * VK_NV_device_diagnostic_checkpoints: vkCmdSetCheckpointNV with the same values; after the loss
//     vkGetQueueCheckpointDataNV lists the checkpoints reached. No buffer.
// Both reduce to "last started, last finished", which gpu_breadcrumbs.hpp turns into a window of sites.
// VK_EXT_device_fault, when enabled, adds the driver's own account (faulting address, vendor fault
// code) beside it. Each piece is optional: a device with none of the three simply never arms.
//
// THE DISPATCH TABLE IS INJECTABLE on purpose. Stock lavapipe exposes neither marker extension, so CI
// can never run this against a real driver; the unit test substitutes recording functions and asserts
// the exact sequence of writes. That proves the emitter writes what the resolver expects and says
// NOTHING about whether a real driver executes those writes before a hang -- that is a local run on
// AMD or NV hardware and is recorded in the PR, never claimed from CI.
#pragma once

#include "diagnostics/env_cache.hpp"
#include "gpu/diagnostics/gpu_breadcrumbs.hpp"
#include "gpu/diagnostics/gpu_memory_budget_vk.hpp"
#include "gpu/memory/memory_type_select.hpp"

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace prosper::gpu {

enum class BreadcrumbMode : uint8_t { off = 0, amd_buffer_marker = 1, nv_checkpoint = 2 };

struct BreadcrumbDispatch {
    PFN_vkCmdWriteBufferMarkerAMD cmd_write_buffer_marker = nullptr;
    PFN_vkCmdSetCheckpointNV cmd_set_checkpoint = nullptr;
    PFN_vkGetQueueCheckpointDataNV get_queue_checkpoint_data = nullptr;
    PFN_vkGetDeviceFaultInfoEXT get_device_fault_info = nullptr;
};

// The one switch. Cached: the device extensions are chosen once, at device creation, so a later write
// to the variable could not take effect and must not appear to.
inline bool breadcrumbs_requested() { return PROSPER_ENV_ON("PROSPER_GPU_BREADCRUMBS"); }

class BreadcrumbEmitter {
public:
    // `slots` points at >= 2 host-visible, host-coherent, zero-initialised uint32 values bound to
    // `buffer` (AMD only; null for NV). Slot 0 is the last "before" marker, slot 1 the last "after".
    void configure(BreadcrumbMode mode, const BreadcrumbDispatch& dispatch, VkDevice device,
                   VkBuffer buffer, volatile uint32_t* slots, bool device_fault_enabled) {
        dispatch_ = dispatch;
        device_ = device;
        buffer_ = buffer;
        slots_ = slots;
        fault_enabled_ = device_fault_enabled && dispatch.get_device_fault_info != nullptr;
        mode_.store(mode, std::memory_order_release);
    }
    void disable() { mode_.store(BreadcrumbMode::off, std::memory_order_release); }

    BreadcrumbMode mode() const { return mode_.load(std::memory_order_acquire); }
    bool active() const { return mode() != BreadcrumbMode::off; }

    // Records the BEFORE marker for `site` into `command` and returns its id; 0 when inactive, so a
    // caller can pass the result straight to end() without testing it.
    uint32_t begin(VkCommandBuffer command, const BreadcrumbSite& site) {
        const BreadcrumbMode current = mode();
        if (current == BreadcrumbMode::off) return 0;
        const uint32_t id = breadcrumb_table().begin_site(site);
        write(command, current, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, /*slot=*/0,
              breadcrumb_before_value(id));
        return id;
    }
    // Records the AFTER marker for the site `begin` returned. A 0 id is a no-op.
    void end(VkCommandBuffer command, uint32_t id) {
        if (id == 0) return;
        const BreadcrumbMode current = mode();
        if (current == BreadcrumbMode::off) return;
        write(command, current, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, /*slot=*/1,
              breadcrumb_after_value(id));
    }

    struct Readback {
        bool ok = false;           // false: nothing could be read (inactive, or no entry point)
        uint32_t started = 0;      // raw "before" value, as the verdict expects it
        uint32_t finished = 0;     // raw "after" value
    };

    // Read the last markers that reached memory. Valid after VK_ERROR_DEVICE_LOST; `queue` is the queue
    // the instrumented work was submitted to (NV reads checkpoints per queue).
    Readback read_back(VkQueue queue) const {
        Readback out;
        switch (mode()) {
        case BreadcrumbMode::off:
            return out;
        case BreadcrumbMode::amd_buffer_marker:
            if (!slots_) return out;
            out.started = slots_[0];
            out.finished = slots_[1];
            out.ok = true;
            return out;
        case BreadcrumbMode::nv_checkpoint: {
            if (!dispatch_.get_queue_checkpoint_data || queue == VK_NULL_HANDLE) return out;
            uint32_t count = 0;
            dispatch_.get_queue_checkpoint_data(queue, &count, nullptr);
            std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
            if (count) dispatch_.get_queue_checkpoint_data(queue, &count, data.data());
            std::vector<uint32_t> values;
            for (uint32_t i = 0; i < count && i < data.size(); ++i)
                values.push_back(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker)));
            reduce_checkpoint_values(values.data(), values.size(), out.started, out.finished);
            out.ok = true;
            return out;
        }
        }
        return out;
    }

    // The full report for a device loss: the verdict line, then the driver's own fault account when
    // VK_EXT_device_fault is enabled. When the emitter is inactive it says so and how to arm it, rather
    // than staying silent: an absent line reads as "nothing to report", which is the wrong conclusion
    // about a run that simply was not instrumented (docs/DIAGNOSTIC_GATE_AUDIT.md, rule 1).
    std::string report_device_loss(VkQueue queue) const {
        if (!active())
            return "[gpu-breadcrumb] not armed: re-run with PROSPER_GPU_BREADCRUMBS=1 to learn where "
                   "the GPU stopped (needs VK_AMD_buffer_marker or VK_NV_device_diagnostic_checkpoints)\n";
        std::string out;
        const Readback read = read_back(queue);
        if (!read.ok) {
            out = "[gpu-breadcrumb] active, but the markers could not be read back\n";
        } else {
            out = format_breadcrumb_verdict(
                      resolve_breadcrumbs(breadcrumb_table(), read.started, read.finished),
                      read.started, read.finished) + "\n";
        }
        if (fault_enabled_) out += device_fault_report();
        return out;
    }

private:
    void write(VkCommandBuffer command, BreadcrumbMode mode, VkPipelineStageFlagBits stage,
               uint32_t slot, uint32_t value) {
        if (mode == BreadcrumbMode::amd_buffer_marker) {
            if (dispatch_.cmd_write_buffer_marker)
                dispatch_.cmd_write_buffer_marker(command, stage, buffer_,
                                                  static_cast<VkDeviceSize>(slot) * sizeof(uint32_t),
                                                  value);
        } else if (mode == BreadcrumbMode::nv_checkpoint) {
            if (dispatch_.cmd_set_checkpoint)
                dispatch_.cmd_set_checkpoint(
                    command, reinterpret_cast<const void*>(static_cast<uintptr_t>(value)));
        }
    }

    // VK_EXT_device_fault: two calls (counts, then data). Numbers and the driver's own description
    // only -- no vendor binary blob, which can be large and is meaningful only to the vendor's tools.
    std::string device_fault_report() const {
        VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        if (dispatch_.get_device_fault_info(device_, &counts, nullptr) != VK_SUCCESS)
            return "[gpu-fault] VK_EXT_device_fault reported no information\n";
        std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
        VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
        info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
        info.pVendorInfos = vendors.empty() ? nullptr : vendors.data();
        if (dispatch_.get_device_fault_info(device_, &counts, &info) != VK_SUCCESS)
            return "[gpu-fault] VK_EXT_device_fault reported no information\n";
        char line[320];
        std::string out;
        std::snprintf(line, sizeof line, "[gpu-fault] driver: %.*s (%u address record(s), %u vendor record(s))\n",
                      static_cast<int>(sizeof info.description), info.description,
                      counts.addressInfoCount, counts.vendorInfoCount);
        out += line;
        for (const VkDeviceFaultAddressInfoEXT& a : addresses) {
            std::snprintf(line, sizeof line,
                          "[gpu-fault]   address type=%d reported=0x%llx precision=0x%llx\n",
                          static_cast<int>(a.addressType),
                          static_cast<unsigned long long>(a.reportedAddress),
                          static_cast<unsigned long long>(a.addressPrecision));
            out += line;
        }
        for (const VkDeviceFaultVendorInfoEXT& v : vendors) {
            std::snprintf(line, sizeof line,
                          "[gpu-fault]   vendor code=0x%llx data=0x%llx %.*s\n",
                          static_cast<unsigned long long>(v.vendorFaultCode),
                          static_cast<unsigned long long>(v.vendorFaultData),
                          static_cast<int>(sizeof v.description), v.description);
            out += line;
        }
        return out;
    }

    std::atomic<BreadcrumbMode> mode_{BreadcrumbMode::off};
    BreadcrumbDispatch dispatch_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    volatile uint32_t* slots_ = nullptr;
    bool fault_enabled_ = false;
};

// The process-wide emitter: the shipped renderer and the live compute backend both record through it.
inline BreadcrumbEmitter& breadcrumb_emitter() {
    static BreadcrumbEmitter emitter;
    return emitter;
}

// Which of the three extensions the device advertised, filled as the extension list is walked and read
// back when the device is created.
struct BreadcrumbDeviceSupport {
    bool amd_buffer_marker = false;
    bool nv_checkpoint = false;
    bool device_fault = false;   // advertised AND its feature bit is set

    // Returns the canonical (static-lifetime) extension name to enable, or nullptr when `name` is not
    // one of ours. Static literals, because the caller stores the pointer until vkCreateDevice and the
    // enumeration vector it came from may be gone by then.
    const char* note_extension(const char* name) {
        if (!std::strcmp(name, VK_AMD_BUFFER_MARKER_EXTENSION_NAME)) {
            amd_buffer_marker = true;
            return VK_AMD_BUFFER_MARKER_EXTENSION_NAME;
        }
        if (!std::strcmp(name, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)) {
            nv_checkpoint = true;
            return VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
        }
        return nullptr;
    }
    // Which path to arm, if any. NV checkpoints need no buffer, so they win when both are advertised.
    BreadcrumbMode chosen() const {
        if (nv_checkpoint) return BreadcrumbMode::nv_checkpoint;
        if (amd_buffer_marker) return BreadcrumbMode::amd_buffer_marker;
        return BreadcrumbMode::off;
    }
};

// Loads the entry points, creates the AMD marker buffer when that path is chosen, and arms the process-
// wide emitter. Call once, right after vkCreateDevice, only when breadcrumbs_requested() and the device
// was created with the extensions `support` recorded. Returns the mode armed; `off` means a requirement
// was not met and the reason was printed, so a run that asked for breadcrumbs and got none says so.
//
// The marker buffer and its memory live for the process: a handful of bytes, and freeing them would
// race the very device loss they exist to survive.
inline BreadcrumbMode breadcrumb_arm_device(VkDevice device, VkPhysicalDevice physical,
                                            const BreadcrumbDeviceSupport& support,
                                            bool fault_enabled, const char* who) {
    const BreadcrumbMode mode = support.chosen();
    if (mode == BreadcrumbMode::off) {
        std::fprintf(stderr,
                     "[gpu-breadcrumb] %s: requested, but the device advertises neither "
                     "VK_AMD_buffer_marker nor VK_NV_device_diagnostic_checkpoints; not armed\n", who);
        return BreadcrumbMode::off;
    }
    BreadcrumbDispatch dispatch;
    dispatch.cmd_write_buffer_marker = reinterpret_cast<PFN_vkCmdWriteBufferMarkerAMD>(
        vkGetDeviceProcAddr(device, "vkCmdWriteBufferMarkerAMD"));
    dispatch.cmd_set_checkpoint = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(
        vkGetDeviceProcAddr(device, "vkCmdSetCheckpointNV"));
    dispatch.get_queue_checkpoint_data = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(
        vkGetDeviceProcAddr(device, "vkGetQueueCheckpointDataNV"));
    if (fault_enabled)
        dispatch.get_device_fault_info = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
            vkGetDeviceProcAddr(device, "vkGetDeviceFaultInfoEXT"));

    VkBuffer buffer = VK_NULL_HANDLE;
    volatile uint32_t* slots = nullptr;
    if (mode == BreadcrumbMode::nv_checkpoint) {
        if (!dispatch.cmd_set_checkpoint || !dispatch.get_queue_checkpoint_data) {
            std::fprintf(stderr, "[gpu-breadcrumb] %s: NV checkpoint entry points missing; not armed\n", who);
            return BreadcrumbMode::off;
        }
    } else {
        if (!dispatch.cmd_write_buffer_marker) {
            std::fprintf(stderr, "[gpu-breadcrumb] %s: vkCmdWriteBufferMarkerAMD missing; not armed\n", who);
            return BreadcrumbMode::off;
        }
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = 2 * sizeof(uint32_t);
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;   // required of a vkCmdWriteBufferMarkerAMD target
        if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS) {
            std::fprintf(stderr, "[gpu-breadcrumb] %s: marker buffer creation failed; not armed\n", who);
            return BreadcrumbMode::off;
        }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buffer, &req);
        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(physical, &props);
        const uint32_t type = select_memory_type(
            props, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        if (type == UINT32_MAX || allocate_device_memory(device, &mai, &memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS ||
            vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped) {
            std::fprintf(stderr, "[gpu-breadcrumb] %s: marker memory setup failed; not armed\n", who);
            return BreadcrumbMode::off;
        }
        slots = static_cast<volatile uint32_t*>(mapped);
        slots[0] = 0;
        slots[1] = 0;
    }
    breadcrumb_emitter().configure(mode, dispatch, device, buffer, slots, fault_enabled);
    std::fprintf(stderr, "[gpu-breadcrumb] %s: ARMED via %s%s\n", who,
                 mode == BreadcrumbMode::nv_checkpoint ? "VK_NV_device_diagnostic_checkpoints"
                                                       : "VK_AMD_buffer_marker",
                 dispatch.get_device_fault_info ? " + VK_EXT_device_fault" : "");
    return mode;
}

} // namespace prosper::gpu

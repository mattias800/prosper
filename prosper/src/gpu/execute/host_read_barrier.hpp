// host_read_barrier.hpp — the availability half of a GPU->CPU readback, shared by every backend.
//
// Waiting a fence -- or vkQueueWaitIdle -- orders EXECUTION. It does not perform the availability
// operation that moves a transfer (or shader, or transform-feedback) write into the HOST domain.
// Vulkan requires an explicit dependency for that: dstStageMask = VK_PIPELINE_STAGE_HOST_BIT with
// dstAccessMask = VK_ACCESS_HOST_READ_BIT (#2944, #3249).
//
// Coherent memory does not exempt a readback from it. HOST_COHERENT removes the need for
// vkInvalidateMappedMemoryRanges; it does not remove the need for the dependency. So the two halves
// are independent: non-coherent memory needs both, coherent memory needs this one. The invalidate
// half lives with whichever allocator can hand out non-coherent memory (today only
// `invalidate_mapped_readback` in tests/fixtures/render_runner.h); this half is universal, which is
// why it lives here rather than beside it.
//
// Every driver this project runs on completes the availability anyway, which is exactly why the gap
// survived unnoticed -- the pixels and the dispatch results come back correct and nothing says the
// code was wrong. The fix is spec conformance, not a chase after a visible symptom. Nor can
// synchronization validation see it: measured 2026-09-02, syncval cannot observe a CPU read through
// a mapped pointer (#3248). A structural assertion that the barrier was RECORDED is the only
// discriminator that fails without the fix.
//
// Consumers: the offscreen render backend (tests/fixtures/render_runner.h, which re-exports these
// into `prosper::test`) and the live compute backend (frontends/shared/live/live_compute.cpp).
// Keep it one helper: two spellings of this rule is how the next readback site gets missed.
#pragma once

#include <vulkan/vulkan.h>

#include <atomic>

namespace prosper::gpu {

struct HostReadBarrier {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;
};

// Pure, so the masks can be asserted with no device (`host_read_barrier` ctest case). The buffer
// scope is whole-buffer: a readback buffer holding several slots (the MRT frame buffer writes up to
// eight colour offsets into one allocation) is made available in one dependency.
inline HostReadBarrier host_read_barrier_for(
        VkBuffer buffer,
        VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkAccessFlags src_access = VK_ACCESS_TRANSFER_WRITE_BIT) {
    HostReadBarrier host_read{};
    host_read.barrier.srcAccessMask = src_access;
    host_read.barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host_read.barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_read.barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host_read.barrier.buffer = buffer;
    host_read.barrier.offset = 0;
    host_read.barrier.size = VK_WHOLE_SIZE;
    host_read.src_stages = src_stages;
    host_read.dst_stages = VK_PIPELINE_STAGE_HOST_BIT;
    return host_read;
}

// How many host-read barriers this process has recorded. Exists so a test can prove the barrier
// reached the command buffer on a path it drives: without it the fix is unfalsifiable here, because
// on a coherent desktop driver the UNFIXED code reads back correct pixels and correct dispatch
// results. Relaxed and process-wide -- it is an instrument, never a control input.
inline std::atomic<uint64_t>& backend_host_read_barrier_count() {
    static std::atomic<uint64_t> count{0};
    return count;
}

// Record the availability dependency for one readback buffer. Call it after the LAST device write
// into that buffer in a command buffer, and before the host maps and reads it.
inline void record_host_read_barrier(
        VkCommandBuffer command, VkBuffer buffer,
        VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkAccessFlags src_access = VK_ACCESS_TRANSFER_WRITE_BIT) {
    if (!command || !buffer) return;
    const HostReadBarrier host_read = host_read_barrier_for(buffer, src_stages, src_access);
    vkCmdPipelineBarrier(command, host_read.src_stages, host_read.dst_stages, 0,
                         0, nullptr, 1, &host_read.barrier, 0, nullptr);
    backend_host_read_barrier_count().fetch_add(1, std::memory_order_relaxed);
}

} // namespace prosper::gpu

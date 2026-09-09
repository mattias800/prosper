// Exact device-side guest tiling, recorded after a storage-image transfer.
#pragma once
#include "gpu/recompiler/spirv_builder.hpp"
#include "gpu/texture/tile.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include <vulkan/vulkan.h>
#include <array>
#include <atomic>
#include <algorithm>
#include <bit>
#include <cstdint>

namespace prosper::frontend {
inline std::atomic<uint64_t>& gpu_retile_recordings() {
    static std::atomic<uint64_t> count{0};
    return count;
}
inline std::atomic<bool>& gpu_retile_poison_output_for_test() {
    static std::atomic<bool> enabled{false};
    return enabled;
}

// Descriptor-only admission for the paired array path. Dimensionality, selected-layer offset,
// exact backend representation and byte/stride bounds are checked by the caller.
// Keep this separate from layout math: a valid base-level fallback does not prove
// that the descriptor declares only that level or sample.
inline bool gpu_retile_paired16_descriptor_supported(const prosper::gpu::ShaderResource& resource,
                                                     uint32_t bpe, uint32_t materialized_mip_levels) {
    return !(bpe != 2 || resource.sample_count != 1 || resource.declared_mip_levels != 1 ||
        materialized_mip_levels != 1 || resource.mip_chain_base_level || resource.mip_chain_max_level ||
        resource.linear_row_pitch_bytes || resource.mip_tail_offset || resource.mip_tail_bytes ||
        resource.mip_tail_x || resource.mip_tail_y || resource.compression_enabled ||
        resource.write_compress_enabled || resource.metadata_addr || resource.dcc_metadata_host_data ||
        (resource.mip_chain_element_width && resource.mip_chain_element_width != resource.width) ||
        (resource.mip_chain_element_height && resource.mip_chain_element_height != resource.height) ||
        (resource.mip_chain_bytes_per_block && resource.mip_chain_bytes_per_block != bpe));
}

struct GpuRetileParameters {
    std::array<uint32_t, 26> words{};
    uint64_t linear_bytes = 0, tiled_bytes = 0;
    uint32_t groups_x = 0, groups_y = 0, groups_z = 1;
    prosper::gpu::RetileShaderKind kind = prosper::gpu::RetileShaderKind::Words2D;

    bool initialize(uint32_t width, uint32_t height, uint32_t bpe, uint32_t mode,
                    const VkPhysicalDeviceLimits& limits) {
        *this = {};
        std::array<uint32_t, 16> equation{};
        uint32_t bw = 0, bh = 0;
        if (!width || !height ||
            !prosper::gpu::tile64_word_equation(mode, bpe, equation, bw, bh)) return false;
        if (!std::has_single_bit(bw) || !std::has_single_bit(bh) ||
            !std::has_single_bit(bpe / 4)) return false;
        const uint64_t row_words = uint64_t(width) * (bpe / 4);
        const uint64_t blocks_x = (uint64_t(width) + bw - 1) / bw;
        const uint64_t blocks_y = (uint64_t(height) + bh - 1) / bh;
        // Bound arithmetic before products; shader byte/word indices stay below 2^32.
        if (row_words > UINT32_MAX / uint64_t(height) ||
            blocks_x > UINT32_MAX / 65536u / blocks_y) return false;
        linear_bytes = row_words * height * 4;
        tiled_bytes = blocks_x * blocks_y * 65536;
        const uint64_t padded_row_words = blocks_x * bw * (bpe / 4);
        const uint64_t padded_height = blocks_y * bh;
        if (linear_bytes > limits.maxStorageBufferRange ||
            tiled_bytes > limits.maxStorageBufferRange ||
            limits.maxComputeWorkGroupSize[0] < 128 ||
            limits.maxComputeWorkGroupInvocations < 128 ||
            limits.maxPushConstantsSize < sizeof(words) ||
            (padded_row_words + 127) / 128 > limits.maxComputeWorkGroupCount[0] ||
            padded_height > limits.maxComputeWorkGroupCount[1]) return false;
        // Power-of-two widths travel as shifts, avoiding per-word integer division.
        words[0] = width; words[1] = height; words[2] = std::countr_zero(bpe / 4);
        words[3] = std::countr_zero(bw); words[4] = std::countr_zero(bh);
        words[5] = uint32_t(blocks_x);
        std::copy(equation.begin(), equation.end(), words.begin() + 6);
        groups_x = uint32_t((padded_row_words + 127) / 128);
        groups_y = uint32_t(padded_height);
        return true;
    }
    bool initialize_paired16_array(uint32_t width, uint32_t height, uint32_t layers,
                                   uint32_t mode, const VkPhysicalDeviceLimits& limits) {
        *this = {};
        std::array<uint32_t, 16> equation{};
        uint32_t bw = 0, bh = 0;
        if (!width || (width & 1) || !height || !layers ||
            !prosper::gpu::tile64_paired16_equation(mode, equation, bw, bh)) return false;
        const uint64_t row_words = width / 2;
        const uint64_t bx = (uint64_t(width) + bw - 1) / bw;
        const uint64_t by = (uint64_t(height) + bh - 1) / bh;
        // Bound each product before multiplication, including every array layer.
        // Conservative byte bounds also keep all shader word indices representable.
        if (row_words > UINT32_MAX / 4u / uint64_t(height) ||
            row_words * height > UINT32_MAX / 4u / uint64_t(layers) ||
            bx > UINT32_MAX / 65536u / by ||
            bx * by > UINT32_MAX / 65536u / uint64_t(layers)) return false;
        const uint64_t linear_words = row_words * height, tiled_words = bx * by * 16384;
        linear_bytes = linear_words * layers * 4; tiled_bytes = tiled_words * layers * 4;
        const uint64_t gx = (bx * (bw / 2) + 127) / 128, gy = by * bh;
        if (linear_bytes > limits.maxStorageBufferRange || tiled_bytes > limits.maxStorageBufferRange ||
            limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupInvocations < 128 ||
            limits.maxPushConstantsSize < sizeof(words) || gx > limits.maxComputeWorkGroupCount[0] ||
            gy > limits.maxComputeWorkGroupCount[1] || layers > limits.maxComputeWorkGroupCount[2])
            return false;
        words[0] = width; words[1] = height;
        words[3] = std::countr_zero(bw); words[4] = std::countr_zero(bh); words[5] = uint32_t(bx);
        std::copy(equation.begin(), equation.end(), words.begin() + 6);
        words[22] = layers; words[23] = uint32_t(linear_words); words[24] = uint32_t(tiled_words);
        groups_x = uint32_t(gx); groups_y = uint32_t(gy); groups_z = layers;
        kind = prosper::gpu::RetileShaderKind::Paired16Array;
        return true;
    }
    bool initialize_volume(uint32_t width, uint32_t height, uint32_t depth,
                           uint32_t bpe, uint32_t mode, const VkPhysicalDeviceLimits& limits) {
        *this = {};
        std::array<uint32_t, 16> equation{};
        uint32_t bw = 0, bh = 0, bd = 0, bits = 0;
        if (!width || !height || !depth ||
            !prosper::gpu::tile_volume_word_equation(mode, bpe, equation, bw, bh, bd, bits))
            return false;
        if (!std::has_single_bit(bw) || !std::has_single_bit(bh) || !std::has_single_bit(bd))
            return false;
        const uint64_t row_words = uint64_t(width) * (bpe / 4);
        const uint64_t bx = (uint64_t(width) + bw - 1) / bw;
        const uint64_t by = (uint64_t(height) + bh - 1) / bh;
        const uint64_t bz = (uint64_t(depth) + bd - 1) / bd;
        const uint64_t block_bytes = uint64_t{1} << bits;
        if (row_words > UINT32_MAX / uint64_t(height) ||
            row_words * height > UINT32_MAX / uint64_t(depth) ||
            bx > UINT32_MAX / block_bytes / by ||
            bx * by > UINT32_MAX / block_bytes / bz) return false;
        linear_bytes = row_words * height * depth * 4;
        tiled_bytes = bx * by * bz * block_bytes;
        const uint64_t gx = (bx * bw * (bpe / 4) + 127) / 128;
        const uint64_t gy = by * bh, gz = bz * bd;
        if (linear_bytes > limits.maxStorageBufferRange || tiled_bytes > limits.maxStorageBufferRange ||
            limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupInvocations < 128 ||
            limits.maxPushConstantsSize < sizeof(words) || gx > limits.maxComputeWorkGroupCount[0] ||
            gy > limits.maxComputeWorkGroupCount[1] || gz > limits.maxComputeWorkGroupCount[2])
            return false;
        words[0] = width; words[1] = height; words[2] = std::countr_zero(bpe / 4);
        words[3] = std::countr_zero(bw); words[4] = std::countr_zero(bh); words[5] = uint32_t(bx);
        std::copy(equation.begin(), equation.end(), words.begin() + 6);
        words[22] = depth; words[23] = std::countr_zero(bd); words[24] = uint32_t(by);
        words[25] = bits - 2; // Word offset of a whole block.
        groups_x = uint32_t(gx); groups_y = uint32_t(gy); groups_z = uint32_t(gz);
        kind = prosper::gpu::RetileShaderKind::Volume3D;
        return true;
    }
};

// Context-owned pipeline. Per-dispatch buffers and descriptors remain owned by
// BoundImage until completion; this helper never submits or waits independently.
struct GpuRetilePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool attempted = false;
    VkResult setup_result = VK_ERROR_INITIALIZATION_FAILED;
    void destroy() {
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
        pipeline = VK_NULL_HANDLE; layout = VK_NULL_HANDLE; descriptors = VK_NULL_HANDLE;
    }
    VkResult initialize(VkDevice dev, VkPipelineCache cache,
                        prosper::gpu::RetileShaderKind kind = prosper::gpu::RetileShaderKind::Words2D) {
        if (attempted) return setup_result;
        attempted = true; device = dev;
        const VkDescriptorSetLayoutBinding bindings[]{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 2; dci.pBindings = bindings;
        setup_result = vkCreateDescriptorSetLayout(device, &dci, nullptr, &descriptors);
        if (setup_result != VK_SUCCESS) return setup_result;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 26 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1; lci.pSetLayouts = &descriptors;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &push;
        setup_result = vkCreatePipelineLayout(device, &lci, nullptr, &layout);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        const auto words = prosper::gpu::build_compute_retile_words(kind);
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = words.size() * sizeof(uint32_t); sci.pCode = words.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        setup_result = vkCreateShaderModule(device, &sci, nullptr, &shader);
        if (setup_result != VK_SUCCESS) { destroy(); return setup_result; }
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = shader; pci.stage.pName = "main"; pci.layout = layout;
        setup_result = vkCreateComputePipelines(device, cache, 1, &pci, nullptr, &pipeline);
        vkDestroyShaderModule(device, shader, nullptr);
        if (setup_result != VK_SUCCESS) destroy();
        return setup_result;
    }
    VkResult bind(VkBuffer source, VkBuffer output, const GpuRetileParameters& p,
                  VkDescriptorPool& pool, VkDescriptorSet& set) const {
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.maxSets = 1; ci.poolSizeCount = 1; ci.pPoolSizes = &size;
        auto result = vkCreateDescriptorPool(device, &ci, nullptr, &pool);
        if (result != VK_SUCCESS) return result;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &descriptors;
        result = vkAllocateDescriptorSets(device, &ai, &set);
        if (result != VK_SUCCESS) return result; // caller releases the partial binding
        const VkDescriptorBufferInfo infos[]{
            {source, 0, p.linear_bytes}, {output, 0, p.tiled_bytes}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        return VK_SUCCESS;
    }
    void record(VkCommandBuffer command, VkBuffer source, VkBuffer output,
                VkDescriptorSet set, const GpuRetileParameters& p) const {
        // A pooled allocation can have a previous device writer under a different
        // buffer handle. Every output word, including padding, is shader-written.
        VkMemoryBarrier recycled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        recycled.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        recycled.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        if (gpu_retile_poison_output_for_test().load(std::memory_order_relaxed)) {
            // Dirty padding makes a missing shader store fail even on fresh memory.
            VkMemoryBarrier before_poison = recycled;
            before_poison.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before_poison, 0, nullptr, 0, nullptr);
            vkCmdFillBuffer(command, output, 0, p.tiled_bytes, 0xa5a5a5a5u);
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &recycled, 0, nullptr, 0, nullptr);
        VkBufferMemoryBarrier before{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        before.srcQueueFamilyIndex = before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        before.size = VK_WHOLE_SIZE; before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        before.buffer = source; before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &before, 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p.words), p.words.data());
        vkCmdDispatch(command, p.groups_x, p.groups_y, p.groups_z);
        gpu_retile_recordings().fetch_add(1, std::memory_order_relaxed);
        // The host reads the entire shader-written allocation, even if exact-result
        // comparison later skips this mapping.
        prosper::gpu::record_host_read_barrier(command, output,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }
};
} // namespace prosper::frontend

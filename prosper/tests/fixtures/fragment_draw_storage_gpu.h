// Shipping same-TU companion of render_runner.h, inside prosper::test. Device-local private
// transaction storage shares the renderer's device/queue and completion lifetime. It is never
// mapped or CPU-read, never serialized as guest authority, and never reused before completion.
#pragma once

struct FragmentDrawDeviceStorage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0, allocation_bytes = 0;
};
struct FragmentDrawDeviceStoragePool {
    std::map<std::pair<VkDevice, VkDeviceSize>, std::vector<FragmentDrawDeviceStorage>> free;
    VkDeviceSize bytes = 0;
};
inline FragmentDrawDeviceStoragePool& fragment_draw_device_storage_pool() {
    static thread_local FragmentDrawDeviceStoragePool pool;
    return pool;
}
inline void destroy_fragment_draw_device_storage(VkDevice device, FragmentDrawDeviceStorage value) {
    if (value.buffer) vkDestroyBuffer(device, value.buffer, nullptr);
    if (value.memory) prosper::gpu::free_device_memory(device, value.memory);
}
inline FragmentDrawDeviceStorage acquire_fragment_draw_device_storage(const RenderVkCtx& context,
                                                                      VkDeviceSize bytes) {
    if (!bytes || bytes > (64u << 20)) return {};
    VkDeviceSize capacity = 256;
    while (capacity < bytes) capacity *= 2;
    auto& pool = fragment_draw_device_storage_pool();
    const auto key = std::pair{context.dev, capacity};
    const auto found = pool.free.find(key);
    if (found != pool.free.end() && !found->second.empty()) {
        const auto result = found->second.back();
        found->second.pop_back();
        pool.bytes -= result.allocation_bytes;
        if (found->second.empty()) pool.free.erase(found);
        return result;
    }
    FragmentDrawDeviceStorage result;
    result.capacity = capacity;
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = capacity;
    buffer.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (vkCreateBuffer(context.dev, &buffer, nullptr, &result.buffer) != VK_SUCCESS) return {};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(context.dev, result.buffer, &requirements);
    result.allocation_bytes = requirements.size;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = render_memory_type(context.phys, requirements.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (allocation.memoryTypeIndex == UINT32_MAX ||
        prosper::gpu::allocate_device_memory(context.dev, &allocation, &result.memory) !=
            VK_SUCCESS ||
        vkBindBufferMemory(context.dev, result.buffer, result.memory, 0) != VK_SUCCESS) {
        destroy_fragment_draw_device_storage(context.dev, result);
        return {};
    }
    return result;
}
inline void release_fragment_draw_device_storage(VkDevice device, FragmentDrawDeviceStorage value) {
    constexpr VkDeviceSize budget = 128u << 20;
    auto& pool = fragment_draw_device_storage_pool();
    if (!value.buffer || !value.memory || !value.allocation_bytes ||
        value.allocation_bytes > budget || pool.bytes > budget - value.allocation_bytes) {
        destroy_fragment_draw_device_storage(device, value);
        return;
    }
    pool.free[{device, value.capacity}].push_back(value);
    pool.bytes += value.allocation_bytes;
}

// All immutable planes, producing owners and command-generation owners are retained by one
// transaction. Replay buffer views additionally retain this owner; no pooled gate/output can
// become a later draw's storage while the original ordered attachment consumer is still live.
class FragmentDrawGpuOwner : public std::enable_shared_from_this<FragmentDrawGpuOwner> {
public:
    enum Plane : uint32_t { Collector, Input, Output, Commit, Planes };
    static std::shared_ptr<FragmentDrawGpuOwner>
    create(const RenderVkCtx& context, prosper::gpu::FragmentDrawTransaction transaction,
           std::string& refusal) {
        refusal.clear();
        if (!context.ok || !transaction.rejection().empty() || !transaction.program() ||
            !transaction.program()->capacity_owner() ||
            transaction.executing_device_identity() != reinterpret_cast<uintptr_t>(context.dev)) {
            refusal = "fragment-draw-device-storage-owner-unavailable";
            return {};
        }
        auto result = std::shared_ptr<FragmentDrawGpuOwner>(
            new FragmentDrawGpuOwner(context, std::move(transaction)));
        const auto& capacity = *result->transaction_.program()->capacity_owner();
        const std::array<uint32_t, Planes> sizes{capacity.collector_words(), capacity.input_words(),
                                                 capacity.output_words(), capacity.commit_words()};
        for (uint32_t plane = 0; plane < Planes; ++plane) {
            const uint64_t bytes = uint64_t(sizes[plane]) * 4;
            if (!bytes || bytes > context.detile_limits.maxStorageBufferRange) {
                refusal = "fragment-draw-device-storage-extent-unavailable";
                return {};
            }
            result->storage_[plane] = acquire_fragment_draw_device_storage(context, bytes);
            result->bytes_[plane] = bytes;
            if (!result->storage_[plane].buffer) {
                refusal = "fragment-draw-device-storage-allocation-failed";
                return {};
            }
        }
        const auto& entry = result->transaction_.entry_words();
        const auto& authority = capacity.authority();
        for (uint32_t upload = 0; upload < 2; ++upload) {
            const void* data = upload ? static_cast<const void*>(entry.data()) : authority.data();
            const uint64_t bytes = uint64_t(upload ? entry.size() : authority.size()) * 4;
            if (!bytes || bytes > context.detile_limits.maxStorageBufferRange) {
                refusal = "fragment-draw-immutable-upload-extent-unavailable";
                return {};
            }
            result->upload_[upload] = acquire_render_host_buffer(context, bytes);
            result->upload_bytes_[upload] = bytes;
            if (!result->upload_[upload].mapped) {
                refusal = "fragment-draw-immutable-upload-allocation-failed";
                return {};
            }
            std::memcpy(result->upload_[upload].mapped, data, bytes);
        }
        const auto& bank = result->transaction_.scalar_bank();
        if (bool(bank) != result->transaction_.program()->requires_scalar_bank()) {
            refusal = "fragment-draw-scalar-bank-upload-owner-unavailable";
            return {};
        }
        if (bank) {
            const uint64_t bytes = uint64_t(bank->wire_words()) * 4;
            if (!bytes || bytes > context.detile_limits.maxStorageBufferRange ||
                bank->wire_metadata().size() > bank->wire_words() ||
                bank->wire_interval_words().size() != bank->intervals().size()) {
                refusal = "fragment-draw-scalar-bank-upload-extent-unavailable";
                return {};
            }
            result->upload_[2] = acquire_render_host_buffer(context, bytes);
            result->upload_bytes_[2] = bytes;
            auto* destination = static_cast<uint8_t*>(result->upload_[2].mapped);
            if (!destination) {
                refusal = "fragment-draw-scalar-bank-upload-allocation-failed";
                return {};
            }
            std::memcpy(destination, bank->wire_metadata().data(),
                        bank->wire_metadata().size() * 4);
            for (uint32_t index = 0; index < bank->intervals().size(); ++index) {
                const auto& interval = bank->intervals()[index];
                const uint64_t offset = uint64_t(bank->wire_interval_words()[index]) * 4;
                if (!interval.bytes || offset > bytes || interval.bytes->size() > bytes - offset) {
                    refusal = "fragment-draw-scalar-bank-upload-interval-invalid";
                    return {};
                }
                // Exact immutable checked source segments, copied once into the normal pooled
                // coherent upload. No guest VA reread, full-buffer concatenation or per-wave copy.
                std::memcpy(destination + offset, interval.bytes->data(), interval.bytes->size());
            }
            auto& stats = prosper::gpu::fragment_draw_cache_stats();
            ++stats.scalar_bank_uploads;
            stats.scalar_bank_payload_bytes += bank->payload_bytes();
        }
        return result;
    }
    ~FragmentDrawGpuOwner() {
        for (auto value : storage_) release_fragment_draw_device_storage(context_->dev, value);
        for (auto value : upload_) release_render_host_buffer(context_->dev, value);
    }
    const prosper::gpu::FragmentDrawTransaction& transaction() const { return transaction_; }
    VkDescriptorBufferInfo plane(Plane plane) const {
        if (uint32_t(plane) >= Planes) return {};
        return {storage_[plane].buffer, 0, bytes_[plane]};
    }
    VkDescriptorBufferInfo upload(uint32_t index) const {
        if (index >= upload_.size()) return {};
        return {upload_[index].buffer, 0, upload_bytes_[index]};
    }
    std::shared_ptr<const FragmentDrawGpuBuffer> view(Plane plane) const {
        if (uint32_t(plane) >= Planes) return {};
        return std::shared_ptr<const FragmentDrawGpuBuffer>(new FragmentDrawGpuBuffer(
            context_->dev, storage_[plane].buffer, bytes_[plane], shared_from_this()));
    }
    std::shared_ptr<const FragmentDrawGpuBuffer> authority_view() const {
        return std::shared_ptr<const FragmentDrawGpuBuffer>(new FragmentDrawGpuBuffer(
            context_->dev, upload_[0].buffer, upload_bytes_[0], shared_from_this()));
    }
    // Called exactly once on a newly leased transaction BEFORE collector/indirect/validation use.
    // Missing/outer-refused stages leave gate0, never a previous pool user's accepted gate1.
    void record_zero_and_upload_visibility(VkCommandBuffer command) const {
        std::array<VkBufferMemoryBarrier, Planes + 3> barriers{};
        const uint32_t barrier_count = Planes + (transaction_.scalar_bank() ? 3u : 2u);
        std::array<VkBufferMemoryBarrier, Planes> recycle{};
        for (uint32_t index = 0; index < Planes; ++index) {
            recycle[index] = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            recycle[index].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            recycle[index].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            recycle[index].srcQueueFamilyIndex = recycle[index].dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            recycle[index].buffer = storage_[index].buffer;
            recycle[index].size = bytes_[index];
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                             uint32_t(recycle.size()), recycle.data(), 0, nullptr);
        for (uint32_t index = 0; index < barrier_count; ++index) {
            auto& barrier = barriers[index];
            barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.offset = 0;
            const bool device_plane = index < Planes;
            const auto info = device_plane ? plane(Plane(index)) : upload(index - Planes);
            barrier.buffer = info.buffer;
            barrier.size = info.range;
            // Collector's four-word header is updated separately, over a disjoint range. A fill
            // followed by an overlapping update would introduce an unnecessary transfer WAW.
            const VkDeviceSize start = index == Collector ? 16 : 0;
            if (device_plane) vkCmdFillBuffer(command, info.buffer, start, info.range - start, 0);
            barrier.srcAccessMask =
                device_plane ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    (device_plane ? VK_ACCESS_SHADER_WRITE_BIT : 0) |
                                    (index == Input ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT : 0);
        }
        const uint32_t header[]{0, 0, transaction_.program()->collector_shape().record_words,
                                prosper::gpu::kRasterQuadMagic};
        vkCmdUpdateBuffer(command, storage_[Collector].buffer, 0, sizeof(header), header);
        vkCmdPipelineBarrier(
            command, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 0, nullptr, barrier_count, barriers.data(), 0, nullptr);
    }

private:
    FragmentDrawGpuOwner(const RenderVkCtx& context,
                         prosper::gpu::FragmentDrawTransaction transaction)
        : context_(&context), transaction_(std::move(transaction)) {}
    const RenderVkCtx* const context_;
    const prosper::gpu::FragmentDrawTransaction transaction_;
    std::array<FragmentDrawDeviceStorage, Planes> storage_{};
    std::array<VkDeviceSize, Planes> bytes_{};
    std::array<RenderHostBuffer, 3> upload_{};
    std::array<VkDeviceSize, 3> upload_bytes_{};
};

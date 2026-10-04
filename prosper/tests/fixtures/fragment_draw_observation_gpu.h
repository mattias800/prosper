// Shipping same-TU diagnostic companion, included after FragmentDrawGpuOwner. These are bounded
// observations of actual private planes, NOT guest permission or an alternative renderer. Unset
// records no commands, allocates no staging, performs no readback and adds no completion wait.
#pragma once

struct FragmentDrawObservationStats {
    uint64_t recorded = 0, reported = 0, unavailable = 0;
};
inline FragmentDrawObservationStats& fragment_draw_observation_stats() {
    static thread_local FragmentDrawObservationStats value;
    return value;
}
inline bool fragment_draw_observation_enabled() {
    return PROSPER_ENV_ON("PROSPER_FRAGMENT_DRAW_OBSERVE");
}

class FragmentDrawGpuObservation {
public:
    using O = FragmentDrawGpuOwner;
    static std::unique_ptr<FragmentDrawGpuObservation>
    create(const RenderVkCtx& context, std::shared_ptr<const O> owner, size_t draw) {
        auto result = std::unique_ptr<FragmentDrawGpuObservation>(
            new FragmentDrawGpuObservation(context, std::move(owner), draw));
        const auto& capacity = *result->owner_->transaction().program()->capacity_owner();
        result->append(O::Collector, UINT32_MAX, 0, 32);
        result->append(O::Input, UINT32_MAX, 0, prosper::gpu::kFragmentDrawHeaderWords);
        result->append(O::Commit, UINT32_MAX, 0, prosper::gpu::kFragmentDrawCommitHeaderWords);
        // This is capacity-bounded sampling, not a CPU count of GPU-produced work. A zero/unrun
        // sampled slot remains an observation of that slot; it is never labeled an executed wave.
        for (uint32_t wave = 0; wave < std::min(capacity.max_waves(), 3u); ++wave) {
            const auto placement = capacity.placement(wave);
            result->append(O::Input, wave, placement.input_base,
                           std::min(capacity.input_span(), 1024u));
            result->append(O::Output, wave, placement.output_base,
                           std::min(capacity.output_span(), 4096u));
        }
        if (!result->bytes_ || result->bytes_ > (64u << 10)) return {};
        std::string error;
        if (!persistent_ds_transfer_buffer(context, result->bytes_,
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT, result->buffer_,
                                           result->memory_, error)) {
            std::fprintf(stderr, "[fragment-draw-observe] unavailable draw=%zu reason=%s\n", draw,
                         error.c_str());
            ++fragment_draw_observation_stats().unavailable;
            return {};
        }
        return result;
    }
    ~FragmentDrawGpuObservation() {
        if (buffer_) vkDestroyBuffer(context_->dev, buffer_, nullptr);
        if (memory_) prosper::gpu::free_device_memory(context_->dev, memory_);
    }
    void record_after_replay(VkCommandBuffer command) {
        // Outside the ordinary render pass and after its last private-plane reads. Source writes
        // include collector/count/assembly/kernel/validation; indirect/header readers are also
        // ordered before the transfer. Nothing below modifies those source planes or admission.
        std::array<VkBufferMemoryBarrier, O::Planes> barriers{};
        for (uint32_t plane = 0; plane < O::Planes; ++plane) {
            const auto source = owner_->plane(O::Plane(plane));
            auto& barrier = barriers[plane];
            barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT |
                                    VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = source.buffer;
            barrier.offset = source.offset;
            barrier.size = source.range;
        }
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, uint32_t(barriers.size()),
            barriers.data(), 0, nullptr);
        for (const auto& range : ranges_) {
            const auto source = owner_->plane(range.plane);
            const VkBufferCopy copy{source.offset + uint64_t(range.word) * 4, range.destination,
                                    uint64_t(range.words) * 4};
            vkCmdCopyBuffer(command, source.buffer, buffer_, 1, &copy);
        }
        record_host_read_barrier(command, buffer_);
        recorded_ = true;
        ++fragment_draw_observation_stats().recorded;
    }
    // Caller invokes ONLY after the SAME batch's existing successful submit/fence. Failure or
    // indeterminate completion publishes nothing and retains this staging with the batch owner.
    void report_completed() const {
        if (!recorded_) return;
        void* mapped = nullptr;
        const auto plan = mapped_readback_plan(bytes_);
        const bool mapped_ok =
            vkMapMemory(context_->dev, memory_, 0, plan.map_size, 0, &mapped) == VK_SUCCESS &&
            mapped;
        const bool readable = mapped_ok && invalidate_mapped_readback(*context_, memory_, plan);
        if (!readable) {
            if (mapped_ok) vkUnmapMemory(context_->dev, memory_);
            ++fragment_draw_observation_stats().unavailable;
            std::fprintf(stderr,
                         "[fragment-draw-observe] unavailable draw=%zu reason=map-invalidate\n",
                         draw_);
            return;
        }
        const auto& capacity = *owner_->transaction().program()->capacity_owner();
        const auto& program = capacity.kernel()->program;
        std::fprintf(stderr,
                     "[fragment-draw-observe] completed draw=%zu capacity_waves=%u "
                     "input_span=%u output_span=%u output_prefix=%u status_offset=%u "
                     "vgpr_status_offset=%u input_sample_words=%u output_sample_words=%u "
                     "omitted_capacity_waves=%u bank_upload_words=%llu bank_sample_words=%zu "
                     "staging_bytes=%llu\n",
                     draw_, capacity.max_waves(), capacity.input_span(), capacity.output_span(),
                     prosper::gpu::kPacketWaveOutputPrefix, program.status_offset,
                     program.packet.vgpr_status_offset, std::min(capacity.input_span(), 1024u),
                     std::min(capacity.output_span(), 4096u),
                     capacity.max_waves() - std::min(capacity.max_waves(), 3u),
                     static_cast<unsigned long long>(owner_->upload(2).range / 4),
                     owner_->diagnostic_uploaded_bank_words().size(),
                     static_cast<unsigned long long>(bytes_));
        const auto& layout = capacity.kernel()->layout;
        for (size_t index = 0; index < layout.sgprs.size(); ++index)
            std::fprintf(stderr,
                         "[fragment-draw-observe] draw=%zu scalar=s%u input_wave_relative_word=%u "
                         "available_wave_relative_word=%u\n",
                         draw_, layout.sgprs[index],
                         index < layout.scalar_offsets.size() ? 2 + layout.scalar_offsets[index]
                                                              : UINT32_MAX,
                         index < layout.scalar_available_offsets.size()
                             ? 2 + layout.scalar_available_offsets[index]
                             : UINT32_MAX);
        // Actual already-owned HOST_COHERENT bank upload. Its immutable source owner remains
        // retained; this neither rereads guest VA nor claims atomicity against guest CPU writes.
        const auto bank = owner_->diagnostic_uploaded_bank_words();
        dump("bank-upload-host", UINT32_MAX, 0, bank);
        const auto* words = static_cast<const uint32_t*>(mapped);
        constexpr const char* names[]{"collector", "input", "output", "commit"};
        for (const auto& range : ranges_)
            dump(names[range.plane], range.wave, range.word,
                 std::span(words + range.destination / 4, range.words));
        vkUnmapMemory(context_->dev, memory_);
        ++fragment_draw_observation_stats().reported;
    }

private:
    struct Range {
        O::Plane plane;
        uint32_t wave, word, words;
        VkDeviceSize destination;
    };
    FragmentDrawGpuObservation(const RenderVkCtx& context, std::shared_ptr<const O> owner,
                               size_t draw)
        : context_(&context), owner_(std::move(owner)), draw_(draw) {}
    void append(O::Plane plane, uint32_t wave, uint32_t word, uint32_t requested) {
        const uint64_t words = owner_->plane(plane).range / 4;
        if (word >= words || !requested) return;
        const uint32_t count = uint32_t(std::min<uint64_t>(words - word, requested));
        ranges_.push_back({plane, wave, word, count, bytes_});
        bytes_ += uint64_t(count) * 4;
    }
    void dump(const char* plane, uint32_t wave, uint32_t word,
              std::span<const uint32_t> values) const {
        for (size_t begin = 0; begin < values.size(); begin += 16) {
            std::fprintf(stderr,
                         "[fragment-draw-observe] draw=%zu plane=%s capacity_wave=%u word=%zu",
                         draw_, plane, wave, size_t(word) + begin);
            for (size_t index = begin; index < std::min(begin + 16, values.size()); ++index)
                std::fprintf(stderr, " %08x", values[index]);
            std::fputc('\n', stderr);
        }
    }
    const RenderVkCtx* const context_;
    const std::shared_ptr<const O> owner_;
    const size_t draw_;
    std::vector<Range> ranges_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize bytes_ = 0;
    bool recorded_ = false;
};

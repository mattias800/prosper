// SHIPPING companion of render_runner.h, inside prosper::test: the backend half of a merged ES+GS
// NGG draw (#3135 phase P4; the draw description is gpu/execute/ngg_subgroup_draw.hpp). Uses the
// renderer's own device, queue, descriptor pool and submission batch; it creates no device.
//
// A BackendDraw carrying `ngg_subgroup` is EXPANDED before the pass is set up: one ordinary draw
// per run of the description, each running that run's pass-through vertex stage over the export
// region its wave-count group writes (set 2 binding 1; the violation counters at binding 2). The
// expanded draws keep the original draw's set-0 resources (the shell's guest inputs), its pixel
// stage and its fixed-function state, with the topology replaced by the pass-through list.
//
// Then, in the pass's own command buffer, before its render pass begins and with no CPU wait:
// fill the export regions and counters with zero, barrier, one dispatch per wave-count group,
// barrier from the compute to the vertex stage. render_draws_rgba makes an NGG draw its own
// segment, so the work of earlier draws is complete in queue order and made visible to the shell
// by this prelude's first barrier.
//
// SCRATCH comes from a ring of retained chunks: launch records and counters in host-visible
// memory, export blocks in device-local memory. A slice returns to its chunk when the batch that
// used it has completed (or was never submitted); no Vulkan object is created per draw once the
// ring and the per-shell pipelines are warm.
//
// The violation counters are read LAZILY: after the batch completes for its own reasons, in its
// completion callback, never by a wait of their own (CLAUDE.md P1). They accumulate into
// ngg_subgroup_backend_stats() and print a bounded warning.
//
// ADMISSION (#3135 P5). The device half of admission is shared: ngg_device_refusal
// (gpu/execute/ngg_draw_admission.hpp) is what the live producer asks before it hands a draw over
// and what render_draws_rgba asks before it splits the batch (ngg_admit_backend_draws), so the
// two never disagree; this file publishes the device's answers (publish_ngg_backend_capabilities).
// render_draws_rgba also drops an NGG draw whose own segment would lose transient depth
// (persist_depth_stencil false) or force a colour readback between segments (no persistent colour
// target: a CPU wait per split, CLAUDE.md P1). Every drop is counted under backend/ngg-subgroup.
//
// ALL OR NOTHING. A prelude is recorded only when every one of its run draws is ready to record;
// otherwise none of its runs draw, and each is counted under ngg-subgroup.
//
// SET 0 travels on the FIRST run draw only: the shell reads it through that draw's descriptor
// writes, and the pass-through and pixel stages never read set 0.
//
// IMAGES. The shell reads plain storage buffers only (ngg-draw-guest-resource-unsupported refuses
// anything else), so of the pass's image barriers only the texture-upload one names COMPUTE. A
// change that admits shell images must audit the detile-restore and compute-image barriers too.
#pragma once

// ---- Statistics ---------------------------------------------------------------------------------

struct NggSubgroupBackendStats {
    std::atomic<uint64_t> draws{0};   // NGG draws whose prelude was recorded
    std::atomic<uint64_t> dispatches{0};
    std::atomic<uint64_t> completed{0};   // preludes whose batch completed and was read
    std::atomic<uint64_t> invalid_blocks{0};
    std::atomic<uint64_t> connectivity{0};
    std::atomic<uint64_t> layer_culled{0};
    std::atomic<uint64_t> refused{0};   // NGG draws the backend refused (with a reason printed)
    // Where the last recorded prelude's first export region lives: lets a test see slice reuse.
    std::atomic<uint64_t> last_export_buffer{0};
    std::atomic<uint64_t> last_export_offset{0};
};
inline NggSubgroupBackendStats& ngg_subgroup_backend_stats() {
    static NggSubgroupBackendStats stats;
    return stats;
}

// Whether this backend can count violations: the pass-through vertex stage's atomics need
// vertexPipelineStoresAndAtomics. A producer sets NggRasterCommitConfig::count_violations from it.
inline bool ngg_backend_counts_violations(const RenderVkCtx& ctx) {
    return ctx.vertex_pipeline_stores;
}

// What this device can do for the merged-NGG path, in the Vulkan-free form admission takes.
inline prosper::gpu::NggHostCapabilities ngg_host_capabilities(const RenderVkCtx& ctx) {
    prosper::gpu::NggHostCapabilities host;
    host.published = ctx.ok;
    host.compute = ctx.queue_supports_compute;
    host.vertex_pipeline_stores = ctx.vertex_pipeline_stores;
    host.shader_output_layer = ctx.shader_output_layer_enabled;
    host.geometry_shader = ctx.geometry_shader_enabled;
    host.native_wave64 = ctx.subgroup_size_control && ctx.compute_full_subgroups &&
                         (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_COMPUTE_BIT) &&
                         ctx.min_subgroup_size <= 64u && ctx.max_subgroup_size >= 64u;
    host.max_compute_workgroup_subgroups = ctx.max_compute_workgroup_subgroups;
    host.max_compute_shared_memory = ctx.detile_limits.maxComputeSharedMemorySize;
    host.max_compute_workgroup_size_x = ctx.max_compute_workgroup_size_x;
    host.max_compute_workgroup_invocations = ctx.max_compute_workgroup_invocations;
    host.max_compute_workgroup_count_x = ctx.detile_limits.maxComputeWorkGroupCount[0];
    host.max_storage_buffer_range = ctx.detile_limits.maxStorageBufferRange;
    host.max_push_constants_size = ctx.detile_limits.maxPushConstantsSize;
    return host;
}

// Called once the renderer's device exists (render_vk_ctx): realization admits nothing before it.
inline void publish_ngg_backend_capabilities(const RenderVkCtx& ctx) {
    prosper::gpu::NggHostCapabilities host = ngg_host_capabilities(ctx);
    host.published = true;
    prosper::gpu::publish_ngg_host_capabilities(host);
}

// One merged-NGG draw lost in the backend: counted under backend/ngg-subgroup in the alarm ledger
// (and, when the draw was already counted seen in a pass, in the draw-disposition census), with a
// bounded log line.
inline void note_ngg_backend_drop(const char* reason, bool in_pass) {
    if (in_pass)
        prosper::gpu::draw_disposition_census().note_dropped(prosper::gpu::DrawDrop::NggSubgroup);
    else
        prosper::diagnostics::perf::drop_draw(
            prosper::diagnostics::perf::DropReason::BackendNggSubgroup);
    ngg_subgroup_backend_stats().refused.fetch_add(1, std::memory_order_relaxed);
    static std::atomic<uint32_t> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 16u)
        std::fprintf(stderr, "[ngg-backend] dropped draw reason=%s (first 16 reported)\n", reason);
}

// ---- Scratch ring ---------------------------------------------------------------------------------

struct NggScratchChunk {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* mapped = nullptr;   // host chunks only
    VkDeviceSize size = 0, head = 0;
    uint32_t leases = 0;
    bool host = false;
};

struct NggScratchSlice {
    NggScratchChunk* chunk = nullptr;
    VkDeviceSize offset = 0, bytes = 0;
    VkBuffer buffer() const { return chunk ? chunk->buffer : VK_NULL_HANDLE; }
    uint32_t* words() const {
        return chunk && chunk->mapped ? reinterpret_cast<uint32_t*>(chunk->mapped + offset)
                                      : nullptr;
    }
};

// Process lifetime, like the render context (see render_vk_context.h): chunks are never destroyed.
class NggScratchRing {
public:
    static NggScratchRing& instance() {
        static NggScratchRing ring;
        return ring;
    }

    // A slice of at least `bytes`, aligned for a storage-buffer descriptor, or an empty slice.
    NggScratchSlice acquire(const RenderVkCtx& ctx, bool host, VkDeviceSize bytes) {
        const VkDeviceSize align =
            std::max<VkDeviceSize>(256u, ctx.detile_limits.minStorageBufferOffsetAlignment);
        bytes = (std::max<VkDeviceSize>(bytes, 4u) + align - 1u) / align * align;
        const std::lock_guard lock(mutex_);
        for (const auto& chunk : chunks_)
            if (chunk->host == host && chunk->size - chunk->head >= bytes)
                return take(*chunk, bytes);
        auto chunk = create(ctx, host, std::max<VkDeviceSize>(kChunkBytes, bytes));
        if (!chunk) return {};
        chunks_.push_back(std::move(chunk));
        return take(*chunks_.back(), bytes);
    }

    void release(const NggScratchSlice& slice) {
        if (!slice.chunk) return;
        const std::lock_guard lock(mutex_);
        if (!slice.chunk->leases) return;
        --slice.chunk->leases;
        if (!slice.chunk->leases) slice.chunk->head = 0;
    }

    size_t chunk_count() const {
        const std::lock_guard lock(mutex_);
        return chunks_.size();
    }

private:
    static constexpr VkDeviceSize kChunkBytes = 4u << 20;

    static NggScratchSlice take(NggScratchChunk& chunk, VkDeviceSize bytes) {
        NggScratchSlice slice{&chunk, chunk.head, bytes};
        chunk.head += bytes;
        ++chunk.leases;
        return slice;
    }

    static std::unique_ptr<NggScratchChunk> create(const RenderVkCtx& ctx, bool host,
                                                   VkDeviceSize bytes) {
        auto chunk = std::make_unique<NggScratchChunk>();
        chunk->host = host;
        chunk->size = bytes;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(ctx.dev, &info, nullptr, &chunk->buffer) != VK_SUCCESS) return nullptr;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(ctx.dev, chunk->buffer, &requirements);
        constexpr VkMemoryPropertyFlags kHost =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = host ? render_memory_type(ctx.phys, requirements.memoryTypeBits, kHost)
                             : render_memory_type(ctx.phys, requirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX && !host)
            type = render_memory_type(ctx.phys, requirements.memoryTypeBits, 0);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        void* mapped = nullptr;
        if (type == UINT32_MAX ||
            prosper::gpu::allocate_device_memory(ctx.dev, &allocate, &chunk->memory) !=
                VK_SUCCESS ||
            vkBindBufferMemory(ctx.dev, chunk->buffer, chunk->memory, 0) != VK_SUCCESS ||
            (host &&
             vkMapMemory(ctx.dev, chunk->memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)) {
            if (chunk->memory) prosper::gpu::free_device_memory(ctx.dev, chunk->memory);
            vkDestroyBuffer(ctx.dev, chunk->buffer, nullptr);
            return nullptr;
        }
        chunk->mapped = static_cast<uint8_t*>(mapped);
        return chunk;
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<NggScratchChunk>> chunks_;
};

// ---- Compute pipelines ----------------------------------------------------------------------------

// One per (shell module, guest bindings, push words, native Wave64). Shared by every prelude that
// uses it; an entry evicted from the bounded cache is destroyed when the last batch holding it
// completes.
struct NggShellPipeline {
    VkDevice dev = VK_NULL_HANDLE;
    VkDescriptorSetLayout guest = VK_NULL_HANDLE, empty = VK_NULL_HANDLE, shell = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t push_bytes = 0;
    std::shared_ptr<const std::vector<uint32_t>> words;   // the module, for an exact key compare
    NggShellPipeline() = default;
    NggShellPipeline(const NggShellPipeline&) = delete;
    NggShellPipeline& operator=(const NggShellPipeline&) = delete;
    ~NggShellPipeline() {
        if (!dev) return;
        if (pipeline) vkDestroyPipeline(dev, pipeline, nullptr);
        if (module) vkDestroyShaderModule(dev, module, nullptr);
        if (layout) vkDestroyPipelineLayout(dev, layout, nullptr);
        for (VkDescriptorSetLayout l : {guest, empty, shell})
            if (l) vkDestroyDescriptorSetLayout(dev, l, nullptr);
    }
};

struct NggShellPipelineCacheStats {
    uint64_t hits = 0, creations = 0, evictions = 0;
    size_t entries = 0;
};
inline NggShellPipelineCacheStats& ngg_shell_pipeline_cache_stats() {
    static NggShellPipelineCacheStats stats;
    return stats;
}
inline constexpr size_t kNggShellPipelineCacheEntries = 64;

// Called with the backend's persistent-resource guard held. The key is the module hash computed
// once when the stages were built (never re-hashed per draw); a hit compares the module by pointer,
// and by content only when two descriptions hold equal modules in different allocations.
inline std::shared_ptr<const NggShellPipeline>
ngg_shell_pipeline(const RenderVkCtx& ctx, const prosper::gpu::NggSubgroupStages& stages,
                   const std::vector<uint32_t>& guest_bindings, uint32_t push_words,
                   bool native_wave64) {
    struct Entry {
        std::shared_ptr<const NggShellPipeline> pipeline;
        std::vector<uint32_t> key;
        uint64_t last_use = 0;
    };
    // Leaked on purpose, like the render context: no exit-time destruction against a torn-down
    // device.
    static auto& cache = *new std::unordered_multimap<uint64_t, Entry>();
    static uint64_t clock = 0;
    auto& stats = ngg_shell_pipeline_cache_stats();
    const auto& shell = *stages.shell;
    std::vector<uint32_t> key = {stages.waves, push_words, native_wave64 ? 1u : 0u,
                                 static_cast<uint32_t>(shell.size()),
                                 static_cast<uint32_t>(guest_bindings.size())};
    key.insert(key.end(), guest_bindings.begin(), guest_bindings.end());
    const auto [first, last] = cache.equal_range(stages.shell_hash);
    for (auto it = first; it != last; ++it) {
        Entry& entry = it->second;
        if (entry.key != key) continue;
        if (entry.pipeline->words != stages.shell && *entry.pipeline->words != shell) continue;
        entry.last_use = ++clock;
        ++stats.hits;
        return entry.pipeline;
    }
    auto entry = std::make_shared<NggShellPipeline>();
    entry->dev = ctx.dev;
    entry->words = stages.shell;
    const auto make_layout = [&](const std::vector<uint32_t>& bindings) {
        std::vector<VkDescriptorSetLayoutBinding> entries;
        entries.reserve(bindings.size());
        for (uint32_t binding : bindings)
            entries.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = static_cast<uint32_t>(entries.size());
        info.pBindings = entries.empty() ? nullptr : entries.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(ctx.dev, &info, nullptr, &layout);
        return layout;
    };
    entry->guest = make_layout(guest_bindings);
    entry->empty = make_layout({});
    entry->shell =
        make_layout({prosper::gpu::kNggShellLaunchBinding, prosper::gpu::kNggShellExportBinding});
    entry->push_bytes = push_words * 4u;
    const VkDescriptorSetLayout sets[3] = {entry->guest, entry->empty, entry->shell};
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, entry->push_bytes};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 3;
    layout_info.pSetLayouts = sets;
    layout_info.pushConstantRangeCount = push_words ? 1u : 0u;
    layout_info.pPushConstantRanges = &push;
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = shell.size() * 4u;
    module_info.pCode = shell.data();
    if (!entry->guest || !entry->empty || !entry->shell ||
        vkCreatePipelineLayout(ctx.dev, &layout_info, nullptr, &entry->layout) != VK_SUCCESS ||
        vkCreateShaderModule(ctx.dev, &module_info, nullptr, &entry->module) != VK_SUCCESS)
        return nullptr;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    required.requiredSubgroupSize = 64;
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization): every stage field is set below.
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = entry->module;
    info.stage.pName = "main";
    if (native_wave64) {
        info.stage.pNext = &required;
        info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    info.layout = entry->layout;
    if (vkCreateComputePipelines(ctx.dev, VK_NULL_HANDLE, 1, &info, nullptr, &entry->pipeline) !=
        VK_SUCCESS)
        return nullptr;
    ++stats.creations;
    while (cache.size() >= kNggShellPipelineCacheEntries) {
        auto oldest = cache.begin();
        for (auto it = std::next(cache.begin()); it != cache.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.erase(oldest);
        ++stats.evictions;
    }
    cache.emplace(stages.shell_hash, Entry{entry, std::move(key), ++clock});
    stats.entries = cache.size();
    return entry;
}

// ---- The per-pass batch ---------------------------------------------------------------------------

class NggSubgroupBackendBatch : public std::enable_shared_from_this<NggSubgroupBackendBatch> {
public:
    explicit NggSubgroupBackendBatch(const RenderVkCtx& ctx) : ctx_(ctx) {}
    NggSubgroupBackendBatch(const NggSubgroupBackendBatch&) = delete;
    NggSubgroupBackendBatch& operator=(const NggSubgroupBackendBatch&) = delete;
    ~NggSubgroupBackendBatch() {
        // Reached only after the batch that used the slices completed or was discarded: the
        // completion callback holds this object until then (an abandoned batch leaks it).
        for (const Prelude& prelude : preludes_) {
            for (const NggScratchSlice& slice : prelude.launch)
                NggScratchRing::instance().release(slice);
            for (const NggScratchSlice& slice : prelude.exports)
                NggScratchRing::instance().release(slice);
            NggScratchRing::instance().release(prelude.counters);
        }
    }

    // Replace every NGG draw of `in` by its run draws in `out`. Returns false with `refusal` set
    // when any NGG draw cannot run on this device; nothing is recorded then.
    static bool expand(const RenderVkCtx& ctx, std::span<const BackendDraw> in,
                       std::vector<BackendDraw>& out,
                       std::shared_ptr<NggSubgroupBackendBatch>& batch, std::string& refusal) {
        batch = std::make_shared<NggSubgroupBackendBatch>(ctx);
        out.clear();
        for (const BackendDraw& draw : in) {
            if (!draw.ngg_subgroup) {
                out.push_back(draw);
                batch->prelude_of_draw_.push_back(-1);
                continue;
            }
            if (!batch->expand_one(draw, out, refusal)) {
                note_ngg_backend_drop(refusal.c_str(), false);
                return false;
            }
        }
        return true;
    }

    uint32_t additional_sets() const {
        uint32_t sets = 0;
        for (const Prelude& prelude : preludes_)
            sets += 1u + static_cast<uint32_t>(prelude.ngg->groups.size());
        return sets;
    }
    uint32_t additional_storage_descriptors() const {
        uint32_t count = 0;
        for (const Prelude& prelude : preludes_)
            count += static_cast<uint32_t>(prelude.ngg->guest_bindings.size() +
                                           2u * prelude.ngg->groups.size());
        return count;
    }

    // Called once per pass-local draw after its descriptor sets are written. The FIRST run draw of
    // an NGG draw arms its prelude: the shell's guest set takes the very buffer descriptors this
    // draw's set 0 received. It never refuses the draw itself: whether the NGG draw records at all
    // is decided once every run draw's pipeline exists (record()), so a failure here only leaves
    // the prelude unarmed, and record() then drops every run.
    bool capture(size_t draw, VkDescriptorSet set0, std::span<const VkWriteDescriptorSet> writes,
                 VkDescriptorPool pool) {
        if (draw >= prelude_of_draw_.size() || prelude_of_draw_[draw] < 0) return true;
        Prelude& prelude = preludes_[static_cast<size_t>(prelude_of_draw_[draw])];
        if (draw != prelude.first_draw) return true;
        if (!pool || !set0) return fail_capture(prelude, "descriptor-pool");
        std::vector<VkDescriptorSetLayout> layouts = {prelude.pipelines.front()->guest};
        for (const auto& pipeline : prelude.pipelines) layouts.push_back(pipeline->shell);
        std::vector<VkDescriptorSet> sets(layouts.size());
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocate.pSetLayouts = layouts.data();
        if (vkAllocateDescriptorSets(ctx_.dev, &allocate, sets.data()) != VK_SUCCESS)
            return fail_capture(prelude, "descriptor-allocation");
        std::vector<VkWriteDescriptorSet> out;
        std::vector<VkDescriptorBufferInfo> infos;
        infos.reserve(2u * prelude.pipelines.size());
        for (uint32_t binding : prelude.ngg->guest_bindings) {
            const auto found = std::find_if(writes.begin(), writes.end(), [&](const auto& w) {
                return w.dstSet == set0 && w.dstBinding == binding;
            });
            if (found == writes.end() || found->descriptorCount != 1 || !found->pBufferInfo ||
                found->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                return fail_capture(prelude, "guest-binding");
            VkWriteDescriptorSet write = *found;
            write.dstSet = sets[0];
            out.push_back(write);
        }
        for (size_t g = 0; g < prelude.pipelines.size(); ++g) {
            infos.push_back(
                {prelude.launch[g].buffer(), prelude.launch[g].offset, prelude.launch[g].bytes});
            infos.push_back(
                {prelude.exports[g].buffer(), prelude.exports[g].offset, prelude.exports[g].bytes});
            for (uint32_t k = 0; k < 2; ++k) {
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = sets[1 + g];
                write.dstBinding = k;   // kNggShellLaunchBinding, kNggShellExportBinding
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &infos[2 * g + k];
                out.push_back(write);
            }
        }
        vkUpdateDescriptorSets(ctx_.dev, static_cast<uint32_t>(out.size()), out.data(), 0, nullptr);
        prelude.guest_set = sets[0];
        prelude.shell_sets.assign(sets.begin() + 1, sets.end());
        prelude.armed = true;
        return true;
    }

    // Before the render pass begins, in the pass's command buffer: fill, barrier, dispatches,
    // barrier. Registers the lazy counter read on `submission`. `draws` is the pass's per-draw
    // state; a prelude is recorded only when it is armed and EVERY one of its run draws is ready
    // (`ok`). Otherwise none of its runs draw: each ready one is turned off here and counted.
    template <typename DrawState>
    void record(VkCommandBuffer cmd, BackendSubmissionBatch& submission,
                std::span<DrawState> draws) {
        for (size_t p = 0; p < preludes_.size(); ++p) {
            Prelude& prelude = preludes_[p];
            if (prelude_ready(prelude.armed, static_cast<int>(p), prelude_of_draw_, draws))
                continue;
            prelude.armed = false;
            for (size_t d = 0; d < prelude_of_draw_.size() && d < draws.size(); ++d)
                if (prelude_of_draw_[d] == static_cast<int>(p) && draws[d].ok) {
                    draws[d].ok = false;
                    note_ngg_backend_drop("ngg-backend-partial-draw", true);
                }
        }
        bool any = false;
        const VkPipelineStageFlags graphics =
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            (ctx_.geometry_shader_enabled ? VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT : 0u);
        const VkPipelineStageFlags raster_stages =
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            (ctx_.geometry_shader_enabled ? VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT : 0u);
        for (Prelude& prelude : preludes_) {
            if (!prelude.armed) continue;
            any = true;
            // Earlier segments of this submission may have written what the shell reads.
            VkMemoryBarrier prior{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            prior.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            prior.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(
                cmd,
                graphics | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &prior,
                0, nullptr, 0, nullptr);
            // 1. Zero every export block (an unwritten flag means "not exported") and counter.
            for (const NggScratchSlice& slice : prelude.exports)
                vkCmdFillBuffer(cmd, slice.buffer(), slice.offset, slice.bytes, 0u);
            if (prelude.counters.chunk)
                vkCmdFillBuffer(cmd, prelude.counters.buffer(), prelude.counters.offset,
                                prelude.counters.bytes, 0u);
            // 2. The fill before the shell and the counting vertex stage.
            VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | raster_stages, 0, 1,
                                 &filled, 0, nullptr, 0, nullptr);
            // 3. One dispatch per wave-count group, ascending W.
            for (size_t g = 0; g < prelude.pipelines.size(); ++g) {
                const NggShellPipeline& pipeline = *prelude.pipelines[g];
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                                        &prelude.guest_set, 0, nullptr);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 2, 1,
                                        &prelude.shell_sets[g], 0, nullptr);
                if (pipeline.push_bytes)
                    vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       pipeline.push_bytes, prelude.ngg->push_constants.data());
                vkCmdDispatch(cmd, prelude.ngg->groups[g].blocks, 1, 1);
                ngg_subgroup_backend_stats().dispatches.fetch_add(1, std::memory_order_relaxed);
            }
            // 4. The shell's records before the pass-through stages read them.
            VkMemoryBarrier exported{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            exported.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            exported.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, raster_stages, 0, 1,
                                 &exported, 0, nullptr, 0, nullptr);
            prelude.recorded = true;
            ngg_subgroup_backend_stats().last_export_buffer.store(
                std::bit_cast<uint64_t>(prelude.exports.front().buffer()));
            ngg_subgroup_backend_stats().last_export_offset.store(prelude.exports.front().offset);
            ngg_subgroup_backend_stats().draws.fetch_add(1, std::memory_order_relaxed);
        }
        if (!any) return;
        auto self = shared_from_this();
        submission.add_failure_cleanup([self] { self->submission_failed_ = true; });
        submission.add_cleanup([self] { self->read_counters(); });
    }

    // Whether prelude `prelude` may record: it is armed and EVERY run draw it owns is ready. A
    // prelude with no run draw in `draws` is not ready.
    template <typename DrawState>
    static bool prelude_ready(bool armed, int prelude, std::span<const int> prelude_of_draw,
                              std::span<DrawState> draws) {
        bool any = false;
        for (size_t d = 0; d < prelude_of_draw.size(); ++d) {
            if (prelude_of_draw[d] != prelude) continue;
            if (d >= draws.size() || !draws[d].ok) return false;
            any = true;
        }
        return armed && any;
    }

    // After the render pass ends: the counters' vertex-stage writes, made available to the host
    // read in the completion callback. No wait is recorded or performed here.
    void record_after_pass(VkCommandBuffer cmd) const {
        for (const Prelude& prelude : preludes_)
            if (prelude.recorded && prelude.counters.chunk)
                prosper::gpu::record_host_read_barrier(cmd, prelude.counters.buffer(),
                                                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                                                       VK_ACCESS_SHADER_WRITE_BIT);
    }

private:
    struct Prelude {
        std::shared_ptr<const prosper::gpu::NggSubgroupDraw> ngg;
        size_t first_draw = 0;
        std::vector<std::shared_ptr<const NggShellPipeline>> pipelines;   // per group
        std::vector<NggScratchSlice> launch, exports;   // per group
        NggScratchSlice counters;
        VkDescriptorSet guest_set = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> shell_sets;
        bool armed = false, recorded = false;
    };

    bool fail_capture(Prelude& prelude, const char* why) {
        static std::atomic<uint32_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 16u)
            std::fprintf(stderr, "[ngg-backend] draw not recorded: prelude %s failed\n", why);
        prelude.armed = false;
        return true;   // the draw goes on to pipeline creation; record() drops all of its runs
    }

    bool refuse(std::string& refusal, const char* reason) {
        refusal = std::string("reason=") + reason;
        return false;
    }

    bool expand_one(const BackendDraw& draw, std::vector<BackendDraw>& out, std::string& refusal) {
        using namespace prosper::gpu;
        const NggSubgroupDraw& ngg = *draw.ngg_subgroup;
        // The device half of admission, shared with the producer (ngg_draw_admission.hpp).
        if (const char* device = ngg_device_refusal(ngg, ngg_host_capabilities(ctx_)))
            return refuse(refusal, device);
        if (draw.mesh_draw || draw.owned_waves || draw.fragment_draw_inputs || draw.raster_quads ||
            draw.index_count())
            return refuse(refusal, "ngg-backend-draw-shape");
        // The shell's guest inputs: exactly one plain buffer per declared set-0 binding, and set 2
        // free for the export and counter views.
        const auto resource_count = [&](uint32_t set, uint32_t binding, bool* buffer) {
            uint32_t count = 0;
            for (const FrameResource& r : draw.R)
                if (r.set == set && r.binding == binding) {
                    ++count;
                    *buffer = !r.is_texture() && r.table_entries.empty() && !r.is_internal_gds;
                }
            for (const FrameBufferResource& r : draw.B)
                if (r.set == set && r.binding == binding) {
                    ++count;
                    *buffer =
                        r.table_entries.empty() && !r.is_internal_gds && !r.fragment_draw_buffer;
                }
            return count;
        };
        for (uint32_t binding : ngg.guest_bindings) {
            bool buffer = false;
            if (resource_count(0, binding, &buffer) != 1 || !buffer)
                return refuse(refusal, "ngg-backend-guest-binding");
        }
        for (const FrameResource& r : draw.R)
            if (r.set == kNggRasterDescriptorSet) return refuse(refusal, "ngg-backend-set2-taken");
        for (const FrameBufferResource& r : draw.B)
            if (r.set == kNggRasterDescriptorSet) return refuse(refusal, "ngg-backend-set2-taken");

        Prelude prelude;
        prelude.ngg = draw.ngg_subgroup;
        prelude.first_draw = out.size();
        const auto release_all = [&] {
            for (const NggScratchSlice& slice : prelude.launch)
                NggScratchRing::instance().release(slice);
            for (const NggScratchSlice& slice : prelude.exports)
                NggScratchRing::instance().release(slice);
            NggScratchRing::instance().release(prelude.counters);
        };
        auto& ring = NggScratchRing::instance();
        for (const NggSubgroupWaveGroup& group : ngg.groups) {
            const VkDeviceSize launch_bytes = group.launch_words.size() * VkDeviceSize{4};
            const VkDeviceSize export_bytes = group.export_words * VkDeviceSize{4};
            // The workgroup and buffer limits were checked by ngg_device_refusal above.
            auto pipeline = ngg_shell_pipeline(ctx_, *group.stages, ngg.guest_bindings,
                                               static_cast<uint32_t>(ngg.push_constants.size()),
                                               ngg.native_wave64);
            prelude.pipelines.push_back(pipeline);
            prelude.launch.push_back(ring.acquire(ctx_, true, launch_bytes));
            prelude.exports.push_back(ring.acquire(ctx_, false, export_bytes));
            if (!pipeline || !prelude.launch.back().chunk || !prelude.exports.back().chunk) {
                release_all();
                return refuse(refusal, "ngg-backend-resources");
            }
            // Ring slices are rounded up; the descriptors cover exactly the meaningful words.
            prelude.launch.back().bytes = launch_bytes;
            prelude.exports.back().bytes = export_bytes;
            std::memcpy(prelude.launch.back().words(), group.launch_words.data(),
                        static_cast<size_t>(launch_bytes));
        }
        if (ngg.count_violations) {
            constexpr VkDeviceSize kCounterBytes = VkDeviceSize{kNggViolationWords} * 4u;
            prelude.counters = ring.acquire(ctx_, true, kCounterBytes);
            if (!prelude.counters.chunk) {
                release_all();
                return refuse(refusal, "ngg-backend-resources");
            }
            prelude.counters.bytes = kCounterBytes;
            // The GPU fill zeroes them before use; zero them here too, so a batch that is never
            // submitted reads back zeros rather than a previous lease's counts.
            std::memset(prelude.counters.words(), 0, static_cast<size_t>(kCounterBytes));
        }

        // The pass-through topology replaces the guest's input topology; everything else of the
        // fixed-function state is the draw's own.
        auto& state = states_.emplace_back(draw.ps ? *draw.ps : ResolvedPipelineState{});
        state.topology = ngg.topology == NggOutputTopology::LineList
                             ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                             : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        const auto token = std::shared_ptr<const void>(shared_from_this());
        const int index = static_cast<int>(preludes_.size());
        // Runs after the first carry no set-0 resources: only the first run's descriptor writes
        // feed the shell, and no pass-through or pixel stage reads set 0.
        BackendDraw later_runs = draw;
        strip_set0(later_runs);
        for (const NggSubgroupRun& run : ngg.runs) {
            const NggSubgroupWaveGroup& group = ngg.groups[run.group];
            const uint32_t per_block = 64u * group.waves * ngg.vertices_per_primitive;
            BackendDraw expanded = &run == &ngg.runs.front() ? draw : later_runs;
            expanded.ngg_subgroup.reset();
            expanded.vs.clear();
            expanded.vs_shared = group.stages->raster_vertex;
            expanded.vs_identity = 0;
            expanded.gs = group.stages->raster_geometry;
            expanded.ps = &state;
            expanded.vcount = run.blocks * per_block;
            expanded.vertex_offset = static_cast<int32_t>(run.first_block * per_block);
            expanded.instance_count = 1;
            const NggScratchSlice& exports = prelude.exports[run.group];
            add_view(expanded, kNggRasterExportBinding,
                     FragmentDrawGpuBuffer::view(ctx_.dev, exports.buffer(), exports.offset,
                                                 exports.bytes, token));
            if (ngg.count_violations)
                add_view(expanded, kNggRasterCounterBinding,
                         FragmentDrawGpuBuffer::view(ctx_.dev, prelude.counters.buffer(),
                                                     prelude.counters.offset,
                                                     prelude.counters.bytes, token));
            out.push_back(std::move(expanded));
            prelude_of_draw_.push_back(index);
        }
        preludes_.push_back(std::move(prelude));
        return true;
    }

    // Removes every set-0 resource, keeping the frontend's binding order of the rest.
    static void strip_set0(BackendDraw& draw) {
        std::vector<FrameResource> full;
        std::vector<FrameBufferResource> buffers;
        std::vector<uint32_t> order;
        const auto keep_full = [&](uint32_t index) {
            if (draw.R[index].set == 0) return;
            order.push_back(static_cast<uint32_t>(full.size()));
            full.push_back(std::move(draw.R[index]));
        };
        const auto keep_buffer = [&](uint32_t index) {
            if (draw.B[index].set == 0) return;
            order.push_back(0x80000000u | static_cast<uint32_t>(buffers.size()));
            buffers.push_back(std::move(draw.B[index]));
        };
        if (draw.resource_order.empty()) {
            for (uint32_t i = 0; i < draw.R.size(); ++i) keep_full(i);
            for (uint32_t i = 0; i < draw.B.size(); ++i) keep_buffer(i);
        } else {
            for (uint32_t token : draw.resource_order) {
                const uint32_t i = token & 0x7fffffffu;
                if (token & 0x80000000u) keep_buffer(i);
                else keep_full(i);
            }
        }
        draw.R = std::move(full);
        draw.B = std::move(buffers);
        draw.resource_order = draw.B.empty() ? std::vector<uint32_t>{} : std::move(order);
    }

    static void add_view(BackendDraw& draw, uint32_t binding,
                         std::shared_ptr<const FragmentDrawGpuBuffer> view) {
        // A compact (B) resource needs explicit order metadata. An empty order means "every
        // resource is in R" (with B empty), so spell that order out before appending.
        if (draw.resource_order.empty())
            for (uint32_t index = 0; index < draw.R.size(); ++index)
                draw.resource_order.push_back(index);
        FrameBufferResource resource;
        resource.set = prosper::gpu::kNggRasterDescriptorSet;
        resource.binding = binding;
        resource.fragment_draw_buffer = std::move(view);
        draw.B.push_back(std::move(resource));
        draw.resource_order.push_back(0x80000000u | static_cast<uint32_t>(draw.B.size() - 1u));
    }

    void read_counters() {
        if (submission_failed_) return;
        auto& stats = ngg_subgroup_backend_stats();
        for (const Prelude& prelude : preludes_) {
            if (!prelude.recorded) continue;
            stats.completed.fetch_add(1, std::memory_order_relaxed);
            const uint32_t* words = prelude.counters.words();
            if (!words) continue;
            stats.invalid_blocks.fetch_add(words[prosper::gpu::kNggViolationInvalidBlocks]);
            stats.connectivity.fetch_add(words[prosper::gpu::kNggViolationConnectivity]);
            stats.layer_culled.fetch_add(words[prosper::gpu::kNggViolationLayerCulled]);
            if (words[0] | words[1] | words[2]) {
                static std::atomic<uint32_t> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 16u)
                    std::fprintf(stderr,
                                 "[ngg-backend] guest protocol violations: invalid-blocks=%u "
                                 "connectivity=%u layer-culled=%u (first 16 reported)\n",
                                 words[0], words[1], words[2]);
            }
        }
    }

    const RenderVkCtx& ctx_;
    std::vector<Prelude> preludes_;
    std::vector<int> prelude_of_draw_;   // per expanded draw: its prelude, or -1
    std::deque<prosper::gpu::ResolvedPipelineState> states_;   // stable addresses
    bool submission_failed_ = false;
};

// A merged-NGG draw is a backend segment of its own: the first split point of `draws` is before
// the first NGG draw, or after it when it comes first. draws.size() when there is none.
inline size_t ngg_segment_split_index(std::span<const BackendDraw> draws) {
    for (size_t i = 0; i < draws.size(); ++i)
        if (draws[i].ngg_subgroup) return i ? i : 1u;
    return draws.size();
}

// Before render_draws_rgba splits a batch: the NGG draws this call can run. An NGG draw is dropped
// (counted under backend/ngg-subgroup) when the device cannot run it (ngg_device_refusal, the same
// answer the producer got), or when it shares the call with other draws and its own segment would
// lose them something: transient depth (`persist_depth_stencil` false, e.g. PROSPER_DUMP_DRAWSTEPS)
// or the colour result (no persistent colour target, so each split reads the pass back with a CPU
// wait -- CLAUDE.md P1). Returns `draws` itself when nothing is dropped, else `kept`.
inline std::span<const BackendDraw> ngg_admit_backend_draws(const std::vector<BackendDraw>& draws,
                                                            bool persist_depth_stencil,
                                                            const BackendColorTarget* color_target,
                                                            std::vector<BackendDraw>& kept) {
    if (std::none_of(draws.begin(), draws.end(),
                     [](const BackendDraw& d) { return bool(d.ngg_subgroup); }))
        return draws;
    const prosper::gpu::NggHostCapabilities host = ngg_host_capabilities(render_vk_ctx());
    const bool splits_safely =
        draws.size() == 1u ||
        (persist_depth_stencil && color_target && color_target->persistent_id);
    const auto refusal = [&](const BackendDraw& d) -> const char* {
        if (!d.ngg_subgroup) return nullptr;
        if (const char* device = prosper::gpu::ngg_device_refusal(*d.ngg_subgroup, host))
            return device;
        if (!persist_depth_stencil && !splits_safely) return "ngg-backend-transient-depth-split";
        if (!splits_safely) return "ngg-backend-readback-split";
        return nullptr;
    };
    if (std::none_of(draws.begin(), draws.end(), refusal)) return draws;
    kept.clear();
    for (const BackendDraw& d : draws) {
        if (const char* reason = refusal(d)) {
            note_ngg_backend_drop(reason, false);
            continue;
        }
        kept.push_back(d);
    }
    return kept;
}

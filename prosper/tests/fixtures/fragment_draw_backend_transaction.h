// Same-TU backend coordinator. Only a sealed original-PS plan may substitute replay SOURCE.
// Geometry/index uploads remain the normal backend's leases and are retained together through
// completion. The first attachment recipe has no DS tests/clears, samples, GS or mutable VB inputs.
#pragma once

struct FragmentDrawBackendStats {
    uint64_t planned = 0, refused = 0, recorded = 0;
};
inline FragmentDrawBackendStats& fragment_draw_backend_stats() {
    static thread_local FragmentDrawBackendStats value;
    return value;
}

class FragmentDrawBackendBatch {
public:
    FragmentDrawBackendBatch(const RenderVkCtx& context, std::span<const BackendDraw> draws,
                             uint32_t width, uint32_t height, uint32_t colors)
        : context_(&context), width_(width), height_(height), states_(draws.size()) {
        bool readonly_pass_checked = false, readonly_pass = false;
        for (size_t index = 0; index < draws.size(); ++index) {
            const auto& draw = draws[index];
            if (!draw.fragment_draw_inputs) continue;
            auto& state = states_[index];
            state.attachment_guard = true;
            const auto& inputs = draw.fragment_draw_inputs;
            const bool producing_match =
                !draw.raster_quad_contract_modified &&
                (inputs->original_fragment_producer
                     ? inputs->original_fragment_producer->matches(*inputs) &&
                           inputs->original_fragment_producer->matches_modules(
                               draw.vs_shared, draw.gs_words(), draw.fs_words())
                     : inputs->source_vs && inputs->source_gs && inputs->source_fs &&
                           *inputs->source_vs == draw.vs_words() &&
                           *inputs->source_gs == draw.gs_words() &&
                           *inputs->source_fs == draw.fs_words());
            const auto prepared =
                prosper::gpu::prepare_fragment_packet_inputs(inputs, producing_match);
            const prosper::gpu::FragmentPacketDeviceContract device{
                reinterpret_cast<uintptr_t>(context.dev), context.shader_int64_enabled, false};
            const auto program =
                prosper::gpu::cached_fragment_draw_program(*inputs, *prepared, device, 4096);
            if (!program->rejection_reason().empty()) {
                refuse(state, draw, program->rejection_reason());
                continue;
            }
            const auto* ps = draw.ps;
            if (!ps || colors != 1 || draw.mesh_draw || !draw.gs_words().empty() ||
                draw.instance_count != 1 || ps->topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
                ps->polygon_mode != VK_POLYGON_MODE_FILL || ps->depth_test_enable ||
                ps->depth_write_enable || ps->stencil_enable || ps->depth_clear_enable ||
                ps->stencil_clear_enable || raster_quad_raster_state_overridden() ||
                PROSPER_ENV_ON("PROSPER_DEPTH_CLEAR_PROBE")) {
                refuse(state, draw, "fragment-draw-attachment-raster-recipe-unimplemented");
                continue;
            }
            bool samples = false;
            for (const auto& target : ps->color_targets) samples |= target.log2_samples != 0;
            const auto indices = draw.index_words();
            const uint64_t vertices = indices.empty() ? draw.vcount : indices.size();
            if (samples || !vertices || vertices > UINT32_MAX || vertices % 3 ||
                (indices.empty() && draw.vertex_offset < 0) ||
                (ps->depth_bias_enable && ps->depth_bias_clamp &&
                 !context.depth_bias_clamp_enabled)) {
                refuse(state, draw, "fragment-draw-geometry-sample-recipe-unimplemented");
                continue;
            }
            // Collection can precede the pass only if no attachment draw can mutate its shared
            // uploaded vertex buffers before replay. This is source-owned negative effect proof,
            // not a residency budget or a guess from the absence of a visible store in the VS.
            if (!readonly_pass_checked) {
                readonly_pass_checked = true;
                const bool original_effects_required =
                    std::any_of(draws.begin(), draws.end(), [](const BackendDraw& value) {
                        return value.fragment_draw_inputs &&
                               value.fragment_draw_inputs->original_fragment_producer;
                    });
                readonly_pass =
                    std::all_of(draws.begin(), draws.end(), [&](const BackendDraw& value) {
                        if (original_effects_required &&
                            (value.raster_quad_contract_modified || value.mesh_draw ||
                             !value.original_graphics_effects ||
                             !value.original_graphics_effects->matches_draw(
                                 value.source_submit, value.command_order, value.vs_shared,
                                 value.fs_shared, value.gs_words(), value.fs_words())))
                            return false;
                        const bool original_bank_fragment =
                            value.fragment_draw_inputs &&
                            value.fragment_draw_inputs->original_fragment_producer &&
                            value.fragment_draw_inputs->original_fragment_producer->matches(
                                *value.fragment_draw_inputs) &&
                            value.fragment_draw_inputs->original_fragment_producer->matches_modules(
                                value.vs_shared, value.gs_words(), value.fs_words());
                        return backend_module_has_readonly_buffers(value.vs_words(),
                                                                   value.vs_shared) &&
                               (original_bank_fragment || backend_module_has_readonly_buffers(
                                                              value.fs_words(), value.fs_shared)) &&
                               (value.gs_words().empty() ||
                                backend_module_has_readonly_buffers(value.gs_words()));
                    });
            }
            if (!readonly_pass) {
                refuse(state, draw, "fragment-draw-replay-geometry-input-mutation-unproved");
                continue;
            }
            std::vector<uint32_t> bindings;
            bool resources = true;
            const auto inspect = [&](const FrameBufferResource& value, bool texture) {
                resources &=
                    !texture && value.set == 0 && !value.is_internal_gds &&
                    value.table_entries.empty() && !value.fragment_draw_buffer &&
                    value.buffer_words_data() && value.buffer_word_count() &&
                    value.buffer_word_count() <= context.detile_limits.maxStorageBufferRange / 4;
                if (resources) bindings.push_back(value.binding);
            };
            for (const auto& value : draw.R) inspect(value, value.is_texture());
            for (const auto& value : draw.B) inspect(value, false);
            if (!resources) {
                refuse(state, draw, "fragment-draw-geometry-resource-lease-unimplemented");
                continue;
            }
            std::string rejection;
            state.collect = FragmentDrawCollectGpuProgram::acquire(
                context, program, inputs->source_vs, std::move(bindings), ps->depth_bias_enable,
                rejection);
            if (!state.collect) {
                refuse(state, draw, rejection);
                continue;
            }
            for (const auto& [binding, required] : state.collect->required_bytes()) {
                const FrameBufferResource* source = nullptr;
                for (const auto& value : draw.R)
                    if (value.binding == binding) source = &value;
                for (const auto& value : draw.B)
                    if (value.binding == binding) source = &value;
                if (!source || required > uint64_t(source->buffer_word_count()) * 4)
                    resources = false;
            }
            if (!resources) {
                refuse(state, draw, "fragment-draw-geometry-buffer-extent-unavailable");
                continue;
            }
            auto transaction = prosper::gpu::instantiate_fragment_draw_transaction(
                program, inputs, *prepared, width, height, uint32_t(vertices / 3),
                reinterpret_cast<uintptr_t>(context.dev));
            if (!transaction.rejection().empty()) {
                refuse(state, draw, transaction.rejection());
                continue;
            }
            // The normal flush decision is made only after resource realization. Prelease the
            // opt-in usage here; staging/copy/readback remain absent until that decision is true.
            const bool observe = fragment_draw_observation_enabled() && index < 8;
            state.owner =
                FragmentDrawGpuOwner::create(context, std::move(transaction), rejection, observe);
            state.compute = FragmentDrawComputeGpuProgram::acquire(context, program, rejection);
            state.target =
                FragmentDrawCollectFramebuffer::acquire(context, state.collect, width, height);
            if (!state.owner || !state.compute || !state.target) {
                refuse(state, draw,
                       rejection.empty() ? "fragment-draw-retained-device-lease-unavailable"
                                         : rejection);
                continue;
            }
            state.replay = std::make_unique<BackendDraw>(draw);
            auto& replay = *state.replay;
            replay.fs.clear();
            replay.fs_shared = program->replay_owner();
            replay.fs_identity = 0;   // never use an original-module identity for different SOURCE
            replay.fragment_wave_policy = prosper::gpu::FragmentWavePolicy::Strict;
            replay.allow_native_fragment_vote_width = replay.allow_partial_wave_fragment = false;
            if (replay.resource_order.empty()) {
                for (uint32_t index = 0; index < replay.R.size(); ++index)
                    replay.resource_order.push_back(index);
                for (uint32_t index = 0; index < replay.B.size(); ++index)
                    replay.resource_order.push_back(0x80000000u | index);
            }
            using O = FragmentDrawGpuOwner;
            const std::array<std::shared_ptr<const FragmentDrawGpuBuffer>, 5> buffers{
                state.owner->view(O::Input), state.owner->authority_view(),
                state.owner->view(O::Collector), state.owner->view(O::Output),
                state.owner->view(O::Commit)};
            for (uint32_t index = 0; index < buffers.size(); ++index) {
                FrameBufferResource resource;
                resource.set = 1;
                resource.binding = index + 1;
                resource.fragment_draw_buffer = buffers[index];
                replay.resource_order.push_back(0x80000000u | uint32_t(replay.B.size()));
                replay.B.push_back(std::move(resource));
            }
            ++fragment_draw_backend_stats().planned;
        }
    }
    const BackendDraw& draw(size_t index, const BackendDraw& original) const {
        return states_[index].replay ? *states_[index].replay : original;
    }
    bool replay(size_t index) const { return bool(states_[index].replay); }
    bool attachment_guard(size_t index) const { return states_[index].attachment_guard; }
    uint64_t additional_sets() const {
        return uint64_t(std::count_if(states_.begin(), states_.end(),
                                      [](const State& state) { return bool(state.replay); })) *
               5;
    }
    uint64_t additional_storage_descriptors() const {
        uint64_t result = 0;
        for (const auto& state : states_)
            if (state.replay) result += uint64_t(state.compute->storage_bindings()) * 4 + 1;
        return result;
    }
    bool allocate(size_t index, VkDescriptorPool pool) {
        auto& state = states_[index];
        if (!state.replay) return true;
        return allocate_fragment_draw_compute_sets(context_->dev, pool, *state.compute,
                                                   *state.owner, state.sets) &&
               allocate_fragment_draw_collect_set(context_->dev, pool, *state.collect, *state.owner,
                                                  state.collector_set);
    }
    // Called only after the existing real flush_now decision, before command recording. Deferred
    // batches never allocate observation staging, record a copy, read a plane or add a wait.
    template <class Draw>
    void prepare_completed_observations(std::span<const Draw> draws) {
        if (!fragment_draw_observation_enabled()) return;
        for (size_t index = 0; index < std::min(states_.size(), size_t(8)); ++index) {
            auto& state = states_[index];
            if (state.replay && state.owner && draws[index].ok && !state.observation)
                state.observation =
                    FragmentDrawGpuObservation::create(*context_, state.owner, index);
        }
    }
    template <class Draw>
    void record(VkCommandBuffer command, std::span<const Draw> draws) const {
        for (size_t index = 0; index < states_.size(); ++index) {
            const auto& state = states_[index];
            if (!state.replay || !draws[index].ok) continue;
            state.owner->record_zero_and_upload_visibility(command);
            record_fragment_draw_collect(command, *state.target, state.collector_set, draws[index],
                                         width_, height_);
            record_fragment_draw_compute_transaction(command, *state.compute, *state.owner,
                                                     state.sets);
            ++fragment_draw_backend_stats()
                  .recorded;   // recording, NOT completion/publication proof
        }
    }
    template <class Draw>
    void record_observations_after_replay(VkCommandBuffer command,
                                          std::span<const Draw> draws) const {
        for (size_t index = 0; index < states_.size(); ++index)
            if (states_[index].observation && draws[index].ok)
                states_[index].observation->record_after_replay(command);
    }
    void report_completed_observations() const {
        for (const auto& state : states_)
            if (state.observation) state.observation->report_completed();
    }

private:
    struct State {
        bool attachment_guard = false;
        std::unique_ptr<BackendDraw> replay;
        std::shared_ptr<FragmentDrawGpuOwner> owner;
        std::unique_ptr<FragmentDrawGpuObservation> observation;
        std::shared_ptr<const FragmentDrawComputeGpuProgram> compute;
        std::shared_ptr<const FragmentDrawCollectGpuProgram> collect;
        std::shared_ptr<const FragmentDrawCollectFramebuffer> target;
        std::array<VkDescriptorSet, FragmentDrawComputeGpuProgram::Stages> sets{};
        VkDescriptorSet collector_set = VK_NULL_HANDLE;
        std::string rejection;
    };
    static void refuse(State& state, const BackendDraw& draw, std::string rejection) {
        state.rejection = std::move(rejection);
        ++fragment_draw_backend_stats().refused;
        // Bounded diagnostics still name each contract class; missing guest authority is not
        // hidden behind the host-width gate or turned into an all-present private scratch slot.
        static thread_local std::set<std::pair<uint64_t, std::string>> reported;
        if (reported.size() < 1024 && reported.emplace(draw.fs_identity, state.rejection).second)
            std::fprintf(
                stderr,
                "[fragment-draw] refused reason=%s draw=%llu; no emulated attachment admission\n",
                state.rejection.c_str(), static_cast<unsigned long long>(draw.draw_index));
    }
    const RenderVkCtx* const context_;
    const uint32_t width_, height_;
    std::vector<State> states_;
};

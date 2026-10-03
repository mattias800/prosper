// Registered direct-stage inputs become an immutable owned logical-wave plan before execution.
// Missing launch/entry/producer/allocation authority refuses; cached native modules are irrelevant.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/memory/guest_memory_topology.hpp"
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr);
namespace prosper::gpu {
// Registered AGC headers publish the shader blob size in bytes. Dynamic descriptor folding used to
// ignore it and hand the decoder a fixed 0x4000-dword window, allowing the walk to read up to 64 KiB
// past a short shader. Retain that historical 64 KiB ceiling as a WORK bound too: CreateShader accepts
// guest metadata, so a corrupt multi-gigabyte shader_size must not become a page-probe/decode/OOM budget.
// Truncate a non-dword-aligned tail rather than reading one byte beyond the blob, and refuse the bounded
// span if it is not wholly readable. A zero result safely disables only the optional fold;
// metadata-described resources remain available to build_stage_table.
size_t dynamic_fold_shader_dwords(uint32_t shader_size_bytes) {
    constexpr size_t kMaxDynamicFoldDwords = 0x4000;
    return std::min<size_t>(shader_size_bytes / sizeof(uint32_t), kMaxDynamicFoldDwords);
}

size_t registered_shader_dwords(const AgcShaderHeader& header, uint64_t code_addr) {
    const size_t dwords = dynamic_fold_shader_dwords(header.shader_size);
    const uint32_t bounded_bytes = static_cast<uint32_t>(dwords * sizeof(uint32_t));
    if (!bounded_bytes || !guest_readable(code_addr, bounded_bytes)) return 0;
    return dwords;
}

bool graphics_program_requires_owned_waves(uint64_t address) {
    const auto* header =
        static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(address));
    if (!header) return false;
    const auto source = registered_graphics_original(address);
    if (!source) return false;
    std::vector<Rdna2Inst> original;
    rdna2_walk(source->data(), source->size(), original);
    return !rdna2_raw_wave_wide_data_loads(original).empty();
}

bool prepare_draw_owned_waves(const GpuState& state, const GpuState::Draw* draw,
                              uint64_t vertex_address, uint64_t fragment_address,
                              uint32_t vertex_count, FloatTransportConfig profile,
                              const GraphicsRawSnapshotContext* context,
                              std::shared_ptr<const GraphicsOwnedWaveDraw>& owned,
                              std::vector<uint32_t>& indices, std::string& refusal) {
    owned.reset();
    indices.clear();
    refusal.clear();
    const auto reject = [&](const char* reason) {
        refusal = reason;
        return false;
    };
    const bool vertex = graphics_program_requires_owned_waves(vertex_address);
    const bool fragment = graphics_program_requires_owned_waves(fragment_address);
    if (!vertex && !fragment) return true;
    if (!context || !context->producers_complete)
        return reject("draw-wave-prior-producers-incomplete");
    const auto render = extract_render_state(state);
    if (fragment && (render.ps_wave32 || !render.ps_launch_rsrc1.available))
        return reject("draw-wave-known-fragment64-launch-unavailable");
    GraphicsRawSnapshotContext isolated = *context;
    namespace P = prosper::agc::Pm4;
    const auto cx = [&](uint32_t reg) {
        auto it = state.cx.find(reg);
        return it == state.cx.end() ? 0u : it->second;
    };
    for (uint32_t slot = 0; slot < render.color_targets.size(); ++slot) {
        const auto& target = render.color_targets[slot];
        if (!target.base) continue;
        const uint64_t extent = color_target_physical_bytes(target);
        if (!extent) return reject("draw-wave-output-physical-extent-unavailable");
        isolated.output_allocations.emplace_back(target.base, extent);
        for (const auto [lo, hi] :
             {std::pair{P::CB_COLOR0_CMASK + slot * 0xfu, P::CB_COLOR0_CMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_FMASK + slot * 0xfu, P::CB_COLOR0_FMASK_BASE_EXT + slot},
              std::pair{P::CB_COLOR0_DCC_BASE + slot * 0xfu, P::CB_COLOR0_DCC_BASE_EXT + slot}})
            if ((uint64_t(cx(lo)) << 8u) | (uint64_t(cx(hi) & 0xffu) << 40u))
                return reject("draw-wave-output-metadata-extent-unavailable");
    }
    if (render.depth_read_base || render.depth_write_base || render.stencil_read_base ||
        render.stencil_write_base || render.htile_data_base)
        return reject("draw-wave-depth-physical-extent-unavailable");
    // Reuse only the established PS0 / merged-VS8 ABI with a known loaded USER_SGPR prefix.
    // Physical USER_DATA presence alone is not a launched entry value. Nonzero range starts,
    // missing launch metadata, and unobserved required words remain explicit refusals.
    const auto entry = [&](uint64_t address, bool pixel, std::vector<uint32_t>& original,
                           std::vector<std::pair<uint32_t, uint32_t>>& scalars) {
        const auto* header =
            static_cast<const AgcShaderHeader*>(prosper_agc_shader_header_for_code(address));
        if (!header) return false;
        if (!header->specials ||
            !guest_readable(uint64_t(uintptr_t(header->specials)), sizeof(AgcShaderSpecials)) ||
            header->specials->user_data_range_start)
            return false;
        const auto rsrc2 =
            state.sh.find(pixel ? P::SPI_SHADER_PGM_RSRC2_PS : P::SPI_SHADER_PGM_RSRC2_GS);
        if (rsrc2 == state.sh.end()) return false;
        uint32_t loaded =
            pixel ? ((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT) &
                     P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MASK) |
                        (((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT) &
                          P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_MASK)
                         << 5u)
                  : ((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT) &
                     P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MASK) |
                        (((rsrc2->second >> P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_SHIFT) &
                          P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_MSB_MASK)
                         << 5u);
        if (pixel && loaded > kUserSgprs) {
            refusal = "draw-wave-fragment-user-sgpr-count-out-of-range";
            return false;
        }
        if (!loaded || loaded > kUserSgprs || !header->specials->user_data_range_end ||
            header->specials->user_data_range_end > loaded)
            return false;
        const auto source = registered_graphics_original(address);
        if (!source || source->empty() || source->size() > 4096u) return false;
        original = *source;
        const uint32_t base = pixel ? P::SPI_SHADER_USER_DATA_PS_0 : P::SPI_SHADER_USER_DATA_GS_0;
        for (uint32_t word = 0; word < loaded; ++word) {
            auto found = state.sh.find(base + word);
            if (found != state.sh.end())
                scalars.emplace_back((pixel ? 0u : 8u) + word, found->second);
        }
        return true;
    };
    if (vertex && draw && draw->indexed) {
        const uint32_t size = index_elem_bytes(state.index_type);
        if (!state.index_type_announced || !draw->index_count || draw->index_count > 4096u ||
            (size != 2u && size != 4u) || !draw->index_addr ||
            draw->index_addr > UINT64_MAX - uint64_t(draw->index_count) * size)
            return reject("draw-wave-index-shape-unavailable");
        GuestMappingLease lease;
        const uint32_t bytes = draw->index_count * size;
        const auto window = guest_memory_direct_readable_window(lease, draw->index_addr);
        if (!window || uint64_t(bytes) > window.virtual_end - draw->index_addr ||
            !graphics_raw_source_is_guest_current(lease, draw->index_addr, bytes))
            return reject("draw-wave-current-index-bytes-unavailable");
        indices.resize(draw->index_count);
        for (uint32_t index = 0; index < draw->index_count; ++index) {
            uint32_t value = 0;
            std::memcpy(
                &value,
                reinterpret_cast<const void*>(uintptr_t(draw->index_addr + uint64_t(index) * size)),
                size);
            indices[index] = value;
        }
    }
    auto pending = std::make_shared<GraphicsOwnedWaveDraw>();
    pending->vertex_pending = vertex;
    pending->fragment_pending = fragment;
    if (vertex) {
        std::vector<uint32_t> original;
        std::vector<std::pair<uint32_t, uint32_t>> scalars;
        if (!entry(vertex_address, false, original, scalars))
            return reject("draw-wave-vertex-entry-unavailable");
        if (!prepare_owned_vertex_waves(original, scalars, indices, vertex_count,
                                        draw && draw->has_vertex_offset_override
                                            ? draw->indirect_vertex_offset
                                            : int32_t(render.ge_indx_offset),
                                        draw ? draw->instance_count : state.num_instances, profile,
                                        &isolated, pending->vertex, refusal))
            return false;
    }
    if (fragment) {
        std::vector<uint32_t> original;
        if (!entry(fragment_address, true, original, pending->fragment_scalars))
            return refusal.empty() ? reject("draw-wave-fragment-entry-unavailable") : false;
        if (!observe_graphics_raw_wave_windows(original, pending->fragment_scalars, &isolated,
                                               pending->fragment_windows, refusal))
            return false;
        pending->fragment_code = std::make_shared<const std::vector<uint32_t>>(std::move(original));
    }
    owned = std::move(pending);
    return true;
}

}   // namespace prosper::gpu

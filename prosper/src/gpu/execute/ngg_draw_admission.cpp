// ngg_draw_admission.cpp -- see ngg_draw_admission.hpp.
#include "gpu/execute/ngg_draw_admission.hpp"

#include "gpu/pm4/vgt_shader_stages.hpp"

#include <cstdint>
#include <mutex>
#include <string>

namespace prosper::gpu {

namespace {

std::mutex& host_mutex() {
    static std::mutex mutex;
    return mutex;
}
NggHostCapabilities& host_slot() {
    static NggHostCapabilities capabilities;
    return capabilities;
}

constexpr uint32_t kPrimTriangleList = 4;
constexpr uint32_t kPrimTriangleStrip = 6;
constexpr uint32_t kWaveLanes = 64;

}   // namespace

void publish_ngg_host_capabilities(const NggHostCapabilities& capabilities) {
    const std::lock_guard lock(host_mutex());
    host_slot() = capabilities;
    host_slot().published = true;
}

NggHostCapabilities published_ngg_host_capabilities() {
    const std::lock_guard lock(host_mutex());
    return host_slot();
}

NggDrawAdmission admit_ngg_draw(const NggDrawRegisters& registers, const NggDrawFacts& facts,
                                const NggHostCapabilities& host) {
    NggDrawAdmission admission;
    const VgtShaderStages stages{registers.vgt_shader_stages_en};
    if (!stages.primgen_enabled()) return admission;
    // #3135 P7: NGG without a GS runs the VS as the primitive shader. Its launch is the merged
    // one (s3 counts, v0/v1 vertex offsets scaled by ESGS_RING_ITEMSIZE, v5 VertexID, v8
    // InstanceID): Kena's culling VS programs read exactly those, and the ABI analysis admits them
    // unchanged. Only the partition differs (NggSubgroupLimits::vs_only).
    admission.vs_only = !stages.gs_enabled();
    admission.applies = true;
    const auto refuse = [&](const char* reason) {
        admission.refusal = reason;
        return admission;
    };
    if (!host.published) return refuse("ngg-host-unpublished");
    if (registers.missing) return refuse("ngg-register-missing");
    if (stages.gs_wave32()) return refuse("ngg-gs-wave32");
    if (ngg_gs_instance_count(registers.vgt_gs_instance_cnt) > 1u)
        return refuse("ngg-gs-instancing");
    if (registers.primitive_type == kPrimTriangleList)
        admission.shape.topology = NggInputTopology::TriangleList;
    else if (registers.primitive_type == kPrimTriangleStrip)
        admission.shape.topology = NggInputTopology::TriangleStrip;
    else
        return refuse("ngg-input-topology");
    // Without a GS there is no GS output primitive type to read: the primitive shader exports the
    // INPUT primitives it keeps (culling drops some, never changes their kind), so a triangle list
    // or strip comes out as triangles. VGT_GS_OUT_PRIM_TYPE is not consulted for it. Kena's culling
    // VS programs run with that register at 0 -- POINTLIST if it applied -- while the title draws
    // their geometry as triangles on PS5. CONFIDENCE: MED (Kena's register values and its PS5
    // picture; no published register reference states the VS-only rule). Inputs other than a
    // triangle list or strip were refused above (ngg-input-topology), by name.
    admission.topology = admission.vs_only ? NggOutputTopology::TriangleList
                                           : ngg_output_topology(registers.vgt_gs_out_prim_type);
    if (admission.topology == NggOutputTopology::Unsupported) return refuse("ngg-output-topology");
    if (facts.indexed) {
        if (facts.index_refusal) return refuse(facts.index_refusal);
        if (!facts.indices) return refuse("ngg-index-unavailable");
    }
    if (facts.indirect) return refuse("ngg-indirect");
    if (facts.vertex_offset) return refuse("ngg-vertex-offset");

    const uint32_t vs_out = registers.pa_cl_vs_out_cntl;
    if (vs_out & kVsOutUseVtxViewportIndx) return refuse("ngg-viewport-index");
    if (vs_out & kVsOutUseVtxPointSize) return refuse("ngg-point-size");
    if (vs_out & (kVsOutClipDistanceMask | kVsOutCullDistanceMask | kVsOutCcDist0VecEna |
                  kVsOutCcDist1VecEna))
        return refuse("ngg-clip-cull-distance");
    if (registers.pa_cl_clip_cntl & 0x3fu) return refuse("ngg-user-clip-plane");   // UCP_ENA [5:0]
    if (vs_out & kVsOutUseVtxKillFlag) return refuse("ngg-vertex-kill-flag");
    if (vs_out & kVsOutUndecodedMask) return refuse("ngg-vs-out-undecoded");
    admission.layer_from_pos1 = (vs_out & kVsOutUseVtxRenderTargetIndx) != 0;
    if (admission.layer_from_pos1) {
        if (facts.target_slices) {
            if (facts.target_first_slice) return refuse("ngg-layer-slice-start");
            admission.layer_slices = facts.target_slices;
        } else if (facts.target_single_slice) {
            // A one-slice view: layer 0 is its only slice. A primitive naming another layer is
            // culled and counted by the raster commit, the rule a volume's out-of-range layers
            // already follow (CONFIDENCE: LOW on cull versus clamp, ngg_raster_commit.hpp) -- here
            // it is visible as a counted cull rather than a draw silently placed at slice 0.
            // Kena's culling VS programs export POS1.z into 2D scene targets (#3135 P7).
            admission.layer_slices = 1;
        } else {
            return refuse("ngg-layer-target-not-layered");
        }
    }
    admission.provoking_vertex_last = (registers.pa_su_sc_mode_cntl >> 19) & 1u;
    if (admission.shape.topology == NggInputTopology::TriangleStrip) {
        const bool culls = (registers.pa_su_sc_mode_cntl & 3u) != 0;   // CULL_FRONT, CULL_BACK
        if (culls || (registers.spi_ps_input_ena & kPsInputFrontFace) || facts.flat_input_mask ||
            facts.raw_vertex_input_mask)
            return refuse("ngg-strip-order-visible");
    }

    if (!facts.user_data_range_known || facts.user_data_range_start != 0 ||
        facts.user_data_range_end > kNggShellMaxPushWords)
        return refuse("ngg-user-data-range");
    // The SPI loads RSRC2_GS.USER_SGPR user SGPRs (s8..) from SPI_SHADER_USER_DATA_GS_*: that is
    // the hardware's count. AGC leaves the field zero on some programs (Kena's LUT producer reads
    // exactly its 8-dword range), and then the range is the count. A count below a longer range is
    // ordinary: the rest of the user data is reached through the user-data address in s0:s1, as
    // Kena's 4324d9f3 does (USER_SGPR 12, range 0..24). CONFIDENCE: HIGH on the field's meaning,
    // MED on reading zero as "the range" (#3135). Whether two more push words are needed for s0:s1
    // depends on the program, so the live producer checks that room (ngg_live_draw.cpp).
    const uint32_t rsrc2_sgprs = ngg_rsrc2_gs_user_sgprs(registers.spi_shader_pgm_rsrc2_gs);
    admission.user_sgprs = rsrc2_sgprs ? rsrc2_sgprs : facts.user_data_range_end;
    if (admission.user_sgprs > kNggShellMaxPushWords) return refuse("ngg-user-sgpr-count");

    admission.lds_granules = ngg_rsrc2_gs_lds_size(registers.spi_shader_pgm_rsrc2_gs);
    const uint64_t lds_bytes = uint64_t{admission.lds_granules} * kNggLdsGranuleDwords * 4u;
    if (admission.lds_granules > kNggMaxLdsGranules || lds_bytes > host.max_compute_shared_memory)
        return refuse("ngg-lds-limit");
    if (!host.compute) return refuse("ngg-host-compute");

    admission.limits = decode_ngg_subgroup_limits(
        registers.vgt_gs_onchip_cntl, registers.ge_cntl, registers.ge_max_output_per_subgroup,
        registers.vgt_gs_max_vert_out, registers.vgt_esgs_ring_itemsize);
    admission.limits.vs_only = admission.vs_only;
    admission.shape.vertex_count = facts.vertex_count;
    if (facts.indexed) {
        admission.shape.indices = facts.indices;
        admission.shape.vertex_count = static_cast<uint32_t>(facts.indices->size());
    }
    admission.shape.instance_count = facts.instance_count;
    admission.shape.first_vertex = 0;

    NggLayerRouteQuery query;
    query.topology = admission.topology;
    query.layer_from_pos1 = admission.layer_from_pos1;
    query.interpolation_geometry_required = facts.interpolation_geometry_required;
    query.shader_output_layer = host.shader_output_layer;
    query.geometry_shader = host.geometry_shader;
    std::string why;
    admission.route = select_ngg_layer_route(query, &why);
    if (why.find("ngg-interpolation-geometry-needs-triangles") != std::string::npos)
        return refuse("ngg-interpolation-geometry-needs-triangles");
    if (!why.empty()) return refuse("ngg-layer-route-unavailable");
    if (facts.interpolation_geometry_required && !host.geometry_shader)
        return refuse("ngg-layer-route-unavailable");
    admission.count_violations = host.vertex_pipeline_stores;
    admission.native_wave64 = host.native_wave64;
    return admission;
}

const char* ngg_device_refusal(const NggSubgroupDraw& draw, const NggHostCapabilities& host) {
    if (!host.compute) return "ngg-backend-no-compute";
    if (draw.groups.empty() || draw.runs.empty() ||
        draw.push_constants.size() * 4u > host.max_push_constants_size)
        return "ngg-backend-description";
    for (const NggSubgroupWaveGroup& group : draw.groups)
        if (!group.stages || !group.stages->shell || !group.stages->raster_vertex)
            return "ngg-backend-description";
    if (draw.count_violations && !host.vertex_pipeline_stores)
        return "ngg-backend-vertex-stores-unavailable";
    const bool geometry_route = draw.route == NggLayerRoute::ForwardingGeometry ||
                                draw.route == NggLayerRoute::InterpolationGeometry;
    bool any_geometry = geometry_route;
    for (const NggSubgroupWaveGroup& group : draw.groups)
        any_geometry |= !group.stages->raster_geometry.empty();
    if ((draw.route == NggLayerRoute::ShaderOutputLayer && !host.shader_output_layer) ||
        (any_geometry && !host.geometry_shader))
        return "ngg-backend-layer-route-unavailable";
    if (draw.native_wave64 && !host.native_wave64) return "ngg-backend-wave64-unavailable";
    if (draw.lds_bytes > host.max_compute_shared_memory) return "ngg-backend-lds-limit";
    for (const NggSubgroupWaveGroup& group : draw.groups) {
        const uint32_t local = kWaveLanes * group.waves;
        if (local > host.max_compute_workgroup_size_x ||
            local > host.max_compute_workgroup_invocations ||
            group.blocks > host.max_compute_workgroup_count_x ||
            (draw.native_wave64 && group.waves > host.max_compute_workgroup_subgroups))
            return "ngg-backend-workgroup-limit";
        const uint64_t launch_bytes = uint64_t{group.launch_words.size()} * 4u;
        const uint64_t export_bytes = uint64_t{group.export_words} * 4u;
        if (launch_bytes > host.max_storage_buffer_range ||
            export_bytes > host.max_storage_buffer_range)
            return "ngg-backend-buffer-range";
    }
    return nullptr;
}

}   // namespace prosper::gpu

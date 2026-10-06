// ngg_subgroup_draw.cpp -- see ngg_subgroup_draw.hpp.
#include "gpu/execute/ngg_subgroup_draw.hpp"

#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {

namespace {

constexpr uint32_t kWaveLanes = 64;

std::shared_ptr<const NggSubgroupDraw> fail(std::string* refusal, std::string reason) {
    if (refusal) *refusal = "reason=" + std::move(reason);
    return nullptr;
}

bool same_layout(const NggExportRecordLayout& a, const NggExportRecordLayout& b) {
    return a.words_per_lane == b.words_per_lane && a.pos1_word == b.pos1_word &&
           a.first_param_word == b.first_param_word && a.param_targets == b.param_targets &&
           a.prim_channels == b.prim_channels && a.pos0_channels == b.pos0_channels &&
           a.pos1_channels == b.pos1_channels && a.param_channels == b.param_channels;
}

}   // namespace

bool ngg_shell_guest_bindings(const std::vector<uint32_t>& spirv, std::vector<uint32_t>* bindings) {
    // SPIR-V opcodes, decorations and storage classes read below.
    constexpr uint32_t kOpTypeImage = 25, kOpTypeSampler = 26, kOpTypeSampledImage = 27;
    constexpr uint32_t kOpTypeArray = 28, kOpTypeRuntimeArray = 29, kOpTypePointer = 32;
    constexpr uint32_t kOpVariable = 59, kOpDecorate = 71;
    constexpr uint32_t kDecBinding = 33, kDecSet = 34;
    constexpr uint32_t kStorageBuffer = 12;
    std::map<uint32_t, uint32_t> set_of, binding_of, pointee_of, variable_type, variable_class;
    std::set<uint32_t> arrays;
    if (spirv.size() < 5) return false;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16, opcode = spirv[word] & 0xffffu;
        if (!count || word + count > spirv.size()) return false;
        if (opcode == kOpTypeImage || opcode == kOpTypeSampler || opcode == kOpTypeSampledImage)
            return false;
        if (opcode == kOpDecorate && count >= 4) {
            if (spirv[word + 2] == kDecSet) set_of[spirv[word + 1]] = spirv[word + 3];
            if (spirv[word + 2] == kDecBinding) binding_of[spirv[word + 1]] = spirv[word + 3];
        }
        if ((opcode == kOpTypeArray || opcode == kOpTypeRuntimeArray) && count >= 3)
            arrays.insert(spirv[word + 1]);
        if (opcode == kOpTypePointer && count >= 4) pointee_of[spirv[word + 1]] = spirv[word + 3];
        if (opcode == kOpVariable && count >= 4) {
            variable_type[spirv[word + 2]] = spirv[word + 1];
            variable_class[spirv[word + 2]] = spirv[word + 3];
        }
        word += count;
    }
    std::vector<uint32_t> out;
    for (const auto& [id, set] : set_of) {
        const auto binding = binding_of.find(id);
        const auto storage = variable_class.find(id);
        if (binding == binding_of.end() || storage == variable_class.end()) return false;
        // A plain storage buffer: a StorageBuffer-class variable that is not a descriptor array.
        // A Uniform or UniformConstant resource (a UBO, an image, a sampler) is not one.
        if (storage->second != kStorageBuffer) return false;
        const auto pointee = pointee_of.find(variable_type[id]);
        if (pointee == pointee_of.end() || arrays.contains(pointee->second)) return false;
        if (set == kNggShellDescriptorSet) continue;   // the shell's own I/O
        if (set != 0) return false;
        out.push_back(binding->second);
    }
    std::sort(out.begin(), out.end());
    if (std::adjacent_find(out.begin(), out.end()) != out.end()) return false;
    if (bindings) *bindings = std::move(out);
    return true;
}

uint64_t ngg_words_hash(const std::vector<uint32_t>& words) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t word : words) hash = (hash ^ word) * 1099511628211ull;
    return hash;
}

std::shared_ptr<const NggSubgroupStages>
compile_ngg_subgroup_stages(const NggSubgroupDrawRequest& request, uint32_t waves,
                            std::string* refusal) {
    const auto refuse = [&](std::string reason) -> std::shared_ptr<const NggSubgroupStages> {
        if (refusal) *refusal = std::move(reason);
        return nullptr;
    };
    auto stages = std::make_shared<NggSubgroupStages>();
    stages->waves = waves;
    NggSubgroupShellConfig shell = request.shell;
    shell.waves = waves;
    std::string why;
    auto module = recompile_ngg_subgroup(request.linked_code, request.dwords, request.resources,
                                         shell, &stages->layout, request.diagnostic, &why);
    if (module.empty()) return refuse(why);
    std::vector<uint32_t> declared;
    if (!ngg_shell_guest_bindings(module, &declared))
        return refuse("reason=ngg-draw-guest-resource-unsupported");
    // Only the bindings the shell ACCESSES: the same reflection the frontend materializes set 0
    // from, which leaves a declared-but-unread binding (a direct V# every load bypasses) unbuilt.
    const DescriptorValidationReport used =
        validate_spirv_descriptor_interface(module, nullptr, 0, SpirvShaderStage::Compute, false);
    for (uint32_t binding : declared)
        if (std::any_of(used.descriptors.begin(), used.descriptors.end(),
                        [&](const auto& d) { return d.set == 0 && d.binding == binding; }))
            stages->guest_bindings.push_back(binding);
    stages->shell_hash = ngg_words_hash(module);
    stages->shell = std::make_shared<const std::vector<uint32_t>>(std::move(module));
    NggRasterCommitConfig raster = request.raster;
    raster.layout = stages->layout;
    raster.waves = waves;
    NggRasterCommitInterface published;
    auto vertex = build_ngg_raster_commit_vertex(raster, &published, &why);
    if (vertex.empty()) return refuse(why);
    stages->raster_vertex = std::make_shared<const std::vector<uint32_t>>(std::move(vertex));
    stages->vertices_per_primitive = published.vertices_per_primitive;
    if (raster.route == NggLayerRoute::ForwardingGeometry)
        stages->raster_geometry =
            build_ngg_layer_forward_geometry(published, raster.float_transport);
    // The interpolation stage is built whenever the request supplies it: for its own layer route,
    // and for a draw that needs it although no layer is read (route None).
    if ((raster.route == NggLayerRoute::InterpolationGeometry ||
         raster.route == NggLayerRoute::None) &&
        request.interpolation_geometry)
        stages->raster_geometry = request.interpolation_geometry(published);
    if ((raster.route == NggLayerRoute::ForwardingGeometry ||
         raster.route == NggLayerRoute::InterpolationGeometry ||
         (raster.route == NggLayerRoute::None && request.interpolation_geometry)) &&
        stages->raster_geometry.empty())
        return refuse("reason=ngg-draw-geometry-unavailable");
    return stages;
}

std::shared_ptr<const NggSubgroupDraw>
assemble_ngg_subgroup_draw(const NggSubgroupDrawRequest& request,
                           const NggSubgroupStagesSource& stages_for, std::string* refusal) {
    // The launch record is v0..v8 then s3 (ngg_subgroup_shell.hpp).
    static_assert(kNggLaunchWordsPerLane == std::size(NggLaneLaunch{}.v) + 1u &&
                      kNggLaunchS3Word == std::size(NggLaneLaunch{}.v),
                  "a launch record is the lane's VGPRs v0..v8 followed by s3");
    auto draw = std::make_shared<NggSubgroupDraw>();
    draw->plan = plan_ngg_subgroups(request.shape, request.limits, request.budget);
    if (!draw->plan.ok()) return fail(refusal, "ngg-draw-plan " + draw->plan.refusal);
    if (draw->plan.subgroups.empty()) return fail(refusal, "ngg-draw-empty");
    const uint32_t push_words =
        request.shell.user_sgprs + (request.shell.user_data_address_known ? 2u : 0u);
    if (request.push_constants.size() != push_words)
        return fail(refusal, "ngg-draw-push-constants");
    draw->push_constants = request.push_constants;
    draw->topology = request.raster.topology;
    draw->route = request.raster.route;
    draw->count_violations = request.raster.count_violations;
    draw->native_wave64 = request.shell.native_wave64;
    draw->lds_bytes = request.shell.rsrc2_gs_lds_size * kNggLdsGranuleDwords * 4u;
    // The interpolation stage's input is Triangles (select_ngg_layer_route refuses the same case).
    const bool interpolation =
        request.raster.route == NggLayerRoute::InterpolationGeometry ||
        (request.raster.route == NggLayerRoute::None && request.interpolation_geometry);
    if (interpolation && request.raster.topology != NggOutputTopology::TriangleList)
        return fail(refusal, "ngg-interpolation-geometry-needs-triangles");

    // One group per distinct wave count, ascending.
    std::map<uint32_t, uint32_t> group_of;   // W -> group index
    for (const NggSubgroup& subgroup : draw->plan.subgroups) group_of[subgroup.waves] = 0;
    for (auto& [waves, index] : group_of) {
        index = static_cast<uint32_t>(draw->groups.size());
        NggSubgroupWaveGroup group;
        group.waves = waves;
        std::string why;
        group.stages = stages_for(waves, &why);
        if (!group.stages || group.stages->waves != waves) {
            if (refusal) *refusal = why.empty() ? "reason=ngg-draw-stages-unavailable" : why;
            return nullptr;
        }
        if (draw->groups.empty()) {
            draw->layout = group.stages->layout;
            draw->guest_bindings = group.stages->guest_bindings;
            draw->vertices_per_primitive = group.stages->vertices_per_primitive;
        } else {
            if (!same_layout(group.stages->layout, draw->layout))
                return fail(refusal, "ngg-draw-layout-varies");
            if (group.stages->guest_bindings != draw->guest_bindings)
                return fail(refusal, "ngg-draw-guest-resource-unsupported");
        }
        draw->groups.push_back(std::move(group));
    }

    // Launch records per group in plan order, and the runs that draw them in plan order.
    for (const NggSubgroup& subgroup : draw->plan.subgroups) {
        const uint32_t index = group_of[subgroup.waves];
        NggSubgroupWaveGroup& group = draw->groups[index];
        for (uint32_t wave = 0; wave < group.waves; ++wave) {
            const uint32_t s3 = ngg_merged_wave_info(subgroup, wave);
            for (uint32_t lane = 0; lane < kWaveLanes; ++lane) {
                const NggLaneLaunch launch = ngg_lane_launch(subgroup, request.limits, wave, lane);
                group.launch_words.insert(group.launch_words.end(), std::begin(launch.v),
                                          std::end(launch.v));
                group.launch_words.push_back(s3);
            }
        }
        if (!draw->runs.empty() && draw->runs.back().group == index)
            ++draw->runs.back().blocks;
        else
            draw->runs.push_back({index, group.blocks, 1u});
        ++group.blocks;
    }
    for (NggSubgroupWaveGroup& group : draw->groups)
        group.export_words = group.blocks * draw->layout.block_words(group.waves);
    return draw;
}

std::shared_ptr<const NggSubgroupDraw>
build_ngg_subgroup_draw(const NggSubgroupDrawRequest& request, std::string* refusal) {
    return assemble_ngg_subgroup_draw(
        request,
        [&](uint32_t waves, std::string* why) {
            return compile_ngg_subgroup_stages(request, waves, why);
        },
        refusal);
}

}   // namespace prosper::gpu

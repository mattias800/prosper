// ngg_subgroup_codec.hpp -- v72: the merged-NGG draw description a realized draw carries (#3135 P5),
// so a frame grab of a title whose world depends on one replays it.
//
// Written: everything the backend reads (ngg_subgroup_gpu.h) -- per wave-count group the compiled
// shell, pass-through vertex and geometry stages, launch records and export size; the runs; the
// record layout; topology, route, K, violation counting, native Wave64, LDS size; the shell's
// guest bindings and the push-constant words. Not written: the planner's subgroups (the launch
// records already encode them, and nothing downstream of the builder reads the plan). The shell
// hash is recomputed on read rather than trusted.
//
// The stored modules are the capture-day compile, exactly like every other realized module in a
// capsule: a replay runs the description the live renderer ran, and the backend re-asks the
// device half of admission (ngg_device_refusal) on the replay device before recording it.
#pragma once

#include "gpu/capture/serialize/capture_codecs.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {

inline constexpr uint32_t kNggCodecMaxGroups = 4;
inline constexpr uint32_t kNggCodecMaxRuns = 1u << 20;
inline constexpr uint32_t kNggCodecMaxSmall = 64;   // bindings, push words, PARAM targets

inline void write_ngg_layout(Writer& w, const NggExportRecordLayout& layout) {
    w.u32(layout.words_per_lane);
    w.u32(layout.pos1_word);
    w.u32(layout.first_param_word);
    w.words(layout.param_targets);
    w.u32(layout.prim_channels);
    w.u32(layout.pos0_channels);
    w.u32(layout.pos1_channels);
    w.words(layout.param_channels);
}

inline bool read_ngg_layout(Reader& r, NggExportRecordLayout& layout) {
    return r.u32(layout.words_per_lane) && r.u32(layout.pos1_word) &&
           r.u32(layout.first_param_word) &&
           r.words_bounded(layout.param_targets, kNggCodecMaxSmall, "invalid NGG layout") &&
           r.u32(layout.prim_channels) && r.u32(layout.pos0_channels) &&
           r.u32(layout.pos1_channels) &&
           r.words_bounded(layout.param_channels, kNggCodecMaxSmall, "invalid NGG layout") &&
           layout.param_channels.size() == layout.param_targets.size() && layout.words_per_lane;
}

// One presence byte, then the description.
inline bool write_ngg_subgroup_draw(Writer& w, const std::shared_ptr<const NggSubgroupDraw>& draw) {
    if (!draw) {
        w.u8(0u);
        return true;
    }
    if (draw->groups.empty() || draw->groups.size() > kNggCodecMaxGroups ||
        draw->runs.size() > kNggCodecMaxRuns)
        return false;
    w.u8(1u);
    write_ngg_layout(w, draw->layout);
    w.u8(static_cast<uint8_t>(draw->topology));
    w.u8(static_cast<uint8_t>(draw->route));
    w.u32(draw->vertices_per_primitive);
    w.u8(
        static_cast<uint8_t>((draw->count_violations ? 1u : 0u) | (draw->native_wave64 ? 2u : 0u)));
    w.u32(draw->lds_bytes);
    w.words(draw->guest_bindings);
    w.words(draw->push_constants);
    w.u32(static_cast<uint32_t>(draw->groups.size()));
    for (const NggSubgroupWaveGroup& group : draw->groups) {
        if (!group.stages || !group.stages->shell || !group.stages->raster_vertex) return false;
        w.u32(group.waves);
        w.u32(group.blocks);
        w.u32(group.export_words);
        w.words(group.launch_words);
        w.words(*group.stages->shell);
        w.words(*group.stages->raster_vertex);
        w.words(group.stages->raster_geometry);
    }
    w.u32(static_cast<uint32_t>(draw->runs.size()));
    for (const NggSubgroupRun& run : draw->runs) {
        w.u32(run.group);
        w.u32(run.first_block);
        w.u32(run.blocks);
    }
    return true;
}

inline bool read_ngg_subgroup_draw(Reader& r, std::shared_ptr<const NggSubgroupDraw>& out) {
    out.reset();
    uint8_t present = 0;
    if (!r.u8(present) || present > 1u) return false;
    if (!present) return true;
    auto draw = std::make_shared<NggSubgroupDraw>();
    uint8_t topology = 0, route = 0, flags = 0;
    uint32_t groups = 0, runs = 0;
    if (!read_ngg_layout(r, draw->layout) || !r.u8(topology) || !r.u8(route) ||
        !r.u32(draw->vertices_per_primitive) || !r.u8(flags) || !r.u32(draw->lds_bytes) ||
        !r.words_bounded(draw->guest_bindings, kNggCodecMaxSmall, "invalid NGG bindings") ||
        !r.words_bounded(draw->push_constants, kNggCodecMaxSmall, "invalid NGG push words") ||
        !r.u32(groups) || !groups || groups > kNggCodecMaxGroups)
        return false;
    if (topology != static_cast<uint8_t>(NggOutputTopology::LineList) &&
        topology != static_cast<uint8_t>(NggOutputTopology::TriangleList))
        return false;
    if (route > static_cast<uint8_t>(NggLayerRoute::ForwardingGeometry) || flags > 3u ||
        (draw->vertices_per_primitive != 2u && draw->vertices_per_primitive != 3u))
        return false;
    draw->topology = static_cast<NggOutputTopology>(topology);
    draw->route = static_cast<NggLayerRoute>(route);
    draw->count_violations = flags & 1u;
    draw->native_wave64 = flags & 2u;
    for (uint32_t g = 0; g < groups; ++g) {
        NggSubgroupWaveGroup group;
        auto stages = std::make_shared<NggSubgroupStages>();
        std::vector<uint32_t> shell, vertex;
        if (!r.u32(group.waves) || !group.waves || group.waves > 4u || !r.u32(group.blocks) ||
            !group.blocks || !r.u32(group.export_words) || !r.words(group.launch_words) ||
            !r.words(shell) || shell.empty() || !r.words(vertex) || vertex.empty() ||
            !r.words(stages->raster_geometry))
            return false;
        if (group.launch_words.size() !=
                uint64_t{group.blocks} * 64u * group.waves * kNggLaunchWordsPerLane ||
            group.export_words != uint64_t{group.blocks} * draw->layout.block_words(group.waves))
            return false;
        stages->waves = group.waves;
        stages->shell_hash = ngg_words_hash(shell);
        stages->shell = std::make_shared<const std::vector<uint32_t>>(std::move(shell));
        stages->raster_vertex = std::make_shared<const std::vector<uint32_t>>(std::move(vertex));
        stages->layout = draw->layout;
        stages->guest_bindings = draw->guest_bindings;
        stages->vertices_per_primitive = draw->vertices_per_primitive;
        group.stages = std::move(stages);
        draw->groups.push_back(std::move(group));
    }
    if (!r.u32(runs) || !runs || runs > kNggCodecMaxRuns || uint64_t{runs} * 12u > r.left)
        return false;
    for (uint32_t i = 0; i < runs; ++i) {
        NggSubgroupRun run;
        if (!r.u32(run.group) || !r.u32(run.first_block) || !r.u32(run.blocks) ||
            run.group >= draw->groups.size() || !run.blocks ||
            uint64_t{run.first_block} + run.blocks > draw->groups[run.group].blocks)
            return false;
        draw->runs.push_back(run);
    }
    out = std::move(draw);
    return true;
}

}   // namespace prosper::gpu

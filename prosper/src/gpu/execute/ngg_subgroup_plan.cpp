// ngg_subgroup_plan.cpp -- see ngg_subgroup_plan.hpp.
#include "gpu/execute/ngg_subgroup_plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace prosper::gpu {

NggSubgroupLimits decode_ngg_subgroup_limits(uint32_t vgt_gs_onchip_cntl, uint32_t ge_cntl,
                                             uint32_t ge_max_output_per_subgroup,
                                             uint32_t vgt_gs_max_vert_out,
                                             uint32_t vgt_esgs_ring_itemsize) {
    NggSubgroupLimits limits;
    limits.es_verts_per_subgroup = vgt_gs_onchip_cntl & 0x7ffu;
    limits.gs_prims_per_subgroup = (vgt_gs_onchip_cntl >> 11) & 0x7ffu;
    limits.prim_group_size = ge_cntl & 0x1ffu;
    limits.vert_group_size = (ge_cntl >> 9) & 0x1ffu;
    limits.max_out_verts_per_subgroup = ge_max_output_per_subgroup & 0x7ffu;
    limits.gs_max_vert_out = vgt_gs_max_vert_out & 0x7ffu;
    limits.esgs_item_size = vgt_esgs_ring_itemsize & 0x7fffu;
    return limits;
}

namespace {

constexpr uint32_t kWaveLanes = 64;

uint32_t prims_per_instance(const NggDrawShape& draw) {
    if (draw.topology == NggInputTopology::TriangleList) return draw.vertex_count / 3u;
    return draw.vertex_count >= 3u ? draw.vertex_count - 2u : 0u;
}

// The three vertex indices of instance-local primitive `p`, in input order. A strip's odd
// triangles are kept in natural order (p, p+1, p+2); whether the hardware swaps them for the GS is
// open question 3 on #3135, so admission accepts strips only where the order cannot matter.
std::array<uint32_t, 3> prim_vertices(const NggDrawShape& draw, uint32_t p) {
    if (draw.topology == NggInputTopology::TriangleList) return {3u * p, 3u * p + 1u, 3u * p + 2u};
    return {p, p + 1u, p + 2u};
}

uint32_t threads_of(const NggSubgroup& s, const NggSubgroupLimits& limits) {
    return std::max({s.es_threads(), s.gs_threads(), s.gs_threads() * limits.gs_max_vert_out});
}

}  // namespace

NggSubgroupPlan plan_ngg_subgroups(const NggDrawShape& draw, const NggSubgroupLimits& limits,
                                   const NggSubgroupBudget& budget) {
    NggSubgroupPlan plan;
    const uint32_t prim_limit = std::min(
        {limits.gs_prims_per_subgroup, limits.prim_group_size,
         limits.gs_max_vert_out ? limits.max_out_verts_per_subgroup / limits.gs_max_vert_out : 0u});
    const uint32_t es_limit = std::min(limits.es_verts_per_subgroup, limits.vert_group_size);
    // A primitive must fit, and a vertex offset (lane x ITEMSIZE) must fit its 16-bit field.
    if (prim_limit == 0 || es_limit < 3u || limits.esgs_item_size == 0 ||
        (es_limit - 1u) * limits.esgs_item_size > 0xffffu) {
        plan.refusal = "ngg-limits-unusable";
        return plan;
    }
    const uint32_t prims = prims_per_instance(draw);
    if (prims == 0 || draw.instance_count == 0) return plan;   // nothing to draw

    for (uint32_t instance = 0; instance < draw.instance_count; ++instance) {
        NggSubgroup current;
        current.instance = instance;
        const auto close = [&]() -> bool {
            if (current.prim_slot.empty()) return true;
            const uint32_t threads = threads_of(current, limits);
            current.waves = (threads + kWaveLanes - 1u) / kWaveLanes;
            if (current.waves > budget.max_waves_per_subgroup) {
                plan.refusal = "ngg-subgroup-too-wide";
                return false;
            }
            if (plan.subgroups.size() >= budget.max_subgroups) {
                plan.refusal = "ngg-subgroup-plan-budget";
                return false;
            }
            plan.subgroups.push_back(std::move(current));
            return true;
        };
        for (uint32_t p = 0; p < prims; ++p) {
            const auto verts = prim_vertices(draw, p);
            uint32_t added = 0;
            for (uint32_t k = 0; k < 3u; ++k) {
                const bool seen = std::find(current.es_vertex.begin(), current.es_vertex.end(),
                                            verts[k]) != current.es_vertex.end();
                // A vertex repeated inside one primitive counts once.
                const bool repeated =
                    (k >= 1u && verts[k] == verts[0]) || (k == 2u && verts[2] == verts[1]);
                if (!seen && !repeated) ++added;
            }
            if (current.gs_threads() + 1u > prim_limit || current.es_threads() + added > es_limit) {
                if (!close()) {
                    plan.subgroups.clear();
                    return plan;
                }
                current = NggSubgroup{};
                current.instance = instance;
            }
            if (current.prim_slot.empty()) current.first_prim = p;
            for (uint32_t v : verts) {
                auto it = std::find(current.es_vertex.begin(), current.es_vertex.end(), v);
                if (it == current.es_vertex.end()) {
                    current.es_vertex.push_back(v);
                    it = current.es_vertex.end() - 1;
                }
                current.prim_slot.push_back(static_cast<uint32_t>(it - current.es_vertex.begin()));
            }
        }
        if (!close()) {
            plan.subgroups.clear();
            return plan;
        }
    }
    return plan;
}

uint32_t ngg_merged_wave_info(const NggSubgroup& subgroup, uint32_t wave) {
    const auto in_wave = [&](uint32_t threads) {
        const uint32_t base = wave * kWaveLanes;
        return threads > base ? std::min(threads - base, kWaveLanes) : 0u;
    };
    return (subgroup.waves & 0xfu) << 28 | (wave & 0xfu) << 24 |
           (in_wave(subgroup.gs_threads()) & 0xffu) << 8 | (in_wave(subgroup.es_threads()) & 0xffu);
}

NggLaneLaunch ngg_lane_launch(const NggSubgroup& subgroup, const NggSubgroupLimits& limits,
                              uint32_t first_vertex, uint32_t wave, uint32_t lane) {
    NggLaneLaunch launch;
    const uint32_t t = wave * kWaveLanes + lane;
    if (t < subgroup.gs_threads()) {
        const uint32_t item = limits.esgs_item_size;
        const uint32_t* slot = subgroup.prim_slot.data() + static_cast<size_t>(t) * 3u;
        launch.v[0] = slot[0] * item | (slot[1] * item) << 16;
        launch.v[1] = slot[2] * item;
        launch.v[2] = subgroup.first_prim + t;
    }
    if (t < subgroup.es_threads()) {
        launch.v[5] = first_vertex + subgroup.es_vertex[t];
        launch.v[8] = subgroup.instance;
    }
    return launch;
}

}  // namespace prosper::gpu

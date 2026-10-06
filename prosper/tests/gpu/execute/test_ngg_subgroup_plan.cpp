// The merged-NGG subgroup planner (ngg_subgroup_plan.hpp, #3135 phase P1): partition, launch SGPR
// s3 and launch VGPRs v0..v8. The Kena arm uses that title's recorded registers
// (KENA_STATUS.md: VGT_GS_ONCHIP_CNTL 0x10020040, GE_CNTL 0x8040, 192 / 3 / ITEMSIZE 4) and the
// LUT producer's draw (a 4-vertex strip, 32 instances).
#include "gpu/execute/ngg_subgroup_plan.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

NggSubgroupLimits kena_limits() {
    return decode_ngg_subgroup_limits(0x10020040u, 0x8040u, 192u, 3u, 4u);
}

TEST(NggSubgroupPlan, DecodesTheRecordedRegisters) {
    const NggSubgroupLimits l = kena_limits();
    EXPECT_EQ(l.es_verts_per_subgroup, 64u);
    EXPECT_EQ(l.gs_prims_per_subgroup, 64u);
    EXPECT_EQ(l.prim_group_size, 64u);
    EXPECT_EQ(l.vert_group_size, 64u);
    EXPECT_EQ(l.max_out_verts_per_subgroup, 192u);
    EXPECT_EQ(l.gs_max_vert_out, 3u);
    EXPECT_EQ(l.esgs_item_size, 4u);
}

TEST(NggSubgroupPlan, KenaLutStripIsOneSubgroupPerInstance) {
    NggDrawShape draw;
    draw.topology = NggInputTopology::TriangleStrip;
    draw.vertex_count = 4;
    draw.instance_count = 32;
    const NggSubgroupPlan plan = plan_ngg_subgroups(draw, kena_limits());
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.subgroups.size(), 32u) << "instances are never packed";
    for (uint32_t i = 0; i < 32u; ++i) {
        const NggSubgroup& s = plan.subgroups[i];
        EXPECT_EQ(s.instance, i);
        EXPECT_EQ(s.first_prim, 0u);
        EXPECT_EQ(s.es_vertex, (std::vector<uint32_t>{0, 1, 2, 3}));
        EXPECT_EQ(s.prim_slot, (std::vector<uint32_t>{0, 1, 2, 1, 2, 3}));
        EXPECT_EQ(s.waves, 1u) << "max(4 ES, 2 GS, 2 x 3 output) = 6 threads";
    }
    const NggSubgroup& s = plan.subgroups[5];
    EXPECT_EQ(ngg_merged_wave_info(s, 0), (1u << 28) | (2u << 8) | 4u);
    const NggLaneLaunch lane0 = ngg_lane_launch(s, kena_limits(), 0, 0, 0);
    EXPECT_EQ(lane0.v[0], 0u | (4u << 16)) << "slots 0,1 scaled by ITEMSIZE";
    EXPECT_EQ(lane0.v[1], 8u) << "slot 2 x 4";
    EXPECT_EQ(lane0.v[2], 0u);
    EXPECT_EQ(lane0.v[5], 0u);
    EXPECT_EQ(lane0.v[8], 5u) << "InstanceID";
    const NggLaneLaunch lane1 = ngg_lane_launch(s, kena_limits(), 0, 0, 1);
    EXPECT_EQ(lane1.v[0], 4u | (8u << 16));
    EXPECT_EQ(lane1.v[1], 12u);
    EXPECT_EQ(lane1.v[2], 1u) << "PrimitiveID";
    const NggLaneLaunch lane3 = ngg_lane_launch(s, kena_limits(), 100, 0, 3);
    EXPECT_EQ(lane3.v[0], 0u) << "lane 3 is an ES thread only";
    EXPECT_EQ(lane3.v[5], 103u) << "VertexID includes first_vertex";
    const NggLaneLaunch lane9 = ngg_lane_launch(s, kena_limits(), 0, 0, 9);
    for (uint32_t r = 0; r < 9u; ++r) EXPECT_EQ(lane9.v[r], 0u) << "v" << r << " of an idle lane";
}

TEST(NggSubgroupPlan, EveryInputPrimitiveAppearsOnceInOrderWithinLimits) {
    NggDrawShape draw;
    draw.vertex_count = 3000;   // a 1,000-triangle list
    draw.instance_count = 2;
    const NggSubgroupLimits limits = kena_limits();
    const NggSubgroupPlan plan = plan_ngg_subgroups(draw, limits);
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    for (uint32_t instance = 0; instance < 2u; ++instance) {
        uint32_t next_prim = 0;
        for (const NggSubgroup& s : plan.subgroups) {
            if (s.instance != instance) continue;
            EXPECT_EQ(s.first_prim, next_prim) << "subgroups continue the primitive sequence";
            ASSERT_LE(s.gs_threads(), 64u);
            ASSERT_LE(s.es_threads(), 64u);
            ASSERT_LE(s.gs_threads() * limits.gs_max_vert_out, limits.max_out_verts_per_subgroup);
            for (uint32_t p = 0; p < s.gs_threads(); ++p)
                for (uint32_t k = 0; k < 3u; ++k)
                    EXPECT_EQ(s.es_vertex[s.prim_slot[3u * p + k]], 3u * (next_prim + p) + k)
                        << "slot -> vertex is the input primitive's own vertex";
            next_prim += s.gs_threads();
        }
        EXPECT_EQ(next_prim, 1000u) << "every primitive exactly once";
    }
}

TEST(NggSubgroupPlan, SubgroupsCloseAtEachLimit) {
    NggDrawShape list;
    list.vertex_count = 3u * 100u;
    // GS_PRIMS_PER_SUBGRP = 10 (and ES large): 100 prims -> 10 subgroups of 10.
    NggSubgroupLimits by_prims = kena_limits();
    by_prims.gs_prims_per_subgroup = 10;
    by_prims.es_verts_per_subgroup = by_prims.vert_group_size = 64;
    const auto a = plan_ngg_subgroups(list, by_prims);
    ASSERT_TRUE(a.ok());
    EXPECT_EQ(a.subgroups.size(), 10u);
    // ES_VERTS_PER_SUBGRP = 9: a list primitive brings 3 new vertices -> 3 prims per subgroup.
    NggSubgroupLimits by_verts = kena_limits();
    by_verts.es_verts_per_subgroup = 9;
    const auto b = plan_ngg_subgroups(list, by_verts);
    ASSERT_TRUE(b.ok());
    EXPECT_EQ(b.subgroups.size(), 34u);   // ceil(100 / 3)
    EXPECT_EQ(b.subgroups[0].es_threads(), 9u);
    // Output vertices: GS_MAX_VERT_OUT 3 and 30 output slots -> 10 prims per subgroup.
    NggSubgroupLimits by_output = kena_limits();
    by_output.max_out_verts_per_subgroup = 30;
    const auto c = plan_ngg_subgroups(list, by_output);
    ASSERT_TRUE(c.ok());
    EXPECT_EQ(c.subgroups.size(), 10u);
    // A strip shares vertices: 8 ES vertices hold 6 strip triangles.
    NggDrawShape strip;
    strip.topology = NggInputTopology::TriangleStrip;
    strip.vertex_count = 20;
    NggSubgroupLimits strip_verts = kena_limits();
    strip_verts.es_verts_per_subgroup = 8;
    const auto d = plan_ngg_subgroups(strip, strip_verts);
    ASSERT_TRUE(d.ok());
    EXPECT_EQ(d.subgroups[0].gs_threads(), 6u);
    EXPECT_EQ(d.subgroups[0].es_threads(), 8u);
}

TEST(NggSubgroupPlan, EveryOutputVertexHasALane) {
    NggDrawShape list;
    list.vertex_count = 3u * 64u;
    const auto plan = plan_ngg_subgroups(list, kena_limits());
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan.subgroups.size(), 4u) << "64 ES vertices hold 21 list triangles: 21+21+21+1";
    const NggSubgroup& s = plan.subgroups[0];
    EXPECT_EQ(s.gs_threads(), 21u);
    EXPECT_EQ(s.waves, 1u) << "max(63, 21, 63) threads";
    // A strip packs 62 triangles into 64 vertices: 62 x 3 = 186 output slots -> 3 waves.
    NggDrawShape strip;
    strip.topology = NggInputTopology::TriangleStrip;
    strip.vertex_count = 64;
    const auto wide = plan_ngg_subgroups(strip, kena_limits());
    ASSERT_TRUE(wide.ok());
    ASSERT_EQ(wide.subgroups.size(), 1u);
    EXPECT_EQ(wide.subgroups[0].waves, 3u);
    EXPECT_EQ(ngg_merged_wave_info(wide.subgroups[0], 0), (3u << 28) | (62u << 8) | 64u);
    EXPECT_EQ(ngg_merged_wave_info(wide.subgroups[0], 1), (3u << 28) | (1u << 24));
    EXPECT_EQ(ngg_merged_wave_info(wide.subgroups[0], 2), (3u << 28) | (2u << 24));
}

TEST(NggSubgroupPlan, RefusesWhatItCannotModel) {
    NggDrawShape list;
    list.vertex_count = 3u * 64u;
    NggSubgroupLimits tiny = kena_limits();
    tiny.es_verts_per_subgroup = 2;
    EXPECT_EQ(plan_ngg_subgroups(list, tiny).refusal, "ngg-limits-unusable");
    NggSubgroupLimits no_item = kena_limits();
    no_item.esgs_item_size = 0;
    EXPECT_EQ(plan_ngg_subgroups(list, no_item).refusal, "ngg-limits-unusable");
    NggSubgroupLimits no_out = kena_limits();
    no_out.max_out_verts_per_subgroup = 2;   // below GS_MAX_VERT_OUT: no primitive fits
    EXPECT_EQ(plan_ngg_subgroups(list, no_out).refusal, "ngg-limits-unusable");
    NggDrawShape strip;
    strip.topology = NggInputTopology::TriangleStrip;
    strip.vertex_count = 64;
    NggSubgroupBudget narrow;
    narrow.max_waves_per_subgroup = 2;
    const auto too_wide = plan_ngg_subgroups(strip, kena_limits(), narrow);
    EXPECT_EQ(too_wide.refusal, "ngg-subgroup-too-wide");
    EXPECT_TRUE(too_wide.subgroups.empty()) << "a refused plan carries no partial subgroups";
    NggSubgroupBudget few;
    few.max_subgroups = 2;
    EXPECT_EQ(plan_ngg_subgroups(list, kena_limits(), few).refusal, "ngg-subgroup-plan-budget");
}

TEST(NggSubgroupPlan, NothingToDrawIsAnEmptyPlan) {
    NggDrawShape strip;
    strip.topology = NggInputTopology::TriangleStrip;
    strip.vertex_count = 2;
    const auto plan = plan_ngg_subgroups(strip, kena_limits());
    EXPECT_TRUE(plan.ok());
    EXPECT_TRUE(plan.subgroups.empty());
    NggDrawShape none;
    none.vertex_count = 3;
    none.instance_count = 0;
    EXPECT_TRUE(plan_ngg_subgroups(none, kena_limits()).subgroups.empty());
}

}  // namespace

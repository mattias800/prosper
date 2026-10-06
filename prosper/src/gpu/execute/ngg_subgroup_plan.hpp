// ngg_subgroup_plan.hpp -- partition a merged ES+GS NGG draw into guest subgroups, and the exact
// launch values each subgroup's waves receive (#3135, phase P1 of the merged-NGG lowering).
//
// Pure CPU, no Vulkan state, and nothing live calls it yet: it is the input to the production
// compute shell (P2), which runs ONE Vulkan workgroup per guest subgroup.
//
// The launch contract (SGPR s3, VGPRs v0..v8) is the one derived from the RDNA2 ISA and two
// independent compilers, checked against Kena's guest code, and recorded on #3135. What the public
// sources do not settle -- how the hardware partitions primitives into subgroups -- is encoded here
// as an explicit modelling POLICY, not as a claim about hardware:
//
//   * input primitives are packed in API order and never reordered;
//   * a subgroup closes when the next primitive would exceed ES_VERTS_PER_SUBGRP unique vertices,
//     GS_PRIMS_PER_SUBGRP (and GE_CNTL's group sizes), or prims x GS_MAX_VERT_OUT output vertices;
//   * ES vertices are deduplicated within a subgroup only;
//   * instances are never packed together (one instance per subgroup);
//   * a subgroup has max(es, gs, gs x GS_MAX_VERT_OUT) threads, so every output vertex a GS lane may
//     export has a lane (open question 2 on #3135).
//
// Why a policy can be right without knowing the hardware's choice (hypothesis, CONFIDENCE: MED):
// NGG code reads its own counts from s3 rather than assuming full subgroups, so it is correct for
// any legal partition, and output order is input order whatever the partition. The admission step
// (P5) refuses the known ways a program can depend on the partition (side effects, GS instancing,
// order-sensitive strip states). Anything this planner cannot model returns a refusal, never a guess.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

// The partition limits, decoded from the draw's registers.
struct NggSubgroupLimits {
    uint32_t es_verts_per_subgroup = 0;   // VGT_GS_ONCHIP_CNTL.ES_VERTS_PER_SUBGRP [10:0]
    uint32_t gs_prims_per_subgroup = 0;   // VGT_GS_ONCHIP_CNTL.GS_PRIMS_PER_SUBGRP [21:11]
    uint32_t prim_group_size = 0;   // GE_CNTL.PRIM_GRP_SIZE [8:0]
    uint32_t vert_group_size = 0;   // GE_CNTL.VERT_GRP_SIZE [17:9]
    uint32_t max_out_verts_per_subgroup = 0;   // GE_MAX_OUTPUT_PER_SUBGROUP [10:0]
    uint32_t gs_max_vert_out = 0;   // VGT_GS_MAX_VERT_OUT [10:0]
    uint32_t esgs_item_size = 0;   // VGT_ESGS_RING_ITEMSIZE [14:0], in dwords
};

NggSubgroupLimits decode_ngg_subgroup_limits(uint32_t vgt_gs_onchip_cntl, uint32_t ge_cntl,
                                             uint32_t ge_max_output_per_subgroup,
                                             uint32_t vgt_gs_max_vert_out,
                                             uint32_t vgt_esgs_ring_itemsize);

// Input topologies the planner models. Anything else is refused.
enum class NggInputTopology : uint8_t { TriangleList, TriangleStrip };

// A non-indexed draw (indexed draws are a later phase).
struct NggDrawShape {
    NggInputTopology topology = NggInputTopology::TriangleList;
    uint32_t vertex_count = 0;
    uint32_t instance_count = 1;
    uint32_t first_vertex = 0;   // added to every VertexID (v5)
};

struct NggSubgroupBudget {
    uint32_t max_subgroups = 1u << 16;
    uint32_t max_waves_per_subgroup = 4;   // 256 threads: the compute shell's workgroup bound
};

// One guest subgroup: a run of consecutive input primitives of one instance.
struct NggSubgroup {
    uint32_t instance = 0;
    uint32_t first_prim = 0;   // the instance-local index of the first primitive
    // Unique vertex indices (VertexID before first_vertex), in first-use order: ES lane e runs
    // es_vertex[e].
    std::vector<uint32_t> es_vertex;
    // Per primitive, the ES lanes of its three vertices in input order.
    std::vector<uint32_t> prim_slot;   // 3 per primitive
    uint32_t waves = 0;
    uint32_t es_threads() const { return static_cast<uint32_t>(es_vertex.size()); }
    uint32_t gs_threads() const { return static_cast<uint32_t>(prim_slot.size() / 3u); }
};

struct NggSubgroupPlan {
    std::vector<NggSubgroup> subgroups;
    std::string refusal;   // empty when the plan is complete
    bool ok() const { return refusal.empty(); }
};

NggSubgroupPlan plan_ngg_subgroups(const NggDrawShape& draw, const NggSubgroupLimits& limits,
                                   const NggSubgroupBudget& budget = {});

// The merged wave info SGPR (s3) for wave `wave` of `subgroup`:
//   [31:28] waves in subgroup, [27:24] wave index, [23:16] GS wave id (0; programs reading it are
//   refused at admission), [15:8] GS threads in this wave, [7:0] ES threads in this wave.
uint32_t ngg_merged_wave_info(const NggSubgroup& subgroup, uint32_t wave);

// The launch VGPRs v0..v8 for lane `lane` (0..63) of wave `wave`. Thread t = wave * 64 + lane.
//   GS threads (t < gs): v0 = slot0*ITEMSIZE | (slot1*ITEMSIZE) << 16, v1 = slot2*ITEMSIZE,
//     v2 = PrimitiveID (instance-local), v3 = 0 (GS instance; GS instancing is refused), v4 = 0.
//   ES threads (t < es): v5 = VertexID (first_vertex + index), v8 = InstanceID.
//   v6/v7 (ES user VGPRs) are not established and stay 0; admission refuses programs reading them.
//   Every other value is 0 (the hardware leaves it undefined; a correct program cannot depend on it).
// CONFIDENCE: HIGH on the layout (ISA + Mesa + LLPC agree, Kena's code reads it this way); MED on
// scaling the vertex offsets by ITEMSIZE (Mesa programs 1 and scales in the shader).
struct NggLaneLaunch {
    uint32_t v[9] = {};
};
NggLaneLaunch ngg_lane_launch(const NggSubgroup& subgroup, const NggSubgroupLimits& limits,
                              uint32_t first_vertex, uint32_t wave, uint32_t lane);

}   // namespace prosper::gpu

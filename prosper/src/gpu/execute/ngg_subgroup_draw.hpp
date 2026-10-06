// ngg_subgroup_draw.hpp -- everything the Vulkan backend needs to run one merged ES+GS NGG draw
// (#3135 phase P4): the guest subgroups the planner made (P1), the compiled subgroup shell per wave
// count (P2), the pass-through raster stages that draw its export records (P3), and the launch
// records. Pure CPU and immutable once built, so a DrawItem and a BackendDraw can share it.
//
// HOW THE BACKEND RUNS IT. A draw carrying an NggSubgroupDraw is its own backend segment. In ONE
// command buffer, with no CPU wait, the backend records:
//   1. vkCmdFillBuffer zeroing every export block and the violation counters;
//   2. a barrier from that fill to the compute and vertex stages;
//   3. one dispatch per wave-count group, ascending W, one workgroup per subgroup of that W;
//   4. a barrier from COMPUTE_SHADER to the vertex (and geometry) stage;
//   5. the pass-through draws, one per RUN, in run order.
//
// WAVE-COUNT GROUPS AND RUNS. The shell is compiled for one W (its LocalSize is 64W), and
// workgroup g of a dispatch reads launch record g * 64W + i and writes export block g. So the
// subgroups of one W are dispatched together, in plan order, into their own export region. The
// planner can interleave wave counts (a strip's last subgroup per instance is often smaller), and
// guest primitive order is plan order, so the draw side visits blocks in PLAN order: a run is a
// maximal stretch of consecutive plan subgroups with one W, drawn by that W's pass-through stage
// with firstVertex = first_block * 64W * K over that W's region.
//
// SET-0 RESOURCES. The shell reads its guest resources through set 0, at the bindings listed in
// `guest_bindings` (storage buffers only; a shell declaring an image, sampler or uniform block is
// refused here). The draw carrying this description supplies them as its own set-0 resources.
#pragma once

#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/ngg_raster_commit.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

// The subgroups of one wave count W.
struct NggSubgroupWaveGroup {
    uint32_t waves = 0;   // W, 1..4
    uint32_t blocks = 0;   // subgroups of this W = workgroups dispatched = export blocks
    std::vector<uint32_t> shell;   // compute module, LocalSize 64W
    std::vector<uint32_t> raster_vertex;   // pass-through vertex stage for W
    std::vector<uint32_t> raster_geometry;   // the route's geometry stage, or empty
    // kNggLaunchWordsPerLane words per invocation, blocks * 64W invocations, plan order.
    std::vector<uint32_t> launch_words;
    uint32_t export_words = 0;   // blocks * layout.block_words(waves)
};

// A maximal stretch of consecutive plan subgroups with one wave count.
struct NggSubgroupRun {
    uint32_t group = 0;   // index into NggSubgroupDraw::groups
    uint32_t first_block = 0;   // the run's first block within that group's export region
    uint32_t blocks = 0;
};

struct NggSubgroupDraw {
    NggSubgroupPlan plan;
    NggExportRecordLayout layout;
    NggOutputTopology topology = NggOutputTopology::TriangleList;
    NggLayerRoute route = NggLayerRoute::None;
    uint32_t vertices_per_primitive = 3;   // K
    bool count_violations = true;   // the vertex stage writes set 2 binding 2
    bool native_wave64 = false;   // every shell requires a 64-lane subgroup
    std::vector<NggSubgroupWaveGroup> groups;   // ascending W: the dispatch order
    std::vector<NggSubgroupRun> runs;   // plan order: the draw order
    std::vector<uint32_t> guest_bindings;   // set-0 storage-buffer bindings the shells read
    std::vector<uint32_t> push_constants;   // s8.. (then s0:s1 when the address is known)
};

struct NggSubgroupDrawRequest {
    const uint32_t* linked_code = nullptr;   // the linked ES+GS program
    size_t dwords = 0;
    const ShaderResourceTable* resources = nullptr;
    NggSubgroupShellConfig shell;   // `waves` is ignored: each group sets its own
    NggSubgroupLimits limits;
    NggDrawShape shape;
    NggSubgroupBudget budget;
    // `layout` and `waves` are ignored: the builder fills them per group from the shell.
    NggRasterCommitConfig raster;
    // The interpolation geometry stage the draw's pixel stage needs, built against the vertex
    // stage's published interface (its layer location). Required for the InterpolationGeometry
    // route; also used with route None (no layer read). Ignored by the other routes, which the
    // route selection only picks when no interpolation stage is required.
    std::function<std::vector<uint32_t>(const NggRasterCommitInterface&)> interpolation_geometry;
    std::vector<uint32_t> push_constants;
    RecompileDiagnosticContext diagnostic{RecompileDiagnosticStage::Vertex, 0};
};

// Plans, compiles and lays out the draw. Returns null with `refusal` set ("reason=<name> ...") on
// any refusal of its parts, or: ngg-draw-plan (the planner refused; its reason follows),
// ngg-draw-empty (no subgroups), ngg-draw-layout-varies (the shells' record layouts differ by W),
// ngg-draw-guest-resource-unsupported (a shell declares an image, sampler or non-storage block, or
// a set other than 0 and 2), ngg-draw-geometry-unavailable (the route needs a geometry stage that
// could not be built), ngg-draw-push-constants (the words disagree with the shell configuration),
// ngg-interpolation-geometry-needs-triangles (an interpolation stage for a line list).
std::shared_ptr<const NggSubgroupDraw>
build_ngg_subgroup_draw(const NggSubgroupDrawRequest& request, std::string* refusal = nullptr);

// The set-0 bindings a shell module declares, or false when it declares anything the backend
// cannot bind as a plain storage buffer (see ngg-draw-guest-resource-unsupported).
bool ngg_shell_guest_bindings(const std::vector<uint32_t>& spirv, std::vector<uint32_t>* bindings);

}   // namespace prosper::gpu

// gpu_execute.hpp — the stage-independent core of the GPU executor (Stage A of docs/GPU_EXECUTOR_DESIGN.md).
//
// Turns a folded GpuState (exactly what agc_driver_submit_dcb produces via run_command_buffer) into a
// rendered frame: extract the RDNA2 render-state, recompile the vertex+pixel shaders straight from their
// SHADER_PGM addresses, resolve fixed-function state to Vulkan-ready values, and invoke a caller-supplied
// render backend. It is deliberately **Vulkan-agnostic** — the backend is a std::function — so this lives
// in prosper_core (which does not link Vulkan) while the live-device renderer is supplied by whoever has a
// device (the app/HLE, or tests via render_runner.h). agc_driver_submit_dcb calls this with the live
// renderer once the device is wired; tests call it with the offscreen renderer to verify the spine.
#pragma once
#include "command_processor.hpp"   // GpuState
#include "render_state.hpp"        // extract_render_state / resolve_pipeline_state / ResolvedPipelineState
#include "rdna2_to_spirv.hpp"      // recompile_vertex / recompile_fragment
#include "shader_resources.hpp"    // ShaderResourceTable
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

namespace prosper::gpu {

struct ShaderResourceTable;   // fwd (shader_resources.hpp); passed to the backend so it can bind resources

// One realized draw of a submit: recompiled VS+PS SPIR-V, the draw's OWN resolved fixed-function
// state, the two stages' resource tables (so the backend can bind the constant/vertex buffers +
// textures the shaders declare, reading their bytes from 1:1-mapped guest memory), and its vertex
// count. execute_gpustate() emits one per realized draw; the backend records ALL of them into ONE
// render pass (clear once, then draw each with its own pipeline + descriptors) so a multi-draw submit
// — e.g. Unity's background + composite, whose per-draw masks/blends/shaders differ — composites
// correctly instead of collapsing onto a single draw. The tables may be null (color-only shaders).
struct DrawItem {
    std::vector<uint32_t> vs, fs;                     // recompiled SPIR-V
    ResolvedPipelineState ps;                         // THIS draw's fixed-function state
    std::shared_ptr<ShaderResourceTable> vrt, prt;    // may be null
    uint32_t vertex_count = 3;
};

// The pluggable Vulkan backend: render the submit's draw items into one image and return W*H*4 RGBA8
// pixels (or {} on failure). Empty list -> {} (nothing to draw).
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items)>;

// Safe guest-address readability probe: write() to /dev/null returns EFAULT for an unmapped source
// (Linux; always-true on Windows), so callers can test a guest pointer without risking a SIGSEGV.
// Implemented in gpu_executor.cpp; shared by the executor's const-eval and HLE diagnostic probes.
bool guest_readable(uint64_t addr, uint32_t bytes);

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign bindings matching the recompiler+backend convention (constant buffer -> binding
// 2, vertex buffer -> binding 3, textures -> binding 4+). Returns null if the stage has no shader header
// or no resources. Implemented in gpu_executor.cpp (needs the AGC registry + descriptor decode).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps);

// Realize ONE draw of `ds` (a register snapshot or the folded state) into a DrawItem: recompile the
// VS+PS, resolve fixed-function state, apply the fullscreen-quad fan heuristic + env overrides. Returns
// false (and leaves `out` untouched) if the draw is a no-op (no PGM bound, recompile failed, or
// color_write_mask==0). Shared by the default (folded-state, one item) and PROSPER_PERDRAW (per-draw)
// paths so their per-draw handling is identical.
inline bool realize_draw_item(const GpuState& ds, uint32_t vcount_hint, uint32_t max_shader_dwords,
                              bool log, DrawItem& out) {
    RenderState rs = extract_render_state(ds);
    if (!rs.es_addr || !rs.ps_addr) {
        if (log) fprintf(stderr, "[exec] skip draw: no PGM bound (es=0x%llx ps=0x%llx)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
        return false;
    }
    std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(ds, rs.es_addr, false);
    std::shared_ptr<ShaderResourceTable> prt = build_stage_table(ds, rs.ps_addr, true);
    std::vector<uint32_t> vs = recompile_vertex((const uint32_t*)(uintptr_t)rs.es_addr, max_shader_dwords, vrt.get());
    std::vector<uint32_t> fs = recompile_fragment((const uint32_t*)(uintptr_t)rs.ps_addr, max_shader_dwords, prt.get());
    if (vs.empty() || fs.empty()) {
        if (log) {
            fprintf(stderr, "[exec] skip draw: recompile failed (vs=%zu fs=%zu; es=0x%llx ps=0x%llx)\n",
                    vs.size(), fs.size(), (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
            for (auto [tag, addr] : {std::pair{"vs", rs.es_addr}, std::pair{"ps", rs.ps_addr}}) {
                RecompileCoverage c = recompile_coverage((const uint32_t*)(uintptr_t)addr, max_shader_dwords);
                fprintf(stderr, "[exec]   %s coverage: total=%u alu=%u exp=%u tabledep=%u unsupported=%u "
                                "first_bad fmt=%d op=0x%x\n", tag, c.total, c.alu, c.exports,
                        c.table_dependent, c.unsupported, c.first_bad_fmt, c.first_bad_op);
                if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                    char fn[512]; snprintf(fn, sizeof fn, "%s/exec_%s_%llx.bin", dd, tag, (unsigned long long)addr);
                    if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)addr, 1, 4096, f); fclose(f); }
                }
            }
        }
        return false;
    }
    ResolvedPipelineState ps = resolve_pipeline_state(rs);
    // Skip a draw with no color writes: it contributes no pixels, so recording it (and, in the single-item
    // case, presenting its bare clear) would just overwrite the last real frame (an art/clear flicker).
    if (ps.color_write_mask == 0) {
        if (log) fprintf(stderr, "[exec] skip draw: color_write_mask==0 (no-op)\n");
        return false;
    }
    uint32_t vertex_count = vcount_hint ? vcount_hint : 3u;
    // Fullscreen-composite quad fill: the game tiles a 4-corner quad buffer into two triangles via indexed
    // draws; rendering only 3 vertices paints half the quad. When the bound vertex buffer is exactly a
    // 4-vertex quad, render its corners as a triangle FAN (perimeter order BL,TL,TR,BR -> tris {0,1,2},
    // {0,2,3} tile the quad). Only triggers for a 4-record buffer, so real meshes are unaffected.
    if (vrt) for (const auto& r : vrt->resources)
        if (r.cls == ResourceClass::VertexBuffer && r.stride && (r.size / r.stride) == 4 && vertex_count <= 4) {
            vertex_count = 4; ps.topology = 5 /*VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN*/; break;
        }
    if (const char* vc = getenv("PROSPER_VCOUNT")) { int v = atoi(vc); if (v > 0) vertex_count = (uint32_t)v; }
    if (const char* tp = getenv("PROSPER_TOPO")) { int t = atoi(tp); if (t >= 0) ps.topology = (uint32_t)t; }
    out.vs = std::move(vs); out.fs = std::move(fs); out.ps = ps;
    out.vrt = std::move(vrt); out.prt = std::move(prt); out.vertex_count = vertex_count;
    return true;
}

// Recompile + resolve a GpuState's draws and render them via `render`. Default: ONE item from the folded
// end state (the fullscreen-quad fan fills the current title composite). PROSPER_PERDRAW=1: ONE item per
// draw, each realized from ITS OWN register snapshot (Draw::state), so per-draw masks/blends/shaders
// composite correctly — the path for multi-geometry scenes (opt-in until the AGC context-log section
// semantics that stage duplicate register writes are fully RE'd; see docs/REAL_FRAMES_FINDINGS.md).
// `max_shader_dwords` bounds the recompiler's walk (it stops at S_ENDPGM).
inline std::vector<uint8_t> execute_gpustate(const GpuState& st, const RenderFn& render,
                                             uint32_t max_shader_dwords = 0x10000) {
    if (st.draws.empty() || !render) return {};              // nothing to render (e.g. a state-only submit)
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;    // bail-point visibility (why no frame?)
    static const bool perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    std::vector<DrawItem> items;
    if (perdraw) {
        for (size_t i = 0; i < st.draws.size(); i++) {
            DrawItem it;
            if (realize_draw_item(st.state_at_draw(i), st.draws[i].index_count, max_shader_dwords, log, it))
                items.push_back(std::move(it));
        }
    } else {
        // Default: render the submit's composite from the folded end state as a single item (the fan
        // heuristic fills the fullscreen quad). Preserves the merged single-draw behavior exactly.
        DrawItem it;
        uint32_t vcount = st.draws.empty() ? 3u : st.draws[0].index_count;
        if (realize_draw_item(st, vcount, max_shader_dwords, log, it)) items.push_back(std::move(it));
    }
    if (getenv("PROSPER_DRAWLOG")) { fprintf(stderr, "[exec] draws=%zu perdraw=%d -> %zu item(s):",
        st.draws.size(), (int)perdraw, items.size());
        for (auto& it : items) fprintf(stderr, " (vcount=%u topo=%u mask=0x%x)", it.vertex_count, it.ps.topology, it.ps.color_write_mask);
        fprintf(stderr, "\n"); fflush(stderr); }
    if (items.empty()) return {};
    if (log) fprintf(stderr, "[exec] rendering %zu draw item(s) (of %zu draws)\n", items.size(), st.draws.size());
    return render(items);
}

// --- Live submit renderer registry (Stage A wiring; implemented in gpu_executor.cpp) --------------------
// The live renderer additionally receives the target width/height (from videoout) so it can size its
// attachments. Registered by whoever owns a persistent Vulkan device — the runtime binary at startup, or a
// test — so prosper_core itself stays Vulkan-free (this just stores a std::function). Same DrawItem-list
// shape as RenderFn, plus (w,h).
using LiveRenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items,
                                                        uint32_t width, uint32_t height)>;

// Register (or clear, with {}) the live render backend that agc_driver_submit_dcb uses on each submit.
void set_submit_renderer(LiveRenderFn fn);
bool have_submit_renderer();

// Render a folded GpuState at (width,height) via the registered live renderer and hand the frame to the
// present path (present_write_frame). Returns true iff a frame was produced and presented. A no-op
// returning false when there is no renderer registered or the state has no draws — so it is inert on the
// game path until the runtime wires a device, yet fully exercised by tests. Stage A of GPU_EXECUTOR_DESIGN.
bool execute_and_present(const GpuState& st, uint32_t width, uint32_t height);

} // namespace prosper::gpu

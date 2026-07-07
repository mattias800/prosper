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
// state, the two stages' resource tables (descriptors from the draw's register snapshot), and its
// vertex count. execute_gpustate() emits one per Draw; the backend records ALL of them into ONE
// render pass (clear once, then draw each with its own pipeline) so a multi-draw submit — e.g.
// Unity's background + composite pair, whose per-draw masks/blends/shaders differ — composites
// correctly instead of collapsing onto the last draw's state.
struct DrawItem {
    std::vector<uint32_t> vs, fs;                        // recompiled SPIR-V
    ResolvedPipelineState ps;                            // THIS draw's fixed-function state
    std::shared_ptr<ShaderResourceTable> vrt, prt;       // may be null (color-only shaders)
    uint32_t vertex_count = 3;
};

// The pluggable Vulkan backend: render the submit's draw items into one image and return W*H*4 RGBA8
// pixels (or {} on failure).
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items)>;

// Safe guest-address readability probe: write() to /dev/null returns EFAULT for an unmapped source
// (Linux; always-true on Windows), so callers can test a guest pointer without risking a SIGSEGV.
// Implemented in gpu_executor.cpp; shared by the executor's const-eval and HLE diagnostic probes.
bool guest_readable(uint64_t addr, uint32_t bytes);

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign each resource its own descriptor binding starting at `binding_base`. Returns
// null if the stage has no shader header or no resources. Implemented in gpu_executor.cpp.
// `binding_base`: BOTH stage tables land in ONE Vulkan descriptor set, so the bases must not overlap —
// the caller gives the VS table base 2 and starts the PS table after the VS table's last binding.
// Duplicate binding numbers in one set layout are invalid Vulkan and observably made the PS's sampled
// texture read zeros (a black/blank composite).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps,
                                                       uint32_t binding_base = 2);

// Recompile + resolve a GpuState's draws and render them via `render`. Each draw uses ITS OWN register
// snapshot (Draw::state; the folded end state for snapshot-less draws, e.g. hand-built tests), so
// per-draw masks/blends/shaders/descriptors are honored. Returns the pixels, or {} if nothing could be
// realized. `max_shader_dwords` bounds the recompiler's walk (it stops at S_ENDPGM).
inline std::vector<uint8_t> execute_gpustate(const GpuState& st, const RenderFn& render,
                                             uint32_t max_shader_dwords = 0x10000) {
    if (st.draws.empty() || !render) return {};              // nothing to render (e.g. a state-only submit)
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;    // bail-point visibility (why no frame?)
    std::vector<DrawItem> items;
    // PROSPER_PERDRAW=1: realize each draw from ITS OWN register snapshot (captured at the draw
    // during the fold). Architecturally correct, but currently OPT-IN: the game's AGC context-log
    // packets carry duplicate register writes in sectioned form (current-draw + staged post-draw
    // state — see command_processor.cpp), and until those section semantics are reverse-engineered,
    // draw-time snapshots resolve some state one draw off (observed: the composite gets the previous
    // draw's PGM/mask). Default: every draw uses the submit's folded end state, which renders the
    // known-good title composite.
    static const bool perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    const GpuState* prev_state = nullptr; uint64_t prev_es = 0, prev_ps = 0;
    for (const auto& d : st.draws) {
        const GpuState& ds = (perdraw && d.state) ? *d.state : st;   // the draw's register snapshot
        RenderState rs = extract_render_state(ds);
        // The draw rasterizes with the PRIMITIVE TYPE in effect AT the draw, and the context log
        // stages a post-draw rewrite (the composite's end state reads 4=trilist while its draw-time
        // value is 7=rect) — so prefer the snapshot's prim type whenever it is a value we recognize.
        // (Snapshot values we don't know yet fall back to the end state rather than degrading to
        // points; see the sectioned-log notes in command_processor.cpp.)
        if (!perdraw && d.state) {
            uint32_t sp = extract_render_state(*d.state).prim_type;
            if ((sp >= 1 && sp <= 7) || sp == 17 || sp == 19) rs.prim_type = sp;
        }
        if (!rs.es_addr || !rs.ps_addr) {                    // no vertex/pixel program bound at this draw
            if (log) fprintf(stderr, "[exec] skip draw: no PGM bound (es=0x%llx ps=0x%llx)\n",
                             (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
            continue;
        }
        // Same snapshot + same shader pair as the previous draw -> reuse its recompile/resolve
        // (the common N-draws-one-pipeline case; recompiling per draw would be pure waste).
        if (!items.empty() && &ds == prev_state && rs.es_addr == prev_es && rs.ps_addr == prev_ps) {
            DrawItem it = items.back();
            it.vertex_count = d.index_count ? d.index_count : 3;
            items.push_back(std::move(it));
            continue;
        }
        std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(ds, rs.es_addr, false, 2);
        // The PS table's bindings start AFTER the VS table's (both land in one descriptor set — see
        // build_stage_table's contract; overlapping bindings zeroed the PS's sampled texture).
        uint32_t ps_base = 2 + (vrt ? (uint32_t)vrt->resources.size() : 0);
        std::shared_ptr<ShaderResourceTable> prt = build_stage_table(ds, rs.ps_addr, true, ps_base);
        DrawItem it;
        it.vs = recompile_vertex((const uint32_t*)(uintptr_t)rs.es_addr, max_shader_dwords, vrt.get());
        it.fs = recompile_fragment((const uint32_t*)(uintptr_t)rs.ps_addr, max_shader_dwords, prt.get());
        if (it.vs.empty() || it.fs.empty()) {                // an unsupported shader — skip THIS draw only
            if (log) {
                fprintf(stderr, "[exec] skip draw: recompile failed (vs=%zu fs=%zu dwords; es=0x%llx ps=0x%llx)\n",
                        it.vs.size(), it.fs.size(), (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
                for (auto [tag, addr] : {std::pair{"vs", rs.es_addr}, std::pair{"ps", rs.ps_addr}}) {
                    RecompileCoverage c = recompile_coverage((const uint32_t*)(uintptr_t)addr, max_shader_dwords);
                    fprintf(stderr, "[exec]   %s coverage: total=%u alu=%u exp=%u tabledep=%u unsupported=%u "
                                    "first_bad fmt=%d op=0x%x\n",
                            tag, c.total, c.alu, c.exports, c.table_dependent, c.unsupported,
                            c.first_bad_fmt, c.first_bad_op);
                    // PROSPER_SHADER_DUMP: write the raw code (4 KB cap) for offline llvm-mc disassembly.
                    if (const char* dd = getenv("PROSPER_SHADER_DUMP")) {
                        char fn[512]; snprintf(fn, sizeof fn, "%s/exec_%s_%llx.bin", dd, tag,
                                               (unsigned long long)addr);
                        if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)addr, 1, 4096, f); fclose(f); }
                    }
                }
            }
            continue;
        }
        it.ps = resolve_pipeline_state(rs);
        it.vrt = vrt; it.prt = prt;
        it.vertex_count = d.index_count ? d.index_count : 3;
        // RECT primitives (raw 7 / 17): the hardware expands a 3-corner draw into the full
        // rectangle. We translate to a triangle strip, so draw the 4th corner too — the game's
        // vertex buffer carries it (num_records=4), and it equals the corner hardware derivation
        // would produce (Kyty's Gen5 rect path draws 4 strip vertices the same way).
        if ((rs.prim_type == 7 || rs.prim_type == 17) && it.vertex_count == 3) it.vertex_count = 4;
        // A draw whose color write mask is 0 cannot contribute pixels — don't realize it (our
        // fresh-cleared target model would otherwise PRESENT the bare clear color over the last
        // real frame; on hardware the render target simply keeps its contents).
        if (it.ps.color_write_mask == 0) {
            if (log) fprintf(stderr, "[exec] skip draw: color_write_mask=0 (contributes no pixels)\n");
            continue;
        }
        if (log) fprintf(stderr, "[exec] draw item %zu: vs=%zu fs=%zu dwords vcount=%u mask=0x%x blend=%d "
                         "fmt=%u prim=%u snapprim=%d es=0x%llx ps=0x%llx\n", items.size(), it.vs.size(),
                         it.fs.size(), it.vertex_count, it.ps.color_write_mask, (int)it.ps.blend_enable,
                         rs.color0_format, rs.prim_type,
                         d.state ? (int)extract_render_state(*d.state).prim_type : -1,
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
        items.push_back(std::move(it));
        prev_state = &ds; prev_es = rs.es_addr; prev_ps = rs.ps_addr;
    }
    if (items.empty()) return {};
    if (log) { fprintf(stderr, "[exec] rendering %zu draw item(s) (of %zu draws)\n",
                       items.size(), st.draws.size()); fflush(stderr); }
    return render(items);
}

// --- Live submit renderer registry (Stage A wiring; implemented in gpu_executor.cpp) --------------------
// The live renderer additionally receives the target width/height (from videoout) so it can size its
// attachments. Registered by whoever owns a persistent Vulkan device — the runtime binary at startup, or a
// test — so prosper_core itself stays Vulkan-free (this just stores a std::function).
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

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

// The pluggable Vulkan backend: given recompiled VS+PS SPIR-V and resolved pipeline state, render and
// return W*H*4 RGBA8 pixels (or {} on failure). Signature matches tests/render_runner.h::render_triangle_rgba.
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<uint32_t>& vs,
                                                    const std::vector<uint32_t>& fs,
                                                    const ResolvedPipelineState& ps)>;

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign bindings matching the recompiler+backend convention (constant buffer -> binding
// 2, vertex buffer -> binding 3, textures -> binding 4+). Returns null if the stage has no shader header
// or no resources. Implemented in gpu_executor.cpp (needs the AGC registry + descriptor decode).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps);

// Recompile + resolve a GpuState and render it via `render`. Returns the pixels, or {} if there is nothing
// to draw or a stage fails to recompile. `max_shader_dwords` bounds the recompiler's walk (it stops at
// S_ENDPGM, so this is just an upper bound). CONFIDENCE: HIGH on the orchestration (mirrors the verified
// test_gpustate_render spine); the general multi-draw / vertex-fetch path is handled by the backend.
inline std::vector<uint8_t> execute_gpustate(const GpuState& st, const RenderFn& render,
                                             uint32_t max_shader_dwords = 0x10000) {
    if (st.draws.empty() || !render) return {};              // nothing to render (e.g. a state-only submit)
    RenderState rs = extract_render_state(st);
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;    // bail-point visibility (why no frame?)
    if (!rs.es_addr || !rs.ps_addr) {                        // no vertex/pixel program bound
        if (log) fprintf(stderr, "[exec] skip: no PGM bound (es=0x%llx ps=0x%llx, %zu draws)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr, st.draws.size());
        return {};
    }
    std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(st, rs.es_addr, false);
    std::shared_ptr<ShaderResourceTable> prt = build_stage_table(st, rs.ps_addr, true);
    std::vector<uint32_t> vs = recompile_vertex((const uint32_t*)(uintptr_t)rs.es_addr, max_shader_dwords, vrt.get());
    std::vector<uint32_t> fs = recompile_fragment((const uint32_t*)(uintptr_t)rs.ps_addr, max_shader_dwords, prt.get());
    if (vs.empty() || fs.empty()) {                          // an unsupported shader — leave frame untouched
        if (log) {
            fprintf(stderr, "[exec] skip: recompile failed (vs=%zu fs=%zu dwords; es=0x%llx ps=0x%llx)\n",
                    vs.size(), fs.size(), (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
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
        return {};
    }
    if (log) { fprintf(stderr, "[exec] BOTH stages recompiled: vs=%zu fs=%zu dwords -> rendering\n",
                       vs.size(), fs.size()); fflush(stderr); }
    ResolvedPipelineState ps = resolve_pipeline_state(rs);
    return render(vs, fs, ps);
}

// --- Live submit renderer registry (Stage A wiring; implemented in gpu_executor.cpp) --------------------
// The live renderer additionally receives the target width/height (from videoout) so it can size its
// attachments. Registered by whoever owns a persistent Vulkan device — the runtime binary at startup, or a
// test — so prosper_core itself stays Vulkan-free (this just stores a std::function). Signature adds w,h to
// RenderFn; the tests' render_triangle_rgba already takes (vs, fs, w, h, &ps).
using LiveRenderFn = std::function<std::vector<uint8_t>(const std::vector<uint32_t>& vs,
                                                        const std::vector<uint32_t>& fs,
                                                        const ResolvedPipelineState& ps,
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

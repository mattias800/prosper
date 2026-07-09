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
#include "agc_shader_layout.hpp"   // DecodedBufferDescriptor (DynFetch)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>
#include <set>

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
    // Indexed draw (sceAgcDcbDrawIndex): the guest index buffer, fetched from 1:1-mapped memory and
    // widened to 32-bit. Non-empty -> the backend must render with vkCmdDrawIndexed (gl_VertexIndex
    // then IS the fetched index, which the recompiled VS uses for its storage-buffer vertex fetch);
    // empty -> plain vkCmdDraw(vertex_count).
    std::vector<uint32_t> indices;
    // Render-to-texture (#167): the CB_COLOR0_BASE this draw renders INTO. The game renders its scene
    // into a color target then samples that same address as a texture in a later composite pass; the
    // live renderer caches each submit's rendered pixels under this address and injects them when a
    // subsequent draw samples a texture at a matching base (otherwise the sample reads empty guest
    // memory — the scene RT is never populated on the CPU side — and the frame is a black composite).
    uint64_t color0_base = 0;
};

// The pluggable Vulkan backend: render the submit's draw items into one image and return W*H*4 RGBA8
// pixels (or {} on failure). Empty list -> {} (nothing to draw).
using RenderFn = std::function<std::vector<uint8_t>(const std::vector<DrawItem>& items)>;

// Safe guest-address readability probe: write() to /dev/null returns EFAULT for an unmapped source
// (Linux; always-true on Windows), so callers can test a guest pointer without risking a SIGSEGV.
// Implemented in gpu_executor.cpp; shared by the executor's const-eval and HLE diagnostic probes.
bool guest_readable(uint64_t addr, uint32_t bytes);

// One resolved bindless-dynamic vertex fetch from the wave-uniform scalar const-fold in
// gpu_executor.cpp: the exact fetch instruction (pc), its SRSRC SGPR, and the V# live in that SGPR
// at that instruction. Exposed (with resolve_dynamic_fetch) so the fold's scalar-ALU semantics are
// unit-testable; production callers stay inside gpu_executor.cpp.
struct DynFetch { uint32_t fetch_pc; int srsrc; DecodedBufferDescriptor desc; uint32_t desc_v3; };
std::vector<DynFetch> resolve_dynamic_fetch(const uint32_t* code, size_t dwords,
                                            const uint32_t* user_sgprs, uint32_t nsgpr,
                                            uint32_t user_sgpr_base);

// Assign each resource in `t` a descriptor binding from `first`: constant/vertex buffers first, then
// textures / storage images — never on binding 2 or 3 (the recompiler's two hardwired cbufs) so a
// texture-first shader can't collide two descriptor types at one binding (#157). Exposed for testing.
void assign_convention_bindings(ShaderResourceTable& t, uint32_t first);

// Build a shader stage's resource table from the folded GpuState: look up the registered shader header
// by its bound code address, read its user-data SGPR block from the sh register file, decode the V#/T#/S#
// descriptors, and assign bindings matching the recompiler+backend convention (constant buffer -> binding
// 2, vertex buffer -> binding 3, textures -> binding 4+). Returns null if the stage has no shader header
// or no resources. Implemented in gpu_executor.cpp (needs the AGC registry + descriptor decode).
std::shared_ptr<ShaderResourceTable> build_stage_table(const GpuState& st, uint64_t code_addr, bool is_ps);

// Byte size of one index element for a GpuState::index_type (the last SetIndexType value).
// 0 -> 16-bit, 1 -> 32-bit, exactly Kyty's index_type_and_size switch (GraphicsRender.cpp:4724) and
// the hardware VGT_INDEX_TYPE encoding; 0 is also the reset default, matching this title, which never
// emits SetIndexType yet packs its quad index streams 12 bytes apart (6 x 2-byte indices) — live
// capture confirms 16-bit. CONFIDENCE: HIGH for 0/1; any other value is unseen -> returns 0 and the
// caller falls back to a non-indexed draw (loudly), rather than mis-reading the buffer.
inline uint32_t index_elem_bytes(uint32_t index_type) {
    return index_type == 0 ? 2u : index_type == 1 ? 4u : 0u;
}

// Realize ONE draw of `ds` (a register snapshot or the folded state) into a DrawItem: recompile the
// VS+PS, resolve fixed-function state, and — for an indexed draw — fetch the guest index buffer.
// `draw` is the PM4 draw record (index count + indexed/index_addr); null means "no record" (hand-built
// states) and renders vcount_hint vertices non-indexed. Returns false (and leaves `out` untouched) if
// the draw is a no-op (no PGM bound, recompile failed, or color_write_mask==0). Shared by the default
// (folded-state, one item) and PROSPER_PERDRAW (per-draw) paths so their per-draw handling is identical.
inline bool realize_draw_item(const GpuState& ds, const GpuState::Draw* draw, uint32_t vcount_hint,
                              uint32_t max_shader_dwords, bool log, DrawItem& out) {
    RenderState rs = extract_render_state(ds);
    if (!rs.es_addr || !rs.ps_addr) {
        if (log) fprintf(stderr, "[exec] skip draw: no PGM bound (es=0x%llx ps=0x%llx)\n",
                         (unsigned long long)rs.es_addr, (unsigned long long)rs.ps_addr);
        return false;
    }
    std::shared_ptr<ShaderResourceTable> vrt = build_stage_table(ds, rs.es_addr, false);
    std::shared_ptr<ShaderResourceTable> prt = build_stage_table(ds, rs.ps_addr, true);
    // PROSPER_RTLOG: correlate this draw's render-target address (CB_COLOR0_BASE) with the addresses of
    // the textures it SAMPLES. If a sampled texture's base equals some draw's color0_base, that surface is
    // a GPU render target (the game renders into it then samples it) -> render-to-texture (#83/#101).
    if (getenv("PROSPER_RTLOG")) {
        fprintf(stderr, "[rt] color0=0x%llx", (unsigned long long)rs.color0_base);
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " tex=0x%llx(%ux%u f%u c%u)", (unsigned long long)r.gpu_addr, r.width, r.height,
                        (unsigned)r.format, r.num_components);
        if (vrt) for (const auto& r : vrt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " vtex=0x%llx(%ux%u f%u)", (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
        fprintf(stderr, "\n");
    }
    std::vector<uint32_t> vs = recompile_vertex((const uint32_t*)(uintptr_t)rs.es_addr, max_shader_dwords, vrt.get());
    std::vector<uint32_t> fs = recompile_fragment((const uint32_t*)(uintptr_t)rs.ps_addr, max_shader_dwords, prt.get());
    if (const char* dd = getenv("PROSPER_VS_DUMP")) {   // diag: dump successful VS SPIR-V + raw RDNA2 for inspection
        static int nd = 0;
        if (nd < 3 && !vs.empty()) {
            char fn[512];
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.spv", dd, nd, (unsigned long long)rs.es_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
            snprintf(fn, sizeof fn, "%s/vs_%d_%llx.bin", dd, nd, (unsigned long long)rs.es_addr);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.es_addr, 1, 4096, f); fclose(f); }
            nd++;
        }
    }
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
    if (ps.color_write_mask == 0 && !getenv("PROSPER_FORCE_COLORWRITE")) {
        if (log) fprintf(stderr, "[exec] skip draw: color_write_mask==0 (no-op) cb_target_mask=0x%x cb_color_control=0x%x color0_fmt=%u\n",
                         rs.cb_target_mask, rs.cb_color_control, ps.color0_format);
        return false;
    }
    // PROSPER_FORCE_COLORWRITE: diagnostic — render color_write_mask==0 draws anyway (force mask to RGBA).
    // The cutscene submits ~66 draws/frame that resolve to mask==0; if that is a mis-decode (not a genuine
    // depth-only pass), rendering them reveals whether they are the cutscene content.
    if (ps.color_write_mask == 0) ps.color_write_mask = 0xf;
    // PROSPER_DRAWDIAG: per-RENDERED-draw geometry/position/texture — to LOCATE specific draws (e.g. the
    // cutscene caption text: small indexed quads, bottom viewport, blended, sampling a font atlas).
    if (getenv("PROSPER_DRAWDIAG")) {
        uint32_t ic = (draw && draw->indexed) ? draw->index_count : vcount_hint;
        fprintf(stderr, "[draw] idx=%u vp=%d y=%.0f h=%.0f blend=%d cwm=0x%x es=0x%llx", ic,
                ps.has_viewport, ps.viewport_y, ps.viewport_h, ps.blend_enable, ps.color_write_mask,
                (unsigned long long)rs.es_addr);
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture)
                fprintf(stderr, " tex=0x%llx(%ux%u f%u)", (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
        fprintf(stderr, "\n");
    }
    uint32_t vertex_count = vcount_hint ? vcount_hint : 3u;
    // The bound vertex buffer's record count (size/stride) — bounds an indexed draw's vertex range, and
    // for a NON-indexed draw is often the truer count: a draw record's index_count can be a low/stale
    // value for these NGG draws (4 of ~20 verts -> a degenerate sliver), while the VB's record count is
    // the whole mesh. A shader fetching past a real vertex reads 0 under robustBufferAccess -> a
    // degenerate, clipped vertex, so a slightly-generous count is harmless.
    uint32_t vb_entries = 0;
    if (vrt) for (const auto& r : vrt->resources)
        if (r.cls == ResourceClass::VertexBuffer && r.stride)
            vb_entries = std::max(vb_entries, r.size / r.stride);
    if (vb_entries > 65536u) vb_entries = 65536u;   // sanity cap: don't stall llvmpipe on an over-sized VB
    // Indexed draw (sceAgcDcbDrawIndex): fetch the real index data from guest memory (1:1-mapped) and
    // hand it to the backend, which renders it with vkCmdDrawIndexed. This replaced the old "4-record
    // VB -> TRIANGLE_FAN" heuristic (issue #64): the sprite quads that heuristic guessed at are in fact
    // DrawIndex packets with a 6-entry index list ([0,1,2, 2,3,0]-style two-triangle quads), so the
    // real indices + the real decoded VGT_PRIMITIVE_TYPE topology render every 4-vertex mesh correctly,
    // whatever its vertex order. Unknown element size or an unreadable buffer falls back (loudly) to a
    // non-indexed draw of the hint count instead of reading garbage.
    static constexpr uint32_t kMaxIndices = 1u << 20;   // sanity cap (largest seen live: 0x61e)
    if (draw && draw->indexed && draw->index_addr && draw->index_count) {
        const uint32_t esz = index_elem_bytes(ds.index_type);
        uint32_t n = std::min(draw->index_count, kMaxIndices);
        if (esz == 0) {
            if (log) fprintf(stderr, "[exec] indexed draw: UNKNOWN index_type=%u — falling back to non-indexed\n",
                             ds.index_type);
        } else if (!guest_readable(draw->index_addr, n * esz)) {
            if (log) fprintf(stderr, "[exec] indexed draw: index buffer 0x%llx (%u x %uB) unreadable — "
                             "falling back to non-indexed\n",
                             (unsigned long long)draw->index_addr, n, esz);
        } else {
            out.indices.resize(n);
            uint32_t max_index = 0;
            if (esz == 2) {
                const uint16_t* src = (const uint16_t*)(uintptr_t)draw->index_addr;
                for (uint32_t i = 0; i < n; i++) { out.indices[i] = src[i]; max_index = std::max(max_index, out.indices[i]); }
            } else {
                const uint32_t* src = (const uint32_t*)(uintptr_t)draw->index_addr;
                for (uint32_t i = 0; i < n; i++) { out.indices[i] = src[i]; max_index = std::max(max_index, out.indices[i]); }
            }
            // Vertex count = the indexed range (max index + 1). The INDEX BUFFER is authoritative for how
            // many vertices the draw touches — do NOT clamp down to the bound VB's record count: the
            // bindless per-glyph fetch resolves a tiny per-glyph V# (num_records=4 = one glyph), so clamping
            // to it would drop every glyph but the first (#257). The VB is grown below to span this range;
            // a truly-unmapped tail still degrades safely (safe_copy stops at the mapping edge -> zero).
            vertex_count = max_index + 1;
        }
    }
    if (out.indices.empty() && vb_entries > vertex_count) vertex_count = vb_entries;
    // Bindless per-glyph vertex fetch (#257): the fetch-shader patches a SMALL per-glyph V# (num_records=4
    // = one glyph's 4 corners, size=304). But the draw indexes ALL vertices (gl_VertexIndex 0..N-1) out of
    // the CONTIGUOUS vertex pool that begins at that base — so uploading only num_records*stride bytes
    // leaves every vertex past the first glyph reading out-of-bounds (robustBufferAccess 0), collapsing the
    // whole caption to the first glyph. Grow each vertex buffer to cover the draw's full vertex range so
    // gl_VertexIndex reads the real per-vertex data. (Correct in general: a VB must span the drawn range.)
    if (vrt) for (auto& r : vrt->resources)
        if (r.cls == ResourceClass::VertexBuffer && r.stride) {
            uint32_t need = vertex_count * r.stride;
            if (r.size < need) r.size = need;
        }
    // PROSPER_CAPTION_DIAG: for the caption text mesh (a draw whose PS samples the 2048x1024 R8 font
    // atlas), dump everything needed to see WHY its geometry collapses — vertex count, the bound vertex
    // buffers (base/stride/size), the index range, and the VS bytecode (for offline llvm-mc disasm of the
    // position export). Once per distinct VS. #102 follow-up (font atlas decodes; glyphs land nowhere).
    // PROSPER_ONLY_ATLAS: render ONLY the caption text draw (samples the 2048x1024 font atlas), skipping
    // every other draw — so the caption geometry renders alone onto a clear frame. With TESTPS this shows
    // whether the caption rasterizes at all (magenta glyphs) or is culled/degenerate/clipped. #257.
    if (getenv("PROSPER_ONLY_ATLAS")) {
        bool sa = false;
        if (prt) for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture && r.width == 2048 && r.height == 1024) sa = true;
        if (!sa) return false;
        // PROSPER_ONLY_IC=<n>: further restrict to draws whose index count == n (isolate one atlas mesh,
        // e.g. the main glyph batch idx=1566 vs a fullscreen atlas draw).
        if (const char* ic_s = getenv("PROSPER_ONLY_IC")) {
            uint32_t want = (uint32_t)atoi(ic_s);
            uint32_t ic = (draw && draw->indexed) ? draw->index_count : vcount_hint;
            if (ic != want) return false;
        }
    }
    if (getenv("PROSPER_CAPTION_DIAG") && prt) {
        bool samples_atlas = false;
        for (const auto& r : prt->resources)
            if (r.cls == ResourceClass::Texture && r.width == 2048 && r.height == 1024) samples_atlas = true;
        static std::set<uint64_t> seen_vs;
        if (samples_atlas && seen_vs.insert(rs.es_addr).second) {
            uint32_t maxi = 0; for (uint32_t v : out.indices) maxi = std::max(maxi, v);
            fprintf(stderr, "[caption] es=0x%llx vertex_count=%u indices=%zu max_index=%u vb_entries=%u\n",
                    (unsigned long long)rs.es_addr, vertex_count, out.indices.size(), maxi, vb_entries);
            if (vrt) for (const auto& r : vrt->resources) {
                fprintf(stderr, "[caption]   %s binding=%u base=0x%llx stride=%u size=%u fmt=%u nc=%u\n",
                        r.cls == ResourceClass::VertexBuffer ? "VB" :
                        r.cls == ResourceClass::ConstantBuffer ? "CB" : "TEX",
                        r.binding, (unsigned long long)r.gpu_addr, r.stride, r.size, (unsigned)r.format, r.num_components);
                // Raw first 3 records as floats — reveals whether the fetched position/attr data is valid or
                // NaN/degenerate (a stride-0 buffer reads record 0 for every vertex -> collapse). #257.
                if (r.cls == ResourceClass::VertexBuffer && r.gpu_addr > 0x10000) {
                    uint32_t st = r.stride ? r.stride : 16;
                    for (int rec = 0; rec < 3; rec++) {
                        const float* f = (const float*)(uintptr_t)(r.gpu_addr + (uint64_t)rec * st);
                        const uint32_t* u = (const uint32_t*)(uintptr_t)(r.gpu_addr + (uint64_t)rec * st);
                        fprintf(stderr, "[caption]       rec%d: %.3f %.3f %.3f %.3f  (raw %08x %08x %08x %08x)\n",
                                rec, f[0], f[1], f[2], f[3], u[0], u[1], u[2], u[3]);
                    }
                }
                // Constant buffers = the transform matrices / uniforms. Dump as vec4 rows (the collapse may
                // be a mis-resolved MVP -> valid glyph coords transform off-screen / to w<=0). #257.
                if (r.cls == ResourceClass::ConstantBuffer && r.gpu_addr > 0x10000) {
                    uint32_t nvec = r.size / 16; if (nvec > 24) nvec = 24;
                    for (uint32_t v = 0; v < nvec; v++) {
                        const float* f = (const float*)(uintptr_t)(r.gpu_addr + (uint64_t)v * 16);
                        fprintf(stderr, "[caption]       cb[%u]: %.4f %.4f %.4f %.4f\n", v, f[0], f[1], f[2], f[3]);
                    }
                }
            }
            fflush(stderr);
            if (const char* dd = getenv("PROSPER_FRAME_DIR")) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/caption_vs_%llx.bin", dd, (unsigned long long)rs.es_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.es_addr, 1, 8192, f); fclose(f); }
                // The RECOMPILED VS SPIR-V — disassemble offline (spirv-dis) to trace the gl_Position export
                // op-by-op against the RDNA2 source and find the mis-modeled op / bad matrix input. #257.
                snprintf(fn, sizeof fn, "%s/caption_recompiled_%llx.spv", dd, (unsigned long long)rs.es_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite(vs.data(), 4, vs.size(), f); fclose(f); }
                // The recompiled PIXEL shader — the text is invisible because the PS discards every fragment
                // (SDF alpha-test). Dump it (spirv-dis) to inspect the discard condition + atlas sample. #257.
                snprintf(fn, sizeof fn, "%s/caption_ps_%llx.spv", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite(fs.data(), 4, fs.size(), f); fclose(f); }
                snprintf(fn, sizeof fn, "%s/caption_ps_raw_%llx.bin", dd, (unsigned long long)rs.ps_addr);
                if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)rs.ps_addr, 1, 8192, f); fclose(f); }
            }
        }
    }
    out.vs = std::move(vs); out.fs = std::move(fs); out.ps = ps;
    out.vrt = std::move(vrt); out.prt = std::move(prt); out.vertex_count = vertex_count;
    out.color0_base = rs.color0_base;   // render-to-texture: the target this draw writes into (#167)
    return true;
}

// Recompile + resolve a GpuState's draws and render them via `render`. Default: ONE item from the folded
// end state, realized as the submit's LAST draw — the folded register file IS the state at the last
// draw, so that is the one draw record (count + index buffer) coherent with it. (The pre-#64 code paired
// the end state with draws[0]'s count instead, and needed a "4-record VB -> TRIANGLE_FAN" heuristic to
// paper over the mismatch; the title composite is in fact the submit's last draw, a 6-index DrawIndex
// quad, which now renders through the real indexed path.) PROSPER_PERDRAW=1: ONE item per draw, each
// realized from ITS OWN register snapshot (Draw::state), so per-draw masks/blends/shaders composite
// correctly — the path for multi-geometry scenes (opt-in until the AGC context-log section semantics
// that stage duplicate register writes are fully RE'd; see docs/REAL_FRAMES_FINDINGS.md).
// `max_shader_dwords` bounds the recompiler's walk (it stops at S_ENDPGM).
// vp_scale_{x,y}: scale each draw's guest viewport by this factor. The guest programs PA_CL_VPORT in
// full present-resolution pixels; when we render into a reduced-resolution framebuffer (PROSPER_RENDER_SCALE)
// the viewport must shrink by the same ratio, or a full-res viewport into a small framebuffer clips the
// image to its bottom-left corner. Default 1.0 (full-res / tests: no change).
inline std::vector<uint8_t> execute_gpustate(const GpuState& st, const RenderFn& render,
                                             uint32_t max_shader_dwords = 0x10000,
                                             float vp_scale_x = 1.0f, float vp_scale_y = 1.0f) {
    if (st.draws.empty() || !render) return {};              // nothing to render (e.g. a state-only submit)
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;    // bail-point visibility (why no frame?)
    // Render each draw from its OWN register snapshot when the submit has MULTIPLE draws — a real
    // multi-geometry scene (the game's in-game/cutscene submits carry 8-11 distinct draws with per-draw
    // shaders/textures/blends). Folding those to just the last draw drops the rest and the frame comes out
    // as the bare clear; per-draw rendering makes the intro cutscene's real content appear (black scene +
    // its geometry) instead of a blank blue clear. A single-draw submit stays folded (nothing to composite).
    // Overridable: PROSPER_PERDRAW forces per-draw always, PROSPER_FOLDED forces the old single-item path.
    // CONFIDENCE: MED — verified multi-draw scenes render their content per-draw vs. a blank clear folded.
    static const bool force_perdraw = getenv("PROSPER_PERDRAW") != nullptr;
    static const bool force_folded  = getenv("PROSPER_FOLDED") != nullptr;
    const bool perdraw = force_perdraw || (!force_folded && st.draws.size() > 1);
    std::vector<DrawItem> items;
    if (perdraw) {
        for (size_t i = 0; i < st.draws.size(); i++) {
            DrawItem it;
            if (realize_draw_item(st.state_at_draw(i), &st.draws[i], st.draws[i].index_count,
                                  max_shader_dwords, log, it))
                items.push_back(std::move(it));
        }
    } else {
        // Default: render the submit's last draw from the folded end state as a single item.
        DrawItem it;
        const GpuState::Draw& last = st.draws.back();
        if (realize_draw_item(st, &last, last.index_count, max_shader_dwords, log, it))
            items.push_back(std::move(it));
    }
    if (getenv("PROSPER_DRAWLOG")) { fprintf(stderr, "[exec] draws=%zu perdraw=%d -> %zu item(s): raw index_counts=[",
        st.draws.size(), (int)perdraw, items.size());
        for (size_t i = 0; i < st.draws.size(); i++) fprintf(stderr, "%s%u%s", i?",":"", st.draws[i].index_count,
                                                             st.draws[i].indexed ? "i" : "");
        fprintf(stderr, "] items:");
        for (auto& it : items) fprintf(stderr, " (vcount=%u nidx=%zu topo=%u mask=0x%x)",
                                       it.vertex_count, it.indices.size(), it.ps.topology, it.ps.color_write_mask);
        fprintf(stderr, "\n"); fflush(stderr); }
    if (items.empty()) return {};
    // Scale each item's guest viewport to the actual (possibly reduced-resolution) framebuffer. Skip if
    // 1.0 (full-res) or if this draw has no guest viewport (the backend then uses the full-target default).
    if (vp_scale_x != 1.0f || vp_scale_y != 1.0f)
        for (auto& it : items)
            if (it.ps.has_viewport) {
                it.ps.viewport_x *= vp_scale_x; it.ps.viewport_w *= vp_scale_x;
                it.ps.viewport_y *= vp_scale_y; it.ps.viewport_h *= vp_scale_y;
            }
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

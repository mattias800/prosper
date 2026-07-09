// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "live_renderer.hpp"

#include "gpu/gpu_execute.hpp"          // DrawItem, set_submit_renderer
#include "gpu/tile.hpp"                 // detile_surface / tiled_surface_bytes / detile_elements
#include "gpu/bc_decode.hpp"            // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/shader_resources.hpp"     // ShaderResourceTable / ResourceClass
#include "gpu/rdna2_to_spirv.hpp"       // recompile_fragment (diagnostic solid-color PS)
#include "render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);

namespace prosper::frontend {

// Render-to-texture surface cache (#167): CB_COLOR0_BASE -> the RGBA pixels we last rendered into it.
// The game renders its scene into a color target then samples that address as a texture in a later
// composite pass. Guest memory at that address is never populated on our (CPU-read) side, so without
// this the composite samples zeros and the frame is black. We cache each submit's rendered pixels under
// its render-target base and inject them when a subsequent draw samples a texture at a matching base.
namespace { struct RttSurf { std::vector<uint8_t> rgba; uint32_t w = 0, h = 0; }; }

void register_live_renderer(const std::string& frame_dir, bool dump_bmps) {
    static std::atomic<int> frame_no{0};
    static std::unordered_map<uint64_t, RttSurf> g_rtt;   // render-to-texture cache (#167)
    static const bool rtt_on = getenv("PROSPER_RTT") != nullptr;
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps](const std::vector<prosper::gpu::DrawItem>& items, uint32_t w, uint32_t h) -> std::vector<uint8_t> {
            using RC = prosper::gpu::ResourceClass;
            // PROSPER_RENDER_FIRST=<N>: skip the slow (~400x) Vulkan render for the first N GPU submits, so
            // the game reaches a LATE scene (e.g. the level1 cutscene, which only starts submitting after
            // ~5000 title-loop submits) at native speed before we begin rendering/dumping. Returning {}
            // means "not rendered this submit". Without this, rendering from boot is far too slow to ever
            // reach a post-loading-screen scene.
            static std::atomic<int> g_submit_idx{0};
            static int g_render_first = getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0;
            if (g_submit_idx++ < g_render_first) return {};
            // Dump the FIRST item's recompiled SPIR-V (diagnostic; survives a mid-render crash).
            if (getenv("PROSPER_SHADER_DUMP") && !items.empty()) {
                std::string d = getenv("PROSPER_SHADER_DUMP");
                if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(items[0].vs.data(), 4, items[0].vs.size(), f); fclose(f); }
                if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(items[0].fs.data(), 4, items[0].fs.size(), f); fclose(f); }
                fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", items[0].vs.size(), items[0].fs.size()); fflush(stderr);
            }
            // Copy [a, a+n) into dst, but stop at the first 64KB block that is NOT within a reserved guest
            // mapping (prosper_reserved_range_state == 0). A resource's declared size (e.g. a 2048x1024
            // atlas = 8 MB) can run past its real committed backing; an unguarded memcpy then walks off
            // the mapping and SIGSEGVs the render thread mid-copy — /dev/null-write "readable" can't catch
            // it because /dev/null discards without faulting. Returns bytes copied; dst is pre-zeroed so a
            // short copy just leaves a transparent/black tail (only the missing region degrades).
            auto safe_copy = [](uint8_t* dst, uint64_t a, size_t n) -> size_t {
                const size_t PG = 0x10000;   // lazy-commit granularity (64 KB)
                size_t done = 0;
                while (done < n) {
                    uint64_t cur = a + done;
                    if (cur < 0x1000 || prosper_reserved_range_state(cur) == 0) break;
                    size_t chunk = std::min(n - done, PG - (size_t)(cur & (PG - 1)));
                    std::memcpy(dst + done, (const void*)(uintptr_t)cur, chunk);
                    done += chunk;
                }
                return done;
            };
            static std::vector<std::vector<uint8_t>> texstore;  // keep every item's texture bytes alive
            texstore.clear();
            // Build one draw's set-tagged resources from its VS (set 0) + PS (set 1) tables — read the
            // bytes from 1:1-mapped guest memory, detile textures. (Each constant/vertex buffer + texture
            // gets its own binding; the recompiler declared a storage buffer / image sampler at each.)
            auto build_R = [&](const prosper::gpu::ShaderResourceTable* vrt, const prosper::gpu::ShaderResourceTable* prt) {
              std::vector<prosper::test::FrameResource> R;
              auto add = [&](const prosper::gpu::ShaderResourceTable* t, uint32_t set){
                if (!t) return;
                for (auto& r : t->resources) {
                    prosper::test::FrameResource fr; fr.binding = r.binding; fr.set = set;
                    if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                        uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                        size_t nb = (size_t)tw * th * 4;
                        texstore.emplace_back(nb, 0x00);
                        // RTT (#167): if this texture's base is a color target we rendered into, inject those
                        // pixels (nearest-scaled to tw x th) instead of reading empty guest memory.
                        bool rtt_hit = false;
                        if (rtt_on) { auto rit = g_rtt.find(r.gpu_addr);
                            if (rit != g_rtt.end() && rit->second.w && rit->second.h && !rit->second.rgba.empty()) {
                                const RttSurf& s = rit->second;
                                for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                    uint32_t sx = (uint32_t)((uint64_t)x * s.w / tw), sy = (uint32_t)((uint64_t)y * s.h / th);
                                    size_t si = ((size_t)sy * s.w + sx) * 4;
                                    if (si + 4 <= s.rgba.size()) std::memcpy(&texstore.back()[((size_t)y * tw + x) * 4], &s.rgba[si], 4);
                                }
                                rtt_hit = true;
                            }
                            if (getenv("PROSPER_RTTLOG"))
                                fprintf(stderr, "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> %s (cache_size=%zu)\n",
                                        (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format,
                                        rtt_hit ? "HIT" : "miss", g_rtt.size());
                        }
                        // Block-compressed (BC1/2/3): read the (possibly tiled) compressed blocks, block-
                        // detile, and decode to RGBA8 in-place. The blocks are the tiled element (BC3 =
                        // 16 bytes -> SW_4KB_S 16x16-block micro-tiles). #121.
                        const uint32_t bcb = rtt_hit ? 0u : prosper::gpu::bc_block_bytes(r.format);
                        if (rtt_hit) { /* pixels already injected from the RTT cache */ }
                        else if (bcb) {
                            uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                            // Block-detile: tiled_elements_bytes/detile_elements now derive the 4KB
                            // micro-tile geometry from the block size (bpe) internally (#119) — 16-byte
                            // blocks -> 16x16, 8-byte -> 32x16 — so no tile_side is passed here.
                            size_t comp_bytes = (size_t)bw * bh * bcb;
                            std::vector<uint8_t> lin(comp_bytes, 0);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = prosper::gpu::tiled_elements_bytes(bw, bh, bcb, r.tile_mode);
                                std::vector<uint8_t> traw(tbytes, 0);
                                safe_copy(traw.data(), r.gpu_addr, tbytes);
                                prosper::gpu::detile_elements(lin.data(), traw.data(), tbytes, bw, bh, bcb, r.tile_mode);
                            } else {
                                safe_copy(lin.data(), r.gpu_addr, comp_bytes);
                            }
                            prosper::gpu::bc_decode_surface(texstore.back().data(), lin.data(), lin.size(), tw, th, r.format);
                        } else {
                            safe_copy(texstore.back().data(), r.gpu_addr, nb);
                        }
                        // PROSPER_DUMP_RAWTEX: write the raw tiled RGBA bytes (pre-detile) to a .bin for
                        // offline swizzle experimentation.
                        if (getenv("PROSPER_DUMP_RAWTEX") && !texstore.back().empty()) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_%ux%u.bin", d.c_str(), tw, th);
                            if (FILE* f = fopen(fn, "wb")) { fwrite(texstore.back().data(), 1, nb, f); fclose(f);
                                fprintf(stderr, "[render] dumped raw tiled bytes -> %s (%zu)\n", fn, nb); fflush(stderr); }
                        }
                        // PROSPER_TESTTEX: overwrite with a recognizable gradient/checker to prove the
                        // sample path (the game's real render-target texture is empty until we execute
                        // the scene draws that fill it).
                        if (getenv("PROSPER_TESTTEX")) {
                            for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                uint8_t* p = &texstore.back()[((size_t)y * tw + x) * 4];
                                bool ck = ((x / 64) ^ (y / 64)) & 1;
                                p[0] = (uint8_t)(255 * x / tw); p[1] = (uint8_t)(255 * y / th);
                                p[2] = ck ? 200 : 40; p[3] = 255;
                            }
                        }
                        if (getenv("PROSPER_GFXLOG")) { const uint8_t* b = texstore.back().data();
                            size_t nz = 0; for (size_t i = 0; i < nb && i < (1u<<16); i++) nz += (b[i] != 0);
                            fprintf(stderr, "[render] tex binding=%u %ux%u first64k-nonzero=%zu\n", r.binding, tw, th, nz); }
                        // AUTO-DETILE: de-swizzle a GPU-tiled sampled surface into the linear texstore,
                        // driven by the T# tile_mode threaded through the resource table (r.tile_mode).
                        bool auto_tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                        const char* dt = getenv("PROSPER_DETILE");
                        // BC textures already block-detiled + decoded above; the 32-bpp detiler must not touch them.
                        if (!rtt_hit && !bcb && !getenv("PROSPER_NODETILE") && (auto_tiled || (dt && atoi(dt) != 0))) {
                            const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                            const uint32_t pitch = getenv("PROSPER_PITCH") ? (uint32_t)atoi(getenv("PROSPER_PITCH")) : 0;
                            size_t tiled_bytes = prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
                            std::vector<uint8_t> tiled(tiled_bytes, 0);
                            size_t got = safe_copy(tiled.data(), r.gpu_addr, tiled_bytes);
                            if (got < nb)
                                // The padded tiled buffer's tail (th rounded to whole 32-row tiles) runs past
                                // the real backing: fall back to the width*height linear bytes copied above
                                // rather than an all-zero buffer (which would BLANK the texture).
                                std::memcpy(tiled.data(), texstore.back().data(), std::min(nb, tiled_bytes));
                            // PROSPER_DUMP_RAWTILE: write the EXACT padded tiled bytes to a .bin (no lossy BMP
                            // round-trip) so the de-swizzle can be reversed offline against a known image (#101).
                            if (getenv("PROSPER_DUMP_RAWTILE") && frame_no < 200) {
                                std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char bn[512]; snprintf(bn, sizeof bn, "%s/tiled_f%04d_b%u_%ux%u.bin",
                                                       dd.c_str(), (int)frame_no, r.binding, tw, th);
                                if (FILE* bf = fopen(bn, "wb")) { fwrite(tiled.data(), 1, tiled.size(), bf); fclose(bf); }
                            }
                            prosper::gpu::detile_surface(texstore.back().data(), tiled.data(), tw, th, tmode, pitch);
                        }
                        // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                        // bypassing the shader — reveals whether the render target is tiled or linear.
                        if (getenv("PROSPER_DUMP_TEX") && !texstore.back().empty() && frame_no < 200) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                            prosper::test::dump_bmp(fn, texstore.back(), tw, th);
                            fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                        }
                        fr.tex_rgba = texstore.back().data(); fr.tw = tw; fr.th = th;
                    } else {
                        uint32_t nb = std::min(r.size ? r.size : 256u, 1u << 20) & ~3u;   // cap 1 MB, dword-aligned
                        if (nb >= 4) {
                            std::vector<uint8_t> tmp(nb, 0);
                            if (safe_copy(tmp.data(), r.gpu_addr, nb) > 0)
                                fr.dwords.assign((const uint32_t*)tmp.data(), (const uint32_t*)(tmp.data() + nb));
                        }
                        if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                    }
                    R.push_back(std::move(fr));
                }
              };
              add(vrt, 0); add(prt, 1);   // VS resources -> descriptor set 0, PS -> set 1
              return R;
            };
            // Diagnostic shader/state overrides (computed once, applied to EVERY draw item):
            //   REFVS  -> a known-good fullscreen-triangle VS (isolates the game's real VS).
            //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading).
            //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
            //   NOPS   -> bypass the resolved pipeline state (default state).
            #include "refvs.inc"
            const bool refvs = getenv("PROSPER_RENDER_REFVS");
            std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
            std::vector<uint32_t> ps_override;
            if (getenv("PROSPER_RENDER_TESTPS")) {
                static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
                    0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
                ps_override = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
            }
            if (const char* fsp = getenv("PROSPER_FS_SPV")) {
                if (FILE* f = fopen(fsp, "rb")) {
                    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                    std::vector<uint32_t> m((size_t)sz / 4);
                    if (sz >= 20 && fread(m.data(), 4, m.size(), f) == m.size()) ps_override = std::move(m);
                    fclose(f);
                }
            }
            const bool nops = getenv("PROSPER_RENDER_NOPS");
            // Assemble the backend draws — one per realized DrawItem, each with its own resources +
            // fixed-function state (or the diagnostic overrides above). render_draws_rgba composites
            // them into ONE framebuffer.
            std::vector<prosper::test::BackendDraw> bds;
            for (const auto& it : items) {
                prosper::test::BackendDraw bd;
                bd.vs     = refvs ? refvs_spv : it.vs;
                bd.fs     = ps_override.empty() ? it.fs : ps_override;
                bd.vcount = refvs ? 3u : it.vertex_count;
                bd.ps     = nops ? nullptr : &it.ps;
                bd.R      = build_R(it.vrt.get(), it.prt.get());
                // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
                // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
                if (!refvs) bd.indices = it.indices;
                if (getenv("PROSPER_GFXLOG")) fprintf(stderr,
                    "[render] item %zu: %zu resources vcount=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
                    bds.size(), bd.R.size(), bd.vcount, bd.indices.size(), it.ps.topology,
                    it.ps.color_write_mask, (int)it.ps.blend_enable);
                bds.push_back(std::move(bd));
            }
            std::vector<uint8_t> px = prosper::test::render_draws_rgba(bds, w, h);
            // RTT (#167): cache these rendered pixels under this submit's render-target base, so a later
            // composite pass that samples that address gets the scene we drew (not empty guest memory).
            if (rtt_on && !px.empty()) {
                uint64_t tgt = 0;
                for (const auto& it : items) if (it.color0_base) { tgt = it.color0_base; break; }
                if (tgt) { RttSurf& s = g_rtt[tgt]; s.rgba = px; s.w = w; s.h = h; }
                if (getenv("PROSPER_RTTLOG")) { size_t nz=0; for(size_t i=0;i<px.size();i++) nz+=(px[i]!=0);
                    fprintf(stderr, "[rtt] store target=0x%llx (%zu items, color0s:", (unsigned long long)tgt, items.size());
                    for (const auto& it : items) fprintf(stderr, " 0x%llx", (unsigned long long)it.color0_base);
                    fprintf(stderr, ") px_nonzero=%zu cache_size=%zu\n", nz, g_rtt.size()); }
            }
            int n = frame_no++;
            if (px.empty()) {
                fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
            } else if (dump_bmps && (n < 60 || n % 10 == 0)) {   // periodic screenshots (headless verification only)
                char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
                prosper::test::dump_bmp(fn, px, w, h);
                fprintf(stderr, "[render] frame %d rendered (%ux%u) -> %s\n", n, w, h, fn);
            }
            return px;
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend

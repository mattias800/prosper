// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "live_renderer.hpp"

#include "gpu/gpu_execute.hpp"          // DrawItem, set_submit_renderer
#include "gpu/tile.hpp"                 // detile_surface / tiled_surface_bytes / detile_elements
#include "gpu/bc_decode.hpp"            // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/shader_resources.hpp"     // ShaderResourceTable / ResourceClass
#include "gpu/rdna2_to_spirv.hpp"       // recompile_fragment (diagnostic solid-color PS)
#include "gpu/videoout_present.hpp"     // present_front_index (flip-anchored present selection)
#include "render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <set>
#include <unordered_map>

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);
// VideoOut scanout registry (hle_graphics.cpp) — which guest buffer the game most recently FLIPPED
// to screen. The flip fires during the Dcb fold (agc_dcb_set_flip -> prosper_vo_flip_from_gpu),
// BEFORE the submit's execute_and_present, so at render time these identify this frame's scanout VA.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint64_t prosper_vo_buffer_addr(int i);

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
    // PROSPER_RTT_PERTARGET renders each distinct CB_COLOR0_BASE into its OWN framebuffer (below) and
    // implies RTT injection — a group can only sample an earlier group if the injection path is live.
    static const bool pertarget = getenv("PROSPER_RTT_PERTARGET") != nullptr;
    static const bool rtt_on = getenv("PROSPER_RTT") != nullptr || pertarget;
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
                        // Real bytes-per-texel of the SAMPLED surface. Single/dual-channel textures (an R8
                        // font/coverage atlas — this game's 2048x1024 c1 surface) are NOT 4 B/texel; reading
                        // them as RGBA8 packs adjacent texels into one pixel and over-reads the allocation,
                        // which is why glyph text rendered as solid white/black BLOCKS (#102). bpt drives a
                        // narrow read+expand path below. StorageImage / unknown formats keep 4 B (bpt=0->4).
                        uint32_t bpt = prosper::gpu::data_format_bytes(r.format) * (r.num_components ? r.num_components : 1);
                        // fp16 surfaces (fmt 71 Float16x4 = 8 B/texel, 29 = 4 B, 13 = 2 B) get a real
                        // half->UNORM8 conversion path below; everything else unknown/oversized keeps
                        // the legacy RGBA8 read (bpt=4).
                        const bool f16 = r.format == prosper::gpu::DataFormat::Float16 &&
                                         (bpt == 2 || bpt == 4 || bpt == 8);
                        if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                        bool narrow_done = false;
                        bool f16_done = false;
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
                        } else if (f16) {
                            // fp16 texture (#290 wall 1): read at the REAL bytes-per-texel and detile
                            // with the REAL element size — the old clamp read an 8-B/texel Float16x4
                            // surface at 4 B AND detiled it with bpe=4 against 8-B tiled elements, so
                            // the result was doubly wrong ("confetti" regions, e.g. DOLL's 960x540 /
                            // 480x270 bloom-chain buffers). Convert half->UNORM8 on upload: the backend
                            // uploads RGBA8 only (the whole render path is 8-bit gamma space — see the
                            // #263 note in render_runner), so HDR values above 1.0 clamp to 255.
                            // Missing components read (0,0,0,1) per the hardware rule; the T# DST_SEL
                            // swizzle still applies. CONFIDENCE: MED — the half decode is exact and
                            // unit-tested, but the [0,1] clamp loses >1.0 bloom energy (native
                            // VK_FORMAT_R16G16B16A16_SFLOAT upload is the documented follow-up).
                            const uint32_t nc = bpt / 2;                    // fp16 components per texel
                            std::vector<uint8_t> hlin((size_t)tw * th * bpt, 0);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                                std::vector<uint8_t> traw(tbytes, 0);
                                size_t got = safe_copy(traw.data(), r.gpu_addr, tbytes);
                                if (got < hlin.size()) safe_copy(hlin.data(), r.gpu_addr, hlin.size());  // short backing -> linear fallback
                                else prosper::gpu::detile_surface(hlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                safe_copy(hlin.data(), r.gpu_addr, hlin.size());
                            }
                            for (size_t t = 0; t < (size_t)tw * th; t++) {
                                uint8_t* p = &texstore.back()[t * 4];
                                for (uint32_t c = 0; c < 4; c++) {
                                    if (c < nc) {
                                        uint16_t hv; std::memcpy(&hv, &hlin[t * bpt + c * 2], 2);
                                        float f = prosper::gpu::half_to_float(hv);
                                        p[c] = (f != f || f <= 0.f) ? 0                     // NaN/neg -> 0
                                             : (f >= 1.f ? 255 : (uint8_t)(f * 255.f + 0.5f));
                                    } else p[c] = (c == 3) ? 255 : 0;       // absent: (.,0,0,1)
                                }
                            }
                            f16_done = true;   // read+detiled at the real element size already
                        } else if (bpt < 4) {
                            // Narrow (single/dual-channel) surface: read at the REAL element size and detile
                            // with the matching bpe geometry (1 B -> 64x64 micro-tiles, #119), then expand
                            // each texel to grayscale RGBA8 (replicate to R,G,B,A) so the sampling shader
                            // reads the coverage in ANY channel it uses (.r for a font atlas, .a for a mask).
                            std::vector<uint8_t> nlin((size_t)tw * th * bpt, 0);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                                std::vector<uint8_t> traw(tbytes, 0);
                                size_t got = safe_copy(traw.data(), r.gpu_addr, tbytes);
                                // PROSPER_DUMP_RAWTILE (narrow path): the single/dual-channel RAW TILED bytes,
                                // once per address, for offline 8-bpp de-swizzle sweeps (the SDF font atlas).
                                if (getenv("PROSPER_DUMP_RAWTILE") && got >= nlin.size() && tw <= 2048 && th <= 1024) {
                                    static std::set<uint64_t> nseen;
                                    if (nseen.insert(r.gpu_addr).second) {
                                        std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                        char bn[512]; snprintf(bn, sizeof bn, "%s/narrowtile_%ux%u_b%u_%llx.bin",
                                                               dd.c_str(), tw, th, bpt, (unsigned long long)r.gpu_addr);
                                        if (FILE* bf = fopen(bn, "wb")) { fwrite(traw.data(), 1, traw.size(), bf); fclose(bf);
                                            fprintf(stderr, "[render] narrow raw tiled -> %s (%zu, bpt=%u)\n", bn, traw.size(), bpt); fflush(stderr); }
                                    }
                                }
                                if (got < nlin.size()) safe_copy(nlin.data(), r.gpu_addr, nlin.size());  // short backing -> linear fallback
                                else prosper::gpu::detile_surface(nlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                safe_copy(nlin.data(), r.gpu_addr, nlin.size());
                            }
                            for (size_t t = 0; t < (size_t)tw * th; t++) {
                                uint8_t v = nlin[t * bpt];   // first (coverage) channel
                                uint8_t* p = &texstore.back()[t * 4];
                                p[0] = p[1] = p[2] = p[3] = v;
                            }
                            narrow_done = true;   // already detiled+expanded; skip the 32-bpp auto-detile below
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
                        if (!rtt_hit && !bcb && !narrow_done && !f16_done && !getenv("PROSPER_NODETILE") && (auto_tiled || (dt && atoi(dt) != 0))) {
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
                            if (getenv("PROSPER_DUMP_RAWTILE") && (frame_no < 200 || (tw <= 2048 && th <= 1024))) {
                                // Early frames by binding, PLUS small textures once per address (the font/UI
                                // atlas can be sampled late — capture whenever first seen).
                                static std::set<uint64_t> rawseen;
                                bool small = tw <= 2048 && th <= 1024;
                                if (!small || rawseen.insert(r.gpu_addr).second) {
                                    std::string dd = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                    char bn[512];
                                    if (small) snprintf(bn, sizeof bn, "%s/rawtile_%ux%u_%llx.bin", dd.c_str(), tw, th, (unsigned long long)r.gpu_addr);
                                    else snprintf(bn, sizeof bn, "%s/tiled_f%04d_b%u_%ux%u.bin", dd.c_str(), (int)frame_no, r.binding, tw, th);
                                    if (FILE* bf = fopen(bn, "wb")) { fwrite(tiled.data(), 1, tiled.size(), bf); fclose(bf); }
                                }
                            }
                            prosper::gpu::detile_surface(texstore.back().data(), tiled.data(), tw, th, tmode, pitch);
                        }
                        // Packed R11G11B10F (Gen5 IMG_FMT 36 -> DataFormat::Float10_11_11, #294): UE4's
                        // scene-color RT format. The texel IS 4 bytes, so the generic read + auto-detile
                        // above already produced linear packed words; unpack each to RGBA8 in place with
                        // the same semantics as the fp16 path (NaN/neg -> 0, clamp [0,1], no alpha -> 255).
                        if (!rtt_hit && r.format == prosper::gpu::DataFormat::Float10_11_11) {
                            uint8_t* tp = texstore.back().data();
                            for (size_t t = 0; t < (size_t)tw * th; t++) {
                                uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                                const float fc[3] = { prosper::gpu::f11_to_float((uint16_t)(v & 0x7FFu)),
                                                      prosper::gpu::f11_to_float((uint16_t)((v >> 11) & 0x7FFu)),
                                                      prosper::gpu::f10_to_float((uint16_t)((v >> 22) & 0x3FFu)) };
                                for (int c = 0; c < 3; c++)
                                    tp[t * 4 + c] = (fc[c] != fc[c] || fc[c] <= 0.f) ? 0
                                                  : (fc[c] >= 1.f ? 255 : (uint8_t)(fc[c] * 255.f + 0.5f));
                                tp[t * 4 + 3] = 255;
                            }
                        }
                        // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                        // bypassing the shader — reveals whether the render target is tiled or linear.
                        if (getenv("PROSPER_DUMP_TEX") && !texstore.back().empty() && frame_no < 200) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                            prosper::test::dump_bmp(fn, texstore.back(), tw, th);
                            fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                        }
                        // PROSPER_DUMP_ATLAS: dump each SMALL sampled texture once per ADDRESS (find the caption
                        // font among same-size UI textures). Capped.
                        if (getenv("PROSPER_DUMP_ATLAS") && !texstore.back().empty() && tw <= 2048 && th <= 1024) {
                            static std::unordered_map<uint64_t,int> seen; static int ndumped = 0;
                            if (seen[r.gpu_addr]++ == 0 && ndumped++ < 60) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/tex_%ux%u_%llx_c%u.bmp", d.c_str(), tw, th,
                                                       (unsigned long long)r.gpu_addr, r.num_components);
                                prosper::test::dump_bmp(fn, texstore.back(), tw, th);
                            }
                        }
                        fr.tex_rgba = texstore.back().data(); fr.tw = tw; fr.th = th;
                        // Carry the decoded S# sampler state (filter/wrap/mip) so the pipeline samples the
                        // way the game asked instead of a fixed LINEAR/clamp sampler (#<sampler-fix>).
                        fr.mag_filter = r.mag_filter; fr.min_filter = r.min_filter; fr.mip_filter = r.mip_filter;
                        fr.addr_uvw[0] = r.addr_uvw[0]; fr.addr_uvw[1] = r.addr_uvw[1]; fr.addr_uvw[2] = r.addr_uvw[2];
                        // Remaining S# sampler fields (#262): border color + LOD clamp/bias (applied where
                        // valid; the decode-only aniso/compare/unnorm stay on ShaderResource).
                        fr.border_color_type = r.border_color_type;
                        fr.min_lod = r.min_lod; fr.max_lod = r.max_lod; fr.lod_bias = r.lod_bias;
                        // T# DST_SEL channel remap (#261): apply only to REAL-RGBA texels. The narrow
                        // R->RGBA replication path (narrow_done) already broadcasts coverage to every
                        // channel — keep it identity so the font/mask path (#102/#256) is untouched.
                        if (narrow_done) { fr.swizzle[0]=4; fr.swizzle[1]=5; fr.swizzle[2]=6; fr.swizzle[3]=7; }
                        else { for (int k=0;k<4;k++) fr.swizzle[k] = r.swizzle[k]; }
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
            // Assemble backend draws for a subset of the submit's items — one BackendDraw per realized
            // DrawItem with its own resources + fixed-function state (or the diagnostic overrides above).
            // build_R reads the CURRENT g_rtt, so calling this AFTER an earlier target-group has been
            // rendered+stored lets the later group sample that group's pixels (a HIT, not empty memory).
            auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group) {
                std::vector<prosper::test::BackendDraw> bds;
                for (const auto* itp : group) {
                    const auto& it = *itp;
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
                return bds;
            };
            std::vector<uint8_t> px;
            if (pertarget) {
                // PER-TARGET RTT: a real frame is a sequence of passes, each rendering into a specific
                // color target (CB_COLOR0_BASE), and a final composite pass SAMPLES the earlier targets.
                // The single-framebuffer path below flattens ALL draws into one image and caches it under
                // only the FIRST item's base — so a draw that targets a different base never gets cached
                // under its OWN address, and a later composite that samples that address misses -> black.
                // Here we group items by color0_base (first-appearance order), render each group into its
                // own framebuffer, and cache each under its base. Because groups render in order and
                // build_bds re-reads g_rtt, a composite group that samples a scene group rendered earlier
                // THIS submit now hits — closing the intra-submit multi-pass gap (and cross-submit too,
                // since g_rtt persists). The final (last) group's pixels — the composite — are presented.
                std::vector<uint64_t> order;                                            // distinct bases, in order
                std::unordered_map<uint64_t, std::vector<const prosper::gpu::DrawItem*>> groups;
                for (const auto& it : items) {
                    uint64_t k = it.color0_base;                                        // 0 => no declared target
                    if (groups.find(k) == groups.end()) order.push_back(k);
                    groups[k].push_back(&it);
                }
                // Flip-anchored present selection: the correct frame is whatever the guest FLIPS to
                // screen, i.e. the RTT group whose color-target VA is a registered VideoOut buffer
                // (preferring the CURRENT front buffer — the in-stream SetFlip fires during the Dcb
                // fold, before this render, so present_front_index() is this frame's scanout choice).
                const int      vo_n     = prosper_vo_buffer_count();
                const int      vo_front = prosper::gpu::present_front_index();
                const uint64_t front_va = vo_front >= 0 ? prosper_vo_buffer_addr(vo_front) : 0;
                if (getenv("PROSPER_RTTLOG")) {
                    fprintf(stderr, "[rtt] flip state: front=%d va=0x%llx of %d registered:",
                            vo_front, (unsigned long long)front_va, vo_n);
                    for (int i = 0; i < vo_n && i < 8; i++)
                        fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
                    fprintf(stderr, "\n");
                }
                std::vector<uint8_t> px_front, px_vo;   // flip-matched / any-scanout-matched candidates
                // Render-target persistence (default on; PROSPER_RTT_NOSEED reverts to per-pass blue
                // clear): seed each group's framebuffer with the pixels last rendered into that SAME
                // target VA, so a pass that draws into an already-written target (UE4's UI pass onto
                // the backbuffer, incremental HUD updates) composites OVER the earlier content instead
                // of starting from the diagnostic clear. Real RT memory persists exactly this way.
                static const bool seed_rtt = getenv("PROSPER_RTT_NOSEED") == nullptr;
                for (uint64_t base : order) {
                    const uint8_t* seed = nullptr;
                    if (seed_rtt && base) { auto sit = g_rtt.find(base);
                        if (sit != g_rtt.end() && sit->second.w == w && sit->second.h == h &&
                            sit->second.rgba.size() == (size_t)w * h * 4) seed = sit->second.rgba.data(); }
                    std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(build_bds(groups[base]), w, h, seed);
                    if (base && !gpx.empty()) { RttSurf& s = g_rtt[base]; s.rgba = gpx; s.w = w; s.h = h; }
                    bool is_vo = false;
                    for (int i = 0; i < vo_n && !is_vo; i++) is_vo = base && base == prosper_vo_buffer_addr(i);
                    if (getenv("PROSPER_RTTLOG")) { size_t nz=0; for (uint8_t b : gpx) nz += (b != 0);
                        fprintf(stderr, "[rtt] group target=0x%llx (%zu draws) px_nonzero=%zu cache_size=%zu%s%s\n",
                                (unsigned long long)base, groups[base].size(), nz, g_rtt.size(),
                                is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : ""); }
                    // PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes>: dump each per-target group's rendered
                    // pixels (rtgrp_<base>_<frame>.bmp in PROSPER_FRAME_DIR) — to inspect an intermediate
                    // pass (e.g. the UI/banner RT) instead of only the presented composite. Diagnostic.
                    if (const char* rg = getenv("PROSPER_DUMP_RTGROUPS"); rg && !gpx.empty()) {
                        size_t nz = 0; for (uint8_t b : gpx) nz += (b != 0);
                        if (nz >= (size_t)atol(rg)) {
                            const char* dd = getenv("PROSPER_FRAME_DIR");
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rtgrp_%llx_%04d.bmp",
                                                   dd ? dd : ".", (unsigned long long)base, frame_no.load());
                            prosper::test::dump_bmp(fn, gpx, w, h);
                        }
                    }
                    if (!gpx.empty()) {
                        if (base && base == front_va) px_front = gpx;                   // the flipped buffer
                        if (is_vo)                    px_vo    = gpx;                   // any registered scanout
                        px = std::move(gpx);                                            // last non-empty (fallback)
                    }
                }
                // Present priority: the flipped front buffer > any registered scanout target > the
                // legacy "last group" fallback (unchanged behavior when no group targets a VO buffer).
                if (!px_front.empty())   px = std::move(px_front);
                else if (!px_vo.empty()) px = std::move(px_vo);
            } else {
                // Single-framebuffer path: render_draws_rgba composites every draw into ONE framebuffer.
                std::vector<const prosper::gpu::DrawItem*> all; all.reserve(items.size());
                for (const auto& it : items) all.push_back(&it);
                px = prosper::test::render_draws_rgba(build_bds(all), w, h);
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
            }
            int n = frame_no++;
            // PROSPER_DUMP_CONTENT=<min-nonzero-bytes>: dump ONLY frames whose framebuffer has at least
            // that many nonzero bytes — catches the intermittent content submits the periodic dump misses.
            size_t content_thr = 0; if (const char* c = getenv("PROSPER_DUMP_CONTENT")) content_thr = (size_t)atol(c);
            size_t px_nz = 0; if (content_thr) for (size_t i = 0; i < px.size(); i++) px_nz += (px[i] != 0);
            if (px.empty()) {
                fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
            } else if (dump_bmps && ((content_thr && px_nz >= content_thr) || (!content_thr && (n < 60 || n % 10 == 0)))) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
                prosper::test::dump_bmp(fn, px, w, h);
                fprintf(stderr, "[render] frame %d rendered (%ux%u) nz=%zu -> %s\n", n, w, h, px_nz, fn);
            }
            return px;
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend

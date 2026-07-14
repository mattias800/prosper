// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "live_renderer.hpp"
#include "live_compute.hpp"

#include "gpu/gpu_execute.hpp"          // DrawItem, set_submit_renderer
#include "gpu/writer_provenance.hpp"
#include "gpu/gpu_capture.hpp"          // temporal RTT capture/replay seeds
#include "gpu/tile.hpp"                 // detile_surface / tiled_surface_bytes / detile_elements
#include "gpu/bc_decode.hpp"            // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/shader_resources.hpp"     // ShaderResourceTable / ResourceClass
#include "gpu/rdna2_to_spirv.hpp"       // recompile_fragment (diagnostic solid-color PS)
#include "gpu/videoout_present.hpp"     // present_front_index (flip-anchored present selection)
#include "render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp

#include <atomic>
#include <chrono>
#include <climits>
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
namespace {
struct RttSurf { std::vector<uint8_t> rgba; uint32_t w = 0, h = 0; };

// Pixel decoding is a pure function of these fields for guest-backed textures. Keep this cache local
// to one renderer callback: graphics spans are split at compute operations, so guest texture bytes
// cannot change within its lifetime. Live RTT inputs are excluded separately because an earlier pass
// in the same callback can replace their pixels.
struct TextureDecodeKey {
    uint64_t gpu_addr = 0;
    uint64_t host_data = 0;
    uint64_t host_data_size = 0;
    uint32_t size = 0;
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0, height = 0;
    uint32_t tile_mode = 0;
    uint32_t img_dim = 0;
    bool operator==(const TextureDecodeKey&) const = default;
};

struct TextureDecodeKeyHash {
    size_t operator()(const TextureDecodeKey& key) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t value) {
            hash ^= static_cast<size_t>(value);
            hash *= 1099511628211ull;
            hash ^= static_cast<size_t>(value >> 32);
            hash *= 1099511628211ull;
        };
        mix(key.gpu_addr); mix(key.host_data); mix(key.host_data_size); mix(key.size);
        mix(key.cls); mix(key.format); mix(key.num_components); mix(key.width); mix(key.height);
        mix(key.tile_mode); mix(key.img_dim);
        return hash;
    }
};

struct DecodedTexture {
    const uint8_t* pixels = nullptr;
    uint32_t output_height = 0;
    bool narrow = false;
};
}

void register_live_renderer(const std::string& frame_dir, bool dump_bmps) {
    register_live_compute();
    const char* ds_invalidate = getenv("PROSPER_DS_GUEST_WRITE_INVALIDATE");
    if (!ds_invalidate || strcmp(ds_invalidate, "0"))
        prosper::gpu::set_guest_gpu_write_observer(
            [](uint64_t addr, uint64_t size) {
                prosper::test::invalidate_persistent_ds_guest_write(addr, size);
            });
    else
        prosper::gpu::set_guest_gpu_write_observer({});
    // Resource tables are built before the submit reaches this callback. Publish the renderer's
    // default mode now so unmapped render-target descriptors remain available for RTT injection.
    // Outside a registered renderer, resource decoding retains its strict unknown-format policy.
    if (!getenv("PROSPER_RTT_SINGLE_TARGET") && !getenv("PROSPER_RTT_PERTARGET")) {
#ifdef _WIN32
        _putenv_s("PROSPER_RTT_PERTARGET", "1");
#else
        setenv("PROSPER_RTT_PERTARGET", "1", 1);
#endif
    }
    static std::atomic<int> frame_no{0};
    static std::unordered_map<uint64_t, RttSurf> g_rtt;   // render-to-texture cache (#167)
    if (getenv("PROSPER_GPU_CAPTURE") || getenv("PROSPER_GPU_TIMELINE_CAPTURE") ||
        getenv("PROSPER_GPU_REPLAY_EXPORT_RTT")) {
        prosper::gpu::set_gpu_capture_rtt_seed_reader([](uint64_t addr, prosper::gpu::GpuCaptureRttSeed& seed) {
            auto it = g_rtt.find(addr); if (it == g_rtt.end()) return false;
            seed.guest_addr = addr; seed.width = it->second.w; seed.height = it->second.h;
            seed.rgba = it->second.rgba; return true;
        });
        prosper::gpu::set_gpu_capture_rtt_seed_snapshot_reader(
            [](std::vector<prosper::gpu::GpuCaptureRttSeed>& seeds, std::string&) {
                seeds.reserve(g_rtt.size());
                for (const auto& [addr, surface] : g_rtt) {
                    prosper::gpu::GpuCaptureRttSeed seed;
                    seed.guest_addr = addr; seed.width = surface.w; seed.height = surface.h;
                    seed.rgba = surface.rgba; seeds.push_back(std::move(seed));
                }
                return true;
            });
    }
    // The compute backend must not sample a surface whose CURRENT pixels live in this renderer's
    // RTT cache (raw guest memory is then empty/stale — the Dead Cells 642x362 lesson): publish the
    // exact-match target query it skips on (#590). Same keying as the RTT injection below.
    prosper::gpu::set_live_target_query([](uint64_t addr) { return g_rtt.count(addr) != 0; });
    if (getenv("PROSPER_GPU_CAPTURE") || getenv("PROSPER_GPU_TIMELINE_CAPTURE") ||
        getenv("PROSPER_GPU_REPLAY_EXPORT_DS"))
        prosper::gpu::set_gpu_capture_ds_seed_snapshot_reader(
            [](std::vector<prosper::gpu::GpuCaptureDsSeed>& seeds, std::string& error) {
                return prosper::test::snapshot_persistent_ds_images(seeds, error);
            });
    if (getenv("PROSPER_GPU_REPLAY_RTT_SEEDS"))
        prosper::gpu::set_gpu_replay_rtt_seed_writer([](const prosper::gpu::GpuCaptureRttSeed& seed, std::string& error) {
            const uint64_t expected = static_cast<uint64_t>(seed.width) * seed.height * 4;
            if (!seed.guest_addr || !seed.width || !seed.height || expected != seed.rgba.size()) {
                error = "invalid temporal RTT seed"; return false;
            }
            RttSurf& surface = g_rtt[seed.guest_addr];
            surface.w = seed.width; surface.h = seed.height; surface.rgba = seed.rgba;
            return true;
        });
    if (getenv("PROSPER_GPU_REPLAY_DS_SEEDS"))
        prosper::gpu::set_gpu_replay_ds_seed_writer(
            [](const prosper::gpu::GpuCaptureDsSeed& seed, std::string& error) {
                return prosper::test::restore_persistent_ds_image(seed, error);
            });
    // A real command stream renders each CB_COLOR0_BASE into its own surface. Keep the old flattened
    // compositor only as a diagnostic fallback; it cannot preserve post chains or target extents.
    static const bool pertarget = getenv("PROSPER_RTT_PERTARGET") != nullptr ||
                                  getenv("PROSPER_RTT_SINGLE_TARGET") == nullptr;
    static const bool rtt_on = getenv("PROSPER_RTT") != nullptr || pertarget;
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps](const std::vector<prosper::gpu::DrawItem>& items, uint32_t w, uint32_t h) -> std::vector<uint8_t> {
            using RC = prosper::gpu::ResourceClass;
            const prosper::gpu::LiveRenderPhase phase = prosper::gpu::live_render_phase();
            // PROSPER_RENDER_FIRST=<N>: skip the slow (~400x) Vulkan render for the first N GPU submits, so
            // the game reaches a LATE scene (e.g. the level1 cutscene, which only starts submitting after
            // ~5000 title-loop submits) at native speed before we begin rendering/dumping. Returning {}
            // means "not rendered this submit". Without this, rendering from boot is far too slow to ever
            // reach a post-loading-screen scene.
            static std::atomic<int> g_submit_idx{0};
            static int g_render_first = getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0;
            // PROSPER_RENDER_DELAY_MS=<N>: wall-clock warmup for titles whose useful scene begins after
            // a variable number of submits. The guest and command decoder keep running at native speed;
            // only the synchronous Vulkan work is skipped. The clock starts at the first GPU submit.
            static const int64_t g_render_delay_ms = getenv("PROSPER_RENDER_DELAY_MS")
                ? std::max<int64_t>(0, atoll(getenv("PROSPER_RENDER_DELAY_MS"))) : 0;
            static const auto g_render_delay_start = std::chrono::steady_clock::now();
            static std::atomic<bool> g_render_delay_announced{false};
            // PROSPER_RENDER_LAST=<N>: stop rendering after submit N (default: unbounded). Bounds the render
            // window so a diagnostic slice at a late stall (RENDER_FIRST..RENDER_LAST) does not accumulate
            // unbounded RTT/GPU resources across tens of thousands of submits (which OOM-kills the process).
            static int g_render_last = getenv("PROSPER_RENDER_LAST") ? atoi(getenv("PROSPER_RENDER_LAST")) : INT_MAX;
            static thread_local int g_this_submit = -1;
            if (phase.first_span || g_this_submit < 0) g_this_submit = g_submit_idx++;
            if (g_this_submit > g_render_last) return {};
            using RenderClock = std::chrono::steady_clock;
            struct RenderTiming {
                uint64_t callbacks = 0;
                double total_ms = 0, build_resources_ms = 0, backend_ms = 0;
                uint64_t textures = 0, texture_reuses = 0, buffers = 0;
                uint64_t texture_bytes = 0, buffer_bytes = 0;
                double texture_ms = 0, buffer_ms = 0;
            };
            static thread_local RenderTiming pending_timing;
            const bool timing_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr;
            if (timing_enabled && phase.first_span) pending_timing = {};
            const auto callback_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            // PROSPER_SUBMITLOG: print the GPU-submit index periodically (at native speed, before the slow
            // render) so it can be correlated with guest-side log lines (e.g. a MsgDialog wait) to find the
            // exact submit at which a scene appears — for aiming PROSPER_RENDER_FIRST at it.
            if (phase.first_span && getenv("PROSPER_SUBMITLOG") && (g_this_submit % 1000 == 0))
                fprintf(stderr, "[submit] index=%d (%zu draw items)\n", g_this_submit, items.size());
            if (const char* sd = getenv("PROSPER_SUBMITLOG_DIM")) {
                uint32_t sw = 0, sh = 0;
                if (sscanf(sd, "%ux%u", &sw, &sh) == 2)
                    for (const auto& it : items)
                        if (it.color0_width == sw && it.color0_height == sh) {
                            fprintf(stderr, "[submit] index=%d target=0x%llx extent=%ux%u (%zu draw items)\n",
                                    g_this_submit, (unsigned long long)it.color0_base, sw, sh, items.size());
                            break;
                        }
            }
            bool force_target = false;
            if (const char* td = getenv("PROSPER_RENDER_TARGET_DIM")) {
                uint32_t tw = 0, th = 0;
                if (sscanf(td, "%ux%u", &tw, &th) == 2)
                    for (const auto& it : items)
                        if (it.color0_width == tw && it.color0_height == th) { force_target = true; break; }
            }
            if (const char* rd = getenv("PROSPER_RENDER_RESOURCE_DIM")) {
                uint32_t rw = 0, rh = 0;
                if (sscanf(rd, "%ux%u", &rw, &rh) == 2)
                    for (const auto& it : items) {
                        auto has_dim = [&](const prosper::gpu::ShaderResourceTable* table) {
                            if (!table) return false;
                            for (const auto& r : table->resources)
                                if (r.width == rw && r.height == rh) return true;
                            return false;
                        };
                        if (has_dim(it.vrt.get()) || has_dim(it.prt.get())) { force_target = true; break; }
                    }
            }
            // A one-time offscreen producer may occur before a much later consumer render window.
            // Render that target even before the warmup ends so its RTT cache entry survives skipped
            // intermediate submits (#526). Submit-count and wall-clock gates are additive.
            const int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - g_render_delay_start).count();
            const bool before_delay = elapsed_ms < g_render_delay_ms;
            if (!before_delay && g_render_delay_ms > 0 &&
                !g_render_delay_announced.exchange(true, std::memory_order_relaxed)) {
                fprintf(stderr, "[render] wall-clock warmup complete after %lld ms at submit %d\n",
                        (long long)elapsed_ms, g_this_submit);
            }
            if ((g_this_submit < g_render_first || before_delay) && !force_target) return {};
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
            // Keep decoded texture storage alive across callbacks. The old clear()+emplace(size, 0)
            // released and zero-filled tens of MiB every submit even though the decode paths overwrite
            // all pixels. Reusing same-sized slots avoids both costs; short guest reads explicitly clear
            // their uncovered tail below so stale scratch bytes can never become sampled pixels.
            static std::vector<std::vector<uint8_t>> texstore;
            size_t texstore_used = 0;
            std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash> decoded_textures;
            uint32_t resource_hash_w = 0, resource_hash_h = 0;
            if (const char* dim = getenv("PROSPER_RESOURCE_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &resource_hash_w, &resource_hash_h) != 2)
                    resource_hash_w = resource_hash_h = 0;
            uint32_t target_step_w = 0, target_step_h = 0;
            if (const char* dim = getenv("PROSPER_TARGET_STEP_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &target_step_w, &target_step_h) != 2)
                    target_step_w = target_step_h = 0;
            const size_t target_step_min_draws = getenv("PROSPER_TARGET_STEP_HASH_MIN_DRAWS")
                ? std::max<long>(2, atol(getenv("PROSPER_TARGET_STEP_HASH_MIN_DRAWS"))) : 2;
            // Build one draw's set-tagged resources from its VS (set 0) + PS (set 1) tables — read the
            // bytes from 1:1-mapped guest memory, detile textures. (Each constant/vertex buffer + texture
            // gets its own binding; the recompiler declared a storage buffer / image sampler at each.)
            auto build_R = [&](const prosper::gpu::DrawItem& draw,
                               const prosper::gpu::ShaderResourceTable* vrt,
                               const prosper::gpu::ShaderResourceTable* prt) {
              std::vector<prosper::test::FrameResource> R;
              auto add = [&](const prosper::gpu::ShaderResourceTable* t, uint32_t set){
                if (!t) return;
                for (auto& r : t->resources) {
                    const auto resource_timing_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    bool resource_rtt_hit = false;
                    prosper::test::FrameResource fr; fr.binding = r.binding; fr.set = set;
                    // Captured replays preserve the logical guest address for RT identity but provide
                    // owned bytes here. Production resources leave host_data null and use guest memory.
                    auto copy_resource = [&](uint8_t* dst, uint64_t addr, size_t n) -> size_t {
                        if (!r.host_data) return safe_copy(dst, addr, n);
                        if (addr < r.gpu_addr) return 0;
                        uint64_t off = addr - r.gpu_addr;
                        if (off >= r.host_data_size) return 0;
                        size_t take = static_cast<size_t>(std::min<uint64_t>(n, r.host_data_size - off));
                        std::memcpy(dst, r.host_data + off, take);
                        return take;
                    };
                    if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                        uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                        const TextureDecodeKey decode_key{
                            r.gpu_addr, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r.host_data)),
                            r.host_data_size, r.size, static_cast<uint32_t>(r.cls),
                            static_cast<uint32_t>(r.format), r.num_components, tw, th,
                            r.tile_mode, r.img_dim,
                        };
                        auto live_rtt = rtt_on ? g_rtt.find(r.gpu_addr) : g_rtt.end();
                        const bool has_live_rtt = live_rtt != g_rtt.end() && live_rtt->second.w &&
                            live_rtt->second.h && !live_rtt->second.rgba.empty();
                        bool narrow_done = false;
                        auto reused = has_live_rtt ? decoded_textures.end()
                                                   : decoded_textures.find(decode_key);
                        if (reused != decoded_textures.end()) {
                            fr.tex_rgba = reused->second.pixels;
                            fr.tw = tw;
                            fr.th = reused->second.output_height;
                            narrow_done = reused->second.narrow;
                            if (timing_enabled) pending_timing.texture_reuses++;
                        } else {
                        const bool is_cube = r.img_dim == 3u;   // CUBE: six faces stacked vertically (#273)
                        size_t nb = (size_t)tw * th * 4 * (is_cube ? 6u : 1u);
                        if (texstore_used == texstore.size()) texstore.emplace_back();
                        std::vector<uint8_t>& texture_pixels = texstore[texstore_used++];
                        texture_pixels.resize(nb);
                        // PROSPER_TEXCOMMIT: log, once per texture base, how much of the sampled surface is
                        // COMMITTED guest memory (the same reserved_range_state safe_copy stops at). If the
                        // level's backgrounds read ~0% committed, they're GPU-DMA'd pages the CPU never
                        // touched, so we read zeros -> the scene samples black (#300 black-gameplay probe).
                        if (getenv("PROSPER_TEXCOMMIT")) {
                            const size_t PG = 0x10000; size_t committed = 0;
                            for (uint64_t a = r.gpu_addr; a < r.gpu_addr + nb; a += PG)
                                if (a >= 0x1000 && prosper_reserved_range_state(a) != 0) committed += PG;
                            static std::set<uint64_t> tcseen;
                            if (tcseen.insert(r.gpu_addr).second) {
                                // Also sample the first 8 dwords and count non-zero bytes over the whole
                                // surface: zero content => the texture was allocated but never filled (a
                                // GPU-side upload/copy we don't execute); non-zero => it's a decode/tiling
                                // problem. tile_mode tells tiled vs linear.
                                uint32_t w0[8] = {0}; size_t nzb = 0;
                                if (committed) {
                                    const uint8_t* p = (const uint8_t*)(uintptr_t)r.gpu_addr;
                                    for (int i = 0; i < 8; i++) w0[i] = ((const uint32_t*)p)[i];
                                    for (size_t i = 0; i < nb; i += 997) nzb += (p[i] != 0);   // sparse scan
                                }
                                fprintf(stderr, "[texcommit] tex 0x%llx %ux%u f%u tile=%u nb=%zu committed=%zu%% "
                                        "nz~%zu/%zu first=%08x %08x %08x %08x\n",
                                        (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format, r.tile_mode, nb,
                                        nb ? (size_t)(100 * committed / nb) : 100, nzb, nb/997,
                                        w0[0], w0[1], w0[2], w0[3]);
                            }
                        }
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
                        bool f16_done = false;
                        // RTT (#167): if this texture's base is a color target we rendered into, inject those
                        // pixels (nearest-scaled to tw x th) instead of reading empty guest memory.
                        bool rtt_hit = false;
                        if (rtt_on) { auto rit = g_rtt.find(r.gpu_addr);
                            if (rit != g_rtt.end() && rit->second.w && rit->second.h && !rit->second.rgba.empty()) {
                                const RttSurf& s = rit->second;
                                const size_t expected = static_cast<size_t>(tw) * th * 4;
                                if (s.w == tw && s.h == th && s.rgba.size() == expected) {
                                    std::memcpy(texture_pixels.data(), s.rgba.data(), expected);
                                } else {
                                    std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                                    for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                        uint32_t sx = (uint32_t)((uint64_t)x * s.w / tw), sy = (uint32_t)((uint64_t)y * s.h / th);
                                        size_t si = ((size_t)sy * s.w + sx) * 4;
                                        if (si + 4 <= s.rgba.size()) std::memcpy(&texture_pixels[((size_t)y * tw + x) * 4], &s.rgba[si], 4);
                                    }
                                }
                                rtt_hit = true;
                                resource_rtt_hit = true;
                            }
                            if (getenv("PROSPER_RTTLOG"))
                                fprintf(stderr, "[rtt] sample tex addr=0x%llx %ux%u fmt=%u -> %s (cache_size=%zu)\n",
                                        (unsigned long long)r.gpu_addr, tw, th, (unsigned)r.format,
                                        rtt_hit ? "HIT" : "miss", g_rtt.size());
                        }
                        // CUBE texture (#273 — DOLL's title reflection probes / skybox): six faces,
                        // each an independently-tiled w x h surface, laid out face-major at
                        // gpu_addr + f*face_stride; decode each into rows [f*th, (f+1)*th) of the
                        // stacked w x 6h image the cube-sample lowering addresses. BCn faces block-
                        // detile + decode; anything else reads 4 B/texel with the surface detiler.
                        // CONFIDENCE: MED on the face stride (padded tiled footprint) — visually
                        // validated; a wrong stride garbles faces 1..5, it cannot crash (safe_copy).
                        bool cube_done = false;
                        if (is_cube && !rtt_hit) {
                            const uint32_t cb = prosper::gpu::bc_block_bytes(r.format);
                            const bool ctiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            for (uint32_t fface = 0; fface < 6; fface++) {
                                uint8_t* slice = texture_pixels.data() + (size_t)fface * tw * th * 4;
                                if (cb) {
                                    uint32_t bw = (tw + 3) / 4, bh = (th + 3) / 4;
                                    size_t comp = (size_t)bw * bh * cb;
                                    size_t stride = ctiled ? prosper::gpu::tiled_elements_bytes(bw, bh, cb, r.tile_mode) : comp;
                                    std::vector<uint8_t> lin(comp, 0);
                                    if (ctiled) {
                                        std::vector<uint8_t> traw(stride, 0);
                                        copy_resource(traw.data(), r.gpu_addr + (uint64_t)fface * stride, stride);
                                        prosper::gpu::detile_elements(lin.data(), traw.data(), stride, bw, bh, cb, r.tile_mode);
                                    } else copy_resource(lin.data(), r.gpu_addr + (uint64_t)fface * stride, comp);
                                    std::vector<uint8_t> face((size_t)tw * th * 4, 0);
                                    prosper::gpu::bc_decode_surface(face.data(), lin.data(), lin.size(), tw, th, r.format);
                                    std::memcpy(slice, face.data(), face.size());
                                } else {
                                    size_t fb = (size_t)tw * th * 4;
                                    size_t stride = ctiled ? prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0) : fb;
                                    if (ctiled) {
                                        std::vector<uint8_t> traw(stride, 0);
                                        copy_resource(traw.data(), r.gpu_addr + (uint64_t)fface * stride, stride);
                                        prosper::gpu::detile_surface(slice, traw.data(), tw, th, r.tile_mode, 0);
                                    } else {
                                        const size_t got = copy_resource(
                                            slice, r.gpu_addr + (uint64_t)fface * stride, fb);
                                        if (got < fb) std::fill(slice + got, slice + fb, 0);
                                    }
                                }
                            }
                            cube_done = true;
                        }
                        // Block-compressed (BC1/2/3): read the (possibly tiled) compressed blocks, block-
                        // detile, and decode to RGBA8 in-place. The blocks are the tiled element (BC3 =
                        // 16 bytes -> SW_4KB_S 16x16-block micro-tiles). #121.
                        const uint32_t bcb = (rtt_hit || cube_done) ? 0u : prosper::gpu::bc_block_bytes(r.format);
                        if (rtt_hit || cube_done) { /* pixels already injected/stacked above */ }
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
                                copy_resource(traw.data(), r.gpu_addr, tbytes);
                                prosper::gpu::detile_elements(lin.data(), traw.data(), tbytes, bw, bh, bcb, r.tile_mode);
                            } else {
                                copy_resource(lin.data(), r.gpu_addr, comp_bytes);
                            }
                            if (!prosper::gpu::bc_decode_surface(
                                    texture_pixels.data(), lin.data(), lin.size(), tw, th, r.format))
                                std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
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
                                size_t got = copy_resource(traw.data(), r.gpu_addr, tbytes);
                                if (got < hlin.size()) copy_resource(hlin.data(), r.gpu_addr, hlin.size());  // short backing -> linear fallback
                                else prosper::gpu::detile_surface(hlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                copy_resource(hlin.data(), r.gpu_addr, hlin.size());
                            }
                            for (size_t t = 0; t < (size_t)tw * th; t++) {
                                uint8_t* p = &texture_pixels[t * 4];
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
                                size_t got = copy_resource(traw.data(), r.gpu_addr, tbytes);
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
                                if (got < nlin.size()) copy_resource(nlin.data(), r.gpu_addr, nlin.size());  // short backing -> linear fallback
                                else prosper::gpu::detile_surface(nlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                copy_resource(nlin.data(), r.gpu_addr, nlin.size());
                            }
                            for (size_t t = 0; t < (size_t)tw * th; t++) {
                                uint8_t v = nlin[t * bpt];   // first (coverage) channel
                                uint8_t* p = &texture_pixels[t * 4];
                                p[0] = p[1] = p[2] = p[3] = v;
                            }
                            narrow_done = true;   // already detiled+expanded; skip the 32-bpp auto-detile below
                        } else {
                            const size_t got = copy_resource(texture_pixels.data(), r.gpu_addr, nb);
                            if (got < nb)
                                std::fill(texture_pixels.begin() + got, texture_pixels.end(), 0);
                        }
                        // PROSPER_DUMP_RAWTEX: write the raw tiled RGBA bytes (pre-detile) to a .bin for
                        // offline swizzle experimentation.
                        if (getenv("PROSPER_DUMP_RAWTEX") && !texture_pixels.empty()) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_%ux%u.bin", d.c_str(), tw, th);
                            if (FILE* f = fopen(fn, "wb")) { fwrite(texture_pixels.data(), 1, nb, f); fclose(f);
                                fprintf(stderr, "[render] dumped raw tiled bytes -> %s (%zu)\n", fn, nb); fflush(stderr); }
                        }
                        if (getenv("PROSPER_GFXLOG")) { const uint8_t* b = texture_pixels.data();
                            size_t nz = 0; for (size_t i = 0; i < nb && i < (1u<<16); i++) nz += (b[i] != 0);
                            fprintf(stderr, "[render] tex binding=%u %ux%u first64k-nonzero=%zu\n", r.binding, tw, th, nz); }
                        // AUTO-DETILE: de-swizzle a GPU-tiled sampled surface into the linear texstore,
                        // driven by the T# tile_mode threaded through the resource table (r.tile_mode).
                        bool auto_tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                        const char* dt = getenv("PROSPER_DETILE");
                        // BC textures already block-detiled + decoded above; the 32-bpp detiler must not touch them.
                        if (!rtt_hit && !cube_done && !bcb && !narrow_done && !f16_done && !getenv("PROSPER_NODETILE") && (auto_tiled || (dt && atoi(dt) != 0))) {
                            const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                            const uint32_t pitch = getenv("PROSPER_PITCH") ? (uint32_t)atoi(getenv("PROSPER_PITCH")) : 0;
                            size_t tiled_bytes = prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
                            std::vector<uint8_t> tiled(tiled_bytes, 0);
                            size_t got = copy_resource(tiled.data(), r.gpu_addr, tiled_bytes);
                            if (got < nb)
                                // The padded tiled buffer's tail (th rounded to whole 32-row tiles) runs past
                                // the real backing: fall back to the width*height linear bytes copied above
                                // rather than an all-zero buffer (which would BLANK the texture).
                                std::memcpy(tiled.data(), texture_pixels.data(), std::min(nb, tiled_bytes));
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
                            prosper::gpu::detile_surface(texture_pixels.data(), tiled.data(), tw, th, tmode, pitch);
                        }
                        // Packed R11G11B10F (Gen5 IMG_FMT 36 -> DataFormat::Float10_11_11, #294): UE4's
                        // scene-color RT format. The texel IS 4 bytes, so the generic read + auto-detile
                        // above already produced linear packed words; unpack each to RGBA8 in place with
                        // the same semantics as the fp16 path (NaN/neg -> 0, clamp [0,1], no alpha -> 255).
                        if (!rtt_hit && r.format == prosper::gpu::DataFormat::Float10_11_11) {
                            uint8_t* tp = texture_pixels.data();
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
                        // Focused per-consumer version probe (#586). Hash both the raw guest backing
                        // and the final decoded/RTT-injected pixels so a live draw identifies whether
                        // divergence precedes format conversion or enters through renderer-owned state.
                        if (resource_hash_w == tw && resource_hash_h == th) {
                            const size_t raw_size = std::min<size_t>(
                                r.size ? r.size : (size_t)tw * th * 4, 64u << 20);
                            std::vector<uint8_t> raw(raw_size, 0);
                            const size_t raw_got = copy_resource(raw.data(), r.gpu_addr, raw.size());
                            auto fnv = [](const uint8_t* data, size_t size) {
                                uint64_t hash = 1469598103934665603ull;
                                for (size_t i = 0; i < size; ++i) {
                                    hash ^= data[i];
                                    hash *= 1099511628211ull;
                                }
                                return hash;
                            };
                            const uint64_t raw_hash = fnv(raw.data(), raw_got);
                            const uint64_t sample_hash = fnv(texture_pixels.data(), texture_pixels.size());
                            const auto writer = prosper::gpu::last_guest_write_overlap(
                                r.gpu_addr, raw_size);
                            fprintf(stderr,
                                    "[resource-version] render-submit=%llu draw=%llu order=%llu set=%u bind=%u "
                                    "addr=0x%llx dims=%ux%u class=%u fmt=%u tile=%u rtt=%d "
                                    "raw=%zu/%zu:%016llx sample=%zu:%016llx "
                                    "writer=%s/%llu/%llu/%llu/0x%llx\n",
                                    (unsigned long long)g_this_submit,
                                    (unsigned long long)draw.draw_index,
                                    (unsigned long long)draw.command_order,
                                    set, r.binding, (unsigned long long)r.gpu_addr, tw, th,
                                    (unsigned)r.cls, (unsigned)r.format, r.tile_mode, (int)rtt_hit,
                                    raw_got, raw_size, (unsigned long long)raw_hash,
                                    texture_pixels.size(), (unsigned long long)sample_hash,
                                    writer ? prosper::gpu::guest_writer_kind_name(writer->kind) : "none",
                                    (unsigned long long)(writer ? writer->submit : 0),
                                    (unsigned long long)(writer ? writer->item : 0),
                                    (unsigned long long)(writer ? writer->order : 0),
                                    (unsigned long long)(writer ? writer->identity : 0));
                        }
                        // PROSPER_PALETTELOG: compact identity/provenance trace for Unity's 256x16
                        // palette textures. Unlike GFXLOG this is cheap enough for a focused render
                        // window and reveals both descriptor-address and decoded-content changes.
                        if (getenv("PROSPER_PALETTELOG") && tw == 256 && th == 16) {
                            uint64_t hash = 1469598103934665603ull;
                            size_t rgb_nonblack = 0;
                            for (size_t t = 0; t < (size_t)tw * th; ++t) {
                                const uint8_t* p = &texture_pixels[t * 4];
                                rgb_nonblack += (p[0] != 0 || p[1] != 0 || p[2] != 0);
                                for (unsigned c = 0; c < 4; ++c) {
                                    hash ^= p[c]; hash *= 1099511628211ull;
                                }
                            }
                            fprintf(stderr, "[palette] binding=%u addr=0x%llx fnv=%016llx rgb_nonblack=%zu\n",
                                    r.binding, (unsigned long long)r.gpu_addr,
                                    (unsigned long long)hash, rgb_nonblack);
                        }
                        // Apply the synthetic texture after every decode/conversion step. Applying it before
                        // auto-detile let the real tiled bytes overwrite the checker, producing a false-negative
                        // sampling diagnosis (#522).
                        if (getenv("PROSPER_TESTTEX")) {
                            for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                                bool ck = ((x / 64) ^ (y / 64)) & 1;
                                p[0] = (uint8_t)(255 * x / tw); p[1] = (uint8_t)(255 * y / th);
                                p[2] = ck ? 200 : 40; p[3] = 255;
                            }
                        }
                        // This 256x16 sparse palette is addressed like 16 blue slices, each 16 texels wide.
                        // The identity probe preserves source color through shaders using
                        // u=r/17+b*15/16, v=1-g, distinguishing lookup contents from a broken
                        // source/geometry/sample path (#522).
                        if (getenv("PROSPER_TESTLUT") && tw == 256 && th == 16) {
                            for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                                p[0] = (uint8_t)((x % 16) * 255 / 15);
                                p[1] = (uint8_t)((15 - y) * 255 / 15);
                                p[2] = (uint8_t)((x / 16) * 255 / 15);
                                p[3] = 255;
                            }
                        }
                        // Unity's post-processing stack flattens a 32^3 grading LUT into a 1024x32
                        // strip (32 red samples per blue slice). This isolates a missing LUT producer
                        // from the persistent post shader and its healthy scene input (#522).
                        if (getenv("PROSPER_TESTLUT32") && tw == 1024 && th == 32) {
                            for (uint32_t y = 0; y < th; ++y) for (uint32_t x = 0; x < tw; ++x) {
                                uint8_t* p = &texture_pixels[((size_t)y * tw + x) * 4];
                                p[0] = (uint8_t)((x % 32) * 255 / 31);
                                p[1] = (uint8_t)(y * 255 / 31);
                                p[2] = (uint8_t)((x / 32) * 255 / 31);
                                p[3] = 255;
                            }
                        }
                        // PROSPER_DUMP_TEX: write the RAW texture memory (interpreted linearly) to a BMP,
                        // bypassing the shader — reveals whether the render target is tiled or linear.
                        if (getenv("PROSPER_DUMP_TEX") && !texture_pixels.empty() && frame_no < 200) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                            prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                        }
                        // PROSPER_DUMP_ATLAS: dump each SMALL sampled texture once per ADDRESS (find the caption
                        // font among same-size UI textures). Capped.
                        if (getenv("PROSPER_DUMP_ATLAS") && !texture_pixels.empty() && tw <= 2048 && th <= 1024) {
                            static std::unordered_map<uint64_t,int> seen; static int ndumped = 0;
                            if (seen[r.gpu_addr]++ == 0 && ndumped++ < 60) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/tex_%ux%u_%llx_c%u.bmp", d.c_str(), tw, th,
                                                       (unsigned long long)r.gpu_addr, r.num_components);
                                prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            }
                        }
                        fr.tex_rgba = texture_pixels.data(); fr.tw = tw; fr.th = cube_done ? th * 6u : th;
                        if (!resource_rtt_hit && !has_live_rtt)
                            decoded_textures.emplace(
                                decode_key, DecodedTexture{fr.tex_rgba, fr.th, narrow_done});
                        }
                        // Carry the decoded S# sampler state (filter/wrap/mip) so the pipeline samples the
                        // way the game asked instead of a fixed LINEAR/clamp sampler (#<sampler-fix>).
                        fr.mag_filter = r.mag_filter; fr.min_filter = r.min_filter; fr.mip_filter = r.mip_filter;
                        fr.addr_uvw[0] = r.addr_uvw[0]; fr.addr_uvw[1] = r.addr_uvw[1]; fr.addr_uvw[2] = r.addr_uvw[2];
                        // Remaining S# sampler fields (#262): border color + LOD clamp/bias (applied where
                        // valid; the decode-only compare/unnorm stay on ShaderResource).
                        fr.border_color_type = r.border_color_type;
                        fr.min_lod = r.min_lod; fr.max_lod = r.max_lod; fr.lod_bias = r.lod_bias;
                        // Anisotropy ratio (#275): applied in render_runner.h when the device supports the
                        // samplerAnisotropy feature and filtering is linear. The Messenger decodes ratio 0
                        // (isotropic) so this is a no-op for it; carries correct behavior for titles that
                        // request anisotropic filtering.
                        fr.max_aniso_ratio = r.max_aniso_ratio;
                        // T# DST_SEL channel remap (#261): apply only to REAL-RGBA texels. The narrow
                        // R->RGBA replication path (narrow_done) already broadcasts coverage to every
                        // channel — keep it identity so the font/mask path (#102/#256) is untouched.
                        if (narrow_done) { fr.swizzle[0]=4; fr.swizzle[1]=5; fr.swizzle[2]=6; fr.swizzle[3]=7; }
                        else { for (int k=0;k<4;k++) fr.swizzle[k] = r.swizzle[k]; }
                        // PROSPER_ALPHA1: force the sampled alpha to constant 1 (opaque). Diagnostic for a
                        // black scene whose textures decode to real RGB but composite to nothing — if the
                        // level appears with this, the alpha channel (decode or DST_SEL swizzle) is the bug (#300).
                        if (getenv("PROSPER_ALPHA1")) fr.swizzle[3] = 1;
                    } else {
                        uint32_t nb = std::min(r.size ? r.size : 256u, 1u << 20) & ~3u;   // cap 1 MB, dword-aligned
                        if (nb >= 4) {
                            std::vector<uint8_t> tmp(nb, 0);
                            if (copy_resource(tmp.data(), r.gpu_addr, nb) > 0)
                                fr.dwords.assign((const uint32_t*)tmp.data(), (const uint32_t*)(tmp.data() + nb));
                        }
                        if (fr.dwords.empty()) fr.dwords.assign(64, 0);
                        // PROSPER_CBLOG: log each constant buffer's first 4 dwords as floats, once per
                        // address. If a scene draw's color/tint CB is (0,0,0,0), the PS outputs black
                        // regardless of the (correctly-decoded) texture — the #300 black-scene suspect.
                        if (getenv("PROSPER_CBLOG") && r.cls == RC::ConstantBuffer) {
                            static std::set<uint64_t> cbseen;
                            if (cbseen.insert(r.gpu_addr).second) {
                                const float* fp = (const float*)fr.dwords.data();
                                size_t n = fr.dwords.size();
                                fprintf(stderr, "[cb] bind=%u addr=0x%llx size=%u dw=%08x %08x %08x %08x  f=%.3f %.3f %.3f %.3f\n",
                                        r.binding, (unsigned long long)r.gpu_addr, (unsigned)r.size,
                                        n>0?fr.dwords[0]:0, n>1?fr.dwords[1]:0, n>2?fr.dwords[2]:0, n>3?fr.dwords[3]:0,
                                        n>0?fp[0]:0.f, n>1?fp[1]:0.f, n>2?fp[2]:0.f, n>3?fp[3]:0.f);
                            }
                        }
                    }
                    if (timing_enabled) {
                        const double elapsed = std::chrono::duration<double, std::milli>(
                            RenderClock::now() - resource_timing_start).count();
                        if (fr.is_texture()) {
                            pending_timing.textures++;
                            pending_timing.texture_bytes += static_cast<uint64_t>(fr.tw) * fr.th * 4;
                            pending_timing.texture_ms += elapsed;
                            if (const char* mode = getenv("PROSPER_RENDER_TIMING");
                                mode && strcmp(mode, "detail") == 0 && elapsed >= 0.5) {
                                static uint64_t detail_lines = 0;
                                if (detail_lines++ < 250) {
                                    fprintf(stderr,
                                            "[render-timing] texture addr=0x%llx %ux%u out=%ux%u "
                                            "fmt=%u comps=%u tile=%u rtt=%d %.2f ms\n",
                                            (unsigned long long)r.gpu_addr, r.width, r.height,
                                            fr.tw, fr.th, (unsigned)r.format, r.num_components,
                                            r.tile_mode, (int)resource_rtt_hit, elapsed);
                                }
                            }
                        } else {
                            pending_timing.buffers++;
                            pending_timing.buffer_bytes += static_cast<uint64_t>(fr.dwords.size()) * 4;
                            pending_timing.buffer_ms += elapsed;
                        }
                    }
                    R.push_back(std::move(fr));
                }
              };
              add(vrt, 0); add(prt, 1);   // VS resources -> descriptor set 0, PS -> set 1
              return R;
            };
            // Poison mode keeps the draw running while making invalid bindings visually/numerically
            // unmistakable: magenta/cyan texels for images and NaN-like dwords for buffers. Missing,
            // duplicate, wrong-type, and undersized bindings are replaced from the reflected manifest.
            auto poison_R = [](std::vector<prosper::test::FrameResource>& resources,
                               const std::vector<uint32_t>& spirv,
                               const prosper::gpu::ShaderResourceTable* table,
                               uint32_t set, prosper::gpu::SpirvShaderStage stage) {
                const char* mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");
                if (!mode || strcmp(mode, "poison")) return;
                auto report = prosper::gpu::validate_spirv_descriptor_interface(spirv, table, set, stage, false);
                static const uint8_t poison_tex[16] = {
                    255, 0, 255, 255,   0, 255, 255, 255,
                    0, 255, 255, 255,   255, 0, 255, 255,
                };
                for (const auto& d : report.descriptors) {
                    bool invalid = false;
                    for (const auto& issue : report.issues)
                        if (issue.error && issue.binding == d.binding) { invalid = true; break; }
                    if (!invalid) continue;
                    auto first = std::find_if(resources.begin(), resources.end(), [&](const auto& r) {
                        return r.set == set && r.binding == d.binding;
                    });
                    size_t count = 0;
                    for (const auto& r : resources) if (r.set == set && r.binding == d.binding) ++count;
                    const bool wants_buffer = d.kind == prosper::gpu::SpirvDescriptorKind::StorageBuffer;
                    const uint64_t available = first == resources.end() ? 0 :
                        (first->is_texture() ? (uint64_t)first->tw * first->th * 4 : first->dwords.size() * 4);
                    const bool wrong_type = first != resources.end() && (first->is_texture() == wants_buffer);
                    const bool undersized = wants_buffer && available < std::max<uint64_t>(d.required_bytes, 4);
                    resources.erase(std::remove_if(resources.begin(), resources.end(), [&](const auto& r) {
                        return r.set == set && r.binding == d.binding;
                    }), resources.end());
                    prosper::test::FrameResource replacement;
                    replacement.set = set; replacement.binding = d.binding;
                    if (wants_buffer) {
                        size_t words = static_cast<size_t>((std::max<uint64_t>(d.required_bytes, 16) + 3) / 4);
                        words = std::min<size_t>(words, 1u << 18); // same 1 MiB upload ceiling as build_R
                        replacement.dwords.assign(words, 0x7FC0CDCDu);
                    } else {
                        replacement.tex_rgba = poison_tex; replacement.tw = 2; replacement.th = 2;
                        replacement.mag_filter = replacement.min_filter = 0;
                    }
                    fprintf(stderr, "[descriptor] poison set=%u binding=%u type=%s reason=%s%s%s%s\n",
                            set, d.binding, prosper::gpu::spirv_descriptor_kind_name(d.kind),
                            count == 0 ? "missing" : "",
                            count > 1 ? "duplicate" : "",
                            wrong_type ? "wrong-type" : "",
                            undersized ? "undersized" : "");
                    resources.push_back(std::move(replacement));
                }
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
                    bd.R      = build_R(it, it.vrt.get(), it.prt.get());
                    if (!prosper::gpu::validate_runtime_descriptor_contract(
                            "VS/backend", bd.vs, it.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex) ||
                        !prosper::gpu::validate_runtime_descriptor_contract(
                            "PS/backend", bd.fs, it.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment))
                        continue;
                    poison_R(bd.R, bd.vs, it.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex);
                    poison_R(bd.R, bd.fs, it.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment);
                    // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
                    // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
                    if (!refvs) bd.indices = it.indices;
                    if (getenv("PROSPER_GFXLOG")) fprintf(stderr,
                        "[render] item %zu: %zu resources vcount=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
                        bds.size(), bd.R.size(), bd.vcount, bd.indices.size(), it.ps.topology,
                        it.ps.color_write_mask, (int)it.ps.blend_enable);
                    // RTTLOG per-draw detail (render-window-only, unlike the GFXLOG firehose): enough
                    // state to diagnose a pass whose inputs HIT the RTT cache yet outputs nothing —
                    // blend factors, write mask, viewport, and each PS-sampled texture address (#319).
                    if (getenv("PROSPER_RTTLOG")) {
                        fprintf(stderr, "[rtt]   draw tgt=0x%llx vcount=%u nidx=%zu topo=%u mask=0x%x "
                                "blend=%d(src=%u dst=%u) vp=%d(%.2f,%.2f %.2fx%.2f) z=%d zw=%d ps=",
                                (unsigned long long)it.color0_base, bd.vcount, bd.indices.size(),
                                it.ps.topology, it.ps.color_write_mask, (int)it.ps.blend_enable,
                                it.ps.src_color_blend_factor, it.ps.dst_color_blend_factor,
                                (int)it.ps.has_viewport,
                                it.ps.viewport_x, it.ps.viewport_y, it.ps.viewport_w, it.ps.viewport_h,
                                (int)it.ps.depth_test_enable, (int)it.ps.depth_write_enable);
                        if (it.prt) for (const auto& r : it.prt->resources)
                            if (r.cls == RC::Texture)
                                fprintf(stderr, " tex@%u=0x%llx(%ux%u f%u)", r.binding,
                                        (unsigned long long)r.gpu_addr, r.width, r.height, (unsigned)r.format);
                        fprintf(stderr, "\n");
                    }
                    bds.push_back(std::move(bd));
                }
                return bds;
            };
            // Clear color for a group: the game's decoded fast-clear taken from the group's first item's
            // resolved pipeline state, or its default opaque black when no fast-clear was programmed.
            // Passing this to render_draws_rgba replaces the old hardcoded debug blue on the live path
            // (#309) — PROSPER_CLEAR_DEBUG still forces blue for spotting unrendered areas.
            auto clear_for = [](const std::vector<const prosper::gpu::DrawItem*>& g) -> const float* {
                return g.empty() ? nullptr : g.front()->ps.clear_color;
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
                // PASSES, not groups (#300): a real frame renders in SUBMIT ORDER as A->B->A->B... (a
                // ping-pong RT chain, each pass sampling the previous target). Collapsing all draws with
                // the same color0_base into one group loses that order — an A-pass that samples B then
                // renders before B exists, reading a stale/empty B -> black cascade (and a 4-deep UE4
                // post chain propagates nothing). So split items into CONTIGUOUS same-target runs below
                // and render each in order, caching its RT immediately so the next pass sampling it hits
                // fresh pixels. (Cross-submit persistence via g_rtt is unchanged.)
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
                size_t pass_i = 0;
                while (pass_i < items.size()) {
                    const uint64_t base = items[pass_i].color0_base;
                    std::vector<const prosper::gpu::DrawItem*> pass;
                    while (pass_i < items.size() && items[pass_i].color0_base == base) { pass.push_back(&items[pass_i]); ++pass_i; }

                    // Gen5 render-target extent (#526). Large scene/scanout surfaces retain the
                    // configured VideoOut render scale; small offscreen targets render at native
                    // resolution so lookup textures (Messenger's 1024x32 grading LUT) preserve every
                    // texel. The viewport was already scaled for the global w/h framebuffer by
                    // execute_gpustate, so correct it from that scale to this pass-local scale.
                    uint32_t native_w = pass.empty() ? 0u : pass.front()->color0_width;
                    uint32_t native_h = pass.empty() ? 0u : pass.front()->color0_height;
                    uint32_t gw = w, gh = h;
                    const uint32_t present_w = prosper::gpu::present_width();
                    const uint32_t present_h = prosper::gpu::present_height();
                    if (native_w && native_h) {
                        uint64_t native_pixels = (uint64_t)native_w * native_h;
                        uint64_t global_pixels = (uint64_t)w * h;
                        if (native_pixels <= global_pixels) {
                            gw = native_w; gh = native_h;
                        } else if (present_w && present_h) {
                            gw = std::max(1u, (uint32_t)(((uint64_t)native_w * w + present_w / 2) / present_w));
                            gh = std::max(1u, (uint32_t)(((uint64_t)native_h * h + present_h / 2) / present_h));
                        }
                    }
                    std::vector<prosper::gpu::DrawItem> adjusted;
                    std::vector<const prosper::gpu::DrawItem*> render_pass = pass;
                    if (native_w && native_h && present_w && present_h && w && h) {
                        const float ax = ((float)gw / native_w) / ((float)w / present_w);
                        const float ay = ((float)gh / native_h) / ((float)h / present_h);
                        if (ax != 1.0f || ay != 1.0f) {
                            adjusted.reserve(pass.size()); render_pass.clear(); render_pass.reserve(pass.size());
                            for (const auto* src : pass) {
                                adjusted.push_back(*src);
                                auto& item = adjusted.back();
                                if (item.ps.has_viewport) {
                                    item.ps.viewport_x *= ax; item.ps.viewport_w *= ax;
                                    item.ps.viewport_y *= ay; item.ps.viewport_h *= ay;
                                }
                                render_pass.push_back(&item);
                            }
                        }
                    }
                    const uint8_t* seed = nullptr;
                    if (seed_rtt && base) { auto sit = g_rtt.find(base);
                        if (sit != g_rtt.end() && sit->second.w == gw && sit->second.h == gh &&
                            sit->second.rgba.size() == (size_t)gw * gh * 4) seed = sit->second.rgba.data(); }
                    const auto build_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    auto backend_draws = build_bds(render_pass);
                    const auto build_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(
                        backend_draws, gw, gh, seed, clear_for(render_pass), true);
                    const auto backend_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    if (timing_enabled) {
                        pending_timing.build_resources_ms +=
                            std::chrono::duration<double, std::milli>(build_done - build_start).count();
                        pending_timing.backend_ms +=
                            std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                    }
                    if (base && !gpx.empty()) { RttSurf& s = g_rtt[base]; s.rgba = gpx; s.w = gw; s.h = gh; }
                    if (native_w == resource_hash_w && native_h == resource_hash_h && !gpx.empty()) {
                        uint64_t hash = 1469598103934665603ull;
                        for (uint8_t byte : gpx) {
                            hash ^= byte;
                            hash *= 1099511628211ull;
                        }
                        const auto* first = render_pass.empty() ? nullptr : render_pass.front();
                        const auto* last = render_pass.empty() ? nullptr : render_pass.back();
                        fprintf(stderr,
                                "[target-version] render-submit=%llu target=0x%llx dims=%ux%u "
                                "draws=%llu-%llu orders=%llu-%llu seed=%d clear=%d "
                                "rgba=%.3f,%.3f,%.3f,%.3f hash=%016llx\n",
                                (unsigned long long)g_this_submit,
                                (unsigned long long)base, native_w, native_h,
                                (unsigned long long)(first ? first->draw_index : 0),
                                (unsigned long long)(last ? last->draw_index : 0),
                                (unsigned long long)(first ? first->command_order : 0),
                                (unsigned long long)(last ? last->command_order : 0),
                                seed != nullptr, first ? (int)first->ps.has_clear_color : 0,
                                first ? first->ps.clear_color[0] : 0.0f,
                                first ? first->ps.clear_color[1] : 0.0f,
                                first ? first->ps.clear_color[2] : 0.0f,
                                first ? first->ps.clear_color[3] : 0.0f,
                                (unsigned long long)hash);
                    }
                    if (native_w == target_step_w && native_h == target_step_h &&
                        render_pass.size() >= target_step_min_draws) {
                        for (size_t k = 1; k <= render_pass.size(); ++k) {
                            std::vector<const prosper::gpu::DrawItem*> prefix(
                                render_pass.begin(), render_pass.begin() + k);
                            texstore_used = 0;
                            decoded_textures.clear();
                            std::vector<uint8_t> step = prosper::test::render_draws_rgba(
                                build_bds(prefix), gw, gh, seed, clear_for(prefix), true);
                            uint64_t hash = 1469598103934665603ull;
                            for (uint8_t byte : step) {
                                hash ^= byte;
                                hash *= 1099511628211ull;
                            }
                            size_t dark = 0, near_white = 0;
                            uint64_t rgb_sum = 0;
                            for (size_t p = 0; p + 3 < step.size(); p += 4) {
                                const uint8_t r = step[p], g = step[p + 1], b = step[p + 2];
                                dark += std::max({r, g, b}) < 64;
                                near_white += std::min({r, g, b}) > 240;
                                rgb_sum += r + g + b;
                            }
                            const double mean_rgb = step.empty() ? 0.0 :
                                (double)rgb_sum / ((step.size() / 4) * 3);
                            const auto* draw = prefix.back();
                            auto spirv_hash = [](const std::vector<uint32_t>& words) {
                                uint64_t hash = 1469598103934665603ull;
                                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(words.data());
                                for (size_t i = 0; i < words.size() * sizeof(uint32_t); ++i) {
                                    hash ^= bytes[i];
                                    hash *= 1099511628211ull;
                                }
                                return hash;
                            };
                            fprintf(stderr,
                                    "[target-step] render-submit=%llu target=0x%llx step=%zu/%zu "
                                    "draw=%llu order=%llu vs=%016llx ps=%016llx mask=0x%x "
                                    "blend=%d/%u/%u hash=%016llx dark=%zu white=%zu mean=%.2f\n",
                                    (unsigned long long)g_this_submit,
                                    (unsigned long long)base, k, render_pass.size(),
                                    (unsigned long long)draw->draw_index,
                                    (unsigned long long)draw->command_order,
                                    (unsigned long long)spirv_hash(draw->vs),
                                    (unsigned long long)spirv_hash(draw->fs),
                                    draw->ps.color_write_mask, (int)draw->ps.blend_enable,
                                    draw->ps.src_color_blend_factor,
                                    draw->ps.dst_color_blend_factor,
                                    (unsigned long long)hash, dark, near_white, mean_rgb);
                        }
                    }
                    bool is_vo = false;
                    for (int i = 0; i < vo_n && !is_vo; i++) is_vo = base && base == prosper_vo_buffer_addr(i);
                    if (getenv("PROSPER_RTTLOG")) {
                        size_t nz = 0, rgb_nz = 0;
                        for (uint8_t b : gpx) nz += (b != 0);
                        for (size_t p = 0; p + 3 < gpx.size(); p += 4)
                            rgb_nz += (gpx[p] != 0 || gpx[p + 1] != 0 || gpx[p + 2] != 0);
                        fprintf(stderr, "[rtt] pass target=0x%llx extent=%ux%u native=%ux%u (%zu draws) "
                                "px_nonzero=%zu rgb_nonblack=%zu cache_size=%zu%s%s\n",
                                (unsigned long long)base, gw, gh, native_w, native_h, pass.size(), nz, rgb_nz, g_rtt.size(),
                                is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : ""); }
                    // PROSPER_DUMP_DRAWSTEPS: for a pass targeting a SCANOUT buffer, re-render the pass
                    // draw-by-draw (prefix 1, prefix 2, ...) and dump each cumulative result — a one-boot
                    // bisect for "which draw of the final composite blacks the screen" (#319). Diagnostic.
                    if (getenv("PROSPER_DUMP_DRAWSTEPS") && is_vo && pass.size() > 1) {
                        for (size_t k = 1; k <= pass.size(); k++) {
                            std::vector<const prosper::gpu::DrawItem*> prefix(render_pass.begin(), render_pass.begin() + k);
                            std::vector<uint8_t> spx = prosper::test::render_draws_rgba(
                                build_bds(prefix), gw, gh, seed, clear_for(prefix));
                            if (spx.empty()) continue;
                            size_t snz = 0; for (uint8_t b : spx) snz += (b != 0);
                            fprintf(stderr, "[rtt] drawstep %zu/%zu tgt=0x%llx px_nonzero=%zu\n",
                                    k, pass.size(), (unsigned long long)base, snz);
                            const char* dd = getenv("PROSPER_FRAME_DIR");
                            char fn[512]; snprintf(fn, sizeof fn, "%s/drawstep_%04d_%zu.bmp",
                                                   dd ? dd : ".", frame_no.load(), k);
                            prosper::test::dump_bmp(fn, spx, gw, gh);
                        }
                    }
                    // PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes>: dump each per-target group's rendered
                    // pixels (rtgrp_<base>_<frame>.bmp in PROSPER_FRAME_DIR) — to inspect an intermediate
                    // pass (e.g. the UI/banner RT) instead of only the presented composite. Diagnostic.
                    if (const char* rg = getenv("PROSPER_DUMP_RTGROUPS"); rg && !gpx.empty()) {
                        size_t nz = 0; for (uint8_t b : gpx) nz += (b != 0);
                        if (nz >= (size_t)atol(rg)) {
                            const char* dd = getenv("PROSPER_FRAME_DIR");
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rtgrp_%llx_%04d.bmp",
                                                   dd ? dd : ".", (unsigned long long)base, frame_no.load());
                            prosper::test::dump_bmp(fn, gpx, gw, gh);
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
                const auto build_start = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                auto backend_draws = build_bds(all);
                const auto build_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                px = prosper::test::render_draws_rgba(backend_draws, w, h, nullptr, clear_for(all), true);
                const auto backend_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                if (timing_enabled) {
                    pending_timing.build_resources_ms +=
                        std::chrono::duration<double, std::milli>(build_done - build_start).count();
                    pending_timing.backend_ms +=
                        std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                }
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
            // Ordered submits may invoke this callback for several graphics spans separated by
            // compute dispatches. Intermediate spans only update g_rtt. At the final span, recover
            // the flipped scanout from that persistent cache even when it was rendered earlier in
            // the transaction, and advance frame/dump state exactly once.
            if (phase.final_span && pertarget) {
                auto cached_scanout = [&](uint64_t addr) -> const RttSurf* {
                    auto it = g_rtt.find(addr);
                    if (it == g_rtt.end() || it->second.w != w || it->second.h != h ||
                        it->second.rgba.size() != (size_t)w * h * 4) return nullptr;
                    return &it->second;
                };
                const int front = prosper::gpu::present_front_index();
                const RttSurf* scanout = front >= 0
                    ? cached_scanout(prosper_vo_buffer_addr(front)) : nullptr;
                if (!scanout) {
                    for (int i = 0; i < prosper_vo_buffer_count(); ++i)
                        if ((scanout = cached_scanout(prosper_vo_buffer_addr(i)))) break;
                }
                if (scanout) px = scanout->rgba;
            }
            if (!phase.final_span) {
                if (timing_enabled) {
                    pending_timing.callbacks++;
                    pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                        RenderClock::now() - callback_timing_start).count();
                }
                return px;
            }
            int n = frame_no++;
            // PROSPER_DUMP_CONTENT=<min-nonzero-bytes>: dump ONLY frames whose framebuffer has at least
            // that many nonzero bytes — catches the intermittent content submits the periodic dump misses.
            size_t content_thr = 0; if (const char* c = getenv("PROSPER_DUMP_CONTENT")) content_thr = (size_t)atol(c);
            size_t px_nz = 0;
            if (dump_bmps) for (uint8_t b : px) px_nz += (b != 0);
            if (px.empty()) {
                fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
            } else if (dump_bmps && ((content_thr && px_nz >= content_thr) || (!content_thr && (n < 60 || n % 10 == 0)))) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
                prosper::test::dump_bmp(fn, px, w, h);
                fprintf(stderr, "[render] frame %d rendered (%ux%u) nz=%zu -> %s\n", n, w, h, px_nz, fn);
            }
            if (timing_enabled) {
                pending_timing.callbacks++;
                pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - callback_timing_start).count();
                struct TimingTotals {
                    uint64_t submits = 0, callbacks = 0;
                    double total_ms = 0, build_resources_ms = 0, backend_ms = 0;
                    uint64_t textures = 0, texture_reuses = 0, buffers = 0;
                    uint64_t texture_bytes = 0, buffer_bytes = 0;
                    double texture_ms = 0, buffer_ms = 0;
                };
                static TimingTotals totals;
                static TimingTotals window;
                auto accumulate = [&](TimingTotals& timing) {
                    timing.submits++;
                    timing.callbacks += pending_timing.callbacks;
                    timing.total_ms += pending_timing.total_ms;
                    timing.build_resources_ms += pending_timing.build_resources_ms;
                    timing.backend_ms += pending_timing.backend_ms;
                    timing.textures += pending_timing.textures;
                    timing.texture_reuses += pending_timing.texture_reuses;
                    timing.buffers += pending_timing.buffers;
                    timing.texture_bytes += pending_timing.texture_bytes;
                    timing.buffer_bytes += pending_timing.buffer_bytes;
                    timing.texture_ms += pending_timing.texture_ms;
                    timing.buffer_ms += pending_timing.buffer_ms;
                };
                accumulate(totals);
                accumulate(window);
                if (totals.submits % 25 == 0) {
                    const double nsub = static_cast<double>(totals.submits);
                    const double other = totals.total_ms - totals.build_resources_ms - totals.backend_ms;
                    fprintf(stderr,
                            "[render-timing] frontend submits=%llu callbacks=%llu avg_ms: total=%.2f "
                            "build_resources=%.2f backend=%.2f other=%.2f\n",
                            (unsigned long long)totals.submits, (unsigned long long)totals.callbacks,
                            totals.total_ms / nsub, totals.build_resources_ms / nsub,
                            totals.backend_ms / nsub, other / nsub);
                    fprintf(stderr,
                            "[render-timing] resources textures=%llu reused=%llu %.1f MiB %.2f ms/submit; "
                            "buffers=%llu %.1f MiB %.2f ms/submit\n",
                            (unsigned long long)totals.textures,
                            (unsigned long long)totals.texture_reuses,
                            totals.texture_bytes / (1024.0 * 1024.0), totals.texture_ms / nsub,
                            (unsigned long long)totals.buffers,
                            totals.buffer_bytes / (1024.0 * 1024.0), totals.buffer_ms / nsub);
                    const double wn = static_cast<double>(window.submits);
                    const double window_other = window.total_ms - window.build_resources_ms -
                                                window.backend_ms;
                    fprintf(stderr,
                            "[render-window] frontend submits=%llu callbacks=%.1f avg_ms: total=%.2f "
                            "build_resources=%.2f backend=%.2f other=%.2f; resources textures=%.1f "
                            "reused=%.1f %.1f MiB %.2f ms buffers=%.1f %.1f MiB %.2f ms\n",
                            (unsigned long long)window.submits, window.callbacks / wn,
                            window.total_ms / wn, window.build_resources_ms / wn,
                            window.backend_ms / wn, window_other / wn, window.textures / wn,
                            window.texture_reuses / wn,
                            window.texture_bytes / (wn * 1024.0 * 1024.0), window.texture_ms / wn,
                            window.buffers / wn, window.buffer_bytes / (wn * 1024.0 * 1024.0),
                            window.buffer_ms / wn);
                    window = {};
                }
            }
            return px;
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend

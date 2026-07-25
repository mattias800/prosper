// render_runner.h — inline helper to render a single triangle (3 vertices, no vertex input) with a
// given vertex + fragment SPIR-V pair into a WxH RGBA8 image, clearing to blue first, and return the
// pixels. Used to verify recompiled shaders end-to-end (render -> readback -> pixel asserts). The
// including test links Vulkan::Vulkan.
#pragma once
#include <vulkan/vulkan.h>
#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/render_state.hpp"
#include "../frontends/shared/vulkan_device_select.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::test {

// Write an RGBA8 framebuffer to a binary PPM (P6) so a rendered frame is viewable as an image. PPM is
// trivially convertible to PNG (e.g. `magick in.ppm out.png`). Used by the render demos to leave
// screenshots on disk. Returns true on success.
inline bool dump_ppm(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    fprintf(f, "P6\n%u %u\n255\n", W, H);
    for (size_t i = 0; i < (size_t)W * H; i++) fwrite(&px[i * 4], 1, 3, f);   // RGB (drop alpha)
    fclose(f); return true;
}

// Write an RGBA8 framebuffer to a 24-bit BMP — natively viewable on Windows (double-click). BMP rows
// are bottom-up and BGR, padded to a 4-byte boundary.
inline bool dump_bmp(const char* path, const std::vector<uint8_t>& px, uint32_t W, uint32_t H) {
    if (px.size() != (size_t)W * H * 4) return false;
    FILE* f = fopen(path, "wb"); if (!f) return false;
    const uint32_t rowpad = (4 - (W * 3) % 4) % 4;
    const uint32_t dataSize = (W * 3 + rowpad) * H;
    const uint32_t fileSize = 54 + dataSize;
    auto u16 = [&](uint32_t v){ uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; fwrite(b, 1, 2, f); };
    auto u32 = [&](uint32_t v){ uint8_t b[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)}; fwrite(b, 1, 4, f); };
    fputc('B', f); fputc('M', f); u32(fileSize); u32(0); u32(54);                 // file header
    u32(40); u32(W); u32(H); u16(1); u16(24); u32(0); u32(dataSize); u32(2835); u32(2835); u32(0); u32(0);  // info header
    for (int y = (int)H - 1; y >= 0; y--) {                                       // bottom-up rows
        for (uint32_t x = 0; x < W; x++) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4];
            fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);                        // BGR
        }
        for (uint32_t k = 0; k < rowpad; k++) fputc(0, f);
    }
    fclose(f); return true;
}

// A texture to bind for a recompiled shader's image_sample: `rgba` points to w*h*4 RGBA8 bytes,
// bound as a COMBINED_IMAGE_SAMPLER (nearest filter, clamp) at descriptor-set 0, `binding`.
struct TexDesc { uint32_t binding; uint32_t w; uint32_t h; const uint8_t* rgba;
                 uint32_t max_aniso_ratio = 0; };   // #275: S# anisotropy ratio (0 = isotropic)

inline uint64_t hash_buffer_words(const uint32_t* words, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t word = 0; word < count; ++word) {
        hash ^= words[word];
        hash *= 1099511628211ull;
    }
    return hash;
}

// One resource for the general N-binding path (render_triangle_rgba's `gres`): a storage buffer
// (dwords non-empty, tex_rgba null), combined image sampler, or storage image at `binding`. Lets a
// real game shader that declares several buffers and images have each bound distinctly.
struct FrameResource {
    uint32_t binding = 0;
    uint32_t set = 0;               // descriptor set: VS resources -> 0, PS resources -> 1 (they must not
                                    // share a set — both stages number bindings from 2, so one set would
                                    // collide binding 2/3 between stages and make the layout invalid).
    std::vector<uint32_t> dwords;   // owned storage-buffer contents (empty -> a 1-dword zero buffer)
    // A production callback may point directly at a fully readable immutable guest/capture range.
    // The backend consumes and uploads this view synchronously; an explicit submission batch retains
    // only the completed Vulkan upload, never this pointer. Tests/replays may use the same contract
    // when their backing outlives render_draws_rgba().
    const uint32_t* dwords_view = nullptr;
    size_t dwords_view_count = 0;
    // Exact guest resource identity for a storage buffer. The backend may share an immutable upload
    // within one synchronous call only when both this identity and the complete captured bytes match;
    // zero keeps synthetic/replay resources conservatively distinct.
    uint64_t buffer_identity = 0;
    // Backend-owned PS5 GDS storage. Unlike guest/capture buffers this is one persistent,
    // zero-initialized 64 KiB allocation shared across ordered render calls.
    bool is_internal_gds = false;
    const uint8_t* tex_rgba = nullptr;   // non-null => a texture; then tw/th are its dimensions
    uint32_t tw = 0, th = 0, td = 1;
    // T#-declared mip-chain length (#1272). 1 (default) = single-level upload, the historical
    // behavior. >1 lets the backend generate a box-filtered chain — bounded by this declared count —
    // for plain-2D RGBA8 sampled textures, so minification stops point-sampling through dense art.
    uint32_t declared_mip_levels = 1;
    uint32_t img_dim = 1;             // ShaderResource/MIMG dim (1=2D, 2=3D); depth-1 3D stays 3D
    // Renderer-owned RTTs keep their native format between producer and consumer. Guest-backed
    // textures still arrive through the existing RGBA8 decoder unless explicitly tagged otherwise.
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    // StorageImage is a distinct Vulkan descriptor contract: no sampler, STORAGE usage, and GENERAL
    // layout. Keep the flag independent of tex_rgba because both sampled and storage images upload
    // decoded pixels through that pointer.
    bool is_storage_image = false;
    // Sampler state (Texture only). Defaults = LINEAR + clamp-to-edge — the harness's prior fixed
    // sampler — so render tests that build FrameResources directly stay byte-identical. The live path
    // fills these from the decoded S# (shader_resources.hpp). filter: 0=nearest, 1=linear; addr = Gen5
    // SQ_TEX CLAMP enum (0=wrap, 1=mirror, 2=clamp-last-texel, 6/7=border).
    uint32_t mag_filter = 1, min_filter = 1, mip_filter = 0;
    uint32_t addr_uvw[3] = {2, 2, 2};
    // Remaining S# sampler fields (#262). Defaults reproduce the current Vulkan sampler exactly (border
    // transparent-black, LOD 0..0, no bias), so FrameResources built directly by tests are byte-identical.
    uint32_t border_color_type = 0;
    float    min_lod = 0.0f, max_lod = 0.0f, lod_bias = 0.0f;
    // Anisotropy ratio (S# WORD0[11:9]; maxAnisotropy = 1<<ratio). 0 = isotropic (the default) -> the
    // sampler is byte-identical to before, so tests building FrameResources directly are unaffected (#275).
    uint32_t max_aniso_ratio = 0;
    // T# DST_SEL channel swizzle (SQ_SEL per channel: 0=0,1=1,4=R,5=G,6=B,7=A). Default = identity
    // (R,G,B,A) == a no-op VkComponentMapping, so tests that build FrameResources directly are unchanged.
    uint32_t swizzle[4] = {4, 5, 6, 7};
    // Non-zero only when the frontend has revalidated the complete guest source backing and can
    // prove these decoded pixels are the same content version as a prior callback. The Vulkan
    // backend may retain an uploaded image under this ID; zero remains callback-local/transient.
    uint64_t persistent_texture_id = 0;
    // Non-zero identifies a color target retained by an earlier backend call. When that exact
    // target is available, bind its GPU image directly instead of uploading `tex_rgba` again.
    // `tex_rgba` remains an optional conservative fallback for an invalidated/missing target.
    uint64_t persistent_render_target_id = 0;
    // Non-zero identifies a persistent depth/stencil surface whose DEPTH plane this resource
    // samples (a shadow map / depth pyramid tap, #1275). The backend binds the retained Vulkan
    // depth image directly — prosper never writes rendered depth back to guest memory, so there
    // is no CPU fallback; an invalidated/missing surface falls through to the guest-byte decode.
    uint64_t persistent_depth_target_id = 0;
    const uint32_t* buffer_words_data() const {
        return dwords_view && dwords_view_count
            ? dwords_view : (dwords.empty() ? nullptr : dwords.data());
    }
    size_t buffer_word_count() const {
        return dwords_view && dwords_view_count ? dwords_view_count : dwords.size();
    }
    bool is_texture() const {
        return tex_rgba != nullptr || persistent_render_target_id != 0 ||
               persistent_depth_target_id != 0;
    }
};

// Optional color-target contract for the live backend. A non-zero ID gives the target a stable
// guest identity across calls. Existing contents are loaded only when both `load_existing` and a
// valid matching image are present. `readback=false` leaves the completed result GPU-resident and
// deliberately returns an empty CPU vector.
struct BackendColorTarget {
    uint64_t persistent_id = 0;
    bool load_existing = true;
    bool readback = true;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

inline VkFormat backend_color_format(VkFormat format) {
    return format == VK_FORMAT_R16G16B16A16_SFLOAT
        ? VK_FORMAT_R16G16B16A16_SFLOAT
        : VK_FORMAT_R8G8B8A8_UNORM;
}

inline uint32_t backend_color_bytes_per_pixel(VkFormat format) {
    return backend_color_format(format) == VK_FORMAT_R16G16B16A16_SFLOAT ? 8u : 4u;
}

struct BackendColorTargetStats {
    uint64_t writes = 0;
    uint64_t write_hits = 0;
    uint64_t sampled_hits = 0;
    uint64_t readbacks = 0;
    uint64_t cached_bytes = 0;
    uint64_t cached_entries = 0;
};

inline BackendColorTargetStats& backend_color_target_stats_storage() {
    static thread_local BackendColorTargetStats stats;
    return stats;
}

inline BackendColorTargetStats backend_color_target_stats() {
    return backend_color_target_stats_storage();
}

// Returns native target bytes (RGBA8 by default, RGBA16F when requested), or {} on failure. When `ps`
// is non-null, the pipeline's fixed-function state (topology, blend, color write mask) is taken from
// the resolved RDNA2 render-state — this is how the back-half realizes a GpuState as a real VkPipeline.
//
// When `vbuf` and/or `cbuf` are non-null, a descriptor set is bound with the constant buffer at
// binding 2 and the vertex buffer at binding 3 (as storage buffers), matching declare_cbufs() in the
// recompiler — this is how a table-recompiled vertex shader fetches real vertex/constant data. Each is
// the raw dword contents of the buffer to bind. When both are null, the pipeline layout is empty and
// the color-only path above is taken unchanged.
//
// When `tex` is non-null, its RGBA8 texels are uploaded to a sampled VkImage and bound as a combined
// image sampler at tex->binding — how a recompiled pixel shader's image_sample reaches a real texture.
// One draw for the multi-draw backend: recompiled VS+PS SPIR-V, its resolved fixed-function state, its
// set-tagged resources, vertex count, and instance count. render_draws_rgba records ALL of a submit's
// draws into ONE
// render pass (clear once, then per-draw pipeline+descriptors+draw) so a multi-draw frame composites
// correctly. render_triangle_rgba is a thin single-draw wrapper (below).
struct BackendDraw {
    std::vector<uint32_t> vs, gs, fs;
    prosper::gpu::SharedShaderWords vs_shared, fs_shared;
    uint64_t vs_identity = 0, fs_identity = 0;
    const prosper::gpu::ResolvedPipelineState* ps = nullptr;   // null -> triangle-list, write RGBA, no depth
    std::vector<FrameResource> R;                              // set-tagged resources (empty -> no descriptors)
    uint32_t vcount = 3;
    uint32_t instance_count = 1;
    int32_t vertex_offset = 0;
    // Indexed draw: 32-bit index data (the executor widens guest 16-bit indices). Non-empty -> the draw
    // is recorded as vkCmdBindIndexBuffer + vkCmdDrawIndexed(indices.size()), so gl_VertexIndex is the
    // fetched index — exactly what the recompiled VS's storage-buffer vertex fetch expects. Empty ->
    // plain vkCmdDraw(vcount). Both paths preserve instance_count.
    std::vector<uint32_t> indices;

    const std::vector<uint32_t>& vs_words() const { return vs_shared ? *vs_shared : vs; }
    const std::vector<uint32_t>& gs_words() const { return gs; }
    const std::vector<uint32_t>& fs_words() const { return fs_shared ? *fs_shared : fs; }
};

struct BackendTextureUploadStats {
    size_t references = 0;
    size_t unique_uploads = 0;
    uint64_t upload_bytes = 0;
    size_t persistent_hits = 0;
    size_t persistent_misses = 0;
    uint64_t persistent_cached_bytes = 0;
};

inline BackendTextureUploadStats& backend_texture_upload_stats_storage() {
    static BackendTextureUploadStats stats;
    return stats;
}

inline BackendTextureUploadStats backend_texture_upload_stats() {
    return backend_texture_upload_stats_storage();
}

// Per-call Vulkan objects that can be shared only when their complete immutable contracts match.
// These counters make the optimization auditable without exposing backend implementation details to
// the live renderer.
struct BackendResourceReuseStats {
    size_t buffer_references = 0;
    size_t unique_buffers = 0;
    size_t texture_binding_references = 0;
    size_t unique_texture_bindings = 0;
    size_t descriptor_set_layout_references = 0;
    size_t unique_descriptor_set_layouts = 0;
    size_t pipeline_layout_references = 0;
    size_t unique_pipeline_layouts = 0;
    size_t descriptor_pools = 0;
    size_t persistent_pipeline_layout_hits = 0;
    size_t persistent_pipeline_layout_misses = 0;
    size_t persistent_pipeline_layout_entries = 0;
    size_t persistent_pipeline_layout_evictions = 0;
    size_t persistent_texture_binding_hits = 0;
    size_t persistent_texture_binding_misses = 0;
    size_t persistent_texture_binding_entries = 0;
    size_t persistent_texture_binding_evictions = 0;
    // Buffer content-hash economics (#1268): the shared-buffer dedup key embeds a full content
    // hash, which profiled as the dominant CPU term on Blue Prince's ~4,000-draw submits. These
    // count, per call like the rest of this struct, how much hashing actually ran and how much
    // was avoided (repeat references resolved by the per-call memo; unique-tag keys that cannot
    // match anything skip hashing entirely).
    uint64_t buffer_hash_calls = 0;
    uint64_t buffer_hash_dwords = 0;
    uint64_t buffer_hash_skipped_unique = 0;
    uint64_t buffer_hash_skipped_large = 0;
    uint64_t buffer_ref_memo_hits = 0;
};

inline BackendResourceReuseStats& backend_resource_reuse_stats_storage() {
    static thread_local BackendResourceReuseStats stats;
    return stats;
}

inline BackendResourceReuseStats backend_resource_reuse_stats() {
    return backend_resource_reuse_stats_storage();
}

// Cumulative across calls (the per-call struct above is reset at every render_draws_rgba entry, which
// tests rely on). PROSPER_HASH_STATS=1 prints these every few seconds from the render path so a live
// route shows the hashing economics without a debugger (#1268).
struct BackendHashStatsTotals {
    uint64_t references = 0;
    uint64_t memo_hits = 0;
    uint64_t hash_calls = 0;
    uint64_t hash_dwords = 0;
    uint64_t skipped_unique = 0;
    uint64_t skipped_large = 0;
    uint64_t unique_buffers = 0;
};
inline BackendHashStatsTotals& backend_hash_stats_totals() {
    static thread_local BackendHashStatsTotals totals;
    return totals;
}
inline void maybe_report_hash_stats() {
    static const bool on = getenv("PROSPER_HASH_STATS") != nullptr;
    if (!on) return;
    static thread_local auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(5)) return;
    last = now;
    const BackendHashStatsTotals& t = backend_hash_stats_totals();
    fprintf(stderr,
            "[hash-stats] refs=%llu memo_hits=%llu hash_calls=%llu hashed_MiB=%.1f "
            "skipped_unique=%llu skipped_large=%llu unique_buffers=%llu\n",
            (unsigned long long)t.references, (unsigned long long)t.memo_hits,
            (unsigned long long)t.hash_calls, (double)t.hash_dwords * 4.0 / (1024.0 * 1024.0),
            (unsigned long long)t.skipped_unique, (unsigned long long)t.skipped_large,
            (unsigned long long)t.unique_buffers);
}

struct BackendPipelineCacheStats {
    uint64_t references = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bypasses = 0;
    uint64_t entries = 0;
    uint64_t evictions = 0;
};

inline BackendPipelineCacheStats& backend_pipeline_cache_stats_storage() {
    static thread_local BackendPipelineCacheStats stats;
    return stats;
}

inline BackendPipelineCacheStats backend_pipeline_cache_stats() {
    return backend_pipeline_cache_stats_storage();
}

struct PersistentPipelineKey {
    static constexpr size_t kInlineWords = 64;
    std::array<uint32_t, kInlineWords> inline_words{};
    std::vector<uint32_t> overflow_words;
    uint32_t word_count = 0;
    uint64_t hash = 1469598103934665603ull;

    bool operator==(const PersistentPipelineKey& other) const {
        if (hash != other.hash || word_count != other.word_count) return false;
        const size_t inline_count = std::min<size_t>(word_count, kInlineWords);
        return std::equal(inline_words.begin(), inline_words.begin() + inline_count,
                          other.inline_words.begin()) &&
               overflow_words == other.overflow_words;
    }
};

struct PersistentPipelineKeyHash {
    size_t operator()(const PersistentPipelineKey& key) const {
        return static_cast<size_t>(key.hash);
    }
};

struct PersistentPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint64_t last_use = 0;
};

inline std::unordered_map<PersistentPipelineKey, PersistentPipeline, PersistentPipelineKeyHash>&
persistent_pipeline_cache() {
    static std::unordered_map<PersistentPipelineKey, PersistentPipeline, PersistentPipelineKeyHash> cache;
    return cache;
}

inline uint64_t& persistent_pipeline_generation() {
    static uint64_t generation = 0;
    return generation;
}

inline bool persistent_pipeline_cache_enabled() {
    return getenv("PROSPER_NO_BACKEND_PIPELINE_CACHE") == nullptr;
}

inline size_t persistent_pipeline_cache_limit() {
    static const size_t limit = [] {
        const char* value = getenv("PROSPER_PIPELINE_CACHE_ENTRIES");
        return value ? static_cast<size_t>(strtoull(value, nullptr, 10)) : size_t{1024};
    }();
    return limit;
}

struct BackendWordVectorHash {
    size_t operator()(const std::vector<uint64_t>& words) const {
        uint64_t hash = 1469598103934665603ull;
        for (uint64_t word : words) {
            hash ^= word;
            hash *= 1099511628211ull;
        }
        return static_cast<size_t>(hash);
    }
};

struct PersistentBackendPipelineLayout {
    VkPipelineLayout handle = VK_NULL_HANDLE;
    uint64_t last_use = 0;
};

inline std::unordered_map<std::vector<uint64_t>, PersistentBackendPipelineLayout,
                          BackendWordVectorHash>&
persistent_backend_pipeline_layout_cache() {
    static std::unordered_map<std::vector<uint64_t>, PersistentBackendPipelineLayout,
                              BackendWordVectorHash> cache;
    return cache;
}

inline uint64_t& persistent_pipeline_layout_generation() {
    static uint64_t generation = 0;
    return generation;
}

inline bool persistent_pipeline_layout_cache_enabled() {
    return getenv("PROSPER_NO_BACKEND_PIPELINE_LAYOUT_CACHE") == nullptr;
}

inline size_t persistent_pipeline_layout_cache_limit() {
    static const size_t limit = [] {
        const char* value = getenv("PROSPER_PIPELINE_LAYOUT_CACHE_ENTRIES");
        return value ? static_cast<size_t>(strtoull(value, nullptr, 10)) : size_t{256};
    }();
    return limit;
}

struct BackendRenderTimingStats {
    uint64_t calls = 0;
    uint64_t draws = 0;
    uint64_t command_buffers = 0;
    uint64_t queue_submits = 0;
    uint64_t fence_waits = 0;
    double target_ms = 0;
    double draw_setup_ms = 0;
    double record_upload_ms = 0;
    double gpu_wait_ms = 0;
    double readback_ms = 0;
    double cleanup_ms = 0;
    double setup_shader_ms = 0;
    double setup_fixed_ms = 0;
    double setup_resources_ms = 0;
    double setup_pipeline_ms = 0;

    double total_ms() const {
        return target_ms + draw_setup_ms + record_upload_ms + gpu_wait_ms + readback_ms + cleanup_ms;
    }
};

inline BackendRenderTimingStats& backend_render_timing_stats_storage() {
    static thread_local BackendRenderTimingStats stats;
    return stats;
}

inline BackendRenderTimingStats backend_render_timing_stats() {
    return backend_render_timing_stats_storage();
}

// `seed_rgba` (optional): native-format pixels to PRELOAD the color attachment with before the draws
// run (loadOp LOAD instead of the blue clear). This is real render-target memory semantics: a game
// pass that draws into a target it (or an earlier submit) already rendered composites OVER that
// content — without it every pass starts from the diagnostic blue clear, so cross-submit
// accumulation (UE4's UI-onto-backbuffer after a separate composite submit) is lost. Null (the
// default) keeps the blue-clear behavior byte-identical for every existing caller.
// `clear_rgba` (optional): 4 floats (RGBA, Vulkan order) to clear the color attachment to when no
// seed is supplied. Null keeps the legacy diagnostic blue — every test harness caller passes null,
// so their behavior is byte-identical. The live renderer passes the game's decoded fast-clear color
// (or opaque black when none), so real frames no longer start from blue (#309). PROSPER_CLEAR_DEBUG
// forces the blue back on regardless, so unrendered areas can still be spotted during development.
// Persistent Vulkan context. Creating a fresh instance+device PER render_draws_rgba call dominated
// wall-clock — every submit paid full device init — which made a many-draw frame (real gameplay is
// hundreds of draws/submit) impossibly slow and blocked headless scene investigation (#320). Create the
// instance/physical-device/device/queue ONCE (lazy, thread-safe static init) and reuse it across every
// call. Per-call Vulkan resources are created independently and are retained until their direct call
// or explicit ordered submission batch completes. The context intentionally leaks at process exit.
// LIFETIME INVARIANT: this context is intentionally never destroyed (no destructor; the device and
// instance leak at process exit). The compute backend BORROWS this device (#1091) and its static
// destructor calls vkDestroyPipeline/vkFreeMemory on it at exit, with unspecified cross-TU
// destruction order. Adding a destructor here that destroys the device would therefore create an
// immediate use-after-free in ~VulkanComputeContext. Do not add one without first giving compute an
// explicit release-before-teardown handshake.
struct RenderVkCtx {
    VkInstance inst = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE; VkQueue queue = VK_NULL_HANDLE; uint32_t qfi = UINT32_MAX;
    VkDeviceSize storage_buffer_alignment = 1;
    bool aniso_enabled = false; float max_aniso_limit = 1.0f;
    bool depth_bias_clamp_enabled = false;   // VkPhysicalDeviceFeatures::depthBiasClamp (#1349)
    bool logic_op_enabled = false; bool ok = false;
    bool fragment_stores_atomics = false;
    // Per-draw "fragment funnel" diagnostic (PROSPER_DRAW_STATS): pipeline-statistics + precise
    // occlusion queries. Enabled at device creation only when advertised; inert otherwise.
    bool pipeline_stats_enabled = false;
    bool occlusion_precise = false;
    // Geometry-probe (PROSPER_GEOM_PROBE): VK_EXT_transform_feedback for capturing gl_Position.
    bool transform_feedback_enabled = false;
    bool subgroup_size_control = false;
    uint32_t min_subgroup_size = 0, max_subgroup_size = 0;
    VkShaderStageFlags required_subgroup_size_stages = 0;
    VkShaderStageFlags subgroup_stages = 0;
    VkSubgroupFeatureFlags subgroup_operations = 0;
    // Present unification (#1270): so prosper-app can adopt THIS device for its swapchain and blit the
    // renderer's front-buffer image straight to the screen (no 4K CPU round-trip). All additive and
    // only when advertised, so the headless test/screenshot path is byte-for-byte unchanged: on a
    // display-less target the surface instance-extensions and VK_KHR_swapchain are simply absent, these
    // stay false, and prosper-app falls back to its own separate present device + CPU pixels.
    bool present_surface_capable = false;   // instance enabled VK_KHR_surface (+ a platform surface ext)
    bool present_swapchain_capable = false; // device enabled VK_KHR_swapchain
    VkQueue present_queue = VK_NULL_HANDLE; // dedicated 2nd queue when the family has >=2, else == queue
    bool present_queue_shared = false;      // present_queue aliases the render queue -> submits need a mutex
};
inline const RenderVkCtx& render_vk_ctx() {
    static RenderVkCtx c = [] {
        RenderVkCtx r;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        // macOS: MoltenVK is linked directly, so VK_KHR_portability_enumeration is neither needed nor
        // accepted here (see prosper-app main.cpp). Only the device-level portability_subset matters.
        //
        // Present unification (#1270): enable the WSI surface instance-extensions when the loader
        // advertises them, so prosper-app can create its window surface on THIS instance and present on
        // this device (see present_surface_capable). "Add-if-available" only: a headless target (llvmpipe
        // CI, no display) advertises none, so the enabled list stays empty and instance creation is
        // exactly as before. This is created before SDL_Init in the app, so we can't consult
        // SDL_Vulkan_GetInstanceExtensions; instead enable every platform surface extension present and
        // let the app fall back to its own device if SDL still needs one we didn't get (#1270 R4).
        std::vector<const char*> inst_exts;
        {
            static const char* const kWsiExts[] = {
                "VK_KHR_surface",
#if defined(_WIN32)
                "VK_KHR_win32_surface",
#elif defined(__APPLE__)
                "VK_EXT_metal_surface", "VK_MVK_macos_surface",
#else
                "VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface",
#endif
            };
            uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
            std::vector<VkExtensionProperties> avail(nie);
            if (nie) vkEnumerateInstanceExtensionProperties(nullptr, &nie, avail.data());
            auto has = [&](const char* name) {
                for (const auto& e : avail) if (!strcmp(e.extensionName, name)) return true;
                return false;
            };
            bool have_surface = has("VK_KHR_surface");
            if (have_surface) {
                bool have_platform = false;
                for (const char* e : kWsiExts) {
                    if (has(e)) { inst_exts.push_back(e); if (strcmp(e, "VK_KHR_surface")) have_platform = true; }
                }
                // A bare VK_KHR_surface with no platform surface is useless for a window; require both.
                r.present_surface_capable = have_platform;
                if (!have_platform) inst_exts.clear();
            }
            if (!inst_exts.empty()) {
                ici.enabledExtensionCount = (uint32_t)inst_exts.size();
                ici.ppEnabledExtensionNames = inst_exts.data();
            }
        }
        // If the driver rejects the surface set (should not happen since each was advertised), retry with
        // no instance extensions so the headless render path never regresses on an unexpected loader.
        if (vkCreateInstance(&ici, nullptr, &r.inst) != VK_SUCCESS || !r.inst) {
            if (!inst_exts.empty()) {
                r.present_surface_capable = false;
                VkInstanceCreateInfo bare{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; bare.pApplicationInfo = &app;
                if (vkCreateInstance(&bare, nullptr, &r.inst) != VK_SUCCESS || !r.inst) return r;
            } else {
                return r;
            }
        }
        uint32_t nd = 0; vkEnumeratePhysicalDevices(r.inst, &nd, nullptr);
        if (!nd) return r;
        std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(r.inst, &nd, devs.data());
        const auto selection = prosper::frontend::select_vulkan_device(devs, VK_QUEUE_GRAPHICS_BIT);
        r.phys = selection.device;
        r.qfi = selection.queue_family;
        if (!r.phys || r.qfi == UINT32_MAX) return r;
        std::fprintf(stderr, "[render] Vulkan device: %s (%s)\n",
                     selection.properties.deviceName,
                     prosper::frontend::vulkan_device_type_name(selection.properties.deviceType));
        // Present unification (#1270): if the graphics family exposes a second queue, dedicate index 1
        // to prosper-app's present so the app's blit/present submits never contend the render queue's
        // external-synchronization. On a single-queue family (RADV STRIX_HALO is queueCount==1) the app
        // instead shares queue index 0 under a submit mutex. Requesting queueCount is harmless to the
        // headless path: the extra queue is simply never fetched by tests/screenshot.
        uint32_t family_queue_count = 1;
        {
            uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, nullptr);
            if (nq) {
                std::vector<VkQueueFamilyProperties> qfp(nq);
                vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, qfp.data());
                if (r.qfi < nq) family_queue_count = qfp[r.qfi].queueCount;
            }
        }
        float prio[2] = {1.0f, 1.0f};
        const uint32_t want_queues = family_queue_count >= 2 ? 2u : 1u;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = r.qfi; qci.queueCount = want_queues; qci.pQueuePriorities = prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        // robustBufferAccess: OOB storage-buffer accesses are well-defined (predicated ops on
        // narrowed-EXEC lanes can't fault).
        VkPhysicalDeviceFeatures feats{}; feats.robustBufferAccess = VK_TRUE; dci.pEnabledFeatures = &feats;
        // samplerAnisotropy (#275): enable only if advertised; maxSamplerAnisotropy is the clamp ceiling.
        VkPhysicalDeviceFeatures supported{}; vkGetPhysicalDeviceFeatures(r.phys, &supported);
        VkPhysicalDeviceProperties phys_props{}; vkGetPhysicalDeviceProperties(r.phys, &phys_props);
        r.storage_buffer_alignment = std::max<VkDeviceSize>(
            1, phys_props.limits.minStorageBufferOffsetAlignment);
        r.aniso_enabled = supported.samplerAnisotropy;
        r.depth_bias_clamp_enabled = supported.depthBiasClamp;
        if (r.depth_bias_clamp_enabled) feats.depthBiasClamp = VK_TRUE;
        r.logic_op_enabled = supported.logicOp;
        r.max_aniso_limit = phys_props.limits.maxSamplerAnisotropy;
        if (r.aniso_enabled) feats.samplerAnisotropy = VK_TRUE;
        if (r.logic_op_enabled) feats.logicOp = VK_TRUE;
        // Portable AMD P0/P10/P20 lowering inserts a descriptor-free geometry pass on devices that
        // lack explicit vertex-parameter fragment extensions. Core geometryShader is available on
        // the Linux llvmpipe headless target and the desktop Vulkan drivers we support.
        feats.geometryShader = supported.geometryShader;
        // Storage-image shaders declare the format-free read/write capabilities. Enable every
        // corresponding core feature advertised by the device; fragment/vertex stores additionally
        // need their pipeline-stage store features.
        feats.shaderStorageImageReadWithoutFormat = supported.shaderStorageImageReadWithoutFormat;
        feats.shaderStorageImageWriteWithoutFormat = supported.shaderStorageImageWriteWithoutFormat;
        feats.vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics;
        feats.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics;
        r.fragment_stores_atomics = supported.fragmentStoresAndAtomics;
        // Per-draw fragment-funnel diagnostic (PROSPER_DRAW_STATS): pipeline statistics + precise
        // occlusion. Enable only when advertised; costs nothing unless the diagnostic is used.
        feats.pipelineStatisticsQuery = supported.pipelineStatisticsQuery;
        r.pipeline_stats_enabled = supported.pipelineStatisticsQuery;
        if (supported.occlusionQueryPrecise) { feats.occlusionQueryPrecise = VK_TRUE; r.occlusion_precise = true; }
        // robustImageAccess (VK_EXT_image_robustness): OpImageRead OOB must return zero (#131). Guarded.
        // Device extensions accumulate into a vector so the (optional) image-robustness and (macOS)
        // portability-subset extensions coexist.
        std::vector<const char*> dev_exts;
        VkPhysicalDeviceImageRobustnessFeaturesEXT irf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_ROBUSTNESS_FEATURES_EXT};
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup_properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup_core_properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceTransformFeedbackFeaturesEXT tf_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT};
        { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, nullptr);
          std::vector<VkExtensionProperties> de(ne);
          vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &ne, de.data());
          for (uint32_t i = 0; i < ne; i++) {
              if (!strcmp(de[i].extensionName, "VK_EXT_image_robustness")) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &irf; vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (irf.robustImageAccess) {
                      irf.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &irf;
                      dev_exts.push_back("VK_EXT_image_robustness");
                  }
              }
              if (!strcmp(de[i].extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &subgroup_features;
                  vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (subgroup_features.subgroupSizeControl) {
                      VkPhysicalDeviceProperties2 p2{
                          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                      p2.pNext = &subgroup_core_properties;
                      subgroup_core_properties.pNext = &subgroup_properties;
                      vkGetPhysicalDeviceProperties2(r.phys, &p2);
                      subgroup_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &subgroup_features;
                      dev_exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
                      r.subgroup_size_control = true;
                      r.min_subgroup_size = subgroup_properties.minSubgroupSize;
                      r.max_subgroup_size = subgroup_properties.maxSubgroupSize;
                      r.required_subgroup_size_stages =
                          subgroup_properties.requiredSubgroupSizeStages;
                      r.subgroup_stages = subgroup_core_properties.supportedStages;
                      r.subgroup_operations = subgroup_core_properties.supportedOperations;
                  }
              }
              // Geometry-probe (PROSPER_GEOM_PROBE): transform feedback to capture gl_Position. Enable
              // only the base transformFeedback feature (VS-only capture; geometryStreams not needed).
              if (!strcmp(de[i].extensionName, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME)) {
                  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                  f2.pNext = &tf_features; vkGetPhysicalDeviceFeatures2(r.phys, &f2);
                  if (tf_features.transformFeedback) {
                      tf_features.geometryStreams = VK_FALSE;
                      tf_features.pNext = const_cast<void*>(dci.pNext);
                      dci.pNext = &tf_features;
                      dev_exts.push_back(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
                      r.transform_feedback_enabled = true;
                  }
              }
              // Present unification (#1270): the swapchain device-extension, only when the instance is
              // surface-capable. Enabling it on the headless render device is harmless (no swapchain is
              // ever created there by tests/screenshot); prosper-app needs it to create its swapchain on
              // this shared device. Gated on present_surface_capable so a display-less build never
              // requests it.
              if (r.present_surface_capable &&
                  !strcmp(de[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                  dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
                  r.present_swapchain_capable = true;
              }
#ifdef __APPLE__
              // Spec-mandated: must be enabled when advertised (MoltenVK always advertises it).
              if (!strcmp(de[i].extensionName, "VK_KHR_portability_subset")) dev_exts.push_back("VK_KHR_portability_subset");
#endif
          } }
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
        if (vkCreateDevice(r.phys, &dci, nullptr, &r.dev) != VK_SUCCESS || !r.dev) return r;
        vkGetDeviceQueue(r.dev, r.qfi, 0, &r.queue);
        // Present unification (#1270): resolve the queue prosper-app will use to present. A dedicated
        // second queue (when the family has >=2) avoids contending the render queue; otherwise the app
        // shares queue 0 under a submit mutex (present_queue_shared). present capability requires the
        // swapchain extension to have been enabled above.
        if (r.present_swapchain_capable) {
            if (want_queues >= 2) {
                vkGetDeviceQueue(r.dev, r.qfi, 1, &r.present_queue);
                r.present_queue_shared = false;
            } else {
                r.present_queue = r.queue;
                r.present_queue_shared = true;
            }
        }
        r.ok = true;
        // Publish this device so the compute backend can adopt it instead of creating a second one
        // (#1091). Only the features compute needs are advertised; it declines the shared context if
        // they are missing and creates its own device exactly as before. Graphics and compute run
        // strictly sequentially on one thread, so sharing the queue needs no extra synchronization.
        if (!std::getenv("PROSPER_NO_SHARED_VULKAN_DEVICE")) {
            prosper::gpu::SharedVulkanContext shared;
            shared.instance = r.inst;
            shared.physical = r.phys;
            shared.device = r.dev;
            shared.queue = r.queue;
            shared.queue_family = r.qfi;
            // Publish what the device was actually CREATED with (feats), not what the physical
            // device merely supports. They are equal today because feats.x = supported.x
            // unconditionally, but the consumer's contract is "enabled here" -- if these ever became
            // gated like aniso/logicOp are, publishing `supported` would let compute emit shaders
            // declaring a capability the device never enabled.
            shared.storage_image_read_without_format = feats.shaderStorageImageReadWithoutFormat;
            shared.storage_image_write_without_format = feats.shaderStorageImageWriteWithoutFormat;
            // Present unification (#1270): advertise present adoption only when the instance is
            // surface-capable AND the device enabled VK_KHR_swapchain AND a present queue was resolved.
            shared.present_capable = r.present_surface_capable && r.present_swapchain_capable &&
                                     r.present_queue != VK_NULL_HANDLE;
            shared.present_queue = r.present_queue;
            shared.present_queue_shared = r.present_queue_shared;
            prosper::gpu::set_shared_vulkan_context(shared);
        }
        return r;
    }();
    return c;
}

// Present unification (#1270): serialize a single queue CALL against prosper-app's present submits when
// they share one VkQueue, and only once the app has adopted the shared queue (shared_present_active());
// otherwise it is a plain call after an acquire atomic load, so the headless/test/screenshot path and
// every non-shared device are unaffected. vkQueueSubmit returns without waiting for GPU work; the
// wait-idle wrapper (used only on the batch fence-timeout and compute-drain ERROR paths) does drain the
// queue under the lock, which briefly blocks the peer thread's submits -- acceptable on those rare paths.
inline VkResult render_locked_queue_submit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f) {
    if (prosper::gpu::shared_present_active()) {
        std::lock_guard<std::mutex> lk(prosper::gpu::shared_present_submit_mutex());
        return vkQueueSubmit(q, n, s, f);
    }
    return vkQueueSubmit(q, n, s, f);
}
inline VkResult render_locked_queue_wait_idle(VkQueue q) {
    if (prosper::gpu::shared_present_active()) {
        std::lock_guard<std::mutex> lk(prosper::gpu::shared_present_submit_mutex());
        return vkQueueWaitIdle(q);
    }
    return vkQueueWaitIdle(q);
}

struct BackendSubmissionBatchResult {
    VkResult submit_result = VK_SUCCESS;
    VkResult wait_result = VK_SUCCESS;
    uint64_t command_buffers = 0;
    uint64_t queue_submits = 0;
    uint64_t fence_waits = 0;
};

// Collect command buffers that belong to one ordered renderer callback. Vulkan queue order preserves
// target producer/consumer dependencies; one fence on the final submission is enough to retain every
// referenced object until the complete callback has finished. Direct test callers keep the established
// synchronous behavior by omitting this object.
class BackendSubmissionBatch {
public:
    BackendSubmissionBatch() = default;
    BackendSubmissionBatch(const BackendSubmissionBatch&) = delete;
    BackendSubmissionBatch& operator=(const BackendSubmissionBatch&) = delete;

    ~BackendSubmissionBatch() {
        if (!commands_.empty()) {
            const RenderVkCtx& ctx = render_vk_ctx();
            if (ctx.ok) (void)submit_and_wait(ctx.dev, ctx.queue, false);
            else discard();
        }
        if (commands_.empty()) complete();
    }

    bool pending() const { return !commands_.empty(); }

    void enqueue(VkCommandBuffer command) {
        commands_.push_back(command);
    }

    void add_cleanup(std::function<void()> cleanup) {
        cleanups_.push_back(std::move(cleanup));
    }

    // Persistent attachment state is updated speculatively so later command buffers in the same
    // ordered batch can LOAD/sample earlier results. If the batch cannot be submitted and completed,
    // every touched entry must be invalidated before its retained resources are released.
    void add_failure_cleanup(std::function<void()> cleanup) {
        failure_cleanups_.push_back(std::move(cleanup));
    }

    void discard() {
        commands_.clear();
        finish_persistent_state(false);
    }

    BackendSubmissionBatchResult submit_and_wait(VkDevice dev, VkQueue queue,
                                                  bool backend_trace) {
        BackendSubmissionBatchResult result;
        result.command_buffers = commands_.size();
        if (commands_.empty()) return result;

        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = static_cast<uint32_t>(commands_.size());
        submit.pCommandBuffers = commands_.data();
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        if (vkCreateFence(dev, &fence_info, nullptr, &fence) != VK_SUCCESS || !fence) {
            result.submit_result = VK_ERROR_INITIALIZATION_FAILED;
            discard();
            return result;
        }
        if (backend_trace) {
            std::fprintf(stderr, "[backend-trace] queue-submit begin command_buffers=%zu\n",
                         commands_.size());
            std::fflush(stderr);
        }
        result.submit_result = render_locked_queue_submit(queue, 1, &submit, fence);
        result.queue_submits = 1;
        if (backend_trace) {
            std::fprintf(stderr,
                         "[backend-trace] queue-submit end result=%d; fence-wait begin\n",
                         static_cast<int>(result.submit_result));
            std::fflush(stderr);
        }
        if (result.submit_result == VK_SUCCESS) {
            result.wait_result = vkWaitForFences(
                dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
            result.fence_waits = 1;
            // Preserve lifetime safety even if the bounded diagnostic wait expires.
            if (result.wait_result != VK_SUCCESS)
                result.wait_result = render_locked_queue_wait_idle(queue);
        }
        if (backend_trace) {
            std::fprintf(stderr, "[backend-trace] fence-wait end result=%d\n",
                         static_cast<int>(result.wait_result));
            std::fflush(stderr);
        }
        vkDestroyFence(dev, fence, nullptr);
        commands_.clear();
        finish_persistent_state(result.submit_result == VK_SUCCESS &&
                                result.wait_result == VK_SUCCESS);
        return result;
    }

    void complete() {
        for (auto& cleanup : cleanups_) cleanup();
        cleanups_.clear();
    }

private:
    void finish_persistent_state(bool completed) {
        if (!completed)
            for (auto cleanup = failure_cleanups_.rbegin();
                 cleanup != failure_cleanups_.rend(); ++cleanup)
                (*cleanup)();
        failure_cleanups_.clear();
    }

    std::vector<VkCommandBuffer> commands_;
    std::vector<std::function<void()>> cleanups_;
    std::vector<std::function<void()>> failure_cleanups_;
};

struct PersistentColorTargetKey {
    uint64_t id = 0;
    uint32_t width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool operator==(const PersistentColorTargetKey&) const = default;
};

struct PersistentColorTargetKeyHash {
    size_t operator()(const PersistentColorTargetKey& key) const {
        size_t h = std::hash<uint64_t>{}(key.id);
        auto mix = [&](uint32_t value) {
            h ^= static_cast<size_t>(value) + 0x9e3779b9u + (h << 6) + (h >> 2);
        };
        mix(key.width); mix(key.height); mix(static_cast<uint32_t>(key.format));
        return h;
    }
};

struct PersistentColorTargetImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceSize bytes = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint64_t last_use = 0;
    uint32_t pin_count = 0;
    bool valid = false;
};

inline std::unordered_map<PersistentColorTargetKey, PersistentColorTargetImage,
                          PersistentColorTargetKeyHash>& persistent_color_target_cache() {
    static std::unordered_map<PersistentColorTargetKey, PersistentColorTargetImage,
                              PersistentColorTargetKeyHash> cache;
    return cache;
}

inline VkDeviceSize& persistent_color_target_bytes() {
    static VkDeviceSize bytes = 0;
    return bytes;
}

inline uint64_t& persistent_color_target_generation() {
    static uint64_t generation = 0;
    return generation;
}

// Device-derived residency budget, initialized once from the physical device's largest DEVICE_LOCAL
// heap (see init_persistent_color_target_device_budget). 0 = not yet initialized -> the 256 MiB legacy
// fallback is used until the first render call has a VkPhysicalDevice in hand.
inline VkDeviceSize& persistent_color_target_device_budget() {
    static VkDeviceSize budget = 0;
    return budget;
}

// The historical fixed 256 MiB budget holds only ~4-8 targets at 4K (33 MiB RGBA8 / 66 MiB RGBA16F),
// so a deferred renderer's targets overflow it, are recreated non-resident, and every later sample of
// them falls to CPU detile (the #1177 bottleneck). Since a GPU-resident sample and a CPU detile produce
// identical pixels, growing this budget is correctness-preserving and only trades a bounded amount of
// device memory for far less per-frame CPU work. Precedence: explicit PROSPER_BACKEND_TARGET_CACHE_MB,
// then the device-derived budget, then the 256 MiB fallback.
inline VkDeviceSize persistent_color_target_limit() {
    static const bool have_env = getenv("PROSPER_BACKEND_TARGET_CACHE_MB") != nullptr;
    static const VkDeviceSize env_limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_TARGET_CACHE_MB");
        const uint64_t mib = value ? strtoull(value, nullptr, 10) : 256ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    if (have_env) return env_limit;
    const VkDeviceSize dev = persistent_color_target_device_budget();
    return dev ? dev : static_cast<VkDeviceSize>(256ull * 1024ull * 1024ull);
}

// Size the default residency budget to a quarter of the largest device-local heap, clamped to
// [256 MiB, 4 GiB]. On an integrated GPU (shared system RAM, a very large heap) this lands at the 4 GiB
// ceiling; on a small discrete GPU it stays proportional so we never claim more than ~25% of VRAM.
// Idempotent: only the first caller (with a valid device) sets it; PROSPER_BACKEND_TARGET_CACHE_MB
// overrides it entirely (handled in persistent_color_target_limit).
inline void init_persistent_color_target_device_budget(
    const VkPhysicalDeviceMemoryProperties& memp) {
    if (persistent_color_target_device_budget() != 0) return;
    VkDeviceSize heap = 0;
    for (uint32_t i = 0; i < memp.memoryHeapCount; i++)
        if (memp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heap = std::max(heap, memp.memoryHeaps[i].size);
    VkDeviceSize budget = heap / 4;
    budget = std::max<VkDeviceSize>(budget, 256ull * 1024ull * 1024ull);
    budget = std::min<VkDeviceSize>(budget, 4096ull * 1024ull * 1024ull);
    persistent_color_target_device_budget() = budget;
    fprintf(stderr, "[render] persistent color-target residency budget = %llu MiB "
            "(device-local heap %llu MiB)\n",
            (unsigned long long)(budget / (1024ull * 1024ull)),
            (unsigned long long)(heap / (1024ull * 1024ull)));
}

// Max number of distinct render targets kept GPU-resident (sampleable) at once. A target that would
// exceed this count is created non-resident (no SAMPLED_BIT), forcing every later sample of it onto the
// CPU detile path. At 4K a deferred renderer (G-buffer MRTs + lighting + post chain) easily needs more
// than the historical 64, so this is configurable via PROSPER_BACKEND_TARGET_CACHE_COUNT (see #1177).
inline size_t persistent_color_target_count_limit() {
    static const size_t count = []() -> size_t {
        const char* value = getenv("PROSPER_BACKEND_TARGET_CACHE_COUNT");
        // Raised from the historical 64 so a 4K deferred renderer's targets are bounded by the memory
        // budget (persistent_color_target_limit) rather than an arbitrarily small count (#1177).
        const uint64_t n = value ? strtoull(value, nullptr, 10) : 256ull;
        return n ? static_cast<size_t>(n) : size_t{256};
    }();
    return count;
}

inline PersistentColorTargetImage* find_persistent_color_target(
    uint64_t id, uint32_t width, uint32_t height, VkFormat format, bool require_valid = true) {
    if (!id) return nullptr;
    auto& cache = persistent_color_target_cache();
    auto found = cache.find({id, width, height, format});
    if (found == cache.end() || (require_valid && !found->second.valid)) return nullptr;
    return &found->second;
}

// Ordered frontend spans may need a GPU-only target to survive later backend calls until the submit's
// final callback materializes or presents it. Pins are explicit and short-lived; invalidation still
// makes the pixels unusable, while eviction waits until every owner releases the allocation.
inline bool pin_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                        VkFormat format) {
    PersistentColorTargetImage* target = find_persistent_color_target(
        id, width, height, backend_color_format(format));
    if (!target || target->pin_count == UINT32_MAX) return false;
    ++target->pin_count;
    return true;
}

inline bool unpin_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                          VkFormat format) {
    auto& cache = persistent_color_target_cache();
    auto found = cache.find({id, width, height, backend_color_format(format)});
    if (found == cache.end() || !found->second.pin_count) return false;
    --found->second.pin_count;
    return true;
}

// Guest/compute writes invalidate the GPU-produced version without immediately freeing the image.
// The next render may reuse the allocation, but it cannot LOAD or sample stale pixels.
inline void invalidate_persistent_color_target(uint64_t id) {
    if (!id) return;
    for (auto& [key, target] : persistent_color_target_cache())
        if (key.id == id) target.valid = false;
}

inline void invalidate_persistent_color_target_guest_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    const uint64_t end = size > UINT64_MAX - addr ? UINT64_MAX : addr + size;
    for (auto& [key, target] : persistent_color_target_cache()) {
        const uint64_t bytes = static_cast<uint64_t>(key.width) * key.height *
                               backend_color_bytes_per_pixel(key.format);
        const uint64_t target_end = bytes > UINT64_MAX - key.id ? UINT64_MAX : key.id + bytes;
        if (addr < target_end && key.id < end) target.valid = false;
    }
}

inline void destroy_persistent_color_target(const RenderVkCtx& ctx,
                                            PersistentColorTargetImage& target) {
    if (target.view) vkDestroyImageView(ctx.dev, target.view, nullptr);
    if (target.image) vkDestroyImage(ctx.dev, target.image, nullptr);
    if (target.memory) vkFreeMemory(ctx.dev, target.memory, nullptr);
    target = {};
}

inline bool readback_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                             VkFormat format, std::vector<uint8_t>& output,
                                             std::string& error);

// With deferred RTT readback (#1284) a valid persistent target can hold the ONLY copy of its
// rendered pixels. Eviction must hand those pixels back to the frontend's CPU cache before the
// image is destroyed, or the content is silently lost. Without a registered sink, eviction keeps
// the historical destroy-only behavior (every pass then still has an authoritative CPU copy).
inline std::function<void(uint64_t, uint32_t, uint32_t, VkFormat, std::vector<uint8_t>&&)>&
persistent_color_target_evict_sink() {
    static std::function<void(uint64_t, uint32_t, uint32_t, VkFormat, std::vector<uint8_t>&&)>
        sink;
    return sink;
}

inline bool evict_persistent_color_target(const RenderVkCtx& ctx, uint64_t current_generation) {
    auto& cache = persistent_color_target_cache();
    auto victim = cache.end();
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->second.last_use == current_generation || it->second.pin_count) continue;
        if (victim == cache.end() || it->second.last_use < victim->second.last_use)
            victim = it;
    }
    if (victim == cache.end()) return false;
    if (victim->second.valid) {
        auto& sink = persistent_color_target_evict_sink();
        if (sink) {
            std::vector<uint8_t> pixels;
            std::string error;
            if (readback_persistent_color_target(victim->first.id, victim->first.width,
                                                 victim->first.height, victim->first.format,
                                                 pixels, error))
                sink(victim->first.id, victim->first.width, victim->first.height,
                     victim->first.format, std::move(pixels));
        }
    }
    persistent_color_target_bytes() -= victim->second.bytes;
    destroy_persistent_color_target(ctx, victim->second);
    cache.erase(victim);
    return true;
}

// Transient allocations return to this pool only after their call or explicit submission batch has
// completed, so they can be safely recycled by exact Vulkan memory requirements. Keeping only the
// memory object avoids changing image layouts or descriptor lifetimes while removing the driver's
// expensive allocate/free churn between batches.
struct RenderMemoryKey {
    VkDeviceSize bytes = 0;
    uint32_t memory_type = UINT32_MAX;
    bool operator==(const RenderMemoryKey& other) const {
        return bytes == other.bytes && memory_type == other.memory_type;
    }
};

struct RenderMemoryKeyHash {
    size_t operator()(const RenderMemoryKey& key) const {
        return std::hash<uint64_t>{}(static_cast<uint64_t>(key.bytes)) ^
               (std::hash<uint32_t>{}(key.memory_type) << 1);
    }
};

struct RenderMemoryPool {
    std::mutex mutex;
    std::unordered_map<RenderMemoryKey, std::vector<VkDeviceMemory>, RenderMemoryKeyHash> available;
    std::unordered_map<VkDeviceMemory, RenderMemoryKey> active;
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

struct RenderMemoryPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

inline RenderMemoryPool& render_memory_pool() {
    static RenderMemoryPool pool;
    return pool;
}

inline bool render_memory_pool_enabled() {
    static const bool enabled = getenv("PROSPER_NO_MEMORY_POOL") == nullptr;
    return enabled;
}

inline VkDeviceSize render_memory_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_MEMORY_POOL_MB");
        const uint64_t mib = value ? strtoull(value, nullptr, 10) : 512ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

inline VkDeviceMemory allocate_transient_render_memory(VkDevice device, VkDeviceSize bytes,
                                                       uint32_t memory_type) {
    if (memory_type == UINT32_MAX) return VK_NULL_HANDLE;
    RenderMemoryKey key{bytes, memory_type};
    if (render_memory_pool_enabled()) {
        RenderMemoryPool& pool = render_memory_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        auto found = pool.available.find(key);
        if (found != pool.available.end() && !found->second.empty()) {
            VkDeviceMemory memory = found->second.back();
            found->second.pop_back();
            if (found->second.empty()) pool.available.erase(found);
            pool.cached_bytes -= bytes;
            --pool.cached_allocations;
            ++pool.hits;
            pool.active.emplace(memory, key);
            return memory;
        }
        ++pool.misses;
    }

    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = bytes;
    allocation.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &allocation, nullptr, &memory) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (render_memory_pool_enabled()) {
        RenderMemoryPool& pool = render_memory_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        pool.active.emplace(memory, key);
    }
    return memory;
}

inline void release_transient_render_memory(VkDevice device, VkDeviceMemory memory) {
    if (!memory) return;
    if (!render_memory_pool_enabled()) {
        vkFreeMemory(device, memory, nullptr);
        return;
    }

    RenderMemoryPool& pool = render_memory_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto found = pool.active.find(memory);
    if (found == pool.active.end()) {
        vkFreeMemory(device, memory, nullptr);
        return;
    }
    const RenderMemoryKey key = found->second;
    pool.active.erase(found);
    constexpr size_t max_cached_allocations = 4096;
    const VkDeviceSize limit = render_memory_pool_limit();
    const VkDeviceSize remaining = pool.cached_bytes < limit ? limit - pool.cached_bytes : 0;
    if (pool.cached_allocations >= max_cached_allocations ||
        key.bytes > remaining) {
        ++pool.discarded;
        vkFreeMemory(device, memory, nullptr);
        return;
    }
    pool.available[key].push_back(memory);
    pool.cached_bytes += key.bytes;
    ++pool.cached_allocations;
}

inline RenderMemoryPoolStats render_memory_pool_stats() {
    RenderMemoryPool& pool = render_memory_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    return {pool.cached_bytes, pool.cached_allocations, pool.hits, pool.misses, pool.discarded};
}

inline uint32_t render_memory_type(VkPhysicalDevice phys, uint32_t bits,
                                   VkMemoryPropertyFlags wanted) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(phys, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return UINT32_MAX;
}

// Storage-buffer contents are rewritten for every synchronous render call, but their Vulkan object
// shapes repeat heavily. Keep capacity-class host-coherent buffers mapped between calls so the hot path
// only copies bytes. The backend normally packs call-local logical uploads into aligned slices of a few
// pooled arenas; the same pool also backs the per-upload fallback. A call or explicit submission batch
// completes before returning buffers, so no in-flight GPU work can observe a later upload. Descriptors
// retain exact logical offsets and ranges, so capacity padding and neighboring arena slices remain
// shader-inaccessible.
struct RenderHostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize bytes = 0;
    VkDeviceSize allocation_bytes = 0;
};

struct RenderHostBufferPool {
    std::unordered_map<VkDeviceSize, std::vector<RenderHostBuffer>> available;
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

struct RenderHostBufferPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_buffers = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

inline RenderHostBufferPool& render_host_buffer_pool() {
    static thread_local RenderHostBufferPool pool;
    return pool;
}

inline bool render_host_buffer_pool_enabled() {
    return getenv("PROSPER_NO_BACKEND_BUFFER_POOL") == nullptr;
}

inline VkDeviceSize render_host_buffer_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_BUFFER_POOL_MB");
        const uint64_t mib = value ? strtoull(value, nullptr, 10) : 256ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

inline VkDeviceSize render_host_buffer_arena_size() {
    static const VkDeviceSize bytes = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_BUFFER_ARENA_KB");
        const uint64_t kib = value ? strtoull(value, nullptr, 10) : 1024ull;
        if (kib > UINT64_MAX / 1024ull) return VkDeviceSize{UINT64_MAX};
        return std::max<VkDeviceSize>(4, static_cast<VkDeviceSize>(kib) * 1024ull);
    }();
    return bytes;
}

inline void destroy_render_host_buffer(VkDevice device, RenderHostBuffer& buffer) {
    if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
    buffer = {};
}

inline RenderHostBuffer acquire_render_host_buffer(const RenderVkCtx& ctx,
                                                   VkDeviceSize bytes) {
    if (!bytes) return {};
    VkDeviceSize capacity = 4;
    while (capacity < bytes && capacity <= UINT64_MAX / 2) capacity *= 2;
    if (capacity < bytes) capacity = bytes;
    RenderHostBufferPool& pool = render_host_buffer_pool();
    auto found = pool.available.find(capacity);
    if (found != pool.available.end() && !found->second.empty()) {
        RenderHostBuffer buffer = found->second.back();
        found->second.pop_back();
        if (found->second.empty()) pool.available.erase(found);
        pool.cached_bytes -= buffer.allocation_bytes;
        --pool.cached_buffers;
        ++pool.hits;
        return buffer;
    }
    ++pool.misses;

    RenderHostBuffer buffer;
    buffer.bytes = capacity;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = capacity;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &buffer.buffer) != VK_SUCCESS)
        return {};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer.buffer, &requirements);
    buffer.allocation_bytes = requirements.size;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = render_memory_type(
        ctx.phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (allocation.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(ctx.dev, &allocation, nullptr, &buffer.memory) != VK_SUCCESS ||
        vkBindBufferMemory(ctx.dev, buffer.buffer, buffer.memory, 0) != VK_SUCCESS ||
        vkMapMemory(ctx.dev, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped) != VK_SUCCESS) {
        destroy_render_host_buffer(ctx.dev, buffer);
        return {};
    }
    return buffer;
}

inline void release_render_host_buffer(VkDevice device, RenderHostBuffer buffer) {
    if (!buffer.buffer || !buffer.memory || !buffer.mapped) {
        destroy_render_host_buffer(device, buffer);
        return;
    }
    constexpr size_t max_cached_buffers = 4096;
    const VkDeviceSize limit = render_host_buffer_pool_limit();
    if (!limit || buffer.allocation_bytes > limit) {
        destroy_render_host_buffer(device, buffer);
        return;
    }

    std::vector<RenderHostBuffer> evicted;
    RenderHostBufferPool& pool = render_host_buffer_pool();
    while ((pool.cached_buffers >= max_cached_buffers ||
            pool.cached_bytes > limit - buffer.allocation_bytes) &&
           !pool.available.empty()) {
        auto victim = pool.available.begin();
        RenderHostBuffer old = victim->second.back();
        victim->second.pop_back();
        if (victim->second.empty()) pool.available.erase(victim);
        pool.cached_bytes -= old.allocation_bytes;
        --pool.cached_buffers;
        ++pool.evictions;
        evicted.push_back(old);
    }
    if (pool.cached_buffers < max_cached_buffers &&
        pool.cached_bytes <= limit - buffer.allocation_bytes) {
        pool.available[buffer.bytes].push_back(buffer);
        pool.cached_bytes += buffer.allocation_bytes;
        ++pool.cached_buffers;
        buffer = {};
    }
    for (RenderHostBuffer& old : evicted) destroy_render_host_buffer(device, old);
    if (buffer.buffer) destroy_render_host_buffer(device, buffer);
}

inline const RenderHostBuffer& render_internal_gds_buffer() {
    static const RenderHostBuffer buffer = [] {
        constexpr VkDeviceSize kGdsBytes = 64u * 1024u;
        RenderHostBuffer result = acquire_render_host_buffer(render_vk_ctx(), kGdsBytes);
        if (result.mapped) std::memset(result.mapped, 0, static_cast<size_t>(kGdsBytes));
        return result;
    }();
    return buffer;
}

inline void reset_internal_gds_for_test() {
    const RenderHostBuffer& buffer = render_internal_gds_buffer();
    if (buffer.mapped) std::memset(buffer.mapped, 0, 64u * 1024u);
}

inline uint32_t read_internal_gds_for_test(uint32_t byte_offset) {
    const RenderHostBuffer& buffer = render_internal_gds_buffer();
    if (!buffer.mapped || byte_offset > 64u * 1024u - sizeof(uint32_t)) return 0;
    uint32_t value = 0;
    std::memcpy(&value, static_cast<const uint8_t*>(buffer.mapped) + byte_offset, sizeof(value));
    return value;
}

inline RenderHostBufferPoolStats render_host_buffer_pool_stats() {
    RenderHostBufferPool& pool = render_host_buffer_pool();
    return {pool.cached_bytes, pool.cached_buffers, pool.hits, pool.misses, pool.evictions};
}

struct PersistentDsKey {
    uint64_t dr = 0, dw = 0, sr = 0, sw = 0, htile = 0;
    uint32_t w = 0, h = 0, fmt = 0;
    bool operator==(const PersistentDsKey& o) const {
        return dr == o.dr && dw == o.dw && sr == o.sr && sw == o.sw && htile == o.htile &&
               w == o.w && h == o.h && fmt == o.fmt;
    }
};

struct PersistentDsKeyHash {
    size_t operator()(const PersistentDsKey& k) const {
        size_t hash = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { hash ^= static_cast<size_t>(v); hash *= 1099511628211ull; };
        mix(k.dr); mix(k.dw); mix(k.sr); mix(k.sw); mix(k.htile);
        mix(k.w); mix(k.h); mix(k.fmt);
        return hash;
    }
};

struct PersistentDsImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool layout_initialized = false;
    bool depth_valid = false;
    bool stencil_valid = false;
    uint64_t last_depth_write = 0;   // sampled-bridge recency (#1275)
};

inline uint64_t& persistent_ds_write_generation() {
    static uint64_t generation = 0;
    return generation;
}

// DB_RENDER_CONTROL.DEPTH_CLEAR_ENABLE substitutes the VALUE of depth writes (DB_DEPTH_CLEAR); it
// does not create writes on its own — the write path still requires Z_ENABLE + Z_WRITE_ENABLE and a
// compare that can pass. Real guest clear draws program exactly that shape (test+write+ALWAYS);
// Blue Prince's per-light shadow loop instead issues a fullscreen rect with DEPTH_CLEAR_ENABLE set
// and DB_DEPTH_CONTROL fully disabled immediately BEFORE sampling the plane its shadow casters
// rendered — treating that as a clear/write destroyed the shadow map it is about to consume and
// banded every light (#1287). The guest's own draw order proves the writes-disabled shape must be
// depth-inert. CONFIDENCE: MED (write-path gating; the exercised title shapes are covered).
constexpr bool depth_clear_effective(bool clear_enabled, bool test_enabled, bool write_enabled,
                                     uint32_t compare_op) {
    return clear_enabled && test_enabled && write_enabled && compare_op != VK_COMPARE_OP_NEVER;
}

constexpr bool persistent_ds_pass_may_write_depth(bool clear_enabled, bool test_enabled,
                                                  bool write_enabled, uint32_t compare_op) {
    (void)clear_enabled;   // an effective clear already satisfies the write-path clause below
    return test_enabled && write_enabled && compare_op != VK_COMPARE_OP_NEVER;
}

// STENCIL_CLEAR_ENABLE is the stencil twin of the rule above (#1355): the bit substitutes the
// VALUE of stencil writes and only acts through the enabled stencil write path — STENCIL_ENABLE
// plus a nonzero write mask (driver stencil-clear draws program enable+ALWAYS+REPLACE+full mask).
// Op-level analysis (a KEEP-everywhere draw writes nothing) is deliberately omitted: requiring
// only enable+mask errs toward honoring clears, the safe direction for real guest clear shapes.
// CONFIDENCE: MED (write-path gating by analogy with the #1352 title evidence; no title observed
// exercising the writes-disabled stencil shape).
constexpr bool stencil_clear_effective(bool clear_enabled, bool stencil_enabled,
                                       uint32_t write_mask_front, uint32_t write_mask_back) {
    return clear_enabled && stencil_enabled && ((write_mask_front | write_mask_back) != 0u);
}

inline void note_persistent_ds_depth_write(PersistentDsImage& image, bool use_depth,
                                           bool depth_may_be_written) {
    if (use_depth && depth_may_be_written)
        image.last_depth_write = ++persistent_ds_write_generation();
}

inline std::unordered_map<PersistentDsKey, PersistentDsImage, PersistentDsKeyHash>&
persistent_ds_cache() {
    static std::unordered_map<PersistentDsKey, PersistentDsImage, PersistentDsKeyHash> cache;
    return cache;
}

// Sampled depth-plane lookup (#1275): a shadow-map / depth-pyramid T# addresses the DEPTH plane of
// a surface prosper rendered into a persistent Vulkan DS image and never wrote back to guest
// memory. Resolve that address (read or write base — the guest aliases both at the same plane) to
// the retained image so the consumer can bind it directly. Extent must match the T# exactly; only
// a valid, initialized depth plane may be sampled.
struct PersistentDsSampled {
    PersistentDsImage* image = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
};
inline PersistentDsSampled find_persistent_ds_sampled(uint64_t addr, uint32_t width,
                                                      uint32_t height) {
    if (!addr) return {};
    static const bool bridge_log = getenv("PROSPER_DSBRIDGE_LOG") != nullptr;
    // Prefer the most recently DEPTH-WRITTEN match: a surface re-keyed (D32 -> D32S8) keeps its
    // stale sibling entry, and unordered_map iteration order must not pick the winner.
    PersistentDsSampled best{};
    uint64_t best_write = 0;
    for (auto& [key, image] : persistent_ds_cache()) {
        if (key.w != width || key.h != height) continue;
        if (key.dr != addr && key.dw != addr) continue;
        if (!image.depth_valid || !image.layout_initialized || !image.image) continue;
        if (!best.image || image.last_depth_write > best_write) {
            best = {&image, static_cast<VkFormat>(key.fmt)};
            best_write = image.last_depth_write;
        }
    }
    if (best.image) {
        if (bridge_log) {
            static int hits = 0;
            if (hits++ < 8)
                fprintf(stderr, "[dsbridge] HIT addr=0x%llx %ux%u fmt=%u\n",
                        (unsigned long long)addr, width, height, (unsigned)best.format);
        }
        return best;
    }
    if (bridge_log) {
        static int misses = 0;
        if (misses++ < 8) {
            fprintf(stderr, "[dsbridge] miss addr=0x%llx %ux%u; cache:\n",
                    (unsigned long long)addr, width, height);
            for (const auto& [key, image] : persistent_ds_cache())
                fprintf(stderr, "[dsbridge]   dr=0x%llx dw=0x%llx %ux%u fmt=%u dvalid=%d init=%d\n",
                        (unsigned long long)key.dr, (unsigned long long)key.dw, key.w, key.h,
                        key.fmt, (int)image.depth_valid, (int)image.layout_initialized);
        }
    }
    return {};
}

inline bool guest_ranges_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
    if (!a || !a_size || !b || !b_size) return false;
    return a < b + b_size && b < a + a_size;
}

// The host image is deliberately detached from guest tiled depth/stencil memory. A guest-side GPU
// write to any plane base therefore makes the cached Vulkan contents stale. HTILE fast clears are
// especially important: Unity fills the metadata allocation without emitting DB_RENDER_CONTROL clear
// bits, and hardware treats the surface as cleared on its next use (#611).
inline size_t invalidate_persistent_ds_guest_write(uint64_t addr, uint64_t size) {
    if (!addr || !size) return 0;
    size_t invalidated = 0;
    for (auto& [key, image] : persistent_ds_cache()) {
        const uint64_t depth_size = static_cast<uint64_t>(key.w) * key.h * 4;
        const uint64_t stencil_size = static_cast<uint64_t>(key.w) * key.h;
        const uint64_t htile_blocks = static_cast<uint64_t>((key.w + 7) / 8) *
                                      ((key.h + 7) / 8);
        const uint64_t htile_size = (htile_blocks * 4 + 0x7fff) & ~0x7fffull;
        if (!guest_ranges_overlap(addr, size, key.dr, depth_size) &&
            !guest_ranges_overlap(addr, size, key.dw, depth_size) &&
            !guest_ranges_overlap(addr, size, key.sr, stencil_size) &&
            !guest_ranges_overlap(addr, size, key.sw, stencil_size) &&
            !guest_ranges_overlap(addr, size, key.htile, htile_size)) continue;
        image.depth_valid = false;
        image.stencil_valid = false;
        ++invalidated;
    }
    if (invalidated && getenv("PROSPER_DSLOG"))
        fprintf(stderr, "[ds] guest-write addr=%llx size=%llu invalidated=%zu\n",
                (unsigned long long)addr, (unsigned long long)size, invalidated);
    return invalidated;
}

// Make device writes visible before the CPU reads a mapped readback buffer. A HOST_CACHED memory
// type is not required to also be HOST_COHERENT, so every mapped READ of a TRANSFER_DST buffer must
// invalidate first. Specified as valid on coherent memory too, so callers can invoke it
// unconditionally rather than tracking the selected memory type.
// NOTE: this header contains a second readback allocator that requires HOST_COHERENT in every tier,
// so its mapped reads need no invalidate. Keep that requirement, or route its reads through this
// helper too -- relaxing it without adding invalidates would silently return stale bytes.
inline bool invalidate_mapped_readback(const RenderVkCtx& ctx, VkDeviceMemory memory) {
    if (!memory) return false;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;   // VK_WHOLE_SIZE is exempt from the nonCoherentAtomSize size rule
    return vkInvalidateMappedMemoryRanges(ctx.dev, 1, &range) == VK_SUCCESS;
}

// CONTRACT: a pure TRANSFER_DST (readback) buffer may be backed by HOST_CACHED memory that is NOT
// HOST_COHERENT. Call invalidate_mapped_readback() after mapping and before reading one, and do not
// host-WRITE through such a mapping without a flush. Any usage including TRANSFER_SRC is always
// backed by HOST_COHERENT memory, so host writes to it need no flush.
inline bool persistent_ds_transfer_buffer(const RenderVkCtx& ctx, VkDeviceSize bytes,
                                          VkBufferUsageFlags usage, VkBuffer& buffer,
                                          VkDeviceMemory& memory, std::string& error) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes; info.usage = usage;
    if (vkCreateBuffer(ctx.dev, &info, nullptr, &buffer) != VK_SUCCESS) {
        error = "cannot create persistent DS transfer buffer"; return false;
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx.dev, buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    // A pure TRANSFER_DST buffer is a GPU->CPU readback: the CPU READS every byte of it. Plain
    // HOST_VISIBLE|HOST_COHERENT memory is typically write-combined and uncached, which streams CPU
    // writes well but reads back at only a few hundred MB/s -- measured here at ~236 MB/s, making the
    // map+copy 99% of a color-target readback. Such a buffer therefore prefers HOST_CACHED so the
    // copy runs at cache speed. A buffer that is ALSO TRANSFER_SRC is host-written (an upload, or a
    // round trip), so it keeps the write-combined coherent selection that is right for writes and
    // needs no host flush.
    const bool readback = (usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0 &&
                          (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0;
    const uint32_t coherent_type = render_memory_type(
        ctx.phys, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t preferred = UINT32_MAX;
    if (readback) {
        preferred = render_memory_type(
            ctx.phys, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (preferred == UINT32_MAX)
            preferred = render_memory_type(
                ctx.phys, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    // Try the preferred (cached) type first, then the previously used coherent type. The retry
    // matters because a cached type can live in a small heap (BAR-style) that the coherent type does
    // not: without it, a cached-heap exhaustion would fail a readback that previously succeeded.
    auto try_allocate = [&](uint32_t type) {
        if (type == UINT32_MAX) return false;
        allocation.memoryTypeIndex = type;
        if (vkAllocateMemory(ctx.dev, &allocation, nullptr, &memory) != VK_SUCCESS) {
            memory = VK_NULL_HANDLE;
            return false;
        }
        if (vkBindBufferMemory(ctx.dev, buffer, memory, 0) != VK_SUCCESS) {
            vkFreeMemory(ctx.dev, memory, nullptr); memory = VK_NULL_HANDLE;
            return false;
        }
        return true;
    };
    if (!try_allocate(preferred) && !(preferred != coherent_type && try_allocate(coherent_type))) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr); buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;
        error = "cannot allocate persistent DS transfer buffer"; return false;
    }
    return true;
}

inline bool submit_persistent_ds_transfer(const RenderVkCtx& ctx, VkImage image,
                                          VkImageAspectFlags aspects, VkImageLayout old_layout,
                                          VkImageLayout transfer_layout, VkBuffer buffer,
                                          uint32_t width, uint32_t height,
                                          bool copy_depth, bool copy_stencil,
                                          bool upload, std::string& error) {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx.qfi;
    if (vkCreateCommandPool(ctx.dev, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        error = "cannot create persistent DS transfer command pool"; return false;
    }
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool; command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(ctx.dev, &command_info, &command) != VK_SUCCESS ||
        vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        vkDestroyCommandPool(ctx.dev, pool, nullptr);
        error = "cannot begin persistent DS transfer command"; return false;
    }
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = old_layout; barrier.newLayout = transfer_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspects, 0, 1, 0, 1};
    barrier.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 :
        (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    barrier.dstAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    const VkPipelineStageFlags source_stage = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
        ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    vkCmdPipelineBarrier(command, source_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    VkDeviceSize offset = 0;
    auto copy_plane = [&](VkImageAspectFlagBits aspect, VkDeviceSize bytes) {
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {static_cast<VkImageAspectFlags>(aspect), 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        if (upload)
            vkCmdCopyBufferToImage(command, buffer, image, transfer_layout, 1, &copy);
        else
            vkCmdCopyImageToBuffer(command, image, transfer_layout, buffer, 1, &copy);
        offset += bytes;
    };
    if (copy_depth) copy_plane(VK_IMAGE_ASPECT_DEPTH_BIT,
                               static_cast<VkDeviceSize>(width) * height * 4);
    if (copy_stencil) copy_plane(VK_IMAGE_ASPECT_STENCIL_BIT,
                                 static_cast<VkDeviceSize>(width) * height);

    barrier.oldLayout = transfer_layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const bool recorded = vkEndCommandBuffer(command) == VK_SUCCESS;
    const bool fenced = recorded &&
        vkCreateFence(ctx.dev, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    const bool submitted = fenced && render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS;
    bool finished = submitted &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (submitted && !finished) finished = render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS;
    if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
    vkDestroyCommandPool(ctx.dev, pool, nullptr);
    if (!finished) { error = "persistent DS transfer did not complete"; return false; }
    return true;
}

// Materialize a valid GPU-only color target on demand. Ordered DMA may consume a target in a later
// submit, which the producing render callback cannot predict; keeping the fast no-readback path and
// synchronizing only at that consumer preserves both the persistent-target contract and DMA versioning.
inline bool readback_persistent_color_target(uint64_t id, uint32_t width, uint32_t height,
                                             VkFormat format, std::vector<uint8_t>& output,
                                             std::string& error) {
    output.clear(); error.clear();
    format = backend_color_format(format);
    PersistentColorTargetImage* target = find_persistent_color_target(
        id, width, height, format);
    if (!target || !target->image || target->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        error = "persistent color target is unavailable";
        return false;
    }
    target->last_use = ++persistent_color_target_generation();
    const uint64_t texels = static_cast<uint64_t>(width) * height;
    const uint64_t bpp = backend_color_bytes_per_pixel(format);
    if (!width || !height || !bpp || texels > UINT64_MAX / bpp || texels * bpp > SIZE_MAX) {
        error = "persistent color target byte size is invalid";
        return false;
    }
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(texels * bpp);
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!persistent_ds_transfer_buffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       buffer, memory, error)) {
        error = "cannot allocate persistent color target readback buffer";
        return false;
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup = [&] {
        if (fence) vkDestroyFence(ctx.dev, fence, nullptr);
        if (pool) vkDestroyCommandPool(ctx.dev, pool, nullptr);
        if (buffer) vkDestroyBuffer(ctx.dev, buffer, nullptr);
        if (memory) vkFreeMemory(ctx.dev, memory, nullptr);
    };
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = ctx.qfi;
    if (vkCreateCommandPool(ctx.dev, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        cleanup(); error = "cannot create persistent color target readback command pool"; return false;
    }
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkAllocateCommandBuffers(ctx.dev, &command_info, &command) != VK_SUCCESS ||
        vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) {
        cleanup(); error = "cannot begin persistent color target readback command"; return false;
    }

    const VkImageLayout saved_layout = target->layout;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = saved_layout;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = target->image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(command, target->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = saved_layout;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const bool recorded = vkEndCommandBuffer(command) == VK_SUCCESS;
    const bool fenced = recorded &&
        vkCreateFence(ctx.dev, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    const bool submitted = fenced && render_locked_queue_submit(ctx.queue, 1, &submit, fence) == VK_SUCCESS;
    bool finished = submitted &&
        vkWaitForFences(ctx.dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000) == VK_SUCCESS;
    if (submitted && !finished) finished = render_locked_queue_wait_idle(ctx.queue) == VK_SUCCESS;
    if (!finished) {
        cleanup(); error = "persistent color target readback did not complete"; return false;
    }
    void* mapped = nullptr;
    if (vkMapMemory(ctx.dev, memory, 0, bytes, 0, &mapped) != VK_SUCCESS || !mapped) {
        cleanup(); error = "cannot map persistent color target readback"; return false;
    }
    if (!invalidate_mapped_readback(ctx, memory)) {
        vkUnmapMemory(ctx.dev, memory);
        cleanup(); error = "cannot invalidate persistent color target readback"; return false;
    }
    const auto* first = static_cast<const uint8_t*>(mapped);
    output.assign(first, first + static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.dev, memory);
    ++backend_color_target_stats_storage().readbacks;
    cleanup();
    return true;
}

inline bool snapshot_persistent_ds_images(std::vector<prosper::gpu::GpuCaptureDsSeed>& seeds,
                                          std::string& error) {
    error.clear(); seeds.clear();
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    for (const auto& [key, image] : persistent_ds_cache()) {
        if (!image.depth_valid && !image.stencil_valid) continue;
        if (!image.image || !image.layout_initialized) {
            error = "valid persistent DS cache entry has no initialized image"; return false;
        }
        prosper::gpu::GpuCaptureDsSeed seed;
        seed.depth_read_base = key.dr; seed.depth_write_base = key.dw;
        seed.stencil_read_base = key.sr; seed.stencil_write_base = key.sw;
        seed.htile_data_base = key.htile; seed.width = key.w; seed.height = key.h;
        if (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT))
            seed.format = prosper::gpu::GpuCaptureDsFormat::D32Float;
        else if (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT))
            seed.format = prosper::gpu::GpuCaptureDsFormat::D32FloatS8;
        else {
            error = "persistent DS cache uses an unsupported capture format"; return false;
        }
        seed.depth_valid = image.depth_valid;
        seed.stencil_valid = image.stencil_valid;
        const size_t depth_bytes = seed.depth_valid ? static_cast<size_t>(key.w) * key.h * 4 : 0;
        const size_t stencil_bytes = seed.stencil_valid ? static_cast<size_t>(key.w) * key.h : 0;
        VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
        if (!persistent_ds_transfer_buffer(ctx, depth_bytes + stencil_bytes,
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           buffer, memory, error)) return false;
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
            (key.fmt == static_cast<uint32_t>(VK_FORMAT_D32_SFLOAT_S8_UINT)
                 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        const bool copied = submit_persistent_ds_transfer(
            ctx, image.image, aspects, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, key.w, key.h,
            seed.depth_valid, seed.stencil_valid, false, error);
        void* mapped = nullptr;
        const bool mapped_ok = copied &&
            vkMapMemory(ctx.dev, memory, 0, depth_bytes + stencil_bytes, 0, &mapped) == VK_SUCCESS;
        const bool invalidated = mapped_ok && invalidate_mapped_readback(ctx, memory);
        if (invalidated) {
            const auto* bytes = static_cast<const uint8_t*>(mapped);
            seed.depth.assign(bytes, bytes + depth_bytes);
            seed.stencil.assign(bytes + depth_bytes, bytes + depth_bytes + stencil_bytes);
            vkUnmapMemory(ctx.dev, memory);
        }
        if (mapped_ok && !invalidated) vkUnmapMemory(ctx.dev, memory);
        vkDestroyBuffer(ctx.dev, buffer, nullptr); vkFreeMemory(ctx.dev, memory, nullptr);
        if (!invalidated) {
            if (error.empty())
                error = mapped_ok ? "cannot invalidate persistent DS readback"
                                  : "cannot map persistent DS readback";
            return false;
        }
        seeds.push_back(std::move(seed));
    }
    return true;
}

inline bool restore_persistent_ds_image(const prosper::gpu::GpuCaptureDsSeed& seed,
                                        std::string& error) {
    error.clear();
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) { error = "Vulkan renderer is unavailable"; return false; }
    const VkFormat format = seed.format == prosper::gpu::GpuCaptureDsFormat::D32Float
        ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D32_SFLOAT_S8_UINT;
    const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
        (format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    PersistentDsKey key{seed.depth_read_base, seed.depth_write_base,
                        seed.stencil_read_base, seed.stencil_write_base,
                        seed.htile_data_base, seed.width, seed.height,
                        static_cast<uint32_t>(format)};
    PersistentDsImage& image = persistent_ds_cache()[key];
    if (!image.image) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D; info.format = format;
        info.extent = {seed.width, seed.height, 1};
        info.mipLevels = 1; info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;   // sampled depth bridge (#1275)
        if (vkCreateImage(ctx.dev, &info, nullptr, &image.image) != VK_SUCCESS) {
            error = "cannot create restored persistent DS image"; return false;
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(ctx.dev, image.image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = render_memory_type(ctx.phys, requirements.memoryTypeBits, 0);
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image.image; view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format; view_info.subresourceRange = {aspects, 0, 1, 0, 1};
        if (allocation.memoryTypeIndex == UINT32_MAX ||
            vkAllocateMemory(ctx.dev, &allocation, nullptr, &image.memory) != VK_SUCCESS ||
            vkBindImageMemory(ctx.dev, image.image, image.memory, 0) != VK_SUCCESS ||
            vkCreateImageView(ctx.dev, &view_info, nullptr, &image.view) != VK_SUCCESS) {
            if (image.view) vkDestroyImageView(ctx.dev, image.view, nullptr);
            vkDestroyImage(ctx.dev, image.image, nullptr);
            if (image.memory) vkFreeMemory(ctx.dev, image.memory, nullptr);
            image = {};
            error = "cannot allocate restored persistent DS image"; return false;
        }
    }
    const size_t transfer_bytes = seed.depth.size() + seed.stencil.size();
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
    if (!persistent_ds_transfer_buffer(ctx, transfer_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                       buffer, memory, error)) return false;
    void* mapped = nullptr;
    if (vkMapMemory(ctx.dev, memory, 0, transfer_bytes, 0, &mapped) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.dev, buffer, nullptr); vkFreeMemory(ctx.dev, memory, nullptr);
        error = "cannot map persistent DS upload"; return false;
    }
    if (!seed.depth.empty()) std::memcpy(mapped, seed.depth.data(), seed.depth.size());
    if (!seed.stencil.empty())
        std::memcpy(static_cast<uint8_t*>(mapped) + seed.depth.size(),
                    seed.stencil.data(), seed.stencil.size());
    vkUnmapMemory(ctx.dev, memory);
    const VkImageLayout old_layout = image.layout_initialized
        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    const bool copied = submit_persistent_ds_transfer(
        ctx, image.image, aspects, old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        buffer, seed.width, seed.height, seed.depth_valid, seed.stencil_valid, true, error);
    vkDestroyBuffer(ctx.dev, buffer, nullptr); vkFreeMemory(ctx.dev, memory, nullptr);
    if (!copied) return false;
    image.layout_initialized = true;
    image.depth_valid = seed.depth_valid; image.stencil_valid = seed.stencil_valid;
    return true;
}

// `submission_batch` is an explicit live-renderer ownership scope. Calls with no requested CPU
// readback may return after recording; `flush_submission_batch` submits every accumulated command
// buffer in order, waits once, and releases all retained resources. Omitting the batch preserves the
// synchronous test/replay contract.
inline std::vector<uint8_t> render_draws_rgba(const std::vector<BackendDraw>& draws, uint32_t W, uint32_t H,
                                              const uint8_t* seed_rgba = nullptr,
                                              const float* clear_rgba = nullptr,
                                              bool persist_depth_stencil = false,
                                              const BackendColorTarget* color_target = nullptr,
                                              const uint8_t* seed_rgba1 = nullptr,
                                              const float* clear_rgba1 = nullptr,
                                              std::vector<uint8_t>* out_rgba1 = nullptr,
                                              BackendSubmissionBatch* submission_batch = nullptr,
                                              bool flush_submission_batch = true) {
    using TimingClock = std::chrono::steady_clock;
    const bool timing_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto timing_start = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    if (timing_enabled) backend_render_timing_stats_storage() = {};
    BackendColorTargetStats& color_target_stats = backend_color_target_stats_storage();
    color_target_stats = {};
    BackendResourceReuseStats& resource_reuse_stats = backend_resource_reuse_stats_storage();
    resource_reuse_stats = {};
    maybe_report_hash_stats();   // gated cumulative hashing economics (#1268)
    std::vector<uint8_t> out;
    if (out_rgba1) out_rgba1->clear();
    if (draws.empty()) return out;
    const RenderVkCtx& ctx = render_vk_ctx();
    if (!ctx.ok) return out;
    BackendSubmissionBatch direct_submission;
    BackendSubmissionBatch& active_submission = submission_batch
        ? *submission_batch : direct_submission;
    const bool avoid_cache_eviction = active_submission.pending();
    const bool persistent_color_targets_enabled =
        getenv("PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS") == nullptr;
    const bool persistent_color_enabled = persistent_color_targets_enabled && color_target &&
                                          color_target->persistent_id;
    const uint64_t color_target_generation = ++persistent_color_target_generation();
    VkInstance inst = ctx.inst; (void)inst; VkPhysicalDevice phys = ctx.phys;
    VkDevice dev = ctx.dev; VkQueue queue = ctx.queue; uint32_t qfi = ctx.qfi;
    const bool aniso_enabled = ctx.aniso_enabled; const float max_aniso_limit = ctx.max_aniso_limit;
    VkPhysicalDeviceMemoryProperties memp; vkGetPhysicalDeviceMemoryProperties(phys, &memp);
    init_persistent_color_target_device_budget(memp);   // size the residency budget once (#1177)
    auto pick = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < memp.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX; };
    const bool use_color1 = out_rgba1 != nullptr;
    const auto first_pipeline_format = [&](bool color1) {
        for (const auto& draw : draws) {
            if (!draw.ps) continue;
            const VkFormat format = static_cast<VkFormat>(
                color1 ? draw.ps->color1_format : draw.ps->color0_format);
            if (format != VK_FORMAT_UNDEFINED) return backend_color_format(format);
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    };
    const VkFormat FMT = color_target && color_target->format != VK_FORMAT_UNDEFINED
        ? backend_color_format(color_target->format)
        : first_pipeline_format(false);
    const VkFormat FMT1 = use_color1 ? first_pipeline_format(true) : VK_FORMAT_UNDEFINED;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(W) * H *
                               backend_color_bytes_per_pixel(FMT);
    const VkDeviceSize bytes1 = use_color1
        ? static_cast<VkDeviceSize>(W) * H * backend_color_bytes_per_pixel(FMT1)
        : 0;
    const uint32_t color_count = use_color1 ? 2u : 1u;
    const uint32_t ds_attachment = color_count;
    // Depth attachment is created if ANY draw enables the depth test (the shared render pass has one
    // fixed attachment set); each draw's pipeline sets its own depthTest/Write/CompareOp. A frame with
    // no depth-using draw takes the color-only path unchanged.
    bool use_depth = false, use_stencil = false;
    // Initial values for a newly-created depth/stencil attachment (#371). Existing guest-identified
    // surfaces LOAD their contents below, and explicit DB_RENDER_CONTROL clears execute in draw order.
    // Latch depth and stencil independently: coupling them let a stencil-only first draw force the
    // wrong reverse-Z depth initial value for a later draw (#457).
    float    depth_clear   = 1.0f;
    uint32_t stencil_clear = 0;
    bool     got_depth_clear = false, got_stencil_clear = false;
    // A DEPTH_CLEAR_ENABLE bit only acts through the enabled depth-write path (see
    // depth_clear_effective) — the writes-disabled Blue Prince light-loop shape is depth-inert and
    // must not force a depth attachment, pick the clear value, or clear in-pass (#1287).
    const auto effective_depth_clear = [](const prosper::gpu::ResolvedPipelineState* ps) {
        return depth_clear_effective(ps->depth_clear_enable, ps->depth_test_enable,
                                     ps->depth_write_enable, ps->depth_compare_op);
    };
    for (const auto& d : draws) {
        if (!d.ps) continue;
        if (d.ps->depth_test_enable || effective_depth_clear(d.ps)) { use_depth = true;
            // DB_DEPTH_CLEAR is only consumed when DB_RENDER_CONTROL requests an actual clear. Merely
            // programming the register does not initialize a newly-created host depth image: it is
            // commonly stale state, and Astro Bot even leaves the packed 1920x1080 max coordinates
            // (0x0437077f) there. Interpreting those bits as a float initialized every LEQUAL surface
            // to 2.15e-36 and rejected the entire scene. Without an explicit clear, approximate the
            // unavailable guest depth contents with the compare-appropriate far value (#371).
            //
            // An explicit clear wins even on ALWAYS/NEVER. Otherwise those compare modes do not depend
            // on the initial value and must not poison the first later meaningful draw (#508).
            // Initial-contents approximation for a NEWLY-CREATED host image: an explicitly
            // programmed clear expresses the guest's intended surface value even when this pass's
            // write path cannot apply it (#508 keeps this for ALWAYS/NEVER too). The write-path
            // gate below governs what the pass DOES (in-pass clears, validity, recency), not what
            // a fresh image starts as. Note this latch sits inside the depth-using block above, so
            // a fully write-path-disabled clear draw (the #1287 inert shape) never reaches it —
            // only a test-enabled clear can supply fresh-image contents.
            if (!got_depth_clear && d.ps->depth_clear_enable) {
                depth_clear = d.ps->depth_clear_value;
                got_depth_clear = true;
            } else if (!got_depth_clear && d.ps->depth_compare_op != VK_COMPARE_OP_ALWAYS
                                            && d.ps->depth_compare_op != VK_COMPARE_OP_NEVER) {
                depth_clear = (d.ps->depth_compare_op == VK_COMPARE_OP_GREATER ||
                               d.ps->depth_compare_op == VK_COMPARE_OP_GREATER_OR_EQUAL)
                    ? 0.0f : 1.0f;
                got_depth_clear = true;
            } }
        // #1355: like the depth gate above, a STENCIL_CLEAR_ENABLE bit with the stencil write
        // path disabled is inert and must not force a stencil attachment or clear in-pass. The
        // initial-value latch stays reachable through stencil_enable only, mirroring the depth
        // fresh-image approximation.
        if (d.ps->stencil_enable ||
            stencil_clear_effective(d.ps->stencil_clear_enable, d.ps->stencil_enable,
                                    d.ps->stencil_write_mask[0], d.ps->stencil_write_mask[1])) {
            use_stencil = true;
            // Mirror the depth contract (#371/#508 via #1361): DB_STENCIL_CLEAR is consumed as a
            // fresh image's initial contents only when the guest explicitly requests a clear.
            // Latching it from every stencil-using draw let a STALE register word poison a new
            // plane (the stencil analog of #371's Astro Bot depth case); without a clear the
            // unknown guest contents are approximated as 0.
            if (!got_stencil_clear && d.ps->stencil_clear_enable) {
                stencil_clear = d.ps->stencil_clear_value; got_stencil_clear = true;
            } }
    }
    if (const char* v = getenv("PROSPER_STENCIL_CLEAR"))
        stencil_clear = static_cast<uint32_t>(strtoul(v, nullptr, 0)) & 0xFFu;
    // Diagnostic A/B twin of PROSPER_STENCIL_CLEAR: override the derived initial depth value.
    // The #371 approximation picks the compare-appropriate always-pass value; sweeping this
    // instead reveals whether a pass's draws sit at DIFFERENT depths (a mid value culls some
    // draws but not others), i.e. whether real guest depth contents would gate them.
    if (const char* v = getenv("PROSPER_DEPTH_CLEAR"))
        depth_clear = strtof(v, nullptr);
    if (getenv("PROSPER_NO_DEPTH"))   use_depth = false;     // diag: isolate depth-test rejection
    if (getenv("PROSPER_NO_STENCIL")) use_stencil = false;   // diag: isolate stencil masking
    const bool use_ds = use_depth || use_stencil;
    // Use a stencil-capable depth format ONLY when a draw actually uses stencil (a UI mask). The
    // depth-only path keeps the original D32 depth-only format + aspect, so existing render tests are
    // byte-identical (#264).
    const prosper::gpu::ResolvedPipelineState* identity = nullptr;
    for (const auto& d : draws)
        if (d.ps && (d.ps->depth_test_enable || d.ps->stencil_enable ||
                     effective_depth_clear(d.ps) ||
                     stencil_clear_effective(d.ps->stencil_clear_enable, d.ps->stencil_enable,
                                             d.ps->stencil_write_mask[0],
                                             d.ps->stencil_write_mask[1]))) { identity = d.ps; break; }
    const bool has_ds_identity = identity && (identity->depth_read_base || identity->depth_write_base ||
                                               identity->stencil_read_base || identity->stencil_write_base);
    const bool persistent_ds = persist_depth_stencil && use_ds && has_ds_identity;
    // A persistent attachment must keep a stable format even across a depth-only call between stencil
    // users. Nonzero stencil identity means this guest surface owns a stencil plane.
    const bool format_has_stencil = use_stencil || (persistent_ds &&
        (identity->stencil_read_base || identity->stencil_write_base));
    const VkFormat DFMT = format_has_stencil ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D32_SFLOAT;
    const VkImageAspectFlags DASPECT = VK_IMAGE_ASPECT_DEPTH_BIT |
                                       (format_has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    VkImage dimg = VK_NULL_HANDLE; VkDeviceMemory dmem = VK_NULL_HANDLE; VkImageView dview = VK_NULL_HANDLE;
    bool ds_layout_initialized = false;
    bool depth_was_valid = false, stencil_was_valid = false;
    bool depth_used_meaningfully = false;
    bool depth_may_be_written = false;
    for (const auto& d : draws) {
        if (!d.ps) continue;
        depth_used_meaningfully |= effective_depth_clear(d.ps) || d.ps->depth_write_enable ||
            (d.ps->depth_test_enable && d.ps->depth_compare_op != VK_COMPARE_OP_ALWAYS &&
                                        d.ps->depth_compare_op != VK_COMPARE_OP_NEVER);
        depth_may_be_written |= persistent_ds_pass_may_write_depth(
            d.ps->depth_clear_enable, d.ps->depth_test_enable, d.ps->depth_write_enable,
            d.ps->depth_compare_op);
    }
    PersistentDsImage* cached_ds = nullptr;
    PersistentDsKey ds_key{};
    if (persistent_ds) {
        uint64_t htile_identity = identity->htile_data_base;
        // Captures through v5 did not serialize DB_HTILE_DATA_BASE. The explicit replay migration
        // switch recovers one preserved artifact whose manifest proves Unity allocated HTILE in the
        // 64 KiB block immediately before stencil. Never infer this layout for live/current state.
        if (!htile_identity && getenv("PROSPER_GPU_REPLAY_LEGACY_HTILE_BEFORE_STENCIL") &&
            identity->stencil_read_base >= 0x10000)
            htile_identity = identity->stencil_read_base - 0x10000;
        ds_key = {identity->depth_read_base, identity->depth_write_base,
                  identity->stencil_read_base, identity->stencil_write_base,
                  htile_identity, W, H, (uint32_t)DFMT};
        cached_ds = &persistent_ds_cache()[ds_key];
        dimg = cached_ds->image; dmem = cached_ds->memory; dview = cached_ds->view;
        ds_layout_initialized = cached_ds->layout_initialized;
        depth_was_valid = cached_ds->depth_valid;
        stencil_was_valid = cached_ds->stencil_valid;
    }
    if (getenv("PROSPER_DSLOG")) {
        static uint64_t call_id = 0;
        const uint64_t id = ++call_id;
        fprintf(stderr,
                "[ds] call=%llu size=%ux%u draws=%zu use=%d/%d persistent=%d valid=%d/%d/%d "
                "key=%llx/%llx/%llx/%llx htile=%llx fmt=%u initial=%g/%u\n",
                (unsigned long long)id, W, H, draws.size(), (int)use_depth, (int)use_stencil,
                (int)persistent_ds, (int)ds_layout_initialized,
                (int)depth_was_valid, (int)stencil_was_valid,
                (unsigned long long)ds_key.dr, (unsigned long long)ds_key.dw,
                (unsigned long long)ds_key.sr, (unsigned long long)ds_key.sw,
                (unsigned long long)ds_key.htile, ds_key.fmt, depth_clear, stencil_clear);
        for (size_t i = 0; i < draws.size(); ++i) {
            const auto* ps = draws[i].ps;
            if (!ps) {
                fprintf(stderr, "[ds] call=%llu draw=%zu state=none\n",
                        (unsigned long long)id, i);
                continue;
            }
            fprintf(stderr,
                    "[ds] call=%llu draw=%zu bases=%llx/%llx/%llx/%llx "
                    "depth=%d/%d/op%u clear=%d/%g stencil=%d clear=%d/%u "
                    "view=%08x htile=%llx hsurf=%08x info=%08x/%08x/%08x "
                    "size=%08x/%08x/%08x\n",
                    (unsigned long long)id, i,
                    (unsigned long long)ps->depth_read_base,
                    (unsigned long long)ps->depth_write_base,
                    (unsigned long long)ps->stencil_read_base,
                    (unsigned long long)ps->stencil_write_base,
                    (int)ps->depth_test_enable, (int)ps->depth_write_enable,
                    ps->depth_compare_op, (int)ps->depth_clear_enable, ps->depth_clear_value,
                    (int)ps->stencil_enable, (int)ps->stencil_clear_enable,
                    ps->stencil_clear_value, ps->db_depth_view,
                    (unsigned long long)ps->htile_data_base, ps->db_htile_surface,
                    ps->db_depth_info, ps->db_z_info, ps->db_stencil_info,
                    ps->db_depth_size_xy, ps->db_depth_size, ps->db_depth_slice);
        }
    }

    // Protect every GPU target sampled by this call from LRU eviction while its descriptors are built.
    for (const auto& draw : draws)
        for (const auto& resource : draw.R)
            if (persistent_color_targets_enabled && resource.persistent_render_target_id)
                if (auto* sampled = find_persistent_color_target(
                        resource.persistent_render_target_id, resource.tw, resource.th,
                        backend_color_format(resource.texture_format)))
                    sampled->last_use = color_target_generation;

    bool persistent_color = persistent_color_enabled;
    PersistentColorTargetKey color_key{};
    PersistentColorTargetImage* cached_color = nullptr;
    if (persistent_color) {
        color_key = {color_target->persistent_id, W, H, FMT};
        auto [found, inserted] = persistent_color_target_cache().try_emplace(color_key);
        cached_color = &found->second;
        cached_color->last_use = color_target_generation;
    }

    VkImage img = cached_color ? cached_color->image : VK_NULL_HANDLE;
    VkDeviceMemory imem = cached_color ? cached_color->memory : VK_NULL_HANDLE;
    VkImageView view = cached_color ? cached_color->view : VK_NULL_HANDLE;
    if (!img) {
        VkImageCreateInfo imgci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgci.imageType = VK_IMAGE_TYPE_2D; imgci.format = FMT; imgci.extent = {W, H, 1};
        imgci.mipLevels = 1; imgci.arrayLayers = 1; imgci.samples = VK_SAMPLE_COUNT_1_BIT;
        imgci.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      (persistent_color ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u) |
                      ((seed_rgba || persistent_color) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
        vkCreateImage(dev, &imgci, nullptr, &img);
        VkMemoryRequirements ir{}; vkGetImageMemoryRequirements(dev, img, &ir);
        VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        iai.allocationSize = ir.size; iai.memoryTypeIndex = pick(ir.memoryTypeBits, 0);
        if (persistent_color) {
            const VkDeviceSize limit = persistent_color_target_limit();
            // NOTE: the unsigned `limit - ir.size` below is guarded by short-circuit ordering, not by the
            // budget floor: `ir.size > limit` (here) and `ir.size <= limit` (retention check) are evaluated
            // first, so the subtraction only runs when `ir.size <= limit`. Keep those operands ahead of it —
            // reordering would let a single over-budget target underflow the subtraction to a huge value.
            while (!avoid_cache_eviction &&
                   (persistent_color_target_cache().size() > persistent_color_target_count_limit() || ir.size > limit ||
                    persistent_color_target_bytes() > limit - ir.size) &&
                   evict_persistent_color_target(ctx, color_target_generation)) {}
            if (ir.size <= limit && persistent_color_target_cache().size() <= persistent_color_target_count_limit() &&
                persistent_color_target_bytes() <= limit - ir.size &&
                vkAllocateMemory(dev, &iai, nullptr, &imem) == VK_SUCCESS) {
                cached_color->bytes = ir.size;
                persistent_color_target_bytes() += ir.size;
            } else {
                vkDestroyImage(dev, img, nullptr);
                img = VK_NULL_HANDLE;
                persistent_color_target_cache().erase(color_key);
                cached_color = nullptr;
                persistent_color = false;
                imgci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              (seed_rgba ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
                vkCreateImage(dev, &imgci, nullptr, &img);
                vkGetImageMemoryRequirements(dev, img, &ir);
                iai.allocationSize = ir.size; iai.memoryTypeIndex = pick(ir.memoryTypeBits, 0);
            }
        }
        if (!imem)
            imem = allocate_transient_render_memory(dev, iai.allocationSize, iai.memoryTypeIndex);
        vkBindImageMemory(dev, img, imem, 0);
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = FMT;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(dev, &ivci, nullptr, &view);
        if (cached_color) {
            cached_color->image = img;
            cached_color->memory = imem;
            cached_color->view = view;
        }
    }
    const bool load_cached_color = cached_color && cached_color->valid &&
                                   color_target->load_existing && !seed_rgba;
    if (cached_color) {
        ++color_target_stats.writes;
        color_target_stats.write_hits = load_cached_color ? 1 : 0;
    }
    if (color_target && color_target->persistent_id && getenv("PROSPER_BACKEND_LOAD_LOG"))
        fprintf(stderr,
                "[backend-load] id=0x%llx %ux%u fmt=%d cached=%d valid=%d load_existing=%d "
                "seed=%d readback=%d -> load=%d\n",
                (unsigned long long)color_target->persistent_id, W, H, (int)FMT,
                cached_color != nullptr, cached_color ? (int)cached_color->valid : -1,
                (int)color_target->load_existing, seed_rgba != nullptr,
                (int)color_target->readback, (int)load_cached_color);

    VkImage img1 = VK_NULL_HANDLE; VkDeviceMemory imem1 = VK_NULL_HANDLE;
    VkImageView view1 = VK_NULL_HANDLE;
    if (use_color1) {
        VkImageCreateInfo color1_ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        color1_ci.imageType = VK_IMAGE_TYPE_2D; color1_ci.format = FMT1;
        color1_ci.extent = {W, H, 1}; color1_ci.mipLevels = 1; color1_ci.arrayLayers = 1;
        color1_ci.samples = VK_SAMPLE_COUNT_1_BIT; color1_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        color1_ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          (seed_rgba1 ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0u);
        vkCreateImage(dev, &color1_ci, nullptr, &img1);
        VkMemoryRequirements color1_requirements{};
        vkGetImageMemoryRequirements(dev, img1, &color1_requirements);
        VkMemoryAllocateInfo color1_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        color1_allocation.allocationSize = color1_requirements.size;
        color1_allocation.memoryTypeIndex = pick(color1_requirements.memoryTypeBits, 0);
        imem1 = allocate_transient_render_memory(
            dev, color1_allocation.allocationSize, color1_allocation.memoryTypeIndex);
        vkBindImageMemory(dev, img1, imem1, 0);
        VkImageViewCreateInfo color1_view_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        color1_view_ci.image = img1; color1_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        color1_view_ci.format = FMT1;
        color1_view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(dev, &color1_view_ci, nullptr, &view1);
    }

    if (use_ds && !dimg) {
        VkImageCreateInfo dci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dci.imageType = VK_IMAGE_TYPE_2D; dci.format = DFMT; dci.extent = {W, H, 1};
        dci.mipLevels = 1; dci.arrayLayers = 1; dci.samples = VK_SAMPLE_COUNT_1_BIT;
        dci.tiling = VK_IMAGE_TILING_OPTIMAL;
        dci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;   // sampled depth bridge (#1275)
        vkCreateImage(dev, &dci, nullptr, &dimg);
        VkMemoryRequirements dr; vkGetImageMemoryRequirements(dev, dimg, &dr);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize = dr.size; dai.memoryTypeIndex = pick(dr.memoryTypeBits, 0);
        if (cached_ds) vkAllocateMemory(dev, &dai, nullptr, &dmem);
        else dmem = allocate_transient_render_memory(dev, dai.allocationSize,
                                                      dai.memoryTypeIndex);
        vkBindImageMemory(dev, dimg, dmem, 0);
        VkImageViewCreateInfo dvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dvci.image = dimg; dvci.viewType = VK_IMAGE_VIEW_TYPE_2D; dvci.format = DFMT;
        dvci.subresourceRange = {DASPECT, 0, 1, 0, 1};
        vkCreateImageView(dev, &dvci, nullptr, &dview);
        if (cached_ds) { cached_ds->image = dimg; cached_ds->memory = dmem; cached_ds->view = dview; }
    }

    VkAttachmentDescription att[3]{};
    att[0].format = FMT; att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    // Seeded or persistent: the attachment already holds valid pixels before this pass, so LOAD them.
    att[0].loadOp = (seed_rgba || load_cached_color) ? VK_ATTACHMENT_LOAD_OP_LOAD
                                                     : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = (seed_rgba || load_cached_color)
        ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = persistent_color ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    if (use_color1) {
        att[1].format = FMT1; att[1].samples = VK_SAMPLE_COUNT_1_BIT;
        att[1].loadOp = seed_rgba1 ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[1].initialLayout = seed_rgba1 ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                          : VK_IMAGE_LAYOUT_UNDEFINED;
        att[1].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    att[ds_attachment].format = DFMT; att[ds_attachment].samples = VK_SAMPLE_COUNT_1_BIT;
    // A guest-identified DS surface survives calls. New attachments get a defined initial value;
    // existing ones LOAD. Explicit DB_RENDER_CONTROL clears execute at their draw below, preserving
    // command order instead of being promoted to an unconditional pass-start clear (#518).
    // Depth and stencil have independent guest lifetimes even when Vulkan stores them in one D32S8
    // image. Using stencil must not make an untouched depth plane valid: Unity can stencil-prime a
    // surface under an ALWAYS, read-only depth test and only later use reverse-Z depth (#540).
    att[ds_attachment].loadOp = depth_was_valid ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[ds_attachment].storeOp = persistent_ds ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[ds_attachment].stencilLoadOp = format_has_stencil
        ? (stencil_was_valid ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR)
        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[ds_attachment].stencilStoreOp = (persistent_ds && format_has_stencil)
        ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[ds_attachment].initialLayout = ds_layout_initialized
        ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    att[ds_attachment].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ar[2] = {
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    };
    VkAttachmentReference dar{ds_attachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = color_count; sub.pColorAttachments = ar;
    if (use_ds) sub.pDepthStencilAttachment = &dar;
    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = color_count + (use_ds ? 1u : 0u); rpci.pAttachments = att;
    rpci.subpassCount = 1; rpci.pSubpasses = &sub;
    VkRenderPass rp; vkCreateRenderPass(dev, &rpci, nullptr, &rp);
    VkImageView fbviews[3] = {view, VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (use_color1) fbviews[1] = view1;
    if (use_ds) fbviews[ds_attachment] = dview;
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass = rp; fbci.attachmentCount = color_count + (use_ds ? 1u : 0u);
    fbci.pAttachments = fbviews; fbci.width = W; fbci.height = H; fbci.layers = 1;
    VkFramebuffer fb; vkCreateFramebuffer(dev, &fbci, nullptr, &fb);

    auto mkmod = [&](const std::vector<uint32_t>& c) -> VkShaderModule {
        VkShaderModuleCreateInfo s{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        s.codeSize = c.size() * 4; s.pCode = c.data(); VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(dev, &s, nullptr, &m); return m; };
    // Per-draw Vulkan objects stay alive until the call or explicit submission batch completes.
    const auto timing_target_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    struct DV {
        VkShaderModule vs = VK_NULL_HANDLE, gs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> dsets;
        VkPipelineLayout layout = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE;
        VkBuffer ibuf = VK_NULL_HANDLE; VkDeviceMemory ibmem = VK_NULL_HANDLE;   // index buffer (indexed draws)
        VkRect2D scissor{};
        uint32_t n_sets = 1, vcount = 3, icount = 0, instance_count = 1;
        int32_t vertex_offset = 0;
        bool use_desc = false, ok = false, pipeline_cached = false;
    };
    struct TextureUploadKey {
        const uint8_t* pixels = nullptr;
        uint64_t render_target_id = 0;
        uint32_t width = 0, height = 0, depth = 1;
        uint32_t img_dim = 1;
        uint32_t mip_levels = 1;   // effective uploaded chain length (#1272)
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool storage_image = false;
        bool operator==(const TextureUploadKey& other) const {
            return pixels == other.pixels && render_target_id == other.render_target_id &&
                   width == other.width && height == other.height && depth == other.depth &&
                   img_dim == other.img_dim && mip_levels == other.mip_levels &&
                   format == other.format && storage_image == other.storage_image;
        }
    };
    struct TextureUploadKeyHash {
        size_t operator()(const TextureUploadKey& key) const {
            size_t h = std::hash<const uint8_t*>{}(key.pixels);
            h ^= std::hash<uint64_t>{}(key.render_target_id) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.width) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.height) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.depth) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.img_dim) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.mip_levels) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.format) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(key.storage_image) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct SharedTextureUpload {
        TextureUploadKey key;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        uint64_t persistent_id = 0;
        VkDeviceSize image_bytes = 0;
        bool persistent_hit = false;
        bool borrowed_target = false;
        // Sampled depth bridge (#1275): image borrowed from the persistent DS cache. The view uses
        // the DEPTH aspect of ds_format, and the call transitions the image DS-attachment ->
        // shader-read around its passes.
        bool borrowed_ds = false;
        VkFormat ds_format = VK_FORMAT_UNDEFINED;
        bool direct_memory = false;
    };
    struct PersistentTextureKey {
        uint64_t id = 0;
        uint32_t width = 0, height = 0, depth = 1, img_dim = 1;
        uint32_t mip_levels = 1;   // a cached image must match the requested chain length (#1272)
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        bool operator==(const PersistentTextureKey&) const = default;
    };
    struct PersistentTextureKeyHash {
        size_t operator()(const PersistentTextureKey& key) const {
            size_t h = std::hash<uint64_t>{}(key.id);
            auto mix = [&](uint32_t value) {
                h ^= static_cast<size_t>(value) + 0x9e3779b9u + (h << 6) + (h >> 2);
            };
            mix(key.width); mix(key.height); mix(key.depth); mix(key.img_dim);
            mix(key.mip_levels);
            mix(static_cast<uint32_t>(key.format));
            return h;
        }
    };
    struct TextureBindingKey {
        std::array<uint64_t, 22> words{};
        bool operator==(const TextureBindingKey&) const = default;
    };
    struct TextureBindingKeyHash {
        size_t operator()(const TextureBindingKey& key) const {
            uint64_t hash = 1469598103934665603ull;
            for (uint64_t word : key.words) {
                hash ^= word;
                hash *= 1099511628211ull;
            }
            return static_cast<size_t>(hash);
        }
    };
    struct PersistentTextureBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        uint64_t last_use = 0;
    };
    struct PersistentTextureImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;
        uint64_t last_use = 0;
        std::unordered_map<TextureBindingKey, PersistentTextureBinding,
                           TextureBindingKeyHash> bindings;
    };
    static std::unordered_map<PersistentTextureKey, PersistentTextureImage,
                              PersistentTextureKeyHash> persistent_texture_images;
    static VkDeviceSize persistent_texture_bytes = 0;
    static uint64_t persistent_texture_generation = 0;
    constexpr size_t persistent_texture_max_entries = 1024;
    const uint64_t texture_generation = ++persistent_texture_generation;
    const bool persistent_textures_enabled =
        getenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES") == nullptr;
    const VkDeviceSize persistent_texture_limit = []() -> VkDeviceSize {
        const char* value = getenv("PROSPER_BACKEND_TEXTURE_CACHE_MB");
        const uint64_t mib = value ? strtoull(value, nullptr, 10) : 1024ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    const bool share_backend_resources =
        getenv("PROSPER_NO_BACKEND_RESOURCE_SHARE") == nullptr;
    const bool reuse_host_buffers = render_host_buffer_pool_enabled();
    struct SharedBufferKey {
        const uint32_t* words = nullptr;
        size_t count = 0;
        uint64_t identity = 0;
        uint64_t hash = 0;
        uint64_t unique_tag = 0;
        bool operator==(const SharedBufferKey& other) const {
            return identity == other.identity && hash == other.hash &&
                   unique_tag == other.unique_tag &&
                   count == other.count &&
                   (!count || std::memcmp(words, other.words, count * sizeof(uint32_t)) == 0);
        }
    };
    struct SharedBufferKeyHash {
        size_t operator()(const SharedBufferKey& key) const {
            return static_cast<size_t>(key.hash ^ key.identity ^
                (key.unique_tag * 0x9e3779b97f4a7c15ull));
        }
    };
    struct SharedBufferUpload {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize offset = 0;
        VkDeviceSize bytes = 0;
        VkDeviceSize allocation_bytes = 0;
        bool pooled = false;
        bool arena = false;
    };
    struct SharedBufferArena {
        RenderHostBuffer buffer;
        VkDeviceSize used = 0;
    };
    struct SharedTextureBinding {
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        bool persistent = false;
    };
    struct SharedPipelineLayout {
        VkPipelineLayout handle = VK_NULL_HANDLE;
        bool persistent = false;
    };
    std::vector<SharedBufferUpload> shared_buffers;
    std::vector<SharedBufferArena> shared_buffer_arenas;
    std::unordered_map<SharedBufferKey, size_t, SharedBufferKeyHash> shared_buffer_indices;
    // Per-call repeat-reference memo (#1268): draws in one pass batch overwhelmingly re-reference
    // the same guest ranges (same VB/UB across hundreds of draws), and the SharedBufferKey lookup
    // pays a FULL content hash per reference because the hash is part of the key. Within one call
    // the referenced guest memory is stable modulo cross-thread racing writes — which are equally
    // nondeterministic on real hardware (the GPU samples the buffer once per draw at execution),
    // so resolving a repeated (words, count, identity) reference to the first upload is within the
    // same latitude the hardware has. First reference still hashes + memcmps as before.
    struct BufferRefMemoKey {
        const uint32_t* words = nullptr;
        size_t count = 0;
        uint64_t identity = 0;
        bool operator==(const BufferRefMemoKey& other) const {
            return words == other.words && count == other.count && identity == other.identity;
        }
    };
    struct BufferRefMemoKeyHash {
        size_t operator()(const BufferRefMemoKey& key) const {
            return static_cast<size_t>((reinterpret_cast<uintptr_t>(key.words) >> 2) ^
                (key.count * 0x9e3779b97f4a7c15ull) ^ key.identity);
        }
    };
    std::unordered_map<BufferRefMemoKey, size_t, BufferRefMemoKeyHash> buffer_ref_memo;
    std::vector<SharedTextureBinding> shared_texture_bindings;
    std::unordered_map<TextureBindingKey, size_t, TextureBindingKeyHash>
        shared_texture_binding_indices;
    std::unordered_map<std::vector<uint64_t>, VkDescriptorSetLayout, BackendWordVectorHash>
        shared_descriptor_set_layouts;
    std::unordered_map<std::vector<uint64_t>, SharedPipelineLayout, BackendWordVectorHash>
        shared_pipeline_layouts;
    uint64_t resource_unique_tag = 0;
    const uint32_t zero_buffer_word = 0;
    const VkDeviceSize storage_buffer_alignment = ctx.storage_buffer_alignment;
    const bool use_buffer_arena = reuse_host_buffers &&
        getenv("PROSPER_NO_BACKEND_BUFFER_ARENA") == nullptr;
    auto align_storage_offset = [storage_buffer_alignment](VkDeviceSize value) {
        const VkDeviceSize remainder = value % storage_buffer_alignment;
        if (!remainder) return value;
        const VkDeviceSize padding = storage_buffer_alignment - remainder;
        return value <= UINT64_MAX - padding ? value + padding : VkDeviceSize{UINT64_MAX};
    };
    auto acquire_buffer_arena_slice = [&](VkDeviceSize bytes, SharedBufferUpload& upload) {
        for (SharedBufferArena& arena : shared_buffer_arenas) {
            const VkDeviceSize offset = align_storage_offset(arena.used);
            if (offset != UINT64_MAX && offset <= arena.buffer.bytes &&
                bytes <= arena.buffer.bytes - offset) {
                upload.buffer = arena.buffer.buffer;
                upload.mapped = arena.buffer.mapped;
                upload.offset = offset;
                upload.arena = true;
                arena.used = offset + bytes;
                return true;
            }
        }
        VkDeviceSize request = std::max(render_host_buffer_arena_size(), bytes);
        if (request < bytes) request = bytes;
        RenderHostBuffer buffer = acquire_render_host_buffer(ctx, request);
        if (!buffer.buffer || !buffer.memory || !buffer.mapped) {
            destroy_render_host_buffer(ctx.dev, buffer);
            return false;
        }
        shared_buffer_arenas.push_back({buffer, bytes});
        upload.buffer = buffer.buffer;
        upload.mapped = buffer.mapped;
        upload.offset = 0;
        upload.arena = true;
        return true;
    };
    auto handle_bits = [](auto handle) {
        uint64_t bits = 0;
        static_assert(sizeof(handle) <= sizeof(bits));
        std::memcpy(&bits, &handle, sizeof(handle));
        return bits;
    };
    auto float_bits = [](float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };
    std::vector<DV> dv(draws.size());
    std::vector<std::vector<FrameResource>> effective_resources(draws.size());
    for (size_t i = 0; i < draws.size(); ++i) {
        effective_resources[i] = draws[i].R;
        if (prosper::gpu::fragment_spirv_uses_internal_gds(draws[i].fs_words())) {
            FrameResource gds;
            gds.set = 1;
            gds.binding = 0;
            gds.is_internal_gds = true;
            effective_resources[i].push_back(std::move(gds));
        }
    }
    std::vector<SharedTextureUpload> texture_uploads;
    std::unordered_map<TextureUploadKey, size_t, TextureUploadKeyHash> texture_upload_indices;
    const bool share_texture_uploads = getenv("PROSPER_NO_BACKEND_TEXTURE_SHARE") == nullptr;
    size_t texture_references = 0;
    size_t persistent_texture_hits = 0;
    size_t persistent_texture_misses = 0;
    double setup_shader_ms = 0.0;
    double setup_fixed_ms = 0.0;
    double setup_resources_ms = 0.0;
    double setup_pipeline_ms = 0.0;
    auto setup_elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    BackendPipelineCacheStats& pipeline_stats = backend_pipeline_cache_stats_storage();
    pipeline_stats = {};
    auto& pipeline_cache = persistent_pipeline_cache();
    const uint64_t pipeline_generation = ++persistent_pipeline_generation();
    const bool pipeline_cache_enabled = persistent_pipeline_cache_enabled();
    const size_t pipeline_cache_limit = persistent_pipeline_cache_limit();
    const bool pipeline_layout_cache_enabled = share_backend_resources &&
        persistent_pipeline_layout_cache_enabled();
    const size_t pipeline_layout_cache_limit = persistent_pipeline_layout_cache_limit();
    const uint64_t pipeline_layout_generation = ++persistent_pipeline_layout_generation();
    auto& persistent_pipeline_layouts = persistent_backend_pipeline_layout_cache();
    auto evict_persistent_pipeline_layout = [&]() {
        auto victim = persistent_pipeline_layouts.end();
        for (auto it = persistent_pipeline_layouts.begin();
             it != persistent_pipeline_layouts.end(); ++it) {
            if (it->second.last_use == pipeline_layout_generation) continue;
            if (victim == persistent_pipeline_layouts.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == persistent_pipeline_layouts.end()) return false;
        vkDestroyPipelineLayout(dev, victim->second.handle, nullptr);
        persistent_pipeline_layouts.erase(victim);
        ++resource_reuse_stats.persistent_pipeline_layout_evictions;
        return true;
    };
    auto evict_pipeline = [&]() {
        auto victim = pipeline_cache.end();
        for (auto it = pipeline_cache.begin(); it != pipeline_cache.end(); ++it) {
            if (it->second.last_use == pipeline_generation) continue;
            if (victim == pipeline_cache.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == pipeline_cache.end()) return false;
        vkDestroyPipeline(dev, victim->second.pipeline, nullptr);
        pipeline_cache.erase(victim);
        ++pipeline_stats.evictions;
        return true;
    };
    uint64_t descriptor_sets = 0;
    uint64_t storage_buffers = 0;
    uint64_t sampled_images = 0;
    uint64_t storage_images = 0;
    for (const auto& resources : effective_resources) {
        if (resources.empty()) continue;
        uint32_t set_count = 1;
        for (const FrameResource& resource : resources) {
            set_count = std::max(set_count, resource.set + 1);
            if (!resource.is_texture()) {
                ++storage_buffers;
            } else if (resource.is_storage_image) {
                ++storage_images;
            } else {
                ++sampled_images;
            }
        }
        descriptor_sets += set_count;
    }
    VkDescriptorPool shared_descriptor_pool = VK_NULL_HANDLE;
    if (descriptor_sets) {
        VkDescriptorPoolSize sizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             static_cast<uint32_t>(std::max<uint64_t>(storage_buffers, 1))},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
             static_cast<uint32_t>(std::max<uint64_t>(sampled_images, 1))},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
             static_cast<uint32_t>(std::max<uint64_t>(storage_images, 1))},
        };
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = static_cast<uint32_t>(descriptor_sets);
        pool_info.poolSizeCount = 3;
        pool_info.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(dev, &pool_info, nullptr, &shared_descriptor_pool) ==
            VK_SUCCESS) {
            resource_reuse_stats.descriptor_pools = 1;
        }
    }
    // Pass 1: create each draw's shader modules, descriptors (with texture staging upload), and pipeline.
    const bool backend_trace = getenv("PROSPER_BACKEND_TRACE") != nullptr;
    for (size_t di = 0; di < draws.size(); di++) {
        const auto setup_begin = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        const BackendDraw& bd = draws[di];
        const std::vector<uint32_t>& bd_vs = bd.vs_words();
        const std::vector<uint32_t>& bd_gs = bd.gs_words();
        const std::vector<uint32_t>& bd_fs = bd.fs_words();
        DV& v = dv[di];
        const uint32_t required_fragment_subgroup_size =
            prosper::gpu::fragment_spirv_required_subgroup_size(bd_fs);
        const bool uses_internal_gds =
            prosper::gpu::fragment_spirv_uses_internal_gds(bd_fs);
        if (required_fragment_subgroup_size &&
            (!ctx.subgroup_size_control ||
             required_fragment_subgroup_size < ctx.min_subgroup_size ||
             required_fragment_subgroup_size > ctx.max_subgroup_size ||
             !(ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
             !(ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
             !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ||
             (uses_internal_gds && !ctx.fragment_stores_atomics))) {
            const uint64_t shader_key = bd.fs_identity
                ? bd.fs_identity : hash_buffer_words(bd_fs.data(), bd_fs.size());
            static std::mutex log_mutex;
            static std::unordered_set<uint64_t> logged;
            std::lock_guard<std::mutex> lock(log_mutex);
            if (logged.insert(shader_key).second)
                std::fprintf(stderr,
                             "[render] skip draw: fragment shader requires subgroup size %u "
                             "(device range %u..%u required-stages=0x%x subgroup-stages=0x%x "
                             "ops=0x%x control=%d gds=%d fragment-atomics=%d)\n",
                             required_fragment_subgroup_size, ctx.min_subgroup_size,
                             ctx.max_subgroup_size, ctx.required_subgroup_size_stages,
                             ctx.subgroup_stages, ctx.subgroup_operations,
                             static_cast<int>(ctx.subgroup_size_control),
                             static_cast<int>(uses_internal_gds),
                             static_cast<int>(ctx.fragment_stores_atomics));
            continue;
        }
        if (uses_internal_gds && !render_internal_gds_buffer().buffer) {
            static std::once_flag logged;
            std::call_once(logged, [] {
                std::fprintf(stderr,
                             "[render] skip draw: failed to allocate persistent GDS buffer\n");
            });
            continue;
        }
        if (backend_trace) {
            fprintf(stderr,
                    "[backend-trace] draw=%zu/%zu begin extent=%ux%u vs=%zu gs=%zu fs=%zu "
                    "vs_id=%016llx fs_id=%016llx resources=%zu\n",
                    di, draws.size(), W, H, bd_vs.size(), bd_gs.size(), bd_fs.size(),
                    (unsigned long long)bd.vs_identity,
                    (unsigned long long)bd.fs_identity, bd.R.size());
            fflush(stderr);
        }
        // Pipeline hits do not need temporary VkShaderModules. Defer module creation until after the
        // persistent lookup; the fixed/resource setup below is also required by the hit pipeline.
        const auto setup_shaders_ready = setup_begin;
        const prosper::gpu::ResolvedPipelineState* ps = bd.ps;
        v.vcount = bd.vcount;
        v.instance_count = bd.instance_count;
        v.vertex_offset = bd.vertex_offset;
        v.scissor = {{0, 0}, {W, H}};
        // PROSPER_IGNORE_EMPTY_SCISSOR (#1287 bring-up diagnostic): render draws whose resolved
        // scissor is empty with a full-target scissor instead, to A/B whether they carry the
        // missing post/shadow content. Off by default; not a fix.
        static const bool ignore_empty_scissor = getenv("PROSPER_IGNORE_EMPTY_SCISSOR") != nullptr;
        const bool scissor_empty = ps && ps->has_scissor &&
            (ps->scissor_right <= ps->scissor_left || ps->scissor_bottom <= ps->scissor_top);
        if (ps && ps->has_scissor && !(ignore_empty_scissor && scissor_empty)) {
            const int64_t left = std::clamp<int64_t>(ps->scissor_left, 0, W);
            const int64_t top = std::clamp<int64_t>(ps->scissor_top, 0, H);
            const int64_t right = std::clamp<int64_t>(ps->scissor_right, left, W);
            const int64_t bottom = std::clamp<int64_t>(ps->scissor_bottom, top, H);
            v.scissor.offset = {static_cast<int32_t>(left), static_cast<int32_t>(top)};
            v.scissor.extent = {static_cast<uint32_t>(right - left),
                                static_cast<uint32_t>(bottom - top)};
        }
        // Indexed draw: upload the 32-bit index data to a host-visible VkIndexBuffer now; the record
        // pass binds it and issues vkCmdDrawIndexed instead of vkCmdDraw.
        if (!bd.indices.empty()) {
            VkDeviceSize isz = (VkDeviceSize)bd.indices.size() * 4;
            VkBufferCreateInfo ibci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ibci.size = isz; ibci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            vkCreateBuffer(dev, &ibci, nullptr, &v.ibuf);
            VkMemoryRequirements imr; vkGetBufferMemoryRequirements(dev, v.ibuf, &imr);
            VkMemoryAllocateInfo imai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; imai.allocationSize = imr.size;
            imai.memoryTypeIndex = pick(imr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            v.ibmem = allocate_transient_render_memory(dev, imai.allocationSize,
                                                        imai.memoryTypeIndex);
            vkBindBufferMemory(dev, v.ibuf, v.ibmem, 0);
            void* ip = nullptr; vkMapMemory(dev, v.ibmem, 0, isz, 0, &ip);
            std::memcpy(ip, bd.indices.data(), (size_t)isz); vkUnmapMemory(dev, v.ibmem);
            v.icount = (uint32_t)bd.indices.size();
        }
        VkPipelineShaderStageCreateInfo st[3]{};
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_fragment_subgroup{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        st[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = v.vs; st[0].pName = "main";
        const uint32_t fragment_stage_index = bd_gs.empty() ? 1u : 2u;
        if (!bd_gs.empty()) {
            st[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            st[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT; st[1].module = v.gs; st[1].pName = "main";
        }
        st[fragment_stage_index] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        st[fragment_stage_index].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[fragment_stage_index].module = v.fs;
        st[fragment_stage_index].pName = "main";
        if (required_fragment_subgroup_size) {
            required_fragment_subgroup.requiredSubgroupSize = required_fragment_subgroup_size;
            st[fragment_stage_index].pNext = &required_fragment_subgroup;
        }
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = ps ? (VkPrimitiveTopology)ps->topology : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
#ifdef __APPLE__
        // Metal always has primitive restart enabled and MoltenVK rejects primitiveRestartEnable=VK_FALSE
        // with VK_ERROR_FEATURE_NOT_PRESENT. Force it on; harmless here because these draws don't use the
        // strip-restart sentinel index (0xFFFF/0xFFFFFFFF), and it has no effect on list topologies.
        ia.primitiveRestartEnable = VK_TRUE;
#endif
        // Default: full-target viewport. When the resolved state carries the guest's PA_CL_VPORT transform,
        // honor it — a guest yscale < 0 arrives as a negative viewport_h (Vulkan core-1.1 flipped viewport),
        // reproducing the hardware's Y orientation (#38; each draw item keeps its own resolved viewport).
        VkViewport vp{0, 0, (float)W, (float)H, 0, 1}; VkRect2D sc{{0, 0}, {W, H}};
        if (ps && ps->has_viewport)
            vp = {ps->viewport_x, ps->viewport_y, ps->viewport_w, ps->viewport_h, ps->min_depth, ps->max_depth};
        VkPipelineViewportStateCreateInfo vpst{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpst.viewportCount = 1; vpst.pViewports = &vp; vpst.scissorCount = 1; vpst.pScissors = &sc;
        const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic_state.dynamicStateCount = 1; dynamic_state.pDynamicStates = dynamic_states;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        // Honor the guest's PA_SU_SC_MODE_CNTL cull/front-face/polygon mode (#456). Resolve encodes these
        // as the Vk enumerators; an absent register resolves to the same NONE/CCW/FILL default above, so
        // the null-ps (test) path and any draw that never programs it are byte-identical. PROSPER_NO_CULL
        // forces CULL_NONE back on; PROSPER_FLIP_FRONT_FACE preserves culling and toggles only winding.
        // Together they isolate a cull-mode problem from a front-face translation problem without a rebuild.
        if (ps) { rs.cullMode  = getenv("PROSPER_NO_CULL") ? VK_CULL_MODE_NONE : (VkCullModeFlags)ps->cull_mode;
                  rs.frontFace = getenv("PROSPER_FLIP_FRONT_FACE")
                      ? (ps->front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                  : VK_FRONT_FACE_CLOCKWISE)
                      : (VkFrontFace)ps->front_face;
                  rs.polygonMode = (VkPolygonMode)ps->polygon_mode; }
        // Depth bias (#1349): the guest's PA_SU_POLY_OFFSET_* — shadow-map passes need it against
        // acne. Clamp requires the depthBiasClamp device feature; without it a non-zero clamp is
        // dropped to 0 (bias still applies, unclamped — the safe direction). PROSPER_NO_DEPTH_BIAS
        // is the A/B diagnostic, symmetric with PROSPER_NO_CULL above.
        if (ps && ps->depth_bias_enable && !getenv("PROSPER_NO_DEPTH_BIAS")) {
            rs.depthBiasEnable         = VK_TRUE;
            rs.depthBiasConstantFactor = ps->depth_bias_constant;
            rs.depthBiasSlopeFactor    = ps->depth_bias_slope;
            rs.depthBiasClamp = render_vk_ctx().depth_bias_clamp_enabled ? ps->depth_bias_clamp : 0.0f;
        }
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba[2]{};
        cba[0].colorWriteMask = 0xF; cba[1].colorWriteMask = 0xF;
        if (ps) {
            cba[0].colorWriteMask = ps->color_write_mask;
            cba[0].blendEnable    = ps->blend_enable ? VK_TRUE : VK_FALSE;
            if (getenv("PROSPER_NO_BLEND")) cba[0].blendEnable = VK_FALSE;   // diag: isolate blend compositing
            cba[0].srcColorBlendFactor = (VkBlendFactor)ps->src_color_blend_factor;
            cba[0].dstColorBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor;
            cba[0].colorBlendOp        = (VkBlendOp)ps->color_blend_op;
            // Alpha channel uses its OWN resolved factors (#381): resolve set these from the separate
            // ALPHA_* blend fields when SEPARATE_ALPHA_BLEND was programmed, else it already mirrored the
            // color factors — so this is correct in both cases without guessing here.
            cba[0].srcAlphaBlendFactor = (VkBlendFactor)ps->src_alpha_blend_factor;
            cba[0].dstAlphaBlendFactor = (VkBlendFactor)ps->dst_alpha_blend_factor;
            cba[0].alphaBlendOp        = (VkBlendOp)ps->alpha_blend_op;
            cba[1].colorWriteMask = ps->color1_write_mask;
            cba[1].blendEnable = ps->blend1_enable ? VK_TRUE : VK_FALSE;
            if (getenv("PROSPER_NO_BLEND")) cba[1].blendEnable = VK_FALSE;
            cba[1].srcColorBlendFactor = (VkBlendFactor)ps->src_color_blend_factor1;
            cba[1].dstColorBlendFactor = (VkBlendFactor)ps->dst_color_blend_factor1;
            cba[1].colorBlendOp = (VkBlendOp)ps->color_blend_op1;
            cba[1].srcAlphaBlendFactor = (VkBlendFactor)ps->src_alpha_blend_factor1;
            cba[1].dstAlphaBlendFactor = (VkBlendFactor)ps->dst_alpha_blend_factor1;
            cba[1].alphaBlendOp = (VkBlendOp)ps->alpha_blend_op1;
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        if (ps && ps->logic_op_enable) {
            if (ctx.logic_op_enabled) {
                cb.logicOpEnable = VK_TRUE;
                cb.logicOp = static_cast<VkLogicOp>(ps->logic_op);
            } else {
                static std::once_flag logged;
                std::call_once(logged, [] {
                    fprintf(stderr, "[gpu] Vulkan device lacks logicOp support -> COPY fallback\n");
                });
            }
        }
        cb.attachmentCount = color_count; cb.pAttachments = cba;
        VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (ps && ps->depth_test_enable) {
            dss.depthTestEnable  = VK_TRUE;
            dss.depthWriteEnable = ps->depth_write_enable ? VK_TRUE : VK_FALSE;
            dss.depthCompareOp   = (VkCompareOp)ps->depth_compare_op;
            // UE4 repeats its reverse-Z depth prepass in a separately translated base-pass shader.
            // A one-ULP position difference between those shaders makes exact EQUAL reject the whole
            // base pass, although the guest hardware accepts the pair. Preserve occlusion by relaxing
            // only a read-only EQUAL against an already-populated, explicitly reverse-Z surface:
            // GEQUAL still rejects geometry behind the prepass instead of disabling depth outright.
            const bool reverse_z_equal_compat = persistent_ds && depth_was_valid &&
                !ps->depth_write_enable && ps->depth_compare_op == VK_COMPARE_OP_EQUAL &&
                ps->has_depth_clear && ps->depth_clear_value <= 0.5f;
            if (reverse_z_equal_compat) {
                dss.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
                if (getenv("PROSPER_DSLOG"))
                    fprintf(stderr, "[ds] reverse-Z read-only EQUAL -> GEQUAL compatibility\n");
            }
        }
        if (ps && ps->stencil_enable) {
            // Wire the front/back stencil op-state so masks clip (e.g. the title shimmer tests the
            // stencil the logo draw wrote). ref/compareMask/writeMask are baked (not dynamic).
            dss.stencilTestEnable = VK_TRUE;
            auto mkop = [&](int fb) {
                VkStencilOpState s{};
                s.failOp      = (VkStencilOp)ps->stencil_fail_op[fb];
                s.passOp      = (VkStencilOp)ps->stencil_pass_op[fb];
                s.depthFailOp = (VkStencilOp)ps->stencil_depth_fail_op[fb];
                s.compareOp   = (VkCompareOp)ps->stencil_compare_op[fb];
                // PROSPER_STENCIL_MIRROR=1 (diagnostic A/B ONLY — NOT hardware semantics): mirror
                // the asymmetric compare ops, i.e. evaluate `stencil OP ref` instead of Vulkan's
                // `ref OP stencil`. RDNA2 STENCILFUNC is ref-on-left 1:1 with VkCompareOp (radeonsi
                // programs PIPE_FUNC straight into the field), so this switch deliberately DIVERGES
                // from hardware — its use is isolating whether a suspect draw's coverage is
                // stencil-gated (flipping GREATER<->LESS suppresses/expands it) without touching
                // any other state. Symmetric ops (EQUAL/NOTEQUAL/ALWAYS/NEVER) are unaffected.
                if (getenv("PROSPER_STENCIL_MIRROR")) {
                    switch (s.compareOp) {
                        case VK_COMPARE_OP_LESS:             s.compareOp = VK_COMPARE_OP_GREATER; break;
                        case VK_COMPARE_OP_GREATER:          s.compareOp = VK_COMPARE_OP_LESS; break;
                        case VK_COMPARE_OP_LESS_OR_EQUAL:    s.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; break;
                        case VK_COMPARE_OP_GREATER_OR_EQUAL: s.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL; break;
                        default: break;
                    }
                }
                s.compareMask = ps->stencil_compare_mask[fb];
                s.writeMask   = ps->stencil_write_mask[fb];
                // AMD splits the stencil reference: STENCILTESTVAL is the COMPARE reference, but a REPLACE
                // op writes STENCILOPVAL. Vulkan has one `reference` for both. When this draw REPLACEs
                // (its whole purpose is to WRITE a mask value, and its compare is typically ALWAYS so the
                // compare-ref is irrelevant), use STENCILOPVAL so the mask is written with the value the
                // game intended — else a mask-write with TESTVAL=0/OPVAL=1 writes 0 and every later
                // test==1 draw is wrongly culled (PPSA02664's whole UI vanished; #270).
                const uint32_t REPLACE = 2;   // VK_STENCIL_OP_REPLACE
                bool does_replace = (ps->stencil_pass_op[fb] == REPLACE || ps->stencil_fail_op[fb] == REPLACE ||
                                     ps->stencil_depth_fail_op[fb] == REPLACE);
                s.reference   = does_replace ? ps->stencil_op_val[fb] : ps->stencil_ref[fb];
                if (does_replace && s.compareOp == VK_COMPARE_OP_ALWAYS)
                    if (const char* v = getenv("PROSPER_STENCIL_REPLACE"))
                        s.reference = static_cast<uint32_t>(strtoul(v, nullptr, 0)) & 0xFFu;
                return s;
            };
            dss.front = mkop(0); dss.back = mkop(1);
            if (getenv("PROSPER_STENCILLOG"))
                fprintf(stderr, "[stencil] front{cmp=%u ref=%u opval=%u cmask=0x%x wmask=0x%x fail=%u pass=%u zfail=%u} back{cmp=%u ref=%u fail=%u pass=%u zfail=%u} vkref=%u/%u cull=%u depth_test=%d\n",
                        ps->stencil_compare_op[0], ps->stencil_ref[0], ps->stencil_op_val[0], ps->stencil_compare_mask[0], ps->stencil_write_mask[0],
                        ps->stencil_fail_op[0], ps->stencil_pass_op[0], ps->stencil_depth_fail_op[0],
                        ps->stencil_compare_op[1], ps->stencil_ref[1],
                        ps->stencil_fail_op[1], ps->stencil_pass_op[1], ps->stencil_depth_fail_op[1],
                        dss.front.reference, dss.back.reference, (unsigned)ps->cull_mode, (int)ps->depth_test_enable);
        }
        // Descriptor resources for this draw (two-set: VS=set0, PS=set1 — same layout as the single path).
        const auto setup_fixed_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) setup_fixed_ms += setup_elapsed_ms(setup_shaders_ready, setup_fixed_ready);
        const auto& R = effective_resources[di];
        v.use_desc = !R.empty();
        for (auto& r : R) v.n_sets = std::max(v.n_sets, r.set + 1);
        std::vector<VkDescriptorSetLayout> dsls(v.n_sets, VK_NULL_HANDLE);
        v.dsets.assign(v.n_sets, VK_NULL_HANDLE);
        std::vector<std::vector<uint64_t>> descriptor_layout_keys(v.n_sets);
        if (v.use_desc) {
            std::vector<VkDescriptorSetLayoutBinding> lb(R.size());
            std::vector<VkDescriptorBufferInfo> dbi(R.size());
            std::vector<VkDescriptorImageInfo> dii(R.size());
            std::vector<VkWriteDescriptorSet> wr(R.size());
            for (size_t i = 0; i < R.size(); i++) {
                const FrameResource& r = R[i];
                lb[i] = {}; lb[i].binding = r.binding; lb[i].descriptorCount = 1;
                if (r.is_texture()) {
                    lb[i].descriptorType = r.is_storage_image
                        ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                        : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    // A set-0 texture belongs to the VERTEX shader (build_R tags VS resources into set 0,
                    // PS into set 1). stageFlags must include every stage that reads the binding, so a
                    // vertex texture fetch (displacement/heightmap, GPU vertex animation) needs
                    // VERTEX_BIT — a fragment-only hardcode made set-0 textures invisible to the VS,
                    // yielding undefined samples / a validation error (#376). Match the storage-buffer path.
                    lb[i].stageFlags = (r.set == 0)
                        ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
                    texture_references++;
                    // #1272: effective generated-mip chain — bounded by the chain the T# itself
                    // declares (declared_mip_levels; 1 = historical single-level behavior), and
                    // restricted to plain-2D RGBA8 sampled guest textures. Cube/volume stacks,
                    // storage images, RTT-backed bindings, and other formats keep one level.
                    uint32_t tex_mip_levels = 1;
                    if (!r.is_storage_image && r.img_dim == 1 && r.td == 1 &&
                        !r.persistent_render_target_id && r.tex_rgba &&
                        // Widening this format gate requires BLIT_SRC/BLIT_DST +
                        // SAMPLED_IMAGE_FILTER_LINEAR on the new format (the blit cascade below).
                        backend_color_format(r.texture_format) == VK_FORMAT_R8G8B8A8_UNORM &&
                        r.declared_mip_levels > 1 && (r.tw > 1 || r.th > 1)) {
                        uint32_t full = 1;
                        for (uint32_t m = r.tw > r.th ? r.tw : r.th; m > 1; m >>= 1) full++;
                        tex_mip_levels = r.declared_mip_levels < full ? r.declared_mip_levels : full;
                        // Symmetry with the 16-region copy cap below; unreachable from guest data
                        // (T# extents are rejected above 16384 -> <= 15 levels).
                        if (tex_mip_levels > 16u) tex_mip_levels = 16u;
                    }
                    // A DS-bridged resource shares the id slot: both are guest plane addresses, and
                    // pixels stays null for either direct bind, so distinct surfaces cannot collide.
                    const TextureUploadKey texture_key{
                        r.tex_rgba,
                        r.persistent_render_target_id ? r.persistent_render_target_id
                                                      : r.persistent_depth_target_id,
                        r.tw, r.th, r.td, r.img_dim,
                        tex_mip_levels, backend_color_format(r.texture_format), r.is_storage_image};
                    size_t upload_index = SIZE_MAX;
                    if (share_texture_uploads) {
                        auto found = texture_upload_indices.find(texture_key);
                        if (found != texture_upload_indices.end()) upload_index = found->second;
                    }
                    if (upload_index == SIZE_MAX) {
                        upload_index = texture_uploads.size();
                        texture_uploads.push_back({});
                        SharedTextureUpload& upload = texture_uploads.back();
                        upload.key = texture_key;
                        if (share_texture_uploads) texture_upload_indices.emplace(texture_key, upload_index);

                        const bool target_feedback = persistent_color &&
                            r.persistent_render_target_id == color_target->persistent_id;
                        if (!r.is_storage_image && persistent_color_targets_enabled && !target_feedback &&
                            r.persistent_render_target_id && r.img_dim == 1) {
                            if (auto* target = find_persistent_color_target(
                                    r.persistent_render_target_id, r.tw, r.th,
                                    backend_color_format(r.texture_format))) {
                                target->last_use = color_target_generation;
                                upload.image = target->image;
                                upload.image_bytes = target->bytes;
                                upload.borrowed_target = true;
                                ++color_target_stats.sampled_hits;
                            }
                        }
                        // Sampled depth bridge (#1275): the T# addresses a depth plane rendered
                        // into a persistent DS image (never written back to guest memory). Bind
                        // that image's depth aspect directly. The pass's own DS attachment must
                        // not be borrowed as a sampled input (feedback) — cached_ds identifies it.
                        if (!upload.image && !upload.borrowed_target && !r.is_storage_image &&
                            r.persistent_depth_target_id && r.img_dim == 1) {
                            const PersistentDsSampled sampled_ds = find_persistent_ds_sampled(
                                r.persistent_depth_target_id, r.tw, r.th);
                            if (sampled_ds.image &&
                                (!cached_ds || sampled_ds.image->image != cached_ds->image)) {
                                upload.image = sampled_ds.image->image;
                                upload.borrowed_ds = true;
                                upload.ds_format = sampled_ds.format;
                            }
                        }
                        upload.persistent_id =
                            r.is_storage_image || upload.borrowed_ds ? 0 : r.persistent_texture_id;
                        const PersistentTextureKey persistent_key{
                            r.persistent_texture_id, r.tw, r.th, r.td, r.img_dim,
                            texture_key.mip_levels, backend_color_format(r.texture_format)};
                        if (!r.is_storage_image && !upload.borrowed_target && !upload.borrowed_ds &&
                            persistent_textures_enabled &&
                            r.persistent_texture_id) {
                            auto cached = persistent_texture_images.find(persistent_key);
                            if (cached != persistent_texture_images.end()) {
                                cached->second.last_use = texture_generation;
                                upload.image = cached->second.image;
                                upload.image_bytes = cached->second.bytes;
                                upload.persistent_hit = true;
                                ++persistent_texture_hits;
                            } else {
                                ++persistent_texture_misses;
                            }
                        }

                        if (!upload.persistent_hit && !upload.borrowed_target && !upload.borrowed_ds) {
                            VkImageCreateInfo tci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                            const bool texture_3d = r.img_dim == 2;
                            tci.imageType = texture_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
                            tci.format = backend_color_format(r.texture_format);
                            tci.extent = {r.tw, r.th, r.td};
                            tci.mipLevels = upload.key.mip_levels; tci.arrayLayers = 1;
                            tci.samples = VK_SAMPLE_COUNT_1_BIT; tci.tiling = VK_IMAGE_TILING_OPTIMAL;
                            tci.usage = (r.is_storage_image ? VK_IMAGE_USAGE_STORAGE_BIT
                                                          : VK_IMAGE_USAGE_SAMPLED_BIT) |
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        (upload.key.mip_levels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                                                   : 0u);   // blit-cascade source (#1272)
                            vkCreateImage(dev, &tci, nullptr, &upload.image);
                            VkMemoryRequirements tr;
                            vkGetImageMemoryRequirements(dev, upload.image, &tr);
                            upload.image_bytes = tr.size;
                            VkMemoryAllocateInfo tai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                            tai.allocationSize = tr.size;
                            tai.memoryTypeIndex = pick(tr.memoryTypeBits, 0);
                            const bool retain = !r.is_storage_image && persistent_textures_enabled &&
                                                upload.persistent_id &&
                                                tr.size <= persistent_texture_limit;
                            if (retain && vkAllocateMemory(dev, &tai, nullptr, &upload.memory) == VK_SUCCESS)
                                upload.direct_memory = true;
                            if (!upload.memory) {
                                upload.persistent_id = 0;
                                upload.memory = allocate_transient_render_memory(
                                    dev, tai.allocationSize, tai.memoryTypeIndex);
                            }
                            vkBindImageMemory(dev, upload.image, upload.memory, 0);

                            // #1272: staging carries level 0 only; levels 1..N-1 are produced on the
                            // GPU by a linear-filtered vkCmdBlitImage cascade at upload time (see the
                            // copy site). A CPU box filter here was the first implementation and
                            // collapsed titles that re-upload large mip-eligible textures per frame
                            // (Evergate's title froze the publish rate — snapshot-gate catch).
                            const VkDeviceSize tbytes =
                                static_cast<VkDeviceSize>(r.tw) * r.th * r.td *
                                backend_color_bytes_per_pixel(r.texture_format);
                            VkBufferCreateInfo stci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                            stci.size = tbytes; stci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                            vkCreateBuffer(dev, &stci, nullptr, &upload.staging);
                            VkMemoryRequirements sr;
                            vkGetBufferMemoryRequirements(dev, upload.staging, &sr);
                            VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                            sai.allocationSize = sr.size;
                            sai.memoryTypeIndex = pick(sr.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                            upload.staging_memory = allocate_transient_render_memory(
                                dev, sai.allocationSize, sai.memoryTypeIndex);
                            vkBindBufferMemory(dev, upload.staging, upload.staging_memory, 0);
                            void* sp = nullptr;
                            vkMapMemory(dev, upload.staging_memory, 0, tbytes, 0, &sp);
                            if (r.tex_rgba) {
                                std::memcpy(sp, r.tex_rgba, static_cast<size_t>(tbytes));
                            } else {
                                // Only a declined/missed depth-plane borrow reaches the creation
                                // path with no CPU pixels (#1275: the bridge deliberately carries
                                // none — e.g. the consumer samples its own bound DS attachment, or
                                // the plane was invalidated between the frontend gate and this
                                // lookup). Bind well-defined zeros — the value the guest-byte
                                // decode of an unwritten depth address produced — and say so.
                                std::memset(sp, 0, static_cast<size_t>(tbytes));
                                static int declined_logged = 0;
                                if (declined_logged++ < 16)
                                    fprintf(stderr,
                                            "[dsbridge] borrow declined/missed for 0x%llx %ux%u -> "
                                            "zero texture\n",
                                            (unsigned long long)r.persistent_depth_target_id,
                                            r.tw, r.th);
                            }
                            vkUnmapMemory(dev, upload.staging_memory);
                        }
                    }
                    const SharedTextureUpload& upload = texture_uploads[upload_index];
                    VkImageViewCreateInfo tvci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                    // NOTE(#263): r.srgb carries whether the T# is a gamma-encoded (sRGB) surface, but we
                    // deliberately keep the view UNORM. This whole renderer works in gamma/sRGB space
                    // end-to-end (this target is UNORM, the frontend blit + swapchain are UNORM), with NO
                    // linear->sRGB encode at present. Sampling an sRGB texture as UNORM passes its encoded
                    // bytes straight through, which MATCHES real-hardware output for pass-through content.
                    // Flipping this to VK_FORMAT_R8G8B8A8_SRGB would apply sRGB->linear on sample with no
                    // matching encode on store -> linear values into a UNORM swapchain -> too dark. A
                    // correct sRGB fix is a coordinated linear-working-space + output-encode change (see the
                    // #263 discussion), NOT a per-view format flip. r.srgb is decoded now as groundwork.
                    tvci.image = upload.image;
                    tvci.viewType = r.img_dim == 2 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
                    tvci.format = backend_color_format(r.texture_format);
                    // T# DST_SEL channel remap (#261): map each SQ_SEL to a VkComponentSwizzle. Identity
                    // (the default, and the narrow/font path) yields IDENTITY == a no-op. PROSPER_NO_SWIZZLE
                    // forces identity for A/B testing against the pre-swizzle behavior.
                    auto vkswz = [](uint32_t s) -> VkComponentSwizzle {
                        switch (s) {
                            case 0:  return VK_COMPONENT_SWIZZLE_ZERO;
                            case 1:  return VK_COMPONENT_SWIZZLE_ONE;
                            case 4:  return VK_COMPONENT_SWIZZLE_R;
                            case 5:  return VK_COMPONENT_SWIZZLE_G;
                            case 6:  return VK_COMPONENT_SWIZZLE_B;
                            case 7:  return VK_COMPONENT_SWIZZLE_A;
                            default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                        }
                    };
                    // Vulkan component mappings do not apply to storage-image accesses.
                    if (!r.is_storage_image && !getenv("PROSPER_NO_SWIZZLE"))
                        tvci.components = {vkswz(r.swizzle[0]), vkswz(r.swizzle[1]), vkswz(r.swizzle[2]), vkswz(r.swizzle[3])};
                    tvci.subresourceRange =
                        {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels, 0, 1};
                    // Sampled depth bridge (#1275): a borrowed DS image is viewed through its own
                    // depth format's DEPTH aspect (one level — DS surfaces have no mip chains here);
                    // the sampled value arrives in R. The T# swizzle above still applies.
                    if (upload.borrowed_ds) {
                        tvci.format = upload.ds_format;
                        tvci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                    }
                    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                    // Honor the game's decoded S# (r.mag/min/mip_filter, r.addr_uvw) instead of a fixed
                    // LINEAR/clamp sampler — point-sampled art (pixel-art titles) no longer gets a blurred
                    // per-texel outline, and real wrap modes work. Gen5 CLAMP enum -> Vk address mode.
                    auto vkflt  = [](uint32_t f){ return f ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; };
                    auto vkaddr = [](uint32_t c) -> VkSamplerAddressMode {
                        switch (c) {
                            case 0:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                            case 1:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                            case 6: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // 2,3,4,5: clamp-ish
                        }
                    };
                    sci.magFilter = vkflt(r.mag_filter); sci.minFilter = vkflt(r.min_filter);
                    sci.mipmapMode = r.mip_filter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    // Sampled depth bridge (#1275): LINEAR filtering of depth formats is an OPTIONAL
                    // Vulkan format feature, and the manual-compare lowering takes single taps
                    // anyway (filter-then-compare would differ from hardware's compare-then-filter
                    // PCF regardless). Force NEAREST for borrowed depth views.
                    if (upload.borrowed_ds) {
                        sci.magFilter = VK_FILTER_NEAREST;
                        sci.minFilter = VK_FILTER_NEAREST;
                        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    }
                    sci.addressModeU = vkaddr(r.addr_uvw[0]);
                    sci.addressModeV = vkaddr(r.addr_uvw[1]);
                    sci.addressModeW = vkaddr(r.addr_uvw[2]);
                    // Remaining S# fields (#262), applied where valid on this color combined-image-sampler.
                    // Defaults (border 0 / LOD 0,0 / bias 0) reproduce the previous fixed sampler exactly.
                    //   border color: only bites with CLAMP_TO_BORDER wrap. 3 = register/custom (needs
                    //     VK_EXT_custom_border_color); fall back to opaque-black.
                    //   LOD min/max/bias: honored; harmless with our single uploaded mip.
                    // Anisotropy (#275): applied when the S# requests a ratio, the device feature is
                    // enabled, and filtering is linear (Vulkan requires anisotropyEnable only with linear
                    // mag/min filters). maxAnisotropy = 1<<ratio, clamped to the device ceiling. ratio 0
                    // (isotropic) leaves anisotropyEnable false -> the sampler is unchanged.
                    if (r.max_aniso_ratio > 0 && aniso_enabled &&
                        sci.magFilter == VK_FILTER_LINEAR && sci.minFilter == VK_FILTER_LINEAR) {
                        sci.anisotropyEnable = VK_TRUE;
                        float want = (float)(1u << r.max_aniso_ratio);
                        sci.maxAnisotropy = want < max_aniso_limit ? want : max_aniso_limit;
                    }
                    // NOT applied here (need machinery the current path lacks — decoded under GFXLOG only):
                    //   depth_compare_func (needs a depth/shadow sampler over a depth image),
                    //   unnormalized coords (strict validity rules + recompiler coord semantics).
                    switch (r.border_color_type) {
                        case 1:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;
                        case 2:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; break;
                        case 3:  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK; break;   // custom unsupported
                        default: sci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK; break;
                    }
                    sci.minLod = r.min_lod; sci.maxLod = r.max_lod; sci.mipLodBias = r.lod_bias;
                    TextureBindingKey binding_key;
                    binding_key.words = {
                        handle_bits(upload.image), static_cast<uint64_t>(tvci.viewType),
                        static_cast<uint64_t>(tvci.format),
                        static_cast<uint64_t>(tvci.components.r),
                        static_cast<uint64_t>(tvci.components.g),
                        static_cast<uint64_t>(tvci.components.b),
                        static_cast<uint64_t>(tvci.components.a),
                        static_cast<uint64_t>(r.is_storage_image),
                        static_cast<uint64_t>(sci.magFilter),
                        static_cast<uint64_t>(sci.minFilter),
                        static_cast<uint64_t>(sci.mipmapMode),
                        static_cast<uint64_t>(sci.addressModeU),
                        static_cast<uint64_t>(sci.addressModeV),
                        static_cast<uint64_t>(sci.addressModeW),
                        static_cast<uint64_t>(sci.borderColor),
                        static_cast<uint64_t>(float_bits(sci.minLod)),
                        static_cast<uint64_t>(float_bits(sci.maxLod)),
                        static_cast<uint64_t>(float_bits(sci.mipLodBias)),
                        static_cast<uint64_t>(sci.anisotropyEnable),
                        static_cast<uint64_t>(float_bits(sci.maxAnisotropy)),
                        0,
                        share_backend_resources ? 0 : ++resource_unique_tag,
                    };
                    ++resource_reuse_stats.texture_binding_references;
                    size_t binding_index = SIZE_MAX;
                    auto binding_found = shared_texture_binding_indices.find(binding_key);
                    if (binding_found != shared_texture_binding_indices.end()) {
                        binding_index = binding_found->second;
                    } else {
                        binding_index = shared_texture_bindings.size();
                        SharedTextureBinding binding;
                        const bool persistent_bindings_enabled = share_backend_resources &&
                            persistent_textures_enabled &&
                            getenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS") == nullptr;
                        auto persistent_image = persistent_texture_images.end();
                        if (persistent_bindings_enabled && upload.persistent_hit &&
                            upload.persistent_id && !r.is_storage_image) {
                            persistent_image = persistent_texture_images.find(
                                {upload.persistent_id, upload.key.width, upload.key.height,
                                 upload.key.depth, upload.key.img_dim,
                                 upload.key.mip_levels, upload.key.format});
                        }
                        if (persistent_image != persistent_texture_images.end()) {
                            auto cached_binding = persistent_image->second.bindings.find(binding_key);
                            if (cached_binding != persistent_image->second.bindings.end()) {
                                cached_binding->second.last_use = texture_generation;
                                binding.view = cached_binding->second.view;
                                binding.sampler = cached_binding->second.sampler;
                                binding.persistent = true;
                                ++resource_reuse_stats.persistent_texture_binding_hits;
                            } else {
                                ++resource_reuse_stats.persistent_texture_binding_misses;
                                constexpr size_t max_bindings_per_texture = 32;
                                if (!avoid_cache_eviction &&
                                    persistent_image->second.bindings.size() >=
                                        max_bindings_per_texture) {
                                    auto victim = persistent_image->second.bindings.end();
                                    for (auto it = persistent_image->second.bindings.begin();
                                         it != persistent_image->second.bindings.end(); ++it) {
                                        if (it->second.last_use == texture_generation) continue;
                                        if (victim == persistent_image->second.bindings.end() ||
                                            it->second.last_use < victim->second.last_use)
                                            victim = it;
                                    }
                                    if (victim != persistent_image->second.bindings.end()) {
                                        if (victim->second.sampler)
                                            vkDestroySampler(dev, victim->second.sampler, nullptr);
                                        if (victim->second.view)
                                            vkDestroyImageView(dev, victim->second.view, nullptr);
                                        persistent_image->second.bindings.erase(victim);
                                        ++resource_reuse_stats.persistent_texture_binding_evictions;
                                    }
                                }
                                vkCreateImageView(dev, &tvci, nullptr, &binding.view);
                                vkCreateSampler(dev, &sci, nullptr, &binding.sampler);
                                if (persistent_image->second.bindings.size() <
                                    max_bindings_per_texture) {
                                    persistent_image->second.bindings.emplace(
                                        binding_key, PersistentTextureBinding{
                                            binding.view, binding.sampler, texture_generation});
                                    binding.persistent = true;
                                }
                            }
                        } else {
                            vkCreateImageView(dev, &tvci, nullptr, &binding.view);
                            if (!r.is_storage_image)
                                vkCreateSampler(dev, &sci, nullptr, &binding.sampler);
                        }
                        shared_texture_bindings.push_back(binding);
                        shared_texture_binding_indices.emplace(binding_key, binding_index);
                        ++resource_reuse_stats.unique_texture_bindings;
                    }
                    const VkImageLayout image_layout = r.is_storage_image
                        ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    dii[i] = {shared_texture_bindings[binding_index].sampler,
                              shared_texture_bindings[binding_index].view, image_layout};
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                    wr[i].descriptorType = lb[i].descriptorType; wr[i].pImageInfo = &dii[i];
                } else {
                    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                    if (r.is_internal_gds) {
                        const RenderHostBuffer& gds = render_internal_gds_buffer();
                        dbi[i] = {gds.buffer, 0, 64u * 1024u};
                        wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        wr[i].pBufferInfo = &dbi[i];
                        continue;
                    }
                    const uint32_t* words = r.buffer_words_data();
                    size_t word_count = r.buffer_word_count();
                    if (!words || !word_count) {
                        words = &zero_buffer_word;
                        word_count = 1;
                    }
                    ++resource_reuse_stats.buffer_references;
                    ++backend_hash_stats_totals().references;
                    const bool shareable = share_backend_resources && r.buffer_identity != 0;
                    size_t buffer_index = SIZE_MAX;
                    const BufferRefMemoKey memo_key{words, word_count, r.buffer_identity};
                    if (shareable) {
                        auto memo_found = buffer_ref_memo.find(memo_key);
                        if (memo_found != buffer_ref_memo.end()) {
                            buffer_index = memo_found->second;
                            ++resource_reuse_stats.buffer_ref_memo_hits;
                            ++backend_hash_stats_totals().memo_hits;
                        }
                    }
                    if (buffer_index != SIZE_MAX) {
                        // repeat reference within this call — resolved without hashing (#1268)
                    } else {
                    // A unique-tag key can never match an existing entry (the tag differs from every
                    // other key and operator== checks all scalars before the memcmp), so its content
                    // hash contributes nothing to the lookup — skip it (#1268).
                    //
                    // Large buffers also take a unique tag: cross-pointer content dedup is kept for
                    // SMALL buffers only (<= kSharedBufferHashDedupMaxDwords). Measured live on Blue
                    // Prince's loading submits, the content-hash lookup had ZERO dedup hits across
                    // 556K hashed references while costing ~1.8 GiB/s of FNV over ~97 KiB average
                    // payloads — pure waste at exactly the draw volume where it hurts. Repeat
                    // references to a large range still resolve through the per-call memo above;
                    // what a large buffer loses is only the merge of two DIFFERENT-pointer,
                    // identical-content references within one call, which the live data shows never
                    // happens. Small buffers (per-draw UBO-sized, the tests' contract) keep the full
                    // hash + memcmp dedup exactly as before.
                    constexpr size_t kSharedBufferHashDedupMaxDwords = 1024;   // 4 KiB
                    const bool hash_dedup =
                        shareable && word_count <= kSharedBufferHashDedupMaxDwords;
                    uint64_t content_hash = 0;
                    if (hash_dedup) {
                        content_hash = hash_buffer_words(words, word_count);
                        ++resource_reuse_stats.buffer_hash_calls;
                        resource_reuse_stats.buffer_hash_dwords += word_count;
                        ++backend_hash_stats_totals().hash_calls;
                        backend_hash_stats_totals().hash_dwords += word_count;
                    } else if (shareable) {
                        ++resource_reuse_stats.buffer_hash_skipped_large;
                        ++backend_hash_stats_totals().skipped_large;
                    } else {
                        ++resource_reuse_stats.buffer_hash_skipped_unique;
                        ++backend_hash_stats_totals().skipped_unique;
                    }
                    SharedBufferKey buffer_key{
                        words, word_count, r.buffer_identity, content_hash,
                        hash_dedup ? 0 : ++resource_unique_tag};
                    auto buffer_found = shared_buffer_indices.find(buffer_key);
                    if (buffer_found != shared_buffer_indices.end()) {
                        buffer_index = buffer_found->second;
                    } else {
                        buffer_index = shared_buffers.size();
                        SharedBufferUpload upload;
                        const VkDeviceSize bytes = static_cast<VkDeviceSize>(word_count) * 4;
                        if (use_buffer_arena)
                            acquire_buffer_arena_slice(bytes, upload);
                        if (!upload.arena && reuse_host_buffers) {
                            RenderHostBuffer pooled = acquire_render_host_buffer(ctx, bytes);
                            upload.buffer = pooled.buffer;
                            upload.memory = pooled.memory;
                            upload.mapped = pooled.mapped;
                            upload.bytes = pooled.bytes;
                            upload.allocation_bytes = pooled.allocation_bytes;
                            upload.pooled = upload.buffer && upload.memory && upload.mapped;
                        }
                        if (!upload.arena && !upload.pooled) {
                            VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                            buffer_info.size = bytes;
                            buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                            vkCreateBuffer(dev, &buffer_info, nullptr, &upload.buffer);
                            VkMemoryRequirements requirements;
                            vkGetBufferMemoryRequirements(dev, upload.buffer, &requirements);
                            const uint32_t memory_type = pick(
                                requirements.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                            upload.memory = allocate_transient_render_memory(
                                dev, requirements.size, memory_type);
                            vkBindBufferMemory(dev, upload.buffer, upload.memory, 0);
                            void* mapped = nullptr;
                            vkMapMemory(dev, upload.memory, 0, bytes, 0, &mapped);
                            std::memcpy(mapped, words, static_cast<size_t>(bytes));
                            vkUnmapMemory(dev, upload.memory);
                        } else {
                            std::memcpy(static_cast<uint8_t*>(upload.mapped) + upload.offset,
                                        words, static_cast<size_t>(bytes));
                        }
                        shared_buffers.push_back(upload);
                        shared_buffer_indices.emplace(buffer_key, buffer_index);
                        ++resource_reuse_stats.unique_buffers;
                        ++backend_hash_stats_totals().unique_buffers;
                    }
                    if (shareable) buffer_ref_memo.emplace(memo_key, buffer_index);
                    }
                    dbi[i] = {shared_buffers[buffer_index].buffer,
                              shared_buffers[buffer_index].offset,
                              static_cast<VkDeviceSize>(word_count) * sizeof(uint32_t)};
                    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstBinding = r.binding; wr[i].descriptorCount = 1;
                    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
                }
            }
            for (uint32_t s = 0; s < v.n_sets; s++) {
                std::vector<VkDescriptorSetLayoutBinding> slb;
                for (size_t i = 0; i < R.size(); i++) if (R[i].set == s) slb.push_back(lb[i]);
                std::sort(slb.begin(), slb.end(), [](const auto& left, const auto& right) {
                    return left.binding < right.binding;
                });
                std::vector<uint64_t> layout_key;
                layout_key.reserve(1 + slb.size() * 4);
                layout_key.push_back(share_backend_resources ? 0 : ++resource_unique_tag);
                for (const auto& binding : slb) {
                    layout_key.push_back(binding.binding);
                    layout_key.push_back(binding.descriptorType);
                    layout_key.push_back(binding.descriptorCount);
                    layout_key.push_back(binding.stageFlags);
                }
                descriptor_layout_keys[s] = layout_key;
                ++resource_reuse_stats.descriptor_set_layout_references;
                auto layout_found = shared_descriptor_set_layouts.find(layout_key);
                if (layout_found != shared_descriptor_set_layouts.end()) {
                    dsls[s] = layout_found->second;
                } else {
                    VkDescriptorSetLayoutCreateInfo layout_info{
                        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                    layout_info.bindingCount = static_cast<uint32_t>(slb.size());
                    layout_info.pBindings = slb.data();
                    vkCreateDescriptorSetLayout(dev, &layout_info, nullptr, &dsls[s]);
                    shared_descriptor_set_layouts.emplace(std::move(layout_key), dsls[s]);
                    ++resource_reuse_stats.unique_descriptor_set_layouts;
                }
            }
            VkDescriptorSetAllocateInfo allocate_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocate_info.descriptorPool = shared_descriptor_pool;
            allocate_info.descriptorSetCount = v.n_sets;
            allocate_info.pSetLayouts = dsls.data();
            vkAllocateDescriptorSets(dev, &allocate_info, v.dsets.data());
            for (size_t i = 0; i < R.size(); i++)
                wr[i].dstSet = v.dsets[R[i].set];
            vkUpdateDescriptorSets(dev, static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
        }
        const auto setup_resources_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) setup_resources_ms += setup_elapsed_ms(setup_fixed_ready, setup_resources_ready);
        std::vector<uint64_t> pipeline_layout_key;
        size_t pipeline_layout_key_words = 1;
        if (v.use_desc) {
            for (const auto& descriptor_layout_key : descriptor_layout_keys)
                pipeline_layout_key_words += 1 + descriptor_layout_key.size();
        }
        pipeline_layout_key.reserve(pipeline_layout_key_words);
        pipeline_layout_key.push_back(share_backend_resources ? 0 : ++resource_unique_tag);
        if (v.use_desc) {
            for (const auto& descriptor_layout_key : descriptor_layout_keys) {
                pipeline_layout_key.push_back(descriptor_layout_key.size());
                pipeline_layout_key.insert(pipeline_layout_key.end(),
                                           descriptor_layout_key.begin(),
                                           descriptor_layout_key.end());
            }
        }
        ++resource_reuse_stats.pipeline_layout_references;
        auto pipeline_layout_found = shared_pipeline_layouts.find(pipeline_layout_key);
        if (pipeline_layout_found != shared_pipeline_layouts.end()) {
            v.layout = pipeline_layout_found->second.handle;
        } else {
            SharedPipelineLayout shared_layout;
            const bool can_persist_pipeline_layout = pipeline_layout_cache_enabled &&
                pipeline_layout_cache_limit;
            auto persistent_layout = can_persist_pipeline_layout
                ? persistent_pipeline_layouts.find(pipeline_layout_key)
                : persistent_pipeline_layouts.end();
            if (persistent_layout != persistent_pipeline_layouts.end()) {
                persistent_layout->second.last_use = pipeline_layout_generation;
                shared_layout.handle = persistent_layout->second.handle;
                shared_layout.persistent = true;
                ++resource_reuse_stats.persistent_pipeline_layout_hits;
            } else {
                VkPipelineLayoutCreateInfo layout_info{
                    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                if (v.use_desc) {
                    layout_info.setLayoutCount = v.n_sets;
                    layout_info.pSetLayouts = dsls.data();
                }
                vkCreatePipelineLayout(dev, &layout_info, nullptr, &shared_layout.handle);
                if (can_persist_pipeline_layout) {
                    ++resource_reuse_stats.persistent_pipeline_layout_misses;
                    while (!avoid_cache_eviction &&
                           persistent_pipeline_layouts.size() >= pipeline_layout_cache_limit &&
                           evict_persistent_pipeline_layout()) {}
                    if (shared_layout.handle &&
                        persistent_pipeline_layouts.size() < pipeline_layout_cache_limit) {
                        persistent_pipeline_layouts.emplace(
                            pipeline_layout_key,
                            PersistentBackendPipelineLayout{
                                shared_layout.handle, pipeline_layout_generation});
                        shared_layout.persistent = true;
                    }
                }
            }
            v.layout = shared_layout.handle;
            shared_pipeline_layouts.emplace(std::move(pipeline_layout_key), shared_layout);
            ++resource_reuse_stats.unique_pipeline_layouts;
        }
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = bd_gs.empty() ? 2u : 3u; gp.pStages = st;
        gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vpst; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb; gp.pDynamicState = &dynamic_state;
        gp.layout = v.layout; gp.renderPass = rp; gp.subpass = 0;
        if (ps && (ps->depth_test_enable || ps->stencil_enable)) gp.pDepthStencilState = &dss;
        ++pipeline_stats.references;
        bool create_pipeline = true;
        PersistentPipelineKey pipeline_key;
        if (pipeline_cache_enabled && pipeline_cache_limit) {
            auto append = [&](uint32_t word) {
                if (pipeline_key.word_count < PersistentPipelineKey::kInlineWords)
                    pipeline_key.inline_words[pipeline_key.word_count] = word;
                else
                    pipeline_key.overflow_words.push_back(word);
                ++pipeline_key.word_count;
                pipeline_key.hash ^= word;
                pipeline_key.hash *= 1099511628211ull;
            };
            auto append_float = [&](float value) {
                uint32_t word = 0;
                static_assert(sizeof word == sizeof value);
                memcpy(&word, &value, sizeof word);
                append(word);
            };
            append(8); // key schema version (depth bias in rasterization state, #1349)
            append(W); append(H); append(color_count); append(static_cast<uint32_t>(FMT));
            append(static_cast<uint32_t>(FMT1)); append(use_ds); append(static_cast<uint32_t>(DFMT));
            const bool exact_shader_identities = bd.vs_identity && bd.fs_identity;
            append(bd.vs_identity != 0);
            if (bd.vs_identity) {
                append(static_cast<uint32_t>(bd.vs_identity));
                append(static_cast<uint32_t>(bd.vs_identity >> 32));
            } else {
                append(static_cast<uint32_t>(bd_vs.size()));
                for (uint32_t word : bd_vs) append(word);
            }
            append(bd.fs_identity != 0);
            if (bd.fs_identity) {
                append(static_cast<uint32_t>(bd.fs_identity));
                append(static_cast<uint32_t>(bd.fs_identity >> 32));
            } else {
                append(static_cast<uint32_t>(bd_fs.size()));
                for (uint32_t word : bd_fs) append(word);
            }
            append(required_fragment_subgroup_size);
            append(static_cast<uint32_t>(bd_gs.size()));
            for (uint32_t word : bd_gs) append(word);
            append(v.use_desc); append(v.n_sets);
            // A pair of shader-cache identities names the exact compile keys, including every
            // descriptor's class and binding. External/replay shaders have identity zero and keep
            // the full layout contract in the fallback key.
            append(!exact_shader_identities);
            if (!exact_shader_identities) {
                append(static_cast<uint32_t>(R.size()));
                for (size_t i = 0; i < R.size(); ++i) {
                    const bool texture = R[i].is_texture();
                    const VkDescriptorType descriptor_type = !texture
                        ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                        : (R[i].is_storage_image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
                    const VkShaderStageFlags stage_flags = !texture || R[i].set == 0
                        ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                        : VK_SHADER_STAGE_FRAGMENT_BIT;
                    append(R[i].set); append(R[i].binding); append(descriptor_type);
                    append(1); append(stage_flags);
                }
            }
            append(ia.topology); append(ia.primitiveRestartEnable);
            append_float(vp.x); append_float(vp.y); append_float(vp.width); append_float(vp.height);
            append_float(vp.minDepth); append_float(vp.maxDepth);
            append(rs.polygonMode); append(rs.cullMode); append(rs.frontFace); append_float(rs.lineWidth);
            append(rs.depthBiasEnable); append_float(rs.depthBiasConstantFactor);
            append_float(rs.depthBiasSlopeFactor); append_float(rs.depthBiasClamp);
            append(cb.logicOpEnable); append(cb.logicOp);
            for (uint32_t attachment = 0; attachment < color_count; ++attachment) {
                append(cba[attachment].blendEnable); append(cba[attachment].srcColorBlendFactor);
                append(cba[attachment].dstColorBlendFactor); append(cba[attachment].colorBlendOp);
                append(cba[attachment].srcAlphaBlendFactor); append(cba[attachment].dstAlphaBlendFactor);
                append(cba[attachment].alphaBlendOp); append(cba[attachment].colorWriteMask);
            }
            append(gp.pDepthStencilState != nullptr);
            if (gp.pDepthStencilState) {
                append(dss.depthTestEnable); append(dss.depthWriteEnable); append(dss.depthCompareOp);
                append(dss.depthBoundsTestEnable); append(dss.stencilTestEnable);
                auto append_stencil = [&](const VkStencilOpState& stencil) {
                    append(stencil.failOp); append(stencil.passOp); append(stencil.depthFailOp);
                    append(stencil.compareOp); append(stencil.compareMask); append(stencil.writeMask);
                    append(stencil.reference);
                };
                append_stencil(dss.front); append_stencil(dss.back);
                append_float(dss.minDepthBounds); append_float(dss.maxDepthBounds);
            }
            auto found = pipeline_cache.find(pipeline_key);
            if (found != pipeline_cache.end()) {
                found->second.last_use = pipeline_generation;
                v.pipe = found->second.pipeline;
                v.pipeline_cached = true;
                v.ok = true;
                create_pipeline = false;
                ++pipeline_stats.hits;
            } else {
                ++pipeline_stats.misses;
            }
        } else {
            ++pipeline_stats.bypasses;
        }
        const auto setup_pipeline_key_ready = timing_enabled
            ? TimingClock::now() : TimingClock::time_point{};
        auto setup_pipeline_create_begin = setup_pipeline_key_ready;
        if (create_pipeline) {
            const auto setup_shader_begin = timing_enabled
                ? TimingClock::now() : TimingClock::time_point{};
            if (backend_trace) {
                fprintf(stderr, "[backend-trace] draw=%zu create-shaders begin\n", di);
                fflush(stderr);
            }
            v.vs = mkmod(bd_vs); v.gs = bd_gs.empty() ? VK_NULL_HANDLE : mkmod(bd_gs);
            v.fs = mkmod(bd_fs);
            if (backend_trace) {
                fprintf(stderr,
                        "[backend-trace] draw=%zu create-shaders end vs=%p gs=%p fs=%p\n",
                        di, (void*)v.vs, (void*)v.gs, (void*)v.fs);
                fflush(stderr);
            }
            const auto setup_shader_ready = timing_enabled
                ? TimingClock::now() : TimingClock::time_point{};
            if (timing_enabled)
                setup_shader_ms += setup_elapsed_ms(setup_shader_begin, setup_shader_ready);
            if (!v.vs || !v.fs || (!bd_gs.empty() && !v.gs)) {
                if (timing_enabled)
                    setup_pipeline_ms += setup_elapsed_ms(
                        setup_resources_ready, setup_pipeline_key_ready);
                continue;   // rejected SPIR-V -> skip this draw
            }
            st[0].module = v.vs;
            if (!bd_gs.empty()) st[1].module = v.gs;
            st[fragment_stage_index].module = v.fs;
            setup_pipeline_create_begin = setup_shader_ready;
            if (backend_trace) {
                fprintf(stderr, "[backend-trace] draw=%zu create-pipeline begin\n", di);
                fflush(stderr);
            }
            const VkResult pipeline_result = vkCreateGraphicsPipelines(
                    dev, VK_NULL_HANDLE, 1, &gp, nullptr, &v.pipe);
            if (backend_trace) {
                fprintf(stderr,
                        "[backend-trace] draw=%zu create-pipeline end result=%d pipeline=%p\n",
                        di, (int)pipeline_result, (void*)v.pipe);
                fflush(stderr);
            }
            if (pipeline_result == VK_SUCCESS) {
                v.ok = true;
                bool retain = pipeline_cache_enabled && pipeline_cache_limit;
                while (retain && !avoid_cache_eviction &&
                       pipeline_cache.size() >= pipeline_cache_limit)
                    if (!evict_pipeline()) retain = false;
                if (retain && pipeline_cache.size() >= pipeline_cache_limit) retain = false;
                if (retain) {
                    pipeline_cache.emplace(std::move(pipeline_key),
                                           PersistentPipeline{v.pipe, pipeline_generation});
                    v.pipeline_cached = true;
                }
            }
        }
        pipeline_stats.entries = pipeline_cache.size();
        const auto setup_pipeline_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
        if (timing_enabled) {
            setup_pipeline_ms += setup_elapsed_ms(setup_resources_ready, setup_pipeline_key_ready);
            setup_pipeline_ms += setup_elapsed_ms(setup_pipeline_create_begin, setup_pipeline_ready);
        }
    }

    BackendTextureUploadStats& texture_stats = backend_texture_upload_stats_storage();
    texture_stats = {};
    texture_stats.references = texture_references;
    texture_stats.persistent_hits = persistent_texture_hits;
    texture_stats.persistent_misses = persistent_texture_misses;
    for (const auto& upload : texture_uploads) {
        if (!upload.staging) continue;
        ++texture_stats.unique_uploads;
        texture_stats.upload_bytes += static_cast<uint64_t>(upload.key.width) * upload.key.height *
                                      upload.key.depth * backend_color_bytes_per_pixel(upload.key.format);
    }

    const auto timing_draws_ready = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const VkDeviceSize readback_bytes = bytes + bytes1;
    // MRT1 is currently retained in the live renderer's CPU RTT cache, so a paired pass always
    // reads both attachments back. Single-target passes keep the persistent no-readback fast path.
    const bool readback_requested = use_color1 || !persistent_color || color_target->readback;
    VkBuffer rb = VK_NULL_HANDLE;
    VkDeviceMemory bmem = VK_NULL_HANDLE;
    if (readback_requested) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = readback_bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &rb);
        VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, rb, &br);
        VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; bai.allocationSize = br.size;
    // Prefer cached host memory for GPU -> CPU readback. Discrete NVIDIA exposes an earlier coherent,
    // write-combined BAR type and a later HOST_CACHED type; the generic first-match selector chose the
    // former, making an 8 MiB 1080p read take roughly 570 ms on Windows. Upload buffers deliberately keep
    // the write-combined type. Integrated GPUs and portability drivers may not expose HOST_CACHED, so fall
    // back to the original required flags.
    constexpr VkMemoryPropertyFlags host_coherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        bai.memoryTypeIndex = pick(br.memoryTypeBits, host_coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (bai.memoryTypeIndex == UINT32_MAX)
            bai.memoryTypeIndex = pick(br.memoryTypeBits, host_coherent);
        bmem = allocate_transient_render_memory(dev, bai.allocationSize, bai.memoryTypeIndex);
        vkBindBufferMemory(dev, rb, bmem, 0);
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = qfi;
    VkCommandPool pool; vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cbai, &cmd);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);
    if (load_cached_color) {
        VkImageMemoryBarrier load{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        load.oldLayout = cached_color->layout;
        load.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        load.image = img;
        load.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // A previous target pass may be an earlier command buffer in the same queue submission.
        // Command-buffer order does not itself make its attachment writes visible, so include the
        // producer access/stage as well as the layouts in which an already-flushed target can rest.
        load.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        load.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &load);
    }
    // Preload the color attachment with the seed pixels (render-target persistence): staging upload +
    // transition to COLOR_ATTACHMENT_OPTIMAL, matching att[0]'s LOAD/initialLayout above.
    VkBuffer seedbuf = VK_NULL_HANDLE; VkDeviceMemory seedmem = VK_NULL_HANDLE;
    if (seed_rgba) {
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &seedbuf);
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, seedbuf, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        seedmem = allocate_transient_render_memory(dev, sai.allocationSize,
                                                    sai.memoryTypeIndex);
        vkBindBufferMemory(dev, seedbuf, seedmem, 0);
        void* sp = nullptr; vkMapMemory(dev, seedmem, 0, bytes, 0, &sp);
        memcpy(sp, seed_rgba, (size_t)bytes); vkUnmapMemory(dev, seedmem);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = img; s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.srcAccessMask = 0; s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, seedbuf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = img; s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    VkBuffer seedbuf1 = VK_NULL_HANDLE; VkDeviceMemory seedmem1 = VK_NULL_HANDLE;
    if (use_color1 && seed_rgba1) {
        VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sci.size = bytes1; sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &sci, nullptr, &seedbuf1);
        VkMemoryRequirements sr; vkGetBufferMemoryRequirements(dev, seedbuf1, &sr);
        VkMemoryAllocateInfo sai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; sai.allocationSize = sr.size;
        sai.memoryTypeIndex = pick(sr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        seedmem1 = allocate_transient_render_memory(dev, sai.allocationSize, sai.memoryTypeIndex);
        vkBindBufferMemory(dev, seedbuf1, seedmem1, 0);
        void* sp = nullptr; vkMapMemory(dev, seedmem1, 0, bytes1, 0, &sp);
        memcpy(sp, seed_rgba1, static_cast<size_t>(bytes1)); vkUnmapMemory(dev, seedmem1);
        VkImageMemoryBarrier s0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; s0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s0.image = img1; s0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &s0);
        VkBufferImageCopy sc{}; sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sc.imageExtent = {W, H, 1};
        vkCmdCopyBufferToImage(cmd, seedbuf1, img1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sc);
        VkImageMemoryBarrier s1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        s1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s1.image = img1; s1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        s1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        s1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &s1);
    }
    // Upload each distinct texture once. Draw descriptors may use separate views/samplers over the
    // same image, preserving per-binding swizzle and sampler state without duplicating pixel storage.
    for (const auto& upload : texture_uploads) {
        if (!upload.staging) continue;  // exact-validated persistent image already has shader-read layout
        VkImageMemoryBarrier b0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b0.image = upload.image;
        b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels, 0, 1};
        b0.srcAccessMask = 0; b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b0);
        VkBufferImageCopy tc{}; tc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        tc.imageExtent = {upload.key.width, upload.key.height, upload.key.depth};
        vkCmdCopyBufferToImage(cmd, upload.staging, upload.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &tc);
        // #1272: generate levels 1..N-1 with a linear-filtered blit cascade (GPU-side, once per
        // upload — a CPU box filter here collapsed titles that re-upload large textures per frame).
        // Each source level transitions DST->SRC before feeding the next; the final barrier below
        // then flips the whole chain to shader-read. RGBA8 linear-blit support is mandatory Vulkan.
        for (uint32_t l = 1; l < upload.key.mip_levels; l++) {
            VkImageMemoryBarrier bs{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            bs.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            bs.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bs.image = upload.image;
            bs.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 1, 0, 1};
            bs.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bs.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &bs);
            const int32_t sw = (int32_t)(upload.key.width >> (l - 1) ? upload.key.width >> (l - 1) : 1u);
            const int32_t sh = (int32_t)(upload.key.height >> (l - 1) ? upload.key.height >> (l - 1) : 1u);
            const int32_t dw = (int32_t)(upload.key.width >> l ? upload.key.width >> l : 1u);
            const int32_t dh = (int32_t)(upload.key.height >> l ? upload.key.height >> l : 1u);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, 1};
            blit.srcOffsets[1] = {sw, sh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            blit.dstOffsets[1] = {dw, dh, 1};
            vkCmdBlitImage(cmd, upload.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           upload.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                           VK_FILTER_LINEAR);
        }
        if (upload.key.mip_levels > 1) {
            // Levels 0..N-2 sit in TRANSFER_SRC after feeding the cascade; return them to
            // TRANSFER_DST so the single final-layout barrier below covers the whole chain.
            VkImageMemoryBarrier br{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            br.image = upload.image;
            br.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels - 1, 0, 1};
            br.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            br.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &br);
        }
        VkImageMemoryBarrier b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b1.newLayout = upload.key.storage_image
            ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b1.image = upload.image;
        b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, upload.key.mip_levels, 0, 1};
        b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            (upload.key.storage_image ? VK_ACCESS_SHADER_WRITE_BIT : 0);
        // The dst stage must cover EVERY stage that samples this image. #376 made set-0 textures
        // VS-visible (stageFlags VERTEX|FRAGMENT), so a vertex texture fetch reads it in the VERTEX
        // stage — a FRAGMENT-only barrier leaves the transfer-write→shader-read dependency unordered
        // for that stage (SYNC-HAZARD-READ-AFTER-WRITE; garbage vertex fetch on GPUs that don't
        // over-synchronize). Include the vertex stage to match the binding's stageFlags (#454).
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b1);
    }
    // Sampled depth bridge (#1275): borrowed DS images live in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    // between passes. Transition each (once) to SHADER_READ_ONLY for this pass's sampling; the
    // matching post-pass barrier below returns it so the next depth pass LOADs it unchanged.
    // Both aspects of a combined image must transition together.
    std::vector<std::pair<VkImage, VkFormat>> borrowed_ds_images;
    for (const SharedTextureUpload& upload : texture_uploads) {
        if (!upload.borrowed_ds || !upload.image) continue;
        if (std::any_of(borrowed_ds_images.begin(), borrowed_ds_images.end(),
                        [&](const auto& entry) { return entry.first == upload.image; }))
            continue;
        borrowed_ds_images.push_back({upload.image, upload.ds_format});
        const bool ds_has_stencil = upload.ds_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkImageMemoryBarrier to_sampled{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_sampled.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sampled.image = upload.image;
        to_sampled.subresourceRange = {
            VK_IMAGE_ASPECT_DEPTH_BIT |
                (ds_has_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u), 0, 1, 0, 1};
        to_sampled.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        to_sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_sampled);
    }
    // Clear color: the caller's clear_rgba (game fast-clear / black on the live path), else the
    // legacy diagnostic blue. PROSPER_CLEAR_DEBUG forces blue back on even when a color is passed.
    float cc[4] = {0.0f, 0.0f, 1.0f, 1.0f};   // diagnostic blue
    if (clear_rgba && getenv("PROSPER_CLEAR_DEBUG") == nullptr)
        for (int i = 0; i < 4; i++) cc[i] = clear_rgba[i];
    float cc1[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    if (clear_rgba1 && getenv("PROSPER_CLEAR_DEBUG") == nullptr)
        for (int i = 0; i < 4; ++i) cc1[i] = clear_rgba1[i];
    VkClearValue clear[3]{}; clear[0].color = {{cc[0], cc[1], cc[2], cc[3]}};
    if (use_color1) clear[1].color = {{cc1[0], cc1[1], cc1[2], cc1[3]}};
    clear[ds_attachment].depthStencil = {depth_clear, stencil_clear}; // guest DB clear/default
    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass = rp; rpbi.framebuffer = fb; rpbi.renderArea = {{0, 0}, {W, H}};
    rpbi.clearValueCount = color_count + (use_ds ? 1u : 0u); rpbi.pClearValues = clear;
    if (getenv("PROSPER_PIPELOG")) {   // diag: how many draws' pipelines built + will be recorded
        int nok = 0; for (auto& v : dv) if (v.ok) nok++;
        fprintf(stderr, "[pipe] %zu draws, %d pipelines OK, use_depth=%d use_stencil=%d; counts:", dv.size(), nok, (int)use_depth, (int)use_stencil);
        for (auto& v : dv) fprintf(stderr, " %s%u", v.ok ? "" : "SKIP", v.icount ? v.icount : v.vcount);
        fprintf(stderr, "\n");
    }
    // Per-draw "fragment funnel" (PROSPER_DRAW_STATS): wrap each recorded draw in pipeline-statistics
    // + occlusion queries to show WHERE its pixels vanish (geometry clipped away, never rasterized,
    // depth/stencil-rejected, or survived) — objective per-draw truth, no oracle needed. Read-only, and
    // only active with the env var AND a real flush in THIS call (so the results are ready to read back;
    // the flush condition below is identical to `flush_now` computed after the pass). Query-pool RESET
    // must be recorded outside a render pass, so it happens here.
    const RenderVkCtx& ds_ctx = render_vk_ctx();
    const bool draw_stats = getenv("PROSPER_DRAW_STATS") && ds_ctx.pipeline_stats_enabled && !dv.empty()
                            && (!submission_batch || readback_requested || flush_submission_batch);
    VkQueryPool ds_stats_pool = VK_NULL_HANDLE, ds_occ_pool = VK_NULL_HANDLE;
    if (draw_stats) {
        const uint32_t nq = static_cast<uint32_t>(dv.size());
        VkQueryPoolCreateInfo sp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        sp.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS; sp.queryCount = nq;
        sp.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        VkQueryPoolCreateInfo op{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        op.queryType = VK_QUERY_TYPE_OCCLUSION; op.queryCount = nq;
        if (vkCreateQueryPool(dev, &sp, nullptr, &ds_stats_pool) == VK_SUCCESS &&
            vkCreateQueryPool(dev, &op, nullptr, &ds_occ_pool) == VK_SUCCESS) {
            vkCmdResetQueryPool(cmd, ds_stats_pool, 0, nq);
            vkCmdResetQueryPool(cmd, ds_occ_pool, 0, nq);
        } else {
            if (ds_stats_pool) { vkDestroyQueryPool(dev, ds_stats_pool, nullptr); ds_stats_pool = VK_NULL_HANDLE; }
            if (ds_occ_pool)   { vkDestroyQueryPool(dev, ds_occ_pool,   nullptr); ds_occ_pool   = VK_NULL_HANDLE; }
        }
    }
    const bool ds_active = ds_stats_pool != VK_NULL_HANDLE && ds_occ_pool != VK_NULL_HANDLE;

    // Geometry probe (PROSPER_GEOM_PROBE=N): capture draw N's post-transform clip-space vertices via
    // transform feedback and report where they land (degenerate / off-screen / behind-camera / NaN).
    // Requires VK_EXT_transform_feedback + the VS recompiled with gl_Position xfb-decorated (gated on
    // the same env var in gpu_executor). Only active with the env var, TF support, AND a flush here.
    const char* geom_env = getenv("PROSPER_GEOM_PROBE");
    const long geom_target = geom_env ? strtol(geom_env, nullptr, 10) : -1;
    const bool geom_probe = geom_target >= 0 && static_cast<size_t>(geom_target) < dv.size()
                            && ds_ctx.transform_feedback_enabled
                            && (!submission_batch || readback_requested || flush_submission_batch);
    static auto p_bindxfb  = reinterpret_cast<PFN_vkCmdBindTransformFeedbackBuffersEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdBindTransformFeedbackBuffersEXT"));
    static auto p_beginxfb = reinterpret_cast<PFN_vkCmdBeginTransformFeedbackEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdBeginTransformFeedbackEXT"));
    static auto p_endxfb   = reinterpret_cast<PFN_vkCmdEndTransformFeedbackEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdEndTransformFeedbackEXT"));
    VkBuffer geom_buf = VK_NULL_HANDLE; VkDeviceMemory geom_mem = VK_NULL_HANDLE;
    VkBuffer geom_counter = VK_NULL_HANDLE; VkDeviceMemory geom_counter_mem = VK_NULL_HANDLE;
    uint32_t geom_cap = 0;   // buffer capacity in vertices; the counter buffer gives the exact count written
    if (geom_probe && p_bindxfb && p_beginxfb && p_endxfb && dv[geom_target].ok) {
        const auto& tv = dv[geom_target];
        const uint64_t per_inst = tv.icount ? tv.icount : tv.vcount;
        // Transform feedback records DECOMPOSED primitives: a triangle strip/fan of N verts emits up to
        // ~3*(N-2) individual vertices. Over-size by 3x so no records are dropped; the counter buffer
        // reports how many were actually written so we never read the uninitialized tail.
        uint64_t total = per_inst * (tv.instance_count ? tv.instance_count : 1u) * 3u;
        if (total > (1u << 20)) total = (1u << 20);   // cap at 1M vertices (16 MiB)
        geom_cap = static_cast<uint32_t>(total);
        std::string gerr;
        if (geom_cap == 0 ||
            !persistent_ds_transfer_buffer(ds_ctx, static_cast<VkDeviceSize>(geom_cap) * 16,
                                           VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT,
                                           geom_buf, geom_mem, gerr) ||
            !persistent_ds_transfer_buffer(ds_ctx, 16,
                                           VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT,
                                           geom_counter, geom_counter_mem, gerr)) {
            if (geom_buf) { vkDestroyBuffer(dev, geom_buf, nullptr); vkFreeMemory(dev, geom_mem, nullptr); }
            if (geom_counter) { vkDestroyBuffer(dev, geom_counter, nullptr); vkFreeMemory(dev, geom_counter_mem, nullptr); }
            geom_buf = VK_NULL_HANDLE; geom_mem = VK_NULL_HANDLE;
            geom_counter = VK_NULL_HANDLE; geom_counter_mem = VK_NULL_HANDLE; geom_cap = 0;
        }
    }
    const bool geom_active = geom_buf != VK_NULL_HANDLE && geom_counter != VK_NULL_HANDLE;

    // ONE render pass (cleared once): record every realized draw with its own pipeline + descriptors.
    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
    for (size_t di = 0; di < dv.size(); di++) {
        auto& v = dv[di];
        const auto* ps = draws[di].ps;
        if (use_ds && ps &&
            (effective_depth_clear(ps) ||
             stencil_clear_effective(ps->stencil_clear_enable, ps->stencil_enable,
                                     ps->stencil_write_mask[0], ps->stencil_write_mask[1]))) {
            VkClearAttachment dsc{};
            if (effective_depth_clear(ps)) dsc.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (stencil_clear_effective(ps->stencil_clear_enable, ps->stencil_enable,
                                        ps->stencil_write_mask[0], ps->stencil_write_mask[1]) &&
                format_has_stencil)
                dsc.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            dsc.clearValue.depthStencil = {ps->depth_clear_value, ps->stencil_clear_value};
            VkClearRect rect{v.scissor, 0, 1};
            // A fully clipped draw legitimately has a zero-area dynamic scissor, but Vulkan requires
            // vkCmdClearAttachments rectangles to have non-zero width and height (VUID 02682/02683).
            if (dsc.aspectMask && rect.rect.extent.width && rect.rect.extent.height)
                vkCmdClearAttachments(cmd, 1, &dsc, 1, &rect);
        }
        if (!v.ok) continue;
        if (ds_active) {
            vkCmdBeginQuery(cmd, ds_stats_pool, static_cast<uint32_t>(di), 0);
            vkCmdBeginQuery(cmd, ds_occ_pool, static_cast<uint32_t>(di),
                            ds_ctx.occlusion_precise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
        vkCmdSetScissor(cmd, 0, 1, &v.scissor);
        if (v.use_desc) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
        const bool geom_here = geom_active && static_cast<long>(di) == geom_target;
        if (geom_here) {
            VkDeviceSize off = 0, sz = static_cast<VkDeviceSize>(geom_cap) * 16;
            p_bindxfb(cmd, 0, 1, &geom_buf, &off, &sz);
            p_beginxfb(cmd, 0, 0, nullptr, nullptr);
        }
        if (v.icount) {
            vkCmdBindIndexBuffer(cmd, v.ibuf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, v.icount, v.instance_count, 0, v.vertex_offset, 0);
        } else {
            vkCmdDraw(cmd, v.vcount, v.instance_count,
                      static_cast<uint32_t>(v.vertex_offset), 0);
        }
        if (geom_here) { VkDeviceSize coff = 0; p_endxfb(cmd, 0, 1, &geom_counter, &coff); }
        if (ds_active) {
            vkCmdEndQuery(cmd, ds_occ_pool, static_cast<uint32_t>(di));
            vkCmdEndQuery(cmd, ds_stats_pool, static_cast<uint32_t>(di));
        }
    }
    vkCmdEndRenderPass(cmd);
    // Fence waits used to provide the device-memory dependency between every target call. Batched
    // command buffers deliberately remove those intermediate waits, and command-buffer/submission
    // order alone permits action commands to overlap. Publish persistent attachment writes here so
    // any later command buffer in the queue can sample or LOAD them without relying on driver-wide
    // serialization. The final render-pass layouts are already correct; these same-layout barriers
    // supply the missing availability/visibility dependency.
    if (persistent_color && !readback_requested) {
        VkImageMemoryBarrier color_ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        color_ready.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        color_ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        color_ready.image = img;
        color_ready.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        color_ready.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        color_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &color_ready);
    }
    if (persistent_ds) {
        VkImageMemoryBarrier ds_ready{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        ds_ready.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ds_ready.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ds_ready.image = dimg;
        ds_ready.subresourceRange = {DASPECT, 0, 1, 0, 1};
        ds_ready.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        ds_ready.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &ds_ready);
    }
    // Sampled depth bridge (#1275): return each borrowed DS image to the layout every depth pass
    // expects, so the bridge is invisible to the existing persistent-DS contract.
    for (const auto& [borrowed, borrowed_format] : borrowed_ds_images) {
        VkImageMemoryBarrier to_ds{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_ds.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_ds.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        to_ds.image = borrowed;
        to_ds.subresourceRange = {
            VK_IMAGE_ASPECT_DEPTH_BIT |
                (borrowed_format == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                 : 0u),
            0, 1, 0, 1};
        to_ds.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_ds.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &to_ds);
    }
    if (readback_requested) {
        if (persistent_color) {
            VkImageMemoryBarrier to_readback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_readback.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_readback.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_readback.image = img;
            to_readback.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            to_readback.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            to_readback.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_readback);
        }
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {W, H, 1};
        vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp);
        if (use_color1) {
            VkBufferImageCopy cp1 = cp; cp1.bufferOffset = bytes;
            vkCmdCopyImageToBuffer(
                cmd, img1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp1);
        }
        if (persistent_color) {
            VkImageMemoryBarrier to_sample{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to_sample.image = img;
            to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            to_sample.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &to_sample);
        }
    }
    vkEndCommandBuffer(cmd);

    // Publish newly uploaded exact-version textures before a later command buffer in the same batch
    // is recorded. The image upload and every consumer remain ordered in the eventual queue submit.
    // Do not evict while an earlier command buffer is pending: it may still reference the candidate.
    auto evict_persistent_texture = [&]() {
        auto victim = persistent_texture_images.end();
        for (auto it = persistent_texture_images.begin();
             it != persistent_texture_images.end(); ++it) {
            if (it->second.last_use == texture_generation) continue;
            if (victim == persistent_texture_images.end() ||
                it->second.last_use < victim->second.last_use) victim = it;
        }
        if (victim == persistent_texture_images.end()) return false;
        for (const auto& [key, binding] : victim->second.bindings) {
            if (binding.sampler) vkDestroySampler(dev, binding.sampler, nullptr);
            if (binding.view) vkDestroyImageView(dev, binding.view, nullptr);
        }
        vkDestroyImage(dev, victim->second.image, nullptr);
        vkFreeMemory(dev, victim->second.memory, nullptr);
        persistent_texture_bytes -= victim->second.bytes;
        persistent_texture_images.erase(victim);
        return true;
    };
    for (auto& upload : texture_uploads) {
        if (!upload.direct_memory || !upload.persistent_id || upload.persistent_hit) continue;
        const PersistentTextureKey key{upload.persistent_id, upload.key.width, upload.key.height,
                                       upload.key.depth, upload.key.img_dim,
                                       upload.key.mip_levels, upload.key.format};
        while (!avoid_cache_eviction &&
               (persistent_texture_images.size() >= persistent_texture_max_entries ||
                (upload.image_bytes <= persistent_texture_limit &&
                 persistent_texture_bytes > persistent_texture_limit - upload.image_bytes)) &&
               evict_persistent_texture()) {}
        if (upload.image_bytes <= persistent_texture_limit &&
            persistent_texture_images.size() < persistent_texture_max_entries &&
            persistent_texture_bytes <= persistent_texture_limit - upload.image_bytes) {
            auto [cached, inserted] = persistent_texture_images.emplace(
                key, PersistentTextureImage{upload.image, upload.memory, upload.image_bytes,
                                            texture_generation});
            if (inserted) {
                persistent_texture_bytes += upload.image_bytes;
                upload.image = VK_NULL_HANDLE;
                upload.memory = VK_NULL_HANDLE;
            }
        }
    }
    if (!avoid_cache_eviction)
        while (persistent_texture_bytes > persistent_texture_limit &&
               evict_persistent_texture()) {}
    for (const auto& [key, image] : persistent_texture_images)
        resource_reuse_stats.persistent_texture_binding_entries += image.bindings.size();
    texture_stats.persistent_cached_bytes = persistent_texture_bytes;

    const auto timing_recorded = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    active_submission.enqueue(cmd);
    if (cached_ds) {
        active_submission.add_failure_cleanup([cached_ds]() {
            cached_ds->layout_initialized = false;
            cached_ds->depth_valid = false;
            cached_ds->stencil_valid = false;
        });
        cached_ds->layout_initialized = true;
        cached_ds->depth_valid |= use_depth && depth_used_meaningfully;
        cached_ds->stencil_valid |= use_stencil;
        // Sampled depth bridge (#1275): recency for find_persistent_ds_sampled — two valid
        // entries can share a plane address (a surface re-keyed D32 -> D32S8 keeps its old
        // entry), and the most recently written one is the live truth.
        note_persistent_ds_depth_write(*cached_ds, use_depth, depth_may_be_written);
    }
    if (cached_color) {
        active_submission.add_failure_cleanup([cached_color]() {
            cached_color->valid = false;
        });
        cached_color->valid = true;
        cached_color->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    const bool flush_now = !submission_batch || readback_requested || flush_submission_batch;
    BackendSubmissionBatchResult batch_result;
    if (flush_now)
        batch_result = active_submission.submit_and_wait(dev, queue, backend_trace);
    const auto timing_gpu_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};
    const bool batch_completed = !flush_now ||
        (batch_result.submit_result == VK_SUCCESS && batch_result.wait_result == VK_SUCCESS);

    // Fragment-funnel readback (PROSPER_DRAW_STATS): one line per realized draw showing where its
    // pixels vanished. ds_active implies flush_now (the pool-creation gate above uses the same flush
    // condition), so `cmd` has completed and the results are ready. Pools are destroyed unconditionally.
    if (ds_active) {
        const uint32_t nq = static_cast<uint32_t>(dv.size());
        if (batch_completed) {
            std::vector<uint64_t> sres(static_cast<size_t>(nq) * 5, 0);  // 4 statistics + availability
            std::vector<uint64_t> ores(static_cast<size_t>(nq) * 2, 0);  // occlusion samples + availability
            vkGetQueryPoolResults(dev, ds_stats_pool, 0, nq, sres.size() * sizeof(uint64_t), sres.data(),
                                  5 * sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            vkGetQueryPoolResults(dev, ds_occ_pool, 0, nq, ores.size() * sizeof(uint64_t), ores.data(),
                                  2 * sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            for (uint32_t i = 0; i < nq; i++) {
                if (!sres[i * 5 + 4]) continue;  // unavailable -> draw was skipped (v.ok == false)
                const uint64_t verts = sres[i * 5 + 0], prims = sres[i * 5 + 1],
                               clip = sres[i * 5 + 2], fs = sres[i * 5 + 3];
                const uint64_t samp = ores[i * 2 + 1] ? ores[i * 2 + 0] : 0;
                // Funnel classification, checked in pipeline order. `samples` (occlusion) is the ground
                // truth for "survived the depth+stencil test" — it is counted even when the fragment
                // shader is optimised out for a colour-write-disabled (stencil-only) draw, so it must be
                // tested before fs_inv or such draws look falsely dead.
                const char* tag =
                    (prims == 0) ? "NO-GEOMETRY(no primitives)" :
                    (clip  == 0) ? "GEOMETRY-VANISH(clipped/degenerate/offscreen)" :
                    (samp  >  0) ? "passed-samples(colour/stencil written)" :
                    (fs    >  0) ? "TEST-KILLED(depth/stencil rejected all)" :
                                   "NO-RASTER(cull/scissor/zero-area)";
                fprintf(stderr,
                        "[draw-stats] draw=%u verts=%llu prims=%llu after_clip=%llu fs_inv=%llu samples=%llu %s\n",
                        i, (unsigned long long)verts, (unsigned long long)prims,
                        (unsigned long long)clip, (unsigned long long)fs, (unsigned long long)samp, tag);
            }
        }
        vkDestroyQueryPool(dev, ds_stats_pool, nullptr);
        vkDestroyQueryPool(dev, ds_occ_pool, nullptr);
    }

    // Geometry-probe readback: report where the probed draw's post-transform clip-space vertices landed.
    // geom_active implies flush_now (same gate as buffer creation), so `cmd` has completed here.
    if (geom_active) {
        if (batch_completed) {
            // The counter buffer holds the byte count transform feedback actually wrote; read only that
            // many vertices so the uninitialized tail of the (3x-oversized) buffer never pollutes stats.
            uint32_t written = 0;
            void* cp = nullptr;
            if (vkMapMemory(dev, geom_counter_mem, 0, 16, 0, &cp) == VK_SUCCESS) {
                uint32_t bytes = 0; std::memcpy(&bytes, cp, sizeof bytes);
                written = std::min(bytes / 16u, geom_cap);
                vkUnmapMemory(dev, geom_counter_mem);
            }
            void* gp = nullptr;
            if (written && vkMapMemory(dev, geom_mem, 0, static_cast<VkDeviceSize>(written) * 16, 0, &gp) == VK_SUCCESS) {
                const float* pos = static_cast<const float*>(gp);
                float minx=1e30f,maxx=-1e30f,miny=1e30f,maxy=-1e30f,minz=1e30f,maxz=-1e30f,minw=1e30f,maxw=-1e30f;
                uint32_t nan=0, wle0=0, offscreen=0, clipped=0, finite=0;
                bool all_same = written > 0;
                for (uint32_t i = 0; i < written; i++) {
                    const float x=pos[i*4+0], y=pos[i*4+1], z=pos[i*4+2], w=pos[i*4+3];
                    if (!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z)||!std::isfinite(w)) { nan++; continue; }
                    finite++;
                    minx=std::min(minx,x); maxx=std::max(maxx,x); miny=std::min(miny,y); maxy=std::max(maxy,y);
                    minz=std::min(minz,z); maxz=std::max(maxz,z); minw=std::min(minw,w); maxw=std::max(maxw,w);
                    if (w <= 0.0f) wle0++;
                    if (std::fabs(x) > std::fabs(w) || std::fabs(y) > std::fabs(w)) offscreen++;
                    // A vertex is on-screen only if it is in front of the camera (w>0) and inside the
                    // clip cube (|x|,|y| <= w). Everything else is clipped and cannot rasterize.
                    if (!(w > 0.0f && std::fabs(x) <= w && std::fabs(y) <= w)) clipped++;
                    if (i && (x!=pos[0]||y!=pos[1]||z!=pos[2]||w!=pos[3])) all_same = false;
                }
                const uint32_t onscreen = finite - clipped;
                // Classification is descriptive (read it WITH the funnel, which owns the vanishes verdict):
                // a large quad whose verts sit just outside the cube still rasterizes via clipping, so
                // "all verts outside" is not the same as "renders nothing" — the bbox tells them apart.
                const char* tag =
                    finite == 0             ? "ALL-NAN/INF(numeric defect)" :
                    all_same                ? "DEGENERATE(all verts collapse to one clip point)" :
                    wle0 == finite          ? "ALL-BEHIND-CAMERA(w<=0: transform/w defect)" :
                    clipped == finite       ? "ALL-VERTS-OUTSIDE-CLIP-CUBE(see bbox: off-screen shift or oversized quad)" :
                    onscreen * 20u < finite ? "MOSTLY-OUTSIDE(<5% verts on-screen; see bbox)" :
                                              "on-screen(geometry spread across clip space)";
                // Shader I/O tap mode (PROSPER_SHADER_TAP): the captured "positions" are actually the tapped
                // intermediate VGPR (dst..dst+3) at that PC, so print the raw hex too (values are often
                // integers/bitfields, not clip floats) and skip the meaningless clip classification.
                const bool is_tap = getenv("PROSPER_SHADER_TAP") != nullptr;
                if (is_tap)
                    fprintf(stderr, "[geom-probe] draw=%ld SHADER-TAP: values below are the tapped VGPR "
                                    "(dst+3) at that PC, not clip positions (bbox/tags meaningless)\n", geom_target);
                else
                    fprintf(stderr, "[geom-probe] draw=%ld verts-written=%u finite=%u on-screen=%u clipped=%u "
                                    "(offscreen=%u w<=0=%u nan/inf=%u)\n"
                                    "[geom-probe]   clip-bbox x[%g,%g] y[%g,%g] z[%g,%g] w[%g,%g] -> %s\n",
                            geom_target, written, finite, onscreen, clipped, offscreen, wle0, nan,
                            minx,maxx, miny,maxy, minz,maxz, minw,maxw, tag);
                for (uint32_t i = 0; i < written && i < (is_tap ? 8u : 4u); i++) {
                    if (is_tap) {
                        uint32_t h[4]; std::memcpy(h, &pos[i*4], 16);
                        fprintf(stderr, "[geom-probe]   v%u = float(%g, %g, %g, %g) hex(%08x %08x %08x %08x)\n",
                                i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3], h[0], h[1], h[2], h[3]);
                    } else {
                        fprintf(stderr, "[geom-probe]   v%u = (%g, %g, %g, %g)\n",
                                i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3]);
                    }
                }
                // Geometry-health metrics (#1257): the transform-feedback buffer already holds every
                // post-transform vertex in primitive-assembly order, so consecutive triples ARE the
                // rasterized triangles (TF decomposes strips/fans into triangles). Report the tells that
                // localize a geometry/vertex-fetch bug — the exact signals that cracked GTA #1163's
                // vertex-count inflation: how many DISTINCT positions the draw really has (a shared VB pool
                // read past its real range collapses most verts onto a few points), what fraction of
                // triangles are DEGENERATE (zero-area stitching / collapsed fetch), and how many triangles
                // are exact DUPLICATES (the same triangle rasterized N times = pure overdraw). Overdraw
                // itself is the funnel's job (occlusion samples > covered pixels): read this WITH
                // PROSPER_DRAW_STATS. Skipped in tap mode (pos holds VGPR values, not positions).
                if (!is_tap && written >= 3) {
                    std::vector<std::array<float, 4>> verts;   // finite positions, for unique/multiplicity
                    verts.reserve(written);
                    for (uint32_t i = 0; i < written; i++) {
                        const float x = pos[i*4+0], y = pos[i*4+1], z = pos[i*4+2], w = pos[i*4+3];
                        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(w))
                            verts.push_back({x, y, z, w});
                    }
                    std::sort(verts.begin(), verts.end());
                    uint32_t unique_pos = 0, max_mult = 0, run = 0;
                    for (size_t i = 0; i < verts.size(); i++) {
                        if (i == 0 || verts[i] != verts[i-1]) { unique_pos++; run = 1; } else run++;
                        max_mult = std::max(max_mult, run);
                    }
                    // Per-triangle (over FINITE triangles only): degenerate (near-zero NDC area) + exact
                    // duplicates (canonical key). Non-finite triangles are a numeric defect already reported
                    // by the probe's nan/inf count, and are skipped here so no NaN enters std::sort (a NaN
                    // breaks the strict-weak-ordering -> UB); their vertices are also excluded from `verts`.
                    const uint32_t ntri = written / 3u;
                    uint32_t degenerate = 0, finite_tri = 0;
                    std::vector<std::array<float, 12>> tris;
                    tris.reserve(ntri);
                    for (uint32_t t = 0; t < ntri; t++) {
                        const float* p = &pos[t*3*4];
                        bool fin = true;
                        for (int k = 0; k < 12; k++) if (!std::isfinite(p[k])) { fin = false; break; }
                        if (!fin) continue;
                        finite_tri++;
                        auto ndc = [&](int k, int c) {   // p[k*4+3] is finite here
                            float v = p[k*4+c], w = p[k*4+3];
                            return w != 0.0f ? v / w : v;
                        };
                        const float ax = ndc(0,0), ay = ndc(0,1), bx = ndc(1,0), by = ndc(1,1),
                                    cx = ndc(2,0), cy = ndc(2,1);
                        const float area2 = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay);
                        if (std::fabs(area2) < 1e-10f) degenerate++;
                        // Canonical key: the triangle's three (x,y,z,w) vertices sorted, so a duplicate
                        // triangle in any winding/rotation collides.
                        std::array<std::array<float,4>,3> v3 = {{
                            {p[0],p[1],p[2],p[3]}, {p[4],p[5],p[6],p[7]}, {p[8],p[9],p[10],p[11]} }};
                        std::sort(v3.begin(), v3.end());
                        tris.push_back({v3[0][0],v3[0][1],v3[0][2],v3[0][3],
                                        v3[1][0],v3[1][1],v3[1][2],v3[1][3],
                                        v3[2][0],v3[2][1],v3[2][2],v3[2][3]});
                    }
                    std::sort(tris.begin(), tris.end());
                    uint32_t dup_tri = 0;
                    for (size_t i = 1; i < tris.size(); i++) if (tris[i] == tris[i-1]) dup_tri++;
                    const uint32_t real_tri = finite_tri - degenerate;
                    const char* health =
                        finite && unique_pos <= 2                      ? "COLLAPSED(<=2 distinct positions - fetch/transform returns a constant)" :
                        finite_tri && degenerate * 5u >= finite_tri*4u ? "DEGENERATE-HEAVY(>=80% zero-area - strip-stitching read as list, or wrong count/stride)" :
                        dup_tri && dup_tri * 4u >= real_tri            ? "DUPLICATE-TRIANGLES(exact repeats = pure overdraw - likely over-count/wrong vertex range)" :
                                                                         "ok(check PROSPER_DRAW_STATS for overdraw: samples>pixels)";
                    fprintf(stderr, "[geom-health] draw=%ld verts=%u unique-pos=%u (max-mult=%u) "
                                    "triangles=%u degenerate=%u(%.0f%%) real=%u duplicate-tri=%u -> %s\n",
                            geom_target, (unsigned)verts.size(), unique_pos, max_mult, finite_tri, degenerate,
                            finite_tri ? 100.0 * degenerate / finite_tri : 0.0, real_tri, dup_tri, health);
                }
                // PROSPER_GEOM_PROBE_DUMP=path (gated, off by default): write EVERY post-transform vertex
                // (x,y,z,w in primitive-assembly order — for a triangle list, consecutive triples are the
                // rasterized triangles) as CSV, so per-triangle overlap/degeneracy can be analyzed offline
                // (e.g. GTA #1163's stencil over-count from self-overlapping mask triangles).
                if (const char* dp = getenv("PROSPER_GEOM_PROBE_DUMP")) {
                    if (FILE* f = fopen(dp, "w")) {
                        fprintf(f, "i,x,y,z,w\n");
                        for (uint32_t i = 0; i < written; i++)
                            fprintf(f, "%u,%.7g,%.7g,%.7g,%.7g\n",
                                    i, pos[i*4+0], pos[i*4+1], pos[i*4+2], pos[i*4+3]);
                        fclose(f);
                        fprintf(stderr, "[geom-probe]   wrote %u verts -> %s\n", written, dp);
                    }
                }
                vkUnmapMemory(dev, geom_mem);
            } else if (!written) {
                fprintf(stderr, "[geom-probe] draw=%ld: transform feedback wrote 0 vertices "
                                "(draw produced no primitives)\n", geom_target);
            }
        }
        vkDestroyBuffer(dev, geom_buf, nullptr); vkFreeMemory(dev, geom_mem, nullptr);
        vkDestroyBuffer(dev, geom_counter, nullptr); vkFreeMemory(dev, geom_counter_mem, nullptr);
    }

    if (readback_requested && batch_completed) {
        void* mp = nullptr; vkMapMemory(dev, bmem, 0, readback_bytes, 0, &mp);
        const auto* readback = static_cast<const uint8_t*>(mp);
        // A range assignment constructs directly from the mapped pixels. resize()+memcpy first zeroed the
        // entire 8.3 MiB 1080p vector even though every byte was immediately overwritten.
        out.assign(readback, readback + static_cast<size_t>(bytes));
        if (use_color1)
            out_rgba1->assign(readback + static_cast<size_t>(bytes),
                              readback + static_cast<size_t>(readback_bytes));
        vkUnmapMemory(dev, bmem);
        color_target_stats.readbacks = persistent_color ? 1 : 0;
    }
    const auto timing_readback_done = timing_enabled ? TimingClock::now() : TimingClock::time_point{};

    // PROSPER_DRAW_ISO + PROSPER_ISO_AT="x,y": per-draw kill isolation (generalizes the #240 title harness
    // to any submit / any target pixel). On the FIRST submit whose rendered pixel at (x,y) is lit
    // (non-background), re-render THIS exact submit once per killed-draw index and report which draw lights
    // that pixel — the kill index that turns (x,y) dark is the culprit. Reuses the built pipelines/
    // descriptors; a fresh clear each pass. Env-gated, no default behavior. Used to locate a stray primitive
    // such as the #298 menu focus-ring sliver. Dumps iso_kill_<k>.bmp to PROSPER_FRAME_DIR.
    if (FMT == VK_FORMAT_R8G8B8A8_UNORM &&
        getenv("PROSPER_DRAW_ISO") && getenv("PROSPER_ISO_AT")) {
        static bool iso_done = false;
        int tx = -1, ty = -1; sscanf(getenv("PROSPER_ISO_AT"), "%d,%d", &tx, &ty);
        // Optional PROSPER_ISO_RGB="r,g,b" (+ PROSPER_ISO_TOL, default 45): the target submit is the first
        // whose pixel at (x,y) matches that color within tol — robust against an earlier full-screen submit
        // (e.g. the intro cutscene) that merely lights the pixel a different color. Unset -> any non-background.
        int wr = -1, wg = 0, wb = 0, tol = getenv("PROSPER_ISO_TOL") ? atoi(getenv("PROSPER_ISO_TOL")) : 45;
        if (getenv("PROSPER_ISO_RGB")) sscanf(getenv("PROSPER_ISO_RGB"), "%d,%d,%d", &wr, &wg, &wb);
        auto lit_at = [&](const std::vector<uint8_t>& buf) -> bool {
            if (tx < 0 || ty < 0 || (uint32_t)tx >= W || (uint32_t)ty >= H) return false;
            const uint8_t* p = &buf[((size_t)ty * W + tx) * 4];
            if (wr >= 0) return abs((int)p[0]-wr) <= tol && abs((int)p[1]-wg) <= tol && abs((int)p[2]-wb) <= tol;
            return p[0] > 40 || p[1] > 40 || p[2] > 40;
        };
        // Optional second reference pixel: reported alongside the target so we can tell whether the culprit
        // draw ALSO paints a legit element (e.g. the active focus ring) or only the stray pixel.
        int rx = -1, ry = -1; if (getenv("PROSPER_ISO_AT2")) sscanf(getenv("PROSPER_ISO_AT2"), "%d,%d", &rx, &ry);
        if (!iso_done && lit_at(out)) {
            iso_done = true;
            const char* fd = getenv("PROSPER_FRAME_DIR"); std::string dir = fd ? fd : ".";
            fprintf(stderr, "[iso] submit lights (%d,%d): %zu draws; re-rendering per killed draw\n", tx, ty, dv.size());
            // Characterize every draw in the target submit (blend/write-mask/viewport/textures/vertex count).
            for (size_t di = 0; di < dv.size(); di++) {
                const prosper::gpu::ResolvedPipelineState* ps = draws[di].ps; DV& v = dv[di];
                fprintf(stderr, "[iso]  draw#%zu %s cnt=%u", di, v.ok ? "OK" : "SKIP", v.icount ? v.icount : v.vcount);
                if (ps) fprintf(stderr, " blend=%d src=%u dst=%u cwm=0x%x vp_y=%.0f vp_h=%.0f depth=%d/%d",
                                (int)ps->blend_enable, ps->src_color_blend_factor, ps->dst_color_blend_factor,
                                ps->color_write_mask, ps->viewport_y, ps->viewport_h,
                                (int)ps->depth_test_enable, (int)ps->depth_write_enable);
                int nt = 0; for (const auto& r : draws[di].R) if (r.is_texture()) { fprintf(stderr, " tex%d=%ux%u", nt, r.tw, r.th); nt++; }
                fprintf(stderr, "\n");
            }
            VkFenceCreateInfo iso_fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence iso_fence = VK_NULL_HANDLE;
            vkCreateFence(dev, &iso_fence_info, nullptr, &iso_fence);
            VkBufferImageCopy cp2{}; cp2.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp2.imageExtent = {W, H, 1};
            for (int kk = -1; kk < (int)dv.size(); kk++) {
                VkCommandBuffer c2; VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
                vkAllocateCommandBuffers(dev, &ai, &c2); vkBeginCommandBuffer(c2, &cbbi);
                vkCmdBeginRenderPass(c2, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
                for (size_t di = 0; di < dv.size(); di++) { auto& v = dv[di]; if (!v.ok) continue; if ((int)di == kk) continue;
                    vkCmdBindPipeline(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
                    if (v.use_desc) vkCmdBindDescriptorSets(c2, VK_PIPELINE_BIND_POINT_GRAPHICS, v.layout, 0, v.n_sets, v.dsets.data(), 0, nullptr);
                    if (v.icount) { vkCmdBindIndexBuffer(c2, v.ibuf, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(c2, v.icount, v.instance_count, 0, v.vertex_offset, 0); }
                    else vkCmdDraw(c2, v.vcount, v.instance_count, static_cast<uint32_t>(v.vertex_offset), 0);
                }
                vkCmdEndRenderPass(c2);
                vkCmdCopyImageToBuffer(c2, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp2);
                vkEndCommandBuffer(c2); vkResetFences(dev, 1, &iso_fence);
                VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si2.commandBufferCount = 1; si2.pCommandBuffers = &c2;
                render_locked_queue_submit(queue, 1, &si2, iso_fence); vkWaitForFences(dev, 1, &iso_fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
                std::vector<uint8_t> px(bytes); void* m2 = nullptr; vkMapMemory(dev, bmem, 0, bytes, 0, &m2);
                for (VkDeviceSize i = 0; i < bytes; i++) px[i] = ((const uint8_t*)m2)[i]; vkUnmapMemory(dev, bmem);
                const uint8_t* tp = &px[((size_t)ty * W + tx) * 4];
                bool lit = tp[0] > 40 || tp[1] > 40 || tp[2] > 40;
                // Total gold pixels (ring color) in the frame: if killing a draw drops this only by the
                // sliver's ~size, that draw paints ONLY the sliver; a large drop means it also paints the ring.
                size_t gold_n = 0;
                for (size_t q = 0; q < (size_t)W * H; q++) { const uint8_t* g = &px[q * 4];
                    if (g[0] > 140 && g[1] > 100 && g[2] < 90 && (int)g[0] - (int)g[2] > 60) gold_n++; }
                char ref[96] = "";
                if (rx >= 0 && ry >= 0 && (uint32_t)rx < W && (uint32_t)ry < H) {
                    const uint8_t* rp = &px[((size_t)ry * W + rx) * 4];
                    snprintf(ref, sizeof ref, "  ref(%d,%d)=%u,%u,%u %s", rx, ry, rp[0], rp[1], rp[2],
                             (rp[0] > 40 || rp[1] > 40 || rp[2] > 40) ? "lit" : "DARK");
                }
                fprintf(stderr, "[iso]  kill=%d -> (%d,%d) rgb=%u,%u,%u %s  gold_px=%zu%s\n", kk, tx, ty, tp[0], tp[1], tp[2],
                        lit ? "LIT" : "dark <<< THIS DRAW paints the pixel", gold_n, ref);
                char fn[512]; snprintf(fn, sizeof fn, "%s/iso_kill_%d.bmp", dir.c_str(), kk); dump_bmp(fn, px, W, H);
                vkFreeCommandBuffers(dev, pool, 1, &c2);
            }
            if (iso_fence) vkDestroyFence(dev, iso_fence, nullptr);
            fprintf(stderr, "[iso] done: the kill index marked 'dark' is the draw painting (%d,%d)\n", tx, ty);
        }
    }

    color_target_stats.cached_bytes = persistent_color_target_bytes();
    color_target_stats.cached_entries = persistent_color_target_cache().size();
    resource_reuse_stats.persistent_pipeline_layout_entries =
        persistent_pipeline_layouts.size();
    const bool transient_color = cached_color == nullptr;
    const bool transient_ds = use_ds && cached_ds == nullptr;
    const RenderVkCtx* ctx_ptr = &ctx;
    active_submission.add_cleanup(
        [dev, pool, dv = std::move(dv), shared_descriptor_pool,
         shared_pipeline_layouts = std::move(shared_pipeline_layouts),
         shared_descriptor_set_layouts = std::move(shared_descriptor_set_layouts),
         shared_texture_bindings = std::move(shared_texture_bindings),
         shared_buffers = std::move(shared_buffers),
         shared_buffer_arenas = std::move(shared_buffer_arenas),
         texture_uploads = std::move(texture_uploads), seedbuf, seedmem, seedbuf1, seedmem1,
         rb, bmem, fb, rp, transient_color, view, img, imem, use_color1, view1, img1,
         imem1, transient_ds, dview, dimg, dmem, ctx_ptr,
         color_target_generation]() mutable {
            vkDestroyCommandPool(dev, pool, nullptr);
            for (auto& v : dv) {
                if (v.pipe && !v.pipeline_cached) vkDestroyPipeline(dev, v.pipe, nullptr);
                if (v.ibuf) vkDestroyBuffer(dev, v.ibuf, nullptr);
                if (v.ibmem) release_transient_render_memory(dev, v.ibmem);
                if (v.vs) vkDestroyShaderModule(dev, v.vs, nullptr);
                if (v.gs) vkDestroyShaderModule(dev, v.gs, nullptr);
                if (v.fs) vkDestroyShaderModule(dev, v.fs, nullptr);
            }
            if (shared_descriptor_pool)
                vkDestroyDescriptorPool(dev, shared_descriptor_pool, nullptr);
            for (const auto& [key, layout] : shared_pipeline_layouts)
                if (layout.handle && !layout.persistent)
                    vkDestroyPipelineLayout(dev, layout.handle, nullptr);
            for (const auto& [key, layout] : shared_descriptor_set_layouts)
                if (layout) vkDestroyDescriptorSetLayout(dev, layout, nullptr);
            for (const SharedTextureBinding& binding : shared_texture_bindings) {
                if (!binding.persistent) {
                    if (binding.sampler) vkDestroySampler(dev, binding.sampler, nullptr);
                    if (binding.view) vkDestroyImageView(dev, binding.view, nullptr);
                }
            }
            for (const SharedBufferUpload& upload : shared_buffers) {
                if (upload.arena) {
                    continue;
                } else if (upload.pooled) {
                    release_render_host_buffer(
                        dev, {upload.buffer, upload.memory, upload.mapped,
                              upload.bytes, upload.allocation_bytes});
                } else {
                    if (upload.buffer) vkDestroyBuffer(dev, upload.buffer, nullptr);
                    if (upload.memory) release_transient_render_memory(dev, upload.memory);
                }
            }
            for (SharedBufferArena& arena : shared_buffer_arenas)
                release_render_host_buffer(dev, arena.buffer);
            for (auto& upload : texture_uploads) {
                if (upload.image && !upload.persistent_hit && !upload.borrowed_target &&
                    !upload.borrowed_ds)
                    vkDestroyImage(dev, upload.image, nullptr);
                if (upload.memory) {
                    if (upload.direct_memory) vkFreeMemory(dev, upload.memory, nullptr);
                    else release_transient_render_memory(dev, upload.memory);
                }
                if (upload.staging) vkDestroyBuffer(dev, upload.staging, nullptr);
                if (upload.staging_memory)
                    release_transient_render_memory(dev, upload.staging_memory);
            }
            if (seedbuf) vkDestroyBuffer(dev, seedbuf, nullptr);
            if (seedmem) release_transient_render_memory(dev, seedmem);
            if (seedbuf1) vkDestroyBuffer(dev, seedbuf1, nullptr);
            if (seedmem1) release_transient_render_memory(dev, seedmem1);
            if (rb) vkDestroyBuffer(dev, rb, nullptr);
            if (bmem) release_transient_render_memory(dev, bmem);
            vkDestroyFramebuffer(dev, fb, nullptr);
            vkDestroyRenderPass(dev, rp, nullptr);
            if (transient_color) {
                vkDestroyImageView(dev, view, nullptr);
                vkDestroyImage(dev, img, nullptr);
                release_transient_render_memory(dev, imem);
            }
            if (use_color1) {
                vkDestroyImageView(dev, view1, nullptr);
                vkDestroyImage(dev, img1, nullptr);
                release_transient_render_memory(dev, imem1);
            }
            if (transient_ds) {
                vkDestroyImageView(dev, dview, nullptr);
                vkDestroyImage(dev, dimg, nullptr);
                release_transient_render_memory(dev, dmem);
            }
            while ((persistent_color_target_cache().size() > persistent_color_target_count_limit() ||
                    persistent_color_target_bytes() > persistent_color_target_limit()) &&
                   evict_persistent_color_target(*ctx_ptr, color_target_generation)) {}
        });
    if (flush_now) active_submission.complete();
    if (timing_enabled) {
        const auto timing_done = TimingClock::now();
        auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        BackendRenderTimingStats& call_timing = backend_render_timing_stats_storage();
        call_timing.calls = 1;
        call_timing.draws = draws.size();
        call_timing.command_buffers = batch_result.command_buffers;
        call_timing.queue_submits = batch_result.queue_submits;
        call_timing.fence_waits = batch_result.fence_waits;
        call_timing.target_ms = ms(timing_start, timing_target_ready);
        call_timing.draw_setup_ms = ms(timing_target_ready, timing_draws_ready);
        call_timing.record_upload_ms = ms(timing_draws_ready, timing_recorded);
        call_timing.gpu_wait_ms = ms(timing_recorded, timing_gpu_done);
        call_timing.readback_ms = ms(timing_gpu_done, timing_readback_done);
        call_timing.cleanup_ms = ms(timing_readback_done, timing_done);
        call_timing.setup_shader_ms = setup_shader_ms;
        call_timing.setup_fixed_ms = setup_fixed_ms;
        call_timing.setup_resources_ms = setup_resources_ms;
        call_timing.setup_pipeline_ms = setup_pipeline_ms;
        struct TimingTotals {
            uint64_t calls = 0, draws = 0;
            uint64_t command_buffers = 0, queue_submits = 0, fence_waits = 0;
            uint64_t texture_references = 0, texture_uploads = 0, texture_upload_bytes = 0;
            uint64_t persistent_hits = 0, persistent_misses = 0, persistent_cached_bytes = 0;
            uint64_t texture_binding_references = 0, unique_texture_bindings = 0;
            uint64_t buffer_references = 0, unique_buffers = 0;
            uint64_t descriptor_layout_references = 0, unique_descriptor_layouts = 0;
            uint64_t pipeline_layout_references = 0, unique_pipeline_layouts = 0;
            uint64_t pipeline_references = 0, pipeline_hits = 0, pipeline_misses = 0;
            uint64_t pipeline_bypasses = 0, pipeline_entries = 0, pipeline_evictions = 0;
            double target = 0, draw_setup = 0, record = 0, gpu_wait = 0, readback = 0, cleanup = 0;
            double setup_shader = 0, setup_fixed = 0, setup_resources = 0, setup_pipeline = 0;
        };
        static TimingTotals totals;
        static TimingTotals window;
        auto accumulate = [&](TimingTotals& timing) {
            timing.calls++;
            timing.draws += draws.size();
            timing.command_buffers += call_timing.command_buffers;
            timing.queue_submits += call_timing.queue_submits;
            timing.fence_waits += call_timing.fence_waits;
            timing.texture_references += texture_stats.references;
            timing.texture_uploads += texture_stats.unique_uploads;
            timing.texture_upload_bytes += texture_stats.upload_bytes;
            timing.persistent_hits += texture_stats.persistent_hits;
            timing.persistent_misses += texture_stats.persistent_misses;
            timing.persistent_cached_bytes = texture_stats.persistent_cached_bytes;
            timing.texture_binding_references += resource_reuse_stats.texture_binding_references;
            timing.unique_texture_bindings += resource_reuse_stats.unique_texture_bindings;
            timing.buffer_references += resource_reuse_stats.buffer_references;
            timing.unique_buffers += resource_reuse_stats.unique_buffers;
            timing.descriptor_layout_references +=
                resource_reuse_stats.descriptor_set_layout_references;
            timing.unique_descriptor_layouts +=
                resource_reuse_stats.unique_descriptor_set_layouts;
            timing.pipeline_layout_references += resource_reuse_stats.pipeline_layout_references;
            timing.unique_pipeline_layouts += resource_reuse_stats.unique_pipeline_layouts;
            timing.pipeline_references += pipeline_stats.references;
            timing.pipeline_hits += pipeline_stats.hits;
            timing.pipeline_misses += pipeline_stats.misses;
            timing.pipeline_bypasses += pipeline_stats.bypasses;
            timing.pipeline_entries = pipeline_stats.entries;
            timing.pipeline_evictions += pipeline_stats.evictions;
            timing.target += call_timing.target_ms;
            timing.draw_setup += call_timing.draw_setup_ms;
            timing.record += call_timing.record_upload_ms;
            timing.gpu_wait += call_timing.gpu_wait_ms;
            timing.readback += call_timing.readback_ms;
            timing.cleanup += call_timing.cleanup_ms;
            timing.setup_shader += call_timing.setup_shader_ms;
            timing.setup_fixed += call_timing.setup_fixed_ms;
            timing.setup_resources += call_timing.setup_resources_ms;
            timing.setup_pipeline += call_timing.setup_pipeline_ms;
        };
        accumulate(totals);
        accumulate(window);
        static const bool print_backend_timing_windows =
            getenv("PROSPER_BACKEND_TIMING_WINDOWS") != nullptr;
        if (print_backend_timing_windows && totals.calls % 25 == 0) {
            const double n = static_cast<double>(totals.calls);
            const double total = totals.target + totals.draw_setup + totals.record +
                                 totals.gpu_wait + totals.readback + totals.cleanup;
            fprintf(stderr,
                    "[render-timing] backend calls=%llu draws=%llu avg_ms: total=%.2f target=%.2f "
                    "draw_setup=%.2f record_upload=%.2f gpu_wait=%.2f readback=%.2f cleanup=%.2f\n",
                    (unsigned long long)totals.calls, (unsigned long long)totals.draws, total / n,
                    totals.target / n, totals.draw_setup / n, totals.record / n,
                    totals.gpu_wait / n, totals.readback / n, totals.cleanup / n);
            fprintf(stderr,
                    "[render-timing] backend synchronization command_buffers=%llu queue_submits=%llu "
                    "fence_waits=%llu\n",
                    (unsigned long long)totals.command_buffers,
                    (unsigned long long)totals.queue_submits,
                    (unsigned long long)totals.fence_waits);
            fprintf(stderr,
                    "[render-timing] draw_setup avg_ms: shaders=%.2f fixed=%.2f resources=%.2f pipeline=%.2f\n",
                    totals.setup_shader / n, totals.setup_fixed / n,
                    totals.setup_resources / n, totals.setup_pipeline / n);
            fprintf(stderr,
                    "[render-timing] backend textures refs=%llu uploads=%llu %.1f MiB "
                    "persistent=%llu/%llu cache=%.1f MiB\n",
                    (unsigned long long)totals.texture_references,
                    (unsigned long long)totals.texture_uploads,
                    totals.texture_upload_bytes / (1024.0 * 1024.0),
                    (unsigned long long)totals.persistent_hits,
                    (unsigned long long)totals.persistent_misses,
                    totals.persistent_cached_bytes / (1024.0 * 1024.0));
            fprintf(stderr,
                    "[render-timing] backend pipelines refs=%llu hits=%llu misses=%llu bypass=%llu "
                    "entries=%llu evictions=%llu\n",
                    (unsigned long long)totals.pipeline_references,
                    (unsigned long long)totals.pipeline_hits,
                    (unsigned long long)totals.pipeline_misses,
                    (unsigned long long)totals.pipeline_bypasses,
                    (unsigned long long)totals.pipeline_entries,
                    (unsigned long long)totals.pipeline_evictions);
            const RenderMemoryPoolStats pool = render_memory_pool_stats();
            fprintf(stderr,
                    "[render-timing] memory_pool hits=%llu misses=%llu cached=%zu %.1f MiB "
                    "discarded=%llu\n",
                    (unsigned long long)pool.hits, (unsigned long long)pool.misses,
                    pool.cached_allocations,
                    static_cast<double>(pool.cached_bytes) / (1024.0 * 1024.0),
                    (unsigned long long)pool.discarded);
            const RenderHostBufferPoolStats buffer_pool = render_host_buffer_pool_stats();
            fprintf(stderr,
                    "[render-timing] backend_buffer_pool hits=%llu misses=%llu cached=%zu %.1f MiB "
                    "evictions=%llu\n",
                    (unsigned long long)buffer_pool.hits,
                    (unsigned long long)buffer_pool.misses,
                    buffer_pool.cached_buffers,
                    static_cast<double>(buffer_pool.cached_bytes) / (1024.0 * 1024.0),
                    (unsigned long long)buffer_pool.evictions);
            const double wn = static_cast<double>(window.calls);
            const double window_total = window.target + window.draw_setup + window.record +
                                        window.gpu_wait + window.readback + window.cleanup;
            fprintf(stderr,
                    "[render-window] backend calls=%llu draws=%.1f avg_ms: total=%.2f target=%.2f "
                    "draw_setup=%.2f record_upload=%.2f gpu_wait=%.2f readback=%.2f cleanup=%.2f\n",
                    (unsigned long long)window.calls, window.draws / wn, window_total / wn,
                    window.target / wn, window.draw_setup / wn, window.record / wn,
                    window.gpu_wait / wn, window.readback / wn, window.cleanup / wn);
            fprintf(stderr,
                    "[render-window] backend synchronization command_buffers=%.1f queue_submits=%.1f "
                    "fence_waits=%.1f\n",
                    window.command_buffers / wn, window.queue_submits / wn,
                    window.fence_waits / wn);
            fprintf(stderr,
                    "[render-window] draw_setup avg_ms: shaders=%.2f fixed=%.2f resources=%.2f pipeline=%.2f\n",
                    window.setup_shader / wn, window.setup_fixed / wn,
                    window.setup_resources / wn, window.setup_pipeline / wn);
            fprintf(stderr,
                    "[render-window] backend textures refs=%.1f uploads=%.1f %.1f MiB "
                    "persistent=%.1f/%.1f cache=%.1f MiB\n",
                    window.texture_references / wn, window.texture_uploads / wn,
                    window.texture_upload_bytes / (wn * 1024.0 * 1024.0),
                    window.persistent_hits / wn, window.persistent_misses / wn,
                    window.persistent_cached_bytes / (1024.0 * 1024.0));
            fprintf(stderr,
                    "[render-window] backend resources texture_bindings=%.1f/%.1f "
                    "buffers=%.1f/%.1f descriptor_layouts=%.1f/%.1f "
                    "pipeline_layouts=%.1f/%.1f\n",
                    window.texture_binding_references / wn,
                    window.unique_texture_bindings / wn,
                    window.buffer_references / wn, window.unique_buffers / wn,
                    window.descriptor_layout_references / wn,
                    window.unique_descriptor_layouts / wn,
                    window.pipeline_layout_references / wn,
                    window.unique_pipeline_layouts / wn);
            fprintf(stderr,
                    "[render-window] backend pipelines refs=%.1f hits=%.1f misses=%.1f bypass=%.1f "
                    "entries=%llu evictions=%.1f\n",
                    window.pipeline_references / wn, window.pipeline_hits / wn,
                    window.pipeline_misses / wn, window.pipeline_bypasses / wn,
                    (unsigned long long)window.pipeline_entries, window.pipeline_evictions / wn);
            window = {};
        }
    }
    // NB: dev/instance are the persistent RenderVkCtx — do NOT destroy them here (reused across calls).
    return out;
}

// Single-draw entry — a thin wrapper over render_draws_rgba, preserving the exact signature/behavior the
// recompiled-shader render tests rely on. Builds one draw's set-tagged resources (`gres`, or the legacy
// cbuf/vbuf@bindings 2/3 mirrored into both sets + optional `tex` in set 1).
inline std::vector<uint8_t> render_triangle_rgba(const std::vector<uint32_t>& vert,
                                                 const std::vector<uint32_t>& frag,
                                                 uint32_t W, uint32_t H,
                                                 const prosper::gpu::ResolvedPipelineState* ps = nullptr,
                                                 const std::vector<uint32_t>* vbuf = nullptr,
                                                 const std::vector<uint32_t>* cbuf = nullptr,
                                                 const TexDesc* tex = nullptr,
                                                 const std::vector<FrameResource>* gres = nullptr,
                                                 uint32_t vcount = 3) {
    BackendDraw d; d.vs = vert; d.fs = frag; d.ps = ps; d.vcount = vcount;
    if (gres && !gres->empty()) { d.R = *gres; }
    else {
        for (uint32_t s2 = 0; s2 < 2; s2++) {
            FrameResource b2; b2.binding = 2; b2.set = s2; if (cbuf) b2.dwords = *cbuf; d.R.push_back(std::move(b2));
            FrameResource b3; b3.binding = 3; b3.set = s2; if (vbuf) b3.dwords = *vbuf; d.R.push_back(std::move(b3));
        }
        if (tex) { FrameResource t; t.binding = tex->binding; t.set = 1; t.tex_rgba = tex->rgba; t.tw = tex->w; t.th = tex->h; t.max_aniso_ratio = tex->max_aniso_ratio; d.R.push_back(std::move(t)); }
    }
    return render_draws_rgba({std::move(d)}, W, H);
}

} // namespace prosper::test

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
#include "host/guest_write_watch.hpp"
#include "render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp

#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);
#ifdef _WIN32
extern "C" int prosper_try_commit_dmem(uint64_t addr, uint64_t len, int write);
#endif
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
struct RttSurf {
    std::shared_ptr<const std::vector<uint8_t>> rgba;
    uint32_t w = 0, h = 0;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    bool gpu_valid = false;
};

using RttCache = std::unordered_map<uint64_t, RttSurf>;

struct PendingGuestGpuWrites {
    std::mutex mutex;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    bool overflowed = false;
};

PendingGuestGpuWrites& pending_guest_gpu_writes() {
    static PendingGuestGpuWrites pending;
    return pending;
}

void queue_guest_gpu_write(uint64_t addr, uint64_t size) {
    auto& pending = pending_guest_gpu_writes();
    std::lock_guard<std::mutex> lock(pending.mutex);
    constexpr size_t kMaxPendingRanges = 65536;
    if (pending.ranges.size() < kMaxPendingRanges)
        pending.ranges.emplace_back(addr, size);
    else
        pending.overflowed = true;
}

void invalidate_cpu_rtt_guest_write(RttCache& cache, uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    const uint64_t end = size > UINT64_MAX - addr ? UINT64_MAX : addr + size;
    for (auto it = cache.begin(); it != cache.end();) {
        const RttSurf& surface = it->second;
        const uint64_t bpp = prosper::test::backend_color_bytes_per_pixel(surface.format);
        const uint64_t pixels = static_cast<uint64_t>(surface.w) * surface.h;
        const uint64_t bytes = pixels > UINT64_MAX / bpp ? UINT64_MAX : pixels * bpp;
        const uint64_t target_end = bytes > UINT64_MAX - it->first
            ? UINT64_MAX : it->first + bytes;
        if (addr < target_end && it->first < end) it = cache.erase(it);
        else ++it;
    }
}

void drain_guest_gpu_writes(RttCache& cache, bool invalidate_ds) {
    auto& pending = pending_guest_gpu_writes();
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    bool overflowed = false;
    {
        std::lock_guard<std::mutex> lock(pending.mutex);
        ranges.swap(pending.ranges);
        overflowed = pending.overflowed;
        pending.overflowed = false;
    }
    if (overflowed) {
        for (auto& [key, target] : prosper::test::persistent_color_target_cache()) {
            (void)key;
            target.valid = false;
        }
        if (invalidate_ds)
            for (auto& [key, image] : prosper::test::persistent_ds_cache()) {
                (void)key;
                image.depth_valid = false;
                image.stencil_valid = false;
            }
        cache.clear();
        return;
    }
    for (const auto& [addr, size] : ranges) {
        if (invalidate_ds)
            prosper::test::invalidate_persistent_ds_guest_write(addr, size);
        prosper::test::invalidate_persistent_color_target_guest_write(addr, size);
        invalidate_cpu_rtt_guest_write(cache, addr, size);
    }
}

prosper::gpu::GpuCaptureColorFormat capture_color_format(VkFormat format) {
    return prosper::test::backend_color_format(format) == VK_FORMAT_R16G16B16A16_SFLOAT
        ? prosper::gpu::GpuCaptureColorFormat::Rgba16Float
        : prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm;
}

VkFormat replay_color_format(prosper::gpu::GpuCaptureColorFormat format) {
    return format == prosper::gpu::GpuCaptureColorFormat::Rgba16Float
        ? VK_FORMAT_R16G16B16A16_SFLOAT
        : VK_FORMAT_R8G8B8A8_UNORM;
}

std::vector<uint8_t> inspection_rgba8(const std::vector<uint8_t>& pixels,
                                      uint32_t width, uint32_t height, VkFormat format) {
    const size_t texels = static_cast<size_t>(width) * height;
    if (format == VK_FORMAT_R8G8B8A8_UNORM && pixels.size() == texels * 4)
        return pixels;
    if (format != VK_FORMAT_R16G16B16A16_SFLOAT || pixels.size() != texels * 8)
        return {};
    std::vector<uint8_t> rgba(texels * 4);
    for (size_t texel = 0; texel < texels; ++texel) {
        for (uint32_t channel = 0; channel < 4; ++channel) {
            uint16_t half = 0;
            std::memcpy(&half, pixels.data() + texel * 8 + channel * 2, sizeof(half));
            const float value = prosper::gpu::half_to_float(half);
            rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
        }
    }
    return rgba;
}

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
    uint32_t width = 0, height = 0, depth = 1;
    uint32_t tile_mode = 0;
    uint32_t img_dim = 0;
    uint32_t max_uncompressed_block_size = 0, max_compressed_block_size = 0;
    uint32_t dcc_flags = 0;
    uint64_t metadata_addr = 0;
    uint64_t metadata_host_data = 0;
    uint64_t metadata_host_data_size = 0;
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
        mix(key.cls); mix(key.format); mix(key.num_components); mix(key.width); mix(key.height); mix(key.depth);
        mix(key.tile_mode); mix(key.img_dim); mix(key.max_uncompressed_block_size);
        mix(key.max_compressed_block_size); mix(key.dcc_flags); mix(key.metadata_addr);
        mix(key.metadata_host_data); mix(key.metadata_host_data_size);
        return hash;
    }
};

struct DecodedTexture {
    const uint8_t* pixels = nullptr;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t persistent_id = 0;
};

struct PersistentDecodedTexture {
    size_t source_size = 0;
    size_t source_prefix_size = 0;
    bool source_matches_pixels = false;
    std::vector<uint8_t> source_prefix;
    std::vector<uint8_t> pixels;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t last_use = 0;
    uint64_t persistent_id = 0;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    prosper::host::GuestWriteWatch source_watch;
    uint32_t source_watch_dirty_count = 0;
    bool source_watch_disabled = false;

    size_t bytes() const { return source_prefix.size() + pixels.size(); }
};
}

void register_live_renderer(const std::string& frame_dir, bool dump_bmps) {
    static RttCache g_rtt;   // render-to-texture cache (#167)
    // Match boot_trace's progression-diagnostic contract: callers may register the graphics
    // renderer while deliberately leaving compute unregistered. This keeps semantic dispatches
    // visible without letting screenshot/prosper-app registration silently undo the A/B.
    if (!getenv("PROSPER_NO_COMPUTE")) register_live_compute();
    const char* ds_invalidate = getenv("PROSPER_DS_GUEST_WRITE_INVALIDATE");
    const bool invalidate_ds = !ds_invalidate || strcmp(ds_invalidate, "0");
    prosper::gpu::set_guest_gpu_write_observer(
        [](uint64_t addr, uint64_t size) { queue_guest_gpu_write(addr, size); });
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
    if (getenv("PROSPER_GPU_CAPTURE") || getenv("PROSPER_GPU_TIMELINE_CAPTURE") ||
        getenv("PROSPER_GPU_REPLAY_EXPORT_RTT")) {
        prosper::gpu::set_gpu_capture_rtt_seed_reader([invalidate_ds](uint64_t addr, prosper::gpu::GpuCaptureRttSeed& seed) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            auto it = g_rtt.find(addr); if (it == g_rtt.end()) return false;
            if (!it->second.rgba) return false;
            seed.guest_addr = addr; seed.width = it->second.w; seed.height = it->second.h;
            seed.format = capture_color_format(it->second.format);
            seed.rgba = *it->second.rgba; return true;
        });
        prosper::gpu::set_gpu_capture_rtt_seed_snapshot_reader(
            [invalidate_ds](std::vector<prosper::gpu::GpuCaptureRttSeed>& seeds, std::string&) {
                drain_guest_gpu_writes(g_rtt, invalidate_ds);
                seeds.reserve(g_rtt.size());
                for (const auto& [addr, surface] : g_rtt) {
                    if (!surface.rgba) continue;
                    prosper::gpu::GpuCaptureRttSeed seed;
                    seed.guest_addr = addr; seed.width = surface.w; seed.height = surface.h;
                    seed.format = capture_color_format(surface.format);
                    seed.rgba = *surface.rgba; seeds.push_back(std::move(seed));
                }
                return true;
            });
    }
    // The compute backend must not sample a surface whose CURRENT pixels live in this renderer's
    // RTT cache (raw guest memory is then empty/stale — the Dead Cells 642x362 lesson): publish the
    // exact-match identity and immutable CPU snapshot used by live compute (#590).
    prosper::gpu::set_live_target_query([invalidate_ds](uint64_t addr) {
        drain_guest_gpu_writes(g_rtt, invalidate_ds);
        return g_rtt.count(addr) != 0;
    });
    prosper::gpu::set_live_target_reader(
        [invalidate_ds](uint64_t addr, prosper::gpu::LiveTargetSnapshot& snapshot) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            const auto it = g_rtt.find(addr);
            if (it == g_rtt.end() || !it->second.rgba || !it->second.w || !it->second.h)
                return false;
            const VkFormat format = prosper::test::backend_color_format(it->second.format);
            const uint32_t bytes_per_pixel = prosper::test::backend_color_bytes_per_pixel(format);
            const uint64_t texels = static_cast<uint64_t>(it->second.w) * it->second.h;
            if (texels > UINT64_MAX / bytes_per_pixel) return false;
            const uint64_t expected = texels * bytes_per_pixel;
            if (expected != it->second.rgba->size()) return false;
            snapshot.width = it->second.w;
            snapshot.height = it->second.h;
            snapshot.format = format == VK_FORMAT_R16G16B16A16_SFLOAT
                ? prosper::gpu::LiveTargetPixelFormat::Rgba16Float
                : prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = it->second.rgba;
            return true;
        });
    prosper::gpu::set_live_target_byte_range_reader(
        [invalidate_ds](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            if (!addr || !bytes) return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
            for (auto& [base, surface] : g_rtt) {
                if (addr < base) continue;
                if (!surface.w || !surface.h) {
                    if (addr == base) return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                    continue;
                }
                const VkFormat format = prosper::test::backend_color_format(surface.format);
                const uint64_t bpp = prosper::test::backend_color_bytes_per_pixel(format);
                const uint64_t pixels = static_cast<uint64_t>(surface.w) * surface.h;
                if (!bpp || pixels > UINT64_MAX / bpp) continue;
                const uint64_t target_bytes = pixels * bpp;
                const uint64_t offset = addr - base;
                if (offset >= target_bytes) continue;
                if ((!surface.rgba || surface.rgba->size() != target_bytes) &&
                    surface.gpu_valid) {
                    std::vector<uint8_t> materialized;
                    std::string error;
                    if (prosper::test::readback_persistent_color_target(
                            base, surface.w, surface.h, format, materialized, error) &&
                        materialized.size() == target_bytes) {
                        surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                            std::move(materialized));
                    } else {
                        static std::atomic<int> warned{0};
                        if (warned.fetch_add(1) < 24)
                            std::fprintf(stderr,
                                         "[rtt] ordered DMA target readback failed: base=0x%llx "
                                         "extent=%ux%u error=%s\n",
                                         static_cast<unsigned long long>(base), surface.w, surface.h,
                                         error.c_str());
                        return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                    }
                }
                if (!surface.rgba || surface.rgba->size() != target_bytes ||
                    bytes > target_bytes - offset)
                    return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
                output.assign(surface.rgba->begin() + static_cast<size_t>(offset),
                              surface.rgba->begin() + static_cast<size_t>(offset + bytes));
                return prosper::gpu::LiveTargetByteReadResult::Success;
            }
            return prosper::gpu::LiveTargetByteReadResult::NotFound;
        });
    if (getenv("PROSPER_GPU_CAPTURE") || getenv("PROSPER_GPU_TIMELINE_CAPTURE") ||
        getenv("PROSPER_GPU_REPLAY_EXPORT_DS"))
        prosper::gpu::set_gpu_capture_ds_seed_snapshot_reader(
            [](std::vector<prosper::gpu::GpuCaptureDsSeed>& seeds, std::string& error) {
                return prosper::test::snapshot_persistent_ds_images(seeds, error);
            });
    if (getenv("PROSPER_GPU_REPLAY_RTT_SEEDS"))
        prosper::gpu::set_gpu_replay_rtt_seed_writer([](const prosper::gpu::GpuCaptureRttSeed& seed, std::string& error) {
            const VkFormat format = replay_color_format(seed.format);
            const uint64_t expected = static_cast<uint64_t>(seed.width) * seed.height *
                                      prosper::test::backend_color_bytes_per_pixel(format);
            if (!seed.guest_addr || !seed.width || !seed.height || expected != seed.rgba.size()) {
                error = "invalid temporal RTT seed"; return false;
            }
            RttSurf& surface = g_rtt[seed.guest_addr];
            surface.w = seed.width; surface.h = seed.height;
            surface.format = format;
            surface.rgba = std::make_shared<const std::vector<uint8_t>>(seed.rgba);
            surface.gpu_valid = false;
            prosper::test::invalidate_persistent_color_target(seed.guest_addr);
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
    // Retain intermediate color targets on the GPU by default. Captures and per-target pixel
    // diagnostics require authoritative CPU pixels at every pass, so they retain the established
    // readback path. The explicit opt-out keeps a direct A/B and a recovery switch for driver issues.
    static const bool live_gpu_targets = pertarget &&
        !getenv("PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS") && !getenv("PROSPER_GPU_CAPTURE") &&
        !getenv("PROSPER_GPU_TIMELINE_CAPTURE") && !getenv("PROSPER_GPU_REPLAY_EXPORT_RTT") &&
        !getenv("PROSPER_GPU_REPLAY_RTT_SEEDS") && !getenv("PROSPER_DUMP_SAMPLED_RTT") &&
        !getenv("PROSPER_DUMP_RTGROUPS") && !getenv("PROSPER_DUMP_RTGROUPS_RGBA") &&
        !getenv("PROSPER_DUMP_DRAWSTEPS") &&
        !getenv("PROSPER_RESOURCE_HASH_DIM") && !getenv("PROSPER_TARGET_STEP_HASH_DIM") &&
        !getenv("PROSPER_RTTLOG");
    if (live_gpu_targets)
        fprintf(stderr, "[render] persistent GPU color targets enabled (experimental)\n");
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps, invalidate_ds](const std::vector<prosper::gpu::DrawItem>& items,
                               uint32_t w, uint32_t h) -> prosper::gpu::RenderedFrame {
            using RC = prosper::gpu::ResourceClass;
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
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
            static thread_local bool g_force_this_submit = false;
            if (phase.first_span || g_this_submit < 0) {
                g_this_submit = g_submit_idx++;
                g_force_this_submit = false;
            }
            if (g_this_submit > g_render_last) return {};
            static const int g_rttlog_min_submit = getenv("PROSPER_RTTLOG_MIN_SUBMIT")
                ? std::max(0, atoi(getenv("PROSPER_RTTLOG_MIN_SUBMIT"))) : 0;
            static const int g_rttlog_max_submit = getenv("PROSPER_RTTLOG_MAX_SUBMIT")
                ? std::max(0, atoi(getenv("PROSPER_RTTLOG_MAX_SUBMIT"))) : INT_MAX;
            const bool rtt_log_in_range =
                g_this_submit >= g_rttlog_min_submit && g_this_submit <= g_rttlog_max_submit;
            const bool rtt_log = getenv("PROSPER_RTTLOG") && rtt_log_in_range;
            using RenderClock = std::chrono::steady_clock;
            struct RenderTiming {
                uint64_t callbacks = 0;
                double total_ms = 0, build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
                uint64_t backend_calls = 0, backend_draws = 0;
                double backend_target_ms = 0, backend_draw_setup_ms = 0;
                double backend_record_upload_ms = 0, backend_gpu_wait_ms = 0;
                double backend_readback_ms = 0, backend_cleanup_ms = 0;
                double backend_setup_shader_ms = 0, backend_setup_fixed_ms = 0;
                double backend_setup_resources_ms = 0, backend_setup_pipeline_ms = 0;
                uint64_t backend_pipeline_refs = 0, backend_pipeline_hits = 0;
                uint64_t backend_pipeline_misses = 0, backend_pipeline_bypasses = 0;
                uint64_t backend_pipeline_entries = 0, backend_pipeline_evictions = 0;
                uint64_t color_target_writes = 0, color_target_write_hits = 0;
                uint64_t color_target_sample_hits = 0, color_target_readbacks = 0;
                uint64_t color_target_cached_bytes = 0, color_target_cached_entries = 0;
                uint64_t textures = 0, texture_reuses = 0, buffers = 0;
                uint64_t persistent_hits = 0, persistent_misses = 0, persistent_invalidations = 0;
                uint64_t persistent_submit_reuses = 0, persistent_validations = 0;
                uint64_t persistent_validation_bytes = 0;
                uint64_t persistent_watch_reuses = 0, persistent_watch_dirty = 0;
                uint64_t persistent_watch_unknown = 0, persistent_watch_disabled = 0;
                uint64_t texture_bytes = 0, buffer_bytes = 0;
                double texture_ms = 0, buffer_ms = 0;
            };
            struct RttTimingRecord {
                int submit = 0;
                uint64_t target = 0;
                uint32_t width = 0, height = 0;
                size_t draws = 0;
                double measured_ms = 0;
                prosper::test::BackendRenderTimingStats timing;
                prosper::test::BackendColorTargetStats color_target;
            };
            static thread_local RenderTiming pending_timing;
            static thread_local std::vector<RttTimingRecord> pending_rtt_timing;
            const bool timing_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr;
            const bool lightweight_rtt_timing = timing_enabled && getenv("PROSPER_RTT_TIMING");
            static const uint64_t rtt_timing_min_draws = getenv("PROSPER_RTT_TIMING_MIN_DRAWS")
                ? strtoull(getenv("PROSPER_RTT_TIMING_MIN_DRAWS"), nullptr, 0) : 0;
            if (timing_enabled && phase.first_span) {
                pending_timing = {};
                pending_rtt_timing.clear();
            }
            const auto callback_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            auto record_backend_timing = [&]
                (const prosper::test::BackendRenderTimingStats& backend,
                 const prosper::test::BackendPipelineCacheStats& pipelines) {
                pending_timing.backend_calls += backend.calls;
                pending_timing.backend_draws += backend.draws;
                pending_timing.backend_target_ms += backend.target_ms;
                pending_timing.backend_draw_setup_ms += backend.draw_setup_ms;
                pending_timing.backend_record_upload_ms += backend.record_upload_ms;
                pending_timing.backend_gpu_wait_ms += backend.gpu_wait_ms;
                pending_timing.backend_readback_ms += backend.readback_ms;
                pending_timing.backend_cleanup_ms += backend.cleanup_ms;
                pending_timing.backend_setup_shader_ms += backend.setup_shader_ms;
                pending_timing.backend_setup_fixed_ms += backend.setup_fixed_ms;
                pending_timing.backend_setup_resources_ms += backend.setup_resources_ms;
                pending_timing.backend_setup_pipeline_ms += backend.setup_pipeline_ms;
                pending_timing.backend_pipeline_refs += pipelines.references;
                pending_timing.backend_pipeline_hits += pipelines.hits;
                pending_timing.backend_pipeline_misses += pipelines.misses;
                pending_timing.backend_pipeline_bypasses += pipelines.bypasses;
                pending_timing.backend_pipeline_entries = pipelines.entries;
                pending_timing.backend_pipeline_evictions += pipelines.evictions;
            };
            auto append_rtt_timing = [](std::string& output, const RttTimingRecord& record) {
                const prosper::test::BackendRenderTimingStats& timing = record.timing;
                const double detail_ms = timing.total_ms();
                char line[512];
                const int length = snprintf(
                    line, sizeof(line),
                    "[rtt-timing] submit=%d target=0x%llx extent=%ux%u draws=%zu "
                    "measured=%.2f detail=%.2f other=%.2f target_setup=%.2f "
                    "draw_setup=%.2f record_upload=%.2f gpu_wait=%.2f "
                    "readback=%.2f cleanup=%.2f gpu_target=%llu load=%llu sample=%llu cpu=%llu\n",
                    record.submit, (unsigned long long)record.target,
                    record.width, record.height, record.draws, record.measured_ms, detail_ms,
                    record.measured_ms - detail_ms,
                    timing.target_ms, timing.draw_setup_ms, timing.record_upload_ms,
                    timing.gpu_wait_ms, timing.readback_ms, timing.cleanup_ms,
                    (unsigned long long)record.color_target.writes,
                    (unsigned long long)record.color_target.write_hits,
                    (unsigned long long)record.color_target.sampled_hits,
                    (unsigned long long)record.color_target.readbacks);
                if (length > 0)
                    output.append(line, std::min<size_t>(length, sizeof(line) - 1));
            };
            auto print_rtt_timing = [&](const RttTimingRecord& record) {
                std::string output;
                append_rtt_timing(output, record);
                if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
            };
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
            // Ordered submits can contain several graphics spans separated by compute work. Once a
            // diagnostic target selects one span, keep rendering the rest of that transaction so the
            // final span can recover and return the scanout assembled in the persistent RTT cache.
            g_force_this_submit |= force_target;
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
            if ((g_this_submit < g_render_first || before_delay) && !g_force_this_submit) return {};
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
#ifdef _WIN32
                // Prepare a complete sparse direct-memory resource once. The per-chunk mapping
                // checks below still stop an over-declared resource at its real guest boundary.
                if (prosper_try_commit_dmem(a, n, 0)) {
                    std::memcpy(dst, (const void*)(uintptr_t)a, n);
                    return n;
                }
#endif
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
            auto safe_equal = [](const uint8_t* expected, uint64_t a, size_t n,
                                 size_t& compared) -> bool {
                const size_t PG = 0x10000;
#ifdef _WIN32
                if (prosper_try_commit_dmem(a, n, 0)) {
                    compared = n;
                    return std::memcmp(expected, (const void*)(uintptr_t)a, n) == 0;
                }
#endif
                compared = 0;
                while (compared < n) {
                    const uint64_t cur = a + compared;
                    if (cur < 0x1000 || prosper_reserved_range_state(cur) == 0) return true;
                    const size_t chunk = std::min(
                        n - compared, PG - static_cast<size_t>(cur & (PG - 1)));
                    if (std::memcmp(expected + compared,
                                    reinterpret_cast<const void*>(static_cast<uintptr_t>(cur)),
                                    chunk)) return false;
                    compared += chunk;
                }
                return true;
            };
            // Keep decoded texture storage alive across callbacks. The old clear()+emplace(size, 0)
            // released and zero-filled tens of MiB every submit even though the decode paths overwrite
            // all pixels. Reusing same-sized slots avoids both costs; short guest reads explicitly clear
            // their uncovered tail below so stale scratch bytes can never become sampled pixels.
            static std::vector<std::vector<uint8_t>> texstore;
            size_t texstore_used = 0;
            std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash> decoded_textures;
            static std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>
                persistent_decoded_textures;
            static size_t persistent_decoded_texture_bytes = 0;
            static uint64_t persistent_decode_generation = 0;
            const uint64_t decode_generation = ++persistent_decode_generation;
            static uint64_t persistent_texture_id = 0;
            static std::vector<uint8_t> persistent_validation_scratch;
            const size_t persistent_decode_limit = [] {
                const char* value = getenv("PROSPER_TEXTURE_DECODE_CACHE_MB");
                const uint64_t mib = value ? strtoull(value, nullptr, 10) : 1024ull;
                return static_cast<size_t>(std::min<uint64_t>(mib, SIZE_MAX / (1024ull * 1024ull))) *
                       1024ull * 1024ull;
            }();
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
                    bool resource_local_reuse = false;
                    bool resource_persistent_hit = false;
                    bool resource_persistent_submit_reuse = false;
                    bool resource_persistent_miss = false;
                    bool resource_persistent_invalidation = false;
                    prosper::test::FrameResource fr; fr.binding = r.binding; fr.set = set;
                    fr.is_storage_image = r.cls == RC::StorageImage;
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
                    auto copy_dcc_metadata = [&](uint8_t* dst, size_t n) -> size_t {
                        if (!r.dcc_metadata_host_data)
                            return safe_copy(dst, r.metadata_addr, n);
                        const size_t take = static_cast<size_t>(std::min<uint64_t>(
                            n, r.dcc_metadata_host_data_size));
                        std::memcpy(dst, r.dcc_metadata_host_data, take);
                        return take;
                    };
                    if (r.cls == RC::Texture || r.cls == RC::StorageImage) {
                        uint32_t tw = r.width ? r.width : 4, th = r.height ? r.height : 4;
                        const TextureDecodeKey decode_key{
                            r.gpu_addr, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(r.host_data)),
                            r.host_data_size, r.size, static_cast<uint32_t>(r.cls),
                            static_cast<uint32_t>(r.format), r.num_components, tw, th, r.depth,
                            r.tile_mode, r.img_dim,
                            r.compression_enabled ? r.max_uncompressed_block_size : 0u,
                            r.compression_enabled ? r.max_compressed_block_size : 0u,
                            r.compression_enabled
                                ? ((r.meta_pipe_aligned ? 1u : 0u) |
                                   (r.write_compress_enabled ? 2u : 0u) | 4u |
                                   (r.alpha_is_on_msb ? 8u : 0u) |
                                   (r.color_transform ? 16u : 0u))
                                : 0u,
                            r.compression_enabled ? r.metadata_addr : 0u,
                            r.compression_enabled
                                ? static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                                      r.dcc_metadata_host_data))
                                : 0u,
                            r.compression_enabled ? r.dcc_metadata_host_data_size : 0u,
                        };
                        auto live_rtt = rtt_on ? g_rtt.find(r.gpu_addr) : g_rtt.end();
                        const bool has_cpu_live_rtt = r.img_dim == 1u && live_rtt != g_rtt.end() &&
                            live_rtt->second.w && live_rtt->second.h && live_rtt->second.rgba &&
                            !live_rtt->second.rgba->empty();
                        // A retained render target was created for color-attachment + sampled usage,
                        // not storage usage. Storage images therefore take the decoded/upload path.
                        const bool has_gpu_live_rtt = !fr.is_storage_image && live_gpu_targets &&
                            r.img_dim == 1u &&
                            live_rtt != g_rtt.end() && live_rtt->second.gpu_valid &&
                            live_rtt->second.w == tw && live_rtt->second.h == th &&
                            r.gpu_addr != draw.color0_base &&
                            prosper::test::find_persistent_color_target(
                                r.gpu_addr, tw, th, live_rtt->second.format) != nullptr;
                        const bool has_live_rtt = has_cpu_live_rtt || has_gpu_live_rtt;
                        const bool is_cube = r.img_dim == 3u;   // CUBE: six faces stacked vertically (#273)
                        const bool is_volume = r.img_dim == 2u;
                        const uint32_t persistent_pitch = getenv("PROSPER_PITCH")
                            ? static_cast<uint32_t>(atoi(getenv("PROSPER_PITCH"))) : 0;
                        const uint32_t persistent_bc_block_bytes =
                            prosper::gpu::bc_block_bytes(r.format);
                        const bool persistent_unorm8_texture =
                            r.format == prosper::gpu::DataFormat::Unorm8 &&
                            r.num_components >= 1 && r.num_components <= 4;
                        const bool persistent_sampled_texture =
                            !has_live_rtt && !r.host_data && r.img_dim == 1u &&
                            r.cls == RC::Texture &&
                            (persistent_unorm8_texture || persistent_bc_block_bytes != 0);
                        const bool persistent_source_is_tiled =
                            persistent_sampled_texture && !getenv("PROSPER_NODETILE") &&
                            prosper::gpu::tile_mode_is_tiled(r.tile_mode);
                        const bool persistent_source_matches_pixels =
                            persistent_unorm8_texture && r.num_components == 4 &&
                            !prosper::gpu::tile_mode_is_tiled(r.tile_mode) &&
                            (!getenv("PROSPER_DETILE") || atoi(getenv("PROSPER_DETILE")) == 0);
                        const size_t persistent_source_size = [&] {
                            if (!persistent_sampled_texture) return size_t{0};
                            if (persistent_bc_block_bytes) {
                                const uint32_t bw = (tw + 3) / 4;
                                const uint32_t bh = (th + 3) / 4;
                                return persistent_source_is_tiled
                                    ? prosper::gpu::tiled_elements_bytes(
                                          bw, bh, persistent_bc_block_bytes, r.tile_mode)
                                    : static_cast<size_t>(bw) * bh * persistent_bc_block_bytes;
                            }
                            const uint32_t source_bpt = r.num_components;
                            if (persistent_source_is_tiled)
                                return prosper::gpu::tiled_surface_bytes(
                                    tw, th, r.tile_mode, persistent_pitch, source_bpt);
                            // Forced detiling treats a nominally linear Unorm8x4 descriptor as tiled.
                            // Its decode source is therefore not the linear byte range computed here.
                            if (source_bpt == 4 && !persistent_source_matches_pixels) return size_t{0};
                            return static_cast<size_t>(tw) * th * source_bpt;
                        }();
                        const bool persistent_cache_eligible = !fr.is_storage_image &&
                            !getenv("PROSPER_NO_TEXTURE_DECODE_CACHE") &&
                            !r.compression_enabled &&
                            persistent_decode_limit && persistent_source_size != 0;
                        static const bool cross_submit_watch_enabled =
                            !getenv("PROSPER_NO_CROSS_SUBMIT_TEXTURE_WRITE_WATCH");
                        static const bool audit_cross_submit_watch =
                            getenv("PROSPER_AUDIT_CROSS_SUBMIT_TEXTURE_WRITE_WATCH") != nullptr;
                        prosper::host::GuestWriteWatch pending_source_watch;
                        bool narrow_done = false;
                        auto reused = has_live_rtt ? decoded_textures.end()
                                                   : decoded_textures.find(decode_key);
                        DecodedTexture persistent_reuse;
                        const DecodedTexture* decoded_reuse = reused != decoded_textures.end()
                            ? &reused->second : nullptr;
                        resource_local_reuse = decoded_reuse != nullptr;
                        if (!decoded_reuse && persistent_cache_eligible) {
                            auto cached = persistent_decoded_textures.find(decode_key);
                            if (cached != persistent_decoded_textures.end() &&
                                cached->second.source_size == persistent_source_size) {
                                auto validate_exact = [&] {
                                    bool matches = false;
                                    size_t validated_bytes = 0;
                                    if (cached->second.source_matches_pixels) {
                                        matches = safe_equal(
                                            cached->second.pixels.data(), r.gpu_addr,
                                            persistent_source_size, validated_bytes) &&
                                            validated_bytes == cached->second.source_prefix_size;
                                    } else if (!getenv("PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY") &&
                                               cached->second.source_prefix.size() ==
                                                   persistent_source_size) {
                                        // The cached prefix owns the complete encoded texture. Compare
                                        // guest memory directly against it: copying the same 100+ MiB
                                        // working set into a scratch buffer before memcmp doubled the
                                        // validation traffic on every Evergate frame. safe_equal keeps
                                        // the same sparse-page/readability guards and exact byte check.
                                        matches = safe_equal(
                                            cached->second.source_prefix.data(), r.gpu_addr,
                                            persistent_source_size, validated_bytes) &&
                                            validated_bytes == persistent_source_size;
                                    } else {
                                        persistent_validation_scratch.resize(persistent_source_size);
                                        validated_bytes = copy_resource(
                                            persistent_validation_scratch.data(), r.gpu_addr,
                                            persistent_source_size);
                                        matches =
                                            validated_bytes == cached->second.source_prefix_size &&
                                            validated_bytes == cached->second.source_prefix.size() &&
                                            (validated_bytes == 0 || !std::memcmp(
                                                persistent_validation_scratch.data(),
                                                cached->second.source_prefix.data(), validated_bytes));
                                    }
                                    if (timing_enabled) {
                                        pending_timing.persistent_validations++;
                                        pending_timing.persistent_validation_bytes += validated_bytes;
                                    }
                                    return matches;
                                };
                                static const bool submit_reuse_enabled =
                                    !getenv("PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE");
                                static const bool audit_submit_reuse =
                                    getenv("PROSPER_AUDIT_SUBMIT_TEXTURE_VALIDATION_REUSE") != nullptr;
                                const bool submit_unchanged = submit_reuse_enabled &&
                                    prosper::gpu::guest_gpu_writes_since(
                                        cached->second.validation_snapshot, r.gpu_addr,
                                        persistent_source_size) ==
                                        prosper::gpu::GuestGpuWriteQuery::Unchanged;
                                prosper::host::GuestWriteWatchQuery watch_query =
                                    prosper::host::GuestWriteWatchQuery::Unknown;
                                if (!submit_unchanged && cross_submit_watch_enabled &&
                                    !cached->second.source_watch_disabled) {
                                    watch_query = cached->second.source_watch.query();
                                    if (timing_enabled) {
                                        if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty)
                                            pending_timing.persistent_watch_dirty++;
                                        else if (watch_query == prosper::host::GuestWriteWatchQuery::Unknown)
                                            pending_timing.persistent_watch_unknown++;
                                    }
                                    if (watch_query == prosper::host::GuestWriteWatchQuery::Dirty &&
                                        ++cached->second.source_watch_dirty_count >= 2) {
                                        cached->second.source_watch.reset();
                                        cached->second.source_watch_disabled = true;
                                        if (timing_enabled)
                                            pending_timing.persistent_watch_disabled++;
                                    } else if (watch_query ==
                                               prosper::host::GuestWriteWatchQuery::Unchanged) {
                                        cached->second.source_watch_dirty_count = 0;
                                    }
                                }
                                const bool watch_unchanged = !submit_unchanged &&
                                    watch_query == prosper::host::GuestWriteWatchQuery::Unchanged;
                                if (!submit_unchanged && !watch_unchanged &&
                                    cross_submit_watch_enabled &&
                                    !cached->second.source_watch_disabled) {
                                    // Arm before reading. A concurrent CPU write during or after the
                                    // authoritative comparison then dirties this registration instead of
                                    // landing in an unprotected compare-to-rearm window.
                                    if (!cached->second.source_watch.rearm())
                                        cached->second.source_watch =
                                            prosper::host::GuestWriteWatch::create(
                                                r.gpu_addr, persistent_source_size);
                                }
                                bool content_matches = false;
                                if (submit_unchanged || watch_unchanged) {
                                    const bool audit = submit_unchanged
                                        ? audit_submit_reuse : audit_cross_submit_watch;
                                    content_matches = audit ? validate_exact() : true;
                                    if (!content_matches) {
                                        fprintf(stderr,
                                                "[render] %s texture validation audit failed "
                                                "addr=0x%llx bytes=%zu\n",
                                                submit_unchanged ? "in-submit" : "cross-submit-watch",
                                                (unsigned long long)r.gpu_addr,
                                                persistent_source_size);
                                    } else {
                                        if (submit_unchanged) {
                                            resource_persistent_submit_reuse = true;
                                            if (timing_enabled)
                                                pending_timing.persistent_submit_reuses++;
                                        } else if (timing_enabled) {
                                            pending_timing.persistent_watch_reuses++;
                                        }
                                    }
                                } else {
                                    content_matches = validate_exact();
                                }
                                if (content_matches) {
                                    cached->second.last_use = decode_generation;
                                    cached->second.validation_snapshot =
                                        prosper::gpu::guest_gpu_write_snapshot();
                                    persistent_reuse = {cached->second.pixels.data(),
                                                        cached->second.output_height,
                                                        cached->second.narrow,
                                                        cached->second.persistent_id};
                                    decoded_reuse = &persistent_reuse;
                                    resource_persistent_hit = true;
                                    if (timing_enabled) pending_timing.persistent_hits++;
                                } else if (timing_enabled) {
                                    resource_persistent_invalidation = true;
                                    pending_timing.persistent_invalidations++;
                                }
                            } else {
                                // Establish the mutation boundary before the initial source read/decode.
                                // If registration is unsupported, the empty watch keeps all later reuse on
                                // the exact fallback.
                                if (cross_submit_watch_enabled)
                                    pending_source_watch = prosper::host::GuestWriteWatch::create(
                                        r.gpu_addr, persistent_source_size);
                                if (timing_enabled) {
                                    resource_persistent_miss = true;
                                    pending_timing.persistent_misses++;
                                }
                            }
                        }
                        if (has_gpu_live_rtt) {
                            fr.persistent_render_target_id = r.gpu_addr;
                            fr.tw = tw; fr.th = th; fr.td = 1; fr.img_dim = r.img_dim;
                            fr.texture_format = live_rtt->second.format;
                            resource_rtt_hit = true;
                        } else if (decoded_reuse) {
                            fr.tex_rgba = decoded_reuse->pixels;
                            fr.tw = tw;
                            fr.th = decoded_reuse->output_height;
                            fr.td = is_volume ? r.depth : 1u;
                            fr.img_dim = r.img_dim;
                            narrow_done = decoded_reuse->narrow;
                            fr.persistent_texture_id = decoded_reuse->persistent_id;
                            decoded_textures.emplace(
                                decode_key, DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                                          fr.persistent_texture_id});
                            if (timing_enabled) pending_timing.texture_reuses++;
                        } else {
                        const size_t volume_texels = (size_t)tw * th * (is_volume ? r.depth : 1u);
                        const VkFormat live_rtt_format = has_cpu_live_rtt
                            ? live_rtt->second.format : VK_FORMAT_R8G8B8A8_UNORM;
                        const uint32_t output_bpp = has_cpu_live_rtt
                            ? prosper::test::backend_color_bytes_per_pixel(live_rtt_format) : 4u;
                        size_t nb = volume_texels * output_bpp * (is_cube ? 6u : 1u);
                        size_t linear_source_prefix_size = 0;
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
                        // half->UNORM8 conversion path below for guest-backed textures. Renderer-owned
                        // Float16x4 RTTs bypass it and retain native bytes. Other unknown formats keep RGBA8.
                        const bool f16 = r.format == prosper::gpu::DataFormat::Float16 &&
                                         (bpt == 2 || bpt == 4 || bpt == 8);
                        if (bpt == 0 || (bpt > 4 && !f16)) bpt = 4;
                        bool f16_done = false;
                        // RTT (#167): if this texture's base is a color target we rendered into, inject those
                        // pixels (nearest-scaled to tw x th) instead of reading empty guest memory.
                        bool rtt_hit = false;
                        if (rtt_on && !is_volume && !r.in_mip_tail) { auto rit = g_rtt.find(r.gpu_addr);
                            if (rit != g_rtt.end() && rit->second.w && rit->second.h && rit->second.rgba &&
                                !rit->second.rgba->empty()) {
                                const RttSurf& s = rit->second;
                                const uint32_t rtt_bpp = prosper::test::backend_color_bytes_per_pixel(s.format);
                                const size_t expected = static_cast<size_t>(tw) * th * rtt_bpp;
                                if (s.w == tw && s.h == th && s.rgba->size() == expected) {
                                    std::memcpy(texture_pixels.data(), s.rgba->data(), expected);
                                } else {
                                    std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                                    for (uint32_t y = 0; y < th; y++) for (uint32_t x = 0; x < tw; x++) {
                                        uint32_t sx = (uint32_t)((uint64_t)x * s.w / tw), sy = (uint32_t)((uint64_t)y * s.h / th);
                                        size_t si = ((size_t)sy * s.w + sx) * rtt_bpp;
                                        if (si + rtt_bpp <= s.rgba->size())
                                            std::memcpy(&texture_pixels[((size_t)y * tw + x) * rtt_bpp],
                                                        &(*s.rgba)[si], rtt_bpp);
                                    }
                                }
                                fr.texture_format = s.format;
                                rtt_hit = true;
                                resource_rtt_hit = true;
                                // PROSPER_DUMP_SAMPLED_RTT (#710/#320): dump the EXACT RTT-layer pixels a
                                // draw samples, at sample time, so we can see whether an already-rendered
                                // layer (e.g. the title's menu panel) arrives bright or dark — disambiguates
                                // "layer rendered dark" from "composite/tint darkens a correct layer". One
                                // BMP per (sampled addr) into PROSPER_FRAME_DIR.
                                if (getenv("PROSPER_DUMP_SAMPLED_RTT")) {
                                    static std::set<uint64_t> seen;
                                    if (seen.insert(r.gpu_addr).second) {
                                        const std::vector<uint8_t> inspected = inspection_rgba8(
                                            texture_pixels, tw, th, s.format);
                                        size_t nz = 0, rgbnz = 0;
                                        for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                                            if (inspected[p] || inspected[p+1] || inspected[p+2]) rgbnz++;
                                            for (int k = 0; k < 4; k++) nz += (inspected[p+k] != 0);
                                        }
                                        const char* dd = getenv("PROSPER_FRAME_DIR");
                                        char fn[512]; snprintf(fn, sizeof fn, "%s/sampledrtt_%llx_%ux%u.bmp",
                                                               dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th);
                                        prosper::test::dump_bmp(fn, inspected, tw, th);
                                        fprintf(stderr, "[sampledrtt] addr=0x%llx %ux%u rgb_nonblack=%zu/%u -> %s\n",
                                                (unsigned long long)r.gpu_addr, tw, th, rgbnz, tw*th, fn);
                                    }
                                }
                            }
                            if (rtt_log)
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
                        // GFX8-GFX10 embeds four self-contained 0/1 fast clears in DCC metadata. A
                        // uniform metadata surface means every compression block has that value, so it
                        // can be materialized without interpreting compressed base bytes. Uniform 0xff
                        // means an emulated writer published ordinary uncompressed base texels and the
                        // normal format/detile path below is authoritative.
                        bool dcc_fast_clear_done = false;
                        bool dcc_uncompressed = false;
                        if (r.compression_enabled && !rtt_hit) {
                            const uint64_t metadata_bytes =
                                prosper::gpu::gpu_capture_dcc_metadata_footprint(r);
                            std::vector<uint8_t> metadata(static_cast<size_t>(metadata_bytes), 0);
                            const size_t metadata_got = metadata_bytes
                                ? copy_dcc_metadata(metadata.data(), metadata.size()) : 0;
                            uint8_t clear_code = 0;
                            dcc_uncompressed = metadata_got == metadata.size() &&
                                !metadata.empty() &&
                                std::all_of(metadata.begin(), metadata.end(),
                                            [](uint8_t code) { return code == 0xff; });
                            if (!dcc_uncompressed && metadata_got == metadata.size() &&
                                prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                                    texture_pixels.data(), texture_pixels.size() / 4,
                                    metadata.data(), metadata.size(), r.num_components,
                                    r.alpha_is_on_msb, &clear_code)) {
                                dcc_fast_clear_done = true;
                                static std::set<std::pair<uint64_t, uint64_t>> decoded_dcc_images;
                                if (decoded_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second)
                                    fprintf(stderr,
                                            "[render] DCC fast-clear addr=0x%llx meta=0x%llx "
                                            "%ux%ux%u fmt=%u code=0x%02x bytes=%zu\n",
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)r.metadata_addr,
                                            tw, th, r.depth, (unsigned)r.format, clear_code,
                                            metadata.size());
                            } else if (!dcc_uncompressed) {
                                static std::set<std::pair<uint64_t, uint64_t>> warned_dcc_images;
                                if (warned_dcc_images.emplace(r.gpu_addr, r.metadata_addr).second)
                                    fprintf(stderr,
                                            "[render] DCC-compressed sampled image addr=0x%llx "
                                            "meta=0x%llx %ux%ux%u fmt=%u tile=%u is unsupported; "
                                            "metadata=%zu/%llu first=0x%02x\n",
                                            (unsigned long long)r.gpu_addr,
                                            (unsigned long long)r.metadata_addr,
                                            tw, th, r.depth, (unsigned)r.format, r.tile_mode,
                                            metadata_got,
                                            (unsigned long long)metadata_bytes,
                                            metadata_got ? metadata[0] : 0u);
                            }
                        }
                        bool cube_done = dcc_fast_clear_done && is_cube;
                        if (is_cube && !rtt_hit && !dcc_fast_clear_done) {
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
                        const uint32_t bcb = (rtt_hit || cube_done || dcc_fast_clear_done)
                            ? 0u : prosper::gpu::bc_block_bytes(r.format);
                        if (rtt_hit || cube_done || dcc_fast_clear_done) { /* pixels already materialized */ }
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
                                if (r.in_mip_tail)
                                    prosper::gpu::detile_elements_level(
                                        lin.data(), traw.data(), tbytes, bw, bh, bcb,
                                        r.tile_mode, r.mip_tail_x, r.mip_tail_y);
                                else
                                    prosper::gpu::detile_elements(
                                        lin.data(), traw.data(), tbytes, bw, bh, bcb, r.tile_mode);
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
                            // 480x270 bloom-chain buffers). Guest memory still converts half->UNORM8 on
                            // upload. That path clamps values above 1.0 and remains separate from native
                            // tiled guest-texture upload and the renderer-owned RTT fix in #773.
                            // Missing components read (0,0,0,1) per the hardware rule; the T# DST_SEL
                            // swizzle still applies. CONFIDENCE: MED — the half decode is exact and
                            // unit-tested, but the [0,1] clamp loses >1.0 bloom energy for guest-backed
                            // textures; renderer-owned RTTs retain RGBA16F through the #773 path.
                            const uint32_t nc = bpt / 2;                    // fp16 components per texel
                            std::vector<uint8_t> hlin(volume_texels * bpt, 0);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = is_volume
                                    ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                                    : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
                                std::vector<uint8_t> traw(tbytes, 0);
                                size_t got = copy_resource(traw.data(), r.gpu_addr, tbytes);
                                if (got < hlin.size()) copy_resource(hlin.data(), r.gpu_addr, hlin.size());  // short backing -> linear fallback
                                else if (is_volume) prosper::gpu::detile_volume(
                                    hlin.data(), traw.data(), got, tw, th, r.depth, r.tile_mode, bpt);
                                else if (r.in_mip_tail) prosper::gpu::detile_surface_level(
                                    hlin.data(), traw.data(), got, tw, th, r.tile_mode, bpt,
                                    r.mip_tail_x, r.mip_tail_y);
                                else prosper::gpu::detile_surface(
                                    hlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                copy_resource(hlin.data(), r.gpu_addr, hlin.size());
                            }
                            for (size_t t = 0; t < volume_texels; t++) {
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
                            std::vector<uint8_t> nlin(volume_texels * bpt, 0);
                            bool tiled = prosper::gpu::tile_mode_is_tiled(r.tile_mode) && !getenv("PROSPER_NODETILE");
                            if (tiled) {
                                size_t tbytes = is_volume
                                    ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, r.tile_mode, bpt)
                                    : prosper::gpu::tiled_surface_bytes(tw, th, r.tile_mode, 0, bpt);
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
                                else if (is_volume) prosper::gpu::detile_volume(
                                    nlin.data(), traw.data(), got, tw, th, r.depth, r.tile_mode, bpt);
                                else if (r.in_mip_tail) prosper::gpu::detile_surface_level(
                                    nlin.data(), traw.data(), got, tw, th, r.tile_mode, bpt,
                                    r.mip_tail_x, r.mip_tail_y);
                                else prosper::gpu::detile_surface(
                                    nlin.data(), traw.data(), tw, th, r.tile_mode, 0, bpt);
                            } else {
                                copy_resource(nlin.data(), r.gpu_addr, nlin.size());
                            }
                            for (size_t t = 0; t < volume_texels; t++) {
                                uint8_t v = nlin[t * bpt];   // first (coverage) channel
                                uint8_t* p = &texture_pixels[t * 4];
                                p[0] = p[1] = p[2] = p[3] = v;
                            }
                            narrow_done = true;   // already detiled+expanded; skip the 32-bpp auto-detile below
                        } else {
                            const size_t got = copy_resource(texture_pixels.data(), r.gpu_addr, nb);
                            linear_source_prefix_size = got;
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
                        if (!rtt_hit && !cube_done && !dcc_fast_clear_done && !bcb && !narrow_done && !f16_done &&
                            !getenv("PROSPER_NODETILE") &&
                            (auto_tiled || (!is_volume && dt && atoi(dt) != 0))) {
                            const uint32_t tmode = auto_tiled ? r.tile_mode : (uint32_t)prosper::gpu::TileMode::Sw4KbS;
                            const uint32_t pitch = getenv("PROSPER_PITCH") ? (uint32_t)atoi(getenv("PROSPER_PITCH")) : 0;
                            size_t tiled_bytes = is_volume
                                ? prosper::gpu::tiled_volume_bytes(tw, th, r.depth, tmode, 4)
                                : prosper::gpu::tiled_surface_bytes(tw, th, tmode, pitch);
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
                            if (is_volume) prosper::gpu::detile_volume(
                                texture_pixels.data(), tiled.data(), got, tw, th, r.depth, tmode, 4);
                            else if (r.in_mip_tail) prosper::gpu::detile_surface_level(
                                texture_pixels.data(), tiled.data(), got, tw, th, tmode, 4,
                                r.mip_tail_x, r.mip_tail_y);
                            else prosper::gpu::detile_surface(
                                texture_pixels.data(), tiled.data(), tw, th, tmode, pitch);
                        }
                        // Packed R11G11B10F (Gen5 IMG_FMT 36 -> DataFormat::Float10_11_11, #294): UE4's
                        // scene-color RT format. The texel IS 4 bytes, so the generic read + auto-detile
                        // above already produced linear packed words; unpack each to RGBA8 in place with
                        // the same semantics as the fp16 path (NaN/neg -> 0, clamp [0,1], no alpha -> 255).
                        if (!rtt_hit && !dcc_fast_clear_done &&
                            r.format == prosper::gpu::DataFormat::Float10_11_11) {
                            uint8_t* tp = texture_pixels.data();
                            for (size_t t = 0; t < volume_texels; t++) {
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
                        // Packed R10G10B10A2 UNORM (GFX10 IMG_FMT 50 / "2_10_10_10_UNORM"):
                        // the generic 4-B read and detile above preserve packed texels; normalize each
                        // field to the RGBA8 image format used by this renderer before sampling.
                        if (!rtt_hit && !dcc_fast_clear_done &&
                            r.format == prosper::gpu::DataFormat::Unorm2_10_10_10) {
                            uint8_t* tp = texture_pixels.data();
                            for (size_t t = 0; t < volume_texels; t++) {
                                uint32_t v; std::memcpy(&v, tp + t * 4, 4);
                                prosper::gpu::unorm2_10_10_10_to_rgba8(v, tp + t * 4);
                            }
                        }
                        // Focused per-consumer version probe (#586). Hash both the raw guest backing
                        // and the final decoded/RTT-injected pixels so a live draw identifies whether
                        // divergence precedes format conversion or enters through renderer-owned state.
                        if (resource_hash_w == tw && resource_hash_h == th) {
                            const size_t raw_size = std::min<size_t>(
                                r.size ? r.size : volume_texels * 4, 64u << 20);
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
                            const std::vector<uint8_t> inspected = inspection_rgba8(
                                texture_pixels, tw, th, fr.texture_format);
                            size_t rgb_nonblack = 0, alpha_nonzero = 0;
                            for (size_t p = 0; p + 3 < inspected.size(); p += 4) {
                                rgb_nonblack += inspected[p] != 0 || inspected[p + 1] != 0 ||
                                                inspected[p + 2] != 0;
                                alpha_nonzero += inspected[p + 3] != 0;
                            }
                            const auto writer = prosper::gpu::last_guest_write_overlap(
                                r.gpu_addr, raw_size);
                            fprintf(stderr,
                                    "[resource-version] render-submit=%llu draw=%llu order=%llu set=%u bind=%u "
                                    "addr=0x%llx dims=%ux%u class=%u fmt=%u tile=%u dcc=%u meta=0x%llx rtt=%d "
                                    "raw=%zu/%zu:%016llx sample=%zu:%016llx rgb_nonblack=%zu alpha_nonzero=%zu "
                                    "writer=%s/%llu/%llu/%llu/0x%llx\n",
                                    (unsigned long long)g_this_submit,
                                    (unsigned long long)draw.draw_index,
                                    (unsigned long long)draw.command_order,
                                    set, r.binding, (unsigned long long)r.gpu_addr, tw, th,
                                    (unsigned)r.cls, (unsigned)r.format, r.tile_mode,
                                    r.compression_enabled, (unsigned long long)r.metadata_addr,
                                    (int)rtt_hit,
                                    raw_got, raw_size, (unsigned long long)raw_hash,
                                    texture_pixels.size(), (unsigned long long)sample_hash,
                                    rgb_nonblack, alpha_nonzero,
                                    writer ? prosper::gpu::guest_writer_kind_name(writer->kind) : "none",
                                    (unsigned long long)(writer ? writer->submit : 0),
                                    (unsigned long long)(writer ? writer->item : 0),
                                    (unsigned long long)(writer ? writer->order : 0),
                                    (unsigned long long)(writer ? writer->identity : 0));
                            if (getenv("PROSPER_DUMP_RESOURCE_VERSION")) {
                                static std::set<std::pair<uint64_t, uint64_t>> dumped_versions;
                                if (dumped_versions.emplace(r.gpu_addr, sample_hash).second) {
                                    const char* dd = getenv("PROSPER_FRAME_DIR");
                                    char fn[512];
                                    snprintf(fn, sizeof fn, "%s/resource_%llx_%ux%u_%016llx.bmp",
                                             dd && *dd ? dd : ".", (unsigned long long)r.gpu_addr, tw, th,
                                             (unsigned long long)sample_hash);
                                    prosper::test::dump_bmp(fn, inspected, tw, th);
                                    fprintf(stderr, "[resource-version] dumped decoded sample -> %s\n", fn);
                                }
                            }
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
                        const char* test_texture = getenv("PROSPER_TESTTEX");
                        const char* test_texture_binding = getenv("PROSPER_TESTTEX_BINDING");
                        const char* test_texture_draw = getenv("PROSPER_TESTTEX_DRAW");
                        const bool test_this_texture = test_texture &&
                            (!test_texture_binding ||
                             strtoul(test_texture_binding, nullptr, 0) == r.binding) &&
                            (!test_texture_draw ||
                             strtoull(test_texture_draw, nullptr, 0) == draw.draw_index);
                        if (test_this_texture) {
                            const uint32_t slices = is_volume ? std::max(r.depth, 1u) : 1u;
                            if (!strcmp(test_texture, "zero")) {
                                std::fill(texture_pixels.begin(), texture_pixels.end(), 0);
                            } else if (fr.texture_format == VK_FORMAT_R16G16B16A16_SFLOAT) {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            uint16_t* p = reinterpret_cast<uint16_t*>(
                                                texture_pixels.data() +
                                                (((size_t)z * th + y) * tw + x) * 8);
                                            const bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            p[0] = prosper::gpu::float_to_half((float)x / tw);
                                            p[1] = prosper::gpu::float_to_half((float)y / th);
                                            p[2] = prosper::gpu::float_to_half(ck ? 0.8f : 0.16f);
                                            p[3] = prosper::gpu::float_to_half(1.0f);
                                        }
                            } else {
                                for (uint32_t z = 0; z < slices; ++z)
                                    for (uint32_t y = 0; y < th; ++y)
                                        for (uint32_t x = 0; x < tw; ++x) {
                                            uint8_t* p = &texture_pixels[
                                                (((size_t)z * th + y) * tw + x) * 4];
                                            bool ck = ((x / 64) ^ (y / 64) ^ z) & 1;
                                            p[0] = (uint8_t)(255 * x / tw);
                                            p[1] = (uint8_t)(255 * y / th);
                                            p[2] = ck ? 200 : 40;
                                            p[3] = 255;
                                        }
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
                        if (getenv("PROSPER_DUMP_TEX") && !texture_pixels.empty() && frame_no < 200 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                            char fn[512]; snprintf(fn, sizeof fn, "%s/rawtex_f%04d_b%u.bmp", d.c_str(), (int)frame_no, r.binding);
                            prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            fprintf(stderr, "[render] dumped raw texture -> %s\n", fn); fflush(stderr);
                        }
                        // PROSPER_DUMP_ATLAS: dump each SMALL sampled texture once per ADDRESS (find the caption
                        // font among same-size UI textures). Capped.
                        if (getenv("PROSPER_DUMP_ATLAS") && !texture_pixels.empty() && tw <= 2048 && th <= 1024 &&
                            fr.texture_format == VK_FORMAT_R8G8B8A8_UNORM) {
                            static std::unordered_map<uint64_t,int> seen; static int ndumped = 0;
                            if (seen[r.gpu_addr]++ == 0 && ndumped++ < 60) {
                                std::string d = getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".";
                                char fn[512]; snprintf(fn, sizeof fn, "%s/tex_%ux%u_%llx_c%u.bmp", d.c_str(), tw, th,
                                                       (unsigned long long)r.gpu_addr, r.num_components);
                                prosper::test::dump_bmp(fn, texture_pixels, tw, th);
                            }
                        }
                        fr.tex_rgba = texture_pixels.data(); fr.tw = tw; fr.th = cube_done ? th * 6u : th;
                        fr.td = is_volume ? r.depth : 1u;
                        fr.img_dim = r.img_dim;
                        if (persistent_cache_eligible) {
                            size_t source_prefix_size = linear_source_prefix_size;
                            if (!persistent_source_matches_pixels) {
                                persistent_validation_scratch.resize(persistent_source_size);
                                source_prefix_size = copy_resource(
                                    persistent_validation_scratch.data(), r.gpu_addr,
                                    persistent_source_size);
                            }
                            auto old = persistent_decoded_textures.find(decode_key);
                            uint32_t inherited_watch_dirty_count = 0;
                            bool inherited_watch_disabled = false;
                            prosper::host::GuestWriteWatch inherited_source_watch =
                                std::move(pending_source_watch);
                            if (old != persistent_decoded_textures.end() &&
                                old->second.source_size == persistent_source_size) {
                                inherited_watch_dirty_count = old->second.source_watch_dirty_count;
                                inherited_watch_disabled = old->second.source_watch_disabled;
                                if (!inherited_watch_disabled)
                                    inherited_source_watch = std::move(old->second.source_watch);
                            }
                            const bool can_replace = old == persistent_decoded_textures.end() ||
                                old->second.last_use != decode_generation;
                            if (old != persistent_decoded_textures.end() && can_replace) {
                                persistent_decoded_texture_bytes -= old->second.bytes();
                                persistent_decoded_textures.erase(old);
                            }
                            const size_t required =
                                (persistent_source_matches_pixels ? 0 : source_prefix_size) +
                                texture_pixels.size();
                            while (required <= persistent_decode_limit &&
                                   persistent_decoded_texture_bytes > persistent_decode_limit - required) {
                                auto victim = persistent_decoded_textures.end();
                                for (auto it = persistent_decoded_textures.begin();
                                     it != persistent_decoded_textures.end(); ++it) {
                                    if (it->second.last_use == decode_generation) continue;
                                    if (victim == persistent_decoded_textures.end() ||
                                        it->second.last_use < victim->second.last_use) victim = it;
                                }
                                if (victim == persistent_decoded_textures.end()) break;
                                persistent_decoded_texture_bytes -= victim->second.bytes();
                                persistent_decoded_textures.erase(victim);
                            }
                            if (can_replace &&
                                required <= persistent_decode_limit &&
                                persistent_decoded_texture_bytes <= persistent_decode_limit - required) {
                                PersistentDecodedTexture cached;
                                cached.source_size = persistent_source_size;
                                cached.source_prefix_size = source_prefix_size;
                                cached.source_matches_pixels = persistent_source_matches_pixels;
                                if (!persistent_source_matches_pixels)
                                    cached.source_prefix.assign(
                                        persistent_validation_scratch.begin(),
                                        persistent_validation_scratch.begin() + source_prefix_size);
                                // A successful persistent insertion owns the decoded allocation from
                                // now on. Moving it avoids retaining the same large atlas in both the
                                // process-wide cache and a reusable scratch slot.
                                cached.pixels = std::move(texture_pixels);
                                cached.output_height = fr.th;
                                cached.narrow = narrow_done;
                                cached.last_use = decode_generation;
                                cached.persistent_id = ++persistent_texture_id;
                                if (!cached.persistent_id)
                                    cached.persistent_id = ++persistent_texture_id;
                                cached.validation_snapshot =
                                    prosper::gpu::guest_gpu_write_snapshot();
                                cached.source_watch_dirty_count = inherited_watch_dirty_count;
                                cached.source_watch_disabled = inherited_watch_disabled;
                                cached.source_watch = std::move(inherited_source_watch);
                                auto [inserted, ok] = persistent_decoded_textures.emplace(
                                    decode_key, std::move(cached));
                                if (ok) {
                                    persistent_decoded_texture_bytes += inserted->second.bytes();
                                    fr.tex_rgba = inserted->second.pixels.data();
                                    fr.persistent_texture_id = inserted->second.persistent_id;
                                }
                            }
                        }
                        if (!resource_rtt_hit && !has_live_rtt)
                            decoded_textures.emplace(
                                decode_key, DecodedTexture{fr.tex_rgba, fr.th, narrow_done,
                                                          fr.persistent_texture_id});
                        }
                        // Carry the decoded S# sampler state (filter/wrap/mip) so the pipeline samples the
                        // way the game asked instead of a fixed LINEAR/clamp sampler (#<sampler-fix>).
                        fr.mag_filter = r.mag_filter; fr.min_filter = r.min_filter; fr.mip_filter = r.mip_filter;
                        // Draw/binding-scoped sampler A/B. This shares TESTTEX's selectors but leaves
                        // the sampled pixels intact, isolating descriptor filtering from texture content.
                        const char* test_filter = getenv("PROSPER_TESTTEX_FILTER");
                        const char* test_filter_binding = getenv("PROSPER_TESTTEX_BINDING");
                        const char* test_filter_draw = getenv("PROSPER_TESTTEX_DRAW");
                        const bool filter_valid = test_filter &&
                            (!strcmp(test_filter, "linear") || !strcmp(test_filter, "point"));
                        if (filter_valid &&
                            (!test_filter_binding ||
                             strtoul(test_filter_binding, nullptr, 0) == r.binding) &&
                            (!test_filter_draw ||
                             strtoull(test_filter_draw, nullptr, 0) == draw.draw_index)) {
                            const uint32_t filter = !strcmp(test_filter, "linear") ? 1u : 0u;
                            fr.mag_filter = fr.min_filter = filter;
                        }
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
                        fr.buffer_identity = r.gpu_addr;
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
                            pending_timing.texture_bytes += static_cast<uint64_t>(fr.tw) * fr.th * fr.td *
                                prosper::test::backend_color_bytes_per_pixel(fr.texture_format);
                            pending_timing.texture_ms += elapsed;
                            const uint64_t detail_min_submit = getenv("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT")
                                ? strtoull(getenv("PROSPER_RENDER_TIMING_DETAIL_MIN_SUBMIT"), nullptr, 0) : 0;
                            if (const char* mode = getenv("PROSPER_RENDER_TIMING");
                                mode && strcmp(mode, "detail") == 0 &&
                                static_cast<uint64_t>(g_this_submit) >= detail_min_submit) {
                                static uint64_t detail_lines = 0;
                                if ((elapsed >= 0.5 || resource_persistent_invalidation) &&
                                    detail_lines++ < 250) {
                                    const char* cache_state = resource_rtt_hit ? "rtt" :
                                        (resource_local_reuse ? "local" :
                                        (resource_persistent_submit_reuse ? "persistent-submit" :
                                        (resource_persistent_hit ? "persistent-hit" :
                                        (resource_persistent_invalidation ? "persistent-invalid" :
                                        (resource_persistent_miss ? "persistent-miss" : "uncached")))));
                                    fprintf(stderr,
                                            "[render-timing] texture addr=0x%llx %ux%u out=%ux%u "
                                            "fmt=%u comps=%u tile=%u cache=%s id=%llu %.2f ms\n",
                                            (unsigned long long)r.gpu_addr, r.width, r.height,
                                            fr.tw, fr.th, (unsigned)r.format, r.num_components,
                                            r.tile_mode, cache_state,
                                            (unsigned long long)fr.persistent_texture_id, elapsed);
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
                    const bool wants_storage_image =
                        d.kind == prosper::gpu::SpirvDescriptorKind::StorageImage;
                    const uint64_t available = first == resources.end() ? 0 :
                        (first->is_texture() ? (uint64_t)first->tw * first->th * first->td * 4
                                             : first->dwords.size() * 4);
                    const bool wrong_type = first != resources.end() &&
                        (wants_buffer ? first->is_texture()
                                      : (!first->is_texture() ||
                                         first->is_storage_image != wants_storage_image));
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
                        replacement.is_storage_image = wants_storage_image;
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
            //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading). The optional
            //             TESTPS_MATCH file restricts it to one exact recompiled guest PS.
            //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
            //   NOPS   -> bypass the resolved pipeline state (default state).
            #include "refvs.inc"
            const bool refvs = getenv("PROSPER_RENDER_REFVS");
            std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
            std::vector<uint32_t> ps_override;
            bool ps_override_is_file = false;   // true only for a valid PROSPER_FS_SPV *file* override
            bool ps_override_is_test = false;
            if (getenv("PROSPER_RENDER_TESTPS")) {
                static const uint32_t kMagentaPs[] = {   // v0=1.0(R) v1=0.0(G) v2=1.0(B) v3=1.0(A); exp mrt0; endpgm
                    0x7E0002F2u, 0x7E020280u, 0x7E0402F2u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
                ps_override = prosper::gpu::recompile_fragment(kMagentaPs, sizeof(kMagentaPs) / 4, nullptr);
                ps_override_is_test = true;
            }
            // Validated SPIR-V file load: require a complete read of a word-aligned file >= 20 bytes
            // (5 words: the minimum SPIR-V header). Validate size BEFORE allocating so a failed ftell
            // (-1 -> huge size_t) cannot trigger a wild allocation, and reject non-word-aligned files
            // rather than silently dropping trailing bytes. Returns false (out untouched) on any failure.
            auto load_spv_file = [](const char* path, std::vector<uint32_t>& out) -> bool {
                FILE* f = fopen(path, "rb");
                if (!f) return false;
                bool ok = false;
                if (fseek(f, 0, SEEK_END) == 0) {
                    long sz = ftell(f);
                    if (sz >= 20 && (sz % 4) == 0 && fseek(f, 0, SEEK_SET) == 0) {
                        std::vector<uint32_t> m(static_cast<size_t>(sz) / 4);
                        if (fread(m.data(), 4, m.size(), f) == m.size()) { out = std::move(m); ok = true; }
                    }
                }
                fclose(f);
                return ok;
            };
            if (const char* fsp = getenv("PROSPER_FS_SPV")) {
                std::vector<uint32_t> m;
                if (load_spv_file(fsp, m)) { ps_override = std::move(m); ps_override_is_file = true; }
                else fprintf(stderr, "[fs-spv] PROSPER_FS_SPV='%s' invalid/unreadable -> no file override\n", fsp);
            }
            // PROSPER_FS_SPV_MATCH=<file>: restrict the PROSPER_FS_SPV *file* override to draws whose
            // recompiled fragment SPIR-V EXACTLY equals this file (a per-draw A/B substitution that does
            // not touch draws with a different descriptor contract). FAILS CLOSED: if requested but the
            // file is missing/short/unaligned/unreadable, NO file override is applied — never a silent
            // global fallback (that would recreate the exact hazard this gate exists to prevent). Does
            // NOT gate PROSPER_RENDER_TESTPS, which stays global by design.
            // mode: 0 = not requested (legacy global file override), 1 = loaded+valid (exact match only),
            //       2 = requested but invalid (file override disabled).
            int fs_match_mode = 0;
            std::vector<uint32_t> fs_match;
            if (const char* mp = getenv("PROSPER_FS_SPV_MATCH")) {
                fs_match_mode = load_spv_file(mp, fs_match) ? 1 : 2;
                if (fs_match_mode == 2)
                    fprintf(stderr, "[fs-match] PROSPER_FS_SPV_MATCH='%s' invalid/unreadable -> applying NO "
                            "fragment file override (fail closed)\n", mp);
            }
            // PROSPER_RENDER_TESTPS_MATCH=<file> is the geometry half of a per-shader A/B test: replace
            // only that exact guest PS with the known solid output while retaining its real VS, indices,
            // viewport, depth and raster state. As with FS_SPV_MATCH, a bad path fails closed.
            int testps_match_mode = 0;
            std::vector<uint32_t> testps_match;
            if (const char* mp = getenv("PROSPER_RENDER_TESTPS_MATCH")) {
                testps_match_mode = load_spv_file(mp, testps_match) ? 1 : 2;
                if (testps_match_mode == 2)
                    fprintf(stderr, "[testps-match] PROSPER_RENDER_TESTPS_MATCH='%s' invalid/unreadable -> "
                            "applying NO test fragment override (fail closed)\n", mp);
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
                    bd.gs     = refvs ? std::vector<uint32_t>{} : it.gs;
                    // File and synthetic overrides have independent exact-match gates.
                    bool fs_ov = !ps_override.empty();
                    if (fs_ov && ps_override_is_file) {
                        if (fs_match_mode == 1)      fs_ov = (it.fs == fs_match);   // valid match -> exact only
                        else if (fs_match_mode == 2) fs_ov = false;                // requested-but-invalid -> off
                        // mode 0 -> legacy global file override (unchanged)
                    }
                    if (fs_ov && ps_override_is_test) {
                        if (testps_match_mode == 1)      fs_ov = (it.fs == testps_match);
                        else if (testps_match_mode == 2) fs_ov = false;
                        // mode 0 -> legacy global TESTPS override (unchanged)
                    }
                    if (fs_ov && ps_override_is_file && fs_match_mode == 1)
                        fprintf(stderr, "[fs-match] file override applied to draw#%llu\n",
                                (unsigned long long)it.draw_index);
                    if (fs_ov && ps_override_is_test && testps_match_mode == 1)
                        fprintf(stderr, "[testps-match] synthetic override applied to draw#%llu\n",
                                (unsigned long long)it.draw_index);
                    bd.fs     = fs_ov ? ps_override : it.fs;
                    bd.vs_identity = refvs ? 0 : it.vs_identity;
                    bd.fs_identity = fs_ov ? 0 : it.fs_identity;
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
                    if (rtt_log) {
                        fprintf(stderr, "[rtt]   draw#%llu vs=0x%llx fs=0x%llx tgt=0x%llx "
                                "vcount=%u nidx=%zu topo=%u mask=0x%x "
                                "blend=%d(src=%u dst=%u) vp=%d(%.2f,%.2f %.2fx%.2f) z=%d zw=%d ps=",
                                (unsigned long long)it.draw_index,
                                (unsigned long long)it.vs_guest_addr,
                                (unsigned long long)it.fs_guest_addr,
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
            std::shared_ptr<const std::vector<uint8_t>> selected_pixels;
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
                if (rtt_log) {
                    fprintf(stderr, "[rtt] flip state: front=%d va=0x%llx of %d registered:",
                            vo_front, (unsigned long long)front_va, vo_n);
                    for (int i = 0; i < vo_n && i < 8; i++)
                        fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
                    fprintf(stderr, "\n");
                }
                std::shared_ptr<const std::vector<uint8_t>> px_front;
                std::shared_ptr<const std::vector<uint8_t>> px_vo;
                std::shared_ptr<const std::vector<uint8_t>> px_last;
                // Render-target persistence (default on; PROSPER_RTT_NOSEED reverts to per-pass blue
                // clear): seed each group's framebuffer with the pixels last rendered into that SAME
                // target VA, so a pass that draws into an already-written target (UE4's UI pass onto
                // the backbuffer, incremental HUD updates) composites OVER the earlier content instead
                // of starting from the diagnostic clear. Real RT memory persists exactly this way.
                static const bool seed_rtt = getenv("PROSPER_RTT_NOSEED") == nullptr;
                auto active_color1 = [](const prosper::gpu::DrawItem& draw) {
                    return draw.ps.color1_write_mask && draw.color1_base
                        ? draw.color1_base : uint64_t{0};
                };
                auto active_format = [](const prosper::gpu::DrawItem& draw, bool color1) {
                    return prosper::test::backend_color_format(static_cast<VkFormat>(
                        color1 ? draw.ps.color1_format : draw.ps.color0_format));
                };
                size_t pass_i = 0;
                while (pass_i < items.size()) {
                    const uint64_t base = items[pass_i].color0_base;
                    const uint64_t base1 = active_color1(items[pass_i]);
                    const VkFormat format0 = active_format(items[pass_i], false);
                    const VkFormat format1 = base1
                        ? active_format(items[pass_i], true) : VK_FORMAT_UNDEFINED;
                    std::vector<const prosper::gpu::DrawItem*> pass;
                    while (pass_i < items.size() && items[pass_i].color0_base == base &&
                           active_color1(items[pass_i]) == base1 &&
                           active_format(items[pass_i], false) == format0 &&
                           (!base1 || active_format(items[pass_i], true) == format1)) {
                        pass.push_back(&items[pass_i]); ++pass_i;
                    }

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
                    // A depth prepass may bind a tiny/dummy color target with CB_TARGET_MASK=0 while
                    // rasterizing the full-size guest depth surface. Keying persistent DS from that
                    // irrelevant color extent creates a small depth image that the later lighting pass
                    // cannot reuse. Recover the attachment extent from the viewport, but only when the
                    // complete inferred extent fits the known presentation surface. Viewport coordinates
                    // position rasterization and are not allocation metadata; accepting translated or
                    // otherwise oversized coordinates here previously created persistent images as large
                    // as 16384x16384 and exhausted the host while Dead Cells loaded its first level.
                    const bool color_disabled = !pass.empty() && std::all_of(
                        pass.begin(), pass.end(), [](const auto* draw) {
                            return draw->ps.color_write_mask == 0;
                        });
                    const bool uses_ds = std::any_of(pass.begin(), pass.end(), [](const auto* draw) {
                        return draw->ps.depth_test_enable || draw->ps.depth_write_enable ||
                               draw->ps.depth_clear_enable || draw->ps.stencil_enable ||
                               draw->ps.stencil_clear_enable;
                    });
                    if (color_disabled && uses_ds && w && h) {
                        float viewport_x = 0.0f, viewport_y = 0.0f;
                        for (const auto* draw : pass) {
                            if (!draw->ps.has_viewport) continue;
                            const float x1 = draw->ps.viewport_x;
                            const float x2 = x1 + draw->ps.viewport_w;
                            const float y1 = draw->ps.viewport_y;
                            const float y2 = y1 + draw->ps.viewport_h;
                            if (std::isfinite(x1) && std::isfinite(x2))
                                viewport_x = std::max(viewport_x, std::max(std::fabs(x1), std::fabs(x2)));
                            if (std::isfinite(y1) && std::isfinite(y2))
                                viewport_y = std::max(viewport_y, std::max(std::fabs(y1), std::fabs(y2)));
                        }
                        const uint32_t max_native_w = present_w ? present_w : w;
                        const uint32_t max_native_h = present_h ? present_h : h;
                        const uint64_t viewport_native_w = static_cast<uint64_t>(
                            std::ceil(viewport_x * static_cast<float>(max_native_w) / w));
                        const uint64_t viewport_native_h = static_cast<uint64_t>(
                            std::ceil(viewport_y * static_cast<float>(max_native_h) / h));
                        const bool viewport_extent_valid = viewport_native_w && viewport_native_h &&
                            viewport_native_w <= max_native_w && viewport_native_h <= max_native_h;
                        if (getenv("PROSPER_DSLOG") && (viewport_native_w || viewport_native_h)) {
                            fprintf(stderr,
                                    "[ds] viewport-derived extent %llux%llu (presentation %ux%u) -> %s\n",
                                    (unsigned long long)viewport_native_w,
                                    (unsigned long long)viewport_native_h,
                                    max_native_w, max_native_h,
                                    viewport_extent_valid ? "accept" : "reject");
                        }
                        if (viewport_extent_valid) {
                            native_w = std::max(native_w, static_cast<uint32_t>(viewport_native_w));
                            native_h = std::max(native_h, static_cast<uint32_t>(viewport_native_h));
                        }
                    }
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
                                prosper::gpu::scale_resolved_render_area(item.ps, ax, ay);
                                render_pass.push_back(&item);
                            }
                        }
                    }
                    const uint8_t* seed = nullptr;
                    bool gpu_seed_available = false;
                    const VkFormat pass_format = format0;
                    const size_t pass_bytes = static_cast<size_t>(gw) * gh *
                        prosper::test::backend_color_bytes_per_pixel(pass_format);
                    if (seed_rtt && base) { auto sit = g_rtt.find(base);
                        gpu_seed_available = live_gpu_targets && sit != g_rtt.end() &&
                            sit->second.gpu_valid && sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format &&
                            prosper::test::find_persistent_color_target(
                                base, gw, gh, pass_format) != nullptr;
                        if (!gpu_seed_available && sit != g_rtt.end() &&
                            sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format && sit->second.rgba &&
                            sit->second.rgba->size() == pass_bytes)
                            seed = sit->second.rgba->data(); }
                    bool is_vo = false;
                    for (int i = 0; i < vo_n && !is_vo; i++)
                        is_vo = base && base == prosper_vo_buffer_addr(i);
                    bool sampled_exact_later = false;
                    bool feedback_later = false;
                    if (live_gpu_targets && base) {
                        for (size_t later = pass_i; later < items.size(); ++later) {
                            auto inspect = [&](const prosper::gpu::ShaderResourceTable* table) {
                                if (!table) return;
                                for (const auto& resource : table->resources) {
                                    if ((resource.cls != RC::Texture &&
                                         resource.cls != RC::StorageImage) ||
                                        resource.gpu_addr != base || resource.img_dim != 1u)
                                        continue;
                                    const uint32_t rw = resource.width ? resource.width : 4u;
                                    const uint32_t rh = resource.height ? resource.height : 4u;
                                    if (rw == gw && rh == gh) {
                                        sampled_exact_later = true;
                                        feedback_later |= items[later].color0_base == base;
                                    }
                                }
                            };
                            inspect(items[later].vrt.get());
                            inspect(items[later].prt.get());
                        }
                    }
                    // Defer only a proven graphics-to-graphics intermediate. A same-submit DMA asks
                    // its producer span for readback; a later-submit DMA materializes the persistent
                    // image synchronously through the byte-range reader above.
                    const bool defer_readback = live_gpu_targets && vo_n > 0 && base && !is_vo &&
                        base != front_va && sampled_exact_later && !feedback_later &&
                        !phase.authoritative_readback;
                    prosper::test::BackendColorTarget backend_target{
                        base, seed_rtt, !defer_readback, pass_format};
                    const bool color1_extent_matches = base1 && !render_pass.empty() &&
                        render_pass.front()->color1_width == native_w &&
                        render_pass.front()->color1_height == native_h;
                    // Keep an exact-capture diagnostic switch so MRT1 can be compared
                    // against the previous single-target behavior without recapturing.
                    const bool use_color1 = base1 && color1_extent_matches &&
                                            getenv("PROSPER_NO_MRT1") == nullptr;
                    const VkFormat pass_format1 = use_color1 ? format1 : VK_FORMAT_UNDEFINED;
                    const uint8_t* seed1 = nullptr;
                    if (seed_rtt && use_color1) { auto sit = g_rtt.find(base1);
                        if (sit != g_rtt.end() && sit->second.w == gw && sit->second.h == gh &&
                            sit->second.format == pass_format1 && sit->second.rgba &&
                            sit->second.rgba->size() == static_cast<size_t>(gw) * gh *
                                prosper::test::backend_color_bytes_per_pixel(pass_format1))
                            seed1 = sit->second.rgba->data(); }
                    if (base1 && !use_color1 && rtt_log) {
                        const auto* first = render_pass.empty() ? nullptr : render_pass.front();
                        fprintf(stderr,
                                "[rtt] skip MRT1 target=0x%llx extent=%ux%u; MRT0 is %ux%u\n",
                                (unsigned long long)base1,
                                first ? first->color1_width : 0u,
                                first ? first->color1_height : 0u, native_w, native_h);
                    }
                    const auto build_start = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    auto backend_draws = build_bds(render_pass);
                    const auto build_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    std::vector<uint8_t> gpx1;
                    std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(
                        backend_draws, gw, gh, seed, clear_for(render_pass), true,
                        live_gpu_targets && base ? &backend_target : nullptr,
                        seed1, use_color1 ? render_pass.front()->ps.clear_color1 : nullptr,
                        use_color1 ? &gpx1 : nullptr);
                    const auto backend_done = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    const prosper::test::BackendColorTargetStats color_target_call =
                        prosper::test::backend_color_target_stats();
                    const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
                        ? prosper::test::backend_render_timing_stats()
                        : prosper::test::BackendRenderTimingStats{};
                    const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
                        ? prosper::test::backend_pipeline_cache_stats()
                        : prosper::test::BackendPipelineCacheStats{};
                    if (timing_enabled) {
                        pending_timing.build_resources_ms +=
                            std::chrono::duration<double, std::milli>(build_done - build_start).count();
                        pending_timing.backend_ms +=
                            std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                        record_backend_timing(backend_call_timing, backend_pipeline_stats);
                        pending_timing.color_target_writes += color_target_call.writes;
                        pending_timing.color_target_write_hits += color_target_call.write_hits;
                        pending_timing.color_target_sample_hits += color_target_call.sampled_hits;
                        pending_timing.color_target_readbacks += color_target_call.readbacks;
                        pending_timing.color_target_cached_bytes = color_target_call.cached_bytes;
                        pending_timing.color_target_cached_entries = color_target_call.cached_entries;
                    }
                    auto pass_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(gpx));
                    if (base && color_target_call.writes) {
                        RttSurf& surface = g_rtt[base];
                        surface.w = gw;
                        surface.h = gh;
                        surface.format = pass_format;
                        surface.gpu_valid = prosper::test::find_persistent_color_target(
                            base, gw, gh, pass_format) != nullptr;
                        if (!pass_pixels->empty()) surface.rgba = pass_pixels;
                        else if (surface.gpu_valid) surface.rgba.reset();
                    } else if (base && !pass_pixels->empty()) {
                        RttSurf& surface = g_rtt[base];
                        surface.rgba = pass_pixels;
                        surface.w = gw;
                        surface.h = gh;
                        surface.format = pass_format;
                        surface.gpu_valid = false;
                    }
                    if (use_color1 && !gpx1.empty()) {
                        if (rtt_log) {
                            size_t nz = 0, rgb_nz = 0;
                            for (uint8_t byte : gpx1) nz += byte != 0;
                            if (pass_format1 == VK_FORMAT_R8G8B8A8_UNORM)
                                for (size_t p = 0; p + 3 < gpx1.size(); p += 4)
                                    rgb_nz += gpx1[p] != 0 || gpx1[p + 1] != 0 ||
                                              gpx1[p + 2] != 0;
                            fprintf(stderr,
                                    "[rtt] pass target1=0x%llx extent=%ux%u (%zu draws) "
                                    "px_nonzero=%zu rgb_nonblack=%zu\n",
                                    (unsigned long long)base1, gw, gh, pass.size(), nz, rgb_nz);
                        }
                        RttSurf& surface = g_rtt[base1];
                        surface.rgba = std::make_shared<const std::vector<uint8_t>>(std::move(gpx1));
                        surface.w = gw; surface.h = gh; surface.format = pass_format1;
                        surface.gpu_valid = false;
                    }
                    const std::vector<uint8_t>& rendered_pixels = *pass_pixels;
                    if (native_w == resource_hash_w && native_h == resource_hash_h &&
                        !rendered_pixels.empty()) {
                        uint64_t hash = 1469598103934665603ull;
                        for (uint8_t byte : rendered_pixels) {
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
                            const VkFormat step_format = prosper::test::backend_color_format(
                                static_cast<VkFormat>(prefix.front()->ps.color0_format));
                            const std::vector<uint8_t> step_rgba = inspection_rgba8(
                                step, gw, gh, step_format);
                            for (size_t p = 0; p + 3 < step_rgba.size(); p += 4) {
                                const uint8_t r = step_rgba[p], g = step_rgba[p + 1], b = step_rgba[p + 2];
                                dark += std::max({r, g, b}) < 64;
                                near_white += std::min({r, g, b}) > 240;
                                rgb_sum += r + g + b;
                            }
                            const double mean_rgb = step_rgba.empty() ? 0.0 :
                                (double)rgb_sum / ((step_rgba.size() / 4) * 3);
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
                    const RttTimingRecord rtt_timing_record{
                        g_this_submit, base, gw, gh, pass.size(),
                        std::chrono::duration<double, std::milli>(
                            backend_done - build_done).count(),
                        backend_call_timing, color_target_call};
                    if (lightweight_rtt_timing) pending_rtt_timing.push_back(rtt_timing_record);
                    else if (timing_enabled && rtt_log) print_rtt_timing(rtt_timing_record);
                    if (rtt_log) {
                        const std::vector<uint8_t> inspected = inspection_rgba8(
                            rendered_pixels, gw, gh, pass_format);
                        size_t nz = 0, rgb_nz = 0;
                        for (uint8_t b : rendered_pixels) nz += (b != 0);
                        for (size_t p = 0; p + 3 < inspected.size(); p += 4)
                            rgb_nz += (inspected[p] != 0 || inspected[p + 1] != 0 ||
                                       inspected[p + 2] != 0);
                        fprintf(stderr, "[rtt] pass target=0x%llx extent=%ux%u native=%ux%u (%zu draws) "
                                "px_nonzero=%zu rgb_nonblack=%zu cache_size=%zu%s%s\n",
                                (unsigned long long)base, gw, gh, native_w, native_h, pass.size(), nz, rgb_nz, g_rtt.size(),
                                is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : "");
                    }
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
                    // Per-target group dumps into PROSPER_FRAME_DIR. Two INDEPENDENT opt-ins (either may
                    // be set alone); PROSPER_DUMP_RTGROUPS_ADDR optionally limits a long replay to one
                    // target VA. Diagnostic; no default behavior change.
                    //  - PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes>: a 24-bit BMP (alpha dropped) of the
                    //    format-inspected pixels — to eyeball an intermediate pass (e.g. a UI/banner RT).
                    //  - PROSPER_DUMP_RTGROUPS_RGBA: the RAW RGBA8 backend bytes (alpha PRESERVED, no
                    //    format inspection) — needed to reason about premultiplied-alpha UI compositing
                    //    that samples an RT's alpha as a blend factor. Non-RGBA8 targets are skipped
                    //    visibly rather than writing native bytes under a misleading .rgba contract.
                    if ((getenv("PROSPER_DUMP_RTGROUPS") || getenv("PROSPER_DUMP_RTGROUPS_RGBA")) &&
                        !rendered_pixels.empty()) {
                        size_t nz = 0; for (uint8_t b : rendered_pixels) nz += (b != 0);
                        const char* address_filter = getenv("PROSPER_DUMP_RTGROUPS_ADDR");
                        const uint64_t wanted_base = address_filter && *address_filter
                            ? strtoull(address_filter, nullptr, 0) : 0;
                        const char* dd = getenv("PROSPER_FRAME_DIR");
                        // Identify the pass by its first..last draw index so multiple passes to the same
                        // target VA in one frame do not silently overwrite each other.
                        const uint64_t pass_d0 = render_pass.empty() ? 0u : render_pass.front()->draw_index;
                        const uint64_t pass_d1 = render_pass.empty() ? 0u : render_pass.back()->draw_index;
                        if (!wanted_base || base == wanted_base) {
                            if (const char* rg = getenv("PROSPER_DUMP_RTGROUPS"); rg && nz >= (size_t)atol(rg)) {
                                const std::vector<uint8_t> inspected = inspection_rgba8(
                                    rendered_pixels, gw, gh, pass_format);
                                char fn[512]; snprintf(fn, sizeof fn, "%s/rtgrp_%llx_%04d.bmp",
                                                       dd ? dd : ".", (unsigned long long)base, frame_no.load());
                                prosper::test::dump_bmp(fn, inspected, gw, gh);
                            }
                            // Independent of the BMP variable AND its nonzero threshold, so a fully
                            // transparent (all-zero) group is captured too. Self-describing filename
                            // (extent + draw range) and failure-visible: a missing file must not be
                            // mistaken for a transparent/empty result.
                            if (getenv("PROSPER_DUMP_RTGROUPS_RGBA")) {
                                const uint64_t expected_bytes_u64 = static_cast<uint64_t>(gw) * gh * 4u;
                                const bool rgba8_format = pass_format == VK_FORMAT_R8G8B8A8_UNORM;
                                const bool size_valid = expected_bytes_u64 <= SIZE_MAX &&
                                    rendered_pixels.size() == static_cast<size_t>(expected_bytes_u64);
                                if (!rgba8_format || !size_valid) {
                                    fprintf(stderr,
                                            "[rtt] rgba-dump skipped target=0x%llx %ux%u draws=%llu..%llu "
                                            "reason=%s format=%d expected=%llu actual=%zu\n",
                                            (unsigned long long)base, gw, gh,
                                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                                            rgba8_format ? "size-mismatch" : "unsupported-format",
                                            static_cast<int>(pass_format),
                                            (unsigned long long)expected_bytes_u64,
                                            rendered_pixels.size());
                                } else {
                                    char rn[600]; snprintf(rn, sizeof rn, "%s/rtgrp_%llx_%ux%u_f%04d_d%04llu-%04llu.rgba",
                                                           dd ? dd : ".", (unsigned long long)base, gw, gh,
                                                           frame_no.load(), (unsigned long long)pass_d0,
                                                           (unsigned long long)pass_d1);
                                    FILE* rf = fopen(rn, "wb");
                                    bool ok = rf != nullptr; size_t wrote = 0;
                                    if (rf) {
                                        wrote = fwrite(rendered_pixels.data(), 1, rendered_pixels.size(), rf);
                                        ok = (wrote == rendered_pixels.size());
                                        if (fclose(rf) != 0) ok = false;
                                    }
                                    fprintf(stderr, "[rtt] rgba-dump %s target=0x%llx %ux%u draws=%llu..%llu "
                                            "bytes=%zu path=%s\n", ok ? "ok" : "FAILED",
                                            (unsigned long long)base, gw, gh,
                                            (unsigned long long)pass_d0, (unsigned long long)pass_d1,
                                            wrote, rn);
                                }
                            }
                        }
                    }
                    if (!rendered_pixels.empty() && pass_format == VK_FORMAT_R8G8B8A8_UNORM) {
                        if (base && base == front_va) px_front = pass_pixels;  // the flipped buffer
                        if (is_vo)                    px_vo = pass_pixels;     // any registered scanout
                        px_last = pass_pixels;                                  // last non-empty (fallback)
                    }
                }
                // Present priority: the flipped front buffer > any registered scanout target > the
                // legacy "last group" fallback (unchanged behavior when no group targets a VO buffer).
                selected_pixels = px_front ? px_front : (px_vo ? px_vo : px_last);
            } else {
                // Single-framebuffer path: render_draws_rgba composites every draw into ONE framebuffer.
                std::vector<const prosper::gpu::DrawItem*> all; all.reserve(items.size());
                for (const auto& it : items) all.push_back(&it);
                const auto build_start = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                auto backend_draws = build_bds(all);
                const auto build_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                auto rendered = prosper::test::render_draws_rgba(
                    backend_draws, w, h, nullptr, clear_for(all), true);
                const auto backend_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
                    ? prosper::test::backend_render_timing_stats()
                    : prosper::test::BackendRenderTimingStats{};
                const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
                    ? prosper::test::backend_pipeline_cache_stats()
                    : prosper::test::BackendPipelineCacheStats{};
                if (timing_enabled) {
                    pending_timing.build_resources_ms +=
                        std::chrono::duration<double, std::milli>(build_done - build_start).count();
                    pending_timing.backend_ms +=
                        std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                    record_backend_timing(backend_call_timing, backend_pipeline_stats);
                }
                // RTT (#167): cache these rendered pixels under this submit's render-target base, so a later
                // composite pass that samples that address gets the scene we drew (not empty guest memory).
                selected_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(rendered));
                if (rtt_on && !selected_pixels->empty()) {
                    uint64_t tgt = 0;
                    for (const auto& it : items) if (it.color0_base) { tgt = it.color0_base; break; }
                    if (tgt) {
                        RttSurf& s = g_rtt[tgt];
                        s.rgba = selected_pixels; s.w = w; s.h = h;
                        s.format = VK_FORMAT_R8G8B8A8_UNORM; s.gpu_valid = false;
                        prosper::test::invalidate_persistent_color_target(tgt);
                    }
                    if (rtt_log) { size_t nz=0;
                        for (uint8_t byte : *selected_pixels) nz += byte != 0;
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
                        it->second.format != VK_FORMAT_R8G8B8A8_UNORM ||
                        !it->second.rgba || it->second.rgba->size() != (size_t)w * h * 4)
                        return nullptr;
                    return &it->second;
                };
                const int front = prosper::gpu::present_front_index();
                const RttSurf* scanout = front >= 0
                    ? cached_scanout(prosper_vo_buffer_addr(front)) : nullptr;
                if (!scanout) {
                    for (int i = 0; i < prosper_vo_buffer_count(); ++i)
                        if ((scanout = cached_scanout(prosper_vo_buffer_addr(i)))) break;
                }
                if (scanout) selected_pixels = scanout->rgba;
            }
            if (phase.final_span && lightweight_rtt_timing && rtt_log_in_range &&
                pending_timing.backend_draws >= rtt_timing_min_draws) {
                std::string output;
                output.reserve(pending_rtt_timing.size() * 320);
                for (const RttTimingRecord& record : pending_rtt_timing)
                    append_rtt_timing(output, record);
                if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
            }
            if (!phase.final_span) {
                if (timing_enabled) {
                    pending_timing.callbacks++;
                    pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                        RenderClock::now() - callback_timing_start).count();
                }
                return {};
            }
            static const std::vector<uint8_t> empty_pixels;
            const std::vector<uint8_t>& px = selected_pixels ? *selected_pixels : empty_pixels;
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
                    double total_ms = 0, build_resources_ms = 0, backend_ms = 0, output_copy_ms = 0;
                    uint64_t backend_calls = 0, backend_draws = 0;
                    double backend_target_ms = 0, backend_draw_setup_ms = 0;
                    double backend_record_upload_ms = 0, backend_gpu_wait_ms = 0;
                    double backend_readback_ms = 0, backend_cleanup_ms = 0;
                    double backend_setup_shader_ms = 0, backend_setup_fixed_ms = 0;
                    double backend_setup_resources_ms = 0, backend_setup_pipeline_ms = 0;
                    uint64_t backend_pipeline_refs = 0, backend_pipeline_hits = 0;
                    uint64_t backend_pipeline_misses = 0, backend_pipeline_bypasses = 0;
                    uint64_t backend_pipeline_entries = 0, backend_pipeline_evictions = 0;
                    uint64_t color_target_writes = 0, color_target_write_hits = 0;
                    uint64_t color_target_sample_hits = 0, color_target_readbacks = 0;
                    uint64_t color_target_cached_bytes = 0, color_target_cached_entries = 0;
                    uint64_t textures = 0, texture_reuses = 0, buffers = 0;
                    uint64_t persistent_hits = 0, persistent_misses = 0, persistent_invalidations = 0;
                    uint64_t persistent_submit_reuses = 0, persistent_validations = 0;
                    uint64_t persistent_validation_bytes = 0;
                    uint64_t persistent_watch_reuses = 0, persistent_watch_dirty = 0;
                    uint64_t persistent_watch_unknown = 0, persistent_watch_disabled = 0;
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
                    timing.output_copy_ms += pending_timing.output_copy_ms;
                    timing.backend_calls += pending_timing.backend_calls;
                    timing.backend_draws += pending_timing.backend_draws;
                    timing.backend_target_ms += pending_timing.backend_target_ms;
                    timing.backend_draw_setup_ms += pending_timing.backend_draw_setup_ms;
                    timing.backend_record_upload_ms += pending_timing.backend_record_upload_ms;
                    timing.backend_gpu_wait_ms += pending_timing.backend_gpu_wait_ms;
                    timing.backend_readback_ms += pending_timing.backend_readback_ms;
                    timing.backend_cleanup_ms += pending_timing.backend_cleanup_ms;
                    timing.backend_setup_shader_ms += pending_timing.backend_setup_shader_ms;
                    timing.backend_setup_fixed_ms += pending_timing.backend_setup_fixed_ms;
                    timing.backend_setup_resources_ms += pending_timing.backend_setup_resources_ms;
                    timing.backend_setup_pipeline_ms += pending_timing.backend_setup_pipeline_ms;
                    timing.backend_pipeline_refs += pending_timing.backend_pipeline_refs;
                    timing.backend_pipeline_hits += pending_timing.backend_pipeline_hits;
                    timing.backend_pipeline_misses += pending_timing.backend_pipeline_misses;
                    timing.backend_pipeline_bypasses += pending_timing.backend_pipeline_bypasses;
                    timing.backend_pipeline_entries = pending_timing.backend_pipeline_entries;
                    timing.backend_pipeline_evictions += pending_timing.backend_pipeline_evictions;
                    timing.color_target_writes += pending_timing.color_target_writes;
                    timing.color_target_write_hits += pending_timing.color_target_write_hits;
                    timing.color_target_sample_hits += pending_timing.color_target_sample_hits;
                    timing.color_target_readbacks += pending_timing.color_target_readbacks;
                    timing.color_target_cached_bytes = pending_timing.color_target_cached_bytes;
                    timing.color_target_cached_entries = pending_timing.color_target_cached_entries;
                    timing.textures += pending_timing.textures;
                    timing.texture_reuses += pending_timing.texture_reuses;
                    timing.persistent_hits += pending_timing.persistent_hits;
                    timing.persistent_misses += pending_timing.persistent_misses;
                    timing.persistent_invalidations += pending_timing.persistent_invalidations;
                    timing.persistent_submit_reuses += pending_timing.persistent_submit_reuses;
                    timing.persistent_validations += pending_timing.persistent_validations;
                    timing.persistent_validation_bytes += pending_timing.persistent_validation_bytes;
                    timing.persistent_watch_reuses += pending_timing.persistent_watch_reuses;
                    timing.persistent_watch_dirty += pending_timing.persistent_watch_dirty;
                    timing.persistent_watch_unknown += pending_timing.persistent_watch_unknown;
                    timing.persistent_watch_disabled += pending_timing.persistent_watch_disabled;
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
                    const double other = totals.total_ms - totals.build_resources_ms - totals.backend_ms -
                                         totals.output_copy_ms;
                    fprintf(stderr,
                            "[render-timing] frontend submits=%llu callbacks=%llu avg_ms: total=%.2f "
                            "build_resources=%.2f backend=%.2f output_copy=%.2f other=%.2f\n",
                            (unsigned long long)totals.submits, (unsigned long long)totals.callbacks,
                            totals.total_ms / nsub, totals.build_resources_ms / nsub,
                            totals.backend_ms / nsub, totals.output_copy_ms / nsub, other / nsub);
                    const double backend_detail_ms = totals.backend_target_ms +
                        totals.backend_draw_setup_ms + totals.backend_record_upload_ms +
                        totals.backend_gpu_wait_ms + totals.backend_readback_ms +
                        totals.backend_cleanup_ms;
                    fprintf(stderr,
                            "[render-timing] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                            "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                            "gpu_wait=%.2f readback=%.2f cleanup=%.2f other=%.2f\n",
                            totals.backend_calls / nsub, totals.backend_draws / nsub,
                            totals.backend_ms / nsub, backend_detail_ms / nsub,
                            totals.backend_target_ms / nsub, totals.backend_draw_setup_ms / nsub,
                            totals.backend_record_upload_ms / nsub, totals.backend_gpu_wait_ms / nsub,
                            totals.backend_readback_ms / nsub, totals.backend_cleanup_ms / nsub,
                            (totals.backend_ms - backend_detail_ms) / nsub);
                    fprintf(stderr,
                            "[render-timing] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                            "resources=%.2f pipeline=%.2f\n",
                            totals.backend_setup_shader_ms / nsub,
                            totals.backend_setup_fixed_ms / nsub,
                            totals.backend_setup_resources_ms / nsub,
                            totals.backend_setup_pipeline_ms / nsub);
                    fprintf(stderr,
                            "[render-timing] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                            "bypass=%.1f entries=%llu evictions=%.1f\n",
                            totals.backend_pipeline_refs / nsub,
                            totals.backend_pipeline_hits / nsub,
                            totals.backend_pipeline_misses / nsub,
                            totals.backend_pipeline_bypasses / nsub,
                            (unsigned long long)totals.backend_pipeline_entries,
                            totals.backend_pipeline_evictions / nsub);
                    fprintf(stderr,
                            "[render-timing] color_targets writes=%llu load_hits=%llu sample_hits=%llu "
                            "readbacks=%llu deferred=%llu cached=%llu %.1f MiB\n",
                            (unsigned long long)totals.color_target_writes,
                            (unsigned long long)totals.color_target_write_hits,
                            (unsigned long long)totals.color_target_sample_hits,
                            (unsigned long long)totals.color_target_readbacks,
                            (unsigned long long)(totals.color_target_writes -
                                                 totals.color_target_readbacks),
                            (unsigned long long)totals.color_target_cached_entries,
                            totals.color_target_cached_bytes / (1024.0 * 1024.0));
                    fprintf(stderr,
                            "[render-timing] resources textures=%llu reused=%llu %.1f MiB %.2f ms/submit; "
                            "buffers=%llu %.1f MiB %.2f ms/submit\n",
                            (unsigned long long)totals.textures,
                            (unsigned long long)totals.texture_reuses,
                            totals.texture_bytes / (1024.0 * 1024.0), totals.texture_ms / nsub,
                            (unsigned long long)totals.buffers,
                            totals.buffer_bytes / (1024.0 * 1024.0), totals.buffer_ms / nsub);
                    fprintf(stderr,
                            "[render-timing] texture_cache hits=%llu submit_reuse=%llu misses=%llu "
                            "watch_reuse=%llu watch_dirty=%llu watch_unknown=%llu watch_disabled=%llu "
                            "invalid=%llu "
                            "validations=%llu %.1f GiB entries=%zu %.1f MiB\n",
                            (unsigned long long)totals.persistent_hits,
                            (unsigned long long)totals.persistent_submit_reuses,
                            (unsigned long long)totals.persistent_misses,
                            (unsigned long long)totals.persistent_watch_reuses,
                            (unsigned long long)totals.persistent_watch_dirty,
                            (unsigned long long)totals.persistent_watch_unknown,
                            (unsigned long long)totals.persistent_watch_disabled,
                            (unsigned long long)totals.persistent_invalidations,
                            (unsigned long long)totals.persistent_validations,
                            totals.persistent_validation_bytes / (1024.0 * 1024.0 * 1024.0),
                            persistent_decoded_textures.size(),
                            persistent_decoded_texture_bytes / (1024.0 * 1024.0));
                    size_t rtt_bytes = 0;
                    for (const auto& [addr, surface] : g_rtt) {
                        (void)addr;
                        if (surface.rgba) rtt_bytes += surface.rgba->size();
                    }
                    size_t scratch_bytes = 0;
                    for (const auto& scratch : texstore)
                        scratch_bytes += scratch.capacity();
                    fprintf(stderr,
                            "[render-timing] host_cache rtt=%zu %.1f MiB decode_scratch=%zu %.1f MiB "
                            "validation=%.1f MiB\n",
                            g_rtt.size(), rtt_bytes / (1024.0 * 1024.0),
                            texstore.size(), scratch_bytes / (1024.0 * 1024.0),
                            persistent_validation_scratch.capacity() / (1024.0 * 1024.0));
                    const auto write_watch = prosper::host::guest_write_watch_stats();
                    fprintf(stderr,
                            "[render-timing] write_watch create=%llu ok=%llu pages=%llu no_map=%llu "
                            "alias=%llu protect=%llu query=%llu unchanged=%llu dirty=%llu "
                            "unknown=%llu faults=%llu physical=%llu rearms=%llu\n",
                            (unsigned long long)write_watch.create_attempts,
                            (unsigned long long)write_watch.registrations,
                            (unsigned long long)write_watch.registered_pages,
                            (unsigned long long)write_watch.create_no_mapping,
                            (unsigned long long)write_watch.create_incomplete_aliases,
                            (unsigned long long)write_watch.create_protect_failures,
                            (unsigned long long)write_watch.queries,
                            (unsigned long long)write_watch.unchanged,
                            (unsigned long long)write_watch.dirty,
                            (unsigned long long)write_watch.unknown,
                            (unsigned long long)write_watch.faults,
                            (unsigned long long)write_watch.physical_writes,
                            (unsigned long long)write_watch.rearms);
                    const double wn = static_cast<double>(window.submits);
                    const double window_other = window.total_ms - window.build_resources_ms -
                                                window.backend_ms - window.output_copy_ms;
                    fprintf(stderr,
                            "[render-window] frontend submits=%llu callbacks=%.1f avg_ms: total=%.2f "
                            "build_resources=%.2f backend=%.2f output_copy=%.2f other=%.2f; "
                            "resources textures=%.1f "
                            "reused=%.1f %.1f MiB %.2f ms buffers=%.1f %.1f MiB %.2f ms\n",
                            (unsigned long long)window.submits, window.callbacks / wn,
                            window.total_ms / wn, window.build_resources_ms / wn,
                            window.backend_ms / wn, window.output_copy_ms / wn, window_other / wn,
                            window.textures / wn,
                            window.texture_reuses / wn,
                            window.texture_bytes / (wn * 1024.0 * 1024.0), window.texture_ms / wn,
                            window.buffers / wn, window.buffer_bytes / (wn * 1024.0 * 1024.0),
                            window.buffer_ms / wn);
                    const double window_backend_detail_ms = window.backend_target_ms +
                        window.backend_draw_setup_ms + window.backend_record_upload_ms +
                        window.backend_gpu_wait_ms + window.backend_readback_ms +
                        window.backend_cleanup_ms;
                    fprintf(stderr,
                            "[render-window] backend-submit calls=%.2f draws=%.1f avg_ms: measured=%.2f "
                            "detail=%.2f target=%.2f draw_setup=%.2f record_upload=%.2f "
                            "gpu_wait=%.2f readback=%.2f cleanup=%.2f other=%.2f\n",
                            window.backend_calls / wn, window.backend_draws / wn,
                            window.backend_ms / wn, window_backend_detail_ms / wn,
                            window.backend_target_ms / wn, window.backend_draw_setup_ms / wn,
                            window.backend_record_upload_ms / wn, window.backend_gpu_wait_ms / wn,
                            window.backend_readback_ms / wn, window.backend_cleanup_ms / wn,
                            (window.backend_ms - window_backend_detail_ms) / wn);
                    fprintf(stderr,
                            "[render-window] backend-submit draw_setup avg_ms: shaders=%.2f fixed=%.2f "
                            "resources=%.2f pipeline=%.2f\n",
                            window.backend_setup_shader_ms / wn,
                            window.backend_setup_fixed_ms / wn,
                            window.backend_setup_resources_ms / wn,
                            window.backend_setup_pipeline_ms / wn);
                    fprintf(stderr,
                            "[render-window] backend-submit pipelines refs=%.1f hits=%.1f misses=%.1f "
                            "bypass=%.1f entries=%llu evictions=%.1f\n",
                            window.backend_pipeline_refs / wn,
                            window.backend_pipeline_hits / wn,
                            window.backend_pipeline_misses / wn,
                            window.backend_pipeline_bypasses / wn,
                            (unsigned long long)window.backend_pipeline_entries,
                            window.backend_pipeline_evictions / wn);
                    fprintf(stderr,
                            "[render-window] color_targets writes=%.1f load_hits=%.1f sample_hits=%.1f "
                            "readbacks=%.1f deferred=%.1f cached=%llu %.1f MiB\n",
                            window.color_target_writes / wn,
                            window.color_target_write_hits / wn,
                            window.color_target_sample_hits / wn,
                            window.color_target_readbacks / wn,
                            (window.color_target_writes - window.color_target_readbacks) / wn,
                            (unsigned long long)window.color_target_cached_entries,
                            window.color_target_cached_bytes / (1024.0 * 1024.0));
                    fprintf(stderr,
                            "[render-window] texture_cache hits=%.1f submit_reuse=%.1f misses=%.1f "
                            "watch_reuse=%.1f watch_dirty=%.1f watch_unknown=%.1f watch_disabled=%.1f "
                            "invalid=%.1f "
                            "validations=%.1f %.1f MiB\n",
                            window.persistent_hits / wn, window.persistent_submit_reuses / wn,
                            window.persistent_misses / wn, window.persistent_watch_reuses / wn,
                            window.persistent_watch_dirty / wn, window.persistent_watch_unknown / wn,
                            window.persistent_watch_disabled / wn,
                            window.persistent_invalidations / wn,
                            window.persistent_validations / wn,
                            window.persistent_validation_bytes / (wn * 1024.0 * 1024.0));
                    window = {};
                }
            }
            return prosper::gpu::RenderedFrame(std::move(selected_pixels));
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend

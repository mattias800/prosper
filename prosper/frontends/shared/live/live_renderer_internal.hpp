#pragma once

// Lifted out of live_renderer.cpp's anonymous namespaces so the code that operates on them
// can live in its own translation units. Declared in prosper/frontends/shared/live; who may include it is
// a decision this banner does not make -- say so with --note if it is restricted.

// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "shared/live/live_renderer.hpp"
#include "diagnostics/env_cache.hpp"   // PROSPER_ENV_ON / _VALUE: cached reads on per-draw paths
#include "diagnostics/env_numeric.hpp" // #3253: a typo must not select a different setting
#include "gpu/resources/metadata_kind_correlation.hpp"  // positive metadata-kind correlation (pure, tested)
#include "diagnostics/watch_list.hpp"   // strict 0x-only watch parsing
#include "gpu/diagnostics/draw_program_skip.hpp"          // PROSPER_SKIP_DRAW_PROGRAM / census
#include "gpu/diagnostics/pass_break_census.hpp"         // why a pass stopped accepting draws
#include "gpu/diagnostics/link_list_census.hpp"          // PROSPER_DRAW_LINKSCAN
#include "gpu/capture/writer_provenance.hpp"              // who last wrote a censused range
#include "shared/rtt/rtt_authority.hpp"
#include "shared/rtt/rtt_injection.hpp"
#include "shared/rtt/rtt_scale.hpp"
#include "shared/rtt/mrt_extent.hpp"
#include "shared/rtt/mrt_binding.hpp"               // which MRT slot may join a pass (measured extents only)
#include "shared/present/readback_policy.hpp"
#include "shared/present/selected_source_identity.hpp"
#include "shared/diagnostics/capture_renderer_policy.hpp"
#include "shared/texture/write_watch_policy.hpp"
#include "shared/texture/validation_census.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/buffer_source_gate.hpp"
#include "shared/live/guest_source_read.hpp"
#include "shared/live/resolve_submission_policy.hpp"
#include "shared/live/texture_source_snapshot.hpp"
#include "shared/live/depth_cube_source_snapshot.hpp"
#include "shared/live/depth_cube_quantize.hpp"
#include "shared/live/decode_scratch.hpp"     // pooled full-surface decode intermediates
#include "shared/live/live_target_format.hpp"       // the one LiveTargetPixelFormat mapping (exhaustive)
#include "shared/perf/performance_capture.hpp"      // bounded F8 post-trigger renderer timing
#include "shared/present/present_handoff_trace.hpp"
#include "shared/perf/performance_timing_gate.hpp"  // turn on render_runner's existing backend clocks
#include "shared/perf/performance_timing_policy.hpp" // retain timing across split semantic submits

#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/execute/gpu_execute.hpp"          // DrawItem, set_submit_renderer
#include "gpu/timeline/gpu_timeline.hpp"         // phase-gated detailed-capture policy
#include "gpu/capture/writer_provenance.hpp"
#include "gpu/capture/gpu_capture.hpp"          // temporal RTT capture/replay seeds
#include "gpu/texture/guest_texture_layout.hpp" // exact pitch for HLE-produced guest textures
#include "gpu/texture/tile.hpp"                 // detile_surface / tiled_surface_bytes / detile_elements
#include "gpu/texture/bc_decode.hpp"            // BC1/2/3 block decompression -> RGBA8 (#121)
#include "gpu/resources/mip_chain_plan.hpp"      // native BCn guest mip chains (#3873)
#include "gpu/resources/shader_resources.hpp"     // ShaderResourceTable / ResourceClass
#include "gpu/recompiler/rdna2_to_spirv.hpp"       // recompile_fragment (diagnostic solid-color PS)
#include "gpu/present/videoout_present.hpp"     // present_front_index (flip-anchored present selection)
#include "shared/present/present_blit.hpp"             // GPU scanout handoff (#1270 unified-device present)
#include "shared/present/present_blit_policy.hpp"      // flip-anchored scanout publication policy
#include "shared/present/present_extent.hpp"           // the publish extent contract with the caller (#1986)
#include "shared/media/avplayer_plane_policy.hpp"    // which sampled resource is AvPlayer's NV12 chroma plane
#include "shared/live/texture_reference_census.hpp"  // PROSPER_TEXREF_CENSUS (#3873)
#include "shared/live/submit_renderer/callback_types.hpp" // the callback's own types (#3892)
#include "shared/live/submit_renderer/guest_reads.hpp"    // safe_span / safe_copy / safe_equal
#include "shared/rtt/volume_target_index.hpp"        // volume-footprint candidates (#3873)
#include "shared/present/guest_scanout_present.hpp"    // publishing the guest's own flipped buffer (#1968)
#include "shared/diagnostics/diagnostic_window.hpp"        // census window by callback ordinal or by elapsed time
#include "shared/diagnostics/persistent_readback_filter.hpp" // bounded retained-target readback
#include "diagnostics/diag_ratelimit.hpp"   // ordinal + sparse tail for capped diagnostics
#include "host/memory/guest_write_watch.hpp"
#include "fixtures/render_runner.h"              // offscreen Vulkan backend (render_draws_rgba) + dump_bmp
#include "diagnostics/perf/perf_ledger.hpp"       // #3891: always-on alarm ledger
#include "fixtures/retained_depth_array_gpu.h"   // GPU-resident retained depth-array snapshots
#include "fixtures/retained_depth_cube_gpu.h"    // GPU-resident retained depth-cube snapshots

#include <atomic>
#include <cerrno>
#include <functional>
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
#include <utility>
#include <tuple>
#include <thread>
#include <map>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// The guest-flip publish bridge. Core reaches the frontend through a REGISTERED pointer, never by
// naming this symbol: prosper_core links into tools that have no frontend at all, and on Mach-O a
// weak declaration with no definition is still an undefined symbol (`nid_census` failed to link on
// the macOS x86_64 job for exactly that reason, while ELF accepted it).
extern "C" void prosper_vo_set_flip_publish_hook(void (*fn)(uint64_t));
extern "C" void prosper_frontend_flip_publish_guest_scanout(uint64_t flip);

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);
// VideoOut scanout registry (hle_graphics.cpp) — which guest buffer the game most recently FLIPPED
// to screen. The flip fires during the Dcb fold (agc_dcb_set_flip -> prosper_vo_flip_from_gpu),
// BEFORE the submit's execute_and_present, so at render time these identify this frame's scanout VA.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint64_t prosper_vo_buffer_addr(int i);
extern "C" uint64_t prosper_vo_flip_count();

namespace prosper::frontend {

// Only live realization has the full CB allocation/array programming. A Vulkan extent or an
// old capture's tiling-presence bit cannot replace this nonserialized, code-derived certificate.
inline uint64_t raw_snapshot_color_footprint(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    return slot < prosper::gpu::kColorTargetCount
        ? draw.color_targets[slot].raw_snapshot_footprint_bytes : 0u;
}

// Default ceiling on a single non-texture (vertex/index/storage/constant) buffer upload. This is
// not borrowed from any other path — it exists only to bound a corrupt descriptor, and it is sized
// against what a 64 MiB read already costs elsewhere (~16K guest_readable page probes). A guest
// descriptor legitimately declares multi-megabyte vertex streams, and anything clamped away reads
// as zeros in the shader and collapses geometry, so the bound stays far above real content and any
// short upload is reported (#1427). PROSPER_MAX_BUFFER_UPLOAD_MB can lower it for an A/B.
constexpr uint32_t kMaxBufferUploadBytes = 64u << 20;

struct RttSurf {
    std::shared_ptr<const std::vector<uint8_t>> rgba;
    // Uniform DCC fast-clears remain compact until a consumer genuinely needs CPU bytes. Graphics
    // sampling and attachment LOADs can realize the value directly on the GPU.
    bool has_uniform_color = false;
    std::array<float, 4> uniform_color{};
    uint32_t w = 0, h = 0;
    uint32_t volume_depth = 0; // zero is a 2D target; never infer volume from guest address
    // Guest MSAA sample count (CB_COLORn_ATTRIB.NUM_SAMPLES); the renderer image is single-sample,
    // but the guest surface is `samples` times larger (#3906 depth-layer overlap footprint).
    uint32_t samples = 1;
    // Outstanding guest footprint of a renderer-produced volume. A later 2D alias may have
    // current 2D pixels, but cannot make the other volume slices valid guest bytes.
    uint64_t volume_guest_bytes = 0;
    bool volume_footprint_proven = false;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    // Raw guest CB_COLOR format before backend_color_format() canonicalizes the Vulkan attachment.
    // Consumers need this to compose their T# DST_SEL with the host image's component order.
    VkFormat guest_format = VK_FORMAT_UNDEFINED;
    bool gpu_valid = false;
    // A color target can be cleared by a compute write to its DCC metadata rather than by a
    // color-plane write. Remember the sampled descriptor's metadata range so that write can
    // invalidate the retained CPU/GPU target just like a write to the color plane itself.
    uint64_t dcc_metadata_addr = 0;
    uint64_t dcc_metadata_bytes = 0;
    bool dcc_metadata_dirty = false;
    // How that descriptor reads the metadata's clear codes (gfx10_dcc_fast_clear_rgba8's inputs)
    // and the guest extent it described, kept so a pass that RENDERS to the cleared target can
    // decode the clear without a descriptor, and only for the surface the descriptor was of.
    uint32_t dcc_num_components = 0;
    bool dcc_alpha_is_on_msb = false;
    uint32_t dcc_width = 0, dcc_height = 0;
    prosper::test::BackendGuestProducerOrigins guest_origins;
    prosper::test::BackendGuestProducerOrigins dcc_guest_origins;
};
// What a retained colour target keeps from a descriptor that samples it with DCC enabled: where
// its metadata is, and how the descriptor reads a clear code. `metadata_bytes` is that
// descriptor's gpu_capture_dcc_metadata_footprint.
inline void note_rtt_dcc_descriptor(RttSurf& surface, const prosper::gpu::ShaderResource& resource,
                                    uint64_t metadata_bytes) {
    surface.dcc_metadata_addr = resource.metadata_addr;
    surface.dcc_metadata_bytes = metadata_bytes;
    surface.dcc_num_components = resource.num_components;
    surface.dcc_alpha_is_on_msb = resource.alpha_is_on_msb;
    surface.dcc_width = resource.width;
    surface.dcc_height = resource.height;
    surface.dcc_guest_origins.observe(resource.metadata_addr, metadata_bytes);
}
// PROSPER_RENDER_SCALE as the renderer reads it: a positive integer, anything else 1.
inline uint32_t configured_render_scale() {
    static const uint32_t scale = [] {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): one cached read of a process-start setting
        const char* value = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
        const long parsed = value ? std::strtol(value, nullptr, 10) : 1;
        return parsed > 0 ? static_cast<uint32_t>(parsed) : 1u;
    }();
    return scale;
}
// Keeps the per-resource overlap scan entirely off the ordinary 2D-only execution path.
inline bool g_ever_volume_target = false;
// Keys of RTT-cache entries that may carry a nonzero volume_guest_bytes. A superset: every write
// that can make a footprint nonzero goes through settle_volume_guest_footprint() or notes the key
// itself (the resolve copy below). See volume_target_index.hpp for why queries stay exact.
inline prosper::frontend::VolumeTargetIndex g_volume_targets;

inline bool unpublished_volume_may_overlap(uint64_t volume_base, uint64_t volume_bytes,
                                    uint64_t address, uint64_t bytes) {
    if (!volume_base || !volume_bytes || !address || !bytes) return false;
    if (prosper::frontend::live_rtt_ranges_overlap(
            volume_base, volume_bytes, address, bytes)) return true;
    // Direct-memory aliases can have disjoint VAs. Only a proven physical disjoint result
    // permits consuming ordinary guest bytes while renderer-only volume slices exist.
    return ::prosper::guest_memory_topology_relation(
        volume_base, volume_bytes, address, bytes) !=
        ::prosper::GuestMemoryTopologyRelation::Disjoint;
}

inline bool materialize_uniform_rtt(RttSurf& surface) {
    if (surface.volume_depth || !surface.has_uniform_color ||
        !surface.w || !surface.h) return false;
    const VkFormat format = prosper::test::backend_color_format(surface.format);
    const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(format);
    const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
    if (!bpp || texels > SIZE_MAX / bpp) return false;
    std::vector<uint8_t> pixels(static_cast<size_t>(texels) * bpp);
    if (format == VK_FORMAT_R8G8B8A8_UNORM) {
        uint8_t native[4];
        for (uint32_t channel = 0; channel < 4; ++channel)
            native[channel] = static_cast<uint8_t>(std::clamp(
                surface.uniform_color[channel], 0.0f, 1.0f) * 255.0f + 0.5f);
        prosper::frontend::fill_repeating_pixel(pixels, native, sizeof(native));
    } else if (format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        uint16_t native[4];
        for (uint32_t channel = 0; channel < 4; ++channel)
            native[channel] = prosper::gpu::float_to_half(surface.uniform_color[channel]);
        prosper::frontend::fill_repeating_pixel(
            pixels, reinterpret_cast<const uint8_t*>(native), sizeof(native));
    } else if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        const uint32_t native =
            static_cast<uint32_t>(prosper::gpu::float_to_f11(surface.uniform_color[0])) |
            (static_cast<uint32_t>(prosper::gpu::float_to_f11(surface.uniform_color[1])) << 11) |
            (static_cast<uint32_t>(prosper::gpu::float_to_f10(surface.uniform_color[2])) << 22);
        prosper::frontend::fill_repeating_pixel(
            pixels, reinterpret_cast<const uint8_t*>(&native), sizeof(native));
    } else {
        return false;
    }
    surface.rgba = std::make_shared<const std::vector<uint8_t>>(std::move(pixels));
    return true;
}

using RttCache = std::unordered_map<uint64_t, RttSurf>;

// "Does any renderer-only volume footprint satisfy `visit`?" -- asked per sampled texture
// reference, per compute unpublished-volume query and per live byte-range read. The indexed walk
// visits only volume candidates; PROSPER_NO_VOLUME_RTT_INDEX=1 restores the full cache walk on the
// same binary. PROSPER_AUDIT_VOLUME_RTT_INDEX=1 additionally walks the whole cache on every query and
// reports (loudly, unconditionally) any volume entry the index does not hold and any disagreement --
// the superset invariant, checked directly rather than inferred from matching answers.
template <class Visit>
bool any_volume_target(RttCache& cache, Visit&& visit) {
    static const bool indexed = PROSPER_ENV_VALUE("PROSPER_NO_VOLUME_RTT_INDEX") == nullptr;
    static const bool audit = PROSPER_ENV_VALUE("PROSPER_AUDIT_VOLUME_RTT_INDEX") != nullptr;
    if (!indexed) return prosper::frontend::any_volume_target_full_scan(cache, visit);
    const bool answer = g_volume_targets.any_of(cache, visit);
    if (audit) {
        static uint64_t audits = 0, missing = 0, disagreements = 0;
        ++audits;
        for (const auto& [base, surface] : cache) {
            if (surface.volume_guest_bytes && !g_volume_targets.contains(base)) {
                if (++missing <= 16)
                    fprintf(stderr, "[volume-index] *** AUDIT: volume entry 0x%llx (%llu bytes) "
                                    "is not indexed\n", (unsigned long long)base,
                            (unsigned long long)surface.volume_guest_bytes);
            }
        }
        if (prosper::frontend::any_volume_target_full_scan(cache, visit) != answer &&
            ++disagreements <= 16)
            fprintf(stderr, "[volume-index] *** AUDIT: indexed answer %d disagrees with the full "
                            "walk\n", answer ? 1 : 0);
        if ((audits & (audits - 1)) == 0 && audits >= 1024)
            fprintf(stderr, "[volume-index] audit: %llu queries, %llu unindexed volume entries, "
                            "%llu disagreements, %zu candidates, cache=%zu\n",
                    (unsigned long long)audits, (unsigned long long)missing,
                    (unsigned long long)disagreements, g_volume_targets.candidates(), cache.size());
    }
    return answer;
}

// A failed array binding can recur on every draw. Bound routine diagnostics while preserving
// the first rejection at each site; the opt-in full log remains available for investigation.
inline bool log_array_rejection(uint32_t& count, const char* site) {
    // Tests arm this between submits; a process-lifetime cache would make the arm vacuous.
    const bool full_log = std::getenv("PROSPER_ARRAY_REJECT_LOG_ALL") != nullptr;
    if (full_log) return true;
    constexpr uint32_t Limit = 64;
    if (count < Limit) { ++count; return true; }
    if (count == Limit) {
        ++count;
        std::fprintf(stderr, "[render-array-reject] site=%s suppressed after %u messages; "
            "set PROSPER_ARRAY_REJECT_LOG_ALL=1 for full logging\n", site, Limit);
    }
    return false;
}

// A queued guest write carries WHO made it. The origin is a thread-local set around
// notify_guest_gpu_write, but DS/RTT invalidation runs later at drain time -- on whatever thread
// drains, long after that thread-local was reset. So an invalidation diagnostic that reads the
// thread-local reports a constant: measured on a routed GTA V boot, `[ds] invalidate` printed
// `origin=gpu` for 30,458 of 30,458 invalidations, including ones made by writers that DO tag
// themselves. The field looked like attribution and was structurally incapable of it.
//
// Capturing the origin at QUEUE time is what makes it real, and it is why the origin must live in
// the record rather than be re-read at the far end of the seam.
struct PendingGuestGpuWrite {
    uint64_t addr = 0;
    uint64_t size = 0;
    const char* origin = "unknown";   // string literal or static storage; never an owned buffer
};

struct PendingGuestGpuWrites {
    std::mutex mutex;
    std::vector<PendingGuestGpuWrite> ranges;
    bool overflowed = false;
};

// Per-thread cumulative work counts. No guest reads, per-call clocks, shared counter locks or
// execution-policy changes: this distinguishes empty queue drains from actual cache traversal.
// The recurring report interval is not a collection cap. Enable with exactly "1" at startup.
struct GuestWriteDrainCensus {
    uint64_t calls = 0, empty = 0, overflows = 0, ranges = 0, max_ranges = 0, ignored = 0;
    uint64_t ds_visits = 0, ds_invalidations = 0, color_visits = 0, color_overlaps = 0;
    uint64_t rtt_visits = 0, rtt_erases = 0, rtt_dcc = 0, dcc_color_visits = 0;
    uint64_t rtt_footprints = 0, prepared_drains = 0, prepared_record_visits = 0;

    void report() const {
        if (calls > 4 && calls % 4096) return;
        static thread_local const size_t thread = std::hash<std::thread::id>{}(
            std::this_thread::get_id());
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        fprintf(stderr, "[guest-write-drain] thread=%zu steady_ns=%lld calls=%llu empty=%llu "
                "overflows=%llu ranges=%llu max_ranges=%llu ignored=%llu ds_visits=%llu "
                "ds_invalidations=%llu color_visits=%llu color_overlaps=%llu rtt_visits=%llu "
                "rtt_erases=%llu rtt_dcc=%llu dcc_color_visits=%llu rtt_footprints=%llu "
                "prepared_drains=%llu prepared_record_visits=%llu\n",
                thread, (long long)ns, (unsigned long long)calls, (unsigned long long)empty,
                (unsigned long long)overflows, (unsigned long long)ranges,
                (unsigned long long)max_ranges, (unsigned long long)ignored,
                (unsigned long long)ds_visits, (unsigned long long)ds_invalidations,
                (unsigned long long)color_visits, (unsigned long long)color_overlaps,
                (unsigned long long)rtt_visits, (unsigned long long)rtt_erases,
                (unsigned long long)rtt_dcc, (unsigned long long)dcc_color_visits,
                (unsigned long long)rtt_footprints, (unsigned long long)prepared_drains,
                (unsigned long long)prepared_record_visits);
    }
};

inline GuestWriteDrainCensus* guest_write_drain_census() {
    static const bool enabled = [] {
        const char* value = std::getenv("PROSPER_WRITE_DRAIN_CENSUS");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return nullptr;
    static thread_local GuestWriteDrainCensus census;
    return &census;
}

inline PendingGuestGpuWrites& pending_guest_gpu_writes() {
    static PendingGuestGpuWrites pending;
    return pending;
}

// Does this draw bind `addr` as ANY active colour target?
//
// The direct-live sampling decision excluded `draw.color0_base` alone. That was complete only while
// slots above 1 could not be persistent; they can now, so sampling an ACTIVE MRT2+ target would
// otherwise borrow the very image the draw is writing -- one VkImage used as shader-read and colour
// attachment at once, bypassing the CPU fallback that exists for exactly this case.
//
// "Active" matches the pass-grouping definition: a bound base whose write mask is non-zero. Slots 0
// and 1 keep their named-field fallbacks, since a draw may carry either form.
// The backend's notion of a defined colour format, supplied to the shared policy so that header
// stays backend-free. One mapping, used by grouping and feedback alike.
inline bool mrt_format_defined(uint32_t raw) {
    return prosper::test::backend_color_format(static_cast<VkFormat>(raw)) != VK_FORMAT_UNDEFINED;
}

inline bool draw_binds_color_target(const prosper::gpu::DrawItem& draw, uint64_t addr,
                             uint32_t sampled_width = 0u, uint32_t sampled_height = 0u) {
    return prosper::frontend::mrt_draw_binds_target_view(
        draw, addr, sampled_width, sampled_height, mrt_format_defined);
}

// PROSPER_RTT_INVALIDATE_WATCH=<0xaddr>[,<0xaddr>…] — why a renderer-owned surface's CPU snapshot is,
// or is not, being invalidated.
//
// The question it answers is the one left when a composite input is empty and the renderer serves its
// own snapshot for it. `PROSPER_RTT_GUESTPEEK` cannot answer it: every renderer-owned surface reports
// 0% non-zero GUEST bytes whether it renders correctly or not, because the renderer owns the pixels.
// So "the guest memory is empty" is not evidence about production, and the live question becomes
// whether the cached copy is ever refreshed.
//
// It reports both directions, which is the point. A write that OVERLAPS prints its effect and its
// origin; a write that misses prints nothing, but the periodic line states how many writes were
// examined and how many hit — so "never invalidated" is distinguishable from "the instrument never
// ran", and a watched address absent from the cache says so explicitly rather than looking like a
// silent zero.
//
// Addresses are parsed by the shared STRICT parser (`watch_list.hpp`), which requires an explicit `0x`,
// full token consumption, no overflow and no zero address, and arms NOTHING on a malformed spec. An
// earlier revision of this comment claimed `strtoull(..., 0)` was deliberate "so `0x` is required" —
// that was simply false, since base 0 falls back to DECIMAL, and the comment made the bug read as
// checked. The same trap is recorded for PROSPER_TARGET_WATCH, which now uses the same parser.
struct RttInvalidateWatch {
    std::vector<uint64_t> addrs;
    std::atomic<uint64_t> writes_examined{0};
    std::atomic<uint64_t> writes_hitting{0};
};

// The counters are atomics, so the watch is populated IN PLACE rather than returned from an
// initializer lambda -- an atomic is neither copyable nor movable.
inline std::vector<uint64_t> parse_rtt_invalidate_watch_addrs() {
    std::vector<uint64_t> addrs;
    const char* spec = getenv("PROSPER_RTT_INVALIDATE_WATCH");
    if (!spec || !*spec) return addrs;
    // Strict: an explicit 0x prefix, full consumption, no overflow, no zero address. A rejected spec
    // arms NOTHING and says so, rather than watching a decimal-parsed address and reporting success.
    if (!prosper::gpu::parse_hex_watch_list(spec, addrs)) {
        fprintf(stderr,
                "[rtt-inval] ignoring malformed PROSPER_RTT_INVALIDATE_WATCH=\"%s\" "
                "(expected 0x-prefixed hex addresses, comma separated) -- NOT armed\n", spec);
        addrs.clear();
        return addrs;
    }
    fprintf(stderr, "[rtt-inval] watching %zu address(es)\n", addrs.size());
    return addrs;
}

inline RttInvalidateWatch& rtt_invalidate_watch() {
    static RttInvalidateWatch watch;
    static const bool once = [] {
        watch.addrs = parse_rtt_invalidate_watch_addrs();
        return true;
    }();
    (void)once;
    return watch;
}

inline const char* rtt_guest_write_effect_name(prosper::frontend::LiveRttGuestWriteEffect effect) {
    switch (effect) {
        case prosper::frontend::LiveRttGuestWriteEffect::color_plane:   return "color-plane";
        case prosper::frontend::LiveRttGuestWriteEffect::dcc_metadata:  return "dcc-metadata";
        default:                                                       return "none";
    }
}

// A footprint is valid only for this drain. Its operations can revoke authority or erase entries,
// but cannot change their extent/metadata or insert new entries. Producers append to the queue;
// they do not mutate the renderer cache. The existing serialized-renderer contract still applies.
struct PreparedRttGuestWrite {
    RttCache::iterator target;
    uint64_t address, bytes, metadata_address, metadata_bytes;
};

inline uint64_t cpu_rtt_guest_write_bytes(const RttSurf& surface, GuestWriteDrainCensus* census) {
    if (census) ++census->rtt_footprints;
    const uint64_t current_bytes = prosper::frontend::live_rtt_color_footprint_bytes(
        surface.w, surface.h, surface.volume_depth,
        prosper::test::backend_color_bytes_per_pixel(surface.format));
    return std::max(current_bytes, surface.volume_guest_bytes);
}

inline RttCache::iterator invalidate_cpu_rtt_entry(
    RttCache& cache, RttCache::iterator it,
    prosper::frontend::LiveRttGuestWriteEffect effect, GuestWriteDrainCensus* census,
    bool complete_volume_overwrite = false) {
    if (effect == prosper::frontend::LiveRttGuestWriteEffect::color_plane) {
        if (it->second.volume_guest_bytes && !complete_volume_overwrite) {
            // A partial write or an unproven native tile footprint cannot make untouched
            // renderer-only slices valid guest bytes.
            prosper::test::invalidate_persistent_color_target(it->first);
            it->second.rgba.reset();
            it->second.has_uniform_color = false;
            it->second.gpu_valid = false;
            it->second.dcc_metadata_dirty = true;
            return ++it;
        }
        if (census) ++census->rtt_erases;
        return cache.erase(it);
    } else if (effect == prosper::frontend::LiveRttGuestWriteEffect::dcc_metadata) {
        // Keep the surface identity/extent long enough to materialize a uniform DCC clear from
        // the descriptor in the following graphics span, but never LOAD or sample stale pixels.
        if (census) {
            ++census->rtt_dcc;
            census->dcc_color_visits += prosper::test::persistent_color_target_cache().size();
        }
        prosper::test::invalidate_persistent_color_target(it->first);
        it->second.rgba.reset();
        it->second.has_uniform_color = false;
        it->second.gpu_valid = false;
        it->second.dcc_metadata_dirty = true;
        return ++it;
    } else {
        return ++it;
    }
}

inline void invalidate_cpu_rtt_guest_write(RttCache& cache, uint64_t addr, uint64_t size,
                                  GuestWriteDrainCensus* census = nullptr,
                                  std::vector<PreparedRttGuestWrite>* prepared = nullptr) {
    if (!addr || !size) return;
    if (prepared) {
        if (census) census->prepared_record_visits += prepared->size();
        for (auto& entry : *prepared) {
            if (entry.target == cache.end()) continue;
            const auto effect = prosper::frontend::live_rtt_guest_write_effect(
                entry.address, entry.bytes, entry.metadata_address, entry.metadata_bytes, addr, size);
            const auto actual_effect = entry.target->second.volume_guest_bytes &&
                    unpublished_volume_may_overlap(entry.address, entry.bytes, addr, size)
                ? prosper::frontend::LiveRttGuestWriteEffect::color_plane : effect;
            if (actual_effect == prosper::frontend::LiveRttGuestWriteEffect::none) continue;
            const bool volume_entry = entry.target->second.volume_guest_bytes != 0;
            const bool complete_volume_overwrite = volume_entry &&
                entry.target->second.volume_footprint_proven &&
                prosper::frontend::live_rtt_complete_guest_overwrite(
                    entry.address, entry.bytes, addr, size);
            invalidate_cpu_rtt_entry(cache, entry.target, actual_effect, census,
                                     complete_volume_overwrite);
            // Other unordered-map iterators survive erasure. This one must never be used again.
            if (actual_effect == prosper::frontend::LiveRttGuestWriteEffect::color_plane &&
                (!volume_entry || complete_volume_overwrite))
                entry.target = cache.end();
        }
        return;
    }
    auto& watch = rtt_invalidate_watch();
    if (!watch.addrs.empty()) {
        watch.writes_examined.fetch_add(1, std::memory_order_relaxed);
        bool write_touched_a_watched_surface = false;
        for (const uint64_t wanted : watch.addrs) {
            const auto entry = cache.find(wanted);
            if (entry == cache.end()) {
                // A watched address that is not a cached RTT surface at all. Reported once so an
                // absence of invalidations is not read as "nothing ever wrote it".
                static std::mutex absent_mutex;
                static std::set<uint64_t> absent_reported;
                bool first = false;
                { std::lock_guard<std::mutex> lock(absent_mutex);
                  first = absent_reported.insert(wanted).second; }
                if (first)
                    fprintf(stderr,
                            "[rtt-inval] 0x%llx is NOT in the RTT cache at this drain -- no snapshot "
                            "to invalidate\n", (unsigned long long)wanted);
                continue;
            }
            const RttSurf& surface = entry->second;
            const uint64_t bytes = cpu_rtt_guest_write_bytes(surface, nullptr);
            const auto effect = prosper::frontend::live_rtt_guest_write_effect(
                wanted, bytes, surface.dcc_metadata_addr, surface.dcc_metadata_bytes, addr, size);
            if (effect == prosper::frontend::LiveRttGuestWriteEffect::none) continue;
            // Counted ONCE per queued write, not once per matching address: with several watched
            // addresses a per-address counter can exceed writes_examined, which makes the ratio
            // nonsense in exactly the summary line that exists to make a zero trustworthy.
            if (!write_touched_a_watched_surface) {
                write_touched_a_watched_surface = true;
                watch.writes_hitting.fetch_add(1, std::memory_order_relaxed);
            }
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 64)
                fprintf(stderr,
                        "[rtt-inval] 0x%llx %ux%u snapshot invalidated: effect=%s by write "
                        "addr=0x%llx size=%llu origin=%s\n",
                        (unsigned long long)wanted, surface.w, surface.h,
                        rtt_guest_write_effect_name(effect),
                        (unsigned long long)addr, (unsigned long long)size,
                        prosper::gpu::guest_gpu_write_origin());
        }
        // Periodic totals, so a zero is a measurement rather than a silence.
        const uint64_t examined = watch.writes_examined.load(std::memory_order_relaxed);
        if (examined && (examined & (examined - 1)) == 0 && examined >= 1024)
            fprintf(stderr, "[rtt-inval] examined=%llu writes, %llu touched a watched surface\n",
                    (unsigned long long)examined,
                    (unsigned long long)watch.writes_hitting.load(std::memory_order_relaxed));
    }
    for (auto it = cache.begin(); it != cache.end();) {
        const RttSurf& surface = it->second;
        const auto effect = prosper::frontend::live_rtt_guest_write_effect(
            it->first, cpu_rtt_guest_write_bytes(surface, census),
            surface.dcc_metadata_addr, surface.dcc_metadata_bytes, addr, size);
        const auto actual_effect = surface.volume_guest_bytes &&
                unpublished_volume_may_overlap(
                    it->first, cpu_rtt_guest_write_bytes(surface, nullptr), addr, size)
            ? prosper::frontend::LiveRttGuestWriteEffect::color_plane : effect;
        const bool complete_volume_overwrite = surface.volume_guest_bytes &&
            surface.volume_footprint_proven &&
            prosper::frontend::live_rtt_complete_guest_overwrite(
                it->first, cpu_rtt_guest_write_bytes(surface, nullptr), addr, size);
        it = invalidate_cpu_rtt_entry(cache, it, actual_effect, census,
                                      complete_volume_overwrite);
    }
}

// Bumped whenever a drain observes at least one queued guest GPU write (or an overflow), so a
// callback-scoped cache of guest bytes can tell that memory it read may have changed (#3893).
inline std::atomic<uint64_t>& guest_gpu_write_drain_epoch() {
    static std::atomic<uint64_t> epoch{0};
    return epoch;
}

inline void drain_guest_gpu_writes(RttCache& cache, bool invalidate_ds) {
    auto& pending = pending_guest_gpu_writes();
    std::vector<PendingGuestGpuWrite> ranges;
    bool overflowed = false;
    {
        std::lock_guard<std::mutex> lock(pending.mutex);
        ranges.swap(pending.ranges);
        overflowed = pending.overflowed;
        pending.overflowed = false;
    }
    if (!ranges.empty() || overflowed)
        guest_gpu_write_drain_epoch().fetch_add(1, std::memory_order_relaxed);
    auto* census = guest_write_drain_census();
    if (census) {
        ++census->calls;
        census->empty += ranges.empty() && !overflowed;
        census->overflows += overflowed;
        census->ranges += ranges.size();
        census->max_ranges = std::max<uint64_t>(census->max_ranges, ranges.size());
    }
    if (overflowed) {
        for (auto& [key, target] : prosper::test::persistent_color_target_cache()) {
            (void)key;
            target.valid = false;
            std::fill(target.valid_volume_slices.begin(),
                      target.valid_volume_slices.end(), 0u);
            prosper::test::invalidate_color_producer(target);
        }
        if (invalidate_ds)
            for (auto& [key, image] : prosper::test::persistent_ds_cache()) {
                (void)key;
                image.depth_valid = false;
                image.stencil_valid = false;
            }
        for (auto it = cache.begin(); it != cache.end();) {
            if (!it->second.volume_guest_bytes) {
                it = cache.erase(it);
                continue;
            }
            // The overflow omits the address of one or more writes. A volume's untouched guest
            // slices are still stale, so keep a tombstone until a future complete overwrite or
            // successful GPU producer restores authority.
            it->second.rgba.reset();
            it->second.has_uniform_color = false;
            it->second.gpu_valid = false;
            it->second.dcc_metadata_dirty = true;
            ++it;
        }
        if (census) census->report();
        return;
    }
    std::vector<PreparedRttGuestWrite> prepared;
    static const bool prepare_enabled = !PROSPER_ENV_ON("PROSPER_NO_RTT_WRITE_BATCH");
    // One useful write cannot amortize copying the footprints. Keep the original watch path so
    // per-write watched-address hits, absences and origins retain exactly their previous meaning.
    size_t useful_writes = 0;
    if (prepare_enabled && ranges.size() >= 2 && !cache.empty() &&
        rtt_invalidate_watch().addrs.empty()) {
        for (const auto& write : ranges)
            if (write.addr && write.size && ++useful_writes == 2) break;
    }
    if (useful_writes == 2) {
        prepared.reserve(cache.size());
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            const auto& surface = it->second;
            prepared.push_back({it, it->first, cpu_rtt_guest_write_bytes(surface, census),
                                surface.dcc_metadata_addr, surface.dcc_metadata_bytes});
        }
        if (census) ++census->prepared_drains;
    }
    for (const auto& write : ranges) {
        if (census) {
            if (!write.addr || !write.size) ++census->ignored;
            else {
                if (invalidate_ds) census->ds_visits += prosper::test::persistent_ds_cache().size();
                census->color_visits += prosper::test::persistent_color_target_cache().size();
                census->rtt_visits += cache.size();
            }
        }
        // Restore the writer's own origin for the duration of this range's invalidation, so the
        // diagnostics inside report who actually wrote the bytes rather than the drain thread's
        // default. Cleared afterwards so a queued origin never leaks into an unrelated later write.
        prosper::gpu::set_guest_gpu_write_origin(write.origin);
        if (invalidate_ds) {
            const prosper::test::BackendPersistentResourceGuard guard;
            const auto invalidations =
                prosper::test::invalidate_persistent_ds_guest_write(write.addr, write.size);
            if (census) census->ds_invalidations += invalidations;
        }
        const auto color_overlaps =
            prosper::test::invalidate_persistent_color_target_guest_write(write.addr, write.size);
        if (census) census->color_overlaps += color_overlaps;
        invalidate_cpu_rtt_guest_write(cache, write.addr, write.size, census,
                                      prepared.empty() ? nullptr : &prepared);
        prosper::gpu::set_guest_gpu_write_origin(nullptr);
    }
    if (census) census->report();
}

inline std::vector<uint8_t> inspection_rgba8(const std::vector<uint8_t>& pixels,
                                      uint32_t width, uint32_t height, VkFormat format) {
    const size_t texels = static_cast<size_t>(width) * height;
    // A native BCn texture (#3873) holds blocks, not texels: decode a copy for inspection only.
    if (const uint32_t block = prosper::test::backend_block_compressed_bytes(format)) {
        using prosper::gpu::DataFormat;
        const DataFormat source =
            format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK ? DataFormat::Bc1
            : format == VK_FORMAT_BC2_UNORM_BLOCK ? DataFormat::Bc2
            : format == VK_FORMAT_BC3_UNORM_BLOCK ? DataFormat::Bc3
            : format == VK_FORMAT_BC4_UNORM_BLOCK ? DataFormat::Bc4
            : format == VK_FORMAT_BC5_UNORM_BLOCK ? DataFormat::Bc5
            : format == VK_FORMAT_BC6H_UFLOAT_BLOCK ? DataFormat::Bc6
                                                    : DataFormat::Bc7;
        std::vector<uint8_t> rgba(texels * 4, 0);
        if (pixels.size() == prosper::test::backend_texture_bytes(format, width, height) &&
            block == prosper::gpu::bc_block_bytes(source))
            prosper::gpu::bc_decode_surface(rgba.data(), pixels.data(), pixels.size(),
                                            width, height, source);
        return rgba;
    }
    if (format == VK_FORMAT_R8G8B8A8_UNORM && pixels.size() == texels * 4)
        return pixels;
    if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && pixels.size() == texels * 4) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            uint32_t packed = 0;
            std::memcpy(&packed, pixels.data() + texel * 4, sizeof(packed));
            const float values[3] = {
                prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)),
                prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
                prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)),
            };
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float value = values[channel];
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R8_UNORM && pixels.size() == texels) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            const uint8_t value = pixels[texel];
            rgba[texel * 4 + 0] = value;
            rgba[texel * 4 + 1] = value;
            rgba[texel * 4 + 2] = value;
            rgba[texel * 4 + 3] = value;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R8G8_UNORM && pixels.size() == texels * 2) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            rgba[texel * 4 + 0] = pixels[texel * 2 + 0];
            rgba[texel * 4 + 1] = pixels[texel * 2 + 1];
            rgba[texel * 4 + 2] = 0;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R16G16_SFLOAT && pixels.size() == texels * 4) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            for (uint32_t channel = 0; channel < 2; ++channel) {
                uint16_t half = 0;
                std::memcpy(&half, pixels.data() + texel * 4 + channel * 2, sizeof(half));
                const float value = prosper::gpu::half_to_float(half);
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
            rgba[texel * 4 + 2] = 0;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R16_SFLOAT && pixels.size() == texels * 2) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            uint16_t half = 0;
            std::memcpy(&half, pixels.data() + texel * 2, sizeof(half));
            const float value = prosper::gpu::half_to_float(half);
            const uint8_t visible = !std::isfinite(value) || value <= 0.0f ? 0
                : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            rgba[texel * 4 + 0] = visible;
            rgba[texel * 4 + 1] = visible;
            rgba[texel * 4 + 2] = visible;
            rgba[texel * 4 + 3] = 255;
        }
        return rgba;
    }
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT && pixels.size() == texels * 16) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                float value = 0.0f;
                std::memcpy(&value, pixels.data() + texel * 16 + channel * 4, sizeof(value));
                rgba[texel * 4 + channel] = !std::isfinite(value) || value <= 0.0f ? 0
                    : value >= 1.0f ? 255 : static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
        }
        return rgba;
    }
    // R16G16B16A16_UNORM: 16-bit fixed-point per channel -> 8-bit (high byte). Without this, a 64-bit
    // UNORM color target (a common non-float HDR/deep surface) inspected to black regardless of content,
    // which is misleading for RTT diagnostics.
    if (format == VK_FORMAT_R16G16B16A16_UNORM && pixels.size() == texels * 8) {
        std::vector<uint8_t> rgba(texels * 4);
        for (size_t texel = 0; texel < texels; ++texel)
            for (uint32_t channel = 0; channel < 4; ++channel) {
                uint16_t v = 0;
                std::memcpy(&v, pixels.data() + texel * 8 + channel * 2, sizeof(v));
                rgba[texel * 4 + channel] = static_cast<uint8_t>(v >> 8);
            }
        return rgba;
    }
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

// Pixel decoding is a pure function of these fields for guest-backed textures. The identity map keyed
// by this tuple lives for one SUBMIT (#1691), not one renderer callback. One callback is one graphics
// span, and a submit is split into a new span at every interleaved compute/DMA operation, so a
// span-scoped map re-resolved every identity once per span: Blue Prince interleaves 21-22 dispatches
// through one frame, and its 56 distinct identities were resolved 853 times.
//
// The span split exists precisely because an interleaved operation can rewrite guest texture bytes.
// Retaining an entry past that boundary therefore requires proof that no such write landed on the
// decoded range, which the ordered in-submit journal (guest_gpu_writes_since) supplies directly and
// more precisely than the span boundary did. An entry reused inside its own span keeps the historical
// guarantee unchanged and needs no query. Writable storage-image callbacks still explicitly invalidate
// every overlapping entry after publishing their results, and live RTT / compute-imported inputs are
// excluded from the map entirely because an earlier pass in the same callback can replace their pixels.
struct TextureDecodeKey {
    uint64_t gpu_addr = 0;
    uint64_t host_data = 0;
    uint64_t host_data_size = 0;
    uint32_t size = 0;
    uint64_t source_span_bytes = 0;
    uint32_t cls = 0;
    uint32_t format = 0;
    uint32_t num_components = 0;
    uint32_t width = 0, height = 0, depth = 1;
    uint32_t sample_count = 1;
    uint32_t tile_mode = 0;
    uint32_t linear_row_pitch_bytes = 0;
    uint32_t img_dim = 0;
    uint32_t mip_tail_bytes = 0, mip_tail_x = 0, mip_tail_y = 0;
    uint32_t layer_stride_bytes = 0, layer_mip_offset_bytes = 0;
    uint32_t max_uncompressed_block_size = 0, max_compressed_block_size = 0;
    uint32_t dcc_flags = 0;
    uint64_t metadata_addr = 0;
    uint64_t metadata_host_data = 0;
    uint64_t metadata_host_data_size = 0;
    bool in_mip_tail = false;
    bool preserve_narrow_channels = false;
    // The same guest T# can be consumed through incompatible shader interfaces. In particular,
    // graphics IMAGE_LOAD uses a formatless raw-uvec4 image (16 host bytes per texel), while an
    // image atomic over that descriptor requires exact R32ui (4 host bytes per texel). These are
    // distinct decoded identities even when every guest descriptor field and source span matches.
    uint32_t reflected_image_numeric_class = 0;
    uint32_t reflected_storage_image_format = 0;
    uint32_t reflected_image_dim = 0;
    bool reflected_image_arrayed = false;
    bool reflected_image_multisampled = false;
    bool reflected_image_writable = false;
    // Renderer-owned layers that are newer than the guest/compute backing participate in the
    // decoded identity. This is zero for ordinary textures.
    uint64_t renderer_overlay_version = 0;
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
        mix(key.source_span_bytes);
        mix(key.cls); mix(key.format); mix(key.num_components); mix(key.width); mix(key.height); mix(key.depth);
        mix(key.sample_count);
        mix(key.tile_mode); mix(key.linear_row_pitch_bytes); mix(key.img_dim);
        mix(key.mip_tail_bytes); mix(key.mip_tail_x); mix(key.mip_tail_y);
        mix(key.layer_stride_bytes); mix(key.layer_mip_offset_bytes);
        mix(key.max_uncompressed_block_size);
        mix(key.max_compressed_block_size); mix(key.dcc_flags); mix(key.metadata_addr);
        mix(key.metadata_host_data); mix(key.metadata_host_data_size);
        mix(key.in_mip_tail);
        mix(key.preserve_narrow_channels);
        mix(key.reflected_image_numeric_class);
        mix(key.reflected_storage_image_format);
        mix(key.reflected_image_dim);
        mix(key.reflected_image_arrayed);
        mix(key.reflected_image_multisampled);
        mix(key.reflected_image_writable);
        mix(key.renderer_overlay_version);
        return hash;
    }
};

inline bool avp_chroma_log() {
    static const bool enabled = getenv("PROSPER_AVPCHROMA_LOG") != nullptr;
    return enabled;
}

inline void log_avp_chroma_candidate(const prosper::gpu::ShaderResource& r,
                              uint32_t tw, uint32_t th, const AvpChromaVerdict& v) {
    static std::mutex mx;
    static std::set<uint64_t> seen;
    {
        std::lock_guard<std::mutex> lock(mx);
        const uint64_t identity = r.gpu_addr ^ (static_cast<uint64_t>(tw) << 12) ^
                                  (static_cast<uint64_t>(th) << 34) ^
                                  (static_cast<uint64_t>(r.num_components) << 56);
        if (!seen.insert(identity).second) return;
    }
    // A one-component plane is a luma candidate, never a chroma one: report it as context rather
    // than as a rejected chroma plane, so the pair can be read off one log without confusion.
    const char* verdict = r.num_components == 1u ? "luma-plane-candidate "
                        : v.match               ? "CHROMA "
                                                : "coverage-broadcast ";
    fprintf(stderr,
            "[avpchroma] addr=%llx %ux%u fmt=%u ncomp=%u tile=%u dim=%u depth=%u "
            "lstride=%u lmipoff=%u miptail=%u size=%u swz=%u,%u,%u,%u "
            "row_bytes=%u pitch_field=%u registered=%u resolved=%u luma=%llx -> %s%s\n",
            (unsigned long long)r.gpu_addr, tw, th, static_cast<unsigned>(r.format),
            r.num_components, r.tile_mode, r.img_dim, r.depth,
            r.layer_stride_bytes, r.layer_mip_offset_bytes, r.in_mip_tail ? 1u : 0u, r.size,
            r.swizzle[0], r.swizzle[1], r.swizzle[2], r.swizzle[3],
            v.row_bytes, r.linear_row_pitch_bytes, v.registered_pitch, v.resolved_pitch,
            (unsigned long long)v.sibling_luma_addr,
            verdict, avp_chroma_reason_name(v.reason));
    fflush(stderr);
}

struct DecodedTexture {
    const uint8_t* pixels = nullptr;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t persistent_id = 0;
    uint64_t persistent_version = 0;
    // Everything below exists so the entry can outlive the span that produced it (#1691).
    // `span` is the renderer-callback ordinal that decoded these pixels: reuse inside that same span
    // is the historical contract and needs no proof. Crossing a span boundary requires
    // guest_gpu_writes_since(snapshot, source_addr, source_size) == Unchanged over the exact range
    // the decode read, so an interleaved compute/DMA/EOP write to the backing forces a fresh resolve
    // instead of serving pixels the guest has since replaced (the #780 failure shape).
    // `texstore_slot` pins the scratch slot when the pixels are NOT owned by the persistent cache;
    // without the pin the next span would hand the same slot to an unrelated decode and this entry
    // would silently alias another texture's bytes.
    uint64_t span = 0;
    prosper::gpu::GuestGpuWriteSnapshot snapshot;
    uint64_t source_addr = 0;
    uint64_t source_size = 0;
    size_t texstore_slot = SIZE_MAX;
    // Exact 2D-MSAA resolves use an owned plane-major allocation rather than a reusable texstore
    // slot. Retaining the owner here gives same-submit cache hits the same lifetime guarantee as a
    // pinned scratch slot, without pinning one ~32 MiB vector per distinct surface forever.
    std::shared_ptr<const std::vector<uint8_t>> pixels_owner;
    // Pixel bytes do not define their Vulkan interpretation. Restore these on every hit rather than
    // re-deriving only the historical sampled R8/RG8/FP16 cases and accidentally reverting a
    // portable UINT storage image to the FrameResource default RGBA8_UNORM.
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    bool storage_image_contract_valid = true;
    // Decoded array layers behind `pixels`. 1 for every non-array texture, which is why this can
    // sit last and default: the positional initializers below stay correct without naming it.
    uint32_t layers = 1;
    // Readable bytes behind `pixels`. The backend's multi-layer span contract REQUIRES this -- it
    // refuses to build a layered image without proof that every layer is readable -- and the
    // renderer had exactly one assignment to fr.tex_byte_size in the whole file, on the MSAA path.
    // So every array served from cache arrived claiming zero readable bytes and was rejected.
    size_t pixels_bytes = 0;
    std::shared_ptr<prosper::test::GpuDetileUpload> gpu_detile;
    // Mip levels packed into `pixels` (FrameResource::uploaded_mip_levels; 0 = level 0 only). A
    // native BCn chain's byte count depends on the T#'s level count, which TextureDecodeKey does not
    // hold, so reuse must match it exactly (native_bc_entry_matches) -- #3883 review.
    uint32_t packed_mip_levels = 0;
};

// Always-on identity-scope accounting; see TextureDecodeScopeStats in the header.
inline thread_local TextureDecodeScopeStats g_texture_decode_scope{};

// #325: the decoded-array footprint budget. Read once at file scope rather than inside the decode,
// because it is CONFIGURATION, not a diagnostic gate -- a getenv lexically enclosing a report makes
// that report depend on two switches, which check_diag_gates.py rightly rejects (TWO-GATE).
inline uint64_t array_decode_budget_bytes() {
    static const uint64_t bytes = [] {
        // atoll turned `-1` into a budget of 0xFFFFFFFFFFF00000 -- i.e. UNBOUNDED, the one setting
        // this knob exists to prevent -- and `1gib` into 1 MiB, which reads in the log as a
        // deliberate choice. Refuse both and keep 1 GiB (#3267).
        const char* e = getenv("PROSPER_ARRAY_DECODE_BUDGET_MIB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_ARRAY_DECODE_BUDGET_MIB", e, 1024ull, UINT64_MAX >> 20, "MiB");
        return mib << 20;
    }();
    return bytes;
}

struct PersistentDecodedTexture {
    // Decoded array layers behind `pixels`; 1 for every non-array texture. The persistent cache is
    // the one that actually serves a world-texture atlas -- a submit-local entry is rebuilt from it
    // -- so a layer count that stops here is a layer count the shader never sees (#325).
    uint32_t layers = 1;
    uint64_t source_addr = 0;
    size_t source_size = 0;
    size_t source_prefix_size = 0;
    bool source_matches_pixels = false;
    DepthCubeSourceSnapshot depth_cube_source;
    std::vector<uint8_t> source_prefix;
    std::shared_ptr<const std::vector<uint8_t>> pixels;
    uint32_t output_height = 0;
    bool narrow = false;
    uint64_t last_use = 0;
    uint64_t persistent_id = 0;
    uint64_t persistent_version = 0;
    VkFormat texture_format = VK_FORMAT_R8G8B8A8_UNORM;
    // Levels packed into `pixels`; see DecodedTexture::packed_mip_levels.
    uint32_t packed_mip_levels = 0;
    bool storage_image_contract_valid = true;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    prosper::host::GuestWriteWatch source_watch;
    uint32_t source_watch_dirty_count = 0;
    uint32_t source_watch_stable_validations = 0;
    bool source_watch_only = false;
    bool source_watch_disabled = false;

    size_t bytes() const { return source_prefix.size() + (pixels ? pixels->size() : 0); }
};

// PROSPER_NO_NATIVE_BC=1 restores the CPU decoder for every BCn texture (the A/B switch).
// PROSPER_TESTTEX writes RGBA8 texels over the decoded surface, so it also keeps the decoder.
inline bool native_bc_sampled_enabled() {
    static const bool enabled = PROSPER_ENV_VALUE("PROSPER_NO_NATIVE_BC") == nullptr &&
        PROSPER_ENV_VALUE("PROSPER_TESTTEX") == nullptr;
    return enabled;
}

// The guest mip chain a native BCn texture uploads (#3873), memoized on every descriptor field the
// placement depends on: this runs once per texture REFERENCE (tens of thousands a second), while the
// plan itself walks tiled_mip_level_layout once per level.
inline const prosper::gpu::MipChainPlan& native_bc_mip_chain_plan(const prosper::gpu::ShaderResource& r,
                                                            uint32_t block_bytes) {
    using Key = std::array<uint64_t, 20>;
    thread_local std::map<Key, prosper::gpu::MipChainPlan> memo;
    const Key key{r.width, r.height, static_cast<uint64_t>(r.format), r.tile_mode,
                  r.declared_mip_levels, r.mip_chain_element_width, r.mip_chain_element_height,
                  r.mip_chain_bytes_per_block, r.mip_chain_max_level, r.mip_chain_base_level,
                  r.in_mip_tail ? 1u : 0u, r.mip_tail_x, r.mip_tail_y, r.mip_tail_bytes,
                  r.img_dim, r.depth, r.layer_stride_bytes, r.layer_mip_offset_bytes,
                  (r.compression_enabled ? 1u : 0u) | (r.metadata_addr ? 2u : 0u) |
                      (static_cast<uint64_t>(r.cls) << 2),
                  (static_cast<uint64_t>(r.sample_count) << 8) | block_bytes};
    auto found = memo.find(key);
    if (found != memo.end()) return found->second;
    if (memo.size() > 8192) memo.clear();
    return memo.emplace(key, prosper::gpu::shader_resource_block_mip_chain_plan(r, block_bytes))
        .first->second;
}

struct DiagnosticAddressSelector {
    bool configured = false;
    bool valid = true;
    std::vector<uint64_t> addresses;

    bool includes(uint64_t address) const {
        return !configured ||
            (valid && std::find(addresses.begin(), addresses.end(), address) != addresses.end());
    }
};

inline DiagnosticAddressSelector parse_diagnostic_address_selector(const char* env_name) {
    DiagnosticAddressSelector selector;
    // Each caller stores the result in a function-local static, so this generic parser keeps the
    // same one-read-per-process contract as PROSPER_ENV_VALUE without passing a runtime name into
    // that macro's captureless lambda.
    const char* spec = std::getenv(env_name);
    if (!spec || !*spec) return selector;
    selector.configured = true;
    selector.valid = prosper::gpu::parse_hex_watch_list(spec, selector.addresses);
    if (!selector.valid) {
        std::fprintf(stderr,
                     "[render] ignoring malformed %s=\"%s\" (expected 0x-prefixed hex "
                     "addresses, comma separated) -- selector matches NOTHING\n",
                     env_name, spec);
        selector.addresses.clear();
    } else {
        std::fprintf(stderr, "[render] %s selects %zu address(es)\n",
                     env_name, selector.addresses.size());
    }
    return selector;
}

inline const DiagnosticAddressSelector& native_r11_sampled_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_NATIVE_R11_SAMPLED_ONLY");
    return selector;
}

inline const DiagnosticAddressSelector& renderer_mip_chain_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_RENDERER_MIP_CHAIN_ONLY");
    return selector;
}


// A draw dropped because a texture reference touches a renderer-owned volume the renderer cannot
// serve. Always reported, rate-limited: these drops were silent, and a whole screen of GTA V UI went
// black for days without a single log line saying why (#3842 regression, 2026-09-27).
inline void report_volume_sample_drop(const char* reason, uint64_t addr, uint64_t volume_bytes) {
    static std::atomic<uint64_t> drops{0};
    const uint64_t n = drops.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 8 || (n & (n - 1)) == 0)
        std::fprintf(stderr, "[volume-sample-drop] #%llu reason=%s texture=0x%llx volume-bytes=%llu "
                             "-- the draw sampling it is dropped\n",
                     (unsigned long long)n, reason, (unsigned long long)addr,
                     (unsigned long long)volume_bytes);
}

// ---- appended by a later promotion out of the same source ----
// The one place a PRODUCER OUTCOME settles a renderer-produced volume's claim over its guest
// footprint, by live_rtt_settle_volume_footprint's rule: a claim needs a valid renderer image, and a pass or
// refusal without one releases any earlier claim. Otherwise every later sample of the address is
// refused and its draw dropped, with guest memory never allowed to stand in (#3842, #3889, #3890).
// Not a global invariant: a partial guest write, the pending-write overflow path and the MSAA-resolve
// destination copy keep or copy a claim without an image on purpose (#3842's tombstone).
inline void settle_volume_guest_footprint(uint64_t base, RttSurf& surface, bool renderer_image_valid,
                                   uint64_t bytes, bool proven) {
    const prosper::frontend::LiveRttVolumeFootprint settled =
        prosper::frontend::live_rtt_settle_volume_footprint(
            {surface.volume_guest_bytes, surface.volume_footprint_proven},
            surface.volume_depth != 0, renderer_image_valid, bytes, proven);
    surface.volume_guest_bytes = settled.bytes;
    surface.volume_footprint_proven = settled.proven;
    if (surface.volume_guest_bytes) g_volume_targets.note(base);
}

// A volume producer pass the renderer declined, or whose view or backend admission it refused. The
// renderer holds no image for the attempted version, so the entry records the shape only and guest
// memory stays authoritative for the footprint (settle_volume_guest_footprint).
inline void note_volume_producer_denied(uint64_t base, RttSurf& denied, uint32_t width, uint32_t height,
                                 uint32_t volume_depth, VkFormat format) {
    denied.w = width;
    denied.h = height;
    denied.volume_depth = volume_depth;
    settle_volume_guest_footprint(base, denied, /*renderer_image_valid=*/false, 0, false);
    denied.format = format;
    denied.rgba.reset();
    denied.has_uniform_color = false;
    denied.gpu_valid = false;
}

// #1330: gpu_replay sets PROSPER_PREFIX_INSPECT for its ordered-prefix modes (--draw N:M,
// --draw-steps, --through-operation) before the first render, so a prefix ending on a non-RGBA8
// color pass publishes that surface (inspection-converted) instead of a stale earlier RGBA8 pass.
// Never set outside those diagnostic replays; cached once — the flag is process-lifetime.
inline bool prefix_inspect_publish() {
    static const bool enabled = getenv("PROSPER_PREFIX_INSPECT") != nullptr;
    return enabled;
}

// Milliseconds since the first ARMED census check of this run — the origin for the `ms:` form of
// PROSPER_PASS_LOG / PROSPER_DUMP_PERSISTENT (diagnostic_window.hpp). Lazily started, so an
// unarmed run never reads the clock and the origin is the same first-callback moment
// PROSPER_GPU_CAPTURE_AFTER_MS uses, letting a capture and a census be aimed at one instant.
inline uint64_t diagnostic_elapsed_ms() {
    static const auto start = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count());
}

inline const DiagnosticAddressSelector& rtt_no_seed_target_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_RTT_NOSEED_TARGET");
    return selector;
}

inline const DiagnosticAddressSelector& rtt_residency_trace_selector() {
    static const DiagnosticAddressSelector selector =
        parse_diagnostic_address_selector("PROSPER_RTT_RESIDENCY_TRACE");
    return selector;
}

// ---- appended by a later promotion out of the same source ----
inline bool parse_diagnostic_address(const char* spec, uint64_t& address) {
    if (!spec) return true;
    if (spec[0] != '0' || (spec[1] != 'x' && spec[1] != 'X') || !spec[2]) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(spec, &end, 16);
    if (errno || end == spec + 2 || *end || !parsed) return false;
    address = parsed;
    return true;
}

inline bool parse_diagnostic_extent(const char* spec, uint32_t& width, uint32_t& height) {
    if (!spec) return true;
    if (*spec < '0' || *spec > '9') return false;
    char* separator = nullptr;
    errno = 0;
    const unsigned long w = std::strtoul(spec, &separator, 10);
    if (errno || !separator || *separator != 'x' || !w || w > UINT32_MAX ||
        separator[1] < '0' || separator[1] > '9') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long h = std::strtoul(separator + 1, &end, 10);
    if (errno || !end || *end || !h || h > UINT32_MAX) return false;
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);
    return true;
}

}  // namespace prosper::frontend

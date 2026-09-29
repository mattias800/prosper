// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "shared/live/live_renderer.hpp"
#include "diagnostics/env_cache.hpp"   // PROSPER_ENV_ON / _VALUE: cached reads on per-draw paths
#include "diagnostics/env_numeric.hpp" // #3253: a typo must not select a different setting
#include "gpu/resources/metadata_kind_correlation.hpp"  // positive metadata-kind correlation (pure, tested)
#include "gpu/diagnostics/watch_list.hpp"                 // strict 0x-only watch parsing
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
#include "shared/present/compute_scanout.hpp"          // #3915: GPU present of a compute-written scanout
#include "shared/present/present_extent.hpp"           // the publish extent contract with the caller (#1986)
#include "shared/media/avplayer_plane_policy.hpp"    // which sampled resource is AvPlayer's NV12 chroma plane
#include "shared/live/texture_reference_census.hpp"  // PROSPER_TEXREF_CENSUS (#3873)
#include "shared/live/submit_renderer/callback_types.hpp" // the callback's own types (#3892)
#include "shared/live/submit_renderer/callback_state.hpp" // process prelude owner (#3892)
#include "shared/live/submit_renderer/guest_reads.hpp"    // safe_span / safe_copy / safe_equal
#include "shared/live/submit_renderer/draw_resources.hpp" // build_draw_frame_resources (#3892)
#include "shared/live/submit_renderer/timing_report.hpp"  // report_render_timing_aggregates (#3892)
#include "shared/live/submit_renderer/backend_timing.hpp" // record_backend_timing_stats, print_rtt_timing (#3892)
#include "shared/live/submit_renderer/backend_draws.hpp"  // build_backend_draws, clear_for (#3892)
#include "shared/live/submit_renderer/per_target_passes.hpp" // render_per_target_passes (#3892)
#include "shared/live/submit_renderer/callback_prelude.hpp" // load_shader_overrides, materialize_dirty_dcc_clears (#3892)
#include "shared/live/submit_renderer/callback_diagnostics.hpp" // the callback's own diagnostics (#3892)
#include "shared/live/submit_renderer/single_framebuffer.hpp" // render_single_framebuffer (#3892)
#include "shared/live/submit_renderer/final_span_present.hpp" // select_final_span_present (#3892)
#include "shared/rtt/volume_target_index.hpp"        // volume-footprint candidates (#3873)
#include "shared/present/guest_scanout_present.hpp"    // publishing the guest's own flipped buffer (#1968)
#include "shared/diagnostics/diagnostic_window.hpp"        // census window by callback ordinal or by elapsed time
#include "shared/diagnostics/persistent_readback_filter.hpp" // bounded retained-target readback
#include "gpu/diagnostics/diag_ratelimit.hpp"       // ordinal + sparse tail for capped diagnostics
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

// Classify a guest address: 0 => not within a reserved/committed guest mapping (see hle_kernel_mem).
extern "C" int prosper_reserved_range_state(uint64_t addr);
// VideoOut scanout registry (hle_graphics.cpp) — which guest buffer the game most recently FLIPPED
// to screen. The flip fires during the Dcb fold (agc_dcb_set_flip -> prosper_vo_flip_from_gpu),
// BEFORE the submit's execute_and_present, so at render time these identify this frame's scanout VA.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint64_t prosper_vo_buffer_addr(int i);
extern "C" uint64_t prosper_vo_flip_count();
#include "live_renderer_internal.hpp"

namespace prosper::frontend {


// Render-to-texture surface cache (#167): CB_COLOR0_BASE -> the RGBA pixels we last rendered into it.
// The game renders its scene into a color target then samples that address as a texture in a later
// composite pass. Guest memory at that address is never populated on our (CPU-read) side, so without
// this the composite samples zeros and the frame is black. We cache each submit's rendered pixels under
// its render-target base and inject them when a subsequent draw samples a texture at a matching base.
namespace {

// Installed by the live-renderer registration below; called from the guest flip path through the
// extern "C" entry at the bottom of this file, which that registration hands to core. Empty when
// the live renderer is not registered.

uint64_t host_physical_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0) return 0;
    const uint64_t count = static_cast<uint64_t>(pages);
    const uint64_t bytes = static_cast<uint64_t>(page_bytes);
    return count <= UINT64_MAX / bytes ? count * bytes : UINT64_MAX;
#endif
}
}

namespace submit_renderer {

CallbackState& CallbackState::instance() {
    static CallbackState state;
    return state;
}

const size_t& CallbackState::write_watch_promotion_budget_bytes() {
    static const auto& value = write_watch_promotion_budget_bytes_.emplace([] {
        // 0 is not "off": an empty budget makes WriteWatchPromotionBudget::try_consume
        // return true unconditionally, i.e. unbounded arming per submit. A typo must keep
        // the default rather than select that (#3253).
        const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", value, 8ull,
            SIZE_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<size_t>(mib * (1024ull * 1024ull));
    }());
    return value;
}

std::atomic<int>& CallbackState::g_submit_idx() {
    static auto& value = g_submit_idx_.emplace(0);
    return value;
}

int& CallbackState::g_render_first() {
    static auto& value = g_render_first_.emplace(getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0);
    return value;
}

const int64_t& CallbackState::g_render_delay_ms() {
    static const auto& value = g_render_delay_ms_.emplace(getenv("PROSPER_RENDER_DELAY_MS")
        ? std::max<int64_t>(0, atoll(PROSPER_ENV_VALUE("PROSPER_RENDER_DELAY_MS"))) : 0);
    return value;
}

const std::chrono::steady_clock::time_point& CallbackState::g_render_delay_start() {
    static const auto& value = g_render_delay_start_.emplace(std::chrono::steady_clock::now());
    return value;
}

std::atomic<bool>& CallbackState::g_render_delay_announced() {
    static auto& value = g_render_delay_announced_.emplace(false);
    return value;
}

int& CallbackState::g_render_last() {
    static auto& value = g_render_last_.emplace(getenv("PROSPER_RENDER_LAST") ? atoi(getenv("PROSPER_RENDER_LAST")) : INT_MAX);
    return value;
}

const int& CallbackState::g_rttlog_min_submit() {
    static const auto& value = g_rttlog_min_submit_.emplace(getenv("PROSPER_RTTLOG_MIN_SUBMIT")
        ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MIN_SUBMIT"))) : 0);
    return value;
}

const int& CallbackState::g_rttlog_max_submit() {
    static const auto& value = g_rttlog_max_submit_.emplace(getenv("PROSPER_RTTLOG_MAX_SUBMIT")
        ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MAX_SUBMIT"))) : INT_MAX);
    return value;
}

const bool& CallbackState::validation_census_requested() {
    static const auto& value = validation_census_requested_.emplace(PROSPER_ENV_VALUE("PROSPER_TEXTURE_VALIDATION_CENSUS") != nullptr);
    return value;
}

const uint64_t& CallbackState::rtt_timing_min_draws() {
    static const auto& value = rtt_timing_min_draws_.emplace(getenv("PROSPER_RTT_TIMING_MIN_DRAWS")
        ? strtoull(PROSPER_ENV_VALUE("PROSPER_RTT_TIMING_MIN_DRAWS"), nullptr, 0) : 0);
    return value;
}

const bool& CallbackState::submit_decode_scope_disabled() {
    static const auto& value = submit_decode_scope_disabled_.emplace(PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE") != nullptr ||
        PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") != nullptr);
    return value;
}

const bool& CallbackState::use_tracked_buffer_membership_cache() {
    static const auto& value = use_tracked_buffer_membership_cache_.emplace(PROSPER_ENV_VALUE("PROSPER_NO_TRACKED_BUFFER_GATE") == nullptr);
    return value;
}

std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>& CallbackState::persistent_decoded_textures() {
    static auto& value = persistent_decoded_textures_.emplace();
    return value;
}

size_t& CallbackState::persistent_decoded_texture_bytes() {
    static auto& value = persistent_decoded_texture_bytes_.emplace(0);
    return value;
}

uint64_t& CallbackState::persistent_decode_generation() {
    static auto& value = persistent_decode_generation_.emplace(0);
    return value;
}

size_t& CallbackState::retired_submit_bytes() {
    static auto& value = retired_submit_bytes_.emplace(0);
    return value;
}

uint64_t& CallbackState::persistent_texture_id() {
    static auto& value = persistent_texture_id_.emplace(0);
    return value;
}

std::vector<uint8_t>& CallbackState::persistent_validation_scratch() {
    static auto& value = persistent_validation_scratch_.emplace();
    return value;
}

const size_t& CallbackState::persistent_decode_limit() {
    static const auto& value = persistent_decode_limit_.emplace([] {
        const uint64_t physical_bytes = host_physical_memory_bytes();
        const size_t limit = texture_decode_cache_limit_bytes(
            PROSPER_ENV_VALUE("PROSPER_TEXTURE_DECODE_CACHE_MB"), physical_bytes);
        fprintf(stderr,
                "[render] decoded texture cache budget = %.1f MiB "
                "(host physical %.1f GiB)\n",
                limit / (1024.0 * 1024.0),
                physical_bytes / (1024.0 * 1024.0 * 1024.0));
        return limit;
    }());
    return value;
}

const bool& CallbackState::reserve_frame_resources() {
    static const auto& value = reserve_frame_resources_.emplace([] {
        const char* setting = std::getenv("PROSPER_FRAME_RESOURCE_RESERVE");
        return !setting || std::strcmp(setting, "0") != 0;
    }());
    return value;
}

const bool& CallbackState::disable_guest_depth_layers() {
    static const auto& value = disable_guest_depth_layers_.emplace(std::getenv("PROSPER_NO_GUEST_DEPTH_LAYERS") != nullptr);
    return value;
}

} // namespace submit_renderer

} // namespace prosper::frontend

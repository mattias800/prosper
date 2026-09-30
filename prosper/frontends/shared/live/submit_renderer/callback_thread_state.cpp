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
#include "shared/live/submit_renderer/callback_thread_state.hpp" // calling-thread prelude owner (#3892)
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
}

namespace submit_renderer {

CallbackThreadState& CallbackThreadState::current() {
    static thread_local CallbackThreadState state;
    return state;
}

prosper::frontend::WriteWatchPromotionBudget& CallbackThreadState::write_watch_promotion_budget() {
    static thread_local auto& value = write_watch_promotion_budget_.emplace();
    return value;
}

std::vector<PinnedScanout>& CallbackThreadState::pinned_scanouts() {
    static thread_local auto& value = pinned_scanouts_.emplace();
    return value;
}

std::vector<PinnedRendererMipTarget>& CallbackThreadState::pinned_renderer_mip_targets() {
    static thread_local auto& value = pinned_renderer_mip_targets_.emplace();
    return value;
}

int& CallbackThreadState::g_this_submit() {
    static thread_local auto& value = g_this_submit_.emplace(-1);
    return value;
}

bool& CallbackThreadState::g_force_this_submit() {
    static thread_local auto& value = g_force_this_submit_.emplace(false);
    return value;
}

RenderTiming& CallbackThreadState::pending_timing() {
    static thread_local auto& value = pending_timing_.emplace();
    return value;
}

std::vector<RttTimingRecord>& CallbackThreadState::pending_rtt_timing() {
    static thread_local auto& value = pending_rtt_timing_.emplace();
    return value;
}

uint64_t& CallbackThreadState::pending_span_start_ns() {
    static thread_local auto& value = pending_span_start_ns_.emplace(0);
    return value;
}

uint64_t& CallbackThreadState::pending_capture_generation() {
    static thread_local auto& value = pending_capture_generation_.emplace(0);
    return value;
}

ValidationCensusLog& CallbackThreadState::validation_census_log() {
    static thread_local auto& value = validation_census_log_.emplace();
    return value;
}

std::vector<std::vector<uint8_t>>& CallbackThreadState::texstore() {
    static thread_local auto& value = texstore_.emplace();
    return value;
}

std::vector<bool>& CallbackThreadState::texstore_pinned() {
    static thread_local auto& value = texstore_pinned_.emplace();
    return value;
}

std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& CallbackThreadState::decoded_textures() {
    static thread_local auto& value = decoded_textures_.emplace();
    return value;
}

uint64_t& CallbackThreadState::decode_span_ordinal() {
    static thread_local auto& value = decode_span_ordinal_.emplace(0);
    return value;
}

int& CallbackThreadState::decode_scope_submit() {
    static thread_local auto& value = decode_scope_submit_.emplace(-1);
    return value;
}

std::vector<std::shared_ptr<const std::vector<uint8_t>>>& CallbackThreadState::retired_submit_pixels() {
    static thread_local auto& value = retired_submit_pixels_.emplace();
    return value;
}

std::unordered_map<uint64_t, ReflectMemoEntry>& CallbackThreadState::reflect_memo() {
    static thread_local auto& value = reflect_memo_.emplace();
    return value;
}

} // namespace submit_renderer

} // namespace prosper::frontend

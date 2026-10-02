// live_renderer.cpp — see live_renderer.hpp. Extracted from boot_trace's PROSPER_RENDER lambda
// (behavior-preserving); Vulkan-backed, so this unit links Vulkan::Vulkan.
#include "shared/live/live_renderer.hpp"
#include "diagnostics/readback_refusal.hpp"
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

using namespace submit_renderer;

bool flush_live_graphics_pipeline_cache() {
    return prosper::test::flush_graphics_pipeline_cache();
}

// Render-to-texture surface cache (#167): CB_COLOR0_BASE -> the RGBA pixels we last rendered into it.
// The game renders its scene into a color target then samples that address as a texture in a later
// composite pass. Guest memory at that address is never populated on our (CPU-read) side, so without
// this the composite samples zeros and the frame is black. We cache each submit's rendered pixels under
// its render-target base and inject them when a subsequent draw samples a texture at a matching base.
namespace {

// Installed by the live-renderer registration below; called from the guest flip path through the
// extern "C" entry at the bottom of this file, which that registration hands to core. Empty when
// the live renderer is not registered.
std::function<void(uint64_t)> g_flip_scanout_hook;

void queue_guest_gpu_write(uint64_t addr, uint64_t size, const char* origin) {
    auto& pending = pending_guest_gpu_writes();
    // The origin arrives AS DATA from the notifier. Reading the thread-local here instead would be
    // correct for the plain notifier and wrong for the byte-preserving one, which knows a
    // classification the thread-local never carried.
    if (!origin) origin = "unknown";
    std::lock_guard<std::mutex> lock(pending.mutex);
    constexpr size_t kMaxPendingRanges = 65536;
    if (pending.ranges.size() < kMaxPendingRanges)
        pending.ranges.push_back(PendingGuestGpuWrite{addr, size, origin});
    else
        pending.overflowed = true;
}

void register_cpu_rtt_dcc_metadata(
    RttCache& cache, const std::vector<prosper::gpu::DrawItem>& items) {
    for (const auto& item : items) {
        const prosper::gpu::ShaderResourceTable* tables[] = {
            item.vrt.get(), item.prt.get(),
        };
        for (const auto* table : tables) {
            if (!table) continue;
            for (const auto& resource : table->resources) {
                if (!resource.compression_enabled || !resource.gpu_addr ||
                    !resource.metadata_addr)
                    continue;
                auto surface = cache.find(resource.gpu_addr);
                if (surface == cache.end()) continue;
                const uint64_t metadata_bytes =
                    prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
                if (!metadata_bytes) continue;
                surface->second.dcc_metadata_addr = resource.metadata_addr;
                surface->second.dcc_metadata_bytes = metadata_bytes;
            }
        }
    }
}

bool capture_color_format(VkFormat format, prosper::gpu::GpuCaptureColorFormat& captured) {
    format = prosper::test::backend_color_format(format);
    if (format == VK_FORMAT_R8G8B8A8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm;
    else if (format == VK_FORMAT_R16G16B16A16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba16Float;
    else if (format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        captured = prosper::gpu::GpuCaptureColorFormat::R11G11B10Float;
    else if (format == VK_FORMAT_R8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::R8Unorm;
    else if (format == VK_FORMAT_R32_UINT)
        captured = prosper::gpu::GpuCaptureColorFormat::R32Uint;
    else if (format == VK_FORMAT_R32_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::R32Float;
    else if (format == VK_FORMAT_R8G8_UNORM)
        captured = prosper::gpu::GpuCaptureColorFormat::Rg8Unorm;
    else if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rgba32Float;
    else if (format == VK_FORMAT_R16G16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::Rg16Float;
    else if (format == VK_FORMAT_R16_SFLOAT)
        captured = prosper::gpu::GpuCaptureColorFormat::R16Float;
    else
        return false;
    return true;
}

VkFormat replay_color_format(prosper::gpu::GpuCaptureColorFormat format) {
    if (format == prosper::gpu::GpuCaptureColorFormat::Rgba16Float)
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R11G11B10Float)
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    if (format == prosper::gpu::GpuCaptureColorFormat::R8Unorm)
        return VK_FORMAT_R8_UNORM;
    if (format == prosper::gpu::GpuCaptureColorFormat::R32Uint)
        return VK_FORMAT_R32_UINT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R32Float)
        return VK_FORMAT_R32_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rg8Unorm)
        return VK_FORMAT_R8G8_UNORM;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rgba32Float)
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::Rg16Float)
        return VK_FORMAT_R16G16_SFLOAT;
    if (format == prosper::gpu::GpuCaptureColorFormat::R16Float)
        return VK_FORMAT_R16_SFLOAT;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

// ---- AvPlayer NV12 chroma-plane recognition diagnostic (#2005) --------------------------------
//
// The classification itself lives in avplayer_plane_policy.hpp (with its own unit test); this is
// only its live log. PROSPER_AVPCHROMA_LOG=1 emits one line per distinct narrow Unorm8 sampled
// texture — both NV12 planes qualify, whatever DIM they declare — naming every field the predicate
// reads and the clause that decided it. Off by default. A wrong verdict here is silent: a chroma
// plane sent down the coverage broadcast still produces correct luma, detail and geometry, so no
// draw census, colour count or non-black metric can see it.
using prosper::frontend::AvpChromaReason;
using prosper::frontend::AvpChromaVerdict;
using prosper::frontend::avp_chroma_reason_name;
using prosper::frontend::classify_avplayer_chroma_plane;
}

size_t texture_decode_cache_limit_bytes(const char* override_mib,
                                        uint64_t physical_memory_bytes) {
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    constexpr uint64_t kMinBytes = 1024ull * kMiB;
    constexpr uint64_t kMaxBytes = 4096ull * kMiB;
    uint64_t bytes = kMinBytes;
    if (override_mib) {
        const uint64_t mib = strtoull(override_mib, nullptr, 10);
        bytes = std::min<uint64_t>(mib, SIZE_MAX / kMiB) * kMiB;
    } else if (physical_memory_bytes) {
        bytes = std::clamp(physical_memory_bytes / 8u, kMinBytes, kMaxBytes);
        bytes -= bytes % kMiB;
    }
    return static_cast<size_t>(std::min<uint64_t>(bytes, SIZE_MAX));
}

bool submit_local_texture_decode_reusable(uint64_t entry_span, uint64_t current_span,
                                          uint64_t entry_source_addr, uint64_t entry_source_size,
                                          uint64_t current_source_addr, uint64_t current_source_size,
                                          prosper::gpu::GuestGpuWriteQuery journal_query) {
    if (entry_span == current_span) return true;
    return current_source_size != 0 &&
        entry_source_addr == current_source_addr &&
        entry_source_size == current_source_size &&
        journal_query == prosper::gpu::GuestGpuWriteQuery::Unchanged;
}

GuestWriteDrainWork guest_write_drain_work_for_thread() {
    const auto* census = guest_write_drain_census();
    if (!census) return {};
    return {true, census->calls, census->rtt_footprints, census->prepared_drains,
            census->rtt_erases, census->rtt_dcc};
}

TextureDecodeScopeStats texture_decode_scope_stats() { return g_texture_decode_scope; }
void reset_texture_decode_scope_stats() { g_texture_decode_scope = {}; }

bool sampled_msaa_fetch_shape_supported(const prosper::gpu::ShaderResource& resource,
                                        bool is_storage_image,
                                        bool reflected_msaa_fetch) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::ResourceClass;
    return resource.cls == ResourceClass::Texture && !is_storage_image &&
        resource.img_dim == 6u && resource.sample_count == 4u &&
        resource.tile_mode == 24u && resource.format == DataFormat::Float32 &&
        resource.num_components == 1u && resource.depth == 1u &&
        resource.declared_mip_levels == 1u && !resource.in_mip_tail &&
        resource.mip_tail_bytes == 0u && resource.layer_stride_bytes == 0u &&
        resource.layer_mip_offset_bytes == 0u && reflected_msaa_fetch;
}

bool native_r11_sampled_upload_supported(const prosper::gpu::ShaderResource& resource) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::ResourceClass;
    return resource.cls == ResourceClass::Texture && resource.img_dim == 1u &&
        resource.depth == 1u && resource.format == DataFormat::Float10_11_11 &&
        resource.num_components == 3u && !resource.compression_enabled && !resource.srgb;
}

uint32_t native_bc_sampled_format(const prosper::gpu::ShaderResource& resource,
                                  bool allow_declared_mips) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::ResourceClass;
    if (resource.cls != ResourceClass::Texture || resource.compression_enabled) return 0u;
    if (!(resource.img_dim == 1u || resource.img_dim == 5u) || resource.depth > 1u) return 0u;
    if (!allow_declared_mips && resource.declared_mip_levels > 1u) return 0u;
    switch (resource.format) {
        case DataFormat::Bc1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case DataFormat::Bc2: return VK_FORMAT_BC2_UNORM_BLOCK;
        case DataFormat::Bc3: return VK_FORMAT_BC3_UNORM_BLOCK;
        case DataFormat::Bc4: return VK_FORMAT_BC4_UNORM_BLOCK;
        case DataFormat::Bc5: return VK_FORMAT_BC5_UNORM_BLOCK;
        case DataFormat::Bc6: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case DataFormat::Bc7: return VK_FORMAT_BC7_UNORM_BLOCK;
        default: return 0u;
    }
}

RendererMipChainLayout renderer_mip_chain_layout(uint64_t level_zero_address,
                                                 uint32_t width, uint32_t height,
                                                 uint32_t bytes_per_texel,
                                                 uint32_t tile_mode,
                                                 uint32_t declared_mip_levels) {
    RendererMipChainLayout result;
    if (!level_zero_address || !width || !height || !bytes_per_texel ||
        declared_mip_levels < 2u || declared_mip_levels > result.level_ids.size())
        return result;
    const uint32_t max_mip = declared_mip_levels - 1u;
    const prosper::gpu::TiledMipLevelLayout level_zero =
        prosper::gpu::tiled_mip_level_layout(
            width, height, bytes_per_texel, tile_mode, max_mip, 0u);
    // This helper starts from a descriptor selecting allocation level zero. A level-zero tail view
    // cannot identify the allocation origin by subtraction and remains on the established fallback.
    if (!level_zero.supported || level_zero.in_tail ||
        level_zero.byte_offset > level_zero_address)
        return result;
    const uint64_t allocation_base = level_zero_address - level_zero.byte_offset;
    for (uint32_t level = 0; level < declared_mip_levels; ++level) {
        const prosper::gpu::TiledMipLevelLayout layout =
            prosper::gpu::tiled_mip_level_layout(
                width, height, bytes_per_texel, tile_mode, max_mip, level);
        if (!layout.supported ||
            (!layout.in_tail && layout.byte_offset > UINT64_MAX - allocation_base))
            return {};
        result.level_ids[level] = layout.in_tail
            ? allocation_base : allocation_base + layout.byte_offset;
    }
    result.level_count = declared_mip_levels;
    return result;
}

bool persistent_texture_decode_cache_eligible(bool guest_decode_candidate,
                                              bool compute_image_hit,
                                              bool is_storage_image,
                                              bool cache_disabled,
                                              bool compression_supported,
                                              size_t cache_limit,
                                              size_t source_size) {
    return guest_decode_candidate && !compute_image_hit && !is_storage_image &&
        !cache_disabled && compression_supported && cache_limit && source_size;
}

// PROSPER_RTT_ALIAS=<sampled>:<served>[,<sampled>:<served>…] — diagnostic ONLY, off by default.
//
// Serve one sampled surface from another surface's cache entry. It answers exactly one question, and
// only by experiment: when a consumer samples an address nothing writes, is that address a distinct
// buffer whose producer is missing, or the SAME buffer the renderer already holds under a different
// base?  Those two have identical signatures in every census -- the pass graph, the target census
// and the write watches all report "written by nothing" either way -- and they call for opposite
// work: find the missing producer, or fix a base decode.
//
// GTA V PPSA04263 is the worked case. Its final composite samples the 4K HDR scene colour
// 0x2063380000 24x per frame, and across 6,561 frames that address is a render target exactly once,
// for a clear. Meanwhile the lit scene sits in 0x20431c0000, which the composite never reads.
// Aliasing one onto the other decides between the two readings in a single run.
//
// This is not a repair and must never become one: it makes a consumer read bytes the guest did not
// point it at, so a picture appearing under it is evidence about the DECODE, never a fix. Leave it
// unset outside an A/B.
// PROSPER_SCAN_DESCRIPTOR=<hex surface base>[,<hex base>…] — count CANDIDATE descriptor words in
// guest memory whose value matches a surface's base field, once, from the render thread.
//
// READ THE NEGATIVES CAREFULLY. This is a candidate-word scan, not a descriptor enumerator, and it is
// not exhaustive in three separate ways:
//   * COVERAGE. It probes 2 MiB-aligned spans in a fixed window and keeps only spans that are
//     readable IN FULL, so a descriptor in a partially mapped span, or in a mapping outside the
//     window, is never examined. The report prints the spans kept and skipped so the coverage is
//     visible rather than assumed.
//   * VALIDATION. A matching dword is only the value a base field WOULD hold; the following words are
//     printed but not checked against a descriptor layout. A hit is a candidate, not proof that a
//     descriptor lives there, and an ordinary integer can match.
//   * DIRECTION. A descriptor does not encode read/write intent -- that lives in the shader
//     instruction -- so a hit cannot distinguish a producer's view of a surface from a consumer's.
// **A zero therefore cannot establish that no producer exists.** It bounds where one was not found,
// which is a weaker and different claim.
//
// COST: it runs SYNCHRONOUSLY on the render callback and may read several GiB, so the frame it fires
// on stalls visibly. Opt-in and one-shot for that reason; never leave it set for a timing run.
//
// The question it exists for is the one left over when every write path comes back empty: a surface
// nothing writes is either one the guest genuinely never produces, or one whose producer prosper
// failed to RESOLVE -- and an unresolved producer is invisible to every instrument that enumerates
// resolved tables. `PROSPER_DYNTRACE_FAIL` prints only the descriptors the const-fold evaluated;
// measured on GTA V's failing kernels it printed 15 of 21 image ops for 0x205b5e8600 and 4 of 6 for
// 0x205b657200, so its null covered most of that population and not all of it.
//
// A descriptor's base field holds the address shifted right by 8, so a surface's candidates are
// findable by matching that dword. Use the result as a lead to follow, not as a census.
void scan_for_descriptors_once(uint64_t submit_index) {
    struct Spec { std::vector<uint64_t> bases; uint64_t at_submit = 0; };
    static const Spec spec = [] {
        Spec parsed;
        const char* text = getenv("PROSPER_SCAN_DESCRIPTOR");
        if (!text || !*text) return parsed;
        for (const char* cursor = text; *cursor;) {
            char* end = nullptr;
            const uint64_t value = strtoull(cursor, &end, 16);
            if (end == cursor) break;
            parsed.bases.push_back(value);
            cursor = (*end == ',') ? end + 1 : end;
        }
        // SUBMITS, not frames: the caller passes g_this_submit. Named for what it counts -- the two
        // differ by more than an order of magnitude on a routed run, so a threshold read as frames
        // fires at a moment the operator did not ask for.
        const char* at = getenv("PROSPER_SCAN_DESCRIPTOR_AT_SUBMIT");
        parsed.at_submit = at ? strtoull(at, nullptr, 10) : 6000;
        return parsed;
    }();
    if (spec.bases.empty()) return;
    static std::atomic<bool> done{false};
    if (submit_index < spec.at_submit || done.exchange(true)) return;

    // The guest's direct-memory mapping, discovered by probing rather than assumed: walk 2 MiB steps
    // over the region PS5 titles map and keep the readable spans.
    constexpr uint64_t kStep = 2ull << 20;
    constexpr uint64_t kBegin = 0x2000000000ull, kEnd = 0x2200000000ull;
    size_t spans = 0, skipped_spans = 0, scanned_mb = 0;
    std::map<uint64_t, uint64_t> hits;   // surface base -> count
    std::map<uint64_t, std::map<std::string, uint64_t>> shapes;  // base -> shape -> count
    for (uint64_t page = kBegin; page < kEnd; page += kStep) {
        if (!prosper::gpu::guest_readable(page, static_cast<uint32_t>(kStep))) {
            ++skipped_spans;   // unmapped OR only partially mapped -- not examined
            continue;
        }
        ++spans; scanned_mb += kStep >> 20;
        const uint32_t* words = reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(page));
        const size_t count = kStep / sizeof(uint32_t);
        for (size_t i = 0; i < count; ++i) {
            for (const uint64_t base : spec.bases) {
                if (words[i] != static_cast<uint32_t>(base >> 8)) continue;
                ++hits[base];
                // A bare dword match is not yet a descriptor: the same bit pattern occurs in ordinary
                // data. Decode the eight words that would follow if this were a T# and group by their
                // SHAPE, so a real descriptor set stands out from coincidence and two descriptors that
                // describe the surface differently (a producer's view versus a consumer's) separate.
                if (i + 8 > count) continue;
                char sig[128];
                snprintf(sig, sizeof sig, "w1=%08x w2=%08x w3=%08x w4=%08x w7=%08x",
                         words[i + 1], words[i + 2], words[i + 3], words[i + 4], words[i + 7]);
                shapes[base][sig] += 1;
            }
        }
    }
    fprintf(stderr,
            "[descr-scan] examined %zu MiB over %zu fully-readable spans at submit %llu; %zu span(s) "
            "SKIPPED (unmapped or partially mapped) -- negatives are NOT exhaustive\n",
            scanned_mb, spans, (unsigned long long)submit_index, skipped_spans);
    for (const uint64_t base : spec.bases) {
        fprintf(stderr,
                "[descr-scan] 0x%llx: %llu CANDIDATE word(s) match its base field, %zu distinct "
                "following shape(s) -- unvalidated, and direction (producer vs consumer) unknown\n",
                (unsigned long long)base, (unsigned long long)hits[base], shapes[base].size());
        size_t shown = 0;
        for (const auto& [sig, n] : shapes[base]) {
            if (shown++ >= 8) break;
            fprintf(stderr, "[descr-scan]     x%-4llu %s\n", (unsigned long long)n, sig.c_str());
        }
    }
}

uint64_t texture_rtt_alias(uint64_t gpu_address) {
    struct Alias { uint64_t from, to; };
    static const std::vector<Alias> aliases = [] {
        std::vector<Alias> parsed;
        const char* spec = getenv("PROSPER_RTT_ALIAS");
        if (!spec || !*spec) return parsed;
        for (const char* cursor = spec; *cursor;) {
            char* end = nullptr;
            const uint64_t from = strtoull(cursor, &end, 16);
            if (end == cursor || *end != ':') break;
            cursor = end + 1;
            const uint64_t to = strtoull(cursor, &end, 16);
            if (end == cursor) break;
            parsed.push_back({from, to});
            fprintf(stderr, "[rtt-alias] sampling 0x%llx will be served from 0x%llx\n",
                    (unsigned long long)from, (unsigned long long)to);
            cursor = (*end == ',') ? end + 1 : end;
        }
        return parsed;
    }();
    for (const Alias& alias : aliases)
        if (alias.from == gpu_address) return alias.to;
    return gpu_address;
}

uint64_t texture_decode_source_address(uint64_t gpu_address,
                                       uint32_t image_dimension,
                                       bool in_mip_tail,
                                       uint32_t layer_mip_offset_bytes) {
    gpu_address = texture_rtt_alias(gpu_address);
    if (image_dimension != 5u || in_mip_tail ||
        gpu_address > UINT64_MAX - layer_mip_offset_bytes)
        return gpu_address;
    return gpu_address + layer_mip_offset_bytes;
}

bool texture_source_snapshot_can_follow_watch(bool source_matches_pixels,
                                              bool validation_audit_enabled,
                                              bool watch_active,
                                              size_t retained_source_bytes,
                                              size_t expected_source_bytes) {
    return !source_matches_pixels && !validation_audit_enabled && watch_active &&
        expected_source_bytes != 0 && retained_source_bytes == expected_source_bytes;
}

uint32_t buffer_upload_bytes(uint32_t declared_bytes) {
    // PROSPER_MAX_BUFFER_UPLOAD_MB=N lowers the ceiling (1..64 MiB) so one build can reproduce the
    // #1427 collapse and its fix back to back; unset/invalid keeps the full ceiling.
    static const uint32_t ceiling = [] {
        const char* value = getenv("PROSPER_MAX_BUFFER_UPLOAD_MB");
        if (!value || !*value) return kMaxBufferUploadBytes;
        const unsigned long megabytes = strtoul(value, nullptr, 10);
        if (megabytes < 1 || megabytes > 64) return kMaxBufferUploadBytes;
        return static_cast<uint32_t>(megabytes) << 20;
    }();
    return std::min(declared_bytes, ceiling) & ~3u;
}

// #2215 instrument -- see live_renderer.hpp for why this is a process-wide slot rather than a
// thread_local (the sampler asks from a different thread than the one it is asking about).
std::atomic<unsigned long>& renderer_callback_tid() {
    static std::atomic<unsigned long> tid{0};
    return tid;
}

bool tid_is_in_renderer_callback(unsigned long native_tid) {
    const unsigned long current = renderer_callback_tid().load(std::memory_order_relaxed);
    return current != 0 && current == native_tid;
}

// Every callback ENTRY, not the ~0.6% of them a sampler happens to catch. See the self-check in
// hle_kernel.cpp: a zero here is real evidence that the strong override never linked, whereas a zero
// count of sampler sightings is just the ordinary outcome at this base rate (#2347).
std::atomic<unsigned long long> g_renderer_callback_entries{0};

extern "C" unsigned long long prosper_renderer_callback_entries() {
    return g_renderer_callback_entries.load(std::memory_order_relaxed);
}

ScopedRendererCallbackTid::ScopedRendererCallbackTid() {
    g_renderer_callback_entries.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    const unsigned long self = (unsigned long)GetCurrentThreadId();
#else
    const unsigned long self = (unsigned long)(uintptr_t)pthread_self();
#endif
    previous = renderer_callback_tid().exchange(self, std::memory_order_relaxed);
}

ScopedRendererCallbackTid::~ScopedRendererCallbackTid() {
    renderer_callback_tid().store(previous, std::memory_order_relaxed);
}

// The strong definition that overrides prosper_core's weak stub once the live renderer is in
// the link. Keeping the override here rather than in prosper_core is what lets a build WITHOUT
// the renderer answer 0 truthfully instead of failing to link (#2215).
extern "C" int prosper_thread_in_renderer_callback(unsigned long native_tid) {
    return prosper::frontend::tid_is_in_renderer_callback(native_tid) ? 1 : 0;
}

void register_live_renderer(const std::string& frame_dir, bool dump_bmps_requested,
                            prosper::gpu::FragmentWavePolicy requested_wave_policy) {
    prosper::diagnostics::readback_refusal::initialize();
    // Keep the legacy global disable authoritative for every frontend, including callers with their
    // own explicit opt-in such as PROSPER_APP_DUMP_FRAMES.
    const bool dump_bmps = frame_dump_request_allowed(
        dump_bmps_requested, PROSPER_ENV_VALUE("PROSPER_NO_FRAME_DUMPS"));
    // Frontends select a guest-semantic contract explicitly. No game ID participates. Replay and
    // direct callers remain Strict by default; a diagnostic opt-out cannot opt them into lowering.
    const auto fragment_wave_policy = PROSPER_ENV_VALUE("PROSPER_STRICT_FRAGMENT_WAVE_WIDTH")
        ? prosper::gpu::FragmentWavePolicy::Strict : requested_wave_policy;
    // Create (and thereby PUBLISH) the renderer's Vulkan device up front so the compute backend can
    // adopt it (#1091). Compute initializes lazily on its first dispatch, and titles routinely
    // dispatch before their first draw -- without this the compute device would be created first and
    // the two would never share. Only reached when the live renderer is registered, so headless
    // compute-only use (tests/gpu/recompiler/test_game_compute.cpp) still creates its own device.
    (void)prosper::test::render_vk_ctx();
    static RttCache g_rtt;   // render-to-texture cache (#167)
    // PROSPER_FLIP_GUEST_SCANOUT=1 (default OFF): let a guest FLIP publish the guest's own scanout
    // buffer when the renderer has produced nothing at all.
    //
    // prosper's present path is SPAN-DRIVEN: every publish decision above, including the
    // guest-scanout fallback that exists precisely for "the guest wrote the framebuffer itself",
    // sits behind `phase.final_span`, i.e. behind a graphics DRAW. A title that composites entirely
    // with compute and never draws therefore never reaches it -- the flips arrive and nothing
    // downstream is ever asked whether there is a frame. Measured on Uncharted (PPSA05684, #3616):
    // 29,283 flips over 540 s, `draws_cum` 0, `[rtt] GUEST SCANOUT` printed ZERO times, and
    // `0 published`.
    //
    // This hook is deliberately the narrowest thing that closes that: it runs only while
    // `present_has_frame()` is false, so the moment the renderer produces anything the span path
    // owns presentation again and this never competes with it. That also removes the cross-thread
    // question -- it does not consult g_rtt, because a renderer that owns a target has necessarily
    // produced a frame, and this path has already returned by then.
    //
    // The POLICY is not re-implemented: guest_scanout_read_warranted / guest_scanout_publishable are
    // the same unit-tested predicates the span path uses, including the authorship test that is the
    // whole point (a buffer whose contents have never differed from what they were at registration
    // is not the guest's frame, whatever is in it).
    g_flip_scanout_hook = [](uint64_t flip) {
        static const bool armed = getenv("PROSPER_FLIP_GUEST_SCANOUT") != nullptr;
        if (!armed) return;
        if (prosper::gpu::present_has_frame()) return;      // the span path owns presentation
        const uint32_t w = prosper::gpu::present_width(), h = prosper::gpu::present_height();
        const size_t display_bytes = static_cast<size_t>(w) * h * 4u;
        if (!display_bytes) return;
        auto decision = prosper::frontend::guest_scanout_read_warranted(
            /*published_gpu=*/false, /*renderer_scanout=*/false, /*have_selected_pixels=*/false,
            display_bytes, display_bytes);
        if (decision != prosper::frontend::GuestScanoutDecision::Publish) return;
        prosper::VideoOutLinearRead read;
        const bool got = prosper::videoout_read_front_linear(read);
        decision = prosper::frontend::guest_scanout_publishable(
            got ? read.pixels.size() : 0u, display_bytes,
            got && read.metadata.address != 0,
            /*renderer_owns_target=*/false,      // no rendered frame exists: see the note above
            read.guest_authored);
        static std::atomic<uint64_t> reports[(size_t)
            prosper::frontend::GuestScanoutDecision::SkipNotAuthored + 1]{};
        const uint64_t ord = reports[(size_t)decision].fetch_add(1) + 1;
        if (prosper::diag_should_print(ord))
            fprintf(stderr, "[flip-scanout] #%llu flip=%llu %s (%ux%u addr=0x%llx authored=%d "
                            "read=%zu want=%zu)\n",
                    (unsigned long long)ord, (unsigned long long)flip,
                    prosper::frontend::guest_scanout_decision_name(decision), w, h,
                    (unsigned long long)read.metadata.address, (int)read.guest_authored,
                    got ? read.pixels.size() : 0u, display_bytes);
        if (decision != prosper::frontend::GuestScanoutDecision::Publish) return;
        prosper::gpu::present_write_frame(
            std::make_shared<const std::vector<uint8_t>>(std::move(read.pixels)), w, h,
            prosper::gpu::PresentFrameOrigin::GuestScanout);
    };
    // Hand core the entry point now that the hook exists. Registration rather than a symbol
    // reference: prosper_core links into tools that have no frontend at all.
    prosper_vo_set_flip_publish_hook(&prosper_frontend_flip_publish_guest_scanout);
    // Shared final-callback ordinal for PROSPER_PASS_LOG (increments where dp_submit does).
    static std::atomic<uint64_t> g_pass_log_submit{0};
    // The census windows those two switches open. Parsed once here rather than per callback so the
    // `ms:` form's latch is one object: PROSPER_PASS_LOG is consulted from two places about the SAME
    // ordinal, and a per-site window would latch them onto different callbacks. Same single-present-
    // thread contract as `g_rtt` and the `dp_submit`/`warned` statics below.
    static prosper::frontend::DiagnosticWindow g_pass_log_window{
        prosper::frontend::parse_diagnostic_window(PROSPER_ENV_VALUE("PROSPER_PASS_LOG"))};
    static prosper::frontend::DiagnosticWindow g_persist_window{
        prosper::frontend::parse_diagnostic_window(PROSPER_ENV_VALUE("PROSPER_DUMP_PERSISTENT"))};
    static const auto g_persist_filter = [] {
        const char* spec = PROSPER_ENV_VALUE("PROSPER_DUMP_PERSISTENT_ADDRS");
        auto filter = prosper::frontend::parse_persistent_readback_filter(spec);
        if (!spec) return filter;
        if (!PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT"))
            fprintf(stderr, "[persist] PROSPER_DUMP_PERSISTENT_ADDRS is a modifier; "
                            "PROSPER_DUMP_PERSISTENT is not armed -- no readback\n");
        if (filter.state == prosper::frontend::PersistentReadbackFilterState::Selected)
            fprintf(stderr, "[persist] address selector armed on %zu target(s)\n",
                    filter.addresses.size());
        else
            fprintf(stderr, "[persist] address selector refused (%s); no retained targets "
                            "will be read back\n",
                    filter.state == prosper::frontend::PersistentReadbackFilterState::TooMany
                        ? "more than eight addresses" : "malformed 0x-prefixed list");
        return filter;
    }();
    static const std::pair<uint32_t, uint32_t> g_persist_extent = [] {
        const char* spec = PROSPER_ENV_VALUE("PROSPER_DUMP_PERSISTENT_EXTENT");
        if (!spec) return std::pair<uint32_t, uint32_t>{0, 0};
        uint32_t width = 0, height = 0;
        if (!parse_diagnostic_extent(spec, width, height)) {
            std::fprintf(stderr, "[persist] extent selector refused: expected WxH\n");
            return std::pair<uint32_t, uint32_t>{UINT32_MAX, UINT32_MAX};
        }
        std::fprintf(stderr, "[persist] extent selector armed on %ux%u\n", width, height);
        return std::pair<uint32_t, uint32_t>{width, height};
    }();
    // Match boot_trace's progression-diagnostic contract: callers may register the graphics
    // renderer while deliberately leaving compute unregistered. This keeps semantic dispatches
    // visible without letting screenshot/prosper-app registration silently undo the A/B.
    if (!PROSPER_ENV_ON("PROSPER_NO_COMPUTE")) register_live_compute();
    const char* ds_invalidate = PROSPER_ENV_VALUE("PROSPER_DS_GUEST_WRITE_INVALIDATE");
    const bool invalidate_ds = !ds_invalidate || strcmp(ds_invalidate, "0");
    prosper::gpu::set_guest_gpu_write_observer(
        [](uint64_t addr, uint64_t size, const char* origin) {
            queue_guest_gpu_write(addr, size, origin);
        });
    // Resource tables are built before the submit reaches this callback. Publish the renderer's
    // default mode now so unmapped render-target descriptors remain available for RTT injection.
    // Outside a registered renderer, resource decoding retains its strict unknown-format policy.
    // PROSPER_RTT_PERTARGET is read LIVE here even though this runs once: the very next lines SET
    // it, so it is a variable this process writes to itself. Caching it would be correct today
    // (each macro use site has its own static, and :1067 reads after this one) but it is correct by
    // ordering rather than by construction, and it is the only thing that would put this name in
    // the cached-vs-armed intersection below. A gate with one standing exception is a gate people
    // stop reading. This site is init-time, so a live read costs nothing.
    if (!PROSPER_ENV_ON("PROSPER_RTT_SINGLE_TARGET") && !getenv("PROSPER_RTT_PERTARGET")) {
#ifdef _WIN32
        _putenv_s("PROSPER_RTT_PERTARGET", "1");
#else
        setenv("PROSPER_RTT_PERTARGET", "1", 1);
#endif
    }
    static std::atomic<int> frame_no{0};
    // Register the capture RTT-seed readers whenever the live renderer is up — not only under the
    // capture env vars — so the interactive F9 frame grab (request_interactive_capture_bundle, which has
    // no env var) can seed the renderer-owned RTTs its frame SAMPLES. Registration is free: these lambdas
    // are only invoked while a capture is actually in flight; a normal render run never calls them.
    // Without the seed, a submit that samples a deferred/temporal renderer-owned RGBA16F target (Blue
    // Prince's presenting pass) replays black because that input is captured as all-zeros (#1291).
    {
        auto materialize_current_rtt = [](uint64_t addr, RttSurf& surface,
                                          std::string& error) {
            // Capture's RTT seed format carries one 2D plane. Never serialize a truncated
            // volume under that identity; its exact 3D image remains authoritative live.
            if (surface.volume_depth) return false;
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            const uint32_t bytes_per_pixel =
                prosper::test::backend_color_bytes_per_pixel(format);
            if (surface.rgba && prosper::frontend::live_rtt_cpu_snapshot_matches(
                    surface.w, surface.h, bytes_per_pixel, surface.rgba->size()))
                return true;
            if (materialize_uniform_rtt(surface)) return true;
            if (!surface.gpu_valid) return false;
            std::vector<uint8_t> materialized;
            if (!prosper::test::readback_persistent_color_target(
                    addr, surface.w, surface.h, format, materialized, error))
                return false;
            if (!prosper::frontend::live_rtt_cpu_snapshot_matches(
                    surface.w, surface.h, bytes_per_pixel, materialized.size())) {
                error = "persistent RTT readback byte count does not match its current identity";
                return false;
            }
            surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                std::move(materialized));
            return true;
        };
        prosper::gpu::set_gpu_capture_rtt_seed_reader(
            [invalidate_ds, materialize_current_rtt](
                uint64_t addr, prosper::gpu::GpuCaptureRttSeed& seed) {
                drain_guest_gpu_writes(g_rtt, invalidate_ds);
                auto it = g_rtt.find(addr); if (it == g_rtt.end()) return false;
                if (it->second.volume_depth) return false;
                prosper::gpu::GpuCaptureColorFormat captured_format;
                if (!capture_color_format(it->second.format, captured_format)) return false;
                std::string error;
                if (!materialize_current_rtt(addr, it->second, error)) return false;
                seed.guest_addr = addr; seed.width = it->second.w; seed.height = it->second.h;
                seed.format = captured_format;
                seed.rgba = *it->second.rgba; return true;
            });
        prosper::gpu::set_gpu_capture_rtt_seed_snapshot_reader(
            [invalidate_ds, materialize_current_rtt](
                std::vector<prosper::gpu::GpuCaptureRttSeed>& seeds, std::string& error) {
                drain_guest_gpu_writes(g_rtt, invalidate_ds);
                seeds.reserve(g_rtt.size());
                for (auto& [addr, surface] : g_rtt) {
                    if (surface.volume_depth) continue;
                    prosper::gpu::GpuCaptureColorFormat captured_format;
                    if (!capture_color_format(surface.format, captured_format)) continue;
                    std::string readback_error;
                    if (!materialize_current_rtt(addr, surface, readback_error)) {
                        if (!readback_error.empty()) {
                            char detail[256];
                            std::snprintf(detail, sizeof(detail),
                                          "RTT 0x%llx %ux%u snapshot failed: %s",
                                          static_cast<unsigned long long>(addr),
                                          surface.w, surface.h, readback_error.c_str());
                            error = detail;
                            return false;
                        }
                        continue;
                    }
                    prosper::gpu::GpuCaptureRttSeed seed;
                    seed.guest_addr = addr; seed.width = surface.w; seed.height = surface.h;
                    seed.format = captured_format;
                    seed.rgba = *surface.rgba; seeds.push_back(std::move(seed));
                }
                return true;
            });
    }
    // Deferred RTT readback (#1284) can leave a persistent Vulkan image as the ONLY copy of a
    // target's pixels. When the backend evicts such an image it reads the pixels back first and
    // hands them here, so the CPU cache regains the authoritative copy the eager path used to keep.
    prosper::test::persistent_color_target_evict_sink() =
        [](uint64_t id, uint32_t width, uint32_t height, VkFormat format,
           std::vector<uint8_t>&& pixels) {
            auto it = g_rtt.find(id);
            if (it == g_rtt.end()) return;
            RttSurf& surface = it->second;
            if (surface.volume_depth) return;
            if (surface.w != width || surface.h != height ||
                prosper::test::backend_color_format(surface.format) !=
                    prosper::test::backend_color_format(format))
                return;
            const size_t expected = static_cast<size_t>(width) * height *
                prosper::test::backend_color_bytes_per_pixel(
                    prosper::test::backend_color_format(format));
            if (pixels.size() != expected) return;
            if (!surface.rgba || surface.rgba->size() != expected)
                surface.rgba =
                    std::make_shared<const std::vector<uint8_t>>(std::move(pixels));
            surface.gpu_valid = false;   // the image is being destroyed
        };
    // The compute backend must not sample a surface whose CURRENT pixels live in this renderer's
    // RTT cache (raw guest memory is then empty/stale — the Dead Cells 642x362 lesson): publish the
    // exact-match identity and immutable CPU snapshot used by live compute (#590).
    // Establish WHICH kind of compression metadata an address is. The decision is a pure function over
    // retained state (metadata_kind_correlation.hpp); this lambda only GATHERS that state, so the rule
    // itself is mutation-testable at the site that ships rather than through this registration.
    //
    // Both answers are positive correlations over metadata AND resource identity. An earlier version
    // answered DCC by elimination -- "not Float32x1, therefore colour" -- which classified an
    // uncorrelated Uint32x1 raw alias as DCC and let its all-0xff bytes authorize reading the base.
    prosper::gpu::set_metadata_kind_query(
        [](const prosper::gpu::MetadataKindRequest& request) {
            std::vector<prosper::gpu::RetainedDepthCorrelation> depth;
            for (const auto& [key, image] : prosper::test::persistent_ds_cache()) {
                (void)image;
                depth.push_back({key.dr, key.dw, key.sr, key.sw, key.htile});
            }
            std::vector<prosper::gpu::RetainedColorCorrelation> color;
            for (const auto& [addr, surface] : g_rtt)
                color.push_back({addr, surface.dcc_metadata_addr});
            return prosper::gpu::correlate_compression_metadata_kind(request, depth, color);
        });

    prosper::gpu::set_live_target_query([invalidate_ds](uint64_t addr) {
        drain_guest_gpu_writes(g_rtt, invalidate_ds);
        auto it = g_rtt.find(addr);
        if (it == g_rtt.end()) return false;
        const RttSurf& surface = it->second;
        if (surface.volume_guest_bytes) return true; // unsupported compute reads must not use stale guest bytes
        const VkFormat format = prosper::test::backend_color_format(surface.format);
        const uint32_t bytes_per_pixel =
            prosper::test::backend_color_bytes_per_pixel(format);
        const bool has_cpu_snapshot = (surface.rgba &&
            prosper::frontend::live_rtt_cpu_snapshot_matches(
                surface.w, surface.h, bytes_per_pixel, surface.rgba->size())) ||
            surface.has_uniform_color;
        return prosper::frontend::live_rtt_compute_authoritative(
            surface.gpu_valid, has_cpu_snapshot);
    });
    prosper::gpu::set_unpublished_volume_query([invalidate_ds](uint64_t addr, uint64_t bytes) {
        drain_guest_gpu_writes(g_rtt, invalidate_ds);
        return any_volume_target(g_rtt, [&](uint64_t base, const RttSurf& surface) {
            return unpublished_volume_may_overlap(base, surface.volume_guest_bytes, addr, bytes);
        });
    });
    prosper::gpu::set_live_target_reader(
        [invalidate_ds](uint64_t addr, prosper::gpu::LiveTargetSnapshot& snapshot) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            auto it = g_rtt.find(addr);
            if (it == g_rtt.end() || !it->second.w || !it->second.h) return false;
            RttSurf& surface = it->second;
            if (surface.volume_depth) return false; // LiveTargetSnapshot is 2D-only.
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            const uint32_t bytes_per_pixel = prosper::test::backend_color_bytes_per_pixel(format);
            const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
            if (!bytes_per_pixel || texels > UINT64_MAX / bytes_per_pixel) return false;
            const uint64_t expected = texels * bytes_per_pixel;
            if ((!surface.rgba || surface.rgba->size() != expected) &&
                surface.has_uniform_color)
                materialize_uniform_rtt(surface);
            // Graphics-to-graphics intermediates normally stay in the persistent Vulkan target.
            // Compute cannot import that color attachment directly, so materialize its current
            // pixels only when an ordered compute dispatch actually consumes the surface.
            if ((!surface.rgba || surface.rgba->size() != expected) && surface.gpu_valid) {
                std::vector<uint8_t> materialized;
                std::string error;
                const bool readback_ok = [&] {
                    namespace refusal = prosper::diagnostics::readback_refusal;
                    if (!refusal::selected(addr))
                        return prosper::test::readback_persistent_color_target(
                            addr, surface.w, surface.h, format, materialized, error);
                    refusal::Record record{};
                    record.context.caller = refusal::Caller::ComputeSnapshot;
                    record.context.frontend_gpu_valid = surface.gpu_valid
                        ? refusal::ObservedBool::Yes : refusal::ObservedBool::No;
                    record.context.frontend_cpu_pixels = surface.rgba
                        ? refusal::ObservedBool::Yes : refusal::ObservedBool::No;
                    const bool result = prosper::test::readback_persistent_color_target(
                        addr, surface.w, surface.h, format, materialized, error, 0, &record);
                    refusal::emit(record);
                    return result;
                }();
                if (readback_ok &&
                    materialized.size() == expected) {
                    surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                        std::move(materialized));
                } else {
                    static std::atomic<int> warned{0};
                    if (warned.fetch_add(1) < 24)
                        std::fprintf(stderr,
                                     "[rtt] live compute target readback failed: base=0x%llx "
                                     "extent=%ux%u error=%s\n",
                                     static_cast<unsigned long long>(addr), surface.w, surface.h,
                                     error.c_str());
                    return false;
                }
            }
            if (!surface.rgba || expected != surface.rgba->size()) return false;
            snapshot.width = surface.w;
            snapshot.height = surface.h;
            if (!prosper::frontend::live_target_pixel_format_from_vk(format, snapshot.format))
                return false;
            snapshot.pixels = surface.rgba;
            return true;
        });
    // Phase 2 of #1091 (#1095): with one shared device the compute backend can sample the renderer's
    // persistent image in place. Offer it only while gpu_valid proves that image is current. A CPU
    // snapshot may coexist as an ordered readback mirror; CPU-newer publications clear gpu_valid,
    // while guest GPU writes erase the cache entry before this callback runs (#780).
    // One dispatch can bind the same target through several descriptors, so pins are counted: the
    // compute backend releases once per successful import and the entry drops at zero.
    struct PinnedImport { uint32_t width, height; VkFormat format; uint32_t count; };
    static std::unordered_map<uint64_t, PinnedImport> pinned_imports;
    // gpu_replay registers a renderer from more than one entry point, so this can run twice in a
    // process. Pins are taken and released within a single dispatch, so the map is empty between
    // them; clear it anyway so a second registration cannot inherit counts for a cache that no
    // longer holds those entries.
    pinned_imports.clear();
    const bool direct_bind = !PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_RTT_BIND");
    prosper::gpu::set_live_target_image_importer(
        [invalidate_ds, direct_bind](uint64_t addr,
                                     const prosper::gpu::LiveTargetImageRequest& request,
                                     prosper::gpu::LiveTargetImageImport& import) {
            if (!direct_bind) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::DirectBindDisabled; return false; }
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            // `allow_depth` is set only for a proven one-component Float32 sample or Uint32 raw-bit
            // view of an ordinary 2D descriptor. Prefer the matching DS plane before consulting the
            // address-only color registry: guest allocations are reused, and a stale RttSurf at the
            // same base must not hide the newer persistent depth image. Persistent DS entries are
            // not evicted, and compute runs in ordered submit execution, so unlike the bounded color
            // cache this path needs no pin.
            if (request.allow_depth && request.width && request.height) {
                const prosper::test::PersistentDsSampled sampled =
                    prosper::test::find_persistent_ds_sampled(
                        addr, request.width, request.height,
                        request.render_scale ? request.render_scale : 1u,
                        request.normalized_sampling);
                const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
                if (sampled.image && ctx.ok) {
                    import.width = sampled.width;
                    import.height = sampled.height;
                    import.kind = prosper::gpu::LiveTargetImageImport::Kind::Depth;
                    import.native_format = static_cast<uint32_t>(sampled.format);
                    import.image = sampled.image->image;
                    import.device = ctx.dev;
                    import.layout = static_cast<uint32_t>(
                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
                    return true;
                }
            }
            auto it = g_rtt.find(addr);
            if (it == g_rtt.end()) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoRttEntry; return false; }
            RttSurf& surface = it->second;
            if (!surface.w || !surface.h) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::ZeroExtent; return false; }
            // image import is a 2D contract
            if (surface.volume_depth) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::VolumeDepth; return false; }
            const VkFormat format = prosper::test::backend_color_format(surface.format);
            // Map the backend format explicitly and fail closed. A direct bind hands the consumer a
            // real VkImage, so an unrecognized format must decline rather than be reported as rgba8:
            // the consumer would then build a mismatched view over the renderer's image.
            prosper::gpu::LiveTargetPixelFormat pixel_format;
            if (!prosper::frontend::live_target_pixel_format_from_vk(format, pixel_format)) {
                import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::UnmappedFormat;
                return false;
            }
            const uint32_t bytes_per_pixel = prosper::test::backend_color_bytes_per_pixel(format);
            if (!bytes_per_pixel) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::UnmappedFormat; return false; }
            const uint64_t texels = static_cast<uint64_t>(surface.w) * surface.h;
            if (texels > UINT64_MAX / bytes_per_pixel) {
                import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::TexelOverflow;
                return false;
            }
            const uint64_t expected = texels * bytes_per_pixel;
            const bool has_cpu_snapshot = surface.rgba && surface.rgba->size() == expected;
            if (!prosper::frontend::live_rtt_gpu_importable(surface.gpu_valid,
                                                             has_cpu_snapshot)) {
                // Split by what the renderer DOES hold. "The pixels are on the CPU" and "the entry
                // is an empty shell" have nothing in common: the first means a GPU-side handoff was
                // possible and was lost, the second means there is nothing to hand over.
                import.refusal = has_cpu_snapshot ? prosper::gpu::LiveTargetImageImport::Refusal::CpuOnlyAuthority
                                                  : prosper::gpu::LiveTargetImageImport::Refusal::NoAuthority;
                return false;
            }
            const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
            if (!ctx.ok) { import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoRenderContext; return false; }
            prosper::test::PersistentColorTargetImage* target =
                prosper::test::find_persistent_color_target(addr, surface.w, surface.h, format);
            if (!target || !target->image || target->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoPersistentImage;
                return false;
            }
            if (!prosper::test::pin_persistent_color_target(addr, surface.w, surface.h, format)) {
                import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::PinRefused;
                return false;
            }
            target->last_use = ++prosper::test::persistent_color_target_generation();
            // The unpin key is {addr, w, h, format}. A repeat import that disagreed with the
            // recorded key would corrupt the outstanding pin's release, so decline instead.
            PinnedImport& pin = pinned_imports[addr];
            if (pin.count &&
                (pin.width != surface.w || pin.height != surface.h || pin.format != format)) {
                prosper::test::unpin_persistent_color_target(addr, surface.w, surface.h, format);
                import.refusal = prosper::gpu::LiveTargetImageImport::Refusal::PinKeyMismatch;
                return false;
            }
            pin = {surface.w, surface.h, format, pin.count + 1};
            import.width = surface.w;
            import.height = surface.h;
            import.format = pixel_format;
            import.native_format = static_cast<uint32_t>(format);
            import.image = target->image;
            import.device = ctx.dev;
            import.layout = static_cast<uint32_t>(target->layout);
            import.transfer_dst = true;
            import.transfer_src = true;
            return true;
        },
        [](uint64_t addr) {
            auto pinned = pinned_imports.find(addr);
            if (pinned == pinned_imports.end()) return;
            PinnedImport& pin = pinned->second;
            prosper::test::unpin_persistent_color_target(addr, pin.width, pin.height, pin.format);
            if (--pin.count == 0) pinned_imports.erase(pinned);
        });
    prosper::gpu::set_live_target_image_destination_borrower(
        [invalidate_ds, direct_bind](uint64_t addr,
                        const prosper::gpu::LiveTargetImageDestinationRequest& request,
                        prosper::gpu::LiveTargetImageImport& destination) {
            if (!direct_bind) { destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::DirectBindDisabled; return false; }
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            auto it = g_rtt.find(addr);
            // A complete compute overwrite can be the first producer of this address. Its
            // destination lease does not read old pixels, and the completion notifier creates
            // the registry entry only after the device copy and guest writeback succeed. Keep
            // the existing-entry requirement for ordinary borrows, which have no such proof.
            static const bool unregistered_destination_disabled =
                std::getenv("PROSPER_NO_COMPUTE_RTT_UNREGISTERED_DEST") != nullptr;
            if (it == g_rtt.end() &&
                (!request.allow_create || unregistered_destination_disabled)) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoRttEntry;
                return false;
            }
            if (!request.width || !request.height) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::ZeroExtent;
                return false;
            }
            if (it != g_rtt.end() &&
                (it->second.w != request.width || it->second.h != request.height)) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::ExtentMismatch;
                return false;
            }
            const VkFormat format = prosper::frontend::live_target_pixel_format_vk(request.format);
            if (format == VK_FORMAT_UNDEFINED ||
                (it != g_rtt.end() &&
                 prosper::test::backend_color_format(it->second.format) != format)) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::FormatMismatch;
                return false;
            }
            const prosper::test::RenderVkCtx& ctx = prosper::test::render_vk_ctx();
            if (!ctx.ok) { destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoRenderContext; return false; }
            if (request.allow_create && request.device != static_cast<void*>(ctx.dev)) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::DeviceMismatch;
                return false;
            }
            auto* target = prosper::test::find_persistent_color_target(
                addr, request.width, request.height, format, false);
            if ((!target || !target->image) && request.allow_create) {
                bool allocation_failed = false;
                target = prosper::test::ensure_persistent_color_target_for_compute_overwrite(
                    addr, request.width, request.height, format, &allocation_failed);
                if (!target) {
                    destination.refusal = allocation_failed
                        ? prosper::gpu::LiveTargetImageImport::Refusal::DestinationAllocationFailed
                        : prosper::gpu::LiveTargetImageImport::Refusal::DestinationCreationRefused;
                    return false;
                }
            }
            if (!target || !target->image ||
                (target->layout == VK_IMAGE_LAYOUT_UNDEFINED &&
                 (!request.allow_create || !target->compute_overwrite_uninitialized))) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoPersistentImage;
                return false;
            }
            const bool fresh_uninitialized = target->layout == VK_IMAGE_LAYOUT_UNDEFINED;
            if (!prosper::test::pin_persistent_color_target_for_overwrite(
                    addr, request.width, request.height, format, fresh_uninitialized)) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::PinRefused;
                return false;
            }
            PinnedImport& pin = pinned_imports[addr];
            if (pin.count && (pin.width != request.width || pin.height != request.height ||
                              pin.format != format)) {
                prosper::test::unpin_persistent_color_target(addr, request.width,
                                                               request.height, format);
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::PinKeyMismatch;
                return false;
            }
            pin = {request.width, request.height, format, pin.count + 1};
            destination.width = request.width;
            destination.height = request.height;
            destination.format = request.format;
            destination.native_format = static_cast<uint32_t>(format);
            destination.image = target->image;
            destination.device = ctx.dev;
            destination.layout = static_cast<uint32_t>(fresh_uninitialized
                ? VK_IMAGE_LAYOUT_GENERAL : target->layout);
            destination.fresh_uninitialized = fresh_uninitialized;
            destination.transfer_dst = true; // persistent color images have TRANSFER_DST usage
            return true;
        },
        [](uint64_t addr, const prosper::gpu::LiveTargetImageImport& destination) {
            const VkFormat format = prosper::frontend::live_target_pixel_format_vk(destination.format);
            auto* target = prosper::test::find_persistent_color_target(
                addr, destination.width, destination.height, format, false);
            if (!target || target->image != destination.image) return;
            target->valid = false;
            auto it = g_rtt.find(addr);
            if (it != g_rtt.end() && it->second.w == destination.width &&
                it->second.h == destination.height &&
                prosper::test::backend_color_format(it->second.format) == format)
                it->second.gpu_valid = false;
        });
    prosper::gpu::set_live_target_image_written_notifier(
        [invalidate_ds](const prosper::gpu::LiveTargetImageWrite& write) {
            auto it = g_rtt.find(write.gpu_addr);
            const auto& residency_trace = rtt_residency_trace_selector();
            const bool trace_write = residency_trace.configured &&
                residency_trace.includes(write.gpu_addr);
            if (trace_write)
                std::fprintf(stderr,
                    "[rtt-residency] producer=compute addr=0x%llx extent=%ux%u "
                    "format=%u linear-bytes=%zu mirrored=%u fresh=%u prior=%u "
                    "prior-gpu-valid=%u\n",
                    (unsigned long long)write.gpu_addr, write.width, write.height,
                    static_cast<unsigned>(write.format),
                    write.linear_pixels ? write.linear_pixels->size() : 0u,
                    write.mirrored_image ? 1u : 0u,
                    write.fresh_uninitialized ? 1u : 0u,
                    it != g_rtt.end() ? 1u : 0u,
                    it != g_rtt.end() && it->second.gpu_valid ? 1u : 0u);
            // A prior completion in this dispatch may have drained and erased this CPU registry
            // entry. An exact pinned destination image still proves which allocation was copied;
            // older address-only mirrors retain the stricter existing-entry requirement.
            if (it == g_rtt.end() && !write.linear_pixels && !write.mirrored_image) return;
            // Name the write's format exhaustively. Reporting an unmapped format as RGBA8 does not
            // merely mislabel it: the mirror-identity check below then rejects a target the compute
            // dispatch really did write, the entry stays invalidated by the ordinary guest-write
            // path, and every later consumer samples guest bytes prosper never wrote. That is the
            // whole Syberia black-3D-menu defect, and #773 in a different format.
            const VkFormat format = prosper::frontend::live_target_pixel_format_vk(write.format);
            if (format == VK_FORMAT_UNDEFINED) return;
            if (write.fresh_uninitialized && !write.mirrored_image) return;
            if (it != g_rtt.end() && !write.linear_pixels &&
                !prosper::frontend::live_rtt_mirror_identity_matches(
                    it->first, it->second.w, it->second.h,
                    static_cast<uint32_t>(
                        prosper::test::backend_color_format(it->second.format)),
                    write.gpu_addr, write.width, write.height,
                    static_cast<uint32_t>(format)))
                return;
            if (write.mirrored_image) {
                auto* target = prosper::test::find_persistent_color_target(
                    write.gpu_addr, write.width, write.height, format, false);
                if (!target || target->image != write.mirrored_image) return;
            }

            // The compute backend also wrote exact guest bytes. Process that ordinary notification
            // first so every color/depth/view alias becomes stale, then restore only the persistent
            // image that received the queue-ordered device copy. If the image disappeared, leave the
            // invalidation in force and let the next graphics use rebuild from the correct guest bytes.
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            // CPU publication and device-mirror restoration are different events (#3727).
            // A linear snapshot means no device mirror received this result. The restore helper
            // only marks an existing image valid; it uploads nothing. Calling it here would discard
            // fresh CPU pixels and authorize stale device contents. This affects any title once
            // graphics has created the persistent image, including later frames in Sonic.
            // Publish the snapshot with GPU authority false so graphics uploads it on the next use.
            if (write.linear_pixels && !write.linear_pixels->empty()) {
                prosper::test::invalidate_persistent_color_target_dimension_aliases(
                    write.gpu_addr, write.width, write.height, format, 0u);
                RttSurf& published = g_rtt[write.gpu_addr];
                published.rgba = write.linear_pixels;
                published.has_uniform_color = false;
                published.w = write.width;
                published.h = write.height;
                published.volume_depth = 0;
                published.format = format;
                published.guest_format = format;
                published.gpu_valid = false;
                if (trace_write)
                    std::fprintf(stderr,
                                 "[rtt-residency] addr=0x%llx result=cpu-only\n",
                                 (unsigned long long)write.gpu_addr);
                return;
            }
            if (write.mirrored_image) {
                auto* target = prosper::test::find_persistent_color_target(
                    write.gpu_addr, write.width, write.height, format, false);
                if (!target || target->image != write.mirrored_image) return;
            }
            if (!prosper::test::restore_persistent_color_target_after_mirrored_write(
                    write.gpu_addr, write.width, write.height, format,
                    write.fresh_uninitialized)) {
                return;
            }
            RttSurf& published = g_rtt[write.gpu_addr];
            published.rgba.reset();
            published.has_uniform_color = false;
            published.w = write.width;
            published.h = write.height;
            published.volume_depth = 0;
            published.format = format;
            published.guest_format = format;
            published.gpu_valid = true;
            if (trace_write)
                std::fprintf(stderr,
                             "[rtt-residency] addr=0x%llx result=gpu-valid\n",
                             (unsigned long long)write.gpu_addr);
        });
    prosper::gpu::set_live_target_byte_range_reader(
        [invalidate_ds](uint64_t addr, uint32_t bytes, std::vector<uint8_t>& output) {
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            if (!addr || !bytes) return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
            // A retained 3D attachment has not been published in the guest's native tiled byte
            // order. Check its entire physical footprint first, including reads beginning just
            // before the base. Returning NotFound would make ordered DMA copy stale guest bytes.
            if (any_volume_target(g_rtt, [&](uint64_t base, const RttSurf& surface) {
                    return unpublished_volume_may_overlap(
                        base, cpu_rtt_guest_write_bytes(surface, nullptr), addr, bytes);
                }))
                return prosper::gpu::LiveTargetByteReadResult::InvalidRange;
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
                    surface.has_uniform_color)
                    materialize_uniform_rtt(surface);
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
    // Register unconditionally, as for RTT seeds above: the interactive F9 bundle has no capture
    // environment variable and snapshots persistent depth/stencil state only after it is armed.
    // Normal rendering pays nothing because the callback is otherwise never invoked (#1307).
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
            surface.volume_depth = 0;
            surface.format = format;
            // Capture seeds contain canonical backend pixels. Older capture versions do not carry
            // the producing CB_COLOR order, so retain the exact format they do name.
            surface.guest_format = format;
            surface.rgba = std::make_shared<const std::vector<uint8_t>>(seed.rgba);
            surface.has_uniform_color = false;
            surface.gpu_valid = false;
            surface.dcc_metadata_dirty = false;
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
                                  PROSPER_ENV_VALUE("PROSPER_RTT_SINGLE_TARGET") == nullptr;
    static const bool rtt_on = getenv("PROSPER_RTT") != nullptr || pertarget;
    static const bool timeline_capture_requested =
        getenv("PROSPER_GPU_TIMELINE_CAPTURE") != nullptr;
    static const bool timeline_capture_phase_gated = timeline_capture_requested &&
        prosper::gpu::gpu_timeline_capture_is_after_compute_gated();
    static const bool timeline_capture_permits_live_targets =
        timeline_capture_allows_persistent_targets(
            timeline_capture_requested, timeline_capture_phase_gated);
    // Retain intermediate color targets on the GPU by default. Per-target pixel diagnostics require
    // authoritative CPU pixels at every pass, so they retain the established readback path. The
    // one-shot PROSPER_GPU_CAPTURE does not: it reads referenced targets back on demand for its one
    // selected submit, so the run is the production path until the capture moment (#3895; the
    // decision and its rationale live in capture_renderer_policy.hpp).
    // PROSPER_GPU_CAPTURE_READBACK=1 restores the pre-#3895 behaviour (a capture run reads back from
    // boot) as a same-binary A/B arm. The explicit opt-out keeps a direct A/B and a recovery switch
    // for driver issues.
    // Bundle replay normally requests CPU-visible RTT checkpoints so its inspection/export modes can
    // observe every pass. That fallback cannot carry MRT2..7 between independent render groups (the
    // public CPU seed parameters historically cover only MRT0/MRT1), and therefore is not an oracle
    // for a deferred frame that accumulates a wider G-buffer. This explicit diagnostic makes replay
    // use the production GPU-resident contract while still restoring the bundle's boundary seeds.
    // It is deliberately opt-in: export/readback diagnostics keep their established behaviour.
    static const bool replay_live_targets =
        PROSPER_ENV_VALUE("PROSPER_GPU_REPLAY_LIVE_TARGETS") != nullptr;
    static const bool gpu_capture_requested = getenv("PROSPER_GPU_CAPTURE") != nullptr;
    static const bool legacy_capture_readback =
        gpu_capture_requested && PROSPER_ENV_VALUE("PROSPER_GPU_CAPTURE_READBACK") != nullptr;
    static const prosper::frontend::LiveColorTargetResidencyInputs live_target_inputs = [&] {
        prosper::frontend::LiveColorTargetResidencyInputs in;
        in.per_target = pertarget;
        in.opted_out = PROSPER_ENV_VALUE("PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS") != nullptr ||
                       legacy_capture_readback;
        in.timeline_capture_permits = timeline_capture_permits_live_targets;
        in.gpu_capture_requested = gpu_capture_requested;
        in.replay_export_rtt = PROSPER_ENV_VALUE("PROSPER_GPU_REPLAY_EXPORT_RTT") != nullptr;
        in.replay_rtt_seeds = getenv("PROSPER_GPU_REPLAY_RTT_SEEDS") != nullptr;
        in.replay_live_targets = replay_live_targets;
        in.per_pass_pixel_diagnostic =
            PROSPER_ENV_VALUE("PROSPER_DUMP_SAMPLED_RTT") || PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS") ||
            getenv("PROSPER_DUMP_RTGROUPS_RGBA") || PROSPER_ENV_VALUE("PROSPER_DUMP_DRAWSTEPS") ||
            PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") ||
            PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_DIM") || PROSPER_ENV_VALUE("PROSPER_RTTLOG");
        return in;
    }();
    static const bool live_gpu_targets =
        prosper::frontend::live_color_targets_enabled(live_target_inputs);
    // #3891 diagnostic-path-active: name the switches that turned the production path off.
    static const bool diagnostic_path_noted = [&] {
        namespace perf = prosper::diagnostics::perf;
        uint64_t mask = 0;
        if (!live_gpu_targets) {
            const auto& in = live_target_inputs;
            auto bit = [&](perf::DiagnosticPathSwitch sw) {
                mask |= 1ull << static_cast<unsigned>(sw);
            };
            if (legacy_capture_readback) bit(perf::DiagnosticPathSwitch::CaptureReadback);
            else if (in.opted_out) bit(perf::DiagnosticPathSwitch::NoLiveTargets);
            if (in.replay_export_rtt || (in.replay_rtt_seeds && !in.replay_live_targets))
                bit(perf::DiagnosticPathSwitch::ReplayExport);
            if (in.per_pass_pixel_diagnostic) bit(perf::DiagnosticPathSwitch::PerPassPixelDump);
            if (!mask) bit(perf::DiagnosticPathSwitch::Other);   // RTT_SINGLE_TARGET, timeline
        }
        perf::set(perf::Gauge::DiagnosticPathSwitches, mask);
        return true;
    }();
    (void)diagnostic_path_noted;
    if (gpu_capture_requested)
        fprintf(stderr, "[render] PROSPER_GPU_CAPTURE: %s\n",
                live_gpu_targets ? "live targets retained; capture readback is on demand"
                : legacy_capture_readback ? "CPU readback path (PROSPER_GPU_CAPTURE_READBACK)"
                                          : "CPU readback path (another diagnostic requires it)");
    if (timeline_capture_phase_gated)
        fprintf(stderr, "[render] phase-gated timeline capture retains live targets; "
                        "capture readback is on demand\n");
    static const bool defer_intermediate_scanout = live_gpu_targets &&
        !PROSPER_ENV_VALUE("PROSPER_NO_INTERMEDIATE_SCANOUT_DEFER");
    // Ordered passes share one queue and keep their attachments GPU-resident, so submit the callback's
    // command buffers in one batch by default. Keep a direct recovery/A-B switch for driver issues.
    static const bool batch_backend_submits = live_gpu_targets &&
        !PROSPER_ENV_VALUE("PROSPER_NO_BACKEND_BATCH_SUBMITS");
    if (live_gpu_targets)
        fprintf(stderr, "[render] persistent GPU color targets enabled (experimental%s)\n",
                replay_live_targets ? ", replay parity" : "");
    if (batch_backend_submits)
        fprintf(stderr, "[render] backend target-submit batching enabled (experimental)\n");
    prosper::gpu::set_deferred_graphics_retirer(
        [] { (void)prosper::test::retire_deferred_backend_submission(); });
    // Every registration shares process state; taking this reference initializes only empty slots.
    auto* const callback_state = &CallbackState::instance();
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps, invalidate_ds, fragment_wave_policy,
         callback_state](const std::vector<prosper::gpu::DrawItem>& items,
                               uint32_t w, uint32_t h) -> prosper::gpu::RenderedFrame {
            const uint64_t callback_capture_generation =
                prosper::perf::interactive_performance_capture().active_generation();
            const bool perf_capture_timing = callback_capture_generation != 0;
            const uint64_t callback_entry_ns = perf_capture_timing
                ? prosper::perf::monotonic_now_ns() : 0;
            // #2215 instrument: publish which thread is inside a submit-render callback right
            // now, so the thread sampler can attribute its samples EXACTLY instead of guessing
            // from a six-frame host-stack walk (a sample taken deep in ucrtbase loses the
            // renderer frame off the end of that walk and is misfiled as "outside").
            //
            // Measured: 2.92 submits/flip x 181.8 ms = 530 ms/flip inside these callbacks
            // against 938 ms/flip wall -- so ~43% of the collapsed frame is time when this
            // callback is NOT running, and nothing has ever instrumented that half.
            //
            // One slot, not a set: the renderer already documents a single-present-thread
            // contract (see g_rtt and the dp_submit statics above). A second concurrent caller
            // would overwrite it, so the reader treats a mismatch as "outside" rather than
            // asserting -- a wrong attribution is better than a crash in a diagnostic.
            prosper::frontend::ScopedRendererCallbackTid scoped_renderer_tid;
            // #3948 stage 2: a previous span's graphics batch may still be pending (its wait was
            // deferred past a dispatch). Everything below reads or mutates renderer state its
            // completion publishes, so retire it first.
            prosper::test::retire_deferred_backend_submission();
            // Compute fast-clears update a target's DCC metadata between graphics spans. Associate
            // those ranges before draining the ordered guest-write notifications so a metadata write
            // cannot leave the previous frame's retained target authoritative.
            register_cpu_rtt_dcc_metadata(g_rtt, items);
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            const prosper::gpu::LiveRenderPhase phase = prosper::gpu::live_render_phase();
            static const auto& write_watch_promotion_budget_bytes = callback_state->write_watch_promotion_budget_bytes();
            auto& callback_thread_state = CallbackThreadState::current();
            static thread_local auto& write_watch_promotion_budget = callback_thread_state.write_watch_promotion_budget();
            if (phase.first_span)
                write_watch_promotion_budget.reset(write_watch_promotion_budget_bytes);
            static thread_local auto& pinned_scanouts = callback_thread_state.pinned_scanouts();
            static thread_local auto& pinned_renderer_mip_targets = callback_thread_state.pinned_renderer_mip_targets();
            auto release_pinned_scanouts = [&] {
                for (const PinnedScanout& target : pinned_scanouts)
                    prosper::test::unpin_persistent_color_target(
                        target.id, target.width, target.height, target.format);
                pinned_scanouts.clear();
            };
            auto release_consumed_renderer_mip_targets = [&] {
                auto target = pinned_renderer_mip_targets.begin();
                while (target != pinned_renderer_mip_targets.end()) {
                    if (!target->consumed) {
                        ++target;
                        continue;
                    }
                    prosper::test::unpin_persistent_color_target(
                        target->id, target->width, target->height, target->format);
                    target = pinned_renderer_mip_targets.erase(target);
                }
            };
            // A terminal callback normally releases every pin. Recover conservatively if an earlier
            // submit was aborted after rendering but before frontend finalization.
            if (phase.first_span && !pinned_scanouts.empty()) release_pinned_scanouts();
            if (phase.first_span && !pinned_renderer_mip_targets.empty())
                release_consumed_renderer_mip_targets();
            // PROSPER_RENDER_FIRST=<N>: skip the slow (~400x) Vulkan render for the first N GPU submits, so
            // the game reaches a LATE scene (e.g. the level1 cutscene, which only starts submitting after
            // ~5000 title-loop submits) at native speed before we begin rendering/dumping. Returning {}
            // means "not rendered this submit". Without this, rendering from boot is far too slow to ever
            // reach a post-loading-screen scene.
            static auto& g_submit_idx = callback_state->g_submit_idx();
            static auto& g_render_first = callback_state->g_render_first();
            // PROSPER_RENDER_DELAY_MS=<N>: wall-clock warmup for titles whose useful scene begins after
            // a variable number of submits. The guest and command decoder keep running at native speed;
            // only the synchronous Vulkan work is skipped. The clock starts at the first GPU submit.
            static const auto& g_render_delay_ms = callback_state->g_render_delay_ms();
            static const auto& g_render_delay_start = callback_state->g_render_delay_start();
            static auto& g_render_delay_announced = callback_state->g_render_delay_announced();
            // PROSPER_RENDER_LAST=<N>: stop rendering after submit N (default: unbounded). Bounds the render
            // window so a diagnostic slice at a late stall (RENDER_FIRST..RENDER_LAST) does not accumulate
            // unbounded RTT/GPU resources across tens of thousands of submits (which OOM-kills the process).
            static auto& g_render_last = callback_state->g_render_last();
            static thread_local auto& g_this_submit = callback_thread_state.g_this_submit();
            static thread_local auto& g_force_this_submit = callback_thread_state.g_force_this_submit();
            if (phase.first_span || g_this_submit < 0) {
                g_this_submit = g_submit_idx++;
                g_force_this_submit = false;
            }
            if (g_this_submit > g_render_last) return {};
            static const auto& g_rttlog_min_submit = callback_state->g_rttlog_min_submit();
            static const auto& g_rttlog_max_submit = callback_state->g_rttlog_max_submit();
            scan_for_descriptors_once(static_cast<uint64_t>(g_this_submit < 0 ? 0 : g_this_submit));
            const bool rtt_log_in_range =
                g_this_submit >= g_rttlog_min_submit && g_this_submit <= g_rttlog_max_submit;
            const bool rtt_log = PROSPER_ENV_VALUE("PROSPER_RTTLOG") && rtt_log_in_range;
            static thread_local auto& pending_timing = callback_thread_state.pending_timing();
            static thread_local auto& pending_rtt_timing = callback_thread_state.pending_rtt_timing();
            static thread_local auto& pending_span_start_ns = callback_thread_state.pending_span_start_ns();
            static thread_local auto& pending_capture_generation = callback_thread_state.pending_capture_generation();
            // The timing bool is sampled per callback, not a capture ID or completion proof.
            // Keep cumulative per-thread populations across captures; a falling edge is only
            // an observed inactive callback. This diagnostic does not change realization policy.
            static const auto& validation_census_requested = callback_state->validation_census_requested();
            ValidationCensusLog* validation_census = nullptr;
            if (validation_census_requested) {
                static thread_local auto& log = callback_thread_state.validation_census_log();
                if (log.active && !perf_capture_timing)
                    log.data.report(stderr, log.thread, "timing-inactive");
                log.active = perf_capture_timing;
                if (perf_capture_timing) {
                    if (!log.thread) {
#ifdef _WIN32
                        log.thread = static_cast<unsigned long>(GetCurrentThreadId());
#else
                        log.thread = static_cast<unsigned long>((uintptr_t)pthread_self());
#endif
                    }
                    validation_census = &log;
                }
            }
            const bool timing_capture_only =
                getenv("PROSPER_RENDER_TIMING_CAPTURE_ONLY") != nullptr;
            const bool timing_log_enabled = getenv("PROSPER_RENDER_TIMING") != nullptr &&
                (!timing_capture_only || perf_capture_timing);
            const prosper::frontend::PerformanceTimingMode timing_mode =
                prosper::frontend::performance_timing_mode(timing_log_enabled,
                                                            perf_capture_timing);
            const bool timing_enabled = timing_mode.measure;
            // Read ONCE per span. The guard below sits inside the per-RESOURCE loop, where a
            // getenv costs a locked environment scan for every resource of every draw -- it was
            // the single hottest line in #2215's profile (26 of 349 primary-thread samples).
            //
            // Deliberately a LIVE read rather than PROSPER_ENV_VALUE: tests arm this variable at
            // runtime, so it has to stay observable. Hoisting removes the per-resource cost
            // WITHOUT changing that -- which is why it is the better fix here than caching.
            const char* const render_timing_env = getenv("PROSPER_RENDER_TIMING");
            const bool render_timing_detail =
                ((render_timing_env && strcmp(render_timing_env, "detail") == 0 &&
                  (!timing_capture_only || perf_capture_timing)) ||
                 (perf_capture_timing &&
                  getenv("PROSPER_RENDER_TIMING_DETAIL_CAPTURE") != nullptr));
            // render_runner.h cannot depend on the app capture singleton: it is also compiled into
            // standalone Vulkan tests. This thread-local scope activates its backend clocks only
            // inside this production callback and restores the prior state on every return path.
            prosper::frontend::ScopedInteractivePerformanceTiming scoped_perf_timing(
                perf_capture_timing);
            const bool lightweight_rtt_timing = timing_mode.log && PROSPER_ENV_VALUE("PROSPER_RTT_TIMING");
            static const auto& rtt_timing_min_draws = callback_state->rtt_timing_min_draws();
            if (phase.first_span) {
                pending_capture_generation = timing_enabled ? callback_capture_generation : 0;
                pending_span_start_ns = pending_capture_generation ? callback_entry_ns : 0;
                if (timing_enabled) {
                    pending_timing = {};
                    pending_rtt_timing.clear();
                }
            }
            const auto callback_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            BackendTimingContext backend_timing_ctx{
                .pending_timing = pending_timing,
                .timing_mode = timing_mode};
            SubmitLogContext log_submit_index_ctx{
                .items = items,
                .phase = phase,
                .g_this_submit = g_this_submit};
            log_submit_index(log_submit_index_ctx);
            bool force_target = false;
            DiagnosticTargetContext diagnostic_target_forces_render_ctx{
                .items = items,
                .force_target = force_target};
            diagnostic_target_forces_render(diagnostic_target_forces_render_ctx);
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
            ShaderDumpContext dump_first_item_spirv_ctx{
                .items = items};
            dump_first_item_spirv(dump_first_item_spirv_ctx);
            DccClearContext materialize_dirty_dcc_clears_ctx{
                .g_rtt = g_rtt,
                .items = items,
                .pending_timing = pending_timing,
                .timing_enabled = timing_enabled};
            materialize_dirty_dcc_clears(materialize_dirty_dcc_clears_ctx);
            // Keep decoded texture storage alive across callbacks. The old clear()+emplace(size, 0)
            // released and zero-filled tens of MiB every submit even though the decode paths overwrite
            // all pixels. Reusing same-sized slots avoids both costs; short guest reads explicitly clear
            // their uncovered tail below so stale scratch bytes can never become sampled pixels.
            static thread_local auto& texstore = callback_thread_state.texstore();
            // Parallel to `texstore`: a slot is pinned while a retained submit-scoped identity entry
            // still points into it. Handing a pinned slot to a later span's decode would overwrite the
            // very bytes that entry promises, so the allocator below skips them and the pins are
            // released together when the submit's identity map is rebuilt.
            static thread_local auto& texstore_pinned = callback_thread_state.texstore_pinned();
            size_t texstore_used = 0;
            // PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE=1 restores the pre-#1691 span-scoped lifetime on
            // the same build, so a routed A/B measures the change and not two compilers' luck.
            //
            // PROSPER_RESOURCE_HASH_DIM does the same, for a different reason: it correlates each
            // decode's raw and sampled hashes with that range's writers, and it reports from inside
            // the decode path. Retaining an identity across a span boundary would silently drop
            // correlation points a previous build emitted, which would make the instrument disagree
            // with itself across builds while investigating exactly the kind of question it exists
            // for. Keep its output identical to pre-#1691 rather than make it cheaper.
            static const auto& submit_decode_scope_disabled = callback_state->submit_decode_scope_disabled();
            static thread_local auto& decoded_textures = callback_thread_state.decoded_textures();
            static thread_local auto& decode_span_ordinal = callback_thread_state.decode_span_ordinal();
            ++decode_span_ordinal;
            // `phase.first_span` is the normal submit boundary, but it is not the only one: the
            // warmup/window gates above (PROSPER_RENDER_FIRST, PROSPER_RENDER_DELAY_MS,
            // PROSPER_RENDER_LAST) return before this point, so a submit can reach here first at a
            // later span. Tracking the submit ordinal as well keeps the rebuild exact under those
            // recipes — without it the pins and the LRU generation would sit frozen across the whole
            // warmup. Retained entries were never unsafe there (a foreign submit serial makes the
            // journal query Unknown, which refuses reuse before dereferencing anything), but stale
            // state that outlives its submit is not something to leave resting on that.
            static thread_local auto& decode_scope_submit = callback_thread_state.decode_scope_submit();
            const bool rebuild_decode_scope = phase.first_span || submit_decode_scope_disabled ||
                g_this_submit != decode_scope_submit;
            decode_scope_submit = g_this_submit;
            const bool use_direct_buffer_views =
                PROSPER_ENV_VALUE("PROSPER_NO_FRONTEND_BUFFER_VIEW") == nullptr;
            // DrawItem retains the widened index words until this synchronous renderer callback
            // returns. The backend consumes them into its host-visible upload before then, so an
            // intermediate BackendDraw vector copy has no ownership or completion role. Keep the
            // owned-vector route as a same-binary control and compatibility fallback.
            const bool use_direct_index_views =
                PROSPER_ENV_VALUE("PROSPER_NO_DIRECT_INDEX_VIEW") == nullptr;
            static const auto& use_tracked_buffer_membership_cache = callback_state->use_tracked_buffer_membership_cache();
            static auto& persistent_decoded_textures = callback_state->persistent_decoded_textures();
            static auto& persistent_decoded_texture_bytes = callback_state->persistent_decoded_texture_bytes();
            static auto& persistent_decode_generation = callback_state->persistent_decode_generation();
            // The persistent cache's LRU generation now advances per SUBMIT rather than per span. That
            // is what keeps a submit-scoped identity entry safe: an entry whose `last_use` equals the
            // current generation is skipped by the eviction scan, so persistent-cache storage a
            // retained pointer refers to cannot be freed under it later in the same submit.
            if (rebuild_decode_scope) ++persistent_decode_generation;
            const uint64_t decode_generation = persistent_decode_generation;
            static thread_local auto& retired_submit_pixels = callback_thread_state.retired_submit_pixels();
            static auto& retired_submit_bytes = callback_state->retired_submit_bytes();
            if (rebuild_decode_scope) {
                decoded_textures.clear();
                texstore_pinned.assign(texstore.size(), false);
                retired_submit_pixels.clear();
                retired_submit_bytes = 0;
            }
            static auto& persistent_texture_id = callback_state->persistent_texture_id();
            static auto& persistent_validation_scratch = callback_state->persistent_validation_scratch();
            static const auto& persistent_decode_limit = callback_state->persistent_decode_limit();
            uint32_t resource_hash_w = 0, resource_hash_h = 0;
            if (const char* dim = PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &resource_hash_w, &resource_hash_h) != 2)
                    resource_hash_w = resource_hash_h = 0;
            uint32_t target_step_w = 0, target_step_h = 0;
            if (const char* dim = PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_DIM"))
                if (sscanf(dim, "%ux%u", &target_step_w, &target_step_h) != 2)
                    target_step_w = target_step_h = 0;
            const size_t target_step_min_draws = PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_MIN_DRAWS")
                ? std::max<long>(2, atol(PROSPER_ENV_VALUE("PROSPER_TARGET_STEP_HASH_MIN_DRAWS"))) : 2;
            prosper::frontend::RttInjectionCache rtt_injection_cache;
            // Build one draw's set-tagged resources from its VS (set 0) + PS (set 1) tables — read the
            // bytes from 1:1-mapped guest memory, detile textures. (Each constant/vertex buffer + texture
            // gets its own binding; the recompiler declared a storage buffer / image sampler at each.)
            // #2256. `validate_spirv_descriptor_interface` is memoized inside shader_resources.cpp,
            // but its HIT path is O(module size) twice -- an FNV-1a walk of every word to build the
            // key, then a full vector equality compare to confirm it -- under a process-global lock,
            // and build_R calls it twice per draw. Measured on a Blue Prince gameplay submit
            // (PPSA25009, 2,103 draws): 30.37 ms/submit, 53% of build_R and 9.3% of the frame,
            // traversing 27.2 million words. 3,231 calls served 523 distinct shaders, so 84% of that
            // work was re-deriving a key for a module already reflected.
            //
            // The caller already holds a cheap stable key. DrawItem::vs_identity/fs_identity come
            // from the exact shader-recompile cache as `cache.next_identity++`
            // (src/gpu/execute/gpu_executor.cpp:1255) -- a monotonic counter that is NEVER reused, so an
            // identity denotes one module for the life of the process even across cache eviction
            // (an evicted shader recompiles to a NEW identity, which misses here and is refilled).
            //
            // This changes no semantics. The upstream cache already returns its stored report for a
            // matching module regardless of which runtime table was passed, so the report was
            // already a pure function of the SPIR-V; and this call site reads only
            // `report.descriptors` (via find_spirv_descriptor_binding), never `report.issues`.
            //
            // Bounded at the same 4,096 entries the shader cache itself uses. On overflow the map is
            // cleared wholesale rather than evicted by age: there is no LRU bookkeeping to pay for on
            // a path this hot, and 523 distinct shaders per submit leaves ample headroom.
            //
            // The overflow behaviour does NOT degrade gracefully, and `memo_clears` should be read
            // accordingly: a title exceeding the cap would clear and refill repeatedly, paying the
            // full reflection cost again PLUS the map churn. So a non-zero `memo_clears` means this
            // optimisation has STOPPED WORKING, not that a threshold was brushed -- do not read a
            // small non-zero value as benign. Observed 0 across a 13-minute Blue Prince route.
            //
            // The entry copies the whole DescriptorValidationReport although this path reads only
            // `descriptors`. Deliberate: storing the subset would bound the footprint to what is
            // used, but it would also make the entry a lie about what it holds the moment a caller
            // reads `issues`, and at 523 entries against a 4,096 cap the footprint is not the
            // constraint. If that ever inverts, narrow the entry rather than raising the cap.
            static thread_local auto& reflect_memo = callback_thread_state.reflect_memo();
            // Read once per submit. Poison mode mutates the completed resource vector, so it keeps
            // the compatibility representation where every resource is a FrameResource. Ordinary
            // rendering stores buffers in the compact carrier and records their original order.
            const char* const descriptor_validate_mode = getenv("PROSPER_DESCRIPTOR_VALIDATE");
            const bool compact_buffer_resources =
                getenv("PROSPER_NO_COMPACT_BUFFER_RESOURCES") == nullptr &&
                (!descriptor_validate_mode || strcmp(descriptor_validate_mode, "poison") != 0);
            // These vectors are rebuilt for every draw. A bounded hint avoids repeated
            // growth for ordinary reflected resource sets without allocating for draws
            // that have no accepted resources. The control permits same-binary timing.
            static const auto& reserve_frame_resources = callback_state->reserve_frame_resources();
            // Immutable CPU snapshots live only for this renderer callback (not later spans or
            // frames). Every lookup still proves the exact retained images/generations. Sharing the
            // owner across draws also lets one backend group deduplicate its uploads by pointer.
            DepthArraySnapshotCensus depth_array_census{std::getenv("PROSPER_DEPTH_ARRAY_SNAPSHOT_CENSUS") != nullptr};
            std::vector<DepthArraySnapshot> depth_array_snapshots;
            size_t depth_array_snapshot_bytes = 0;
            bool depth_array_snapshot_admitted = false;
            const bool reuse_depth_arrays_across_draws =
                std::getenv("PROSPER_NO_SUBMIT_DEPTH_ARRAY_SNAPSHOT_REUSE") == nullptr;
            const bool compact_depth_array_snapshots =
                std::getenv("PROSPER_NO_COMPACT_DEPTH_ARRAY_SNAPSHOT") == nullptr;
            // PROSPER_NO_GPU_DEPTH_ARRAY=1 restores the CPU readback + re-upload bridge.
            const bool gpu_depth_array_snapshots =
                prosper::test::gpu_depth_array_snapshots_enabled();
            // PROSPER_NO_GUEST_DEPTH_LAYERS=1 restores the pre-#3893 refusal of a retained depth
            // array with never-rendered layers (same-binary A/B).
            static const auto& disable_guest_depth_layers = callback_state->disable_guest_depth_layers();
            // Callback-scoped guest-layer scans (#3893), dropped when a guest GPU write drains.
            prosper::test::DepthArrayGuestScanMemo depth_array_guest_scans;
            uint64_t depth_array_guest_scan_epoch =
                guest_gpu_write_drain_epoch().load(std::memory_order_relaxed);
            // PROSPER_NO_GPU_DEPTH_CUBE=1 restores the CPU readback + quantise + re-upload bridge
            // for fully renderer-owned depth cubes (tests/fixtures/retained_depth_cube_gpu.h).
            // PROSPER_DS_UNBRIDGED_FAR is a diagnostic that fills unbridged retained planes in
            // the decode path; keep its cubes on that path so the lever keeps its meaning.
            const bool gpu_depth_cube_snapshots =
                prosper::test::gpu_depth_cube_snapshots_enabled() &&
                !PROSPER_ENV_VALUE("PROSPER_DS_UNBRIDGED_FAR");
            // Callback-local like depth_array_snapshots: a snapshot is served again only to a
            // consumer recorded into the batch that carries its gather, and only while the exact
            // six retained images and their depth-write generations are unchanged.
            std::vector<DepthCubeGpuSnapshot> depth_cube_gpu_snapshots;
            DrawResourceContext draw_resource_ctx{
                .g_rtt = g_rtt,
                .g_pass_log_submit = g_pass_log_submit,
                .invalidate_ds = invalidate_ds,
                .frame_no = frame_no,
                .rtt_on = rtt_on,
                .live_gpu_targets = live_gpu_targets,
                .write_watch_promotion_budget = write_watch_promotion_budget,
                .pinned_renderer_mip_targets = pinned_renderer_mip_targets,
                .g_this_submit = g_this_submit,
                .rtt_log = rtt_log,
                .pending_timing = pending_timing,
                .validation_census = validation_census,
                .timing_enabled = timing_enabled,
                .render_timing_detail = render_timing_detail,
                .texstore = texstore,
                .texstore_pinned = texstore_pinned,
                .texstore_used = texstore_used,
                .decoded_textures = decoded_textures,
                .decode_span_ordinal = decode_span_ordinal,
                .use_direct_buffer_views = use_direct_buffer_views,
                .use_tracked_buffer_membership_cache = use_tracked_buffer_membership_cache,
                .persistent_decoded_textures = persistent_decoded_textures,
                .persistent_decoded_texture_bytes = persistent_decoded_texture_bytes,
                .decode_generation = decode_generation,
                .retired_submit_pixels = retired_submit_pixels,
                .retired_submit_bytes = retired_submit_bytes,
                .persistent_texture_id = persistent_texture_id,
                .persistent_validation_scratch = persistent_validation_scratch,
                .persistent_decode_limit = persistent_decode_limit,
                .resource_hash_w = resource_hash_w,
                .resource_hash_h = resource_hash_h,
                .rtt_injection_cache = rtt_injection_cache,
                .reflect_memo = reflect_memo,
                .compact_buffer_resources = compact_buffer_resources,
                .reserve_frame_resources = reserve_frame_resources,
                .depth_array_census = depth_array_census,
                .depth_array_snapshots = depth_array_snapshots,
                .depth_array_snapshot_bytes = depth_array_snapshot_bytes,
                .depth_array_snapshot_admitted = depth_array_snapshot_admitted,
                .reuse_depth_arrays_across_draws = reuse_depth_arrays_across_draws,
                .compact_depth_array_snapshots = compact_depth_array_snapshots,
                .gpu_depth_array_snapshots = gpu_depth_array_snapshots,
                .disable_guest_depth_layers = disable_guest_depth_layers,
                .depth_array_guest_scans = depth_array_guest_scans,
                .depth_array_guest_scan_epoch = depth_array_guest_scan_epoch,
                .gpu_depth_cube_snapshots = gpu_depth_cube_snapshots,
                .depth_cube_gpu_snapshots = depth_cube_gpu_snapshots};
            // Diagnostic shader/state overrides (REFVS, TESTPS, FS_SPV, NOPS, SKIP_DRAW, ...): see
            // load_shader_overrides. Computed once per callback, applied to every draw item.
            ShaderOverrides shader_overrides = load_shader_overrides();
            BackendDrawContext backend_draw_ctx{
                .fragment_wave_policy = fragment_wave_policy,
                .phase = phase,
                .rtt_log = rtt_log,
                .pending_timing = pending_timing,
                .timing_enabled = timing_enabled,
                .use_direct_index_views = use_direct_index_views,
                .descriptor_validate_mode = descriptor_validate_mode,
                .draw_resource_ctx = draw_resource_ctx,
                .overrides = shader_overrides};
            // Declared beside pass_timing_start, not at the loop's end: the group loop lives in an
            // inner scope that closes before the pass span is accumulated, so a tail marker declared
            // there is out of scope where it is needed.
            RenderClock::time_point pass_tail_start{};
            double pass_tail_measured_before = 0.0;
            const auto pass_timing_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            if (timing_enabled)
                pending_timing.prelude_ms += std::chrono::duration<double, std::milli>(
                    pass_timing_start - callback_timing_start).count();
            std::shared_ptr<const std::vector<uint8_t>> selected_pixels;
            uint64_t selected_source_submit = 0;
            // Provenance of whatever `selected_pixels` ends up holding. It travels with the frame to
            // the present layer so a verification gate can tell a composited frame from a
            // republished guest scanout — `--require-composited-frame` asserts prosper rendered
            // something, and #2026 was reverted (#2044) for letting the one frame that assertion
            // exists to reject satisfy it.
            auto frame_origin = prosper::gpu::PresentFrameOrigin::Composited;
            bool published_gpu = false;
            // The extent contract with our caller (#1986). A submit headed for the publish gate is
            // published only as exactly w*h*4 bytes and anything else is silently discarded, so a
            // rendered pass of a different extent is not a present source at all — see
            // present_extent.hpp for what that changes, and PresentSubmitScope in gpu_execute.hpp for
            // why the answer cannot be derived here: render_submit_items (gpu_replay's ordered-prefix
            // inspection, the renderer's own tests) deliberately wants the last pass AT ITS OWN
            // EXTENT, and is indistinguishable from a present by anything visible in this callback.
            // Zero means "no contract", which is that path and not a degenerate case.
            const size_t present_extent_bytes = prosper::gpu::present_submit_in_progress()
                ? static_cast<size_t>(w) * h * 4u : 0u;
            // The chosen source and each candidate's own extent, so the report below can name what was
            // available rather than only that nothing was.
            prosper::frontend::PresentSourceChoice present_choice =
                prosper::frontend::PresentSourceChoice::None;
            uint32_t px_front_w = 0, px_front_h = 0;
            uint32_t px_vo_w = 0, px_vo_h = 0;
            uint32_t px_last_w = 0, px_last_h = 0;
            uint64_t px_front_base = 0, px_vo_base = 0, px_last_base = 0;
            // Provenance belongs to each candidate's immutable pixels, not the final callback.
            // A retained image can be served by a later submit, and its source must travel with it.
            uint64_t px_front_source_submit = 0, px_vo_source_submit = 0,
                     px_last_source_submit = 0;
            // The raw CB_COLOR0_INFO.FORMAT of the pass that produced each present candidate, so the
            // SELECTED one's format can be reported (#2283). UINT32_MAX is "no candidate recorded",
            // deliberately not 0: 0 is CB_COLOR_INVALID, a real and meaningful value here, and the
            // whole question is how often it reaches the publish source. Absent and disabled must
            // not be the same number.
            uint32_t px_front_fmt = kNoPassFormat, px_vo_fmt = kNoPassFormat,
                     px_last_fmt = kNoPassFormat;
            if (pertarget) {
                PerTargetPassContext per_target_ctx{
                    .g_rtt = g_rtt,
                    .g_pass_log_submit = g_pass_log_submit,
                    .g_pass_log_window = g_pass_log_window,
                    .g_persist_window = g_persist_window,
                    .g_persist_filter = g_persist_filter,
                    .g_persist_extent = g_persist_extent,
                    .frame_no = frame_no,
                    .live_gpu_targets = live_gpu_targets,
                    .defer_intermediate_scanout = defer_intermediate_scanout,
                    .batch_backend_submits = batch_backend_submits,
                    .items = items,
                    .w = w,
                    .h = h,
                    .phase = phase,
                    .pinned_scanouts = pinned_scanouts,
                    .pinned_renderer_mip_targets = pinned_renderer_mip_targets,
                    .g_this_submit = g_this_submit,
                    .rtt_log = rtt_log,
                    .pending_timing = pending_timing,
                    .pending_rtt_timing = pending_rtt_timing,
                    .timing_enabled = timing_enabled,
                    .lightweight_rtt_timing = lightweight_rtt_timing,
                    .texstore = texstore,
                    .texstore_pinned = texstore_pinned,
                    .texstore_used = texstore_used,
                    .decoded_textures = decoded_textures,
                    .resource_hash_w = resource_hash_w,
                    .resource_hash_h = resource_hash_h,
                    .target_step_w = target_step_w,
                    .target_step_h = target_step_h,
                    .target_step_min_draws = target_step_min_draws,
                    .pass_tail_start = pass_tail_start,
                    .pass_tail_measured_before = pass_tail_measured_before,
                    .pass_timing_start = pass_timing_start,
                    .selected_pixels = selected_pixels,
                    .selected_source_submit = selected_source_submit,
                    .present_extent_bytes = present_extent_bytes,
                    .present_choice = present_choice,
                    .px_front_w = px_front_w,
                    .px_front_h = px_front_h,
                    .px_vo_w = px_vo_w,
                    .px_vo_h = px_vo_h,
                    .px_last_w = px_last_w,
                    .px_last_h = px_last_h,
                    .px_front_base = px_front_base,
                    .px_vo_base = px_vo_base,
                    .px_last_base = px_last_base,
                    .px_front_source_submit = px_front_source_submit,
                    .px_vo_source_submit = px_vo_source_submit,
                    .px_last_source_submit = px_last_source_submit,
                    .px_front_fmt = px_front_fmt,
                    .px_vo_fmt = px_vo_fmt,
                    .px_last_fmt = px_last_fmt,
                    .backend_draw_ctx = backend_draw_ctx,
                    .backend_timing_ctx = backend_timing_ctx};
                render_per_target_passes(per_target_ctx);
            } else {
                SingleFramebufferContext single_framebuffer_ctx{
                    .items = items,
                    .w = w,
                    .h = h,
                    .phase = phase,
                    .pending_timing = pending_timing,
                    .timing_enabled = timing_enabled,
                    .backend_timing_ctx = backend_timing_ctx,
                    .backend_draw_ctx = backend_draw_ctx,
                    .selected_pixels = selected_pixels,
                    .selected_source_submit = selected_source_submit};
                render_single_framebuffer(single_framebuffer_ctx);
                if (rtt_on && !selected_pixels->empty()) {
                    uint64_t tgt = 0;
                    for (const auto& it : items) if (it.color0_base) { tgt = it.color0_base; break; }
                    if (tgt) {
                        RttSurf& s = g_rtt[tgt];
                        s.rgba = selected_pixels; s.w = w; s.h = h;
                        s.volume_depth = 0;
                        s.format = VK_FORMAT_R8G8B8A8_UNORM; s.gpu_valid = false;
                        s.guest_format = items.empty()
                            ? VK_FORMAT_R8G8B8A8_UNORM
                            : static_cast<VkFormat>(
                                  prosper::frontend::mrt_raw_format(items.front(), 0));
                        s.has_uniform_color = false;
                        s.dcc_metadata_dirty = false;
                        prosper::test::invalidate_persistent_color_target(tgt);
                    }
                    if (rtt_log) { size_t nz=0;
                        for (uint8_t byte : *selected_pixels) nz += byte != 0;
                        fprintf(stderr, "[rtt] store target=0x%llx (%zu items, color0s:", (unsigned long long)tgt, items.size());
                        for (const auto& it : items) fprintf(stderr, " 0x%llx", (unsigned long long)it.color0_base);
                        fprintf(stderr, ") px_nonzero=%zu cache_size=%zu\n", nz, g_rtt.size()); }
                }
            }
            if (timing_enabled)
                pending_timing.pass_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - pass_timing_start).count();
            if (timing_enabled)
                pending_timing.pass_tail_ms +=
                    std::chrono::duration<double, std::milli>(
                        RenderClock::now() - pass_tail_start).count() -
                    (pending_timing.build_resources_ms + pending_timing.backend_ms -
                     pass_tail_measured_before);
            // Ordered submits may invoke this callback for several graphics spans separated by
            // compute dispatches. Intermediate spans only update g_rtt. At the final span, recover
            // the flipped scanout from that persistent cache even when it was rendered earlier in
            // the transaction, and advance frame/dump state exactly once.
            const auto output_copy_start = timing_enabled && phase.final_span
                ? RenderClock::now() : RenderClock::time_point{};
            if (phase.final_span && pertarget) {
                FinalSpanPresentContext final_span_present_ctx{
                    .g_rtt = g_rtt,
                    .w = w,
                    .h = h,
                    .selected_pixels = selected_pixels,
                    .selected_source_submit = selected_source_submit,
                    .frame_origin = frame_origin,
                    .published_gpu = published_gpu,
                    .present_extent_bytes = present_extent_bytes,
                    .present_choice = present_choice,
                    .px_front_w = px_front_w,
                    .px_front_h = px_front_h,
                    .px_vo_w = px_vo_w,
                    .px_vo_h = px_vo_h,
                    .px_last_w = px_last_w,
                    .px_last_h = px_last_h,
                    .px_front_base = px_front_base,
                    .px_vo_base = px_vo_base,
                    .px_last_base = px_last_base,
                    .px_front_fmt = px_front_fmt,
                    .px_vo_fmt = px_vo_fmt,
                    .px_last_fmt = px_last_fmt};
                select_final_span_present(final_span_present_ctx);
            }
            if (timing_enabled && phase.final_span)
                pending_timing.output_copy_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - output_copy_start).count();
            if (phase.final_span) {
                release_pinned_scanouts();
                release_consumed_renderer_mip_targets();
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
                prosper::frontend::reset_performance_timing_after_span(
                    pending_timing, timing_enabled, phase.final_span);
                return {};
            }
            static const std::vector<uint8_t> empty_pixels;
            const std::vector<uint8_t>& px = selected_pixels ? *selected_pixels : empty_pixels;
            int n = frame_no++;
            PresentedFrameDumpContext dump_presented_frame_ctx{
                .frame_dir = frame_dir,
                .dump_bmps = dump_bmps,
                .w = w,
                .h = h,
                .published_gpu = published_gpu,
                .px = px,
                .n = n};
            dump_presented_frame(dump_presented_frame_ctx);
            if (timing_enabled) {
                pending_timing.callbacks++;
                pending_timing.total_ms += std::chrono::duration<double, std::milli>(
                    RenderClock::now() - callback_timing_start).count();
                // One record per complete semantic submit. A submit may be split into several
                // graphics spans by interleaved compute/DMA; recording every callback would count
                // the growing pending total repeatedly and make the capture itself misattribute time.
                if (perf_capture_timing && phase.final_span &&
                    prosper::frontend::complete_renderer_span_belongs_to_capture(
                        pending_capture_generation, callback_capture_generation)) {
                    RendererTimingRecordContext publish_renderer_timing_record_ctx{
                        .pending_timing = pending_timing,
                        .pending_span_start_ns = pending_span_start_ns,
                        .pending_capture_generation = pending_capture_generation};
                    publish_renderer_timing_record(publish_renderer_timing_record_ctx);
                }
                if (!timing_mode.log) {
                    // F8 consumes the structured record above, then leaves without touching the
                    // lifetime aggregates or their large periodic stderr summaries.
                    prosper::frontend::reset_performance_timing_after_span(
                        pending_timing, timing_enabled, phase.final_span);
                    pending_span_start_ns = 0;
                    pending_capture_generation = 0;
                    prosper::gpu::RenderedFrame frame(std::move(selected_pixels));
                    frame.origin = frame_origin;
                    frame.source_submit = selected_source_submit;
                    return frame;
                }
                RenderTimingReportContext timing_report_ctx{
                    .g_rtt = g_rtt,
                    .pending_timing = pending_timing,
                    .texstore = texstore,
                    .submit_decode_scope_disabled = submit_decode_scope_disabled,
                    .persistent_decoded_textures = persistent_decoded_textures,
                    .persistent_decoded_texture_bytes = persistent_decoded_texture_bytes,
                    .persistent_validation_scratch = persistent_validation_scratch};
                report_render_timing_aggregates(timing_report_ctx);
                // The complete semantic submit has now been recorded and added to aggregate logs.
                // Intermediate spans take the earlier return and retain this same population.
                prosper::frontend::reset_performance_timing_after_span(
                    pending_timing, timing_enabled, phase.final_span);
            }
            pending_span_start_ns = 0;
            pending_capture_generation = 0;
            prosper::gpu::RenderedFrame frame(std::move(selected_pixels));
            frame.origin = frame_origin;
            frame.source_submit = selected_source_submit;
            return frame;
        });
    fprintf(stderr, "[render] live Vulkan submit renderer registered (dump=%d, frames -> %s)\n",
            (int)dump_bmps, frame_dir.c_str());
}

} // namespace prosper::frontend

// Guest flip -> "is there a frame to publish?", for a title that composites with COMPUTE and never
// draws. Defined here because the decision and the publish both live in this translation unit, and
// REGISTERED with core rather than named by it: core must not reference a frontend symbol, or every
// tool that links prosper_core without a frontend fails to link (it did, on Mach-O, where a weak
// declaration with no definition is still an undefined symbol).
extern "C" void prosper_frontend_flip_publish_guest_scanout(uint64_t flip) {
    if (prosper::frontend::g_flip_scanout_hook) prosper::frontend::g_flip_scanout_hook(flip);
}

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
#include "shared/live/submit_renderer/guest_reads.hpp"    // safe_span / safe_copy / safe_equal
#include "shared/live/submit_renderer/draw_resources.hpp" // build_draw_frame_resources (#3892)
#include "shared/live/submit_renderer/timing_report.hpp"  // report_render_timing_aggregates (#3892)
#include "shared/live/submit_renderer/backend_timing.hpp" // record_backend_timing_stats, print_rtt_timing (#3892)
#include "shared/live/submit_renderer/backend_draws.hpp"  // build_backend_draws, clear_for (#3892)
#include "shared/live/submit_renderer/per_target_passes.hpp" // render_per_target_passes (#3892)
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
bool parse_diagnostic_address(const char* spec, uint64_t& address) {
    if (!spec) return true;
    if (spec[0] != '0' || (spec[1] != 'x' && spec[1] != 'X') || !spec[2]) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(spec, &end, 16);
    if (errno || end == spec + 2 || *end || !parsed) return false;
    address = parsed;
    return true;
}

bool parse_diagnostic_extent(const char* spec, uint32_t& width, uint32_t& height) {
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
                            const std::string& title_id) {
    // Keep the legacy global disable authoritative for every frontend, including callers with their
    // own explicit opt-in such as PROSPER_APP_DUMP_FRAMES.
    const bool dump_bmps = frame_dump_request_allowed(
        dump_bmps_requested, PROSPER_ENV_VALUE("PROSPER_NO_FRAME_DUMPS"));
    // Titles whose WaveAny-only fragment programs may run at the host's native Wave32.
    //
    // The classifier (render_runner.h) admits a reason set of EXACTLY kFragmentWaveReasonWaveAny --
    // no lane identity, no ballot, no shuffle, no scalar reduction. What this list controls is which
    // titles that classifier is trusted on, and it is a list rather than a switch for a measured
    // reason: every entry has a before/after survey on a reviewed route on this project's hardware,
    // and joining it is one run (`tools/shader_inspect/skip_survey.py --from-snapshots`).
    //
    // Why not simply always-on, which is what #3464 W5 first proposed. The surveyed set is NOT a
    // random sample of the corpus -- it is the titles that have snapshot routes, which biases toward
    // simple, mature fragment work. That is exactly the population least likely to contain the
    // shader this classifier still mis-admits, because the "control-flow only" property the safety
    // argument rests on is NOT what the classifier tests (it tests the reason set, and the vote taint
    // that would raise ScalarReduce has known escapes: `sel()` drops the taint `bsel()` propagates,
    // and a vote routed into VCC is read per-lane by v_cndmask_b32 with no marker). Until the
    // classifier tests the property its safety argument claims, each title is admitted on evidence.
    //
    // Keeping it title-keyed also preserves an invariant an always-on switch silently broke: tests
    // and gpu_replay call register_live_renderer without a title_id (live_renderer.hpp:156 defaults
    // it to {}), so they match nothing and keep the strict exact-width contract. gpu_replay gates on
    // expected_output_hash, so admitting there would have moved the project's offline oracle.
    //
    // Measured on Windows/NVIDIA, reviewed routes (#3464, PR #3480). Modules reporting exactly
    // WaveAny, and how many of them are PROVABLY width-independent:
    //   PPSA01885 Evergate         11 of 11 (title + gameplay routes; the title screen's missing
    //                                        3D content is restored, confirmed by eye)
    //   PPSA02664 Alex Kidd DX       8 of 8  (title screen confirmed correct by eye; its WORLD has
    //                                        a separate, unrelated defect -- #3479)
    //   PPSA13579 Blasphemous 2      1 of 8
    //   PPSA25009 Blue Prince       18 of 22
    //   PPSA04263 Grand Theft Auto V         the original reviewed bank route
    //
    // Two of those titles are on the list although most or some of their modules do NOT clear. That
    // is safe BECAUSE the decision is per module: under THIS tier the 11 that cannot be proved keep
    // the exact-width contract and are refused individually. Admitting a title no longer means trusting
    // all of its shaders, which is what made a title the wrong unit before. (The partial-wave tier
    // below admits those 11 on its own, different argument; measured on all five titles, #3797.)
    //
    // NOT on this list: PPSA21564 (6 measured refusals, no dump and no after-arm yet). One survey
    // run away, and with the per-module proof the run is confirmation rather than a gamble.
    static const char* const kNativeFragmentVoteTitles[] = {
        "PPSA01885", "PPSA02664", "PPSA04263", "PPSA13579", "PPSA25009",
    };
    // The opt-out exists so the A/B stays reproducible, following PROSPER_NO_GUEST_FS: a default-on
    // behaviour whose disable switch is for bisection, not for routine use.
    const bool native_fragment_vote_width =
        !PROSPER_ENV_VALUE("PROSPER_STRICT_FRAGMENT_WAVE_WIDTH") &&
        std::any_of(std::begin(kNativeFragmentVoteTitles), std::end(kNativeFragmentVoteTitles),
                    [&](const char* id) { return title_id == id; });
    // The partial-wave tier (#3464, rdna2_to_spirv.hpp kFragmentWavePartialWaveExactReasons) runs on
    // every title above. Its argument does not depend on the title, but each title was still measured
    // before joining, because on these titles it admits exactly the modules the vote tier's proof
    // declined. Same binary, tier on vs PROSPER_NO_PARTIAL_WAVE_FRAGMENT=1, fresh saves, the reviewed
    // snapshot routes on Windows/NVIDIA (32..32):
    //   PPSA04263 GTA V          188 refused -> 0; the bank world renders instead of black (#3794)
    //   PPSA25009 Blue Prince      4 refused -> 0 on the hall route; the bouquet and the edge lens
    //                              fringe return and blue-prince-hall passes (it FAILS with the tier
    //                              off: 13,609 colours against a 40,000 floor) (#3797)
    //   PPSA01885 Evergate         4 refused -> 0 on the gameplay route; the save-slot portals show
    //                              their world preview instead of being empty (#3797)
    //   PPSA13579 Blasphemous 2    7 refused -> 0 (about 5% of gameplay draws, at a 640x360 target); the
    //                              composite does not change beyond frame-to-frame noise (#3797)
    //   PPSA02664 Alex Kidd DX     0 refused either way; the tier changes no admission (#3797)
    // Every guard passes with the tier on. It stays a title list, not a rule: the other titles have
    // no before/after, and a title joins with one survey run.
    //
    // Two switches turn it off: PROSPER_NO_PARTIAL_WAVE_FRAGMENT isolates THIS tier for an A/B on one
    // binary (the renderer then behaves exactly as before the tier existed), and
    // PROSPER_STRICT_FRAGMENT_WAVE_WIDTH, which disables both tiers through native_fragment_vote_width.
    const bool partial_wave_fragment = native_fragment_vote_width &&
        !PROSPER_ENV_VALUE("PROSPER_NO_PARTIAL_WAVE_FRAGMENT");
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
                if (prosper::test::readback_persistent_color_target(
                        addr, surface.w, surface.h, format, materialized, error) &&
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
            if (it == g_rtt.end()) { destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::NoRttEntry; return false; }
            if (!request.width || !request.height) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::ZeroExtent;
                return false;
            }
            if (it->second.w != request.width || it->second.h != request.height) {
                destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::ExtentMismatch;
                return false;
            }
            const VkFormat format = prosper::frontend::live_target_pixel_format_vk(request.format);
            if (format == VK_FORMAT_UNDEFINED ||
                prosper::test::backend_color_format(it->second.format) != format) {
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
                target = prosper::test::ensure_persistent_color_target_for_compute_overwrite(
                    addr, request.width, request.height, format);
                if (!target) {
                    destination.refusal = prosper::gpu::LiveTargetImageImport::Refusal::
                        DestinationCreationRefused;
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
    prosper::gpu::set_submit_renderer(
        [frame_dir, dump_bmps, invalidate_ds, native_fragment_vote_width,
         partial_wave_fragment](const std::vector<prosper::gpu::DrawItem>& items,
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
            // Compute fast-clears update a target's DCC metadata between graphics spans. Associate
            // those ranges before draining the ordered guest-write notifications so a metadata write
            // cannot leave the previous frame's retained target authoritative.
            register_cpu_rtt_dcc_metadata(g_rtt, items);
            drain_guest_gpu_writes(g_rtt, invalidate_ds);
            const prosper::gpu::LiveRenderPhase phase = prosper::gpu::live_render_phase();
            static const size_t write_watch_promotion_budget_bytes = [] {
                // 0 is not "off": an empty budget makes WriteWatchPromotionBudget::try_consume
                // return true unconditionally, i.e. unbounded arming per submit. A typo must keep
                // the default rather than select that (#3253).
                const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB");
                const uint64_t mib = prosper::diag::env_u64_or_default_capped(
                    "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", value, 8ull,
                    SIZE_MAX / (1024ull * 1024ull), "MiB");
                return static_cast<size_t>(mib * (1024ull * 1024ull));
            }();
            static thread_local prosper::frontend::WriteWatchPromotionBudget
                write_watch_promotion_budget;
            if (phase.first_span)
                write_watch_promotion_budget.reset(write_watch_promotion_budget_bytes);
            static thread_local std::vector<PinnedScanout> pinned_scanouts;
            static thread_local std::vector<PinnedRendererMipTarget>
                pinned_renderer_mip_targets;
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
            static std::atomic<int> g_submit_idx{0};
            static int g_render_first = getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0;
            // PROSPER_RENDER_DELAY_MS=<N>: wall-clock warmup for titles whose useful scene begins after
            // a variable number of submits. The guest and command decoder keep running at native speed;
            // only the synchronous Vulkan work is skipped. The clock starts at the first GPU submit.
            static const int64_t g_render_delay_ms = getenv("PROSPER_RENDER_DELAY_MS")
                ? std::max<int64_t>(0, atoll(PROSPER_ENV_VALUE("PROSPER_RENDER_DELAY_MS"))) : 0;
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
                ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MIN_SUBMIT"))) : 0;
            static const int g_rttlog_max_submit = getenv("PROSPER_RTTLOG_MAX_SUBMIT")
                ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MAX_SUBMIT"))) : INT_MAX;
            scan_for_descriptors_once(static_cast<uint64_t>(g_this_submit < 0 ? 0 : g_this_submit));
            const bool rtt_log_in_range =
                g_this_submit >= g_rttlog_min_submit && g_this_submit <= g_rttlog_max_submit;
            const bool rtt_log = PROSPER_ENV_VALUE("PROSPER_RTTLOG") && rtt_log_in_range;
            static thread_local RenderTiming pending_timing;
            static thread_local std::vector<RttTimingRecord> pending_rtt_timing;
            static thread_local uint64_t pending_span_start_ns = 0;
            static thread_local uint64_t pending_capture_generation = 0;
            // The timing bool is sampled per callback, not a capture ID or completion proof.
            // Keep cumulative per-thread populations across captures; a falling edge is only
            // an observed inactive callback. This diagnostic does not change realization policy.
            static const bool validation_census_requested =
                PROSPER_ENV_VALUE("PROSPER_TEXTURE_VALIDATION_CENSUS") != nullptr;
            ValidationCensusLog* validation_census = nullptr;
            if (validation_census_requested) {
                static thread_local ValidationCensusLog log;
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
            static const uint64_t rtt_timing_min_draws = getenv("PROSPER_RTT_TIMING_MIN_DRAWS")
                ? strtoull(PROSPER_ENV_VALUE("PROSPER_RTT_TIMING_MIN_DRAWS"), nullptr, 0) : 0;
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
            auto record_backend_timing = [&]
                (const prosper::test::BackendRenderTimingStats& backend,
                 const prosper::test::BackendTextureUploadStats& textures,
                 const prosper::test::BackendPipelineCacheStats& pipelines,
                 const prosper::test::BackendResourceReuseStats& reuse) {
                return record_backend_timing_stats(backend_timing_ctx, backend, textures, pipelines, reuse);
            };
            // PROSPER_SUBMITLOG: print the GPU-submit index periodically (at native speed, before the slow
            // render) so it can be correlated with guest-side log lines (e.g. a MsgDialog wait) to find the
            // exact submit at which a scene appears — for aiming PROSPER_RENDER_FIRST at it.
            if (phase.first_span && PROSPER_ENV_ON("PROSPER_SUBMITLOG") && (g_this_submit % 1000 == 0))
                fprintf(stderr, "[submit] index=%d (%zu draw items)\n", g_this_submit, items.size());
            if (const char* sd = PROSPER_ENV_VALUE("PROSPER_SUBMITLOG_DIM")) {
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
            if (const char* td = PROSPER_ENV_VALUE("PROSPER_RENDER_TARGET_DIM")) {
                uint32_t tw = 0, th = 0;
                if (sscanf(td, "%ux%u", &tw, &th) == 2)
                    for (const auto& it : items)
                        if (it.color0_width == tw && it.color0_height == th) { force_target = true; break; }
            }
            if (const char* rd = PROSPER_ENV_VALUE("PROSPER_RENDER_RESOURCE_DIM")) {
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
            if (PROSPER_ENV_ON("PROSPER_SHADER_DUMP") && !items.empty()) {
                std::string d = PROSPER_ENV_VALUE("PROSPER_SHADER_DUMP");
                const auto& dump_vs = items[0].vs_words();
                const auto& dump_fs = items[0].fs_words();
                if (FILE* f = fopen((d + "/frame_vs.spv").c_str(), "wb")) { fwrite(dump_vs.data(), 4, dump_vs.size(), f); fclose(f); }
                if (FILE* f = fopen((d + "/frame_fs.spv").c_str(), "wb")) { fwrite(dump_fs.data(), 4, dump_fs.size(), f); fclose(f); }
                fprintf(stderr, "[render] dumped SPIR-V vs=%zu fs=%zu dwords\n", dump_vs.size(), dump_fs.size()); fflush(stderr);
            }
            // A uniform DCC clear is self-contained in metadata. If the ordered compute span dirtied
            // a retained target's metadata, materialize that clear now so the following graphics pass
            // loads the clear value rather than stale pixels (or an arbitrary fallback clear).
            const auto dcc_materialize_start = timing_enabled
                ? RenderClock::now() : RenderClock::time_point{};
            uint64_t dcc_materialize_surfaces = 0;
            uint64_t dcc_materialize_bytes = 0;
            for (const auto& item : items) {
                const prosper::gpu::ShaderResourceTable* tables[] = {
                    item.vrt.get(), item.prt.get(),
                };
                for (const auto* table : tables) {
                    if (!table) continue;
                    for (const auto& resource : table->resources) {
                        auto found = g_rtt.find(resource.gpu_addr);
                        if (found == g_rtt.end() || !found->second.dcc_metadata_dirty ||
                            !resource.compression_enabled ||
                            resource.metadata_addr != found->second.dcc_metadata_addr ||
                            resource.width != found->second.w ||
                            resource.height != found->second.h)
                            continue;
                        const uint64_t metadata_bytes =
                            prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
                        if (!metadata_bytes || metadata_bytes > SIZE_MAX) continue;
                        std::vector<uint8_t> metadata(static_cast<size_t>(metadata_bytes));
                        size_t copied = 0;
                        if (resource.dcc_metadata_host_data) {
                            copied = static_cast<size_t>(std::min<uint64_t>(
                                metadata.size(), resource.dcc_metadata_host_data_size));
                            std::memcpy(metadata.data(), resource.dcc_metadata_host_data, copied);
                        } else {
                            copied = safe_copy(metadata.data(), resource.metadata_addr,
                                               metadata.size());
                        }
                        const uint64_t texels = static_cast<uint64_t>(found->second.w) *
                                                found->second.h;
                        const VkFormat format = prosper::test::backend_color_format(
                            found->second.format);
                        const uint32_t bpp =
                            prosper::test::backend_color_bytes_per_pixel(format);
                        if (copied != metadata.size() || !bpp || texels > SIZE_MAX / bpp)
                            continue;
                        uint8_t clear_rgba[4]{};
                        if (!prosper::gpu::gfx10_dcc_fast_clear_rgba8(
                                clear_rgba, 1, metadata.data(), metadata.size(),
                            resource.num_components, resource.alpha_is_on_msb))
                            continue;
                        if (format != VK_FORMAT_R8G8B8A8_UNORM &&
                            format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                            format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
                            continue;
                        }
                        found->second.rgba.reset();
                        found->second.has_uniform_color = true;
                        for (uint32_t channel = 0; channel < 4; ++channel)
                            found->second.uniform_color[channel] =
                                clear_rgba[channel] ? 1.0f : 0.0f;
                        // PROSPER_DCCLOG=1 -- diagnostic only, no behaviour change. A surface
                        // materialised from a DCC fast-clear code becomes a UNIFORM colour for the
                        // whole target, so if this decode is wrong the entire frame is one wrong
                        // colour with no content -- which is exactly Little Nightmares III's
                        // uniform-yellow presents (#2014). Deduped per (address, decoded colour) so a
                        // run costs a handful of lines.
                        if (const char* dcclog = PROSPER_ENV_VALUE("PROSPER_DCCLOG")) {
                            if (dcclog[0] == '1' && dcclog[1] == '\0') {
                                static std::mutex dcc_mutex;
                                static std::set<std::pair<uint64_t, uint32_t>> dcc_seen;
                                const uint32_t packed = (uint32_t)clear_rgba[0] |
                                    ((uint32_t)clear_rgba[1] << 8) |
                                    ((uint32_t)clear_rgba[2] << 16) |
                                    ((uint32_t)clear_rgba[3] << 24);
                                bool first = false;
                                {
                                    std::lock_guard<std::mutex> lock(dcc_mutex);
                                    first = dcc_seen.emplace((uint64_t)found->first, packed).second;
                                }
                                if (first)
                                    fprintf(stderr,
                                            "[dcclog] addr=0x%llx %ux%u fmt=%d ncomp=%u "
                                            "alpha_msb=%d clear_rgba=(%u,%u,%u,%u) -> uniform=(%.0f,%.0f,%.0f,%.0f)\n",
                                            (unsigned long long)found->first,
                                            found->second.w, found->second.h, (int)format,
                                            resource.num_components, (int)resource.alpha_is_on_msb,
                                            clear_rgba[0], clear_rgba[1], clear_rgba[2], clear_rgba[3],
                                            found->second.uniform_color[0], found->second.uniform_color[1],
                                            found->second.uniform_color[2], found->second.uniform_color[3]);
                            }
                        }
                        found->second.dcc_metadata_dirty = false;
                        ++dcc_materialize_surfaces;
                        dcc_materialize_bytes += sizeof(found->second.uniform_color);
                    }
                }
            }
            if (timing_enabled) {
                pending_timing.dcc_materialize_ms +=
                    std::chrono::duration<double, std::milli>(
                        RenderClock::now() - dcc_materialize_start).count();
                pending_timing.dcc_materialize_surfaces += dcc_materialize_surfaces;
                pending_timing.dcc_materialize_bytes += dcc_materialize_bytes;
            }
            // Keep decoded texture storage alive across callbacks. The old clear()+emplace(size, 0)
            // released and zero-filled tens of MiB every submit even though the decode paths overwrite
            // all pixels. Reusing same-sized slots avoids both costs; short guest reads explicitly clear
            // their uncovered tail below so stale scratch bytes can never become sampled pixels.
            static thread_local std::vector<std::vector<uint8_t>> texstore;
            // Parallel to `texstore`: a slot is pinned while a retained submit-scoped identity entry
            // still points into it. Handing a pinned slot to a later span's decode would overwrite the
            // very bytes that entry promises, so the allocator below skips them and the pins are
            // released together when the submit's identity map is rebuilt.
            static thread_local std::vector<bool> texstore_pinned;
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
            static const bool submit_decode_scope_disabled =
                PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE") != nullptr ||
                PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") != nullptr;
            static thread_local std::unordered_map<TextureDecodeKey, DecodedTexture,
                                                   TextureDecodeKeyHash> decoded_textures;
            static thread_local uint64_t decode_span_ordinal = 0;
            ++decode_span_ordinal;
            // `phase.first_span` is the normal submit boundary, but it is not the only one: the
            // warmup/window gates above (PROSPER_RENDER_FIRST, PROSPER_RENDER_DELAY_MS,
            // PROSPER_RENDER_LAST) return before this point, so a submit can reach here first at a
            // later span. Tracking the submit ordinal as well keeps the rebuild exact under those
            // recipes — without it the pins and the LRU generation would sit frozen across the whole
            // warmup. Retained entries were never unsafe there (a foreign submit serial makes the
            // journal query Unknown, which refuses reuse before dereferencing anything), but stale
            // state that outlives its submit is not something to leave resting on that.
            static thread_local int decode_scope_submit = -1;
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
            static const bool use_tracked_buffer_membership_cache =
                PROSPER_ENV_VALUE("PROSPER_NO_TRACKED_BUFFER_GATE") == nullptr;
            static std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>
                persistent_decoded_textures;
            static size_t persistent_decoded_texture_bytes = 0;
            static uint64_t persistent_decode_generation = 0;
            // The persistent cache's LRU generation now advances per SUBMIT rather than per span. That
            // is what keeps a submit-scoped identity entry safe: an entry whose `last_use` equals the
            // current generation is skipped by the eviction scan, so persistent-cache storage a
            // retained pointer refers to cannot be freed under it later in the same submit.
            if (rebuild_decode_scope) ++persistent_decode_generation;
            const uint64_t decode_generation = persistent_decode_generation;
            static thread_local std::vector<std::shared_ptr<const std::vector<uint8_t>>>
                retired_submit_pixels;
            static size_t retired_submit_bytes = 0;
            if (rebuild_decode_scope) {
                decoded_textures.clear();
                texstore_pinned.assign(texstore.size(), false);
                retired_submit_pixels.clear();
                retired_submit_bytes = 0;
            }
            static uint64_t persistent_texture_id = 0;
            static std::vector<uint8_t> persistent_validation_scratch;
            static const size_t persistent_decode_limit = [] {
                const uint64_t physical_bytes = host_physical_memory_bytes();
                const size_t limit = texture_decode_cache_limit_bytes(
                    PROSPER_ENV_VALUE("PROSPER_TEXTURE_DECODE_CACHE_MB"), physical_bytes);
                fprintf(stderr,
                        "[render] decoded texture cache budget = %.1f MiB "
                        "(host physical %.1f GiB)\n",
                        limit / (1024.0 * 1024.0),
                        physical_bytes / (1024.0 * 1024.0 * 1024.0));
                return limit;
            }();
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
            static thread_local std::unordered_map<uint64_t, ReflectMemoEntry> reflect_memo;
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
            static const bool reserve_frame_resources = [] {
                const char* setting = std::getenv("PROSPER_FRAME_RESOURCE_RESERVE");
                return !setting || std::strcmp(setting, "0") != 0;
            }();
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
            static const bool disable_guest_depth_layers =
                std::getenv("PROSPER_NO_GUEST_DEPTH_LAYERS") != nullptr;
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
            // Diagnostic shader/state overrides (computed once, applied to EVERY draw item):
            //   REFVS  -> a known-good fullscreen-triangle VS (isolates the game's real VS).
            //   TESTPS -> a solid-magenta PS (isolates VS geometry from PS shading). The optional
            //             TESTPS_MATCH file restricts it to one exact recompiled guest PS.
            //   FS_SPV -> a caller-supplied PS SPIR-V (e.g. a UV visualizer).
            //   NOPS   -> bypass the resolved pipeline state (default state).
            #include "refvs.inc"
            const bool refvs = PROSPER_ENV_VALUE("PROSPER_RENDER_REFVS");
            std::vector<uint32_t> refvs_spv(kRefVs, kRefVs + sizeof(kRefVs) / 4);
            std::vector<uint32_t> ps_override;
            bool ps_override_is_file = false;   // true only for a valid PROSPER_FS_SPV *file* override
            bool ps_override_is_test = false;
            if (PROSPER_ENV_ON("PROSPER_RENDER_TESTPS")) {
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
            if (const char* fsp = PROSPER_ENV_VALUE("PROSPER_FS_SPV")) {
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
            if (const char* mp = PROSPER_ENV_VALUE("PROSPER_FS_SPV_MATCH")) {
                fs_match_mode = load_spv_file(mp, fs_match) ? 1 : 2;
                if (fs_match_mode == 2)
                    fprintf(stderr, "[fs-match] PROSPER_FS_SPV_MATCH='%s' invalid/unreadable -> applying NO "
                            "fragment file override (fail closed)\n", mp);
            }
            // Optional second selector for a live exact-program experiment. A matching SPIR-V
            // module can be reused by unrelated draws, so a shader-only A/B is not necessarily
            // an exact draw substitution. Invalid input fails closed.
            const char* fs_guest_addr_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_GUEST_ADDR");
            uint64_t fs_guest_addr = 0;
            const bool fs_guest_addr_valid =
                parse_diagnostic_address(fs_guest_addr_text, fs_guest_addr);
            if (!fs_guest_addr_valid) {
                static std::atomic_flag warned = ATOMIC_FLAG_INIT;
                if (!warned.test_and_set())
                    std::fprintf(stderr,
                                 "[fs-match] PROSPER_FS_SPV_GUEST_ADDR invalid -> no file override\n");
            }
            const char* fs_target_addr_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_TARGET_ADDR");
            uint64_t fs_target_addr = 0;
            const bool fs_target_addr_valid =
                parse_diagnostic_address(fs_target_addr_text, fs_target_addr);
            if (!fs_target_addr_valid) {
                static std::atomic_flag warned = ATOMIC_FLAG_INIT;
                if (!warned.test_and_set())
                    std::fprintf(stderr,
                                 "[fs-match] PROSPER_FS_SPV_TARGET_ADDR invalid -> no file override\n");
            }
            const char* fs_target_dim_text = PROSPER_ENV_VALUE("PROSPER_FS_SPV_TARGET_DIM");
            uint32_t fs_target_width = 0, fs_target_height = 0;
            const bool fs_target_dim_valid =
                parse_diagnostic_extent(fs_target_dim_text, fs_target_width, fs_target_height);
            if (!fs_target_dim_valid) {
                static std::atomic_flag warned = ATOMIC_FLAG_INIT;
                if (!warned.test_and_set())
                    std::fprintf(stderr,
                                 "[fs-match] PROSPER_FS_SPV_TARGET_DIM invalid -> no file override\n");
            }
            // PROSPER_RENDER_TESTPS_MATCH=<file> is the geometry half of a per-shader A/B test: replace
            // only that exact guest PS with the known solid output while retaining its real VS, indices,
            // viewport, depth and raster state. As with FS_SPV_MATCH, a bad path fails closed.
            int testps_match_mode = 0;
            std::vector<uint32_t> testps_match;
            if (const char* mp = PROSPER_ENV_VALUE("PROSPER_RENDER_TESTPS_MATCH")) {
                testps_match_mode = load_spv_file(mp, testps_match) ? 1 : 2;
                if (testps_match_mode == 2)
                    fprintf(stderr, "[testps-match] PROSPER_RENDER_TESTPS_MATCH='%s' invalid/unreadable -> "
                            "applying NO test fragment override (fail closed)\n", mp);
            }
            const bool nops = PROSPER_ENV_VALUE("PROSPER_RENDER_NOPS");
            // Assemble backend draws for a subset of the submit's items — one BackendDraw per realized
            // DrawItem with its own resources + fixed-function state (or the diagnostic overrides above).
            // build_R reads the CURRENT g_rtt, so calling this AFTER an earlier target-group has been
            // rendered+stored lets the later group sample that group's pixels (a HIT, not empty memory).
            // PROSPER_SKIP_DRAW="N[,N...]" (diagnostic): drop these semantic draw_index values from
            // every pass — isolate whether a specific draw (e.g. a suspected opaque UI backdrop that
            // hides the composited world) is what corrupts the frame, without touching any state.
            static const char* skip_draws_env = getenv("PROSPER_SKIP_DRAW");
            // PROSPER_SKIP_DRAW_PROGRAM / PROSPER_DRAW_PROGRAM_CENSUS: decline draws by shader
            // PROGRAM identity, and enumerate the programs a title draws with. Both are process
            // singletons configured once from the environment (see draw_program_skip.hpp for the
            // contract and the four limits a reader of a skipped run cannot see in the output).
            // Hoisted here for the same reason descriptor_validate_mode is: the accessor is one
            // function-local-static test, but calling it per draw on a 2,100-draw submit is a
            // measurable cost for a variable nobody set.
            auto& program_skip = prosper::gpu::draw_program_skip_selector();
            const bool program_skip_armed = program_skip.armed();
            const bool program_census = prosper::gpu::draw_program_census_enabled();
            // PROSPER_DRAW_LINKSCAN: the graphics counterpart of PROSPER_COMPUTE_PARENTSCAN.
            // Hoisted for the same reason as the two above -- disarmed it is one bool.
            auto& link_scan = prosper::gpu::draw_link_scan_selector();
            const bool link_scan_armed = prosper::gpu::draw_link_scan_enabled();
            BackendDrawContext backend_draw_ctx{
                .native_fragment_vote_width = native_fragment_vote_width,
                .partial_wave_fragment = partial_wave_fragment,
                .phase = phase,
                .rtt_log = rtt_log,
                .pending_timing = pending_timing,
                .timing_enabled = timing_enabled,
                .use_direct_index_views = use_direct_index_views,
                .descriptor_validate_mode = descriptor_validate_mode,
                .draw_resource_ctx = draw_resource_ctx,
                .refvs = refvs,
                .refvs_spv = refvs_spv,
                .ps_override = ps_override,
                .ps_override_is_file = ps_override_is_file,
                .ps_override_is_test = ps_override_is_test,
                .fs_match_mode = fs_match_mode,
                .fs_match = fs_match,
                .fs_guest_addr_text = fs_guest_addr_text,
                .fs_guest_addr = fs_guest_addr,
                .fs_guest_addr_valid = fs_guest_addr_valid,
                .fs_target_addr_text = fs_target_addr_text,
                .fs_target_addr = fs_target_addr,
                .fs_target_addr_valid = fs_target_addr_valid,
                .fs_target_dim_text = fs_target_dim_text,
                .fs_target_width = fs_target_width,
                .fs_target_height = fs_target_height,
                .fs_target_dim_valid = fs_target_dim_valid,
                .testps_match_mode = testps_match_mode,
                .testps_match = testps_match,
                .nops = nops,
                .skip_draws_env = skip_draws_env,
                .program_skip = program_skip,
                .program_skip_armed = program_skip_armed,
                .program_census = program_census,
                .link_scan = link_scan,
                .link_scan_armed = link_scan_armed};
            auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group,
                                 prosper::test::BackendSubmissionBatch* producer_batch = nullptr) {
                return build_backend_draws(backend_draw_ctx, group, producer_batch);
            };
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
            // Fail-visible: a submit that renders yet offers no present-extent source is exactly the
            // Frontiers wall, and before #1968's diagnostics nothing anywhere said so — the publish
            // counter simply stopped while the guest kept submitting. Never expected, so it reports
            // unconditionally; capped per the rate-limit contract (ordinal on every line so the last
            // one bounds the population from below, plus a power-of-two tail so the tail exists at
            // all — src/gpu/diagnostics/diag_ratelimit.hpp, instrument trap 49).
            //
            // The two totals on every line settle a question the frame counter cannot: whether a title
            // whose publish rate looks healthy is publishing FRESH frames or re-serving one retained
            // frame. `frame_seq` climbing says nothing about which (instrument trap 90), and the answer
            // decides where the next investigation looks, so the counters are carried by the instrument
            // rather than left to be inferred.
            static std::atomic<uint64_t> present_frames_stored{0};   // publishable, this submit's own
            static std::atomic<uint64_t> present_frames_served{0};   // the retained frame, re-served
            static std::atomic<uint64_t> present_extent_reports{0};
            auto report_present_extent_shortfall = [&](const char* outcome, size_t offered_bytes) {
                const uint64_t ord = present_extent_reports.fetch_add(1) + 1;
                if (!prosper::diag_should_print(ord)) return;
                auto describe = [](char* out, size_t n, uint32_t cw, uint32_t ch, uint64_t cbase) {
                    if (!cw || !ch) snprintf(out, n, "none");
                    else snprintf(out, n, "%ux%u@0x%llx", cw, ch, (unsigned long long)cbase);
                };
                char front_text[64], vo_text[64], last_text[64];
                describe(front_text, sizeof front_text, px_front_w, px_front_h, px_front_base);
                describe(vo_text, sizeof vo_text, px_vo_w, px_vo_h, px_vo_base);
                describe(last_text, sizeof last_text, px_last_w, px_last_h, px_last_base);
                fprintf(stderr,
                        "[rtt] PRESENT SOURCE EXTENT MISMATCH #%llu: no pass produced a %ux%u "
                        "(%zu-byte) present source — px_front=%s px_vo=%s px_last=%s, offered %zu "
                        "bytes; %s (published so far: fresh=%llu retained=%llu)\n",
                        (unsigned long long)ord, w, h, present_extent_bytes,
                        front_text, vo_text, last_text, offered_bytes, outcome,
                        (unsigned long long)present_frames_stored.load(std::memory_order_relaxed),
                        (unsigned long long)present_frames_served.load(std::memory_order_relaxed));
            };
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
                // Single-framebuffer path: render_draws_rgba composites every draw into ONE framebuffer.
                std::vector<const prosper::gpu::DrawItem*> all; all.reserve(items.size());
                for (const auto& it : items) all.push_back(&it);
                const bool perf_build_clock = prosper::diagnostics::perf::enabled();   // #3891
                const auto build_start = timing_enabled || perf_build_clock
                    ? RenderClock::now() : RenderClock::time_point{};
                auto backend_draws = build_bds(all);
                const auto build_done = timing_enabled || perf_build_clock
                    ? RenderClock::now() : RenderClock::time_point{};
                if (perf_build_clock) {
                    prosper::diagnostics::perf::add_cost(
                        prosper::diagnostics::perf::Cost::FrontendBuild,
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            build_done - build_start).count()));
                    prosper::diagnostics::perf::flush_thread_texture_references();
                }
                auto rendered = prosper::test::render_draws_rgba(
                    backend_draws, w, h, nullptr, clear_for(all), true);
                const auto backend_done = timing_enabled
                    ? RenderClock::now() : RenderClock::time_point{};
                const prosper::test::BackendRenderTimingStats backend_call_timing = timing_enabled
                    ? prosper::test::backend_render_timing_stats()
                    : prosper::test::BackendRenderTimingStats{};
                const prosper::test::BackendTextureUploadStats backend_texture_stats = timing_enabled
                    ? prosper::test::backend_texture_upload_stats()
                    : prosper::test::BackendTextureUploadStats{};
                const prosper::test::BackendPipelineCacheStats backend_pipeline_stats = timing_enabled
                    ? prosper::test::backend_pipeline_cache_stats()
                    : prosper::test::BackendPipelineCacheStats{};
                // Captured HERE, beside the pipeline stats, and PASSED IN -- not read inside
                // record_backend_timing. These are thread_local and published by the backend call;
                // reading them from wherever the recorder happens to run returns a different,
                // nearly empty instance. The first draft did exactly that and reported refs=11 for
                // a 2,102-draw submit -- the kind of small plausible number that gets believed.
                const prosper::test::BackendResourceReuseStats backend_reuse_stats = timing_enabled
                    ? prosper::test::backend_resource_reuse_stats()
                    : prosper::test::BackendResourceReuseStats{};
                if (timing_enabled) {
                    pending_timing.build_resources_ms +=
                        std::chrono::duration<double, std::milli>(build_done - build_start).count();
                    pending_timing.backend_ms +=
                        std::chrono::duration<double, std::milli>(backend_done - build_done).count();
                    record_backend_timing(backend_call_timing, backend_texture_stats,
                                          backend_pipeline_stats, backend_reuse_stats);
                }
                // RTT (#167): cache these rendered pixels under this submit's render-target base, so a later
                // composite pass that samples that address gets the scene we drew (not empty guest memory).
                selected_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(rendered));
                selected_source_submit = !items.empty() ? phase.source_submit : 0;
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
                auto cached_scanout = [&](uint64_t addr) -> const RttSurf* {
                    auto it = g_rtt.find(addr);
                    if (it == g_rtt.end() || it->second.w != w || it->second.h != h ||
                        it->second.format != VK_FORMAT_R8G8B8A8_UNORM ||
                        it->second.volume_depth)
                        return nullptr;
                    RttSurf& surface = it->second;
                    const size_t expected = static_cast<size_t>(w) * h * 4;
                    if ((!surface.rgba || surface.rgba->size() != expected) &&
                        surface.has_uniform_color)
                        materialize_uniform_rtt(surface);
                    if ((!surface.rgba || surface.rgba->size() != expected) && surface.gpu_valid) {
                        std::vector<uint8_t> materialized;
                        std::string error;
                        if (prosper::test::readback_persistent_color_target(
                                addr, w, h, surface.format, materialized, error) &&
                            materialized.size() == expected) {
                            surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                                std::move(materialized));
                        } else {
                            static std::atomic<int> warned{0};
                            if (warned.fetch_add(1) < 24)
                                std::fprintf(stderr,
                                             "[rtt] final scanout readback failed: base=0x%llx "
                                             "extent=%ux%u error=%s\n",
                                             static_cast<unsigned long long>(addr), w, h,
                                             error.c_str());
                            return nullptr;
                        }
                    }
                    return surface.rgba && surface.rgba->size() == expected ? &surface : nullptr;
                };
                prosper::VideoOutBufferSnapshot front_snapshot;
                const bool have_front = prosper::videoout_front_snapshot(front_snapshot);
                const int front = have_front ? front_snapshot.buffer_index : -1;
                // Selection, address, registration generation, and originating HLE flip are one
                // registry-locked snapshot. The global flip and present counters can cross-pair
                // when guest and GPU flip submitters interleave, so neither may label this image.
                const uint64_t front_flip = have_front ? front_snapshot.source_flip_seq : 0;
                static uint64_t last_gpu_publish_flip = UINT64_MAX;
                const uint64_t current_flip = prosper_vo_flip_count();
                const bool new_gpu_flip = front_flip && prosper::frontend::present_blit_has_new_flip(
                    last_gpu_publish_flip, front_flip);
                PresentHandoffTrace handoff_trace(current_flip);
                if (handoff_trace.active)
                    handoff_trace.emit(prosper::perf::PresentHandoffEvent::RendererGate, front, 0, prosper::gpu::present_count(),
                                       front >= 0 ? prosper_vo_buffer_addr(front) : 0);
                // GPU present (#1270): when prosper-app has adopted this device and is consuming the
                // front-buffer image directly, blit it into a scanout slot on the GPU and SKIP the CPU
                // readback+reupload entirely. gpu_present_active() is false in every headless/test/
                // screenshot process, so this whole branch is inert there and the CPU path below is
                // byte-for-byte unchanged. On a MISS (image not resident/valid this frame) this falls
                // through to the CPU readback below, which still publishes a CPU frame; prosper-app
                // presents that CPU frame when no GPU frame was published (main.cpp), so a miss degrades to
                // the CPU present path rather than freezing the window.
                // A submit whose one-shot capture is pending presents through the CPU path below,
                // which materializes the scanout on demand: the capsule needs that frame as its
                // output oracle (#3895). Every other submit keeps GPU present.
                const bool gpu_present_now = prosper::frontend::gpu_present_allowed_during_capture(
                    prosper::gpu::gpu_present_active(),
                    prosper::gpu::gpu_capture_requires_cpu_output());
                using prosper::frontend::GpuPresentOutcome;
                GpuPresentOutcome gpu_outcome = GpuPresentOutcome::Published;
                // Diagnostic context for a decline line; filled as far as the checks get.
                uint64_t decline_va = front_snapshot.address;
                const RttSurf* decline_surf = nullptr;
                VkFormat decline_fmt = VK_FORMAT_UNDEFINED;
                bool decline_have_target = false;
                if (!prosper::gpu::gpu_present_active()) {
                    gpu_outcome = GpuPresentOutcome::Inactive;
                } else if (!gpu_present_now) {
                    gpu_outcome = GpuPresentOutcome::CaptureNeedsCpu;
                } else if (front < 0) {
                    gpu_outcome = GpuPresentOutcome::NoFront;
                } else if (!front_flip) {
                    gpu_outcome = GpuPresentOutcome::NoFlipIdentity;
                } else if (!new_gpu_flip) {
                    // The previously published slot remains the correct scanout for this guest
                    // flip. Treat it as a successful GPU publication so intermediate render
                    // submissions do not fall through to the expensive CPU readback path.
                    published_gpu = true;
                    gpu_outcome = GpuPresentOutcome::SameFlip;
                    handoff_trace.emit(prosper::perf::PresentHandoffEvent::SameFlipSuppressed, 0, 0, last_gpu_publish_flip);
                } else {
                    const uint64_t front_va = front_snapshot.address;
                    auto rit = g_rtt.find(front_va);
                    if (rit != g_rtt.end()) decline_surf = &rit->second;
                    if (rit == g_rtt.end()) {
                        gpu_outcome = GpuPresentOutcome::NoRenderTarget;
                    } else if (rit->second.volume_depth) {
                        gpu_outcome = GpuPresentOutcome::VolumeTarget;
                    } else if (!rit->second.gpu_valid || !rit->second.w || !rit->second.h) {
                        gpu_outcome = GpuPresentOutcome::NotGpuResident;
                    } else {
                        const VkFormat fmt = prosper::test::backend_color_format(rit->second.format);
                        decline_fmt = fmt;
                        // The cache key is only a lookup hint. Hold its resource-domain lock from
                        // the exact image/provenance snapshot through the synchronous scanout copy;
                        // eviction or allocation reuse must not substitute another image mid-handoff.
                        std::lock_guard resource_lock(
                            prosper::test::backend_persistent_resource_mutex());
                        prosper::test::PersistentColorTargetImage* tgt =
                            prosper::test::find_persistent_color_target(
                                front_va, rit->second.w, rit->second.h, fmt);
                        decline_have_target = tgt != nullptr;
                        if (!tgt || !tgt->image) {
                            gpu_outcome = GpuPresentOutcome::NoPersistentImage;
                        } else if (tgt->layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                            gpu_outcome = GpuPresentOutcome::UndefinedLayout;
                        } else {
                            published_gpu = prosper::frontend::present_blit_publish(
                                tgt->image, tgt->layout, fmt, rit->second.w, rit->second.h,
                                front_flip,
                                prosper::test::persistent_color_producer_source(*tgt),
                                &front_snapshot);
                            if (published_gpu) last_gpu_publish_flip = front_flip;
                            else gpu_outcome = GpuPresentOutcome::PublishFailed;
                        }
                    }
                }
                // #3915: a display buffer written by a compute dispatch has no render target, but
                // the compute backend may have left a GPU mirror of exactly its bytes. The mirror
                // stands in only where the CPU fallback would itself present the guest buffer: no
                // renderer entry at the front address at all. Any entry, a pixel-less tombstone
                // included, makes the CPU path keep the previous frame, so the decision declines it
                // (RendererOwnsTarget) and the renderer's own reason stands.
                if (gpu_outcome == GpuPresentOutcome::NoRenderTarget ||
                    gpu_outcome == GpuPresentOutcome::NotGpuResident) {
                    // The CPU path's own priority (cached_scanout below): a render target at ANY
                    // registered buffer that it could serve beats the guest buffer.
                    const size_t scanout_bytes = static_cast<size_t>(w) * h * 4;
                    bool renderer_scanout = false;
                    for (int i = 0; i < prosper_vo_buffer_count() && !renderer_scanout; ++i) {
                        auto it = g_rtt.find(prosper_vo_buffer_addr(i));
                        renderer_scanout = it != g_rtt.end() && it->second.w == w &&
                            it->second.h == h && it->second.format == VK_FORMAT_R8G8B8A8_UNORM &&
                            !it->second.volume_depth &&
                            ((it->second.rgba && it->second.rgba->size() == scanout_bytes) ||
                             it->second.has_uniform_color || it->second.gpu_valid);
                    }
                    prosper::frontend::ComputeScanoutPresentInputs renderer_inputs;
                    renderer_inputs.have_selected_pixels = static_cast<bool>(selected_pixels);
                    renderer_inputs.renderer_scanout = renderer_scanout;
                    renderer_inputs.renderer_owns_front = decline_surf != nullptr;
                    renderer_inputs.present_extent_bytes = present_extent_bytes;
                    renderer_inputs.display_bytes =
                        static_cast<uint64_t>(prosper::gpu::present_width()) *
                        prosper::gpu::present_height() * 4u;
                    const auto compute = prosper::frontend::compute_scanout_publish(
                        front_snapshot, front_flip, renderer_inputs);
                    using prosper::frontend::ComputeScanoutPresent;
                    switch (compute.decision) {
                    case ComputeScanoutPresent::Publish:
                        gpu_outcome = compute.published ? GpuPresentOutcome::Published
                                                        : GpuPresentOutcome::PublishFailed;
                        break;
                    case ComputeScanoutPresent::Absent:               // keep the renderer's reason
                    case ComputeScanoutPresent::RendererOwnsTarget: break;
                    case ComputeScanoutPresent::Stale:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutStale; break;
                    case ComputeScanoutPresent::Unwatched:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutUnwatched; break;
                    case ComputeScanoutPresent::ExtentMismatch:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutExtentMismatch; break;
                    case ComputeScanoutPresent::TileMismatch:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutTileMismatch; break;
                    case ComputeScanoutPresent::RendererSource:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutRendererSource; break;
                    case ComputeScanoutPresent::ScaledPresent:
                        gpu_outcome = GpuPresentOutcome::ComputeScanoutScaled; break;
                    }
                    if (compute.published) {
                        published_gpu = true;
                        last_gpu_publish_flip = front_flip;
                    }
                }
                if (prosper::frontend::gpu_present_outcome_is_decline(gpu_outcome)) {
                    prosper::diagnostics::perf::note_present_decline(
                        static_cast<size_t>(gpu_outcome),
                        prosper::frontend::gpu_present_outcome_name(gpu_outcome));
                    static_assert(static_cast<size_t>(GpuPresentOutcome::Count) <=
                                      prosper::diagnostics::perf::kPresentDeclineSlots,
                                  "every GPU-present outcome needs a perf-ledger decline slot");
                    // Budgeted per reason, so one noisy reason cannot hide the first of another.
                    // Counted per final render SPAN; a flip can end several, so this is not a
                    // count of fallback presents (that is the fps line's cpu-fallback count).
                    static std::atomic<uint64_t> decline_reports[(size_t)GpuPresentOutcome::Count]{};
                    const uint64_t ord = decline_reports[(size_t)gpu_outcome].fetch_add(1) + 1;
                    if (prosper::diag_should_print(ord))
                        fprintf(stderr,
                                "[present] GPU PRESENT DECLINED #%llu: %s -- front=%d flip=%llu "
                                "va=0x%llx rtt=%s%ux%u fmt=%d gpu_valid=%d uniform=%d cpu_px=%d "
                                "persistent=%d display=%ux%u; this span's frame goes to the CPU fallback\n",
                                (unsigned long long)ord,
                                prosper::frontend::gpu_present_outcome_name(gpu_outcome), front,
                                (unsigned long long)front_flip, (unsigned long long)decline_va,
                                decline_surf ? "" : "none:", decline_surf ? decline_surf->w : 0,
                                decline_surf ? decline_surf->h : 0,
                                decline_surf ? (int)decline_surf->format : (int)decline_fmt,
                                decline_surf ? (int)decline_surf->gpu_valid : 0,
                                decline_surf ? (int)decline_surf->has_uniform_color : 0,
                                decline_surf && decline_surf->rgba ? 1 : 0,
                                (int)decline_have_target, w, h);
                }
                if (!published_gpu) handoff_trace.emit(prosper::perf::PresentHandoffEvent::CpuFallbackNeeded);
                const RttSurf* scanout = (!published_gpu && front >= 0)
                    ? cached_scanout(prosper_vo_buffer_addr(front)) : nullptr;
                if (!published_gpu && !scanout) {
                    for (int i = 0; i < prosper_vo_buffer_count(); ++i)
                        if ((scanout = cached_scanout(prosper_vo_buffer_addr(i)))) break;
                }
                if (scanout) {
                    const auto prior_selected_pixels = selected_pixels;
                    const uint64_t prior_source_submit = selected_source_submit;
                    selected_pixels = scanout->rgba;
                    // Only the same immutable allocation from this callback's completed pass
                    // inherits its submit. A cached/materialized surface with the same address but
                    // different storage has no source proof here.
                    selected_source_submit = selected_pixels && selected_pixels == prior_selected_pixels
                        ? prior_source_submit : 0;
                }
                // Last resort before the retained frame: the flipped buffer's own guest memory.
                // The decision itself lives in guest_scanout_present.hpp so it can be unit-tested;
                // read that header for why each condition is there. Only the mechanism is here.
                //
                // Read once per guest flip, not once per submit: this callback runs for every span
                // of every submit (eleven per frame on Frontiers), the buffer only changes when the
                // guest flips a new one, and a 4K copy plus de-swizzle is not free. These statics
                // have the same single-present-thread contract as `last_scanout_present` below.
                static uint64_t guest_scanout_flip = UINT64_MAX;
                static std::shared_ptr<const std::vector<uint8_t>> guest_scanout_pixels;
                const size_t display_bytes =
                    static_cast<size_t>(prosper::gpu::present_width()) *
                    prosper::gpu::present_height() * 4u;
                if (prosper::frontend::guest_scanout_read_warranted(
                        published_gpu, scanout != nullptr, (bool)selected_pixels,
                        present_extent_bytes, display_bytes) ==
                    prosper::frontend::GuestScanoutDecision::Publish) {
                    if (guest_scanout_flip != current_flip) {
                        guest_scanout_flip = current_flip;
                        guest_scanout_pixels.reset();
                        prosper::VideoOutLinearRead read;
                        const bool got = prosper::videoout_read_front_linear(read);
                        // Named rather than inlined so the #2932 probe below can REPORT it. Whether
                        // the renderer owns a target at the flipped VA is the first thing
                        // guest_scanout_publishable tests, and it short-circuits ahead of
                        // guest_authored -- so a decline of SkipNotAuthored is itself evidence that
                        // this was false at that flip. Reading that off the decision name is
                        // inference; printing the input is measurement.
                        const bool renderer_owns_flip =
                            got && g_rtt.find(read.metadata.address) != g_rtt.end();
                        const auto decision = prosper::frontend::guest_scanout_publishable(
                            got ? read.pixels.size() : 0u, present_extent_bytes,
                            got && read.metadata.address != 0,
                            renderer_owns_flip,
                            read.guest_authored);
                        if (decision == prosper::frontend::GuestScanoutDecision::Publish)
                            guest_scanout_pixels = std::make_shared<const std::vector<uint8_t>>(
                                std::move(read.pixels));
                        // Report the OUTCOME, publish or decline, and name which. A negative
                        // control that can only observe silence cannot tell "correctly declined"
                        // from "never reached" — and a decline is exactly what the Bendy
                        // (PPSA27616) reused-memory case must produce, so it has to be legible.
                        //
                        // Budgeted PER DECISION, which is the whole point on this line and not
                        // ceremony (diag_ratelimit.hpp: "budget per key, or a noisy key exhausts the
                        // log before the one under investigation gets a line"). A title that
                        // declines on every flip would otherwise push the first PUBLISH past the
                        // cap, and "no publish line" is exactly the reading the negative control
                        // rests on. With its own counter the first publish is ordinal 1 and always
                        // prints, so the absence of a `publish` line means zero publishes.
                        static std::atomic<uint64_t>
                            guest_scanout_reports[(size_t)
                                prosper::frontend::GuestScanoutDecision::SkipNotAuthored + 1]{};
                        const uint64_t ord =
                            guest_scanout_reports[(size_t)decision].fetch_add(1) + 1;
                        if (prosper::diag_should_print(ord))
                            fprintf(stderr,
                                    "[rtt] GUEST SCANOUT #%llu: no present source and no renderer "
                                    "target at the flipped buffer 0x%llx — %s (%ux%u tiling=%u "
                                    "authored=%d footprint=%s)\n",
                                    (unsigned long long)ord,
                                    (unsigned long long)read.metadata.address,
                                    prosper::frontend::guest_scanout_decision_name(decision), w, h,
                                    read.metadata.tiling_mode, (int)read.guest_authored,
                                    read.padded_footprint ? "padded" : "nominal");
                        // #2932: what is actually true of the flipped buffer at the moment we
                        // decline to publish it? Three facts, printed together because separately
                        // each one invites a wrong inference:
                        //
                        //   phys=      the physical page behind the flipped VA. This began as a test
                        //              of "scanout and the render targets are two mappings of one
                        //              allocation"; that is FALSE (STRAY_STATUS.md, Ruled out).
                        //   aliases=N  the SEARCHED DOMAIN. The table is populated only while
                        //              page-protection watches are armed, so N=0 means "nothing was
                        //              searched" and says nothing about this VA. A miss reported
                        //              without its domain is how an unarmed instrument gets recorded
                        //              as a negative result.
                        //   rtt_owns=  whether g_rtt holds a target at this VA -- the FIRST input
                        //              guest_scanout_publishable tests, short-circuiting ahead of
                        //              guest_authored. Measured, not inferred from the decision name.
                        //
                        // Deliberately NOT asserted here: that no VA-keyed lookup can connect the
                        // flipped buffer to a render target. Stray's scanout VAs appear in the
                        // colour-target census as 4K attachments in passes carrying draws, so that
                        // sentence -- which earlier revisions of this comment stated as fact -- is
                        // withdrawn. rtt_owns is printed precisely so nobody has to assume it again.
                        if (read.metadata.address && prosper::diag_should_print(ord)) {
                            uint64_t flip_phys = 0;
                            size_t alias_n = 0;
                            const bool ok = prosper::host::guest_write_watch_va_to_phys(
                                read.metadata.address, flip_phys, &alias_n);
                            if (ok)
                                fprintf(stderr,
                                        "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> phys=0x%llx "
                                        "(aliases=%zu rtt_owns=%d decision=%s)\n",
                                        (unsigned long long)ord,
                                        (unsigned long long)read.metadata.address,
                                        (unsigned long long)flip_phys, alias_n,
                                        (int)renderer_owns_flip,
                                        prosper::frontend::guest_scanout_decision_name(decision));
                            else
                                fprintf(stderr,
                                        "[rtt] GUEST SCANOUT #%llu phys: va=0x%llx -> UNRESOLVED "
                                        "(aliases=%zu rtt_owns=%d decision=%s; %s)\n",
                                        (unsigned long long)ord,
                                        (unsigned long long)read.metadata.address, alias_n,
                                        (int)renderer_owns_flip,
                                        prosper::frontend::guest_scanout_decision_name(decision),
                                        alias_n ? "no alias range covers this VA"
                                                : "alias table EMPTY -- nothing was searched");
                        }
                    }
                    // Re-check the size on the cached path too: a two-set geometry switch can change
                    // the present extent between spans of one flip, and the cache is keyed on the
                    // flip alone.
                    if (guest_scanout_pixels && guest_scanout_pixels->size() == present_extent_bytes) {
                        selected_pixels = guest_scanout_pixels;
                        frame_origin = prosper::gpu::PresentFrameOrigin::GuestScanout;
                        selected_source_submit = 0;
                    }
                }
                // Hold the last good scanout across VideoOut buffer rotation. Bendy (PPSA27616) rapidly
                // re-registers its scanout buffers; on the frames where the guest has registered new
                // buffers but not yet rendered+flipped them, none is gpu_valid and the present would be
                // a black flicker. Presenting the previous frame instead keeps a stable image (the new
                // buffers get drawn and flipped within a frame or two). CONFIDENCE: MED.
                // Thread-safety: this static is a plain (non-atomic) shared_ptr, correct only under the
                // renderer's single present thread (this callback and its sibling statics — dp_submit,
                // warned, frame_no — all assume the one serialized present path). It must not be read or
                // assigned from another thread; a concurrent present would race the object assignment.
                // The net's own test used to be `!empty()`, which is not the publish predicate (#1986):
                // a non-empty WRONG-EXTENT frame took the store branch, overwrote the retained good
                // frame, and made the recovery branch unreachable for the rest of the process. Sonic
                // Frontiers' 4K publish wall was permanent for exactly that reason. Retaining is now
                // the same predicate as publishing, so a frame prosper could not publish can never
                // become the frame it serves later.
                static std::shared_ptr<const std::vector<uint8_t>> last_scanout_present;
                // Provenance is a property of the RETAINED frame, not of the submit that re-serves
                // it. Post-wall, Frontiers re-serves one retained frame thousands of times (fresh +1
                // against retained +2,048 between two shortfall ordinals), so a label that reset on
                // re-serve would report the guest's own buffer as a composited frame for all but the
                // first of them — and `--require-composited-frame` would pass on the strength of the
                // copies.
                static bool last_scanout_present_from_guest = false;
                static uint64_t last_scanout_source_submit = 0;
                if (!published_gpu) {
                    const size_t current_bytes = selected_pixels ? selected_pixels->size() : 0u;
                    const size_t retained_bytes =
                        last_scanout_present ? last_scanout_present->size() : 0u;
                    switch (prosper::frontend::retained_frame_action(
                                current_bytes, retained_bytes, present_extent_bytes)) {
                        case prosper::frontend::RetainedFrameAction::StoreCurrent:
                            last_scanout_present = selected_pixels;
                            last_scanout_present_from_guest =
                                frame_origin == prosper::gpu::PresentFrameOrigin::GuestScanout;
                            last_scanout_source_submit = selected_source_submit;
                            present_frames_stored.fetch_add(1, std::memory_order_relaxed);
                            // `retained_frame_action` decides on BYTE SIZES ALONE -- by design, since
                            // it cannot know what a frame should look like. The consequence is that a
                            // correctly-sized but CONTENTLESS frame is retained just as readily as a
                            // good one, and is then re-served on every later submit that produces no
                            // present source. Little Nightmares III presents a uniform (255,255,0)
                            // frame on ~2/3 of samples for exactly that reason (#2014): the retained
                            // frame is itself uniform.
                            //
                            // This names the ORIGIN of a uniform retained frame, which is the one
                            // fact the existing counters cannot supply -- `fresh=N retained=M` says a
                            // substitution happened, not what was substituted or where it came from.
                            // Sampled, not exhaustive: a full uniformity scan of a 33 MB frame on the
                            // present path would cost more than the render.
                            if (const char* uni = PROSPER_ENV_VALUE("PROSPER_UNIFORMLOG")) {
                                if (uni[0] == '1' && uni[1] == '\0' && selected_pixels &&
                                    selected_pixels->size() >= 4) {
                                    const uint8_t* px = selected_pixels->data();
                                    const size_t n = selected_pixels->size();
                                    bool uniform = true;
                                    for (size_t off = 4; off + 4 <= n; off += ((n / 4096) | 4u) & ~3u)
                                        if (px[off] != px[0] || px[off+1] != px[1] ||
                                            px[off+2] != px[2] || px[off+3] != px[3]) {
                                            uniform = false; break;
                                        }
                                    if (uniform) {
                                        static std::atomic<uint64_t> uniform_stores{0};
                                        const uint64_t ord =
                                            uniform_stores.fetch_add(1, std::memory_order_relaxed) + 1;
                                        // WHICH pass handed us the uniform frame. `origin=` only
                                        // separates a composite from a republished guest scanout; it
                                        // cannot say which colour target the composite read back, and
                                        // that is the one fact needed to go from "a uniform frame was
                                        // retained" to "this surface is uniform" (#2014).
                                        using prosper::frontend::PresentSourceChoice;
                                        const uint32_t src_w =
                                            present_choice == PresentSourceChoice::Front ? px_front_w
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_w
                                          : present_choice == PresentSourceChoice::Last  ? px_last_w : 0u;
                                        const uint32_t src_h =
                                            present_choice == PresentSourceChoice::Front ? px_front_h
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_h
                                          : present_choice == PresentSourceChoice::Last  ? px_last_h : 0u;
                                        const uint64_t src_base =
                                            present_choice == PresentSourceChoice::Front ? px_front_base
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_base
                                          : present_choice == PresentSourceChoice::Last  ? px_last_base : 0u;
                                        const uint32_t src_fmt =
                                            present_choice == PresentSourceChoice::Front ? px_front_fmt
                                          : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
                                          : present_choice == PresentSourceChoice::Last  ? px_last_fmt
                                                                                        : kNoPassFormat;
                                        // A VkFormat, not the raw CB_COLOR0_INFO.FORMAT:
                                        // `ResolvedPipelineState::color0_format` is already
                                        // `vk_color_format(...)`'s output (render_state.cpp:789).
                                        // Naming it `cbfmt` cost a wrong reading once -- 44 is
                                        // VK_FORMAT_B8G8R8A8_UNORM, which is not a CB format value
                                        // at all.
                                        char fmt_text[16];
                                        if (src_fmt == kNoPassFormat) snprintf(fmt_text, sizeof fmt_text, "none");
                                        else snprintf(fmt_text, sizeof fmt_text, "%u", src_fmt);
                                        if (prosper::diag_should_print(ord))
                                            fprintf(stderr,
                                                    "[uniformlog] #%llu retaining a UNIFORM frame "
                                                    "rgba=(%u,%u,%u,%u) bytes=%zu origin=%s src=%s "
                                                    "%ux%u@0x%llx vkfmt=%s\n",
                                                    (unsigned long long)ord,
                                                    px[0], px[1], px[2], px[3], n,
                                                    frame_origin ==
                                                        prosper::gpu::PresentFrameOrigin::GuestScanout
                                                        ? "GuestScanout" : "Composited",
                                                    prosper::frontend::present_source_name(present_choice),
                                                    src_w, src_h, (unsigned long long)src_base, fmt_text);
                                    }
                                }
                            }
                            break;
                        case prosper::frontend::RetainedFrameAction::ServeRetained:
                            selected_pixels = last_scanout_present;
                            frame_origin = last_scanout_present_from_guest
                                ? prosper::gpu::PresentFrameOrigin::GuestScanout
                                : prosper::gpu::PresentFrameOrigin::Composited;
                            selected_source_submit = last_scanout_source_submit;
                            present_frames_served.fetch_add(1, std::memory_order_relaxed);
                            // A stale-but-correct frame IS what the guest sees, so say so rather than
                            // letting a silent substitution read as a healthy present. Under no extent
                            // contract this branch is the historical empty-frame recovery and is not
                            // itself a defect, so it stays quiet there.
                            if (present_extent_bytes)
                                report_present_extent_shortfall(
                                    "serving the retained frame instead", current_bytes);
                            break;
                        case prosper::frontend::RetainedFrameAction::NoUsableFrame:
                            // Nothing publishable this submit and nothing retained to fall back on, so
                            // the screen does not advance. selected_pixels is left as selection found
                            // it — normally empty on this path, since selection already refused every
                            // wrong-extent candidate — which keeps the caller's `[render] … Vulkan
                            // render FAILED` and `[agc] PUBLISH DROPPED` backstops intact. This is the
                            // one outcome that is a lost frame rather than a substituted one, so it is
                            // reported whenever the contract applies, including when nothing rendered
                            // at all: "no pass produced a present-extent source" is the finding either
                            // way, and Frontiers' wall was invisible for exactly as long as it was not
                            // stated anywhere.
                            if (present_extent_bytes)
                                report_present_extent_shortfall(
                                    "nothing is published for this submit", current_bytes);
                            break;
                    }
                }
                if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
                    size_t nb = 0;
                    if (selected_pixels)
                        for (size_t p = 0; p + 3 < selected_pixels->size(); p += 4)
                            if ((*selected_pixels)[p] || (*selected_pixels)[p+1] || (*selected_pixels)[p+2]) nb++;
                    fprintf(stderr, "[persist] present: front=%d front_va=0x%llx scanout=%s rgb_nonblack=%zu\n",
                            front, (unsigned long long)(front >= 0 ? prosper_vo_buffer_addr(front) : 0),
                            scanout ? "HIT" : "MISS", nb);
                }
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
            // PROSPER_DUMP_CONTENT=<min-nonzero-bytes>: dump ONLY frames whose framebuffer has at least
            // that many nonzero bytes — catches the intermittent content submits the periodic dump misses.
            size_t content_thr = 0; if (const char* c = PROSPER_ENV_VALUE("PROSPER_DUMP_CONTENT")) content_thr = (size_t)atol(c);
            // Sparse long-route captures can override the default first-60/every-10 cadence. This is
            // particularly useful for 4K titles, where capturing itself would otherwise add gigabytes
            // of readback I/O before the scene under investigation is reached. Zero disables a phase.
            static const int dump_first = [] { const char* e = getenv("PROSPER_FRAME_DUMP_FIRST");
                                                return e ? (int)atol(e) : 60; }();
            static const int dump_every = [] { const char* e = getenv("PROSPER_FRAME_DUMP_EVERY");
                                                return e ? (int)atol(e) : 10; }();
            // PROSPER_PRESENT_NZLOG=N: log the presented frame's nonzero-byte count every N frames WITHOUT
            // writing any image. A memory-safe content proxy for long progression runs — dumping BMPs to a
            // tmpfs frame dir exhausts RAM, this does not. 0/unset disables.
            static const int nzlog_every = [] { const char* e = getenv("PROSPER_PRESENT_NZLOG");
                                                return e ? (int)atol(e) : 0; }();
            size_t px_nz = 0;
            if (dump_bmps || nzlog_every) for (uint8_t b : px) px_nz += (b != 0);
            if (px.empty() && !published_gpu) {
                fprintf(stderr, "[render] frame %d: Vulkan render FAILED (%ux%u)\n", n, w, h);
            } else if (dump_bmps && ((content_thr && px_nz >= content_thr) ||
                       (!content_thr && ((dump_first > 0 && n < dump_first) ||
                                         (dump_every > 0 && n % dump_every == 0))))) {
                char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%04d.bmp", frame_dir.c_str(), n);
                prosper::test::dump_bmp(fn, px, w, h);
                fprintf(stderr, "[render] frame %d rendered (%ux%u) nz=%zu -> %s\n", n, w, h, px_nz, fn);
            } else if (nzlog_every && !px.empty() && (n % nzlog_every == 0)) {
                fprintf(stderr, "[render-nz] frame %d (%ux%u) nz=%zu\n", n, w, h, px_nz);
            }
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
                    prosper::perf::RendererTimingRecord record;
                    record.span_start_monotonic_ns = pending_span_start_ns;
                    record.capture_generation = pending_capture_generation;
                    record.callbacks = pending_timing.callbacks;
                    record.draws = pending_timing.backend_draws;
                    record.texture_bytes = pending_timing.texture_bytes;
                    record.buffer_bytes = pending_timing.buffer_bytes;
                    record.total_ms = pending_timing.total_ms;
                    record.prelude_ms = pending_timing.prelude_ms;
                    record.pass_ms = pending_timing.pass_ms;
                    record.build_resources_ms = pending_timing.build_resources_ms;
                    record.backend_ms = pending_timing.backend_ms;
                    record.output_copy_ms = pending_timing.output_copy_ms;
                    record.pass_head_ms = pending_timing.pass_head_ms;
                    record.pass_loop_ms = pending_timing.pass_loop_ms;
                    record.pass_pre_ms = pending_timing.pass_pre_ms;
                    record.pass_post_ms = pending_timing.pass_post_ms;
                    record.post_stats_ms = pending_timing.post_stats_ms;
                    record.post_slot0_ms = pending_timing.post_slot0_ms;
                    record.post_mrt_ms = pending_timing.post_mrt_ms;
                    record.post_rest_ms = pending_timing.post_rest_ms;
                    record.pass_tail_ms = pending_timing.pass_tail_ms;
                    record.resolve_stall_ms = pending_timing.resolve_stall_ms;
                    record.resolve_read_ms = pending_timing.resolve_read_ms;
                    record.resolve_copy_stall_ms = pending_timing.resolve_copy_stall_ms;
                    record.resolve_copy_ms = pending_timing.resolve_copy_ms;
                    record.resolve_count = pending_timing.resolve_n;
                    record.resolve_read_count = pending_timing.resolve_read_n;
                    record.resolve_bytes = pending_timing.resolve_bytes;
                    record.gpu_wait_ms = pending_timing.backend_gpu_wait_ms;
                    record.gpu_timestamp_samples =
                        pending_timing.backend_gpu_timestamp_samples;
                    record.gpu_device_ms = pending_timing.backend_gpu_device_ms;
                    record.readback_ms = pending_timing.backend_readback_ms;
                    record.backend_target_ms = pending_timing.backend_target_ms;
                    record.backend_draw_setup_ms = pending_timing.backend_draw_setup_ms;
                    record.backend_record_upload_ms = pending_timing.backend_record_upload_ms;
                    record.backend_cleanup_ms = pending_timing.backend_cleanup_ms;
                    record.backend_setup_shader_ms = pending_timing.backend_setup_shader_ms;
                    record.backend_setup_fixed_ms = pending_timing.backend_setup_fixed_ms;
                    record.setup_resources_ms = pending_timing.backend_setup_resources_ms;
                    record.backend_setup_pipeline_ms =
                        pending_timing.backend_setup_pipeline_ms;
                    record.backend_pipeline_refs = pending_timing.backend_pipeline_refs;
                    record.backend_pipeline_hits = pending_timing.backend_pipeline_hits;
                    record.backend_pipeline_misses = pending_timing.backend_pipeline_misses;
                    record.backend_pipeline_bypasses = pending_timing.backend_pipeline_bypasses;
                    record.backend_pipeline_entries = pending_timing.backend_pipeline_entries;
                    record.backend_pipeline_evictions = pending_timing.backend_pipeline_evictions;
                    // Frontend and backend, kept apart by name. These two are build_resources' own
                    // texture/buffer time and are NOT parts of setup_resources_ms above.
                    record.frontend_texture_ms = pending_timing.texture_ms;
                    record.frontend_buffer_ms = pending_timing.buffer_ms;
                    record.frontend_buffer_tracked_cache_hits =
                        pending_timing.buffer_source_gate.tracked_cache_hits;
                    record.frontend_buffer_tracked_cache_fills =
                        pending_timing.buffer_source_gate.tracked_cache_fills;
                    record.frontend_buffer_tracked_untracked_misses =
                        pending_timing.buffer_source_gate.tracked_untracked_misses;
                    record.frontend_buffer_reserved_state_queries =
                        pending_timing.buffer_source_gate.reserved_state_queries;
                    record.frontend_buffer_compact_resources =
                        pending_timing.compact_buffer_resources;
                    record.frontend_buffer_full_resources =
                        pending_timing.full_buffer_resources;
                    record.frontend_tex_rtt_ms = pending_timing.tex_rtt_ms;
                    record.frontend_tex_compute_ms = pending_timing.tex_compute_ms;
                    record.frontend_tex_local_ms = pending_timing.tex_local_ms;
                    record.frontend_tex_persist_hit_ms = pending_timing.tex_persist_hit_ms;
                    record.frontend_tex_persist_reuse_ms = pending_timing.tex_persist_reuse_ms;
                    record.frontend_tex_persist_miss_ms = pending_timing.tex_persist_miss_ms;
                    record.frontend_tex_source_snapshot_handoff_ms = pending_timing.tex_source_snapshot_handoff_ms;
                    record.frontend_tex_source_snapshot_copied_bytes = pending_timing.tex_source_snapshot_copied_bytes;
                    record.frontend_tex_source_snapshot_transferred_bytes = pending_timing.tex_source_snapshot_transferred_bytes;
                    record.frontend_tex_persist_invalid_ms = pending_timing.tex_persist_invalid_ms;
                    record.frontend_tex_persist_invalid_n = pending_timing.tex_persist_invalid_n;
                    record.frontend_tex_other_n = pending_timing.tex_other_n;
                    record.frontend_tex_other_slowest_ms =
                        pending_timing.tex_other_slowest_ms;
                    record.frontend_tex_other_addr = pending_timing.tex_other_addr;
                    record.frontend_tex_other_source_bytes =
                        pending_timing.tex_other_source_bytes;
                    record.frontend_tex_other_width = pending_timing.tex_other_width;
                    record.frontend_tex_other_height = pending_timing.tex_other_height;
                    record.frontend_tex_other_depth = pending_timing.tex_other_depth;
                    record.frontend_tex_other_format = pending_timing.tex_other_format;
                    record.frontend_tex_other_components = pending_timing.tex_other_components;
                    record.frontend_tex_other_tile_mode = pending_timing.tex_other_tile_mode;
                    record.frontend_tex_other_img_dim = pending_timing.tex_other_img_dim;
                    record.frontend_tex_other_class = pending_timing.tex_other_class;
                    record.frontend_tex_other_compute_candidate =
                        pending_timing.tex_other_compute_candidate;
                    record.frontend_tex_other_persistent_candidate =
                        pending_timing.tex_other_persistent_candidate;
                    record.frontend_tex_other_compressed =
                        pending_timing.tex_other_compressed;
                    record.frontend_tex_other_depth_compare =
                        pending_timing.tex_other_depth_compare;
                    record.frontend_tex_other_host_backed =
                        pending_timing.tex_other_host_backed;
                    record.frontend_build_draw_ms = pending_timing.build_r_ms;
                    record.frontend_validate_ms = pending_timing.build_validate_ms;
                    record.frontend_poison_ms = pending_timing.build_poison_ms;
                    record.frontend_indices_ms = pending_timing.build_indices_ms;
                    record.frontend_reflect_ms = pending_timing.build_reflect_ms;
                    // These four DO decompose setup_resources_ms, so an offline report can attribute
                    // the largest bucket in the capture instead of leaving a plausible residue.
                    record.frontend_gpu_detile_preparations = pending_timing.gpu_detile_preparations;
                    record.frontend_gpu_detile_2d_preparations = pending_timing.gpu_detile_2d_preparations;
                    record.frontend_gpu_detile_source_bytes = pending_timing.gpu_detile_source_bytes;
                    record.res_texture_ms = pending_timing.backend_res_texture_ms;
                    record.res_texture_upload_ms = pending_timing.backend_res_texture_upload_ms;
                    record.res_texture_bind_ms = pending_timing.backend_res_texture_bind_ms;
                    record.backend_texture_refs = pending_timing.backend_texture_refs;
                    record.backend_texture_uploads = pending_timing.backend_texture_uploads;
                    record.backend_texture_upload_bytes = pending_timing.backend_texture_upload_bytes;
                    record.backend_texture_persistent_hits =
                        pending_timing.backend_texture_persistent_hits;
                    record.backend_texture_persistent_misses =
                        pending_timing.backend_texture_persistent_misses;
                    record.backend_texture_binding_refs = pending_timing.backend_texture_binding_refs;
                    record.backend_texture_binding_unique = pending_timing.backend_texture_binding_unique;
                    record.backend_texture_binding_persistent_hits =
                        pending_timing.backend_texture_binding_persistent_hits;
                    record.backend_texture_binding_persistent_misses =
                        pending_timing.backend_texture_binding_persistent_misses;
                    record.res_buffer_ms = pending_timing.backend_res_buffer_ms;
                    record.res_buffer_range_plan_ms = pending_timing.backend_res_buffer_range_plan_ms;
                    record.res_buffer_copy_ms = pending_timing.backend_res_buffer_copy_ms;
                    record.res_buffer_resident_ms = pending_timing.backend_res_buffer_resident_ms;
                    record.res_buffer_watch_ms = pending_timing.backend_res_buffer_watch_ms;
                    record.buffer_range_uploads = pending_timing.buffer_range_uploads;
                    record.buffer_range_bindings = pending_timing.buffer_range_bindings;
                    record.buffer_range_upload_bytes = pending_timing.buffer_range_upload_bytes;
                    record.buffer_range_bound_bytes = pending_timing.buffer_range_bound_bytes;
                    record.buffer_upload_bytes = pending_timing.buffer_upload_bytes;
                    record.buffer_resident_hits = pending_timing.buffer_resident_hits;
                    record.buffer_resident_compared_bytes = pending_timing.buffer_resident_compared_bytes;
                    record.buffer_resident_reused_bytes = pending_timing.buffer_resident_reused_bytes;
                    record.buffer_resident_admitted_bytes = pending_timing.buffer_resident_admitted_bytes;
                    record.buffer_resident_refreshed_bytes = pending_timing.buffer_resident_refreshed_bytes;
                    record.buffer_resident_watched_bytes = pending_timing.buffer_resident_watched_bytes;
                    record.buffer_resident_declined_bytes = pending_timing.buffer_resident_declined_bytes;
                    record.buffer_resident_ineligible_bytes = pending_timing.buffer_resident_ineligible_bytes;
                    record.res_buffer_create_ms = pending_timing.backend_res_buffer_create_ms;
                    record.res_buffer_index_find_ms = pending_timing.backend_res_buffer_index_find_ms;
                    record.res_buffer_index_insert_ms = pending_timing.backend_res_buffer_index_insert_ms;
                    record.res_buffer_hash_ms = pending_timing.backend_res_buffer_hash_ms;
                    record.res_descriptor_ms = pending_timing.backend_res_descriptor_ms;
                    prosper::perf::interactive_performance_capture().record_renderer(record);
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
extern "C" void prosper_vo_set_flip_publish_hook(void (*fn)(uint64_t));
extern "C" void prosper_frontend_flip_publish_guest_scanout(uint64_t flip) {
    if (prosper::frontend::g_flip_scanout_hook) prosper::frontend::g_flip_scanout_hook(flip);
}

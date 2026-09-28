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
#include "shared/present/present_extent.hpp"           // the publish extent contract with the caller (#1986)
#include "shared/media/avplayer_plane_policy.hpp"    // which sampled resource is AvPlayer's NV12 chroma plane
#include "shared/live/texture_reference_census.hpp"  // PROSPER_TEXREF_CENSUS (#3873)
#include "shared/live/submit_renderer/callback_types.hpp" // the callback's own types (#3892)
#include "shared/live/submit_renderer/guest_reads.hpp"    // safe_span / safe_copy / safe_equal
#include "shared/live/submit_renderer/draw_resources.hpp" // build_draw_frame_resources (#3892)
#include "shared/live/submit_renderer/timing_report.hpp"  // report_render_timing_aggregates (#3892)
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

// ---- Backend timing accumulation and the per-pass helpers (#3892) --------------------------------
//
// record_backend_timing_stats folds one backend call's thread-local timing, texture, pipeline and
// reuse counters into the submit's RenderTiming (and, under timing logging, the cumulative buffer
// reuse report). append_rtt_timing / print_rtt_timing format a PROSPER_RTT_TIMING record, and
// clear_for picks a pass group's clear colour. These were lambdas in the submit callback; their
// bodies moved here verbatim. record_backend_timing stays in the callback as a forwarder.
namespace {
// The submit callback's state that record_backend_timing_stats reads and writes, one reference per object.
struct BackendTimingContext {
    RenderTiming& pending_timing;
    const prosper::frontend::PerformanceTimingMode& timing_mode;
};

void record_backend_timing_stats(BackendTimingContext& ctx,
                                 const prosper::test::BackendRenderTimingStats& backend,
                                 const prosper::test::BackendTextureUploadStats& textures,
                                 const prosper::test::BackendPipelineCacheStats& pipelines,
                                 const prosper::test::BackendResourceReuseStats& reuse) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& pending_timing = ctx.pending_timing;
    auto& timing_mode = ctx.timing_mode;
    pending_timing.backend_calls += backend.calls;
    pending_timing.backend_draws += backend.draws;
    pending_timing.backend_command_buffers += backend.command_buffers;
    pending_timing.backend_queue_submits += backend.queue_submits;
    pending_timing.backend_fence_waits += backend.fence_waits;
    pending_timing.flush_no_batch += backend.flush_no_batch;
    pending_timing.flush_readback += backend.flush_readback;
    pending_timing.flush_storage_writeback += backend.flush_storage_writeback;
    pending_timing.flush_explicit += backend.flush_explicit;
    pending_timing.flush_cache_pressure += backend.flush_cache_pressure;
    pending_timing.cache_pressure_ms += backend.cache_pressure_ms;
    pending_timing.backend_gpu_timestamp_samples += backend.gpu_timestamp_samples;
    pending_timing.backend_target_ms += backend.target_ms;
    pending_timing.backend_draw_setup_ms += backend.draw_setup_ms;
    pending_timing.backend_record_upload_ms += backend.record_upload_ms;
    pending_timing.backend_gpu_wait_ms += backend.gpu_wait_ms;
    pending_timing.backend_gpu_device_ms += backend.gpu_device_ms;
    pending_timing.backend_readback_ms += backend.readback_ms;
    pending_timing.backend_cleanup_ms += backend.cleanup_ms;
    pending_timing.backend_setup_shader_ms += backend.setup_shader_ms;
    pending_timing.backend_setup_fixed_ms += backend.setup_fixed_ms;
    pending_timing.backend_res_fixed_index_upload_ms += backend.res_fixed_index_upload_ms;
    pending_timing.backend_res_fixed_blend_ms += backend.res_fixed_blend_ms;
    pending_timing.backend_res_fixed_depth_stencil_ms += backend.res_fixed_depth_stencil_ms;
    pending_timing.backend_res_fixed_viewport_ms += backend.res_fixed_viewport_ms;
    pending_timing.backend_res_fixed_stages_ms += backend.res_fixed_stages_ms;
    pending_timing.backend_res_fixed_prologue_ms += backend.res_fixed_prologue_ms;
    pending_timing.backend_res_prologue_subgroup_scan_ms += backend.res_prologue_subgroup_scan_ms;
    pending_timing.backend_setup_resources_ms += backend.setup_resources_ms;
    pending_timing.backend_res_texture_ms += backend.res_texture_ms;
    pending_timing.backend_res_texture_upload_ms += backend.res_texture_upload_ms;
    pending_timing.backend_res_texture_bind_ms += backend.res_texture_bind_ms;
    pending_timing.backend_texture_refs += textures.references;
    pending_timing.backend_texture_uploads += textures.unique_uploads;
    pending_timing.backend_texture_upload_bytes += textures.upload_bytes;
    pending_timing.backend_texture_persistent_hits += textures.persistent_hits;
    pending_timing.backend_texture_persistent_misses += textures.persistent_misses;
    pending_timing.backend_texture_binding_refs += reuse.texture_binding_references;
    pending_timing.backend_texture_binding_unique += reuse.unique_texture_bindings;
    pending_timing.backend_texture_binding_persistent_hits +=
        reuse.persistent_texture_binding_hits;
    pending_timing.backend_texture_binding_persistent_misses +=
        reuse.persistent_texture_binding_misses;
    pending_timing.backend_res_buffer_ms += backend.res_buffer_ms;
    pending_timing.backend_res_buffer_acquire_ms += backend.res_buffer_acquire_ms;
    pending_timing.backend_res_buffer_range_plan_ms += backend.res_buffer_range_plan_ms;
    pending_timing.backend_res_buffer_copy_ms += backend.res_buffer_copy_ms;
    pending_timing.backend_res_buffer_resident_ms += backend.res_buffer_resident_ms;
    pending_timing.backend_res_buffer_watch_ms += backend.res_buffer_watch_ms;
    pending_timing.buffer_range_uploads += reuse.buffer_range_uploads;
    pending_timing.buffer_range_bindings += reuse.buffer_range_bindings;
    pending_timing.buffer_range_upload_bytes += reuse.buffer_range_upload_bytes;
    pending_timing.buffer_range_bound_bytes += reuse.buffer_range_bound_bytes;
    pending_timing.buffer_upload_bytes += reuse.buffer_upload_bytes;
    pending_timing.buffer_resident_hits += reuse.buffer_resident_hits;
    pending_timing.buffer_resident_compared_bytes += reuse.buffer_resident_compared_bytes;
    pending_timing.buffer_resident_reused_bytes += reuse.buffer_resident_reused_bytes;
    pending_timing.buffer_resident_admitted_bytes += reuse.buffer_resident_admitted_bytes;
    pending_timing.buffer_resident_refreshed_bytes += reuse.buffer_resident_refreshed_bytes;
    pending_timing.buffer_resident_watched_bytes += reuse.buffer_resident_watched_bytes;
    pending_timing.buffer_resident_declined_bytes += reuse.buffer_resident_declined_bytes;
    pending_timing.buffer_resident_ineligible_bytes += reuse.buffer_resident_ineligible_bytes;
    pending_timing.backend_res_buffer_create_ms += backend.res_buffer_create_ms;
    pending_timing.backend_res_buffer_index_find_ms += backend.res_buffer_index_find_ms;
    pending_timing.backend_res_buffer_index_insert_ms += backend.res_buffer_index_insert_ms;
    pending_timing.backend_res_buffer_hash_ms += backend.res_buffer_hash_ms;
    pending_timing.backend_res_descriptor_ms += backend.res_descriptor_ms;
    pending_timing.backend_setup_pipeline_ms += backend.setup_pipeline_ms;
    pending_timing.backend_pipeline_refs += pipelines.references;
    pending_timing.backend_pipeline_hits += pipelines.hits;
    pending_timing.backend_pipeline_misses += pipelines.misses;
    pending_timing.backend_pipeline_bypasses += pipelines.bypasses;
    pending_timing.backend_pipeline_entries = pipelines.entries;
    pending_timing.backend_pipeline_evictions += pipelines.evictions;
    // Buffer-copy economics. This leaf dominates the Blue Prince FMV collapse -- 539 ms
    // of 802 ms of backend resource setup, 67% (#2215) -- and nothing could say WHY: the
    // counters are collected per call and aggregated into the thread-local storage, and
    // the ONLY consumers in the tree were assertions in test_multidraw_render.cpp. The
    // sibling resource kind has reported its cache economics for ages
    // ("[render-timing] texture_cache hits=... misses=..."), while buffers -- an order of
    // magnitude more expensive in that regime -- reported nothing at all.
    //
    // Per SUBMIT rather than as a window mean, deliberately. The question is which of
    // three mutually exclusive stories produces the copy volume, and they need different
    // fixes, so an average across them answers none of it:
    //   skipped_unique dominates -> buffer_identity == 0, nothing can dedup, fix upstream
    //   skipped_large dominates  -> the 4 KiB hash-dedup cap is working as designed and
    //                               the payloads are genuinely distinct; the fix is not
    //                               dedup at all but not staging through a mapped copy
    //   refs >> memo_hits        -> repeat references within one call are being missed
    if (timing_mode.log) {
        // CUMULATIVE, not a per-call snapshot. The first draft printed one call's
        // numbers at diag_should_print's ordinals (first 64, then powers of two) --
        // which is a UNIFORM sample of a heavily skewed distribution, so it could not
        // see the expensive calls by construction. The sampled calls summed to ~290 KB
        // per submit against 30 ms of measured copy: 290 KB cannot take 30 ms, and that
        // contradiction is the only reason the sampling flaw was caught.
        //
        // Totals cannot miss a call. The rate limit now controls how often the running
        // total is PRINTED, never which calls it counts.
        static uint64_t refs = 0, uniq = 0, memo = 0, hashed = 0, dwords = 0;
        static uint64_t skipped_large = 0, skipped_unique = 0, fallbacks = 0, calls = 0;
        static uint64_t large_dwords = 0, upload_bytes = 0;
        static PassBufferLookupStats lookup;
        refs += reuse.buffer_references;      uniq += reuse.unique_buffers;
        memo += reuse.buffer_ref_memo_hits;   hashed += reuse.buffer_hash_calls;
        dwords += reuse.buffer_hash_dwords;   fallbacks += reuse.buffer_upload_fallbacks;
        skipped_large += reuse.buffer_hash_skipped_large;
        skipped_unique += reuse.buffer_hash_skipped_unique;
        large_dwords += reuse.buffer_skipped_large_dwords;
        upload_bytes += reuse.buffer_upload_bytes;
        lookup.add(reuse.buffer_lookup);
        if (prosper::diag_should_print(++calls)) {
            fprintf(stderr,
                    "[render-timing] buffer_reuse (cumulative over %llu backend calls) "
                    "refs=%llu unique=%llu memo_hits=%llu hashed=%llu (%.1f MiB) "
                    "skipped_large=%llu (%.1f MiB) skipped_unique=%llu "
                    "COPIED=%.1f MiB upload_fallbacks=%llu\n",
                    (unsigned long long)calls, (unsigned long long)refs,
                    (unsigned long long)uniq, (unsigned long long)memo,
                    (unsigned long long)hashed, (double)dwords * 4.0 / (1024.0 * 1024.0),
                    (unsigned long long)skipped_large,
                    (double)large_dwords * 4.0 / (1024.0 * 1024.0),
                    (unsigned long long)skipped_unique,
                    (double)upload_bytes / (1024.0 * 1024.0),
                    (unsigned long long)fallbacks);
            fprintf(stderr,
                    "[render-timing] buffer_lookup (cumulative over %llu backend calls) "
                    "observed_passes=%llu arena_passes=%llu allocations=%llu "
                    "allocation_bytes=%llu deallocations=%llu deallocation_bytes=%llu\n",
                    (unsigned long long)calls,
                    (unsigned long long)lookup.observed_passes,
                    (unsigned long long)lookup.arena_passes,
                    (unsigned long long)lookup.allocations,
                    (unsigned long long)lookup.allocation_bytes,
                    (unsigned long long)lookup.deallocations,
                    (unsigned long long)lookup.deallocation_bytes);
        }
    }
}

void append_rtt_timing(std::string& output, const RttTimingRecord& record) {
    const prosper::test::BackendRenderTimingStats& timing = record.timing;
    const double detail_ms = timing.total_ms();
    char line[512];
    const int length = snprintf(
        line, sizeof(line),
        "[rtt-timing] submit=%d target=0x%llx extent=%ux%u draws=%zu "
        "span=%d/%d authoritative=%d deferred=%d cmd=%llu submit=%llu wait=%llu "
        "measured=%.2f detail=%.2f other=%.2f target_setup=%.2f "
        "draw_setup=%.2f record_upload=%.2f batch_wait=%.2f "
        "batch_device=%.2f batch_overhead=%.2f batch_timestamps=%llu "
        "readback=%.2f cleanup=%.2f gpu_target=%llu load=%llu sample=%llu cpu=%llu\n",
        record.submit, (unsigned long long)record.target,
        record.width, record.height, record.draws,
        record.first_span, record.final_span, record.authoritative_readback,
        record.deferred_readback,
        (unsigned long long)timing.command_buffers,
        (unsigned long long)timing.queue_submits,
        (unsigned long long)timing.fence_waits,
        record.measured_ms, detail_ms,
        record.measured_ms - detail_ms,
        timing.target_ms, timing.draw_setup_ms, timing.record_upload_ms,
        timing.gpu_wait_ms, timing.gpu_device_ms,
        std::max(0.0, timing.gpu_wait_ms - timing.gpu_device_ms),
        (unsigned long long)timing.gpu_timestamp_samples,
        timing.readback_ms, timing.cleanup_ms,
        (unsigned long long)record.color_target.writes,
        (unsigned long long)record.color_target.write_hits,
        (unsigned long long)record.color_target.sampled_hits,
        (unsigned long long)record.color_target.readbacks);
    if (length > 0)
        output.append(line, std::min<size_t>(length, sizeof(line) - 1));
}

void print_rtt_timing(const RttTimingRecord& record) {
    std::string output;
    append_rtt_timing(output, record);
    if (!output.empty()) fwrite(output.data(), 1, output.size(), stderr);
}

// Clear color for a group: the game's decoded fast-clear taken from the group's first item's
// resolved pipeline state, or its default opaque black when no fast-clear was programmed.
// Passing this to render_draws_rgba replaces the old hardcoded debug blue on the live path
// (#309) — PROSPER_CLEAR_DEBUG still forces blue for spotting unrendered areas.
const float* clear_for(const std::vector<const prosper::gpu::DrawItem*>& g) {
    return g.empty() ? nullptr : g.front()->ps.clear_color;
}
} // namespace

// ---- Backend draw assembly (#3892) ---------------------------------------------------------------
//
// One BackendDraw per realized DrawItem of a pass group: shaders (or the diagnostic overrides),
// the frame resources build_draw_frame_resources resolves, descriptor-contract validation,
// poison mode, indices, and the per-draw census/skip diagnostics. This was the body of the submit
// callback's build_bds lambda, moved here verbatim with the three lambdas only it used; build_bds
// stays in the callback as a one-line forwarder, so its call sites are unchanged.
namespace {
// The submit callback's state that build_backend_draws reads and writes, one reference per object.
struct BackendDrawContext {
    const bool& native_fragment_vote_width;
    const bool& partial_wave_fragment;
    const prosper::gpu::LiveRenderPhase& phase;
    const bool& rtt_log;
    RenderTiming& pending_timing;
    const bool& timing_enabled;
    const bool& use_direct_index_views;
    const char *const& descriptor_validate_mode;
    DrawResourceContext& draw_resource_ctx;
    const bool& refvs;
    std::vector<uint32_t>& refvs_spv;
    std::vector<uint32_t>& ps_override;
    bool& ps_override_is_file;
    bool& ps_override_is_test;
    int& fs_match_mode;
    std::vector<uint32_t>& fs_match;
    const char *& fs_guest_addr_text;
    uint64_t& fs_guest_addr;
    const bool& fs_guest_addr_valid;
    const char *& fs_target_addr_text;
    uint64_t& fs_target_addr;
    const bool& fs_target_addr_valid;
    const char *& fs_target_dim_text;
    uint32_t& fs_target_width;
    uint32_t& fs_target_height;
    const bool& fs_target_dim_valid;
    int& testps_match_mode;
    std::vector<uint32_t>& testps_match;
    const bool& nops;
    const char *& skip_draws_env;
    prosper::gpu::DrawProgramSkipSelector& program_skip;
    const bool& program_skip_armed;
    const bool& program_census;
    prosper::gpu::DrawLinkScanSelector& link_scan;
    const bool& link_scan_armed;
};

std::vector<prosper::test::BackendDraw> build_backend_draws(BackendDrawContext& ctx,
                                                            const std::vector<const prosper::gpu::DrawItem*>& group,
                                                            prosper::test::BackendSubmissionBatch* producer_batch) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& native_fragment_vote_width = ctx.native_fragment_vote_width;
    auto& partial_wave_fragment = ctx.partial_wave_fragment;
    auto& phase = ctx.phase;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& use_direct_index_views = ctx.use_direct_index_views;
    auto& descriptor_validate_mode = ctx.descriptor_validate_mode;
    auto& draw_resource_ctx = ctx.draw_resource_ctx;
    auto& refvs = ctx.refvs;
    auto& refvs_spv = ctx.refvs_spv;
    auto& ps_override = ctx.ps_override;
    auto& ps_override_is_file = ctx.ps_override_is_file;
    auto& ps_override_is_test = ctx.ps_override_is_test;
    auto& fs_match_mode = ctx.fs_match_mode;
    auto& fs_match = ctx.fs_match;
    auto& fs_guest_addr_text = ctx.fs_guest_addr_text;
    auto& fs_guest_addr = ctx.fs_guest_addr;
    auto& fs_guest_addr_valid = ctx.fs_guest_addr_valid;
    auto& fs_target_addr_text = ctx.fs_target_addr_text;
    auto& fs_target_addr = ctx.fs_target_addr;
    auto& fs_target_addr_valid = ctx.fs_target_addr_valid;
    auto& fs_target_dim_text = ctx.fs_target_dim_text;
    auto& fs_target_width = ctx.fs_target_width;
    auto& fs_target_height = ctx.fs_target_height;
    auto& fs_target_dim_valid = ctx.fs_target_dim_valid;
    auto& testps_match_mode = ctx.testps_match_mode;
    auto& testps_match = ctx.testps_match;
    auto& nops = ctx.nops;
    auto& program_skip = ctx.program_skip;
    auto& program_skip_armed = ctx.program_skip_armed;
    auto& program_census = ctx.program_census;
    auto& link_scan = ctx.link_scan;
    auto& link_scan_armed = ctx.link_scan_armed;
    // The callback's process-lifetime skip_draws_env is named, uncaptured, by the captureless
    // draw_is_skipped lambda (a variable with static storage duration needs no capture). A
    // function-scope static reference binds once to that same object, so the lambda keeps
    // reaching it exactly as it did inside the callback.
    static const char *& skip_draws_env = ctx.skip_draws_env;
    auto build_R = [&](const prosper::gpu::DrawItem& draw,
                       const prosper::gpu::ShaderResourceTable* vrt,
                       const prosper::gpu::ShaderResourceTable* prt,
                       prosper::test::BackendSubmissionBatch* producer_batch) {
        return build_draw_frame_resources(draw_resource_ctx, draw, vrt, prt, producer_batch);
    };
    // Poison mode keeps the draw running while making invalid bindings visually/numerically
    // unmistakable: magenta/cyan texels for images and NaN-like dwords for buffers. Missing,
    // duplicate, wrong-type, and undersized bindings are replaced from the reflected manifest.
    // Read once per submit, not once per call. poison_R is invoked TWICE per draw (VS and
    // PS), so at 2,179 draws in a submit its getenv was 4,358 calls -- and on Windows
    // getenv takes a process-wide lock and walks the environment block, which is why #2214
    // measured +43% from removing exactly this shape from the per-resource path.
    //
    // Hoisted rather than wrapped in PROSPER_ENV_VALUE, and the difference is not stylistic:
    // test_gpu_capture_render.cpp and test_shader_resources.cpp ARM this variable at runtime
    // between phases, and a process-lifetime cache would not observe the second write. Those
    // tests would not fail -- they would go VACUOUS and keep printing [ok] against a stale
    // value, which is the #2214 defect that `cached_env_arming_logic` now gates. A per-submit
    // hoist keeps every runtime write observable from the next submit on.
    auto poison_R = [descriptor_validate_mode](
                       std::vector<prosper::test::FrameResource>& resources,
                       const std::vector<uint32_t>& spirv,
                       const prosper::gpu::ShaderResourceTable* table,
                       uint32_t set, prosper::gpu::SpirvShaderStage stage) {
        const char* mode = descriptor_validate_mode;
        if (!mode || strcmp(mode, "poison")) return;
        auto report = prosper::gpu::validate_spirv_descriptor_interface(spirv, table, set, stage, false);
        static const uint8_t poison_tex[16] = {
            255, 0, 255, 255,   0, 255, 255, 255,
            0, 255, 255, 255,   255, 0, 255, 255,
        };
        static const uint32_t poison_storage_uint[16] = {
            0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
            0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0u, 0x3f800000u, 0x3f800000u, 0x3f800000u,
            0x3f800000u, 0u, 0x3f800000u, 0x3f800000u,
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
                                     : first->buffer_word_count() * 4);
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
                // Poison replacements stay deliberately small: this is a diagnostic
                // substitute for an invalid binding, not a real upload, so it does not
                // follow build_R's ceiling (which is now the declared range, #1427).
                words = std::min<size_t>(words, 1u << 18);
                replacement.dwords.assign(words, 0x7FC0CDCDu);
            } else {
                replacement.storage_image_numeric_class = d.image_numeric_class;
                replacement.storage_image_contract_valid =
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Float ||
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Uint;
                if (wants_storage_image &&
                    d.image_numeric_class ==
                        prosper::gpu::SpirvImageNumericClass::Uint) {
                    replacement.tex_rgba = reinterpret_cast<const uint8_t*>(
                        poison_storage_uint);
                    replacement.texture_format = VK_FORMAT_R32G32B32A32_UINT;
                } else {
                    replacement.tex_rgba = poison_tex;
                }
                replacement.tw = 2; replacement.th = 2;
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
    auto draw_is_skipped = [](uint64_t idx) -> bool {
        if (!skip_draws_env) return false;
        for (const char* s = skip_draws_env; *s;) {
            char* end = nullptr; unsigned long long v = strtoull(s, &end, 0);
            if (end != s && v == idx) return true;
            s = (end && *end) ? end + 1 : end;
            if (!s || !*s) break;
        }
        return false;
    };
    std::vector<prosper::test::BackendDraw> bds;
    // Once per call, not once per draw -- see the note on descriptor_validate_mode above
    // for why this is a hoist and not a PROSPER_ENV_* cache (test_eop_write.cpp arms
    // PROSPER_GFXLOG at runtime, and a cached read would make its arms vacuous).
    const bool gfxlog = getenv("PROSPER_GFXLOG") != nullptr;
    for (const auto* itp : group) {
        const auto& it = *itp;
        if (draw_is_skipped(it.draw_index)) continue;
        prosper::test::BackendDraw bd;
        bd.source_submit = phase.source_submit;
        if (refvs) {
            bd.vs = refvs_spv;
        } else if (it.vs_shared) {
            bd.vs_shared = it.vs_shared;
        } else {
            bd.vs = it.vs;
        }
        bd.gs = refvs ? std::vector<uint32_t>{} : it.gs;
        // File and synthetic overrides have independent exact-match gates.
        bool fs_ov = !ps_override.empty();
        if (fs_ov && ps_override_is_file) {
            if (fs_match_mode == 1)      fs_ov = (it.fs_words() == fs_match);   // valid match -> exact only
            else if (fs_match_mode == 2) fs_ov = false;                // requested-but-invalid -> off
            // mode 0 -> legacy global file override (unchanged)
            if (!fs_guest_addr_valid || !fs_target_addr_valid || !fs_target_dim_valid ||
                (fs_guest_addr_text && it.fs_guest_addr != fs_guest_addr) ||
                (fs_target_addr_text && it.color0_base != fs_target_addr) ||
                (fs_target_dim_text &&
                 (it.color0_width != fs_target_width ||
                  it.color0_height != fs_target_height)))
                fs_ov = false;
        }
        if (fs_ov && ps_override_is_test) {
            if (testps_match_mode == 1)      fs_ov = (it.fs_words() == testps_match);
            else if (testps_match_mode == 2) fs_ov = false;
            // mode 0 -> legacy global TESTPS override (unchanged)
        }
        if (fs_ov && ps_override_is_file && fs_match_mode == 1)
            fprintf(stderr, "[fs-match] file override applied to draw#%llu\n",
                    (unsigned long long)it.draw_index);
        if (fs_ov && ps_override_is_test && testps_match_mode == 1)
            fprintf(stderr, "[testps-match] synthetic override applied to draw#%llu\n",
                    (unsigned long long)it.draw_index);
        if (fs_ov) {
            bd.fs = ps_override;
        } else if (it.fs_shared) {
            bd.fs_shared = it.fs_shared;
        } else {
            bd.fs = it.fs;
        }
        bd.vs_identity = refvs ? 0 : it.vs_identity;
        bd.fs_identity = fs_ov ? 0 : it.fs_identity;
        bd.allow_native_fragment_vote_width =
            !fs_ov && native_fragment_vote_width;
        bd.allow_partial_wave_fragment = !fs_ov && partial_wave_fragment;
        bd.draw_index = it.draw_index;
        bd.command_order = it.command_order;
        bd.vcount = refvs ? 3u : it.vertex_count;
        bd.instance_count = it.instance_count;
        bd.vertex_offset = refvs ? 0 : it.vertex_offset;
        bd.ps     = nops ? nullptr : &it.ps;
        // Five clock reads bounding four spans, only when timing is armed -- ~0.9% of
        // this bucket at 2,100 draws a submit. It inflates what it measures while
        // armed, as every timer here does; read the shares, not the totals.
        const auto bt0 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        auto built_resources = build_R(it, it.vrt.get(), it.prt.get(), producer_batch);
        bd.R = std::move(built_resources.full);
        bd.B = std::move(built_resources.buffers);
        bd.resource_order = std::move(built_resources.order);
        const auto bt1 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        // && of the positives rather than || of the negations: identical short-circuit
        // and identical outcome, but it puts the accounting below on a path the
        // `continue` cannot skip.
        // descriptor_validate_mode is the per-submit hoist made for poison_R above, and
        // it serves this pair for the same reason: without it each call re-reads getenv,
        // and with the variable unset that read IS the whole call. It measured
        // 5.77 ms/submit in the build_resources partition -- larger than the validation
        // it was declining to do.
        const bool contract_ok =
            prosper::gpu::validate_runtime_descriptor_contract(
                "VS/backend", bd.vs_words(), it.vrt.get(), 0,
                prosper::gpu::SpirvShaderStage::Vertex, descriptor_validate_mode) &&
            prosper::gpu::validate_runtime_descriptor_contract(
                "PS/backend", bd.fs_words(), it.prt.get(), 1,
                prosper::gpu::SpirvShaderStage::Fragment, descriptor_validate_mode);
        const auto bt2 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        if (timing_enabled) {
            pending_timing.build_r_ms +=
                std::chrono::duration<double, std::milli>(bt1 - bt0).count();
            pending_timing.build_validate_ms +=
                std::chrono::duration<double, std::milli>(bt2 - bt1).count();
        }
        // A rejected draw has already spent build_R and both validations, and they are
        // counted above. Dropping it without accounting would move that time into the
        // residual, where it reads as unattributed work -- the exact defect this
        // partition exists to make visible.
        if (!built_resources.complete || !contract_ok) {
            // #3891: a dropped draw is a correctness alarm, not only a timing bucket,
            // and it carries the site that dropped it.
            prosper::diagnostics::perf::drop_draw(built_resources.complete
                ? DropReason::ContractMismatch : built_resources.drop_reason);
            if (timing_enabled) ++pending_timing.build_rejected;
            continue;
        }
        poison_R(bd.R, bd.vs_words(), it.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex);
        poison_R(bd.R, bd.fs_words(), it.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment);
        const auto bt3 = timing_enabled ? RenderClock::now() : RenderClock::time_point{};
        // Indexed draw: hand the executor-fetched index data to the backend (vkCmdDrawIndexed).
        // Skipped under REFVS — the reference VS is a 3-vertex non-indexed fullscreen triangle.
        if (!refvs) {
            if (use_direct_index_views) bd.borrow_indices(it.indices);
            else bd.indices = it.indices;
        }
        if (timing_enabled) {
            const auto bt4 = RenderClock::now();
            pending_timing.build_poison_ms +=
                std::chrono::duration<double, std::milli>(bt3 - bt2).count();
            pending_timing.build_indices_ms +=
                std::chrono::duration<double, std::milli>(bt4 - bt3).count();
            // Counted AFTER the reject, so `build_draws` is accepted draws — and the
            // leaves do not all share it. `build_R` and `validate` ran for
            // draws + rejected; `poison` and `indices` ran for draws alone. Both
            // numbers are printed so the reader can pick the right denominator, and
            // the report line says which is which — a per-draw figure divided by the
            // wrong one over-states silently, and with rejected=0 nobody would notice
            // the rule until the first run where it is not.
            ++pending_timing.build_draws;
        }
        if (gfxlog) fprintf(stderr,
            "[render] item %zu: %zu resources vcount=%u instances=%u nidx=%zu topo=%u mask=0x%x blend=%d\n",
            bds.size(), bd.R.size() + bd.B.size(), bd.vcount, bd.instance_count,
            bd.index_count(), it.ps.topology,
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
                    (unsigned long long)it.color0_base, bd.vcount, bd.index_count(),
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
        // PROSPER_DRAW_PROGRAM_CENSUS: which programs does this title actually draw
        // with? One line per distinct (vs, vs-chain, ps) triple, then powers of two --
        // bounded by the program count, not the draw count. This is the census that
        // supplies PROSPER_SKIP_DRAW_PROGRAM's input.
        if (program_census) {
            const auto sighting = prosper::gpu::draw_program_census().observe(
                it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
            if (sighting.print)
                fprintf(stderr, "[draw-program] %s vs=0x%llx chain=0x%llx ps=0x%llx "
                        "draws=%llu distinct=%llu vcount=%u nidx=%zu topo=%u tgt=0x%llx\n",
                        sighting.first ? "NEW " : "    ",
                        (unsigned long long)it.vs_guest_addr,
                        (unsigned long long)it.vs_chain_guest_addr,
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)sighting.ordinal,
                        (unsigned long long)sighting.distinct,
                        bd.vcount, bd.index_count(), it.ps.topology,
                        (unsigned long long)it.color0_base);
        }
        // PROSPER_DRAW_LINKSCAN: census the LINKED LISTS this draw's scalar buffers
        // contain, taken from the exact bytes prosper is about to upload -- not from
        // guest memory, because the question is what the SHADER sees. Ordered with the
        // other observers and before the skip, for the trap-166 reason below.
        if (link_scan_armed &&
            link_scan.matches(it.vs_guest_addr, it.vs_chain_guest_addr,
                              it.fs_guest_addr)) {
            const auto& lcfg = prosper::gpu::draw_link_scan_settings();
            const uint32_t* heads_words = nullptr; size_t heads_count = 0;
            const uint32_t* rec_words = nullptr;   size_t rec_count = 0;
            size_t buffers_seen = 0;
            std::vector<const prosper::test::FrameBufferResource*> link_resources;
            link_resources.reserve(bd.R.size() + bd.B.size());
            if (bd.resource_order.empty()) {
                for (const auto& fr : bd.R)
                    if (!fr.is_texture() && !fr.is_storage_image)
                        link_resources.push_back(&fr);
                for (const auto& fr : bd.B) link_resources.push_back(&fr);
            } else {
                for (uint32_t token : bd.resource_order) {
                    const bool compact = (token & kCompactBufferResourceBit) != 0;
                    const uint32_t index = token & ~kCompactBufferResourceBit;
                    if (compact) link_resources.push_back(&bd.B[index]);
                    else if (!bd.R[index].is_texture() && !bd.R[index].is_storage_image)
                        link_resources.push_back(&bd.R[index]);
                }
            }
            for (const auto* fr_ptr : link_resources) {
                const auto& fr = *fr_ptr;
                const uint32_t* words = fr.buffer_words_data();
                const size_t nwords = fr.buffer_word_count();
                if (!words || nwords == 0) continue;
                ++buffers_seen;
                // The DECLARED descriptor beside the UPLOADED bytes. A short upload is
                // the mis-resolved-descriptor case and is invisible in the census alone:
                // the shader reads zeros past the end and a zero link is not an exit.
                uint64_t res_addr = 0; uint32_t res_declared = 0; int res_cls = -1;
                const prosper::gpu::ShaderResourceTable* table =
                    fr.set == 0 ? it.vrt.get() : it.prt.get();
                if (table)
                    for (const auto& r : table->resources)
                        if (r.binding == fr.binding) {
                            res_addr = r.gpu_addr; res_declared = r.size;
                            res_cls = static_cast<int>(r.cls);
                            break;
                        }
                if (fr.set == 1 && fr.binding == lcfg.heads_binding) {
                    heads_words = words; heads_count = nwords;
                }
                if (fr.set == 1 && fr.binding == lcfg.records_binding) {
                    rec_words = words; rec_count = nwords;
                }
                bool exhausted = false;
                const uint32_t scan_ordinal = link_scan.should_scan(
                    it.fs_guest_addr, fr.set, fr.binding,
                    lcfg.max_scans_per_buffer, &exhausted);
                if (!scan_ordinal) {
                    if (exhausted)
                        fprintf(stderr,
                                "[linkscan] capped ps=0x%llx set=%u binding=%u "
                                "addr=0x%llx after %u scan(s) -- later draws of this "
                                "buffer are NOT scanned\n",
                                (unsigned long long)it.fs_guest_addr, fr.set,
                                fr.binding, (unsigned long long)res_addr,
                                lcfg.max_scans_per_buffer);
                    continue;
                }
                const std::span<const uint32_t> span(words, nwords);
                const auto hist = prosper::gpu::histogram_words(
                    span, lcfg.encoding.terminator);
                const auto self = prosper::gpu::census_self_walk(span, lcfg.encoding);
                fprintf(stderr,
                        "[linkscan] ps=0x%llx draw#%llu scan=%u set=%u binding=%u "
                        "cls=%d addr=0x%llx declared=%u uploaded_dw=%zu | "
                        "hist zero=%u term=%u other=%u first_other=[%u]=0x%08x "
                        "other_range=0x%08x..0x%08x | self-walk stride=%u next=+%u "
                        "term=0x%08x records=%u starts=%u terminating=%u cyclic=%u "
                        "cycle-nodes=%u oob-starts=%u longest=%u\n",
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)it.draw_index, scan_ordinal, fr.set,
                        fr.binding, res_cls, (unsigned long long)res_addr, res_declared,
                        nwords, hist.zero, hist.terminator, hist.other,
                        hist.first_other_index, hist.first_other_value, hist.min_other,
                        hist.max_other, lcfg.encoding.record_stride_dwords,
                        lcfg.encoding.next_dword_offset, lcfg.encoding.terminator,
                        self.records, self.starts, self.terminating, self.cyclic,
                        self.cycle_nodes, self.oob_starts, self.longest);
                for (uint32_t k = 0; k < self.sample_count; ++k)
                    fprintf(stderr,
                            "[linkscan]   cycle-member[%u] link=%u -> next=%u\n", k,
                            self.sample_link[k], self.sample_next[k]);
                // WHO last wrote these bytes. An all-zero buffer has two completely
                // different causes -- a producer that never ran, and a producer that
                // ran and wrote zeros -- and the census alone cannot tell them apart.
                // The recorder summary is printed with the answer, unconditionally,
                // because a bounded and per-kind-gated history makes "no writer"
                // a statement about the INSTRUMENT unless the reader can see which
                // recorders fired (writer_provenance.hpp's scope list, #2111).
                if (res_addr && res_declared) {
                    if (!prosper::gpu::writer_provenance_enabled()) {
                        static bool provenance_note = false;
                        if (!provenance_note) {
                            provenance_note = true;
                            fprintf(stderr,
                                    "[linkscan]   writer provenance is OFF -- set "
                                    "PROSPER_WRITER_PROVENANCE=1 to learn whether an "
                                    "all-zero buffer was never written or was written "
                                    "with zeros\n");
                        }
                    } else {
                        const auto writer = prosper::gpu::last_guest_write_overlap(
                            res_addr, res_declared);
                        if (writer)
                            fprintf(stderr,
                                    "[linkscan]   last writer binding=%u kind=%s "
                                    "addr=0x%llx size=%llu submit=%llu item=%llu "
                                    "order=%llu identity=0x%llx | recorders %s\n",
                                    fr.binding,
                                    prosper::gpu::guest_writer_kind_name(writer->kind),
                                    (unsigned long long)writer->addr,
                                    (unsigned long long)writer->size,
                                    (unsigned long long)writer->submit,
                                    (unsigned long long)writer->item,
                                    (unsigned long long)writer->order,
                                    (unsigned long long)writer->identity,
                                    prosper::gpu::guest_write_recorder_summary());
                        else
                            fprintf(stderr,
                                    "[linkscan]   last writer binding=%u NONE -- no "
                                    "recorded write overlaps 0x%llx+%u | history=%zu "
                                    "recorders %s\n",
                                    fr.binding, (unsigned long long)res_addr,
                                    res_declared,
                                    prosper::gpu::guest_write_history_size(),
                                    prosper::gpu::guest_write_recorder_summary());
                    }
                }
            }
            if (heads_words && rec_words) {
                const auto heads = prosper::gpu::census_head_walk(
                    std::span<const uint32_t>(rec_words, rec_count),
                    std::span<const uint32_t>(heads_words, heads_count),
                    lcfg.encoding);
                fprintf(stderr,
                        "[linkscan] ps=0x%llx draw#%llu HEAD-WALK heads_binding=%u "
                        "(%zu dw) records_binding=%u (%zu dw) starts=%u "
                        "terminating=%u cyclic=%u cycle-nodes=%u oob-starts=%u "
                        "longest=%u\n",
                        (unsigned long long)it.fs_guest_addr,
                        (unsigned long long)it.draw_index, lcfg.heads_binding,
                        heads_count, lcfg.records_binding, rec_count, heads.starts,
                        heads.terminating, heads.cyclic, heads.cycle_nodes,
                        heads.oob_starts, heads.longest);
            }
            // Silence would otherwise be indistinguishable between "the program never
            // drew" and "it drew with no buffers at all". Say which, once.
            if (buffers_seen == 0) {
                static std::set<uint64_t> no_buffer_reported;
                if (no_buffer_reported.insert(it.fs_guest_addr).second)
                    fprintf(stderr,
                            "[linkscan] ps=0x%llx draw#%llu matched but has NO storage "
                            "buffer resources -- nothing to census\n",
                            (unsigned long long)it.fs_guest_addr,
                            (unsigned long long)it.draw_index);
            }
        }
        // PROSPER_SKIP_DRAW_PROGRAM: withhold this draw from the GPU because one of its
        // programs is named. LAST in the per-draw build on purpose -- instrument trap
        // 166: a diagnostic skip ordered before the dump blinds every other instrument
        // to the thing under suspicion, and the program worth declining is precisely the
        // one that hangs the GPU. So the draw is fully realized, fully validated and
        // fully logged by [render]/[rtt]/[draw-program] first; only the Vulkan draw call
        // is withheld. Its CPU-side cost is therefore real, and the timing partition
        // above has already counted it in build_draws, which is arithmetically right --
        // it did spend build_R, validation, poison and indices.
        if (program_skip_armed) {
            const auto decision = program_skip.evaluate(
                it.vs_guest_addr, it.vs_chain_guest_addr, it.fs_guest_addr);
            if (decision.skip) {
                if (decision.print)
                    fprintf(stderr,
                            "[draw-decline] program=0x%llx stage=%s "
                            "reason=skipped-by-selector count=%llu draw#%llu order=%llu "
                            "vs=0x%llx chain=0x%llx ps=0x%llx vcount=%u nidx=%zu "
                            "tgt=0x%llx\n",
                            (unsigned long long)decision.address,
                            prosper::gpu::draw_program_stage_name(decision.stage),
                            (unsigned long long)decision.ordinal,
                            (unsigned long long)it.draw_index,
                            (unsigned long long)it.command_order,
                            (unsigned long long)it.vs_guest_addr,
                            (unsigned long long)it.vs_chain_guest_addr,
                            (unsigned long long)it.fs_guest_addr,
                            bd.vcount, bd.index_count(),
                            (unsigned long long)it.color0_base);
                continue;
            }
        }
        bds.push_back(std::move(bd));
    }
    return bds;
}
} // namespace

// ---- The per-target pass loop (#3892) ------------------------------------------------------------
//
// PROSPER_RTT per-target rendering: splits the submit's draws into passes by color target,
// renders each pass into its own retained target (resolves, MRT slots, RTT publication and the
// present-candidate bookkeeping included), and leaves the composite's pixels for the present
// selection that follows. This was the body of the submit callback's `if (pertarget)` branch; it
// moved here verbatim. What it shares with the rest of the callback is named, not templated:
// build_backend_draws and record_backend_timing_stats over the callback's contexts, plus the
// clear_for / print_rtt_timing helpers.
namespace {
// The submit callback's state that render_per_target_passes reads and writes, one reference per object.
struct PerTargetPassContext {
    RttCache& g_rtt;
    std::atomic<uint64_t>& g_pass_log_submit;
    prosper::frontend::DiagnosticWindow& g_pass_log_window;
    prosper::frontend::DiagnosticWindow& g_persist_window;
    const PersistentReadbackFilter& g_persist_filter;
    const std::pair<uint32_t, uint32_t>& g_persist_extent;
    std::atomic<int>& frame_no;
    const bool& live_gpu_targets;
    const bool& defer_intermediate_scanout;
    const bool& batch_backend_submits;
    const std::vector<prosper::gpu::DrawItem> & items;
    uint32_t& w;
    uint32_t& h;
    const prosper::gpu::LiveRenderPhase& phase;
    std::vector<PinnedScanout>& pinned_scanouts;
    std::vector<PinnedRendererMipTarget>& pinned_renderer_mip_targets;
    int& g_this_submit;
    const bool& rtt_log;
    RenderTiming& pending_timing;
    std::vector<RttTimingRecord>& pending_rtt_timing;
    const bool& timing_enabled;
    const bool& lightweight_rtt_timing;
    std::vector<std::vector<uint8_t>>& texstore;
    std::vector<bool>& texstore_pinned;
    size_t& texstore_used;
    std::unordered_map<TextureDecodeKey, DecodedTexture, TextureDecodeKeyHash>& decoded_textures;
    uint32_t& resource_hash_w;
    uint32_t& resource_hash_h;
    uint32_t& target_step_w;
    uint32_t& target_step_h;
    const size_t& target_step_min_draws;
    RenderClock::time_point& pass_tail_start;
    double& pass_tail_measured_before;
    const RenderClock::time_point& pass_timing_start;
    std::shared_ptr<const std::vector<uint8_t>>& selected_pixels;
    uint64_t& selected_source_submit;
    const size_t& present_extent_bytes;
    prosper::frontend::PresentSourceChoice& present_choice;
    uint32_t& px_front_w;
    uint32_t& px_front_h;
    uint32_t& px_vo_w;
    uint32_t& px_vo_h;
    uint32_t& px_last_w;
    uint32_t& px_last_h;
    uint64_t& px_front_base;
    uint64_t& px_vo_base;
    uint64_t& px_last_base;
    uint64_t& px_front_source_submit;
    uint64_t& px_vo_source_submit;
    uint64_t& px_last_source_submit;
    uint32_t& px_front_fmt;
    uint32_t& px_vo_fmt;
    uint32_t& px_last_fmt;
    BackendDrawContext& backend_draw_ctx;
    BackendTimingContext& backend_timing_ctx;
};

void render_per_target_passes(PerTargetPassContext& ctx) {
    // Every name the moved body used from the callback, bound once to the same object.
    auto& g_pass_log_submit = ctx.g_pass_log_submit;
    auto& g_pass_log_window = ctx.g_pass_log_window;
    auto& g_persist_window = ctx.g_persist_window;
    auto& g_persist_filter = ctx.g_persist_filter;
    auto& g_persist_extent = ctx.g_persist_extent;
    auto& frame_no = ctx.frame_no;
    auto& live_gpu_targets = ctx.live_gpu_targets;
    auto& defer_intermediate_scanout = ctx.defer_intermediate_scanout;
    auto& batch_backend_submits = ctx.batch_backend_submits;
    auto& items = ctx.items;
    auto& w = ctx.w;
    auto& h = ctx.h;
    auto& phase = ctx.phase;
    auto& pinned_scanouts = ctx.pinned_scanouts;
    auto& pinned_renderer_mip_targets = ctx.pinned_renderer_mip_targets;
    auto& g_this_submit = ctx.g_this_submit;
    auto& rtt_log = ctx.rtt_log;
    auto& pending_timing = ctx.pending_timing;
    auto& pending_rtt_timing = ctx.pending_rtt_timing;
    auto& timing_enabled = ctx.timing_enabled;
    auto& lightweight_rtt_timing = ctx.lightweight_rtt_timing;
    auto& texstore = ctx.texstore;
    auto& texstore_pinned = ctx.texstore_pinned;
    auto& texstore_used = ctx.texstore_used;
    auto& decoded_textures = ctx.decoded_textures;
    auto& resource_hash_w = ctx.resource_hash_w;
    auto& resource_hash_h = ctx.resource_hash_h;
    auto& target_step_w = ctx.target_step_w;
    auto& target_step_h = ctx.target_step_h;
    auto& target_step_min_draws = ctx.target_step_min_draws;
    auto& pass_tail_start = ctx.pass_tail_start;
    auto& pass_tail_measured_before = ctx.pass_tail_measured_before;
    auto& pass_timing_start = ctx.pass_timing_start;
    auto& selected_pixels = ctx.selected_pixels;
    auto& selected_source_submit = ctx.selected_source_submit;
    auto& present_extent_bytes = ctx.present_extent_bytes;
    auto& present_choice = ctx.present_choice;
    auto& px_front_w = ctx.px_front_w;
    auto& px_front_h = ctx.px_front_h;
    auto& px_vo_w = ctx.px_vo_w;
    auto& px_vo_h = ctx.px_vo_h;
    auto& px_last_w = ctx.px_last_w;
    auto& px_last_h = ctx.px_last_h;
    auto& px_front_base = ctx.px_front_base;
    auto& px_vo_base = ctx.px_vo_base;
    auto& px_last_base = ctx.px_last_base;
    auto& px_front_source_submit = ctx.px_front_source_submit;
    auto& px_vo_source_submit = ctx.px_vo_source_submit;
    auto& px_last_source_submit = ctx.px_last_source_submit;
    auto& px_front_fmt = ctx.px_front_fmt;
    auto& px_vo_fmt = ctx.px_vo_fmt;
    auto& px_last_fmt = ctx.px_last_fmt;
    // The callback's process-lifetime g_rtt is named, uncaptured, by a failure-cleanup lambda
    // the backend may run after this function returns (a variable with static storage duration
    // cannot be captured). A function-scope static reference binds once to that same object,
    // so the lambda keeps reaching it exactly as it did inside the callback.
    static RttCache& g_rtt = ctx.g_rtt;
    // build_bds and record_backend_timing, over the same contexts the callback's forwarders use:
    // each call is still a direct call into build_backend_draws / record_backend_timing_stats.
    auto build_bds = [&](const std::vector<const prosper::gpu::DrawItem*>& group,
                         prosper::test::BackendSubmissionBatch* producer_batch = nullptr) {
        return build_backend_draws(ctx.backend_draw_ctx, group, producer_batch);
    };
    auto record_backend_timing = [&](const prosper::test::BackendRenderTimingStats& backend,
                                     const prosper::test::BackendTextureUploadStats& textures,
                                     const prosper::test::BackendPipelineCacheStats& pipelines,
                                     const prosper::test::BackendResourceReuseStats& reuse) {
        return record_backend_timing_stats(ctx.backend_timing_ctx, backend, textures, pipelines, reuse);
    };
    auto pin_renderer_mip_target = [&](uint64_t id, uint32_t width, uint32_t height,
                                       VkFormat format, bool gpu_valid) {
        if (!gpu_valid || !id || !width || !height ||
            !prosper::frontend::renderer_mip_target_requires_submit_retention(format))
            return;
        const auto already_pinned = std::find_if(
            pinned_renderer_mip_targets.begin(), pinned_renderer_mip_targets.end(),
            [&](const PinnedRendererMipTarget& target) {
                return target.id == id && target.width == width &&
                       target.height == height && target.format == format;
            });
        if (already_pinned != pinned_renderer_mip_targets.end()) {
            // A new production supersedes any earlier consumption of the same identity.
            // It must remain retained for the next mip consumer, which may be in a later
            // logical submit.
            already_pinned->consumed = false;
            return;
        }
        // This is a narrow bridge for independently rendered packed-HDR mip levels, not a
        // second unbounded residency cache. GTA V's observed scene uses 22 identities;
        // retaining substantially more without one being consumed indicates a different
        // workload and must fall back visibly to the ordinary LRU contract.
        constexpr size_t kMaxPendingRendererMipTargets = 64;
        if (pinned_renderer_mip_targets.size() >= kMaxPendingRendererMipTargets) {
            const PinnedRendererMipTarget oldest = pinned_renderer_mip_targets.front();
            prosper::test::unpin_persistent_color_target(
                oldest.id, oldest.width, oldest.height, oldest.format);
            pinned_renderer_mip_targets.erase(pinned_renderer_mip_targets.begin());
            static std::atomic<uint64_t> overflows{0};
            const uint64_t occurrence = overflows.fetch_add(1) + 1;
            if (prosper::diag_should_print(occurrence))
                fprintf(stderr,
                        "[renderer-mips] pending producer cap reached #%llu; "
                        "released unconsumed target=0x%llx extent=%ux%u\n",
                        (unsigned long long)occurrence,
                        (unsigned long long)oldest.id, oldest.width, oldest.height);
        }
        if (prosper::test::pin_persistent_color_target(id, width, height, format)) {
            pinned_renderer_mip_targets.push_back({id, width, height, format, false});
            return;
        }
        // The target was proven GPU-valid immediately before this call, so a failed pin
        // means frontend/backend cache identity drift. Report it even without diagnostics:
        // silently continuing would recreate the missing-middle-mip corruption this pin
        // exists to prevent.
        static std::atomic<uint64_t> failures{0};
        const uint64_t occurrence = failures.fetch_add(1) + 1;
        if (prosper::diag_should_print(occurrence))
            fprintf(stderr,
                    "[renderer-mips] submit-retention pin FAILED #%llu: "
                    "target=0x%llx extent=%ux%u format=%d\n",
                    (unsigned long long)occurrence, (unsigned long long)id,
                    width, height, (int)format);
    };
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
    // px_front_*/px_vo_*/px_last_* (each candidate's own target extent, and px_last's
    // base) are declared at callback scope above so the final-span report can name which
    // sources this submit actually offered.
    // Render-target persistence (default on; PROSPER_RTT_NOSEED reverts to per-pass blue
    // clear): seed each group's framebuffer with the pixels last rendered into that SAME
    // target VA, so a pass that draws into an already-written target (UE4's UI pass onto
    // the backbuffer, incremental HUD updates) composites OVER the earlier content instead
    // of starting from the diagnostic clear. Real RT memory persists exactly this way.
    static const bool seed_rtt = getenv("PROSPER_RTT_NOSEED") == nullptr;
    const auto& no_seed_targets = rtt_no_seed_target_selector();
    const auto seed_target = [&](uint64_t target) {
        return seed_rtt && !(no_seed_targets.configured &&
            no_seed_targets.includes(target));
    };
    // Pass grouping and same-pass feedback detection must not disagree about what an
    // active binding is, so both go through frontends/shared/rtt/mrt_binding.hpp. These
    // were duplicated lambdas; a second, looser copy in the feedback path classified
    // stale named state as a live binding (#2550 review).
    auto color_binding = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::frontend::mrt_color_binding(draw, slot);
    };
    auto active_format = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::test::backend_color_format(static_cast<VkFormat>(
            prosper::frontend::mrt_raw_format(draw, slot)));
    };
    auto active_color = [](const prosper::gpu::DrawItem& draw, uint32_t slot) {
        return prosper::frontend::mrt_active_color(draw, slot, mrt_format_defined);
    };
    auto active_color_count = [](const prosper::gpu::DrawItem& draw) {
        return prosper::frontend::mrt_active_color_count(draw, mrt_format_defined);
    };
    // #3026 -- the slot-0/1 alias mirror, checked HERE, at the consumer.
    //
    // `DrawItem::color_targets[0]`/`[1]` mirror the named `color0_*`/`color1_*` triples
    // or are absent; every producer honours that and nothing enforced it. The moment
    // one does not, the readers below answer differently: this loop's pass target is
    // the RAW named field (`base`, and `native_w`/`native_h` for the framebuffer
    // extent), grouping is named-first (`mrt_pass_color_binding`), and the
    // active-binding rule behind the attachment count and feedback detection is
    // array-first (`mrt_color_binding`). #3023 is what one such disagreement cost: two
    // draws rendering to different addresses grouped into one pass, and the second
    // draw's surface was discarded with no pass, no published pixels and no diagnostic.
    //
    // It is checked here and not beside the assignments that establish it because an
    // assertion at `realize_draw_item`'s success exit would sit two lines below its own
    // mirror and could never fail. This is the place a divergence would have had to
    // survive to matter.
    //
    // Report-only, deliberately. Which representation should win is undecided, and
    // deciding it here would change the surface the renderer renders to.
    for (const auto& alias_draw : items) {
        prosper::frontend::mrt_check_color_alias_mirror(
            alias_draw,
            [](uint32_t slot,
               const prosper::gpu::DrawItem::ColorTargetBinding& carried,
               const prosper::gpu::DrawItem::ColorTargetBinding& named,
               uint64_t ordinal) {
                static const bool alias_log =
                    PROSPER_ENV_VALUE("PROSPER_MRT_ALIAS_LOG") != nullptr;
                if (ordinal > 8 && !alias_log) return;
                fprintf(stderr,
                        "[mrt-alias] c%u array=0x%llx %ux%u vs named=0x%llx %ux%u -- "
                        "the pass renders to the NAMED surface; grouping, feedback and "
                        "the attachment count may read the other one (#3026) x%llu%s\n",
                        slot, (unsigned long long)carried.base, carried.width,
                        carried.height, (unsigned long long)named.base, named.width,
                        named.height, (unsigned long long)ordinal,
                        ordinal == 8 && !alias_log
                            ? "  (further reports need PROSPER_MRT_ALIAS_LOG=1)" : "");
            });
    }
    size_t pass_i = 0;
    prosper::test::BackendSubmissionBatch backend_submission;
    if (timing_enabled)
        pending_timing.pass_head_ms += std::chrono::duration<double, std::milli>(
            RenderClock::now() - pass_timing_start).count();
    const auto pass_loop_start = timing_enabled
        ? RenderClock::now() : RenderClock::time_point{};
    const double pass_loop_measured_before = timing_enabled
        ? pending_timing.build_resources_ms + pending_timing.backend_ms : 0.0;
    while (pass_i < items.size()) {
        const auto group_start = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        // Counted at the TOP, before any `continue` can skip it -- so the difference
        // from pass_groups is exactly the number of iterations that returned without
        // rendering, which is the population the missing time must belong to.
        if (timing_enabled) ++pending_timing.pass_groups_seen;
        uint64_t base = items[pass_i].color0_base;
        const uint32_t requested_color_count = active_color_count(items[pass_i]);
        // PROSPER_MRT_CENSUS=1 — per slot, why an attachment did or did not become
        // active. A slot needs all three of base, a known format, and a non-zero write
        // mask; if any is absent the attachment is dropped silently and the draws that
        // wrote it produce nothing. On GTA V only c1 is ever published, while an exact
        // draw census shows 122,028 of 131,072 draws binding a colour target at SLOT 4 --
        // so one of these three is missing for that slot and nothing said which.
        if (PROSPER_ENV_ON("PROSPER_MRT_CENSUS")) {
            struct SlotCensus {
                std::atomic<uint64_t> seen{0}, has_base{0}, has_format{0},
                    has_mask{0}, active{0};
            };
            static std::array<SlotCensus, prosper::gpu::kColorTargetCount> census;
            static std::atomic<uint64_t> groups{0};
            const auto& d = items[pass_i];
            for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
                auto& c = census[slot];
                c.seen.fetch_add(1, std::memory_order_relaxed);
                if (color_binding(d, slot).base)
                    c.has_base.fetch_add(1, std::memory_order_relaxed);
                if (active_format(d, slot) != VK_FORMAT_UNDEFINED)
                    c.has_format.fetch_add(1, std::memory_order_relaxed);
                if (d.ps.color_targets[slot].write_mask ||
                    (slot == 0 && d.ps.color_write_mask) ||
                    (slot == 1 && d.ps.color1_write_mask))
                    c.has_mask.fetch_add(1, std::memory_order_relaxed);
                // The per-slot mask the DrawItem CARRIES, against the one its own two
                // mask registers IMPLY. render_state.cpp computes the first as
                // (cb_target_mask & cb_shader_mask) >> (slot*4) & 0xf, so a disagreement
                // means the per-slot array did not survive the path that built this
                // DrawItem -- and the array is what decides whether the attachment
                // exists at all.
                {
                    const uint32_t implied =
                        ((d.ps.cb_target_mask & d.ps.cb_shader_mask) >> (slot * 4u))
                        & 0xfu;
                    const uint32_t carried = d.ps.color_targets[slot].write_mask;
                    if (implied != carried && color_binding(d, slot).base) {
                        static std::mutex disagree_mutex;
                        static std::map<std::tuple<uint32_t, uint32_t, uint32_t>,
                                        uint64_t> disagree;
                        std::lock_guard lock(disagree_mutex);
                        if (disagree.size() < 48)
                            ++disagree[{slot, implied, carried}];
                        static uint64_t n = 0;
                        if (++n % 8192 == 0) {
                            fprintf(stderr,
                                    "[mrt-mask-disagree] per-slot write_mask carried by "
                                    "the DrawItem vs implied by its own registers:\n");
                            for (const auto& e : disagree)
                                fprintf(stderr,
                                        "[mrt-mask-disagree]   c%u implied=0x%x "
                                        "carried=0x%x  x%llu\n",
                                        std::get<0>(e.first), std::get<1>(e.first),
                                        std::get<2>(e.first),
                                        (unsigned long long)e.second);
                        }
                    }
                }
                if (active_color(d, slot))
                    c.active.fetch_add(1, std::memory_order_relaxed);
            }
            // WHICH of the two masks is narrow. The write mask is
            // cb_target_mask & cb_shader_mask, and both default to 0xffffffff when the
            // register was never seen -- so a zero nibble means a register really was
            // programmed narrow, and the fix differs completely depending on which.
            // Histogrammed only over groups where slot 4 HAS a base, i.e. exactly the
            // population where the dropped attachment matters.
            if (color_binding(d, 4).base) {
                static std::mutex mask_mutex;
                static std::map<std::pair<uint32_t, uint32_t>, uint64_t> masks;
                std::lock_guard lock(mask_mutex);
                if (masks.size() < 64)
                    ++masks[{d.ps.cb_target_mask, d.ps.cb_shader_mask}];
                static uint64_t reported = 0;
                if (++reported % 4096 == 0) {
                    fprintf(stderr, "[mrt-census] masks where c4 has a base:\n");
                    for (const auto& e : masks)
                        fprintf(stderr,
                                "[mrt-census]   cb_target_mask=0x%08x "
                                "cb_shader_mask=0x%08x -> effective=0x%08x  x%llu\n",
                                e.first.first, e.first.second,
                                e.first.first & e.first.second,
                                (unsigned long long)e.second);
                }
            }
            // The SHAPE of each distinct pass: its eight slot bases next to the two
            // mask registers. The aggregate above says how often a slot activates; it
            // cannot say which surfaces a given pass meant to write, which is what
            // "buffer X is sampled 23 times and never written" actually needs.
            {
                static std::mutex shape_mutex;
                static std::map<std::array<uint64_t, 10>, uint64_t> shapes;
                std::array<uint64_t, 10> shape{};
                for (uint32_t slot = 0; slot < 8u; ++slot)
                    shape[slot] = color_binding(d, slot).base;
                shape[8] = d.ps.cb_target_mask;
                shape[9] = d.ps.cb_shader_mask;
                // PROSPER_MRT_SHAPE_FOR=0xADDR[,...] restricts recording to passes
                // whose slot-0 base is named. Without it the map fills with startup and
                // scanout passes long before the gameplay G-buffer pass appears, and an
                // unfiltered top-10 is then a list of the most FREQUENT shapes rather
                // than the ones being asked about.
                static const std::string shape_for =
                    PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR")
                        ? PROSPER_ENV_VALUE("PROSPER_MRT_SHAPE_FOR") : "";
                bool shape_wanted = shape_for.empty();
                if (!shape_wanted && shape[0]) {
                    char needle[24];
                    std::snprintf(needle, sizeof needle, "0x%llx",
                                  (unsigned long long)shape[0]);
                    shape_wanted = shape_for.find(needle) != std::string::npos;
                }
                if (!shape_wanted) goto shape_done;
                {
                std::lock_guard lock(shape_mutex);
                if (shapes.size() < 96 || shapes.count(shape)) ++shapes[shape];
                static uint64_t shape_reports = 0;
                if (++shape_reports % (shape_for.empty() ? 8192 : 256) == 0) {
                    std::vector<std::pair<uint64_t, std::array<uint64_t, 10>>> ranked;
                    for (const auto& e : shapes) ranked.push_back({e.second, e.first});
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first; });
                    fprintf(stderr, "[mrt-shape] %zu distinct pass shapes\n",
                            shapes.size());
                    for (size_t i = 0; i < ranked.size() && i < 10; ++i) {
                        fprintf(stderr, "[mrt-shape]   x%-6llu tmask=0x%08llx "
                                "smask=0x%08llx bases:",
                                (unsigned long long)ranked[i].first,
                                (unsigned long long)ranked[i].second[8],
                                (unsigned long long)ranked[i].second[9]);
                        for (uint32_t slot = 0; slot < 8u; ++slot)
                            if (ranked[i].second[slot])
                                fprintf(stderr, " c%u=0x%llx", slot,
                                        (unsigned long long)ranked[i].second[slot]);
                        fprintf(stderr, "\n");
                    }
                }
                }
                shape_done: ;
            }
            const uint64_t g = groups.fetch_add(1) + 1;
            if ((g & (g - 1)) == 0 && g >= 4096) {
                fprintf(stderr, "[mrt-census] pass groups=%llu\n",
                        (unsigned long long)g);
                for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot)
                    fprintf(stderr,
                            "[mrt-census]   c%u base=%llu format=%llu mask=%llu "
                            "ACTIVE=%llu\n",
                            slot,
                            (unsigned long long)census[slot].has_base.load(),
                            (unsigned long long)census[slot].has_format.load(),
                            (unsigned long long)census[slot].has_mask.load(),
                            (unsigned long long)census[slot].active.load());
            }
        }
        std::array<uint64_t, prosper::gpu::kColorTargetCount> pass_bases{};
        std::array<VkFormat, prosper::gpu::kColorTargetCount> pass_formats{};
        for (uint32_t slot = 0; slot < requested_color_count; ++slot) {
            pass_bases[slot] = slot ? active_color(items[pass_i], slot) : base;
            pass_formats[slot] = active_format(items[pass_i], slot);
        }
        const uint64_t base1 = pass_bases[1];
        const VkFormat format0 = pass_formats[0];
        const VkFormat format1 = base1
            ? pass_formats[1] : VK_FORMAT_UNDEFINED;
        // The DEPTH/STENCIL attachment is part of a pass's target identity, not just its
        // colour attachments.
        //
        // A render pass has ONE depth attachment, and the backend picks one cached DS
        // image for the whole grouped call from its first meaningful draw. Grouping on
        // colour alone therefore let a single call span draws that name different DS
        // surfaces -- or, for a layered surface, different DB_DEPTH_VIEW SLICES of one
        // allocation. Every such draw then rendered into the first face the call
        // happened to select.
        //
        // Measured on GTA V (Codex, #2542): one grouped call crosses DB_DEPTH_VIEW from
        // slice 0 to slice 1 at draw 65, so several guest cube faces were being rendered
        // into one host face. That also makes any "the cube has six valid faces"
        // measurement void: the handles existed, their contents did not correspond to
        // six guest faces.
        auto ds_identity = [](const prosper::gpu::DrawItem& draw) {
            return std::tuple(draw.ps.depth_read_base, draw.ps.depth_write_base,
                              draw.ps.stencil_read_base, draw.ps.stencil_write_base,
                              draw.ps.htile_data_base, draw.ps.db_depth_size_xy,
                              prosper::test::ds_depth_view_slice_start(
                                  draw.ps.db_depth_view));
        };
        const prosper::gpu::DrawItem& pass_head = items[pass_i];
        const auto ds0 = ds_identity(pass_head);
        std::vector<const prosper::gpu::DrawItem*> pass;
        auto same_targets = [&](const prosper::gpu::DrawItem& draw) {
            if (!prosper::frontend::mrt_same_color_pass(
                    pass_head, draw, mrt_format_defined, active_format))
                return false;
            // One 3D allocation can be written through several bounded slice views.
            // Grouping them into one framebuffer would route later draws to the
            // first view even though their guest addresses and 2D extents match.
            const auto& first_view = pass_head.color_targets[0];
            const auto& next_view = draw.color_targets[0];
            if (std::tuple(first_view.selected_mip_depth,
                           first_view.first_slice, first_view.slice_count) !=
                std::tuple(next_view.selected_mip_depth,
                           next_view.first_slice, next_view.slice_count))
                return false;
            if (first_view.selected_mip_depth &&
                std::tuple(first_view.tile_mode, first_view.mip_level,
                           first_view.in_mip_tail,
                           first_view.native_layout_known) !=
                std::tuple(next_view.tile_mode, next_view.mip_level,
                           next_view.in_mip_tail,
                           next_view.native_layout_known))
                return false;
            if (ds_identity(draw) != ds0) return false;
            return true;
        };
        // A MODE=RESOLVE draw is answered by a COPY below, not by a render, so its
        // pass identity is the source/destination pair that copy acts with rather than
        // the attachment set `same_targets` compares. `mrt_same_resolve_pass` carries
        // that whole rule -- keeping a resolve out of a scene group, and keeping two
        // resolves with different destinations out of each other's (#3025) -- and its
        // header states why each half exists.
        // Array snapshots are materialized by build_bds BEFORE this group's
        // commands are recorded. Flush a preceding writer as its own group before
        // preparing an aliased consumer, even when all attachment identities match.
        // The bridge can then submit the pending producer batch before reading it.
        bool pass_writes_depth = false;
        const auto samples_pass_depth_array = [&](const prosper::gpu::DrawItem& draw) {
            const auto aliases = [&](const prosper::gpu::ShaderResourceTable* table) {
                if (!table) return false;
                for (const auto& resource : table->resources) {
                    if (resource.cls != RC::Texture || resource.img_dim != 5u ||
                        resource.depth <= 1u ||
                        resource.format != prosper::gpu::DataFormat::Float32 ||
                        !resource.gpu_addr)
                        continue;
                    if (resource.gpu_addr == pass_head.ps.depth_read_base ||
                        resource.gpu_addr == pass_head.ps.depth_write_base)
                        return true;
                }
                return false;
            };
            return aliases(draw.vrt.get()) || aliases(draw.prt.get());
        };
        // Same four conditions in the same short-circuit order as before; the only
        // change is that the loop now records WHICH one ended the pass. Pass length
        // sets what every fixed per-pass cost actually costs over a run, and the
        // aggregate cannot distinguish an irreducible target change from an
        // over-strict predicate. See pass_break_census.hpp.
        const size_t pass_begin = pass_i;
        prosper::gpu::PassBreak pass_break = prosper::gpu::PassBreak::EndOfItems;
        while (true) {
            if (pass_i >= items.size()) {
                pass_break = prosper::gpu::PassBreak::EndOfItems; break;
            }
            if (!same_targets(items[pass_i])) {
                pass_break = prosper::gpu::PassBreak::TargetsChanged; break;
            }
            if (!prosper::frontend::mrt_same_resolve_pass(pass_head, items[pass_i])) {
                pass_break = prosper::gpu::PassBreak::MrtResolveDiffers; break;
            }
            const auto& draw = items[pass_i];
            if (pass_writes_depth && samples_pass_depth_array(draw)) {
                pass_break = prosper::gpu::PassBreak::DepthFeedback; break;
            }
            pass.push_back(&draw); ++pass_i;
            pass_writes_depth |= prosper::test::persistent_ds_pass_may_write_depth(
                draw.ps.depth_clear_enable, draw.ps.depth_test_enable,
                draw.ps.depth_write_enable, draw.ps.depth_compare_op);
        }
        prosper::gpu::pass_break_census().note_break(pass_break, pass_i - pass_begin);

        // CB_COLOR_CONTROL.MODE=RESOLVE(3): the guest resolves an MSAA color0 surface into a
        // single-sample color1 destination (Blue Prince PPSA25009 resolves its 4x-MSAA scene
        // this way). prosper renders single-sample, so the resolve is a straight copy of the
        // already-rendered color0 surface into color1. Use the RAW color1_base, not
        // active_color1(): a fixed-function resolve exports nothing, so its color1_write_mask
        // is 0 and active_color1 would report no destination. Without this the resolved surface
        // the display later samples never receives the scene and the frame is a uniform fill.
        // Census every MODE=RESOLVE pass: source, destination and whether the
        // destination is one prosper can even express. prosper takes the destination
        // from `color1_base` alone, so a guest resolving into any other slot is
        // invisible to the resolve path -- a surface filled that way would read as
        // "written by nothing" in every other census.
        if (!pass.empty() && pass.front()->ps.cb_resolve &&
            PROSPER_ENV_ON("PROSPER_RESOLVE_CENSUS")) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint64_t>, uint64_t> seen;
            const uint64_t rsrc = pass.front()->color0_base;
            const uint64_t rdst = pass.front()->color1_base;
            std::lock_guard lock(mutex);
            const uint64_t n = ++seen[{rsrc, rdst}];
            if (n <= 2 || (n & (n - 1)) == 0)
                fprintf(stderr,
                        "[resolve-census] src=0x%llx dst(color1)=0x%llx x%llu%s\n",
                        (unsigned long long)rsrc, (unsigned long long)rdst,
                        (unsigned long long)n,
                        rdst ? "" : "  <-- NO EXPRESSIBLE DESTINATION");
        }
        static const bool no_resolve = getenv("PROSPER_NO_RESOLVE") != nullptr;
        if (!pass.empty() && pass.front()->ps.cb_resolve && !no_resolve) {
            const uint64_t rsrc = pass.front()->color0_base;
            const uint64_t rdst = pass.front()->color1_base;
            auto src_it = rsrc ? g_rtt.find(rsrc) : g_rtt.end();
            if (src_it != g_rtt.end() && src_it->second.volume_depth) {
                // Fixed-function resolve currently copies one 2D plane. A retained
                // volume cannot be inherited as a 2D destination or read back through
                // that path. Revoke any old destination instead of publishing an alias.
                prosper::test::invalidate_persistent_color_target(rdst);
                auto dst_it = rdst ? g_rtt.find(rdst) : g_rtt.end();
                if (dst_it != g_rtt.end() && dst_it->second.volume_guest_bytes) {
                    dst_it->second.rgba.reset();
                    dst_it->second.has_uniform_color = false;
                    dst_it->second.gpu_valid = false;
                } else if (dst_it != g_rtt.end()) {
                    g_rtt.erase(dst_it);
                }
                src_it = g_rtt.end();
            }
            if (src_it != g_rtt.end() && src_it->second.has_uniform_color)
                materialize_uniform_rtt(src_it->second);
            // Deferred RTT readback (#1284): the resolve is a CPU copy of the source's
            // pixels, and the source pass usually rendered EARLIER IN THIS BATCH with its
            // readback deferred. The consumer scan cannot see a resolve (it consumes via
            // cb_resolve, not a sampled resource), so materialize here: flush the pending
            // batch first — a mid-batch readback would otherwise return stale pixels —
            // then read the persistent image back once.
            // PROSPER_EAGER_RESOLVE_READBACK: restore the pre-#2266 behaviour, where the
            // CPU mirror was materialised here unconditionally. Kept as a permanent
            // bisection lever so the A/B for this change is single-variable on ONE binary,
            // and because this path sits on the #1334/#1382/#1287 evidence chain -- if a
            // title regresses, the first question is whether the lazy mirror caused it and
            // that has to be answerable without a rebuild.
            //
            // Measured before the change, Blue Prince gameplay window (#2266):
            //   resolve 5.1/submit [stall=20.76 read=20.63 copy=1.54] 63.3 MiB/submit
            // 63.3 MiB at 3.1 GB/s -- a PCIe readback rate -- for a CPU mirror that this
            // file materialises ON DEMAND everywhere else (see the readback at the compute
            // consumer, whose comment says intermediates "normally stay in the persistent
            // Vulkan target" and are materialised "only when an ordered compute dispatch
            // actually consumes the surface"). The resolve was doing eagerly what the rest
            // of the renderer does lazily, and its own comment already calls the CPU pixels
            // the failure case: "on failure the shared CPU pixels are the only truth".
            static const bool eager_resolve_readback =
                getenv("PROSPER_EAGER_RESOLVE_READBACK") != nullptr;
            if (src_it != g_rtt.end() && src_it->second.gpu_valid &&
                src_it->second.w && src_it->second.h && rdst) {
                RttSurf& src_surface = src_it->second;
                const VkFormat src_format =
                    prosper::test::backend_color_format(src_surface.format);
                const uint32_t src_bpp =
                    prosper::test::backend_color_bytes_per_pixel(src_format);
                const size_t src_bytes = static_cast<size_t>(src_surface.w) *
                    src_surface.h * src_bpp;
                if (src_bpp && eager_resolve_readback &&
                    (!src_surface.rgba || src_surface.rgba->size() != src_bytes)) {
                    const prosper::test::RenderVkCtx& ctx =
                        prosper::test::render_vk_ctx();
                    const auto rs0 = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    if (ctx.ok && backend_submission.pending())
                        backend_submission.submit_and_wait(ctx.dev, ctx.queue, false);
                    const auto rs1 = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    std::vector<uint8_t> materialized;
                    std::string error;
                    const bool read_ok = prosper::test::readback_persistent_color_target(
                            rsrc, src_surface.w, src_surface.h, src_format,
                            materialized, error);
                    if (timing_enabled) {
                        const auto rs2 = RenderClock::now();
                        pending_timing.resolve_stall_ms +=
                            std::chrono::duration<double, std::milli>(rs1 - rs0).count();
                        pending_timing.resolve_read_ms +=
                            std::chrono::duration<double, std::milli>(rs2 - rs1).count();
                        ++pending_timing.resolve_read_n;
                        pending_timing.resolve_bytes += src_bytes;
                    }
                    if (read_ok &&
                        materialized.size() == src_bytes) {
                        src_surface.rgba =
                            std::make_shared<const std::vector<uint8_t>>(
                                std::move(materialized));
                    } else {
                        static std::atomic<int> warned{0};
                        if (warned.fetch_add(1) < 24)
                            fprintf(stderr,
                                    "[rtt] resolve source readback failed: "
                                    "base=0x%llx extent=%ux%u error=%s\n",
                                    (unsigned long long)rsrc, src_surface.w,
                                    src_surface.h, error.c_str());
                    }
                }
            }
            // A GPU-only source is now the ordinary case, so the destination no longer
            // requires CPU pixels to exist before the resolve. `gpu_valid` is what the
            // copy below needs; `rgba` is inherited when the source happens to have it
            // and is otherwise materialised on demand by the consumer path, exactly as
            // for every other graphics-to-graphics intermediate.
            if (rsrc && rdst && src_it != g_rtt.end() &&
                (src_it->second.rgba || src_it->second.gpu_valid)) {
                // Copy the source RttSurf out before the g_rtt[rdst] insert: operator[] may
                // rehash and invalidate src_it before the assignment reads it.
                RttSurf resolved = src_it->second;   // shares pixels (shared_ptr), no deep copy
                // A resolve copies one current 2D plane. It does not transfer the
                // source's older unpublished volume to another address, nor does it
                // publish slices of a volume formerly owned by the destination.
                const auto old_destination = g_rtt.find(rdst);
                resolved.volume_depth = 0;
                resolved.volume_guest_bytes = old_destination != g_rtt.end()
                    ? old_destination->second.volume_guest_bytes : 0u;
                resolved.volume_footprint_proven = old_destination != g_rtt.end() &&
                    old_destination->second.volume_footprint_proven;
                // The resolve destination has its own descriptor/DCC allocation. Do not
                // transfer the source surface's metadata identity to an unrelated base.
                resolved.dcc_metadata_addr = 0;
                resolved.dcc_metadata_bytes = 0;
                resolved.dcc_metadata_dirty = false;
                const uint32_t rw = resolved.w, rh = resolved.h;
                bool batched_copy_recorded = false;
                // #1334: the destination-keyed persistent GPU image did NOT receive these
                // pixels — inheriting gpu_valid from the source let a later #780 CPU-copy
                // discard leave consumers importing a stale/zero image (Blue Prince's
                // compute tonemap read black; #1287/#1381 evidence chain). Copy the
                // device-local pixels into the destination identity so gpu_valid is
                // genuinely true; on failure the shared CPU pixels are the only truth.
                if (resolved.gpu_valid) {
                    // The control submits the copy out of band, so every earlier pass —
                    // including one targeting the destination — must complete first.
                    // The default records the copy into the same ordered batch instead;
                    // its barriers supply visibility and the final batch fence owns
                    // publication, failure invalidation and resource lifetime.
                    static const bool no_batched_resolve_copy =
                        PROSPER_ENV_VALUE("PROSPER_NO_BATCHED_RESOLVE_COPY") != nullptr;
                    const bool batch_resolve_copy =
                        prosper::frontend::resolve_copy_may_batch(
                            batch_backend_submits, no_batched_resolve_copy);
                    if (!batch_resolve_copy) {
                        const prosper::test::RenderVkCtx& copy_ctx =
                            prosper::test::render_vk_ctx();
                        const auto cs0 = timing_enabled
                            ? RenderClock::now() : RenderClock::time_point{};
                        if (copy_ctx.ok && backend_submission.pending())
                            backend_submission.submit_and_wait(copy_ctx.dev,
                                                               copy_ctx.queue, false);
                        if (timing_enabled)
                            pending_timing.resolve_copy_stall_ms +=
                                std::chrono::duration<double, std::milli>(
                                    RenderClock::now() - cs0).count();
                    }
                    std::string copy_error;
                    const auto rc0 = timing_enabled
                        ? RenderClock::now() : RenderClock::time_point{};
                    const bool copy_ok = prosper::test::copy_persistent_color_target(
                            rsrc, rdst, rw, rh,
                            prosper::test::backend_color_format(resolved.format),
                            copy_error,
                            batch_resolve_copy ? &backend_submission : nullptr);
                    batched_copy_recorded = copy_ok && batch_resolve_copy;
                    if (timing_enabled)
                        pending_timing.resolve_copy_ms +=
                            std::chrono::duration<double, std::milli>(
                                RenderClock::now() - rc0).count();
                    if (!copy_ok) {
                        resolved.gpu_valid = false;
                        // The CPU mirror is the documented fallback -- "on failure the
                        // shared CPU pixels are the only truth" -- so with the eager
                        // readback gone it has to be materialised HERE, on the failure
                        // path it was always meant for. Without this the destination
                        // would have neither a valid GPU image nor pixels, which is
                        // strictly worse than before this change rather than merely
                        // slower. The flush is required for the same reason the eager
                        // path needed one: a mid-batch readback returns stale pixels.
                        if (!resolved.rgba || resolved.rgba->size() !=
                                static_cast<size_t>(rw) * rh *
                                prosper::test::backend_color_bytes_per_pixel(
                                    prosper::test::backend_color_format(resolved.format))) {
                            const prosper::test::RenderVkCtx& fb_ctx =
                                prosper::test::render_vk_ctx();
                            if (fb_ctx.ok && backend_submission.pending())
                                backend_submission.submit_and_wait(fb_ctx.dev,
                                                                   fb_ctx.queue, false);
                            std::vector<uint8_t> fallback;
                            std::string fb_error;
                            if (prosper::test::readback_persistent_color_target(
                                    rsrc, rw, rh,
                                    prosper::test::backend_color_format(resolved.format),
                                    fallback, fb_error))
                                resolved.rgba =
                                    std::make_shared<const std::vector<uint8_t>>(
                                        std::move(fallback));
                        }
                        static std::atomic<int> warned{0};
                        if (warned.fetch_add(1) < 8)
                            fprintf(stderr,
                                    "[msaa] resolve GPU copy failed (%s) — destination "
                                    "keeps CPU pixels only (#1334)\n",
                                    copy_error.c_str());
                    }
                }
                g_rtt[rdst] = std::move(resolved);    // dest inherits src content/extent/format
                // The one whole-entry copy into the cache: it may carry a footprint.
                if (g_rtt[rdst].volume_guest_bytes) g_volume_targets.note(rdst);
                if (batched_copy_recorded) {
                    const VkFormat resolved_format = g_rtt[rdst].format;
                    backend_submission.add_failure_cleanup(
                        [rdst, rw, rh, resolved_format]() {
                            auto failed = g_rtt.find(rdst);
                            if (failed != g_rtt.end() && failed->second.w == rw &&
                                failed->second.h == rh &&
                                failed->second.format == resolved_format)
                                failed->second.gpu_valid = false;
                        });
                }
                if (PROSPER_ENV_ON("PROSPER_MSAA_LOG"))
                    fprintf(stderr, "[msaa] resolve copy 0x%llx -> 0x%llx (%ux%u)\n",
                            (unsigned long long)rsrc, (unsigned long long)rdst, rw, rh);
            } else {
                // A resolve that performs no copy is DROPPED RENDERED CONTENT: the
                // destination keeps whatever it held before, and every later census
                // reads it as "written by nothing". Report it unconditionally, bounded
                // like the copy-failure warning above, so the one remaining way a
                // resolve can produce nothing is fail-visible rather than silent.
                // PROSPER_MSAA_LOG still reports every occurrence.
                static std::atomic<int> skipped{0};
                if (PROSPER_ENV_VALUE("PROSPER_MSAA_LOG") ||
                    skipped.fetch_add(1) < 8)
                    fprintf(stderr, "[msaa] resolve SKIP src=0x%llx dst=0x%llx (%s)\n",
                            (unsigned long long)rsrc, (unsigned long long)rdst,
                            !rsrc || !rdst ? "missing base"
                                           : "source has no rendered surface");
            }
            if (timing_enabled) ++pending_timing.resolve_n;
            // A normal final render group flushes the ordered backend batch inside
            // render_draws_rgba. A resolve returns before that call, so a terminal
            // resolve must establish the same boundary here. Otherwise the tail below
            // can read back or publish g_rtt's speculative destination before the copy
            // has even reached the queue.
            if (prosper::frontend::terminal_resolve_must_flush(
                    batch_backend_submits, pass_i == items.size(),
                    backend_submission.pending())) {
                const prosper::test::RenderVkCtx& terminal_ctx =
                    prosper::test::render_vk_ctx();
                if (terminal_ctx.ok)
                    backend_submission.submit_and_wait(
                        terminal_ctx.dev, terminal_ctx.queue, false);
                else
                    backend_submission.discard();
            }
            continue;   // resolve is a copy, not an ordinary-draw render
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
        bool is_vo = false;
        for (int i = 0; i < vo_n && !is_vo; i++)
            is_vo = base && base == prosper_vo_buffer_addr(i);
        // VideoOut registration is the visible scanout contract. CB_COLOR0_ATTRIB2 can
        // describe an overallocated backing surface (Terminator reports 4096x4096 for a
        // 1920x1080 scanout), which must not become the persistent target extent or the
        // final cache lookup will reject the rendered frame.
        if (is_vo && present_w && present_h) {
            native_w = present_w;
            native_h = present_h;
        }
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
                for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot)
                    if (prosper::frontend::mrt_write_mask(*draw, slot)) return false;
                return true;
            });
        const bool uses_ds = std::any_of(pass.begin(), pass.end(), [](const auto* draw) {
            return draw->ps.depth_test_enable || draw->ps.depth_write_enable ||
                   draw->ps.depth_clear_enable || draw->ps.stencil_enable ||
                   draw->ps.stencil_clear_enable;
        });
        // A masked color attachment does not determine the depth allocation's size.
        // DB_DEPTH_SIZE_XY is the guest extent; both clear and caster passes must name
        // the same persistent depth image even when their inactive color bindings differ.
        // Resolved/captured state does not retain this register's presence flag. Zero
        // therefore remains ambiguous (absent versus explicit 1x1) and keeps the legacy
        // fallback. Pass grouping above keeps distinct DB extents separate.
        const auto* depth_state = pass.empty() ? nullptr : &pass.front()->ps;
        const bool explicit_depth_extent = depth_state && color_disabled && uses_ds &&
            (depth_state->depth_read_base || depth_state->depth_write_base ||
             depth_state->stencil_read_base || depth_state->stencil_write_base) &&
            depth_state->db_depth_size_xy != 0;
        if (explicit_depth_extent) {
            native_w = PM4_FIELD(depth_state->db_depth_size_xy, DB_DEPTH_SIZE_XY, X_MAX) + 1u;
            native_h = PM4_FIELD(depth_state->db_depth_size_xy, DB_DEPTH_SIZE_XY, Y_MAX) + 1u;
            // All color writes are disabled. The backend still needs a dummy color
            // attachment, but resizing it to the depth extent must not create or
            // publish a new color authority under an inactive guest CB address.
            // Zero pass identities select transient attachments and bypass color
            // seeding, retained-target publication and scanout selection below.
            base = 0;
            pass_bases.fill(0);
            is_vo = false;
        }
        if (color_disabled && uses_ds && !explicit_depth_extent && w && h) {
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
            // Depth-only surfaces legitimately exceed the presentation extent: Blue
            // Prince's directional-shadow cascades render translated 512x512/1024x1024
            // viewports into 2048x2048 and 4096x4096 atlases (#1275). Capping at the
            // presentation surface collapsed those atlases to 1x1, erasing every shadow.
            // The per-axis bound (each axis against its own presentation axis, or 4096)
            // keeps the Dead Cells pathology (translated viewports inferring
            // 16384x16384, the reason this cap exists) rejected and fail-visible.
            const bool viewport_extent_valid = viewport_native_w && viewport_native_h &&
                viewport_native_w <= std::max<uint64_t>(4096u, max_native_w) &&
                viewport_native_h <= std::max<uint64_t>(4096u, max_native_h);
            // Log the UNDECIDABLE case too. Gating this on a non-zero derived extent hid
            // the only outcome that silently changes the DS identity: a depth-only pass
            // whose draws carry no viewport register derives 0x0, falls through to the
            // global frame extent, and mints a second cache entry for a surface that
            // already has a correctly-sized one. A diagnostic that prints only when the
            // inference succeeded cannot report the inference not happening.
            if (PROSPER_ENV_ON("PROSPER_DSLOG")) {
                const size_t with_viewport = static_cast<size_t>(std::count_if(
                    pass.begin(), pass.end(),
                    [](const auto* draw) { return draw->ps.has_viewport; }));
                fprintf(stderr,
                        "[ds] viewport-derived extent %llux%llu (presentation %ux%u, "
                        "%zu/%zu draws with viewport) -> %s\n",
                        (unsigned long long)viewport_native_w,
                        (unsigned long long)viewport_native_h,
                        max_native_w, max_native_h, with_viewport, pass.size(),
                        viewport_extent_valid ? "accept"
                            : (viewport_native_w || viewport_native_h) ? "reject"
                                                                       : "undecidable");
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
            } else if (color_disabled && uses_ds) {
                // No presentation extent (offline replay): render scale is 1, so a
                // depth-only surface's viewport-derived extent is exact. Falling back
                // to the global w/h here silently truncated over-presentation atlases —
                // Blue Prince's 2048x2048 shadow atlas became a 1920x1080 DS image
                // whose sampled taps then missed the bridge (#1275). Color passes keep
                // the historical capture-extent truncation (their over-allocated
                // CB_COLOR0_ATTRIB2 backings are a different, hash-pinned contract).
                gw = native_w; gh = native_h;
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
        const float* retained_uniform_clear = nullptr;
        bool gpu_seed_available = false;
        const bool seed_rtt0 = seed_target(base);
        const VkFormat pass_format = format0;
        const auto& primary_volume_view = pass.front()->color_targets[0];
        const uint32_t producer_volume_depth =
            primary_volume_view.selected_mip_depth;
        g_ever_volume_target |= producer_volume_depth != 0u;
        const uint32_t volume_bpp =
            prosper::test::backend_color_bytes_per_pixel(format0);
        const uint32_t volume_native_w = native_w ? native_w : gw;
        const uint32_t volume_native_h = native_h ? native_h : gh;
        const bool producer_volume_layout_supported = producer_volume_depth &&
            primary_volume_view.native_layout_known &&
            !primary_volume_view.mip_level &&
            !primary_volume_view.in_mip_tail &&
            prosper::gpu::tile_mode_supports_volume(primary_volume_view.tile_mode);
        const uint64_t producer_volume_physical_bytes =
            producer_volume_layout_supported
                ? prosper::gpu::tiled_volume_bytes(
                      volume_native_w, volume_native_h, producer_volume_depth,
                      primary_volume_view.tile_mode, volume_bpp)
                : 0u;
        const bool producer_volume_footprint_proven =
            producer_volume_physical_bytes != 0u;
        const uint64_t producer_volume_guard_bytes =
            producer_volume_footprint_proven
                ? producer_volume_physical_bytes
                : prosper::frontend::live_rtt_color_footprint_bytes(
                      volume_native_w, volume_native_h,
                      producer_volume_depth, volume_bpp);
        uint32_t mrt_count = requested_color_count;
        if (PROSPER_ENV_ON("PROSPER_NO_MRT1") || PROSPER_ENV_ON("PROSPER_NO_MRT")) mrt_count = 1;
        if (!render_pass.empty()) {
            for (uint32_t slot = 1; slot < mrt_count; ++slot) {
                const auto binding = color_binding(*render_pass.front(), slot);
                // Sparse exports retain their native Location through dummy attachments.
                // Only a real target whose extent is KNOWN to differ from the pass
                // extent truncates the prefix. A 0x0 on either side means the guest's
                // CB_COLORn_ATTRIB2 was never seen, not a 0-pixel surface, and an
                // unmeasured extent is no evidence of a conflict — comparing it for
                // equality dropped the whole attachment silently (#2114).
                if (pass_bases[slot] && prosper::frontend::mrt_extent_conflicts(
                        binding.width, binding.height, native_w, native_h)) {
                    if (rtt_log)
                        fprintf(stderr,
                                "[rtt] truncate MRT prefix at c%u target=0x%llx "
                                "extent=%ux%u; MRT0 is %ux%u\n",
                                slot, (unsigned long long)pass_bases[slot],
                                binding.width, binding.height, native_w, native_h);
                    mrt_count = slot;
                    break;
                }
                // Fail-visible: name the attachment that was kept on missing extent
                // data, so an unmeasured extent is a reported state rather than a
                // silent one in either direction.
                if (rtt_log && pass_bases[slot] &&
                    (!prosper::frontend::mrt_extent_known(binding.width, binding.height) ||
                     !prosper::frontend::mrt_extent_known(native_w, native_h)))
                    fprintf(stderr,
                            "[rtt] keep MRT c%u target=0x%llx on unmeasured extent "
                            "%ux%u; MRT0 is %ux%u\n",
                            slot, (unsigned long long)pass_bases[slot],
                            binding.width, binding.height, native_w, native_h);
            }
        }
        const bool use_color1 = mrt_count > 1;
        const VkFormat pass_format1 = use_color1 ? format1 : VK_FORMAT_UNDEFINED;
        const size_t pass_bytes = static_cast<size_t>(gw) * gh *
            prosper::test::backend_color_bytes_per_pixel(pass_format);
        if (seed_rtt0 && base) { auto sit = g_rtt.find(base);
            gpu_seed_available = live_gpu_targets && sit != g_rtt.end() &&
                sit->second.gpu_valid && sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format &&
                sit->second.volume_depth == producer_volume_depth &&
                prosper::test::find_persistent_color_target(
                    base, gw, gh, pass_format, true,
                    producer_volume_depth) != nullptr;
            if (!gpu_seed_available && sit != g_rtt.end() &&
                !producer_volume_depth && !sit->second.volume_depth &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format && sit->second.rgba &&
                sit->second.rgba->size() == pass_bytes)
                seed = sit->second.rgba->data();
            if (!gpu_seed_available && !seed && sit != g_rtt.end() &&
                !producer_volume_depth && !sit->second.volume_depth &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format &&
                sit->second.has_uniform_color)
                retained_uniform_clear = sit->second.uniform_color.data();
            // Gated seed-decision diagnostic: a pass that should LOAD prior target
            // content but silently falls back to its clear color erases everything the
            // earlier pass produced (an opaque-black clear wipes a transparent UI RT —
            // #320's dialogue overlay). Make the decision and its reason visible.
            if (rtt_log && !seed && !gpu_seed_available) {
                if (sit == g_rtt.end())
                    fprintf(stderr, "[rtt] seed miss target=0x%llx reason=no-entry\n",
                            (unsigned long long)base);
                else
                    fprintf(stderr,
                            "[rtt] seed miss target=0x%llx reason=mismatch "
                            "entry=%ux%u fmt=%d rgba=%zu want=%ux%u fmt=%d bytes=%zu\n",
                            (unsigned long long)base, sit->second.w, sit->second.h,
                            (int)sit->second.format,
                            sit->second.rgba ? sit->second.rgba->size() : (size_t)0,
                            gw, gh, (int)pass_format, pass_bytes);
            } }
        struct LaterTargetConsumers {
            bool sampled_exact = false;
            bool feedback = false;
            bool cpu_needed = false;
            uint32_t storage_references = 0;
            uint32_t dimension_mismatches = 0;
            uint32_t extent_mismatches = 0;
            uint32_t feedback_references = 0;
        };
        // A later pass in THIS batch that reads the target's CPU bytes cannot be served
        // lazily: the producing commands are still unsubmitted when it binds, so a
        // mid-batch readback would return stale pixels. Only the direct GPU bind (2D
        // texture, exact extent, not feedback, not storage) is batch-ordered; every
        // other same-batch consumer forces the eager readback. A later color attachment
        // LOAD is now direct for both MRT attachments and needs no CPU copy. An exact
        // 3D texture may also borrow this producer's complete retained volume; other
        // volume consumers need a guest publication path that is not yet implemented.
        // pass_i has advanced past the current pass, so scanned items are genuine.
        // Cross-batch consumers materialize on demand at bind/seed/compute/DMA time
        // (#1284).
        const auto inspect_later_consumers = [&](uint64_t target_base) {
            LaterTargetConsumers result;
            if (!live_gpu_targets || !target_base) return result;
            static const uint32_t render_scale = [] {
                const char* e = PROSPER_ENV_VALUE("PROSPER_RENDER_SCALE");
                const long v = e ? std::strtol(e, nullptr, 10) : 1;
                return v > 0 ? static_cast<uint32_t>(v) : 1u;
            }();
            for (size_t later = pass_i; later < items.size(); ++later) {
                auto inspect = [&](const prosper::gpu::ShaderResourceTable* table) {
                    if (!table) return;
                    for (const auto& resource : table->resources) {
                        if ((resource.cls != RC::Texture &&
                             resource.cls != RC::StorageImage) ||
                            resource.gpu_addr != target_base ||
                            (resource.img_dim == 2u &&
                             (!producer_volume_depth || target_base != base)))
                            continue;
                        const uint32_t rw = resource.width ? resource.width : 4u;
                        const uint32_t rh = resource.height ? resource.height : 4u;
                        bool same_pass_target =
                            items[later].color0_base == target_base;
                        for (uint32_t slot = 1;
                             slot < prosper::gpu::kColorTargetCount; ++slot)
                            same_pass_target |=
                                active_color(items[later], slot) == target_base;
                        const bool sampled_extent_compatible =
                            prosper::frontend::rtt_sampled_extent_compatible(
                                rw, rh, gw, gh, render_scale, false);
                        const bool sampled_shape = producer_volume_depth &&
                                                   target_base == base
                            ? resource.img_dim == 2u &&
                              resource.depth == producer_volume_depth &&
                              resource.sample_count == 1u &&
                              resource.declared_mip_levels == 1u &&
                              !resource.in_mip_tail
                            : prosper::frontend::rtt_single_layer_sample_shape(
                                  resource.img_dim, resource.depth,
                                  resource.sample_count);
                        if (sampled_shape && sampled_extent_compatible) {
                            result.sampled_exact = true;
                            result.feedback |= same_pass_target;
                        }
                        // Exact 2D color feedback is GPU-bindable: the backend copies
                        // the prior attachment version to a distinct sampled image before
                        // the render pass. Storage and dimension mismatches still
                        // require the CPU representation. An extent mismatch is an alias
                        // rather than a consumer of this target.
                        const bool direct_bindable =
                            resource.cls == RC::Texture && sampled_shape &&
                            (resource.img_dim != 2u || !same_pass_target);
                        if (!sampled_extent_compatible) {
                            result.extent_mismatches++;
                        } else if (!direct_bindable) {
                            result.cpu_needed = true;
                            result.storage_references +=
                                resource.cls == RC::StorageImage;
                            result.dimension_mismatches += !sampled_shape;
                            result.feedback_references += same_pass_target;
                            static const uint64_t diagnose_min_submit = [] {
                                const char* text = PROSPER_ENV_VALUE(
                                    "PROSPER_READBACK_WHY_MIN_SUBMIT");
                                return text ? std::strtoull(text, nullptr, 0) : 0ull;
                            }();
                            static std::atomic<uint64_t> diagnose_lines{0};
                            if (PROSPER_ENV_ON("PROSPER_READBACK_WHY") &&
                                static_cast<uint64_t>(g_this_submit) >=
                                    diagnose_min_submit &&
                                diagnose_lines.fetch_add(
                                    1, std::memory_order_relaxed) < 64) {
                                fprintf(stderr,
                                        "[readback-consumer] submit=%d target=0x%llx "
                                        "producer-pass=%zu consumer-draw=%zu "
                                        "class=%s dim=%u extent=%ux%u target-extent=%ux%u "
                                        "storage=%u dimension=%u extent-mismatch=%u "
                                        "feedback=%u\n",
                                        g_this_submit,
                                        (unsigned long long)target_base, pass_i - 1,
                                        later,
                                        resource.cls == RC::StorageImage
                                            ? "storage" : "texture",
                                        resource.img_dim, rw, rh, gw, gh,
                                        resource.cls == RC::StorageImage ? 1u : 0u,
                                        !sampled_shape ? 1u : 0u,
                                        !sampled_extent_compatible ? 1u : 0u,
                                        same_pass_target ? 1u : 0u);
                            }
                        }
                    }
                };
                inspect(items[later].vrt.get());
                inspect(items[later].prt.get());
            }
            return result;
        };
        const LaterTargetConsumers consumers0 = inspect_later_consumers(base);
        const LaterTargetConsumers consumers1 = use_color1
            ? inspect_later_consumers(base1) : LaterTargetConsumers{};
        std::array<LaterTargetConsumers, prosper::gpu::kColorTargetCount>
            consumers_slots{};
        for (uint32_t slot = 2; slot < mrt_count; ++slot)
            consumers_slots[slot] = inspect_later_consumers(pass_bases[slot]);
        const bool sampled_exact_later = consumers0.sampled_exact;
        const bool feedback_later = consumers0.feedback;
        const bool cpu_needed_same_batch = consumers0.cpu_needed;
        if (producer_volume_depth &&
            (!producer_volume_footprint_proven || !live_gpu_targets ||
             phase.authoritative_readback ||
             cpu_needed_same_batch)) {
            // No 3D guest publication or same-pass feedback snapshot exists yet, so
            // the renderer cannot produce this volume in a form a consumer here can
            // read. Decline the producer. With no image, the volume claims nothing and
            // consumers read guest memory, which lacks this pass's writes. The
            // alternative was worse: claiming the footprint dropped every draw that
            // samples it, which kept GTA V's menus black whenever live GPU targets are
            // off (PROSPER_DUMP_*, replay seeding, and PROSPER_GPU_CAPTURE before
            // #3895; #3890).
            prosper::test::invalidate_persistent_color_target(base);
            if (base)
                note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                            producer_volume_depth, pass_format);
            static std::atomic<uint32_t> volume_refusals{0};
            if (volume_refusals.fetch_add(1, std::memory_order_relaxed) < 16u)
                std::fprintf(stderr,
                    "[render-volume] guest-observed or unsupported volume pass "
                    "target=0x%llx authoritative=%d cpu-consumer=%d live=%d "
                    "physical=%d mode=%u\n",
                    static_cast<unsigned long long>(base),
                    phase.authoritative_readback, cpu_needed_same_batch,
                    live_gpu_targets, producer_volume_footprint_proven,
                    primary_volume_view.tile_mode);
            continue;
        }
        static const bool defer_rtt_readback =
            !PROSPER_ENV_VALUE("PROSPER_NO_RTT_READBACK_DEFER");
        const bool rtt_defer_ok = defer_rtt_readback
            ? !cpu_needed_same_batch
            : (sampled_exact_later && !feedback_later);
        const bool rtt_defer_ok1 = defer_rtt_readback
            ? !consumers1.cpu_needed
            : (consumers1.sampled_exact && !consumers1.feedback);
        // Keep intermediate scanout spans GPU-resident too: they cannot publish until the
        // final callback, where the cache is materialized on demand if no later scanout
        // pass already requested CPU pixels. A same-submit DMA asks its producer span for
        // authoritative readback, and compute consumers use the lazy target reader above.
        // A pending one-shot capture reads back every pass's colour0 for its one
        // submit, so the publish candidates (front / scanout / last) that become the
        // capsule's output oracle are exactly the readback path's (#3895).
        const bool capture_cpu_output = prosper::gpu::gpu_capture_requires_cpu_output();
        const bool final_gpu_present = phase.final_span &&
            !phase.authoritative_readback &&
            prosper::frontend::gpu_present_allowed_during_capture(
                prosper::gpu::gpu_present_active(), capture_cpu_output);
        const bool defer_readback = live_gpu_targets && vo_n > 0 && base &&
            !phase.authoritative_readback &&
            prosper::frontend::pass_readback_deferral_allowed_during_capture(
                capture_cpu_output) &&
            ((is_vo && can_defer_scanout_readback(
                           phase.allows_deferred_scanout_readback(),
                           final_gpu_present,
                           defer_intermediate_scanout, cpu_needed_same_batch)) ||
             (!is_vo && (base != front_va || final_gpu_present) && rtt_defer_ok));
        const bool defer_readback1 = live_gpu_targets && vo_n > 0 && use_color1 &&
            base1 && base1 != base && !phase.authoritative_readback &&
            (base1 != front_va || final_gpu_present) && rtt_defer_ok1;
        std::array<bool, prosper::gpu::kColorTargetCount> defer_readback_slots{};
        for (uint32_t slot = 2; slot < mrt_count; ++slot) {
            const LaterTargetConsumers& consumers = consumers_slots[slot];
            const bool rtt_defer_ok_slot = defer_rtt_readback
                ? !consumers.cpu_needed
                : (consumers.sampled_exact && !consumers.feedback);
            const uint64_t slot_base = pass_bases[slot];
            defer_readback_slots[slot] = live_gpu_targets && vo_n > 0 &&
                slot_base && !phase.authoritative_readback &&
                (slot_base != front_va || final_gpu_present) && rtt_defer_ok_slot;
        }
        // PROSPER_READBACK_WHY (#1284): classify WHY each non-deferred pass takes the
        // synchronous CPU readback (75-79 ms/window on Blue Prince's Day One frame).
        // The dominant reason selects the next optimization; behavior unchanged.
        if (!defer_readback) {
            static const bool why = getenv("PROSPER_READBACK_WHY") != nullptr;
            if (why) {
                // Counts identify the failing defer condition; bytes weight each bucket
                // by the actual copy size, since readback time scales with bytes.
                enum { kNoLive, kNoVo, kNoBase, kAuth, kVoFinal, kFront,
                       kSameBatchCpu, kNotSampledExact, kFeedback, kBuckets };
                static std::atomic<uint64_t> counts[kBuckets], bytes[kBuckets];
                static std::atomic<uint64_t> same_batch_storage{0};
                static std::atomic<uint64_t> same_batch_dimension{0};
                static std::atomic<uint64_t> same_batch_extent{0};
                static std::atomic<uint64_t> same_batch_feedback{0};
                static std::atomic<uint64_t> c_total{0};
                int bucket = kNotSampledExact;
                if (!live_gpu_targets) bucket = kNoLive;
                else if (vo_n <= 0) bucket = kNoVo;
                else if (!base) bucket = kNoBase;
                else if (phase.authoritative_readback) bucket = kAuth;
                else if (is_vo) bucket = kVoFinal;
                else if (base == front_va) bucket = kFront;
                else if (cpu_needed_same_batch) bucket = kSameBatchCpu;
                else if (sampled_exact_later && feedback_later) bucket = kFeedback;
                ++counts[bucket];
                if (bucket == kSameBatchCpu) {
                    same_batch_storage += consumers0.storage_references;
                    same_batch_dimension += consumers0.dimension_mismatches;
                    same_batch_extent += consumers0.extent_mismatches;
                    same_batch_feedback += consumers0.feedback_references;
                }
                // #2398: bill bytes only when colour pixels are ACTUALLY requested.
                // #2283 narrowed the readback to `want_color_readback` (the same
                // any_of over pass_bases used at the call below) and this accounting
                // never followed it -- so a depth-only pass, whose slot bases are all
                // zero and which therefore lands in the `no_base` bucket BY
                // CONSTRUCTION, was billed a full frame of colour it never copied.
                //
                // That is not a rounding error: on Blue Prince it reported 15.6 GB
                // against `no_base` and made depth passes look like the frame-rate
                // wall. The count was right; the column that made it actionable was
                // measuring a world from before #2283. An instrument that drifts from
                // the code it measures reads as evidence, which is worse than silence.
                const bool billed_color = std::any_of(
                    pass_bases.begin(),
                    pass_bases.begin() + std::min<size_t>(mrt_count, pass_bases.size()),
                    [](uint64_t slot_base) { return slot_base != 0; });
                if (billed_color)
                    bytes[bucket] += static_cast<uint64_t>(gw) * gh *
                        prosper::test::backend_color_bytes_per_pixel(pass_format);
                static std::atomic<uint64_t> last_report{0};
                const uint64_t t = ++c_total;
                if (t >= last_report.load(std::memory_order_relaxed) + 200) {
                    last_report.store(t, std::memory_order_relaxed);
                    static const char* names[kBuckets] = {
                        "no_live", "no_vo", "no_base", "auth", "vo_final",
                        "front", "same_batch_cpu", "not_sampled_exact", "feedback"};
                    fprintf(stderr, "[readback-why] total=%llu",
                            (unsigned long long)t);
                    for (int i = 0; i < kBuckets; ++i)
                        fprintf(stderr, " %s=%llu/%lluMB", names[i],
                                (unsigned long long)counts[i].load(),
                                (unsigned long long)(bytes[i].load() >> 20));
                    fprintf(stderr,
                            " same_batch_reasons=storage:%llu,dimension:%llu,"
                            "extent:%llu,feedback:%llu\n",
                            (unsigned long long)same_batch_storage.load(),
                            (unsigned long long)same_batch_dimension.load(),
                            (unsigned long long)same_batch_extent.load(),
                            (unsigned long long)same_batch_feedback.load());
                }
            }
        }
        const uint8_t* seed1 = nullptr;
        const float* retained_uniform_clear1 = nullptr;
        bool gpu_seed1_available = false;
        const bool seed_rtt1 = seed_target(base1);
        if (seed_rtt1 && use_color1) { auto sit = g_rtt.find(base1);
            gpu_seed1_available = live_gpu_targets && sit != g_rtt.end() &&
                sit->second.gpu_valid &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 &&
                prosper::test::find_persistent_color_target(
                    base1, gw, gh, pass_format1) != nullptr;
            if (!gpu_seed1_available && sit != g_rtt.end() &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 && sit->second.rgba &&
                sit->second.rgba->size() == static_cast<size_t>(gw) * gh *
                    prosper::test::backend_color_bytes_per_pixel(pass_format1))
                seed1 = sit->second.rgba->data();
            if (!gpu_seed1_available && !seed1 && sit != g_rtt.end() &&
                sit->second.w == gw && sit->second.h == gh &&
                sit->second.format == pass_format1 &&
                sit->second.has_uniform_color)
                retained_uniform_clear1 = sit->second.uniform_color.data(); }
        if (base1 && !use_color1 && rtt_log) {
            const auto* first = render_pass.empty() ? nullptr : render_pass.front();
            fprintf(stderr,
                    "[rtt] skip MRT1 target=0x%llx extent=%ux%u; MRT0 is %ux%u\n",
                    (unsigned long long)base1,
                    first ? first->color1_width : 0u,
                    first ? first->color1_height : 0u, native_w, native_h);
        }
        // #3891: one clock pair per pass group feeds the always-on ledger too.
        const bool perf_build_clock = prosper::diagnostics::perf::enabled();
        const auto build_start = timing_enabled || perf_build_clock
            ? RenderClock::now() : RenderClock::time_point{};
        prosper::test::BackendColorTarget backend_target{
            base, seed_rtt0, base != 0 && !defer_readback, pass_format};
        const auto& volume_view = primary_volume_view;
        if (base && volume_view.selected_mip_depth) {
            if (!volume_view.volume_view_consistent() || !volume_view.slice_count) {
                // This draw attempted to replace the retained version. Refusing
                // its view must also revoke the earlier version, including when
                // no backend call is made.
                prosper::test::invalidate_persistent_color_target(base);
                note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                            volume_view.selected_mip_depth,
                                            pass_format);
                std::fprintf(stderr,
                    "[render-volume] invalid slot0 view target=0x%llx\n",
                    static_cast<unsigned long long>(base));
                continue;
            }
            backend_target.volume_depth = volume_view.selected_mip_depth;
            backend_target.volume_first_slice = volume_view.first_slice;
            backend_target.volume_slice_count = volume_view.slice_count;
            backend_target.volume_guest_bytes = producer_volume_physical_bytes;
            backend_target.readback = false;
        }
        backend_target.persistent_id1 = use_color1 ? base1 : 0;
        backend_target.load_existing1 = seed_rtt1;
        backend_target.readback1 = use_color1 && base1 != 0 && !defer_readback1;
        backend_target.format1 = pass_format1;
        // Slots 2..7 retain across render groups on the same terms as slots 0 and 1.
        // A G-buffer built by several groups against one set of allocations otherwise
        // loses every group's work but the last, because slots above 1 were transient
        // images cleared per backend call.
        for (uint32_t slot = 2; slot < mrt_count; ++slot) {
            backend_target.persistent_id_slots[slot] = pass_bases[slot];
            backend_target.load_existing_slots[slot] = seed_target(pass_bases[slot]);
            backend_target.readback_slots[slot] = pass_bases[slot] != 0 && !defer_readback_slots[slot];
        }
        auto backend_draws = build_bds(
            render_pass, batch_backend_submits ? &backend_submission : nullptr);
        const auto build_done = timing_enabled || perf_build_clock
            ? RenderClock::now() : RenderClock::time_point{};
        if (perf_build_clock) {
            prosper::diagnostics::perf::add_cost(
                prosper::diagnostics::perf::Cost::FrontendBuild,
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    build_done - build_start).count()));
            prosper::diagnostics::perf::flush_thread_texture_references();
        }
        prosper::test::BackendMrtOutputs mrt_outputs;
        mrt_outputs.color_count = mrt_count;
        // The colour target is passed whenever ANY slot is bound, not colour-0 alone.
        // The same union the readback flag below already uses, and for the same reason
        // the comment there gives: a pass with colour-0 unbound and a higher slot bound
        // is reachable by construction. On `base != 0` alone such a pass populated
        // persistent_id_slots[2..7] and then handed the backend a nullptr, so those
        // active slots stayed transient and were cleared by the next group -- exactly
        // the defect the persistence contract exists to remove, reintroduced at the one
        // call site that decides whether the contract is used at all.
        const bool any_slot_bound = prosper::frontend::mrt_any_slot_bound(
            pass_bases.data(),
            static_cast<uint32_t>(std::min<size_t>(mrt_count, pass_bases.size())));
        std::vector<uint8_t> gpx = prosper::test::render_draws_rgba(
            backend_draws, gw, gh, seed,
            retained_uniform_clear ? retained_uniform_clear : clear_for(render_pass), true,
            live_gpu_targets && any_slot_bound ? &backend_target : nullptr,
            seed1, retained_uniform_clear1 ? retained_uniform_clear1
                : (use_color1 ? render_pass.front()->ps.clear_color1 : nullptr),
            nullptr,
            batch_backend_submits ? &backend_submission : nullptr,
            pass_i == items.size(), &mrt_outputs,
            // #2283: only ask for colour pixels when something will read them.
            //
            // Keyed on the UNION of every bound slot, not on colour-0 alone. Review
            // caught that: `gpx`'s only consumer is indeed inside an `if (base ...)`
            // pair with no else, but `gpx` is not all the call returns. `mrt_outputs`
            // is filled by the same call and its consumer is keyed per slot on
            // `pass_bases[slot]`, where an EMPTY vector is not a no-op -- it calls
            // `surface.rgba.reset()` and drops that surface's cached pixels.
            //
            // `mrt_count` does not depend on `base`, so a pass with colour-0 unbound
            // and colour-1 bound is reachable by construction -- the same sparse export
            // hole the MRT loop already acknowledges, at slot 0 instead of slot 1. On
            // `base != 0` alone such a pass would silently lose a live colour-1 RTT.
            // Structural rather than observed, which is exactly why the 457/457
            // depth-only measurement could not clear it: that evidence is all slot 0.
            //
            // Depth-only passes have every slot zero, so the win is unchanged.
            // Keyed on the base rather than the draws' colour write masks for the
            // separate reason that 457/457 is evidence about THIS route, and a future
            // title could legally mix a colour-writing draw into a pass that has a base.
            /*want_color_readback=*/any_slot_bound);
        const auto backend_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        const prosper::test::BackendColorTargetStats color_target_call =
            prosper::test::backend_color_target_stats();
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
            // Everything this group did BEFORE the measured pair -- target grouping,
            // format/extent resolution, resolve handling, seed selection. Recorded here
            // rather than at the loop head so a `continue` between the two points cannot
            // skip it and silently under-report (#2250's lesson, applied one level up).
            pending_timing.pass_pre_ms +=
                std::chrono::duration<double, std::milli>(build_start - group_start).count();
            ++pending_timing.pass_groups;
            record_backend_timing(backend_call_timing, backend_texture_stats,
                                  backend_pipeline_stats, backend_reuse_stats);
            pending_timing.color_target_writes += color_target_call.writes;
            pending_timing.color_target_write_hits += color_target_call.write_hits;
            pending_timing.color_target_sample_hits += color_target_call.sampled_hits;
            pending_timing.color_target_readbacks += color_target_call.readbacks;
            pending_timing.color_target_cached_bytes = color_target_call.cached_bytes;
            pending_timing.color_target_cached_entries = color_target_call.cached_entries;
        }
        const auto post_stats_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        static const std::string dump_spec =
            PROSPER_ENV_VALUE("PROSPER_DUMP_PASS")
                ? PROSPER_ENV_VALUE("PROSPER_DUMP_PASS") : "";
        static const int dump_pass_every =
            PROSPER_ENV_VALUE("PROSPER_DUMP_PASS_EVERY")
                ? std::max(1, atoi(getenv("PROSPER_DUMP_PASS_EVERY"))) : 60;
        auto pass_pixels = std::make_shared<const std::vector<uint8_t>>(std::move(gpx));
        if (base && backend_target.volume_depth && !color_target_call.writes) {
            // Backend admission failed before it could record a write (e.g. a device
            // without 2D-on-3D views), so the renderer has no image to serve.
            note_volume_producer_denied(base, g_rtt[base], gw, gh,
                                        backend_target.volume_depth, pass_format);
        }
        const auto* completed_target = base
            ? prosper::test::find_persistent_color_target(
                  base, gw, gh, pass_format, true,
                  backend_target.volume_depth)
            : nullptr;
        const uint64_t pass_source_submit = completed_target
            ? prosper::frontend::completed_source_submit(
                  prosper::test::persistent_color_producer_source(*completed_target))
            : 0;
        if (base && color_target_call.writes) {
            RttSurf& surface = g_rtt[base];
            surface.w = gw;
            surface.h = gh;
            surface.samples = pass.empty()
                ? 1u : 1u << pass.front()->ps.color_targets[0].log2_samples;
            surface.volume_depth = backend_target.volume_depth;
            surface.format = pass_format;
            surface.guest_format = pass.empty()
                ? pass_format
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), 0));
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
            surface.gpu_valid = prosper::test::find_persistent_color_target(
                base, gw, gh, pass_format, true,
                backend_target.volume_depth) != nullptr;
            // Claim renderer authority over the volume's guest footprint ONLY when the
            // renderer now holds a valid image to serve it from. The backend marks a
            // retained volume valid only once every slice is proven written, so a
            // volume whose slices are never proven keeps guest memory authoritative, as
            // before #3842; a proven-complete volume keeps its protection while its
            // image survives. GTA V's 32x32x32 colour-grading LUT is the worked case:
            // its only raster pass writes slice 0 of 32, and a compute program writes
            // the LUT back to guest memory. Claiming it made that compute skip and every draw that
            // samples the LUT drop, blacking out the menus and HUD (#3842, #3889).
            settle_volume_guest_footprint(base, surface, surface.gpu_valid,
                                          producer_volume_guard_bytes,
                                          producer_volume_footprint_proven);
            if (!pass_pixels->empty()) surface.rgba = pass_pixels;
            else surface.rgba.reset();
            // GTA V builds its packed-HDR bloom pyramid as separate CB_COLOR targets,
            // then samples them as one mipmapped T#. A target may be hundreds of LRU
            // insertions old by that final draw. Pin every produced level now, while its
            // exact identity is known, and release the small set at the submit boundary.
            pin_renderer_mip_target(base, gw, gh, pass_format, surface.gpu_valid);
            if (defer_readback && is_vo && surface.gpu_valid) {
                const auto already_pinned = std::find_if(
                    pinned_scanouts.begin(), pinned_scanouts.end(),
                    [&](const PinnedScanout& target) {
                        return target.id == base && target.width == gw &&
                               target.height == gh && target.format == pass_format;
                    });
                if (already_pinned == pinned_scanouts.end()) {
                    if (prosper::test::pin_persistent_color_target(
                            base, gw, gh, pass_format)) {
                        pinned_scanouts.push_back({base, gw, gh, pass_format});
                    } else {
                        // Pinning is expected to succeed for the target just written. If
                        // cache state is inconsistent, preserve correctness by immediately
                        // restoring the authoritative CPU fallback.
                        std::vector<uint8_t> materialized;
                        std::string error;
                        if (prosper::test::readback_persistent_color_target(
                                base, gw, gh, pass_format, materialized, error))
                            surface.rgba =
                                std::make_shared<const std::vector<uint8_t>>(
                                    std::move(materialized));
                    }
                }
            }
        } else if (base && !pass_pixels->empty()) {
            RttSurf& surface = g_rtt[base];
            surface.rgba = pass_pixels;
            surface.w = gw;
            surface.h = gh;
            surface.volume_depth = 0;
            surface.format = pass_format;
            surface.guest_format = pass.empty()
                ? pass_format
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), 0));
            surface.gpu_valid = false;
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
        }
        const auto post_slot0_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        for (uint32_t slot = 1; slot < mrt_count; ++slot) {
            auto& pixels = mrt_outputs.colors[slot];
            if (!pass_bases[slot]) continue;  // transient attachment for a sparse export hole
            if (rtt_log && !pixels.empty()) {
                size_t nz = 0, rgb_nz = 0;
                for (uint8_t byte : pixels) nz += byte != 0;
                if (pass_formats[slot] == VK_FORMAT_R8G8B8A8_UNORM)
                    for (size_t p = 0; p + 3 < pixels.size(); p += 4)
                        rgb_nz += pixels[p] != 0 || pixels[p + 1] != 0 ||
                                  pixels[p + 2] != 0;
                fprintf(stderr,
                        "[rtt] pass c%u=0x%llx extent=%ux%u (%zu draws) "
                        "px_nonzero=%zu rgb_nonblack=%zu\n",
                        slot, (unsigned long long)pass_bases[slot],
                        gw, gh, pass.size(), nz, rgb_nz);
            }
            // PROSPER_DUMP_PASS covers slots 1..7 too. The first version dumped only
            // `base`, so the busiest target in GTA V's frame -- 0x2085de0000, written by
            // 130,290 of 131,072 draws, 96% of them in SLOT 4 -- could not be dumped at
            // all, and its absence read as "that pass does not run".
            if (!dump_spec.empty() && pass_bases[slot]) {
                char needle[24];
                std::snprintf(needle, sizeof needle, "0x%llx",
                              (unsigned long long)pass_bases[slot]);
                if (dump_spec.find(needle) != std::string::npos) {
                    static std::map<uint64_t, size_t> slot_busiest;
                    static std::map<uint64_t, int> slot_counted;
                    const bool slot_peak = pass.size() > slot_busiest[pass_bases[slot]];
                    if (slot_peak) slot_busiest[pass_bases[slot]] = pass.size();
                    if (slot_peak &&
                        slot_counted[pass_bases[slot]]++ % dump_pass_every == 0) {
                        const std::vector<uint8_t> shot = inspection_rgba8(
                            pixels, gw, gh, pass_formats[slot]);
                        const char* dir = getenv("PROSPER_FRAME_DIR");
                        char path[512];
                        std::snprintf(path, sizeof path, "%s/pass_%llx_s%u_%ux%u.bmp",
                                      dir ? dir : ".",
                                      (unsigned long long)pass_bases[slot], slot,
                                      gw, gh);
                        const size_t want = size_t(gw) * gh * 4;
                        const bool ok = shot.size() == want &&
                            prosper::test::dump_bmp(path, shot, gw, gh);
                        fprintf(stderr,
                                "[dump-pass] target=0x%llx slot=%u %ux%u src=%zuB -> %s\n",
                                (unsigned long long)pass_bases[slot], slot, gw, gh,
                                pixels.size(),
                                ok ? path : "NO CPU PIXELS (readback deferred)");
                    }
                }
            }
            RttSurf& surface = g_rtt[pass_bases[slot]];
            surface.w = gw;
            surface.h = gh;
            surface.samples = pass.empty()
                ? 1u : 1u << pass.front()->ps.color_targets[slot].log2_samples;
            surface.volume_depth = 0;
            surface.format = pass_formats[slot];
            surface.guest_format = pass.empty()
                ? pass_formats[slot]
                : static_cast<VkFormat>(
                      prosper::frontend::mrt_raw_format(*pass.front(), slot));
            surface.has_uniform_color = false;
            surface.dcc_metadata_dirty = false;
            // Any slot with a retained target is GPU-valid, not only slot 1. The
            // `slot == 1` clause dated from when slots above 1 had no persistent image
            // to be valid about.
            surface.gpu_valid =
                prosper::test::find_persistent_color_target(
                    pass_bases[slot], gw, gh, pass_formats[slot]) != nullptr;
            pin_renderer_mip_target(pass_bases[slot], gw, gh, pass_formats[slot],
                                    surface.gpu_valid);
            if (!pixels.empty())
                surface.rgba = std::make_shared<const std::vector<uint8_t>>(
                    std::move(pixels));
            else
                surface.rgba.reset();
        }
        // A CPU-only RTT import can come from a producer that never retained an image,
        // an invalidated image, or an evicted one. Trace only named addresses without
        // enabling PROSPER_RTTLOG, which itself turns off the GPU-resident path.
        const auto& residency_trace = rtt_residency_trace_selector();
        if (residency_trace.configured && residency_trace.valid) {
            for (uint32_t slot = 0; slot < mrt_count; ++slot) {
                const uint64_t address = pass_bases[slot];
                if (!address || !residency_trace.includes(address)) continue;
                const auto found = g_rtt.find(address);
                if (found == g_rtt.end()) continue;
                const RttSurf& surface = found->second;
                const auto* retained = prosper::test::find_persistent_color_target(
                    address, surface.w, surface.h, pass_formats[slot], false,
                    slot == 0 ? surface.volume_depth : 0);
                std::fprintf(stderr,
                    "[rtt-residency] submit=%llu pass=%zu slot=%u addr=0x%llx "
                    "extent=%ux%u format=%u live=%u writes=%llu readbacks=%llu "
                    "gpu-valid=%u cpu-bytes=%zu cached=%u cache-valid=%u "
                    "cache-image=%u\n",
                    (unsigned long long)g_pass_log_submit.load(
                        std::memory_order_relaxed), pass_i, slot,
                    (unsigned long long)address, surface.w, surface.h,
                    static_cast<unsigned>(pass_formats[slot]),
                    live_gpu_targets ? 1u : 0u,
                    (unsigned long long)color_target_call.writes,
                    (unsigned long long)color_target_call.readbacks,
                    surface.gpu_valid ? 1u : 0u,
                    surface.rgba ? surface.rgba->size() : 0u,
                    retained ? 1u : 0u,
                    retained && retained->valid ? 1u : 0u,
                    retained && retained->image ? 1u : 0u);
            }
        }
        const auto post_mrt_done = timing_enabled
            ? RenderClock::now() : RenderClock::time_point{};
        const std::vector<uint8_t>& rendered_pixels = *pass_pixels;
        // Both dims parse to 0 when PROSPER_RESOURCE_HASH_DIM is UNSET, so an extent
        // comparison alone made every 0x0 pass satisfy the filter and switch the
        // diagnostic on by itself. Require it to have actually been requested.
        if (resource_hash_w && resource_hash_h &&
            native_w == resource_hash_w && native_h == resource_hash_h &&
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
        // Same unset-is-0 trap as the resource hash above, and far more expensive here:
        // the loop below re-renders every growing draw prefix, so a 0x0 pass silently
        // turned one pass into O(draws^2) rendering plus a readback and pixel census per
        // step. Require PROSPER_TARGET_STEP_HASH_DIM to have actually been requested.
        if (target_step_w && target_step_h &&
            native_w == target_step_w && native_h == target_step_h &&
            render_pass.size() >= target_step_min_draws) {
            for (size_t k = 1; k <= render_pass.size(); ++k) {
                std::vector<const prosper::gpu::DrawItem*> prefix(
                    render_pass.begin(), render_pass.begin() + k);
                // PROSPER_TARGET_STEP_HASH_DIM re-renders growing draw prefixes through
                // build_bds(), so every prefix must resolve its own resources from
                // scratch. Release the scratch pins with the map they belonged to,
                // otherwise each prefix would strand another set of slots.
                texstore_used = 0;
                texstore_pinned.assign(texstore.size(), false);
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
                        (unsigned long long)spirv_hash(draw->vs_words()),
                        (unsigned long long)spirv_hash(draw->fs_words()),
                        draw->ps.color_write_mask, (int)draw->ps.blend_enable,
                        draw->ps.src_color_blend_factor,
                        draw->ps.dst_color_blend_factor,
                        (unsigned long long)hash, dark, near_white, mean_rgb);
            }
        }
        const RttTimingRecord rtt_timing_record{
            g_this_submit, base, gw, gh, pass.size(),
            phase.first_span, phase.final_span, phase.authoritative_readback,
            defer_readback,
            std::chrono::duration<double, std::milli>(
                backend_done - build_done).count(),
            backend_call_timing, color_target_call};
        if (lightweight_rtt_timing) pending_rtt_timing.push_back(rtt_timing_record);
        else if (timing_enabled && rtt_log) print_rtt_timing(rtt_timing_record);
        // PROSPER_DUMP_PASS=0xADDR[,0xADDR…] — write what a pass actually produced, as an
        // image, for the named targets. `rgb_nonblack` alone cannot answer "is the world
        // there": it is computed from an RGBA8 conversion, so an HDR f16 target whose
        // values are all below 1/255 reports EXACTLY ZERO while carrying a complete
        // scene. That is not hypothetical here — GTA V's 0x20431c0000 reports
        // rgb_nonblack=0 while the very next pass extracts 47.5% non-black from it, so
        // reading the metric as "black" is a false negative on the whole lighting stage.
        //
        // Overwrites one file per target so the last write is the latest frame, and
        // dumps every PROSPER_DUMP_PASS_EVERY-th occurrence (default 60) because a 4K
        // BMP is 24 MB and a routed boot renders each of these a few hundred times.
        {
            if (!dump_spec.empty() && base) {
                char needle[24];
                std::snprintf(needle, sizeof needle, "0x%llx",
                              (unsigned long long)base);
                if (dump_spec.find(needle) != std::string::npos) {
                    // Keep the occurrence with the MOST DRAWS, not the last. A
                    // G-buffer target is written by one heavy pass and then touched by
                    // several small ones, so "last write wins" reliably captures a
                    // near-empty tail and reads as "this channel is empty" -- which it
                    // did, for three channels that a draw census showed at 54%.
                    static std::map<uint64_t, size_t> busiest;
                    static std::map<uint64_t, int> counted;
                    const bool new_peak = pass.size() > busiest[base];
                    if (new_peak) busiest[base] = pass.size();
                    if (new_peak && counted[base]++ % dump_pass_every == 0) {
                        const std::vector<uint8_t> shot = inspection_rgba8(
                            rendered_pixels, gw, gh, pass_format);
                        const char* dir = getenv("PROSPER_FRAME_DIR");
                        char path[512];
                        std::snprintf(path, sizeof path, "%s/pass_%llx_%ux%u.bmp",
                                      dir ? dir : ".", (unsigned long long)base,
                                      gw, gh);
                        // Report the source size, not just the intent. A pass whose
                        // readback was deferred has EMPTY cpu pixels, and both this dump
                        // and `rgb_nonblack` are then reporting on nothing — which reads
                        // as "the target is black" rather than "we never looked".
                        const size_t want = size_t(gw) * gh * 4;
                        const bool ok = shot.size() == want &&
                            prosper::test::dump_bmp(path, shot, gw, gh);
                        fprintf(stderr,
                                "[dump-pass] target=0x%llx %ux%u src=%zuB rgba=%zuB/%zuB"
                                " -> %s\n",
                                (unsigned long long)base, gw, gh,
                                rendered_pixels.size(), shot.size(), want,
                                ok ? path : "NO CPU PIXELS (readback deferred)");
                    }
                }
            }
        }
        if (rtt_log) {
            const std::vector<uint8_t> inspected = inspection_rgba8(
                rendered_pixels, gw, gh, pass_format);
            size_t nz = 0, rgb_nz = 0;
            for (uint8_t b : rendered_pixels) nz += (b != 0);
            for (size_t p = 0; p + 3 < inspected.size(); p += 4)
                rgb_nz += (inspected[p] != 0 || inspected[p + 1] != 0 ||
                           inspected[p + 2] != 0);
            // `src=` is load-bearing, not decoration. Both counters are computed from
            // `rendered_pixels`, so a pass whose readback was deferred reports
            // `px_nonzero=0 rgb_nonblack=0` -- identical to a genuinely black target, and
            // indistinguishable from it without this field. The `[dump-pass]` line four
            // lines up already guards exactly this ("as 'the target is black' rather than
            // 'we never looked'"); this line did not, and readbacks are deferred routinely
            // -- `[readback-why]` buckets nine distinct reasons.
            //
            // It matters most for the conclusion it silently supports. Every "this pass
            // reads populated inputs and emits nothing" reading on GTA V rests on these
            // two counters, and "we never looked" produces that reading for free.
            fprintf(stderr, "[rtt] pass target=0x%llx extent=%ux%u native=%ux%u (%zu draws) "
                    "src=%zuB px_nonzero=%zu rgb_nonblack=%zu cache_size=%zu%s%s\n",
                    (unsigned long long)base, gw, gh, native_w, native_h, pass.size(),
                    rendered_pixels.size(), nz, rgb_nz, g_rtt.size(),
                    is_vo ? " SCANOUT" : "", base && base == front_va ? " FRONT" : "");
        }
        // PROSPER_DUMP_DRAWSTEPS: for a pass targeting a SCANOUT buffer, re-render the pass
        // draw-by-draw (prefix 1, prefix 2, ...) and dump each cumulative result — a one-boot
        // bisect for "which draw of the final composite blacks the screen" (#319). Diagnostic.
        if (PROSPER_ENV_ON("PROSPER_DUMP_DRAWSTEPS") && is_vo && pass.size() > 1) {
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
        if ((PROSPER_ENV_ON("PROSPER_DUMP_RTGROUPS") || (getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) &&
            !rendered_pixels.empty()) {
            size_t nz = 0; for (uint8_t b : rendered_pixels) nz += (b != 0);
            const char* address_filter = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS_ADDR");
            const uint64_t wanted_base = address_filter && *address_filter
                ? strtoull(address_filter, nullptr, 0) : 0;
            // Match the address against EVERY MRT slot, not just slot 0.
            //
            // `base` is the pass's slot-0 attachment. A G-buffer names different
            // surfaces per slot, so filtering by a slot-2 address matched ONE pass in a
            // full routed run for a target the draw census credits with 6,672 draws --
            // and a near-empty result reads as "that target is empty" rather than as
            // "the filter asked the wrong question". The dumped pixels are still slot
            // 0's; what this fixes is WHICH PASSES are selected, so naming any slot of a
            // G-buffer selects that G-buffer's passes.
            bool slot_match = !wanted_base || base == wanted_base;
            uint32_t matched_slot = 0;
            if (wanted_base && !slot_match)
                for (uint32_t slot = 1; slot < pass_bases.size(); ++slot)
                    if (pass_bases[slot] == wanted_base) {
                        slot_match = true; matched_slot = slot; break;
                    }
            const char* dd = getenv("PROSPER_FRAME_DIR");
            // Identify the pass by its first..last draw index so multiple passes to the same
            // target VA in one frame do not silently overwrite each other.
            const uint64_t pass_d0 = render_pass.empty() ? 0u : render_pass.front()->draw_index;
            const uint64_t pass_d1 = render_pass.empty() ? 0u : render_pass.back()->draw_index;
            if (slot_match) {
                if (wanted_base && matched_slot) {
                    static std::set<std::pair<uint64_t, uint32_t>> slot_seen;
                    if (slot_seen.emplace(wanted_base, matched_slot).second)
                        fprintf(stderr,
                                "[rtt] rtgroup filter 0x%llx matched slot %u of a pass "
                                "whose slot-0 base is 0x%llx; dumped pixels are SLOT 0\n",
                                (unsigned long long)wanted_base, matched_slot,
                                (unsigned long long)base);
                }
                if (const char* rg = PROSPER_ENV_VALUE("PROSPER_DUMP_RTGROUPS"); rg && nz >= (size_t)atol(rg)) {
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
                if ((getenv("PROSPER_DUMP_RTGROUPS_RGBA") != nullptr)) {
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
        // The POPULATION that the publish zero is a zero OVER, counted BEFORE the
        // emptiness/format gates below -- so it sees every pass whose colour target
        // is disabled, not just those that already qualified as candidates.
        //
        // This placement is the whole point. A same-source positive control proves
        // the DISCRIMINATOR fires; it can never prove the DOMAIN contains the case.
        // Counting the population in the same run is what separates "disabled-target
        // passes exist and none is ever published" from "none exists, so the zero
        // says nothing" -- and those are opposite conclusions about whether #2283's
        // readback skip is safe.
        if (timing_enabled && !pass.empty() && pass.front()->ps.color0_format == 0)
            ++pending_timing.publish_candidate_fmt0;
        if (!explicit_depth_extent && !rendered_pixels.empty() &&
            pass_format == VK_FORMAT_R8G8B8A8_UNORM) {
            // Recorded alongside each candidate rather than re-derived at selection
            // time: by then the pass loop has moved on and `pass` no longer refers to
            // the pass that produced these pixels (#2283).
            const uint32_t candidate_fmt =
                pass.empty() ? kNoPassFormat : pass.front()->ps.color0_format;
            if (base && base == front_va) {                          // the flipped buffer
                px_front = pass_pixels; px_front_w = gw; px_front_h = gh;
                px_front_base = base; px_front_fmt = candidate_fmt;
                px_front_source_submit = pass_source_submit;
            }
            if (is_vo) {                                            // any registered scanout
                px_vo = pass_pixels; px_vo_w = gw; px_vo_h = gh;
                px_vo_base = base; px_vo_fmt = candidate_fmt;
                px_vo_source_submit = pass_source_submit;
            }
            px_last = pass_pixels;                                  // last non-empty (fallback)
            px_last_w = gw; px_last_h = gh; px_last_base = base; px_last_fmt = candidate_fmt;
            px_last_source_submit = pass_source_submit;
        } else if (!explicit_depth_extent && !rendered_pixels.empty() &&
                   prefix_inspect_publish()) {
            // #1330: under gpu_replay's ordered-prefix inspection (--draw/--draw-steps/
            // --through-operation set PROSPER_PREFIX_INSPECT), a prefix ending on a
            // non-RGBA8 pass (an FP16 HDR scene target) must return THAT surface, not
            // whatever stale RGBA8 pass ran earlier. Publish an inspection-converted copy
            // as the weakest fallback: any RGBA8 pass afterwards still overwrites it, and
            // with the env unset (every live/normal replay run) behavior is byte-identical.
            std::vector<uint8_t> converted =
                inspection_rgba8(rendered_pixels, gw, gh, pass_format);
            if (!converted.empty())
                px_last = std::make_shared<const std::vector<uint8_t>>(std::move(converted));
            px_last_source_submit = 0; // inspection conversion is not the native pixels
        }
        // PROSPER_PASS_LOG=<min-submit>|ms:<millis>: per-pass publish provenance for 3
        // submits — which pass produced pixels, its target identity, and the defer
        // decision. The `ms:` form aims the window by wall time (diagnostic_window.hpp).
        // PROSPER_PASS_LOG_NODEFER opens this block too. It was added INSIDE it and
        // armed by its own switch, so setting only PROSPER_PASS_LOG_NODEFER produced
        // ZERO lines -- a silent run indistinguishable from "there are no such passes".
        // That is #2149's class, committed by the author who had been citing #2149 all
        // session: a diagnostic whose producer and printer are armed by different
        // switches reports unarmed state as a confident zero.
        if (PROSPER_ENV_ON("PROSPER_PASS_LOG") ||
            PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER")) {
            const uint64_t at = g_pass_log_submit.load(std::memory_order_relaxed);
            const bool in_window =
                g_pass_log_window.contains(at, diagnostic_elapsed_ms());
            // px_nonblack must be counted in the PASS's OWN format. The original loop
            // stepped 4 bytes and tested bytes `p`, `p+1`, `p+2` — never `p+3` — as if
            // every target were 8-bit RGBA. Over an 8-byte FP16 texel that reads
            // {R_lo, R_hi, G_lo} and then {B_lo, B_hi, A_lo}, so whether a texel counts
            // depends only on where a half-float happens to put its non-zero bytes.
            // Measured on Sonic Origins' 3840x2160 R16G16B16A16_SFLOAT scene target, whose
            // texels are RGB bit-zero with alpha 0x3c05 (1.00488): byte 6 is that alpha's
            // low mantissa byte 0x05, which lands on the second group's `p+2`, so exactly
            // one of the two groups per texel counted and the line reported 8,294,400 —
            // precisely w*h, reading as "every pixel has content" for a frame with no
            // colour in it at all. An alpha of exactly 1.0 would have counted ZERO:
            // 0x3c00's only non-zero byte is the `p+3` this loop skipped. The number was
            // never a property of the image, only of its bit layout — and it survived long
            // enough to become a hypothesis (#1905). Count through the same inspection
            // conversion the persistent dump uses; -1 says the format has no conversion,
            // which is honest where a wrong number is not.
            long long nz = 0;
            const bool direct_rgba8 = pass_format == VK_FORMAT_R8G8B8A8_UNORM &&
                rendered_pixels.size() == static_cast<size_t>(gw) * gh * 4u;
            const std::vector<uint8_t> inspected = direct_rgba8
                ? std::vector<uint8_t>{}
                : inspection_rgba8(rendered_pixels, gw, gh, pass_format);
            const std::vector<uint8_t>& counted =
                direct_rgba8 ? rendered_pixels : inspected;
            if (!rendered_pixels.empty() && counted.empty()) nz = -1;
            else
                for (size_t p = 0; p + 3 < counted.size(); p += 4)
                    if (counted[p] || counted[p + 1] || counted[p + 2]) nz++;
            // In-window: every pass. Out of window: only content-bearing (or deferred)
            // passes, so the publish source can be found without knowing the callback —
            // plus `nz < 0`, a pass whose format the inspection path cannot convert. That
            // last case has to stay visible out of window: it is the one where this line
            // cannot say whether the pass carries content, which is exactly when the
            // reader needs to know the pass existed.
            // PROSPER_PASS_LOG_NODEFER: print the NON-DEFERRED passes specifically.
            // They are the population #2276 needs and the only one this line cannot
            // currently reach: a deferred pass always prints, a content-bearing pass
            // prints on `nz > 100`, and the window covers three callbacks -- but a
            // non-deferred pass with little content satisfies none of those, so 2,999
            // of them a route were invisible while a classifier counted 21 GB of
            // readback against them.
            //
            // Opt-in and separate from PROSPER_PASS_LOG rather than widening the window,
            // because the window's three-callback span is deliberate (its comment says
            // why) and this needs the whole run, not a wider sample. ~9.7 lines a submit.
            static const bool pass_log_nodefer =
                PROSPER_ENV_ON("PROSPER_PASS_LOG_NODEFER");
            if (in_window || nz > 100 || nz < 0 || defer_readback ||
                (pass_log_nodefer && !defer_readback))
                fprintf(stderr,
                        // `bytes` disambiguates px_nonblack=0, which today means EITHER
                        // "the readback returned a near-black surface" OR "the readback
                        // returned nothing and the counting loop never ran". Those are
                        // opposite facts wearing the same number -- #2255's absent-vs-zero
                        // collapse, and instrument trap 116's -- and on #2276 it is
                        // precisely the ambiguity blocking the diagnosis: 2,999 passes a
                        // route take a ~7.1 MB readback and report px_nonblack under 100,
                        // and nobody can say from this line whether those surfaces are
                        // black or absent.
                        "[pass] cb=%llu pass=%zu/%zu base=0x%llx %ux%u fmt=%d vo=%d "
                        // raw_fmt is CB_COLOR0_INFO.FORMAT BEFORE backend_color_format
                        // touches it, and it is the field that decides #2283. That
                        // converter falls back to R8G8B8A8_UNORM for anything it does
                        // not recognise -- INCLUDING 0 -- and R8G8B8A8_UNORM is itself
                        // enumerator 37, so the most common real format and "unknown"
                        // print identically. `fmt` above therefore cannot distinguish a
                        // live colour target from a disabled one, and I nearly published
                        // a correctness claim on it.
                        //
                        // raw_fmt=0 is CB_COLOR_INVALID: the target is DISABLED, the
                        // extent in CB_COLOR0_ATTRIB2 is stale from an earlier pass
                        // (context registers are sticky), and a base of 0 is correct --
                        // the readback is then simply waste. raw_fmt non-zero with
                        // base=0 is the opposite: the guest declared a live colour
                        // target and prosper does not have its address.
                        "seed=%d defer=%d writes=%llu px_nonblack=%lld bytes=%zu "
                        "raw_fmt=%u\n",
                        (unsigned long long)at, pass_i, items.size(),
                        (unsigned long long)base, gw, gh, (int)pass_format,
                        (int)is_vo, (int)seed_rtt0, (int)defer_readback,
                        (unsigned long long)color_target_call.writes, nz,
                        rendered_pixels.size(),
                        pass.empty() ? 0u : pass.front()->ps.color0_format);
        }
        // Everything this group did AFTER the backend call returned -- RTT store,
        // scanout selection, per-pass diagnostics. `backend_done` is in scope only on
        // the paths that reached the render, which are exactly the paths `pre` counted,
        // so the two spans cover the same population and their sum is comparable to
        // pass_groups.
        if (timing_enabled) {
            const auto post_done = RenderClock::now();
            pending_timing.pass_post_ms += std::chrono::duration<double, std::milli>(
                post_done - backend_done).count();
            pending_timing.post_stats_ms += std::chrono::duration<double, std::milli>(
                post_stats_done - backend_done).count();
            pending_timing.post_slot0_ms += std::chrono::duration<double, std::milli>(
                post_slot0_done - post_stats_done).count();
            pending_timing.post_mrt_ms += std::chrono::duration<double, std::milli>(
                post_mrt_done - post_slot0_done).count();
            pending_timing.post_rest_ms += std::chrono::duration<double, std::milli>(
                post_done - post_mrt_done).count();
        }
    }
    // Everything after the group loop: present selection, scanout resolution, the final
    // render, RTT publication. The tail performs its own build_resources+backend, and
    // pass_control already excludes those -- so subtract what they grew by across this
    // span, or the leaves would exceed their parent and the guard would fire on a
    // correct run.
    if (timing_enabled) {
        pending_timing.pass_loop_ms += std::chrono::duration<double, std::milli>(
            RenderClock::now() - pass_loop_start).count() -
            (pending_timing.build_resources_ms + pending_timing.backend_ms -
             pass_loop_measured_before);
        pass_tail_start = RenderClock::now();
        pass_tail_measured_before =
            pending_timing.build_resources_ms + pending_timing.backend_ms;
    }
    // Present priority: the flipped front buffer > any registered scanout target > the
    // legacy "last group" fallback (unchanged behavior when no group targets a VO buffer).
    // Under the publish extent contract (#1986) a candidate whose byte count disagrees is
    // not a present source, so a correctly sized lower-priority candidate is preferred over
    // an incorrectly sized higher-priority one, and "nothing qualified" yields no source at
    // all rather than a frame the caller must discard. Outside a publish-gate submit
    // (offline replay, render_submit_items tests) this is byte-for-byte the historical
    // identity priority.
    auto candidate_bytes = [](const std::shared_ptr<const std::vector<uint8_t>>& p) {
        return p ? p->size() : size_t{0};
    };
    present_choice = prosper::frontend::select_present_source(
        candidate_bytes(px_front), candidate_bytes(px_vo), candidate_bytes(px_last),
        present_extent_bytes);
    using prosper::frontend::PresentSourceChoice;
    selected_pixels = present_choice == PresentSourceChoice::Front ? px_front
                    : present_choice == PresentSourceChoice::Vo    ? px_vo
                    : present_choice == PresentSourceChoice::Last  ? px_last
                    : nullptr;
    selected_source_submit = prosper::frontend::selected_source_submit(
        selected_pixels, {{{px_front, px_front_source_submit},
                           {px_vo, px_vo_source_submit},
                           {px_last, px_last_source_submit}}});
    // #2283's blocking arm, measured rather than argued: is a pass whose colour target is
    // DISABLED (CB_COLOR0_INFO.FORMAT == 0, CB_COLOR_INVALID) ever chosen as the frame
    // that gets published? The proposed fix stops forcing a readback on such a pass, and
    // it is safe by CONSTRUCTION only if the answer is never. Nothing checked it before,
    // and the guard that looks like it would -- pass_format == R8G8B8A8_UNORM at the
    // candidate site -- does NOT exclude a disabled target, because backend_color_format()
    // FALLS BACK to R8G8B8A8_UNORM for anything it does not recognise, including 0.
    if (timing_enabled) {
        const uint32_t chosen_fmt =
            present_choice == PresentSourceChoice::Front ? px_front_fmt
          : present_choice == PresentSourceChoice::Vo    ? px_vo_fmt
          : present_choice == PresentSourceChoice::Last  ? px_last_fmt
          : kNoPassFormat;
        if (selected_pixels) {
            ++pending_timing.publish_selected;
            if (chosen_fmt == 0) ++pending_timing.publish_selected_fmt0;
            else if (chosen_fmt == kNoPassFormat) ++pending_timing.publish_selected_unknown;
        }
    }
    // The one new semantic path: a non-empty higher-priority candidate was passed over on
    // extent. Unreachable today (a VO/front-buffer pass has its extent pinned to the present
    // extent at :5060-5063 before it renders, so those candidates always fit), which is
    // exactly why it gets a report rather than a comment — this PR's whole thesis is that a
    // silent substitution is what cost 31 consecutive submits, and that applies to its own
    // substitution too. If the pin is ever relaxed, this says so instead of quietly changing
    // which surface reaches the screen.
    if (prosper::frontend::present_source_demoted(
            present_choice, candidate_bytes(px_front), candidate_bytes(px_vo))) {
        static std::atomic<uint64_t> demotions{0};
        const uint64_t ord = demotions.fetch_add(1) + 1;
        if (prosper::diag_should_print(ord))
            fprintf(stderr,
                    "[rtt] PRESENT SOURCE DEMOTED #%llu: chose %s for a requested %ux%u "
                    "(%zu-byte) frame because a higher-priority candidate did not fit — "
                    "px_front=%ux%u(%zu bytes) px_vo=%ux%u(%zu bytes); this should be "
                    "unreachable while VO passes are pinned to the present extent "
                    "(live_renderer.cpp:5060) — check that pin first\n",
                    (unsigned long long)ord,
                    prosper::frontend::present_source_name(present_choice), w, h,
                    present_extent_bytes, px_front_w, px_front_h,
                    candidate_bytes(px_front), px_vo_w, px_vo_h, candidate_bytes(px_vo));
    }
    {
        const uint64_t at = g_pass_log_submit.fetch_add(1);
        if (PROSPER_ENV_ON("PROSPER_PASS_LOG")) {
            // The same window object and the same ordinal the per-pass report used, so
            // the two lines describe one set of callbacks — with one seam: if a `ms:`
            // deadline is crossed between the pass loop and this site, the latching
            // callback carries `selected=` with no per-pass lines beside it.
            if (g_pass_log_window.contains(at, diagnostic_elapsed_ms()))
                fprintf(stderr, "[pass] cb=%llu selected=%s\n", (unsigned long long)at,
                        prosper::frontend::present_source_name(present_choice));
        }
    }
    // PROSPER_DUMP_PERSISTENT=<min-submit>|ms:<millis>: read back persistent color
    // targets after this submit's passes render. The default is the original broad
    // census; PROSPER_DUMP_PERSISTENT_ADDRS limits it to explicitly named addresses.
    // Unlike
    // PROSPER_RTT*/DUMP_*, this flag is NOT in the live_gpu_targets disable list
    // (:1078-1085), so it observes the NORMAL persistent-render path (all other CPU-pixel
    // diagnostics change that path -> #1103). readback_persistent_color_target restores
    // the image layout, so rendering is unaffected. The `ms:` form aims the window by
    // wall time, because the ordinal below is a renderer-internal counter with no
    // published rate and the states worth censusing are named in seconds
    // (diagnostic_window.hpp).
    if (PROSPER_ENV_ON("PROSPER_DUMP_PERSISTENT")) {
        static std::atomic<uint64_t> dp_submit{0};
        const uint64_t sub = dp_submit.fetch_add(1);
        if (g_persist_window.contains(sub, diagnostic_elapsed_ms())) {
            const char* dd = getenv("PROSPER_FRAME_DIR");
            prosper::frontend::PersistentReadbackBudget selected_readback_budget;
            fprintf(stderr, "[persist] submit=%llu present: front=%d/%d front_va=0x%llx "
                    "selected=%s vo:", (unsigned long long)sub, vo_front, vo_n,
                    (unsigned long long)front_va,
                    prosper::frontend::present_source_name(present_choice));
            for (int i = 0; i < vo_n && i < 8; i++)
                fprintf(stderr, " [%d]=0x%llx", i, (unsigned long long)prosper_vo_buffer_addr(i));
            fprintf(stderr, "\n");
            for (auto& kv : g_rtt) {
                if (!g_persist_filter.allows(kv.first)) continue;
                RttSurf& s = kv.second;
                if (s.volume_depth) continue; // this diagnostic writes 2D BMPs
                if (g_persist_extent.first &&
                    (s.w != g_persist_extent.first ||
                     s.h != g_persist_extent.second)) continue;
                if (!s.gpu_valid || !s.w || !s.h ||
                    static_cast<uint64_t>(s.w) * s.h < 64u * 64u) {
                    if (g_persist_filter.state ==
                            prosper::frontend::PersistentReadbackFilterState::Selected)
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "not GPU-valid or below 64x64\n",
                                (unsigned long long)sub,
                                (unsigned long long)kv.first);
                    continue;
                }
                const VkFormat fmt = prosper::test::backend_color_format(s.format);
                const uint32_t bpp = prosper::test::backend_color_bytes_per_pixel(fmt);
                uint64_t expected = 0;
                if (g_persist_extent.first || g_persist_filter.state ==
                    prosper::frontend::PersistentReadbackFilterState::Selected) {
                    const auto charge = selected_readback_budget.admit(s.w, s.h, bpp,
                                                                       expected);
                    if (charge != prosper::frontend::PersistentReadbackCharge::Admitted) {
                        const char* reason = "budget 256 MiB";
                        if (charge == prosper::frontend::PersistentReadbackCharge::InvalidSize)
                            reason = "invalid size";
                        else if (charge == prosper::frontend::PersistentReadbackCharge::SizeOverflow)
                            reason = "size overflow";
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "skipped: selected readback %s\n",
                                (unsigned long long)sub, (unsigned long long)kv.first,
                                reason);
                        continue;
                    }
                } else
                    expected = static_cast<uint64_t>(s.w) * s.h * bpp;
                std::vector<uint8_t> px; std::string err;
                if (!prosper::test::readback_persistent_color_target(
                        kv.first, s.w, s.h, fmt, px, err) || px.size() != expected) {
                    if (g_persist_filter.state ==
                            prosper::frontend::PersistentReadbackFilterState::Selected)
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "readback failed: %s (got=%zu expected=%llu)\n",
                                (unsigned long long)sub,
                                (unsigned long long)kv.first, err.c_str(), px.size(),
                                (unsigned long long)expected);
                    continue;
                }
                const std::vector<uint8_t> rgba = inspection_rgba8(px, s.w, s.h, fmt);
                size_t rgbnz = 0;
                for (size_t p = 0; p + 3 < rgba.size(); p += 4)
                    if (rgba[p] || rgba[p + 1] || rgba[p + 2]) rgbnz++;
                // "Black" and "empty" are different findings, and `rgb_nonblack` alone
                // cannot tell them apart: an HDR target whose texels are negative,
                // non-finite or below 1/510 converts to black from bits that are NOT
                // zero, which is a different defect signature from a zero-filled buffer
                // (#1905 recorded the wrong one from the converted count alone). Report
                // the PRE-conversion evidence beside it: how many raw bytes are non-zero,
                // and the first non-zero texel's own bytes, so the row a lane writes down
                // is a measurement.
                size_t rawnz = 0;
                size_t first_nz_texel = 0;
                bool have_first = false;
                const uint32_t texel_bytes = px.size() && s.w && s.h
                    ? (uint32_t)(px.size() / ((size_t)s.w * s.h)) : 0;
                for (size_t i = 0; i < px.size(); i++) {
                    if (!px[i]) continue;
                    rawnz++;
                    if (!have_first && texel_bytes) {
                        first_nz_texel = i / texel_bytes;
                        have_first = true;
                    }
                }
                // Two distinct sentinels, not one: "none" is the finding "no byte of this
                // target is non-zero", while an unprintable texel stride is the
                // instrument declining to answer. Sharing one string would let the
                // second read as the first — a silent zero, which is the class of lie
                // this whole diagnostic exists to stop reporting. Unreachable today
                // (every format the census dumps is <= 16 B/texel), so the point is that
                // it stays unambiguous if a wider one ever appears.
                char first_text[96] = "none";
                if (have_first && (!texel_bytes || texel_bytes > 16))
                    std::snprintf(first_text, sizeof first_text,
                                  "unprintable (%u bytes/texel)", texel_bytes);
                if (have_first && texel_bytes && texel_bytes <= 16) {
                    int at = std::snprintf(first_text, sizeof first_text,
                                           "texel %zu =", first_nz_texel);
                    for (uint32_t b = 0; b < texel_bytes && at > 0 &&
                                        at < (int)sizeof first_text - 4; b++)
                        at += std::snprintf(first_text + at, sizeof first_text - at,
                                            " %02x", px[first_nz_texel * texel_bytes + b]);
                }
                char fn[512];
                std::snprintf(fn, sizeof fn, "%s/persist_s%04llu_%llx_%ux%u.bmp",
                              dd ? dd : ".", (unsigned long long)sub,
                              (unsigned long long)kv.first, s.w, s.h);
                prosper::test::dump_bmp(fn, rgba, s.w, s.h);
                fprintf(stderr, "[persist] submit=%llu addr=0x%llx %ux%u fmt=%u "
                        "rgb_nonblack=%zu/%u raw_nonzero_bytes=%zu/%zu first_nonzero=%s\n",
                        (unsigned long long)sub,
                        (unsigned long long)kv.first, s.w, s.h, (unsigned)s.format,
                        rgbnz, s.w * s.h, rawnz, px.size(), first_text);
            }
            if (g_persist_filter.state ==
                    prosper::frontend::PersistentReadbackFilterState::Selected)
                for (uint64_t addr : g_persist_filter.addresses)
                    if (!g_rtt.contains(addr))
                        fprintf(stderr, "[persist] submit=%llu addr=0x%llx "
                                        "not retained by the renderer\n",
                                (unsigned long long)sub,
                                (unsigned long long)addr);
        }
    }
}
} // namespace

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
                if (front >= 0 && front_flip && gpu_present_now && !new_gpu_flip) {
                    // The previously published slot remains the correct scanout for this guest
                    // flip. Treat it as a successful GPU publication so intermediate render
                    // submissions do not fall through to the expensive CPU readback path.
                    published_gpu = true;
                    handoff_trace.emit(prosper::perf::PresentHandoffEvent::SameFlipSuppressed, 0, 0, last_gpu_publish_flip);
                } else if (front >= 0 && front_flip && gpu_present_now) {
                    const uint64_t front_va = front_snapshot.address;
                    auto rit = g_rtt.find(front_va);
                    if (rit != g_rtt.end() && !rit->second.volume_depth &&
                        rit->second.gpu_valid && rit->second.w && rit->second.h) {
                        const VkFormat fmt = prosper::test::backend_color_format(rit->second.format);
                        // The cache key is only a lookup hint. Hold its resource-domain lock from
                        // the exact image/provenance snapshot through the synchronous scanout copy;
                        // eviction or allocation reuse must not substitute another image mid-handoff.
                        std::lock_guard resource_lock(
                            prosper::test::backend_persistent_resource_mutex());
                        prosper::test::PersistentColorTargetImage* tgt =
                            prosper::test::find_persistent_color_target(
                                front_va, rit->second.w, rit->second.h, fmt);
                        if (tgt && tgt->image && tgt->layout != VK_IMAGE_LAYOUT_UNDEFINED) {
                            published_gpu = prosper::frontend::present_blit_publish(
                                tgt->image, tgt->layout, fmt, rit->second.w, rit->second.h,
                                front_flip,
                                prosper::test::persistent_color_producer_source(*tgt),
                                &front_snapshot);
                            if (published_gpu) last_gpu_publish_flip = front_flip;
                        }
                    }
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

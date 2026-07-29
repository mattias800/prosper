// live_renderer.hpp — register prosper's live Vulkan submit renderer.
//
// Extracted from boot_trace so boot_trace and prosper-app share ONE renderer instead of duplicating
// the ~200-line DrawItem→Vulkan compositor. On every guest GPU submit with draws, it composites the
// draws (recompiled VS/PS + detiled textures + resources) into an RGBA frame and hands it to the
// present layer (present_write_frame → present_readback). All the boot-time diagnostics
// (PROSPER_RENDER_*, PROSPER_DUMP_*, PROSPER_TESTTEX, …) are preserved and remain env-gated.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::frontend {

// Bytes actually uploaded for one non-texture (vertex/index/storage/constant) buffer binding whose
// descriptor declares `declared_bytes`. A vertex fetch may index anywhere inside the declared range,
// so anything clamped away reads as zeros in the shader and collapses geometry to a point — the
// #1427 failure, where a 1 MiB clamp erased 44 of 248 Blue Prince hall draws with no diagnostic.
// The result is dword-aligned and bounded by a ceiling far above real content, which
// PROSPER_MAX_BUFFER_UPLOAD_MB (1..64) may lower for a same-build A/B of the #1427 collapse. Any
// short upload — from that ceiling or the override — is reported by the caller rather than
// silently dropped; only the sub-dword alignment remainder is dropped without a report.
uint32_t buffer_upload_bytes(uint32_t declared_bytes);

// Resolve the bounded decoded-texture cache budget. An explicit MiB string preserves the
// PROSPER_TEXTURE_DECODE_CACHE_MB contract; without one, the budget follows one eighth of host
// physical memory, clamped to [1 GiB, 2 GiB]. Keeping this policy pure makes the memory tradeoff
// directly testable without constructing a renderer or depending on the test host's RAM size.
size_t texture_decode_cache_limit_bytes(const char* override_mib,
                                        uint64_t physical_memory_bytes);

// Decide whether guest bytes are the authoritative source for a sampled-texture decode. Retained
// color and depth targets are already represented by Vulkan images and must bypass CPU decode-cache
// validation; captured host backing and non-2D/non-texture resources follow their dedicated paths.
bool texture_decode_cache_candidate(bool has_live_color_target,
                                    bool has_live_depth_target,
                                    bool has_captured_host_data,
                                    uint32_t image_dimension,
                                    bool is_sampled_texture,
                                    bool format_supported);

// Register the live renderer. `frame_dir` is where periodic BMP screenshots are written; pass
// `dump_bmps = false` (the frontend app) to suppress them — a windowed app presents to screen and
// doesn't want the periodic disk writes the headless runner uses for verification.
void register_live_renderer(const std::string& frame_dir = ".", bool dump_bmps = true);

} // namespace prosper::frontend

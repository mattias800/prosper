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

#include "frame_dump_policy.hpp"
#include "gpu/gpu_execute.hpp"          // GuestGpuWriteQuery — the in-submit mutation proof

namespace prosper::frontend {

// The decoded-texture identity map lives for one SUBMIT (#1691). A submit is cut into a new graphics
// span at every interleaved compute/DMA operation, and that split is exactly what could rewrite guest
// texture bytes mid-submit, so an entry crossing a span boundary must carry proof that nothing wrote
// the range it decoded.
//
// Reuse inside the span that produced the entry keeps the historical span-local guarantee and needs
// no proof. Crossing a span boundary needs all three of:
//
//   * `current_source_size` != 0 — a known validated source range. It is the persistent decode
//     cache's own range for this identity, and is deliberately zero for resources that cache does
//     not validate (captured replay backing, storage images, unsupported DCC states). Those keep the
//     pre-#1691 span lifetime rather than being retained against a range nobody established.
//   * the entry's range still being THIS binding's range. The resolved range is not a pure function
//     of `TextureDecodeKey`: it switches to the DCC metadata plane when live metadata content reads
//     as a fast clear, and it can follow the HLE allocation registry's pitch — neither is in the key.
//     An entry retained over the base texels must therefore not be served to a binding that now
//     resolves to the metadata plane, or a metadata-only rewrite between spans would read as
//     journal-clean over a range that is no longer the authority. The persistent cache makes exactly
//     this comparison before its own reuse; a fast path that short-circuits it must not assert less.
//   * `journal_query` == Unchanged — the ordered in-submit journal proves no retained GPU operation
//     wrote that range since the entry was established. Overlap means the guest GPU did write it
//     (the #780 stale-copy shape); Unknown (journal inactive, overflowed, or from a different submit)
//     is not evidence of anything. Both force a fresh resolve through the persistent cache's exact
//     validation, which is the pre-#1691 behavior.
//
// `journal_query` is ignored when the spans match, so a same-span caller may pass any value rather
// than paying for a query whose verdict cannot matter — this runs once per texture reference.
bool submit_local_texture_decode_reusable(uint64_t entry_span, uint64_t current_span,
                                          uint64_t entry_source_addr, uint64_t entry_source_size,
                                          uint64_t current_source_addr, uint64_t current_source_size,
                                          prosper::gpu::GuestGpuWriteQuery journal_query);

// Identity-scope accounting for the decoded-texture map, maintained unconditionally so a routed run
// and a regression test can both assert on it. `decodes` counts full CPU resolves (the work #1691
// removes), `same_span_reuses` is what the old span-scoped map already served, `cross_span_reuses` is
// what submit scope adds, and `invalidations` counts entries dropped because the journal could not
// prove their backing unchanged. Per-thread, like the renderer's other submit-scoped state.
struct TextureDecodeScopeStats {
    uint64_t decodes = 0;
    uint64_t same_span_reuses = 0;
    uint64_t cross_span_reuses = 0;
    uint64_t invalidations = 0;
    // Retained entries whose pixels the persistent cache did NOT take, so the scratch slot is their
    // only storage and had to be pinned against a later span's decode. Exported because it is the
    // only observable that distinguishes "the pin path ran" from "the pin path was never reached":
    // a test can otherwise pass while covering nothing, since an entry backed by persistent storage
    // survives a span boundary whether or not pinning works.
    uint64_t scratch_pins = 0;
};
TextureDecodeScopeStats texture_decode_scope_stats();
void reset_texture_decode_scope_stats();

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
// validation; captured host backing, cube textures, and non-texture resources follow their dedicated
// paths. Supported 3D volume inputs and the graphics backend's base-slice 2D-array view are pure
// guest-byte decodes and may be retained like ordinary 2D inputs.
bool texture_decode_cache_candidate(bool has_live_color_target,
                                    bool has_live_depth_target,
                                    bool has_captured_host_data,
                                    uint32_t image_dimension,
                                    bool is_sampled_texture,
                                    bool format_supported);

// Exact frontend shape for guest 2D_MSAA IMAGE_LOAD represented as four R32F array layers.
// Exposed so a regression can prove nearby counts/formats/layouts remain fail-visible before upload.
bool sampled_msaa_fetch_shape_supported(const prosper::gpu::ShaderResource& resource,
                                        bool is_storage_image,
                                        bool reflected_msaa_fetch);

// Apply the remaining runtime gates to a texture_decode_cache_candidate(). Keeping the candidate
// explicit is important: DCC metadata can provide a nonzero source span even when renderer-owned
// color is authoritative, but that metadata must never make guest decode-cache state eligible.
bool persistent_texture_decode_cache_eligible(bool guest_decode_candidate,
                                              bool compute_image_hit,
                                              bool is_storage_image,
                                              bool cache_disabled,
                                              bool compression_supported,
                                              size_t cache_limit,
                                              size_t source_size);

// Graphics currently lowers sampled 2D arrays to a 2D base-slice image. Array descriptors retain
// the allocation base plus the selected mip's per-layer offset, so CPU decode must begin at that
// offset (except for packed mip tails, whose coordinates are relative to the shared block base).
uint64_t texture_decode_source_address(uint64_t gpu_address,
                                       uint32_t image_dimension,
                                       bool in_mip_tail,
                                       uint32_t layer_mip_offset_bytes);

// Once an exact source has been validated while a working write watch is armed, the retained encoded
// copy is redundant: Unchanged permits reuse, while Dirty/Unknown conservatively forces a re-decode.
// Sources that double as decoded pixels and audit modes still need their byte-exact baseline.
bool texture_source_snapshot_can_follow_watch(bool source_matches_pixels,
                                              bool validation_audit_enabled,
                                              bool watch_active,
                                              size_t retained_source_bytes,
                                              size_t expected_source_bytes);

// Register the live renderer. `frame_dir` only selects where explicitly requested periodic BMPs are
// written; it does not enable them. PROSPER_NO_FRAME_DUMPS remains a final kill switch even when a
// caller explicitly requests dumps.
void register_live_renderer(const std::string& frame_dir = ".",
                            bool dump_bmps = kFrameDumpsByDefault);

} // namespace prosper::frontend

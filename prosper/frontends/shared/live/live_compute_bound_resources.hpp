// live_compute_bound_resources.hpp -- the per-dispatch view of one bound buffer and one bound image.
//
// Pure data owned by a single dispatch: the Vulkan handles it created or borrowed, the proofs it
// gathered, and the timing it reports. Lifted out of live_compute.cpp so the dispatch tail's
// explicit state (ADR 0009 Stage 1) can name them without the whole backend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "shared/live/compute_buffer_cache_key.hpp"
#include "shared/compute/compute_buffer_timing.hpp"
#include "shared/compute/compute_image_cache_key.hpp"
#include "shared/live/gpu_retile.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/present/compute_scanout.hpp"

namespace prosper::frontend {

struct BoundBuffer {
    const prosper::gpu::ShaderResource* resource = nullptr;
    size_t descriptor_index = SIZE_MAX; // reflected binding that owns this flattened table entry
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // Offset of this window inside a shared arena allocation (0 when it has its own buffer).
    VkDeviceSize buffer_offset = 0;
    size_t alias_of = SIZE_MAX;         // exact guest range sharing an earlier storage buffer
    size_t bytes = 0;                   // Vulkan buffer bytes (may be a detiled image view)
    size_t guest_bytes = 0;             // physical guest backing (may exceed logical image bytes)
    bool output_conflict = false;       // independent buffer writeback can overwrite this range
    bool writable = false;              // reflected OpStore/writing-atomic reachability
    bool atomic_image = false;           // R32_UINT StorageImage exposed as a linear atomic SSBO
    uint32_t atomic_layers = 1;          // #2265: array layers staged for that view (1 when plain 2D)
    size_t atomic_slice_bytes = 0;       // physical guest bytes PER LAYER (tiled slices are padded)
    bool persistent = false;
    bool upload_skipped = false;
    VkBuffer result_baseline = VK_NULL_HANDLE;
    size_t compare_flag_index = SIZE_MAX;
    bool gpu_result_unchanged = false;
    uint32_t dirty_watch_chunks = 0;
    uint32_t total_watch_chunks = 0;
    ComputeBufferTiming timing;
    ComputeBufferCacheKey cache_key{};
    std::vector<uint8_t> linear_seed;    // detiled upload for an atomic-image buffer
    uint64_t before_hash = 0, after_hash = 0;
    uint64_t changed_bytes = 0;
};

// One image binding (#590): a sampled texture (usually RGBA8, with native UINT8x4 and R11G11B10F
// views where shader-visible numeric semantics require them) or a storage image. Storage normally
// uses R32G32B32A32_UINT format-free texels; packed R11G11B10 can instead use exact R32_UINT words.
struct BoundImage {
    const prosper::gpu::ShaderResource* resource = nullptr;
    uint32_t binding = 0;
    VkBuffer retile_buffer = VK_NULL_HANDLE;
    VkDeviceMemory retile_memory = VK_NULL_HANDLE;
    VkDescriptorPool retile_pool = VK_NULL_HANDLE;
    VkDescriptorSet retile_set = VK_NULL_HANDLE;
    bool retile_binding_pending = false;
    GpuRetilePipeline* fused_retile = nullptr;
    GpuRetileParameters retile_parameters{};
    bool direct_retile = false;
    uint32_t image_retile_index = 0;
    VkFormat materialized_format = VK_FORMAT_UNDEFINED;
    bool storage = false;               // descriptor/representation, independent of access
    bool storage_writeback = false;     // whole-alias output obligation, conservatively proven
    bool prior_output_conflict = false; // earlier writeback can invalidate setup-time equality
    bool final_output_conflict = false; // later writeback or self metadata changes final interpretation
    bool native_float_storage = false;  // Vulkan performs exact UNORM/float conversion at native width
    bool native_uint_storage = false;   // exact guest-width integer texels
    bool packed_r11_storage = false;    // shader packs exact R11G11B10 words into typed R32_UINT
    bool unorm_rtt_value_reuse = false; // float R16 view reuses authoritative RGBA8 values
    bool graphics_sampled_usage = false;// native image was created with SAMPLED usage for export
    bool exact_storage_bytes() const {
        return native_float_storage || native_uint_storage || packed_r11_storage;
    }
    uint32_t texel_depth = 1;           // realized 3D depth; ordinary array layers are counted separately
    uint32_t array_layers = 1;           // Vulkan array-layer count (3D depth remains one layer)
    // #3048: the guest-declared mip chain this image materializes, and where each level past zero
    // begins in the staging buffer. 1 (with an empty offset table) is the historical single-level
    // image; the recompiler reads the same derivation before emitting an explicit LOD.
    uint32_t mip_levels = 1;
    std::vector<VkDeviceSize> mip_staging_offsets;
    bool arrayed_2d = false;            // SPIR-V requires a real 2D-array view (not base-slice fallback)
    // Every image access to this binding is an OpImageQuery*: it needs a correctly SHAPED image
    // to answer the query and no texels at all. Uploading them is not merely wasted work -- the
    // staging is sized for the guest surface while the image is the declared query shape, which
    // is VUID-vkCmdCopyBufferToImage-imageSubresource-07972, measured (#657).
    bool query_only_shape = false;
    bool stacked_cube = false;          // cube lowering addresses six faces as one w x 6h 2D image
    bool depth_view = false;             // reflected SPIR-V uses a true depth image/sampler contract
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE; // combined image sampler only
    VkDeviceSize row_pitch = 0;         // LINEAR-tiling row pitch (bytes), from vkGetImageSubresourceLayout
    size_t guest_bytes = 0;             // real linear/tiled guest backing footprint
    size_t alias_of = SIZE_MAX;         // exact sampled/storage binding sharing an earlier image/view
    uint8_t* dcc_metadata = nullptr;    // DCC control bytes to mark uncompressed after writeback
    size_t dcc_metadata_bytes = 0;
    const uint8_t* sampled_metadata = nullptr; // input dependency, never a reset obligation
    size_t sampled_metadata_bytes = 0;
    // Borrowed renderer-owned image bound in place (#1095). `image` is then owned by the live
    // renderer: it must not be destroyed here, its layout must be restored, and the pin taken at
    // import time must be released.
    bool imported = false;
    bool imported_component_order_bgra = false;   // #4291: see LiveTargetImageImport
    bool imported_depth = false;        // borrowed persistent DS depth plane, not a color RTT
    // A one-component Uint32 T# can alias a renderer-owned D32 depth plane byte-for-byte. Vulkan
    // cannot create an R32_UINT view of a depth image, so keep the borrowed DS image as a transfer
    // source and materialize the guest-declared integer view through a device buffer. This remains
    // entirely on the shared GPU; no stale guest backing or synchronous host readback is involved.
    bool depth_bits_source = false;
    VkImage depth_bits_image = VK_NULL_HANDLE;
    VkFormat depth_bits_format = VK_FORMAT_UNDEFINED;
    uint32_t depth_bits_saved_layout = 0;
    // RGBA8 UNORM -> UINT preserves the CPU snapshot's bytes, but must sample an INTEGER
    // image. Copy into an owned UINT image; no mutable image or mismatched UNORM view is needed.
    bool packed10_source = false;
    VkDescriptorPool packed10_pool = VK_NULL_HANDLE;
    VkDescriptorSet packed10_set = VK_NULL_HANDLE;
    bool color_bits_source = false;
    VkImage color_bits_image = VK_NULL_HANDLE;
    uint32_t color_bits_saved_layout = 0;
    VkFormat imported_format = VK_FORMAT_UNDEFINED;
    prosper::gpu::LiveTargetPixelFormat imported_pixel_format =
        prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm;
    bool imported_transfer_dst = false;
    bool persistent = false;            // guest-backed sampled image retained across dispatches
    bool cache_candidate = false;
    bool post_writeback_promotion_candidate = false;
    bool renderer_seeded_result_candidate = false;
    // Exact cached allocation leased for a DCC-unsafe producer. Source authority was invalidated,
    // upload_skipped remains false, and cache publication still waits for post-writeback metadata.
    bool forced_seed_allocation_reused = false;
    bool direct_storage_detile_used = false;
    // A linear 2D image's row pitch in guest memory (compute_linear_row_pitch), or 0 for tight rows.
    // The sampled upload and storage seed gather rows from it; the storage writeback scatters to it.
    size_t guest_row_pitch = 0;
    bool watch_backed_snapshot_skip_requested = false;
    bool upload_skipped = false;         // write watch proved the cached source unchanged
    VkDeviceSize allocation_bytes = 0;
    VkDeviceSize staging_allocation_bytes = 0;
    // Exact bytes copied from the Vulkan image into staging. Native typed storage already has the
    // guest's row-major byte width; the raw-uvec4 fallback has four uint32_t channels per texel.
    // Raw results may retain noncanonical channel precision only for write-only alias groups;
    // current-source validation still proves their packed bytes match the required guest input.
    VkDeviceSize exact_result_bytes = 0;
    VkBuffer result_baseline = VK_NULL_HANDLE;
    size_t compare_flag_index = SIZE_MAX;
    bool gpu_result_unchanged = false;
    ComputeImageCacheKey cache_key{};
    // A prior native storage result can seed the sampled cache with a device-local copy. It remains
    // a distinct image because a dispatch may sample the old guest value while writing a new value
    // to the same address; binding one VkImage for both would introduce an in-dispatch data race.
    VkImage compute_transfer_seed = VK_NULL_HANDLE;
    ComputeImageCacheKey compute_transfer_seed_key{};
    bool compute_transfer_seed_borrowed = false;
    std::vector<uint8_t> cache_source_snapshot; // first-use source captured before the transfer
    size_t seed_from_imported = SIZE_MAX; // renderer image copied on-device into this partial-write target
    // Standalone source-only pin: no sampled sibling is needed to preserve current RTT inputs.
    // The private storage image still owns writes and publishes the ordinary guest mirror.
    prosper::gpu::LiveTargetImageImport standalone_seed{};
    bool standalone_seed_swap_rb = false;   // BGRA source: seeded through BgraSeedScratch
    const char* standalone_seed_decision = "not-requested";
    // Separate write-only pin. It may refer to an invalid renderer image, so it must never seed a
    // compute input; only a completed full staging result may replace that allocation's pixels.
    prosper::gpu::LiveTargetImageImport mirror_destination{};
    bool mirror_destination_revoked = false;
    bool mirror_destination_recorded = false;
    // A read-only imported binding can sample the old image during this dispatch. The result copy
    // then leaves that image in GENERAL for the import owner's ordinary layout restoration.
    bool mirror_destination_shared_import = false;
    // #3915: GPU-present mirror of a complete display-buffer result (compute_scanout.hpp).
    prosper::frontend::ComputeScanoutTarget scanout_mirror{};
    bool scanout_mirror_recorded = false;
    bool scanout_mirror_committed = false;
    bool has_renderer_seed() const {
        return seed_from_imported != SIZE_MAX || standalone_seed.valid();
    }
    bool mirror_result_to_imported = false;
    bool storage_write_only = false; // access union of every descriptor sharing this image
    const std::vector<uint32_t>* storage_write_mask = nullptr;
    std::vector<uint8_t> untouched_seed; // exact linear guest bytes for non-injective raw conversions
    uint64_t imported_addr = 0;
    uint32_t imported_width = 0, imported_height = 0; // actual renderer VkImage extent
    uint32_t imported_saved_layout = 0;      // VkImageLayout the renderer left the image in
    // Several bindings can borrow the SAME renderer image without being folded together, because
    // the import contract is looser than the alias contract (it ignores sampler state, T# size and
    // img_dim 1-vs-5). Exactly one of them must emit the layout transitions: a second barrier pair
    // would declare oldLayout=saved on an image already in GENERAL, which is an invalid transition a
    // driver may treat as a discard. Ownership is derived at the barrier loops rather than stored
    // here -- see imported_barrier_owner().
    uint64_t before_hash = 0, after_hash = 0; // trace-only storage-image writeback evidence
    uint64_t nonzero_channels = 0;
};

}  // namespace prosper::frontend

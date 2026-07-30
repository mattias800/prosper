#pragma once
#include "gpu/gpu_execute.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::frontend {

enum class ComputeImageCacheClass : uint8_t { sampled, storage };

// Read-only sampled inputs are often tiny, numerous, and cheap to upload; retaining all of them
// wastes cache identities and device memory. A storage target has a different cost model: even a
// small repeated output otherwise incurs staging readback, guest-format packing, and layout work.
// One 4 KiB host page is the measured storage crossover and avoids retaining sub-page Vulkan
// objects. Keep the default crossover policy explicit and independently testable.
constexpr uint64_t compute_image_cache_default_minimum_bytes(
    ComputeImageCacheClass image_class) {
    return image_class == ComputeImageCacheClass::storage ? 4ull * 1024ull
                                                          : 1024ull * 1024ull;
}

constexpr bool compute_image_cache_default_eligible(
    uint64_t bytes, ComputeImageCacheClass image_class) {
    return bytes >= compute_image_cache_default_minimum_bytes(image_class);
}

// Pack one raw float32 channel to UNORM8 using the storage-image conversion contract. Kept public
// so the optimized scalar conversion can be checked directly against the previous lround path.
uint8_t storage_pack_unorm8(uint32_t float_bits);
uint16_t storage_pack_unorm16(uint32_t float_bits);
void storage_pack_unorm8_range(const uint32_t* channels, uint32_t components,
                               size_t texels, uint8_t* packed);

// Hot storage/sampled-image conversions. These retain the scalar reference semantics while using
// exhaustive binary16 lookup tables or exact runtime-dispatched vector paths in production loops.
uint32_t storage_unpack_float16_bits(uint16_t half_bits);
// Expand tightly-strided RGBA16F to RGBA32F channel bits. The F16C path repairs signaling-NaN
// payloads so every output bit remains identical to half_to_float.
void storage_unpack_float16x4_range(const uint8_t* rgba16f, size_t texels, uint32_t* channels);
uint8_t sampled_float16_to_unorm8(uint16_t half_bits);
void sampled_float16_to_unorm8_range(const uint8_t* source, uint32_t components,
                                     size_t texels, uint8_t* rgba);
// Pack tightly-strided RGBA32F channel bits to RGBA16F. Uses a runtime-dispatched F16C path where
// available and preserves float_to_half's exact NaN payload/rounding contract.
void storage_pack_float16x4_range(const uint32_t* channels, size_t texels, uint8_t* rgba16f);

// A typed Vulkan storage image already exposes the guest format as exact row-major bytes. For a
// tiled guest surface the tiler can therefore read the mapped staging image directly, unless a
// poison-proving dispatch still needs a mutable linear copy to restore untouched texels.
constexpr bool storage_writeback_can_tile_mapped_bytes(bool native_float_storage,
                                                       uint32_t tile_mode,
                                                       bool poison_verify,
                                                       bool disabled) {
    return native_float_storage && tile_mode != 0 && !poison_verify && !disabled;
}

// Whether a sampled guest view can bind a renderer-owned target without a CPU readback/conversion.
// The formats must describe the same Vulkan texels exactly; aliases or numeric conversions fall back
// to the snapshot upload path.
bool direct_sampled_rtt_compatible(prosper::gpu::DataFormat format, uint32_t components,
                                   prosper::gpu::LiveTargetPixelFormat target_format);

// Reconstruct a packed R11G11B10 sampled surface from the renderer's canonical color snapshot.
// The renderer keeps float targets as RGBA16F and ordinary targets as RGBA8; compute descriptors
// can subsequently alias that same target as GFX10 10_11_11_FLOAT. This conversion restores the
// descriptor-visible texel representation without reading stale guest backing.
bool pack_live_target_r11g11b10(const prosper::gpu::LiveTargetSnapshot& snapshot,
                                uint8_t* packed, size_t packed_size);

// Execute already-realized compute items synchronously. Exposed for the production-backend test.
bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items);

// Borrow an exact native storage result for a later sampled graphics binding. The import is
// deliberately narrower than the compute image cache: only a successful typed-storage dispatch can
// publish one, the complete descriptor identity must match, and either the current submit journal or
// the cache's page watch must prove that no later guest write overlapped the result. The lease pins
// the Vulkan image until the graphics submission has completed. Handles remain opaque here so this
// shared interface does not expose Vulkan types.
struct LiveComputeImageImport {
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t native_format = 0; // opaque VkFormat
    uint32_t layout = 0;        // opaque VkImageLayout; currently GENERAL
    void* image = nullptr;      // borrowed VkImage
    void* device = nullptr;     // VkDevice that owns image
    std::shared_ptr<void> lease;

    bool valid() const {
        return width && height && depth && native_format && image && device && lease;
    }
};
bool import_live_compute_storage_image(const prosper::gpu::ShaderResource& sampled_resource,
                                       uint64_t guest_bytes,
                                       LiveComputeImageImport& import);

// Monotonic diagnostic count of writable-buffer results whose exact GPU comparison avoided a host
// mapping/scan. Exposed so the production-backend test can prove that optimization path executes.
uint64_t live_compute_buffer_gpu_result_skips();

// Deterministic failure injection for the storage-image recovery regression test. The next storage
// readback fails after dispatch, exercising retained-image invalidation without a Vulkan fault.
void live_compute_fail_next_storage_readback_for_test();

// Register the synchronous Vulkan compute backend used by AGC submit processing.
void register_live_compute();

} // namespace prosper::frontend

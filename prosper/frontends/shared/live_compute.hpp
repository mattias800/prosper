#pragma once
#include "gpu/gpu_execute.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::frontend {

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

// Register the synchronous Vulkan compute backend used by AGC submit processing.
void register_live_compute();

} // namespace prosper::frontend

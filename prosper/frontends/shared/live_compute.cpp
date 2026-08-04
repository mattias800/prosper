#include "live_compute.hpp"
#include "compute_authority_live_census.hpp"
#include "compute_timing_selector.hpp"
#include "compute_transfer_gate_census.hpp"
#include "live_target_format.hpp"
#include "rtt_scale.hpp"
#include "rtt_authority.hpp"
#include "seed_reprove.hpp"
#include "vulkan_device_select.hpp"
#include "write_watch_policy.hpp"
#include "performance_capture.hpp"      // bounded F8 post-trigger compute timing
#include "performance_timing_policy.hpp" // F8 measures without enabling verbose timing logs

#include "gpu/bc_decode.hpp"
#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/shader_resources.hpp"
#include "gpu/spirv_builder.hpp"
#include "gpu/tile.hpp"
#include "gpu/writer_provenance.hpp"
#include "host/guest_write_watch.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_HAVE_TARGET_F16C 1
#endif

namespace prosper::frontend {

bool compute_sampled_dcc_fast_clear_rgba8(
    const prosper::gpu::ShaderResource& resource,
    bool ordinary_guest_backed_sampled_view,
    bool arrayed_sampled_view,
    bool disabled,
    uint8_t* rgba,
    size_t texel_count,
    const uint8_t* metadata,
    size_t metadata_bytes,
    uint8_t* clear_code) {
    const uint32_t components = resource.num_components ? resource.num_components : 1u;
    if (disabled || !ordinary_guest_backed_sampled_view || arrayed_sampled_view ||
        resource.cls != prosper::gpu::ResourceClass::Texture ||
        resource.format != prosper::gpu::DataFormat::Float16 || components != 4u ||
        resource.img_dim != 1u || resource.depth != 1u ||
        resource.declared_mip_levels != 1u || resource.in_mip_tail ||
        resource.layer_stride_bytes || resource.layer_mip_offset_bytes ||
        resource.srgb || resource.depth_compare || !resource.compression_enabled ||
        !resource.metadata_addr || !rgba || !texel_count)
        return false;
    const uint64_t expected_metadata = prosper::gpu::gpu_capture_dcc_metadata_footprint(resource);
    if (!expected_metadata || expected_metadata > SIZE_MAX ||
        metadata_bytes != static_cast<size_t>(expected_metadata))
        return false;
    return prosper::gpu::gfx10_dcc_fast_clear_rgba8(
        rgba, texel_count, metadata, metadata_bytes, components,
        resource.alpha_is_on_msb, clear_code);
}

uint8_t storage_pack_unorm8(uint32_t float_bits) {
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));
    if (!(value > 0.0f)) return 0; // Includes negative values and NaN.
    if (value >= 1.0f) return 255;
    const float scaled = value * 255.0f;
    const uint32_t whole = static_cast<uint32_t>(scaled);
    return static_cast<uint8_t>(whole + (scaled - static_cast<float>(whole) >= 0.5f));
}

uint16_t storage_pack_unorm16(uint32_t float_bits) {
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));
    if (!(value > 0.0f)) return 0; // Includes negative values and NaN.
    if (value >= 1.0f) return 65535;
    const float scaled = value * 65535.0f;
    const uint32_t whole = static_cast<uint32_t>(scaled);
    return static_cast<uint16_t>(whole +
        (scaled - static_cast<float>(whole) >= 0.5f));
}

uint32_t storage_unpack_float16_bits(uint16_t half_bits) {
    // Full binary16 lookup: the source domain is only 64 Ki entries, while Astro Bot expands more
    // than 33 million FP16 channels for one 4K storage image. Keeping the already-converted float
    // bits in a 256 KiB table removes the branchy scalar decoder from that inner loop without
    // changing a single NaN payload, signed zero, subnormal, or infinity bit.
    static const std::array<uint32_t, 65536> table = [] {
        std::array<uint32_t, 65536> result{};
        for (uint32_t bits = 0; bits < result.size(); ++bits) {
            const float value = prosper::gpu::half_to_float(static_cast<uint16_t>(bits));
            std::memcpy(&result[bits], &value, sizeof(value));
        }
        return result;
    }();
    return table[half_bits];
}

bool pack_live_target_r11g11b10(const prosper::gpu::LiveTargetSnapshot& snapshot,
                                uint8_t* packed, size_t packed_size) {
    if (!snapshot.width || !snapshot.height || !snapshot.pixels) return false;
    const uint64_t texels = static_cast<uint64_t>(snapshot.width) * snapshot.height;
    const uint32_t source_bytes =
        prosper::frontend::live_target_pixel_format_bytes(snapshot.format);
    if (!source_bytes) return false;
    const prosper::frontend::LiveTargetSourceLayout layout =
        prosper::frontend::live_target_source_layout(snapshot.format);
    if (texels > SIZE_MAX / source_bytes || texels > SIZE_MAX / sizeof(uint32_t) ||
        snapshot.pixels->size() != static_cast<size_t>(texels) * source_bytes ||
        !packed || packed_size != static_cast<size_t>(texels) * sizeof(uint32_t))
        return false;
    if (layout == prosper::frontend::LiveTargetSourceLayout::PackedR11G11B10) {
        std::memcpy(packed, snapshot.pixels->data(), packed_size);
        return true;
    }
    for (size_t t = 0; t < static_cast<size_t>(texels); ++t) {
        float rgb[3]{};
        if (layout == prosper::frontend::LiveTargetSourceLayout::Float16x4) {
            for (uint32_t c = 0; c < 3; ++c) {
                uint16_t half = 0;
                std::memcpy(&half, snapshot.pixels->data() + t * 8 + c * 2, sizeof(half));
                rgb[c] = prosper::gpu::half_to_float(half);
            }
        } else {
            for (uint32_t c = 0; c < 3; ++c)
                rgb[c] = (*snapshot.pixels)[t * 4 + c] / 255.0f;
        }
        const uint32_t word = static_cast<uint32_t>(prosper::gpu::float_to_f11(rgb[0])) |
                              (static_cast<uint32_t>(prosper::gpu::float_to_f11(rgb[1])) << 11) |
                              (static_cast<uint32_t>(prosper::gpu::float_to_f10(rgb[2])) << 22);
        std::memcpy(packed + t * sizeof(word), &word, sizeof(word));
    }
    return true;
}

namespace {

const std::array<uint8_t, 65536>& sampled_float16_unorm8_table() {
    static const std::array<uint8_t, 65536> table = [] {
        std::array<uint8_t, 65536> result{};
        for (uint32_t bits = 0; bits < result.size(); ++bits) {
            float value = prosper::gpu::half_to_float(static_cast<uint16_t>(bits));
            if (std::isnan(value)) value = 0.0f;
            value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
            result[bits] = static_cast<uint8_t>(std::lround(value * 255.0f));
        }
        return result;
    }();
    return table;
}

} // namespace

uint8_t sampled_float16_to_unorm8(uint16_t half_bits) {
    // The guest-backed sampled fallback maps NaN/negative to zero and positive infinity to one,
    // matching its historical scalar clamp exactly. Cache every possible input so it avoids
    // half_to_float + lround for every channel. Direct-bound renderer targets bypass this entirely.
    return sampled_float16_unorm8_table()[half_bits];
}

bool direct_sampled_rtt_compatible(prosper::gpu::DataFormat format, uint32_t components,
                                   prosper::gpu::LiveTargetPixelFormat target_format,
                                   bool float_sampled_values) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::LiveTargetPixelFormat;
    const bool exact =
        (components == 4 &&
         ((format == DataFormat::Unorm8 &&
           target_format == LiveTargetPixelFormat::Rgba8Unorm) ||
          (format == DataFormat::Float16 &&
           target_format == LiveTargetPixelFormat::Rgba16Float))) ||
        (components == 3 && format == DataFormat::Float10_11_11 &&
         target_format == LiveTargetPixelFormat::R11G11B10Float);
    // The renderer's RGBA8 fallback already stores the numeric UNORM value. Expanding each byte to
    // uint16 as byte*257 and reading R16_UNORM produces exactly byte/255 again. Vulkan performs
    // that UNORM-to-float conversion for normalized sampling and integer-coordinate OpImageFetch,
    // so either operation may consume the canonical RGBA8 image. Coordinate/extent compatibility
    // is checked separately before a renderer image is borrowed.
    const bool equivalent_unorm_values = float_sampled_values && components == 4 &&
        format == DataFormat::Unorm16 &&
        target_format == LiveTargetPixelFormat::Rgba8Unorm;
    return exact || equivalent_unorm_values;
}

namespace {

std::atomic<uint64_t> g_buffer_gpu_result_skips{0};
std::atomic<uint64_t> g_compute_storage_transfer_seeds{0};
std::atomic<uint64_t> g_dcc_post_writeback_promotions{0};
std::atomic<uint64_t> g_dcc_forced_seed_allocation_reuses{0};
std::atomic<uint64_t> g_dcc_post_writeback_replacements{0};
std::atomic<bool> g_fail_next_storage_readback_for_test{false};
std::atomic<bool> g_leave_next_dcc_metadata_compressed_for_test{false};
std::atomic<bool> g_disable_next_dcc_allocation_reuse_for_test{false};
std::atomic<bool> g_limit_next_image_replacement_for_test{false};
std::atomic<bool> g_zero_next_cold_storage_snapshot_minimum_for_test{false};
std::atomic<bool> g_force_next_image_result_host_fallback_for_test{false};
std::atomic<bool> g_fail_next_image_result_buffer_retain_for_test{false};
std::atomic<bool> g_force_next_queue_submit_device_lost_for_test{false};
std::atomic<uint64_t> g_live_compute_queue_submit_attempts{0};

VkFormat native_storage_vk_format(prosper::gpu::DataFormat format, uint32_t components) {
    using prosper::gpu::DataFormat;
    switch (format) {
    case DataFormat::Unorm8:
        if (components == 1) return VK_FORMAT_R8_UNORM;
        if (components == 2) return VK_FORMAT_R8G8_UNORM;
        if (components == 4) return VK_FORMAT_R8G8B8A8_UNORM;
        break;
    case DataFormat::Float16:
        if (components == 1) return VK_FORMAT_R16_SFLOAT;
        if (components == 2) return VK_FORMAT_R16G16_SFLOAT;
        if (components == 4) return VK_FORMAT_R16G16B16A16_SFLOAT;
        break;
    case DataFormat::Float32:
        if (components == 1) return VK_FORMAT_R32_SFLOAT;
        if (components == 2) return VK_FORMAT_R32G32_SFLOAT;
        if (components == 4) return VK_FORMAT_R32G32B32A32_SFLOAT;
        break;
    case DataFormat::Float10_11_11:
        if (components == 3) return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        break;
    default:
        break;
    }
    return VK_FORMAT_UNDEFINED;
}

bool native_storage_image_create_supported(VkPhysicalDevice physical, VkFormat format,
                                           VkImageType image_type, uint32_t width,
                                           uint32_t height, uint32_t depth,
                                           uint32_t array_layers,
                                           VkImageUsageFlags extra_usage = 0) {
    if (!physical || format == VK_FORMAT_UNDEFINED || !width || !height || !depth ||
        !array_layers)
        return false;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extra_usage;
    VkImageFormatProperties properties{};
    return vkGetPhysicalDeviceImageFormatProperties(
               physical, format, image_type, VK_IMAGE_TILING_OPTIMAL,
               usage, 0, &properties) == VK_SUCCESS &&
           width <= properties.maxExtent.width && height <= properties.maxExtent.height &&
           depth <= properties.maxExtent.depth &&
           array_layers <= properties.maxArrayLayers;
}

// Storage images use an RGBA32_UINT interchange surface so format conversion remains bit-exact on
// readback. A valid 10K x 2K single-channel PS5 surface consequently expands from 40 MiB of guest
// memory to 320 MiB on the host. Keep a hard allocation guard, but leave enough room for one such
// console-sized resource instead of rejecting it solely because of the interchange representation.
constexpr VkDeviceSize kMaxComputeImageBytes = 512ull << 20;
constexpr size_t kMaxCachedComputePipelines = 4096;

// Large storage-image format conversions are independent per texel. Astro Bot's 4K FP16 target is
// 8.3 million texels, so leaving its lookup/pack walk on one core made a ~5 ms GPU dispatch wait on
// hundreds of milliseconds of host work. Keep small surfaces scalar and cap large conversions at
// sixteen workers because both the detiler and these loops eventually become memory-bandwidth-bound.
// Astro Bot's 4K path is still measurably faster at 16 on a 16-core Strix Halo; 32 regresses.
template <class Body>
void parallel_compute_texels(size_t count, size_t work_bytes, Body&& body,
                             unsigned max_threads = 16u) {
    if (!count) return;
    static const unsigned configured = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_CONVERSION_THREADS");
        if (!value || !*value) return 0u;
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        return static_cast<unsigned>(std::min(parsed, 32ul));
    }();
    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned wanted = std::min(
        configured ? configured : std::min(hardware ? hardware : 4u, 16u), max_threads);
    const unsigned by_work = static_cast<unsigned>(std::min<size_t>(
        count, std::max<size_t>(1, work_bytes / (512u * 1024u))));
    const unsigned threads = std::max(1u, std::min(wanted ? wanted : 1u, by_work));
    if (threads <= 1) {
        body(size_t{0}, count);
        return;
    }
    const size_t chunk = (count + threads - 1) / threads;
    // jthread makes a partially-created set exception-safe: if the next OS thread cannot be
    // created, already-started workers are still joined. Finish the unspawned ranges on this
    // thread so transient resource pressure degrades to less parallelism instead of aborting.
    std::vector<std::jthread> workers;
    workers.reserve(threads - 1);
    unsigned next_worker = 1;
    try {
        for (; next_worker < threads; ++next_worker) {
            const size_t begin = static_cast<size_t>(next_worker) * chunk;
            const size_t end = std::min(count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back([&body, begin, end] { body(begin, end); });
        }
    } catch (const std::system_error&) {
        // Fall through: ranges that did not get a worker run synchronously below.
    }
    body(size_t{0}, std::min(count, chunk));
    for (; next_worker < threads; ++next_worker) {
        const size_t begin = static_cast<size_t>(next_worker) * chunk;
        const size_t end = std::min(count, begin + chunk);
        if (begin >= end) break;
        body(begin, end);
    }
}

bool compute_buffers_equal(const void* lhs, const void* rhs, size_t bytes) {
    // glibc's vectorized memcmp is excellent for ordinary bindings. A 32 MiB persistent SSBO still
    // costs several milliseconds on one core, though, and the common maintenance-kernel result is
    // unchanged. Eight independent ranges measured best on the target APU (more workers became
    // memory-bandwidth/thread-start limited). This remains an exact comparison of every byte.
    constexpr size_t kParallelThreshold = 8u << 20;
    if (bytes < kParallelThreshold) return std::memcmp(lhs, rhs, bytes) == 0;
    const auto* a = static_cast<const uint8_t*>(lhs);
    const auto* b = static_cast<const uint8_t*>(rhs);
    std::atomic<bool> equal{true};
    parallel_compute_texels(bytes, bytes,
        [&](size_t begin, size_t end) {
            if (std::memcmp(a + begin, b + begin, end - begin) != 0)
                equal.store(false, std::memory_order_relaxed);
        }, 8u);
    return equal.load(std::memory_order_relaxed);
}

void copy_compute_buffer(void* destination, const void* source, size_t bytes) {
    constexpr size_t kParallelThreshold = 8u << 20;
    if (bytes < kParallelThreshold) {
        std::memcpy(destination, source, bytes);
        return;
    }
    auto* dst = static_cast<uint8_t*>(destination);
    const auto* src = static_cast<const uint8_t*>(source);
    parallel_compute_texels(bytes, bytes,
        [&](size_t begin, size_t end) {
            std::memcpy(dst + begin, src + begin, end - begin);
        }, 8u);
}

#if defined(PROSPER_HAVE_TARGET_F16C)
__attribute__((target("avx2")))
__m128i storage_pack_unorm8x8_avx2(__m256 values) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    // Reproduce storage_pack_unorm8 exactly: unordered/negative/zero -> 0, >=1 -> 255,
    // otherwise multiply in float32 and round a .5 tie upward. Clamp before CVTTPS2DQ so NaN
    // and infinities never raise an invalid-conversion exception.
    const __m256 positive = _mm256_cmp_ps(values, zero, _CMP_GT_OQ);
    const __m256 upper = _mm256_cmp_ps(values, one, _CMP_GE_OQ);
    __m256 bounded = _mm256_blendv_ps(zero, values, positive);
    bounded = _mm256_blendv_ps(bounded, one, upper);
    const __m256 scaled = _mm256_mul_ps(bounded, scale);
    const __m256i whole = _mm256_cvttps_epi32(scaled);
    const __m256 fraction = _mm256_sub_ps(scaled, _mm256_cvtepi32_ps(whole));
    const __m256i increment = _mm256_castps_si256(
        _mm256_cmp_ps(fraction, half, _CMP_GE_OQ));
    const __m256i rounded = _mm256_sub_epi32(whole, increment); // true mask is -1
    const __m128i words = _mm_packus_epi32(
        _mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1));
    return _mm_packus_epi16(words, _mm_setzero_si128());
}

__attribute__((target("avx2")))
void storage_pack_unorm8_avx2(const uint32_t* channels, uint32_t components,
                              size_t begin, size_t end, uint8_t* packed) {
    size_t texel = begin;
    size_t step = 0;
    if (components == 2) {
        step = 4;
    } else if (components == 4) {
        step = 2;
    }
    for (; step && texel + step <= end; texel += step) {
        __m256i bits;
        if (components == 4) {
            bits = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(channels + texel * 4));
        } else {
            const uint32_t* source = channels + texel * 4;
            const __m128i texels01 = _mm_unpacklo_epi64(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)),
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 4)));
            const __m128i texels23 = _mm_unpacklo_epi64(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 8)),
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 12)));
            bits = _mm256_inserti128_si256(_mm256_castsi128_si256(texels01), texels23, 1);
        }
        const __m256 values = _mm256_castsi256_ps(bits);
        const __m128i bytes = storage_pack_unorm8x8_avx2(values);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(packed + texel * components), bytes);
    }
    for (; texel < end; ++texel)
        for (uint32_t channel = 0; channel < components; ++channel)
            packed[texel * components + channel] =
                storage_pack_unorm8(channels[texel * 4 + channel]);
}

__attribute__((target("avx2,f16c")))
void sampled_float16x4_to_unorm8_f16c(const uint8_t* source, size_t begin, size_t end,
                                      uint8_t* rgba,
                                      const std::array<uint8_t, 65536>& fallback_table) {
    size_t channel = begin * 4;
    const size_t channel_end = end * 4;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 rounding = _mm256_set1_ps(0.5f);
    for (; channel + 8 <= channel_end; channel += 8) {
        const __m128i half = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source + channel * sizeof(uint16_t)));
        __m256 value = _mm256_cvtph_ps(half);
        // Match the scalar contract exactly: NaN/negative -> 0, +infinity/>1 -> 1, then
        // round non-negative value*255 halfway away from zero (floor(x + 0.5)).
        value = _mm256_and_ps(value, _mm256_cmp_ps(value, value, _CMP_ORD_Q));
        value = _mm256_min_ps(_mm256_max_ps(value, zero), one);
        const __m256i integers = _mm256_cvttps_epi32(
            _mm256_add_ps(_mm256_mul_ps(value, scale), rounding));
        const __m128i words = _mm_packus_epi32(
            _mm256_castsi256_si128(integers), _mm256_extracti128_si256(integers, 1));
        const __m128i bytes = _mm_packus_epi16(words, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(rgba + channel), bytes);
    }
    for (; channel < channel_end; ++channel) {
        uint16_t half = 0;
        std::memcpy(&half, source + channel * sizeof(half), sizeof(half));
        rgba[channel] = fallback_table[half];
    }
}

__attribute__((target("avx2,f16c")))
void storage_pack_float16x4_f16c(const uint32_t* channels, size_t begin, size_t end,
                                 uint8_t* rgba16f) {
    size_t texel = begin;
    for (; texel + 2 <= end; texel += 2) {
        const __m256 values = _mm256_castsi256_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(channels + texel * 4)));
        const __m128i half = _mm256_cvtps_ph(
            values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgba16f + texel * 8), half);

        // CVTPS2PH quiets signaling NaNs, while the established scalar contract preserves the top
        // payload bits verbatim. That scalar also historically keeps an odd half exponent unchanged
        // when mantissa rounding carries into bit 10 (rather than incrementing it). Preserve both
        // observable behaviors by repairing only those rare lanes after the vector conversion.
        for (unsigned lane = 0; lane < 8; ++lane) {
            const uint32_t bits = channels[texel * 4 + lane];
            const uint32_t exponent = (bits >> 23) & 0xffu;
            const uint32_t mantissa = bits & 0x7fffffu;
            bool scalar_lane = exponent == 0xffu && mantissa != 0;
            const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
            if (!scalar_lane && half_exponent > 0 && half_exponent < 31 &&
                (half_exponent & 1) && (mantissa >> 13) == 0x3ffu) {
                const uint32_t remainder = mantissa & 0x1fffu;
                scalar_lane = remainder >= 0x1000u; // tie rounds up: 0x3ff is odd
            }
            if (!scalar_lane) continue;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            const uint16_t scalar = prosper::gpu::float_to_half(value);
            std::memcpy(rgba16f + texel * 8 + lane * sizeof(scalar), &scalar, sizeof(scalar));
        }
    }
    for (; texel < end; ++texel) {
        for (uint32_t channel = 0; channel < 4; ++channel) {
            float value;
            std::memcpy(&value, channels + texel * 4 + channel, sizeof(value));
            const uint16_t half = prosper::gpu::float_to_half(value);
            std::memcpy(rgba16f + texel * 8 + channel * sizeof(half), &half, sizeof(half));
        }
    }
}

__attribute__((target("avx2,f16c")))
void storage_unpack_float16x4_f16c(const uint8_t* rgba16f, size_t begin, size_t end,
                                   uint32_t* channels) {
    size_t texel = begin;
    const __m128i exponent_mask = _mm_set1_epi16(0x7c00);
    const __m128i mantissa_mask = _mm_set1_epi16(0x03ff);
    for (; texel + 2 <= end; texel += 2) {
        const __m128i half = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(rgba16f + texel * 8));
        const __m256 values = _mm256_cvtph_ps(half);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(channels + texel * 4),
                            _mm256_castps_si256(values));

        // CVTPH2PS quiets signaling NaNs; half_to_float intentionally preserves their payload bits.
        // Detect the uncommon NaN lanes as packed 16-bit values and repair only those through the
        // exhaustive table. Normal numbers, subnormals, infinities and quiet NaNs are already exact.
        const __m128i exponent = _mm_and_si128(half, exponent_mask);
        const __m128i mantissa = _mm_and_si128(half, mantissa_mask);
        const __m128i exponent_all_ones = _mm_cmpeq_epi16(exponent, exponent_mask);
        const __m128i mantissa_zero = _mm_cmpeq_epi16(mantissa, _mm_setzero_si128());
        const __m128i nan_lanes = _mm_andnot_si128(mantissa_zero, exponent_all_ones);
        if (_mm_movemask_epi8(nan_lanes)) {
            for (unsigned lane = 0; lane < 8; ++lane) {
                uint16_t bits = 0;
                std::memcpy(&bits, rgba16f + texel * 8 + lane * sizeof(bits), sizeof(bits));
                if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0)
                    channels[texel * 4 + lane] = storage_unpack_float16_bits(bits);
            }
        }
    }
    for (; texel < end; ++texel) {
        for (uint32_t channel = 0; channel < 4; ++channel) {
            uint16_t bits = 0;
            std::memcpy(&bits, rgba16f + texel * 8 + channel * sizeof(bits), sizeof(bits));
            channels[texel * 4 + channel] = storage_unpack_float16_bits(bits);
        }
    }
}
#endif

size_t linear_array_row_pitch(const prosper::gpu::ShaderResource& resource,
                              uint32_t bytes_per_texel) {
    const size_t tight = static_cast<size_t>(resource.width) * bytes_per_texel;
    const size_t pitch = resource.linear_row_pitch_bytes
        ? resource.linear_row_pitch_bytes
        : prosper::gpu::linear_sampled_row_pitch(resource.width, bytes_per_texel);
    return pitch >= tight ? pitch : 0;
}

size_t linear_array_surface_bytes(const prosper::gpu::ShaderResource& resource,
                                  uint32_t bytes_per_texel) {
    const size_t pitch = linear_array_row_pitch(resource, bytes_per_texel);
    if (!pitch || !resource.height || pitch > SIZE_MAX / resource.height) return 0;
    return pitch * resource.height;
}

// A non-depth, one-layer T# array has historically been lowered as an ordinary 2D SPIR-V image:
// its guest bytes are identical and several shaders use DIM=2D despite the descriptor type. Keep
// that established contract while materializing real layered/depth-array views as Vulkan arrays.
bool backend_uses_2d_array(const prosper::gpu::ShaderResource& resource) {
    return resource.img_dim == 5 &&
           (resource.depth_compare || resource.depth > 1 || resource.layer_stride_bytes != 0);
}

uint16_t storage_pack_unorm16(uint32_t float_bits) {
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));
    if (!(value > 0.0f)) return 0; // Includes negative values and NaN.
    if (value >= 1.0f) return UINT16_MAX;
    const float scaled = value * 65535.0f;
    const uint32_t whole = static_cast<uint32_t>(scaled);
    return static_cast<uint16_t>(whole + (scaled - static_cast<float>(whole) >= 0.5f));
}

template <typename T>
T storage_pack_snorm(uint32_t float_bits, int32_t positive_max) {
    float value;
    std::memcpy(&value, &float_bits, sizeof(value));
    if (std::isnan(value)) return 0;
    if (value <= -1.0f) return static_cast<T>(-positive_max);
    if (value >= 1.0f) return static_cast<T>(positive_max);
    const float scaled = value * static_cast<float>(positive_max);
    return static_cast<T>(static_cast<int32_t>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f)));
}
struct ComputeMemoryKey {
    VkDeviceSize bytes = 0;
    uint32_t memory_type = UINT32_MAX;
    bool operator==(const ComputeMemoryKey& other) const {
        return bytes == other.bytes && memory_type == other.memory_type;
    }
};

struct ComputeMemoryKeyHash {
    size_t operator()(const ComputeMemoryKey& key) const {
        return std::hash<uint64_t>{}(static_cast<uint64_t>(key.bytes)) ^
               (std::hash<uint32_t>{}(key.memory_type) << 1);
    }
};

struct ComputeMemoryPool {
    std::mutex mutex;
    std::unordered_map<ComputeMemoryKey, std::vector<VkDeviceMemory>, ComputeMemoryKeyHash> available;
    std::unordered_map<VkDeviceMemory, ComputeMemoryKey> active;
    std::unordered_map<VkDeviceMemory, void*> persistent_mappings;
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

struct ComputeMemoryPoolStats {
    VkDeviceSize cached_bytes = 0;
    size_t cached_allocations = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t discarded = 0;
};

struct ComputeBufferCacheKey {
    uint64_t gpu_addr = 0;
    uintptr_t host_data = 0;
    uint32_t bytes = 0;
    ComputeBufferMaterializationDiscriminator materialization;
    bool operator==(const ComputeBufferCacheKey& other) const {
        return gpu_addr == other.gpu_addr && host_data == other.host_data &&
               bytes == other.bytes && materialization == other.materialization;
    }
};

struct ComputeBufferCacheKeyHash {
    size_t operator()(const ComputeBufferCacheKey& key) const {
        size_t result = std::hash<uint64_t>{}(key.gpu_addr);
        result ^= std::hash<uintptr_t>{}(key.host_data) << 1;
        result ^= std::hash<uint32_t>{}(key.bytes) << 2;
        result ^= std::hash<uint64_t>{}(key.materialization.logical_bytes) << 3;
        result ^= std::hash<uint64_t>{}(key.materialization.binding_bytes) << 4;
        result ^= std::hash<uint32_t>{}(
            static_cast<uint32_t>(key.materialization.semantic)) << 5;
        return result;
    }
};

constexpr uint32_t kComputeBufferWriteWatchChunkBytes = 1u << 20;

struct ComputeBufferWriteWatchChunk {
    uint32_t offset = 0;
    uint32_t bytes = 0;
    prosper::host::GuestWriteWatch watch;
};

struct CachedComputeBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocation_bytes = 0;
    uint64_t last_use = 0;
    uint32_t pins = 0;
    bool content_valid = true;
    // GuestWriteWatch is page-granular internally, but its public query historically collapsed a
    // registration to one Dirty bit. A four-byte guest write could therefore make the persistent
    // compute cache compare an entire 32 MiB buffer. Split large sources into moderately-sized
    // registrations so an exact refresh only scans chunks containing a dirtied page. An unavailable
    // chunk watch remains fail-closed: acquire_cached_buffer falls back to the full byte comparison.
    std::vector<ComputeBufferWriteWatchChunk> write_watches;
    uint32_t write_watch_stable_validations = 0;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    // Exact previous writable result. Persistent buffers are host-visible because their ordinary
    // fallback must publish guest bytes, but mapping and comparing a 32 MiB result every frame is
    // still expensive. Keep a second storage buffer so the shared exact comparator can reduce an
    // unchanged result to one flag on the GPU. This is a byte-for-byte baseline, never a hash.
    VkBuffer result_buffer = VK_NULL_HANDLE;
    VkDeviceMemory result_memory = VK_NULL_HANDLE;
    VkDeviceSize result_bytes = 0;
};

struct ComputeImageCacheKey {
    uint64_t gpu_addr = 0;
    // Replay/capture resources preserve the architectural address for descriptor identity, but
    // expose their bytes through owned host storage. Keep that storage identity in the key just as
    // the persistent buffer cache does: two loaded captures may reuse a guest address while their
    // owned byte arrays have unrelated lifetimes.
    uintptr_t host_data = 0;
    uint32_t guest_bytes = 0;
    uint32_t resource_bytes = 0;
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t format = 0, components = 0, tile_mode = 0, img_dim = 0;
    uint32_t linear_row_pitch = 0;
    uint32_t layer_stride = 0, layer_mip_offset = 0;
    uint32_t mip_tail_offset = 0, mip_tail_bytes = 0;
    uint32_t mip_tail_x = 0, mip_tail_y = 0;
    uint32_t vk_format = 0;
    bool storage = false;
    bool in_mip_tail = false;
    bool srgb = false;
    bool depth_compare = false;

    bool operator==(const ComputeImageCacheKey& other) const = default;
};

struct ComputeImageCacheKeyHash {
    size_t operator()(const ComputeImageCacheKey& key) const {
        size_t result = std::hash<uint64_t>{}(key.gpu_addr);
        const auto mix = [&](uint64_t value) {
            result ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull +
                      (result << 6) + (result >> 2);
        };
        mix(key.host_data); mix(key.guest_bytes); mix(key.resource_bytes);
        mix(key.width); mix(key.height); mix(key.depth);
        mix(key.format); mix(key.components); mix(key.tile_mode); mix(key.img_dim);
        mix(key.linear_row_pitch); mix(key.layer_stride); mix(key.layer_mip_offset);
        mix(key.mip_tail_offset); mix(key.mip_tail_bytes);
        mix(key.mip_tail_x); mix(key.mip_tail_y); mix(key.vk_format);
        mix(key.storage); mix(key.in_mip_tail); mix(key.srgb); mix(key.depth_compare);
        return result;
    }
};

ComputeImageCacheKey storage_image_cache_key(const prosper::gpu::ShaderResource& resource,
                                              uint32_t guest_bytes,
                                              VkFormat native_format) {
    return {
        resource.gpu_addr, reinterpret_cast<uintptr_t>(resource.host_data),
        guest_bytes, resource.size,
        resource.width, resource.height, resource.depth,
        static_cast<uint32_t>(resource.format),
        resource.num_components ? resource.num_components : 1u,
        resource.tile_mode, resource.img_dim, resource.linear_row_pitch_bytes,
        resource.layer_stride_bytes, resource.layer_mip_offset_bytes,
        resource.mip_tail_offset, resource.mip_tail_bytes,
        resource.mip_tail_x, resource.mip_tail_y,
        static_cast<uint32_t>(native_format), true, resource.in_mip_tail,
        resource.srgb, resource.depth_compare};
}

struct CachedComputeImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocation_bytes = 0;
    uint64_t last_use = 0;
    uint32_t pins = 0;
    bool content_valid = true;
    // Several descriptors in one dispatch can name the same image with different sampler state.
    // They need distinct views/samplers but validate the same guest byte range. Remember the first
    // exact result for that setup epoch so later bindings do not rescan the complete surface.
    uint64_t validation_epoch = 0;
    bool validation_result = false;
    prosper::host::GuestWriteWatch write_watch;
    uint32_t write_watch_stable_validations = 0;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    // A successful typed-storage dispatch leaves `image` in GENERAL with the exact result. Graphics
    // may borrow it only while a current-submit journal or this watch proves that no later guest
    // write changed the architectural backing. This is separate from source validation: a failed
    // dispatch can leave the old guest mirror valid while making the Vulkan result unknowable.
    prosper::gpu::GuestGpuWriteSnapshot graphics_export_snapshot;
    bool graphics_export_valid = false;
    prosper::gpu::GuestGpuWriteSnapshot compute_transfer_snapshot;
    bool compute_transfer_valid = false;
    std::vector<uint8_t> source_snapshot;
    // Exact row-major bytes produced by the last storage dispatch. Full-overwrite post-processes
    // commonly reproduce the same large target on successive frames. Retaining the result lets a
    // later dispatch prove byte-for-byte that the guest mirror is already current before omitting
    // its CPU pack/retile/writeback. This is deliberately not a hash: a collision must never turn a
    // changed GPU result into a missed guest write or cache invalidation.
    std::vector<uint8_t> result_snapshot;
    // Exact GPU-side result baseline for native storage targets and proven-full raw write-only
    // targets. The next dispatch compares its transfer buffer to this one word-for-word and returns
    // a four-byte changed flag to the CPU.
    // Keeping the baseline as a buffer avoids rereading 30-70 MiB of host-visible staging memory
    // merely to discover that a full-screen post-process reproduced the previous frame exactly.
    VkBuffer result_buffer = VK_NULL_HANDLE;
    VkDeviceMemory result_memory = VK_NULL_HANDLE;
    VkDeviceSize result_allocation_bytes = 0;
    VkDeviceSize result_bytes = 0;
};

enum class ComputeTransferBorrowResult : uint8_t {
    NotAttempted,
    Hit,
    NoCache,
    InvalidCache,
    AuthorityChanged,
};

bool persistent_compute_buffer_enabled(uint32_t bytes) {
    static const bool enabled =
        std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_BUFFERS") == nullptr;
    // Small bindings are cheap and numerous. Residency targets the multi-megabyte SSBOs whose
    // create/bind/full-compare cycle is visible in every Astro Bot frame.
    return enabled && bytes >= (1u << 20);
}

bool persistent_compute_buffer_result_enabled(uint32_t bytes) {
    static const bool enabled =
        std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS") == nullptr;
    static const uint64_t minimum = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB");
        uint64_t mib = 16;
        if (value && *value) {
            char* end = nullptr;
            const uint64_t parsed = std::strtoull(value, &end, 10);
            if (end && !*end) mib = parsed;
        }
        return mib > UINT64_MAX / (1024ull * 1024ull)
            ? UINT64_MAX : mib * 1024ull * 1024ull;
    }();
    // Below 16 MiB the extra GPU dispatch is neutral on the measured integrated device; the exact
    // CPU comparison is cheaper and avoids doubling that cache entry's allocation. Keep the
    // crossover configurable for discrete GPUs and for compact production-backend tests.
    return enabled && bytes >= minimum;
}

VkDeviceSize persistent_compute_buffer_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = std::getenv("PROSPER_COMPUTE_BUFFER_CACHE_MB");
        const uint64_t mib = value ? std::strtoull(value, nullptr, 10) : 256ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

VkDeviceSize compute_image_cache_minimum(const char* value, VkDeviceSize fallback) {
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const uint64_t kib = std::strtoull(value, &end, 10);
    if (!end || *end) return fallback;
    if (kib > UINT64_MAX / 1024ull) return VkDeviceSize{UINT64_MAX};
    return static_cast<VkDeviceSize>(kib) * 1024ull;
}

bool persistent_compute_image_enabled(VkDeviceSize bytes,
                                      ComputeImageCacheClass image_class) {
    static const bool enabled =
        std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_IMAGES") == nullptr;
    static const VkDeviceSize sampled_minimum = [] {
        return compute_image_cache_minimum(
            std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB"),
            compute_image_cache_default_minimum_bytes(ComputeImageCacheClass::sampled));
    }();
    static const VkDeviceSize storage_minimum = [] {
        const char* value = std::getenv("PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB");
        // The historical generic override applied to both image roles. Preserve that contract when
        // no storage-specific value is supplied, while allowing the production defaults to diverge.
        if (!value || !*value)
            value = std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB");
        return compute_image_cache_minimum(
            value,
            compute_image_cache_default_minimum_bytes(ComputeImageCacheClass::storage));
    }();
    const VkDeviceSize minimum = image_class == ComputeImageCacheClass::storage
        ? storage_minimum : sampled_minimum;
    // Both thresholds are configurable so production-backend tests can exercise residency without
    // large allocations and device-specific profiling can retune either crossover independently.
    return enabled && bytes >= minimum;
}

VkDeviceSize persistent_compute_image_limit(
    const VkPhysicalDeviceMemoryProperties& memory) {
    static const std::optional<VkDeviceSize> override_limit = []()
        -> std::optional<VkDeviceSize> {
        const char* value = std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MB");
        if (!value) return std::nullopt;
        const uint64_t mib = std::strtoull(value, nullptr, 10);
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    if (override_limit) return *override_limit;

    VkDeviceSize largest_local_heap = 0;
    for (uint32_t index = 0; index < memory.memoryHeapCount; ++index) {
        if (memory.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            largest_local_heap = std::max(largest_local_heap,
                                          memory.memoryHeaps[index].size);
    }
    return compute_image_cache_default_limit_bytes(largest_local_heap);
}

uint32_t compute_write_watch_promotion_validations() {
    static const uint32_t value = [] {
        const char* text = std::getenv("PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS");
        const uint64_t parsed = text ? std::strtoull(text, nullptr, 10) : 3ull;
        return static_cast<uint32_t>(std::min<uint64_t>(parsed, UINT32_MAX));
    }();
    return value;
}

size_t compute_write_watch_promotion_budget_bytes() {
    static const size_t value = [] {
        const char* text = std::getenv("PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_MB");
        const uint64_t mib = text ? std::strtoull(text, nullptr, 10) : 8ull;
        return static_cast<size_t>(
            std::min<uint64_t>(mib, SIZE_MAX / (1024ull * 1024ull)) *
            (1024ull * 1024ull));
    }();
    return value;
}

bool adaptive_storage_result_validation_enabled() {
    static const bool enabled =
        std::getenv("PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION") == nullptr;
    return enabled;
}

bool native_3d_transfer_enabled() {
    static const bool enabled =
        std::getenv("PROSPER_NO_NATIVE_3D_COMPUTE_TRANSFER") == nullptr;
    return enabled;
}

bool native_2d_transfer_enabled() {
    static const bool enabled =
        std::getenv("PROSPER_NO_NATIVE_2D_COMPUTE_TRANSFER") == nullptr;
    return enabled;
}

size_t cold_storage_result_snapshot_defer_min_bytes() {
    if (g_zero_next_cold_storage_snapshot_minimum_for_test.exchange(
            false, std::memory_order_acq_rel))
        return 0;
    static const size_t bytes = [] {
        const char* value = std::getenv("PROSPER_COLD_STORAGE_SNAPSHOT_MIN_MB");
        char* end = nullptr;
        const uint64_t parsed = value ? std::strtoull(value, &end, 10) : 16ull;
        const uint64_t mib = value && (!end || *end) ? 16ull : parsed;
        return static_cast<size_t>(
            std::min<uint64_t>(mib, SIZE_MAX / (1024ull * 1024ull)) *
            (1024ull * 1024ull));
    }();
    return bytes;
}

struct CachedComputePipeline {
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

bool compute_memory_pool_enabled() {
    static const bool enabled = std::getenv("PROSPER_NO_MEMORY_POOL") == nullptr;
    return enabled;
}

VkDeviceSize compute_memory_pool_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = std::getenv("PROSPER_COMPUTE_MEMORY_POOL_MB");
        // Astro Bot's first 120 frames stabilize at about 614 MiB of differently-sized image and
        // staging allocations. A 256 MiB cache discarded about 200 allocations and repeatedly paid
        // AMD BO page initialization; 640 MiB holds that measured working set with zero discards,
        // while remaining a bounded fraction of the renderer's persistent-target budget.
        const uint64_t mib = value ? std::strtoull(value, nullptr, 10) : 640ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
}

struct VulkanComputeContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence dispatch_fence = VK_NULL_HANDLE;
    uint32_t descriptor_buffer_capacity = 0;
    uint32_t descriptor_sampled_capacity = 0;
    uint32_t descriptor_storage_capacity = 0;
    std::unordered_map<std::string, CachedComputePipeline> pipelines;
    uint32_t queue_family = UINT32_MAX;
    VkPhysicalDeviceMemoryProperties memory{};
    ComputeMemoryPool memory_pool;
    std::unordered_map<ComputeBufferCacheKey, CachedComputeBuffer,
                       ComputeBufferCacheKeyHash> buffer_cache;
    VkDeviceSize buffer_cache_bytes = 0;
    uint64_t buffer_cache_clock = 0;
    std::unordered_map<ComputeImageCacheKey, CachedComputeImage,
                       ComputeImageCacheKeyHash> image_cache;
    VkDeviceSize image_cache_bytes = 0;
    uint64_t image_cache_clock = 0;
    uint64_t image_validation_clock = 0;
    WriteWatchPromotionBudget write_watch_promotion_budget;
    uint64_t image_source_snapshot_copies = 0;
    uint64_t image_source_snapshot_bytes = 0;
    uint64_t storage_result_snapshot_copies = 0;
    uint64_t storage_result_snapshot_bytes = 0;
    uint64_t image_result_snapshot_copies = 0;
    uint64_t image_result_snapshot_bytes = 0;
    VkDescriptorSetLayout compare_descriptor_layout = VK_NULL_HANDLE;
    VkShaderModule compare_shader = VK_NULL_HANDLE;
    VkPipelineLayout compare_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compare_pipeline = VK_NULL_HANDLE;
    // Storage-image support (#590): the recompiler's storage path declares the
    // StorageImageRead/WriteWithoutFormat capabilities (raw uvec4 texel model — see
    // tests/image_compute_runner.h, the exec-diff harness for that contract). When the device lacks
    // the features, image-binding dispatches are skipped loudly instead of creating an invalid device.
    bool image_support = false;
    uint32_t subgroup_size = 0;
    VkShaderStageFlags subgroup_stages = 0;
    VkSubgroupFeatureFlags subgroup_operations = 0;
    std::array<uint32_t, 3> max_compute_workgroup_count{};
    // Exact-subgroup pipelines are valid only on an adopted renderer device that enabled the
    // published size-control/full-subgroup contract. Private fallback devices deliberately do not
    // enable that optional extension.
    bool native_subgroup_contract = false;
    uint32_t min_native_subgroup_size = 0, max_native_subgroup_size = 0;
    // True when instance/device/queue were ADOPTED from the live renderer (#1091). A borrowed
    // context is owned by the renderer: destroy our own pipelines/pools/memory, never its device.
    bool borrowed = false;
    // VK_ERROR_DEVICE_LOST is permanent for a VkDevice. Keep the first failure latched so later
    // PM4 dispatches cannot spend minutes rebuilding resources and submitting work that Vulkan is
    // required to reject. AGC submit execution serializes access to this context.
    bool device_lost = false;
    // A queue API was entered and no fence or queue-idle result proved completion. The current item
    // and this context then retain objects that Vulkan may still own; neither may be destroyed.
    bool completion_unproven = false;

    ~VulkanComputeContext() {
        release_cached_buffers();
        release_cached_images();
        release_cached_memory();
        if (dispatch_fence) vkDestroyFence(device, dispatch_fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        if (compare_pipeline) vkDestroyPipeline(device, compare_pipeline, nullptr);
        if (compare_pipeline_layout)
            vkDestroyPipelineLayout(device, compare_pipeline_layout, nullptr);
        if (compare_shader) vkDestroyShaderModule(device, compare_shader, nullptr);
        if (compare_descriptor_layout)
            vkDestroyDescriptorSetLayout(device, compare_descriptor_layout, nullptr);
        for (const auto& [key, cached] : pipelines) {
            (void)key;
            if (cached.pipeline) vkDestroyPipeline(device, cached.pipeline, nullptr);
            if (cached.pipeline_layout)
                vkDestroyPipelineLayout(device, cached.pipeline_layout, nullptr);
            if (cached.shader) vkDestroyShaderModule(device, cached.shader, nullptr);
            if (cached.descriptor_layout)
                vkDestroyDescriptorSetLayout(device, cached.descriptor_layout, nullptr);
        }
        if (pipeline_cache) vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        if (borrowed) return;                       // renderer owns the device/instance
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    static bool descriptor_pool_reuse_enabled() {
        static const bool enabled = std::getenv("PROSPER_NO_DESCRIPTOR_POOL_REUSE") == nullptr;
        return enabled;
    }

    bool prepare_dispatch_commands() {
        if (!command_pool) {
            VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool_info.queueFamilyIndex = queue_family;
            if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
                return false;
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = command_pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device, &allocate, &command_buffer) != VK_SUCCESS)
                return false;
        } else if (vkResetCommandPool(device, command_pool, 0) != VK_SUCCESS) {
            return false;
        }
        if (!dispatch_fence) {
            VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (vkCreateFence(device, &fence_info, nullptr, &dispatch_fence) != VK_SUCCESS)
                return false;
        } else if (vkResetFences(device, 1, &dispatch_fence) != VK_SUCCESS) {
            return false;
        }
        return command_buffer != VK_NULL_HANDLE;
    }

    VkDescriptorPool prepare_descriptor_pool(uint32_t buffers, uint32_t sampled,
                                             uint32_t storage) {
        if (!descriptor_pool_reuse_enabled()) return VK_NULL_HANDLE;
        if (descriptor_pool && buffers <= descriptor_buffer_capacity &&
            sampled <= descriptor_sampled_capacity && storage <= descriptor_storage_capacity) {
            if (vkResetDescriptorPool(device, descriptor_pool, 0) == VK_SUCCESS)
                return descriptor_pool;
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            descriptor_pool = VK_NULL_HANDLE;
        } else if (descriptor_pool) {
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            descriptor_pool = VK_NULL_HANDLE;
        }
        descriptor_buffer_capacity = std::max(buffers, descriptor_buffer_capacity);
        descriptor_sampled_capacity = std::max(sampled, descriptor_sampled_capacity);
        descriptor_storage_capacity = std::max(storage, descriptor_storage_capacity);
        VkDescriptorPoolSize sizes[3];
        uint32_t count = 0;
        if (descriptor_buffer_capacity)
            sizes[count++] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_buffer_capacity};
        if (descriptor_sampled_capacity)
            sizes[count++] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                              descriptor_sampled_capacity};
        if (descriptor_storage_capacity)
            sizes[count++] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptor_storage_capacity};
        VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        info.maxSets = 1;
        info.poolSizeCount = count;
        info.pPoolSizes = sizes;
        if (!count || vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS)
            descriptor_pool = VK_NULL_HANDLE;
        return descriptor_pool;
    }

    static bool persistent_mapping_enabled() {
        static const bool enabled =
            std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_MAP") == nullptr;
        return enabled;
    }

    size_t release_available_memory() {
        if (!device) return 0;
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        size_t released = 0;
        for (const auto& [key, allocations] : memory_pool.available) {
            (void)key;
            for (VkDeviceMemory allocation : allocations) {
                if (memory_pool.persistent_mappings.erase(allocation))
                    vkUnmapMemory(device, allocation);
                vkFreeMemory(device, allocation, nullptr);
                ++released;
            }
        }
        memory_pool.available.clear();
        memory_pool.cached_bytes = 0;
        memory_pool.cached_allocations = 0;
        memory_pool.discarded += released;
        return released;
    }

    VkDeviceMemory allocate_memory(VkDeviceSize bytes, uint32_t memory_type,
                                   bool persistently_map = false) {
        if (memory_type == UINT32_MAX) return VK_NULL_HANDLE;
        const ComputeMemoryKey key{bytes, memory_type};
        static const bool best_fit_reuse =
            std::getenv("PROSPER_COMPUTE_MEMORY_POOL_EXACT") == nullptr;
        if (compute_memory_pool_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            auto found = memory_pool.available.find(key);
            // Vulkan permits binding an allocation larger than the resource's memory requirement.
            // Exact-size-only reuse fragmented Astro Bot's 256 MiB pool across many nearly-identical
            // image/staging sizes, forcing hundreds of fresh AMD BO allocations and kernel page-zero
            // passes. On an exact miss, reuse the smallest available allocation of the same memory
            // type that is large enough. Keep its REAL size in active so release/accounting remain
            // exact and a large allocation is never accidentally treated as a smaller one.
            if ((found == memory_pool.available.end() || found->second.empty()) &&
                best_fit_reuse) {
                auto best = memory_pool.available.end();
                for (auto candidate = memory_pool.available.begin();
                     candidate != memory_pool.available.end(); ++candidate) {
                    if (candidate->second.empty() || candidate->first.memory_type != memory_type ||
                        candidate->first.bytes < bytes)
                        continue;
                    if (best == memory_pool.available.end() ||
                        candidate->first.bytes < best->first.bytes)
                        best = candidate;
                }
                found = best;
            }
            if (found != memory_pool.available.end() && !found->second.empty()) {
                const ComputeMemoryKey allocation_key = found->first;
                const VkDeviceMemory allocation = found->second.back();
                found->second.pop_back();
                if (found->second.empty()) memory_pool.available.erase(found);
                memory_pool.cached_bytes -= allocation_key.bytes;
                --memory_pool.cached_allocations;
                ++memory_pool.hits;
                memory_pool.active.emplace(allocation, allocation_key);
                if (persistently_map && persistent_mapping_enabled() &&
                    memory_pool.persistent_mappings.find(allocation) ==
                        memory_pool.persistent_mappings.end()) {
                    void* mapping = nullptr;
                    if (vkMapMemory(device, allocation, 0, allocation_key.bytes, 0, &mapping) ==
                        VK_SUCCESS)
                        memory_pool.persistent_mappings.emplace(allocation, mapping);
                }
                return allocation;
            }
            ++memory_pool.misses;
        }

        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = bytes;
        allocation.memoryTypeIndex = memory_type;
        VkDeviceMemory result = VK_NULL_HANDLE;
        VkResult allocation_result = vkAllocateMemory(device, &allocation, nullptr, &result);
        if (allocation_result != VK_SUCCESS && compute_memory_pool_enabled()) {
            // Cached allocations are expendable. Under real heap pressure, release them before
            // propagating OOM so an enlarged reuse cache can never strand memory needed now.
            const size_t released = release_available_memory();
            if (released) {
                fprintf(stderr,
                        "[compute] allocation failed (%d); evicted %zu cached allocation(s) and retrying\n",
                        static_cast<int>(allocation_result), released);
                allocation_result = vkAllocateMemory(device, &allocation, nullptr, &result);
            }
        }
        if (allocation_result != VK_SUCCESS) return VK_NULL_HANDLE;
        if (compute_memory_pool_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            memory_pool.active.emplace(result, key);
            if (persistently_map && persistent_mapping_enabled()) {
                void* mapping = nullptr;
                if (vkMapMemory(device, result, 0, bytes, 0, &mapping) == VK_SUCCESS)
                    memory_pool.persistent_mappings.emplace(result, mapping);
            }
        }
        return result;
    }

    VkResult map_memory(VkDeviceMemory allocation, VkDeviceSize offset, VkDeviceSize bytes,
                        void** mapping) {
        if (compute_memory_pool_enabled() && persistent_mapping_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            const auto found = memory_pool.persistent_mappings.find(allocation);
            if (found != memory_pool.persistent_mappings.end()) {
                *mapping = static_cast<uint8_t*>(found->second) + offset;
                return VK_SUCCESS;
            }
        }
        return vkMapMemory(device, allocation, offset, bytes, 0, mapping);
    }

    void unmap_memory(VkDeviceMemory allocation) {
        if (compute_memory_pool_enabled() && persistent_mapping_enabled()) {
            std::lock_guard<std::mutex> lock(memory_pool.mutex);
            if (memory_pool.persistent_mappings.find(allocation) !=
                memory_pool.persistent_mappings.end())
                return;
        }
        vkUnmapMemory(device, allocation);
    }

    void release_memory(VkDeviceMemory allocation) {
        if (!allocation) return;
        if (!compute_memory_pool_enabled()) {
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        auto found = memory_pool.active.find(allocation);
        if (found == memory_pool.active.end()) {
            if (memory_pool.persistent_mappings.erase(allocation))
                vkUnmapMemory(device, allocation);
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        const ComputeMemoryKey key = found->second;
        memory_pool.active.erase(found);
        constexpr size_t max_cached_allocations = 2048;
        const VkDeviceSize limit = compute_memory_pool_limit();
        const VkDeviceSize remaining = memory_pool.cached_bytes < limit
            ? limit - memory_pool.cached_bytes : 0;
        if (memory_pool.cached_allocations >= max_cached_allocations || key.bytes > remaining) {
            ++memory_pool.discarded;
            if (memory_pool.persistent_mappings.erase(allocation))
                vkUnmapMemory(device, allocation);
            vkFreeMemory(device, allocation, nullptr);
            return;
        }
        memory_pool.available[key].push_back(allocation);
        memory_pool.cached_bytes += key.bytes;
        ++memory_pool.cached_allocations;
    }

    ComputeMemoryPoolStats memory_pool_stats() {
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        return {memory_pool.cached_bytes, memory_pool.cached_allocations, memory_pool.hits,
                memory_pool.misses, memory_pool.discarded};
    }

    void release_cached_memory() {
        if (!device) return;
        std::lock_guard<std::mutex> lock(memory_pool.mutex);
        for (const auto& [key, allocations] : memory_pool.available) {
            (void)key;
            for (VkDeviceMemory allocation : allocations) {
                if (memory_pool.persistent_mappings.erase(allocation))
                    vkUnmapMemory(device, allocation);
                vkFreeMemory(device, allocation, nullptr);
            }
        }
        for (const auto& [allocation, key] : memory_pool.active) {
            (void)key;
            if (memory_pool.persistent_mappings.erase(allocation))
                vkUnmapMemory(device, allocation);
            vkFreeMemory(device, allocation, nullptr);
        }
        memory_pool.available.clear();
        memory_pool.active.clear();
        memory_pool.persistent_mappings.clear();
        memory_pool.cached_bytes = 0;
        memory_pool.cached_allocations = 0;
    }

    void release_cached_buffers() {
        if (!device) return;
        for (auto& [key, cached] : buffer_cache) {
            (void)key;
            cached.write_watches.clear();
            if (cached.buffer) vkDestroyBuffer(device, cached.buffer, nullptr);
            if (cached.memory) release_memory(cached.memory);
            if (cached.result_buffer) vkDestroyBuffer(device, cached.result_buffer, nullptr);
            if (cached.result_memory) release_memory(cached.result_memory);
        }
        buffer_cache.clear();
        buffer_cache_bytes = 0;
    }

    void release_cached_images() {
        if (!device) return;
        for (auto& [key, cached] : image_cache) {
            (void)key;
            destroy_cached_image_resources(cached);
        }
        image_cache.clear();
        image_cache_bytes = 0;
    }

    void begin_write_watch_promotions() {
        write_watch_promotion_budget.reset(compute_write_watch_promotion_budget_bytes());
    }

    bool may_promote_write_watch_before_exact(size_t source_bytes,
                                              uint32_t stable_validations) {
        const uint32_t promotion_validations =
            compute_write_watch_promotion_validations();
        const uint32_t prospective_stability = update_write_watch_stability(
            stable_validations, true, promotion_validations);
        return should_promote_write_watch(
                   source_bytes, prospective_stability, 1,
                   promotion_validations) &&
               write_watch_promotion_budget.try_consume(source_bytes);
    }

    void remember_image_source_snapshot(CachedComputeImage& cached,
                                        const uint8_t* source, size_t bytes,
                                        bool storage_result = false) {
        if (!source) return;
        cached.source_snapshot.assign(source, source + bytes);
        ++image_source_snapshot_copies;
        image_source_snapshot_bytes += bytes;
        if (storage_result) {
            ++storage_result_snapshot_copies;
            storage_result_snapshot_bytes += bytes;
        }
    }

    bool make_buffer_cache_room(VkDeviceSize bytes) {
        const VkDeviceSize limit = persistent_compute_buffer_limit();
        if (bytes > limit) return false;
        while (buffer_cache_bytes > limit - bytes) {
            auto victim = buffer_cache.end();
            for (auto it = buffer_cache.begin(); it != buffer_cache.end(); ++it) {
                if (it->second.pins) continue;
                if (victim == buffer_cache.end() ||
                    it->second.last_use < victim->second.last_use)
                    victim = it;
            }
            if (victim == buffer_cache.end()) return false;
            victim->second.write_watches.clear();
            if (victim->second.buffer) vkDestroyBuffer(device, victim->second.buffer, nullptr);
            if (victim->second.memory) release_memory(victim->second.memory);
            if (victim->second.result_buffer)
                vkDestroyBuffer(device, victim->second.result_buffer, nullptr);
            if (victim->second.result_memory)
                release_memory(victim->second.result_memory);
            buffer_cache_bytes -= victim->second.allocation_bytes;
            buffer_cache.erase(victim);
        }
        return true;
    }

    bool acquire_cached_buffer(const ComputeBufferCacheKey& key, const uint8_t* source,
                               VkBuffer& buffer, VkDeviceMemory& memory, bool& upload_skipped,
                               uint32_t& dirty_watch_chunks, uint32_t& total_watch_chunks) {
        auto found = buffer_cache.find(key);
        if (found == buffer_cache.end()) return false;
        CachedComputeBuffer& cached = found->second;
        cached.last_use = ++buffer_cache_clock;
        ++cached.pins;
        buffer = cached.buffer;
        memory = cached.memory;
        const bool submit_unchanged = cached.content_valid && !key.host_data &&
            prosper::gpu::guest_gpu_writes_since(cached.validation_snapshot,
                                                  key.gpu_addr, key.bytes) ==
                prosper::gpu::GuestGpuWriteQuery::Unchanged;
        std::vector<size_t> dirty_chunks;
        bool watches_complete = cached.content_valid && !cached.write_watches.empty();
        if (!submit_unchanged) {
            for (size_t i = 0; i < cached.write_watches.size(); ++i) {
                ++total_watch_chunks;
                const auto query = cached.write_watches[i].watch.query();
                if (query == prosper::host::GuestWriteWatchQuery::Unknown) {
                    watches_complete = false;
                } else if (query == prosper::host::GuestWriteWatchQuery::Dirty) {
                    dirty_chunks.push_back(i);
                }
            }
        }
        dirty_watch_chunks = static_cast<uint32_t>(dirty_chunks.size());
        upload_skipped = submit_unchanged || (watches_complete && dirty_chunks.empty());
        if (!upload_skipped) {
            // Establish the mutation boundary before the authoritative guest-byte comparison.
            // Arming after memcmp would leave a compare-to-arm gap where a concurrent guest CPU
            // write could become permanently invisible to this cache entry.
            prepare_cached_buffer_write_watches_before_exact(key);
            void* mapped = nullptr;
            if (map_memory(cached.memory, 0, key.bytes, &mapped) != VK_SUCCESS) {
                --cached.pins;
                buffer = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                return false;
            }
            if (watches_complete) {
                bool changed = false;
                auto* destination = static_cast<uint8_t*>(mapped);
                for (size_t index : dirty_chunks) {
                    const ComputeBufferWriteWatchChunk& chunk = cached.write_watches[index];
                    if (std::memcmp(destination + chunk.offset, source + chunk.offset,
                                    chunk.bytes) == 0)
                        continue;
                    std::memcpy(destination + chunk.offset, source + chunk.offset, chunk.bytes);
                    changed = true;
                }
                upload_skipped = !changed;
            } else {
                const bool changed = !compute_buffers_equal(mapped, source, key.bytes);
                if (changed) copy_compute_buffer(mapped, source, key.bytes);
                upload_skipped = !changed;
            }
            unmap_memory(cached.memory);
            cached.content_valid = true;
            if (!key.host_data)
                validate_cached_buffer_source(key, upload_skipped, true);
        }
        return true;
    }

    bool retain_buffer(const ComputeBufferCacheKey& key, VkBuffer buffer, VkDeviceMemory memory,
                       VkDeviceSize allocation_bytes) {
        if (!make_buffer_cache_room(allocation_bytes)) return false;
        CachedComputeBuffer cached;
        cached.buffer = buffer;
        cached.memory = memory;
        cached.allocation_bytes = allocation_bytes;
        cached.last_use = ++buffer_cache_clock;
        cached.pins = 1;
        if (!key.host_data)
            cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        auto [it, inserted] = buffer_cache.emplace(key, std::move(cached));
        if (!inserted) return false;
        buffer_cache_bytes += allocation_bytes;
        return true;
    }

    void release_cached_buffer(const ComputeBufferCacheKey& key) {
        auto found = buffer_cache.find(key);
        if (found != buffer_cache.end() && found->second.pins) --found->second.pins;
    }

    void prepare_cached_buffer_write_watches_before_exact(
        const ComputeBufferCacheKey& key) {
        auto found = buffer_cache.find(key);
        if (found == buffer_cache.end() || key.host_data) return;
        CachedComputeBuffer& cached = found->second;
        if (cached.write_watches.empty()) {
            if (!may_promote_write_watch_before_exact(
                    key.bytes, cached.write_watch_stable_validations))
                return;
            for (uint32_t offset = 0; offset < key.bytes;) {
                const uint32_t bytes = std::min(kComputeBufferWriteWatchChunkBytes,
                                                key.bytes - offset);
                cached.write_watches.push_back({
                    offset, bytes,
                    prosper::host::GuestWriteWatch::create(key.gpu_addr + offset, bytes)});
                offset += bytes;
            }
            return;
        }
        for (ComputeBufferWriteWatchChunk& chunk : cached.write_watches) {
            if (chunk.watch && chunk.watch.rearm()) continue;
            chunk.watch.reset();
            chunk.watch = prosper::host::GuestWriteWatch::create(
                key.gpu_addr + chunk.offset, chunk.bytes);
        }
    }

    void validate_cached_buffer_source(const ComputeBufferCacheKey& key,
                                       bool content_unchanged = false,
                                       bool watch_prepared_before_validation = false) {
        auto found = buffer_cache.find(key);
        if (found == buffer_cache.end() || key.host_data) return;
        CachedComputeBuffer& cached = found->second;
        cached.content_valid = true;
        cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        cached.write_watch_stable_validations = update_write_watch_stability(
            cached.write_watch_stable_validations, content_unchanged,
            compute_write_watch_promotion_validations());
        if (watch_prepared_before_validation || cached.write_watches.empty()) return;
        for (ComputeBufferWriteWatchChunk& chunk : cached.write_watches) {
            if (chunk.watch && chunk.watch.rearm()) continue;
            chunk.watch.reset();
            chunk.watch = prosper::host::GuestWriteWatch::create(
                key.gpu_addr + chunk.offset, chunk.bytes);
        }
    }

    void invalidate_cached_buffer_source(const ComputeBufferCacheKey& key) {
        const auto found = buffer_cache.find(key);
        if (found != buffer_cache.end()) found->second.content_valid = false;
    }

    bool cached_buffer_result_buffer(const ComputeBufferCacheKey& key, VkDeviceSize bytes,
                                     VkBuffer& result) const {
        const auto found = buffer_cache.find(key);
        if (found == buffer_cache.end() || found->second.result_bytes != bytes ||
            !found->second.result_buffer)
            return false;
        result = found->second.result_buffer;
        return true;
    }

    bool retain_cached_buffer_result(const ComputeBufferCacheKey& key,
                                     const uint8_t* result) {
        if (!persistent_compute_buffer_result_enabled(key.bytes)) return false;
        if (!result || !key.bytes || (key.bytes & 15u)) return false;
        auto found = buffer_cache.find(key);
        if (found == buffer_cache.end() || found->second.result_buffer) return false;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = key.bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(device, &bci, nullptr, &buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        if (!make_buffer_cache_room(requirements.size)) {
            vkDestroyBuffer(device, buffer, nullptr);
            return false;
        }
        const uint32_t memory_type = host_memory_type(requirements.memoryTypeBits);
        VkDeviceMemory memory = allocate_memory(requirements.size, memory_type, true);
        if (!memory || vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
            if (memory) release_memory(memory);
            vkDestroyBuffer(device, buffer, nullptr);
            return false;
        }
        void* mapped = nullptr;
        if (map_memory(memory, 0, key.bytes, &mapped) != VK_SUCCESS) {
            release_memory(memory);
            vkDestroyBuffer(device, buffer, nullptr);
            return false;
        }
        copy_compute_buffer(mapped, result, key.bytes);
        unmap_memory(memory);

        // make_buffer_cache_room can evict other entries, so reacquire the pinned current entry
        // before publishing ownership of the allocation.
        found = buffer_cache.find(key);
        if (found == buffer_cache.end() || found->second.result_buffer) {
            release_memory(memory);
            vkDestroyBuffer(device, buffer, nullptr);
            return false;
        }
        found->second.result_buffer = buffer;
        found->second.result_memory = memory;
        found->second.result_bytes = key.bytes;
        found->second.allocation_bytes += requirements.size;
        buffer_cache_bytes += requirements.size;
        return true;
    }

    void destroy_cached_image_resources(CachedComputeImage& cached) {
        cached.write_watch.reset();
        if (cached.image) vkDestroyImage(device, cached.image, nullptr);
        if (cached.memory) release_memory(cached.memory);
        if (cached.result_buffer) vkDestroyBuffer(device, cached.result_buffer, nullptr);
        if (cached.result_memory) release_memory(cached.result_memory);
        cached.image = VK_NULL_HANDLE;
        cached.memory = VK_NULL_HANDLE;
        cached.result_buffer = VK_NULL_HANDLE;
        cached.result_memory = VK_NULL_HANDLE;
    }

    bool make_image_cache_room(VkDeviceSize bytes) {
        const VkDeviceSize limit = persistent_compute_image_limit(memory);
        if (bytes > limit) return false;
        while (image_cache_bytes > limit - bytes) {
            auto victim = image_cache.end();
            for (auto it = image_cache.begin(); it != image_cache.end(); ++it) {
                if (it->second.pins) continue;
                if (victim == image_cache.end() ||
                    it->second.last_use < victim->second.last_use)
                    victim = it;
            }
            if (victim == image_cache.end()) return false;
            destroy_cached_image_resources(victim->second);
            image_cache_bytes -= victim->second.allocation_bytes;
            image_cache.erase(victim);
        }
        return true;
    }

    bool acquire_cached_image(const ComputeImageCacheKey& key, const uint8_t* source,
                              uint64_t validation_epoch, VkImage& image,
                              VkDeviceMemory& memory, bool& upload_skipped,
                              bool source_snapshot_required = true) {
        auto found = image_cache.find(key);
        if (found == image_cache.end()) return false;
        CachedComputeImage& cached = found->second;
        cached.last_use = ++image_cache_clock;
        ++cached.pins;
        image = cached.image;
        memory = cached.memory;
        if (!key.host_data && validation_epoch && cached.validation_epoch == validation_epoch) {
            upload_skipped = cached.validation_result;
            return true;
        }
        // Capture-owned bytes are not architectural guest mappings: direct writes to host_data do
        // not enter the submit journal and cannot be protected by GuestWriteWatch. Validate those
        // entries only against their exact retained snapshot. This makes warm replay exercise the
        // same image residency as live execution without ever trusting an unrelated guest address.
        const bool exact_source_only = key.host_data != 0;
        const bool submit_unchanged = cached.content_valid && !exact_source_only &&
            prosper::gpu::guest_gpu_writes_since(cached.validation_snapshot,
                                                  key.gpu_addr, key.guest_bytes) ==
                prosper::gpu::GuestGpuWriteQuery::Unchanged;
        const prosper::host::GuestWriteWatchQuery watch_query =
            !exact_source_only && !submit_unchanged && cached.content_valid && cached.write_watch
                ? cached.write_watch.query()
                : prosper::host::GuestWriteWatchQuery::Unknown;
        const bool watch_unchanged = !submit_unchanged &&
            watch_query == prosper::host::GuestWriteWatchQuery::Unchanged;
        if (!exact_source_only && cached.content_valid && !submit_unchanged &&
            !watch_unchanged && source) {
            // Rearm a dirtied watch, or promote a repeatedly stable exact source, before memcmp.
            // A write racing the comparison then remains Dirty for the next acquisition instead of
            // being erased by a post-comparison rearm.
            if (cached.write_watch && !cached.write_watch.rearm())
                cached.write_watch.reset();
            if (!cached.write_watch && may_promote_write_watch_before_exact(
                    key.guest_bytes, cached.write_watch_stable_validations)) {
                cached.write_watch = prosper::host::GuestWriteWatch::create(
                    key.gpu_addr, key.guest_bytes);
            }
        }
        const bool exact_unchanged = cached.content_valid && !submit_unchanged &&
            !watch_unchanged && source &&
            cached.source_snapshot.size() == key.guest_bytes &&
            std::memcmp(cached.source_snapshot.data(), source, key.guest_bytes) == 0;
        upload_skipped = submit_unchanged || watch_unchanged || exact_unchanged;
        cached.validation_epoch = validation_epoch;
        cached.validation_result = upload_skipped;
        if (exact_unchanged) {
            if (!exact_source_only) {
                cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
                cached.write_watch_stable_validations = update_write_watch_stability(
                    cached.write_watch_stable_validations, true,
                    compute_write_watch_promotion_validations());
            }
        } else if (!upload_skipped && source && source_snapshot_required) {
            cached.write_watch_stable_validations = 0;
            remember_image_source_snapshot(
                cached, source, key.guest_bytes, key.storage);
            // Do not trust the new mirror until the corresponding transfer completes. A failed
            // submit leaves this false, so the next use refreshes instead of skipping stale pixels.
            cached.content_valid = false;
            cached.graphics_export_valid = false;
            cached.compute_transfer_valid = false;
        }
        return true;
    }

    // DCC-compressed guest bytes cannot authorize an upload skip, but an exact unpinned cache entry
    // still owns a perfectly compatible Vulkan allocation. Lease that allocation before preparing
    // the mandatory seed so a changing producer does not destroy and recreate tens of MiB after
    // every writeback. Invalidate all source/export authority immediately: only this dispatch's
    // forced upload plus successful DCC publication may authorize the reused allocation again.
    bool acquire_cached_image_allocation_for_forced_seed(
        const ComputeImageCacheKey& key, VkImage& image, VkDeviceMemory& memory,
        VkDeviceSize& allocation_bytes) {
        auto found = image_cache.find(key);
        if (found == image_cache.end() || found->second.pins || !found->second.image ||
            !found->second.memory)
            return false;
        if (g_disable_next_dcc_allocation_reuse_for_test.exchange(
                false, std::memory_order_acq_rel))
            return false;
        CachedComputeImage& cached = found->second;
        cached.last_use = ++image_cache_clock;
        ++cached.pins;
        image = cached.image;
        memory = cached.memory;
        allocation_bytes = cached.allocation_bytes;
        // A zero-sized snapshot carries no authority, but retaining its capacity lets successful
        // writeback refresh the exact bytes without a second large allocation in the hot path.
        invalidate_cached_image_source(key, true);
        g_dcc_forced_seed_allocation_reuses.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void invalidate_cached_image_export(const ComputeImageCacheKey& key) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        found->second.graphics_export_valid = false;
        found->second.compute_transfer_valid = false;
    }

    void authorize_cached_image_export(const ComputeImageCacheKey& key) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end() || !found->second.content_valid || !found->second.image)
            return;
        found->second.graphics_export_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        found->second.graphics_export_valid = true;
    }

    bool authorize_cached_image_compute_transfer(const ComputeImageCacheKey& key) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end() || !found->second.content_valid || !found->second.image)
            return false;
        found->second.compute_transfer_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        found->second.compute_transfer_valid = true;
        return true;
    }

    bool borrow_cached_image_for_graphics(const ComputeImageCacheKey& key,
                                          VkImage& image) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return false;
        CachedComputeImage& cached = found->second;
        if (!cached.graphics_export_valid || !cached.content_valid || !cached.image) return false;
        const bool submit_unchanged =
            prosper::gpu::guest_gpu_writes_since(cached.graphics_export_snapshot,
                                                  key.gpu_addr, key.guest_bytes) ==
            prosper::gpu::GuestGpuWriteQuery::Unchanged;
        const bool watch_unchanged = !submit_unchanged && cached.write_watch &&
            cached.write_watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged;
        if (!submit_unchanged && !watch_unchanged) return false;
        cached.last_use = ++image_cache_clock;
        ++cached.pins;
        image = cached.image;
        return true;
    }

    bool borrow_cached_image_for_compute_transfer(const ComputeImageCacheKey& key,
                                                  VkImage& image, bool trace = false,
                                                  ComputeTransferBorrowResult* result = nullptr) {
        if (result) *result = ComputeTransferBorrowResult::NotAttempted;
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) {
            if (result) *result = ComputeTransferBorrowResult::NoCache;
            if (trace)
                std::fprintf(stderr,
                             "[compute]   native storage transfer miss: no storage cache\n");
            return false;
        }
        CachedComputeImage& cached = found->second;
        if (!cached.compute_transfer_valid || !cached.content_valid || !cached.image) {
            if (result) *result = ComputeTransferBorrowResult::InvalidCache;
            if (trace)
                std::fprintf(stderr,
                             "[compute]   native storage transfer miss: "
                             "export=%u content=%u image=%u\n",
                             cached.compute_transfer_valid ? 1u : 0u,
                             cached.content_valid ? 1u : 0u, cached.image ? 1u : 0u);
            return false;
        }
        const prosper::gpu::GuestGpuWriteQuery submit_query =
            prosper::gpu::guest_gpu_writes_since(cached.compute_transfer_snapshot,
                                                  key.gpu_addr, key.guest_bytes);
        const bool submit_unchanged =
            submit_query == prosper::gpu::GuestGpuWriteQuery::Unchanged;
        const bool watch_unchanged = !submit_unchanged && cached.write_watch &&
            cached.write_watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged;
        if (!submit_unchanged && !watch_unchanged) {
            if (result) *result = ComputeTransferBorrowResult::AuthorityChanged;
            if (trace)
                std::fprintf(stderr,
                             "[compute]   native storage transfer miss: "
                             "submit-query=%u watch=%u\n",
                             static_cast<unsigned>(submit_query),
                             cached.write_watch ? 1u : 0u);
            return false;
        }
        cached.last_use = ++image_cache_clock;
        ++cached.pins;
        image = cached.image;
        if (result) *result = ComputeTransferBorrowResult::Hit;
        return true;
    }

    bool retain_image(const ComputeImageCacheKey& key, VkImage image, VkDeviceMemory memory,
                      VkDeviceSize allocation_bytes,
                      std::vector<uint8_t>&& prepared_source_snapshot) {
        if (!prepared_source_snapshot.empty() &&
            prepared_source_snapshot.size() != key.guest_bytes)
            return false;
        if (!make_image_cache_room(allocation_bytes)) return false;
        CachedComputeImage cached;
        cached.image = image;
        cached.memory = memory;
        cached.allocation_bytes = allocation_bytes;
        cached.last_use = ++image_cache_clock;
        cached.pins = 1;
        if (!key.host_data)
            cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        if (!prepared_source_snapshot.empty()) {
            cached.source_snapshot = std::move(prepared_source_snapshot);
            ++image_source_snapshot_copies;
            image_source_snapshot_bytes += key.guest_bytes;
            if (key.storage) {
                ++storage_result_snapshot_copies;
                storage_result_snapshot_bytes += key.guest_bytes;
            }
        }
        auto [it, inserted] = image_cache.emplace(key, std::move(cached));
        if (!inserted) return false;
        image_cache_bytes += allocation_bytes;
        return true;
    }

    bool retain_image(const ComputeImageCacheKey& key, VkImage image, VkDeviceMemory memory,
                      VkDeviceSize allocation_bytes, const uint8_t* source) {
        if (!make_image_cache_room(allocation_bytes)) return false;
        CachedComputeImage cached;
        cached.image = image;
        cached.memory = memory;
        cached.allocation_bytes = allocation_bytes;
        cached.last_use = ++image_cache_clock;
        cached.pins = 1;
        if (!key.host_data)
            cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        if (source)
            remember_image_source_snapshot(
                cached, source, key.guest_bytes, key.storage);
        auto [it, inserted] = image_cache.emplace(key, std::move(cached));
        if (!inserted) return false;
        image_cache_bytes += allocation_bytes;
        return true;
    }

    // A compressed producer can follow a consumer that already retained the same exact storage
    // identity.  The producer's transient image must replace that stale entry after successful DCC
    // writeback; plain emplace would decline on the collision and forfeit the producer-to-consumer
    // device-local handoff.  Never disturb a pinned entry.  Preflight all required eviction before
    // mutating the cache so capacity failure preserves both the old cache authority and the caller's
    // transient fallback.
    bool replace_or_retain_image(const ComputeImageCacheKey& key,
                                 VkImage image, VkDeviceMemory memory,
                                 VkDeviceSize allocation_bytes, const uint8_t* source) {
        auto exact = image_cache.find(key);
        if (exact == image_cache.end())
            return retain_image(key, image, memory, allocation_bytes, source);
        if (exact->second.pins) return false;

        VkDeviceSize limit = persistent_compute_image_limit(this->memory);
        // Exercise the real capacity preflight, rather than bypassing this function at its caller.
        // The exact entry must remain untouched when the incoming allocation cannot fit.
        if (g_limit_next_image_replacement_for_test.exchange(
                false, std::memory_order_acq_rel))
            limit = allocation_bytes ? allocation_bytes - 1u : 0u;
        if (allocation_bytes > limit || exact->second.allocation_bytes > image_cache_bytes)
            return false;
        const VkDeviceSize retained_without_exact =
            image_cache_bytes - exact->second.allocation_bytes;
        const VkDeviceSize allowed_without_exact = limit - allocation_bytes;
        if (retained_without_exact > allowed_without_exact) {
            VkDeviceSize reclaimable = 0;
            for (const auto& [candidate_key, cached] : image_cache) {
                if (candidate_key == key || cached.pins) continue;
                if (cached.allocation_bytes >=
                    retained_without_exact - allowed_without_exact - reclaimable) {
                    reclaimable = retained_without_exact - allowed_without_exact;
                    break;
                }
                reclaimable += cached.allocation_bytes;
            }
            if (reclaimable < retained_without_exact - allowed_without_exact)
                return false;
        }

        CachedComputeImage replacement;
        replacement.image = image;
        replacement.memory = memory;
        replacement.allocation_bytes = allocation_bytes;
        replacement.last_use = ++image_cache_clock;
        replacement.pins = 1;
        if (!key.host_data)
            replacement.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        if (source)
            remember_image_source_snapshot(
                replacement, source, key.guest_bytes, key.storage);

        // Capacity was preflighted while the exact entry remained protected.  Evict only the
        // required other entries; the exact entry's bytes are replaced, not added a second time.
        while (image_cache_bytes - exact->second.allocation_bytes > allowed_without_exact) {
            auto victim = image_cache.end();
            for (auto it = image_cache.begin(); it != image_cache.end(); ++it) {
                if (it->first == key || it->second.pins) continue;
                if (victim == image_cache.end() ||
                    it->second.last_use < victim->second.last_use)
                    victim = it;
            }
            if (victim == image_cache.end()) return false;
            destroy_cached_image_resources(victim->second);
            image_cache_bytes -= victim->second.allocation_bytes;
            image_cache.erase(victim);
        }
        exact = image_cache.find(key);
        if (exact == image_cache.end() || exact->second.pins) return false;
        image_cache_bytes -= exact->second.allocation_bytes;
        destroy_cached_image_resources(exact->second);
        exact->second = std::move(replacement);
        image_cache_bytes += allocation_bytes;
        g_dcc_post_writeback_replacements.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void release_cached_image(const ComputeImageCacheKey& key) {
        auto found = image_cache.find(key);
        if (found != image_cache.end() && found->second.pins) --found->second.pins;
    }

    void invalidate_cached_image_source(const ComputeImageCacheKey& key,
                                        bool preserve_snapshot_capacity = false) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        CachedComputeImage& cached = found->second;
        cached.content_valid = false;
        cached.graphics_export_valid = false;
        cached.compute_transfer_valid = false;
        cached.write_watch.reset();
        if (preserve_snapshot_capacity)
            cached.source_snapshot.clear();
        else if (!cached.source_snapshot.empty())
            std::vector<uint8_t>().swap(cached.source_snapshot);
        // acquire_cached_image may otherwise reuse a same-submit validation result before checking
        // content_valid. A failed post-submit readback can leave the retained image/result buffer
        // newer than guest memory, so every source/export authority must be rebuilt by the retry.
        cached.validation_epoch = 0;
        cached.validation_result = false;
    }

    void validate_cached_image_source_from_compute_transfer(
        const ComputeImageCacheKey& key) {
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        CachedComputeImage& cached = found->second;
        cached.content_valid = true;
        cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        cached.write_watch_stable_validations = 0;
        cached.write_watch.reset();
        if (!cached.source_snapshot.empty())
            std::vector<uint8_t>().swap(cached.source_snapshot);
    }

    void validate_cached_image_source(const ComputeImageCacheKey& key,
                                      const uint8_t* current_source = nullptr,
                                      bool result_unchanged = false,
                                      bool source_snapshot_required = true,
                                      bool compute_transfer_watch = false) {
        auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        CachedComputeImage& cached = found->second;
        const bool source_was_valid = cached.content_valid;
        // A storage dispatch replaces both the image and its guest-memory mirror. Retain those
        // result bytes as the exact comparison baseline; keeping the pre-dispatch input here could
        // misclassify a later guest write that restores that old input as "unchanged".
        cached.content_valid = true;
        if (!key.host_data)
            cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        cached.write_watch_stable_validations = 0;
        if (key.host_data) {
            if (current_source)
                remember_image_source_snapshot(
                    cached, current_source, key.guest_bytes, true);
            cached.write_watch.reset();
            return;
        }
        if (!adaptive_storage_result_validation_enabled()) {
            // Diagnostic/control path matching the pre-optimization policy: every successful
            // storage writeback retains exact guest bytes, and only an already-existing generic
            // source watch may be rearmed.
            if (current_source)
                remember_image_source_snapshot(
                    cached, current_source, key.guest_bytes, true);
            if (cached.write_watch && cached.write_watch.rearm()) return;
            cached.write_watch.reset();
            return;
        }
        if (source_snapshot_required) {
            // Read/modify/write and partial storage targets still need the ordinary exact source
            // contract because their prior guest bytes are observable by the next dispatch.
            if (current_source)
                remember_image_source_snapshot(
                    cached, current_source, key.guest_bytes, true);
            if (cached.write_watch && cached.write_watch.rearm()) return;
            cached.write_watch.reset();
            return;
        }
        // Storage-result adaptation deliberately uses exact bytes, not a page watch. Page-watch
        // queries walk one record per host page and were slower than memcmp for Plucky's stable
        // post-process targets. A repeated result keeps one collision-free baseline; an identical
        // GPU skip never reaches this function, so that baseline is not recopied.
        if (compute_transfer_watch) {
            if (cached.write_watch && !cached.write_watch.rearm())
                cached.write_watch.reset();
            if (!cached.write_watch)
                cached.write_watch = prosper::host::GuestWriteWatch::create(
                    key.gpu_addr, key.guest_bytes);
        } else {
            cached.write_watch.reset();
        }
        if (!source_was_valid) {
            // A failed dispatch/readback may have advanced the retained GPU result while leaving
            // guest memory at the old baseline. The successful repair must replace that invalidated
            // authority even when the GPU comparator calls the retried result "unchanged".
            if (current_source)
                remember_image_source_snapshot(
                    cached, current_source, key.guest_bytes, true);
        } else if (result_unchanged) {
            if (current_source && cached.source_snapshot.empty())
                remember_image_source_snapshot(
                    cached, current_source, key.guest_bytes, true);
        } else if (!cached.source_snapshot.empty()) {
            std::vector<uint8_t>().swap(cached.source_snapshot);
        }
    }

    bool cached_image_result_matches(const ComputeImageCacheKey& key,
                                     const uint8_t* result, size_t bytes) const {
        if (!result) return false;
        const auto found = image_cache.find(key);
        return found != image_cache.end() && found->second.content_valid &&
               found->second.result_snapshot.size() == bytes &&
               std::memcmp(found->second.result_snapshot.data(), result, bytes) == 0;
    }

    bool cached_image_result_buffer(const ComputeImageCacheKey& key, VkDeviceSize bytes,
                                    VkBuffer& result) const {
        const auto found = image_cache.find(key);
        if (found == image_cache.end() || found->second.result_bytes != bytes ||
            !found->second.result_buffer)
            return false;
        result = found->second.result_buffer;
        return true;
    }

    bool retain_cached_image_result_buffer(const ComputeImageCacheKey& key,
                                           VkBuffer& buffer, VkDeviceMemory& memory,
                                           VkDeviceSize allocation_bytes,
                                           VkDeviceSize result_bytes) {
        if (!buffer || !memory || !result_bytes || (result_bytes & 3u)) return false;
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return false;
        CachedComputeImage& cached = found->second;
        if (cached.result_buffer) return false;
        // This staging buffer contains the current dispatch result. Once the caller chooses it as
        // the replacement baseline, an older host fallback is no longer authoritative even when
        // ownership fails (for example because every cache entry is pinned). Leaving that fallback
        // behind could let a later A result match stale A after failed result B remained in guest
        // memory, incorrectly suppressing the required A writeback.
        std::vector<uint8_t>().swap(cached.result_snapshot);
        if (g_fail_next_image_result_buffer_retain_for_test.exchange(
                false, std::memory_order_acq_rel))
            return false;
        if (!make_image_cache_room(allocation_bytes)) return false;
        cached.result_buffer = buffer;
        cached.result_memory = memory;
        cached.result_allocation_bytes = allocation_bytes;
        cached.result_bytes = result_bytes;
        cached.allocation_bytes += allocation_bytes;
        image_cache_bytes += allocation_bytes;
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return true;
    }

    void remember_cached_image_result(const ComputeImageCacheKey& key,
                                      const uint8_t* result, size_t bytes) {
        if (!result) return;
        const auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        // A retained GPU baseline is updated in the dispatch command buffer. Keep the old host
        // snapshot only as the exact fallback for small/unaligned results that cannot use it.
        if (!found->second.result_buffer) {
            found->second.result_snapshot.assign(result, result + bytes);
            ++image_result_snapshot_copies;
            image_result_snapshot_bytes += bytes;
        }
    }

    bool prepare_compare_pipeline() {
        if (compare_pipeline) return true;
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = 3;
        dlci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &dlci, nullptr,
                                        &compare_descriptor_layout) != VK_SUCCESS)
            return false;
        const std::vector<uint32_t> spirv = prosper::gpu::build_compute_compare_uvec4();
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();
        if (vkCreateShaderModule(device, &smci, nullptr, &compare_shader) != VK_SUCCESS)
            return false;
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = sizeof(uint32_t);
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &compare_descriptor_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device, &plci, nullptr,
                                   &compare_pipeline_layout) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = compare_shader;
        cpci.stage.pName = "main";
        cpci.layout = compare_pipeline_layout;
        return vkCreateComputePipelines(device, pipeline_cache, 1, &cpci, nullptr,
                                        &compare_pipeline) == VK_SUCCESS;
    }

    // A GRAPHICS queue family is not required by spec to also advertise COMPUTE, and the renderer
    // selects its family on GRAPHICS alone. Dispatching on a family without COMPUTE is invalid usage,
    // so verify before adopting rather than assuming the common family-0 layout.
    static bool queue_family_supports_compute(VkPhysicalDevice phys, uint32_t family) {
        if (!phys || family == UINT32_MAX) return false;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
        if (family >= count) return false;
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());
        return (props[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
    }

    void query_subgroup_support() {
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical, &properties);
        subgroup_size = subgroup.subgroupSize;
        subgroup_stages = subgroup.supportedStages;
        subgroup_operations = subgroup.supportedOperations;
        std::copy_n(properties.properties.limits.maxComputeWorkGroupCount, 3,
                    max_compute_workgroup_count.begin());
    }

    bool init() {
        // Adopt the live renderer's device when it published one (#1091). Sharing a device is what
        // makes it possible for a dispatch to bind a renderer-owned image at all; without it every
        // such binding must round-trip through host memory. Declined when the renderer's device
        // lacks the storage-image features this backend needs, and absent entirely in headless
        // compute-only use (tests/test_game_compute.cpp), where the private device below is created
        // exactly as before.
        const prosper::gpu::SharedVulkanContext shared = prosper::gpu::shared_vulkan_context();
        if (shared.valid() && shared.storage_image_read_without_format &&
            shared.storage_image_write_without_format &&
            queue_family_supports_compute(static_cast<VkPhysicalDevice>(shared.physical),
                                          shared.queue_family)) {
            instance = static_cast<VkInstance>(shared.instance);
            physical = static_cast<VkPhysicalDevice>(shared.physical);
            device = static_cast<VkDevice>(shared.device);
            queue = static_cast<VkQueue>(shared.queue);
            queue_family = shared.queue_family;
            borrowed = true;
            image_support = true;
            native_subgroup_contract = shared.compute_subgroup_size_control &&
                shared.compute_full_subgroups && shared.compute_subgroup_vote &&
                shared.compute_subgroup_arithmetic;
            min_native_subgroup_size = shared.min_compute_subgroup_size;
            max_native_subgroup_size = shared.max_compute_subgroup_size;
            VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            if (vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache) == VK_SUCCESS) {
                vkGetPhysicalDeviceMemoryProperties(physical, &memory);
                query_subgroup_support();
                std::fprintf(stderr, "[compute] Vulkan device: adopted the renderer's device "
                                     "(shared, queue family %u)\n", queue_family);
                return true;
            }
            // Anything failing here must fall through to a private device rather than killing the
            // whole compute backend for the run: adoption is an optimization, never a requirement.
            // Release only what we created (nothing yet: the pipeline cache is what failed) and drop
            // the borrowed handles so the private path below starts from a clean context.
            instance = VK_NULL_HANDLE; physical = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE; queue = VK_NULL_HANDLE;
            queue_family = UINT32_MAX; borrowed = false; image_support = false;
            native_subgroup_contract = false;
            min_native_subgroup_size = max_native_subgroup_size = 0;
            pipeline_cache = VK_NULL_HANDLE;
        }
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (!device_count) return false;
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        const auto selection = select_vulkan_device(devices, VK_QUEUE_COMPUTE_BIT);
        physical = selection.device;
        queue_family = selection.queue_family;
        if (!physical || queue_family == UINT32_MAX) return false;
        std::fprintf(stderr, "[compute] Vulkan device: %s (%s)\n",
                     selection.properties.deviceName,
                     vulkan_device_type_name(selection.properties.deviceType));

        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(physical, &supported);
        if (!supported.robustBufferAccess) {
            std::fprintf(stderr, "[compute] device lacks robustBufferAccess\n");
            return false;
        }
        VkPhysicalDeviceFeatures enabled{};
        enabled.robustBufferAccess = VK_TRUE;
        // Image bindings (#590): enable the format-free storage-image features when available.
        image_support = supported.shaderStorageImageReadWithoutFormat &&
                        supported.shaderStorageImageWriteWithoutFormat;
        if (image_support) {
            enabled.shaderStorageImageReadWithoutFormat = VK_TRUE;
            enabled.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        }
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &enabled;
        std::vector<const char*> dev_exts;
#ifdef __APPLE__
        // Spec-mandated on MoltenVK: enable VK_KHR_portability_subset when advertised (always is).
        { uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(physical, nullptr, &ne, nullptr);
          std::vector<VkExtensionProperties> de(ne);
          vkEnumerateDeviceExtensionProperties(physical, nullptr, &ne, de.data());
          for (auto& e : de) if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset")) {
              dev_exts.push_back("VK_KHR_portability_subset"); break; } }
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
#endif
        if (vkCreateDevice(physical, &dci, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        if (vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache) != VK_SUCCESS)
            return false;
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        query_subgroup_support();
        return true;
    }

    uint32_t host_memory_type(uint32_t bits) const {
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const VkMemoryPropertyFlags cached = wanted | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & cached) == cached)
                return i;
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted)
                return i;
        return UINT32_MAX;
    }

};

VulkanComputeContext* g_live_compute_context = nullptr;
std::atomic<uint64_t> g_sampled_image_upload_skips{0};
std::atomic<uint64_t> g_cpu_fill_dispatches{0};

struct BorrowedComputeImageLease {
    VulkanComputeContext* context = nullptr;
    ComputeImageCacheKey key{};
    ~BorrowedComputeImageLease() {
        if (context) context->release_cached_image(key);
    }
};

struct BoundBuffer {
    const prosper::gpu::ShaderResource* resource = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t alias_of = SIZE_MAX;         // exact guest range sharing an earlier storage buffer
    size_t bytes = 0;                   // Vulkan buffer bytes (may be a detiled image view)
    size_t guest_bytes = 0;             // physical guest backing (may exceed logical image bytes)
    bool writable = false;              // reflected OpStore/writing-atomic reachability
    bool atomic_image = false;           // R32_UINT StorageImage exposed as a linear atomic SSBO
    bool persistent = false;
    bool upload_skipped = false;
    VkBuffer result_baseline = VK_NULL_HANDLE;
    size_t compare_flag_index = SIZE_MAX;
    bool gpu_result_unchanged = false;
    uint32_t dirty_watch_chunks = 0;
    uint32_t total_watch_chunks = 0;
    ComputeBufferCacheKey cache_key{};
    std::vector<uint8_t> linear_seed;    // detiled upload for an atomic-image buffer
    uint64_t before_hash = 0, after_hash = 0;
    uint64_t changed_bytes = 0;
};

// Image conversion writes straight into the host-visible Vulkan staging allocation. Keep the map
// scoped so every validation/error exit unmaps it before cleanup releases the pooled memory.
struct ScopedMappedMemory {
    explicit ScopedMappedMemory(VulkanComputeContext& c) : context(c) {}
    ~ScopedMappedMemory() { unmap(); }
    ScopedMappedMemory(const ScopedMappedMemory&) = delete;
    ScopedMappedMemory& operator=(const ScopedMappedMemory&) = delete;

    void unmap() {
        if (!data) return;
        context.unmap_memory(memory);
        data = nullptr;
    }

    VulkanComputeContext& context;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* data = nullptr;
};

// One image binding (#590): a sampled texture (usually RGBA8, with native UINT8x4 and R11G11B10F
// views where shader-visible numeric semantics require them) or a storage image. Storage normally
// uses R32G32B32A32_UINT format-free texels; packed R11G11B10 can instead use exact R32_UINT words.
struct BoundImage {
    const prosper::gpu::ShaderResource* resource = nullptr;
    uint32_t binding = 0;
    bool storage = false;               // storage image: read back + pack to guest after the dispatch
    bool native_float_storage = false;  // Vulkan performs exact UNORM/float conversion at native width
    bool packed_r11_storage = false;    // shader packs exact R11G11B10 words into typed R32_UINT
    bool unorm_rtt_value_reuse = false; // float R16 view reuses authoritative RGBA8 values
    bool graphics_sampled_usage = false;// native image was created with SAMPLED usage for export
    bool exact_storage_bytes() const { return native_float_storage || packed_r11_storage; }
    uint32_t texel_depth = 1;           // logical Z/layer count represented in the staging buffer
    uint32_t array_layers = 1;           // Vulkan array-layer count (3D depth remains one layer)
    bool arrayed_2d = false;            // SPIR-V requires a real 2D-array view (not base-slice fallback)
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
    // Borrowed renderer-owned image bound in place (#1095). `image` is then owned by the live
    // renderer: it must not be destroyed here, its layout must be restored, and the pin taken at
    // import time must be released.
    bool imported = false;
    bool imported_depth = false;        // borrowed persistent DS depth plane, not a color RTT
    // A one-component Uint32 T# can alias a renderer-owned D32 depth plane byte-for-byte. Vulkan
    // cannot create an R32_UINT view of a depth image, so keep the borrowed DS image as a transfer
    // source and materialize the guest-declared integer view through a device buffer. This remains
    // entirely on the shared GPU; no stale guest backing or synchronous host readback is involved.
    bool depth_bits_source = false;
    VkImage depth_bits_image = VK_NULL_HANDLE;
    VkFormat depth_bits_format = VK_FORMAT_UNDEFINED;
    uint32_t depth_bits_saved_layout = 0;
    VkFormat imported_format = VK_FORMAT_UNDEFINED;
    prosper::gpu::LiveTargetPixelFormat imported_pixel_format =
        prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm;
    bool imported_transfer_dst = false;
    bool persistent = false;            // guest-backed sampled image retained across dispatches
    bool cache_candidate = false;
    bool post_writeback_promotion_candidate = false;
    // Exact cached allocation leased for a DCC-unsafe producer. Source authority was invalidated,
    // upload_skipped remains false, and cache publication still waits for post-writeback metadata.
    bool forced_seed_allocation_reused = false;
    bool upload_skipped = false;         // write watch proved the cached source unchanged
    VkDeviceSize allocation_bytes = 0;
    VkDeviceSize staging_allocation_bytes = 0;
    // Exact bytes copied from the Vulkan image into staging. Native typed storage already has the
    // guest's row-major byte width; the raw-uvec4 fallback has four uint32_t channels per texel.
    // The latter is safe to retain/compare only for proven-full write-only outputs, whose previous
    // raw image contents cannot be observed by the guest shader.
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
    bool seed_skip = false;             // #1122: write-only full-coverage storage image; no seed needed
    bool poison_verify = false;         // #1122: proving frame -- seed poison, prove full coverage
    size_t seed_from_imported = SIZE_MAX; // renderer image copied on-device into this partial-write target
    bool mirror_result_to_imported = false;
    std::vector<uint8_t> seed_linear;   // #1122: detiled guest seed, kept on a proving frame so any
                                        // texel the write leaves untouched is restored (not corrupted)
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

// --- Storage-image channel model (#590) -------------------------------------------------------------
// The portable recompiler path moves image texels as RAW 32-bit VGPR channel values (uvec4 per texel;
// the VkImage is R32G32B32A32_UINT with format-free reads/writes). Real hardware format-converts per
// the T#, so the
// guest surface's bytes must be UNPACKED to channel dwords on upload and PACKED back on writeback:
//   Unorm8/16 -> float(u/max) bits        <- clamp(bitcast float,0,1)*max rounded
//   Snorm8/16 -> max(float(s/max),-1) bits <- clamp(bitcast float,-1,1)*max rounded
//   Float16 -> half->float bits          <- round-to-nearest-even float->half
//   Float32/Uint32/Sint32 -> raw 4-byte move both ways.
//   Uint8/Sint8/Uint16/Sint16 -> integer channel widen (sign-extend for Sint) <- truncate to width.
//     A UINT/SINT image_load returns the stored integer directly and image_store writes the low
//     N bits with no normalization or saturation (Vulkan integer-format store contract), so the
//     host move is a width-aware zero/sign extend on upload and a low-bit truncation on writeback.
//   Unorm2_10_10_10 -> per-field float(bits/max) bits <- clamp(bitcast float,0,1)*max rounded, packed
//     high-to-low A2/B10/G10/R10 (GFX10 IMG_FMT 50 layout, matching unorm2_10_10_10_to_rgba8).
// UE4's post-process color-grading writes its 3D LUT / exposure volumes as these formats (DOLL: a
// 32x32x32 Uint8/Unorm2_10_10_10 LUT + 1x1x1 exposure + a 16x16x16 Uint16 volume); skipping the
// dispatch left the tonemap sampling an unproduced volume -> a near-zero grade -> black title (#590).
// Missing channels read the hardware default (0,0,0,1.0f). Anything else is unsupported -> the caller
// skips the dispatch loudly (never a silent wrong-layout write — correctness-first).
bool storage_unpack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Unorm16 || f == DF::Snorm8 || f == DF::Snorm16 ||
           f == DF::Float16 || f == DF::Float32 || f == DF::Uint32 ||
           f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
bool storage_pack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Unorm16 || f == DF::Snorm8 || f == DF::Snorm16 ||
           f == DF::Float16 || f == DF::Float32 ||
           f == DF::Uint32 || f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
void storage_unpack_texel(const uint8_t* src, prosper::gpu::DataFormat f, uint32_t ncomp, uint32_t out[4]) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t one_f32 = 0x3f800000u;                 // hardware default: missing channels = (0,0,0,1)
    // A missing alpha reads 1 — as the FLOAT 1.0 bits for float/unorm formats, but the INTEGER 1 for
    // UINT/SINT formats (an integer image_load returns raw integer channels, not normalized floats).
    const bool integer_fmt = f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 ||
                             f == DF::Sint16 || f == DF::Uint32 || f == DF::Sint32;
    out[0] = out[1] = out[2] = 0; out[3] = integer_fmt ? 1u : one_f32;
    if (f == DF::Float10_11_11) {
        uint32_t packed = 0; std::memcpy(&packed, src, sizeof(packed));
        const float values[3] = { prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)),
                                  prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
                                  prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)) };
        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&out[c], &values[c], sizeof(values[c]));
        return;
    }
    if (f == DF::Unorm2_10_10_10) {
        uint32_t packed = 0; std::memcpy(&packed, src, sizeof(packed));
        const float values[4] = { ((packed >>  0) & 0x3ffu) / 1023.0f,
                                  ((packed >> 10) & 0x3ffu) / 1023.0f,
                                  ((packed >> 20) & 0x3ffu) / 1023.0f,
                                  ((packed >> 30) & 0x3u)   / 3.0f };
        for (uint32_t c = 0; c < 4; ++c) std::memcpy(&out[c], &values[c], sizeof(values[c]));
        return;
    }
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: { float v = src[c] / 255.0f; std::memcpy(&out[c], &v, 4); break; }
            case DF::Unorm16: { const uint16_t raw = static_cast<uint16_t>(src[c * 2] |
                                      (static_cast<uint16_t>(src[c * 2 + 1]) << 8));
                                const float v = raw / 65535.0f; std::memcpy(&out[c], &v, 4); break; }
            case DF::Snorm8: { const float v = std::max(static_cast<int8_t>(src[c]) / 127.0f, -1.0f);
                               std::memcpy(&out[c], &v, 4); break; }
            case DF::Snorm16: { const int16_t raw = static_cast<int16_t>(src[c * 2] |
                                      (static_cast<uint16_t>(src[c * 2 + 1]) << 8));
                                const float v = std::max(raw / 32767.0f, -1.0f);
                                std::memcpy(&out[c], &v, 4); break; }
            case DF::Float16:
                out[c] = storage_unpack_float16_bits(
                    static_cast<uint16_t>(src[c * 2] | (src[c * 2 + 1] << 8)));
                break;
            // Integer formats carry the raw channel value; a UINT/SINT image_load reads the stored
            // integer directly (zero-extend for Uint, sign-extend for Sint). No normalization.
            case DF::Uint8:  out[c] = src[c]; break;
            case DF::Sint8:  out[c] = static_cast<uint32_t>(static_cast<int32_t>(
                                          static_cast<int8_t>(src[c]))); break;
            case DF::Uint16: out[c] = static_cast<uint32_t>(src[c * 2] | (src[c * 2 + 1] << 8)); break;
            case DF::Sint16: out[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(
                                          src[c * 2] | (src[c * 2 + 1] << 8)))); break;
            default: std::memcpy(&out[c], src + c * 4, 4); break;   // 32-bit raw
        }
    }
}
// Unpack `count` consecutive texels (source stride `src_stride`) into `count` RGBA32 quads.
//
// storage_unpack_texel re-derives the format for every texel AND re-enters a switch for every
// component, so a full-resolution storage image pays ~texels function calls plus ~4x texels switch
// dispatches. That dominated compute image setup (measured: ~56 ms per image-bearing dispatch on
// Blasphemous 2's menu, against 0.39 ms of actual GPU dispatch). This hoists the format dispatch out
// of the loop so each specialized path is a tight typed loop.
//
// Every specialized path is bit-identical to storage_unpack_texel by construction; formats without a
// specialization fall through to the per-texel helper, and PROSPER_VERIFY_UNPACK=1 checks the two
// against each other at runtime.
void storage_unpack_range(const uint8_t* src, size_t src_stride, prosper::gpu::DataFormat f,
                          uint32_t ncomp, size_t count, uint32_t* out) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t one_f32 = 0x3f800000u;
    const bool integer_fmt = f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 ||
                             f == DF::Sint16 || f == DF::Uint32 || f == DF::Sint32;
    const uint32_t alpha_default = integer_fmt ? 1u : one_f32;
    const uint32_t n = ncomp < 4u ? ncomp : 4u;
    auto defaults = [&](uint32_t* o) { o[0] = o[1] = o[2] = 0; o[3] = alpha_default; };
    switch (f) {
        case DF::Unorm8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) { float v = p[c] / 255.0f; std::memcpy(&o[c], &v, 4); }
            }
            return;
        case DF::Unorm16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) {
                    const uint16_t raw = static_cast<uint16_t>(p[c * 2] |
                        (static_cast<uint16_t>(p[c * 2 + 1]) << 8));
                    const float v = raw / 65535.0f; std::memcpy(&o[c], &v, 4);
                }
            }
            return;
        case DF::Snorm8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) {
                    const float v = std::max(static_cast<int8_t>(p[c]) / 127.0f, -1.0f);
                    std::memcpy(&o[c], &v, 4);
                }
            }
            return;
        case DF::Snorm16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) {
                    const int16_t raw = static_cast<int16_t>(p[c * 2] |
                        (static_cast<uint16_t>(p[c * 2 + 1]) << 8));
                    const float v = std::max(raw / 32767.0f, -1.0f);
                    std::memcpy(&o[c], &v, 4);
                }
            }
            return;
        case DF::Float16:
            if (n == 4 && src_stride == 8) {
                storage_unpack_float16x4_range(src, count, out);
                return;
            }
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride;
                        uint32_t* o = out + t * 4;
                        defaults(o);
                        for (uint32_t c = 0; c < n; ++c)
                            o[c] = storage_unpack_float16_bits(
                                static_cast<uint16_t>(p[c * 2] | (p[c * 2 + 1] << 8)));
                    }
                });
            return;
        case DF::Uint8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) o[c] = p[c];
            }
            return;
        case DF::Sint8:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(p[c])));
            }
            return;
        case DF::Uint16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(p[c * 2] | (p[c * 2 + 1] << 8));
            }
            return;
        case DF::Sint16:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c)
                    o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(
                        p[c * 2] | (p[c * 2 + 1] << 8))));
            }
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            for (size_t t = 0; t < count; ++t) {
                const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                for (uint32_t c = 0; c < n; ++c) std::memcpy(&o[c], p + c * 4, 4);
            }
            return;
        default:                                  // packed formats keep the general per-texel path
            for (size_t t = 0; t < count; ++t)
                storage_unpack_texel(src + t * src_stride, f, ncomp, out + t * 4);
            return;
    }
}
// Range-specialized pack (#1101): the writeback mirror of storage_unpack_range (#1092). Hoists the
// per-texel format dispatch out of the loop with per-format inner loops; packed formats keep the
// general per-texel path. Semantics are IDENTICAL to storage_pack_texel over the range -- asserted
// format-by-format by test_storage_pack_range.
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst);
void storage_pack_range(const uint32_t* channels, prosper::gpu::DataFormat f, uint32_t ncomp,
                        size_t count, uint8_t* dst, size_t dst_stride) {
    using DF = prosper::gpu::DataFormat;
    const uint32_t n = ncomp < 4u ? ncomp : 4u;
    switch (f) {
        case DF::Unorm8:
            if (dst_stride == n && n >= 1 && n <= 4) {
                storage_pack_unorm8_range(channels, n, count, dst);
                return;
            }
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4;
                        uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c)
                            p[c] = storage_pack_unorm8(in[c]);
                    }
                });
            return;
        case DF::Unorm16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    const uint16_t raw = storage_pack_unorm16(in[c]);
                    p[c * 2] = static_cast<uint8_t>(raw);
                    p[c * 2 + 1] = static_cast<uint8_t>(raw >> 8);
                }
            }
            return;
        case DF::Snorm8:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c)
                    p[c] = static_cast<uint8_t>(storage_pack_snorm<int8_t>(in[c], 127));
            }
            return;
        case DF::Snorm16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    const uint16_t raw = static_cast<uint16_t>(
                        storage_pack_snorm<int16_t>(in[c], 32767));
                    p[c * 2] = static_cast<uint8_t>(raw);
                    p[c * 2 + 1] = static_cast<uint8_t>(raw >> 8);
                }
            }
            return;
        case DF::Float16:
            if (n == 4 && dst_stride == 8) {
                storage_pack_float16x4_range(channels, count, dst);
                return;
            }
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4;
                        uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) {
                            float v; std::memcpy(&v, &in[c], 4);
                            const uint16_t h = prosper::gpu::float_to_half(v);
                            p[c * 2] = static_cast<uint8_t>(h);
                            p[c * 2 + 1] = static_cast<uint8_t>(h >> 8);
                        }
                    }
                });
            return;
        case DF::Uint8: case DF::Sint8:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) p[c] = static_cast<uint8_t>(in[c]);
            }
            return;
        case DF::Uint16: case DF::Sint16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    p[c * 2] = static_cast<uint8_t>(in[c]);
                    p[c * 2 + 1] = static_cast<uint8_t>(in[c] >> 8);
                }
            }
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) std::memcpy(p + c * 4, &in[c], 4);
            }
            return;
        default:                                  // packed formats keep the general per-texel path
            for (size_t t = 0; t < count; ++t)
                storage_pack_texel(channels + t * 4, f, ncomp, dst + t * dst_stride);
            return;
    }
}
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst) {
    using DF = prosper::gpu::DataFormat;
    if (f == DF::Float10_11_11) {
        float values[3];
        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&values[c], &in[c], sizeof(values[c]));
        const uint32_t packed = static_cast<uint32_t>(prosper::gpu::float_to_f11(values[0])) |
                                (static_cast<uint32_t>(prosper::gpu::float_to_f11(values[1])) << 11) |
                                (static_cast<uint32_t>(prosper::gpu::float_to_f10(values[2])) << 22);
        std::memcpy(dst, &packed, sizeof(packed));
        return;
    }
    if (f == DF::Unorm2_10_10_10) {
        auto q = [](const uint32_t bits, float scale) -> uint32_t {
            float v; std::memcpy(&v, &bits, 4);
            v = !(v > 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);   // NaN and negatives clamp to 0
            return static_cast<uint32_t>(v * scale + 0.5f);
        };
        const uint32_t packed = (q(in[0], 1023.0f) & 0x3ffu)        |
                                ((q(in[1], 1023.0f) & 0x3ffu) << 10) |
                                ((q(in[2], 1023.0f) & 0x3ffu) << 20) |
                                ((q(in[3], 3.0f)    & 0x3u)   << 30);
        std::memcpy(dst, &packed, sizeof(packed));
        return;
    }
    for (uint32_t c = 0; c < ncomp && c < 4; c++) {
        switch (f) {
            case DF::Unorm8: dst[c] = storage_pack_unorm8(in[c]); break;
            case DF::Unorm16: { const uint16_t raw = storage_pack_unorm16(in[c]);
                                dst[c * 2] = static_cast<uint8_t>(raw);
                                dst[c * 2 + 1] = static_cast<uint8_t>(raw >> 8); break; }
            case DF::Snorm8: { dst[c] = static_cast<uint8_t>(
                                   storage_pack_snorm<int8_t>(in[c], 127)); break; }
            case DF::Snorm16: { const uint16_t raw = static_cast<uint16_t>(
                                    storage_pack_snorm<int16_t>(in[c], 32767));
                                dst[c * 2] = static_cast<uint8_t>(raw);
                                dst[c * 2 + 1] = static_cast<uint8_t>(raw >> 8); break; }
            case DF::Float16: { float v; std::memcpy(&v, &in[c], 4);
                                const uint16_t h = prosper::gpu::float_to_half(v);
                                dst[c * 2] = static_cast<uint8_t>(h);
                                dst[c * 2 + 1] = static_cast<uint8_t>(h >> 8); break; }
            // Integer image_store writes the low N bits with no saturation (mirrors the 32-bit raw
            // move truncated to the format width).
            case DF::Uint8: case DF::Sint8:
                dst[c] = static_cast<uint8_t>(in[c]); break;
            case DF::Uint16: case DF::Sint16:
                dst[c * 2] = static_cast<uint8_t>(in[c]);
                dst[c * 2 + 1] = static_cast<uint8_t>(in[c] >> 8); break;
            default: std::memcpy(dst + c * 4, &in[c], 4); break;    // 32-bit raw
        }
    }
}

uint64_t fnv1a(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool trace_compute_item(const prosper::gpu::ComputeItem& item) {
    if (std::getenv("PROSPER_COMPUTELOG")) return true;
    const char* code_env = std::getenv("PROSPER_COMPUTELOG_CODE");
    const char* size_env = std::getenv("PROSPER_COMPUTELOG_SIZE");
    if ((!code_env || !*code_env) && (!size_env || !*size_env)) return false;
    if (code_env && *code_env) {
        char* end = nullptr;
        const uint64_t wanted = std::strtoull(code_env, &end, 0);
        if (!end || *end || item.code_addr != wanted) return false;
    }
    if (size_env && *size_env) {
        char* end = nullptr;
        const unsigned long wanted = std::strtoul(size_env, &end, 0);
        if (!end || *end || !item.resources) return false;
        const auto found = std::find_if(item.resources->resources.begin(),
                                       item.resources->resources.end(),
            [&](const auto& resource) { return resource.size == wanted; });
        if (found == item.resources->resources.end()) return false;
    }
    return true;
}

bool time_compute_address_matches(const prosper::gpu::ComputeItem& item) {
    const char* code_env = std::getenv("PROSPER_COMPUTE_TIMING_CODE");
    if (!code_env || !*code_env) return true;
    char* end = nullptr;
    errno = 0;
    const uint64_t wanted = std::strtoull(code_env, &end, 0);
    return !errno && end != code_env && end && !*end && item.code_addr == wanted;
}

class RuntimeComputeTimingSelector {
public:
    RuntimeComputeTimingSelector() {
        const char* hash_env = std::getenv("PROSPER_COMPUTE_TIMING_HASH");
        if (!hash_env) return;
        hash_requested_ = true;
        const ComputeTimingSelectorParseResult parsed =
            parse_compute_timing_selector_u64(hash_env);
        hash_error_ = parsed.error;
        selector_.hash_requested = true;
        selector_.hash_valid = parsed.accepted();
        selector_.hash = parsed.value;
        const char* address_env = std::getenv("PROSPER_COMPUTE_TIMING_CODE");
        const bool address_requested = address_env && *address_env;
        if (parsed.accepted()) {
            std::fprintf(stderr,
                         "[compute-timing-filter] PROSPER_COMPUTE_TIMING_HASH accepted "
                         "value=0x%016llx mode=%s\n",
                         static_cast<unsigned long long>(parsed.value),
                         address_requested ? "address+hash-AND" : "hash-only");
        } else {
            std::fprintf(stderr,
                         "[compute-timing-filter] PROSPER_COMPUTE_TIMING_HASH ignored "
                         "reason=%s; selector fails closed\n",
                         compute_timing_selector_parse_error_name(parsed.error));
        }
    }

    ~RuntimeComputeTimingSelector() {
        report_summary();
    }

    void report_summary() {
        if (!hash_requested_) return;
        std::lock_guard lock(mutex_);
        if (!claim_compute_timing_selector_summary(counters_)) return;
        if (!selector_.hash_valid) {
            std::fprintf(stderr,
                         "[compute-timing-filter] summary status=ignored reason=%s "
                         "seen=%llu matched=%llu verdict=INVALID-selector-not-armed\n",
                         compute_timing_selector_parse_error_name(hash_error_),
                         static_cast<unsigned long long>(counters_.seen),
                         static_cast<unsigned long long>(counters_.matched));
            return;
        }
        std::fprintf(stderr,
                     "[compute-timing-filter] summary status=accepted hash=0x%016llx "
                     "seen=%llu matched=%llu verdict=%s\n",
                     static_cast<unsigned long long>(selector_.hash),
                     static_cast<unsigned long long>(counters_.seen),
                     static_cast<unsigned long long>(counters_.matched),
                     compute_timing_zero_match_is_invalid(counters_)
                         ? "INVALID-zero-matches" : "matched");
    }

    bool matches(const prosper::gpu::ComputeItem& item, uint64_t program_hash) {
        if (!hash_requested_) return time_compute_address_matches(item);

        ComputeTimingSelector current = selector_;
        const char* code_env = std::getenv("PROSPER_COMPUTE_TIMING_CODE");
        if (code_env && *code_env) {
            current.address_enabled = true;
            char* end = nullptr;
            errno = 0;
            current.address = std::strtoull(code_env, &end, 0);
            current.address_valid = !errno && end != code_env && end && !*end;
        }
        std::lock_guard lock(mutex_);
        const ComputeTimingSelectorObservation observation =
            observe_compute_timing_selector(
                current, counters_, item.code_addr, program_hash);
        if (observation.first_match) {
            std::fprintf(stderr,
                         "[compute-timing-filter] first-match hash=0x%016llx "
                         "code=0x%llx seen=%llu matched=%llu\n",
                         static_cast<unsigned long long>(program_hash),
                         static_cast<unsigned long long>(item.code_addr),
                         static_cast<unsigned long long>(counters_.seen),
                         static_cast<unsigned long long>(counters_.matched));
        }
        return observation.matched;
    }

private:
    bool hash_requested_ = false;
    ComputeTimingSelectorParseError hash_error_ =
        ComputeTimingSelectorParseError::Unset;
    ComputeTimingSelector selector_;
    ComputeTimingSelectorCounters counters_;
    std::mutex mutex_;
};

RuntimeComputeTimingSelector& runtime_compute_timing_selector() {
    static RuntimeComputeTimingSelector selector;
    return selector;
}

const char* compute_transfer_gate_role_name(ComputeTransferGateRole role) {
    switch (role) {
        case ComputeTransferGateRole::Producer: return "producer";
        case ComputeTransferGateRole::Consumer: return "consumer";
        case ComputeTransferGateRole::None: return "none";
    }
    return "none";
}

const char* compute_transfer_borrow_result_name(ComputeTransferBorrowResult result) {
    switch (result) {
        case ComputeTransferBorrowResult::NotAttempted: return "not-attempted";
        case ComputeTransferBorrowResult::Hit: return "hit";
        case ComputeTransferBorrowResult::NoCache: return "no-cache";
        case ComputeTransferBorrowResult::InvalidCache: return "invalid-cache";
        case ComputeTransferBorrowResult::AuthorityChanged: return "authority-changed";
    }
    return "unknown";
}

struct ComputeTransferGateStats {
    uint64_t reflected_images = 0;
    uint64_t reflected_storage = 0;
    uint64_t reflected_sampled = 0;
    uint64_t reflected_dim_2d = 0;
    uint64_t reflected_nonarrayed = 0;
    uint64_t reflected_nonmsaa = 0;
    uint64_t ordinary_2d = 0;
    uint64_t storage_float = 0;
    uint64_t storage_native_semantic = 0;
    uint64_t storage_native_device = 0;
    uint64_t storage_cache_evaluated = 0;
    uint64_t storage_renderer_owned = 0;
    uint64_t storage_dcc_cache_safe = 0;
    uint64_t storage_poison_verify = 0;
    uint64_t storage_exact = 0;
    uint64_t storage_seed_skip = 0;
    uint64_t storage_persistent_enabled = 0;
    uint64_t storage_cache_candidate = 0;
    uint64_t storage_persistent_setup = 0;
    uint64_t storage_publish_evaluated = 0;
    uint64_t storage_publish_native = 0;
    uint64_t storage_publish_unique = 0;
    uint64_t storage_publish_candidate = 0;
    uint64_t storage_publish_persistent = 0;
    uint64_t storage_publish_authorized = 0;
    uint64_t sampled_gate_evaluated = 0;
    uint64_t sampled_cache_candidate = 0;
    uint64_t sampled_persistent_setup = 0;
    uint64_t sampled_ordinary_2d = 0;
    uint64_t sampled_format_compatible = 0;
    uint64_t sampled_transfer_dimension = 0;
    uint64_t sampled_hostless = 0;
    uint64_t sampled_format_match = 0;
    uint64_t sampled_validation_enabled = 0;
    uint64_t sampled_native_defined = 0;
    uint64_t borrow_attempts = 0;
    uint64_t borrow_hits = 0;
    uint64_t borrow_no_cache = 0;
    uint64_t borrow_invalid_cache = 0;
    uint64_t borrow_authority_changed = 0;
};

void increment_gate_counter(uint64_t& value, bool condition = true) {
    if (condition) value = saturating_increment(value);
}

class RuntimeComputeTransferGateCensus {
public:
    RuntimeComputeTransferGateCensus() {
        const char* producer_env =
            std::getenv("PROSPER_COMPUTE_TRANSFER_PRODUCER_HASH");
        const char* consumer_env =
            std::getenv("PROSPER_COMPUTE_TRANSFER_CONSUMER_HASH");
        if (!producer_env && !consumer_env) return;
        selector_.requested = true;
        const ComputeTimingSelectorParseResult producer =
            parse_compute_timing_selector_u64(producer_env);
        const ComputeTimingSelectorParseResult consumer =
            parse_compute_timing_selector_u64(consumer_env);
        selector_.producer_hash = producer.value;
        selector_.consumer_hash = consumer.value;
        selector_.valid = producer.accepted() && consumer.accepted() &&
                          producer.value != consumer.value;
        producer_error_ = producer.error;
        consumer_error_ = consumer.error;
        if (selector_.valid) {
            std::fprintf(stderr,
                         "[compute-transfer-gates] accepted producer=0x%016llx "
                         "consumer=0x%016llx mode=exact-hash\n",
                         static_cast<unsigned long long>(selector_.producer_hash),
                         static_cast<unsigned long long>(selector_.consumer_hash));
        } else {
            std::fprintf(stderr,
                         "[compute-transfer-gates] ignored producer-reason=%s "
                         "consumer-reason=%s distinct=%u; selector fails closed\n",
                         compute_timing_selector_parse_error_name(producer.error),
                         compute_timing_selector_parse_error_name(consumer.error),
                         producer.value != consumer.value ? 1u : 0u);
        }
    }

    ~RuntimeComputeTransferGateCensus() {
        report_summary();
    }

    bool requested() const {
        return selector_.requested;
    }

    ComputeTransferGateSelectorObservation observe(
        const prosper::gpu::ComputeItem& item, uint64_t program_hash) {
        if (!selector_.requested) return {};
        std::lock_guard lock(mutex_);
        const ComputeTransferGateSelectorObservation observation =
            observe_compute_transfer_gate_selector(selector_, counters_, program_hash);
        if (observation.first_match) {
            const uint64_t matches = observation.role == ComputeTransferGateRole::Producer
                ? counters_.producer_matches : counters_.consumer_matches;
            std::fprintf(stderr,
                         "[compute-transfer-gates] first-match role=%s hash=0x%016llx "
                         "code=0x%llx seen=%llu role-matched=%llu\n",
                         compute_transfer_gate_role_name(observation.role),
                         static_cast<unsigned long long>(program_hash),
                         static_cast<unsigned long long>(item.code_addr),
                         static_cast<unsigned long long>(counters_.seen),
                         static_cast<unsigned long long>(matches));
        }
        return observation;
    }

    void record_reflection(ComputeTransferGateRole role,
                           const prosper::gpu::SpirvDescriptorBinding& descriptor,
                           const prosper::gpu::ShaderResource& resource,
                           bool storage, bool ordinary_2d,
                           bool native_semantic, bool native_device,
                           bool detail) {
        if (role == ComputeTransferGateRole::None) return;
        std::lock_guard lock(mutex_);
        ComputeTransferGateStats& stats = stats_for(role);
        increment_gate_counter(stats.reflected_images);
        increment_gate_counter(stats.reflected_storage, storage);
        increment_gate_counter(stats.reflected_sampled, !storage);
        increment_gate_counter(stats.reflected_dim_2d, descriptor.image_dim == 1u);
        increment_gate_counter(stats.reflected_nonarrayed, !descriptor.image_arrayed);
        increment_gate_counter(stats.reflected_nonmsaa, !descriptor.image_multisampled);
        increment_gate_counter(stats.ordinary_2d, ordinary_2d);
        increment_gate_counter(stats.storage_float, storage && descriptor.storage_float);
        increment_gate_counter(stats.storage_native_semantic, storage && native_semantic);
        increment_gate_counter(stats.storage_native_device, storage && native_device);
        if (detail) {
            std::fprintf(stderr,
                         "[compute-transfer-gates] detail role=%s stage=reflection "
                         "binding=%u class=%s addr=0x%llx extent=%ux%ux%u "
                         "resource-dim=%u format=%u components=%u reflected-dim=%u "
                         "arrayed=%u msaa=%u storage-float=%u ordinary-2d=%u "
                         "native-semantic=%u native-device=%u\n",
                         compute_transfer_gate_role_name(role), descriptor.binding,
                         storage ? "storage" : "sampled",
                         static_cast<unsigned long long>(resource.gpu_addr),
                         resource.width, resource.height, resource.depth, resource.img_dim,
                         static_cast<unsigned>(resource.format), resource.num_components,
                         descriptor.image_dim, descriptor.image_arrayed ? 1u : 0u,
                         descriptor.image_multisampled ? 1u : 0u,
                         descriptor.storage_float ? 1u : 0u, ordinary_2d ? 1u : 0u,
                         native_semantic ? 1u : 0u, native_device ? 1u : 0u);
        }
    }

    void record_sampled_gates(ComputeTransferGateRole role,
                              const prosper::gpu::ShaderResource& resource,
                              uint32_t binding, bool cache_candidate,
                              bool persistent_setup, bool ordinary_2d,
                              bool format_compatible, bool transfer_dimension,
                              bool hostless, bool format_match,
                              bool validation_enabled, bool native_defined,
                              ComputeTransferBorrowResult borrow_result,
                              bool detail) {
        if (role == ComputeTransferGateRole::None) return;
        std::lock_guard lock(mutex_);
        ComputeTransferGateStats& stats = stats_for(role);
        increment_gate_counter(stats.sampled_gate_evaluated);
        increment_gate_counter(stats.sampled_cache_candidate, cache_candidate);
        increment_gate_counter(stats.sampled_persistent_setup, persistent_setup);
        increment_gate_counter(stats.sampled_ordinary_2d, ordinary_2d);
        increment_gate_counter(stats.sampled_format_compatible, format_compatible);
        increment_gate_counter(stats.sampled_transfer_dimension, transfer_dimension);
        increment_gate_counter(stats.sampled_hostless, hostless);
        increment_gate_counter(stats.sampled_format_match, format_match);
        increment_gate_counter(stats.sampled_validation_enabled, validation_enabled);
        increment_gate_counter(stats.sampled_native_defined, native_defined);
        const bool attempted = borrow_result != ComputeTransferBorrowResult::NotAttempted;
        increment_gate_counter(stats.borrow_attempts, attempted);
        increment_gate_counter(stats.borrow_hits,
                               borrow_result == ComputeTransferBorrowResult::Hit);
        increment_gate_counter(stats.borrow_no_cache,
                               borrow_result == ComputeTransferBorrowResult::NoCache);
        increment_gate_counter(stats.borrow_invalid_cache,
                               borrow_result == ComputeTransferBorrowResult::InvalidCache);
        increment_gate_counter(stats.borrow_authority_changed,
                               borrow_result == ComputeTransferBorrowResult::AuthorityChanged);
        if (detail) {
            std::fprintf(stderr,
                         "[compute-transfer-gates] detail role=%s stage=sampled-gates "
                         "binding=%u addr=0x%llx cache-candidate=%u persistent=%u "
                         "ordinary-2d=%u format-compatible=%u transfer-dimension=%u "
                         "hostless=%u format-match=%u validation=%u native-defined=%u "
                         "borrow=%s\n",
                         compute_transfer_gate_role_name(role), binding,
                         static_cast<unsigned long long>(resource.gpu_addr),
                         cache_candidate ? 1u : 0u, persistent_setup ? 1u : 0u,
                         ordinary_2d ? 1u : 0u, format_compatible ? 1u : 0u,
                         transfer_dimension ? 1u : 0u, hostless ? 1u : 0u,
                         format_match ? 1u : 0u, validation_enabled ? 1u : 0u,
                         native_defined ? 1u : 0u,
                         compute_transfer_borrow_result_name(borrow_result));
        }
    }

    void record_storage_cache(ComputeTransferGateRole role,
                              const prosper::gpu::ShaderResource& resource,
                              uint32_t binding,
                              const ComputeStorageCacheGateInputs& inputs,
                              bool cache_candidate,
                              bool persistent_setup) {
        if (role == ComputeTransferGateRole::None) return;
        std::lock_guard lock(mutex_);
        const bool detail =
            record_compute_transfer_storage_gate_observation(role, counters_);
        ComputeTransferGateStats& stats = stats_for(role);
        increment_gate_counter(stats.storage_cache_evaluated);
        increment_gate_counter(stats.storage_renderer_owned, inputs.renderer_owned);
        increment_gate_counter(stats.storage_dcc_cache_safe, inputs.dcc_cache_safe);
        increment_gate_counter(stats.storage_poison_verify, inputs.poison_verify);
        increment_gate_counter(stats.storage_exact, inputs.exact_storage);
        increment_gate_counter(stats.storage_seed_skip, inputs.seed_skip);
        increment_gate_counter(stats.storage_persistent_enabled,
                               inputs.persistent_enabled);
        increment_gate_counter(stats.storage_cache_candidate, cache_candidate);
        increment_gate_counter(stats.storage_persistent_setup, persistent_setup);
        if (detail) {
            std::fprintf(stderr,
                         "[compute-transfer-gates] detail role=%s stage=storage-cache "
                         "binding=%u addr=0x%llx renderer-owned=%u dcc-cache-safe=%u "
                         "poison-verify=%u exact-storage=%u seed-skip=%u "
                         "persistent-enabled=%u cache-candidate=%u persistent=%u\n",
                         compute_transfer_gate_role_name(role), binding,
                         static_cast<unsigned long long>(resource.gpu_addr),
                         inputs.renderer_owned ? 1u : 0u,
                         inputs.dcc_cache_safe ? 1u : 0u,
                         inputs.poison_verify ? 1u : 0u,
                         inputs.exact_storage ? 1u : 0u,
                         inputs.seed_skip ? 1u : 0u,
                         inputs.persistent_enabled ? 1u : 0u,
                         cache_candidate ? 1u : 0u, persistent_setup ? 1u : 0u);
        }
    }

    void record_storage_publish(ComputeTransferGateRole role, bool native,
                                bool unique, bool cache_candidate,
                                bool persistent, bool authorized) {
        if (role == ComputeTransferGateRole::None) return;
        std::lock_guard lock(mutex_);
        ComputeTransferGateStats& stats = stats_for(role);
        increment_gate_counter(stats.storage_publish_evaluated);
        increment_gate_counter(stats.storage_publish_native, native);
        increment_gate_counter(stats.storage_publish_unique, unique);
        increment_gate_counter(stats.storage_publish_candidate, cache_candidate);
        increment_gate_counter(stats.storage_publish_persistent, persistent);
        increment_gate_counter(stats.storage_publish_authorized, authorized);
    }

    void report_summary() {
        if (!selector_.requested) return;
        std::lock_guard lock(mutex_);
        if (!claim_compute_transfer_gate_selector_summary(counters_)) return;
        const bool invalid = compute_transfer_gate_selector_is_invalid(selector_, counters_);
        std::fprintf(stderr,
                     "[compute-transfer-gates] summary producer=0x%016llx "
                     "consumer=0x%016llx seen=%llu producer-matched=%llu "
                     "consumer-matched=%llu producer-storage-gates=%llu "
                     "consumer-storage-gates=%llu verdict=%s producer-reason=%s "
                     "consumer-reason=%s\n",
                     static_cast<unsigned long long>(selector_.producer_hash),
                     static_cast<unsigned long long>(selector_.consumer_hash),
                     static_cast<unsigned long long>(counters_.seen),
                     static_cast<unsigned long long>(counters_.producer_matches),
                     static_cast<unsigned long long>(counters_.consumer_matches),
                     static_cast<unsigned long long>(
                         counters_.producer_storage_gate_observations),
                     static_cast<unsigned long long>(
                         counters_.consumer_storage_gate_observations),
                     invalid ? "INVALID" : "matched",
                     compute_timing_selector_parse_error_name(producer_error_),
                     compute_timing_selector_parse_error_name(consumer_error_));
        report_role_summary(ComputeTransferGateRole::Producer, producer_stats_);
        report_role_summary(ComputeTransferGateRole::Consumer, consumer_stats_);
    }

private:
    ComputeTransferGateStats& stats_for(ComputeTransferGateRole role) {
        return role == ComputeTransferGateRole::Producer
            ? producer_stats_ : consumer_stats_;
    }

    static void report_role_summary(ComputeTransferGateRole role,
                                    const ComputeTransferGateStats& s) {
        std::fprintf(stderr,
                     "[compute-transfer-gates] summary role=%s reflected=%llu storage=%llu "
                     "sampled=%llu dim2d=%llu nonarrayed=%llu nonmsaa=%llu ordinary2d=%llu "
                     "storage-float=%llu native-semantic=%llu native-device=%llu "
                     "storage-cache-eval=%llu renderer-owned=%llu "
                     "dcc-cache-safe=%llu poison-verify=%llu exact-storage=%llu "
                     "seed-skip=%llu persistent-enabled=%llu storage-candidate=%llu "
                     "storage-persistent-setup=%llu publish-eval=%llu publish-native=%llu "
                     "publish-unique=%llu publish-candidate=%llu publish-persistent=%llu "
                     "publish-authorized=%llu\n",
                     compute_transfer_gate_role_name(role),
                     static_cast<unsigned long long>(s.reflected_images),
                     static_cast<unsigned long long>(s.reflected_storage),
                     static_cast<unsigned long long>(s.reflected_sampled),
                     static_cast<unsigned long long>(s.reflected_dim_2d),
                     static_cast<unsigned long long>(s.reflected_nonarrayed),
                     static_cast<unsigned long long>(s.reflected_nonmsaa),
                     static_cast<unsigned long long>(s.ordinary_2d),
                     static_cast<unsigned long long>(s.storage_float),
                     static_cast<unsigned long long>(s.storage_native_semantic),
                     static_cast<unsigned long long>(s.storage_native_device),
                     static_cast<unsigned long long>(s.storage_cache_evaluated),
                     static_cast<unsigned long long>(s.storage_renderer_owned),
                     static_cast<unsigned long long>(s.storage_dcc_cache_safe),
                     static_cast<unsigned long long>(s.storage_poison_verify),
                     static_cast<unsigned long long>(s.storage_exact),
                     static_cast<unsigned long long>(s.storage_seed_skip),
                     static_cast<unsigned long long>(s.storage_persistent_enabled),
                     static_cast<unsigned long long>(s.storage_cache_candidate),
                     static_cast<unsigned long long>(s.storage_persistent_setup),
                     static_cast<unsigned long long>(s.storage_publish_evaluated),
                     static_cast<unsigned long long>(s.storage_publish_native),
                     static_cast<unsigned long long>(s.storage_publish_unique),
                     static_cast<unsigned long long>(s.storage_publish_candidate),
                     static_cast<unsigned long long>(s.storage_publish_persistent),
                     static_cast<unsigned long long>(s.storage_publish_authorized));
        std::fprintf(stderr,
                     "[compute-transfer-gates] summary role=%s sampled-gate-eval=%llu "
                     "sampled-candidate=%llu sampled-persistent-setup=%llu "
                     "sampled-ordinary2d=%llu format-compatible=%llu "
                     "transfer-dimension=%llu hostless=%llu format-match=%llu "
                     "validation=%llu native-defined=%llu borrow-attempts=%llu hits=%llu "
                     "no-cache=%llu invalid-cache=%llu authority-changed=%llu\n",
                     compute_transfer_gate_role_name(role),
                     static_cast<unsigned long long>(s.sampled_gate_evaluated),
                     static_cast<unsigned long long>(s.sampled_cache_candidate),
                     static_cast<unsigned long long>(s.sampled_persistent_setup),
                     static_cast<unsigned long long>(s.sampled_ordinary_2d),
                     static_cast<unsigned long long>(s.sampled_format_compatible),
                     static_cast<unsigned long long>(s.sampled_transfer_dimension),
                     static_cast<unsigned long long>(s.sampled_hostless),
                     static_cast<unsigned long long>(s.sampled_format_match),
                     static_cast<unsigned long long>(s.sampled_validation_enabled),
                     static_cast<unsigned long long>(s.sampled_native_defined),
                     static_cast<unsigned long long>(s.borrow_attempts),
                     static_cast<unsigned long long>(s.borrow_hits),
                     static_cast<unsigned long long>(s.borrow_no_cache),
                     static_cast<unsigned long long>(s.borrow_invalid_cache),
                     static_cast<unsigned long long>(s.borrow_authority_changed));
    }

    ComputeTransferGateSelector selector_;
    ComputeTransferGateSelectorCounters counters_;
    ComputeTimingSelectorParseError producer_error_ =
        ComputeTimingSelectorParseError::Unset;
    ComputeTimingSelectorParseError consumer_error_ =
        ComputeTimingSelectorParseError::Unset;
    ComputeTransferGateStats producer_stats_;
    ComputeTransferGateStats consumer_stats_;
    std::mutex mutex_;
};

RuntimeComputeTransferGateCensus& runtime_compute_transfer_gate_census() {
    static RuntimeComputeTransferGateCensus census;
    return census;
}

const char* compute_authority_action_name(ShadowComputeAuthorityAction action) {
    switch (action) {
        case ShadowComputeAuthorityAction::NoPendingResult: return "no-pending";
        case ShadowComputeAuthorityAction::TrackPendingResult: return "track";
        case ShadowComputeAuthorityAction::ReplacePendingResult: return "replace";
        case ShadowComputeAuthorityAction::KeepGpuAuthority: return "keep-gpu";
        case ShadowComputeAuthorityAction::MaterializeGuestMirror: return "materialize";
        case ShadowComputeAuthorityAction::MaterializeAndTrackResult:
            return "materialize+track";
        case ShadowComputeAuthorityAction::RejectResult: return "reject";
    }
    return "unknown";
}

const char* compute_authority_reason_name(ShadowComputeAuthorityReason reason) {
    switch (reason) {
        case ShadowComputeAuthorityReason::NoPendingResult: return "no-pending";
        case ShadowComputeAuthorityReason::ResultAdmitted: return "result-admitted";
        case ShadowComputeAuthorityReason::ResultReplaced: return "result-replaced";
        case ShadowComputeAuthorityReason::ProvenGpuConsumer: return "proven-gpu-consumer";
        case ShadowComputeAuthorityReason::UnrelatedConsumer: return "unrelated-consumer";
        case ShadowComputeAuthorityReason::OverlappingGuestImageConsumer:
            return "guest-image-overlap";
        case ShadowComputeAuthorityReason::OverlappingRawBufferConsumer:
            return "raw-buffer-overlap";
        case ShadowComputeAuthorityReason::OverlappingDrawConsumer: return "draw-overlap";
        case ShadowComputeAuthorityReason::OverlappingDmaConsumer: return "dma-overlap";
        case ShadowComputeAuthorityReason::OverlappingOrderedMemoryEffectConsumer:
            return "ordered-memory-effect-overlap";
        case ShadowComputeAuthorityReason::CaptureConsumer: return "capture";
        case ShadowComputeAuthorityReason::UnknownConsumerRange: return "unknown-range";
        case ShadowComputeAuthorityReason::UnknownConsumer: return "unknown-consumer";
        case ShadowComputeAuthorityReason::SubmitEnd: return "submit-end";
        case ShadowComputeAuthorityReason::ResultRangeChanged: return "result-range-changed";
        case ShadowComputeAuthorityReason::InvalidResultRange: return "invalid-result-range";
    }
    return "unknown";
}

const char* compute_authority_boundary_name(
        prosper::gpu::ComputeAuthorityBoundaryKind kind) {
    using Kind = prosper::gpu::ComputeAuthorityBoundaryKind;
    switch (kind) {
        case Kind::SubmitBegin: return "submit-begin";
        case Kind::Draw: return "draw";
        case Kind::DrawResource: return "draw-resource";
        case Kind::DrawResourceEnd: return "draw-resource-end";
        case Kind::Dma: return "dma";
        case Kind::OrderedMemoryEffect: return "ordered-memory-effect";
        case Kind::Capture: return "capture";
        case Kind::SubmitEnd: return "submit-end";
        case Kind::Compute: return "compute";
    }
    return "unknown";
}

class RuntimeComputeAuthorityCensus {
public:
    RuntimeComputeAuthorityCensus()
        : census_(parse_compute_authority_live_selector(
              std::getenv("PROSPER_COMPUTE_AUTHORITY_HASH"))) {
        const ComputeAuthorityLiveSelector& selector = census_.selector();
        if (!selector.requested) return;
        if (selector.valid) {
            std::fprintf(stderr,
                         "[compute-authority] PROSPER_COMPUTE_AUTHORITY_HASH accepted "
                         "producer=0x%016llx mode=exact-stable-hash shadow-only\n",
                         static_cast<unsigned long long>(selector.producer_hash));
        } else {
            std::fprintf(stderr,
                         "[compute-authority] PROSPER_COMPUTE_AUTHORITY_HASH ignored "
                         "reason=%s; selector fails closed\n",
                         compute_timing_selector_parse_error_name(selector.error));
        }
    }

    ~RuntimeComputeAuthorityCensus() { report_summary(); }

    bool requested() const { return census_.selector().requested; }

    ComputeAuthorityLiveObservation observe_program(
            const prosper::gpu::ComputeItem& item, uint64_t program_hash) {
        if (!requested()) return {};
        std::lock_guard lock(mutex_);
        verify_submit_locked(item.submit_no, "dispatch");
        const ComputeAuthorityLiveObservation observation =
            census_.observe_program(program_hash);
        if (observation.first_match) {
            std::fprintf(stderr,
                         "[compute-authority] first-match producer=0x%016llx "
                         "code=0x%llx submit=%llu dispatch=%llu order=%llu seen=%llu\n",
                         static_cast<unsigned long long>(program_hash),
                         static_cast<unsigned long long>(item.code_addr),
                         static_cast<unsigned long long>(item.submit_no),
                         static_cast<unsigned long long>(item.dispatch_index),
                         static_cast<unsigned long long>(item.command_order),
                         static_cast<unsigned long long>(
                             census_.counters().programs_seen));
        }
        return observation;
    }

    void observe_compute_access(const prosper::gpu::ComputeItem& item,
                                uint32_t binding,
                                ShadowComputeAuthorityConsumerKind kind,
                                const ShadowComputeAuthorityRange& range,
                                const char* stage) {
        if (!requested()) return;
        std::lock_guard lock(mutex_);
        verify_submit_locked(item.submit_no, stage);
        const ShadowComputeAuthorityTransition transition = census_.observe(kind, range);
        record_detail_locked(stage, item.submit_no, item.command_order, binding,
                             range, transition);
    }

    void observe_compute_image_access(
            const prosper::gpu::ComputeItem& item,
            const ShadowComputeAuthorityRange& range,
            const ComputeAuthorityImageSourceFacts& facts) {
        if (!requested()) return;
        const ComputeAuthorityImageSourceDecision decision =
            classify_compute_authority_image_source(facts);
        if (!decision.observe) return;
        std::lock_guard lock(mutex_);
        verify_submit_locked(item.submit_no, "compute-image-input");
        const ShadowComputeAuthorityTransition transition = census_.observe(
            decision.consumer_kind(), range);
        record_detail_locked(decision.proven_gpu ? "compute-image-input-gpu" :
                                                   "compute-image-input-guest",
                             item.submit_no, item.command_order, facts.binding,
                             range, transition);
        record_image_source_detail_locked(item, facts, decision, transition);
    }

    void record_selected_storage_output(
            const prosper::gpu::ComputeItem& item, uint32_t binding,
            const ShadowComputeAuthorityRange& range, bool retained) {
        if (!requested()) return;
        std::lock_guard lock(mutex_);
        verify_submit_locked(item.submit_no, "storage-output");
        const ShadowComputeAuthorityTransition transition =
            census_.record_selected_storage_output(range, retained);
        record_detail_locked(retained ? "retained-storage-output" :
                                      "synchronous-storage-output",
                             item.submit_no, item.command_order, binding,
                             range, transition);
    }

    void observe_boundary(const prosper::gpu::ComputeAuthorityBoundary& boundary) {
        if (!requested()) return;
        std::lock_guard lock(mutex_);
        using BoundaryKind = prosper::gpu::ComputeAuthorityBoundaryKind;
        boundary_counts_[static_cast<size_t>(boundary.kind)] =
            shadow_compute_authority_increment(
                boundary_counts_[static_cast<size_t>(boundary.kind)]);
        if (boundary.kind == BoundaryKind::SubmitBegin) {
            submit_draw_probe_.begin_submit(boundary.submit_no);
            census_.begin_submit(boundary.submit_no);
            return;
        }
        verify_submit_locked(boundary.submit_no,
                             compute_authority_boundary_name(boundary.kind));
        const ShadowComputeAuthorityRange range = boundary.range_known
            ? ShadowComputeAuthorityRange::from(boundary.address, boundary.bytes)
            : ShadowComputeAuthorityRange::unknown();
        if (boundary.kind == BoundaryKind::DrawResource) {
            const ShadowComputeAuthorityDrawProbeAction action =
                draw_probe_.observe_resource(
                    boundary.submit_no, boundary.command_order, range);
            if (action == ShadowComputeAuthorityDrawProbeAction::OverlappingRange) {
                draw_probe_overlap_events_ = shadow_compute_authority_increment(
                    draw_probe_overlap_events_);
                if (draw_probe_overlap_events_ <= 16 ||
                    (draw_probe_overlap_events_ & (draw_probe_overlap_events_ - 1)) == 0) {
                    draw_probe_overlap_detail_lines_ = shadow_compute_authority_increment(
                        draw_probe_overlap_detail_lines_);
                    std::fprintf(stderr,
                                 "[compute-authority] draw-resource-overlap n=%llu "
                                 "submit=%llu order=%llu binding=%u class=%u "
                                 "range=0x%llx+%llu\n",
                                 static_cast<unsigned long long>(
                                     draw_probe_overlap_events_),
                                 static_cast<unsigned long long>(boundary.submit_no),
                                 static_cast<unsigned long long>(boundary.command_order),
                                 boundary.binding, boundary.resource_class,
                                 static_cast<unsigned long long>(range.address),
                                 static_cast<unsigned long long>(range.bytes));
                }
            }
            const uint64_t draw_ordinal = submit_draw_probe_.next_draw_ordinal();
            const ShadowComputeAuthoritySubmitDrawProbeAction submit_action =
                submit_draw_probe_.observe_resource(
                    boundary.submit_no, boundary.command_order, range);
            if (submit_action ==
                    ShadowComputeAuthoritySubmitDrawProbeAction::OverlappingRange) {
                submit_draw_probe_overlap_events_ = shadow_compute_authority_increment(
                    submit_draw_probe_overlap_events_);
                if (submit_draw_probe_overlap_events_ <= 16 ||
                    (submit_draw_probe_overlap_events_ &
                     (submit_draw_probe_overlap_events_ - 1)) == 0) {
                    submit_draw_probe_overlap_detail_lines_ =
                        shadow_compute_authority_increment(
                            submit_draw_probe_overlap_detail_lines_);
                    std::fprintf(stderr,
                                 "[compute-authority] submit-draw-resource-overlap n=%llu "
                                 "submit=%llu draw-ordinal=%llu order=%llu binding=%u class=%u "
                                 "range=0x%llx+%llu\n",
                                 static_cast<unsigned long long>(
                                     submit_draw_probe_overlap_events_),
                                 static_cast<unsigned long long>(boundary.submit_no),
                                 static_cast<unsigned long long>(draw_ordinal),
                                 static_cast<unsigned long long>(boundary.command_order),
                                 boundary.binding, boundary.resource_class,
                                 static_cast<unsigned long long>(range.address),
                                 static_cast<unsigned long long>(range.bytes));
                }
            }
            return;
        }
        if (boundary.kind == BoundaryKind::DrawResourceEnd) {
            (void)draw_probe_.complete(
                boundary.submit_no, boundary.command_order, boundary.draw_realized);
            (void)submit_draw_probe_.complete_draw(
                boundary.submit_no, boundary.command_order, boundary.draw_realized);
            return;
        }
        ShadowComputeAuthorityTransition transition;
        switch (boundary.kind) {
            case BoundaryKind::Draw: {
                const ShadowComputeAuthorityRange pending_range =
                    census_.pending_range();
                transition = census_.observe(
                    ShadowComputeAuthorityConsumerKind::Draw, range);
                if (transition.pending_before &&
                    transition.reason ==
                        ShadowComputeAuthorityReason::UnknownConsumerRange)
                    draw_probe_.arm(
                        boundary.submit_no, boundary.command_order, pending_range);
                if (transition.pending_before &&
                    transition.reason ==
                        ShadowComputeAuthorityReason::UnknownConsumerRange)
                    submit_draw_probe_.arm(
                        boundary.submit_no, boundary.command_order, pending_range);
                break;
            }
            case BoundaryKind::Dma:
                transition = census_.observe(
                    ShadowComputeAuthorityConsumerKind::Dma, range);
                break;
            case BoundaryKind::OrderedMemoryEffect:
                transition = census_.observe(
                    ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect, range);
                break;
            case BoundaryKind::Capture:
                transition = census_.observe(
                    ShadowComputeAuthorityConsumerKind::Capture);
                break;
            case BoundaryKind::SubmitEnd:
                (void)submit_draw_probe_.end_submit(boundary.submit_no);
                transition = census_.end_submit();
                break;
            case BoundaryKind::Compute:
                transition = range.valid()
                    ? census_.observe(
                          ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect, range)
                    : census_.observe(
                          ShadowComputeAuthorityConsumerKind::Unknown);
                break;
            case BoundaryKind::DrawResource:
            case BoundaryKind::DrawResourceEnd:
            case BoundaryKind::SubmitBegin:
                return;
        }
        record_detail_locked(compute_authority_boundary_name(boundary.kind),
                             boundary.submit_no, boundary.command_order, UINT32_MAX,
                             range, transition);
    }

    void report_summary() {
        if (!requested()) return;
        std::lock_guard lock(mutex_);
        if (!census_.claim_summary()) return;
        using BoundaryKind = prosper::gpu::ComputeAuthorityBoundaryKind;
        const ComputeAuthorityLiveSelector& selector = census_.selector();
        const ComputeAuthorityLiveCounters& c = census_.counters();
        const ShadowComputeAuthorityCounters& a = c.authority;
        const char* verdict = !selector.valid ? "INVALID-selector-not-armed" :
            c.programs_matched == 0 ? "INVALID-zero-matches" :
            c.retained_storage_outputs == 0 || a.admitted_results == 0
                ? "INVALID-zero-authority-lever" :
            census_.active_submit() ? "INVALID-unfinished-submit" :
            !submit_draw_probe_.apparatus_valid()
                ? "INVALID-full-submit-draw-probe" :
            !c.apparatus_valid ? "INVALID-apparatus" :
            c.pending_after_submit_end != 0 ? "INVALID-pending-after-submit" : "matched";
        std::fprintf(stderr,
                     "[compute-authority] summary producer=0x%016llx reason=%s "
                     "seen=%llu matched=%llu submits=%llu/%llu interrupted=%llu "
                     "outside-submit=%llu active=%u selected-outputs=%llu retained=%llu "
                     "synchronous=%llu pending-before-end=%llu pending-after-end=%llu "
                     "verdict=%s\n",
                     static_cast<unsigned long long>(selector.producer_hash),
                     compute_timing_selector_parse_error_name(selector.error),
                     static_cast<unsigned long long>(c.programs_seen),
                     static_cast<unsigned long long>(c.programs_matched),
                     static_cast<unsigned long long>(c.submits_completed),
                     static_cast<unsigned long long>(c.submits_started),
                     static_cast<unsigned long long>(c.interrupted_submits),
                     static_cast<unsigned long long>(c.observations_without_submit),
                     census_.active_submit() ? 1u : 0u,
                     static_cast<unsigned long long>(c.selected_storage_outputs),
                     static_cast<unsigned long long>(c.retained_storage_outputs),
                     static_cast<unsigned long long>(c.synchronous_storage_outputs),
                     static_cast<unsigned long long>(c.pending_before_submit_end),
                     static_cast<unsigned long long>(c.pending_after_submit_end), verdict);
        std::fprintf(stderr,
                     "[compute-authority] summary results candidates=%llu admitted=%llu "
                     "replaced=%llu rejected=%llu observations=%llu gpu-keeps=%llu "
                     "unrelated-keeps=%llu materializations=%llu overlap=%llu "
                     "guest-image=%llu raw-buffer=%llu draw=%llu dma=%llu memory-effect=%llu "
                     "capture=%llu unknown=%llu submit-end=%llu unknown-range=%llu\n",
                     static_cast<unsigned long long>(a.result_candidates),
                     static_cast<unsigned long long>(a.admitted_results),
                     static_cast<unsigned long long>(a.replaced_results),
                     static_cast<unsigned long long>(a.rejected_results),
                     static_cast<unsigned long long>(a.consumer_observations),
                     static_cast<unsigned long long>(a.proven_gpu_keeps),
                     static_cast<unsigned long long>(a.unrelated_keeps),
                     static_cast<unsigned long long>(a.materializations),
                     static_cast<unsigned long long>(a.overlap_materializations),
                     static_cast<unsigned long long>(a.guest_image_materializations),
                     static_cast<unsigned long long>(a.raw_buffer_materializations),
                     static_cast<unsigned long long>(a.draw_materializations),
                     static_cast<unsigned long long>(a.dma_materializations),
                     static_cast<unsigned long long>(
                         a.ordered_memory_effect_materializations),
                     static_cast<unsigned long long>(a.capture_materializations),
                     static_cast<unsigned long long>(a.unknown_consumer_materializations),
                     static_cast<unsigned long long>(a.submit_end_materializations),
                     static_cast<unsigned long long>(a.unknown_range_materializations));
        std::fprintf(stderr,
                     "[compute-authority] summary boundaries begin=%llu draw=%llu "
                     "draw-resource=%llu draw-resource-end=%llu dma=%llu "
                     "memory-effect=%llu capture=%llu end=%llu compute=%llu "
                     "detail-events=%llu\n",
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::SubmitBegin)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::Draw)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::DrawResource)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::DrawResourceEnd)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::Dma)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::OrderedMemoryEffect)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::Capture)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::SubmitEnd)]),
                     static_cast<unsigned long long>(boundary_counts_[
                         static_cast<size_t>(BoundaryKind::Compute)]),
                     static_cast<unsigned long long>(detail_events_));
        const ShadowComputeAuthorityDrawProbeCounters& draw_probe =
            draw_probe_.counters();
        std::fprintf(stderr,
                     "[compute-authority] summary draw-probe armed=%llu completed=%llu "
                     "with-overlap=%llu without-overlap=%llu unrealized=%llu "
                     "resources=%llu overlaps=%llu unrelated=%llu invalid=%llu "
                     "superseded=%llu active=%u detail-lines=%llu\n",
                     static_cast<unsigned long long>(draw_probe.armed),
                     static_cast<unsigned long long>(draw_probe.completed),
                     static_cast<unsigned long long>(
                         draw_probe.completed_with_overlap),
                     static_cast<unsigned long long>(
                         draw_probe.completed_without_overlap),
                     static_cast<unsigned long long>(draw_probe.unrealized),
                     static_cast<unsigned long long>(
                         draw_probe.resource_observations),
                     static_cast<unsigned long long>(draw_probe.overlapping_ranges),
                     static_cast<unsigned long long>(draw_probe.unrelated_ranges),
                     static_cast<unsigned long long>(draw_probe.invalid_ranges),
                     static_cast<unsigned long long>(draw_probe.superseded),
                     draw_probe_.active() ? 1u : 0u,
                     static_cast<unsigned long long>(
                         draw_probe_overlap_detail_lines_));
        const ShadowComputeAuthoritySubmitDrawProbeCounters& submit_draw_probe =
            submit_draw_probe_.counters();
        std::fprintf(stderr,
                     "[compute-authority] summary submit-draw-probe armed=%llu "
                     "epochs=%llu with-overlap=%llu without-overlap=%llu "
                     "draws=%llu later-draws=%llu realized=%llu unrealized=%llu "
                     "resources=%llu overlaps=%llu unrealized-resources=%llu "
                     "unrealized-overlaps=%llu first-overlaps=%llu later-overlaps=%llu "
                     "unrelated=%llu invalid=%llu superseded=%llu interrupted=%llu "
                     "active=%u detail-lines=%llu\n",
                     static_cast<unsigned long long>(submit_draw_probe.armed),
                     static_cast<unsigned long long>(
                         submit_draw_probe.epochs_completed),
                     static_cast<unsigned long long>(
                         submit_draw_probe.epochs_with_overlap),
                     static_cast<unsigned long long>(
                         submit_draw_probe.epochs_without_overlap),
                     static_cast<unsigned long long>(
                         submit_draw_probe.draws_completed),
                     static_cast<unsigned long long>(
                         submit_draw_probe.later_draws_completed),
                     static_cast<unsigned long long>(
                         submit_draw_probe.realized_draws),
                     static_cast<unsigned long long>(
                         submit_draw_probe.unrealized_draws),
                     static_cast<unsigned long long>(
                         submit_draw_probe.resource_observations),
                     static_cast<unsigned long long>(
                         submit_draw_probe.overlapping_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.unrealized_resource_observations),
                     static_cast<unsigned long long>(
                         submit_draw_probe.unrealized_overlapping_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.first_draw_overlapping_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.later_draw_overlapping_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.unrelated_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.invalid_ranges),
                     static_cast<unsigned long long>(
                         submit_draw_probe.superseded),
                     static_cast<unsigned long long>(
                         submit_draw_probe.interrupted),
                     submit_draw_probe_.active() ? 1u : 0u,
                     static_cast<unsigned long long>(
                         submit_draw_probe_overlap_detail_lines_));
        std::fprintf(stderr,
                     "[compute-authority] summary image-sources selected-events=%llu "
                     "aliases=%llu direct-borrow=%llu owner-borrow=%llu "
                     "direct-persistent=%llu owner-persistent=%llu "
                     "direct-upload-skipped=%llu owner-upload-skipped=%llu "
                     "same-image=%llu same-view=%llu gpu=%llu guest=%llu detail-lines=%llu\n",
                     static_cast<unsigned long long>(image_sources_.observations),
                     static_cast<unsigned long long>(image_sources_.aliases),
                     static_cast<unsigned long long>(
                         image_sources_.direct_transfer_borrows),
                     static_cast<unsigned long long>(
                         image_sources_.owner_transfer_borrows),
                     static_cast<unsigned long long>(image_sources_.direct_persistent),
                     static_cast<unsigned long long>(image_sources_.owner_persistent),
                     static_cast<unsigned long long>(image_sources_.direct_upload_skips),
                     static_cast<unsigned long long>(image_sources_.owner_upload_skips),
                     static_cast<unsigned long long>(image_sources_.same_images),
                     static_cast<unsigned long long>(image_sources_.same_views),
                     static_cast<unsigned long long>(image_sources_.proven_gpu),
                     static_cast<unsigned long long>(image_sources_.guest),
                     static_cast<unsigned long long>(image_source_detail_lines_));
    }

private:
    void verify_submit_locked(uint64_t submit_no, const char* stage) {
        if (!census_.active_submit() || census_.submit_no() == submit_no) return;
        census_.invalidate_apparatus(true);
        if (!submit_mismatch_reported_) {
            submit_mismatch_reported_ = true;
            std::fprintf(stderr,
                         "[compute-authority] apparatus-invalid stage=%s active-submit=%llu "
                         "observed-submit=%llu; pending authority closed as unknown\n",
                         stage,
                         static_cast<unsigned long long>(census_.submit_no()),
                         static_cast<unsigned long long>(submit_no));
        }
    }

    void record_detail_locked(const char* stage, uint64_t submit_no,
                              uint64_t command_order, uint32_t binding,
                              const ShadowComputeAuthorityRange& range,
                              const ShadowComputeAuthorityTransition& transition) {
        if (!transition.pending_before && !transition.pending_after &&
            transition.action == ShadowComputeAuthorityAction::NoPendingResult)
            return;
        detail_events_ = shadow_compute_authority_increment(detail_events_);
        if (detail_events_ > 16 && (detail_events_ & (detail_events_ - 1)) != 0) return;
        std::fprintf(stderr,
                     "[compute-authority] detail n=%llu stage=%s submit=%llu order=%llu "
                     "binding=%s%u range=%s0x%llx+%llu action=%s reason=%s pending=%u->%u\n",
                     static_cast<unsigned long long>(detail_events_), stage,
                     static_cast<unsigned long long>(submit_no),
                     static_cast<unsigned long long>(command_order),
                     binding == UINT32_MAX ? "-" : "",
                     binding == UINT32_MAX ? 0u : binding,
                     range.valid() ? "" : "unknown:",
                     static_cast<unsigned long long>(range.address),
                     static_cast<unsigned long long>(range.bytes),
                     compute_authority_action_name(transition.action),
                     compute_authority_reason_name(transition.reason),
                     transition.pending_before ? 1u : 0u,
                     transition.pending_after ? 1u : 0u);
    }

    void record_image_source_detail_locked(
            const prosper::gpu::ComputeItem& item,
            const ComputeAuthorityImageSourceFacts& facts,
            const ComputeAuthorityImageSourceDecision& decision,
            const ShadowComputeAuthorityTransition& transition) {
        if (!transition.pending_before && !transition.pending_after &&
            transition.action == ShadowComputeAuthorityAction::NoPendingResult)
            return;
        record_compute_authority_image_source(image_sources_, facts, decision);
        const uint64_t n = image_sources_.observations;
        if (n > 16 && (n & (n - 1)) != 0) return;
        image_source_detail_lines_ = shadow_compute_authority_increment(
            image_source_detail_lines_);
        std::fprintf(stderr,
                     "[compute-authority] image-source n=%llu submit=%llu order=%llu "
                     "binding=%u alias=%u owner-binding=%u "
                     "direct-borrow=%u owner-borrow=%u "
                     "direct-persistent=%u owner-persistent=%u "
                     "direct-upload-skipped=%u owner-upload-skipped=%u "
                     "same-image=%u same-view=%u class=%s\n",
                     static_cast<unsigned long long>(n),
                     static_cast<unsigned long long>(item.submit_no),
                     static_cast<unsigned long long>(item.command_order),
                     facts.binding, facts.alias ? 1u : 0u, facts.owner_binding,
                     facts.direct_transfer_borrowed ? 1u : 0u,
                     facts.owner_transfer_borrowed ? 1u : 0u,
                     facts.direct_persistent ? 1u : 0u,
                     facts.owner_persistent ? 1u : 0u,
                     facts.direct_upload_skipped ? 1u : 0u,
                     facts.owner_upload_skipped ? 1u : 0u,
                     facts.same_image ? 1u : 0u, facts.same_view ? 1u : 0u,
                     decision.proven_gpu ? "gpu" : "guest");
    }

    ComputeAuthorityLiveCensus census_;
    std::array<uint64_t,
               static_cast<size_t>(
                   prosper::gpu::ComputeAuthorityBoundaryKind::Compute) + 1>
        boundary_counts_{};
    ShadowComputeAuthorityDrawProbe draw_probe_;
    ShadowComputeAuthoritySubmitDrawProbe submit_draw_probe_;
    ComputeAuthorityImageSourceCounters image_sources_{};
    uint64_t detail_events_ = 0;
    uint64_t draw_probe_overlap_events_ = 0;
    uint64_t draw_probe_overlap_detail_lines_ = 0;
    uint64_t submit_draw_probe_overlap_events_ = 0;
    uint64_t submit_draw_probe_overlap_detail_lines_ = 0;
    uint64_t image_source_detail_lines_ = 0;
    bool submit_mismatch_reported_ = false;
    std::mutex mutex_;
};

RuntimeComputeAuthorityCensus& runtime_compute_authority_census() {
    static RuntimeComputeAuthorityCensus census;
    return census;
}

void maybe_dump_traced_compute_spirv(const prosper::gpu::ComputeItem& item, bool trace) {
    const char* path = std::getenv("PROSPER_COMPUTELOG_SPIRV");
    if (!trace || !path || !*path || item.spirv.empty()) return;
    uint64_t module_hash = 1469598103934665603ull;
    for (uint32_t word : item.spirv) {
        module_hash ^= word;
        module_hash *= 1099511628211ull;
    }
    static std::mutex mutex;
    static std::unordered_set<uint64_t> dumped;
    std::lock_guard lock(mutex);
    if (dumped.contains(module_hash)) return;
    FILE* file = std::fopen(path, "wb");
    const size_t bytes = item.spirv.size() * sizeof(uint32_t);
    const bool ok = file && std::fwrite(item.spirv.data(), 1, bytes, file) == bytes;
    if (file) std::fclose(file);
    if (ok) dumped.insert(module_hash);
    std::fprintf(stderr,
                 "[compute]   traced SPIR-V program=0x%llx module=%016llx words=%zu "
                 "path=%s result=%s\n",
                 static_cast<unsigned long long>(item.code_addr),
                 static_cast<unsigned long long>(module_hash), item.spirv.size(), path,
                 ok ? "written" : "failed");
}

std::optional<bool> execute_cpu_fast_path(const prosper::gpu::ComputeItem& item) {
    using prosper::gpu::ComputeCpuFastPath;
    using prosper::gpu::ResourceClass;
    if (item.cpu_fast_path == ComputeCpuFastPath::None) return std::nullopt;
    if (item.cpu_fast_path != ComputeCpuFastPath::FillSgprUvec4 ||
        !item.resources || item.resources->resources.size() != 1 ||
        item.user_sgprs.size() != 8 || !item.recompile_config_available ||
        !item.recompile_config.tgid_x_en || item.recompile_config.tgid_y_en ||
        item.recompile_config.tgid_z_en ||
        item.launch.local_x != 64 || item.launch.local_y != 1 || item.launch.local_z != 1 ||
        !item.launch.groups_x || item.launch.groups_y != 1 || item.launch.groups_z != 1 ||
        item.launch.threads_y != 1 || item.launch.threads_z != 1)
        return std::nullopt;

    const prosper::gpu::ShaderResource* resource = item.resources->by_binding(2);
    // The matched MUBUF instruction uses idxen with the direct V# in s[0:3]. Its element address and
    // store conversion therefore still depend on the live descriptor: only the observed 16-byte,
    // four-dword contract is equivalent to the raw record writes below. Similar shaders bound to a
    // different descriptor must retain the normal translator/backend semantics.
    if (!resource || !resource->size || resource->cls != ResourceClass::ConstantBuffer ||
        resource->sgpr_base != 0 || item.resources->by_sgpr_base(0) != resource ||
        resource->stride != 16 ||
        resource->num_components != 4 ||
        prosper::gpu::data_format_bytes(resource->format) != sizeof(uint32_t))
        return std::nullopt;

    const uint64_t dispatched_records =
        static_cast<uint64_t>(item.launch.groups_x) * item.launch.local_x;
    const uint64_t records = item.launch.threads_x
        ? std::min<uint64_t>(item.launch.threads_x, dispatched_records)
        : dispatched_records;
    if (!records || records > UINT64_MAX / 16u) return std::nullopt;
    const uint64_t written_bytes = records * 16u;
    if (written_bytes > resource->size || written_bytes > SIZE_MAX) return std::nullopt;

    uint8_t* destination = nullptr;
    if (resource->host_data && resource->host_data_size >= written_bytes) {
        destination = resource->host_data;
    } else if (resource->gpu_addr && written_bytes <= UINT32_MAX &&
               prosper::gpu::guest_writable(
                   resource->gpu_addr, static_cast<uint32_t>(written_bytes))) {
        destination = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(resource->gpu_addr));
    }
    if (!destination) return std::nullopt;

    // The CPU shortcut bypasses execute_item's reflected-resource census entirely. Publish its
    // exact write boundary before touching bytes so pending storage authority cannot survive an
    // overlapping fast fill. With the diagnostic unarmed this is the same single atomic check as
    // other executor boundaries; no hashing, lock, or callback copy occurs.
    const bool authority_range_known = resource->gpu_addr != 0 && written_bytes != 0 &&
        resource->gpu_addr <= UINT64_MAX - (written_bytes - 1);
    prosper::gpu::notify_compute_authority_boundary({
        prosper::gpu::ComputeAuthorityBoundaryKind::Compute,
        item.submit_no, item.command_order, resource->gpu_addr, written_bytes,
        authority_range_known});

    if (!resource->host_data && resource->gpu_addr)
        prosper::host::guest_write_watch_notify_host_write(
            resource->gpu_addr, static_cast<size_t>(written_bytes));
    const uint32_t pattern[4] = {
        item.user_sgprs[4], item.user_sgprs[5], item.user_sgprs[6], item.user_sgprs[7]};
    if (!(pattern[0] | pattern[1] | pattern[2] | pattern[3])) {
        std::memset(destination, 0, static_cast<size_t>(written_bytes));
    } else {
        for (uint64_t record = 0; record < records; ++record)
            std::memcpy(destination + record * sizeof(pattern), pattern, sizeof(pattern));
    }
    // Match the Vulkan path's conservative invalidation contract: padding beyond the exact launch
    // remains untouched, but every alias of the declared resource must be considered stale.
    if (resource->gpu_addr)
        prosper::gpu::notify_guest_gpu_write(resource->gpu_addr, resource->size);
    if (!resource->host_data && prosper::gpu::writer_provenance_enabled())
        prosper::gpu::record_guest_write(
            prosper::gpu::GuestWriterKind::ComputeBuffer,
            resource->gpu_addr, resource->size, item.submit_no, item.dispatch_index,
            item.command_order, item.code_addr);
    g_cpu_fill_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (trace_compute_item(item))
        std::fprintf(stderr,
                     "[compute] CPU fill program 0x%llx records=%llu bytes=%llu\n",
                     static_cast<unsigned long long>(item.code_addr),
                     static_cast<unsigned long long>(records),
                     static_cast<unsigned long long>(written_bytes));
    return true;
}

bool execute_item(VulkanComputeContext& ctx, const prosper::gpu::ComputeItem& item) {
    using namespace prosper::gpu;
    using ComputeClock = std::chrono::steady_clock;
    const auto phase_start = ComputeClock::now();
    auto phase_setup = phase_start;
    auto phase_pipeline = phase_start;
    auto phase_dispatch = phase_start;
    auto phase_writeback = phase_start;
    double pack_ms = 0.0;
    double layout_ms = 0.0;
    double writeback_prepare_ms = 0.0;
    double writeback_buffers_ms = 0.0;
    double writeback_images_ms = 0.0;
    double writeback_publish_ms = 0.0;
    double image_map_ms = 0.0;
    double image_prepare_ms = 0.0;
    double image_watch_ms = 0.0;
    double image_notify_ms = 0.0;
    double image_cache_ms = 0.0;
    const std::vector<uint32_t>& spirv = item.spirv;
    const bool trace = trace_compute_item(item);
    maybe_dump_traced_compute_spirv(item, trace);
    if (item.required_subgroup_size &&
        (!ctx.borrowed || !ctx.native_subgroup_contract ||
         item.required_subgroup_size < ctx.min_native_subgroup_size ||
         item.required_subgroup_size > ctx.max_native_subgroup_size)) {
        std::fprintf(stderr,
                     "[compute] program 0x%llx requires subgroup=%u on a context without "
                     "that enabled contract -> dispatch skipped\n",
                     (unsigned long long)item.code_addr, item.required_subgroup_size);
        return false;
    }
    const uint32_t dispatch_groups[3] = {
        item.launch.groups_x, item.launch.groups_y, item.launch.groups_z};
    for (uint32_t axis = 0; axis < 3; ++axis) {
        if (dispatch_groups[axis] &&
            dispatch_groups[axis] <= ctx.max_compute_workgroup_count[axis])
            continue;
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[compute] program 0x%llx dispatch=%ux%ux%u exceeds device "
                         "workgroup-count limit=%ux%ux%u -> dispatch skipped\n",
                         (unsigned long long)item.code_addr,
                         item.launch.groups_x, item.launch.groups_y, item.launch.groups_z,
                         ctx.max_compute_workgroup_count[0],
                         ctx.max_compute_workgroup_count[1],
                         ctx.max_compute_workgroup_count[2]);
        return false;
    }
    const uint64_t image_validation_epoch = ++ctx.image_validation_clock;
    const uint32_t min_subgroup = compute_spirv_min_subgroup_size(item.spirv);
    const uint32_t effective_subgroup = item.required_subgroup_size
        ? item.required_subgroup_size : ctx.subgroup_size;
    if (min_subgroup &&
        (effective_subgroup < min_subgroup ||
         !(ctx.subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT) ||
         !(ctx.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT))) {
        std::fprintf(stderr,
                     "[compute] skip program 0x%llx: lane operation requires a compute subgroup "
                     "of at least %u lanes (host=%u stages=0x%x operations=0x%x)\n",
                     static_cast<unsigned long long>(item.code_addr), min_subgroup,
                     effective_subgroup, ctx.subgroup_stages, ctx.subgroup_operations);
        return false;
    }
    // #1122 review B1: covers_extent (dispatch grid >= image extent) is NECESSARY but NOT SUFFICIENT
    // for skipping the seed. A write-only shader can store a subset of its grid (a masked composite:
    // `if (mask) imageStore(...)`, or a scatter store), covering the grid yet leaving untouched texels
    // undefined. Skipping the seed there would pack reused-pool garbage to the guest -- silent
    // corruption. So prove full coverage ONCE per (shader, binding, extent): seed the image with
    // poison, confirm the write overwrites every texel (0 poison survives), cache the verdict, and
    // only then fast-skip. The proving frame itself stays correct -- untouched texels are restored
    // from the seed (see the poison_verify writeback path).
    //
    // The cached verdict is trusted for a DATA-INDEPENDENT store pattern (unconditional per-gid store,
    // or a store bounded by gid vs a constant/the image extent) -- the exercised full-screen composites
    // are exactly this. The extent is part of the key, so the same shader reused for a larger target
    // re-proves (a hard-coded store bound cannot silently under-cover a bigger extent). What this key
    // does NOT catch is a store whose predicate depends on per-frame INPUT (`if (buffer[gid] > k)
    // store`) that is full on the proving frame but partial later; that residual soundness gap is
    // tracked in #1127 -- no exercised title shader triggers it, and a shader first seen partial is
    // cached Partial and always seeds (safe).
    enum class SeedCoverage : uint8_t { Full, Partial };
    // #1127: 'prove once, trust forever' is unsound for a DATA-DEPENDENT store (full on the proving
    // frame, partial later). Re-prove a Full verdict every kSeedReproveInterval fast-skips: a shader
    // that ever covers partially is then re-cached Partial and always seeds, bounding the corruption
    // window from unbounded to <= interval fast-skips. A genuinely data-independent full-writer (the
    // exercised full-screen composites) re-proves to Full each time -- no rendered-output change, ~1
    // extra poison frame per interval. skips counts fast-skips taken since the last (re-)proof.
    struct SeedVerdict { SeedCoverage cov = SeedCoverage::Partial; uint32_t skips = 0; };
    // key = (shader code_addr, output binding, width, height, depth) -- collision-free by construction.
    using SeedCoverageKey = std::tuple<uint64_t, uint32_t, uint32_t, uint32_t, uint32_t>;
    static std::mutex seed_coverage_mu;
    static std::map<SeedCoverageKey, SeedVerdict> seed_coverage_proof;
    // PROSPER_SEED_REPROVE=N: re-prove every N fast-skips (default 256; explicit 0 disables = old
    // prove-once). Parsed fail-safe -- garbage/overflow keeps the 256 default rather than silently
    // disabling the safety (see seed_reprove_interval_from_env).
    static const uint32_t kSeedReproveInterval =
        seed_reprove_interval_from_env(std::getenv("PROSPER_SEED_REPROVE"), 256u);
    // #1122: a compute post-process that only image_stores an output never reads that output's seed.
    // Reflection tracks image access per descriptor: a shader may read several input storage images
    // while binding a distinct write-only output, so the old shader-wide OpImageRead test needlessly
    // seeded the output (66 MiB/frame for Astro Bot's 4K FP16 post target).
    const bool seed_skip_enabled = std::getenv("PROSPER_NO_SKIP_SEED") == nullptr;
    const auto setup_validate_start = ComputeClock::now();
    auto report = validate_spirv_descriptor_interface(
        spirv, item.resources.get(), 0, SpirvShaderStage::Compute, false);
    if (!report.ok()) return false;
    double setup_validate_ms = std::chrono::duration<double, std::milli>(
        ComputeClock::now() - setup_validate_start).count();
    double setup_buffers_ms = 0.0;

    std::vector<SpirvDescriptorBinding> descriptors;       // storage buffers
    std::vector<SpirvDescriptorBinding> image_descriptors; // sampled + storage images (#590)
    for (const auto& descriptor : report.descriptors) {
        switch (descriptor.kind) {
            case SpirvDescriptorKind::StorageBuffer:
                descriptors.push_back(descriptor);
                break;
            case SpirvDescriptorKind::CombinedImageSampler:
            case SpirvDescriptorKind::StorageImage:
                image_descriptors.push_back(descriptor);
                break;
            default:
                std::fprintf(stderr, "[compute] program 0x%llx uses unsupported %s binding %u\n",
                             (unsigned long long)item.code_addr,
                             spirv_descriptor_kind_name(descriptor.kind), descriptor.binding);
                return false;
        }
    }
    if (descriptors.empty() && image_descriptors.empty()) return false;
    const bool has_storage_images = std::any_of(
        image_descriptors.begin(), image_descriptors.end(), [](const auto& descriptor) {
            return descriptor.kind == SpirvDescriptorKind::StorageImage;
        });
    const bool phase_timing_requested =
        std::getenv("PROSPER_COMPUTE_PHASE_TIMING") != nullptr;
    const bool image_timing_requested =
        std::getenv("PROSPER_COMPUTE_IMAGE_TIMING") != nullptr;
    // Preserve the cheap timing address pre-filter. The transfer gate census deliberately hashes
    // every reflected compute candidate because its two stable hashes have no address constraint.
    uint64_t timing_program_hash = 0;
    bool timing_item_selected = false;
    RuntimeComputeTransferGateCensus& transfer_gate_census =
        runtime_compute_transfer_gate_census();
    const bool transfer_gate_requested = transfer_gate_census.requested();
    RuntimeComputeAuthorityCensus& authority_census =
        runtime_compute_authority_census();
    const bool authority_requested = authority_census.requested();
    const bool timing_address_matches = time_compute_address_matches(item);
    if (transfer_gate_requested || authority_requested ||
        ((phase_timing_requested || image_timing_requested) && timing_address_matches)) {
        timing_program_hash = gpu_capture_hash(
            reinterpret_cast<const uint8_t*>(spirv.data()),
            spirv.size() * sizeof(uint32_t));
    }
    if ((phase_timing_requested || image_timing_requested) && timing_address_matches) {
        timing_item_selected =
            runtime_compute_timing_selector().matches(item, timing_program_hash);
    }
    const ComputeTransferGateSelectorObservation transfer_gate_observation =
        transfer_gate_requested
            ? transfer_gate_census.observe(item, timing_program_hash)
            : ComputeTransferGateSelectorObservation{};
    const ComputeAuthorityLiveObservation authority_observation =
        authority_requested
            ? authority_census.observe_program(item, timing_program_hash)
            : ComputeAuthorityLiveObservation{};
    // The phase timer is also useful for buffer-only kernels. Astro Bot's heaviest dispatcher writes
    // buffers exclusively, so the historical storage-image gate hid the actual frame bottleneck.
    const bool timing_trace_only =
        std::getenv("PROSPER_COMPUTE_TIMING_TRACE_ONLY") != nullptr;
    const bool phase_timing =
        phase_timing_requested && timing_item_selected &&
        (!timing_trace_only || trace);
    if (has_storage_images && !ctx.image_support) {
        static bool warned = false;
        if (!warned) { warned = true;
            std::fprintf(stderr, "[compute] device lacks shaderStorageImageRead/WriteWithoutFormat; "
                                 "image-binding dispatches are skipped\n"); }
        return false;
    }
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });
    std::sort(image_descriptors.begin(), image_descriptors.end(), [](const auto& a, const auto& b) {
        return a.binding < b.binding;
    });

    std::vector<BoundBuffer> buffers(descriptors.size());
    std::vector<BoundImage> images(image_descriptors.size());
    std::vector<VkBuffer> staging(image_descriptors.size(), VK_NULL_HANDLE);          // upload/readback
    std::vector<VkDeviceMemory> staging_memory(image_descriptors.size(), VK_NULL_HANDLE);
    std::vector<VkDeviceSize> staging_bytes(image_descriptors.size(), 0);
    VkBuffer compare_flags = VK_NULL_HANDLE;
    VkDeviceMemory compare_flags_memory = VK_NULL_HANDLE;
    VkDescriptorPool compare_descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> compare_descriptor_sets;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    bool descriptor_pool_reused = false;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool pipeline_cached = false;
    std::string pipeline_key;
    bool ok = false;
    bool submission_entered = false;
    bool completion_proven = false;
    auto vk_ok = [&](VkResult result, const char* stage) {
        if (result == VK_SUCCESS) return true;
        if (result == VK_ERROR_DEVICE_LOST && !ctx.device_lost) {
            ctx.device_lost = true;
            std::fprintf(stderr,
                         "[compute] fatal Vulkan device loss stage=%s "
                         "result=VK_ERROR_DEVICE_LOST(%d) program=0x%llx submit=%llu "
                         "dispatch=%llu order=%llu; disabling live compute for this process\n",
                         stage, static_cast<int>(result),
                         static_cast<unsigned long long>(item.code_addr),
                         static_cast<unsigned long long>(item.submit_no),
                         static_cast<unsigned long long>(item.dispatch_index),
                         static_cast<unsigned long long>(item.command_order));
        }
        if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=%s result=%d\n",
                                stage, static_cast<int>(result));
        return false;
    };
    auto vk_handle_ok = [&](auto handle, const char* stage) {
        if (handle != VK_NULL_HANDLE) return true;
        if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=%s result=null-handle\n", stage);
        return false;
    };

    auto cleanup = [&] {
        if (compare_descriptor_pool)
            vkDestroyDescriptorPool(ctx.device, compare_descriptor_pool, nullptr);
        if (compare_flags) vkDestroyBuffer(ctx.device, compare_flags, nullptr);
        if (compare_flags_memory) ctx.release_memory(compare_flags_memory);
        if (!pipeline_cached) {
            if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
            if (shader) vkDestroyShaderModule(ctx.device, shader, nullptr);
            if (descriptor_layout)
                vkDestroyDescriptorSetLayout(ctx.device, descriptor_layout, nullptr);
        }
        if (descriptor_pool && !descriptor_pool_reused)
            vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        for (auto& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX) continue;
            if (buffer.persistent) {
                // A failed command/submit/readback may have modified the retained host-visible
                // buffer without publishing matching guest bytes. Force exact source validation on
                // its next use rather than trusting the pre-dispatch write snapshot.
                if (!ok) ctx.invalidate_cached_buffer_source(buffer.cache_key);
                ctx.release_cached_buffer(buffer.cache_key);
                continue;
            }
            if (buffer.buffer) vkDestroyBuffer(ctx.device, buffer.buffer, nullptr);
            if (buffer.memory) ctx.release_memory(buffer.memory);
        }
        for (size_t i = 0; i < images.size(); i++) {
            // A pin is taken per successful import, so it is released per import -- including for a
            // binding that a later alias check folded into an earlier one (#1095).
            if (images[i].imported || images[i].depth_bits_source)
                release_live_render_target_image(images[i].imported_addr);
            if (images[i].alias_of != SIZE_MAX) continue;
            if (images[i].sampler) vkDestroySampler(ctx.device, images[i].sampler, nullptr);
            if (images[i].view) vkDestroyImageView(ctx.device, images[i].view, nullptr);
            if (images[i].compute_transfer_seed_borrowed)
                ctx.release_cached_image(images[i].compute_transfer_seed_key);
            if (images[i].persistent) {
                // A failed submit/readback can leave the retained VkImage and its exact-result
                // baseline newer than guest memory. Force the retry to upload and publish again;
                // otherwise it could compare against that newer baseline and skip forever.
                if (!ok) ctx.invalidate_cached_image_source(images[i].cache_key);
                ctx.release_cached_image(images[i].cache_key);
            } else {
            // An imported image belongs to the live renderer: release the pin, destroy nothing.
                if (images[i].image && !images[i].imported)
                    vkDestroyImage(ctx.device, images[i].image, nullptr);
                if (images[i].memory) ctx.release_memory(images[i].memory);
            }
            if (staging[i]) vkDestroyBuffer(ctx.device, staging[i], nullptr);
            if (staging_memory[i]) ctx.release_memory(staging_memory[i]);
        }
    };
    auto resource_bytes = [](const ShaderResource* resource) -> uint8_t* {
        if (resource->host_data && resource->host_data_size >= resource->size)
            return resource->host_data;
        return reinterpret_cast<uint8_t*>(uintptr_t(resource->gpu_addr));
    };
    auto resource_bytes_for = [](const ShaderResource* resource, size_t required) -> uint8_t* {
        if (resource->host_data && resource->host_data_size >= required)
            return resource->host_data;
        return reinterpret_cast<uint8_t*>(uintptr_t(resource->gpu_addr));
    };

    do {
        std::vector<VkDescriptorSetLayoutBinding> layout_bindings(descriptors.size());
        bool buffer_setup_failure_reported = false;
        auto skip_buffer = [&](uint32_t binding, const ShaderResource* resource,
                               const char* why) {
            buffer_setup_failure_reported = true;
            if (!trace) return;
            std::fprintf(stderr,
                         "[compute]   buffer setup failed binding=%u addr=0x%llx size=%u: %s\n",
                         binding,
                         (unsigned long long)(resource ? resource->gpu_addr : 0),
                         resource ? resource->size : 0, why);
        };
        for (size_t i = 0; i < descriptors.size(); i++) {
            const ShaderResource* resource = item.resources->by_binding(descriptors[i].binding);
            if (!resource || !resource->size ||
                ((!resource->host_data || resource->host_data_size < resource->size) &&
                 !guest_readable(resource->gpu_addr, resource->size))) {
                skip_buffer(descriptors[i].binding, resource,
                            !resource ? "missing resource" :
                            !resource->size ? "empty resource" : "unreadable backing");
                break;
            }
            buffers[i].resource = resource;
            buffers[i].guest_bytes = resource->size;
            buffers[i].writable = descriptors[i].writable;
            const StorageBufferMaterializationPlan materialization =
                plan_storage_buffer_materialization(descriptors[i], *resource);
            if (!materialization.valid) {
                skip_buffer(descriptors[i].binding, resource,
                            "invalid storage-buffer materialization contract");
                break;
            }
            buffers[i].atomic_image = descriptors[i].atomic_access &&
                resource->cls == ResourceClass::StorageImage &&
                resource->format == DataFormat::Uint32 && resource->num_components == 1 &&
                resource->img_dim == 1 && resource->depth == 1 &&
                !resource->depth_compare && !resource->in_mip_tail &&
                !resource->compression_enabled && resource->width && resource->height;
            const uint64_t linear_image_bytes = static_cast<uint64_t>(resource->width) *
                resource->height * sizeof(uint32_t);
            if (buffers[i].atomic_image) {
                const size_t tight_pitch = static_cast<size_t>(resource->width) * sizeof(uint32_t);
                const size_t guest_image_bytes = resource->tile_mode
                    ? tiled_surface_bytes(resource->width, resource->height,
                                          resource->tile_mode, 0, sizeof(uint32_t))
                    : (resource->linear_row_pitch_bytes
                           ? static_cast<size_t>(resource->linear_row_pitch_bytes)
                           : tight_pitch) * resource->height;
                if (linear_image_bytes > UINT32_MAX ||
                    (!resource->tile_mode && resource->linear_row_pitch_bytes &&
                     resource->linear_row_pitch_bytes < tight_pitch) ||
                    !guest_image_bytes || guest_image_bytes > UINT32_MAX) {
                    skip_buffer(descriptors[i].binding, resource,
                                "invalid atomic-image buffer layout");
                    break;
                }
                if ((!resource->host_data || resource->host_data_size < guest_image_bytes) &&
                    !guest_readable(resource->gpu_addr,
                                    static_cast<uint32_t>(guest_image_bytes))) {
                    skip_buffer(descriptors[i].binding, resource,
                                "unreadable atomic-image physical backing");
                    break;
                }
                buffers[i].guest_bytes = guest_image_bytes;
            }
            buffers[i].bytes = buffers[i].atomic_image
                ? static_cast<size_t>(linear_image_bytes)
                : static_cast<size_t>(materialization.binding_bytes);
            for (size_t j = 0; j < i; ++j) {
                const ShaderResource* prior = buffers[j].resource;
                if (!prior || prior->gpu_addr != resource->gpu_addr ||
                    prior->size != resource->size ||
                    buffers[j].atomic_image != buffers[i].atomic_image ||
                    buffers[j].bytes != buffers[i].bytes ||
                    buffers[j].guest_bytes != buffers[i].guest_bytes ||
                    prior->host_data != resource->host_data ||
                    prior->host_data_size != resource->host_data_size)
                    continue;
                buffers[i].alias_of = buffers[j].alias_of == SIZE_MAX ? j : buffers[j].alias_of;
                const BoundBuffer& owner = buffers[buffers[i].alias_of];
                buffers[i].buffer = owner.buffer;
                buffers[i].memory = owner.memory;
                buffers[buffers[i].alias_of].writable |= buffers[i].writable;
                break;
            }
            if (buffers[i].alias_of == SIZE_MAX) {
                const uint8_t* source = buffers[i].atomic_image
                    ? resource_bytes_for(resource, buffers[i].guest_bytes)
                    : resource_bytes(resource);
                if (buffers[i].atomic_image) {
                    buffers[i].linear_seed.resize(buffers[i].bytes);
                    if (resource->tile_mode) {
                        detile_surface(buffers[i].linear_seed.data(), source,
                                       resource->width, resource->height,
                                       resource->tile_mode, 0, sizeof(uint32_t));
                    } else {
                        const size_t tight_pitch = static_cast<size_t>(resource->width) * 4u;
                        const size_t source_pitch = resource->linear_row_pitch_bytes
                            ? resource->linear_row_pitch_bytes : tight_pitch;
                        for (uint32_t y = 0; y < resource->height; ++y)
                            std::memcpy(buffers[i].linear_seed.data() + y * tight_pitch,
                                        source + y * source_pitch, tight_pitch);
                    }
                    source = buffers[i].linear_seed.data();
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   atomic-image buffer binding=%u addr=0x%llx "
                                     "extent=%ux%u tile=%u bytes=%zu\n",
                                     resource->binding,
                                     (unsigned long long)resource->gpu_addr,
                                     resource->width, resource->height,
                                     resource->tile_mode, buffers[i].bytes);
                } else if (materialization.zero_padded_tail) {
                    buffers[i].linear_seed.resize(buffers[i].bytes);
                    if (!materialize_storage_buffer_bytes(
                            materialization, source, buffers[i].guest_bytes,
                            buffers[i].linear_seed.data(), buffers[i].linear_seed.size())) {
                        skip_buffer(descriptors[i].binding, resource,
                                    "failed zero-padded storage-buffer materialization");
                        break;
                    }
                    source = buffers[i].linear_seed.data();
                }
                if (trace) buffers[i].before_hash = fnv1a(source, buffers[i].bytes);
                buffers[i].cache_key = {
                    resource->gpu_addr, reinterpret_cast<uintptr_t>(resource->host_data),
                    static_cast<uint32_t>(buffers[i].bytes),
                    compute_buffer_materialization_discriminator(materialization)};
                const bool cache_candidate = !buffers[i].atomic_image &&
                    persistent_compute_buffer_enabled(static_cast<uint32_t>(buffers[i].bytes));
                if (cache_candidate && ctx.acquire_cached_buffer(
                        buffers[i].cache_key, source, buffers[i].buffer, buffers[i].memory,
                        buffers[i].upload_skipped, buffers[i].dirty_watch_chunks,
                        buffers[i].total_watch_chunks)) {
                    buffers[i].persistent = true;
                } else {
                    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                    bci.size = buffers[i].bytes;
                    // A persistent buffer may later become writable through an exact alias binding
                    // and need to copy into its result baseline after an external guest update.
                    // Keep one canonical representation transfer-source capable from allocation.
                    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    if (!vk_ok(vkCreateBuffer(ctx.device, &bci, nullptr, &buffers[i].buffer),
                               "buffer-create"))
                        break;
                    VkMemoryRequirements requirements{};
                    vkGetBufferMemoryRequirements(ctx.device, buffers[i].buffer, &requirements);
                    const uint32_t memory_type = ctx.host_memory_type(requirements.memoryTypeBits);
                    if (memory_type == UINT32_MAX) {
                        skip_buffer(descriptors[i].binding, resource,
                                    "no host-visible memory type");
                        break;
                    }
                    buffers[i].memory = ctx.allocate_memory(requirements.size, memory_type, true);
                    if (!buffers[i].memory) {
                        skip_buffer(descriptors[i].binding, resource,
                                    "host-visible memory allocation failed");
                        break;
                    }
                    if (!vk_ok(vkBindBufferMemory(ctx.device, buffers[i].buffer,
                                                  buffers[i].memory, 0),
                               "buffer-bind"))
                        break;
                    void* mapped = nullptr;
                    if (!vk_ok(ctx.map_memory(buffers[i].memory, 0, buffers[i].bytes, &mapped),
                               "buffer-map"))
                        break;
                    // Pooled host-visible allocations retain their previous contents. Compare them
                    // with current guest memory before uploading: any mutation takes the exact copy.
                    if (!compute_buffers_equal(mapped, source, buffers[i].bytes))
                        copy_compute_buffer(mapped, source, buffers[i].bytes);
                    ctx.unmap_memory(buffers[i].memory);
                    if (cache_candidate && ctx.retain_buffer(
                            buffers[i].cache_key, buffers[i].buffer, buffers[i].memory,
                            requirements.size))
                        buffers[i].persistent = true;
                }
                if (trace && buffers[i].persistent)
                    std::fprintf(stderr,
                                 "[compute]   persistent buffer binding=%u addr=0x%llx size=%u "
                                 "upload-skipped=%u dirty-watch-chunks=%u/%u\n",
                                 resource->binding, (unsigned long long)resource->gpu_addr,
                                 resource->size, buffers[i].upload_skipped ? 1u : 0u,
                                 buffers[i].dirty_watch_chunks,
                                 buffers[i].total_watch_chunks);
            }

            layout_bindings[i].binding = descriptors[i].binding;
            layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layout_bindings[i].descriptorCount = 1;
            layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        for (BoundBuffer& buffer : buffers)
            if (buffer.alias_of == SIZE_MAX && buffer.persistent && buffer.writable &&
                !buffer.result_baseline)
                ctx.cached_buffer_result_buffer(
                    buffer.cache_key, buffer.resource->size, buffer.result_baseline);
        bool buffers_ready = true;
        for (const auto& buffer : buffers) buffers_ready &= buffer.resource && buffer.memory;
        if (!buffers_ready) {
            if (trace && !buffer_setup_failure_reported) {
                for (size_t i = 0; i < buffers.size(); ++i) {
                    if (buffers[i].resource && buffers[i].memory) continue;
                    skip_buffer(descriptors[i].binding, buffers[i].resource,
                                "binding was not prepared");
                    break;
                }
            }
            break;
        }
        setup_buffers_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - setup_validate_start).count() - setup_validate_ms;

        // --- Image bindings (#590): sampled textures use RGBA8 unless integer/packed-float semantics
        // require a native view. Storage images normally use format-free R32G32B32A32_UINT channels;
        // exact native-width paths are selected from reflected SPIR-V. Everything not provably correct
        // skips LOUDLY. ---
        bool images_ready = !ctx.device ? false : true;
        auto device_memory_type = [&](uint32_t bits) -> uint32_t {
            for (uint32_t i = 0; i < ctx.memory.memoryTypeCount; i++)
                if ((bits & (1u << i)) &&
                    (ctx.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    return i;
            for (uint32_t i = 0; i < ctx.memory.memoryTypeCount; i++)
                if (bits & (1u << i)) return i;
            return UINT32_MAX;
        };
        auto skip_image = [&](const prosper::gpu::ShaderResource* r, const char* why) {
            static std::vector<uint64_t> warned;
            const uint64_t key = r ? r->gpu_addr : 0;
            if (std::find(warned.begin(), warned.end(), key) == warned.end()) {
                warned.push_back(key);
                std::fprintf(stderr, "[compute] program 0x%llx image 0x%llx %ux%ux%u dim=%u "
                                     "class=%u fmt=%u comps=%u tile=%u size=%u: %s -> "
                                     "dispatch skipped (#590)\n",
                             (unsigned long long)item.code_addr, (unsigned long long)key,
                             r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0,
                             r ? r->img_dim : 0u, r ? static_cast<unsigned>(r->cls) : 0u,
                             r ? (unsigned)r->format : 0u, r ? r->num_components : 0u,
                             r ? r->tile_mode : 0u, r ? r->size : 0u, why);
            }
            images_ready = false;
        };
        const bool image_timing =
            image_timing_requested && timing_item_selected &&
            (!timing_trace_only || trace);
        for (size_t i = 0; i < image_descriptors.size() && images_ready; i++) {
            const auto image_start = ComputeClock::now();
            double import_ms = 0.0;
            double query_ms = 0.0;
            double cache_lookup_ms = 0.0;
            double staging_ms = 0.0;
            double prepare_upload_ms = 0.0;
            double image_allocation_ms = 0.0;
            double view_ms = 0.0;
            double sampler_ms = 0.0;
            const ShaderResource* r = item.resources->by_binding(image_descriptors[i].binding);
            if (!r || !r->width || !r->height) { skip_image(r, "no/degenerate resource"); break; }
            BoundImage& bi = images[i];
            bi.resource = r;
            bi.binding = image_descriptors[i].binding;
            bi.storage = image_descriptors[i].kind == SpirvDescriptorKind::StorageImage;
            if (bi.storage &&
                image_descriptors[i].image_numeric_class != SpirvImageNumericClass::Float &&
                image_descriptors[i].image_numeric_class != SpirvImageNumericClass::Uint) {
                skip_image(r, "storage image has unknown or unsupported signed numeric class");
                break;
            }
            const uint32_t descriptor_components =
                r->num_components ? r->num_components : 1;
            const VkFormat native_storage_format =
                native_storage_vk_format(r->format, descriptor_components);
            const bool spirv_native_storage = bi.storage &&
                image_descriptors[i].image_numeric_class == SpirvImageNumericClass::Float;
            bi.packed_r11_storage = bi.storage && !spirv_native_storage &&
                !image_descriptors[i].atomic_access &&
                image_descriptors[i].storage_image_format == kSpirvImageFormatR32ui &&
                r->format == DataFormat::Float10_11_11 && descriptor_components == 3;
            const bool ordinary_2d_view = shader_resource_uses_ordinary_2d_image(
                *r, image_descriptors[i].image_dim == 1u,
                image_descriptors[i].image_arrayed,
                image_descriptors[i].image_multisampled);
            const bool native_2d_storage = bi.storage &&
                shader_resource_uses_native_2d_storage_image(
                    *r, image_descriptors[i].image_dim == 1u,
                    image_descriptors[i].image_arrayed,
                    image_descriptors[i].image_multisampled);
            const bool ordinary_3d_storage = r->img_dim == 2 && r->depth &&
                                             !r->depth_compare;
            const VkImageType native_storage_type = ordinary_3d_storage
                ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
            const bool native_storage_semantic = native_float_storage_image(
                r->format, descriptor_components, r->srgb);
            // vkGetPhysicalDeviceImageFormatProperties is a driver query, not a cheap metadata
            // lookup. Sampled and raw-uvec4 descriptors cannot take this typed-storage path, so do
            // not repeat that query for every one of their per-frame bindings.
            const bool native_storage_device = spirv_native_storage &&
                (native_2d_storage || ordinary_3d_storage) &&
                native_storage_image_create_supported(
                    ctx.physical, native_storage_format, native_storage_type,
                    r->width, r->height,
                    ordinary_3d_storage ? r->depth : 1u,
                    native_2d_storage && image_descriptors[i].image_arrayed
                        ? r->depth : 1u);
            const bool native_storage_supported = spirv_native_storage &&
                native_float_storage_image_supported(
                    r->format, descriptor_components, r->srgb,
                    native_storage_device);
            transfer_gate_census.record_reflection(
                transfer_gate_observation.role, image_descriptors[i], *r, bi.storage,
                ordinary_2d_view, native_storage_semantic, native_storage_device,
                transfer_gate_observation.first_match);
            if (spirv_native_storage && !native_storage_supported) {
                skip_image(r, "compiled typed storage format is unsupported by this device");
                break;
            }
            // SPIR-V is authoritative here: a raw-uvec4 module must keep the conversion path even
            // if this replay device supports the optional typed format, while a live module is only
            // emitted as float after the frontend's physical-device feature query.
            bi.native_float_storage = spirv_native_storage;
            bi.graphics_sampled_usage = bi.native_float_storage &&
                (ordinary_2d_view || ordinary_3d_storage) &&
                native_storage_image_create_supported(
                    ctx.physical, native_storage_format, native_storage_type,
                    r->width, r->height, ordinary_3d_storage ? r->depth : 1u, 1u,
                    VK_IMAGE_USAGE_SAMPLED_BIT);
            if (trace)
                std::fprintf(stderr,
                             "[compute]   image binding=%u class=%s addr=0x%llx "
                             "extent=%ux%ux%u format=%u components=%u tile=%u native-storage=%u\n",
                             bi.binding, bi.storage ? "storage" : "sampled",
                             (unsigned long long)r->gpu_addr, r->width, r->height, r->depth,
                             (unsigned)r->format, r->num_components, r->tile_mode,
                             bi.native_float_storage ? 1u : 0u);
            if (trace && bi.packed_r11_storage)
                std::fprintf(stderr,
                             "[compute]   image binding=%u exact packed R11G11B10 storage via R32_UINT\n",
                             bi.binding);
            // A surface whose CURRENT pixels live in the renderer's RTT cache must not be read from
            // raw guest memory (empty/stale — the Dead Cells 642x362 lesson).
            const bool dim_1d = r->img_dim == 0;
            const bool dim_3d = r->img_dim == 2;
            const bool dim_2d_array = r->img_dim == 5 &&
                                      image_descriptors[i].image_dim == 1u &&
                                      image_descriptors[i].image_arrayed;
            bi.arrayed_2d = dim_2d_array;
            // The recompiler's established cube lowering converts AMD's cube-processed
            // [x, y, face] coordinates to a plain 2D sample over six vertically stacked faces.
            // Compute must expose the same w x 6h image contract as live_renderer. Astro Bot's
            // world-map irradiance kernel binds a 192-face BC6H probe atlas but this packet samples
            // its first cube, so only the six addressed faces need to be materialized here.
            const bool shader_2d_cube_view = r->img_dim == 3 && !bi.storage &&
                                             image_descriptors[i].image_dim == 1u &&
                                             !image_descriptors[i].image_arrayed &&
                                             !image_descriptors[i].image_multisampled;
            // Cube-array sampling lowers one addressed cube to six vertically stacked faces. A
            // single six-face allocation can also be deliberately viewed by a DIM=2D instruction;
            // retain that established face-zero contract instead of consuming sibling faces.
            const bool dim_cube_stacked = shader_2d_cube_view && r->depth > 6u;
            const bool cube_face_as_2d = shader_2d_cube_view && !dim_cube_stacked;
            bi.stacked_cube = dim_cube_stacked;
            // A SINGLE-LAYER 2D array (img_dim==5, depth==1, no depth-compare) is byte-identical to a
            // plain 2D image (one layer, same tiling), so it flows through the 2D path below unchanged.
            // Reflection distinguishes that fallback from a shader which retains the layer coordinate.
            const bool dim_2d_single = r->img_dim == 5 && ordinary_2d_view;
            if (!dim_1d && r->img_dim != 1 && !dim_3d && !dim_2d_array &&
                !dim_2d_single && !dim_cube_stacked && !cube_face_as_2d) {
                skip_image(r, "layered image deferred to #657"); break;
            }
            if (dim_1d && r->height != 1) { skip_image(r, "1D image has non-unit height"); break; }
            if (!r->depth || (!dim_3d && !dim_2d_array && !dim_cube_stacked && r->depth != 1)) {
                if (!cube_face_as_2d) {
                    skip_image(r, "image depth does not match its dimensionality"); break;
                }
            }
            if (dim_cube_stacked && (r->depth < 6 || r->height > UINT32_MAX / 6u)) {
                skip_image(r, "cube image does not contain six valid faces"); break;
            }
            bi.texel_depth = dim_3d ? r->depth : 1u;
            bi.array_layers = dim_2d_array ? r->depth : 1u;
            if (cube_face_as_2d && trace)
                std::fprintf(stderr,
                             "[compute]   binding cube face zero as shader-declared 2D image "
                             "binding=%u addr=0x%llx\n",
                             bi.binding, (unsigned long long)r->gpu_addr);
            if (dim_3d && r->depth > 1 && r->tile_mode &&
                !tile_mode_supports_volume(r->tile_mode)) {
                skip_image(r, "3D tile mode has no volume address pattern"); break;
            }
            // The backend writes an ordinary tiled base allocation, not hardware-compressed blocks.
            // A compressed U# therefore also needs writable DCC metadata so successful writeback can
            // publish the hardware's 0xff (uncompressed) state before a later sampled descriptor sees it.
            if (bi.storage && r->compression_enabled) {
                const uint64_t metadata_bytes = gpu_capture_dcc_metadata_footprint(*r);
                if (!r->metadata_addr || !metadata_bytes || metadata_bytes > SIZE_MAX ||
                    metadata_bytes > UINT32_MAX) {
                    skip_image(r, "DCC metadata extent is unsupported"); break;
                }
                bi.dcc_metadata_bytes = static_cast<size_t>(metadata_bytes);
                if (r->dcc_metadata_host_data) {
                    if (r->dcc_metadata_host_data_size < metadata_bytes) {
                        skip_image(r, "replay DCC metadata backing is truncated"); break;
                    }
                    bi.dcc_metadata = r->dcc_metadata_host_data;
                } else {
                    if (!guest_readable(r->metadata_addr, static_cast<uint32_t>(metadata_bytes))) {
                        skip_image(r, "live DCC metadata backing is unreadable"); break;
                    }
                    bi.dcc_metadata = reinterpret_cast<uint8_t*>(uintptr_t(r->metadata_addr));
                }
            }
            // A renderer-owned target's current pixels are not in raw guest memory. Sampled
            // descriptors may borrow its Vulkan image directly; storage descriptors use the CPU
            // snapshot path below so their guest writeback keeps overlapping aliases coherent.
            LiveTargetSnapshot live_target;
            const auto query_start = ComputeClock::now();
            bool renderer_owned = !r->in_mip_tail && is_live_render_target(r->gpu_addr);
            query_ms = std::chrono::duration<double, std::milli>(
                ComputeClock::now() - query_start).count();
            static const uint32_t render_scale = [] {
                const char* e = std::getenv("PROSPER_RENDER_SCALE");
                const long v = e ? std::strtol(e, nullptr, 10) : 1;
                return v > 0 ? static_cast<uint32_t>(v) : 1u;
            }();
            static const bool unorm_rtt_value_reuse_enabled =
                std::getenv("PROSPER_NO_UNORM_RTT_VALUE_REUSE") == nullptr &&
                // Preserve the recovery switch published with the normalized-sampling first step.
                std::getenv("PROSPER_NO_NORMALIZED_UNORM_RTT_BIND") == nullptr;
            if (renderer_owned && (dim_3d || dim_2d_array || r->depth != 1)) {
                // The renderer cache represents one concrete 2D color image. An address match from
                // a 3D/layered descriptor is therefore a resource-lifetime alias, not that cached
                // surface: Vulkan/guest allocators routinely recycle one base across incompatible
                // views. Its canonical 2D pixels cannot seed the layered layout, while the new
                // resource's guest backing is the only representation with the requested shape.
                renderer_owned = false;
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   renderer RTT dimensional-view miss binding=%u "
                                 "addr=0x%llx requested=%ux%ux%u dim=%u -> guest backing\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 r->width, r->height, r->depth, r->img_dim);
            }
            // Bind the renderer's own image when it is authoritative and either exactly matches the
            // sampled view (#1095, phase 2 of #1091) or carries proven value-equivalent UNORM data.
            // RGBA8 and RGBA16F exact views avoid a CPU snapshot entirely; a float RGBA16_UNORM view
            // may also consume canonical RGBA8 because byte*257/65535 == byte/255. The independent
            // extent check below still rejects a scaled image for texel fetch/query access. Other
            // aliases and numeric conversions keep the snapshot path below.
            const bool depth_float_import_eligible = !bi.storage && !dim_1d && !dim_3d &&
                !dim_2d_array && r->depth == 1 && r->img_dim == 1 &&
                r->format == DataFormat::Float32 &&
                (r->num_components ? r->num_components : 1u) == 1u;
            const bool depth_bits_import_eligible = !bi.storage && !dim_1d && !dim_3d &&
                !dim_2d_array && r->depth == 1 && r->img_dim == 1 &&
                r->format == DataFormat::Uint32 &&
                (r->num_components ? r->num_components : 1u) == 1u;
            const bool depth_import_eligible =
                depth_float_import_eligible || depth_bits_import_eligible;
            // Persistent renderer images do not carry VK_IMAGE_USAGE_STORAGE_BIT, and a writable
            // storage import would also leave overlapping guest buffer aliases stale. Storage
            // descriptors therefore retain the owned-image + guest-writeback path.
            if (!bi.storage && (renderer_owned || depth_import_eligible) &&
                !dim_1d && !dim_3d && !dim_2d_array &&
                r->depth == 1 && !r->depth_compare) {
                LiveTargetImageImport import;
                const bool scalable_normalized_sampling =
                    image_descriptors[i].normalized_sampling &&
                    !image_descriptors[i].texel_access;
                const bool format_float_sampling = image_descriptors[i].sampled_float;
                const LiveTargetImageRequest import_request{
                    r->width, r->height, render_scale, depth_import_eligible,
                    scalable_normalized_sampling};
                const auto import_start = ComputeClock::now();
                const bool import_available = import_live_render_target_image(
                    r->gpu_addr, import_request, import);
                import_ms = std::chrono::duration<double, std::milli>(
                    ComputeClock::now() - import_start).count();
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   direct RTT candidate binding=%u addr=0x%llx "
                                 "requested=f%u/c%u dim=%u extent=%ux%u available=%u "
                                 "imported=%ux%u/%u kind=%s native=%u\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned)r->format, r->num_components, r->img_dim,
                                 r->width, r->height, import_available ? 1u : 0u,
                                 import.width, import.height, (unsigned)import.format,
                                 import.kind == LiveTargetImageImport::Kind::Depth
                                     ? "depth" : "color",
                                 import.native_format);
                if (import_available) {
                    const bool depth_import =
                        import.kind == LiveTargetImageImport::Kind::Depth;
                    const VkFormat depth_format = static_cast<VkFormat>(import.native_format);
                    const bool compatible_format = depth_import
                        ? depth_import_eligible &&
                              (depth_format == VK_FORMAT_D32_SFLOAT ||
                               depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT)
                        : direct_sampled_rtt_compatible(
                              r->format, r->num_components ? r->num_components : 1,
                              import.format,
                              format_float_sampling && unorm_rtt_value_reuse_enabled);
                    const bool compatible_device =
                        import.device == static_cast<void*>(ctx.device);
                    const bool direct_extent =
                        prosper::frontend::rtt_direct_import_compatible(
                            bi.storage, r->width, r->height, import.width, import.height,
                            render_scale, scalable_normalized_sampling);
                    // Raw bit views cannot use the normalized-sampling scale exception: the
                    // transfer below copies exact texels and deliberately performs no resampling.
                    const bool depth_bits_extent =
                        import.width == r->width && import.height == r->height;
                    if (compatible_format && direct_extent && compatible_device &&
                        (!depth_bits_import_eligible || !depth_import || depth_bits_extent)) {
                        if (depth_import && depth_bits_import_eligible) {
                            bi.depth_bits_source = true;
                            bi.depth_bits_image = static_cast<VkImage>(import.image);
                            bi.depth_bits_format = depth_format;
                            bi.depth_bits_saved_layout = import.layout;
                            bi.imported_addr = r->gpu_addr;
                            bi.imported_width = import.width;
                            bi.imported_height = import.height;
                            if (trace)
                                std::fprintf(stderr,
                                             "[compute]   bridged renderer depth bits binding=%u "
                                             "addr=0x%llx extent=%ux%u depth-format=%u "
                                             "sampled-format=R32_UINT\n",
                                             bi.binding,
                                             (unsigned long long)r->gpu_addr,
                                             import.width, import.height,
                                             static_cast<unsigned>(depth_format));
                        } else {
                            bi.imported = true;
                            bi.imported_depth = depth_import;
                            bi.imported_format = depth_import ? depth_format : VK_FORMAT_UNDEFINED;
                            bi.imported_pixel_format = import.format;
                            bi.imported_transfer_dst = import.transfer_dst;
                            bi.imported_addr = r->gpu_addr;
                            bi.imported_width = import.width;
                            bi.imported_height = import.height;
                            bi.imported_saved_layout = import.layout;
                            bi.image = static_cast<VkImage>(import.image);
                            live_target.width = import.width;
                            live_target.height = import.height;
                            live_target.format = import.format;
                            if (trace)
                                std::fprintf(stderr,
                                             "[compute]   bound renderer RTT in place binding=%u "
                                             "class=%s addr=0x%llx extent=%ux%u format=%s\n",
                                             bi.binding, bi.storage ? "storage" : "sampled",
                                             (unsigned long long)r->gpu_addr,
                                             import.width, import.height,
                                             depth_import ? "depth"
                                                 : prosper::frontend::live_target_pixel_format_name(
                                                       import.format));
                        }
                    } else {
                        // Not our exact contract (a different device or a stale aliased view).
                        release_live_render_target_image(r->gpu_addr);
                    }
                }
            }
            // #1122: skip the seed entirely for a write-only storage image whose dispatch fully
            // covers the target -- BUT only after proving (once per shader) that the write actually
            // stores every texel (see the SeedCoverage cache above). Enough total invocations is a
            // NECESSARY condition (a smaller grid always seeds), but the mapping need not be one
            // invocation per same-axis texel: vectorized/swizzled kernels are common. The poison
            // proving frame establishes actual full coverage for this shader/binding/extent.
            const bool enough_threads = dispatch_has_enough_threads_for_texels(
                item.launch.threads_x, item.launch.threads_y, item.launch.threads_z,
                r->width, r->height, r->depth);
            // Diagnostic: force the proving (poison) path on every eligible dispatch, never fast-skip.
            static const bool force_verify = std::getenv("PROSPER_VERIFY_SEED_SKIP") != nullptr;
            if (bi.storage && seed_skip_enabled && !image_descriptors[i].readable &&
                image_descriptors[i].writable && enough_threads) {
                const SeedCoverageKey proof_key{item.code_addr, bi.binding,
                                                r->width, r->height, r->depth};
                bool proven_full = false, known = false, reprove_due = false;
                {
                    std::lock_guard<std::mutex> lk(seed_coverage_mu);
                    auto it = seed_coverage_proof.find(proof_key);
                    if (it != seed_coverage_proof.end()) {
                        known = true;
                        proven_full = it->second.cov == SeedCoverage::Full;
                        // #1127: periodically re-prove a Full verdict so a data-dependent store that
                        // later under-covers is caught (re-cached Partial, then always seeds). The
                        // helper resets the counter, so concurrent dispatches on this key don't all
                        // re-prove at once.
                        if (proven_full && seed_reprove_due(it->second.skips, kSeedReproveInterval))
                            reprove_due = true;
                    }
                }
                if (force_verify || reprove_due) {
                    bi.poison_verify = true;    // (re-)prove; the writeback caches the verdict
                } else if (proven_full) {
                    bi.seed_skip = true;        // proven: every texel is written, the seed is unobserved
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   seed-skip write-only storage binding=%u addr=0x%llx "
                                     "extent=%ux%u threads=%ux%ux%u renderer_owned=%d\n",
                                     bi.binding, (unsigned long long)r->gpu_addr, r->width, r->height,
                                     item.launch.threads_x, item.launch.threads_y, item.launch.threads_z,
                                     renderer_owned ? 1 : 0);
                } else if (!known) {
                    bi.poison_verify = true;    // unknown: prove coverage this frame (still correct)
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   seed-skip PROVING coverage binding=%u addr=0x%llx "
                                     "code=0x%llx extent=%ux%u\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned long long)item.code_addr, r->width, r->height);
                }
                // Proven Partial: fall through, seed normally every frame.
            }
            // A partial storage write needs the old target contents, but an exact sampled binding
            // earlier in this dispatch may already have borrowed those pixels directly from the
            // renderer. Seed the private writable image from that Vulkan image below instead of
            // materializing the complete RTT on the CPU and immediately uploading it again. Keep
            // proving frames on the CPU poison path: their coverage verdict depends on the seed.
            if (renderer_owned && bi.storage && bi.native_float_storage &&
                !bi.seed_skip && !bi.poison_verify) {
                for (size_t j = 0; j < i; ++j) {
                    const BoundImage& source = images[j];
                    const ShaderResource* p = source.resource;
                    if (!source.imported || source.imported_depth || source.storage || !p)
                        continue;
                    if (p->gpu_addr != r->gpu_addr || p->width != r->width ||
                        p->height != r->height || p->depth != r->depth ||
                        p->format != r->format || p->num_components != r->num_components)
                        continue;
                    if (!prosper::frontend::rtt_gpu_seed_import_extent_compatible(
                            r->width, r->height,
                            source.imported_width, source.imported_height))
                        continue;
                    bi.seed_from_imported = j;
                    bi.mirror_result_to_imported =
                        prosper::frontend::live_rtt_compute_mirror_eligible(
                            true, source.imported_transfer_dst,
                            std::getenv("PROSPER_NO_COMPUTE_RTT_MIRROR") != nullptr);
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   GPU-seeded writable renderer RTT binding=%u "
                                     "from binding=%u addr=0x%llx extent=%ux%u\n",
                                     bi.binding, source.binding,
                                     (unsigned long long)r->gpu_addr, r->width, r->height);
                    break;
                }
            }
            if (renderer_owned && !bi.imported && !bi.seed_skip &&
                bi.seed_from_imported == SIZE_MAX) {
                if (dim_3d || r->depth != 1 ||
                    !read_live_render_target(r->gpu_addr, live_target) || !live_target.pixels) {
                    skip_image(r, "renderer-owned RTT has no readable snapshot"); break;
                }
                // A dimension mismatch is either (a) an exact PROSPER_RENDER_SCALE downscale (the renderer
                // rendered this same target at 1/scale; a compute op sampling it at native res sees e.g.
                // 480x270 cached for a 1920x1080 request), or (b) a genuine view ALIAS at a reused base
                // (Astro Bot renders 960x540 R11G11B10, then dispatches a 1216x684 R32 Z_X view at the same
                // base after a dynamic-resolution change — the snapshot is NOT this view). Only (a) upscales;
                // (b) still falls back to guest backing.
                //
                // The renderer rounds each native axis independently when it constructs a scaled target.
                // Requiring an exact integer ratio rejects legitimate non-divisible extents (for example
                // 1216x684 -> 405x228 at scale 3), so use the shared rounded-extent proof. A stale view alias
                // still fails unless BOTH cached axes equal the configured-scale result.
                if (live_target.width != r->width || live_target.height != r->height) {
                    const bool scaled_extent = prosper::frontend::rtt_scaled_extent_compatible(
                        r->width, r->height, live_target.width, live_target.height, render_scale);
                    const uint64_t bpp =
                        prosper::frontend::live_target_pixel_format_bytes(live_target.format);
                    if (bpp && scaled_extent && live_target.pixels &&
                        live_target.pixels->size() == (uint64_t)live_target.width * live_target.height * bpp) {
                        const uint32_t sw = live_target.width, sh = live_target.height;
                        auto up = std::make_shared<std::vector<uint8_t>>(
                            (size_t)r->width * r->height * (size_t)bpp);
                        const uint8_t* src = live_target.pixels->data();
                        uint8_t* dst = up->data();
                        for (uint32_t y = 0; y < r->height; y++) {
                            const uint32_t sy = std::min<uint32_t>(
                                sh - 1u, static_cast<uint32_t>(
                                    static_cast<uint64_t>(y) * sh / r->height));
                            for (uint32_t x = 0; x < r->width; x++) {
                                const uint32_t sx = std::min<uint32_t>(
                                    sw - 1u, static_cast<uint32_t>(
                                        static_cast<uint64_t>(x) * sw / r->width));
                                std::memcpy(dst + ((size_t)y * r->width + x) * bpp,
                                            src + ((size_t)sy * sw + sx) * bpp, bpp);
                            }
                        }
                        live_target.pixels = up;
                        live_target.width = r->width; live_target.height = r->height;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   renderer RTT upscaled binding=%u addr=0x%llx "
                                         "%ux%u -> %ux%u (RENDER_SCALE x%u)\n",
                                         bi.binding, (unsigned long long)r->gpu_addr, sw, sh,
                                         r->width, r->height, render_scale);
                    } else {
                        renderer_owned = false;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   renderer RTT view miss binding=%u addr=0x%llx "
                                         "cached=%ux%u requested=%ux%u -> guest backing\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         live_target.width, live_target.height, r->width, r->height);
                    }
                }
                if (renderer_owned) {
                    const uint64_t bpp =
                        prosper::frontend::live_target_pixel_format_bytes(live_target.format);
                    if (!bpp) {
                        skip_image(r, "renderer-owned RTT snapshot has no mapped texel width"); break;
                    }
                    const uint64_t texels = static_cast<uint64_t>(r->width) * r->height;
                    if (texels > UINT64_MAX / bpp) {
                        skip_image(r, "renderer-owned RTT snapshot size overflow"); break;
                    }
                    const uint64_t expected = texels * bpp;
                    if (expected != live_target.pixels->size()) {
                        skip_image(r, "renderer-owned RTT snapshot byte count mismatch"); break;
                    }
                }
                if (renderer_owned &&
                    live_target.format == LiveTargetPixelFormat::R11G11B10Float &&
                    (r->format != DataFormat::Float10_11_11 ||
                     (r->num_components ? r->num_components : 1u) != 3u)) {
                    // Keep native packed targets exact. Supporting a different typed view requires
                    // an explicit byte-compatible alias contract; interpreting these four bytes as
                    // RGBA8/FP16 would silently corrupt the compute input.
                    renderer_owned = false;
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   renderer RTT packed-view miss binding=%u "
                                     "addr=0x%llx requested=f%u/c%u -> guest backing\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned)r->format,
                                     r->num_components ? r->num_components : 1u);
                }
                if (renderer_owned && !bi.storage &&
                    (r->format == DataFormat::Float32 || r->format == DataFormat::Uint16 ||
                     r->format == DataFormat::Uint32)) {
                    const uint64_t cached_bpp =
                        prosper::frontend::live_target_pixel_format_bytes(live_target.format);
                    const uint64_t requested_bpp =
                        static_cast<uint64_t>(data_format_bytes(r->format)) *
                        (r->num_components ? r->num_components : 1u);
                    if (cached_bpp != requested_bpp) {
                        // A typed view is a bit reinterpretation of the target allocation. When the
                        // cached and requested texel widths differ, the renderer snapshot is a
                        // different alias and cannot be expanded or numerically converted safely.
                        renderer_owned = false;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   renderer RTT typed-view miss binding=%u "
                                         "addr=0x%llx cached-bpp=%llu requested-bpp=%llu "
                                         "-> guest backing\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         (unsigned long long)cached_bpp,
                                         (unsigned long long)requested_bpp);
                    }
                }
                if (renderer_owned && bi.storage) {
                    const uint32_t nc = r->num_components ? r->num_components : 1;
                    const bool compatible =
                        (live_target.format == LiveTargetPixelFormat::Rgba8Unorm &&
                         r->format == DataFormat::Unorm8 && nc == 4) ||
                        (live_target.format == LiveTargetPixelFormat::Rgba16Float &&
                         r->format == DataFormat::Float16 && nc == 4) ||
                        (live_target.format == LiveTargetPixelFormat::R11G11B10Float &&
                         r->format == DataFormat::Float10_11_11 && nc == 3);
                    if (!compatible) {
                        // The same allocation can carry another target view before this compute
                        // operation (Astro Bot uses R8G8 and RGBA16F views at one base). A snapshot
                        // in the wrong storage format is not current bytes for this view.
                        renderer_owned = false;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   renderer RTT format miss binding=%u "
                                         "addr=0x%llx cached=%s requested=f%u/c%u -> guest backing\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         prosper::frontend::live_target_pixel_format_name(
                                             live_target.format),
                                         (unsigned)r->format, nc);
                    }
                }
                bi.unorm_rtt_value_reuse = renderer_owned && !bi.storage &&
                    r->format == DataFormat::Unorm16 &&
                    direct_sampled_rtt_compatible(
                        r->format, r->num_components ? r->num_components : 1u,
                        live_target.format,
                        unorm_rtt_value_reuse_enabled && image_descriptors[i].sampled_float);
            }
            // Exact descriptor aliases share one Vulkan image. Storage aliases must observe the same
            // read/modify/write state and produce one guest writeback; sampled aliases avoid repeatedly
            // detiling and uploading Astro Bot's full-resolution source through dozens of SRT slots.
            for (size_t j = 0; j < i; j++) {
                const BoundImage& prior = images[j];
                const ShaderResource* p = prior.resource;
                // Identical guest descriptors can appear at multiple bindings with different
                // reflected access contracts. Do not fold an exact R16 upload into the value-equivalent
                // RGBA8 representation (or vice versa), and only share borrowed renderer images
                // when both bindings acquired the same concrete image/view format.
                const bool same_backing_representation =
                    prior.imported == bi.imported &&
                    prior.depth_bits_source == bi.depth_bits_source &&
                    prior.unorm_rtt_value_reuse == bi.unorm_rtt_value_reuse &&
                    (!bi.imported ||
                     (prior.image == bi.image &&
                      prior.imported_depth == bi.imported_depth &&
                      prior.imported_format == bi.imported_format &&
                      prior.imported_pixel_format == bi.imported_pixel_format)) &&
                    (!bi.depth_bits_source ||
                     (prior.depth_bits_image == bi.depth_bits_image &&
                      prior.depth_bits_format == bi.depth_bits_format));
                const bool same_dcc_identity = p &&
                    p->max_uncompressed_block_size == r->max_uncompressed_block_size &&
                    p->max_compressed_block_size == r->max_compressed_block_size &&
                    p->meta_pipe_aligned == r->meta_pipe_aligned &&
                    p->write_compress_enabled == r->write_compress_enabled &&
                    p->compression_enabled == r->compression_enabled &&
                    p->alpha_is_on_msb == r->alpha_is_on_msb &&
                    p->color_transform == r->color_transform &&
                    p->metadata_addr == r->metadata_addr &&
                    p->dcc_metadata_size == r->dcc_metadata_size &&
                    p->dcc_metadata_host_data == r->dcc_metadata_host_data &&
                    p->dcc_metadata_host_data_size == r->dcc_metadata_host_data_size;
                const bool same_view = p && same_backing_representation &&
                    prior.storage == bi.storage &&
                    p->gpu_addr == r->gpu_addr && p->size == r->size &&
                    p->width == r->width && p->height == r->height && p->depth == r->depth &&
                    prior.texel_depth == bi.texel_depth && prior.array_layers == bi.array_layers &&
                    p->format == r->format && p->num_components == r->num_components &&
                    p->tile_mode == r->tile_mode && p->img_dim == r->img_dim &&
                    p->layer_stride_bytes == r->layer_stride_bytes &&
                    p->layer_mip_offset_bytes == r->layer_mip_offset_bytes &&
                    p->in_mip_tail == r->in_mip_tail &&
                    p->mip_tail_x == r->mip_tail_x && p->mip_tail_y == r->mip_tail_y &&
                    p->srgb == r->srgb && p->host_data == r->host_data &&
                    p->host_data_size == r->host_data_size && same_dcc_identity;
                bool same_sampler = true;
                if (!bi.storage && same_view) {
                    same_sampler = p->mag_filter == r->mag_filter &&
                        p->min_filter == r->min_filter && p->mip_filter == r->mip_filter &&
                        p->addr_uvw[0] == r->addr_uvw[0] && p->addr_uvw[1] == r->addr_uvw[1] &&
                        p->addr_uvw[2] == r->addr_uvw[2] &&
                        p->border_color_type == r->border_color_type &&
                        p->min_lod == r->min_lod && p->max_lod == r->max_lod &&
                        p->lod_bias == r->lod_bias && p->max_aniso_ratio == r->max_aniso_ratio &&
                        p->depth_compare_func == r->depth_compare_func &&
                        p->depth_compare == r->depth_compare && p->unnormalized == r->unnormalized &&
                        p->swizzle[0] == r->swizzle[0] && p->swizzle[1] == r->swizzle[1] &&
                        p->swizzle[2] == r->swizzle[2] && p->swizzle[3] == r->swizzle[3];
                }
                if (!same_view || !same_sampler) continue;
                bi.alias_of = prior.alias_of == SIZE_MAX ? j : prior.alias_of;
                const BoundImage& owner = images[bi.alias_of];
                bi.image = owner.image; bi.memory = owner.memory; bi.view = owner.view;
                bi.sampler = owner.sampler; bi.guest_bytes = owner.guest_bytes;
                bi.texel_depth = owner.texel_depth;
                bi.array_layers = owner.array_layers;
                bi.dcc_metadata = owner.dcc_metadata;
                bi.dcc_metadata_bytes = owner.dcc_metadata_bytes;
                staging_bytes[i] = staging_bytes[bi.alias_of];
                if (trace)
                    std::fprintf(stderr, "[compute]   image-alias binding=%u -> binding=%u addr=0x%llx\n",
                                 bi.binding, owner.binding, (unsigned long long)r->gpu_addr);
                break;
            }
            if (bi.alias_of != SIZE_MAX) {
                const BoundImage& alias_owner = images[bi.alias_of];
                if (image_timing)
                    std::fprintf(stderr,
                                 "[compute-image] code=0x%llx hash=0x%016llx "
                                 "binding=%u class=%s alias=1 addr=0x%llx alias_of=%u "
                                 "persistent=%u upload-skipped=%u "
                                 "extent=%ux%ux%u ms=%.3f\n",
                                 (unsigned long long)item.code_addr,
                                 (unsigned long long)timing_program_hash, bi.binding,
                                 bi.storage ? "storage" : "sampled",
                                 (unsigned long long)r->gpu_addr, alias_owner.binding,
                                 alias_owner.persistent ? 1u : 0u,
                                 alias_owner.upload_skipped ? 1u : 0u,
                                 r->width, r->height, r->depth,
                                 std::chrono::duration<double, std::milli>(
                                     ComputeClock::now() - image_start).count());
                continue;
            }
            const uint32_t sampled_layers = dim_cube_stacked ? 6u
                                            : dim_2d_array ? r->depth : 1u;
            const VkDeviceSize volume_texels = static_cast<VkDeviceSize>(r->width) * r->height *
                                               (dim_3d ? r->depth : sampled_layers);
            const uint32_t sampled_components = r->num_components ? r->num_components : 1;
            // ShaderResource::depth_compare describes the guest MIMG operation, not the Vulkan
            // descriptor type. Manual IMAGE_SAMPLE_C* lowering declares an ordinary color image
            // and compares its R channel in SPIR-V; only an actual Depth=1 OpTypeImage may use a
            // depth view and compare-enabled sampler here.
            const bool sampled_depth = !bi.storage && image_descriptors[i].image_depth &&
                                       dim_2d_array &&
                                       sampled_components == 1 &&
                                       (r->format == DataFormat::Float32 ||
                                        r->format == DataFormat::Unorm16);
            bi.depth_view = sampled_depth;
            if (image_descriptors[i].image_depth && !sampled_depth) {
                skip_image(r, "depth-compare image format/dimension unsupported"); break;
            }
            const bool sampled_float32 = !bi.storage && !sampled_depth &&
                                         r->format == DataFormat::Float32 &&
                                         sampled_components >= 1 && sampled_components <= 4;
            // Vulkan supplies (0, 0, 1) for missing sampled-image channels, exactly matching the
            // GCN texture result. Keep native one/two/four-channel widths instead of expanding an
            // R32 image to RGBA32F every dispatch. Three-channel optimal images are not universally
            // supported, so retain the portable four-channel expansion for that uncommon case.
            const bool sampled_float32_native = sampled_float32 && sampled_components != 3;
            // Renderer imports and compute 3D RGBA16F textures can use their exact native sampled
            // representation. Narrowing a volume to RGBA8 discarded its HDR range and prevented a
            // retained native storage result from seeding the next ping-pong sample on the GPU.
            // Ordinary guest-backed 2D FP16 keeps its historical RGBA8 conversion: native RGBA16F
            // sampling was measured 7x slower in Astro Bot's full-resolution composite on RADV.
            const bool sampled_float16_native = !bi.storage &&
                r->format == DataFormat::Float16 && sampled_components == 4 &&
                (bi.imported || dim_3d);
            const bool sampled_unorm8x2 = !bi.storage && r->format == DataFormat::Unorm8 &&
                                          sampled_components == 2;
            const bool sampled_unorm16_native = !bi.storage && r->format == DataFormat::Unorm16 &&
                                                (sampled_components == 1 || sampled_components == 2 ||
                                                 sampled_components == 4);
            const bool sampled_uint8_native = !bi.storage && r->format == DataFormat::Uint8 &&
                                              (sampled_components == 1 || sampled_components == 2 ||
                                               sampled_components == 4);
            const bool sampled_uint16_native = !bi.storage && r->format == DataFormat::Uint16 &&
                                               (sampled_components == 1 || sampled_components == 2 ||
                                                sampled_components == 4);
            const bool sampled_uint32_native = !bi.storage && r->format == DataFormat::Uint32 &&
                                               (sampled_components == 1 || sampled_components == 2 ||
                                                sampled_components == 4);
            const uint32_t native_storage_bytes = bi.exact_storage_bytes()
                ? (r->format == DataFormat::Float10_11_11
                       ? 4u : data_format_bytes(r->format) * sampled_components)
                : 0u;
            // The one bytes call site where 0 does NOT mean decline: zero is already this
            // expression's sentinel for "not an imported color binding", so an unmapped format would
            // silently fall through to the guest resource's texel width below rather than refuse.
            // Safe only because the importer never produces an unmapped format and -Werror=switch
            // stops a new enumerator from reaching here without a real width. Any future edit that
            // weakens either of those has to give this site its own explicit decline.
            const uint32_t imported_color_bytes = bi.imported && !bi.imported_depth
                ? prosper::frontend::live_target_pixel_format_bytes(bi.imported_pixel_format)
                : 0u;
            const uint32_t texel_bytes = imported_color_bytes ? imported_color_bytes
                                         : bi.unorm_rtt_value_reuse ? 4u
                                         : bi.exact_storage_bytes() ? native_storage_bytes
                                         : bi.storage ? 16u
                                         : sampled_float32_native ? sampled_components * 4u
                                         : sampled_float32 ? 16u
                                         : sampled_depth ? data_format_bytes(r->format)
                                         : sampled_float16_native ? 8u
                                         : sampled_uint32_native ? sampled_components * 4u
                                         : sampled_uint16_native ? sampled_components * 2u
                                         : sampled_unorm16_native ? sampled_components * 2u
                                         : sampled_uint8_native ? sampled_components
                                         : sampled_unorm8x2 ? 2u : 4u;
            const bool sampled_r11g11b10 = !bi.storage &&
                r->format == DataFormat::Float10_11_11 && sampled_components == 3;
            const VkFormat image_format = bi.imported_depth ? bi.imported_format
                : bi.imported
                    ? prosper::frontend::live_target_pixel_format_vk(bi.imported_pixel_format)
                : bi.unorm_rtt_value_reuse ? VK_FORMAT_R8G8B8A8_UNORM
                : bi.packed_r11_storage ? VK_FORMAT_R32_UINT
                : bi.native_float_storage
                ? (r->format == DataFormat::Unorm8
                       ? (sampled_components == 1 ? VK_FORMAT_R8_UNORM
                          : sampled_components == 2 ? VK_FORMAT_R8G8_UNORM
                                                     : VK_FORMAT_R8G8B8A8_UNORM)
                   : r->format == DataFormat::Float16
                       ? (sampled_components == 1 ? VK_FORMAT_R16_SFLOAT
                          : sampled_components == 2 ? VK_FORMAT_R16G16_SFLOAT
                                                     : VK_FORMAT_R16G16B16A16_SFLOAT)
                   : r->format == DataFormat::Float10_11_11
                       ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                       : (sampled_components == 1 ? VK_FORMAT_R32_SFLOAT
                          : sampled_components == 2 ? VK_FORMAT_R32G32_SFLOAT
                                                     : VK_FORMAT_R32G32B32A32_SFLOAT))
                : bi.storage ? VK_FORMAT_R32G32B32A32_UINT
                : sampled_depth
                    ? (r->format == DataFormat::Float32
                           ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D16_UNORM)
                : sampled_float16_native ? VK_FORMAT_R16G16B16A16_SFLOAT
                : sampled_uint8_native
                    ? (sampled_components == 1 ? VK_FORMAT_R8_UINT
                       : sampled_components == 2 ? VK_FORMAT_R8G8_UINT
                                                  : VK_FORMAT_R8G8B8A8_UINT)
                : sampled_uint16_native
                    ? (sampled_components == 1 ? VK_FORMAT_R16_UINT
                       : sampled_components == 2 ? VK_FORMAT_R16G16_UINT
                                                  : VK_FORMAT_R16G16B16A16_UINT)
                : sampled_uint32_native
                    ? (sampled_components == 1 ? VK_FORMAT_R32_UINT
                       : sampled_components == 2 ? VK_FORMAT_R32G32_UINT
                                                  : VK_FORMAT_R32G32B32A32_UINT)
                : sampled_r11g11b10 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32
                : sampled_unorm8x2 ? VK_FORMAT_R8G8_UNORM
                : sampled_unorm16_native
                    ? (sampled_components == 1 ? VK_FORMAT_R16_UNORM
                       : sampled_components == 2 ? VK_FORMAT_R16G16_UNORM
                                                  : VK_FORMAT_R16G16B16A16_UNORM)
                : sampled_float32_native
                    ? (sampled_components == 1 ? VK_FORMAT_R32_SFLOAT
                       : sampled_components == 2 ? VK_FORMAT_R32G32_SFLOAT
                                                  : VK_FORMAT_R32G32B32A32_SFLOAT)
                : sampled_float32 ? VK_FORMAT_R32G32B32A32_SFLOAT
                                  : VK_FORMAT_R8G8B8A8_UNORM;
            // Most storage images use raw uvec4 channels; reflected exact paths keep native-width
            // bytes. Most sampled formats normalize to RGBA8, while HDR/integer formats stay native.
            const VkDeviceSize sbytes = volume_texels * texel_bytes;
            if (!sbytes || sbytes > kMaxComputeImageBytes) {
                skip_image(r, "expanded image exceeds the 512 MiB backend bound"); break;
            }
            staging_bytes[i] = sbytes;

            const uint32_t sampled_bpb = bc_block_bytes(r->format);
            const uint32_t sampled_cb = data_format_bytes(r->format);
            const uint32_t sampled_nc = sampled_components;
            const bool sampled_rgba8 = r->format == DataFormat::Unorm8 && sampled_nc == 4;
            const bool sampled_uint8 = r->format == DataFormat::Uint8 && sampled_nc == 4;
            const bool sampled_r8 = r->format == DataFormat::Unorm8 && sampled_nc == 1;
            const bool sampled_f16 = r->format == DataFormat::Float16 &&
                                     sampled_nc >= 1 && sampled_nc <= 4;
            const bool sampled_f32 = sampled_float32;
            size_t sampled_guest_need = 0;
            const uint8_t* sampled_guest_source = nullptr;
            const uint8_t* sampled_dcc_metadata = nullptr;
            size_t sampled_dcc_metadata_bytes = 0;
            bool sampled_dcc_fast_clear = false;
            static const bool imported_guest_bypass_disabled =
                std::getenv("PROSPER_NO_IMPORTED_IMAGE_GUEST_BYPASS") != nullptr;
            static const bool sampled_dcc_fast_clear_disabled =
                std::getenv("PROSPER_NO_COMPUTE_DCC_FAST_CLEAR") != nullptr;
            if (!bi.depth_bits_source && compute_sampled_guest_prepare_required(
                    bi.storage, renderer_owned, bi.imported,
                    imported_guest_bypass_disabled)) {
                // Resolve and validate the guest source before allocating staging. A proven cache
                // hit can then avoid the staging buffer/map as well as conversion and GPU upload.
                if (sampled_bpb && dim_3d) {
                    skip_image(r, "block-compressed 3D texture deferred"); break;
                } else if (sampled_bpb) {
                    const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                    const size_t slice = r->tile_mode
                        ? tiled_elements_bytes(bw, bh, sampled_bpb, r->tile_mode)
                        : static_cast<size_t>(bw) * bh * sampled_bpb;
                    if (cube_face_as_2d && r->layer_stride_bytes) {
                        const size_t selected_slice = r->in_mip_tail
                            ? r->mip_tail_bytes
                            : (r->tile_mode
                                   ? slice
                                   : linear_sampled_row_pitch(bw, sampled_bpb) * bh);
                        const size_t level_offset = r->in_mip_tail
                            ? 0u : r->layer_mip_offset_bytes;
                        if (!selected_slice || level_offset > SIZE_MAX - selected_slice) {
                            skip_image(r, "sampled cube-face backing size overflows"); break;
                        }
                        sampled_guest_need = level_offset + selected_slice;
                    } else if ((dim_2d_array || dim_cube_stacked) && sampled_layers > 1) {
                        const size_t selected_slice = r->in_mip_tail
                            ? r->mip_tail_bytes : slice;
                        const size_t layer_stride = r->layer_stride_bytes
                            ? r->layer_stride_bytes : selected_slice;
                        const size_t level_offset = r->in_mip_tail
                            ? 0u : r->layer_mip_offset_bytes;
                        if (!selected_slice ||
                            layer_stride > SIZE_MAX / (sampled_layers - 1u) ||
                            level_offset > SIZE_MAX - layer_stride * (sampled_layers - 1u) ||
                            selected_slice > SIZE_MAX -
                                (layer_stride * (sampled_layers - 1u) + level_offset)) {
                            skip_image(r, "sampled array backing size overflows"); break;
                        }
                        sampled_guest_need =
                            layer_stride * (sampled_layers - 1u) + level_offset + selected_slice;
                    } else {
                        sampled_guest_need = slice;
                    }
                } else if (sampled_rgba8 || sampled_uint8 || sampled_r8 || sampled_f16 ||
                           sampled_f32 || sampled_r11g11b10 || sampled_unorm8x2 ||
                           sampled_unorm16_native || sampled_uint8_native ||
                           sampled_uint16_native || sampled_uint32_native || sampled_depth) {
                    const uint32_t bpt = sampled_r11g11b10 ? 4u : sampled_cb * sampled_nc;
                    if (cube_face_as_2d && r->layer_stride_bytes) {
                        const size_t selected_slice = r->in_mip_tail
                            ? r->mip_tail_bytes
                            : (r->tile_mode
                                   ? tiled_surface_bytes(
                                         r->width, r->height, r->tile_mode, 0, bpt)
                                   : (r->linear_row_pitch_bytes
                                          ? linear_array_surface_bytes(*r, bpt)
                                          : static_cast<size_t>(r->width) * r->height * bpt));
                        const size_t level_offset = r->in_mip_tail
                            ? 0u : r->layer_mip_offset_bytes;
                        if (!selected_slice || level_offset > SIZE_MAX - selected_slice) {
                            skip_image(r, "sampled cube-face backing size overflows"); break;
                        }
                        sampled_guest_need = level_offset + selected_slice;
                    } else if (r->tile_mode && dim_3d && r->depth > 1) {
                        sampled_guest_need = tiled_volume_bytes(
                            r->width, r->height, r->depth, r->tile_mode, bpt);
                    } else if ((dim_2d_array || dim_cube_stacked) && sampled_layers > 1) {
                        const size_t slice = r->in_mip_tail
                            ? r->mip_tail_bytes
                            : (r->tile_mode
                                   ? tiled_surface_bytes(
                                         r->width, r->height, r->tile_mode, 0, bpt)
                                   : (r->layer_stride_bytes
                                          ? linear_array_surface_bytes(*r, bpt)
                                          : static_cast<size_t>(r->width) * r->height * bpt));
                        const size_t layer_stride = r->layer_stride_bytes
                            ? r->layer_stride_bytes : slice;
                        const size_t level_offset = r->in_mip_tail
                            ? 0u : r->layer_mip_offset_bytes;
                        if (!slice || layer_stride > SIZE_MAX / (sampled_layers - 1u) ||
                            level_offset > SIZE_MAX - layer_stride * (sampled_layers - 1u) ||
                            slice > SIZE_MAX -
                                (layer_stride * (sampled_layers - 1u) + level_offset)) {
                            skip_image(r, "sampled array backing size overflows"); break;
                        }
                        sampled_guest_need =
                            layer_stride * (sampled_layers - 1u) + level_offset + slice;
                    } else {
                        sampled_guest_need = r->tile_mode
                            ? tiled_surface_bytes(r->width, r->height, r->tile_mode, 0, bpt)
                            : static_cast<size_t>(volume_texels) * bpt;
                    }
                } else {
                    skip_image(r, "sampled format not decodable yet"); break;
                }
                if (!sampled_guest_need || sampled_guest_need > kMaxComputeImageBytes ||
                    sampled_guest_need > UINT32_MAX) {
                    skip_image(r, "sampled backing exceeds the 512 MiB backend bound"); break;
                }
                bi.guest_bytes = sampled_guest_need;

                // A complete uniform DCC fast-clear plane is the authoritative image contents;
                // compressed base bytes are intentionally meaningless in that state. Restrict the
                // first compute materialization to the ordinary 2D RGBA16F -> RGBA8 sampled path,
                // where the renderer already proves the same embedded 0/1 clear codes. This probe
                // happens before resolving the base pointer so a valid fast clear never reads or
                // validates tens of MiB of irrelevant compressed allocation bytes.
                if (r->compression_enabled) {
                    const uint64_t metadata_bytes = gpu_capture_dcc_metadata_footprint(*r);
                    if (metadata_bytes && metadata_bytes <= SIZE_MAX &&
                        metadata_bytes <= UINT32_MAX) {
                        sampled_dcc_metadata_bytes = static_cast<size_t>(metadata_bytes);
                        if (r->dcc_metadata_host_data &&
                            r->dcc_metadata_host_data_size >= metadata_bytes) {
                            sampled_dcc_metadata = r->dcc_metadata_host_data;
                        } else if (guest_readable(
                                       r->metadata_addr,
                                       static_cast<uint32_t>(metadata_bytes))) {
                            sampled_dcc_metadata = reinterpret_cast<const uint8_t*>(
                                uintptr_t(r->metadata_addr));
                        }
                    }
                }
                uint8_t sampled_dcc_clear_pixel[4]{};
                uint8_t sampled_dcc_clear_code = 0;
                sampled_dcc_fast_clear = compute_sampled_dcc_fast_clear_rgba8(
                    *r, !bi.storage && !renderer_owned && !bi.imported,
                    image_descriptors[i].image_arrayed,
                    sampled_dcc_fast_clear_disabled,
                    sampled_dcc_clear_pixel, 1,
                    sampled_dcc_metadata, sampled_dcc_metadata_bytes,
                    &sampled_dcc_clear_code);
                if (sampled_dcc_fast_clear && trace)
                    std::fprintf(stderr,
                                 "[compute]   sampled DCC fast-clear binding=%u addr=0x%llx "
                                 "meta=0x%llx code=0x%02x -> RGBA8 staging\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned long long)r->metadata_addr,
                                 sampled_dcc_clear_code);

                if (!sampled_dcc_fast_clear) {
                    sampled_guest_source = resource_bytes_for(r, sampled_guest_need);
                    const bool readable =
                        (r->host_data && r->host_data_size >= sampled_guest_need) ||
                        guest_readable(r->gpu_addr, static_cast<uint32_t>(sampled_guest_need));
                    if (!readable) { skip_image(r, "sampled surface unreadable"); break; }
                }

                bool dcc_cache_safe = !r->compression_enabled;
                if (r->compression_enabled) {
                    dcc_cache_safe = sampled_dcc_metadata && sampled_dcc_metadata_bytes &&
                        std::all_of(sampled_dcc_metadata,
                                    sampled_dcc_metadata + sampled_dcc_metadata_bytes,
                                    [](uint8_t value) { return value == 0xff; });
                }
                // Fast clears are keyed by metadata, not by the inactive base allocation. Keep
                // this first narrow path uncached rather than teaching the existing base-byte key
                // an unsafe partial identity; staging materialization is still far cheaper than
                // detiling and converting the 2x-larger FP16 base.
                bi.cache_candidate = !sampled_dcc_fast_clear && dcc_cache_safe &&
                    persistent_compute_image_enabled(sbytes, ComputeImageCacheClass::sampled);
                const VkFormat transfer_native_format =
                    native_storage_vk_format(r->format, sampled_components);
                const bool transfer_format_compatible =
                    compute_native_2d_transfer_format_compatible(
                        r->format, sampled_components);
                const bool native_transfer_dimension =
                    (dim_3d && native_3d_transfer_enabled()) ||
                    (ordinary_2d_view && native_2d_transfer_enabled() &&
                     transfer_format_compatible);
                const bool transfer_hostless = !r->host_data;
                const bool transfer_format_match = transfer_native_format == image_format;
                const bool transfer_validation_enabled =
                    adaptive_storage_result_validation_enabled();
                const bool transfer_native_defined =
                    transfer_native_format != VK_FORMAT_UNDEFINED;
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   native transfer gate binding=%u dimension=%u "
                                 "hostless=%u format-match=%u validation=%u defined=%u "
                                 "storage-format=%u sampled-format=%u\n",
                                 bi.binding, native_transfer_dimension ? 1u : 0u,
                                 transfer_hostless ? 1u : 0u,
                                 transfer_format_match ? 1u : 0u,
                                 transfer_validation_enabled ? 1u : 0u,
                                 transfer_native_defined ? 1u : 0u,
                                 static_cast<unsigned>(transfer_native_format),
                                 static_cast<unsigned>(image_format));
                ComputeTransferBorrowResult transfer_borrow_result =
                    ComputeTransferBorrowResult::NotAttempted;
                if (bi.cache_candidate) {
                    const auto cache_lookup_start = ComputeClock::now();
                    bi.cache_key = {
                        r->gpu_addr, reinterpret_cast<uintptr_t>(r->host_data),
                        static_cast<uint32_t>(sampled_guest_need), r->size,
                        r->width, r->height, r->depth,
                        static_cast<uint32_t>(r->format), sampled_components,
                        r->tile_mode, r->img_dim, r->linear_row_pitch_bytes,
                        r->layer_stride_bytes, r->layer_mip_offset_bytes,
                        r->mip_tail_offset, r->mip_tail_bytes,
                        r->mip_tail_x, r->mip_tail_y,
                        static_cast<uint32_t>(image_format), bi.storage, r->in_mip_tail,
                        r->srgb, r->depth_compare};
                    // An ordinary native typed 2D image or native typed 3D volume is byte- and
                    // format-identical to the sampled upload that follows it. Borrow that retained
                    // result only as a TRANSFER source: the sampled cache remains a separate image so
                    // a read/modify/write dispatch can sample the old image while producing its
                    // replacement at the same guest address. Exact key lookup plus the ordered-submit
                    // journal (or an independently clean write watch) makes an intervening guest write
                    // fail closed to the existing upload path.
                    if (native_transfer_dimension && transfer_hostless &&
                        transfer_format_match && transfer_validation_enabled &&
                        transfer_native_defined) {
                        const ComputeImageCacheKey storage_key = storage_image_cache_key(
                            *r, static_cast<uint32_t>(sampled_guest_need),
                            transfer_native_format);
                        if (ctx.borrow_cached_image_for_compute_transfer(
                                storage_key, bi.compute_transfer_seed, trace,
                                &transfer_borrow_result)) {
                            bi.compute_transfer_seed_key = storage_key;
                            bi.compute_transfer_seed_borrowed = true;
                            g_compute_storage_transfer_seeds.fetch_add(
                                1, std::memory_order_relaxed);
                            if (trace)
                                std::fprintf(stderr,
                                             "[compute]   sampled %s binding=%u addr=0x%llx "
                                             "seeded from retained native storage image\n",
                                             dim_3d ? "3D" : "2D", bi.binding,
                                             (unsigned long long)r->gpu_addr);
                        }
                    }
                    bi.persistent = ctx.acquire_cached_image(
                        bi.cache_key,
                        bi.compute_transfer_seed_borrowed ? nullptr : sampled_guest_source,
                        image_validation_epoch, bi.image, bi.memory, bi.upload_skipped,
                        !bi.compute_transfer_seed_borrowed);
                    if (bi.persistent && bi.upload_skipped)
                        g_sampled_image_upload_skips.fetch_add(1, std::memory_order_relaxed);
                    if (trace && bi.persistent)
                        std::fprintf(stderr,
                                     "[compute]   persistent sampled image binding=%u "
                                     "addr=0x%llx guest=%zu upload-skipped=%u\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     sampled_guest_need, bi.upload_skipped ? 1u : 0u);
                    if (!bi.persistent && !bi.compute_transfer_seed_borrowed)
                        bi.cache_source_snapshot.assign(
                            sampled_guest_source,
                            sampled_guest_source + sampled_guest_need);
                    cache_lookup_ms = std::chrono::duration<double, std::milli>(
                        ComputeClock::now() - cache_lookup_start).count();
                }
                transfer_gate_census.record_sampled_gates(
                    transfer_gate_observation.role, *r, bi.binding,
                    bi.cache_candidate, bi.persistent, ordinary_2d_view,
                    transfer_format_compatible, native_transfer_dimension,
                    transfer_hostless, transfer_format_match,
                    transfer_validation_enabled, transfer_native_defined,
                    transfer_borrow_result, transfer_gate_observation.first_match);
            }

            // Allocate the host-visible staging buffer before conversion and write into its mapping
            // directly. The old path first built a heap upload and then memcpy'd the complete result
            // here -- an extra 132 MiB CPU pass for Astro Bot's 4K RGBA32 storage representation.
            ScopedMappedMemory upload_mapping(ctx);
            const size_t upload_size = (bi.imported || bi.depth_bits_source || bi.seed_skip ||
                                        bi.seed_from_imported != SIZE_MAX ||
                                        bi.compute_transfer_seed_borrowed)
                ? size_t{0} : (size_t)sbytes;
            // A retained storage image still needs a fresh destination for its post-dispatch
            // transfer. Only a read-only sampled cache hit can omit staging altogether.
            const auto staging_start = ComputeClock::now();
            if (!bi.imported &&
                (bi.storage || (!bi.compute_transfer_seed_borrowed &&
                                !(bi.persistent && bi.upload_skipped)))) {
                VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                sci.size = sbytes;
                sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                if (!vk_ok(vkCreateBuffer(ctx.device, &sci, nullptr, &staging[i]),
                           "image-staging-buffer")) { images_ready = false; break; }
                VkMemoryRequirements sreq{};
                vkGetBufferMemoryRequirements(ctx.device, staging[i], &sreq);
                bi.staging_allocation_bytes = sreq.size;
                const uint32_t staging_memory_type = ctx.host_memory_type(sreq.memoryTypeBits);
                staging_memory[i] = ctx.allocate_memory(sreq.size, staging_memory_type, true);
                if (!vk_handle_ok(staging_memory[i], "image-staging-memory") ||
                    !vk_ok(vkBindBufferMemory(ctx.device, staging[i], staging_memory[i], 0),
                           "image-staging-bind")) {
                    images_ready = false; break;
                }
                upload_mapping.memory = staging_memory[i];
                if (!bi.depth_bits_source && !bi.seed_skip &&
                    bi.seed_from_imported == SIZE_MAX &&
                    !bi.compute_transfer_seed_borrowed &&
                    !(bi.persistent && bi.upload_skipped) &&
                    !vk_ok(ctx.map_memory(staging_memory[i], 0, sbytes,
                                          &upload_mapping.data), "image-staging-map")) {
                    images_ready = false; break;
                }
            }
            staging_ms = std::chrono::duration<double, std::milli>(
                ComputeClock::now() - staging_start).count();
            auto* upload = static_cast<uint8_t*>(upload_mapping.data);
            const auto prepare_upload_start = ComputeClock::now();
            if (bi.imported) {
                // The renderer's sampled image is the source, so direct imports need no transfer.
                bi.guest_bytes = 0;
            } else if (bi.depth_bits_source) {
                // The command buffer below copies the borrowed D32 depth plane through this
                // binding's device buffer into its owned R32_UINT image. There are deliberately no
                // host upload bytes to prepare: touching sampled_guest_source here would reinstate
                // the stale guest-memory defect this bridge exists to close.
                bi.guest_bytes = 0;
            } else if (bi.storage) {
                if (r->tile_mode && !tile_mode_is_tiled(r->tile_mode)) {
                    skip_image(r, "storage tile mode has no supported address pattern"); break;
                }
                if (r->tile_mode && !dim_3d &&
                    std::getenv("PROSPER_DISABLE_COMPUTE_TILED_2D_STORAGE")) {
                    skip_image(r, "tiled 1D/2D storage writeback disabled"); break;
                }
                if (!storage_unpack_supported(r->format) || !storage_pack_supported(r->format)) {
                    skip_image(r, "storage format has no channel pack/unpack yet"); break; }
                const uint32_t cb = data_format_bytes(r->format);
                const uint32_t nc = r->num_components ? r->num_components : 1;
                const size_t guest_texel = (r->format == DataFormat::Float10_11_11 ||
                                            r->format == DataFormat::Unorm2_10_10_10)
                    ? 4u : (size_t)cb * nc;
                const size_t texels = (size_t)volume_texels;
                const uint64_t linear_guest_bytes = static_cast<uint64_t>(texels) * guest_texel;
                bi.exact_result_bytes = bi.exact_storage_bytes()
                    ? static_cast<VkDeviceSize>(linear_guest_bytes) : sbytes;
                size_t guest_bytes = r->tile_mode
                    ? (dim_3d && r->depth > 1
                           ? tiled_volume_bytes(r->width, r->height, r->depth, r->tile_mode,
                                                static_cast<uint32_t>(guest_texel))
                           : tiled_surface_bytes(r->width, r->height, r->tile_mode, 0,
                                                 static_cast<uint32_t>(guest_texel)))
                    : static_cast<size_t>(linear_guest_bytes);
                size_t array_slice_bytes = 0;
                if (dim_2d_array && r->depth > 1) {
                    array_slice_bytes = r->in_mip_tail
                        ? r->mip_tail_bytes
                        : (r->tile_mode
                               ? tiled_surface_bytes(r->width, r->height, r->tile_mode, 0,
                                                     static_cast<uint32_t>(guest_texel))
                               : (r->layer_stride_bytes
                                      ? linear_array_surface_bytes(
                                            *r, static_cast<uint32_t>(guest_texel))
                                      : static_cast<size_t>(r->width) * r->height * guest_texel));
                    const size_t layer_stride = r->layer_stride_bytes
                        ? r->layer_stride_bytes : array_slice_bytes;
                    const size_t level_offset = r->in_mip_tail
                        ? 0u : r->layer_mip_offset_bytes;
                    if (!array_slice_bytes || layer_stride > SIZE_MAX / (r->depth - 1u) ||
                        level_offset > SIZE_MAX - layer_stride * (r->depth - 1u) ||
                        array_slice_bytes > SIZE_MAX -
                            (layer_stride * (r->depth - 1u) + level_offset)) {
                        skip_image(r, "storage array backing size overflows"); break;
                    }
                    guest_bytes = layer_stride * (r->depth - 1u) + level_offset + array_slice_bytes;
                }
                if (!linear_guest_bytes || linear_guest_bytes > SIZE_MAX || !guest_bytes ||
                    guest_bytes > UINT32_MAX ||
                    (!r->tile_mode && !r->layer_stride_bytes && guest_bytes > r->size)) {
                    skip_image(r, "storage backing size is invalid"); break;
                }
                bi.guest_bytes = guest_bytes;
                const uint8_t* src = (renderer_owned || bi.seed_skip)
                    ? nullptr : resource_bytes_for(r, guest_bytes);
                // The writeback still writes up to guest_bytes at r->gpu_addr, so a guest-backed
                // target must be a valid mapped range even when the SEED read is skipped (#1122
                // review B2): keep the guard for the writeback target, not the seed source.
                const bool readable = renderer_owned ||
                                      (r->host_data && r->host_data_size >= guest_bytes) ||
                                      guest_readable(r->gpu_addr, static_cast<uint32_t>(guest_bytes));
                if (!readable) { skip_image(r, "storage backing unreadable"); break; }
                // Exact-width storage targets are byte-identical to guest row-major bytes. A raw-uvec4
                // target is also safe to retain when seed_skip proves the shader is a full, write-only
                // producer: the previous raw values are then unobservable, while retaining them lets
                // the exact GPU comparator recognize a repeated result before CPU pack/retile.
                // Poison proving deliberately stays transient because a partial proof repairs
                // untouched texels only in the CPU mirror after readback.
                const bool dcc_cache_safe = !r->compression_enabled ||
                    (bi.dcc_metadata && bi.dcc_metadata_bytes &&
                     std::all_of(bi.dcc_metadata, bi.dcc_metadata + bi.dcc_metadata_bytes,
                                 [](uint8_t value) { return value == 0xff; }));
                const ComputeStorageCacheGateInputs storage_cache_gates{
                    renderer_owned,
                    dcc_cache_safe,
                    bi.poison_verify,
                    bi.exact_storage_bytes(),
                    bi.seed_skip,
                    persistent_compute_image_enabled(
                        sbytes, ComputeImageCacheClass::storage),
                };
                // Build the same exact identity used by ordinary admission before choosing either
                // path.  A compressed target may become eligible only after successful writeback;
                // deriving a second, looser key there could replace an unrelated cached image.
                bi.cache_key = storage_image_cache_key(
                    *r, static_cast<uint32_t>(guest_bytes), image_format);
                bi.cache_candidate =
                    compute_storage_cache_gate_candidate(storage_cache_gates);
                bi.post_writeback_promotion_candidate =
                    compute_storage_post_writeback_promotion_candidate({
                        storage_cache_gates,
                        bi.dcc_metadata && bi.dcc_metadata_bytes,
                        bi.alias_of == SIZE_MAX,
                    });
                if (bi.cache_candidate) {
                    const auto cache_lookup_start = ComputeClock::now();
                    bi.persistent = ctx.acquire_cached_image(
                        bi.cache_key, resource_bytes_for(r, guest_bytes), image_validation_epoch,
                        bi.image, bi.memory, bi.upload_skipped,
                        !bi.seed_skip || !adaptive_storage_result_validation_enabled());
                    if (bi.persistent)
                        ctx.cached_image_result_buffer(
                            bi.cache_key, bi.exact_result_bytes, bi.result_baseline);
                    if (trace && bi.persistent)
                        std::fprintf(stderr,
                                     "[compute]   persistent storage image binding=%u "
                                     "addr=0x%llx guest=%zu upload-skipped=%u\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     guest_bytes, bi.upload_skipped ? 1u : 0u);
                    cache_lookup_ms = std::chrono::duration<double, std::milli>(
                        ComputeClock::now() - cache_lookup_start).count();
                }
                transfer_gate_census.record_storage_cache(
                    transfer_gate_observation.role, *r, bi.binding,
                    storage_cache_gates,
                    bi.cache_candidate, bi.persistent);
                if (!bi.cache_candidate && bi.post_writeback_promotion_candidate) {
                    const auto cache_lookup_start = ComputeClock::now();
                    bi.forced_seed_allocation_reused =
                        ctx.acquire_cached_image_allocation_for_forced_seed(
                            bi.cache_key, bi.image, bi.memory, bi.allocation_bytes);
                    bi.persistent = bi.forced_seed_allocation_reused;
                    if (bi.persistent)
                        ctx.cached_image_result_buffer(
                            bi.cache_key, bi.exact_result_bytes, bi.result_baseline);
                    if (trace && bi.persistent)
                        std::fprintf(stderr,
                                     "[compute]   reused exact storage allocation with forced "
                                     "seed binding=%u addr=0x%llx guest=%zu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     guest_bytes);
                    cache_lookup_ms += std::chrono::duration<double, std::milli>(
                        ComputeClock::now() - cache_lookup_start).count();
                }
                if (!(bi.persistent && bi.upload_skipped)) {
                const size_t linear_size = (bi.seed_skip || bi.seed_from_imported != SIZE_MAX)
                    ? size_t{0} : static_cast<size_t>(linear_guest_bytes);
                std::unique_ptr<uint8_t[]> linear;
                if (linear_size && !renderer_owned &&
                    (r->tile_mode || (dim_2d_array && r->depth > 1)))
                    linear = std::make_unique_for_overwrite<uint8_t[]>(linear_size);
                const uint8_t* unpack_source = nullptr;
                if (bi.seed_skip) {
                    // #1122: write-only full-coverage target -- the shader overwrites every texel, so
                    // the image is created but never seeded, uploaded, or read. Nothing to fill.
                } else if (bi.seed_from_imported != SIZE_MAX) {
                    // The command buffer copies the renderer's exact native image into this target.
                    // Staging remains allocated because the result still has to be read back below.
                } else if (renderer_owned) {
                    unpack_source = live_target.pixels->data();
                    if (trace) {
                        bi.before_hash = fnv1a(unpack_source, linear_size);
                        std::fprintf(stderr,
                                     "[compute]   imported writable renderer RTT binding=%u "
                                     "addr=0x%llx extent=%ux%u format=%s\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     r->width, r->height,
                                     prosper::frontend::live_target_pixel_format_name(
                                         live_target.format));
                    }
                } else if (r->tile_mode && dim_3d && r->depth > 1) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    if (!detile_volume(linear.get(), src, guest_bytes, r->width, r->height,
                                       r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                        skip_image(r, "storage volume detile failed"); break;
                    }
                    unpack_source = linear.get();
                } else if (dim_2d_array && r->depth > 1) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    const size_t linear_slice = static_cast<size_t>(r->width) * r->height * guest_texel;
                    const size_t layer_stride = r->layer_stride_bytes
                        ? r->layer_stride_bytes : array_slice_bytes;
                    for (uint32_t layer = 0; layer < r->depth; ++layer) {
                        const uint8_t* layer_base = src + layer_stride * layer;
                        if (!r->tile_mode) {
                            const size_t row_pitch = r->layer_stride_bytes
                                ? linear_array_row_pitch(
                                      *r, static_cast<uint32_t>(guest_texel))
                                : static_cast<size_t>(r->width) * guest_texel;
                            for (uint32_t y = 0; y < r->height; ++y)
                                std::memcpy(
                                    linear.get() + linear_slice * layer +
                                        static_cast<size_t>(y) * r->width * guest_texel,
                                    layer_base + r->layer_mip_offset_bytes + y * row_pitch,
                                    static_cast<size_t>(r->width) * guest_texel);
                        } else if (r->in_mip_tail) {
                            detile_surface_level(
                                linear.get() + linear_slice * layer, layer_base,
                                r->mip_tail_bytes, r->width, r->height, r->tile_mode,
                                static_cast<uint32_t>(guest_texel), r->mip_tail_x, r->mip_tail_y);
                        } else {
                            detile_surface(
                                linear.get() + linear_slice * layer,
                                layer_base + r->layer_mip_offset_bytes,
                                r->width, r->height, r->tile_mode, 0,
                                static_cast<uint32_t>(guest_texel));
                        }
                    }
                    unpack_source = linear.get();
                } else if (r->tile_mode && r->in_mip_tail) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    detile_surface_level(linear.get(), src, guest_bytes,
                                         r->width, r->height, r->tile_mode,
                                         static_cast<uint32_t>(guest_texel),
                                         r->mip_tail_x, r->mip_tail_y);
                    unpack_source = linear.get();
                } else if (r->tile_mode) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    detile_surface(linear.get(), src, r->width, r->height, r->tile_mode, 0,
                                   static_cast<uint32_t>(guest_texel));
                    unpack_source = linear.get();
                } else {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    // Linear guest storage is already in the row-major layout consumed by unpack.
                    // Decode it in place instead of copying the complete surface to a temporary first.
                    unpack_source = src;
                }
                if (!bi.seed_skip && bi.seed_from_imported == SIZE_MAX) {
                    if (bi.exact_storage_bytes()) {
                        parallel_compute_texels(texels, static_cast<size_t>(linear_guest_bytes) * 2,
                            [&](size_t begin, size_t end) {
                                std::memcpy(upload + begin * guest_texel,
                                            unpack_source + begin * guest_texel,
                                            (end - begin) * guest_texel);
                            });
                    } else
                        storage_unpack_range(unpack_source, guest_texel, r->format, nc, texels,
                                             reinterpret_cast<uint32_t*>(upload));
                }
                if (bi.poison_verify) {   // #1122 coverage proof: poison every uvec4 channel
                    // Keep the clean detiled guest seed: on a partial-coverage proving frame the
                    // writeback restores every un-stored (still-poison) texel from it, so proving
                    // never corrupts the guest -- only the GPU `upload` is poisoned, `linear` is not.
                    bi.seed_linear.assign(unpack_source, unpack_source + linear_size);
                    if (bi.exact_storage_bytes()) {
                        // Transfers preserve this finite, non-special native texel pattern exactly.
                        // A texel that remains all 0x5a after the dispatch was not stored.
                        std::memset(upload, 0x5a, static_cast<size_t>(sbytes));
                    } else {
                        uint32_t* pp = reinterpret_cast<uint32_t*>(upload);
                        for (size_t t = 0; t < texels * 4; ++t) pp[t] = 0xDEADBEEFu;
                    }
                }
                static const bool verify_unpack =
                    std::getenv("PROSPER_VERIFY_UNPACK") != nullptr && !bi.seed_skip &&
                    !bi.exact_storage_bytes();
                if (verify_unpack) {
                    // Fail-visible A/B: the specialized range unpack must be bit-identical to the
                    // per-texel path it replaces. Reports the whole divergence (count + first texel)
                    // and identifies the binding, and logs the clean case too so a verified run is
                    // self-proving rather than merely silent.
                    uint32_t expect[4];
                    const uint32_t* got = reinterpret_cast<const uint32_t*>(upload);
                    size_t bad = 0, first_bad = 0;
                    for (size_t t = 0; t < texels; ++t) {
                        storage_unpack_texel(unpack_source + t * guest_texel,
                                             r->format, nc, expect);
                        if (std::memcmp(expect, got + t * 4, sizeof expect) != 0) {
                            if (!bad) first_bad = t;
                            ++bad;
                        }
                    }
                    std::fprintf(stderr,
                                 "[compute] unpack-verify binding=%u addr=0x%llx fmt=%u nc=%u "
                                 "texels=%zu mismatches=%zu%s\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned)r->format, nc, texels, bad,
                                 bad ? " MISMATCH" : "");
                    if (bad)
                        std::fprintf(stderr, "[compute]   first mismatching texel=%zu\n", first_bad);
                }
                }
            } else {
                const uint32_t bpb = sampled_bpb;
                const uint32_t cb = sampled_cb;
                const uint32_t nc = sampled_nc;
                const bool rgba8 = sampled_rgba8;
                const bool uint8 = sampled_uint8;
                const bool r8 = sampled_r8;
                const bool f16 = sampled_f16;
                const bool f32 = sampled_f32;
                const bool r11g11b10 = sampled_r11g11b10;
                if (renderer_owned) {
                    const std::vector<uint8_t>& pixels = *live_target.pixels;
                    // Classify the snapshot's texel layout once. Every conversion below reads it as
                    // either UNORM8x4 or FLOAT16x4; the packed R11G11B10 layout is served by the
                    // byte-copy and reconstruction branches, and a packed renderer target under any
                    // other view already declined ownership above. Testing the layout instead of
                    // "is it RGBA8, else assume FP16" keeps a future pixel format from silently
                    // reading eight bytes out of a four-byte texel.
                    using prosper::frontend::LiveTargetSourceLayout;
                    const LiveTargetSourceLayout source_layout =
                        prosper::frontend::live_target_source_layout(live_target.format);
                    const bool source_unorm8 = source_layout == LiveTargetSourceLayout::Unorm8x4;
                    const bool source_float16 = source_layout == LiveTargetSourceLayout::Float16x4;
                    // DO NOT DELETE AS DEAD CODE. This never fires for today's three enumerators,
                    // but it is not redundant: the packed-view check above is written as
                    // `format == R11G11B10Float && <incompatible view>`, which for any NEW format
                    // is false and therefore RETAINS ownership. That predicate fails OPEN, so this
                    // guard and the `!bpp` decline above are what actually keep an unclassified
                    // layout out of the two-layout conversions below - where "not UNORM8" means
                    // "read eight bytes per texel" and a four-byte snapshot is read out of bounds.
                    if (!bi.unorm_rtt_value_reuse && !r11g11b10 &&
                        !source_unorm8 && !source_float16) {
                        skip_image(r, "renderer-owned RTT layout has no sampled conversion"); break;
                    }
                    if (bi.unorm_rtt_value_reuse) {
                        std::memcpy(upload, pixels.data(), upload_size);
                    } else if (r11g11b10) {
                        if (!pack_live_target_r11g11b10(live_target, upload, upload_size)) {
                            skip_image(r, "renderer RTT R11G11B10 reconstruction failed");
                            break;
                        }
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   reconstructed renderer RTT binding=%u "
                                         "addr=0x%llx extent=%ux%u %s -> R11G11B10\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         r->width, r->height,
                                         prosper::frontend::live_target_pixel_format_name(
                                             live_target.format));
                    } else if (sampled_float32_native || sampled_uint16_native ||
                               sampled_uint32_native) {
                        // A renderer-owned target is authoritative only when its cached texel is
                        // byte-identical to the UINT view.  Sonic aliases an RGBA16F render target
                        // as RGBA16_UINT for a compute resolve: those eight bytes must be
                        // reinterpreted, not numerically converted through float or UNORM.
                        std::memcpy(upload, pixels.data(), upload_size);
                    } else if (sampled_uint8_native) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < sampled_components; ++c) {
                                if (source_unorm8) {
                                    upload[t * sampled_components + c] = pixels[t * 4 + c];
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float value = half_to_float(half);
                                    if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                    else if (value >= 255.0f) value = 255.0f;
                                    upload[t * sampled_components + c] = static_cast<uint8_t>(
                                        std::lround(value));
                                }
                            }
                        }
                    } else if (sampled_unorm8x2) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < 2; ++c) {
                                if (source_unorm8) {
                                    upload[t * 2 + c] = pixels[t * 4 + c];
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float value = half_to_float(half);
                                    if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                    else if (value >= 1.0f) value = 1.0f;
                                    upload[t * 2 + c] = static_cast<uint8_t>(
                                        std::lround(value * 255.0f));
                                }
                            }
                        }
                    } else if (sampled_unorm16_native) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < sampled_components; ++c) {
                                uint16_t value = 0;
                                if (source_unorm8) {
                                    value = static_cast<uint16_t>(pixels[t * 4 + c]) * 257u;
                                } else {
                                    uint16_t half = 0;
                                    std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                    float f = half_to_float(half);
                                    if (!std::isfinite(f) || f <= 0.0f) f = 0.0f;
                                    else if (f >= 1.0f) f = 1.0f;
                                    value = static_cast<uint16_t>(std::lround(f * 65535.0f));
                                }
                                std::memcpy(upload +
                                                (t * sampled_components + c) * sizeof(value),
                                            &value, sizeof(value));
                            }
                        }
                    } else if (f32) {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        const uint32_t output_components =
                            sampled_float32_native ? sampled_components : 4u;
                        parallel_compute_texels(
                            texels,
                            texels * (output_components * sizeof(float) + 8u),
                            [&](size_t begin, size_t end) {
                                for (size_t t = begin; t < end; ++t) {
                                    for (uint32_t c = 0; c < output_components; ++c) {
                                        float value = 0.0f;
                                        if (source_unorm8) {
                                            value = pixels[t * 4 + c] / 255.0f;
                                        } else {
                                            uint16_t half = 0;
                                            std::memcpy(&half, pixels.data() + t * 8 + c * 2,
                                                        sizeof(half));
                                            value = half_to_float(half);
                                            if (std::isnan(value)) value = 0.0f;
                                        }
                                        std::memcpy(upload +
                                                        (t * output_components + c) * sizeof(float),
                                                    &value, sizeof(value));
                                    }
                                }
                            });
                    } else if (source_unorm8) {
                        std::memcpy(upload, pixels.data(), upload_size);
                    } else {
                        const size_t texels = static_cast<size_t>(volume_texels);
                        for (size_t t = 0; t < texels; ++t) {
                            for (uint32_t c = 0; c < 4; ++c) {
                                uint16_t half = 0;
                                std::memcpy(&half, pixels.data() + t * 8 + c * 2, sizeof(half));
                                float value = half_to_float(half);
                                if (!std::isfinite(value) || value <= 0.0f) value = 0.0f;
                                else if (value >= 1.0f) value = 1.0f;
                                upload[t * 4 + c] = static_cast<uint8_t>(
                                    std::lround(value * 255.0f));
                            }
                        }
                    }
                    if (trace) {
                        const uint64_t snapshot_hash = fnv1a(live_target.pixels->data(),
                                                             live_target.pixels->size());
                        const size_t nonzero_bytes = static_cast<size_t>(std::count_if(
                            live_target.pixels->begin(), live_target.pixels->end(),
                            [](uint8_t value) { return value != 0; }));
                        std::fprintf(stderr,
                                     "[compute]   imported renderer RTT binding=%u addr=0x%llx "
                                     "extent=%ux%u format=%s hash=%016llx nonzero-bytes=%zu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr, r->width, r->height,
                                     prosper::frontend::live_target_pixel_format_name(
                                         live_target.format),
                                     (unsigned long long)snapshot_hash, nonzero_bytes);
                    }
                    bi.guest_bytes = 0;
                } else {
                    const size_t need = sampled_guest_need;
                    const uint8_t* src = sampled_guest_source;
                    const bool needs_sampled_upload = !bi.upload_skipped &&
                        !bi.compute_transfer_seed_borrowed;
                    if (needs_sampled_upload && sampled_dcc_fast_clear) {
                        if (!compute_sampled_dcc_fast_clear_rgba8(
                                *r, true, image_descriptors[i].image_arrayed,
                                sampled_dcc_fast_clear_disabled,
                                upload, static_cast<size_t>(volume_texels),
                                sampled_dcc_metadata, sampled_dcc_metadata_bytes)) {
                            skip_image(r, "sampled DCC fast-clear materialization changed");
                            break;
                        }
                    } else if (needs_sampled_upload && bpb) {        // BCn: (block-detile ->) decode
                        const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                        const size_t linear_slice = static_cast<size_t>(bw) * bh * bpb;
                        const uint32_t layers = sampled_layers;
                        std::unique_ptr<uint8_t[]> linear;
                        const bool layered = layers > 1;
                        const size_t selected_slice = r->in_mip_tail
                            ? r->mip_tail_bytes
                            : (r->tile_mode
                                   ? tiled_elements_bytes(bw, bh, bpb, r->tile_mode)
                                   : linear_slice);
                        const size_t layer_stride = r->layer_stride_bytes
                            ? r->layer_stride_bytes : selected_slice;
                        const size_t level_offset = r->in_mip_tail
                            ? 0u : r->layer_mip_offset_bytes;
                        const bool padded_linear = !r->tile_mode &&
                            (r->layer_stride_bytes || r->linear_row_pitch_bytes);
                        const bool remap = r->tile_mode || padded_linear || level_offset ||
                            (layered && layer_stride != linear_slice);
                        const uint8_t* decode_source = src;
                        if (remap) {
                            linear = std::make_unique_for_overwrite<uint8_t[]>(linear_slice * layers);
                            decode_source = linear.get();
                        }
                        if (remap) {
                            for (uint32_t layer = 0; layer < layers; ++layer) {
                                const uint8_t* selected = src + layer_stride * layer + level_offset;
                                uint8_t* linear_layer = linear.get() + linear_slice * layer;
                                if (r->tile_mode && r->in_mip_tail) {
                                    detile_elements_level(
                                        linear_layer, selected, selected_slice,
                                        bw, bh, bpb, r->tile_mode,
                                        r->mip_tail_x, r->mip_tail_y);
                                } else if (r->tile_mode) {
                                    detile_elements(linear_layer, selected, selected_slice,
                                                    bw, bh, bpb, r->tile_mode);
                                } else {
                                    const size_t row_pitch = linear_sampled_row_pitch(bw, bpb);
                                    for (uint32_t y = 0; y < bh; ++y)
                                        std::memcpy(
                                            linear_layer + static_cast<size_t>(y) * bw * bpb,
                                            selected + static_cast<size_t>(y) * row_pitch,
                                            static_cast<size_t>(bw) * bpb);
                                }
                            }
                        }
                        for (uint32_t layer = 0; layer < layers; ++layer) {
                            if (!bc_decode_surface(
                                    upload + static_cast<size_t>(layer) * r->width * r->height * 4u,
                                    decode_source + linear_slice * layer, linear_slice,
                                    r->width, r->height, r->format)) {
                                skip_image(r, "BC decode unsupported"); break;
                            }
                        }
                        if (!images_ready) break;
                    } else if (needs_sampled_upload) {
                        const uint32_t bpt = r11g11b10 ? 4u : cb * nc;
                        const size_t linear_bytes = static_cast<size_t>(volume_texels) * bpt;
                        std::unique_ptr<uint8_t[]> linear;
                        const uint8_t* sampled_source = src;
                        const bool remap = r->tile_mode ||
                            (cube_face_as_2d && r->layer_stride_bytes) ||
                            (dim_2d_array && r->depth > 1);
                        if (remap) {
                            linear = std::make_unique_for_overwrite<uint8_t[]>(linear_bytes);
                            sampled_source = linear.get();
                        }
                        if (cube_face_as_2d && r->layer_stride_bytes) {
                            const size_t level_offset = r->in_mip_tail
                                ? 0u : r->layer_mip_offset_bytes;
                            const uint8_t* selected = src + level_offset;
                            const size_t selected_bytes = need - level_offset;
                            if (r->tile_mode && r->in_mip_tail) {
                                detile_surface_level(
                                    linear.get(), selected, selected_bytes,
                                    r->width, r->height, r->tile_mode, bpt,
                                    r->mip_tail_x, r->mip_tail_y);
                            } else if (r->tile_mode) {
                                detile_surface(
                                    linear.get(), selected, r->width, r->height,
                                    r->tile_mode, 0, bpt);
                            } else {
                                const size_t row_pitch = r->linear_row_pitch_bytes
                                    ? linear_array_row_pitch(*r, bpt)
                                    : static_cast<size_t>(r->width) * bpt;
                                for (uint32_t y = 0; y < r->height; ++y)
                                    std::memcpy(
                                        linear.get() + static_cast<size_t>(y) * r->width * bpt,
                                        selected + static_cast<size_t>(y) * row_pitch,
                                        static_cast<size_t>(r->width) * bpt);
                            }
                        } else if (r->tile_mode && dim_3d && r->depth > 1) {
                            if (!detile_volume(linear.get(), src, need,
                                               r->width, r->height, r->depth,
                                               r->tile_mode, bpt)) {
                                skip_image(r, "sampled volume detile failed"); break;
                            }
                        } else if ((dim_2d_array || dim_cube_stacked) && sampled_layers > 1) {
                            const size_t selected_slice = r->in_mip_tail
                                ? r->mip_tail_bytes
                                : (r->tile_mode
                                       ? tiled_surface_bytes(
                                             r->width, r->height, r->tile_mode, 0, bpt)
                                       : (r->layer_stride_bytes
                                              ? linear_array_surface_bytes(*r, bpt)
                                              : static_cast<size_t>(r->width) * r->height * bpt));
                            const size_t layer_stride = r->layer_stride_bytes
                                ? r->layer_stride_bytes : selected_slice;
                            const size_t linear_slice = static_cast<size_t>(r->width) * r->height * bpt;
                            for (uint32_t layer = 0; layer < sampled_layers; ++layer) {
                                const uint8_t* layer_base = src + layer_stride * layer;
                                if (!r->tile_mode) {
                                    const size_t row_pitch = r->layer_stride_bytes
                                        ? linear_array_row_pitch(*r, bpt)
                                        : static_cast<size_t>(r->width) * bpt;
                                    for (uint32_t y = 0; y < r->height; ++y)
                                        std::memcpy(
                                            linear.get() + linear_slice * layer +
                                                static_cast<size_t>(y) * r->width * bpt,
                                            layer_base + r->layer_mip_offset_bytes + y * row_pitch,
                                            static_cast<size_t>(r->width) * bpt);
                                } else if (r->in_mip_tail) {
                                    detile_surface_level(
                                        linear.get() + linear_slice * layer, layer_base,
                                        r->mip_tail_bytes, r->width, r->height,
                                        r->tile_mode, bpt, r->mip_tail_x, r->mip_tail_y);
                                } else {
                                    detile_surface(
                                        linear.get() + linear_slice * layer,
                                        layer_base + r->layer_mip_offset_bytes,
                                        r->width, r->height, r->tile_mode, 0, bpt);
                                }
                            }
                        } else if (r->tile_mode && r->in_mip_tail) {
                            detile_surface_level(linear.get(), src, need, r->width, r->height,
                                                 r->tile_mode, bpt, r->mip_tail_x,
                                                 r->mip_tail_y);
                        } else if (r->tile_mode) {
                            detile_surface(linear.get(), src, r->width, r->height,
                                           r->tile_mode, 0, bpt);
                        }
                        const size_t texels = (size_t)volume_texels;
                        if (rgba8 || uint8 || r11g11b10 ||
                            sampled_unorm8x2 || sampled_unorm16_native ||
                            sampled_uint8_native || sampled_uint16_native ||
                            sampled_uint32_native || sampled_float32_native ||
                            sampled_float16_native || sampled_depth) {  // Native sampled texels
                            std::memcpy(upload, sampled_source, linear_bytes);
                        } else if (f32) {                           // Native float channels + default fill
                            parallel_compute_texels(texels, linear_bytes + texels * 16u,
                                [&](size_t begin, size_t end) {
                                    for (size_t t = begin; t < end; ++t) {
                                        for (uint32_t c = 0; c < 4; ++c) {
                                            float value = c == 3 ? 1.0f : 0.0f;
                                            if (c < nc)
                                                std::memcpy(
                                                    &value,
                                                    sampled_source + (t * nc + c) * sizeof(float),
                                                    sizeof(value));
                                            std::memcpy(
                                                upload + (t * 4 + c) * sizeof(float),
                                                &value, sizeof(value));
                                        }
                                    }
                                });
                        } else if (r8) {                            // R8: broadcast coverage to RGBA
                            parallel_compute_texels(texels, linear_bytes + texels * 4u,
                                [&](size_t begin, size_t end) {
                                    for (size_t t = begin; t < end; ++t) {
                                        const uint8_t v = sampled_source[t];
                                        upload[t * 4 + 0] = v; upload[t * 4 + 1] = v;
                                        upload[t * 4 + 2] = v; upload[t * 4 + 3] = v;
                                    }
                                });
                        } else {                                    // Float16: half -> UNORM8 + default fill
                            sampled_float16_to_unorm8_range(
                                sampled_source, nc, texels, upload);
                        }
                    }
                }
            }

            // Host writes are complete before this allocation is consumed by vkCmdCopyBufferToImage.
            upload_mapping.unmap();
            prepare_upload_ms = std::chrono::duration<double, std::milli>(
                ComputeClock::now() - prepare_upload_start).count();

            // Device-local image.
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = dim_1d ? VK_IMAGE_TYPE_1D : (dim_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D);
            ici.format = image_format;
            ici.extent = {r->width, dim_cube_stacked ? r->height * 6u : r->height,
                          dim_3d ? bi.texel_depth : 1u};
            ici.mipLevels = 1;
            ici.arrayLayers = bi.array_layers;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = (bi.storage ? (VK_IMAGE_USAGE_STORAGE_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                       (bi.graphics_sampled_usage
                                            ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u))
                                    : VK_IMAGE_USAGE_SAMPLED_BIT) |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            // An imported binding already holds the renderer's image; only the view/sampler below
            // are ours to create. `ici` is still filled above so the view matches its format/layers.
            const auto image_allocation_start = ComputeClock::now();
            if (!bi.imported && !bi.image) {
                if (!vk_ok(vkCreateImage(ctx.device, &ici, nullptr, &bi.image),
                           "image-create")) { images_ready = false; break; }
                VkMemoryRequirements ireq{};
                vkGetImageMemoryRequirements(ctx.device, bi.image, &ireq);
                bi.allocation_bytes = ireq.size;
                const uint32_t image_memory_type = device_memory_type(ireq.memoryTypeBits);
                bi.memory = ctx.allocate_memory(ireq.size, image_memory_type);
                if (!vk_handle_ok(bi.memory, "image-memory") ||
                    !vk_ok(vkBindImageMemory(ctx.device, bi.image, bi.memory, 0), "image-bind")) {
                    images_ready = false; break; }
            }
            image_allocation_ms = std::chrono::duration<double, std::milli>(
                ComputeClock::now() - image_allocation_start).count();
            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image = bi.image;
            vci.viewType = dim_1d ? VK_IMAGE_VIEW_TYPE_1D
                                  : (dim_3d ? VK_IMAGE_VIEW_TYPE_3D
                                     : dim_2d_array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                    : VK_IMAGE_VIEW_TYPE_2D);
            vci.format = ici.format;
            if (!bi.storage) {
                // T# DST_SEL channel routing (SQ_SEL: 0=0, 1=1, 4=R, 5=G, 6=B, 7=A) — same mapping
                // the renderer applies on its sampled views.
                auto sel = [&](uint32_t s) {
                    switch (s) {
                        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
                        case 1: return VK_COMPONENT_SWIZZLE_ONE;
                        case 4: return VK_COMPONENT_SWIZZLE_R;
                        case 5: return VK_COMPONENT_SWIZZLE_G;
                        case 6: return VK_COMPONENT_SWIZZLE_B;
                        case 7: return VK_COMPONENT_SWIZZLE_A;
                        default: return VK_COMPONENT_SWIZZLE_IDENTITY;
                    }
                };
                vci.components = {sel(r->swizzle[0]), sel(r->swizzle[1]),
                                  sel(r->swizzle[2]), sel(r->swizzle[3])};
            }
            const VkImageAspectFlags image_aspect = (sampled_depth || bi.imported_depth)
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange = {image_aspect, 0, 1, 0, ici.arrayLayers};
            const auto view_start = ComputeClock::now();
            if (!vk_ok(vkCreateImageView(ctx.device, &vci, nullptr, &bi.view),
                       "image-view")) { images_ready = false; break; }
            view_ms = std::chrono::duration<double, std::milli>(
                ComputeClock::now() - view_start).count();
            if (!bi.storage) {
                // Sampler from the decoded S# (mag/min filter, SQ_TEX CLAMP wrap enums) — the same
                // fields the renderer honors; defaults reproduce LINEAR/clamp.
                auto wrap = [&](uint32_t m) {
                    switch (m) {
                        case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                        case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                        case 6: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                        default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    }
                };
                VkSamplerCreateInfo smci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                // Integer sampled formats cannot be linearly filtered in Vulkan. DOLL's UINT8x4
                // volume is consumed by image_load (texel fetch), so the sampler is unused there;
                // nearest also preserves integer semantics if a shader samples such a descriptor.
                const bool sampled_uint_native = sampled_uint8_native ||
                    sampled_uint16_native || sampled_uint32_native;
                smci.magFilter = sampled_uint_native ? VK_FILTER_NEAREST
                                               : (r->mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.minFilter = sampled_uint_native ? VK_FILTER_NEAREST
                                               : (r->min_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
                smci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                smci.addressModeU = wrap(r->addr_uvw[0]);
                smci.addressModeV = wrap(r->addr_uvw[1]);
                smci.addressModeW = wrap(r->addr_uvw[2]);
                smci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                smci.compareEnable = sampled_depth ? VK_TRUE : VK_FALSE;
                smci.compareOp = static_cast<VkCompareOp>(r->depth_compare_func & 0x7u);
                const auto sampler_start = ComputeClock::now();
                if (!vk_ok(vkCreateSampler(ctx.device, &smci, nullptr, &bi.sampler), "image-sampler")) {
                    images_ready = false; break; }
                sampler_ms = std::chrono::duration<double, std::milli>(
                    ComputeClock::now() - sampler_start).count();
            }
            if (image_timing)
                std::fprintf(stderr,
                             "[compute-image] code=0x%llx hash=0x%016llx "
                             "binding=%u class=%s imported=%u addr=0x%llx "
                             "persistent=%u allocation-reused=%u upload-skipped=%u "
                             "extent=%ux%ux%u guest=%zu staging=%llu "
                             "normalized=%u texel=%u sampled-float=%u rgba8-reuse=%u "
                             "query_ms=%.3f import_ms=%.3f cache_ms=%.3f "
                             "staging_ms=%.3f prepare_ms=%.3f allocation_ms=%.3f "
                             "view_ms=%.3f sampler_ms=%.3f ms=%.3f\n",
                             (unsigned long long)item.code_addr,
                             (unsigned long long)timing_program_hash, bi.binding,
                             bi.storage ? "storage" : "sampled", bi.imported ? 1u : 0u,
                             (unsigned long long)r->gpu_addr,
                             bi.persistent ? 1u : 0u,
                             bi.forced_seed_allocation_reused ? 1u : 0u,
                             bi.upload_skipped ? 1u : 0u,
                             r->width, r->height, r->depth, bi.guest_bytes,
                             (unsigned long long)sbytes,
                             image_descriptors[i].normalized_sampling ? 1u : 0u,
                             image_descriptors[i].texel_access ? 1u : 0u,
                             image_descriptors[i].sampled_float ? 1u : 0u,
                             bi.unorm_rtt_value_reuse ? 1u : 0u,
                             query_ms, import_ms, cache_lookup_ms,
                             staging_ms, prepare_upload_ms, image_allocation_ms,
                             view_ms, sampler_ms,
                             std::chrono::duration<double, std::milli>(
                                 ComputeClock::now() - image_start).count());
        }
        if (!images_ready) break;

        // #1854 shadow census: all bindings are now finalized, including aliases, persistent-cache
        // reuse, and device-local compute-transfer seeds. Observe consumers before the dispatch can
        // execute, but do not change any upload, barrier, wait, readback, or guest writeback. Writable
        // buffers and unselected image outputs are ordered memory effects; selected storage-image
        // results are classified only on the complete success/publish path below.
        if (authority_requested) {
            for (size_t i = 0; i < buffers.size(); ++i) {
                const BoundBuffer& buffer = buffers[i];
                if (!buffer.resource) continue;
                const ShadowComputeAuthorityRange range =
                    ShadowComputeAuthorityRange::from(
                        buffer.resource->gpu_addr, buffer.guest_bytes);
                if (descriptors[i].readable)
                    authority_census.observe_compute_access(
                        item, descriptors[i].binding,
                        ShadowComputeAuthorityConsumerKind::RawBuffer,
                        range, "compute-buffer-input");
                if (buffer.writable)
                    authority_census.observe_compute_access(
                        item, descriptors[i].binding,
                        ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect,
                        range, "compute-buffer-output");
            }
            for (size_t i = 0; i < images.size(); ++i) {
                const BoundImage& image = images[i];
                if (!image.resource) continue;
                const ShadowComputeAuthorityRange range =
                    ShadowComputeAuthorityRange::from(
                        image.resource->gpu_addr, image.guest_bytes);
                const BoundImage& owner = image.alias_of == SIZE_MAX
                    ? image : images[image.alias_of];
                authority_census.observe_compute_image_access(
                    item, range,
                    {image.binding,
                     owner.binding,
                     image_descriptors[i].readable,
                     image.storage,
                     image.alias_of != SIZE_MAX,
                     image.compute_transfer_seed_borrowed,
                     owner.compute_transfer_seed_borrowed,
                     image.persistent,
                     owner.persistent,
                     image.upload_skipped,
                     owner.upload_skipped,
                     image.image == owner.image,
                     image.view == owner.view});
                if (image.storage && !authority_observation.selected)
                    authority_census.observe_compute_access(
                        item, image.binding,
                        ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect,
                        range, "compute-image-output-unselected");
            }
        }
        phase_setup = ComputeClock::now();

        // Layout: the buffer bindings (filled above) + one entry per image binding (#590).
        for (size_t i = 0; i < images.size(); i++) {
            VkDescriptorSetLayoutBinding b{};
            b.binding = images[i].binding;
            b.descriptorType = images[i].storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layout_bindings.push_back(b);
        }
        pipeline_key.clear();
        auto append_pipeline_key_u32 = [&](uint32_t value) {
            pipeline_key.append(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        append_pipeline_key_u32(static_cast<uint32_t>(spirv.size()));
        pipeline_key.append(reinterpret_cast<const char*>(spirv.data()),
                            spirv.size() * sizeof(uint32_t));
        append_pipeline_key_u32(static_cast<uint32_t>(item.user_sgprs.size()));
        append_pipeline_key_u32(item.required_subgroup_size);
        append_pipeline_key_u32(static_cast<uint32_t>(layout_bindings.size()));
        for (const auto& binding : layout_bindings) {
            append_pipeline_key_u32(binding.binding);
            append_pipeline_key_u32(static_cast<uint32_t>(binding.descriptorType));
            append_pipeline_key_u32(binding.descriptorCount);
            append_pipeline_key_u32(binding.stageFlags);
        }
        if (const auto found = ctx.pipelines.find(pipeline_key); found != ctx.pipelines.end()) {
            descriptor_layout = found->second.descriptor_layout;
            shader = found->second.shader;
            pipeline_layout = found->second.pipeline_layout;
            pipeline = found->second.pipeline;
            pipeline_cached = true;
        } else {
            VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            dlci.bindingCount = static_cast<uint32_t>(layout_bindings.size());
            dlci.pBindings = layout_bindings.data();
            if (!vk_ok(vkCreateDescriptorSetLayout(ctx.device, &dlci, nullptr, &descriptor_layout),
                       "descriptor-layout")) break;
        }
        uint32_t sampled_count = 0, storage_image_count = 0;
        for (const auto& im : images) (im.storage ? storage_image_count : sampled_count)++;
        VkDescriptorPoolSize pool_sizes[3]; uint32_t pool_size_count = 0;
        if (!buffers.empty())
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                             static_cast<uint32_t>(buffers.size())};
        if (sampled_count)
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampled_count};
        if (storage_image_count)
            pool_sizes[pool_size_count++] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storage_image_count};
        if (VulkanComputeContext::descriptor_pool_reuse_enabled()) {
            descriptor_pool = ctx.prepare_descriptor_pool(
                static_cast<uint32_t>(buffers.size()), sampled_count, storage_image_count);
            descriptor_pool_reused = descriptor_pool != VK_NULL_HANDLE;
            if (!vk_handle_ok(descriptor_pool, "descriptor-pool-reuse")) break;
        } else {
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpci.maxSets = 1;
            dpci.poolSizeCount = pool_size_count;
            dpci.pPoolSizes = pool_sizes;
            if (!vk_ok(vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &descriptor_pool),
                       "descriptor-pool")) break;
        }
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptor_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &descriptor_layout;
        if (!vk_ok(vkAllocateDescriptorSets(ctx.device, &dsai, &descriptor_set),
                   "descriptor-set")) break;

        std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
        std::vector<VkDescriptorImageInfo> image_infos(images.size());
        std::vector<VkWriteDescriptorSet> writes(buffers.size() + images.size());
        for (size_t i = 0; i < buffers.size(); i++) {
            buffer_infos[i] = {buffers[i].buffer, 0, buffers[i].bytes};
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = descriptors[i].binding;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        for (size_t i = 0; i < images.size(); i++) {
            image_infos[i] = {images[i].sampler, images[i].view, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet& w = writes[buffers.size() + i];
            w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descriptor_set;
            w.dstBinding = images[i].binding;
            w.descriptorCount = 1;
            w.descriptorType = images[i].storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &image_infos[i];
        }
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        if (!pipeline_cached) {
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = spirv.size() * sizeof(uint32_t);
            smci.pCode = spirv.data();
            if (!vk_ok(vkCreateShaderModule(ctx.device, &smci, nullptr, &shader), "shader-module"))
                break;
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &descriptor_layout;
            VkPushConstantRange push_range{};
            if (!item.user_sgprs.empty()) {
                push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                push_range.size = static_cast<uint32_t>(item.user_sgprs.size() * sizeof(uint32_t));
                plci.pushConstantRangeCount = 1;
                plci.pPushConstantRanges = &push_range;
            }
            if (!vk_ok(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &pipeline_layout),
                       "pipeline-layout")) break;
            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = shader;
            cpci.stage.pName = "main";
            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
            if (item.required_subgroup_size) {
                required_subgroup.requiredSubgroupSize = item.required_subgroup_size;
                cpci.stage.pNext = &required_subgroup;
                cpci.stage.flags |=
                    VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
            }
            cpci.layout = pipeline_layout;
            if (trace)
                std::fprintf(stderr,
                             "[compute]   creating compute pipeline words=%zu descriptors=%zu "
                             "subgroup=%u\n",
                             spirv.size(), descriptors.size(), item.required_subgroup_size);
            if (!vk_ok(vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1, &cpci, nullptr,
                                                &pipeline), "compute-pipeline")) break;
            if (trace) std::fprintf(stderr, "[compute]   compute pipeline ready\n");
            if (ctx.pipelines.size() < kMaxCachedComputePipelines) {
                ctx.pipelines.emplace(std::move(pipeline_key), CachedComputePipeline{
                    descriptor_layout, shader, pipeline_layout, pipeline});
                pipeline_cached = true;
            }
        }
        phase_pipeline = ComputeClock::now();

        // Retained writable buffers and storage images can carry an exact prior-result buffer.
        // Compare the current result against it on the GPU, reducing millions of word equalities to
        // one flag. Exact-width image bytes are canonical; raw-uvec4 image bytes enter this path
        // only for proven-full write-only targets. The optimization is deliberately limited to
        // whole uvec4s; unaligned byte counts retain the collision-free CPU comparison below.
        struct CompareTarget {
            VkBuffer current = VK_NULL_HANDLE;
            VkBuffer baseline = VK_NULL_HANDLE;
            VkDeviceSize bytes = 0;
            VkAccessFlags current_src_access = 0;
            BoundBuffer* buffer = nullptr;
            BoundImage* image = nullptr;
        };
        std::vector<CompareTarget> compare_targets;
        for (BoundBuffer& buffer : buffers) {
            if (buffer.alias_of == SIZE_MAX && buffer.writable && buffer.persistent &&
                buffer.upload_skipped && buffer.result_baseline && buffer.resource->size &&
                !(buffer.resource->size & 15u))
                compare_targets.push_back({
                    buffer.buffer, buffer.result_baseline, buffer.resource->size,
                    VK_ACCESS_SHADER_WRITE_BIT, &buffer, nullptr});
        }
        for (size_t i = 0; i < images.size(); ++i) {
            BoundImage& image = images[i];
            if (image.storage && image.cache_candidate && image.persistent &&
                (image.upload_skipped ||
                 (image.seed_skip && adaptive_storage_result_validation_enabled())) &&
                image.result_baseline && image.exact_result_bytes &&
                !(image.exact_result_bytes & 15u) && staging[i])
                compare_targets.push_back({
                    staging[i], image.result_baseline, image.exact_result_bytes,
                    VK_ACCESS_TRANSFER_WRITE_BIT, nullptr, &image});
        }
        bool compare_ready = compare_targets.empty();
        if (!compare_targets.empty() && ctx.prepare_compare_pipeline()) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = compare_targets.size() * sizeof(uint32_t);
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            compare_ready = vk_ok(vkCreateBuffer(ctx.device, &bci, nullptr, &compare_flags),
                                  "compare-flags-buffer");
            VkMemoryRequirements requirements{};
            if (compare_ready) {
                vkGetBufferMemoryRequirements(ctx.device, compare_flags, &requirements);
                compare_flags_memory = ctx.allocate_memory(
                    requirements.size, ctx.host_memory_type(requirements.memoryTypeBits), true);
                compare_ready = vk_handle_ok(compare_flags_memory, "compare-flags-memory") &&
                    vk_ok(vkBindBufferMemory(ctx.device, compare_flags,
                                            compare_flags_memory, 0),
                          "compare-flags-bind");
            }
            if (compare_ready) {
                VkDescriptorPoolSize size{
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    static_cast<uint32_t>(compare_targets.size() * 3)};
                VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                dpci.maxSets = static_cast<uint32_t>(compare_targets.size());
                dpci.poolSizeCount = 1;
                dpci.pPoolSizes = &size;
                compare_ready = vk_ok(vkCreateDescriptorPool(
                    ctx.device, &dpci, nullptr, &compare_descriptor_pool),
                    "compare-descriptor-pool");
            }
            if (compare_ready) {
                compare_descriptor_sets.resize(compare_targets.size());
                std::vector<VkDescriptorSetLayout> layouts(
                    compare_targets.size(), ctx.compare_descriptor_layout);
                VkDescriptorSetAllocateInfo allocate{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocate.descriptorPool = compare_descriptor_pool;
                allocate.descriptorSetCount = static_cast<uint32_t>(layouts.size());
                allocate.pSetLayouts = layouts.data();
                compare_ready = vk_ok(vkAllocateDescriptorSets(
                    ctx.device, &allocate, compare_descriptor_sets.data()),
                    "compare-descriptor-sets");
            }
            if (compare_ready) {
                std::vector<std::array<VkDescriptorBufferInfo, 3>> infos(compare_targets.size());
                std::vector<VkWriteDescriptorSet> writes(compare_targets.size() * 3);
                for (size_t j = 0; j < compare_targets.size(); ++j) {
                    CompareTarget& target = compare_targets[j];
                    infos[j][0] = {target.current, 0, target.bytes};
                    infos[j][1] = {target.baseline, 0, target.bytes};
                    infos[j][2] = {compare_flags, j * sizeof(uint32_t), sizeof(uint32_t)};
                    for (uint32_t binding = 0; binding < 3; ++binding) {
                        VkWriteDescriptorSet& write = writes[j * 3 + binding];
                        write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        write.dstSet = compare_descriptor_sets[j];
                        write.dstBinding = binding;
                        write.descriptorCount = 1;
                        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        write.pBufferInfo = &infos[j][binding];
                    }
                    if (target.buffer) target.buffer->compare_flag_index = j;
                    if (target.image) target.image->compare_flag_index = j;
                }
                vkUpdateDescriptorSets(
                    ctx.device, static_cast<uint32_t>(writes.size()), writes.data(),
                    0, nullptr);
            }
        }
        if (!compare_ready) {
            for (CompareTarget& target : compare_targets) {
                if (target.buffer) target.buffer->compare_flag_index = SIZE_MAX;
                if (target.image) target.image->compare_flag_index = SIZE_MAX;
            }
            compare_targets.clear();
        }

        if (!ctx.prepare_dispatch_commands()) {
            if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=command-reuse\n");
            break;
        }
        const VkCommandBuffer command = ctx.command_buffer;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!vk_ok(vkBeginCommandBuffer(command, &begin), "command-begin")) break;
        // From this point a submitted storage dispatch can replace a retained image even if a
        // later fence/readback step fails. Revoke its graphics authority up front; only the complete
        // success path below republishes it.
        for (const BoundImage& image : images) {
            if (image.storage && image.alias_of == SIZE_MAX && image.cache_candidate &&
                image.persistent)
                ctx.invalidate_cached_image_export(image.cache_key);
        }
        // Exactly one binding emits the layout transitions for each borrowed renderer image. The
        // hazard is per-VkImage, so ownership is keyed on the handle, and it is derived over the
        // bindings that actually REACH these loops (non-aliased ones) rather than decided at import
        // time -- an owner chosen earlier could later be folded into an alias, leaving the image
        // transitioned zero times while descriptors still declare GENERAL.
        auto imported_barrier_owner = [&](size_t index) {
            const BoundImage& self = images[index];
            for (size_t prior = 0; prior < index; prior++)
                if (images[prior].imported && images[prior].alias_of == SIZE_MAX &&
                    images[prior].image == self.image)
                    return false;
            return true;
        };
        auto imported_image_is_writable = [&](VkImage image) {
            return std::any_of(images.begin(), images.end(), [&](const BoundImage& candidate) {
                return candidate.imported && candidate.alias_of == SIZE_MAX &&
                       candidate.image == image && candidate.storage;
            });
        };
        auto imported_aspects = [](const BoundImage& image) {
            if (!image.imported_depth) return VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT};
            return VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT |
                (image.imported_format == VK_FORMAT_D32_SFLOAT_S8_UINT
                     ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u)};
        };
        // Upload every image: UNDEFINED -> TRANSFER_DST, copy the staged texels in, -> GENERAL.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (bi.alias_of != SIZE_MAX) continue;
            const ShaderResource* r = bi.resource;
            if (bi.imported) {
                if (!imported_barrier_owner(i)) continue;   // another binding transitions this image
                // Borrowed renderer image: nothing to upload. Include shader-write access when an
                // exact native storage binding aliases the sampled view in this dispatch.
                VkImageMemoryBarrier to_general{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_general.srcAccessMask = bi.imported_depth
                    ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                    : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                    (imported_image_is_writable(bi.image) ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
                to_general.oldLayout = static_cast<VkImageLayout>(bi.imported_saved_layout);
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_general.srcQueueFamilyIndex = to_general.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                to_general.image = bi.image;
                to_general.subresourceRange = {imported_aspects(bi), 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &to_general);
                continue;
            }
            if (bi.depth_bits_source) {
                // D32 and R32_UINT have the same four-byte texel payload but are not Vulkan
                // view-compatible. Preserve the guest's raw-bit alias with two transfer copies:
                // depth image -> device buffer -> integer image. The buffer is never host-mapped.
                const bool source_already_general = [&] {
                    for (size_t prior = 0; prior < i; ++prior) {
                        const BoundImage& candidate = images[prior];
                        if (candidate.imported && candidate.alias_of == SIZE_MAX &&
                            candidate.image == bi.depth_bits_image &&
                            imported_barrier_owner(prior))
                            return true;
                    }
                    return false;
                }();
                const VkImageLayout source_saved = source_already_general
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : static_cast<VkImageLayout>(bi.depth_bits_saved_layout);
                const VkImageAspectFlags source_aspects = VK_IMAGE_ASPECT_DEPTH_BIT |
                    (bi.depth_bits_format == VK_FORMAT_D32_SFLOAT_S8_UINT
                         ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);

                VkImageMemoryBarrier source_to_copy{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                source_to_copy.srcAccessMask = source_already_general
                    ? VK_ACCESS_SHADER_READ_BIT
                    : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                source_to_copy.oldLayout = source_saved;
                source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                source_to_copy.image = bi.depth_bits_image;
                source_to_copy.subresourceRange = {source_aspects, 0, 1, 0, 1};

                VkImageMemoryBarrier target_to_copy{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                target_to_copy.srcAccessMask = 0;
                target_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                target_to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                target_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                target_to_copy.srcQueueFamilyIndex = target_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                target_to_copy.image = bi.image;
                target_to_copy.subresourceRange = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkImageMemoryBarrier to_copy[2]{source_to_copy, target_to_copy};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 2, to_copy);

                VkBufferImageCopy source_copy{};
                source_copy.imageSubresource = {
                    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
                source_copy.imageExtent = {r->width, r->height, 1};
                vkCmdCopyImageToBuffer(command, bi.depth_bits_image,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       staging[i], 1, &source_copy);
                VkBufferMemoryBarrier buffer_ready{
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                buffer_ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                buffer_ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                buffer_ready.srcQueueFamilyIndex = buffer_ready.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                buffer_ready.buffer = staging[i];
                buffer_ready.size = staging_bytes[i];
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                                     &buffer_ready, 0, nullptr);

                VkBufferImageCopy target_copy{};
                target_copy.imageSubresource = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                target_copy.imageExtent = {r->width, r->height, 1};
                vkCmdCopyBufferToImage(command, staging[i], bi.image,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       1, &target_copy);

                VkImageMemoryBarrier ready[2]{source_to_copy, target_to_copy};
                ready[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                ready[0].dstAccessMask = source_already_general
                    ? VK_ACCESS_SHADER_READ_BIT
                    : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                ready[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ready[0].newLayout = source_saved;
                ready[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                ready[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                ready[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                ready[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                                     nullptr, 2, ready);
                continue;
            }
            if (bi.persistent && (bi.upload_skipped || bi.seed_skip)) {
                // The previous synchronous dispatch left this read-only sampled image in GENERAL,
                // and either the guest source is unchanged or the proven-full shader cannot observe
                // that source. No transfer or layout transition is needed.
                continue;
            }
            if (bi.seed_skip) {
                // #1122: nothing uploaded -- take the never-seeded image straight to GENERAL for the
                // write-only shader (which overwrites every texel). The result is read back below.
                VkImageMemoryBarrier to_general{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_general.srcAccessMask = 0;
                to_general.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_general.srcQueueFamilyIndex = to_general.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                to_general.image = bi.image;
                to_general.subresourceRange = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, bi.array_layers};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &to_general);
                continue;
            }
            if (bi.compute_transfer_seed_borrowed) {
                VkImageMemoryBarrier source_to_copy{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                source_to_copy.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_TRANSFER_WRITE_BIT;
                source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                source_to_copy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                source_to_copy.image = bi.compute_transfer_seed;
                source_to_copy.subresourceRange = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, bi.array_layers};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 1, &source_to_copy);

                VkImageMemoryBarrier target_to_copy{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                target_to_copy.srcAccessMask = bi.persistent
                    ? VK_ACCESS_SHADER_READ_BIT : 0u;
                target_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                target_to_copy.oldLayout = bi.persistent
                    ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
                target_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                target_to_copy.srcQueueFamilyIndex = target_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                target_to_copy.image = bi.image;
                target_to_copy.subresourceRange = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, bi.array_layers};
                vkCmdPipelineBarrier(command,
                                     bi.persistent ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                   : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 1, &target_to_copy);

                VkImageCopy copy{};
                copy.srcSubresource = {
                    VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, bi.array_layers};
                copy.dstSubresource = copy.srcSubresource;
                copy.extent = {r->width,
                               bi.stacked_cube ? r->height * 6u : r->height,
                               bi.array_layers > 1 ? 1u : bi.texel_depth};
                vkCmdCopyImage(command, bi.compute_transfer_seed,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               bi.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

                VkImageMemoryBarrier ready[2]{source_to_copy, target_to_copy};
                ready[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                ready[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                         VK_ACCESS_SHADER_WRITE_BIT;
                ready[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ready[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                ready[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                ready[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                ready[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                ready[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                     0, nullptr, 2, ready);
                continue;
            }
            if (bi.seed_from_imported != SIZE_MAX) {
                const BoundImage& source = images[bi.seed_from_imported];
                VkImageMemoryBarrier source_to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                source_to_copy.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                source_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                source_to_copy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                source_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                source_to_copy.srcQueueFamilyIndex = source_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                source_to_copy.image = source.image;
                source_to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &source_to_copy);

                VkImageMemoryBarrier target_to_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                target_to_copy.srcAccessMask = 0;
                target_to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                target_to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                target_to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                target_to_copy.srcQueueFamilyIndex = target_to_copy.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                target_to_copy.image = bi.image;
                target_to_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &target_to_copy);

                VkImageCopy copy{};
                copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.extent = {r->width, r->height, r->depth};
                vkCmdCopyImage(command, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               bi.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

                VkImageMemoryBarrier ready[2]{};
                ready[0] = source_to_copy;
                ready[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                ready[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                ready[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ready[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                ready[1] = target_to_copy;
                ready[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                ready[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                         VK_ACCESS_SHADER_WRITE_BIT;
                ready[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                ready[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 2, ready);
                continue;
            }
            VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_dst.srcAccessMask = bi.persistent ? VK_ACCESS_SHADER_READ_BIT : 0;
            to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_dst.oldLayout = bi.persistent ? VK_IMAGE_LAYOUT_GENERAL
                                             : VK_IMAGE_LAYOUT_UNDEFINED;
            to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_dst.image = bi.image;
            const VkImageAspectFlags aspect = bi.depth_view
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            to_dst.subresourceRange = {aspect, 0, 1, 0, bi.array_layers};
            vkCmdPipelineBarrier(command,
                                 bi.persistent ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
            VkBufferImageCopy region{};
            region.imageSubresource = {aspect, 0, 0, bi.array_layers};
            region.imageExtent = {r->width, bi.stacked_cube ? r->height * 6u : r->height,
                                  bi.array_layers > 1 ? 1u : bi.texel_depth};
            vkCmdCopyBufferToImage(command, staging[i], bi.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            VkImageMemoryBarrier to_general = to_dst;
            to_general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                       (bi.storage ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
            to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &to_general);
        }
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                                0, 1, &descriptor_set, 0, nullptr);
        if (!item.user_sgprs.empty())
            vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               static_cast<uint32_t>(item.user_sgprs.size() * sizeof(uint32_t)),
                               item.user_sgprs.data());
        vkCmdDispatch(command, item.launch.groups_x, item.launch.groups_y, item.launch.groups_z);
        // Storage images: copy the written texels back into the staging buffer for guest writeback.
        // When this private image was seeded from an exact borrowed renderer target, copy the final
        // result back to that same image as well. The CPU writeback below remains mandatory for
        // overlapping guest aliases; this second device-local copy prevents graphics from having to
        // detile those identical bytes again on the next pass.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (!bi.storage || bi.alias_of != SIZE_MAX || bi.imported) continue;
            const ShaderResource* r = bi.resource;
            VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_src.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            to_src.srcQueueFamilyIndex = to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_src.image = bi.image;
            to_src.subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, bi.array_layers};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_src);
            VkBufferImageCopy region{};
            region.imageSubresource = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, bi.array_layers};
            region.imageExtent = {r->width, r->height,
                                  bi.array_layers > 1 ? 1u : bi.texel_depth};
            vkCmdCopyImageToBuffer(command, bi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging[i], 1, &region);
            if (bi.mirror_result_to_imported) {
                const BoundImage& mirror = images[bi.seed_from_imported];
                VkImageMemoryBarrier mirror_to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                mirror_to_dst.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                mirror_to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mirror_to_dst.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                mirror_to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                mirror_to_dst.srcQueueFamilyIndex = mirror_to_dst.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                mirror_to_dst.image = mirror.image;
                mirror_to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &mirror_to_dst);
                VkImageCopy mirror_copy{};
                mirror_copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                mirror_copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                mirror_copy.extent = {r->width, r->height, r->depth};
                vkCmdCopyImage(command, bi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               mirror.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &mirror_copy);
                VkImageMemoryBarrier mirror_to_general = mirror_to_dst;
                mirror_to_general.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                mirror_to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                mirror_to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                mirror_to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 1, &mirror_to_general);
            }
            if (bi.cache_candidate || bi.persistent ||
                bi.post_writeback_promotion_candidate) {
                VkImageMemoryBarrier to_general = to_src;
                to_general.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                to_general.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                           VK_ACCESS_SHADER_WRITE_BIT;
                to_general.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                     0, nullptr, 1, &to_general);
            }
        }
        // Compare retained exact results before replacing their baselines. Guest-buffer shader
        // writes and image transfer writes become comparator reads; one atomic flag per target then
        // becomes a four-byte host read after the fence.
        if (!compare_targets.empty()) {
            vkCmdFillBuffer(command, compare_flags, 0,
                            compare_targets.size() * sizeof(uint32_t), 0);
            std::vector<VkBufferMemoryBarrier> before_compare;
            before_compare.reserve(compare_targets.size() * 2 + 1);
            for (const CompareTarget& target : compare_targets) {
                VkBufferMemoryBarrier current{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                current.srcAccessMask = target.current_src_access;
                current.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                current.srcQueueFamilyIndex = current.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                current.buffer = target.current;
                current.size = target.bytes;
                before_compare.push_back(current);
                VkBufferMemoryBarrier prior = current;
                prior.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_WRITE_BIT |
                                      VK_ACCESS_SHADER_WRITE_BIT;
                prior.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                      VK_ACCESS_SHADER_WRITE_BIT;
                prior.buffer = target.baseline;
                before_compare.push_back(prior);
            }
            VkBufferMemoryBarrier flags{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            flags.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            flags.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            flags.srcQueueFamilyIndex = flags.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            flags.buffer = compare_flags;
            flags.size = compare_targets.size() * sizeof(uint32_t);
            before_compare.push_back(flags);
            vkCmdPipelineBarrier(command,
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                 static_cast<uint32_t>(before_compare.size()),
                                 before_compare.data(), 0, nullptr);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              ctx.compare_pipeline);
            for (size_t j = 0; j < compare_targets.size(); ++j) {
                const CompareTarget& target = compare_targets[j];
                vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        ctx.compare_pipeline_layout, 0, 1,
                                        &compare_descriptor_sets[j], 0, nullptr);
                const uint32_t vectors =
                    static_cast<uint32_t>(target.bytes / 16u);
                vkCmdPushConstants(command, ctx.compare_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vectors), &vectors);
                vkCmdDispatch(command, (vectors + 255u) / 256u, 1, 1);
            }
        }
        // If source validation found an external guest change, the exact comparator is not allowed
        // to suppress guest repair. Still advance the retained baseline to this dispatch's result so
        // the next unchanged use compares against the right bytes. The same fallback handles any
        // target whose comparator setup failed.
        auto copy_result_baseline = [&](VkBuffer current, VkBuffer baseline, VkDeviceSize bytes,
                                        VkAccessFlags current_src_access) {
            VkBufferMemoryBarrier barriers[2]{};
            for (auto& barrier : barriers) {
                barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                barrier.size = bytes;
            }
            barriers[0].srcAccessMask = current_src_access;
            barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barriers[0].buffer = current;
            barriers[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                                        VK_ACCESS_TRANSFER_WRITE_BIT |
                                        VK_ACCESS_SHADER_WRITE_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barriers[1].buffer = baseline;
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(command, current, baseline, 1, &copy);
        };
        for (BoundBuffer& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX || !buffer.writable || !buffer.result_baseline ||
                buffer.compare_flag_index != SIZE_MAX)
                continue;
            copy_result_baseline(buffer.buffer, buffer.result_baseline,
                                 buffer.resource->size, VK_ACCESS_SHADER_WRITE_BIT);
        }
        for (size_t i = 0; i < images.size(); ++i) {
            BoundImage& image = images[i];
            if (!image.storage || !image.result_baseline || !staging[i]) continue;
            // Compared words update the baseline in the comparator only when they differ. An
            // identical result therefore has no redundant 66 MiB baseline copy at all.
            if (image.compare_flag_index != SIZE_MAX) continue;
            copy_result_baseline(staging[i], image.result_baseline,
                                 image.exact_result_bytes, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        if (!compare_targets.empty()) {
            VkBufferMemoryBarrier flags{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            flags.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            flags.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            flags.srcQueueFamilyIndex = flags.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            flags.buffer = compare_flags;
            flags.size = compare_targets.size() * sizeof(uint32_t);
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &flags,
                                 0, nullptr);
        }
        // Hand every borrowed renderer image back in the layout its owner left it in (#1095), so the
        // renderer's own layout tracking stays true whether the dispatch only sampled it or the
        // device-local mirror above replaced its pixels. Include transfer writes in the source scope.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (!bi.imported || bi.alias_of != SIZE_MAX || !imported_barrier_owner(i)) continue;
            VkImageMemoryBarrier restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            restore.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                (imported_image_is_writable(bi.image) ? VK_ACCESS_SHADER_WRITE_BIT : 0u);
            // The renderer's next use of a persistent target is frequently vkCmdCopyImageToBuffer
            // or a scanout blit, so make the transition visible to transfer access as well.
            restore.dstAccessMask = bi.imported_depth
                ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                : VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            restore.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            restore.newLayout = static_cast<VkImageLayout>(bi.imported_saved_layout);
            restore.srcQueueFamilyIndex = restore.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            restore.image = bi.image;
            restore.subresourceRange = {imported_aspects(bi), 0, 1, 0, 1};
            vkCmdPipelineBarrier(command,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &restore);
        }
        if (!vk_ok(vkEndCommandBuffer(command), "command-end")) break;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (trace) std::fprintf(stderr, "[compute]   submitting dispatch\n");
        // #1270: when prosper-app presents on this same (shared) queue, serialize the submit CALL against
        // its present submits. No-op relaxed atomic load until the app adopts the shared queue.
        VkResult compute_submit_rc;
        {
            std::unique_lock<std::mutex> lk(prosper::gpu::shared_present_submit_mutex(), std::defer_lock);
            if (prosper::gpu::shared_present_active()) lk.lock();
            g_live_compute_queue_submit_attempts.fetch_add(1, std::memory_order_relaxed);
            submission_entered = true;
            compute_submit_rc = g_force_next_queue_submit_device_lost_for_test.exchange(
                                    false, std::memory_order_acq_rel)
                ? VK_ERROR_DEVICE_LOST
                : vkQueueSubmit(ctx.queue, 1, &submit, ctx.dispatch_fence);
        }
        if (!vk_ok(compute_submit_rc, "queue-submit")) break;
        if (trace) std::fprintf(stderr, "[compute]   waiting for dispatch\n");
        const VkResult wait_result = vkWaitForFences(
            ctx.device, 1, &ctx.dispatch_fence, VK_TRUE, 30ull * 1000 * 1000 * 1000);
        if (!vk_ok(wait_result, "queue-wait")) {
            // cleanup() releases resources referenced by the command buffer, including a borrowed
            // renderer image's pin. With that image bound, in-flight work still references it and
            // its layout transitions, so drain the queue first rather than freeing it underneath
            // the GPU.
            // readback_persistent_color_target uses the same drain but PROMOTES a successful drain
            // to success; this item stays failed either way, which is the conservative choice for a
            // dispatch whose results we can no longer trust.
            if (!ctx.device_lost) {
                std::unique_lock<std::mutex> qlk(
                    prosper::gpu::shared_present_submit_mutex(), std::defer_lock);
                if (prosper::gpu::shared_present_active()) qlk.lock();
                const VkResult drain_result = vkQueueWaitIdle(ctx.queue);
                completion_proven = drain_result == VK_SUCCESS;
                if (!vk_ok(drain_result, "queue-drain") && trace)
                    std::fprintf(stderr, "[compute]   queue drain after fence timeout failed\n");
            }
            break;
        }
        completion_proven = true;
        if (trace) std::fprintf(stderr, "[compute]   dispatch complete\n");
        phase_dispatch = ComputeClock::now();
        const auto writeback_prepare_start = phase_dispatch;

        if (!compare_targets.empty()) {
            void* mapped = nullptr;
            if (ctx.map_memory(compare_flags_memory, 0,
                               compare_targets.size() * sizeof(uint32_t), &mapped) == VK_SUCCESS) {
                const auto* changed = static_cast<const uint32_t*>(mapped);
                for (size_t j = 0; j < compare_targets.size(); ++j) {
                    CompareTarget& target = compare_targets[j];
                    if (target.buffer) target.buffer->gpu_result_unchanged = changed[j] == 0;
                    if (target.image) target.image->gpu_result_unchanged = changed[j] == 0;
                }
                ctx.unmap_memory(compare_flags_memory);
            }
        }

        // The fence proves every upload is complete and every sampled image is back in GENERAL.
        // New entries become cache-owned only here, so an earlier Vulkan failure cannot retain an
        // uninitialized image. A dirty hit rearms its source watch after the refreshed upload.
        for (BoundImage& image : images) {
            if (image.storage || image.imported || image.alias_of != SIZE_MAX ||
                !image.cache_candidate)
                continue;
            // When this dispatch samples and writes the same guest view through distinct bindings,
            // the post-dispatch storage image is the cache authority. Retaining the sampled seed here
            // would occupy the identical key before storage writeback can retain the actual result.
            const bool replaced_by_storage = std::any_of(
                images.begin(), images.end(), [&](const BoundImage& candidate) {
                    return candidate.storage && candidate.cache_candidate &&
                           candidate.cache_key == image.cache_key;
                });
            if (replaced_by_storage) continue;
            if (image.persistent) {
                if (!image.upload_skipped) {
                    if (image.compute_transfer_seed_borrowed)
                        ctx.validate_cached_image_source_from_compute_transfer(
                            image.cache_key);
                    else
                        ctx.validate_cached_image_source(image.cache_key);
                }
            } else if (image.image && image.memory && image.allocation_bytes &&
                       (image.cache_source_snapshot.empty()
                            ? ctx.retain_image(image.cache_key, image.image, image.memory,
                                               image.allocation_bytes,
                                               static_cast<const uint8_t*>(nullptr))
                            : ctx.retain_image(image.cache_key, image.image, image.memory,
                                               image.allocation_bytes,
                                               std::move(image.cache_source_snapshot)))) {
                image.persistent = true;
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   retained sampled image binding=%u addr=0x%llx "
                                 "allocation=%llu\n",
                                 image.binding,
                                 (unsigned long long)image.resource->gpu_addr,
                                 (unsigned long long)image.allocation_bytes);
            }
        }

        writeback_prepare_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - writeback_prepare_start).count();
        const auto writeback_buffers_start = ComputeClock::now();
        bool readback_ok = true;
        for (auto& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX || !buffer.writable) continue;
            // The exact GPU comparator saw the same bytes as the retained baseline, while source
            // validation independently proved that the guest mirror still contains that baseline.
            // Preserve architectural write notification, but avoid mapping and scanning the whole
            // host-visible buffer merely to rediscover equality.
            if (buffer.gpu_result_unchanged) {
                g_buffer_gpu_result_skips.fetch_add(1, std::memory_order_relaxed);
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   skipped GPU-identical buffer writeback binding=%u "
                                 "addr=0x%llx bytes=%u\n",
                                 buffer.resource->binding,
                                 (unsigned long long)buffer.resource->gpu_addr,
                                 buffer.resource->size);
                if (buffer.resource->gpu_addr)
                    notify_guest_gpu_write_preserving_bytes(
                        buffer.resource->gpu_addr, buffer.resource->size);
                if (buffer.persistent)
                    ctx.validate_cached_buffer_source(buffer.cache_key);
                if (!buffer.resource->host_data && writer_provenance_enabled())
                    record_guest_write(GuestWriterKind::ComputeBuffer,
                                       buffer.resource->gpu_addr, buffer.resource->size,
                                       item.submit_no, item.dispatch_index,
                                       item.command_order, item.code_addr);
                continue;
            }
            void* mapped = nullptr;
            if (ctx.map_memory(buffer.memory, 0, buffer.bytes, &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            uint8_t* destination = buffer.atomic_image
                ? resource_bytes_for(buffer.resource, buffer.guest_bytes)
                : resource_bytes(buffer.resource);
            const auto* result = static_cast<const uint8_t*>(mapped);
            if (buffer.atomic_image) {
                if (trace) {
                    buffer.after_hash = fnv1a(result, buffer.bytes);
                    for (size_t i = 0; i < buffer.bytes; ++i)
                        buffer.changed_bytes += buffer.linear_seed[i] != result[i];
                }
                if (!buffer.resource->host_data && buffer.resource->gpu_addr)
                    prosper::host::guest_write_watch_notify_host_write(
                        buffer.resource->gpu_addr, buffer.guest_bytes);
                if (buffer.resource->tile_mode) {
                    tile_surface(destination, result,
                                 buffer.resource->width, buffer.resource->height,
                                 buffer.resource->tile_mode, 0, sizeof(uint32_t));
                } else {
                    const size_t tight_pitch = static_cast<size_t>(buffer.resource->width) * 4u;
                    const size_t destination_pitch = buffer.resource->linear_row_pitch_bytes
                        ? buffer.resource->linear_row_pitch_bytes : tight_pitch;
                    for (uint32_t y = 0; y < buffer.resource->height; ++y)
                        std::memcpy(destination + y * destination_pitch,
                                    result + y * tight_pitch, tight_pitch);
                }
                ctx.unmap_memory(buffer.memory);
                if (buffer.resource->gpu_addr)
                    notify_guest_gpu_write(buffer.resource->gpu_addr, buffer.guest_bytes);
                if (!buffer.resource->host_data && writer_provenance_enabled())
                    record_guest_write(GuestWriterKind::ComputeBuffer,
                                       buffer.resource->gpu_addr, buffer.guest_bytes,
                                       item.submit_no, item.dispatch_index,
                                       item.command_order, item.code_addr);
                continue;
            }
            const bool changed = !compute_buffers_equal(
                destination, result, buffer.bytes);
            if (trace) {
                buffer.after_hash = fnv1a(result, buffer.bytes);
                for (size_t i = 0; i < buffer.bytes; i++)
                    buffer.changed_bytes += destination[i] != result[i];
            }
            // Synchronous Unity maintenance kernels commonly rewrite a large persistent buffer with
            // the values it already contains. Terminator 2D's startup kernel binds 8,847,360 bytes;
            // after its first dispatch all later readbacks are identical. Avoiding the redundant host
            // write removes one full pass over both source and destination. Renderer-alias
            // invalidation and writer provenance remain unconditional: renderer-resident state can
            // differ from guest RAM even when consecutive compute readbacks contain identical bytes.
            if (changed) {
                if (!buffer.resource->host_data && buffer.resource->gpu_addr)
                    prosper::host::guest_write_watch_notify_host_write(
                        buffer.resource->gpu_addr, buffer.resource->size);
                copy_compute_buffer(destination, result, buffer.bytes);
            }
            if (buffer.persistent && !buffer.result_baseline &&
                ctx.retain_cached_buffer_result(buffer.cache_key, result) && trace)
                std::fprintf(stderr,
                             "[compute]   retained exact GPU buffer result baseline binding=%u "
                             "addr=0x%llx bytes=%u\n",
                             buffer.resource->binding,
                             (unsigned long long)buffer.resource->gpu_addr,
                             buffer.resource->size);
            ctx.unmap_memory(buffer.memory);
            if (buffer.resource->gpu_addr) {
                if (changed)
                    notify_guest_gpu_write(buffer.resource->gpu_addr, buffer.resource->size);
                else
                    notify_guest_gpu_write_preserving_bytes(
                        buffer.resource->gpu_addr, buffer.resource->size);
            }
            if (buffer.persistent)
                ctx.validate_cached_buffer_source(buffer.cache_key);
            if (!buffer.resource->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   buffer.resource->gpu_addr, buffer.resource->size,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
        }
        writeback_buffers_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - writeback_buffers_start).count();
        if (!readback_ok) break;
        const auto writeback_images_start = ComputeClock::now();
        // Storage-image writeback (#590): copy exact-width texels or pack raw uvec4 channels back
        // into the guest format, then restore its linear or 3D tiled address layout and notify the
        // render side exactly like the buffer path.
        for (size_t i = 0; i < images.size() && readback_ok; i++) {
            BoundImage& bi = images[i];
            if (!bi.storage || bi.alias_of != SIZE_MAX || bi.imported) continue;
            const auto image_writeback_start = ComputeClock::now();
            const bool image_cache_hit = bi.persistent;
            const ShaderResource* r = bi.resource;
            const uint32_t cb = data_format_bytes(r->format);
            const uint32_t nc = r->num_components ? r->num_components : 1;
            const size_t guest_texel = (r->format == DataFormat::Float10_11_11 ||
                                        r->format == DataFormat::Unorm2_10_10_10)
                ? 4u : (size_t)cb * nc;
            const size_t texels = (size_t)r->width * r->height * r->depth;
            const size_t linear_bytes = texels * guest_texel;
            uint8_t* destination = resource_bytes_for(r, bi.guest_bytes);
            // GPU comparison is an exact word-for-word equality reduction and acquire_cached_image
            // independently proved that the guest mirror still contains that baseline. This path
            // therefore needs neither a large staging mapping nor a CPU memory pass.
            if (bi.gpu_result_unchanged && bi.upload_skipped) {
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   skipped GPU-identical storage writeback binding=%u "
                                 "addr=0x%llx bytes=%llu\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned long long)bi.exact_result_bytes);
                if (r->gpu_addr)
                    notify_guest_gpu_write_preserving_bytes(r->gpu_addr, bi.guest_bytes);
                continue;
            }
            void* mapped = nullptr;
            const auto map_start = ComputeClock::now();
            if (g_fail_next_storage_readback_for_test.exchange(
                    false, std::memory_order_acq_rel)) {
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   injected storage readback failure binding=%u\n",
                                 bi.binding);
                readback_ok = false;
                break;
            }
            if (ctx.map_memory(staging_memory[i], 0, staging_bytes[i], &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            const auto map_done = ComputeClock::now();
            const bool array_image = backend_uses_2d_array(*r);
            static const bool direct_tiled_writeback_disabled =
                std::getenv("PROSPER_NO_DIRECT_TILED_WRITEBACK") != nullptr;
            const bool tile_mapped_bytes = storage_writeback_can_tile_mapped_bytes(
                bi.exact_storage_bytes(), r->tile_mode, bi.poison_verify,
                direct_tiled_writeback_disabled);
            std::unique_ptr<uint8_t[]> linear;
            uint8_t* packed = destination;
            if ((r->tile_mode && !tile_mapped_bytes) ||
                (!r->tile_mode && array_image && r->depth > 1)) {
                linear = std::make_unique_for_overwrite<uint8_t[]>(linear_bytes);
                packed = linear.get();
            }
            const uint32_t* channels = static_cast<const uint32_t*>(mapped);
            const uint8_t* native_texels = static_cast<const uint8_t*>(mapped);
            // A retained, proven-full output can avoid the expensive CPU pack/retile and renderer
            // invalidation when BOTH sides of the contract are exact: acquire_cached_image proved
            // that guest memory still contains the prior result, and this dispatch reproduced the
            // same row-major bytes. If either comparison fails, take the ordinary writeback below.
            const bool repeated_output = bi.cache_candidate && bi.persistent &&
                bi.upload_skipped && bi.exact_storage_bytes() &&
                ctx.cached_image_result_matches(bi.cache_key, native_texels, linear_bytes);
            const auto prepare_done = ComputeClock::now();
            if (repeated_output) {
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   skipped identical storage writeback binding=%u "
                                 "addr=0x%llx bytes=%llu\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 (unsigned long long)bi.exact_result_bytes);
                ctx.unmap_memory(staging_memory[i]);
                if (r->gpu_addr)
                    notify_guest_gpu_write_preserving_bytes(r->gpu_addr, bi.guest_bytes);
                continue;
            }
            // Notify page-based dirty trackers only when bytes will actually be written. Doing this
            // before the exact repeated-output check dirtied and rearmed tens of thousands of pages
            // even on the no-write path, defeating the validation that made that path safe.
            if (!r->host_data && r->gpu_addr)
                prosper::host::guest_write_watch_notify_host_write(
                    r->gpu_addr, bi.guest_bytes);
            const auto watch_done = ComputeClock::now();
            // #1122 proving-frame poison scan: a texel still fully poison was NOT stored by the write.
            // Zero survivors == the shader covers every texel (safe to fast-skip its seed henceforth);
            // any survivor == partial coverage (must always seed). Cache the verdict per (code,binding)
            // and, for a partial write, restore the un-stored texels from the clean seed after packing.
            std::vector<uint8_t> poison_texel;   // 1 == this texel survived as poison (untouched)
            if (bi.poison_verify) {
                size_t survived = 0;
                poison_texel.assign(texels, 0);
                for (size_t t = 0; t < texels; ++t) {
                    bool all = true;
                    if (bi.exact_storage_bytes()) {
                        const uint8_t* texel = native_texels + t * guest_texel;
                        for (size_t b = 0; b < guest_texel; ++b)
                            if (texel[b] != 0x5a) all = false;
                    } else {
                        for (uint32_t c = 0; c < 4; ++c)
                            if (channels[t * 4 + c] != 0xDEADBEEFu) all = false;
                    }
                    if (all) { ++survived; poison_texel[t] = 1; }
                }
                {
                    const SeedCoverageKey proof_key{item.code_addr, bi.binding,
                                                    r->width, r->height, r->depth};
                    std::lock_guard<std::mutex> lk(seed_coverage_mu);
                    // Re-cache the freshly-proven verdict; skips=0 restarts the #1127 re-prove interval.
                    seed_coverage_proof[proof_key] =
                        SeedVerdict{ survived ? SeedCoverage::Partial : SeedCoverage::Full, 0 };
                }
                std::fprintf(stderr,
                             "[seed-skip-verify] code=0x%llx binding=%u texels=%zu poison_survived=%zu %s\n",
                             (unsigned long long)item.code_addr, bi.binding, texels, survived,
                             survived ? "PARTIAL-COVERAGE (will always seed)" : "full-coverage (seed-skip proven)");
            }
            if (trace) {
                if (bi.exact_storage_bytes()) {
                    for (size_t t = 0; t < texels; ++t) {
                        const uint8_t* texel = native_texels + t * guest_texel;
                        for (size_t b = 0; b < guest_texel; ++b)
                            bi.nonzero_channels += texel[b] != 0;
                    }
                } else {
                    for (size_t t = 0; t < texels; t++)
                        for (uint32_t c = 0; c < 4; c++)
                            bi.nonzero_channels += channels[t * 4 + c] != 0;
                }
            }
            const auto pack_start = ComputeClock::now();
            static const bool pack_range_enabled = !std::getenv("PROSPER_NO_PACK_RANGE");
            if (bi.exact_storage_bytes()) {
                // The typed Vulkan image has already applied the PS5 descriptor's UNORM/float
                // conversion. Its transfer bytes are the guest's exact row-major texels. A tiled
                // non-proving write can feed those bytes straight to the tiler, avoiding a second
                // full-surface allocation and memcpy (66.8 MiB for Astro Bot's 4K RGBA16F target).
                if (!tile_mapped_bytes)
                    parallel_compute_texels(texels, linear_bytes * 2,
                        [&](size_t begin, size_t end) {
                            std::memcpy(packed + begin * guest_texel,
                                        native_texels + begin * guest_texel,
                                        (end - begin) * guest_texel);
                        });
            } else if (pack_range_enabled) {
                storage_pack_range(channels, r->format, nc, texels, packed, guest_texel);
            } else {
                for (size_t t = 0; t < texels; t++)
                    storage_pack_texel(channels + t * 4, r->format, nc,
                                       packed + t * guest_texel);
            }
            // #1122: a proving frame that turned out partial-coverage packed poison garbage into the
            // un-stored texels. Restore each from the clean seed so the guest keeps its real prior
            // content (exactly a partial write's contract). Full-coverage proving frames have no
            // survivors, so this is a no-op there. bi.seed_linear shares this row-major texel layout.
            if (bi.poison_verify && !poison_texel.empty() &&
                bi.seed_linear.size() == linear_bytes) {
                for (size_t t = 0; t < texels; ++t) {
                    if (poison_texel[t])
                        std::memcpy(packed + t * guest_texel,
                                    bi.seed_linear.data() + t * guest_texel, guest_texel);
                }
            }
            const auto pack_done = ComputeClock::now();
            static const bool verify_pack = std::getenv("PROSPER_VERIFY_PACK") != nullptr;
            if (verify_pack && !bi.exact_storage_bytes()) {
                // Fail-visible A/B (mirrors PROSPER_VERIFY_UNPACK): the specialized range pack must
                // be bit-identical to the per-texel path it replaces, verified against the real
                // workload's texels. Logs the clean case too, so a verified run is self-proving.
                std::vector<uint8_t> expect(guest_texel);
                size_t bad = 0, first_bad = 0;
                for (size_t t = 0; t < texels; ++t) {
                    std::memset(expect.data(), 0, expect.size());
                    storage_pack_texel(channels + t * 4, r->format, nc, expect.data());
                    if (std::memcmp(expect.data(), packed + t * guest_texel,
                                    guest_texel) != 0) {
                        if (!bad) first_bad = t;
                        ++bad;
                    }
                }
                std::fprintf(stderr,
                             "[compute] pack-verify binding=%u addr=0x%llx fmt=%u nc=%u "
                             "texels=%zu mismatches=%zu%s\n",
                             bi.binding, (unsigned long long)r->gpu_addr, (unsigned)r->format,
                             nc, texels, bad, bad ? " MISMATCH" : "");
                if (bad)
                    std::fprintf(stderr, "[compute] pack-verify first mismatch texel=%zu\n",
                                 first_bad);
            }
            const uint8_t* layout_source = tile_mapped_bytes ? native_texels : packed;
            if (r->tile_mode && r->img_dim == 2 && r->depth > 1) {
                if (!tile_volume(destination, bi.guest_bytes, layout_source, r->width, r->height,
                                 r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                    readback_ok = false;
                    ctx.unmap_memory(staging_memory[i]);
                    break;
                }
            } else if (array_image && r->depth > 1) {
                const size_t linear_slice = static_cast<size_t>(r->width) * r->height * guest_texel;
                const size_t selected_slice = r->in_mip_tail
                    ? r->mip_tail_bytes
                    : (r->tile_mode
                           ? tiled_surface_bytes(r->width, r->height, r->tile_mode, 0,
                                                 static_cast<uint32_t>(guest_texel))
                           : (r->layer_stride_bytes
                                  ? linear_array_surface_bytes(
                                        *r, static_cast<uint32_t>(guest_texel))
                                  : linear_slice));
                const size_t layer_stride = r->layer_stride_bytes
                    ? r->layer_stride_bytes : selected_slice;
                for (uint32_t layer = 0; layer < r->depth; ++layer) {
                    uint8_t* layer_base = destination + layer_stride * layer;
                    if (!r->tile_mode) {
                        const size_t row_pitch = r->layer_stride_bytes
                            ? linear_array_row_pitch(
                                  *r, static_cast<uint32_t>(guest_texel))
                            : static_cast<size_t>(r->width) * guest_texel;
                        for (uint32_t y = 0; y < r->height; ++y)
                            std::memcpy(
                                layer_base + r->layer_mip_offset_bytes + y * row_pitch,
                                layout_source + linear_slice * layer +
                                    static_cast<size_t>(y) * r->width * guest_texel,
                                static_cast<size_t>(r->width) * guest_texel);
                    } else if (r->in_mip_tail) {
                        tile_surface_level(
                            layer_base, r->mip_tail_bytes,
                            layout_source + linear_slice * layer,
                            r->width, r->height, r->tile_mode,
                            static_cast<uint32_t>(guest_texel), r->mip_tail_x, r->mip_tail_y);
                    } else {
                        tile_surface(
                            layer_base + r->layer_mip_offset_bytes,
                            layout_source + linear_slice * layer,
                            r->width, r->height, r->tile_mode, 0,
                            static_cast<uint32_t>(guest_texel));
                    }
                }
            } else if (r->tile_mode && r->in_mip_tail) {
                tile_surface_level(destination, bi.guest_bytes, layout_source,
                                   r->width, r->height, r->tile_mode,
                                   static_cast<uint32_t>(guest_texel),
                                   r->mip_tail_x, r->mip_tail_y);
            } else if (r->tile_mode) {
                tile_surface(destination, layout_source, r->width, r->height, r->tile_mode, 0,
                             static_cast<uint32_t>(guest_texel));
            }
            const auto layout_done = ComputeClock::now();
            pack_ms += std::chrono::duration<double, std::milli>(pack_done - pack_start).count();
            layout_ms += std::chrono::duration<double, std::milli>(layout_done - pack_done).count();
            if (trace) bi.after_hash = fnv1a(destination, bi.guest_bytes);
            const auto notify_start = ComputeClock::now();
            notify_guest_gpu_write(r->gpu_addr, bi.guest_bytes);
            if (!r->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   r->gpu_addr, bi.guest_bytes,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
            if (bi.dcc_metadata && bi.dcc_metadata_bytes) {
                const bool leave_compressed_for_test =
                    g_leave_next_dcc_metadata_compressed_for_test.exchange(
                        false, std::memory_order_acq_rel);
                if (!leave_compressed_for_test) {
                    std::memset(bi.dcc_metadata, 0xff, bi.dcc_metadata_bytes);
                    notify_guest_gpu_write(r->metadata_addr, bi.dcc_metadata_bytes);
                    if (!r->dcc_metadata_host_data && writer_provenance_enabled())
                        record_guest_write(GuestWriterKind::ComputeBuffer,
                                           r->metadata_addr, bi.dcc_metadata_bytes,
                                           item.submit_no, item.dispatch_index,
                                           item.command_order, item.code_addr);
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   DCC uncompressed binding=%u meta=0x%llx "
                                     "bytes=%zu code=0xff\n",
                                     bi.binding, (unsigned long long)r->metadata_addr,
                                     bi.dcc_metadata_bytes);
                } else if (trace) {
                    std::fprintf(stderr,
                                 "[compute]   injected unresolved DCC writeback binding=%u\n",
                                 bi.binding);
                }
            }
            if (bi.mirror_result_to_imported) {
                const BoundImage& mirror = images[bi.seed_from_imported];
                notify_live_render_target_image_written({
                    r->gpu_addr, mirror.imported_width, mirror.imported_height,
                    mirror.imported_pixel_format});
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   mirrored storage result into renderer RTT "
                                 "binding=%u addr=0x%llx extent=%ux%u\n",
                                 bi.binding, (unsigned long long)r->gpu_addr,
                                 mirror.imported_width, mirror.imported_height);
            }
            const auto notify_done = ComputeClock::now();
            bool retain_gpu_result_baseline = false;
            bool promoted_after_writeback = false;
            const bool final_dcc_cache_safe = bi.dcc_metadata && bi.dcc_metadata_bytes &&
                std::all_of(bi.dcc_metadata, bi.dcc_metadata + bi.dcc_metadata_bytes,
                            [](uint8_t value) { return value == 0xff; });
            if (bi.post_writeback_promotion_candidate && final_dcc_cache_safe) {
                const uint8_t* retained_source =
                    (adaptive_storage_result_validation_enabled() &&
                     cold_storage_result_snapshot_can_defer(
                         r->host_data != nullptr, bi.seed_skip, bi.guest_bytes,
                         cold_storage_result_snapshot_defer_min_bytes()))
                    ? nullptr : destination;
                if (bi.forced_seed_allocation_reused) {
                    // The exact cache entry was already pinned and forcibly reseeded before this
                    // dispatch. Successful writeback plus the final all-uncompressed metadata scan
                    // may now restore source/transfer authority without replacing its allocation.
                    bi.cache_candidate = true;
                    g_dcc_post_writeback_promotions.fetch_add(
                        1, std::memory_order_relaxed);
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   promoted reused post-writeback DCC storage "
                                     "image binding=%u addr=0x%llx allocation=%llu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned long long)bi.allocation_bytes);
                } else if (bi.image && bi.memory && bi.allocation_bytes &&
                    ctx.replace_or_retain_image(
                        bi.cache_key, bi.image, bi.memory,
                        bi.allocation_bytes, retained_source)) {
                    bi.cache_candidate = true;
                    bi.persistent = true;
                    promoted_after_writeback = true;
                    g_dcc_post_writeback_promotions.fetch_add(
                        1, std::memory_order_relaxed);
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   promoted post-writeback DCC storage image "
                                     "binding=%u addr=0x%llx allocation=%llu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned long long)bi.allocation_bytes);
                }
            }
            if (bi.forced_seed_allocation_reused && !final_dcc_cache_safe)
                ctx.invalidate_cached_image_source(bi.cache_key);
            if (bi.cache_candidate) {
                if (bi.persistent && !promoted_after_writeback) {
                    // Guest bytes now mirror the retained image again. A changing full-overwrite
                    // target needs no source snapshot; the first repeated result retains one exact
                    // baseline, and subsequent identical GPU skips do not recopy it.
                    ctx.validate_cached_image_source(
                        bi.cache_key, destination, bi.gpu_result_unchanged, !bi.seed_skip,
                        bi.seed_skip && bi.native_float_storage && r->img_dim == 2 &&
                            native_3d_transfer_enabled());
                } else if (bi.image && bi.memory && bi.allocation_bytes &&
                           ctx.retain_image(bi.cache_key, bi.image, bi.memory,
                                            bi.allocation_bytes,
                                            (adaptive_storage_result_validation_enabled() &&
                                             cold_storage_result_snapshot_can_defer(
                                                 r->host_data != nullptr, bi.seed_skip,
                                                 bi.guest_bytes,
                                                 cold_storage_result_snapshot_defer_min_bytes()))
                                                ? nullptr : destination)) {
                    bi.persistent = true;
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   retained storage image binding=%u "
                                     "addr=0x%llx allocation=%llu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned long long)bi.allocation_bytes);
                }
                // An aligned exact result is retained as the staging buffer immediately below.
                // Copying it into a host vector first only to clear that vector after ownership
                // transfer is pure churn (66.4 MiB for a native 4K RGBA16F target). Unavailable GPU
                // setup keeps the exact current host fallback; a failed ownership attempt invalidates
                // any older fallback so the next dispatch takes the ordinary writeback path.
                const bool force_host_result_fallback =
                    bi.persistent && !bi.result_baseline && bi.exact_result_bytes &&
                    !(bi.exact_result_bytes & 15u) &&
                    g_force_next_image_result_host_fallback_for_test.exchange(
                        false, std::memory_order_acq_rel);
                retain_gpu_result_baseline = bi.persistent && !bi.result_baseline &&
                    bi.exact_result_bytes && !(bi.exact_result_bytes & 15u) &&
                    !force_host_result_fallback && ctx.prepare_compare_pipeline();
                if (bi.persistent && !retain_gpu_result_baseline)
                    ctx.remember_cached_image_result(
                        bi.cache_key, native_texels,
                        static_cast<size_t>(bi.exact_result_bytes));
            }
            const auto cache_done = ComputeClock::now();
            ctx.unmap_memory(staging_memory[i]);
            const VkBuffer retained_result = staging[i];
            if (retain_gpu_result_baseline &&
                ctx.retain_cached_image_result_buffer(
                    bi.cache_key, staging[i], staging_memory[i],
                    bi.staging_allocation_bytes, bi.exact_result_bytes)) {
                bi.result_baseline = retained_result;
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   retained exact GPU result baseline binding=%u "
                                 "addr=0x%llx bytes=%zu\n",
                                 bi.binding, (unsigned long long)r->gpu_addr, linear_bytes);
            }
            const auto image_writeback_done = ComputeClock::now();
            const auto image_milliseconds = [](auto begin, auto end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
            image_map_ms += image_milliseconds(map_start, map_done);
            image_prepare_ms += image_milliseconds(map_done, prepare_done);
            image_watch_ms += image_milliseconds(prepare_done, watch_done);
            image_notify_ms += image_milliseconds(notify_start, notify_done);
            image_cache_ms += image_milliseconds(notify_done, cache_done);
            if (image_timing)
                std::fprintf(stderr,
                             "[compute-image-writeback] code=0x%llx hash=0x%016llx "
                             "binding=%u addr=0x%llx "
                             "fmt=%u comps=%u tile=%u bytes=%zu cache-hit=%u seed-skip=%u "
                             "poison=%u map_ms=%.3f prepare_ms=%.3f watch_ms=%.3f "
                             "pack_ms=%.3f layout_ms=%.3f notify_ms=%.3f cache_ms=%.3f "
                             "total_ms=%.3f\n",
                             (unsigned long long)item.code_addr,
                             (unsigned long long)timing_program_hash, bi.binding,
                             (unsigned long long)r->gpu_addr, (unsigned)r->format, nc,
                             r->tile_mode, bi.guest_bytes, image_cache_hit ? 1u : 0u,
                             bi.seed_skip ? 1u : 0u, bi.poison_verify ? 1u : 0u,
                             image_milliseconds(map_start, map_done),
                             image_milliseconds(map_done, prepare_done),
                             image_milliseconds(prepare_done, watch_done),
                             image_milliseconds(pack_start, pack_done),
                             image_milliseconds(pack_done, layout_done),
                             image_milliseconds(notify_start, notify_done),
                             image_milliseconds(notify_done, cache_done),
                             image_milliseconds(image_writeback_start, image_writeback_done));
        }
        writeback_images_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - writeback_images_start).count();
        if (!readback_ok) break;
        const auto writeback_publish_start = ComputeClock::now();
        // Every storage image is back in GENERAL, all exact guest writebacks/notifications have
        // completed, and a failed dispatch cannot reach here. Native results may seed a later
        // sampled cache with a device-local copy; exact 2D/3D images created with SAMPLED usage may
        // also be exported directly to graphics. Raw interchange images are compatible with neither.
        for (const BoundImage& image : images) {
            if (!image.storage) continue;
            const bool unique = image.alias_of == SIZE_MAX;
            const bool publish_eligible = image.native_float_storage && unique &&
                image.cache_candidate && image.persistent;
            const bool authorized = publish_eligible &&
                ctx.authorize_cached_image_compute_transfer(image.cache_key);
            transfer_gate_census.record_storage_publish(
                transfer_gate_observation.role, image.native_float_storage, unique,
                image.cache_candidate, image.persistent, authorized);
            if (authority_observation.selected && unique && image.resource) {
                authority_census.record_selected_storage_output(
                    item, image.binding,
                    ShadowComputeAuthorityRange::from(
                        image.resource->gpu_addr, image.guest_bytes),
                    authorized);
            }
            if (publish_eligible && image.graphics_sampled_usage)
                ctx.authorize_cached_image_export(image.cache_key);
        }
        writeback_publish_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - writeback_publish_start).count();
        ok = true;
        phase_writeback = ComputeClock::now();
    } while (false);

    if (trace)
        std::fprintf(stderr, "[compute] execute submit=%llu dispatch=%llu code=0x%llx "
                     "threads=%ux%ux%u local=%ux%ux%u groups=%ux%ux%u "
                     "buffers=%zu images=%zu spirv=%zu/%016llx result=%s\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.dispatch_index,
                     (unsigned long long)item.code_addr, item.launch.threads_x,
                     item.launch.threads_y, item.launch.threads_z, item.launch.local_x,
                     item.launch.local_y, item.launch.local_z, item.launch.groups_x,
                     item.launch.groups_y, item.launch.groups_z, buffers.size(), images.size(),
                     spirv.size(), (unsigned long long)gpu_capture_hash(
                         reinterpret_cast<const uint8_t*>(spirv.data()),
                         spirv.size() * sizeof(uint32_t)),
                     ok ? "ok" : "failed");
    if (trace) {
        for (const auto& buffer : buffers) {
            if (!buffer.resource || buffer.alias_of != SIZE_MAX) continue;
            const uint8_t* bytes = resource_bytes(buffer.resource);
            std::fprintf(stderr,
                         "[compute]   writeback binding=%u addr=0x%llx size=%u changed=%llu "
                         "hash=%016llx->%016llx first=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
                         buffer.resource->binding, (unsigned long long)buffer.resource->gpu_addr,
                         buffer.resource->size, (unsigned long long)buffer.changed_bytes,
                         (unsigned long long)buffer.before_hash, (unsigned long long)buffer.after_hash,
                         buffer.resource->size >= 4 ? reinterpret_cast<const uint32_t*>(bytes)[0] : 0,
                         buffer.resource->size >= 8 ? reinterpret_cast<const uint32_t*>(bytes)[1] : 0,
                         buffer.resource->size >= 12 ? reinterpret_cast<const uint32_t*>(bytes)[2] : 0,
                         buffer.resource->size >= 16 ? reinterpret_cast<const uint32_t*>(bytes)[3] : 0,
                         buffer.resource->size >= 20 ? reinterpret_cast<const uint32_t*>(bytes)[4] : 0,
                         buffer.resource->size >= 24 ? reinterpret_cast<const uint32_t*>(bytes)[5] : 0,
                         buffer.resource->size >= 28 ? reinterpret_cast<const uint32_t*>(bytes)[6] : 0,
                         buffer.resource->size >= 32 ? reinterpret_cast<const uint32_t*>(bytes)[7] : 0);
        }
        for (const auto& image : images) {
            if (!image.storage || !image.resource || image.alias_of != SIZE_MAX) continue;
            const uint8_t* bytes = resource_bytes_for(image.resource, image.guest_bytes);
            std::fprintf(stderr,
                         "[compute]   image-writeback binding=%u addr=0x%llx size=%zu "
                         "hash=%016llx->%016llx nonzero-ch=%llu "
                         "first=%08x,%08x,%08x,%08x\n",
                         image.binding, (unsigned long long)image.resource->gpu_addr,
                         image.guest_bytes, (unsigned long long)image.before_hash,
                         (unsigned long long)image.after_hash,
                         (unsigned long long)image.nonzero_channels,
                         image.guest_bytes >= 4 ? reinterpret_cast<const uint32_t*>(bytes)[0] : 0,
                         image.guest_bytes >= 8 ? reinterpret_cast<const uint32_t*>(bytes)[1] : 0,
                         image.guest_bytes >= 12 ? reinterpret_cast<const uint32_t*>(bytes)[2] : 0,
                         image.guest_bytes >= 16 ? reinterpret_cast<const uint32_t*>(bytes)[3] : 0);
        }
    }
    if (ctx.device_lost && submission_entered && !completion_proven)
        ctx.completion_unproven = true;
    // VK_ERROR_DEVICE_LOST from a queue call gives no completion proof for the affected submission.
    // cleanup() would destroy/recycle command resources and release borrowed renderer-image pins
    // that the GPU may still own. Deliberately retain this raw-handle closure; the sticky entry
    // check prevents reuse and the atexit handler retains the context itself.
    if (!ctx.completion_unproven) cleanup();
    if (phase_timing) {
        const auto phase_cleanup = ComputeClock::now();
        auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::fprintf(stderr,
                     "[compute-phase] submit=%llu code=0x%llx hash=0x%016llx ok=%u "
                     "setup_ms=%.2f setup_validate_ms=%.2f setup_buffers_ms=%.2f "
                     "pipeline_ms=%.2f dispatch_ms=%.2f "
                     "writeback_ms=%.2f writeback_prepare_ms=%.2f "
                     "writeback_buffers_ms=%.2f writeback_images_ms=%.2f "
                     "writeback_publish_ms=%.2f map_ms=%.2f prepare_ms=%.2f watch_ms=%.2f "
                     "pack_ms=%.2f layout_ms=%.2f notify_ms=%.2f cache_ms=%.2f "
                     "cleanup_ms=%.2f total_ms=%.2f subgroup=%u\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.code_addr,
                     (unsigned long long)timing_program_hash, ok ? 1u : 0u,
                     milliseconds(phase_start, phase_setup),
                     setup_validate_ms, setup_buffers_ms,
                     milliseconds(phase_setup, phase_pipeline),
                     milliseconds(phase_pipeline, phase_dispatch),
                     milliseconds(phase_dispatch, phase_writeback),
                     writeback_prepare_ms, writeback_buffers_ms,
                     writeback_images_ms, writeback_publish_ms,
                     image_map_ms, image_prepare_ms, image_watch_ms,
                     pack_ms, layout_ms, image_notify_ms, image_cache_ms,
                     milliseconds(phase_writeback, phase_cleanup),
                     milliseconds(phase_start, phase_cleanup), item.required_subgroup_size);
    }
    return ok;
}

} // namespace

uint32_t storage_image_guest_texel_bytes(prosper::gpu::DataFormat format,
                                         uint32_t components) {
    using prosper::gpu::DataFormat;
    if (components < 1u || components > 4u || !storage_unpack_supported(format)) return 0;
    if (format == DataFormat::Float10_11_11 ||
        format == DataFormat::Unorm2_10_10_10)
        return 4u;
    const uint32_t component_bytes = prosper::gpu::data_format_bytes(format);
    if (!component_bytes || component_bytes > UINT32_MAX / components) return 0;
    return component_bytes * components;
}

bool storage_image_unpack_raw_uvec4(const uint8_t* source, size_t source_bytes,
                                    prosper::gpu::DataFormat format, uint32_t components,
                                    size_t texels, uint32_t* channels,
                                    size_t channel_dwords) {
    const uint32_t guest_texel = storage_image_guest_texel_bytes(format, components);
    if (!source || !channels || !guest_texel || texels > SIZE_MAX / guest_texel ||
        texels > SIZE_MAX / 4u || source_bytes < texels * guest_texel ||
        channel_dwords < texels * 4u)
        return false;
    storage_unpack_range(source, guest_texel, format, components, texels, channels);
    return true;
}

size_t storage_image_raw_uvec4_source_bytes(
    prosper::gpu::DataFormat format, uint32_t components,
    uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_mode,
    bool in_mip_tail, uint32_t mip_tail_bytes) {
    const uint32_t guest_texel = storage_image_guest_texel_bytes(format, components);
    if (!guest_texel || !width || !height || !depth) return 0;
    if (prosper::gpu::tile_mode_is_tiled(tile_mode)) {
        if (depth > 1u)
            return prosper::gpu::tiled_volume_bytes(
                width, height, depth, tile_mode, guest_texel);
        if (in_mip_tail) return mip_tail_bytes;
        return prosper::gpu::tiled_surface_bytes(
            width, height, tile_mode, 0, guest_texel);
    }
    if (width > SIZE_MAX / height) return 0;
    const size_t plane_texels = static_cast<size_t>(width) * height;
    if (plane_texels > SIZE_MAX / depth) return 0;
    const size_t texels = plane_texels * depth;
    return texels <= SIZE_MAX / guest_texel ? texels * guest_texel : 0;
}

bool storage_image_materialize_raw_uvec4(
    const uint8_t* source, size_t source_bytes,
    prosper::gpu::DataFormat format, uint32_t components,
    uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_mode,
    bool in_mip_tail, uint32_t mip_tail_bytes,
    uint32_t mip_tail_x, uint32_t mip_tail_y,
    uint32_t* channels, size_t channel_dwords) {
    const uint32_t guest_texel = storage_image_guest_texel_bytes(format, components);
    const size_t required_source = storage_image_raw_uvec4_source_bytes(
        format, components, width, height, depth, tile_mode,
        in_mip_tail, mip_tail_bytes);
    if (!source || !channels || !guest_texel || !required_source ||
        source_bytes < required_source || width > SIZE_MAX / height)
        return false;
    const size_t plane_texels = static_cast<size_t>(width) * height;
    if (plane_texels > SIZE_MAX / depth) return false;
    const size_t texels = plane_texels * depth;
    if (texels > SIZE_MAX / guest_texel) return false;
    const size_t linear_bytes = texels * guest_texel;

    if (!prosper::gpu::tile_mode_is_tiled(tile_mode))
        return storage_image_unpack_raw_uvec4(
            source, source_bytes, format, components, texels,
            channels, channel_dwords);

    std::vector<uint8_t> linear(linear_bytes);
    bool detiled = true;
    if (depth > 1u) {
        detiled = prosper::gpu::detile_volume(
            linear.data(), source, source_bytes, width, height, depth,
            tile_mode, guest_texel);
    } else if (in_mip_tail) {
        prosper::gpu::detile_surface_level(
            linear.data(), source, source_bytes, width, height, tile_mode,
            guest_texel, mip_tail_x, mip_tail_y);
    } else {
        prosper::gpu::detile_surface(
            linear.data(), source, width, height, tile_mode, 0, guest_texel);
    }
    return detiled && storage_image_unpack_raw_uvec4(
        linear.data(), linear.size(), format, components, texels,
        channels, channel_dwords);
}

bool compute_native_2d_transfer_format_compatible(prosper::gpu::DataFormat format,
                                                  uint32_t components) {
    // Keep this intentionally narrower than every format the sampled uploader can materialize.
    // Ordinary 2D FP16 is deliberately converted to RGBA8 on the sampled side, and packed R11 uses
    // its exact R32_UINT recovery representation on the storage side. Only formats whose current
    // ordinary-2D sampled and typed-storage paths select the same native VkFormat may transfer.
    using prosper::gpu::DataFormat;
    if (format != DataFormat::Float32 && format != DataFormat::Unorm8) return false;
    if (components != 1 && components != 2 && components != 4) return false;
    return native_storage_vk_format(format, components) != VK_FORMAT_UNDEFINED;
}

void report_live_compute_timing_selector_summary() {
    runtime_compute_timing_selector().report_summary();
    runtime_compute_transfer_gate_census().report_summary();
    runtime_compute_authority_census().report_summary();
}

bool import_live_compute_storage_image(const prosper::gpu::ShaderResource& sampled_resource,
                                       uint64_t guest_bytes,
                                       LiveComputeImageImport& import) {
    import = {};
    VulkanComputeContext* context = g_live_compute_context;
    if (!context || !context->device || !sampled_resource.gpu_addr ||
        sampled_resource.cls != prosper::gpu::ResourceClass::Texture ||
        sampled_resource.host_data || !guest_bytes || guest_bytes > UINT32_MAX ||
        !((sampled_resource.img_dim == 1 && sampled_resource.depth == 1) ||
          (sampled_resource.img_dim == 2 && sampled_resource.depth != 0)) ||
        sampled_resource.declared_mip_levels != 1 || sampled_resource.in_mip_tail ||
        sampled_resource.srgb ||
        sampled_resource.depth_compare)
        return false;
    const uint32_t components = sampled_resource.num_components
        ? sampled_resource.num_components : 1u;
    const VkFormat native_format =
        native_storage_vk_format(sampled_resource.format, components);
    if (native_format == VK_FORMAT_UNDEFINED) return false;
    const ComputeImageCacheKey key = storage_image_cache_key(
        sampled_resource, static_cast<uint32_t>(guest_bytes), native_format);
    VkImage image = VK_NULL_HANDLE;
    if (!context->borrow_cached_image_for_graphics(key, image)) return false;
    try {
        auto lease = std::make_shared<BorrowedComputeImageLease>();
        lease->context = context;
        lease->key = key;
        import.width = sampled_resource.width;
        import.height = sampled_resource.height;
        import.depth = sampled_resource.depth;
        import.native_format = static_cast<uint32_t>(native_format);
        import.layout = static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL);
        import.image = static_cast<void*>(image);
        import.device = static_cast<void*>(context->device);
        import.lease = std::move(lease);
    } catch (...) {
        context->release_cached_image(key);
        import = {};
        return false;
    }
    return import.valid();
}

uint64_t live_compute_buffer_gpu_result_skips() {
    return g_buffer_gpu_result_skips.load(std::memory_order_relaxed);
}

uint64_t live_compute_sampled_image_upload_skips() {
    return g_sampled_image_upload_skips.load(std::memory_order_relaxed);
}

uint64_t live_compute_storage_transfer_seeds() {
    return g_compute_storage_transfer_seeds.load(std::memory_order_relaxed);
}

uint64_t live_compute_dcc_post_writeback_promotions() {
    return g_dcc_post_writeback_promotions.load(std::memory_order_relaxed);
}

uint64_t live_compute_dcc_forced_seed_allocation_reuses() {
    return g_dcc_forced_seed_allocation_reuses.load(std::memory_order_relaxed);
}

uint64_t live_compute_dcc_post_writeback_replacements() {
    return g_dcc_post_writeback_replacements.load(std::memory_order_relaxed);
}

bool live_compute_native_storage_3d_supported(prosper::gpu::DataFormat format,
                                              uint32_t components,
                                              uint32_t width, uint32_t height,
                                              uint32_t depth) {
    VulkanComputeContext* context = g_live_compute_context;
    if (!context || !context->physical) return false;
    const VkFormat native_format = native_storage_vk_format(format, components);
    return native_storage_image_create_supported(
        context->physical, native_format, VK_IMAGE_TYPE_3D,
        width, height, depth, 1u);
}

uint64_t live_compute_cpu_fill_dispatches() {
    return g_cpu_fill_dispatches.load(std::memory_order_relaxed);
}

uint64_t live_compute_storage_result_snapshot_bytes() {
    return g_live_compute_context ? g_live_compute_context->storage_result_snapshot_bytes : 0;
}

uint64_t live_compute_image_result_snapshot_bytes() {
    return g_live_compute_context ? g_live_compute_context->image_result_snapshot_bytes : 0;
}

bool cold_storage_result_snapshot_can_defer(bool host_data, bool full_overwrite,
                                            size_t guest_bytes, size_t minimum_bytes) {
    return !host_data && full_overwrite && guest_bytes >= minimum_bytes;
}

void live_compute_fail_next_storage_readback_for_test() {
    g_fail_next_storage_readback_for_test.store(true, std::memory_order_release);
}

void live_compute_leave_next_dcc_metadata_compressed_for_test() {
    g_leave_next_dcc_metadata_compressed_for_test.store(
        true, std::memory_order_release);
}

void live_compute_disable_next_dcc_allocation_reuse_for_test() {
    g_disable_next_dcc_allocation_reuse_for_test.store(
        true, std::memory_order_release);
}

void live_compute_limit_next_image_replacement_for_test() {
    g_limit_next_image_replacement_for_test.store(
        true, std::memory_order_release);
}

void live_compute_zero_next_cold_storage_snapshot_minimum_for_test() {
    g_zero_next_cold_storage_snapshot_minimum_for_test.store(
        true, std::memory_order_release);
}

void live_compute_force_next_image_result_host_fallback_for_test() {
    g_force_next_image_result_host_fallback_for_test.store(true, std::memory_order_release);
}

void live_compute_fail_next_image_result_buffer_retain_for_test() {
    g_fail_next_image_result_buffer_retain_for_test.store(true, std::memory_order_release);
}

void live_compute_force_next_queue_submit_device_lost_for_test() {
    g_force_next_queue_submit_device_lost_for_test.store(true, std::memory_order_release);
}

uint64_t live_compute_queue_submit_attempts() {
    return g_live_compute_queue_submit_attempts.load(std::memory_order_relaxed);
}

void storage_pack_unorm8_range(const uint32_t* channels, uint32_t components,
                               size_t texels, uint8_t* packed) {
    if (!channels || !packed || !texels || !components || components > 4) return;
#if defined(PROSPER_HAVE_TARGET_F16C)
    static const bool use_avx2 = std::getenv("PROSPER_NO_AVX2_UNORM8_PACK") == nullptr &&
                                  __builtin_cpu_supports("avx2");
#endif
    parallel_compute_texels(texels, texels * (sizeof(uint32_t) * 4u + components),
        [&](size_t begin, size_t end) {
#if defined(PROSPER_HAVE_TARGET_F16C)
            if (use_avx2) {
                storage_pack_unorm8_avx2(channels, components, begin, end, packed);
                return;
            }
#endif
            for (size_t texel = begin; texel < end; ++texel)
                for (uint32_t channel = 0; channel < components; ++channel)
                    packed[texel * components + channel] =
                        storage_pack_unorm8(channels[texel * 4 + channel]);
        });
}

void storage_unpack_float16x4_range(const uint8_t* rgba16f, size_t texels, uint32_t* channels) {
    if (!rgba16f || !channels || !texels) return;
#if defined(PROSPER_HAVE_TARGET_F16C)
    static const bool use_f16c = std::getenv("PROSPER_NO_F16C_STORAGE_UNPACK") == nullptr &&
                                 __builtin_cpu_supports("avx2") &&
                                 __builtin_cpu_supports("f16c");
#endif
    parallel_compute_texels(texels, texels * (sizeof(uint16_t) * 4u + sizeof(uint32_t) * 4u),
        [&](size_t begin, size_t end) {
#if defined(PROSPER_HAVE_TARGET_F16C)
            if (use_f16c) {
                storage_unpack_float16x4_f16c(rgba16f, begin, end, channels);
                return;
            }
#endif
            for (size_t texel = begin; texel < end; ++texel)
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    uint16_t bits = 0;
                    std::memcpy(&bits,
                                rgba16f + texel * 8 + channel * sizeof(bits), sizeof(bits));
                    channels[texel * 4 + channel] = storage_unpack_float16_bits(bits);
                }
        });
}

void storage_pack_float16x4_range(const uint32_t* channels, size_t texels, uint8_t* rgba16f) {
    if (!channels || !rgba16f || !texels) return;
#if defined(PROSPER_HAVE_TARGET_F16C)
    static const bool use_f16c = std::getenv("PROSPER_NO_F16C_STORAGE_PACK") == nullptr &&
                                 __builtin_cpu_supports("avx2") &&
                                 __builtin_cpu_supports("f16c");
#endif
    parallel_compute_texels(texels, texels * (sizeof(uint32_t) * 4u + sizeof(uint16_t) * 4u),
        [&](size_t begin, size_t end) {
#if defined(PROSPER_HAVE_TARGET_F16C)
            if (use_f16c) {
                storage_pack_float16x4_f16c(channels, begin, end, rgba16f);
                return;
            }
#endif
            for (size_t texel = begin; texel < end; ++texel) {
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    float value;
                    std::memcpy(&value, channels + texel * 4 + channel, sizeof(value));
                    const uint16_t half = prosper::gpu::float_to_half(value);
                    std::memcpy(rgba16f + texel * 8 + channel * sizeof(half),
                                &half, sizeof(half));
                }
            }
        });
}

void sampled_float16_to_unorm8_range(const uint8_t* source, uint32_t components,
                                     size_t texels, uint8_t* rgba) {
    if (!source || !rgba || !components || components > 4) return;
    const auto& table = sampled_float16_unorm8_table();
#if defined(PROSPER_HAVE_TARGET_F16C)
    static const bool use_f16c = std::getenv("PROSPER_NO_F16C_SAMPLED") == nullptr &&
                                 __builtin_cpu_supports("avx2") &&
                                 __builtin_cpu_supports("f16c");
#else
    constexpr bool use_f16c = false;
#endif
    parallel_compute_texels(texels, texels * (components * 2u + 4u),
        [&](size_t begin, size_t end) {
#if defined(PROSPER_HAVE_TARGET_F16C)
            if (components == 4 && use_f16c) {
                sampled_float16x4_to_unorm8_f16c(source, begin, end, rgba, table);
                return;
            }
#endif
            for (size_t texel = begin; texel < end; ++texel) {
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    if (channel >= components) {
                        rgba[texel * 4 + channel] = channel == 3 ? 255 : 0;
                        continue;
                    }
                    uint16_t half = 0;
                    std::memcpy(&half,
                                source + (texel * components + channel) * sizeof(half),
                                sizeof(half));
                    rgba[texel * 4 + channel] = table[half];
                }
            }
        });
}

bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items) {
    auto fail_closed_items = [&]() {
        for (const auto& item : items)
            prosper::gpu::notify_compute_authority_boundary({
                prosper::gpu::ComputeAuthorityBoundaryKind::Compute,
                item.submit_no, item.command_order, 0, 0, false});
        return false;
    };
    // Keep the Vulkan device alive across dispatch spans. Constructing an instance, device, and queue for
    // every callback cost roughly 25 ms/frame on the native Windows frontend before any kernel work ran.
    // Function-local static initialization is thread-safe; AGC submit execution serializes subsequent use.
    //
    // The context is heap-allocated and torn down from a std::atexit handler registered AFTER
    // vkCreateInstance rather than from a static destructor, because ~VulkanComputeContext() calls
    // into Vulkan (#1704). Destructors of objects with static storage duration and atexit handlers
    // run in one list, in reverse order of registration ([basic.start.term]/3): a static object
    // constructed here would be registered BEFORE vkCreateInstance dlopens any enabled layer, so it
    // would be destroyed AFTER that layer's own statics, and the first vkDestroyBuffer would then
    // dereference the layer's freed dispatch map. That is a deterministic SIGSEGV at exit under
    // VK_LAYER_KHRONOS_validation (reproduced on layers 1.4.341, inside vvl::dispatch::GetData), and
    // the same hazard applies to any implicit layer a user has enabled — MangoHud, vkBasalt, an
    // overlay. Registering the teardown once the instance exists puts it ahead of the layer's in
    // that same reverse-order list.
    //
    // Three things this deliberately does NOT claim:
    //  * The guarantee covers layer state initialized up to and including init(). A layer static
    //    first constructed later — at a first vkCmdDispatch, say — would register after this handler
    //    and be destroyed before it. For VVL the dispatch map is built during instance/device
    //    creation, so that is theoretical, but it is where the argument stops.
    //  * Only statics constructed between the old registration point (completion of this object's
    //    constructor) and the new one (return from init()) change position relative to compute
    //    teardown — in practice the loader, the ICD and any layer. g_rtt, the executor caches and
    //    the renderer context keep exactly the order they had, which is why nothing observable
    //    outside teardown can move.
    //  * Not tearing down at all — what render_vk_ctx() deliberately does for the sibling device —
    //    would also remove the hazard, and is NOT what this does. Teardown is kept because it
    //    releases this context's own pipelines, pools, images and memory on a device that outlives
    //    it, which a deliberate leak would not. It additionally lets a layer report objects still
    //    outstanding at vkDestroyDevice/vkDestroyInstance, but ONLY on the private-device path:
    //    when the device is borrowed the destructor returns at `if (borrowed) return;` before
    //    either call, so that particular benefit does not apply to the live-renderer path.
    static VulkanComputeContext* context_ptr = new VulkanComputeContext();
    static const bool context_ready = [] {
        const bool ok = context_ptr->init();
        // Registered even when init() failed: partially constructed Vulkan state still has to be
        // released while the loader is alive. If std::atexit itself fails there is nothing useful
        // to do but say so — the context then simply outlives the process, as RenderVkCtx does.
        if (std::atexit([] {
                VulkanComputeContext* doomed = context_ptr;
                context_ptr = nullptr;
                g_live_compute_context = nullptr;
                // Returning early from ~VulkanComputeContext would still destroy all of its member
                // caches after the destructor body. When a lost-device submission has no completion
                // proof, retain the object itself so no GPU-facing member ownership is released.
                if (doomed && !doomed->completion_unproven) delete doomed;
            }) != 0) {
            std::fprintf(stderr, "[compute] std::atexit refused the Vulkan teardown handler — "
                                 "the compute device will not be destroyed at exit\n");
        }
        return ok;
    }();
    if (!context_ready) {
        std::fprintf(stderr, "[compute] Vulkan initialization failed\n");
        return fail_closed_items();
    }
    // The teardown handler nulls context_ptr, so a dispatch submitted from another thread once
    // exit() has begun must decline instead of dereferencing it. Read once: the handler writes this
    // pointer without synchronization, and the narrow race it leaves (teardown landing between this
    // load and the first Vulkan call below) is not closable cheaply — but declining loudly in the
    // common case is strictly better than the old static, which left destroyed-but-addressable
    // memory here and simply used it.
    VulkanComputeContext* const live_context = context_ptr;
    if (!live_context) {
        static std::once_flag teardown_logged;
        std::call_once(teardown_logged, [] {
            std::fprintf(stderr, "[compute] dispatch after Vulkan teardown — declining\n");
        });
        return fail_closed_items();
    }
    VulkanComputeContext& context = *live_context;
    g_live_compute_context = &context;
    if (context.device_lost) return fail_closed_items();
    const bool perf_capture_timing =
        prosper::perf::interactive_performance_capture().detailed_timing_active();
    const bool timing_log_enabled = std::getenv("PROSPER_RENDER_TIMING") != nullptr;
    const prosper::frontend::PerformanceTimingMode timing_mode =
        prosper::frontend::performance_timing_mode(timing_log_enabled, perf_capture_timing);
    const bool timing_enabled = timing_mode.measure;
    const auto timing_start = timing_enabled
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    context.begin_write_watch_promotions();
    // Dispatches are independent PM4-order operations: one item failing (e.g. an image shape the
    // backend can't bind yet, #590) must not abort the rest of the batch — that would regress
    // dispatches that executed before image bindings existed. Run all; report all-succeeded.
    bool all_ok = true;
    RuntimeComputeAuthorityCensus& authority_census =
        runtime_compute_authority_census();
    const bool authority_requested = authority_census.requested();
    for (const auto& item : items) {
        if (const std::optional<bool> cpu_result = execute_cpu_fast_path(item)) {
            if (authority_requested) {
                const uint64_t program_hash = prosper::gpu::gpu_capture_hash(
                    reinterpret_cast<const uint8_t*>(item.spirv.data()),
                    item.spirv.size() * sizeof(uint32_t));
                (void)authority_census.observe_program(item, program_hash);
            }
            all_ok &= *cpu_result;
        } else {
            const bool item_ok = execute_item(context, item);
            all_ok &= item_ok;
            if (!item_ok)
                prosper::gpu::notify_compute_authority_boundary({
                    prosper::gpu::ComputeAuthorityBoundaryKind::Compute,
                    item.submit_no, item.command_order, 0, 0, false});
        }
        if (context.device_lost) break;
    }
    if (timing_enabled) {
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timing_start).count();
        if (perf_capture_timing) {
            prosper::perf::ComputeTimingRecord record;
            record.dispatches = items.size();
            record.cpu_fast_total = g_cpu_fill_dispatches.load(std::memory_order_relaxed);
            if (!items.empty() && items.front().code_addr && !items.front().spirv.empty()) {
                const auto& first = items.front();
                const bool homogeneous = std::all_of(
                    items.begin() + 1, items.end(), [&](const auto& item) {
                        return item.code_addr == first.code_addr && item.spirv == first.spirv;
                    });
                if (homogeneous) {
                    record.program_addr = first.code_addr;
                    record.program_hash = prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(first.spirv.data()),
                        first.spirv.size() * sizeof(uint32_t));
                }
            }
            record.total_ms = elapsed;
            prosper::perf::interactive_performance_capture().record_compute(record);
        }
        if (timing_mode.log) {
            struct TimingTotals { uint64_t calls = 0, dispatches = 0; double milliseconds = 0; };
            static TimingTotals totals;
            static TimingTotals window;
            auto accumulate = [&](TimingTotals& timing) {
                timing.calls++;
                timing.dispatches += items.size();
                timing.milliseconds += elapsed;
            };
            accumulate(totals);
            accumulate(window);
            if (totals.calls % 25 == 0) {
                std::fprintf(stderr,
                    "[render-timing] compute calls=%llu dispatches=%llu avg_ms=%.2f\n",
                    (unsigned long long)totals.calls,
                    (unsigned long long)totals.dispatches,
                    totals.milliseconds / static_cast<double>(totals.calls));
                std::fprintf(stderr,
                    "[render-timing] compute_cpu_fast fills=%llu\n",
                    (unsigned long long)g_cpu_fill_dispatches.load(
                        std::memory_order_relaxed));
                const ComputeMemoryPoolStats pool = context.memory_pool_stats();
                std::fprintf(stderr,
                    "[render-timing] compute_memory_pool hits=%llu misses=%llu cached=%zu "
                    "%.1f MiB discarded=%llu\n",
                    (unsigned long long)pool.hits, (unsigned long long)pool.misses,
                    pool.cached_allocations,
                    static_cast<double>(pool.cached_bytes) / (1024.0 * 1024.0),
                    (unsigned long long)pool.discarded);
                std::fprintf(stderr,
                    "[render-timing] compute_image_source snapshots=%llu %.1f MiB "
                    "storage_results=%llu %.1f MiB result_fallbacks=%llu %.1f MiB "
                    "gpu_transfer_seeds=%llu\n",
                    (unsigned long long)context.image_source_snapshot_copies,
                    static_cast<double>(context.image_source_snapshot_bytes) /
                        (1024.0 * 1024.0),
                    (unsigned long long)context.storage_result_snapshot_copies,
                    static_cast<double>(context.storage_result_snapshot_bytes) /
                        (1024.0 * 1024.0),
                    (unsigned long long)context.image_result_snapshot_copies,
                    static_cast<double>(context.image_result_snapshot_bytes) /
                        (1024.0 * 1024.0),
                    (unsigned long long)g_compute_storage_transfer_seeds.load(
                        std::memory_order_relaxed));
                std::fprintf(stderr,
                    "[render-window] compute calls=%llu dispatches=%.1f avg_ms=%.2f\n",
                    (unsigned long long)window.calls,
                    window.dispatches / static_cast<double>(window.calls),
                    window.milliseconds / static_cast<double>(window.calls));
                window = {};
            }
        }
    }
    return all_ok;
}

void register_live_compute() {
    static bool attempted = false;
    if (attempted) return;
    attempted = true;
    // Parse and announce a stable timing selector at backend registration, not at the first matching
    // dispatch. A title that never reaches compute must still prove whether the instrument armed.
    (void)runtime_compute_timing_selector();
    RuntimeComputeAuthorityCensus& authority_census =
        runtime_compute_authority_census();
    if (authority_census.requested()) {
        prosper::gpu::set_compute_authority_boundary_observer(
            [](const prosper::gpu::ComputeAuthorityBoundary& boundary) {
                runtime_compute_authority_census().observe_boundary(boundary);
            });
    }
    const char* enabled = std::getenv("PROSPER_COMPUTE");
    if (enabled && (!std::strcmp(enabled, "0") || !std::strcmp(enabled, "off"))) {
        std::fprintf(stderr, "[compute] live execution disabled by PROSPER_COMPUTE=%s\n", enabled);
        return;
    }
    prosper::gpu::set_submit_compute(execute_live_compute_items);
    std::fprintf(stderr, "[compute] live Vulkan compute backend registered\n");
}

} // namespace prosper::frontend

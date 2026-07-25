#include "live_compute.hpp"
#include "rtt_scale.hpp"
#include "seed_reprove.hpp"
#include "vulkan_device_select.hpp"

#include "gpu/bc_decode.hpp"
#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/shader_resources.hpp"
#include "gpu/tile.hpp"
#include "gpu/writer_provenance.hpp"
#include "host/guest_write_watch.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
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
    const uint32_t source_bytes = snapshot.format == prosper::gpu::LiveTargetPixelFormat::Rgba16Float
        ? 8u : 4u;
    if (texels > SIZE_MAX / source_bytes || texels > SIZE_MAX / sizeof(uint32_t) ||
        snapshot.pixels->size() != static_cast<size_t>(texels) * source_bytes ||
        !packed || packed_size != static_cast<size_t>(texels) * sizeof(uint32_t))
        return false;
    for (size_t t = 0; t < static_cast<size_t>(texels); ++t) {
        float rgb[3]{};
        if (snapshot.format == prosper::gpu::LiveTargetPixelFormat::Rgba16Float) {
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
                                   prosper::gpu::LiveTargetPixelFormat target_format) {
    using prosper::gpu::DataFormat;
    using prosper::gpu::LiveTargetPixelFormat;
    return components == 4 &&
           ((format == DataFormat::Unorm8 &&
             target_format == LiveTargetPixelFormat::Rgba8Unorm) ||
            (format == DataFormat::Float16 &&
             target_format == LiveTargetPixelFormat::Rgba16Float));
}

namespace {

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
                                           uint32_t width, uint32_t height) {
    if (!physical || format == VK_FORMAT_UNDEFINED || !width || !height) return false;
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties properties{};
    return vkGetPhysicalDeviceImageFormatProperties(
               physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
               usage, 0, &properties) == VK_SUCCESS &&
           width <= properties.maxExtent.width && height <= properties.maxExtent.height &&
           properties.maxExtent.depth >= 1 && properties.maxArrayLayers >= 1;
}

constexpr VkDeviceSize kMaxComputeImageBytes = 256ull << 20;
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
    bool operator==(const ComputeBufferCacheKey& other) const {
        return gpu_addr == other.gpu_addr && host_data == other.host_data && bytes == other.bytes;
    }
};

struct ComputeBufferCacheKeyHash {
    size_t operator()(const ComputeBufferCacheKey& key) const {
        size_t result = std::hash<uint64_t>{}(key.gpu_addr);
        result ^= std::hash<uintptr_t>{}(key.host_data) << 1;
        result ^= std::hash<uint32_t>{}(key.bytes) << 2;
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
    // GuestWriteWatch is page-granular internally, but its public query historically collapsed a
    // registration to one Dirty bit. A four-byte guest write could therefore make the persistent
    // compute cache compare an entire 32 MiB buffer. Split large sources into moderately-sized
    // registrations so an exact refresh only scans chunks containing a dirtied page. An unavailable
    // chunk watch remains fail-closed: acquire_cached_buffer falls back to the full byte comparison.
    std::vector<ComputeBufferWriteWatchChunk> write_watches;
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
};

struct ComputeImageCacheKey {
    uint64_t gpu_addr = 0;
    uint32_t guest_bytes = 0;
    uint32_t resource_bytes = 0;
    uint32_t width = 0, height = 0, depth = 0;
    uint32_t format = 0, components = 0, tile_mode = 0, img_dim = 0;
    uint32_t linear_row_pitch = 0;
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
        mix(key.guest_bytes); mix(key.resource_bytes);
        mix(key.width); mix(key.height); mix(key.depth);
        mix(key.format); mix(key.components); mix(key.tile_mode); mix(key.img_dim);
        mix(key.linear_row_pitch); mix(key.mip_tail_offset); mix(key.mip_tail_bytes);
        mix(key.mip_tail_x); mix(key.mip_tail_y); mix(key.vk_format);
        mix(key.storage); mix(key.in_mip_tail); mix(key.srgb); mix(key.depth_compare);
        return result;
    }
};

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
    prosper::gpu::GuestGpuWriteSnapshot validation_snapshot;
    std::vector<uint8_t> source_snapshot;
};

bool persistent_compute_buffer_enabled(uint32_t bytes) {
    static const bool enabled =
        std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_BUFFERS") == nullptr;
    // Small bindings are cheap and numerous. Residency targets the multi-megabyte SSBOs whose
    // create/bind/full-compare cycle is visible in every Astro Bot frame.
    return enabled && bytes >= (1u << 20);
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

bool persistent_compute_image_enabled(VkDeviceSize bytes) {
    static const bool enabled =
        std::getenv("PROSPER_NO_PERSISTENT_COMPUTE_IMAGES") == nullptr;
    // The cache exists to remove full-surface detile/conversion/upload work. Tiny textures are both
    // cheap and numerous, so leave them on the pooled transient path.
    return enabled && bytes >= (1u << 20);
}

VkDeviceSize persistent_compute_image_limit() {
    static const VkDeviceSize limit = []() -> VkDeviceSize {
        const char* value = std::getenv("PROSPER_COMPUTE_IMAGE_CACHE_MB");
        const uint64_t mib = value ? std::strtoull(value, nullptr, 10) : 512ull;
        if (mib > UINT64_MAX / (1024ull * 1024ull)) return VkDeviceSize{UINT64_MAX};
        return static_cast<VkDeviceSize>(mib) * 1024ull * 1024ull;
    }();
    return limit;
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
    // Storage-image support (#590): the recompiler's storage path declares the
    // StorageImageRead/WriteWithoutFormat capabilities (raw uvec4 texel model — see
    // tests/image_compute_runner.h, the exec-diff harness for that contract). When the device lacks
    // the features, image-binding dispatches are skipped loudly instead of creating an invalid device.
    bool image_support = false;
    // True when instance/device/queue were ADOPTED from the live renderer (#1091). A borrowed
    // context is owned by the renderer: destroy our own pipelines/pools/memory, never its device.
    bool borrowed = false;

    ~VulkanComputeContext() {
        release_cached_buffers();
        release_cached_images();
        release_cached_memory();
        if (dispatch_fence) vkDestroyFence(device, dispatch_fence, nullptr);
        if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
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
        }
        buffer_cache.clear();
        buffer_cache_bytes = 0;
    }

    void release_cached_images() {
        if (!device) return;
        for (auto& [key, cached] : image_cache) {
            (void)key;
            cached.write_watch.reset();
            if (cached.image) vkDestroyImage(device, cached.image, nullptr);
            if (cached.memory) release_memory(cached.memory);
        }
        image_cache.clear();
        image_cache_bytes = 0;
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
        const bool submit_unchanged = !key.host_data &&
            prosper::gpu::guest_gpu_writes_since(cached.validation_snapshot,
                                                  key.gpu_addr, key.bytes) ==
                prosper::gpu::GuestGpuWriteQuery::Unchanged;
        std::vector<size_t> dirty_chunks;
        bool watches_complete = !cached.write_watches.empty();
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
            if (!key.host_data) validate_cached_buffer_source(key);
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
        if (!key.host_data) {
            for (uint32_t offset = 0; offset < key.bytes;) {
                const uint32_t bytes = std::min(kComputeBufferWriteWatchChunkBytes,
                                                key.bytes - offset);
                cached.write_watches.push_back({
                    offset, bytes,
                    prosper::host::GuestWriteWatch::create(key.gpu_addr + offset, bytes)});
                offset += bytes;
            }
        }
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

    void validate_cached_buffer_source(const ComputeBufferCacheKey& key) {
        auto found = buffer_cache.find(key);
        if (found == buffer_cache.end() || key.host_data) return;
        CachedComputeBuffer& cached = found->second;
        cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        for (ComputeBufferWriteWatchChunk& chunk : cached.write_watches) {
            if (chunk.watch && chunk.watch.rearm()) continue;
            chunk.watch.reset();
            chunk.watch = prosper::host::GuestWriteWatch::create(
                key.gpu_addr + chunk.offset, chunk.bytes);
        }
    }

    bool make_image_cache_room(VkDeviceSize bytes) {
        const VkDeviceSize limit = persistent_compute_image_limit();
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
            victim->second.write_watch.reset();
            if (victim->second.image) vkDestroyImage(device, victim->second.image, nullptr);
            if (victim->second.memory) release_memory(victim->second.memory);
            image_cache_bytes -= victim->second.allocation_bytes;
            image_cache.erase(victim);
        }
        return true;
    }

    bool acquire_cached_image(const ComputeImageCacheKey& key, const uint8_t* source,
                              uint64_t validation_epoch, VkImage& image,
                              VkDeviceMemory& memory, bool& upload_skipped) {
        auto found = image_cache.find(key);
        if (found == image_cache.end()) return false;
        CachedComputeImage& cached = found->second;
        cached.last_use = ++image_cache_clock;
        ++cached.pins;
        image = cached.image;
        memory = cached.memory;
        if (validation_epoch && cached.validation_epoch == validation_epoch) {
            upload_skipped = cached.validation_result;
            return true;
        }
        const bool submit_unchanged = cached.content_valid &&
            prosper::gpu::guest_gpu_writes_since(cached.validation_snapshot,
                                                  key.gpu_addr, key.guest_bytes) ==
                prosper::gpu::GuestGpuWriteQuery::Unchanged;
        const bool watch_unchanged = !submit_unchanged && cached.content_valid &&
            cached.write_watch &&
            cached.write_watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged;
        const bool exact_unchanged = cached.content_valid && !submit_unchanged &&
            !watch_unchanged && source &&
            cached.source_snapshot.size() == key.guest_bytes &&
            std::memcmp(cached.source_snapshot.data(), source, key.guest_bytes) == 0;
        upload_skipped = submit_unchanged || watch_unchanged || exact_unchanged;
        cached.validation_epoch = validation_epoch;
        cached.validation_result = upload_skipped;
        if (exact_unchanged) {
            cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
            if (!cached.write_watch || !cached.write_watch.rearm()) {
                cached.write_watch.reset();
                cached.write_watch = prosper::host::GuestWriteWatch::create(
                    key.gpu_addr, key.guest_bytes);
            }
        } else if (!upload_skipped && source) {
            cached.source_snapshot.assign(source, source + key.guest_bytes);
            // Do not trust the new mirror until the corresponding transfer completes. A failed
            // submit leaves this false, so the next use refreshes instead of skipping stale pixels.
            cached.content_valid = false;
        }
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
        cached.write_watch = prosper::host::GuestWriteWatch::create(
            key.gpu_addr, key.guest_bytes);
        cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        if (source)
            cached.source_snapshot.assign(source, source + key.guest_bytes);
        auto [it, inserted] = image_cache.emplace(key, std::move(cached));
        if (!inserted) return false;
        image_cache_bytes += allocation_bytes;
        return true;
    }

    void release_cached_image(const ComputeImageCacheKey& key) {
        auto found = image_cache.find(key);
        if (found != image_cache.end() && found->second.pins) --found->second.pins;
    }

    void validate_cached_image_source(const ComputeImageCacheKey& key,
                                      const uint8_t* current_source = nullptr) {
        auto found = image_cache.find(key);
        if (found == image_cache.end()) return;
        CachedComputeImage& cached = found->second;
        // A storage dispatch replaces both the image and its guest-memory mirror. Retain those
        // result bytes as the exact comparison baseline; keeping the pre-dispatch input here could
        // misclassify a later guest write that restores that old input as "unchanged".
        if (current_source)
            cached.source_snapshot.assign(current_source,
                                          current_source + key.guest_bytes);
        cached.content_valid = true;
        cached.validation_snapshot = prosper::gpu::guest_gpu_write_snapshot();
        if (cached.write_watch && cached.write_watch.rearm()) return;
        cached.write_watch.reset();
        cached.write_watch = prosper::host::GuestWriteWatch::create(
            key.gpu_addr, key.guest_bytes);
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
            VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            if (vkCreatePipelineCache(device, &pcci, nullptr, &pipeline_cache) == VK_SUCCESS) {
                vkGetPhysicalDeviceMemoryProperties(physical, &memory);
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

struct BoundBuffer {
    const prosper::gpu::ShaderResource* resource = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t alias_of = SIZE_MAX;         // exact guest range sharing an earlier storage buffer
    bool writable = false;              // reflected OpStore/writing-atomic reachability
    bool persistent = false;
    bool upload_skipped = false;
    uint32_t dirty_watch_chunks = 0;
    uint32_t total_watch_chunks = 0;
    ComputeBufferCacheKey cache_key{};
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
// views where shader-visible numeric semantics require them) or a storage image (R32G32B32A32_UINT
// texels — the recompiler's format-free contract, exec-diff proven by tests/image_compute_runner.h).
struct BoundImage {
    const prosper::gpu::ShaderResource* resource = nullptr;
    uint32_t binding = 0;
    bool storage = false;               // storage image: read back + pack to guest after the dispatch
    bool native_float_storage = false;  // Vulkan performs exact UNORM/float conversion at native width
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
    VkFormat imported_format = VK_FORMAT_UNDEFINED;
    bool persistent = false;            // guest-backed sampled image retained across dispatches
    bool cache_candidate = false;
    bool upload_skipped = false;         // write watch proved the cached source unchanged
    VkDeviceSize allocation_bytes = 0;
    ComputeImageCacheKey cache_key{};
    std::vector<uint8_t> cache_source_snapshot; // first-use source captured before the transfer
    bool seed_skip = false;             // #1122: write-only full-coverage storage image; no seed needed
    bool poison_verify = false;         // #1122: proving frame -- seed poison, prove full coverage
    std::vector<uint8_t> seed_linear;   // #1122: detiled guest seed, kept on a proving frame so any
                                        // texel the write leaves untouched is restored (not corrupted)
    uint64_t imported_addr = 0;
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
// The recompiler moves image texels as RAW 32-bit VGPR channel values (uvec4 per texel; the VkImage is
// R32G32B32A32_UINT with format-free reads/writes). Real hardware format-converts per the T#, so the
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
        case DF::Unorm16:
            for (size_t t = 0; t < count; ++t) {
                const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                for (uint32_t c = 0; c < n; ++c) {
                    const uint16_t value = storage_pack_unorm16(in[c]);
                    p[c * 2] = static_cast<uint8_t>(value);
                    p[c * 2 + 1] = static_cast<uint8_t>(value >> 8);
                }
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
    const std::vector<uint32_t>& spirv = item.spirv;
    const bool trace = trace_compute_item(item);
    const uint64_t image_validation_epoch = ++ctx.image_validation_clock;
    // #1122 review B1: enough invocations for all texels is NECESSARY but NOT SUFFICIENT
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
    // The phase timer is also useful for buffer-only kernels. Astro Bot's heaviest dispatcher writes
    // buffers exclusively, so the historical storage-image gate hid the actual frame bottleneck.
    const bool phase_timing = std::getenv("PROSPER_COMPUTE_PHASE_TIMING") != nullptr;
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
    auto vk_ok = [&](VkResult result, const char* stage) {
        if (result == VK_SUCCESS) return true;
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
                ctx.release_cached_buffer(buffer.cache_key);
                continue;
            }
            if (buffer.buffer) vkDestroyBuffer(ctx.device, buffer.buffer, nullptr);
            if (buffer.memory) ctx.release_memory(buffer.memory);
        }
        for (size_t i = 0; i < images.size(); i++) {
            // A pin is taken per successful import, so it is released per import -- including for a
            // binding that a later alias check folded into an earlier one (#1095).
            if (images[i].imported)
                release_live_render_target_image(images[i].imported_addr);
            if (images[i].alias_of != SIZE_MAX) continue;
            if (images[i].sampler) vkDestroySampler(ctx.device, images[i].sampler, nullptr);
            if (images[i].view) vkDestroyImageView(ctx.device, images[i].view, nullptr);
            if (images[i].persistent) {
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
        for (size_t i = 0; i < descriptors.size(); i++) {
            const ShaderResource* resource = item.resources->by_binding(descriptors[i].binding);
            if (!resource || !resource->size ||
                ((!resource->host_data || resource->host_data_size < resource->size) &&
                 !guest_readable(resource->gpu_addr, resource->size))) break;
            buffers[i].resource = resource;
            buffers[i].writable = descriptors[i].writable;
            for (size_t j = 0; j < i; ++j) {
                const ShaderResource* prior = buffers[j].resource;
                if (!prior || prior->gpu_addr != resource->gpu_addr ||
                    prior->size != resource->size ||
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
                const uint8_t* source = resource_bytes(resource);
                if (trace) buffers[i].before_hash = fnv1a(source, resource->size);
                buffers[i].cache_key = {
                    resource->gpu_addr, reinterpret_cast<uintptr_t>(resource->host_data),
                    resource->size};
                const bool cache_candidate = persistent_compute_buffer_enabled(resource->size);
                if (cache_candidate && ctx.acquire_cached_buffer(
                        buffers[i].cache_key, source, buffers[i].buffer, buffers[i].memory,
                        buffers[i].upload_skipped, buffers[i].dirty_watch_chunks,
                        buffers[i].total_watch_chunks)) {
                    buffers[i].persistent = true;
                } else {
                    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                    bci.size = resource->size;
                    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                    if (vkCreateBuffer(ctx.device, &bci, nullptr, &buffers[i].buffer) != VK_SUCCESS)
                        break;
                    VkMemoryRequirements requirements{};
                    vkGetBufferMemoryRequirements(ctx.device, buffers[i].buffer, &requirements);
                    const uint32_t memory_type = ctx.host_memory_type(requirements.memoryTypeBits);
                    if (memory_type == UINT32_MAX) break;
                    buffers[i].memory = ctx.allocate_memory(requirements.size, memory_type, true);
                    if (!buffers[i].memory) break;
                    if (vkBindBufferMemory(ctx.device, buffers[i].buffer, buffers[i].memory, 0) !=
                        VK_SUCCESS)
                        break;
                    void* mapped = nullptr;
                    if (ctx.map_memory(buffers[i].memory, 0, resource->size, &mapped) != VK_SUCCESS)
                        break;
                    // Pooled host-visible allocations retain their previous contents. Compare them
                    // with current guest memory before uploading: any mutation takes the exact copy.
                    if (!compute_buffers_equal(mapped, source, resource->size))
                        copy_compute_buffer(mapped, source, resource->size);
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
        bool buffers_ready = true;
        for (const auto& buffer : buffers) buffers_ready &= buffer.resource && buffer.memory;
        if (!buffers_ready) break;
        setup_buffers_ms = std::chrono::duration<double, std::milli>(
            ComputeClock::now() - setup_validate_start).count() - setup_validate_ms;

        // --- Image bindings (#590): sampled textures use RGBA8 unless integer/packed-float semantics
        // require a native view; storage images are R32G32B32A32_UINT raw-channel texels (the recompiler's
        // format-free contract, exec-diff proven by tests/image_compute_runner.h) with per-format
        // pack/unpack against the guest surface. Everything not provably correct skips LOUDLY. ---
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
                std::fprintf(stderr, "[compute] program 0x%llx image 0x%llx %ux%ux%u fmt=%u comps=%u "
                                     "tile=%u size=%u: %s -> "
                                     "dispatch skipped (#590)\n",
                             (unsigned long long)item.code_addr, (unsigned long long)key,
                             r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0,
                             r ? (unsigned)r->format : 0u, r ? r->num_components : 0u,
                             r ? r->tile_mode : 0u, r ? r->size : 0u, why);
            }
            images_ready = false;
        };
        const bool image_timing = std::getenv("PROSPER_COMPUTE_IMAGE_TIMING") != nullptr;
        for (size_t i = 0; i < image_descriptors.size() && images_ready; i++) {
            const auto image_start = ComputeClock::now();
            const ShaderResource* r = item.resources->by_binding(image_descriptors[i].binding);
            if (!r || !r->width || !r->height) { skip_image(r, "no/degenerate resource"); break; }
            BoundImage& bi = images[i];
            bi.resource = r;
            bi.binding = image_descriptors[i].binding;
            bi.storage = image_descriptors[i].kind == SpirvDescriptorKind::StorageImage;
            const uint32_t descriptor_components =
                r->num_components ? r->num_components : 1;
            const VkFormat native_storage_format =
                native_storage_vk_format(r->format, descriptor_components);
            const bool spirv_native_storage =
                bi.storage && image_descriptors[i].storage_float;
            const bool ordinary_2d_storage = r->img_dim == 1 && r->depth == 1 &&
                                             !r->depth_compare;
            const bool native_storage_supported = native_float_storage_image_supported(
                r->format, descriptor_components, r->srgb,
                ordinary_2d_storage && native_storage_image_create_supported(
                    ctx.physical, native_storage_format, r->width, r->height));
            if (spirv_native_storage && !native_storage_supported) {
                skip_image(r, "compiled typed storage format is unsupported by this device");
                break;
            }
            // SPIR-V is authoritative here: a raw-uvec4 module must keep the conversion path even
            // if this replay device supports the optional typed format, while a live module is only
            // emitted as float after the frontend's physical-device feature query.
            bi.native_float_storage = spirv_native_storage;
            if (trace)
                std::fprintf(stderr,
                             "[compute]   image binding=%u class=%s addr=0x%llx "
                             "extent=%ux%ux%u format=%u components=%u tile=%u native-storage=%u\n",
                             bi.binding, bi.storage ? "storage" : "sampled",
                             (unsigned long long)r->gpu_addr, r->width, r->height, r->depth,
                             (unsigned)r->format, r->num_components, r->tile_mode,
                             bi.native_float_storage ? 1u : 0u);
            // A surface whose CURRENT pixels live in the renderer's RTT cache must not be read from
            // raw guest memory (empty/stale — the Dead Cells 642x362 lesson).
            const bool dim_1d = r->img_dim == 0;
            const bool dim_3d = r->img_dim == 2;
            const bool dim_2d_array = r->img_dim == 5 && r->depth_compare;
            // A SINGLE-LAYER 2D array (img_dim==5, depth==1, no depth-compare) is byte-identical to a
            // plain 2D image (one layer, same tiling), so it flows through the 2D path below unchanged.
            // The general multi-layer/array case stays deferred to #657. DOLL's post-process compute
            // declares its mask/LUT surfaces this way; skipping them left the tonemap sampling an empty
            // surface -> a zero mask -> black composite even though the scene renders (#319/#657).
            const bool dim_2d_single = r->img_dim == 5 && !r->depth_compare && r->depth == 1;
            if (!dim_1d && r->img_dim != 1 && !dim_3d && !dim_2d_array && !dim_2d_single) {
                skip_image(r, "layered image deferred to #657"); break;
            }
            if (dim_1d && r->height != 1) { skip_image(r, "1D image has non-unit height"); break; }
            if (!r->depth || (!dim_3d && !dim_2d_array && r->depth != 1)) {
                skip_image(r, "image depth does not match its dimensionality"); break;
            }
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
            bool renderer_owned = !r->in_mip_tail && is_live_render_target(r->gpu_addr);
            static const uint32_t render_scale = [] {
                const char* e = std::getenv("PROSPER_RENDER_SCALE");
                const long v = e ? std::strtol(e, nullptr, 10) : 1;
                return v > 0 ? static_cast<uint32_t>(v) : 1u;
            }();
            // Bind the renderer's own image when it is the authoritative copy and this binding
            // matches it EXACTLY (#1095, phase 2 of #1091). RGBA8 and RGBA16F targets both have a
            // byte-identical sampled Vulkan view, so neither needs to round-trip through a CPU
            // snapshot. The FP16 case is especially important for full-resolution post processing:
            // converting a 4K target down to UNORM8 cost more than the compute dispatch itself and
            // also discarded HDR values. Aliases and numerically converted views keep the snapshot
            // path below.
            const bool depth_import_eligible = !bi.storage && !dim_1d && !dim_3d &&
                !dim_2d_array && r->depth == 1 && r->img_dim == 1 &&
                r->format == DataFormat::Float32 &&
                (r->num_components ? r->num_components : 1u) == 1u;
            // Persistent renderer images do not carry VK_IMAGE_USAGE_STORAGE_BIT, and a writable
            // storage import would also leave overlapping guest buffer aliases stale. Storage
            // descriptors therefore retain the owned-image + guest-writeback path.
            if (!bi.storage && (renderer_owned || depth_import_eligible) &&
                !dim_1d && !dim_3d && !dim_2d_array &&
                r->depth == 1 && !r->depth_compare) {
                LiveTargetImageImport import;
                const bool normalized_sampling =
                    image_descriptors[i].normalized_sampling &&
                    !image_descriptors[i].texel_access;
                const LiveTargetImageRequest import_request{
                    r->width, r->height, render_scale, depth_import_eligible,
                    normalized_sampling};
                const bool import_available = import_live_render_target_image(
                    r->gpu_addr, import_request, import);
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
                              import.format);
                    const bool compatible_device =
                        import.device == static_cast<void*>(ctx.device);
                    const bool direct_extent =
                        prosper::frontend::rtt_direct_import_compatible(
                            bi.storage, r->width, r->height, import.width, import.height,
                            render_scale, normalized_sampling);
                    if (compatible_format && direct_extent && compatible_device) {
                        bi.imported = true;
                        bi.imported_depth = depth_import;
                        bi.imported_format = depth_import ? depth_format : VK_FORMAT_UNDEFINED;
                        bi.imported_addr = r->gpu_addr;
                        bi.imported_saved_layout = import.layout;
                        bi.image = static_cast<VkImage>(import.image);
                        live_target.width = import.width;
                        live_target.height = import.height;
                        live_target.format = import.format;
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   bound renderer RTT in place binding=%u class=%s "
                                         "addr=0x%llx extent=%ux%u format=%s\n",
                                         bi.binding, bi.storage ? "storage" : "sampled",
                                         (unsigned long long)r->gpu_addr,
                                         import.width, import.height,
                                         depth_import ? "depth"
                                             : import.format == LiveTargetPixelFormat::Rgba16Float
                                                   ? "rgba16f" : "rgba8");
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
            if (renderer_owned && !bi.imported && !bi.seed_skip) {
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
                    const uint64_t bpp = live_target.format == LiveTargetPixelFormat::Rgba16Float ? 8u : 4u;
                    if (scaled_extent && live_target.pixels &&
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
                    const uint64_t bpp = live_target.format == LiveTargetPixelFormat::Rgba16Float ? 8u : 4u;
                    const uint64_t texels = static_cast<uint64_t>(r->width) * r->height;
                    if (texels > UINT64_MAX / bpp) {
                        skip_image(r, "renderer-owned RTT snapshot size overflow"); break;
                    }
                    const uint64_t expected = texels * bpp;
                    if (expected != live_target.pixels->size()) {
                        skip_image(r, "renderer-owned RTT snapshot byte count mismatch"); break;
                    }
                }
                if (renderer_owned && !bi.storage &&
                    (r->format == DataFormat::Float32 || r->format == DataFormat::Uint16 ||
                     r->format == DataFormat::Uint32)) {
                    const uint64_t cached_bpp =
                        live_target.format == LiveTargetPixelFormat::Rgba16Float ? 8u : 4u;
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
                         r->format == DataFormat::Float16 && nc == 4);
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
                                         live_target.format == LiveTargetPixelFormat::Rgba16Float
                                             ? "rgba16f" : "rgba8",
                                         (unsigned)r->format, nc);
                    }
                }
            }
            // Exact descriptor aliases share one Vulkan image. Storage aliases must observe the same
            // read/modify/write state and produce one guest writeback; sampled aliases avoid repeatedly
            // detiling and uploading Astro Bot's full-resolution source through dozens of SRT slots.
            for (size_t j = 0; j < i; j++) {
                const BoundImage& prior = images[j];
                const ShaderResource* p = prior.resource;
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
                const bool same_view = p && prior.storage == bi.storage &&
                    p->gpu_addr == r->gpu_addr && p->size == r->size &&
                    p->width == r->width && p->height == r->height && p->depth == r->depth &&
                    p->format == r->format && p->num_components == r->num_components &&
                    p->tile_mode == r->tile_mode && p->img_dim == r->img_dim &&
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
                bi.dcc_metadata = owner.dcc_metadata;
                bi.dcc_metadata_bytes = owner.dcc_metadata_bytes;
                staging_bytes[i] = staging_bytes[bi.alias_of];
                if (trace)
                    std::fprintf(stderr, "[compute]   image-alias binding=%u -> binding=%u addr=0x%llx\n",
                                 bi.binding, owner.binding, (unsigned long long)r->gpu_addr);
                break;
            }
            if (bi.alias_of != SIZE_MAX) {
                if (image_timing)
                    std::fprintf(stderr,
                                 "[compute-image] code=0x%llx binding=%u class=%s alias=1 "
                                 "extent=%ux%ux%u ms=%.3f\n",
                                 (unsigned long long)item.code_addr, bi.binding,
                                 bi.storage ? "storage" : "sampled", r->width, r->height, r->depth,
                                 std::chrono::duration<double, std::milli>(
                                     ComputeClock::now() - image_start).count());
                continue;
            }
            const VkDeviceSize volume_texels = static_cast<VkDeviceSize>(r->width) * r->height * r->depth;
            const uint32_t sampled_components = r->num_components ? r->num_components : 1;
            const bool sampled_depth = !bi.storage && r->depth_compare && dim_2d_array &&
                                       sampled_components == 1 &&
                                       (r->format == DataFormat::Float32 ||
                                        r->format == DataFormat::Unorm16);
            if (r->depth_compare && !sampled_depth) {
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
            // Imported renderer-owned RGBA16F is already the exact native sampled representation.
            // Guest-backed FP16 retains the historical RGBA8 conversion: native RG16F sampling is
            // exact, but was measured 7x slower in Astro Bot's full-resolution composite on RADV.
            const bool sampled_float16_native = !bi.storage && bi.imported &&
                                                r->format == DataFormat::Float16 &&
                                                sampled_components == 4;
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
            const uint32_t native_storage_bytes = bi.native_float_storage
                ? (r->format == DataFormat::Float10_11_11
                       ? 4u : data_format_bytes(r->format) * sampled_components)
                : 0u;
            const uint32_t texel_bytes = bi.native_float_storage ? native_storage_bytes
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
            // Storage images use raw uvec4 channels. Most sampled formats are normalized to RGBA8,
            // but Float32 stays native so exposure/HDR kernels do not lose sign or dynamic range.
            const VkDeviceSize sbytes = volume_texels * texel_bytes;
            if (!sbytes || sbytes > kMaxComputeImageBytes) {
                skip_image(r, "expanded image exceeds the 256 MiB backend bound"); break;
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
            if (!bi.storage && !renderer_owned) {
                // Resolve and validate the guest source before allocating staging. A proven cache
                // hit can then avoid the staging buffer/map as well as conversion and GPU upload.
                if (sampled_bpb && dim_3d) {
                    skip_image(r, "block-compressed 3D texture deferred"); break;
                } else if (sampled_bpb) {
                    const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                    const size_t slice = r->tile_mode
                        ? tiled_elements_bytes(bw, bh, sampled_bpb, r->tile_mode)
                        : static_cast<size_t>(bw) * bh * sampled_bpb;
                    sampled_guest_need = dim_2d_array ? slice * r->depth : slice;
                } else if (sampled_rgba8 || sampled_uint8 || sampled_r8 || sampled_f16 ||
                           sampled_f32 || sampled_r11g11b10 || sampled_unorm8x2 ||
                           sampled_unorm16_native || sampled_uint8_native ||
                           sampled_uint16_native || sampled_uint32_native || sampled_depth) {
                    const uint32_t bpt = sampled_r11g11b10 ? 4u : sampled_cb * sampled_nc;
                    if (r->tile_mode && dim_3d && r->depth > 1) {
                        sampled_guest_need = tiled_volume_bytes(
                            r->width, r->height, r->depth, r->tile_mode, bpt);
                    } else if (r->tile_mode && dim_2d_array) {
                        sampled_guest_need = tiled_surface_bytes(
                            r->width, r->height, r->tile_mode, 0, bpt) * r->depth;
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
                    skip_image(r, "sampled backing exceeds the 256 MiB backend bound"); break;
                }
                if (sampled_bpb && dim_2d_array) {
                    skip_image(r, "block-compressed sampled arrays deferred"); break;
                }
                bi.guest_bytes = sampled_guest_need;
                sampled_guest_source = resource_bytes_for(r, sampled_guest_need);
                const bool readable =
                    (r->host_data && r->host_data_size >= sampled_guest_need) ||
                    guest_readable(r->gpu_addr, static_cast<uint32_t>(sampled_guest_need));
                if (!readable) { skip_image(r, "sampled surface unreadable"); break; }

                bool dcc_cache_safe = !r->compression_enabled;
                if (r->compression_enabled) {
                    const uint64_t metadata_bytes = gpu_capture_dcc_metadata_footprint(*r);
                    const uint8_t* metadata =
                        r->dcc_metadata_host_data &&
                                r->dcc_metadata_host_data_size >= metadata_bytes
                            ? r->dcc_metadata_host_data : nullptr;
                    if (!metadata && metadata_bytes && metadata_bytes <= UINT32_MAX &&
                        guest_readable(r->metadata_addr, static_cast<uint32_t>(metadata_bytes)))
                        metadata = reinterpret_cast<const uint8_t*>(uintptr_t(r->metadata_addr));
                    dcc_cache_safe = metadata && metadata_bytes &&
                        std::all_of(metadata, metadata + metadata_bytes,
                                    [](uint8_t value) { return value == 0xff; });
                }
                bi.cache_candidate = !r->host_data && dcc_cache_safe &&
                    persistent_compute_image_enabled(sbytes);
                if (bi.cache_candidate) {
                    bi.cache_key = {
                        r->gpu_addr, static_cast<uint32_t>(sampled_guest_need), r->size,
                        r->width, r->height, r->depth,
                        static_cast<uint32_t>(r->format), sampled_components,
                        r->tile_mode, r->img_dim, r->linear_row_pitch_bytes,
                        r->mip_tail_offset, r->mip_tail_bytes,
                        r->mip_tail_x, r->mip_tail_y,
                        static_cast<uint32_t>(image_format), bi.storage, r->in_mip_tail,
                        r->srgb, r->depth_compare};
                    bi.persistent = ctx.acquire_cached_image(
                        bi.cache_key, sampled_guest_source, image_validation_epoch,
                        bi.image, bi.memory, bi.upload_skipped);
                    if (trace && bi.persistent)
                        std::fprintf(stderr,
                                     "[compute]   persistent sampled image binding=%u "
                                     "addr=0x%llx guest=%zu upload-skipped=%u\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     sampled_guest_need, bi.upload_skipped ? 1u : 0u);
                    if (!bi.persistent)
                        bi.cache_source_snapshot.assign(
                            sampled_guest_source,
                            sampled_guest_source + sampled_guest_need);
                }
            }

            // Allocate the host-visible staging buffer before conversion and write into its mapping
            // directly. The old path first built a heap upload and then memcpy'd the complete result
            // here -- an extra 132 MiB CPU pass for Astro Bot's 4K RGBA32 storage representation.
            ScopedMappedMemory upload_mapping(ctx);
            const size_t upload_size = (bi.imported || bi.seed_skip)
                ? size_t{0} : (size_t)sbytes;
            if (!bi.imported && !(bi.persistent && bi.upload_skipped)) {
                VkBufferCreateInfo sci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                sci.size = sbytes;
                sci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                if (!vk_ok(vkCreateBuffer(ctx.device, &sci, nullptr, &staging[i]),
                           "image-staging-buffer")) { images_ready = false; break; }
                VkMemoryRequirements sreq{};
                vkGetBufferMemoryRequirements(ctx.device, staging[i], &sreq);
                const uint32_t staging_memory_type = ctx.host_memory_type(sreq.memoryTypeBits);
                staging_memory[i] = ctx.allocate_memory(sreq.size, staging_memory_type, true);
                if (!vk_handle_ok(staging_memory[i], "image-staging-memory") ||
                    !vk_ok(vkBindBufferMemory(ctx.device, staging[i], staging_memory[i], 0),
                           "image-staging-bind")) {
                    images_ready = false; break;
                }
                upload_mapping.memory = staging_memory[i];
                if (!bi.seed_skip &&
                    !vk_ok(ctx.map_memory(staging_memory[i], 0, sbytes,
                                          &upload_mapping.data), "image-staging-map")) {
                    images_ready = false; break;
                }
            }
            auto* upload = static_cast<uint8_t*>(upload_mapping.data);
            if (bi.imported) {
                // The renderer's sampled image is the source, so direct imports need no transfer.
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
                const size_t guest_bytes = r->tile_mode
                    ? (r->depth > 1
                           ? tiled_volume_bytes(r->width, r->height, r->depth, r->tile_mode,
                                                static_cast<uint32_t>(guest_texel))
                           : tiled_surface_bytes(r->width, r->height, r->tile_mode, 0,
                                                 static_cast<uint32_t>(guest_texel)))
                    : static_cast<size_t>(linear_guest_bytes);
                if (!linear_guest_bytes || linear_guest_bytes > SIZE_MAX || !guest_bytes ||
                    guest_bytes > UINT32_MAX || (!r->tile_mode && guest_bytes > r->size)) {
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
                // Native storage targets are byte-identical to their sampled view. Retain them after
                // writeback so the next dispatch/frame can consume the GPU result directly instead
                // of detiling and uploading the same multi-megabyte surface again. Poison proving
                // deliberately stays transient: a partial proof repairs untouched texels on the CPU
                // after readback, so its device image is not yet the repaired authoritative value.
                const bool dcc_cache_safe = !r->compression_enabled ||
                    (bi.dcc_metadata && bi.dcc_metadata_bytes &&
                     std::all_of(bi.dcc_metadata, bi.dcc_metadata + bi.dcc_metadata_bytes,
                                 [](uint8_t value) { return value == 0xff; }));
                bi.cache_candidate = !renderer_owned && !r->host_data && dcc_cache_safe &&
                    !bi.seed_skip && !bi.poison_verify &&
                    bi.native_float_storage && persistent_compute_image_enabled(sbytes);
                if (bi.cache_candidate) {
                    bi.cache_key = {
                        r->gpu_addr, static_cast<uint32_t>(guest_bytes), r->size,
                        r->width, r->height, r->depth,
                        static_cast<uint32_t>(r->format), sampled_components,
                        r->tile_mode, r->img_dim, r->linear_row_pitch_bytes,
                        r->mip_tail_offset, r->mip_tail_bytes,
                        r->mip_tail_x, r->mip_tail_y,
                        static_cast<uint32_t>(image_format), bi.storage, r->in_mip_tail,
                        r->srgb, r->depth_compare};
                    bi.persistent = ctx.acquire_cached_image(
                        bi.cache_key, resource_bytes_for(r, guest_bytes), image_validation_epoch,
                        bi.image, bi.memory, bi.upload_skipped);
                    if (trace && bi.persistent)
                        std::fprintf(stderr,
                                     "[compute]   persistent storage image binding=%u "
                                     "addr=0x%llx guest=%zu upload-skipped=%u\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     guest_bytes, bi.upload_skipped ? 1u : 0u);
                }
                if (!(bi.persistent && bi.upload_skipped)) {
                const size_t linear_size = bi.seed_skip
                    ? size_t{0} : static_cast<size_t>(linear_guest_bytes);
                std::unique_ptr<uint8_t[]> linear;
                if (linear_size && !renderer_owned && r->tile_mode)
                    linear = std::make_unique_for_overwrite<uint8_t[]>(linear_size);
                const uint8_t* unpack_source = nullptr;
                if (bi.seed_skip) {
                    // #1122: write-only full-coverage target -- the shader overwrites every texel, so
                    // the image is created but never seeded, uploaded, or read. Nothing to fill.
                } else if (renderer_owned) {
                    unpack_source = live_target.pixels->data();
                    if (trace) {
                        bi.before_hash = fnv1a(unpack_source, linear_size);
                        std::fprintf(stderr,
                                     "[compute]   imported writable renderer RTT binding=%u "
                                     "addr=0x%llx extent=%ux%u format=%s\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     r->width, r->height,
                                     live_target.format == LiveTargetPixelFormat::Rgba16Float
                                         ? "rgba16f" : "rgba8");
                    }
                } else if (r->tile_mode && r->depth > 1) {
                    if (trace) bi.before_hash = fnv1a(src, guest_bytes);
                    if (!detile_volume(linear.get(), src, guest_bytes, r->width, r->height,
                                       r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                        skip_image(r, "storage volume detile failed"); break;
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
                if (!bi.seed_skip) {
                    if (bi.native_float_storage) {
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
                    if (bi.native_float_storage) {
                        // Transfers preserve this finite, non-special native texel pattern exactly.
                        // A texel that remains all 0x5a after the dispatch was not stored.
                        std::memset(upload, 0x5a, static_cast<size_t>(sbytes));
                    } else {
                        uint32_t* pp = reinterpret_cast<uint32_t*>(upload);
                        for (size_t t = 0; t < texels * 4; ++t) pp[t] = 0xDEADBEEFu;
                    }
                }
                static const bool verify_unpack =
                    std::getenv("PROSPER_VERIFY_UNPACK") != nullptr && !bi.seed_skip;
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
                    if (r11g11b10) {
                        if (!pack_live_target_r11g11b10(live_target, upload, upload_size)) {
                            skip_image(r, "renderer RTT R11G11B10 reconstruction failed");
                            break;
                        }
                        if (trace)
                            std::fprintf(stderr,
                                         "[compute]   reconstructed renderer RTT binding=%u "
                                         "addr=0x%llx extent=%ux%u rgba%s -> R11G11B10\n",
                                         bi.binding, (unsigned long long)r->gpu_addr,
                                         r->width, r->height,
                                         live_target.format == LiveTargetPixelFormat::Rgba16Float
                                             ? "16f" : "8");
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
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
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
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
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
                                if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
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
                                        if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
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
                    } else if (live_target.format == LiveTargetPixelFormat::Rgba8Unorm) {
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
                                     live_target.format == LiveTargetPixelFormat::Rgba16Float
                                         ? "rgba16f" : "rgba8",
                                     (unsigned long long)snapshot_hash, nonzero_bytes);
                    }
                    bi.guest_bytes = 0;
                } else {
                    const size_t need = sampled_guest_need;
                    const uint8_t* src = sampled_guest_source;
                    if (!bi.upload_skipped && bpb) {                 // BCn: (block-detile ->) decode
                        const uint32_t bw = (r->width + 3) / 4, bh = (r->height + 3) / 4;
                        if (dim_2d_array) {
                            skip_image(r, "block-compressed sampled arrays deferred"); break;
                        }
                        std::unique_ptr<uint8_t[]> linear;
                        const uint8_t* decode_source = src;
                        if (r->tile_mode) {
                            linear = std::make_unique_for_overwrite<uint8_t[]>((size_t)bw * bh * bpb);
                            decode_source = linear.get();
                        }
                        if (r->tile_mode && r->in_mip_tail)
                            detile_elements_level(linear.get(), src, need, bw, bh, bpb,
                                                  r->tile_mode, r->mip_tail_x,
                                                  r->mip_tail_y);
                        else if (r->tile_mode)
                            detile_elements(linear.get(), src, need, bw, bh, bpb, r->tile_mode);
                        if (!bc_decode_surface(upload, decode_source, (size_t)bw * bh * bpb,
                                               r->width, r->height, r->format)) {
                            skip_image(r, "BC decode unsupported"); break; }
                    } else if (!bi.upload_skipped) {
                        const uint32_t bpt = r11g11b10 ? 4u : cb * nc;
                        const size_t linear_bytes = static_cast<size_t>(volume_texels) * bpt;
                        std::unique_ptr<uint8_t[]> linear;
                        const uint8_t* sampled_source = src;
                        if (r->tile_mode) {
                            linear = std::make_unique_for_overwrite<uint8_t[]>(linear_bytes);
                            sampled_source = linear.get();
                        }
                        if (r->tile_mode && dim_3d && r->depth > 1) {
                            if (!detile_volume(linear.get(), src, need,
                                               r->width, r->height, r->depth,
                                               r->tile_mode, bpt)) {
                                skip_image(r, "sampled volume detile failed"); break;
                            }
                        } else if (r->tile_mode && dim_2d_array) {
                            const size_t tiled_slice = tiled_surface_bytes(
                                r->width, r->height, r->tile_mode, 0, bpt);
                            const size_t linear_slice = static_cast<size_t>(r->width) * r->height * bpt;
                            for (uint32_t layer = 0; layer < r->depth; ++layer)
                                detile_surface(linear.get() + linear_slice * layer,
                                               src + tiled_slice * layer,
                                               r->width, r->height, r->tile_mode, 0, bpt);
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
                            sampled_depth) {                            // Native sampled texels
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

            // Device-local image.
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = dim_1d ? VK_IMAGE_TYPE_1D : (dim_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D);
            ici.format = image_format;
            ici.extent = {r->width, r->height, r->depth};
            ici.mipLevels = 1;
            if (dim_2d_array) ici.extent.depth = 1;
            ici.arrayLayers = dim_2d_array ? r->depth : 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = (bi.storage ? (VK_IMAGE_USAGE_STORAGE_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                                    : VK_IMAGE_USAGE_SAMPLED_BIT) |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            // An imported binding already holds the renderer's image; only the view/sampler below
            // are ours to create. `ici` is still filled above so the view matches its format/layers.
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
            if (!vk_ok(vkCreateImageView(ctx.device, &vci, nullptr, &bi.view),
                       "image-view")) { images_ready = false; break; }
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
                if (!vk_ok(vkCreateSampler(ctx.device, &smci, nullptr, &bi.sampler), "image-sampler")) {
                    images_ready = false; break; }
            }
            if (image_timing)
                std::fprintf(stderr,
                             "[compute-image] code=0x%llx binding=%u class=%s imported=%u "
                             "extent=%ux%ux%u guest=%zu staging=%llu ms=%.3f\n",
                             (unsigned long long)item.code_addr, bi.binding,
                             bi.storage ? "storage" : "sampled", bi.imported ? 1u : 0u,
                             r->width, r->height, r->depth, bi.guest_bytes,
                             (unsigned long long)sbytes,
                             std::chrono::duration<double, std::milli>(
                                 ComputeClock::now() - image_start).count());
        }
        if (!images_ready) break;
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
            buffer_infos[i] = {buffers[i].buffer, 0, buffers[i].resource->size};
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
            }
            cpci.layout = pipeline_layout;
            if (trace) std::fprintf(stderr, "[compute]   creating compute pipeline\n");
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

        if (!ctx.prepare_dispatch_commands()) {
            if (trace) std::fprintf(stderr, "[compute]   Vulkan failure stage=command-reuse\n");
            break;
        }
        const VkCommandBuffer command = ctx.command_buffer;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!vk_ok(vkBeginCommandBuffer(command, &begin), "command-begin")) break;
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
            if (bi.persistent && bi.upload_skipped) {
                // The previous synchronous dispatch left this read-only sampled image in GENERAL,
                // and the write watch proved its complete guest source unchanged. No transfer or
                // layout transition is needed.
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
                to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                                     1, &to_general);
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
            const bool array_depth = r->img_dim == 5 && r->depth_compare;
            const VkImageAspectFlags aspect = r->depth_compare
                ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            to_dst.subresourceRange = {aspect, 0, 1, 0, array_depth ? r->depth : 1u};
            vkCmdPipelineBarrier(command,
                                 bi.persistent ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                               : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
            VkBufferImageCopy region{};
            region.imageSubresource = {aspect, 0, 0, array_depth ? r->depth : 1u};
            region.imageExtent = {r->width, r->height, array_depth ? 1u : r->depth};
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
        // Hand every borrowed renderer image back in the layout its owner left it in (#1095), so the
        // renderer's own layout tracking stays true whether or not a dispatch consumed the target.
        for (size_t i = 0; i < images.size(); i++) {
            const BoundImage& bi = images[i];
            if (!bi.imported || bi.alias_of != SIZE_MAX || !imported_barrier_owner(i)) continue;
            VkImageMemoryBarrier restore{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            restore.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
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
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &restore);
        }
        // Storage images: copy the written texels back into the staging buffer for guest writeback.
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
            to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_src);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {r->width, r->height, r->depth};
            vkCmdCopyImageToBuffer(command, bi.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging[i], 1, &region);
            if (bi.cache_candidate || bi.persistent) {
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
            compute_submit_rc = vkQueueSubmit(ctx.queue, 1, &submit, ctx.dispatch_fence);
        }
        if (!vk_ok(compute_submit_rc, "queue-submit")) break;
        if (trace) std::fprintf(stderr, "[compute]   waiting for dispatch\n");
        if (!vk_ok(vkWaitForFences(ctx.device, 1, &ctx.dispatch_fence, VK_TRUE,
                                   30ull * 1000 * 1000 * 1000), "queue-wait")) {
            // cleanup() destroys the command buffer and releases the borrowed image's pin. With a
            // renderer-owned image bound, in-flight work still references it and its layout
            // transitions, so drain the queue first rather than freeing it underneath the GPU.
            // readback_persistent_color_target uses the same drain but PROMOTES a successful drain
            // to success; this item stays failed either way, which is the conservative choice for a
            // dispatch whose results we can no longer trust.
            std::unique_lock<std::mutex> qlk(prosper::gpu::shared_present_submit_mutex(), std::defer_lock);
            if (prosper::gpu::shared_present_active()) qlk.lock();
            if (vkQueueWaitIdle(ctx.queue) != VK_SUCCESS && trace)
                std::fprintf(stderr, "[compute]   queue drain after fence timeout failed\n");
            break;
        }
        if (trace) std::fprintf(stderr, "[compute]   dispatch complete\n");
        phase_dispatch = ComputeClock::now();

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
                if (!image.upload_skipped)
                    ctx.validate_cached_image_source(image.cache_key);
            } else if (image.image && image.memory && image.allocation_bytes &&
                       ctx.retain_image(image.cache_key, image.image, image.memory,
                                        image.allocation_bytes,
                                        image.cache_source_snapshot.empty()
                                            ? nullptr : image.cache_source_snapshot.data())) {
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

        bool readback_ok = true;
        for (auto& buffer : buffers) {
            if (buffer.alias_of != SIZE_MAX || !buffer.writable) continue;
            void* mapped = nullptr;
            if (ctx.map_memory(buffer.memory, 0, buffer.resource->size, &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            uint8_t* destination = resource_bytes(buffer.resource);
            const auto* result = static_cast<const uint8_t*>(mapped);
            const bool changed = !compute_buffers_equal(
                destination, result, buffer.resource->size);
            if (trace) {
                buffer.after_hash = fnv1a(result, buffer.resource->size);
                for (uint32_t i = 0; i < buffer.resource->size; i++)
                    buffer.changed_bytes += destination[i] != result[i];
            }
            // Synchronous Unity maintenance kernels commonly rewrite a large persistent buffer with
            // the values it already contains. Terminator 2D's startup kernel binds 8,847,360 bytes;
            // after its first dispatch all later readbacks are identical. Avoiding the redundant host
            // write removes one full pass over both source and destination. Cache invalidation and
            // writer provenance remain unconditional: renderer-resident state can differ from guest
            // RAM even when consecutive compute readbacks contain identical bytes.
            if (changed) {
                if (!buffer.resource->host_data && buffer.resource->gpu_addr)
                    prosper::host::guest_write_watch_notify_host_write(
                        buffer.resource->gpu_addr, buffer.resource->size);
                copy_compute_buffer(destination, result, buffer.resource->size);
            }
            ctx.unmap_memory(buffer.memory);
            if (buffer.resource->gpu_addr)
                notify_guest_gpu_write(buffer.resource->gpu_addr, buffer.resource->size);
            if (buffer.persistent)
                ctx.validate_cached_buffer_source(buffer.cache_key);
            if (!buffer.resource->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   buffer.resource->gpu_addr, buffer.resource->size,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
        }
        if (!readback_ok) break;
        // Storage-image writeback (#590): pack the kernel's raw uvec4 channel texels back into the
        // guest surface's real format, then restore its linear or 3D tiled address layout and notify
        // the render side exactly like the buffer path.
        for (size_t i = 0; i < images.size() && readback_ok; i++) {
            BoundImage& bi = images[i];
            if (!bi.storage || bi.alias_of != SIZE_MAX || bi.imported) continue;
            const ShaderResource* r = bi.resource;
            void* mapped = nullptr;
            if (ctx.map_memory(staging_memory[i], 0, staging_bytes[i], &mapped) != VK_SUCCESS) {
                readback_ok = false;
                break;
            }
            const uint32_t cb = data_format_bytes(r->format);
            const uint32_t nc = r->num_components ? r->num_components : 1;
            const size_t guest_texel = (r->format == DataFormat::Float10_11_11 ||
                                        r->format == DataFormat::Unorm2_10_10_10)
                ? 4u : (size_t)cb * nc;
            const size_t texels = (size_t)r->width * r->height * r->depth;
            const size_t linear_bytes = texels * guest_texel;
            uint8_t* destination = resource_bytes_for(r, bi.guest_bytes);
            if (!r->host_data && r->gpu_addr)
                prosper::host::guest_write_watch_notify_host_write(
                    r->gpu_addr, bi.guest_bytes);
            std::unique_ptr<uint8_t[]> linear;
            uint8_t* packed = destination;
            if (r->tile_mode) {
                linear = std::make_unique_for_overwrite<uint8_t[]>(linear_bytes);
                packed = linear.get();
            }
            const uint32_t* channels = static_cast<const uint32_t*>(mapped);
            const uint8_t* native_texels = static_cast<const uint8_t*>(mapped);
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
                    if (bi.native_float_storage) {
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
                if (bi.native_float_storage) {
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
            if (bi.native_float_storage) {
                // The typed Vulkan image has already applied the PS5 descriptor's UNORM/float
                // conversion. Its transfer bytes are the guest's exact row-major texels.
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
            if (verify_pack && !bi.native_float_storage) {
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
            if (r->tile_mode && r->depth > 1) {
                if (!tile_volume(destination, bi.guest_bytes, packed, r->width, r->height,
                                 r->depth, r->tile_mode, static_cast<uint32_t>(guest_texel))) {
                    readback_ok = false;
                    ctx.unmap_memory(staging_memory[i]);
                    break;
                }
            } else if (r->tile_mode && r->in_mip_tail) {
                tile_surface_level(destination, bi.guest_bytes, packed,
                                   r->width, r->height, r->tile_mode,
                                   static_cast<uint32_t>(guest_texel),
                                   r->mip_tail_x, r->mip_tail_y);
            } else if (r->tile_mode) {
                tile_surface(destination, packed, r->width, r->height, r->tile_mode, 0,
                             static_cast<uint32_t>(guest_texel));
            }
            const auto layout_done = ComputeClock::now();
            pack_ms += std::chrono::duration<double, std::milli>(pack_done - pack_start).count();
            layout_ms += std::chrono::duration<double, std::milli>(layout_done - pack_done).count();
            if (trace) bi.after_hash = fnv1a(destination, bi.guest_bytes);
            notify_guest_gpu_write(r->gpu_addr, bi.guest_bytes);
            if (!r->host_data && writer_provenance_enabled())
                record_guest_write(GuestWriterKind::ComputeBuffer,
                                   r->gpu_addr, bi.guest_bytes,
                                   item.submit_no, item.dispatch_index,
                                   item.command_order, item.code_addr);
            if (bi.dcc_metadata && bi.dcc_metadata_bytes) {
                std::memset(bi.dcc_metadata, 0xff, bi.dcc_metadata_bytes);
                notify_guest_gpu_write(r->metadata_addr, bi.dcc_metadata_bytes);
                if (!r->dcc_metadata_host_data && writer_provenance_enabled())
                    record_guest_write(GuestWriterKind::ComputeBuffer,
                                       r->metadata_addr, bi.dcc_metadata_bytes,
                                       item.submit_no, item.dispatch_index,
                                       item.command_order, item.code_addr);
                if (trace)
                    std::fprintf(stderr,
                                 "[compute]   DCC uncompressed binding=%u meta=0x%llx bytes=%zu code=0xff\n",
                                 bi.binding, (unsigned long long)r->metadata_addr,
                                 bi.dcc_metadata_bytes);
            }
            if (bi.cache_candidate) {
                if (bi.persistent) {
                    // Guest bytes now mirror the retained image again; discard the self-write
                    // notification and arm the next external-write check from this exact state.
                    ctx.validate_cached_image_source(bi.cache_key, destination);
                } else if (bi.image && bi.memory && bi.allocation_bytes &&
                           ctx.retain_image(bi.cache_key, bi.image, bi.memory,
                                            bi.allocation_bytes, destination)) {
                    bi.persistent = true;
                    if (trace)
                        std::fprintf(stderr,
                                     "[compute]   retained storage image binding=%u "
                                     "addr=0x%llx allocation=%llu\n",
                                     bi.binding, (unsigned long long)r->gpu_addr,
                                     (unsigned long long)bi.allocation_bytes);
                }
            }
            ctx.unmap_memory(staging_memory[i]);
        }
        if (!readback_ok) break;
        ok = true;
        phase_writeback = ComputeClock::now();
    } while (false);

    if (trace)
        std::fprintf(stderr, "[compute] execute submit=%llu dispatch=%llu code=0x%llx "
                     "threads=%ux%ux%u local=%ux%ux%u groups=%ux%ux%u "
                     "buffers=%zu images=%zu result=%s\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.dispatch_index,
                     (unsigned long long)item.code_addr, item.launch.threads_x,
                     item.launch.threads_y, item.launch.threads_z, item.launch.local_x,
                     item.launch.local_y, item.launch.local_z, item.launch.groups_x,
                     item.launch.groups_y, item.launch.groups_z, buffers.size(), images.size(),
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
    cleanup();
    if (phase_timing) {
        const auto phase_cleanup = ComputeClock::now();
        auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::fprintf(stderr,
                     "[compute-phase] submit=%llu code=0x%llx ok=%u "
                     "setup_ms=%.2f setup_validate_ms=%.2f setup_buffers_ms=%.2f "
                     "pipeline_ms=%.2f dispatch_ms=%.2f "
                     "writeback_ms=%.2f pack_ms=%.2f layout_ms=%.2f "
                     "cleanup_ms=%.2f total_ms=%.2f subgroup=%u\n",
                     (unsigned long long)item.submit_no, (unsigned long long)item.code_addr,
                     ok ? 1u : 0u, milliseconds(phase_start, phase_setup),
                     setup_validate_ms, setup_buffers_ms,
                     milliseconds(phase_setup, phase_pipeline),
                     milliseconds(phase_pipeline, phase_dispatch),
                     milliseconds(phase_dispatch, phase_writeback),
                     pack_ms, layout_ms,
                     milliseconds(phase_writeback, phase_cleanup),
                     milliseconds(phase_start, phase_cleanup), item.required_subgroup_size);
    }
    return ok;
}

} // namespace

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
    // Keep the Vulkan device alive across dispatch spans. Constructing an instance, device, and queue for
    // every callback cost roughly 25 ms/frame on the native Windows frontend before any kernel work ran.
    // Function-local static initialization is thread-safe; AGC submit execution serializes subsequent use.
    static VulkanComputeContext context;
    static const bool context_ready = context.init();
    if (!context_ready) {
        std::fprintf(stderr, "[compute] Vulkan initialization failed\n");
        return false;
    }
    const bool timing_enabled = std::getenv("PROSPER_RENDER_TIMING") != nullptr;
    const auto timing_start = timing_enabled
        ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Dispatches are independent PM4-order operations: one item failing (e.g. an image shape the
    // backend can't bind yet, #590) must not abort the rest of the batch — that would regress
    // dispatches that executed before image bindings existed. Run all; report all-succeeded.
    bool all_ok = true;
    for (const auto& item : items)
        all_ok &= execute_item(context, item);
    if (timing_enabled) {
        struct TimingTotals { uint64_t calls = 0, dispatches = 0; double milliseconds = 0; };
        static TimingTotals totals;
        static TimingTotals window;
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timing_start).count();
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
            const ComputeMemoryPoolStats pool = context.memory_pool_stats();
            std::fprintf(stderr,
                         "[render-timing] compute_memory_pool hits=%llu misses=%llu cached=%zu "
                         "%.1f MiB discarded=%llu\n",
                         (unsigned long long)pool.hits, (unsigned long long)pool.misses,
                         pool.cached_allocations,
                         static_cast<double>(pool.cached_bytes) / (1024.0 * 1024.0),
                         (unsigned long long)pool.discarded);
            std::fprintf(stderr,
                         "[render-window] compute calls=%llu dispatches=%.1f avg_ms=%.2f\n",
                         (unsigned long long)window.calls,
                         window.dispatches / static_cast<double>(window.calls),
                         window.milliseconds / static_cast<double>(window.calls));
            window = {};
        }
    }
    return all_ok;
}

void register_live_compute() {
    static bool attempted = false;
    if (attempted) return;
    attempted = true;
    const char* enabled = std::getenv("PROSPER_COMPUTE");
    if (enabled && (!std::strcmp(enabled, "0") || !std::strcmp(enabled, "off"))) {
        std::fprintf(stderr, "[compute] live execution disabled by PROSPER_COMPUTE=%s\n", enabled);
        return;
    }
    prosper::gpu::set_submit_compute(execute_live_compute_items);
    std::fprintf(stderr, "[compute] live Vulkan compute backend registered\n");
}

} // namespace prosper::frontend

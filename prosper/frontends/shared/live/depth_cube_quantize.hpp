// Numeric D32_FLOAT -> RGBA8 staging for renderer-owned depth-cube faces.
// The Vulkan readback is mapped bytes, not an array of C++ float objects.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_DEPTH_CUBE_AVX2 1
#else
#define PROSPER_DEPTH_CUBE_AVX2 0
#endif

namespace prosper::frontend {

inline void quantize_depth_cube_rgba8_scalar(uint8_t* dst, const uint8_t* source,
                                              size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        float depth;
        std::memcpy(&depth, source + i * sizeof(depth), sizeof(depth));
        depth = depth < 0.0f ? 0.0f : (depth > 1.0f ? 1.0f : depth);
        const uint8_t q = static_cast<uint8_t>(depth * 255.0f + 0.5f);
        dst[i * 4 + 0] = q;
        dst[i * 4 + 1] = q;
        dst[i * 4 + 2] = q;
        dst[i * 4 + 3] = 0xffu;
    }
}

#if PROSPER_DEPTH_CUBE_AVX2
inline bool depth_cube_avx2_available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}

// Only x86's little-endian byte order enters the packed-lane store. The scalar
// path above remains available on every architecture and as a same-binary control.
__attribute__((target("avx2")))
inline void quantize_depth_cube_rgba8_avx2(uint8_t* dst, const uint8_t* source,
                                            size_t pixels) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256i replicate = _mm256_set1_epi32(0x010101);
    const __m256i opaque = _mm256_set1_epi32(0xff000000u);
    size_t i = 0;
    for (; i + 8 <= pixels; i += 8) {
        __m256 depth;
        std::memcpy(&depth, source + i * sizeof(float), sizeof(depth));
        // The original scalar conversion has no defined uint8_t result for NaN.
        // Keep that rare case on the original path instead of changing its behavior.
        if (_mm256_movemask_ps(_mm256_cmp_ps(depth, depth, _CMP_UNORD_Q))) {
            quantize_depth_cube_rgba8_scalar(dst + i * 4, source + i * sizeof(float), 8);
            continue;
        }
        depth = _mm256_max_ps(zero, _mm256_min_ps(depth, one));
        const __m256i q = _mm256_cvttps_epi32(
            _mm256_add_ps(_mm256_mul_ps(depth, scale), half));
        const __m256i pixel = _mm256_or_si256(_mm256_mullo_epi32(q, replicate), opaque);
        std::memcpy(dst + i * 4, &pixel, sizeof(pixel));
    }
    quantize_depth_cube_rgba8_scalar(dst + i * 4, source + i * sizeof(float), pixels - i);
}
#else
inline bool depth_cube_avx2_available() { return false; }
#endif

inline bool depth_cube_simd_enabled() {
    return depth_cube_avx2_available() && !std::getenv("PROSPER_NO_DEPTH_CUBE_SIMD");
}

inline void quantize_depth_cube_rgba8(uint8_t* dst, const uint8_t* source, size_t pixels) {
#if PROSPER_DEPTH_CUBE_AVX2
    if (depth_cube_simd_enabled()) {
        quantize_depth_cube_rgba8_avx2(dst, source, pixels);
        return;
    }
#endif
    quantize_depth_cube_rgba8_scalar(dst, source, pixels);
}

} // namespace prosper::frontend

#undef PROSPER_DEPTH_CUBE_AVX2

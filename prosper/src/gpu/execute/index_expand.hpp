#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_INDEX_EXPAND_AVX2 1
#else
#define PROSPER_INDEX_EXPAND_AVX2 0
#endif

namespace prosper::gpu {

// dst is a separately allocated host vector; src is the guest's validated 16-bit index range.
// Read exactly count indices so a valid range ending at a guest mapping boundary stays valid.
inline uint32_t copy_indices_u16_max_scalar(uint32_t* dst, const uint16_t* src, size_t count) {
    uint32_t maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t index = src[i];
        dst[i] = index;
        maximum = std::max(maximum, index);
    }
    return maximum;
}

#if PROSPER_INDEX_EXPAND_AVX2
inline bool index_expand_avx2_available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}

__attribute__((target("avx2")))
inline uint32_t copy_indices_u16_max_avx2(uint32_t* dst, const uint16_t* src, size_t count) {
    __m256i vector_max = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i packed;
        std::memcpy(&packed, src + i, sizeof(packed));
        const __m256i widened = _mm256_cvtepu16_epi32(packed);
        std::memcpy(dst + i, &widened, sizeof(widened));
        vector_max = _mm256_max_epu32(vector_max, widened);
    }
    alignas(32) uint32_t lanes[8];
    std::memcpy(lanes, &vector_max, sizeof(lanes));
    uint32_t maximum = 0;
    for (uint32_t lane : lanes) maximum = std::max(maximum, lane);
    for (; i < count; ++i) {
        const uint32_t index = src[i];
        dst[i] = index;
        maximum = std::max(maximum, index);
    }
    return maximum;
}
#else
inline bool index_expand_avx2_available() { return false; }
#endif

inline uint32_t copy_indices_u16_max(uint32_t* dst, const uint16_t* src, size_t count) {
#if PROSPER_INDEX_EXPAND_AVX2
    // `0` is a same-binary scalar control. Unset and `1` admit AVX2 only on capable hosts.
    static const bool use_avx2 = [] {
        const char* setting = std::getenv("PROSPER_INDEX_EXPAND_SIMD");
        return index_expand_avx2_available() &&
               !(setting && setting[0] == '0' && setting[1] == '\0');
    }();
    if (use_avx2 && count >= 8) return copy_indices_u16_max_avx2(dst, src, count);
#endif
    return copy_indices_u16_max_scalar(dst, src, count);
}

} // namespace prosper::gpu

#undef PROSPER_INDEX_EXPAND_AVX2

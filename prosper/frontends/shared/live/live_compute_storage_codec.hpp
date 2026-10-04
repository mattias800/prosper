#pragma once

// Live compute's storage texel codec: guest storage-image texels <-> the RGBA32 words compute
// operates on (unpack/pack ranges and single texels, plus the AVX2/F16C fast paths). Lifted out
// of live_compute.cpp's anonymous namespace by promote_internal.py; internal to live compute --
// include it only from shared/live compute code.

#include "shared/live/live_compute.hpp"
#include "shared/compute/compute_buffer_bytes.hpp"   // parallel_compute_texels
#include "gpu/resources/shader_resources.hpp"        // DataFormat, f10/f11/half conversions
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_HAVE_TARGET_F16C 1
#endif

namespace prosper::frontend {

// Declared here, DEFINED in the .cpp this header was lifted out of: other
// translation units link against those definitions, so they must not move.
uint8_t storage_pack_unorm8(uint32_t float_bits);
uint16_t storage_pack_unorm16(uint32_t float_bits);
uint32_t storage_unpack_float16_bits(uint16_t half_bits);
void storage_pack_unorm8_range(const uint32_t* channels, uint32_t components, size_t texels, uint8_t* packed);
void storage_unpack_float16x4_range(const uint8_t* rgba16f, size_t texels, uint32_t* channels);
void storage_pack_float16x4_range(const uint32_t* channels, size_t texels, uint8_t* rgba16f);


#if defined(PROSPER_HAVE_TARGET_F16C)
// MinGW's out-of-line AVX target-function call may spill the by-value YMM argument with VMOVAPS to
// a worker thread's only 16-byte-aligned stack. Keep this leaf in the caller so no cross-function
// YMM spill exists; the generated arithmetic and the runtime AVX2 gate remain unchanged.
inline __attribute__((target("avx2"), always_inline))
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

inline __attribute__((target("avx2")))
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

inline __attribute__((target("avx2,f16c")))
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

inline __attribute__((target("avx2,f16c")))
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

inline __attribute__((target("avx2,f16c")))
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
inline bool storage_unpack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Unorm16 || f == DF::Snorm8 || f == DF::Snorm16 ||
           f == DF::Float16 || f == DF::Float32 || f == DF::Uint32 ||
           f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
inline bool storage_pack_supported(prosper::gpu::DataFormat f) {
    using DF = prosper::gpu::DataFormat;
    return f == DF::Unorm8 || f == DF::Unorm16 || f == DF::Snorm8 || f == DF::Snorm16 ||
           f == DF::Float16 || f == DF::Float32 ||
           f == DF::Uint32 || f == DF::Sint32 || f == DF::Float10_11_11 ||
           f == DF::Uint8 || f == DF::Sint8 || f == DF::Uint16 || f == DF::Sint16 ||
           f == DF::Unorm2_10_10_10;
}
inline void storage_unpack_texel(const uint8_t* src, prosper::gpu::DataFormat f, uint32_t ncomp, uint32_t out[4]) {
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

inline void storage_unpack_range(const uint8_t* src, size_t src_stride, prosper::gpu::DataFormat f,
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
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) { float v = p[c] / 255.0f; std::memcpy(&o[c], &v, 4); }
                    }
                });
            return;
        case DF::Unorm16:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) {
                            const uint16_t raw = static_cast<uint16_t>(p[c * 2] |
                                (static_cast<uint16_t>(p[c * 2 + 1]) << 8));
                            const float v = raw / 65535.0f; std::memcpy(&o[c], &v, 4);
                        }
                    }
                });
            return;
        case DF::Snorm8:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) {
                            const float v = std::max(static_cast<int8_t>(p[c]) / 127.0f, -1.0f);
                            std::memcpy(&o[c], &v, 4);
                        }
                    }
                });
            return;
        case DF::Snorm16:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) {
                            const int16_t raw = static_cast<int16_t>(p[c * 2] |
                                (static_cast<uint16_t>(p[c * 2 + 1]) << 8));
                            const float v = std::max(raw / 32767.0f, -1.0f);
                            std::memcpy(&o[c], &v, 4);
                        }
                    }
                });
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
        case DF::Float10_11_11:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        uint32_t packed = 0; std::memcpy(&packed, src + t * src_stride, sizeof(packed));
                        const float values[3] = {
                            prosper::gpu::f11_to_float(static_cast<uint16_t>(packed)),
                            prosper::gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
                            prosper::gpu::f10_to_float(static_cast<uint16_t>(packed >> 22))
                        };
                        uint32_t* o = out + t * 4;
                        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&o[c], &values[c], sizeof(values[c]));
                        o[3] = one_f32;
                    }
                });
            return;
        case DF::Unorm2_10_10_10:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        uint32_t packed = 0; std::memcpy(&packed, src + t * src_stride, sizeof(packed));
                        const float values[4] = {
                            ((packed >>  0) & 0x3ffu) / 1023.0f,
                            ((packed >> 10) & 0x3ffu) / 1023.0f,
                            ((packed >> 20) & 0x3ffu) / 1023.0f,
                            ((packed >> 30) & 0x3u)   / 3.0f
                        };
                        uint32_t* o = out + t * 4;
                        for (uint32_t c = 0; c < 4; ++c) std::memcpy(&o[c], &values[c], sizeof(values[c]));
                    }
                });
            return;
        case DF::Uint8:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) o[c] = p[c];
                    }
                });
            return;
        case DF::Sint8:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c)
                            o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(p[c])));
                    }
                });
            return;
        case DF::Uint16:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c)
                            o[c] = static_cast<uint32_t>(p[c * 2] | (p[c * 2 + 1] << 8));
                    }
                });
            return;
        case DF::Sint16:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c)
                            o[c] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(
                                p[c * 2] | (p[c * 2 + 1] << 8))));
                    }
                });
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint8_t* p = src + t * src_stride; uint32_t* o = out + t * 4; defaults(o);
                        for (uint32_t c = 0; c < n; ++c) std::memcpy(&o[c], p + c * 4, 4);
                    }
                });
            return;
        default:                                  // packed formats keep the general per-texel path
            parallel_compute_texels(count, count * (src_stride + sizeof(uint32_t) * 4),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t)
                        storage_unpack_texel(src + t * src_stride, f, ncomp, out + t * 4);
                });
            return;
    }
}
// Range-specialized pack (#1101): the writeback mirror of storage_unpack_range (#1092). Hoists the
// per-texel format dispatch out of the loop with per-format inner loops; packed formats keep the
// general per-texel path. Semantics are IDENTICAL to storage_pack_texel over the range -- asserted
// format-by-format by test_storage_pack_range.
void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst);
inline void storage_pack_range(const uint32_t* channels, prosper::gpu::DataFormat f, uint32_t ncomp,
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
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) {
                            const uint16_t raw = storage_pack_unorm16(in[c]);
                            p[c * 2] = static_cast<uint8_t>(raw);
                            p[c * 2 + 1] = static_cast<uint8_t>(raw >> 8);
                        }
                    }
                });
            return;
        case DF::Snorm8:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c)
                            p[c] = static_cast<uint8_t>(storage_pack_snorm<int8_t>(in[c], 127));
                    }
                });
            return;
        case DF::Snorm16:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) {
                            const uint16_t raw = static_cast<uint16_t>(
                                storage_pack_snorm<int16_t>(in[c], 32767));
                            p[c * 2] = static_cast<uint8_t>(raw);
                            p[c * 2 + 1] = static_cast<uint8_t>(raw >> 8);
                        }
                    }
                });
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
        case DF::Float10_11_11:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4;
                        float values[3];
                        for (uint32_t c = 0; c < 3; ++c) std::memcpy(&values[c], &in[c], sizeof(values[c]));
                        const uint32_t packed = static_cast<uint32_t>(prosper::gpu::float_to_f11(values[0])) |
                                                (static_cast<uint32_t>(prosper::gpu::float_to_f11(values[1])) << 11) |
                                                (static_cast<uint32_t>(prosper::gpu::float_to_f10(values[2])) << 22);
                        std::memcpy(dst + t * dst_stride, &packed, sizeof(packed));
                    }
                });
            return;
        case DF::Unorm2_10_10_10:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    auto q = [](const uint32_t bits, float scale) -> uint32_t {
                        float v; std::memcpy(&v, &bits, 4);
                        v = !(v > 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);
                        return static_cast<uint32_t>(v * scale + 0.5f);
                    };
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4;
                        const uint32_t packed = (q(in[0], 1023.0f) & 0x3ffu)        |
                                                ((q(in[1], 1023.0f) & 0x3ffu) << 10) |
                                                ((q(in[2], 1023.0f) & 0x3ffu) << 20) |
                                                ((q(in[3], 3.0f)    & 0x3u)   << 30);
                        std::memcpy(dst + t * dst_stride, &packed, sizeof(packed));
                    }
                });
            return;
        case DF::Uint8: case DF::Sint8:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) p[c] = static_cast<uint8_t>(in[c]);
                    }
                });
            return;
        case DF::Uint16: case DF::Sint16:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) {
                            p[c * 2] = static_cast<uint8_t>(in[c]);
                            p[c * 2 + 1] = static_cast<uint8_t>(in[c] >> 8);
                        }
                    }
                });
            return;
        case DF::Float32: case DF::Uint32: case DF::Sint32:
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t) {
                        const uint32_t* in = channels + t * 4; uint8_t* p = dst + t * dst_stride;
                        for (uint32_t c = 0; c < n; ++c) std::memcpy(p + c * 4, &in[c], 4);
                    }
                });
            return;
        default:                                  // packed formats keep the general per-texel path
            parallel_compute_texels(count, count * (sizeof(uint32_t) * 4 + dst_stride),
                [&](size_t begin, size_t end) {
                    for (size_t t = begin; t < end; ++t)
                        storage_pack_texel(channels + t * 4, f, ncomp, dst + t * dst_stride);
                });
            return;
    }
}
inline void storage_pack_texel(const uint32_t in[4], prosper::gpu::DataFormat f, uint32_t ncomp, uint8_t* dst) {
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

}  // namespace prosper::frontend

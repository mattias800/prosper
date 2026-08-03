#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::frontend {

// Explain why a texture reached the CPU decode path after the persistent-cache lookup. Keep this
// classification separate from the logging code: a diagnostic that labels a supported DCC source
// as generically "compressed", or an exact-content rejection as a cold miss, sends performance
// work toward the wrong subsystem.
enum class TextureDecodeMissReason : uint8_t {
    LiveRenderTarget,
    UnsupportedCandidate,
    UnsupportedCompression,
    EmptySource,
    CacheDisabled,
    CacheLimitZero,
    ContentInvalidated,
    ColdOrEvicted,
    Other,
};

constexpr TextureDecodeMissReason texture_decode_miss_reason(bool has_live_render_target,
                                                              bool decode_candidate,
                                                              bool compression_supported,
                                                              size_t source_size,
                                                              bool cache_disabled,
                                                              size_t cache_limit,
                                                              bool matching_cache_entry,
                                                              bool cache_eligible) {
    if (has_live_render_target) return TextureDecodeMissReason::LiveRenderTarget;
    if (!decode_candidate) return TextureDecodeMissReason::UnsupportedCandidate;
    if (!compression_supported) return TextureDecodeMissReason::UnsupportedCompression;
    if (!source_size) return TextureDecodeMissReason::EmptySource;
    if (cache_disabled) return TextureDecodeMissReason::CacheDisabled;
    if (!cache_limit) return TextureDecodeMissReason::CacheLimitZero;
    if (matching_cache_entry) return TextureDecodeMissReason::ContentInvalidated;
    if (cache_eligible) return TextureDecodeMissReason::ColdOrEvicted;
    return TextureDecodeMissReason::Other;
}

// PROSPER_DETILE_STATS must expose rare expensive repeats without becoming an unbounded per-frame
// trace. The global stream reports its initial events plus periodic summaries. A CPU-expensive
// block decode (currently BC6H) additionally reports the first repeats for each guest address and
// then powers of two, so a once-per-second miss remains visible even after a long boot sequence.
constexpr bool should_report_texture_decode_miss(uint64_t global_ordinal,
                                                 uint64_t address_ordinal,
                                                 bool expensive_block_decode) {
    if (!global_ordinal || !address_ordinal) return false;
    if (global_ordinal <= 16u || (global_ordinal % 3000u) == 0u) return true;
    if (!expensive_block_decode) return false;
    return address_ordinal <= 8u ||
        (address_ordinal & (address_ordinal - 1u)) == 0u;
}

// A successful persistent hit is normally too fast for PROSPER_RENDER_TIMING=detail's 0.5 ms
// resource threshold. Give PROSPER_DETILE_STATS one identity-specific witness without turning every
// cache reuse into a trace: at most the first hit for each of the first 64 expensive BC cube
// identities is printed. Both ordinals are included in the record so a capped stream cannot be
// mistaken for a population count.
constexpr bool should_report_texture_decode_hit(uint64_t global_ordinal,
                                                uint64_t address_ordinal,
                                                bool expensive_block_cube) {
    return expensive_block_cube && global_ordinal >= 1u && global_ordinal <= 64u &&
        address_ordinal == 1u;
}

// An unsupported candidate has no validated persistent source size by construction. Do not let
// that zero hide the exact expensive event this diagnostic exists to expose: a separately derived
// descriptor footprint is still an observable source-size signal. The threshold is diagnostic-only;
// it never changes cache eligibility or decode behavior.
constexpr bool texture_decode_miss_is_expensive_block(bool expensive_codec,
                                                       size_t persistent_source_size,
                                                       size_t fallback_source_size) {
    constexpr size_t kDiagnosticThreshold = 64u * 1024u;
    return expensive_codec &&
        (persistent_source_size >= kDiagnosticThreshold ||
         fallback_source_size >= kDiagnosticThreshold);
}

} // namespace prosper::frontend

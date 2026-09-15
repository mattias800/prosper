#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace prosper::frontend {

// Diagnostic populations, never content authority. Each acquisition belongs to one bucket;
// the query precedes rearming, while disabled/active flags reflect the state at validation.
// ExactFailure also includes unavailable/short sources and extent mismatches, not only unequal bytes.
enum class TextureValidationOutcome { Match, ExactFailure, WatchOnlyRefusal, Count };
enum class TextureValidationWatch {
    ControlDisabled, BelowMinimum, DisabledAfterDirty, Unknown, Dirty, Unchanged, NotQueried, Count
};

struct TextureValidationBucket {
    uint64_t calls = 0;
    uint64_t source_bytes = 0;
    // The existing safe_equal/copy helper's reported extent. Not physical bytes read: on Linux
    // a failing compare does not count its last chunk; the Windows fast path counts the request.
    uint64_t reported_validated_bytes = 0;
    // Acquisition observations, not unique entries or watches.
    uint64_t active_after = 0; // After any rearm attempt.
    std::array<uint64_t, 4> stability{}; // Before this validation updates stability.
    double validation_ms = 0;
};

class TextureValidationCensus {
public:
    static constexpr size_t kBands = 3;
    static constexpr size_t kWatches = static_cast<size_t>(TextureValidationWatch::Count);
    static constexpr size_t kOutcomes = static_cast<size_t>(TextureValidationOutcome::Count);
    static constexpr size_t band(size_t bytes) {
        return bytes < 1024 * 1024 ? 0 : bytes < 8 * 1024 * 1024 ? 1 : 2;
    }

    void record(TextureValidationOutcome outcome, TextureValidationWatch watch,
                size_t source_bytes, size_t reported_bytes, double milliseconds,
                bool active_after, uint32_t stability) {
        auto& b = buckets_[index(outcome, watch, band(source_bytes))];
        ++b.calls;
        b.source_bytes += source_bytes;
        b.reported_validated_bytes += reported_bytes;
        b.validation_ms += milliseconds;
        b.active_after += active_after;
        ++b.stability[stability < 3 ? stability : 3];
        ++calls_;
    }
    void reset() { *this = {}; }
    uint64_t calls() const { return calls_; }
    const TextureValidationBucket& bucket(TextureValidationOutcome outcome,
                                          TextureValidationWatch watch, size_t size_band) const {
        return buckets_[index(outcome, watch, size_band)];
    }

    void report(FILE* stream, unsigned long thread, const char* reason) const {
        static constexpr const char* outcomes[] = {"match", "exact-failure", "watch-only-refusal"};
        static constexpr const char* watches[] = {
            "control-disabled", "below-minimum", "disabled-after-dirty", "unknown",
            "dirty", "unchanged", "not-queried"};
        static constexpr const char* bands[] = {"below-1MiB", "1-to-8MiB", "at-least-8MiB"};
        std::fprintf(stream, "[texture-validation-census] scope=thread-cumulative thread=%lu "
                     "reason=%s calls=%llu\n", thread, reason, (unsigned long long)calls_);
        for (size_t o = 0; o < kOutcomes; ++o)
            for (size_t w = 0; w < kWatches; ++w)
                for (size_t s = 0; s < kBands; ++s) {
                    const auto& b = buckets_[(o * kWatches + w) * kBands + s];
                    if (!b.calls) continue;
                    std::fprintf(stream,
                        "[texture-validation-bucket] thread=%lu cumulative_calls=%llu outcome=%s watch=%s "
                        "size=%s calls=%llu source_bytes=%llu reported_validated_bytes=%llu "
                        "validation_ms=%.6f active_after=%llu stable0=%llu stable1=%llu "
                        "stable2=%llu stable3plus=%llu\n",
                        thread, (unsigned long long)calls_, outcomes[o], watches[w], bands[s],
                        (unsigned long long)b.calls, (unsigned long long)b.source_bytes,
                        (unsigned long long)b.reported_validated_bytes, b.validation_ms,
                        (unsigned long long)b.active_after, (unsigned long long)b.stability[0],
                        (unsigned long long)b.stability[1], (unsigned long long)b.stability[2],
                        (unsigned long long)b.stability[3]);
                }
    }

private:
    static constexpr size_t index(TextureValidationOutcome outcome,
                                  TextureValidationWatch watch, size_t size_band) {
        return (static_cast<size_t>(outcome) * kWatches + static_cast<size_t>(watch)) * kBands + size_band;
    }
    std::array<TextureValidationBucket, kOutcomes * kWatches * kBands> buckets_{};
    uint64_t calls_ = 0;
};

} // namespace prosper::frontend

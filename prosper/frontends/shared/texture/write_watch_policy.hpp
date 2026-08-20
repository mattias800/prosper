#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace prosper::frontend {

// Page-protection watches avoid repeatedly comparing a hot texture, but arming a very large cold
// texture is more expensive than its first few exact validations. Promote large sources only after
// repeated unchanged reuse; smaller sources still arm immediately.
inline bool should_promote_write_watch(size_t source_bytes,
                                       uint32_t stable_exact_validations,
                                       size_t defer_min_bytes,
                                       uint32_t promotion_validations) {
    return defer_min_bytes == 0 || source_bytes < defer_min_bytes ||
           stable_exact_validations >= promotion_validations;
}

inline uint32_t update_write_watch_stability(uint32_t stable_exact_validations,
                                             bool content_unchanged,
                                             uint32_t promotion_validations) {
    if (!content_unchanged) return 0;
    if (stable_exact_validations >= promotion_validations) return promotion_validations;
    return stable_exact_validations + 1;
}

class WriteWatchPromotionBudget {
public:
    void reset(size_t byte_limit) {
        byte_limit_ = byte_limit;
        remaining_ = byte_limit;
        promoted_ = false;
    }

    bool try_consume(size_t source_bytes) {
        if (!byte_limit_) return true;
        if (source_bytes <= remaining_) {
            remaining_ -= source_bytes;
            promoted_ = true;
            return true;
        }
        // A source larger than the whole budget must not starve forever. Let the first promotion in a
        // submit exceed the byte limit, then stop; this still bounds the burst to one large watch.
        if (!promoted_) {
            remaining_ = 0;
            promoted_ = true;
            return true;
        }
        return false;
    }

private:
    size_t byte_limit_ = 0;
    size_t remaining_ = 0;
    bool promoted_ = false;
};

} // namespace prosper::frontend

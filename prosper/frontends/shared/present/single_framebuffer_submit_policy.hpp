#pragma once

#include "shared/present/selected_source_identity.hpp"

#include <cstddef>
#include <limits>
#include <utility>

namespace prosper::frontend {

// Internal graphics spans can finish before a dispatch/DMA tail. Keep their completed CPU pixels
// until the submit's actual final callback, without borrowing a prior submit's frame or source.
class SingleFramebufferSubmitFrame {
public:
    void begin_span(bool first, uint64_t source_submit, uint32_t width, uint32_t height) {
        if (first) {
            reset();
            const uint64_t area = static_cast<uint64_t>(width) * height;
            if (!width || !height || area > std::numeric_limits<size_t>::max() / 4u) return;
            active_ = true;
            source_submit_ = source_submit;
            width_ = width;
            height_ = height;
            bytes_ = static_cast<size_t>(area) * 4u;
        } else if (source_submit != source_submit_ || width != width_ || height != height_) {
            reset();
        }
    }

    PixelSourceCandidate select(bool final, bool has_draws, PixelSourceCandidate current) {
        if (active_ && has_draws && current.pixels && current.pixels->size() == bytes_ &&
            current.source_submit == source_submit_)
            retained_ = std::move(current);
        if (!final) return {};
        PixelSourceCandidate selected = std::move(retained_);
        reset();
        return selected;
    }

    void reset() {
        retained_ = {};
        active_ = false;
        source_submit_ = 0;
        width_ = height_ = 0;
        bytes_ = 0;
    }

private:
    PixelSourceCandidate retained_;
    bool active_ = false;
    uint64_t source_submit_ = 0;
    uint32_t width_ = 0, height_ = 0;
    size_t bytes_ = 0;
};

} // namespace prosper::frontend

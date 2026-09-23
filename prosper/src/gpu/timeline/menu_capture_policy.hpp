#pragma once

#include "gpu/timeline/menu_frame_gate.hpp"

#include <cstdint>

namespace prosper::gpu {

enum class MenuCapturePhase : uint8_t { WaitingForMenu, Armed, Pending, Accepted, Refused };
enum class MenuCaptureRefusal : uint8_t {
    None, WaitExpired, CandidateExpired, CandidateLimit, ByteBudget, MissingPresentation,
    AmbiguousDraw, SameSubmitDependency, UnsupportedAttachment, CaptureFailed, WriteFailed,
};

struct MenuCaptureLimits {
    uint32_t max_candidates = 3;
    // Sum of guest blobs and renderer RTT seeds retained across all candidates. Guest ranges are
    // preflighted against this limit before copying. The existing RTT collector independently
    // limits one seed set to 1 GiB, so it may read an overlarge seed before this stricter aggregate
    // budget refuses; no capsule is installed in that case.
    uint64_t max_total_candidate_bytes = 512ull << 20;
    uint64_t max_wait_ms = 180000;
    uint64_t max_after_menu_ms = 30000;
};

// The live AGC submit counter starts at one; zero is the RenderedFrame unknown-source sentinel.
// Keep this predicate shared by the capsule write and policy acceptance paths.
constexpr bool menu_capture_exact_candidate(
    uint64_t execution_submit, uint64_t image_source_submit,
    uint64_t pending_submit, bool candidate_pending) {
    return image_source_submit != 0 && candidate_pending &&
           execution_submit == image_source_submit &&
           pending_submit == image_source_submit;
}

// A menu image may be a retained publication from an older callback. Inspect its draw state only
// when the same callback owns the source submit; otherwise retry for a bounded number of positive
// images and a bounded interval. This is a diagnostic census, not capture admission.
enum class MenuSourceCensusPhase : uint8_t {
    WaitingForMenu, WaitingForNewSource, Exact, AttemptLimit, WaitExpired,
};

struct MenuSourceCensusLimits {
    uint32_t max_positive_images = 8;
    uint64_t max_wait_ms = 30000;
};

class MenuSourceCensusPolicy {
public:
    explicit MenuSourceCensusPolicy(MenuSourceCensusLimits limits = {}) : limits_(limits) {}

    MenuSourceCensusPhase phase() const { return phase_; }
    uint32_t attempts() const { return attempts_; }

    void tick(uint64_t now_ms) {
        if (phase_ != MenuSourceCensusPhase::WaitingForNewSource) return;
        if (now_ms < started_ms_ || now_ms - started_ms_ > limits_.max_wait_ms)
            phase_ = MenuSourceCensusPhase::WaitExpired;
    }

    bool observe(uint64_t execution_submit, uint64_t source_submit,
                 bool menu_positive, uint64_t now_ms) {
        tick(now_ms);
        if (!menu_positive || phase_ == MenuSourceCensusPhase::Exact ||
            phase_ == MenuSourceCensusPhase::AttemptLimit ||
            phase_ == MenuSourceCensusPhase::WaitExpired) return false;
        if (phase_ == MenuSourceCensusPhase::WaitingForMenu) {
            phase_ = MenuSourceCensusPhase::WaitingForNewSource;
            started_ms_ = now_ms;
        }
        ++attempts_;
        if (source_submit != 0 && source_submit == execution_submit) {
            phase_ = MenuSourceCensusPhase::Exact;
            return true;
        }
        if (attempts_ >= limits_.max_positive_images)
            phase_ = MenuSourceCensusPhase::AttemptLimit;
        return false;
    }

private:
    MenuSourceCensusLimits limits_;
    MenuSourceCensusPhase phase_ = MenuSourceCensusPhase::WaitingForMenu;
    uint32_t attempts_ = 0;
    uint64_t started_ms_ = 0;
};

// This policy never invents a relationship between a menu frame and a candidate. The first
// positive publication only arms future work; the candidate is accepted solely by its own exact
// completed producer submit and a separately classified image published by that submit.
class MenuCapturePolicy {
public:
    explicit MenuCapturePolicy(MenuCaptureLimits limits, uint64_t started_ms)
        : limits_(limits), started_ms_(started_ms) {}

    MenuCapturePhase phase() const { return phase_; }
    MenuCaptureRefusal refusal() const { return refusal_; }
    uint32_t attempted_candidates() const { return attempted_; }
    uint64_t copied_candidate_bytes() const { return bytes_; }
    uint64_t pending_submit() const { return pending_submit_; }
    uint64_t remaining_bytes() const {
        return bytes_ <= limits_.max_total_candidate_bytes
            ? limits_.max_total_candidate_bytes - bytes_ : 0;
    }
    void abort(MenuCaptureRefusal reason) { refuse(reason); }

    void tick(uint64_t now_ms) {
        if (phase_ == MenuCapturePhase::Accepted || phase_ == MenuCapturePhase::Refused) return;
        if (now_ms < started_ms_ || now_ms - started_ms_ > limits_.max_wait_ms) {
            refuse(MenuCaptureRefusal::WaitExpired);
            return;
        }
        if (phase_ != MenuCapturePhase::WaitingForMenu &&
            (now_ms < armed_ms_ || now_ms - armed_ms_ > limits_.max_after_menu_ms))
            refuse(MenuCaptureRefusal::CandidateExpired);
    }

    // Call only after a real publication of `source_submit`; a skipped present has no pixels and
    // cannot arm or accept. `composited` excludes republished raw guest scanout.
    void published(uint64_t source_submit, bool source_submit_known,
                   MenuFrameGateVerdict frame, bool composited,
                   uint64_t now_ms) {
        tick(now_ms);
        if (phase_ == MenuCapturePhase::Refused || phase_ == MenuCapturePhase::Accepted) return;
        const bool menu = composited && source_submit_known &&
            (frame == MenuFrameGateVerdict::MenuWithLogo ||
             frame == MenuFrameGateVerdict::MenuWithoutLogo);
        if (phase_ == MenuCapturePhase::WaitingForMenu) {
            if (menu) { phase_ = MenuCapturePhase::Armed; armed_ms_ = now_ms; }
            return;
        }
        if (!source_submit_known || phase_ != MenuCapturePhase::Pending ||
            source_submit != pending_submit_) return;
        pending_submit_ = 0;
        if (menu) phase_ = MenuCapturePhase::Accepted;
        else if (attempted_ >= limits_.max_candidates)
            refuse(MenuCaptureRefusal::CandidateLimit);
        else phase_ = MenuCapturePhase::Armed;
    }

    bool begin_candidate(uint64_t submit, uint64_t copied_bytes, uint64_t now_ms) {
        tick(now_ms);
        if (phase_ != MenuCapturePhase::Armed) return false;
        if (attempted_ >= limits_.max_candidates) {
            refuse(MenuCaptureRefusal::CandidateLimit);
            return false;
        }
        if (copied_bytes > limits_.max_total_candidate_bytes - bytes_) {
            refuse(MenuCaptureRefusal::ByteBudget);
            return false;
        }
        ++attempted_;
        bytes_ += copied_bytes;
        pending_submit_ = submit;
        phase_ = MenuCapturePhase::Pending;
        return true;
    }

    // The renderer completed the selected submit but did not publish its result. Never certify a
    // later retained/repeated image under this candidate's identity.
    void missing_presentation(uint64_t submit, uint64_t now_ms) {
        tick(now_ms);
        if (phase_ != MenuCapturePhase::Pending || pending_submit_ != submit) return;
        pending_submit_ = 0;
        if (attempted_ >= limits_.max_candidates)
            refuse(MenuCaptureRefusal::MissingPresentation);
        else phase_ = MenuCapturePhase::Armed;
    }

private:
    void refuse(MenuCaptureRefusal reason) {
        refusal_ = reason;
        phase_ = MenuCapturePhase::Refused;
        pending_submit_ = 0;
    }

    MenuCaptureLimits limits_;
    uint64_t started_ms_ = 0;
    uint64_t armed_ms_ = 0;
    uint64_t pending_submit_ = 0;
    uint64_t bytes_ = 0;
    uint32_t attempted_ = 0;
    MenuCapturePhase phase_ = MenuCapturePhase::WaitingForMenu;
    MenuCaptureRefusal refusal_ = MenuCaptureRefusal::None;
};

} // namespace prosper::gpu

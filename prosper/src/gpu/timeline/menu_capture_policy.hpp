#pragma once

#include "gpu/timeline/menu_frame_gate.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace prosper::gpu {

enum class MenuCapturePhase : uint8_t { WaitingForMenu, Armed, Pending, Accepted, Refused };
enum class MenuCaptureRefusal : uint8_t {
    None, WaitExpired, CandidateExpired, CandidateLimit, ByteBudget, MissingPresentation,
    AmbiguousDraw, SameSubmitDependency, UnsupportedAttachment, MenuTargetUnresolved,
    CaptureFailed, WriteFailed,
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
    uint32_t max_distinct_sources = 8;
    uint64_t max_wait_ms = 30000;
};

class MenuSourceCensusPolicy {
public:
    explicit MenuSourceCensusPolicy(MenuSourceCensusLimits limits = {}) : limits_(limits) {}

    MenuSourceCensusPhase phase() const { return phase_; }
    uint32_t distinct_retained_sources() const { return seen_count_; }

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
        if (source_submit != 0 && source_submit == execution_submit) {
            phase_ = MenuSourceCensusPhase::Exact;
            return true;
        }
        // An unknown (zero) source can be delivered at presentation rate before a renderer-owned
        // image exists. Repeated delivery of one older producer also gives no new opportunity.
        // The wall deadline bounds both cases; the count bounds distinct known retained sources.
        if (source_submit == 0) return false;
        for (uint32_t i = 0; i < seen_count_; ++i)
            if (seen_sources_[i] == source_submit) return false;
        if (seen_count_ >= seen_sources_.size()) {
            phase_ = MenuSourceCensusPhase::AttemptLimit;
            return false;
        }
        seen_sources_[seen_count_++] = source_submit;
        if (seen_count_ >= limits_.max_distinct_sources)
            phase_ = MenuSourceCensusPhase::AttemptLimit;
        return false;
    }

private:
    MenuSourceCensusLimits limits_;
    MenuSourceCensusPhase phase_ = MenuSourceCensusPhase::WaitingForMenu;
    uint32_t seen_count_ = 0;
    uint64_t started_ms_ = 0;
    std::array<uint64_t, 8> seen_sources_{};
};

struct MenuWriterEvidence {
    uint64_t draw_index = 0;
    uint64_t ps_addr = 0, target_addr = 0;
    uint32_t width = 0, height = 0, format = 0, raw_write_mask = 0;
    bool realized_exact = false;
    uint32_t realized_write_mask = 0;

    constexpr bool semantic_write_candidate() const {
        return target_addr && format && raw_write_mask;
    }
    constexpr bool active_candidate() const {
        return semantic_write_candidate() && realized_exact &&
               (raw_write_mask & realized_write_mask);
    }
};

enum class MenuTargetReject : uint8_t {
    None, MissingScene, UnrealizedFinalScene, WrongSceneProgram, MissingScreen,
    UnrealizedLaterWriter, OtherWriterAfterScene, TooManyDraws,
};

struct MenuTargetChoice {
    uint64_t target_addr = 0, draw_index = 0;
    MenuTargetReject rejection = MenuTargetReject::MissingScene;
};

// Use the exact menu-positive source submit, not an address copied from another run. Inspect the
// last semantically write-enabled scene-size draw even when realization failed; otherwise an
// earlier admitted draw could be mislabeled as the final one. Only admitted screen-size writers may
// follow. This identifies a diagnostic candidate, not proof those pixels reached scanout.
inline MenuTargetChoice menu_capture_select_target(
    std::span<const MenuWriterEvidence> draws, uint64_t expected_ps,
    uint32_t scene_width, uint32_t scene_height,
    uint32_t screen_width, uint32_t screen_height) {
    const MenuWriterEvidence* last_scene = nullptr;
    for (const auto& draw : draws)
        if (draw.semantic_write_candidate() &&
            draw.width == scene_width && draw.height == scene_height)
            last_scene = &draw;
    if (!last_scene) return {};
    if (!last_scene->active_candidate())
        return {0, last_scene->draw_index, MenuTargetReject::UnrealizedFinalScene};
    if (last_scene->ps_addr != expected_ps)
        return {0, last_scene->draw_index, MenuTargetReject::WrongSceneProgram};
    bool screen_seen = false;
    for (const auto& draw : draws) {
        if (draw.draw_index <= last_scene->draw_index || !draw.semantic_write_candidate())
            continue;
        if (!draw.active_candidate())
            return {0, last_scene->draw_index, MenuTargetReject::UnrealizedLaterWriter};
        if (draw.width != screen_width || draw.height != screen_height)
            return {0, last_scene->draw_index, MenuTargetReject::OtherWriterAfterScene};
        screen_seen = true;
    }
    if (!screen_seen)
        return {0, last_scene->draw_index, MenuTargetReject::MissingScreen};
    return {last_scene->target_addr, last_scene->draw_index, MenuTargetReject::None};
}

constexpr bool menu_capture_target_matches_request(MenuTargetChoice choice,
                                                    uint64_t configured_target) {
    return choice.rejection == MenuTargetReject::None && choice.target_addr &&
           (!configured_target || configured_target == choice.target_addr);
}

// VA-only diagnostic relation for declared byte ranges. An empty, missing, or overflowing range is
// unknown, never disjoint. Physical aliases and indirect pointer dereferences remain outside this
// predicate; callers must not use a disjoint result to admit an incomplete replay capsule.
enum class MenuDeclaredRangeRelation : uint8_t { Overlap, Disjoint, Unknown };

constexpr MenuDeclaredRangeRelation menu_declared_range_relation(
    uint64_t left, uint64_t left_bytes, uint64_t right, uint64_t right_bytes) {
    if (!left || !right || !left_bytes || !right_bytes ||
        left_bytes > std::numeric_limits<uint64_t>::max() - left ||
        right_bytes > std::numeric_limits<uint64_t>::max() - right)
        return MenuDeclaredRangeRelation::Unknown;
    return left < right + right_bytes && right < left + left_bytes
        ? MenuDeclaredRangeRelation::Overlap : MenuDeclaredRangeRelation::Disjoint;
}

// The dependency census may call an effect "before" a draw only after the exact selected draw
// exists in the ordered submit. Both index and command order are needed: a reused index from a
// different packet is not the same operation.
template <typename Operation, typename Kind>
constexpr std::size_t menu_capture_selected_draw_position(
    std::span<const Operation> operations, Kind draw_kind,
    uint64_t draw_index, uint64_t command_order) {
    for (std::size_t i = 0; i < operations.size(); ++i)
        if (operations[i].kind == draw_kind && operations[i].index == draw_index &&
            operations[i].command_order == command_order)
            return i;
    return std::numeric_limits<std::size_t>::max();
}

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

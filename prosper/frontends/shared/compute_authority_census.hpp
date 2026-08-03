#pragma once

#include <cstdint>
#include <limits>

namespace prosper::frontend {

// Behavior-neutral model for a future deferred compute writeback. The live backend still publishes
// every storage result synchronously; this state machine answers only whether a hypothetical pending
// Vulkan result has reached an observer that would require the architectural guest mirror.
struct ShadowComputeAuthorityRange {
    uint64_t address = 0;
    uint64_t bytes = 0;
    bool known = false;

    static constexpr ShadowComputeAuthorityRange from(uint64_t address, uint64_t bytes) {
        const bool representable = bytes != 0 &&
            address <= std::numeric_limits<uint64_t>::max() - (bytes - 1);
        return {address, bytes, representable};
    }

    static constexpr ShadowComputeAuthorityRange unknown() {
        return {};
    }

    constexpr bool operator==(const ShadowComputeAuthorityRange&) const = default;
};

constexpr bool shadow_compute_authority_ranges_overlap(
    const ShadowComputeAuthorityRange& first,
    const ShadowComputeAuthorityRange& second) {
    if (!first.known || !second.known) return false;
    return first.address <= second.address
        ? second.address - first.address < first.bytes
        : first.address - second.address < second.bytes;
}

enum class ShadowComputeAuthorityConsumerKind : uint8_t {
    ProvenGpuImage,
    GuestImage,
    RawBuffer,
    Draw,
    Dma,
    OrderedMemoryEffect,
    Capture,
    Unknown,
    SubmitEnd,
};

enum class ShadowComputeAuthorityAccess : uint8_t {
    ProvenGpu,
    GuestMirror,
    AlwaysMaterialize,
};

constexpr ShadowComputeAuthorityAccess shadow_compute_authority_access(
    ShadowComputeAuthorityConsumerKind kind) {
    switch (kind) {
        case ShadowComputeAuthorityConsumerKind::ProvenGpuImage:
            return ShadowComputeAuthorityAccess::ProvenGpu;
        case ShadowComputeAuthorityConsumerKind::GuestImage:
            return ShadowComputeAuthorityAccess::GuestMirror;
        case ShadowComputeAuthorityConsumerKind::RawBuffer:
            return ShadowComputeAuthorityAccess::GuestMirror;
        case ShadowComputeAuthorityConsumerKind::Draw:
            return ShadowComputeAuthorityAccess::GuestMirror;
        case ShadowComputeAuthorityConsumerKind::Dma:
            return ShadowComputeAuthorityAccess::GuestMirror;
        case ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect:
            return ShadowComputeAuthorityAccess::GuestMirror;
        case ShadowComputeAuthorityConsumerKind::Capture:
        case ShadowComputeAuthorityConsumerKind::Unknown:
        case ShadowComputeAuthorityConsumerKind::SubmitEnd:
            return ShadowComputeAuthorityAccess::AlwaysMaterialize;
    }
    return ShadowComputeAuthorityAccess::AlwaysMaterialize;
}

enum class ShadowComputeAuthorityAction : uint8_t {
    NoPendingResult,
    TrackPendingResult,
    ReplacePendingResult,
    KeepGpuAuthority,
    MaterializeGuestMirror,
    MaterializeAndTrackResult,
    RejectResult,
};

enum class ShadowComputeAuthorityReason : uint8_t {
    NoPendingResult,
    ResultAdmitted,
    ResultReplaced,
    ProvenGpuConsumer,
    UnrelatedConsumer,
    OverlappingGuestConsumer,
    OverlappingRawBufferConsumer,
    UnknownConsumerRange,
    UnknownConsumer,
    SubmitEnd,
    ResultRangeChanged,
    InvalidResultRange,
};

struct ShadowComputeAuthorityTransition {
    ShadowComputeAuthorityAction action =
        ShadowComputeAuthorityAction::NoPendingResult;
    ShadowComputeAuthorityReason reason =
        ShadowComputeAuthorityReason::NoPendingResult;
    bool pending_before = false;
    bool pending_after = false;
};

struct ShadowComputeAuthorityState {
    ShadowComputeAuthorityRange range;
    bool vulkan_result_valid = false;
    bool guest_mirror_current = true;

    constexpr bool pending() const {
        return range.known && vulkan_result_valid && !guest_mirror_current;
    }
};

struct ShadowComputeAuthorityCounters {
    uint64_t result_candidates = 0;
    uint64_t admitted_results = 0;
    uint64_t replaced_results = 0;
    uint64_t rejected_results = 0;
    uint64_t consumer_observations = 0;
    uint64_t proven_gpu_keeps = 0;
    uint64_t unrelated_keeps = 0;
    uint64_t materializations = 0;
    uint64_t overlap_materializations = 0;
    uint64_t raw_buffer_materializations = 0;
    uint64_t unknown_materializations = 0;
    uint64_t submit_end_materializations = 0;
    uint64_t range_change_materializations = 0;
};

constexpr uint64_t shadow_compute_authority_increment(uint64_t value) {
    return value == std::numeric_limits<uint64_t>::max() ? value : value + 1;
}

class ShadowComputeAuthorityCensus {
public:
    constexpr const ShadowComputeAuthorityState& state() const { return state_; }
    constexpr const ShadowComputeAuthorityCounters& counters() const { return counters_; }

    constexpr ShadowComputeAuthorityTransition note_retained_result(
        const ShadowComputeAuthorityRange& range) {
        counters_.result_candidates =
            shadow_compute_authority_increment(counters_.result_candidates);
        const bool pending_before = state_.pending();
        if (!range.known) {
            counters_.rejected_results =
                shadow_compute_authority_increment(counters_.rejected_results);
            if (pending_before) {
                record_materialization(ShadowComputeAuthorityReason::InvalidResultRange);
                state_ = {};
                return {ShadowComputeAuthorityAction::MaterializeGuestMirror,
                        ShadowComputeAuthorityReason::InvalidResultRange, true, false};
            }
            return {ShadowComputeAuthorityAction::RejectResult,
                    ShadowComputeAuthorityReason::InvalidResultRange, false, false};
        }

        if (!pending_before) {
            state_ = {range, true, false};
            counters_.admitted_results =
                shadow_compute_authority_increment(counters_.admitted_results);
            return {ShadowComputeAuthorityAction::TrackPendingResult,
                    ShadowComputeAuthorityReason::ResultAdmitted, false, true};
        }

        if (state_.range == range) {
            counters_.replaced_results =
                shadow_compute_authority_increment(counters_.replaced_results);
            return {ShadowComputeAuthorityAction::ReplacePendingResult,
                    ShadowComputeAuthorityReason::ResultReplaced, true, true};
        }

        record_materialization(ShadowComputeAuthorityReason::ResultRangeChanged);
        state_ = {range, true, false};
        counters_.admitted_results =
            shadow_compute_authority_increment(counters_.admitted_results);
        return {ShadowComputeAuthorityAction::MaterializeAndTrackResult,
                ShadowComputeAuthorityReason::ResultRangeChanged, true, true};
    }

    constexpr ShadowComputeAuthorityTransition observe(
        ShadowComputeAuthorityConsumerKind kind,
        const ShadowComputeAuthorityRange& range =
            ShadowComputeAuthorityRange::unknown()) {
        counters_.consumer_observations =
            shadow_compute_authority_increment(counters_.consumer_observations);
        const bool pending_before = state_.pending();
        if (!pending_before)
            return {ShadowComputeAuthorityAction::NoPendingResult,
                    ShadowComputeAuthorityReason::NoPendingResult, false, false};

        if (kind == ShadowComputeAuthorityConsumerKind::SubmitEnd)
            return materialize(ShadowComputeAuthorityReason::SubmitEnd);

        const ShadowComputeAuthorityAccess access =
            shadow_compute_authority_access(kind);
        if (access == ShadowComputeAuthorityAccess::AlwaysMaterialize)
            return materialize(ShadowComputeAuthorityReason::UnknownConsumer);
        if (!range.known)
            return materialize(ShadowComputeAuthorityReason::UnknownConsumerRange);

        if (!shadow_compute_authority_ranges_overlap(state_.range, range)) {
            counters_.unrelated_keeps =
                shadow_compute_authority_increment(counters_.unrelated_keeps);
            return {ShadowComputeAuthorityAction::KeepGpuAuthority,
                    ShadowComputeAuthorityReason::UnrelatedConsumer, true, true};
        }

        if (access == ShadowComputeAuthorityAccess::ProvenGpu) {
            counters_.proven_gpu_keeps =
                shadow_compute_authority_increment(counters_.proven_gpu_keeps);
            return {ShadowComputeAuthorityAction::KeepGpuAuthority,
                    ShadowComputeAuthorityReason::ProvenGpuConsumer, true, true};
        }

        const ShadowComputeAuthorityReason reason =
            kind == ShadowComputeAuthorityConsumerKind::RawBuffer
                ? ShadowComputeAuthorityReason::OverlappingRawBufferConsumer
                : ShadowComputeAuthorityReason::OverlappingGuestConsumer;
        return materialize(reason);
    }

private:
    constexpr void record_materialization(ShadowComputeAuthorityReason reason) {
        counters_.materializations =
            shadow_compute_authority_increment(counters_.materializations);
        switch (reason) {
            case ShadowComputeAuthorityReason::OverlappingGuestConsumer:
                counters_.overlap_materializations = shadow_compute_authority_increment(
                    counters_.overlap_materializations);
                break;
            case ShadowComputeAuthorityReason::OverlappingRawBufferConsumer:
                counters_.overlap_materializations = shadow_compute_authority_increment(
                    counters_.overlap_materializations);
                counters_.raw_buffer_materializations = shadow_compute_authority_increment(
                    counters_.raw_buffer_materializations);
                break;
            case ShadowComputeAuthorityReason::UnknownConsumerRange:
            case ShadowComputeAuthorityReason::UnknownConsumer:
            case ShadowComputeAuthorityReason::InvalidResultRange:
                counters_.unknown_materializations = shadow_compute_authority_increment(
                    counters_.unknown_materializations);
                break;
            case ShadowComputeAuthorityReason::SubmitEnd:
                counters_.submit_end_materializations = shadow_compute_authority_increment(
                    counters_.submit_end_materializations);
                break;
            case ShadowComputeAuthorityReason::ResultRangeChanged:
                counters_.range_change_materializations = shadow_compute_authority_increment(
                    counters_.range_change_materializations);
                break;
            case ShadowComputeAuthorityReason::NoPendingResult:
            case ShadowComputeAuthorityReason::ResultAdmitted:
            case ShadowComputeAuthorityReason::ResultReplaced:
            case ShadowComputeAuthorityReason::ProvenGpuConsumer:
            case ShadowComputeAuthorityReason::UnrelatedConsumer:
                break;
        }
        state_.guest_mirror_current = true;
    }

    constexpr ShadowComputeAuthorityTransition materialize(
        ShadowComputeAuthorityReason reason) {
        record_materialization(reason);
        return {ShadowComputeAuthorityAction::MaterializeGuestMirror,
                reason, true, false};
    }

    ShadowComputeAuthorityState state_;
    ShadowComputeAuthorityCounters counters_;
};

} // namespace prosper::frontend

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

    constexpr bool valid() const {
        return known && bytes != 0 &&
               address <= std::numeric_limits<uint64_t>::max() - (bytes - 1);
    }

    constexpr bool operator==(const ShadowComputeAuthorityRange&) const = default;
};

constexpr bool shadow_compute_authority_ranges_overlap(
    const ShadowComputeAuthorityRange& first,
    const ShadowComputeAuthorityRange& second) {
    if (!first.valid() || !second.valid()) return false;
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
        default:
            return ShadowComputeAuthorityAccess::AlwaysMaterialize;
    }
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
    OverlappingGuestImageConsumer,
    OverlappingRawBufferConsumer,
    OverlappingDrawConsumer,
    OverlappingDmaConsumer,
    OverlappingOrderedMemoryEffectConsumer,
    CaptureConsumer,
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
        return range.valid() && vulkan_result_valid && !guest_mirror_current;
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
    uint64_t proven_gpu_image_materializations = 0;
    uint64_t guest_image_materializations = 0;
    uint64_t raw_buffer_materializations = 0;
    uint64_t draw_materializations = 0;
    uint64_t dma_materializations = 0;
    uint64_t ordered_memory_effect_materializations = 0;
    uint64_t capture_materializations = 0;
    uint64_t unknown_consumer_materializations = 0;
    uint64_t submit_end_materializations = 0;
    uint64_t unknown_range_materializations = 0;
    uint64_t invalid_result_materializations = 0;
    uint64_t range_change_materializations = 0;
};

constexpr uint64_t shadow_compute_authority_increment(uint64_t value) {
    return value == std::numeric_limits<uint64_t>::max() ? value : value + 1;
}

// Post-realization audit for the conservative unknown-range Draw boundary. The authority census
// still fails closed before draw realization; this companion only asks whether the subsequently
// realized shader-resource tables name the pending result that forced that materialization. Keeping
// the audit separate is intentional: an exact resource-table match is evidence about the draw, not
// yet proof that descriptor/index/shader reads made realization itself safe to cross.
enum class ShadowComputeAuthorityDrawProbeAction : uint8_t {
    Ignored,
    InvalidRange,
    UnrelatedRange,
    OverlappingRange,
    Completed,
};

struct ShadowComputeAuthorityDrawProbeCounters {
    uint64_t armed = 0;
    uint64_t superseded = 0;
    uint64_t resource_observations = 0;
    uint64_t invalid_ranges = 0;
    uint64_t unrelated_ranges = 0;
    uint64_t overlapping_ranges = 0;
    uint64_t completed = 0;
    uint64_t completed_with_overlap = 0;
    uint64_t completed_without_overlap = 0;
    uint64_t unrealized = 0;
};

class ShadowComputeAuthorityDrawProbe {
public:
    constexpr const ShadowComputeAuthorityDrawProbeCounters& counters() const {
        return counters_;
    }

    constexpr bool active() const { return active_; }
    constexpr bool saw_overlap() const { return saw_overlap_; }

    constexpr void arm(uint64_t submit_no, uint64_t command_order,
                       const ShadowComputeAuthorityRange& pending_range) {
        if (active_)
            counters_.superseded = shadow_compute_authority_increment(
                counters_.superseded);
        active_ = pending_range.valid();
        submit_no_ = submit_no;
        command_order_ = command_order;
        pending_range_ = active_ ? pending_range : ShadowComputeAuthorityRange::unknown();
        saw_overlap_ = false;
        if (active_)
            counters_.armed = shadow_compute_authority_increment(counters_.armed);
    }

    constexpr ShadowComputeAuthorityDrawProbeAction observe_resource(
            uint64_t submit_no, uint64_t command_order,
            const ShadowComputeAuthorityRange& resource_range) {
        if (!matches(submit_no, command_order))
            return ShadowComputeAuthorityDrawProbeAction::Ignored;
        counters_.resource_observations = shadow_compute_authority_increment(
            counters_.resource_observations);
        if (!resource_range.valid()) {
            counters_.invalid_ranges = shadow_compute_authority_increment(
                counters_.invalid_ranges);
            return ShadowComputeAuthorityDrawProbeAction::InvalidRange;
        }
        if (!shadow_compute_authority_ranges_overlap(pending_range_, resource_range)) {
            counters_.unrelated_ranges = shadow_compute_authority_increment(
                counters_.unrelated_ranges);
            return ShadowComputeAuthorityDrawProbeAction::UnrelatedRange;
        }
        saw_overlap_ = true;
        counters_.overlapping_ranges = shadow_compute_authority_increment(
            counters_.overlapping_ranges);
        return ShadowComputeAuthorityDrawProbeAction::OverlappingRange;
    }

    constexpr ShadowComputeAuthorityDrawProbeAction complete(
            uint64_t submit_no, uint64_t command_order, bool realized) {
        if (!matches(submit_no, command_order))
            return ShadowComputeAuthorityDrawProbeAction::Ignored;
        counters_.completed = shadow_compute_authority_increment(counters_.completed);
        if (!realized) {
            counters_.unrealized = shadow_compute_authority_increment(counters_.unrealized);
        } else if (saw_overlap_) {
            counters_.completed_with_overlap = shadow_compute_authority_increment(
                counters_.completed_with_overlap);
        } else {
            counters_.completed_without_overlap = shadow_compute_authority_increment(
                counters_.completed_without_overlap);
        }
        active_ = false;
        return ShadowComputeAuthorityDrawProbeAction::Completed;
    }

private:
    constexpr bool matches(uint64_t submit_no, uint64_t command_order) const {
        return active_ && submit_no == submit_no_ && command_order == command_order_;
    }

    ShadowComputeAuthorityDrawProbeCounters counters_;
    ShadowComputeAuthorityRange pending_range_;
    uint64_t submit_no_ = 0;
    uint64_t command_order_ = 0;
    bool active_ = false;
    bool saw_overlap_ = false;
};

class ShadowComputeAuthorityCensus {
public:
    constexpr const ShadowComputeAuthorityState& state() const { return state_; }
    constexpr const ShadowComputeAuthorityCounters& counters() const { return counters_; }

    constexpr ShadowComputeAuthorityTransition note_retained_result(
        const ShadowComputeAuthorityRange& range) {
        counters_.result_candidates =
            shadow_compute_authority_increment(counters_.result_candidates);
        const bool pending_before = state_.pending();
        if (!range.valid()) {
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
            return materialize_consumer(kind, ShadowComputeAuthorityReason::SubmitEnd);

        const ShadowComputeAuthorityAccess access =
            shadow_compute_authority_access(kind);
        if (access == ShadowComputeAuthorityAccess::AlwaysMaterialize) {
            const ShadowComputeAuthorityReason reason =
                kind == ShadowComputeAuthorityConsumerKind::Capture
                    ? ShadowComputeAuthorityReason::CaptureConsumer
                    : ShadowComputeAuthorityReason::UnknownConsumer;
            return materialize_consumer(kind, reason);
        }
        if (!range.valid())
            return materialize_consumer(
                kind, ShadowComputeAuthorityReason::UnknownConsumerRange);

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

        ShadowComputeAuthorityReason reason =
            ShadowComputeAuthorityReason::UnknownConsumer;
        switch (kind) {
            case ShadowComputeAuthorityConsumerKind::GuestImage:
                reason = ShadowComputeAuthorityReason::OverlappingGuestImageConsumer;
                break;
            case ShadowComputeAuthorityConsumerKind::RawBuffer:
                reason = ShadowComputeAuthorityReason::OverlappingRawBufferConsumer;
                break;
            case ShadowComputeAuthorityConsumerKind::Draw:
                reason = ShadowComputeAuthorityReason::OverlappingDrawConsumer;
                break;
            case ShadowComputeAuthorityConsumerKind::Dma:
                reason = ShadowComputeAuthorityReason::OverlappingDmaConsumer;
                break;
            case ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect:
                reason =
                    ShadowComputeAuthorityReason::OverlappingOrderedMemoryEffectConsumer;
                break;
            case ShadowComputeAuthorityConsumerKind::ProvenGpuImage:
            case ShadowComputeAuthorityConsumerKind::Capture:
            case ShadowComputeAuthorityConsumerKind::Unknown:
            case ShadowComputeAuthorityConsumerKind::SubmitEnd:
                break;
        }
        return materialize_consumer(kind, reason);
    }

private:
    constexpr void record_materialization(ShadowComputeAuthorityReason reason) {
        counters_.materializations =
            shadow_compute_authority_increment(counters_.materializations);
        switch (reason) {
            case ShadowComputeAuthorityReason::OverlappingGuestImageConsumer:
            case ShadowComputeAuthorityReason::OverlappingRawBufferConsumer:
            case ShadowComputeAuthorityReason::OverlappingDrawConsumer:
            case ShadowComputeAuthorityReason::OverlappingDmaConsumer:
            case ShadowComputeAuthorityReason::OverlappingOrderedMemoryEffectConsumer:
                counters_.overlap_materializations = shadow_compute_authority_increment(
                    counters_.overlap_materializations);
                break;
            case ShadowComputeAuthorityReason::UnknownConsumerRange:
                counters_.unknown_range_materializations = shadow_compute_authority_increment(
                    counters_.unknown_range_materializations);
                break;
            case ShadowComputeAuthorityReason::InvalidResultRange:
                counters_.invalid_result_materializations =
                    shadow_compute_authority_increment(
                        counters_.invalid_result_materializations);
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
            case ShadowComputeAuthorityReason::CaptureConsumer:
            case ShadowComputeAuthorityReason::UnknownConsumer:
            case ShadowComputeAuthorityReason::SubmitEnd:
                break;
        }
        state_.guest_mirror_current = true;
    }

    constexpr ShadowComputeAuthorityTransition materialize_consumer(
        ShadowComputeAuthorityConsumerKind kind,
        ShadowComputeAuthorityReason reason) {
        record_materialization(reason);
        switch (kind) {
            case ShadowComputeAuthorityConsumerKind::GuestImage:
                counters_.guest_image_materializations = shadow_compute_authority_increment(
                    counters_.guest_image_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::RawBuffer:
                counters_.raw_buffer_materializations = shadow_compute_authority_increment(
                    counters_.raw_buffer_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::Draw:
                counters_.draw_materializations = shadow_compute_authority_increment(
                    counters_.draw_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::Dma:
                counters_.dma_materializations = shadow_compute_authority_increment(
                    counters_.dma_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect:
                counters_.ordered_memory_effect_materializations =
                    shadow_compute_authority_increment(
                        counters_.ordered_memory_effect_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::Capture:
                counters_.capture_materializations = shadow_compute_authority_increment(
                    counters_.capture_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::Unknown:
                counters_.unknown_consumer_materializations =
                    shadow_compute_authority_increment(
                        counters_.unknown_consumer_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::SubmitEnd:
                counters_.submit_end_materializations = shadow_compute_authority_increment(
                    counters_.submit_end_materializations);
                break;
            case ShadowComputeAuthorityConsumerKind::ProvenGpuImage:
                counters_.proven_gpu_image_materializations =
                    shadow_compute_authority_increment(
                        counters_.proven_gpu_image_materializations);
                break;
            default:
                counters_.unknown_consumer_materializations =
                    shadow_compute_authority_increment(
                        counters_.unknown_consumer_materializations);
                break;
        }
        return {ShadowComputeAuthorityAction::MaterializeGuestMirror,
                reason, true, false};
    }

    ShadowComputeAuthorityState state_;
    ShadowComputeAuthorityCounters counters_;
};

} // namespace prosper::frontend

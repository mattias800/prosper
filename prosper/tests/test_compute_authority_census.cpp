#include "compute_authority_census.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(condition, name) do { \
    if (condition) std::printf("  PASS %s\n", name); \
    else { std::printf("  FAIL %s\n", name); ++failures; } \
} while (0)

int main() {
    std::printf("shadow compute authority census policy\n");

    constexpr auto atlas = ShadowComputeAuthorityRange::from(0x200000, 0x10000);
    constexpr auto atlas_tail = ShadowComputeAuthorityRange::from(0x20f000, 0x2000);
    constexpr auto adjacent = ShadowComputeAuthorityRange::from(0x210000, 0x1000);
    constexpr auto unrelated = ShadowComputeAuthorityRange::from(0x400000, 0x1000);
    constexpr auto zero = ShadowComputeAuthorityRange::from(0x200000, 0);
    constexpr auto wrapped = ShadowComputeAuthorityRange::from(
        std::numeric_limits<uint64_t>::max() - 3, 8);
    CHECK(atlas.known && atlas_tail.known && adjacent.known && unrelated.known &&
              !zero.known && !wrapped.known &&
              shadow_compute_authority_ranges_overlap(atlas, atlas_tail) &&
              !shadow_compute_authority_ranges_overlap(atlas, adjacent) &&
              !shadow_compute_authority_ranges_overlap(
                  atlas, ShadowComputeAuthorityRange::unknown()),
          "half-open authority ranges reject zero/wrap and distinguish overlap from adjacency");

    ShadowComputeAuthorityCensus chain;
    const auto admitted = chain.note_retained_result(atlas);
    const auto first_gpu = chain.observe(
        ShadowComputeAuthorityConsumerKind::ProvenGpuImage, atlas);
    const auto replaced = chain.note_retained_result(atlas);
    const auto second_gpu = chain.observe(
        ShadowComputeAuthorityConsumerKind::ProvenGpuImage, atlas_tail);
    CHECK(admitted.action == ShadowComputeAuthorityAction::TrackPendingResult &&
              admitted.reason == ShadowComputeAuthorityReason::ResultAdmitted &&
              first_gpu.action == ShadowComputeAuthorityAction::KeepGpuAuthority &&
              first_gpu.reason == ShadowComputeAuthorityReason::ProvenGpuConsumer &&
              replaced.action == ShadowComputeAuthorityAction::ReplacePendingResult &&
              second_gpu.action == ShadowComputeAuthorityAction::KeepGpuAuthority &&
              chain.state().pending() && chain.state().vulkan_result_valid &&
              !chain.state().guest_mirror_current &&
              chain.counters().admitted_results == 1 &&
              chain.counters().replaced_results == 1 &&
              chain.counters().proven_gpu_keeps == 2 &&
              chain.counters().materializations == 0,
          "proven GPU consumers keep a same-range chain deferred");

    ShadowComputeAuthorityCensus disjoint;
    (void)disjoint.note_retained_result(atlas);
    const auto disjoint_read = disjoint.observe(
        ShadowComputeAuthorityConsumerKind::RawBuffer, unrelated);
    CHECK(disjoint_read.action == ShadowComputeAuthorityAction::KeepGpuAuthority &&
              disjoint_read.reason == ShadowComputeAuthorityReason::UnrelatedConsumer &&
              disjoint.state().pending() && disjoint.counters().unrelated_keeps == 1 &&
              disjoint.counters().materializations == 0,
          "unrelated guest reads leave pending authority intact");

    ShadowComputeAuthorityCensus raw_overlap;
    (void)raw_overlap.note_retained_result(atlas);
    const auto raw_read = raw_overlap.observe(
        ShadowComputeAuthorityConsumerKind::RawBuffer, atlas_tail);
    CHECK(raw_read.action == ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              raw_read.reason ==
                  ShadowComputeAuthorityReason::OverlappingRawBufferConsumer &&
              raw_read.pending_before && !raw_read.pending_after &&
              !raw_overlap.state().pending() &&
              raw_overlap.state().vulkan_result_valid &&
              raw_overlap.state().guest_mirror_current &&
              raw_overlap.counters().materializations == 1 &&
              raw_overlap.counters().overlap_materializations == 1 &&
              raw_overlap.counters().raw_buffer_materializations == 1,
          "overlapping raw-buffer consumer forces guest materialization");

    bool all_known_guest_observers_materialize = true;
    constexpr std::array guest_observers{
        ShadowComputeAuthorityConsumerKind::GuestImage,
        ShadowComputeAuthorityConsumerKind::Draw,
        ShadowComputeAuthorityConsumerKind::Dma,
        ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect,
    };
    for (const auto kind : guest_observers) {
        ShadowComputeAuthorityCensus observer;
        (void)observer.note_retained_result(atlas);
        const auto transition = observer.observe(kind, atlas_tail);
        all_known_guest_observers_materialize &=
            transition.action == ShadowComputeAuthorityAction::MaterializeGuestMirror &&
            transition.reason == ShadowComputeAuthorityReason::OverlappingGuestConsumer &&
            observer.counters().overlap_materializations == 1 &&
            observer.counters().materializations == 1 && !observer.state().pending();
    }
    CHECK(all_known_guest_observers_materialize,
          "known guest-image draw DMA and memory-effect overlaps fail closed");

    ShadowComputeAuthorityCensus unknown_kind;
    (void)unknown_kind.note_retained_result(atlas);
    const auto unknown_consumer = unknown_kind.observe(
        ShadowComputeAuthorityConsumerKind::Unknown, unrelated);
    ShadowComputeAuthorityCensus capture;
    (void)capture.note_retained_result(atlas);
    const auto capture_consumer = capture.observe(
        ShadowComputeAuthorityConsumerKind::Capture, unrelated);
    CHECK(unknown_consumer.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              unknown_consumer.reason == ShadowComputeAuthorityReason::UnknownConsumer &&
              capture_consumer.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              capture_consumer.reason == ShadowComputeAuthorityReason::UnknownConsumer &&
              unknown_kind.counters().unknown_materializations == 1 &&
              capture.counters().unknown_materializations == 1,
          "unknown and capture consumers materialize even with a disjoint claimed range");

    ShadowComputeAuthorityCensus unknown_range;
    (void)unknown_range.note_retained_result(atlas);
    const auto unbounded_draw = unknown_range.observe(
        ShadowComputeAuthorityConsumerKind::Draw,
        ShadowComputeAuthorityRange::unknown());
    CHECK(unbounded_draw.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              unbounded_draw.reason ==
                  ShadowComputeAuthorityReason::UnknownConsumerRange &&
              unknown_range.counters().unknown_materializations == 1,
          "a guest consumer without an exact range fails closed");

    ShadowComputeAuthorityCensus submit;
    (void)submit.note_retained_result(atlas);
    const auto submit_end = submit.observe(
        ShadowComputeAuthorityConsumerKind::SubmitEnd);
    const auto after_submit = submit.observe(
        ShadowComputeAuthorityConsumerKind::SubmitEnd);
    CHECK(submit_end.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              submit_end.reason == ShadowComputeAuthorityReason::SubmitEnd &&
              submit_end.pending_before && !submit_end.pending_after &&
              submit.counters().submit_end_materializations == 1 &&
              submit.counters().materializations == 1 &&
              after_submit.action == ShadowComputeAuthorityAction::NoPendingResult,
          "submit end materializes each pending result exactly once");

    ShadowComputeAuthorityCensus changed_range;
    (void)changed_range.note_retained_result(atlas);
    const auto changed = changed_range.note_retained_result(unrelated);
    CHECK(changed.action ==
                  ShadowComputeAuthorityAction::MaterializeAndTrackResult &&
              changed.reason == ShadowComputeAuthorityReason::ResultRangeChanged &&
              changed.pending_before && changed.pending_after &&
              changed_range.state().pending() &&
              changed_range.state().range == unrelated &&
              changed_range.counters().materializations == 1 &&
              changed_range.counters().range_change_materializations == 1 &&
              changed_range.counters().admitted_results == 2,
          "a changed result range flushes old authority before tracking the new result");

    ShadowComputeAuthorityCensus rejected;
    const auto invalid = rejected.note_retained_result(
        ShadowComputeAuthorityRange::unknown());
    CHECK(invalid.action == ShadowComputeAuthorityAction::RejectResult &&
              invalid.reason == ShadowComputeAuthorityReason::InvalidResultRange &&
              !rejected.state().pending() && rejected.counters().result_candidates == 1 &&
              rejected.counters().rejected_results == 1,
          "an invalid result range cannot create shadow authority");

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

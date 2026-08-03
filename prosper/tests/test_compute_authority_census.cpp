#include "compute_authority_census.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>

using namespace prosper::frontend;

static int failures = 0;
#define CHECK(condition, name) do { \
    if (condition) std::printf("  PASS %s\n", name); \
    else { std::printf("  FAIL %s\n", name); ++failures; } \
} while (0)

static uint64_t attributed_materializations(
    const ShadowComputeAuthorityCounters& counters) {
    return counters.proven_gpu_image_materializations +
           counters.guest_image_materializations +
           counters.raw_buffer_materializations +
           counters.draw_materializations +
           counters.dma_materializations +
           counters.ordered_memory_effect_materializations +
           counters.capture_materializations +
           counters.unknown_consumer_materializations +
           counters.submit_end_materializations;
}

int main() {
    std::printf("shadow compute authority census policy\n");

    constexpr auto atlas = ShadowComputeAuthorityRange::from(0x200000, 0x10000);
    constexpr auto atlas_tail = ShadowComputeAuthorityRange::from(0x20f000, 0x2000);
    constexpr auto adjacent = ShadowComputeAuthorityRange::from(0x210000, 0x1000);
    constexpr auto unrelated = ShadowComputeAuthorityRange::from(0x400000, 0x1000);
    constexpr auto zero = ShadowComputeAuthorityRange::from(0x200000, 0);
    constexpr auto wrapped = ShadowComputeAuthorityRange::from(
        std::numeric_limits<uint64_t>::max() - 3, 8);
    constexpr ShadowComputeAuthorityRange forged_wrapped{
        std::numeric_limits<uint64_t>::max() - 3, 8, true};
    CHECK(atlas.valid() && atlas_tail.valid() && adjacent.valid() &&
              unrelated.valid() && !zero.valid() && !wrapped.valid() &&
              shadow_compute_authority_ranges_overlap(atlas, atlas_tail) &&
              !shadow_compute_authority_ranges_overlap(atlas, adjacent) &&
              !shadow_compute_authority_ranges_overlap(
                  atlas, ShadowComputeAuthorityRange::unknown()),
          "half-open authority ranges reject zero/wrap and distinguish overlap from adjacency");
    CHECK(forged_wrapped.known && !forged_wrapped.valid() &&
              !shadow_compute_authority_ranges_overlap(atlas, forged_wrapped),
          "a caller-forged known overflowing range is still rejected");

    CHECK(shadow_compute_authority_increment(0) == 1 &&
              shadow_compute_authority_increment(
                  std::numeric_limits<uint64_t>::max() - 1) ==
                  std::numeric_limits<uint64_t>::max() &&
              shadow_compute_authority_increment(
                  std::numeric_limits<uint64_t>::max()) ==
                  std::numeric_limits<uint64_t>::max(),
          "authority counters saturate at UINT64_MAX instead of wrapping");

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
    const auto after_raw = raw_overlap.observe(
        ShadowComputeAuthorityConsumerKind::Draw, atlas_tail);
    CHECK(raw_read.action == ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              raw_read.reason ==
                  ShadowComputeAuthorityReason::OverlappingRawBufferConsumer &&
              raw_read.pending_before && !raw_read.pending_after &&
              !raw_overlap.state().pending() &&
              raw_overlap.state().vulkan_result_valid &&
              raw_overlap.state().guest_mirror_current &&
              raw_overlap.counters().materializations == 1 &&
              raw_overlap.counters().overlap_materializations == 1 &&
              raw_overlap.counters().raw_buffer_materializations == 1 &&
              raw_overlap.counters().draw_materializations == 0 &&
              attributed_materializations(raw_overlap.counters()) == 1 &&
              after_raw.action == ShadowComputeAuthorityAction::NoPendingResult,
          "overlapping raw-buffer consumer forces guest materialization");

    ShadowComputeAuthorityCensus guest_image;
    (void)guest_image.note_retained_result(atlas);
    const auto guest_image_read = guest_image.observe(
        ShadowComputeAuthorityConsumerKind::GuestImage, atlas_tail);
    CHECK(guest_image_read.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              guest_image_read.reason ==
                  ShadowComputeAuthorityReason::OverlappingGuestImageConsumer &&
              guest_image.counters().guest_image_materializations == 1 &&
              guest_image.counters().overlap_materializations == 1 &&
              guest_image.counters().materializations == 1 &&
              attributed_materializations(guest_image.counters()) == 1,
          "overlapping guest-image consumer is attributed exactly");

    ShadowComputeAuthorityCensus draw;
    (void)draw.note_retained_result(atlas);
    const auto draw_read = draw.observe(
        ShadowComputeAuthorityConsumerKind::Draw, atlas_tail);
    CHECK(draw_read.action == ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              draw_read.reason == ShadowComputeAuthorityReason::OverlappingDrawConsumer &&
              draw.counters().draw_materializations == 1 &&
              draw.counters().overlap_materializations == 1 &&
              draw.counters().materializations == 1 &&
              attributed_materializations(draw.counters()) == 1,
          "overlapping draw consumer is attributed exactly");

    ShadowComputeAuthorityCensus dma;
    (void)dma.note_retained_result(atlas);
    const auto dma_read = dma.observe(
        ShadowComputeAuthorityConsumerKind::Dma, atlas_tail);
    CHECK(dma_read.action == ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              dma_read.reason == ShadowComputeAuthorityReason::OverlappingDmaConsumer &&
              dma.counters().dma_materializations == 1 &&
              dma.counters().overlap_materializations == 1 &&
              dma.counters().materializations == 1 &&
              attributed_materializations(dma.counters()) == 1,
          "overlapping DMA consumer is attributed exactly");

    ShadowComputeAuthorityCensus memory_effect;
    (void)memory_effect.note_retained_result(atlas);
    const auto memory_effect_write = memory_effect.observe(
        ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect, atlas_tail);
    CHECK(memory_effect_write.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              memory_effect_write.reason ==
                  ShadowComputeAuthorityReason::OverlappingOrderedMemoryEffectConsumer &&
              memory_effect.counters().ordered_memory_effect_materializations == 1 &&
              memory_effect.counters().overlap_materializations == 1 &&
              memory_effect.counters().materializations == 1 &&
              attributed_materializations(memory_effect.counters()) == 1,
          "overlapping ordered memory effect is attributed exactly");

    ShadowComputeAuthorityCensus unknown_kind;
    (void)unknown_kind.note_retained_result(atlas);
    const auto unknown_consumer = unknown_kind.observe(
        ShadowComputeAuthorityConsumerKind::Unknown, unrelated);
    ShadowComputeAuthorityCensus capture;
    (void)capture.note_retained_result(atlas);
    const auto capture_consumer = capture.observe(
        ShadowComputeAuthorityConsumerKind::Capture, unrelated);
    CHECK(capture_consumer.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              capture_consumer.reason == ShadowComputeAuthorityReason::CaptureConsumer &&
              capture.counters().capture_materializations == 1 &&
              capture.counters().materializations == 1 &&
              attributed_materializations(capture.counters()) == 1,
          "capture consumer is attributed exactly even with a claimed disjoint range");
    CHECK(unknown_consumer.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              unknown_consumer.reason == ShadowComputeAuthorityReason::UnknownConsumer &&
              unknown_kind.counters().unknown_consumer_materializations == 1 &&
              unknown_kind.counters().materializations == 1 &&
              attributed_materializations(unknown_kind.counters()) == 1,
          "unknown consumer is attributed exactly even with a claimed disjoint range");

    ShadowComputeAuthorityCensus unknown_range;
    (void)unknown_range.note_retained_result(atlas);
    const auto unbounded_draw = unknown_range.observe(
        ShadowComputeAuthorityConsumerKind::Draw,
        ShadowComputeAuthorityRange::unknown());
    CHECK(unbounded_draw.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              unbounded_draw.reason ==
                  ShadowComputeAuthorityReason::UnknownConsumerRange &&
              unknown_range.counters().draw_materializations == 1 &&
              unknown_range.counters().unknown_range_materializations == 1,
          "an unbounded draw remains attributed while failing closed on its range");

    ShadowComputeAuthorityCensus unbounded_gpu;
    (void)unbounded_gpu.note_retained_result(atlas);
    const auto unknown_gpu_range = unbounded_gpu.observe(
        ShadowComputeAuthorityConsumerKind::ProvenGpuImage,
        ShadowComputeAuthorityRange::unknown());
    CHECK(unknown_gpu_range.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              unknown_gpu_range.reason ==
                  ShadowComputeAuthorityReason::UnknownConsumerRange &&
              unbounded_gpu.counters().proven_gpu_image_materializations == 1 &&
              unbounded_gpu.counters().unknown_range_materializations == 1 &&
              attributed_materializations(unbounded_gpu.counters()) == 1,
          "an unbounded proven-GPU consumer retains exact kind attribution");

    ShadowComputeAuthorityCensus invalid_kind;
    (void)invalid_kind.note_retained_result(atlas);
    const auto invalid_consumer = invalid_kind.observe(
        static_cast<ShadowComputeAuthorityConsumerKind>(0xff), unrelated);
    CHECK(invalid_consumer.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              invalid_consumer.reason == ShadowComputeAuthorityReason::UnknownConsumer &&
              invalid_kind.counters().unknown_consumer_materializations == 1 &&
              attributed_materializations(invalid_kind.counters()) == 1,
          "an invalid consumer enum is attributed as unknown and fails closed");

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
              attributed_materializations(submit.counters()) == 1 &&
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

    ShadowComputeAuthorityCensus rejected_pending;
    (void)rejected_pending.note_retained_result(atlas);
    const auto invalid_pending = rejected_pending.note_retained_result(forged_wrapped);
    CHECK(invalid_pending.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              invalid_pending.reason == ShadowComputeAuthorityReason::InvalidResultRange &&
              invalid_pending.pending_before && !invalid_pending.pending_after &&
              !rejected_pending.state().pending() &&
              rejected_pending.counters().result_candidates == 2 &&
              rejected_pending.counters().rejected_results == 1 &&
              rejected_pending.counters().invalid_result_materializations == 1 &&
              rejected_pending.counters().materializations == 1,
          "a forged invalid result flushes an existing pending authority");

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

#include "compute_authority_census.hpp"
#include "compute_authority_live_census.hpp"

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

    ComputeAuthorityImageSourceFacts alias_source;
    alias_source.binding = 5;
    alias_source.owner_binding = 4;
    alias_source.readable = true;
    alias_source.alias = true;
    alias_source.owner_transfer_borrowed = true;
    alias_source.owner_persistent = true;
    alias_source.same_image = true;
    alias_source.same_view = true;
    const ComputeAuthorityImageSourceDecision alias_decision =
        classify_compute_authority_image_source(alias_source);
    ComputeAuthorityImageSourceCounters alias_counters;
    record_compute_authority_image_source(
        alias_counters, alias_source, alias_decision);
    ShadowComputeAuthorityCensus alias_chain;
    (void)alias_chain.note_retained_result(atlas);
    const ShadowComputeAuthorityTransition alias_read = alias_chain.observe(
        alias_decision.consumer_kind(), atlas);
    CHECK(alias_decision.observe && alias_decision.proven_gpu &&
              alias_decision.consumer_kind() ==
                  ShadowComputeAuthorityConsumerKind::ProvenGpuImage &&
              alias_read.action == ShadowComputeAuthorityAction::KeepGpuAuthority &&
              alias_read.reason == ShadowComputeAuthorityReason::ProvenGpuConsumer &&
              alias_chain.state().pending() &&
              alias_counters.observations == 1 && alias_counters.aliases == 1 &&
              alias_counters.direct_transfer_borrows == 0 &&
              alias_counters.owner_transfer_borrows == 1 &&
              alias_counters.same_images == 1 && alias_counters.same_views == 1 &&
              alias_counters.proven_gpu == 1 && alias_counters.guest == 0,
          "readable exact image alias inherits its owner GPU authority proof");

    ComputeAuthorityImageSourceFacts unreadable_alias = alias_source;
    unreadable_alias.readable = false;
    const ComputeAuthorityImageSourceDecision unreadable_decision =
        classify_compute_authority_image_source(unreadable_alias);
    ComputeAuthorityImageSourceFacts mismatched_alias = alias_source;
    mismatched_alias.same_view = false;
    const ComputeAuthorityImageSourceDecision mismatch_decision =
        classify_compute_authority_image_source(mismatched_alias);
    CHECK(!unreadable_decision.observe && mismatch_decision.observe &&
              !mismatch_decision.proven_gpu &&
              mismatch_decision.consumer_kind() ==
                  ShadowComputeAuthorityConsumerKind::GuestImage,
          "image authority keeps reflection readability and exact source identity fail-closed");

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

    ShadowComputeAuthorityDrawProbe draw_probe;
    draw_probe.arm(77, 900, atlas);
    const auto wrong_draw_resource = draw_probe.observe_resource(77, 901, atlas_tail);
    const auto unrelated_draw_resource = draw_probe.observe_resource(77, 900, unrelated);
    const auto atlas_draw_resource = draw_probe.observe_resource(77, 900, atlas_tail);
    const auto completed_draw_probe = draw_probe.complete(77, 900, true);
    CHECK(wrong_draw_resource == ShadowComputeAuthorityDrawProbeAction::Ignored &&
              unrelated_draw_resource ==
                  ShadowComputeAuthorityDrawProbeAction::UnrelatedRange &&
              atlas_draw_resource ==
                  ShadowComputeAuthorityDrawProbeAction::OverlappingRange &&
              completed_draw_probe == ShadowComputeAuthorityDrawProbeAction::Completed &&
              !draw_probe.active() && draw_probe.counters().armed == 1 &&
              draw_probe.counters().resource_observations == 2 &&
              draw_probe.counters().unrelated_ranges == 1 &&
              draw_probe.counters().overlapping_ranges == 1 &&
              draw_probe.counters().completed == 1 &&
              draw_probe.counters().completed_with_overlap == 1 &&
              draw_probe.counters().completed_without_overlap == 0,
          "post-realization draw probe identifies an exact atlas alias after the conservative boundary");

    draw_probe.arm(78, 910, atlas);
    const auto invalid_draw_resource = draw_probe.observe_resource(
        78, 910, ShadowComputeAuthorityRange::unknown());
    const auto completed_unrelated_draw = draw_probe.complete(78, 910, true);
    draw_probe.arm(79, 920, atlas);
    const auto completed_unrealized_draw = draw_probe.complete(79, 920, false);
    CHECK(invalid_draw_resource ==
                  ShadowComputeAuthorityDrawProbeAction::InvalidRange &&
              completed_unrelated_draw ==
                  ShadowComputeAuthorityDrawProbeAction::Completed &&
              completed_unrealized_draw ==
                  ShadowComputeAuthorityDrawProbeAction::Completed &&
              draw_probe.counters().armed == 3 &&
              draw_probe.counters().invalid_ranges == 1 &&
              draw_probe.counters().completed == 3 &&
              draw_probe.counters().completed_without_overlap == 1 &&
              draw_probe.counters().unrealized == 1,
          "draw probe distinguishes no-overlap evidence from an unrealized draw");

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

    const ComputeAuthorityLiveSelector selector =
        parse_compute_authority_live_selector("0x45e5d145e1a35af9");
    ComputeAuthorityLiveCensus live(selector);
    live.begin_submit(41);
    const auto wrong_program = live.observe_program(0xee8584cf839a5b44ull);
    const auto producer = live.observe_program(0x45e5d145e1a35af9ull);
    const auto selected_output = live.record_selected_storage_output(atlas, true);
    const auto gpu_consumer = live.observe(
        ShadowComputeAuthorityConsumerKind::ProvenGpuImage, atlas);
    const auto capture_boundary = live.observe(
        ShadowComputeAuthorityConsumerKind::Capture);
    const auto clean_end = live.end_submit();
    CHECK(selector.requested && selector.valid &&
              !wrong_program.selected && producer.selected && producer.first_match &&
              selected_output.action ==
                  ShadowComputeAuthorityAction::TrackPendingResult &&
              gpu_consumer.action == ShadowComputeAuthorityAction::KeepGpuAuthority &&
              capture_boundary.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              clean_end.action == ShadowComputeAuthorityAction::NoPendingResult &&
              live.counters().programs_seen == 2 &&
              live.counters().programs_matched == 1 &&
              live.counters().selected_storage_outputs == 1 &&
              live.counters().retained_storage_outputs == 1 &&
              live.counters().authority.proven_gpu_keeps == 1 &&
              live.counters().authority.capture_materializations == 1 &&
              live.counters().pending_after_submit_end == 0 && live.summary_valid(),
          "live exact-hash chain closes at capture and ends with zero pending authority");

    ComputeAuthorityLiveCensus ordered(selector);
    ordered.begin_submit(42);
    (void)ordered.observe_program(selector.producer_hash);
    (void)ordered.record_selected_storage_output(atlas, true);
    const auto disjoint_dma_source = ordered.observe(
        ShadowComputeAuthorityConsumerKind::Dma, unrelated);
    const auto overlapping_dma_destination = ordered.observe(
        ShadowComputeAuthorityConsumerKind::Dma, atlas_tail);
    const auto ordered_end = ordered.end_submit();
    CHECK(disjoint_dma_source.action ==
                  ShadowComputeAuthorityAction::KeepGpuAuthority &&
              overlapping_dma_destination.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              overlapping_dma_destination.reason ==
                  ShadowComputeAuthorityReason::OverlappingDmaConsumer &&
              ordered_end.action == ShadowComputeAuthorityAction::NoPendingResult &&
              ordered.counters().authority.dma_materializations == 1 &&
              ordered.counters().authority.unrelated_keeps == 1 &&
              ordered.counters().pending_after_submit_end == 0 &&
              ordered.summary_valid(),
          "live DMA source and destination are observed separately at ordered positions");

    ComputeAuthorityLiveCensus submit_only(selector);
    submit_only.begin_submit(420);
    (void)submit_only.observe_program(selector.producer_hash);
    (void)submit_only.record_selected_storage_output(atlas, true);
    const auto mandatory_submit_end = submit_only.end_submit();
    CHECK(mandatory_submit_end.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              mandatory_submit_end.reason == ShadowComputeAuthorityReason::SubmitEnd &&
              submit_only.counters().authority.submit_end_materializations == 1 &&
              submit_only.counters().authority.capture_materializations == 0 &&
              submit_only.counters().pending_before_submit_end == 1 &&
              submit_only.counters().pending_after_submit_end == 0 &&
              submit_only.summary_valid(),
          "live submit end unconditionally closes pending authority with exact attribution");

    ComputeAuthorityLiveCensus unknown_draw(selector);
    unknown_draw.begin_submit(43);
    (void)unknown_draw.observe_program(selector.producer_hash);
    (void)unknown_draw.record_selected_storage_output(atlas, true);
    const auto draw_boundary = unknown_draw.observe(
        ShadowComputeAuthorityConsumerKind::Draw);
    (void)unknown_draw.end_submit();
    CHECK(draw_boundary.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              draw_boundary.reason ==
                  ShadowComputeAuthorityReason::UnknownConsumerRange &&
              unknown_draw.counters().authority.draw_materializations == 1 &&
              unknown_draw.counters().authority.unknown_range_materializations == 1,
          "live draw with no proven range fails closed at the exact boundary");

    ComputeAuthorityLiveCensus sync_output(selector);
    sync_output.begin_submit(44);
    (void)sync_output.observe_program(selector.producer_hash);
    (void)sync_output.record_selected_storage_output(atlas, true);
    const auto synchronous_replacement =
        sync_output.record_selected_storage_output(atlas_tail, false);
    (void)sync_output.end_submit();
    CHECK(synchronous_replacement.action ==
                  ShadowComputeAuthorityAction::MaterializeGuestMirror &&
              synchronous_replacement.reason ==
                  ShadowComputeAuthorityReason::OverlappingOrderedMemoryEffectConsumer &&
              sync_output.counters().selected_storage_outputs == 2 &&
              sync_output.counters().retained_storage_outputs == 1 &&
              sync_output.counters().synchronous_storage_outputs == 1,
          "a selected output without retained authority remains a visible synchronous write");

    ComputeAuthorityLiveCensus zero_match(selector);
    zero_match.begin_submit(45);
    (void)zero_match.observe_program(0xee8584cf839a5b44ull);
    (void)zero_match.end_submit();
    CHECK(!zero_match.summary_valid(),
          "a live authority run with zero producer matches is apparatus-invalid");

    ComputeAuthorityLiveCensus zero_output(selector);
    zero_output.begin_submit(451);
    (void)zero_output.observe_program(selector.producer_hash);
    (void)zero_output.end_submit();
    CHECK(zero_output.counters().programs_matched == 1 &&
              zero_output.counters().retained_storage_outputs == 0 &&
              zero_output.counters().authority.admitted_results == 0 &&
              !zero_output.summary_valid(),
          "a matched producer with no admitted retained result is apparatus-invalid");

    ComputeAuthorityLiveCensus invalid_selector(
        parse_compute_authority_live_selector("0xnot-a-hash"));
    invalid_selector.begin_submit(46);
    CHECK(!invalid_selector.observe_program(selector.producer_hash).selected,
          "an invalid live selector fails closed instead of selecting a producer");
    (void)invalid_selector.end_submit();
    CHECK(!invalid_selector.summary_valid(),
          "an invalid live selector cannot produce a valid census verdict");

    ComputeAuthorityLiveCensus interrupted(selector);
    interrupted.begin_submit(47);
    (void)interrupted.observe_program(selector.producer_hash);
    (void)interrupted.record_selected_storage_output(atlas, true);
    interrupted.begin_submit(48);
    (void)interrupted.end_submit();
    CHECK(interrupted.counters().interrupted_submits == 1 &&
              interrupted.counters().authority.unknown_consumer_materializations == 1 &&
              interrupted.counters().pending_after_submit_end == 0 &&
              !interrupted.summary_valid(),
          "a nested submit start closes pending authority and invalidates the apparatus");

    ComputeAuthorityLiveCensus no_submit(selector);
    (void)no_submit.observe(ShadowComputeAuthorityConsumerKind::Draw);
    CHECK(no_submit.counters().observations_without_submit == 1 &&
              !no_submit.summary_valid(),
          "a boundary outside a live submit is counted and fails closed");

    CHECK(live.claim_summary() && !live.claim_summary(),
          "explicit and destructor-style live summaries can publish only once");

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

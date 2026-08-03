#pragma once

#include "compute_authority_census.hpp"
#include "compute_timing_selector.hpp"

#include <cstdint>
#include <limits>

namespace prosper::frontend {

// Process-start selector for the live, behavior-neutral authority census.  The stable SPIR-V hash
// names only the producer whose retained storage results are admitted.  Every later compute input,
// write, draw, DMA, capture, and submit boundary is still observed while one result is pending.
struct ComputeAuthorityLiveSelector {
    bool requested = false;
    bool valid = false;
    uint64_t producer_hash = 0;
    ComputeTimingSelectorParseError error = ComputeTimingSelectorParseError::Unset;
};

inline ComputeAuthorityLiveSelector parse_compute_authority_live_selector(const char* text) {
    if (!text) return {};
    const ComputeTimingSelectorParseResult parsed =
        parse_compute_timing_selector_u64(text);
    return {true, parsed.accepted(), parsed.value, parsed.error};
}

struct ComputeAuthorityLiveObservation {
    bool selected = false;
    bool first_match = false;
};

struct ComputeAuthorityLiveCounters {
    uint64_t programs_seen = 0;
    uint64_t programs_matched = 0;
    uint64_t submits_started = 0;
    uint64_t submits_completed = 0;
    uint64_t interrupted_submits = 0;
    uint64_t observations_without_submit = 0;
    uint64_t selected_storage_outputs = 0;
    uint64_t retained_storage_outputs = 0;
    uint64_t synchronous_storage_outputs = 0;
    uint64_t pending_before_submit_end = 0;
    uint64_t pending_after_submit_end = 0;
    ShadowComputeAuthorityCounters authority;
    bool apparatus_valid = true;
    bool summary_reported = false;
};

// Source provenance for one reflected readable image binding.  Exact descriptor aliases bind the
// owner's concrete Vulkan image/view but do not repeat the owner's cache/transfer setup, so their
// per-binding flags are intentionally empty.  The shadow census must therefore distinguish direct
// provenance from the provenance of the source that is actually bound.
struct ComputeAuthorityImageSourceFacts {
    uint32_t binding = 0;
    uint32_t owner_binding = 0;
    bool readable = false;
    bool storage = false;
    bool alias = false;
    bool direct_transfer_borrowed = false;
    bool owner_transfer_borrowed = false;
    bool direct_persistent = false;
    bool owner_persistent = false;
    bool direct_upload_skipped = false;
    bool owner_upload_skipped = false;
    bool same_image = false;
    bool same_view = false;
};

struct ComputeAuthorityImageSourceDecision {
    bool observe = false;
    bool proven_gpu = false;

    constexpr ShadowComputeAuthorityConsumerKind consumer_kind() const {
        return proven_gpu ? ShadowComputeAuthorityConsumerKind::ProvenGpuImage
                          : ShadowComputeAuthorityConsumerKind::GuestImage;
    }
};

constexpr ComputeAuthorityImageSourceDecision classify_compute_authority_image_source(
    const ComputeAuthorityImageSourceFacts& facts) {
    if (!facts.readable) return {};
    const bool direct_proven = facts.direct_transfer_borrowed ||
        (facts.storage && facts.direct_persistent && facts.direct_upload_skipped);
    const bool exact_alias_source = facts.alias && facts.same_image && facts.same_view;
    const bool owner_proven = facts.owner_transfer_borrowed ||
        (facts.storage && facts.owner_persistent && facts.owner_upload_skipped);
    return {true, direct_proven || (exact_alias_source && owner_proven)};
}

struct ComputeAuthorityImageSourceCounters {
    uint64_t observations = 0;
    uint64_t aliases = 0;
    uint64_t direct_transfer_borrows = 0;
    uint64_t owner_transfer_borrows = 0;
    uint64_t direct_persistent = 0;
    uint64_t owner_persistent = 0;
    uint64_t direct_upload_skips = 0;
    uint64_t owner_upload_skips = 0;
    uint64_t same_images = 0;
    uint64_t same_views = 0;
    uint64_t proven_gpu = 0;
    uint64_t guest = 0;
};

inline void record_compute_authority_image_source(
    ComputeAuthorityImageSourceCounters& counters,
    const ComputeAuthorityImageSourceFacts& facts,
    const ComputeAuthorityImageSourceDecision& decision) {
    if (!decision.observe) return;
    const auto increment_if = [](uint64_t& counter, bool condition) {
        if (condition) counter = shadow_compute_authority_increment(counter);
    };
    counters.observations = shadow_compute_authority_increment(counters.observations);
    increment_if(counters.aliases, facts.alias);
    increment_if(counters.direct_transfer_borrows, facts.direct_transfer_borrowed);
    increment_if(counters.owner_transfer_borrows, facts.owner_transfer_borrowed);
    increment_if(counters.direct_persistent, facts.direct_persistent);
    increment_if(counters.owner_persistent, facts.owner_persistent);
    increment_if(counters.direct_upload_skips, facts.direct_upload_skipped);
    increment_if(counters.owner_upload_skips, facts.owner_upload_skipped);
    increment_if(counters.same_images, facts.same_image);
    increment_if(counters.same_views, facts.same_view);
    increment_if(counters.proven_gpu, decision.proven_gpu);
    increment_if(counters.guest, !decision.proven_gpu);
}

constexpr uint64_t compute_authority_saturating_add(uint64_t first, uint64_t second) {
    return second > std::numeric_limits<uint64_t>::max() - first
        ? std::numeric_limits<uint64_t>::max() : first + second;
}

inline void accumulate_compute_authority_counters(
    ShadowComputeAuthorityCounters& destination,
    const ShadowComputeAuthorityCounters& source) {
#define PROSPER_ADD_AUTHORITY_COUNTER(name) \
    destination.name = compute_authority_saturating_add(destination.name, source.name)
    PROSPER_ADD_AUTHORITY_COUNTER(result_candidates);
    PROSPER_ADD_AUTHORITY_COUNTER(admitted_results);
    PROSPER_ADD_AUTHORITY_COUNTER(replaced_results);
    PROSPER_ADD_AUTHORITY_COUNTER(rejected_results);
    PROSPER_ADD_AUTHORITY_COUNTER(consumer_observations);
    PROSPER_ADD_AUTHORITY_COUNTER(proven_gpu_keeps);
    PROSPER_ADD_AUTHORITY_COUNTER(unrelated_keeps);
    PROSPER_ADD_AUTHORITY_COUNTER(materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(overlap_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(proven_gpu_image_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(guest_image_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(raw_buffer_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(draw_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(dma_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(ordered_memory_effect_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(capture_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(unknown_consumer_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(submit_end_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(unknown_range_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(invalid_result_materializations);
    PROSPER_ADD_AUTHORITY_COUNTER(range_change_materializations);
#undef PROSPER_ADD_AUTHORITY_COUNTER
}

// Per-submit wrapper around ShadowComputeAuthorityCensus.  It does not own or alter any Vulkan or
// guest bytes: the live backend continues its existing synchronous readback/writeback.  This class
// merely asks which exact ordered observer would have forced that writeback if GPU authority were
// deferred in a future change.
class ComputeAuthorityLiveCensus {
public:
    explicit ComputeAuthorityLiveCensus(ComputeAuthorityLiveSelector selector)
        : selector_(selector) {}

    const ComputeAuthorityLiveSelector& selector() const { return selector_; }
    const ComputeAuthorityLiveCounters& counters() const { return counters_; }
    bool active_submit() const { return active_; }
    uint64_t submit_no() const { return submit_no_; }
    bool pending() const { return active_ && submit_.state().pending(); }

    void invalidate_apparatus(bool close_pending_as_unknown = false) {
        if (!selector_.requested) return;
        counters_.apparatus_valid = false;
        if (close_pending_as_unknown && active_)
            (void)submit_.observe(ShadowComputeAuthorityConsumerKind::Unknown);
    }

    ComputeAuthorityLiveObservation observe_program(uint64_t program_hash) {
        if (!selector_.requested) return {};
        if (!active_) {
            counters_.apparatus_valid = false;
            counters_.observations_without_submit = shadow_compute_authority_increment(
                counters_.observations_without_submit);
        }
        counters_.programs_seen = shadow_compute_authority_increment(
            counters_.programs_seen);
        const bool selected = selector_.valid && program_hash == selector_.producer_hash;
        const bool first_match = selected && counters_.programs_matched == 0;
        if (selected)
            counters_.programs_matched = shadow_compute_authority_increment(
                counters_.programs_matched);
        return {selected, first_match};
    }

    void begin_submit(uint64_t submit_no) {
        if (!selector_.requested) return;
        if (active_) {
            counters_.apparatus_valid = false;
            counters_.interrupted_submits = shadow_compute_authority_increment(
                counters_.interrupted_submits);
            // A missing end cannot let authority leak into the next submit.  Attribute the forced
            // close as unknown, then still exercise the ordinary SubmitEnd invariant.
            (void)submit_.observe(ShadowComputeAuthorityConsumerKind::Unknown);
            close_submit();
        }
        active_ = true;
        submit_no_ = submit_no;
        submit_ = {};
        counters_.submits_started = shadow_compute_authority_increment(
            counters_.submits_started);
    }

    ShadowComputeAuthorityTransition observe(
        ShadowComputeAuthorityConsumerKind kind,
        const ShadowComputeAuthorityRange& range =
            ShadowComputeAuthorityRange::unknown()) {
        if (!selector_.requested) return {};
        if (!active_) {
            counters_.apparatus_valid = false;
            counters_.observations_without_submit = shadow_compute_authority_increment(
                counters_.observations_without_submit);
            return {};
        }
        return submit_.observe(kind, range);
    }

    ShadowComputeAuthorityTransition record_selected_storage_output(
        const ShadowComputeAuthorityRange& range, bool retained) {
        if (!selector_.requested) return {};
        counters_.selected_storage_outputs = shadow_compute_authority_increment(
            counters_.selected_storage_outputs);
        if (!active_) {
            counters_.apparatus_valid = false;
            counters_.observations_without_submit = shadow_compute_authority_increment(
                counters_.observations_without_submit);
            return {};
        }
        if (retained) {
            counters_.retained_storage_outputs = shadow_compute_authority_increment(
                counters_.retained_storage_outputs);
            return submit_.note_retained_result(range);
        }
        counters_.synchronous_storage_outputs = shadow_compute_authority_increment(
            counters_.synchronous_storage_outputs);
        return submit_.observe(
            ShadowComputeAuthorityConsumerKind::OrderedMemoryEffect, range);
    }

    ShadowComputeAuthorityTransition end_submit() {
        if (!selector_.requested) return {};
        if (!active_) {
            counters_.apparatus_valid = false;
            counters_.observations_without_submit = shadow_compute_authority_increment(
                counters_.observations_without_submit);
            return {};
        }
        return close_submit();
    }

    bool summary_valid() const {
        return selector_.requested && selector_.valid &&
            counters_.programs_matched != 0 &&
            counters_.retained_storage_outputs != 0 &&
            counters_.authority.admitted_results != 0 &&
            counters_.apparatus_valid && !active_ &&
            counters_.pending_after_submit_end == 0;
    }

    bool claim_summary() {
        if (counters_.summary_reported) return false;
        counters_.summary_reported = true;
        return true;
    }

private:
    ShadowComputeAuthorityTransition close_submit() {
        const bool pending_before = submit_.state().pending();
        const ShadowComputeAuthorityTransition transition = submit_.observe(
            ShadowComputeAuthorityConsumerKind::SubmitEnd);
        const bool pending_after = submit_.state().pending();
        if (pending_before)
            counters_.pending_before_submit_end = shadow_compute_authority_increment(
                counters_.pending_before_submit_end);
        if (pending_after) {
            counters_.pending_after_submit_end = shadow_compute_authority_increment(
                counters_.pending_after_submit_end);
            counters_.apparatus_valid = false;
        }
        accumulate_compute_authority_counters(
            counters_.authority, submit_.counters());
        counters_.submits_completed = shadow_compute_authority_increment(
            counters_.submits_completed);
        active_ = false;
        submit_no_ = 0;
        submit_ = {};
        return transition;
    }

    ComputeAuthorityLiveSelector selector_;
    ComputeAuthorityLiveCounters counters_;
    bool active_ = false;
    uint64_t submit_no_ = 0;
    ShadowComputeAuthorityCensus submit_;
};

} // namespace prosper::frontend

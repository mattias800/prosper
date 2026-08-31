#pragma once

#include "shared/compute/compute_timing_selector.hpp"

#include <cstdint>

namespace prosper::frontend {

enum class ComputeTransferGateRole : uint8_t {
    None,
    Producer,
    Consumer,
};

// Keep the storage-image cache decision and its diagnostic inputs together.  A transfer-gate
// census must expose every input to this policy: reporting only the derived candidate bit cannot
// distinguish a real eligibility difference from an instrument that never observed the gate.
struct ComputeStorageCacheGateInputs {
    bool renderer_owned = false;
    bool dcc_cache_safe = false;
    bool poison_verify = false;
    bool exact_storage = false;
    bool seed_skip = false;
    bool persistent_enabled = false;
};

constexpr bool compute_storage_cache_gate_candidate(
    const ComputeStorageCacheGateInputs& inputs) {
    return !inputs.renderer_owned && inputs.dcc_cache_safe &&
           !inputs.poison_verify && (inputs.exact_storage || inputs.seed_skip) &&
           inputs.persistent_enabled;
}

// The sampled-image cache decision, in the same shape and for the same reason: a census that
// reports only the candidate bit cannot tell "this surface was ineligible" from "the gate was never
// reached".
//
// `dcc_cache_safe` is deliberately NOT required here, and that asymmetry against the storage gate
// above is the substance of #3149 rather than an oversight.  A storage target's base bytes may not
// be authoritative while a DCC plane is live, so that gate must consult it.  A *sampled* upload is
// detile + unpack of the base bytes and never reads the metadata plane -- the fast-clear helper is
// its only consumer, and `sampled_dcc_fast_clear` excludes exactly the surfaces that reach it.  So
// base bytes are the whole identity of a cacheable sampled entry, which is what the cache already
// validates, and requiring an all-0xff plane on top of that only cost hit rate.
//
// That last sentence is a live invariant, not a historical note, and it cuts both ways:
//   * if a sampled decode ever starts CONSULTING the DCC metadata plane, this gate has to consult
//     it too, or the cache will replay a decode whose input it never validated;
//   * if a sampled decode ever starts DECLINING on metadata state, this gate has to decline with
//     it.  That seam already exists and is unwired:
//     `gpu/resources/compressed_source_authority.hpp`'s `sampled_source_decision` has no production
//     caller today, and it classifies exactly the surfaces this gate now admits.  **Wiring it must
//     move this gate in the same commit.**  Deliberately not stated more precisely than that: the
//     two do not stand in a clean correspondence.  That function has five distinct decline reasons
//     and a non-declining `DccUncompressedBase` outcome, its `CompressionMetadataKind` is an
//     adapter-supplied input so `Unknown` is reachable, and an import bypass pre-empts the metadata
//     reasons entirely -- so a compressed surface admitted here can land on several of its
//     outcomes, including non-declining ones.  An earlier draft of this comment named one specific
//     reason and was wrong in both directions (#3150 review).
// Nothing here can assert either mechanically, so both are written where somebody touching that
// decision will be looking.
//
// `dcc_cache_disabled` (PROSPER_NO_DCC_IMAGE_CACHE) restores the older, stricter behaviour exactly,
// so a rendering report can be bisected against this change with one variable.
struct ComputeSampledCacheGateInputs {
    bool sampled_dcc_fast_clear = false;
    bool dcc_cache_safe = false;
    bool dcc_cache_disabled = false;
    bool persistent_enabled = false;
};

constexpr bool compute_sampled_cache_gate_candidate(
    const ComputeSampledCacheGateInputs& inputs) {
    return !inputs.sampled_dcc_fast_clear &&
           (inputs.dcc_cache_safe || !inputs.dcc_cache_disabled) &&
           inputs.persistent_enabled;
}

// A compressed writable target cannot enter the persistent cache before dispatch because its base
// bytes are not yet authoritative.  A successful ordinary writeback changes that fact: it publishes
// exact base bytes and marks the complete DCC plane uncompressed.  Admit that result only when DCC
// was the sole failed normal gate, the metadata plane is writable, and this binding owns the image's
// writeback obligation.  The caller still has to prove fence completion, exact data/metadata
// publication, and a final all-0xff metadata state before retaining the image.
struct ComputeStoragePostWritebackPromotionInputs {
    ComputeStorageCacheGateInputs pre_dispatch{};
    bool writable_dcc_metadata = false;
    bool unique_alias_owner = false;
};

constexpr bool compute_storage_post_writeback_promotion_candidate(
    const ComputeStoragePostWritebackPromotionInputs& inputs) {
    const ComputeStorageCacheGateInputs& gates = inputs.pre_dispatch;
    return !gates.renderer_owned && !gates.dcc_cache_safe &&
           !gates.poison_verify && gates.exact_storage &&
           gates.persistent_enabled && inputs.writable_dcc_metadata &&
           inputs.unique_alias_owner;
}

struct ComputeTransferGateSelector {
    bool requested = false;
    bool valid = false;
    uint64_t producer_hash = 0;
    uint64_t consumer_hash = 0;
};

constexpr ComputeTransferGateRole select_compute_transfer_gate_role(
    const ComputeTransferGateSelector& selector,
    uint64_t program_hash) {
    if (!selector.requested || !selector.valid ||
        selector.producer_hash == selector.consumer_hash)
        return ComputeTransferGateRole::None;
    if (program_hash == selector.producer_hash)
        return ComputeTransferGateRole::Producer;
    if (program_hash == selector.consumer_hash)
        return ComputeTransferGateRole::Consumer;
    return ComputeTransferGateRole::None;
}

struct ComputeTransferGateSelectorCounters {
    uint64_t seen = 0;
    uint64_t producer_matches = 0;
    uint64_t consumer_matches = 0;
    uint64_t producer_storage_gate_observations = 0;
    uint64_t consumer_storage_gate_observations = 0;
    bool summary_reported = false;
};

struct ComputeTransferGateSelectorObservation {
    ComputeTransferGateRole role = ComputeTransferGateRole::None;
    bool first_match = false;
};

inline ComputeTransferGateSelectorObservation observe_compute_transfer_gate_selector(
    const ComputeTransferGateSelector& selector,
    ComputeTransferGateSelectorCounters& counters,
    uint64_t program_hash) {
    counters.seen = saturating_increment(counters.seen);
    const ComputeTransferGateRole role =
        select_compute_transfer_gate_role(selector, program_hash);
    uint64_t* matches = role == ComputeTransferGateRole::Producer
        ? &counters.producer_matches
        : role == ComputeTransferGateRole::Consumer
            ? &counters.consumer_matches : nullptr;
    const bool first_match = matches && *matches == 0;
    if (matches) *matches = saturating_increment(*matches);
    return {role, first_match};
}

inline bool record_compute_transfer_storage_gate_observation(
    ComputeTransferGateRole role,
    ComputeTransferGateSelectorCounters& counters) {
    uint64_t* observations = role == ComputeTransferGateRole::Producer
        ? &counters.producer_storage_gate_observations
        : role == ComputeTransferGateRole::Consumer
            ? &counters.consumer_storage_gate_observations : nullptr;
    if (!observations) return false;
    const bool first_observation = *observations == 0;
    *observations = saturating_increment(*observations);
    return first_observation;
}

constexpr bool compute_transfer_gate_selector_is_invalid(
    const ComputeTransferGateSelector& selector,
    const ComputeTransferGateSelectorCounters& counters) {
    return selector.requested &&
           (!selector.valid || selector.producer_hash == selector.consumer_hash ||
            counters.producer_matches == 0 || counters.consumer_matches == 0 ||
            counters.producer_storage_gate_observations == 0 ||
            counters.consumer_storage_gate_observations == 0);
}

inline bool claim_compute_transfer_gate_selector_summary(
    ComputeTransferGateSelectorCounters& counters) {
    if (counters.summary_reported) return false;
    counters.summary_reported = true;
    return true;
}

} // namespace prosper::frontend

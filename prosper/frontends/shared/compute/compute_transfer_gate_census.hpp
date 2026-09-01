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

// #3157: a guest-memory range, used to measure whether one dispatch's SEED sources overlap the
// PREVIOUS dispatch's WRITEBACK targets.
//
// Why this exists: the per-dispatch `vkWaitForFences` in the live compute path costs ~330 us
// against ~171 us of GPU execution, so ~160 us per dispatch is scheduling latency. Deferring the
// waits (pipelining several dispatches before consuming any results) is the obvious fix, but it
// is only sound where a later dispatch does not seed from guest bytes an earlier one writes
// back -- both address `r->gpu_addr` for `guest_bytes`, so an overlap is a read-after-write
// hazard through guest memory that would silently seed from stale bytes.
//
// The overlap RATE bounds the achievable win: if consecutive dispatches usually alias, a
// correctness-preserving pipeline drains constantly and buys nothing. Measure before building.
//
// KNOWN BIASES, stated so a reader can weigh the number instead of trusting it. All four were
// found by review rather than by the instrument, which is itself the reason to write them down:
//   * UNDER-counts: the sampled DCC fast-clear path reads guest memory through `metadata_addr`
//     and its writeback writes the same plane, and neither side is recorded (covering it needs
//     the metadata range, not `gpu_addr`).
//   * OVER-counts: `compute_transfer_seed_borrowed` bindings are scored as guest seeds, but they
//     read a retained GPU image rather than guest memory.
//   * ASYMMETRIC: `SpirvDescriptorBinding::readable` marks a write-only output that need not be
//     seeded. The image path consults it; `BoundBuffer` copies `writable` but never `readable`,
//     so write-only IMAGES are excluded from seeds while write-only BUFFERS are included. That
//     matches what prosper's buffer upload actually does today -- it does not consult `readable`
//     either, so those bytes really are read -- but the two paths do not agree, and if the upload
//     ever starts honouring `readable` this census must change with it.
//   * DEPTH-1 ONLY: overlap is measured against the immediately preceding dispatch, so the rate
//     is a lower bound for any ring depth greater than one.
struct ComputeGuestRange {
    uint64_t addr = 0;
    uint64_t bytes = 0;
};

// Address 0 is never a guest address, so a range at 0 is never a real one.
//
// Extracted and named so the rule is PINNED rather than living as a bare `if (r->gpu_addr)` at two
// call sites. An explicit null V# entry is bound as a 4-byte zero source backed by a stack-local
// array, and that synthesis deliberately keeps `gpu_addr == 0` -- its own comment says the point is
// to bind a null buffer "without inventing a guest address". Feeding it to the census invented one:
// two dispatches each holding a writable null array entry both produce {0, 4}, which overlaps
// itself, so the census reported a FABRICATED alias at a non-address.
//
// `gpu_addr` rather than `host_data` is the right discriminator, and not only because it is
// simpler: `resource_bytes_for` returns `host_data` for legitimate MIRRORED guest resources too, so
// a host_data test would drop real guest ranges. The synthesis sets `host_data` and leaves
// `gpu_addr` at 0 -- one field separates the cases, and it is this one. It is also the predicate the
// code that performs the writes already uses, at both
// `notify_guest_gpu_write_preserving_bytes` call sites, which predate this census.
constexpr bool compute_guest_range_is_real(const ComputeGuestRange& r) {
    return r.addr != 0;
}

// Half-open [addr, addr+bytes) intersection. A zero-length range touches nothing and so can
// never alias -- that is the `seed_skip`/`renderer_owned` case, where nothing is read from guest
// memory at all, and counting it as an overlap would overstate the hazard.
constexpr bool compute_guest_ranges_overlap(const ComputeGuestRange& a,
                                            const ComputeGuestRange& b) {
    if (a.bytes == 0 || b.bytes == 0) return false;
    return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes;
}

// Plain (non-atomic) counters, deliberately: the census is read and written only from the thread
// that runs `execute_live_compute_items`, and making them atomic would put a lock-prefixed RMW on
// a path whose cost this instrument exists to measure. If a second producer thread is ever added,
// these need revisiting -- they are not thread-safe and do not claim to be.
struct ComputeAliasCensusCounters {
    uint64_t dispatches = 0;          // dispatches that declared at least one guest seed source
    uint64_t aliasing_dispatches = 0; // ...of those, ones seeding from the previous writeback set
    uint64_t seed_ranges = 0;
    uint64_t write_ranges = 0;
};

} // namespace prosper::frontend

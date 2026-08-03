#pragma once

#include "compute_timing_selector.hpp"

#include <cstdint>

namespace prosper::frontend {

enum class ComputeTransferGateRole : uint8_t {
    None,
    Producer,
    Consumer,
};

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

constexpr bool compute_transfer_gate_selector_is_invalid(
    const ComputeTransferGateSelector& selector,
    const ComputeTransferGateSelectorCounters& counters) {
    return selector.requested &&
           (!selector.valid || selector.producer_hash == selector.consumer_hash ||
            counters.producer_matches == 0 || counters.consumer_matches == 0);
}

inline bool claim_compute_transfer_gate_selector_summary(
    ComputeTransferGateSelectorCounters& counters) {
    if (counters.summary_reported) return false;
    counters.summary_reported = true;
    return true;
}

} // namespace prosper::frontend

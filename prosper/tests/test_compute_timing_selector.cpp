#include "compute_timing_selector.hpp"
#include "compute_transfer_gate_census.hpp"

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
    std::printf("compute timing selector policy\n");

    const auto unset = parse_compute_timing_selector_u64(nullptr);
    const auto empty = parse_compute_timing_selector_u64("");
    const auto decimal_zero = parse_compute_timing_selector_u64("0");
    const auto decimal_max =
        parse_compute_timing_selector_u64("18446744073709551615");
    const auto hex_zero = parse_compute_timing_selector_u64("0x0");
    const auto hex_max =
        parse_compute_timing_selector_u64("0Xffffffffffffffff");
    CHECK(unset.error == ComputeTimingSelectorParseError::Unset &&
              empty.error == ComputeTimingSelectorParseError::Empty,
          "unset and empty selectors remain distinguishable");
    CHECK(decimal_zero.accepted() && decimal_zero.value == 0 &&
              decimal_max.accepted() &&
              decimal_max.value == std::numeric_limits<uint64_t>::max() &&
              hex_zero.accepted() && hex_zero.value == 0 &&
              hex_max.accepted() &&
              hex_max.value == std::numeric_limits<uint64_t>::max(),
          "decimal and hexadecimal uint64 endpoints are accepted exactly");

    CHECK(parse_compute_timing_selector_u64("+1").error ==
                  ComputeTimingSelectorParseError::Sign &&
              parse_compute_timing_selector_u64("-1").error ==
                  ComputeTimingSelectorParseError::Sign,
          "unsigned selectors reject both signs");
    CHECK(parse_compute_timing_selector_u64("0x").error ==
                  ComputeTimingSelectorParseError::MissingDigits &&
              parse_compute_timing_selector_u64("xyz").error ==
                  ComputeTimingSelectorParseError::InvalidDigit,
          "missing and invalid digits are rejected explicitly");
    CHECK(parse_compute_timing_selector_u64("42x").error ==
                  ComputeTimingSelectorParseError::TrailingCharacters &&
              parse_compute_timing_selector_u64("0x2a ").error ==
                  ComputeTimingSelectorParseError::TrailingCharacters,
          "trailing bytes and whitespace are rejected");
    CHECK(parse_compute_timing_selector_u64("18446744073709551616").error ==
                  ComputeTimingSelectorParseError::Overflow &&
              parse_compute_timing_selector_u64("0x10000000000000000").error ==
                  ComputeTimingSelectorParseError::Overflow,
          "decimal and hexadecimal overflow are rejected");

    ComputeTimingSelector address_only;
    address_only.address_enabled = true;
    address_only.address = 0x1234;
    CHECK(compute_timing_selector_matches(address_only, 0x1234, 0xaaaa) &&
              !compute_timing_selector_matches(address_only, 0x1235, 0xaaaa),
          "address-only selection preserves exact legacy behavior");
    address_only.address_valid = false;
    CHECK(!compute_timing_selector_matches(address_only, 0x1234, 0xaaaa),
          "invalid address selectors fail closed");

    ComputeTimingSelector hash_only;
    hash_only.hash_requested = true;
    hash_only.hash = 0xfeedface;
    ComputeTimingSelector both = hash_only;
    both.address_enabled = true;
    both.address = 0x1234;
    CHECK(compute_timing_selector_matches(hash_only, 0x9999, 0xfeedface) &&
              compute_timing_selector_matches(both, 0x1234, 0xfeedface),
          "stable hash match selects hash-only and address-AND-hash modes");
    CHECK(!compute_timing_selector_matches(hash_only, 0x9999, 0xfeedfacf) &&
              !compute_timing_selector_matches(both, 0x1234, 0xfeedfacf) &&
              !compute_timing_selector_matches(both, 0x1235, 0xfeedface),
          "stable hash mismatches never select");
    hash_only.hash_valid = false;
    CHECK(!compute_timing_selector_matches(hash_only, 0x9999, 0xfeedface),
          "invalid hash selectors fail closed");

    ComputeTimingSelectorCounters counters;
    const auto miss = observe_compute_timing_selector(
        both, counters, 0x1235, 0xfeedface);
    const auto first = observe_compute_timing_selector(
        both, counters, 0x1234, 0xfeedface);
    const auto later = observe_compute_timing_selector(
        both, counters, 0x1234, 0xfeedface);
    CHECK(!miss.matched && !miss.first_match && first.matched && first.first_match &&
              later.matched && !later.first_match &&
              counters.seen == 3 && counters.matched == 2,
          "seen and matched counters emit exactly one first-match proof");

    ComputeTimingSelectorCounters saturated{
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()};
    const auto saturated_match = observe_compute_timing_selector(
        both, saturated, 0x1234, 0xfeedface);
    CHECK(saturated_match.matched && !saturated_match.first_match &&
              saturated.seen == std::numeric_limits<uint64_t>::max() &&
              saturated.matched == std::numeric_limits<uint64_t>::max(),
          "diagnostic counters saturate instead of wrapping");

    ComputeTimingSelectorCounters reported_once{7, 3};
    const bool explicit_report = claim_compute_timing_selector_summary(reported_once);
    const bool destructor_fallback = claim_compute_timing_selector_summary(reported_once);
    CHECK(explicit_report && !destructor_fallback && reported_once.summary_reported,
          "explicit report and destructor fallback emit exactly once");

    ComputeTimingSelectorCounters zero_matches{7, 0};
    ComputeTimingSelectorCounters one_match{7, 1};
    CHECK(claim_compute_timing_selector_summary(zero_matches) &&
              compute_timing_zero_match_is_invalid(zero_matches) &&
              !compute_timing_zero_match_is_invalid(one_match),
          "zero-match summary is apparatus-invalid");

    const ComputeStorageCacheGateInputs cache_eligible{
        false, true, false, true, false, true};
    CHECK(compute_storage_cache_gate_candidate(cache_eligible),
          "storage cache candidate accepts all gates with exact storage");
    ComputeStorageCacheGateInputs cache_gates = cache_eligible;
    cache_gates.renderer_owned = true;
    CHECK(!compute_storage_cache_gate_candidate(cache_gates),
          "renderer ownership blocks storage cache candidate");
    cache_gates = cache_eligible;
    cache_gates.dcc_cache_safe = false;
    CHECK(!compute_storage_cache_gate_candidate(cache_gates),
          "DCC unsafety blocks storage cache candidate");
    cache_gates = cache_eligible;
    cache_gates.poison_verify = true;
    CHECK(!compute_storage_cache_gate_candidate(cache_gates),
          "poison verification blocks storage cache candidate");
    cache_gates = cache_eligible;
    cache_gates.exact_storage = false;
    CHECK(!compute_storage_cache_gate_candidate(cache_gates),
          "neither exact storage nor seed skip blocks storage cache candidate");
    cache_gates.seed_skip = true;
    CHECK(compute_storage_cache_gate_candidate(cache_gates),
          "seed skip satisfies the storage representation gate");
    cache_gates = cache_eligible;
    cache_gates.persistent_enabled = false;
    CHECK(!compute_storage_cache_gate_candidate(cache_gates),
          "persistent cache disable blocks storage cache candidate");

    ComputeTransferGateSelector gate_selector{
        true, true, 0x2c595d5aada78398ull, 0xa57c763ae4d70d1dull};
    ComputeTransferGateSelectorCounters gate_counters;
    const auto gate_miss = observe_compute_transfer_gate_selector(
        gate_selector, gate_counters, 0x1234);
    const auto producer_first = observe_compute_transfer_gate_selector(
        gate_selector, gate_counters, gate_selector.producer_hash);
    const auto producer_later = observe_compute_transfer_gate_selector(
        gate_selector, gate_counters, gate_selector.producer_hash);
    const auto consumer_first = observe_compute_transfer_gate_selector(
        gate_selector, gate_counters, gate_selector.consumer_hash);
    CHECK(gate_miss.role == ComputeTransferGateRole::None && !gate_miss.first_match &&
              producer_first.role == ComputeTransferGateRole::Producer &&
              producer_first.first_match &&
              producer_later.role == ComputeTransferGateRole::Producer &&
              !producer_later.first_match &&
              consumer_first.role == ComputeTransferGateRole::Consumer &&
              consumer_first.first_match && gate_counters.seen == 4 &&
              gate_counters.producer_matches == 2 &&
              gate_counters.consumer_matches == 1,
          "dual transfer selectors prove each exact hash and first match independently");
    CHECK(compute_transfer_gate_selector_is_invalid(gate_selector, gate_counters),
          "hash matches without storage gate observations are apparatus-invalid");
    const bool delayed_producer_detail =
        record_compute_transfer_storage_gate_observation(
            ComputeTransferGateRole::Producer, gate_counters);
    (void)observe_compute_transfer_gate_selector(
        gate_selector, gate_counters, gate_selector.producer_hash);
    const bool repeated_producer_detail =
        record_compute_transfer_storage_gate_observation(
            ComputeTransferGateRole::Producer, gate_counters);
    CHECK(delayed_producer_detail && !repeated_producer_detail &&
              gate_counters.producer_storage_gate_observations == 2,
          "storage detail survives an earlier hash-only match and emits exactly once");
    CHECK(compute_transfer_gate_selector_is_invalid(gate_selector, gate_counters),
          "one-sided storage gate observation is apparatus-invalid");
    (void)record_compute_transfer_storage_gate_observation(
        ComputeTransferGateRole::Consumer, gate_counters);
    CHECK(!compute_transfer_gate_selector_is_invalid(gate_selector, gate_counters),
          "dual transfer selector requires both hashes and storage gate observations");

    ComputeTransferGateSelectorCounters missing_consumer;
    (void)observe_compute_transfer_gate_selector(
        gate_selector, missing_consumer, gate_selector.producer_hash);
    ComputeTransferGateSelector duplicate_selector{
        true, true, gate_selector.producer_hash, gate_selector.producer_hash};
    CHECK(compute_transfer_gate_selector_is_invalid(gate_selector, missing_consumer) &&
              compute_transfer_gate_selector_is_invalid(
                  duplicate_selector, missing_consumer),
          "zero-match and duplicate transfer selectors fail visibly");
    CHECK(claim_compute_transfer_gate_selector_summary(missing_consumer) &&
              !claim_compute_transfer_gate_selector_summary(missing_consumer),
          "transfer selector summary is claimed exactly once");

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}

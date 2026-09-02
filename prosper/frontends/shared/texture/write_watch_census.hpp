#pragma once

// Why this file exists, in one paragraph, because the reason is the finding.
//
// #3155 asked whether write-watch promotion ever fires, and answered it by counting `memcmp` from
// OUTSIDE the process: an `LD_PRELOAD` interposer that dumped its running tally every 2^20 calls.
// Two arms were then read at whichever dump happened to have fired last, so the headline number
// compared a tally taken at 2,097,152 calls against one taken at 1,048,576 -- different amounts of
// run, not different behaviour. The measurement was retracted; the exact powers of two in both arms
// were the tell. A counter that lives inside the decision it measures cannot make that mistake: it
// counts the decisions themselves rather than a proxy, it always carries its own denominator, and
// every ratio it reports is normalised by that denominator.
//
// What it answers, in one run and without a private patch: of the acquisitions that had to prove a
// cached source unchanged, how many were decided by each of the three proofs (the intra-submit GPU
// write journal, an armed page watch, a full byte compare), how many bytes each of those cost, and
// -- at every point where the promotion policy was consulted -- what the entry's stability counter
// actually held. The last part is the one that separates "the threshold is too high" from "the
// ladder is unclimbable", which are different defects with different fixes.
//
// Counting is unconditional. These are per-ACQUISITION relaxed increments, not per-byte, against a
// path whose cheapest branch is a page-watch query and whose expensive branch is a multi-megabyte
// `memcmp`; a gate would cost about as much as the counter. Only the periodic REPORT is gated, on
// PROSPER_WATCH_PROMOTE_CENSUS.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstddef>

namespace prosper::frontend {

struct WriteWatchCensusSnapshot {
    // Promotion decisions: one per call into the policy, whatever it answered.
    uint64_t decisions = 0;
    uint64_t stability_0 = 0;
    uint64_t stability_1 = 0;
    uint64_t stability_2 = 0;
    uint64_t stability_3_plus = 0;
    uint64_t threshold_met = 0;     // the stability/size policy said yes
    uint64_t granted = 0;           // ...and the per-submit byte budget also said yes
    uint64_t budget_refused = 0;    // threshold_met - granted

    // How each validated acquisition was actually decided, and what it cost.
    uint64_t journal_skips = 0;
    uint64_t journal_skip_bytes = 0;
    uint64_t watch_skips = 0;
    uint64_t watch_skip_bytes = 0;
    uint64_t exact_compares = 0;
    uint64_t exact_compare_bytes = 0;
};

class WriteWatchCensus {
public:
    void record_promotion_decision(uint32_t stable_validations, bool threshold_met, bool granted) {
        bump(decisions_);
        if (stable_validations == 0) bump(stability_0_);
        else if (stable_validations == 1) bump(stability_1_);
        else if (stable_validations == 2) bump(stability_2_);
        else bump(stability_3_plus_);
        if (threshold_met) bump(threshold_met_);
        if (granted) bump(granted_);
    }

    // The intra-submit GPU write journal decided it: O(1), no bytes touched. `bytes` is the source
    // size the compare would have cost, so the three cost lines are comparable.
    void record_journal_skip(uint64_t bytes) { bump(journal_skips_); add(journal_skip_bytes_, bytes); }
    // An armed page watch decided it: O(1). This is the line promotion exists to move.
    void record_watch_skip(uint64_t bytes) { bump(watch_skips_); add(watch_skip_bytes_, bytes); }
    // Neither cheap proof was available, so every byte was read. Record it where the comparison
    // actually runs, never where its result is consumed -- a short-circuited `&&` chain reaches the
    // consumer without having compared anything.
    void record_exact_compare(uint64_t bytes) { bump(exact_compares_); add(exact_compare_bytes_, bytes); }

    WriteWatchCensusSnapshot snapshot() const {
        WriteWatchCensusSnapshot out;
        out.decisions = get(decisions_);
        out.stability_0 = get(stability_0_);
        out.stability_1 = get(stability_1_);
        out.stability_2 = get(stability_2_);
        out.stability_3_plus = get(stability_3_plus_);
        out.threshold_met = get(threshold_met_);
        out.granted = get(granted_);
        out.budget_refused = out.threshold_met - out.granted;
        out.journal_skips = get(journal_skips_);
        out.journal_skip_bytes = get(journal_skip_bytes_);
        out.watch_skips = get(watch_skips_);
        out.watch_skip_bytes = get(watch_skip_bytes_);
        out.exact_compares = get(exact_compares_);
        out.exact_compare_bytes = get(exact_compare_bytes_);
        return out;
    }

private:
    static void bump(std::atomic<uint64_t>& counter) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
    static void add(std::atomic<uint64_t>& counter, uint64_t value) {
        counter.fetch_add(value, std::memory_order_relaxed);
    }
    static uint64_t get(const std::atomic<uint64_t>& counter) {
        return counter.load(std::memory_order_relaxed);
    }

    std::atomic<uint64_t> decisions_{0};
    std::atomic<uint64_t> stability_0_{0};
    std::atomic<uint64_t> stability_1_{0};
    std::atomic<uint64_t> stability_2_{0};
    std::atomic<uint64_t> stability_3_plus_{0};
    std::atomic<uint64_t> threshold_met_{0};
    std::atomic<uint64_t> granted_{0};
    std::atomic<uint64_t> journal_skips_{0};
    std::atomic<uint64_t> journal_skip_bytes_{0};
    std::atomic<uint64_t> watch_skips_{0};
    std::atomic<uint64_t> watch_skip_bytes_{0};
    std::atomic<uint64_t> exact_compares_{0};
    std::atomic<uint64_t> exact_compare_bytes_{0};
};

// Every line is a RUNNING TOTAL and prints the denominator it is a fraction of, so two runs whose
// reports fired at different points can still be compared -- the failure mode that produced #3155's
// retracted numbers. `decisions=` and `validated=` are those denominators; read them first.
inline size_t format_write_watch_census(const WriteWatchCensusSnapshot& census,
                                        char* output, size_t capacity) {
    if (!output || !capacity) return 0;
    const auto percent = [](uint64_t part, uint64_t whole) {
        return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    const uint64_t validated = census.journal_skips + census.watch_skips + census.exact_compares;
    const double mib = 1024.0 * 1024.0;
    const int written = std::snprintf(
        output, capacity,
        "[watch-promote-census] running totals -- decisions=%llu stability0=%llu (%.1f%%) "
        "stability1=%llu stability2=%llu stability3+=%llu threshold_met=%llu granted=%llu "
        "budget_refused=%llu\n"
        "[watch-promote-census] running totals -- validated=%llu journal=%llu (%.1f%%, %.1f MiB "
        "spared) watch=%llu (%.1f%%, %.1f MiB spared) exact=%llu (%.1f%%, %.1f MiB compared)\n",
        (unsigned long long)census.decisions,
        (unsigned long long)census.stability_0, percent(census.stability_0, census.decisions),
        (unsigned long long)census.stability_1,
        (unsigned long long)census.stability_2,
        (unsigned long long)census.stability_3_plus,
        (unsigned long long)census.threshold_met,
        (unsigned long long)census.granted,
        (unsigned long long)census.budget_refused,
        (unsigned long long)validated,
        (unsigned long long)census.journal_skips, percent(census.journal_skips, validated),
        static_cast<double>(census.journal_skip_bytes) / mib,
        (unsigned long long)census.watch_skips, percent(census.watch_skips, validated),
        static_cast<double>(census.watch_skip_bytes) / mib,
        (unsigned long long)census.exact_compares, percent(census.exact_compares, validated),
        static_cast<double>(census.exact_compare_bytes) / mib);
    if (written < 0) return 0;
    return static_cast<size_t>(written) < capacity ? static_cast<size_t>(written) : capacity - 1;
}

} // namespace prosper::frontend

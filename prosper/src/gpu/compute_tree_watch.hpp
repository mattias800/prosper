// Address-keyed watch over a guest binary-hierarchy parent table.
//
// Distinct from compute_parent_walk.hpp, which is PROGRAM-keyed: it evaluates one selected
// program's own resource immediately before that program runs. This watch is keyed on a guest
// ADDRESS and evaluated around EVERY realized dispatch, so a change is attributed to the dispatch
// that made it rather than to "somewhere between two observations of one consumer".
//
// The distinction matters for the case it was built for: the consumer that hangs is not
// necessarily the program that corrupted what it consumes, and a program that changes these bytes
// without binding a range containing them is a finding in itself (an out-of-bounds write, or a
// binding path the resource traversal does not model). Reporting whether the running program's
// table contains the address alongside the change is what makes that visible instead of assumed.
//
// Pure analysis lives here so it is testable without a device; gpu_executor owns the guest reads.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "compute_parent_walk.hpp"

namespace prosper::gpu {

struct ComputeTreeWatchSelector {
    uint64_t addr = 0;
    uint32_t records = 0;
    uint32_t index_shift = 3;
    uint32_t index_mask = (1u << 27u) - 1u;
};

// Strict selector syntax:
//   ADDR:RECORDS[:SHIFT:MASK]
// All fields accept C integer syntax. Shift/mask default to the guest's own bfe(rec, 3, 27).
// `records` is a dword count, not a byte count.
std::optional<ComputeTreeWatchSelector> parse_compute_tree_watch_selector(const char* value);

struct ComputeTreeWatchDelta {
    uint32_t index = 0;
    uint32_t before = 0;
    uint32_t after = 0;
};

// Changed dword indices, ascending, capped at `limit` entries. `total_changed` receives the
// uncapped count so a truncated report never reads as a complete one.
std::vector<ComputeTreeWatchDelta> compute_tree_watch_deltas(
    std::span<const uint32_t> before, std::span<const uint32_t> after, uint32_t limit,
    uint32_t* total_changed);

enum class ComputeTreeWatchTransition {
    Unchanged,
    CleanToClean,
    CleanToCyclic,
    CyclicToClean,
    CyclicToCyclic,
};

const char* compute_tree_watch_transition_name(ComputeTreeWatchTransition transition);

// Cyclicity is the only axis this classifies. Depth is reported separately by the walk report;
// folding it in here would make "clean" mean two different things in two different runs.
ComputeTreeWatchTransition classify_compute_tree_watch_transition(
    const ComputeParentWalkReport& before, const ComputeParentWalkReport& after,
    bool bytes_changed);

// Sibling structure under the LBVH reading (#2542): a full binary tree over 1,032 leaves has
// 1,031 internal nodes, adjacent records are siblings sharing a parent, and bit 30 identifies the
// child side. `pairs` counts k where records[k+1] == records[k] | kSiblingSideBit; `unpaired_side`
// counts records with the side bit set whose k-1 mate does not match.
//
// This is a QUANTITATIVE CORRELATION, not a correctness rule. Clean samples tolerate a nonzero
// unpaired count and the slot-level causal test on it failed, so a nonzero value is a signal to
// look, never on its own a defect.
inline constexpr uint32_t kComputeTreeSiblingSideBit = 0x40000000u;

struct ComputeTreeSiblingReport {
    uint32_t records = 0;
    uint32_t pairs = 0;
    uint32_t side_bit_set = 0;
    uint32_t unpaired_side = 0;
};

ComputeTreeSiblingReport analyze_compute_tree_siblings(std::span<const uint32_t> words);

} // namespace prosper::gpu

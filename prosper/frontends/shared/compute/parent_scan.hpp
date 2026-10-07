// parent_scan.hpp -- CPU-side cyclicity census of a link/parent array (PROSPER_COMPUTE_PARENTSCAN).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

namespace prosper::frontend {

// PROSPER_COMPUTE_PARENTSCAN=0xADDR — CPU-side cyclicity census of a link/parent array, taken
// PRE-dispatch from the exact bytes the dispatch is about to read.
//
// This exists because the obvious instrument does not work. Deciding whether a runaway dispatch's INPUT
// is ALREADY cyclic needs the array and the runaway record from the SAME run, and arming
// PROSPER_GPU_CAPTURE to obtain the array changes which dispatches run away — measured on GTA V's
// 0x413dc6700: 11 dispatches of both parities without a capture, reproducibly, versus 5 odd-only
// ordinals with one armed. An instrument that alters the phenomenon cannot establish its cause.
//
// A walk of bytes the front half has already materialized touches no GPU state, issues no submit and
// cannot reorder one, so this can run alongside PROSPER_CFG_TRIP_BOUND's witness and be read against it.
//
// The link encoding is the title's own: next = (word >> 3) & 0x7FFFFFF, terminating on 0 or on an index
// at/after the record count — an out-of-range RDNA2 buffer load returns zero, which is exactly what
// exits the guest's pc88..97 walk. The line reports `records` and the encoding's shift/mask so a wrong
// guess is visible rather than silent. CONFIDENCE: HIGH on the encoding (it is the guest's own
// `v_bfe_u32 v1, v1, 3, 27` with NUM_RECORDS as the bound).
struct ParentScanResult {
    // `terminating + cyclic == records` -- index 0 is the terminator and is classified as
    // terminating, not skipped. An earlier revision broke out of the walk before pushing it, so the
    // two columns silently summed to records-1 on EVERY array (measured 400/400), which is the kind of
    // off-by-one that reads as a rounding difference rather than a bug.
    uint32_t records = 0, terminating = 0, cyclic = 0, cycle_nodes = 0;
    // Longest terminating CHAIN, computed as a depth, not the longest walk this scan happened to take.
    // The walk length depends on the order starts are visited -- memoisation truncates later walks --
    // so a single 2047-link chain reports 1 or 2047 purely by link direction.
    uint32_t longest = 0;
    uint32_t sample_count = 0;
    uint32_t sample_idx[6]{}, sample_word[6]{}, sample_next[6]{};
};

inline ParentScanResult scan_parent_array(const uint8_t* bytes, size_t byte_count) {
    ParentScanResult out;
    if (!bytes || byte_count < 4) return out;
    const uint32_t records = static_cast<uint32_t>(byte_count / 4);
    out.records = records;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(bytes);
    // 0 = unclassified, 1 = reaches a terminator, 2 = enters a cycle. Memoized so the whole array is
    // classified in O(records) rather than O(records * path length).
    std::vector<uint8_t> state(records, 0);
    std::vector<uint32_t> path;
    std::vector<uint32_t> seen_at(records, UINT32_MAX);
    std::set<uint32_t> cycle_nodes;
    for (uint32_t start = 0; start < records; ++start) {
        if (state[start]) continue;
        path.clear();
        uint32_t i = start;
        uint8_t verdict = 1;
        while (true) {
            if (i == 0 || i >= records) {   // terminator / OOB read -> 0
                verdict = 1;
                break;
            }
            if (state[i]) {   // already classified
                verdict = state[i];
                break;
            }
            if (seen_at[i] != UINT32_MAX) {   // revisited on THIS walk
                for (size_t k = seen_at[i]; k < path.size(); ++k) cycle_nodes.insert(path[k]);
                verdict = 2;
                break;
            }
            seen_at[i] = static_cast<uint32_t>(path.size());
            path.push_back(i);
            i = (words[i] >> 3) & 0x7FFFFFFu;
        }
        for (uint32_t node : path) {
            state[node] = verdict;
            seen_at[node] = UINT32_MAX;
        }
    }
    state[0] = 1;   // the terminator itself terminates
    // Depth of each terminating node, memoised: depth(i) = 1 + depth(next(i)), 0 for a cyclic node.
    // Independent of visit order, unlike the walk length it replaces.
    std::vector<uint32_t> depth(records, 0);
    for (uint32_t start = 0; start < records; ++start) {
        if (state[start] != 1 || depth[start]) continue;
        std::vector<uint32_t> chain;
        uint32_t i = start;
        while (i != 0 && i < records && state[i] == 1 && !depth[i]) {
            chain.push_back(i);
            i = (words[i] >> 3) & 0x7FFFFFFu;
        }
        uint32_t d = (i < records) ? depth[i] : 0u;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            d += 1u;
            depth[*it] = d;
        }
        if (d > out.longest) out.longest = d;
    }
    for (uint32_t k = 0; k < records; ++k) {
        if (state[k] == 1)
            ++out.terminating;
        else if (state[k] == 2)
            ++out.cyclic;
    }
    out.cycle_nodes = static_cast<uint32_t>(cycle_nodes.size());
    // Keep a few actual ring members. A count says corruption happened; the entries say what SHAPE it
    // is, and the shape is usually the mechanism -- a self-loop (parent[i]==i), a 2-cycle, or a ring of
    // stale generation are three different bugs and the count cannot tell them apart.
    for (uint32_t node : cycle_nodes) {
        if (out.sample_count >= 6u) break;
        out.sample_idx[out.sample_count] = node;
        out.sample_word[out.sample_count] = words[node];
        out.sample_next[out.sample_count] = (words[node] >> 3) & 0x7FFFFFFu;
        ++out.sample_count;
    }
    return out;
}

}   // namespace prosper::frontend

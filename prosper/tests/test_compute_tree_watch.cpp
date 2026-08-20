// Pure-analysis arms for the address-keyed tree watch (src/gpu/diagnostics/compute_tree_watch.hpp).
//
// Every arm here is written so that it FAILS if the property it names stops holding. That is not
// a truism in this file's history: three earlier diagnostics in this investigation shipped with
// assertions satisfied by something other than the mechanism under test (an inequality satisfied
// by a counter variable, an arm passing through an earlier empty() exit, a wrap guard whose wrap
// produced a value the comparison rejected anyway). Where an arm could pass for the wrong reason,
// it carries a companion arm that must fail if the mechanism is removed.
#include <cstdio>
#include <cstdint>
#include <span>
#include <vector>

#include "gpu/diagnostics/compute_tree_watch.hpp"

using namespace prosper::gpu;

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "  %s -> FAILED\n", what);
    ++g_failures;
}

void test_selector_parsing() {
    const auto minimal = parse_compute_tree_watch_selector("0x20f848417c:2063");
    check(minimal.has_value(), "minimal selector parses");
    if (minimal) {
        check(minimal->addr == 0x20f848417cull, "address parsed");
        check(minimal->records == 2063u, "record count parsed");
        check(minimal->index_shift == 3u, "shift defaults to the guest's bfe shift");
        check(minimal->index_mask == (1u << 27u) - 1u, "mask defaults to the guest's bfe width");
    }

    const auto full = parse_compute_tree_watch_selector("0x1000:16:5:0xff");
    check(full.has_value(), "full selector parses");
    if (full) {
        check(full->index_shift == 5u, "explicit shift overrides the default");
        check(full->index_mask == 0xffu, "explicit mask overrides the default");
    }

    check(!parse_compute_tree_watch_selector(nullptr).has_value(), "null rejected");
    check(!parse_compute_tree_watch_selector("").has_value(), "empty rejected");
    check(!parse_compute_tree_watch_selector("0x1000").has_value(), "missing record count rejected");
    check(!parse_compute_tree_watch_selector("0:16").has_value(), "zero address rejected");
    check(!parse_compute_tree_watch_selector("0x1000:0").has_value(), "zero record count rejected");
    check(!parse_compute_tree_watch_selector("0x1000:16:5").has_value(),
          "a shift without a mask is rejected rather than silently keeping the default mask");
    check(!parse_compute_tree_watch_selector("0x1000:16:32:0xff").has_value(),
          "a shift of 32 is rejected");
    check(!parse_compute_tree_watch_selector("0x1000:16:5:0").has_value(), "zero mask rejected");
    check(!parse_compute_tree_watch_selector("0x1000:16:5:0xff:9").has_value(),
          "trailing garbage rejected");
    check(!parse_compute_tree_watch_selector("0x1000:99999999").has_value(),
          "an absurd record count is refused at parse time, not discovered as a routed-run stall");
}

void test_deltas() {
    const std::vector<uint32_t> before{1, 2, 3, 4, 5};
    std::vector<uint32_t> after = before;
    uint32_t total = 0xdeadbeefu;

    auto none = compute_tree_watch_deltas(before, after, 8, &total);
    check(none.empty(), "identical buffers report no deltas");
    check(total == 0u, "identical buffers report zero total");

    after[1] = 20;
    after[4] = 50;
    total = 0;
    auto some = compute_tree_watch_deltas(before, after, 8, &total);
    check(some.size() == 2u, "two changed slots reported");
    check(total == 2u, "total matches the reported count when under the cap");
    if (some.size() == 2u) {
        check(some[0].index == 1u && some[0].before == 2u && some[0].after == 20u,
              "first delta carries index and both values");
        check(some[1].index == 4u && some[1].before == 5u && some[1].after == 50u,
              "second delta carries index and both values");
        check(some[0].index < some[1].index, "deltas ascend by index");
    }

    // The cap must truncate the LIST without truncating the COUNT -- a capped report that also
    // capped its total would read as a complete small change.
    total = 0;
    auto capped = compute_tree_watch_deltas(before, after, 1, &total);
    check(capped.size() == 1u, "the cap truncates the reported list");
    check(total == 2u, "the uncapped total survives truncation");

    // Sizes differing must compare only the overlap, or a resized observation reports every
    // trailing slot as a change and buries the real one.
    const std::vector<uint32_t> shorter{1, 2, 3};
    total = 0;
    auto overlap = compute_tree_watch_deltas(before, shorter, 8, &total);
    check(total == 0u, "a shorter observation compares only the overlap");
    check(overlap.empty(), "no deltas from a pure size difference");
}

ComputeParentWalkReport with_cycles(uint32_t cycles) {
    ComputeParentWalkReport report;
    report.distinct_cycles = cycles;
    return report;
}

void test_transition_classification() {
    const auto clean = with_cycles(0);
    const auto cyclic = with_cycles(3);

    check(classify_compute_tree_watch_transition(clean, cyclic, false) ==
              ComputeTreeWatchTransition::Unchanged,
          "unchanged bytes classify as unchanged even when the reports disagree");
    check(classify_compute_tree_watch_transition(clean, cyclic, true) ==
              ComputeTreeWatchTransition::CleanToCyclic,
          "clean to cyclic is named");
    check(classify_compute_tree_watch_transition(cyclic, clean, true) ==
              ComputeTreeWatchTransition::CyclicToClean,
          "cyclic to clean is named");
    check(classify_compute_tree_watch_transition(cyclic, cyclic, true) ==
              ComputeTreeWatchTransition::CyclicToCyclic,
          "cyclic to cyclic is named");
    check(classify_compute_tree_watch_transition(clean, clean, true) ==
              ComputeTreeWatchTransition::CleanToClean,
          "clean to clean is named");

    // Cyclicity is the ONLY axis. A depth change must not masquerade as a cyclicity transition,
    // or "clean" means one thing in a shallow run and another in a deep one.
    ComputeParentWalkReport deep = clean;
    deep.max_depth = 4096;
    check(classify_compute_tree_watch_transition(clean, deep, true) ==
              ComputeTreeWatchTransition::CleanToClean,
          "a depth change alone is not a cyclicity transition");
}

void test_sibling_structure() {
    // A full binary tree over N leaves has 2N-1 nodes. Build a small exact instance: pairs at
    // (1,2), (3,4), (5,6) with index 0 as the unpaired root.
    std::vector<uint32_t> words{0x00000010u,
                                0x00000020u, 0x40000020u,
                                0x00000030u, 0x40000030u,
                                0x00000040u, 0x40000040u};
    auto report = analyze_compute_tree_siblings(words);
    check(report.records == 7u, "record count reported");
    check(report.pairs == 3u, "three sibling pairs found");
    check(report.side_bit_set == 3u, "three side-bit records");
    check(report.unpaired_side == 0u, "a well-formed tree has no unpaired side records");

    // Breaking one head must be visible as exactly one unpaired record, and must NOT change the
    // side-bit population -- if it did, the metric would be counting the wrong thing.
    words[3] = 0x00000031u;
    auto broken = analyze_compute_tree_siblings(words);
    check(broken.pairs == 2u, "breaking a head loses exactly one pair");
    check(broken.unpaired_side == 1u, "breaking a head yields exactly one unpaired record");
    check(broken.side_bit_set == 3u, "breaking a head does not change the side-bit population");

    // Two identical adjacent records both carrying the side bit are duplicates, not siblings.
    // Without the head-clear clause `words[i] == (words[i-1] | bit)` is trivially satisfied here,
    // so this arm fails if that clause is removed.
    const std::vector<uint32_t> duplicates{0x00000010u, 0x40000020u, 0x40000020u};
    auto dup = analyze_compute_tree_siblings(duplicates);
    check(dup.pairs == 0u,
          "two identical side-bit records are duplicates, not a sibling pair");
    check(dup.unpaired_side == 2u, "both duplicates count as unpaired");

    // Index 0 carrying the side bit cannot pair: it has no predecessor.
    const std::vector<uint32_t> leading{0x40000010u, 0x00000020u};
    auto lead = analyze_compute_tree_siblings(leading);
    check(lead.pairs == 0u, "index zero cannot form a pair");
    check(lead.unpaired_side == 1u, "index zero with the side bit counts as unpaired");

    check(analyze_compute_tree_siblings({}).records == 0u, "an empty span is handled");
}

// The watch's whole purpose is to separate "this dispatch made the table cyclic" from "the table
// was already cyclic". That separation is only real if the underlying walk actually detects the
// cycle in a table shaped like the guest's, so build one and confirm both halves.
void test_walk_agrees_on_a_guest_shaped_table() {
    constexpr uint32_t kRecords = 63;   // 2 * 32 - 1
    constexpr uint32_t kShift = 3;
    constexpr uint32_t kMask = (1u << 27u) - 1u;
    auto make_record = [](uint32_t parent) { return parent << kShift; };

    // A proper tree: every node points at floor(i/2), so every walk terminates at index 0.
    std::vector<uint32_t> tree(kRecords, 0u);
    for (uint32_t i = 1; i < kRecords; ++i) tree[i] = make_record(i / 2u);
    auto clean = analyze_compute_parent_walk(tree, kRecords, kShift, kMask);
    check(clean.distinct_cycles == 0u, "a proper parent tree walks clean");
    check(clean.cyclic_roots == 0u, "a proper parent tree has no cyclic roots");
    check(clean.max_depth > 0u, "a proper parent tree has nonzero depth");

    // The observed corruption shape: a record's parent link points back down at its own child, so
    // two nodes point at each other and every root below them never reaches zero.
    std::vector<uint32_t> cyclic = tree;
    cyclic[10] = make_record(21u);   // 21's parent is 10; now 10's parent is 21
    auto broken = analyze_compute_parent_walk(cyclic, kRecords, kShift, kMask);
    check(broken.distinct_cycles == 1u, "a mutual parent link is detected as one cycle");
    check(broken.cyclic_roots > 0u, "the cycle traps at least one root");

    uint32_t total = 0;
    auto deltas = compute_tree_watch_deltas(tree, cyclic, 8, &total);
    check(total == 1u, "exactly one dword separates the clean and cyclic tables");
    if (!deltas.empty()) check(deltas[0].index == 10u, "the delta names the corrupted slot");
    check(classify_compute_tree_watch_transition(clean, broken, total != 0u) ==
              ComputeTreeWatchTransition::CleanToCyclic,
          "the clean-to-cyclic transition is classified on a guest-shaped table");
}

} // namespace

int main() {
    std::printf("test_compute_tree_watch\n");
    test_selector_parsing();
    test_deltas();
    test_transition_classification();
    test_sibling_structure();
    test_walk_agrees_on_a_guest_shaped_table();
    if (g_failures) {
        std::fprintf(stderr, "== FAILURES: %d ==\n", g_failures);
        return 1;
    }
    std::printf("  all passed\n");
    return 0;
}

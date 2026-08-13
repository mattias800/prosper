#include "gpu/compute_parent_walk.hpp"

#include <array>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

namespace {

int failures = 0;
int checks = 0;
#define CHECK(condition, message) do { \
    ++checks; \
    if (!(condition)) { std::printf("FAIL: %s\n", message); ++failures; } \
} while (0)

} // namespace

int main() {
    const auto selector = parse_compute_parent_walk_selector(
        "0x413dc6700:91:3:0x07ffffff");
    CHECK(selector && selector->program_addr == 0x413dc6700ull &&
              selector->fetch_pc == 91u && selector->index_shift == 3u &&
              selector->index_mask == 0x07ffffffu && selector->deep_threshold == 64u,
          "four-field selector parses with the conservative default depth");
    const auto explicit_depth = parse_compute_parent_walk_selector(
        "0x413dc6700:91:3:0x07ffffff:96");
    CHECK(explicit_depth && explicit_depth->deep_threshold == 96u,
          "selector accepts an explicit deep-walk threshold");
    CHECK(!parse_compute_parent_walk_selector("0:91:3:0x07ffffff") &&
              !parse_compute_parent_walk_selector("1:91:32:1") &&
              !parse_compute_parent_walk_selector("1:91:3:0") &&
              !parse_compute_parent_walk_selector("1:91:3:1:0") &&
              !parse_compute_parent_walk_selector("1:91:3:1:2:3"),
          "invalid selectors fail closed");

    // The same next-index representation used by GTA V 0x413dc6700. Root 1 follows a six-load
    // in-range chain; root 2 performs one robust OOB load before zero terminates it.
    std::array<uint32_t, 8> healthy{};
    healthy[1] = 3u << 3u;
    healthy[3] = 4u << 3u;
    healthy[4] = 5u << 3u;
    healthy[5] = 6u << 3u;
    healthy[6] = 7u << 3u;
    healthy[7] = 0u;
    healthy[2] = 99u << 3u;
    const ComputeParentWalkReport healthy_report = analyze_compute_parent_walk(
        healthy, healthy.size(), 3u, 0x07ffffffu);
    CHECK(healthy_report.records == 8u && healthy_report.roots == 8u &&
              healthy_report.distinct_cycles == 0u && healthy_report.cyclic_roots == 0u &&
              healthy_report.oob_roots == 1u && healthy_report.max_depth == 6u &&
              healthy_report.max_depth_root == 1u &&
              !compute_parent_walk_suspicious(healthy_report, 64u),
          "acyclic graph reports robust OOB termination and the exact maximum depth");

    // Mutation arm: perturb the exact parent word consumed by the analyzer, not a helper or an
    // assertion. Changing the end of root 1's chain into a self-link must flip the skip policy.
    auto cyclic = healthy;
    cyclic[7] = 7u << 3u;
    const ComputeParentWalkReport cyclic_report = analyze_compute_parent_walk(
        cyclic, cyclic.size(), 3u, 0x07ffffffu);
    CHECK(cyclic_report.distinct_cycles == 1u && cyclic_report.cyclic_roots == 6u &&
              compute_parent_walk_suspicious(cyclic_report, 64u),
          "same-site parent-word mutation exposes a cycle and trips the diagnostic skip policy");

    std::vector<uint32_t> deep(70u, 0u);
    for (uint32_t index = 1; index + 1u < deep.size(); ++index)
        deep[index] = (index + 1u) << 3u;
    const ComputeParentWalkReport deep_report = analyze_compute_parent_walk(
        deep, deep.size(), 3u, 0x07ffffffu);
    CHECK(deep_report.distinct_cycles == 0u && deep_report.max_depth == 69u &&
              compute_parent_walk_suspicious(deep_report, 64u) &&
              !compute_parent_walk_suspicious(deep_report, 69u),
          "acyclic depth is diagnostic-thresholded without changing walk semantics");

    if (failures) {
        std::printf("== FAIL: %d == (%d assertions executed)\n", failures, checks);
        return 1;
    }
    std::printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}

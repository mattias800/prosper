#include "fixtures/buffer_range_plan.h"
#include <cstdio>
#include <limits>
using namespace prosper::test;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* message) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
        failures += !ok;
    };
    constexpr uint64_t Max = 65536;
    auto groups = plan_buffer_ranges({{4096,8192},{5120,8192},{4096,8192}},256,Max);
    check(groups.size() == 1 && groups[0].address == 4096 && groups[0].bytes == 9216 &&
          groups[0].distinct_bytes == 16384, "deduplicate exact spans before counting saved bytes");
    check(find_buffer_range(groups,5120,8192,256) == 0 &&
          find_buffer_range(groups,5124,8192,256) == groups.size() &&
          find_buffer_range(groups,5120,8196,256) == groups.size(),
          "lookups require aligned containment of the exact requested range");
    check(plan_buffer_ranges({{4096,8192},{4096,8192}},4,Max).empty(),
          "exact duplicates alone add no benefit over existing reference memo");
    check(plan_buffer_ranges({{4096,8192},{4100,8192}},4,Max).size() == 1 &&
          plan_buffer_ranges({{4096,8192},{4100,8192}},16,Max).empty() &&
          plan_buffer_ranges({{4096,8192},{4112,8192}},16,Max).size() == 1,
          "real device alignment controls admissible shifted spans");
    groups = plan_buffer_ranges({{4096,16384},{8192,8192}},4,Max);
    check(groups.size() == 1 && groups[0].bytes == 16384 && groups[0].distinct_bytes == 24576,
          "fully contained distinct ranges share without changing outer extent");
    groups = plan_buffer_ranges({{4096,8192},{8192,8192},{12288,8192}},256,Max);
    check(groups.size() == 1 && groups[0].bytes == 16384,
          "transitive connected overlaps form one gap-free upload");
    check(plan_buffer_ranges({{4096,8192},{12288,8192}},4,Max).empty() &&
          plan_buffer_ranges({{4096,8192},{12304,8192}},4,Max).empty(),
          "adjacency and gaps do not save any copied bytes");
    check(plan_buffer_ranges({{4096,8192},{8192,8192}},4,8192).empty(),
          "a union exceeding the configured size limit falls back");
    groups = plan_buffer_ranges({{4096,8192},{8192,8192},{12288,8192},{16384,8192}},4,12288);
    check(groups.size() == 2 && find_buffer_range(groups,12288,8192,4) == 1 &&
          find_buffer_range(groups,24576,4096,4) == groups.size(),
          "capped overlapping groups select only a complete containing owner");
    constexpr auto U64 = std::numeric_limits<uint64_t>::max();
    check(plan_buffer_ranges({{0,8192},{4096,0},{4096,4095},{U64-4096,8192},
                             {4096,U64}},4,Max).empty() &&
          plan_buffer_ranges({{4096,8192},{4100,8192}},0,Max).empty(),
          "invalid, small, overflowing and zero-alignment metadata is declined");
    return failures ? 1 : 0;
}

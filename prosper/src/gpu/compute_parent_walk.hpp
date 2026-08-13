// Command-ordered diagnostic for compute kernels that chase an index through a buffer.
// Pure analysis and selector parsing live here; gpu_executor owns live resource access and skipping.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace prosper::gpu {

struct ComputeParentWalkSelector {
    uint64_t program_addr = 0;
    uint32_t fetch_pc = UINT32_MAX;
    uint32_t index_shift = 0;
    uint32_t index_mask = UINT32_MAX;
    uint32_t deep_threshold = 64;
};

// Strict diagnostic selector syntax:
//   PROGRAM:FETCH_PC:SHIFT:MASK[:DEEP_THRESHOLD]
// All fields accept C integer syntax. The optional threshold defaults to 64.
std::optional<ComputeParentWalkSelector> parse_compute_parent_walk_selector(
    const char* value);

struct ComputeParentWalkReport {
    uint32_t records = 0;
    uint32_t roots = 0;
    uint32_t distinct_cycles = 0;
    uint32_t cyclic_roots = 0;
    uint32_t oob_roots = 0;
    uint32_t max_depth = 0;
    uint32_t max_depth_root = 0;
};

// Model the common loop:
//   while (index != 0) index = (records[index] >> shift) & mask;
// An out-of-range raw-buffer load is modelled as returning zero and terminating. `root_count`
// starts roots [0, root_count), bounded to the record array.
ComputeParentWalkReport analyze_compute_parent_walk(
    std::span<const uint32_t> records, uint32_t root_count,
    uint32_t index_shift, uint32_t index_mask);

bool compute_parent_walk_suspicious(
    const ComputeParentWalkReport& report, uint32_t deep_threshold);

} // namespace prosper::gpu

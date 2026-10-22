// Barrier-separated compute phase recognition; private to the recompiler.
#pragma once
#include <cstddef>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;

struct BarrierPhasedCompute {
    size_t guard_index = 0;
    size_t end_index = 0;
    std::vector<size_t> barriers;
    bool guarded = false;
    bool found = false;
};

bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, size_t guard_index);
BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins);
}  // namespace prosper::gpu

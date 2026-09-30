#pragma once
// The value used by one WAIT_REG_MEM predicate, retained before later diagnostic reads.
// This does not publish memory or change the wait policy. Readers and overlays are supplied by
// the command processor so sampling keeps its existing probe/copy and pending-queue lock.
#include "gpu/pm4/pm4_decode.hpp"

namespace prosper::gpu {

struct WaitRegMemPredicateSample {
    bool readable = false;
    bool overlay_touched = false;
    bool comparison_supported = false;
    bool satisfied = false;
    uint64_t raw_value = 0;
    uint64_t effective_value = 0;
    uint64_t masked_value = 0;
};

inline bool wait_regmem_value_satisfied(const Pm4Command& c, uint64_t memory_value) {
    const uint64_t v = memory_value & c.wm_mask, r = c.wm_ref;
    switch (c.wm_func) {
        case 0: return true;
        case 1: return v <  r;
        case 2: return v <= r;
        case 3: return v == r;
        case 4: return v != r;
        case 5: return v >= r;
        case 6: return v >  r;
        default: return false;
    }
}

template<class ReadQword, class OverlayQword>
WaitRegMemPredicateSample sample_wait_regmem_predicate(
        const Pm4Command& c, ReadQword&& read_qword, OverlayQword&& overlay_qword) {
    WaitRegMemPredicateSample sample;
    sample.comparison_supported = c.wm_func <= 6;
    if (!read_qword(c.wm_addr, &sample.raw_value)) return sample;
    sample.readable = true;
    sample.effective_value = sample.raw_value;
    sample.overlay_touched = overlay_qword(c.wm_addr, &sample.effective_value);
    sample.masked_value = sample.effective_value & c.wm_mask;
    sample.satisfied = wait_regmem_value_satisfied(c, sample.effective_value);
    return sample;
}

}  // namespace prosper::gpu

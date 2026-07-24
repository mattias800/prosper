#pragma once

namespace prosper::frontend {

// An intermediate scanout normally stays GPU-resident until the final presentation callback.
// A later non-direct consumer in the same submit is different: it needs authoritative CPU bytes
// before the batch is submitted, so deferring here would make it observe the previous submit.
constexpr bool can_defer_intermediate_scanout_readback(bool phase_allows_defer,
                                                        bool defer_intermediate_scanout,
                                                        bool cpu_needed_same_batch) {
    return phase_allows_defer && defer_intermediate_scanout && !cpu_needed_same_batch;
}

} // namespace prosper::frontend

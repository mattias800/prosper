#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;

// A structured join can carry a predicate only when both predecessors have its current
// Bool spelling. An absent spelling is not the physical pair's value on that predecessor.
bool join_structured_saved_masks(SpirvCompute& b, RegState& current,
                                 const std::unordered_map<int, uint32_t>& other,
                                 uint32_t other_block, uint32_t current_block,
                                 const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                 uint32_t merge_pc);
} // namespace prosper::gpu

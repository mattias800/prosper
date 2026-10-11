#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;

struct StructuredSavedMaskEdge {
    std::unordered_map<int, uint32_t> predicates;
    std::unordered_set<int> b32;
    std::set<int> current_data_words;
};
// Capture before scalar PHIs can seed an absent physical word with zero. Current DATA is a
// separate predecessor fact; an old launch spelling or the merged register map is not one.
StructuredSavedMaskEdge structured_saved_mask_edge(const SpirvCompute& b, const RegState& rs);

// A structured join can carry a predicate only when both predecessors have its current
// Bool spelling. An absent spelling is not the physical pair's value on that predecessor.
bool join_structured_saved_masks(SpirvCompute& b, RegState& current,
                                 const StructuredSavedMaskEdge& other,
                                 const StructuredSavedMaskEdge& here, uint32_t other_block,
                                 uint32_t current_block, const std::vector<Rdna2Inst>& ins,
                                 uint32_t branch_pc, uint32_t merge_pc);
} // namespace prosper::gpu

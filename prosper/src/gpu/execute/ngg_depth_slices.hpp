// ngg_depth_slices.hpp -- expanding a layered depth-only NGG draw into one draw per depth slice
// (#3135 layered NGG depth output).
//
// A depth-only NGG draw whose primitives choose their slice (the layer the primitive shader writes)
// into a depth array is realized as one replay per slice (ngg_live_draw.hpp): replay k draws only
// the primitives whose layer is k. The backend's depth model is one image per (surface, slice), the
// same shape a guest's own face-by-face cube render takes, so each replay becomes an ordinary item
// whose DB_DEPTH_VIEW names exactly its slice. The registered submit renderer is wrapped so it
// receives every submit expanded (gpu_executor.cpp set_submit_renderer), and every existing path
// (pass grouping by depth identity, the persistent depth key, sampling) treats it as the face
// render it is.
//
// Order. Draws into different slices touch different images, so only the order WITHIN a slice is
// observable. A run of consecutive expanded items sharing one depth surface and view is emitted
// slice-major -- every item's slice 0, then every item's slice 1, ... -- which keeps each slice's
// draws in guest order and makes them contiguous, so the pass grouper forms one pass per slice
// rather than one per (item, slice). Nothing else is reordered. CONFIDENCE: HIGH for the per-slice
// order; the depth-only precondition is the admission's (ngg_draw_admission.hpp), and a depth-only
// draw writes no colour, so moving one past another can change nothing but its own slice.
#pragma once

#include <cstdint>
#include <vector>

namespace prosper::gpu {

struct DrawItem;

// DB_DEPTH_VIEW narrowed to `slice`: SLICE_START = SLICE_MAX = slice (both with their high bits),
// every other field (read-only flags, MIPID) unchanged.
uint32_t depth_view_for_slice(uint32_t db_depth_view, uint32_t slice);

// When any item carries ngg_depth_slices, writes the expanded list to `out` and returns true: each
// such item becomes one item per slice (item k's ngg_subgroup is slice k's replay and its
// DB_DEPTH_VIEW names slice first + k; none still carries ngg_depth_slices), runs ordered as above.
// Returns false and leaves `out` untouched when there is nothing to expand.
bool expand_ngg_depth_slices(const std::vector<DrawItem>& items, std::vector<DrawItem>& out);

}   // namespace prosper::gpu

// ngg_depth_slices.cpp -- see ngg_depth_slices.hpp.
#include "gpu/execute/ngg_depth_slices.hpp"

#include <tuple>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"

namespace prosper::gpu {

uint32_t depth_view_for_slice(uint32_t db_depth_view, uint32_t slice) {
    namespace P = prosper::agc::Pm4;
    const uint32_t lo = slice & P::DB_DEPTH_VIEW_SLICE_START_MASK;
    const uint32_t hi = (slice >> 11) & P::DB_DEPTH_VIEW_SLICE_START_HI_MASK;
    const uint32_t cleared =
        db_depth_view &
        ~((P::DB_DEPTH_VIEW_SLICE_START_MASK << P::DB_DEPTH_VIEW_SLICE_START_SHIFT) |
          (P::DB_DEPTH_VIEW_SLICE_START_HI_MASK << P::DB_DEPTH_VIEW_SLICE_START_HI_SHIFT) |
          (P::DB_DEPTH_VIEW_SLICE_MAX_MASK << P::DB_DEPTH_VIEW_SLICE_MAX_SHIFT) |
          (P::DB_DEPTH_VIEW_SLICE_MAX_HI_MASK << P::DB_DEPTH_VIEW_SLICE_MAX_HI_SHIFT));
    return cleared | (lo << P::DB_DEPTH_VIEW_SLICE_START_SHIFT) |
           (hi << P::DB_DEPTH_VIEW_SLICE_START_HI_SHIFT) |
           (lo << P::DB_DEPTH_VIEW_SLICE_MAX_SHIFT) | (hi << P::DB_DEPTH_VIEW_SLICE_MAX_HI_SHIFT);
}

namespace {

bool fans_out(const DrawItem& item) {
    return item.ngg_subgroup && item.ngg_depth_slice_count > 1u;
}

// Items of one run share every input of the depth pass identity and the slice range, so their
// slice k items all land in the same slice of the same surface.
auto run_key(const DrawItem& item) {
    return std::tuple(item.ps.depth_read_base, item.ps.depth_write_base, item.ps.stencil_read_base,
                      item.ps.stencil_write_base, item.ps.htile_data_base, item.ps.db_depth_size_xy,
                      item.ps.db_depth_view, item.ngg_depth_first_slice,
                      item.ngg_depth_slice_count);
}

// Turns `item` into its slice-k replay in place: same description, layer k, its own slice.
void make_slice(DrawItem& item, uint32_t first, uint32_t view, uint32_t k) {
    item.ngg_layer_select = k;
    item.ngg_depth_slice_count = 0;
    item.ngg_depth_first_slice = 0;
    item.ps.db_depth_view = depth_view_for_slice(view, first + k);
}

// The expansion of `items`, slice-major per run. `fetch(j, last)` yields item j: a copy, or -- on
// its last use, when the caller owns the items -- the item itself.
template <typename Fetch>
std::vector<DrawItem> expand(const std::vector<DrawItem>& items, Fetch fetch) {
    std::vector<DrawItem> out;
    out.reserve(items.size() + 8u);
    for (size_t i = 0; i < items.size();) {
        if (!fans_out(items[i])) {
            out.push_back(fetch(i, true));
            out.back().ngg_depth_slice_count = 0;   // one slice is the draw itself
            ++i;
            continue;
        }
        const auto key = run_key(items[i]);
        size_t end = i + 1;
        while (end < items.size() && fans_out(items[end]) && run_key(items[end]) == key) ++end;
        const uint32_t slices = items[i].ngg_depth_slice_count;
        const uint32_t first = items[i].ngg_depth_first_slice;
        const uint32_t view = items[i].ps.db_depth_view;
        for (uint32_t k = 0; k < slices; ++k)
            for (size_t j = i; j < end; ++j) {
                out.push_back(fetch(j, k + 1u == slices));
                make_slice(out.back(), first, view, k);
            }
        i = end;
    }
    return out;
}

bool any_fans_out(const std::vector<DrawItem>& items) {
    for (const DrawItem& item : items)
        if (fans_out(item)) return true;
    return false;
}

}  // namespace

bool expand_ngg_depth_slices(const std::vector<DrawItem>& items, std::vector<DrawItem>& out) {
    if (!any_fans_out(items)) return false;
    out = expand(items, [&](size_t j, bool) { return items[j]; });
    return true;
}

bool expand_ngg_depth_slices_in_place(std::vector<DrawItem>& items) {
    if (!any_fans_out(items)) return false;
    std::vector<DrawItem> expanded = expand(items, [&](size_t j, bool last) {
        return last ? std::move(items[j]) : DrawItem(items[j]);
    });
    items = std::move(expanded);
    return true;
}

}  // namespace prosper::gpu

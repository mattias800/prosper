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
    return item.ngg_subgroup && item.ngg_depth_slices.size() > 1u;
}

// Items of one run share every input of the depth pass identity and the slice range, so their
// slice k items all land in the same slice of the same surface.
auto run_key(const DrawItem& item) {
    return std::tuple(item.ps.depth_read_base, item.ps.depth_write_base, item.ps.stencil_read_base,
                      item.ps.stencil_write_base, item.ps.htile_data_base, item.ps.db_depth_size_xy,
                      item.ps.db_depth_view, item.ngg_depth_first_slice,
                      item.ngg_depth_slices.size());
}

DrawItem slice_item(const DrawItem& item, size_t k) {
    DrawItem out = item;
    out.ngg_subgroup = item.ngg_depth_slices[k];
    out.ngg_depth_slices.clear();
    out.ngg_depth_first_slice = 0;
    out.ps.db_depth_view = depth_view_for_slice(
        item.ps.db_depth_view, item.ngg_depth_first_slice + static_cast<uint32_t>(k));
    return out;
}

}  // namespace

bool expand_ngg_depth_slices(const std::vector<DrawItem>& items, std::vector<DrawItem>& out) {
    bool any = false;
    for (const DrawItem& item : items) any = any || fans_out(item);
    if (!any) return false;
    std::vector<DrawItem> expanded;
    expanded.reserve(items.size() + 8u);
    for (size_t i = 0; i < items.size();) {
        if (!fans_out(items[i])) {
            DrawItem plain = items[i];
            plain.ngg_depth_slices.clear();   // a one-slice list is the draw itself
            expanded.push_back(std::move(plain));
            ++i;
            continue;
        }
        const auto key = run_key(items[i]);
        size_t end = i + 1;
        while (end < items.size() && fans_out(items[end]) && run_key(items[end]) == key) ++end;
        const size_t slices = items[i].ngg_depth_slices.size();
        for (size_t k = 0; k < slices; ++k)
            for (size_t j = i; j < end; ++j) expanded.push_back(slice_item(items[j], k));
        i = end;
    }
    out = std::move(expanded);
    return true;
}

}  // namespace prosper::gpu

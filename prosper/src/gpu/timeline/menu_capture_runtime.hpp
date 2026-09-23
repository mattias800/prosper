#pragma once

#include "gpu/present/videoout_present.hpp"

#include <cstdint>
#include <span>

namespace prosper::gpu {
struct GpuState;

// Tiny identity trace from draws actually admitted by the ordered executor. It contains no shader,
// descriptor, or guest resource payload and is only populated for an armed diagnostic run.
struct MenuRealizedDrawIdentity {
    uint64_t draw_index = 0;
    uint64_t fs_guest_addr = 0;
    uint64_t color0_base = 0;
    uint32_t color0_width = 0, color0_height = 0;
    uint32_t write_mask = 0;
};

// Opt-in diagnostic. The submit hook runs before renderer sampling; the publication hook receives
// the exact completed submit and its immutable native pixels. Neither hook selects by elapsed time
// or assumes an old screenshot belongs to a later captured draw.
void menu_draw_capture_on_submit(const GpuState& state, uint64_t submit_no);
void menu_draw_capture_on_publication(const GpuState& state,
                                      std::span<const MenuRealizedDrawIdentity> realized_draws,
                                      uint64_t execution_submit, uint64_t source_submit,
                                      bool published,
                                      PresentFrameOrigin origin,
                                      std::span<const uint8_t> rgba,
                                      uint32_t width, uint32_t height);
} // namespace prosper::gpu

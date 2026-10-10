// CPU-visible indirect draw argument realization. Ordered producer admission remains with the
// submit executor; this function neither retires producers nor authorizes stale memory reads.
#pragma once
#include "gpu/pm4/command_processor.hpp"

namespace prosper::gpu {

bool resolve_indirect_draw_arguments(const GpuState& submit, const GpuState::Draw& source,
                                     GpuState::Draw& resolved);

} // namespace prosper::gpu

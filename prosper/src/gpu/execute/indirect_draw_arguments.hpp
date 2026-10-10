// Indirect draw source selection and realization. Ordered producer admission remains with the
// submit executor; these helpers neither retire producers nor authorize stale memory reads.
#pragma once
#include "gpu/pm4/command_processor.hpp"
#include <span>

namespace prosper::gpu {

// Called only after ordered producer retirement. A claimed renderer range owns the bytes;
// InvalidRange or a truncated response must not authorize a stale CPU fallback.
bool read_indirect_draw_argument_source(const GpuState::Draw& source, std::vector<uint8_t>& owned);

bool resolve_indirect_draw_arguments(const GpuState& submit, const GpuState::Draw& source,
                                     GpuState::Draw& resolved, std::span<const uint8_t> owned = {});

} // namespace prosper::gpu

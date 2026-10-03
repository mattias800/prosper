#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <utility>

namespace prosper::gpu {
struct SpirvCompute;

// Each static row event publishes only SRC0 for peer lookup. SRC1 stays at its own invocation.
struct ComputeDppRowRor8PhaseVariables {
    uint32_t pending, active, source0, source1, operation, destination, event;
};

bool emit_portable_compute_dpp_row_ror8_phase(
    SpirvCompute& builder, const ComputeDppRowRor8PhaseVariables& variables, uint32_t value_base,
    uint32_t metadata_base, const std::set<int>& destinations, const std::map<int, uint32_t>& vgprs,
    const std::map<std::pair<int, int>, uint32_t>& numeric_lane_aliases,
    const std::map<std::pair<int, int>, uint32_t>& mask_lane_aliases, bool floating_add);
}  // namespace prosper::gpu

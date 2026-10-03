#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <map>
#include <set>
#include <utility>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;

// Full-mask, unmodified BC0 integer maximum whose two inputs and output are one VGPR.
// Distinct registers and bounded/masked forms need their own invalid-source write contract.
bool is_inplace_vmax_u32_dpp_row_shr(const Rdna2Inst& instruction);

// The caller still owns EXEC predication and physical VGPR alias expiration. Maximum sites
// require a wave-uniform emission point; divergent dispatcher sites need tagged common phases.
bool emit_compute_inplace_dpp_row_shr(SpirvCompute& builder, const RegState& registers,
                                      const Rdna2Inst& instruction, uint32_t source,
                                      uint32_t old_destination, bool wave_uniform,
                                      uint32_t& result);
// Function-variable IDs shared by the event-isolated dispatcher phase.
struct ComputeDppRowShrPhaseVariables {
    uint32_t pending, active, source, amount, destination, event;
};

bool emit_portable_compute_dpp_row_shr_phase(
    SpirvCompute& builder, const ComputeDppRowShrPhaseVariables& variables,
    uint32_t value_base, uint32_t metadata_base, const std::set<int>& destinations,
    const std::map<int, uint32_t>& vgprs,
    const std::map<std::pair<int, int>, uint32_t>& numeric_lane_aliases,
    const std::map<std::pair<int, int>, uint32_t>& mask_lane_aliases);
}  // namespace prosper::gpu

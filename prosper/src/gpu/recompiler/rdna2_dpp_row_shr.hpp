#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

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
}  // namespace prosper::gpu

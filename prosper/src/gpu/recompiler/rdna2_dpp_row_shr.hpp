#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;

// Full-mask, unmodified BC0 integer maximum whose two inputs and output are one VGPR.
// Distinct registers and bounded/masked forms need their own invalid-source write contract.
bool is_inplace_vmax_u32_dpp_row_shr(const Rdna2Inst& instruction);
bool is_inplace_vor_b32_dpp_row_shr(const Rdna2Inst& instruction);
bool is_compute_dpp_row_shr(const Rdna2Inst& instruction, bool ngg_workgroup_shell);
bool needs_compute_dpp_row_shr_dispatcher(const std::vector<Rdna2Inst>& instructions,
                                          bool ngg_workgroup_shell, uint32_t native_subgroup_size);

// Current ordinary per-lane sources, excluding a spill still live on any CFG predecessor.
// This tracks spill lifetimes; it does not replace operand initialization or entry-ABI checks.
std::map<uint32_t, std::set<int>>
ngg_row_sources_without_live_spills(const std::vector<Rdna2Inst>& instructions,
                                    const RegState& entry, bool* terminal_full_exec = nullptr,
                                    std::set<uint32_t>* full_exec_pcs = nullptr);
// Prepare from the complete selected body before barrier splitting, not a reloaded phase entry.
void prepare_ngg_row_source_proof(SpirvCompute& builder, const RegState& entry,
                                  const std::vector<Rdna2Inst>& instructions);
void restore_ngg_row_source_state(SpirvCompute& builder, RegState& registers,
                                  const std::map<uint32_t, std::set<int>>& ordinary_sources,
                                  const std::vector<uint32_t>& block_starts,
                                  const std::vector<std::vector<uint32_t>>& dispatch_blocks,
                                  uint32_t dispatch, uint32_t end_pc, bool terminal_full_exec);
bool resolve_compute_dpp_row_source(SpirvCompute& builder, RegState& registers,
                                    const Rdna2Inst& instruction,
                                    const std::map<uint32_t, std::set<int>>& ordinary_sources,
                                    uint32_t& source);

// The caller still owns EXEC predication and physical VGPR alias expiration. MAX/OR sites
// require a wave-uniform emission point; divergent dispatcher sites need tagged common phases.
bool emit_compute_inplace_dpp_row_shr(SpirvCompute& builder, const RegState& registers,
                                      const Rdna2Inst& instruction, uint32_t source,
                                      uint32_t old_destination, bool wave_uniform,
                                      uint32_t& result);
// Function-variable IDs shared by the synchronized dispatcher. maximum=0 preserves the
// established ADD-only emission; otherwise each pending static site supplies ADD=false/MAX=true.
// bitwise_or=0 likewise preserves every stream without an OR ladder.
struct ComputeDppRowShrPhaseVariables {
    uint32_t pending, active, source, amount, destination, event, maximum, bitwise_or;
};

bool emit_portable_compute_dpp_row_shr_phase(
    SpirvCompute& builder, const ComputeDppRowShrPhaseVariables& variables, uint32_t value_base,
    uint32_t metadata_base, const std::set<int>& destinations, const std::map<int, uint32_t>& vgprs,
    const std::map<std::pair<int, int>, uint32_t>& numeric_lane_aliases,
    const std::map<std::pair<int, int>, uint32_t>& mask_lane_aliases);
}  // namespace prosper::gpu

// The SMEM pointer/raw-data provenance seed every compute and fragment shell runs once before
// emission. Moved out of the capped CFG translation unit (rdna2_emit_cfg.cpp) unchanged.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

void seed_smem_pointer_provenance(RegState& rs, const std::vector<Rdna2Inst>& ins) {
    if (rs.smem_pointer_analysis_done) return;
    rs.smem_pointer_loads = rdna2_proven_smem_pointer_loads(ins);
    rs.smem_owned_raw_x2_chains = rdna2_owned_raw_x2_chains(ins);
    for (const auto& chain : rs.smem_owned_raw_x2_chains) {
        rs.smem_raw_x2_data_loads.insert(chain.parent_pc);
        rs.smem_raw_x2_data_loads.insert(chain.child_pc);
    }
    const auto raw_x2_data = rdna2_proven_raw_x2_data_loads(ins);
    rs.smem_raw_x2_data_loads.insert(raw_x2_data.begin(), raw_x2_data.end());
    const auto raw_immediate_wide_data = rdna2_proven_raw_immediate_wide_data_loads(ins);
    rs.smem_raw_immediate_wide_data_loads.insert(raw_immediate_wide_data.begin(),
                                                 raw_immediate_wide_data.end());
    const auto owned_wide_data = rdna2_owned_raw_wide_data_loads(ins);
    rs.smem_raw_owned_wide_data_loads.insert(owned_wide_data.begin(), owned_wide_data.end());
    std::vector<uint32_t> raw_offset_scalar_sources;
    const auto raw_register_wide_data =
        rdna2_proven_raw_register_wide_data_loads(ins, &raw_offset_scalar_sources);
    rs.smem_raw_offset_scalar_source_pcs.insert(raw_offset_scalar_sources.begin(),
                                                raw_offset_scalar_sources.end());
    rs.smem_raw_register_wide_data_loads.insert(raw_register_wide_data.begin(),
                                                raw_register_wide_data.end());
    const auto raw_nested_wide_data = rdna2_proven_raw_nested_wide_data_loads(ins);
    rs.smem_owned_nested_wide_chains = rdna2_owned_nested_wide_chains(ins);
    rs.smem_raw_nested_wide_data_loads.insert(raw_nested_wide_data.begin(),
                                              raw_nested_wide_data.end());
    const auto raw_wide_data = rdna2_raw_wide_data_loads(ins);
    rs.smem_raw_wide_data_loads.insert(raw_wide_data.begin(), raw_wide_data.end());
    rs.smem_pointer_analysis_done = true;
}

}  // namespace prosper::gpu

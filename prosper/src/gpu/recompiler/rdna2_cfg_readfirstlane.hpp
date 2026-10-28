// Existing common-phase READFIRST gather; fields are builder IDs and scratch locations only.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct CfgReadfirstlanePhase {
    uint32_t initial_active;
    uint32_t padded_lanes;
    uint32_t mbcnt_lane;
    uint32_t mbcnt_wave_index;
    uint32_t wave_result_base;
    uint32_t zero;
    uint32_t first_pending_var;
    uint32_t first_event_var;
    uint32_t first_exec_var;
    uint32_t first_source_var;
    uint32_t first_dst_var;
};
struct CfgReadfirstlaneResult {
    uint32_t pending, dst, result;
};
CfgReadfirstlaneResult emit_cfg_readfirstlane_gather(SpirvCompute&, const CfgReadfirstlanePhase&);
} // namespace prosper::gpu

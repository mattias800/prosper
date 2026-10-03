#include "gpu/execute/shader_cache_internal.hpp"

namespace prosper::gpu {
std::shared_ptr<const FragmentPacketVgprRequirements> shader_analysis_packet_vgpr_requirements(
        const SharedShaderAnalysis& analysis) {
    // Aliasing ownership pins BOTH the immutable requirements and the exact raw byte version.
    // Warm draws only copy this owner: no reparse, guest-memory read or added process-global lock.
    return analysis ? std::shared_ptr<const FragmentPacketVgprRequirements>(analysis,
        &analysis->packet_vgpr_requirements) : nullptr;
}
} // namespace prosper::gpu

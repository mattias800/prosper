// A missing register retains the legacy raw zero, never an observed FP16 overflow mode.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/pm4_registers.hpp"

namespace prosper::gpu {
void populate_compute_launch_rsrc1(ComputeShaderConfig& config, const RegisterFile& registers) {
    const auto found = registers.find(prosper::agc::Pm4::COMPUTE_PGM_RSRC1);
    const bool observed = found != registers.end();
    config.compute_pgm_rsrc1 = observed ? found->second : 0u;
    config.fp16_overflow = compute_fp16_overflow_from_rsrc1(observed, config.compute_pgm_rsrc1);
}
} // namespace prosper::gpu

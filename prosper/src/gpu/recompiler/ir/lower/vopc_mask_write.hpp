// vopc_mask_write.hpp — the shared lane-mask write of a VOPC compare (vopc_mask_write.cpp).
#pragma once

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {

// Writes one compare's per-lane result `cmp` (a Bool id) where `in` sends it: EXEC &= cmp for a
// V_CMPX, otherwise EXEC & cmp into VCC or the explicit SGPR destination. `compare_wave_uniform`
// says both sources were wave-uniform scalars, which publishes VCC's wave-uniformity proof. The
// call must come after every check that can clear `ok`: this function cannot refuse.
void write_vopc_mask(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t cmp,
                     bool compare_wave_uniform);

}   // namespace prosper::gpu::ir

// Retire the mask views overwritten by a scalar DATA write to physical VCC words.
#pragma once

namespace prosper::gpu {
struct RegState;
struct Rdna2Inst;
void retire_scalar_vcc_mask_view(RegState& state, const Rdna2Inst& instruction);
}   // namespace prosper::gpu

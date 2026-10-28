// Retire the mask views overwritten by a scalar DATA write to physical VCC words.
#pragma once

namespace prosper::gpu {
struct RegState;
void retire_scalar_vcc_mask_view(RegState& state, int untouched_word);
}   // namespace prosper::gpu

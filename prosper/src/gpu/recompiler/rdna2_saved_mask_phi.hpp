// Immutable Bool PHI identities; these facts provide no physical mask words.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
// A completed PHI of the same value or a saved value and itself is that saved value.
uint32_t saved_mask_phi_identity(const SpirvCompute& b, uint32_t value);
void record_saved_mask_phi_subset(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in);
} // namespace prosper::gpu

// Dead 64-bit VCC mask logicals the CFG emitter may elide (moved out of rdna2_emit_cfg.cpp).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// Some GFX10 pixel shaders leave scheduled 64-bit mask operations whose VCC/SCC results feed only
// other dead mask operations and are overwritten before an observable read. Astro Bot's SSAO shader does this with
// `s_and_b64 vcc, s[0:1], vcc`, where s[0:1] is also a live T# descriptor; attempting to reinterpret
// the descriptor bits as a per-lane mask is both impossible in descriptor-backed SPIR-V and pointless.
// Elide only the mechanically proven dead form: the shader has no SCC consumer anywhere, and CFG
// liveness proves the VCC pair cannot reach a non-mask read before redefinition. This deliberately
// does not become a general scalar-pair-to-wave-mask fallback.
std::unordered_set<uint32_t> dead_wave_mask_writes(const std::vector<Rdna2Inst>& ins);

}   // namespace prosper::gpu

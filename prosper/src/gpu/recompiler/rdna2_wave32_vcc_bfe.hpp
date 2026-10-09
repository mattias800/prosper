// rdna2_wave32_vcc_bfe.hpp -- which Wave32 `s_bfe_u64 vcc, ...` results are scalar DATA (#4808).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// Wave32 VCC-as-data BFE (#4808). s_bfe_u64 into VCC keeps its exact 64-bit result as scalar data
// only where its sources are proven real values (emit_alu: retain_vcc_scalar); otherwise VCC is left
// a per-lane mask and a later scalar read of VCC_LO is unresolved. The Wave64 record pass in
// emit_cfg_state_machine proves that for Wave64 only. This is the local half of the same proof, for
// every Wave32 emission path: a source word written by an earlier instruction of the SAME basic
// block holds that instruction's value, never a structured-PHI or dispatcher placeholder (those
// enter only where control flow joins or a barrier phase begins, and the walk forgets everything
// there). emit_alu still requires each source word to be present as scalar data, so a mask-domain
// writer leaves the result a mask. Yakuza Kiwami's Wave32 NGG VS programs build a buffer
// descriptor's word 3 from `s_bfe_u64 vcc, s[8:9], vcc_hi` with s[8:9] and vcc_hi set a few
// instructions earlier. CONFIDENCE: HIGH (a same-block definition dominates its use).
std::unordered_set<uint32_t> proven_wave32_local_vcc_bfe_pcs(const std::vector<Rdna2Inst>& ins);

}   // namespace prosper::gpu

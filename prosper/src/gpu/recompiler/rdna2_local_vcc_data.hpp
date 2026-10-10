// rdna2_local_vcc_data.hpp -- which scalar ALU writes into VCC have only real-value sources (#4808).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// A scalar ALU write into VCC keeps its result as scalar DATA only where its sources are proven
// real values; otherwise VCC is left a per-lane mask and a later scalar read of VCC_LO (a buffer
// descriptor word, an SMEM offset) is unresolved. emit_alu asks that of
//   * `s_bfe_u64 vcc, ...` (retain_vcc_scalar), at either wave width, and
//   * a B32 SOP2 write to VCC_LO/VCC_HI in Wave64 compute (has_proven_scalar_sources),
// and the Wave64 record pass in emit_cfg_state_machine answers it only for Wave64 and only on the
// state-machine path. This is the local half of the same proof, for every emission path: a source
// word written by an earlier instruction of the SAME basic block holds that instruction's value,
// never a structured-PHI or dispatcher placeholder (those enter only where control flow joins or a
// barrier phase begins, and the walk forgets everything there). emit_alu still requires each
// source word to be present as scalar data, so a mask-domain writer leaves the result a mask.
//
// Seen live: Yakuza Kiwami's Wave32 NGG VS builds a buffer descriptor's word 3 from
// `s_bfe_u64 vcc, s[8:9], vcc_hi`; The Pathless's Wave64 ES prolog builds an SMEM offset with
// `s_and_b32 vcc_lo, s64, 31; s_lshl_b32 vcc_lo, vcc_lo, 4` after `s_cselect_b64 vcc, exec, 0`.
// CONFIDENCE: HIGH (a same-block definition dominates its use).
std::unordered_set<uint32_t> proven_local_vcc_scalar_write_pcs(const std::vector<Rdna2Inst>& ins);

}   // namespace prosper::gpu

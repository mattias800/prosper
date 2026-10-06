// vgt_shader_stages.hpp -- decode VGT_SHADER_STAGES_EN (gfx10.3) for the fields prosper routes on.
//
// LS_EN [1:0], HS_EN [2], ES_EN [4:3], GS_EN [5], VS_EN [7:6], PRIMGEN_EN [13], HS_W32_EN [21],
// GS_W32_EN [22], VS_W32_EN [23]. CONFIDENCE: HIGH for GS_EN/ES_EN (independent register
// definitions agree) and for the W32 bits (AGC's half-matching check reads bits 21/22); MED for
// PRIMGEN_EN at bit 13 (Mesa's gfx10 definition; Kena's merged NGG program carries 0x2030).
#pragma once

#include <cstdint>

namespace prosper::gpu {

struct VgtShaderStages {
    uint32_t raw = 0;
    bool gs_enabled() const { return (raw >> 5) & 1u; }
    bool primgen_enabled() const { return (raw >> 13) & 1u; }   // NGG primitive generation
    bool gs_wave32() const { return (raw >> 22) & 1u; }
    uint32_t es_stage() const { return (raw >> 3) & 3u; }
};

// The NGG shape of a draw, for diagnostics and routing:
//   "merged-gs" NGG with a GS stage (ES and GS run merged in one wave group),
//   "ngg-vs"    NGG without GS (the VS is the primitive shader),
//   "legacy"    no primitive generation.
inline const char* ngg_stage_class(VgtShaderStages stages) {
    if (!stages.primgen_enabled()) return "legacy";
    return stages.gs_enabled() ? "merged-gs" : "ngg-vs";
}

}  // namespace prosper::gpu

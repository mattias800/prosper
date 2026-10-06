#pragma once

#include "gpu/diagnostics/refused_shader_dump.hpp"

#include <memory>

namespace prosper::gpu {

struct ShaderCodeAnalysis;
struct ComputeProgramFacts;

// Original stage observations only. These fields confer no compilation or draw admission.
struct RefusedDrawShaders {
    RefusedShaderSource vs, ps;
    std::shared_ptr<const ShaderCodeAnalysis> ps_analysis;
    uint64_t vs_address, ps_address, es_address, command_order;
    size_t max_dwords, vs_words, gs_words, fs_words;
    bool vs_failed, ps_failed;
    // Set when the draw was refused before any recompile was attempted (the owned-wave gate), so
    // the index line says why. Its first_bad_* fields are the generic coverage census, not a cause.
    const char* refusal = nullptr;
    // The draw's NGG shape (vgt_shader_stages.hpp) and how its vertex program is linked: "prolog"
    // (a prolog chained to a main by s_setpc), "fused" (AGC's fused-back form, VS address differs
    // from the ES address) or "none". A chained main is dumped beside its prolog as "vsmain", so an
    // offline reader has both halves (#3135 P0).
    const char* ngg_class = nullptr;
    const char* link = nullptr;
    uint64_t chain_address = 0;
};

void note_refused_draw_shaders(const RefusedDrawShaders& shaders);
void note_refused_compute_program(const std::shared_ptr<const ComputeProgramFacts>& facts,
                                  uint64_t address, uint32_t groups_x, uint32_t groups_y,
                                  uint32_t groups_z);

}  // namespace prosper::gpu

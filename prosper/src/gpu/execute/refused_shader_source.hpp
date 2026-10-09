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

// The live compute backend's reason for declining the dispatch it is executing. The backend runs
// synchronously on the executor's thread, so the executor reads the slot back right after a
// declined call. `reason` must have static storage (a string literal); null clears the slot.
void set_compute_decline_reason(const char* reason);
// The recorded reason, or null when the backend declined without naming one. Clears the slot.
const char* take_compute_decline_reason();
// A dispatch that recompiled and was realized, then declined by the live backend: keep its program
// in the refused-shader dump with "refusal=backend-declined:<reason>" (#4808). Until this, only
// recompile refusals were dumped, so the largest skip category left no program evidence at all.
// Diagnostic only; the dump's usual (stage, content) dedup and its cap apply.
void note_backend_declined_compute(uint64_t address, uint32_t dwords, uint32_t groups_x,
                                   uint32_t groups_y, uint32_t groups_z, const char* reason);

}  // namespace prosper::gpu

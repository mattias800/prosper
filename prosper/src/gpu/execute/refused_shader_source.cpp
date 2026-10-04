#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/refused_shader_source.hpp"
#include "gpu/execute/compute_program_facts.hpp"
#include "gpu/execute/shader_source_window.hpp"

#include <cstdio>
#include <tuple>

namespace prosper::gpu {

namespace {
RefusedShaderSource shader_analysis_refused_source(const SharedShaderAnalysis& analysis) {
    // Both aliases share ownership with the exact immutable analysis. This observation cannot
    // grant shader/resource admission; early refusals use the same bounded byte-validating cache.
    return analysis
               ? RefusedShaderSource{{analysis, &analysis->code}, &analysis->refused_shader_memo}
               : RefusedShaderSource{};
}
}   // namespace

void note_refused_draw_shaders(const RefusedDrawShaders& shaders) {
    if (refused_shader_dump_full()) return;
    for (auto [tag, address, failed, original] :
         {std::tuple{"vs", shaders.vs_address, shaders.vs_failed, shaders.vs},
          std::tuple{"ps", shaders.ps_address, shaders.ps_failed, shaders.ps}}) {
        if (!failed || !address) continue;
        // Pre-key guards have no compile-key owner. Observe their original readable program
        // through the same existing bounded analysis cache, not a guessed address generation.
        if (!original.words) {
            const size_t source_dwords = native_shader_source_dwords(address, shaders.max_dwords);
            if (!source_dwords) continue;
            original = shader_analysis_refused_source(
                tag[0] == 'p' && shaders.ps_analysis
                    ? shaders.ps_analysis
                    : acquire_shader_analysis(
                          reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(address)),
                          source_dwords));
        }
        if (refused_shader_already_noted(tag, original)) continue;
        char detail[160];
        std::snprintf(
            detail, sizeof detail, "draw-order=%llu vs=%zu gs=%zu fs=%zu es=0x%llx ps=0x%llx",
            static_cast<unsigned long long>(shaders.command_order), shaders.vs_words,
            shaders.gs_words, shaders.fs_words, static_cast<unsigned long long>(shaders.es_address),
            static_cast<unsigned long long>(shaders.ps_address));
        note_refused_shader(tag, address, original, detail);
    }
}

void note_refused_compute_program(const std::shared_ptr<const ComputeProgramFacts>& facts,
                                  uint64_t address, uint32_t groups_x, uint32_t groups_y,
                                  uint32_t groups_z) {
    if (!facts) return;
    const RefusedShaderSource source{{facts, &facts->code}, &facts->refused_shader_memo};
    note_refused_compute_shader(address, source, groups_x, groups_y, groups_z);
}

}   // namespace prosper::gpu

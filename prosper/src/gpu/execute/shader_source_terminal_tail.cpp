#include "gpu/execute/shader_source_window.hpp"

#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/recompiler/rdna2_terminal_tail_span.hpp"

#include <cstring>

namespace prosper::gpu {

size_t shader_analysis_source_span(ShaderCodeAnalysis& analysis, const uint32_t* code,
                                   size_t dwords) {
    const size_t span = shader_source_code_span(code, dwords);
    if (!code || !span) return span;
    std::vector<Rdna2Inst> body;
    rdna2_walk(code, dwords, body);
    const auto terminal = rdna2_terminal_tail_span(code, dwords, body);
    analysis.terminal_tail_limit_required = terminal.source_limit_required;
    if (terminal.inspected_dwords > span)
        analysis.terminal_tail_validation.assign(code + span, code + terminal.inspected_dwords);
    return span;
}

bool shader_analysis_source_matches(const ShaderCodeAnalysis& analysis, const uint32_t* code,
                                    size_t dwords) {
    if (analysis.terminal_tail_limit_required && analysis.source_dwords != dwords) return false;
    const size_t code_words = analysis.code.size();
    const size_t tail_words = analysis.terminal_tail_validation.size();
    if (!code && (code_words || tail_words)) return false;
    if (code_words > dwords || tail_words > dwords - code_words) return false;
    if (tail_words > UINT32_MAX / sizeof(uint32_t)) return false;
    if (code_words && std::memcmp(code, analysis.code.data(), code_words * sizeof(uint32_t)))
        return false;
    if (!tail_words) return true;
    // A failed terminal probe is only validation evidence. Its bytes are never compiler input,
    // but a later replacement at the same address must not reuse its old prefix-only verdict.
    const uint64_t tail_address = uint64_t(uintptr_t(code + code_words));
    const auto tail_bytes = static_cast<uint32_t>(tail_words * sizeof(uint32_t));
    return guest_readable(tail_address, tail_bytes) &&
           std::memcmp(code + code_words, analysis.terminal_tail_validation.data(), tail_bytes) ==
               0;
}

} // namespace prosper::gpu

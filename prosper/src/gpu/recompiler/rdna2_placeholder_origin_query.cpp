#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_placeholder_origin_trace.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include <algorithm>

namespace prosper::gpu {
namespace {
std::string describe(const PlaceholderOriginProgram* program, uint32_t pc,
                     const std::vector<PlaceholderWord>& words) {
    if (program) return format_placeholder_origins(placeholder_roots_before(*program, pc, words));
    PlaceholderRoots roots;
    for (auto word : words) roots.insert({PlaceholderOrigin::UnknownLaunch, word});
    return format_placeholder_origins(roots);
}
} // namespace

std::string fresh_placeholder_origins(const RegState& rs, const Rdna2Inst& in,
                                      const PlaceholderOriginProgram* fallback) {
    std::vector<PlaceholderWord> words;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& source = in.src[k];
        if (source.kind == OperandKind::VGPR) {
            const auto width = vector_provenance_source_span(in, k);
            for (uint32_t n = 0; n < width; ++n) {
                const int reg = source.value + static_cast<int>(n);
                const auto mark = rs.vreg_source_marks.find(reg);
                if (!rs.vreg.contains(reg) ||
                    (mark != rs.vreg_source_marks.end() && mark->second.placeholder))
                    words.push_back({PlaceholderFile::Vector, reg});
            }
        } else if (source.kind == OperandKind::SGPR ||
                   (source.kind == OperandKind::Special && source.value <= 124)) {
            const auto width = scalar_source_mark_words(in, k, rs.vector_wave_size);
            if (width == UINT32_MAX) continue;
            for (uint32_t n = 0; n < width; ++n) {
                const int reg = source.value + static_cast<int>(n);
                if (sreg_word_may_be_fabricated(rs, reg))
                    words.push_back({PlaceholderFile::Scalar, reg});
            }
        } else if (source.kind == OperandKind::Special && source.value == 253 &&
                   (!rs.scc || rs.scc_merge_placeholder)) {
            words.push_back({PlaceholderFile::Scc});
        }
    }
    const auto& effects = rdna2_instruction_effects(in);
    if (effects.implicit_reads & kRdna2ImplicitVcc)
        for (int reg : {106, 107})
            if (sreg_word_may_be_fabricated(rs, reg))
                words.push_back({PlaceholderFile::Scalar, reg});
    if ((effects.implicit_reads & kRdna2ImplicitScc) && (!rs.scc || rs.scc_merge_placeholder))
        words.push_back({PlaceholderFile::Scc});
    if (words.empty())
        return format_placeholder_origins({{PlaceholderOrigin::UnmodelledTransfer,
                                            {PlaceholderFile::Vector, in.dst.value},
                                            in.pc}});
    return describe(rs.placeholder_origin_program ? rs.placeholder_origin_program.get() : fallback,
                    in.pc, words);
}

std::string loop_placeholder_origins(const RegState& rs, uint32_t header,
                                     const std::vector<int>& carried,
                                     const PlaceholderOriginProgram* fallback) {
    const auto* program =
        rs.placeholder_origin_program ? rs.placeholder_origin_program.get() : fallback;
    std::vector<PlaceholderWord> words;
    for (auto reg : carried) {
        if (reg >= kVectorBlanketBase)
            words.push_back({PlaceholderFile::Vector,
                             (reg - kVectorBlanketBase) % kArbitraryOutsideRootOffset});
        else
            words.push_back({PlaceholderFile::Scalar, reg});
    }
    if (program) {
        const detail::PlaceholderOriginTrace trace(*program);
        PlaceholderRoots roots;
        for (const auto& [from, to] : trace.cfg.back_edges) {
            if (trace.cfg.blocks[to].pc != header) continue;
            const auto pc = program->instructions[trace.cfg.blocks[from].last].pc;
            const auto incoming = trace.before(pc, words);
            roots.insert(incoming.begin(), incoming.end());
        }
        if (!roots.empty()) return format_placeholder_origins(roots);
    }
    return describe(program, header, words);
}
} // namespace prosper::gpu

#include "gpu/recompiler/rdna2_predicate_operand_roots.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"

#include <cstdint>
#include <optional>
#include <set>

namespace prosper::gpu {
namespace {
bool globally_genuine_scalar(const RegState& rs, const Rdna2Inst& in, uint32_t source) {
    const auto& operand = in.src[source];
    if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat)
        return true;
    if (operand.kind == OperandKind::Literal) return in.has_literal;
    if (operand.kind != OperandKind::SGPR) return false;
    const uint32_t width = scalar_source_mark_words(in, source, rs.vector_wave_size);
    if (!width || width > 2 || operand.value < 0 || operand.value + static_cast<int>(width) > 106)
        return false;
    for (uint32_t word = 0; word < width; ++word) {
        const int reg = operand.value + static_cast<int>(word);
        const auto value = rs.sreg.find(reg);
        const auto roots = rs.sreg_loop_blanket.find(reg);
        if ((!rs.sreg_input.contains(reg) && (value == rs.sreg.end() || !value->second)) ||
            sreg_word_may_be_fabricated(rs, reg) ||
            (roots != rs.sreg_loop_blanket.end() && !roots->second.empty()))
            return false;
    }
    return true;
}
}  // namespace
std::optional<std::set<uint64_t>> predicate_operand_roots(const RegState& rs, const Rdna2Inst& in,
                                                          PredicateRootReader read_roots) {
    // Explicit lane-local inputs; scalar siblings must have a global proof with no roots.
    // Definedness and the EXEC gate remain with the caller. Other forms keep the legacy proof.
    if (in.fmt != Rdna2Format::VOPC || in.has_dpp || in.has_modifier || in.has_sdwa ||
        in.n_src != 2 || !read_roots)
        return std::nullopt;
    std::set<uint64_t> roots;
    for (uint32_t k = 0; k < 2; ++k) {
        if (in.src[k].kind != OperandKind::VGPR) {
            if (!globally_genuine_scalar(rs, in, k)) return std::nullopt;
            continue;
        }
        const uint32_t width = rdna2_valu_vgpr_source_span(in, k);
        if (!width || in.src[k].value < 0 || static_cast<uint32_t>(in.src[k].value) + width > 256)
            return std::nullopt;
        for (uint32_t word = 0; word < width; ++word) {
            const auto source = rs.vreg_source_marks.find(in.src[k].value + static_cast<int>(word));
            if (source == rs.vreg_source_marks.end()) continue;
            const auto& current = read_roots(rs, source->second, rs.exec);
            roots.insert(current.begin(), current.end());
        }
    }
    return roots;
}
} // namespace prosper::gpu

#include "gpu/recompiler/rdna2_predicate_operand_roots.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"

namespace prosper::gpu {
std::optional<std::set<uint64_t>> predicate_operand_roots(const RegState& rs, const Rdna2Inst& in,
                                                          PredicateRootReader read_roots) {
    // These compares read two explicit lane-local vector data operands; their EXEC
    // gate is handled by the caller. DPP/SDWA and other families keep the existing proof.
    if (in.fmt != Rdna2Format::VOPC || in.has_dpp || in.has_modifier || in.has_sdwa ||
        in.n_src != 2 || in.src[0].kind != OperandKind::VGPR || in.src[1].kind != OperandKind::VGPR)
        return std::nullopt;
    std::set<uint64_t> roots;
    for (uint32_t k = 0; k < 2; ++k) {
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

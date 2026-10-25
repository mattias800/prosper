#include "gpu/recompiler/rdna2_current_word_roots.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <utility>

namespace prosper::gpu {
namespace {
bool original_instruction(const Rdna2Inst& in) {
    if (!in.len_dwords || in.len_dwords > std::size(in.words) || in.synthetic_terminator)
        return false;
    auto decoded = rdna2_decode_one(in.words, in.len_dwords);
    decoded.pc = in.pc;
    return decoded == in;
}

bool data_word_roots(const RegState& rs, int reg, std::set<uint64_t>& roots,
                     WordRootReader read_roots) {
    const auto value = rs.vreg.find(reg);
    if (reg < 0 || reg > 255 || value == rs.vreg.end() || !value->second) return false;
    const auto mark = rs.vreg_source_marks.find(reg);
    if (mark == rs.vreg_source_marks.end()) return true;
    if (!vector_word_defined_in_exec(rs, mark->second, rs.exec)) return false;
    const auto& current = read_roots(rs, mark->second, rs.exec);
    roots.insert(current.begin(), current.end());
    return true;
}

bool plain_numeric_roots(const RegState& rs, const Rdna2Inst& in, std::set<uint64_t>& roots,
                         WordRootReader read_roots) {
    const auto& effects = rdna2_instruction_effects(in);
    // Use the existing VALU inventory, excluding hidden mask/index/old-destination inputs.
    if (!effects.describes_vector_result() || effects.lane_flow != Rdna2LaneFlow::LaneWise ||
        effects.vdst_dwords != 1 || effects.vdst_half || effects.accumulates ||
        effects.writes_src0 || effects.implicit_reads || effects.lane_mask_src2 ||
        effects.m0_relative != Rdna2M0Relative::None || !vector_write_completes_active_word(in) ||
        in.n_src > 3)
        return false;
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto& operand = in.src[k];
        if (operand.kind == OperandKind::VGPR) {
            if (rdna2_valu_vgpr_source_span(in, k) != 1 ||
                !data_word_roots(rs, operand.value, roots, read_roots))
                return false;
        } else if (operand.kind != OperandKind::InlineInt &&
                   operand.kind != OperandKind::InlineFloat &&
                   !(operand.kind == OperandKind::Literal && in.has_literal)) {
            return false;
        }
    }
    return true;
}

bool atomic_result_roots(const RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source,
                         std::set<uint64_t>& roots, WordRootReader read_roots) {
    // These x1 services return old memory or their genuine old VDATA fallback. This is not a
    // memory-effect classifier: address/resource admission stays with the existing lowering.
    if (in.fmt != Rdna2Format::MUBUF || (in.opcode != 0x30u && in.opcode != 0x32u) ||
        !in.mubuf_glc || in.mubuf_tfe || in.mubuf_lds || in.len_dwords != 2 ||
        !arbitrary_read_resource_genuine(rs, in) ||
        !data_word_roots(rs, in.dst.value, roots, read_roots))
        return false;
    const uint32_t words = rdna2_vgpr_source_span(in, 0);
    if (words > 2) return false;
    for (uint32_t word = 0; word < words; ++word)
        if (!data_word_roots(rs, in.src[0].value + static_cast<int>(word), roots, read_roots))
            return false;
    const auto& address = read_roots(rs, source, rs.exec);
    roots.insert(address.begin(), address.end());
    return true;
}
}  // namespace

CurrentWordRootPublication snapshot_current_word_roots(const RegState& rs, const Rdna2Inst& in,
                                                       const ScalarSourceMarks& source,
                                                       WordRootReader read_roots) {
    if (!read_roots || rs.vector_wave_size != 64 || !rs.exec || !rs.exec_narrowed ||
        rs.vector_full_exec || in.dst.kind != OperandKind::VGPR ||
        rdna2_vgpr_write_count(in) != 1 || in.has_modifier || in.has_dpp || in.has_sdwa ||
        !original_instruction(in) || !vector_word_defined_in_exec(rs, source, rs.exec))
        return {};
    const auto value = rs.vreg.find(in.dst.value);
    if (value == rs.vreg.end() || !value->second) return {};
    CurrentWordRootPublication result;
    if (in.fmt == Rdna2Format::MUBUF) {
        if (!atomic_result_roots(rs, in, source, result.view.roots, read_roots)) return {};
        result.atomic_result = true;
    } else if (!plain_numeric_roots(rs, in, result.view.roots, read_roots)) {
        return {};
    }
    result.view.exec = rs.exec;
    return result;
}

void retain_current_word_roots(ScalarSourceMarks& result, const RegState& rs,
                               const CurrentWordRootPublication& publication,
                               WordRootReader read_roots) {
    result.current_roots.reset();
    if (!publication.view.exec || !read_roots || !result.scoped_complete_word ||
        !vector_word_defined_in_exec(rs, result, publication.view.exec))
        return;
    const auto& prior = read_roots(rs, result, publication.view.exec);
    const auto& current = publication.view.roots;
    // A refinement cannot add an observation that the legacy view did not already require.
    if (prior == current ||
        !std::includes(prior.begin(), prior.end(), current.begin(), current.end()))
        return;
    result.current_roots = std::make_shared<const CurrentWordRoots>(publication.view);
}

const std::set<uint64_t>* current_word_roots(const RegState& rs, const ScalarSourceMarks& marks,
                                             uint32_t reader) {
    if (!marks.scoped_complete_word || !marks.current_roots || !marks.current_roots->exec ||
        !reader || !vector_word_defined_in_exec(rs, marks, reader) ||
        !exec_within(rs, reader, marks.current_roots->exec))
        return nullptr;
    return &marks.current_roots->roots;
}
}  // namespace prosper::gpu

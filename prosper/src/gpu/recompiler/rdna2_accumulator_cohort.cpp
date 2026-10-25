#include "gpu/recompiler/rdna2_accumulator_cohort.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <cstdint>
#include <iterator>
namespace prosper::gpu {
namespace {
uint32_t accumulator_cohort(const RegState& rs, const Rdna2Inst& in,
                            const ScalarSourceMarks& source) {
    if (rs.vector_wave_size != 64 || !rs.exec_narrowed || rs.vector_full_exec || !rs.exec ||
        (in.fmt != Rdna2Format::VOP2 && in.fmt != Rdna2Format::VOP3) || in.has_modifier ||
        in.has_dpp || in.has_sdwa || in.dst.kind != OperandKind::VGPR || in.dst.value < 0 ||
        in.dst.value > 255 || rdna2_vgpr_write_count(in) != 1 || !in.len_dwords ||
        in.len_dwords > std::size(in.words) || rdna2_instruction_may_change_exec(in) ||
        rdna2_valu_lane_mapped(in.fmt, in.opcode) != Rdna2Answer::Yes ||
        !vector_write_depends_on_old_word(in, true))
        return 0;
    auto decoded = rdna2_decode_one(in.words, in.len_dwords);
    decoded.pc = in.pc;
    if (decoded != in) return 0;
    const auto value = rs.vreg.find(in.dst.value);
    if (value == rs.vreg.end() || !value->second) return 0;
    const auto before = rs.vreg_source_marks.find(in.dst.value);
    if (before == rs.vreg_source_marks.end()) return 0;
    const auto& old = before->second;
    if (!old.placeholder || !old.hard || !old.scoped_complete_word || !old.outside_exec ||
        old.outside_exec == rs.exec || !exec_within(rs, rs.exec, old.outside_exec) ||
        !vector_word_defined_in_exec(rs, old, rs.exec) ||
        !vector_word_defined_in_exec(rs, source, rs.exec))
        return 0;
    return old.outside_exec;
}
void retain_accumulator_cohort(RegState& rs, const Rdna2Inst& in, uint32_t owner) {
    if (!owner) return;
    const auto current = rs.vreg_source_marks.find(in.dst.value);
    if (current == rs.vreg_source_marks.end() || !current->second.hard ||
        !current->second.scoped_complete_word || !current->second.outside_exec ||
        current->second.outside_exec == owner ||
        !exec_within(rs, rs.exec, current->second.outside_exec) ||
        !exec_within(rs, current->second.outside_exec, owner))
        return;
    // The ordinary transfer already met the input and old word's active roots. Widen only its owner.
    auto& result = current->second;
    result.outside_exec = owner;
}
}   // namespace
void record_vector_word_marks_preserving_cohort(RegState& rs, const Rdna2Inst& in,
                                                const ScalarSourceMarks& source) {
    const uint32_t owner = accumulator_cohort(rs, in, source);
    record_vector_word_marks(rs, in, source);
    retain_accumulator_cohort(rs, in, owner);
}
}   // namespace prosper::gpu

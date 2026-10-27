// A new arbitrary admission cannot expose an unrelated missing hardware input as zero.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_placeholder_origin_trace.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
bool reject_input(SpirvCompute& b, const Rdna2Inst& in, const PlaceholderRoots& roots, bool* ok) {
    b.diagnostic.placeholder_origins =
        std::make_shared<const std::string>(format_placeholder_origins(roots));
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = "hard-placeholder-read";
    log_recompile_diagnostic(b.diagnostic, "hard-placeholder-read", "alu",
                             "pc=%u an arbitrary retry still requires every real input", in.pc);
    if (ok) *ok = false;
    return true;
}

bool hard_operand(const RegState& rs, const Operand& operand, PlaceholderWord& word,
                  bool& missing) {
    if (operand.kind == OperandKind::VGPR) {
        word = {PlaceholderFile::Vector, operand.value};
        const auto value = rs.vreg.find(operand.value);
        // A lowering may reserve VDST before reading an aliased source. ID 0 reserves storage;
        // it is no SSA value. A real constant with zero bits still has a nonzero SPIR-V ID.
        missing = value == rs.vreg.end() || !value->second;
        const auto mark = rs.vreg_source_marks.find(operand.value);
        return missing || (mark != rs.vreg_source_marks.end() && mark->second.hard &&
                           !vector_word_defined_in_exec(rs, mark->second, rs.exec));
    }
    if (operand.kind == OperandKind::SGPR ||
        (operand.kind == OperandKind::Special && operand.value >= 106 && operand.value <= 124)) {
        word = {PlaceholderFile::Scalar, operand.value};
        missing = !rs.sreg.contains(operand.value) && !rs.sreg_input.contains(operand.value);
        // Existing mask-domain reads and their physical-word conversion keep their own guards.
        if (missing &&
            (rs.sreg_bool.contains(operand.value) || rs.sreg_bool.contains(operand.value - 1)))
            return false;
        return sreg_word_may_be_fabricated(rs, operand.value) &&
               !sreg_word_blanket_only(rs, operand.value);
    }
    if (operand.kind == OperandKind::Special && operand.value == 253) {
        word = {PlaceholderFile::Scc};
        missing = !rs.scc;
        return missing || (rs.scc_merge_placeholder && rs.scc_loop_blanket.empty());
    }
    return false;
}

void retain_dropped_definition(PlaceholderRoots& roots, const detail::PlaceholderOriginTrace& trace,
                               uint32_t pc, PlaceholderWord word) {
    const auto& ins = trace.program.instructions;
    const auto at =
        std::find_if(ins.begin(), ins.end(), [&](const auto& in) { return in.pc == pc; });
    if (at == ins.end()) return;
    std::vector<uint32_t> work = trace.predecessors[at - ins.begin()];
    std::set<uint32_t> seen;
    while (!work.empty()) {
        const auto index = work.back();
        work.pop_back();
        if (!seen.insert(index).second) continue;
        if (seen.size() > 100000) {
            roots.insert({PlaceholderOrigin::TraceBudget, word, pc});
            return;
        }
        if (trace.writes(ins[index], word)) {
            // A real write reached this read, but the lowering discarded its value. Its input
            // being arbitrary does not permit replacing the computed result with storage zero.
            roots.insert({PlaceholderOrigin::LostDefinition, word, ins[index].pc});
            return;
        }
        const auto& predecessors = trace.predecessors[index];
        work.insert(work.end(), predecessors.begin(), predecessors.end());
    }
}
}   // namespace

bool refuse_arbitrary_placeholder_operand(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                          const Operand& operand, bool* ok) {
    if (!arbitrary_placeholder_retry_active() || b.is_fragment_packet()) return false;
    PlaceholderWord word;
    bool missing = false;
    if (!hard_operand(rs, operand, word, missing)) return false;
    const auto* program = rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                                        : b.placeholder_origin_program.get();
    PlaceholderRoots roots;
    if (program) {
        const detail::PlaceholderOriginTrace trace(*program);
        roots = trace.before(in.pc, {word});
        if (missing && placeholder_roots_are_unloaded(roots))
            retain_dropped_definition(roots, trace, in.pc, word);
    } else {
        roots.insert({PlaceholderOrigin::UnknownLaunch, word});
    }
    if (placeholder_roots_are_unloaded(roots)) return false;
    return reject_input(b, in, roots, ok);
}

bool refuse_arbitrary_placeholder_lane_read(SpirvCompute& b, const RegState& rs,
                                            const Rdna2Inst& in, bool& ok) {
    if (!arbitrary_placeholder_retry_active() || b.is_fragment_packet() ||
        !vector_lane_read_marks(rs, in).hard)
        return false;
    const auto* program = rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                                        : b.placeholder_origin_program.get();
    const int reg = in.src[0].value;
    PlaceholderRoots roots;
    if (program) {
        const detail::PlaceholderOriginTrace trace(*program);
        const bool spill = rs.vgpr_lane_slots.contains(reg) ||
                           rs.vgpr_lane_mask_slots.contains(reg) ||
                           (!rs.vreg.contains(reg) && trace.writelane_destinations.contains(reg));
        const int lane = in.src[1].kind == OperandKind::InlineInt ? in.src[1].value : -1;
        roots = trace.before(in.pc, {{spill ? PlaceholderFile::Spill : PlaceholderFile::Vector, reg,
                                      spill ? lane : -1}});
    } else {
        roots.insert({PlaceholderOrigin::UnknownLaunch, {PlaceholderFile::Vector, reg}});
    }
    if (placeholder_roots_are_unloaded(roots)) return false;
    return reject_input(b, in, roots, &ok);
}

bool refuse_arbitrary_placeholder_export(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                         bool& ok) {
    if (!arbitrary_placeholder_retry_active() || in.fmt != Rdna2Format::EXP) return false;
    // Exports read the register map directly instead of going through operand_bits.
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const uint32_t components = in.exp_compr ? (3u << (2u * k)) : (1u << k);
        if ((in.exp_en & components) &&
            refuse_arbitrary_placeholder_operand(b, rs, in, in.src[k], &ok))
            return true;
    }
    return false;
}
bool refuse_arbitrary_placeholder_old_destination(SpirvCompute& b, const RegState& rs,
                                                  const Rdna2Inst& in, bool& ok) {
    if (!arbitrary_placeholder_retry_active() || b.is_fragment_packet() ||
        vector_write_leaves_old_word(in, rs.vector_native_interpolation) ||
        !vector_write_depends_on_old_word(in, !rs.exec_narrowed) ||
        rdna2_instruction_effects(in).lane_flow == Rdna2LaneFlow::WriteLane)
        return false;
    // The existing predicate names actual old-word reads, including inactive lanes, partial
    // words and accumulators. Lane-slot writers preserve their slot map, not ordinary SSA.
    bool refused = false;
    for_each_possible_vector_write(in, 255, [&](int reg) {
        if (!refused)
            refused =
                refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::VGPR, reg}, &ok);
    });
    return refused;
}
uint32_t read_vector_for_arbitrary_retry(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                         int reg, bool& ok) {
    if (refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::VGPR, reg}, &ok))
        return b.uconst(0);
    return vreg_old(b, rs, reg);
}

uint32_t read_mimg_source_for_arbitrary_retry(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                              int reg, bool& ok) {
    if (refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::VGPR, reg}, &ok))
        return b.uconst(0);
    return read_mimg_source_word(b, rs, reg);
}
}   // namespace prosper::gpu

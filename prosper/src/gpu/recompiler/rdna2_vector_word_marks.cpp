#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {

bool memory_result(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
           in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG ||
           in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::DS;
}

ScalarSourceMarks vector_word(const RegState& rs, int reg) {
    // This relay map follows marked scalar/memory origins, not ordinary per-lane definedness.
    // An unmarked VGPR grants no entry-value authority; that remains the compiler's existing domain.
    const auto found = rs.vreg_source_marks.find(reg);
    return found == rs.vreg_source_marks.end() ? ScalarSourceMarks{} : found->second;
}

void set_word(VectorWordMarks& marks, int reg, ScalarSourceMarks source) {
    if (source.placeholder || source.memory)
        marks[reg] = std::move(source);
    else
        marks.erase(reg);
}

}   // namespace

void union_source_marks(ScalarSourceMarks& into, const ScalarSourceMarks& other) {
    into.placeholder = into.placeholder || other.placeholder;
    into.memory = into.memory || other.memory;
    into.hard = into.hard || other.hard;
    if (into.hard)
        into.blanket_roots.clear();
    else
        into.blanket_roots.insert(other.blanket_roots.begin(), other.blanket_roots.end());
}

ScalarSourceMarks vector_source_marks(const RegState& rs, const Rdna2Inst& in) {
    ScalarSourceMarks marks;
    // Load addresses are not loaded data. A genuine read still carries MEMORY, which only the
    // fragment lane-bit projection rejects; compute keeps its actual guest-lane identity.
    if (memory_result(in)) {
        marks.memory = true;
        if (in.fmt != Rdna2Format::DS) return marks;
    }
    for (uint32_t source = 0; source < in.n_src; ++source)
        for (uint32_t word = 0; word < rdna2_vgpr_source_span(in, source); ++word)
            union_source_marks(marks,
                               vector_word(rs, in.src[source].value + static_cast<int>(word)));
    // Relative source addressing can select any observable source lifetime, not only the encoded
    // base VGPR. No selector value is guessed here.
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43)
        for (const auto& [reg, source] : rs.vreg_source_marks)
            if (reg >= in.src[0].value) union_source_marks(marks, source);
    return marks;
}

ScalarSourceMarks vector_lane_read_marks(const RegState& rs, const Rdna2Inst& in) {
    const int reg = in.src[0].value;
    if (in.src[1].kind == OperandKind::InlineInt &&
        (rs.vgpr_lane_slots.contains(reg) || rs.vgpr_lane_mask_slots.contains(reg))) {
        const std::pair<int, int> slot{reg, in.src[1].value};
        ScalarSourceMarks marks;
        marks.placeholder = rs.lane_slot_merge_placeholder.contains(slot);
        marks.hard = marks.placeholder;
        marks.memory = rs.lane_slot_memory_pattern.contains(slot);
        return marks;
    }
    // Ordinary constant AND dynamic lane reads have the same possible source words. A clean
    // overwritten constant spill slot above remains separate from other lanes' marks.
    return vector_word(rs, reg);
}

void record_vector_word_marks(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source) {
    const uint32_t width = rdna2_vgpr_write_count(in);
    if (!width) return;
    const bool writelane = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361;
    const bool relative = in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
    const bool partial = in.sdwa_dst_sel != 6 && in.sdwa_dst_unused == 2;
    const int end = relative ? rs.max_vgpr + 1 : in.dst.value + static_cast<int>(width);
    for (int reg = in.dst.value; reg < end; ++reg) {
        ScalarSourceMarks result = source;
        // CONFIDENCE: HIGH -- RDNA2 70648's ordinary VALU writes retain inactive lanes; WRITELANE
        // replaces one selected lane regardless of EXEC. Partial/relative writes retain old bits.
        if (rs.exec_narrowed || writelane || relative || partial)
            union_source_marks(result, vector_word(rs, reg));
        set_word(rs.vreg_source_marks, reg, std::move(result));
    }
}

void join_vector_word_marks(VectorWordMarks& into, const VectorWordMarks& other) {
    for (const auto& [reg, source] : other) union_source_marks(into[reg], source);
}

void mark_vector_loop_carried(RegState& rs, const std::vector<Rdna2Inst>& ins, uint32_t lo,
                              uint32_t hi, std::map<int, std::set<uint64_t>>& carried) {
    std::set<int> vectors, scalars;
    loop_written_regs(ins, lo, hi, vectors, scalars);
    for (int reg : vectors) {
        auto source = vector_word(rs, reg);
        const int root_word = kVectorBlanketBase + reg;
        if (rs.vreg.contains(reg) && !source.hard) {
            carried[root_word] = source.blanket_roots;
            source.blanket_roots.insert(blanket_root(lo, root_word));
        } else {
            source.hard = true;
            source.blanket_roots.clear();
        }
        source.placeholder = true;
        rs.vreg_source_marks[reg] = std::move(source);
    }
}

ScalarSourceMarks carried_word_marks(const RegState& rs, int word) {
    if (word >= kVectorBlanketBase) return vector_word(rs, word - kVectorBlanketBase);
    ScalarSourceMarks marks;
    marks.placeholder = merge_edge_word_fabricated(rs, word);
    const auto blanket = rs.sreg_loop_blanket.find(word);
    marks.hard = marks.placeholder && !sreg_word_blanket_only(rs, word);
    if (marks.placeholder && !marks.hard) marks.blanket_roots = blanket->second;
    return marks;
}

}   // namespace prosper::gpu

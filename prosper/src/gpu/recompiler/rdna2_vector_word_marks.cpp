#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {

// True when each result lane depends only on that same lane's sources and old word: no DPP,
// no cross-lane permute, no lane-selecting write, no relative destination, no LDS exchange.
// Accumulators (VOP2/VOP3 mac/fmac) qualify: they read the old destination, but only their own
// lane's. SDWA field-select and per-lane memory loads qualify for the same reason.
bool lane_mapped_vector_write(const Rdna2Inst& in) {
    if (in.has_dpp) return false;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            return in.opcode != 0x02 && in.opcode != kVop1OpcodeMovreldB32 &&
                   !(in.opcode >= 0x40 && in.opcode <= 0x47);
        case Rdna2Format::VOP2: return true;
        case Rdna2Format::VOPC: return true;
        case Rdna2Format::VOP3: return in.opcode < 0x300;
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::FLAT: return true;
        default: return false;
    }
}

// True only for plain lane-mapped arithmetic without SDWA partial destinations: DPP and
// unmodeled SDWA/DPP forms (has_modifier) may move a value across lanes or preserve bits.
bool lane_wise_vector_op(const Rdna2Inst& in) {
    return !in.has_modifier && lane_mapped_vector_write(in);
}

// True when a lane-wise form writes the complete 32-bit word of every lane it executes: no
// partial destination, no preserved old bits (vector_write_depends_on_old_word, with EXEC treated
// as full so only the form's own dependence counts), and not an indexed relative write.
bool complete_full_word_write(const Rdna2Inst& in) {
    // A plain buffer load writes each active lane's dwords (out-of-range lanes return zero); a TFE
    // form also writes a status word, which is not modelled here, so it is not complete.
    if (in.fmt == Rdna2Format::MUBUF) return rdna2_tfe_status_vgpr(in) < 0;
    // A modeled SDWA form writes the complete word only for a full-DWORD destination with
    // zero-fill (UNUSED_PAD). PRESERVE keeps old bits and sub-dword selects write fields, and
    // unmodeled forms keep has_modifier; none of those is complete.
    if (in.has_sdwa) return !in.has_modifier && in.sdwa_dst_sel == 6 && in.sdwa_dst_unused == 0;
    if (!lane_wise_vector_op(in)) return false;
    if (in.fmt == Rdna2Format::VOP1 && (in.opcode == 0x02 || in.opcode == kVop1OpcodeMovreldB32))
        return false;
    return !vector_write_depends_on_old_word(in, /*full_exec=*/true);
}

// True when the EXEC value named by reader is the owner value or a subset of it, by the derivation
// facts recorded at v_cmpx. A placeholder confined outside owner is then also outside reader.
bool exec_within(const RegState& rs, uint32_t reader, uint32_t owner) {
    uint32_t current = reader;
    for (int hop = 0; hop < 64 && current != 0; ++hop) {
        if (current == owner) return true;
        const auto parent = rs.exec_subset_of.find(current);
        if (parent == rs.exec_subset_of.end()) return false;
        current = parent->second;
    }
    return false;
}
bool memory_result(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
           in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG ||
           in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::DS;
}

ScalarSourceMarks vector_word(const RegState& rs, int reg) {
    const auto found = rs.vreg_source_marks.find(reg);
    return found == rs.vreg_source_marks.end() ? ScalarSourceMarks{} : found->second;
}

void set_word(VectorWordMarks& marks, int reg, ScalarSourceMarks source) {
    if (source.placeholder || source.memory)
        marks[reg] = std::move(source);
    else
        marks.erase(reg);
}

bool relative_constant_offset(const SpirvCompute& b, uint32_t value, uint32_t& offset) {
    if (b.uconst_literal(value, &offset)) return true;
    // A completed two-edge PHI of the same genuine constant has that exact value too. Pending
    // loop backedges have zero value/label placeholders and cannot certify an index here.
    for (std::size_t pc = 0; pc < b.code.size();) {
        const uint32_t words = b.code[pc] >> 16;
        if (!words || words > b.code.size() - pc) return false;
        if ((b.code[pc] & 0xffffu) == Op_Phi && words == 7 && b.code[pc + 1] == b.t_u32 &&
            b.code[pc + 2] == value && b.code[pc + 3] == b.code[pc + 5] && b.code[pc + 4] &&
            b.code[pc + 6] && b.code[pc + 4] != b.code[pc + 6])
            return b.uconst_literal(b.code[pc + 3], &offset);
        pc += words;
    }
    return false;
}

}   // namespace

void seed_vector_word_definedness(RegState& rs, const std::vector<Rdna2Inst>& instructions,
                                  bool native_interpolation, uint32_t wave_size) {
    // CONFIDENCE: HIGH -- operand_bits/vreg_old supply zero for absent ordinary VGPRs. That
    // placeholder is not a guest value. Seed before PHIs or dispatcher storage can hide absence;
    // later complete writers retire the mark, while preserved old bits/lanes retain it.
    const int highest = shader_max_vgpr(instructions);
    rs.vector_wave_size = wave_size;
    if (wave_size == 32) {
        // These producing entries supply no architectural SCC input; bfalse is storage only.
        rs.scc_merge_placeholder = true;
        rs.scc_loop_blanket.clear();
    }
    rs.vector_full_exec = rs.exec && !rs.exec_narrowed;
    rs.vector_native_interpolation = native_interpolation;
    for (int reg = 0; reg <= highest; ++reg) {
        const auto value = rs.vreg.find(reg);
        if (value != rs.vreg.end() && value->second) continue;
        auto& mark = rs.vreg_source_marks[reg];
        mark.placeholder = mark.hard = true;
        mark.blanket_roots.clear();
    }
}

void union_source_marks(ScalarSourceMarks& into, const ScalarSourceMarks& other) {
    // EXEC confinement survives a join only when every placeholder side shares it. A clean side
    // contributes no placeholder lanes; a soft placeholder may sit in any lane.
    const uint32_t into_side = into.hard ? into.outside_exec : 0u;
    const uint32_t other_side = other.hard ? other.outside_exec : 0u;
    if (into.placeholder && other.placeholder)
        into.outside_exec = into_side == other_side ? into_side : 0u;
    else if (other.placeholder)
        into.outside_exec = other_side;
    else
        into.outside_exec = into_side;
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
    if (vector_write_leaves_old_word(in, rs.vector_native_interpolation) ||
        (rs.vector_native_interpolation && in.fmt == Rdna2Format::VINTRP))
        return marks;
    // Load addresses are not loaded data. A genuine read still carries MEMORY, which only the
    // fragment lane-bit projection rejects; compute keeps its actual guest-lane identity.
    if (memory_result(in)) {
        marks.memory = true;
        if (in.fmt != Rdna2Format::DS) return marks;
    }
    for (uint32_t source = 0; source < in.n_src; ++source)
        for (uint32_t word = 0; word < vector_provenance_source_span(in, source); ++word)
            union_source_marks(marks,
                               vector_word(rs, in.src[source].value + static_cast<int>(word)));
    // Relative indexing can select an absent word beyond the decoder's storage inventory. A
    // constant current M0 may refine this at the actual emission boundary; maps alone cannot.
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43) {
        marks.placeholder = marks.hard = true;
        marks.blanket_roots.clear();
    }
    return marks;
}

ScalarSourceMarks vector_relative_source_marks(const SpirvCompute& b, const RegState& rs,
                                               const Rdna2Inst& in) {
    ScalarSourceMarks missing;
    missing.placeholder = missing.hard = true;
    const auto m0 = rs.sreg.find(124);
    uint32_t offset = 0;
    if (m0 == rs.sreg.end() || sreg_word_may_be_fabricated(rs, 124) ||
        !relative_constant_offset(b, m0->second, offset))
        return missing;
    const uint64_t selected = static_cast<uint64_t>(in.src[0].value) + offset;
    if (selected > 255) return missing;
    const auto value = rs.vreg.find(static_cast<int>(selected));
    if (value == rs.vreg.end() || !value->second) return missing;
    auto marks = vector_word(rs, static_cast<int>(selected));
    marks.memory = marks.memory || rs.sreg_memory_pattern.contains(124);
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

bool fresh_mask_sources_defined(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    if (!(b.is_compute || b.is_fragment) || (b.wave_size != 64 && b.wave_size != 32)) return true;
    const bool predicate = in.fmt == Rdna2Format::VOPC || vop3_writes_mask_sdst(in) ||
                           (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a);
    if (!predicate) return true;
    // Read before the destination replaces its prior lifetime. This uses decoded source counts and
    // physical widths, including both words of wide comparisons and marked ordinary VALU relays.
    // A memory-loaded value can form a fresh predicate; a missing spill's zero storage cannot.
    const auto marks = scalar_source_marks(rs, in);
    if (!marks.placeholder) return true;
    // The predicate reads only the lanes its own EXEC enables, and its result is masked by that EXEC,
    // so a placeholder confined to the lanes outside the same EXEC value is never observed.
    if (marks.hard && marks.outside_exec != 0 && rs.exec_narrowed &&
        exec_within(rs, rs.exec, marks.outside_exec))
        return true;
    if (marks.hard || marks.blanket_roots.empty()) return false;
    // These roots name defined loop seeds, whose back edges the existing loop proof checks.
    b.loop_blanket_roots_read.insert(marks.blanket_roots.begin(), marks.blanket_roots.end());
    return true;
}

void record_vector_word_marks(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source) {
    const bool full_exec = !rs.exec_narrowed || rs.vector_full_exec;
    if (rdna2_instruction_may_change_exec(in))
        rs.vector_full_exec = vector_exec_is_full_after_write(in, full_exec, rs.vector_wave_size);
    if (vector_write_leaves_old_word(in, rs.vector_native_interpolation)) return;
    const uint32_t width = rdna2_vgpr_write_count(in);
    if (!width) return;
    const bool relative = in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
    const bool preserve = vector_write_depends_on_old_word(in, full_exec);
    const int end = relative ? rs.max_vgpr + 1 : in.dst.value + static_cast<int>(width);
    for (int reg = in.dst.value; reg < end; ++reg) {
        ScalarSourceMarks result = source;
        // CONFIDENCE: HIGH -- RDNA2 70648's ordinary VALU writes retain inactive lanes; WRITELANE
        // replaces one selected lane regardless of EXEC. Partial/relative writes retain old bits.
        if (preserve) union_source_marks(result, vector_word(rs, reg));
        // A write defines its word only in the lanes it writes. Confinement is granted only when the
        // form writes the complete word of every active lane; any other form (preserved bits,
        // selected lanes, relative destinations, cross-lane forms) leaves the old word unconfined.
        const bool complete = complete_full_word_write(in);
        const bool narrowed_write = rs.exec_narrowed && !rdna2_instruction_may_change_exec(in);
        if (!complete && !lane_mapped_vector_write(in)) {
            result.outside_exec = 0;
        } else if (narrowed_write && complete) {
            // Outside EXEC the old word survives; inside it the result is the source relayed.
            const bool defined_inside =
                !source.placeholder ||
                (source.hard && exec_within(rs, rs.exec, source.outside_exec));
            result.outside_exec = defined_inside ? rs.exec : 0u;
        }
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

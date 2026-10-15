#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"

#include <algorithm>
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
// lane's. SDWA field-select qualifies for the same reason. Memory result marks do not
// authenticate their addresses; their dedicated source proofs must grant confinement.
// The opcode half of the answer is the effects table's (#5143: this used to be "VOP3 below
// 0x300", which left v_add3_u32 and every other VOP3-only lane-wise operation out).
bool lane_mapped_vector_write(const Rdna2Inst& in) {
    return !in.has_dpp && rdna2_valu_lane_mapped(in.fmt, in.opcode) == Rdna2Answer::Yes;
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
    // CONFIDENCE: HIGH -- a completed ordinary image reader writes every selected result dword
    // in every active lane. Its predication preserves only lanes outside EXEC, not destination
    // bits inside EXEC. Empty DMASK has only a conservative storage inventory and assigns no
    // result words. Atomics, packed results and status-bearing forms retain their old proof.
    if (in.fmt == Rdna2Format::MIMG)
        return in.mimg_dmask != 0 && !rdna2_instruction_may_write_memory(in) && !in.mimg_d16 &&
               !in.mimg_tfe && !in.mimg_lwe;
    // A DWORD SDWA destination still replaces all 32 bits, but an accumulator still needs its
    // old destination. Apply the shared old-word check below even for a full SDWA destination.
    if (!lane_wise_vector_op(in) || (in.has_sdwa && in.sdwa_dst_sel != 6)) return false;
    return !vector_write_depends_on_old_word(in, /*full_exec=*/true);
}

// True for the ordinary memory reads, beyond the buffer loads complete_full_word_write already
// names, whose admitted lowering writes every result dword in each EXEC lane from that lane's own
// address and predicates only inactive lanes back to the old word (rdna2_emit_alu.cpp): the plain
// LDS reads, and the typed buffer loads that share the untyped format-load path. Their address
// inputs are the source marks, as for a buffer load. GDS, swizzle, bpermute, append/consume, the
// M0/TID-addressed forms, D16 packing and TFE status writes are excluded: their result depends on
// other lanes, on state these marks do not describe, or keeps part of the old word.
bool memory_read_completes_active_word(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::MTBUF) return in.opcode <= 0x03 && !in.mtbuf_tfe;
    if (in.fmt != Rdna2Format::DS || in.ds_gds) return false;
    switch (in.opcode) {
        case 0x36:   // ds_read_b32
        case 0x37:   // ds_read2_b32
        case 0x38:   // ds_read2st64_b32
        case 0x76:   // ds_read_b64
        case 0x77:   // ds_read2_b64
        case 0xfe:   // ds_read_b96
        case 0xff: return true;   // ds_read_b128
        default: return false;
    }
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

void union_marks(ScalarSourceMarks& into, const ScalarSourceMarks& other,
                 const RegState* current = nullptr, bool explicit_sources = false) {
    const uint32_t into_side = into.hard || into.scoped_complete_word ? into.outside_exec : 0u;
    const uint32_t other_side = other.hard || other.scoped_complete_word ? other.outside_exec : 0u;
    auto scoped = into.placeholder && !into.hard ? into.scoped_blanket_roots : into.blanket_roots;
    const auto& other_scoped =
        other.placeholder && !other.hard ? other.scoped_blanket_roots : other.blanket_roots;
    scoped.insert(other_scoped.begin(), other_scoped.end());
    if (into.placeholder && other.placeholder) {
        // CONFIDENCE: HIGH -- outside A union outside B is outside B when B is a proved
        // subset of A. Each subset fact is about two immutable EXEC SSA values, so it holds on
        // every path reaching them: current instruction sources and the structured merges that
        // pass their state use it. A merge that passes no state keeps the equality-only meet. An
        // owner defined on one arm only cannot match any reader after the merge, which sees a
        // phi instead.
        uint32_t owner = into_side == other_side ? into_side : 0u;
        if (!owner && current && into_side && other_side) {
            if (exec_within(*current, into_side, other_side))
                owner = into_side;
            else if (exec_within(*current, other_side, into_side))
                owner = other_side;
        }
        // CONFIDENCE: HIGH -- a soft side with roots has a global proof that covers every lane:
        // its placeholder is only the loop-header assumption those blanket_roots name, which the
        // back edge checks (vector_word_defined_in_exec treats a soft mark the same way). It
        // therefore restricts no confined side's domain; it adds those global roots to the
        // obligation of that domain. Without this, a complete load or relay confined under a
        // narrowed EXEC lost its owner the first time it met an ordinary loop-carried word. A
        // soft mark without roots proves nothing and still vetoes the owner.
        if (!owner && explicit_sources) {
            if (!other.hard && !other.blanket_roots.empty() && into_side) {
                owner = into_side;
                scoped = into.hard ? into.blanket_roots : into.scoped_blanket_roots;
                scoped.insert(other.blanket_roots.begin(), other.blanket_roots.end());
            } else if (!into.hard && !into.blanket_roots.empty() && other_side) {
                owner = other_side;
                scoped = other.hard ? other.blanket_roots : other.scoped_blanket_roots;
                scoped.insert(into.blanket_roots.begin(), into.blanket_roots.end());
            }
        }
        into.outside_exec = owner;
    } else if (other.placeholder) {
        into.outside_exec = other_side;
    } else {
        into.outside_exec = into_side;
    }
    into.placeholder = into.placeholder || other.placeholder;
    into.memory = into.memory || other.memory;
    into.hard = into.hard || other.hard;
    if (into.hard && !into.outside_exec)
        into.blanket_roots.clear();
    else
        into.blanket_roots.insert(other.blanket_roots.begin(), other.blanket_roots.end());
    // CONFIDENCE: HIGH -- a complete soft writer has two proofs: the global carried value and
    // its own active domain. Hard-source unions need only the latter. Predecessors retain it only
    // for an equal owner on every marked edge; an unconfined soft source has no scoped owner.
    into.scoped_complete_word = !into.hard && into.outside_exec != 0;
    if (into.outside_exec) {
        if (into.hard)
            into.blanket_roots = std::move(scoped);
        else
            into.scoped_blanket_roots = std::move(scoped);
    }
    if (into.hard || !into.outside_exec) into.scoped_blanket_roots.clear();
}

const std::set<uint64_t>& source_roots_in_exec(const RegState& rs, const ScalarSourceMarks& marks,
                                               uint32_t exec) {
    if (!marks.hard && marks.scoped_complete_word && marks.outside_exec &&
        exec_within(rs, exec, marks.outside_exec))
        return marks.scoped_blanket_roots;
    return marks.blanket_roots;
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

bool vector_write_completes_active_word(const Rdna2Inst& in) {
    return complete_full_word_write(in) ||
           (in.fmt == Rdna2Format::FLAT && in.opcode >= 0x08 && in.opcode <= 0x0f &&
            (in.flat_segment == 0 || in.flat_segment == 2) && !in.flat_lds &&
            in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 && in.src[0].value < 255 &&
            in.src[1].kind == OperandKind::Special && in.src[1].value == 125);
}

bool vector_read_completes_active_word(const Rdna2Inst& in) {
    return memory_read_completes_active_word(in);
}

void record_exec_mask_subset(RegState& rs, uint32_t result, uint32_t left, uint32_t right,
                             uint32_t opcode, uint32_t previous_exec) {
    // CONFIDENCE: HIGH -- A&B is a subset of each operand; A&~B is a subset of A.
    // A|B is a subset of A only when B was already proved a subset of A (and vice versa).
    // These relations do not authenticate an operand or invent its missing physical words.
    uint32_t parent = 0;
    if (opcode == 0x0f)
        parent = right == previous_exec ? right : left;
    else if (opcode == 0x15)
        parent = left;
    else if (opcode == 0x11) {
        if (exec_within(rs, left, right))
            parent = right;
        else if (exec_within(rs, right, left))
            parent = left;
    }
    if (parent && result && result != parent) rs.exec_subset_of[result] = parent;
    // CONFIDENCE: HIGH -- when B is a proved subset of A, A&~B and B partition A: their union is
    // exactly A. An if/else lowering's second arm (S_ANDN2 EXEC, saved, EXEC) runs under this.
    if (opcode == 0x15 && result && result != left && result != right && left != right &&
        exec_within(rs, right, left))
        rs.exec_complement_in[result] = {left, right};
}

void seed_vector_word_definedness(RegState& rs, const std::vector<Rdna2Inst>& instructions,
                                  bool native_interpolation, uint32_t wave_size) {
    // CONFIDENCE: HIGH -- operand_bits/vreg_old supply zero for absent ordinary VGPRs. That
    // placeholder is not a guest value. Seed before PHIs or dispatcher storage can hide absence;
    // later complete writers retire the mark, while preserved old bits/lanes retain it.
    const int highest = shader_max_vgpr(instructions);
    rs.vector_wave_size = wave_size;
    const bool scc_offset =
        std::any_of(instructions.begin(), instructions.end(), [](const auto& in) {
            return (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                   in.n_src >= 3 && in.src[2].kind == OperandKind::Special &&
                   in.src[2].value == 253;
        });
    if (scc_offset) rs.track_scc_definition = true;
    if (wave_size == 32 || scc_offset) {
        // These producing entries supply no architectural SCC input; bfalse is storage only.
        // A numeric buffer offset requires its actual producer even in ordinary Wave64.
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
    union_marks(into, other);
}

ScalarSourceMarks vector_source_marks(const RegState& rs, const Rdna2Inst& in) {
    ScalarSourceMarks marks;
    if (vector_write_leaves_old_word(in, rs.vector_native_interpolation) ||
        (rs.vector_native_interpolation && in.fmt == Rdna2Format::VINTRP))
        return marks;
    // Load addresses are not loaded data, but an unavailable address cannot produce genuine
    // loaded data. Capture that obligation before an overlapping destination replaces it.
    // MEMORY remains a separate classification, never proof that the address was supplied.
    if (memory_result(in)) {
        marks.memory = true;
        const auto missing_address = [&] {
            marks.placeholder = marks.hard = true;
            marks.outside_exec = 0;
            marks.blanket_roots.clear();
        };
        const auto address_word = [&](int reg) {
            const auto value = rs.vreg.find(reg);
            const auto source = vector_word(rs, reg);
            if (reg < 0 || reg > 255 || value == rs.vreg.end() || !value->second ||
                !vector_word_defined_in_exec(rs, source, rs.exec)) {
                missing_address();
            } else if (!marks.hard && !source.blanket_roots.empty()) {
                marks.blanket_roots.insert(source.blanket_roots.begin(),
                                           source.blanket_roots.end());
            }
        };
        if (in.fmt == Rdna2Format::FLAT && in.src[0].kind == OperandKind::VGPR) {
            // The architectural FLAT/GLOBAL pointer is a pair even when a resolved window
            // lowers its offset using only the low word. A binding cannot supply either word.
            address_word(in.src[0].value);
            address_word(in.src[0].value + 1);
        } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
            const uint32_t words = ((in.literal >> 12) & 1u) + ((in.literal >> 13) & 1u);
            for (uint32_t word = 0; word < words; ++word)
                address_word(in.src[0].value + static_cast<int>(word));
            const auto& offset = in.src[2];
            // SOFFSET consumes scalar DATA: VCC scratch, ttmp and M0 carry the same current
            // unavailable/relay/loop marks as ordinary SGPRs despite decoding as Special.
            const bool scalar_word =
                offset.kind == OperandKind::SGPR ||
                (offset.kind == OperandKind::Special && offset.value >= 106 && offset.value <= 124);
            if (scalar_word) {
                if (rs.unavailable_saved_mask_words.contains(offset.value)) {
                    missing_address();
                } else if (sreg_word_blanket_only(rs, offset.value)) {
                    const auto& roots = rs.sreg_loop_blanket.at(offset.value);
                    if (!marks.hard) marks.blanket_roots.insert(roots.begin(), roots.end());
                } else {
                    // An untracked VCC word is materialized from the current mask only by the
                    // emitter's exact native-wave gate. Its Bool presence grants no DATA here.
                    const bool vcc_mask_word = offset.kind == OperandKind::Special &&
                                               offset.value <= 107 &&
                                               !rs.sreg.contains(offset.value);
                    if (sreg_word_may_be_fabricated(rs, offset.value) ||
                        (!vcc_mask_word && !b32_source_word_defined(rs, offset)))
                        missing_address();
                }
            } else if (offset.kind == OperandKind::Special && offset.value == 253) {
                if (!rs.scc || (rs.scc_merge_placeholder && rs.scc_loop_blanket.empty())) {
                    missing_address();
                } else if (rs.scc_merge_placeholder && !marks.hard) {
                    marks.blanket_roots.insert(rs.scc_loop_blanket.begin(),
                                               rs.scc_loop_blanket.end());
                }
            }
        }
        if (in.fmt != Rdna2Format::DS) return marks;
    }
    for (uint32_t source = 0; source < in.n_src; ++source)
        for (uint32_t word = 0; word < vector_provenance_source_span(in, source); ++word)
            union_marks(marks, vector_word(rs, in.src[source].value + static_cast<int>(word)), &rs,
                        true);
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
    if ((marks.hard || marks.scoped_complete_word) && marks.outside_exec != 0 && rs.exec_narrowed &&
        exec_within(rs, rs.exec, marks.outside_exec)) {
        // An unchanged SSA EXEC owner does not prove the next iteration's carried value. Record
        // the header obligation before accepting this scoped read; the back edge checks it.
        const auto& roots = source_roots_in_exec(rs, marks, rs.exec);
        b.loop_blanket_roots_read.insert(roots.begin(), roots.end());
        return true;
    }
    if (marks.hard || marks.blanket_roots.empty()) return false;
    // These roots name defined loop seeds, whose back edges the existing loop proof checks.
    b.loop_blanket_roots_read.insert(marks.blanket_roots.begin(), marks.blanket_roots.end());
    return true;
}

void record_native_interpolation_publication(const SpirvCompute& b, RegState& rs,
                                             const Rdna2Inst& in) {
    rs.native_interpolation_publication = {};
    // CONFIDENCE: HIGH -- only the existing native fragment lowering publishes this Input
    // word. P1 remains its no-op, and owned packets still need the architectural P1/old-P2/M0
    // inputs. Neither a decoder write span nor a CFG storage reload authenticates this fact.
    if (!b.is_fragment || b.is_compute || b.is_vertex || b.ngg_workgroup_shell || b.ngg_one_lane ||
        b.guest_stage != GuestShaderStage::Fragment ||
        b.physical_domain != PhysicalExecutionDomain::Invocation ||
        !rs.vector_native_interpolation || !rs.exec || in.fmt != Rdna2Format::VINTRP ||
        (in.opcode != 1 && in.opcode != 2) || in.len_dwords != 1 || in.n_src != 1 ||
        in.dst.kind != OperandKind::VGPR || in.dst.value < 0 || in.dst.value > 255 ||
        in.src[0].kind != OperandKind::VGPR || in.src[0].value < 0 || in.src[0].value > 255 ||
        in.vintrp_attr >= 32 || in.vintrp_chan > 3 || in.has_modifier || in.has_sdwa ||
        in.has_dpp || in.has_literal || in.sdst.kind != OperandKind::None ||
        in.words[0] != (0xc8000000u | (in.opcode << 16u) |
                        (static_cast<uint32_t>(in.dst.value) << 18u) | (in.vintrp_attr << 10u) |
                        (in.vintrp_chan << 8u) | static_cast<uint32_t>(in.src[0].value)))
        return;
    const auto* layout = b.fragment_interpolation;
    const uint32_t attribute = 1u << in.vintrp_attr;
    if (!layout || !layout->valid || !(layout->attribute_mask & attribute)) return;
    if (in.opcode == 1) {
        if (!(layout->smooth_mask & attribute)) return;
    } else if (in.src[0].value >= 3 || (!layout->requires_geometry && in.src[0].value != 2) ||
               (layout->requires_geometry &&
                layout->parameter_locations[in.vintrp_attr][in.src[0].value] >= 32)) {
        return;
    }
    const auto value = rs.vreg.find(in.dst.value);
    if (value == rs.vreg.end() || !value->second) return;
    rs.native_interpolation_publication = {in.pc, in.dst.value, rs.exec, value->second};
}

void record_vector_word_marks(RegState& rs, const Rdna2Inst& in, const ScalarSourceMarks& source) {
    const auto interpolation = std::exchange(rs.native_interpolation_publication, {});
    const auto value = rs.vreg.find(in.dst.value);
    const bool published_interpolation =
        rs.vector_native_interpolation && in.fmt == Rdna2Format::VINTRP &&
        (in.opcode == 1 || in.opcode == 2) && interpolation.pc == in.pc &&
        interpolation.destination == in.dst.value && interpolation.exec == rs.exec &&
        interpolation.value && value != rs.vreg.end() && interpolation.value == value->second;
    const bool full_exec = !rs.exec_narrowed || rs.vector_full_exec;
    if (rdna2_instruction_may_change_exec(in))
        rs.vector_full_exec = vector_exec_is_full_after_write(in, full_exec, rs.vector_wave_size);
    if (vector_write_leaves_old_word(in, rs.vector_native_interpolation)) return;
    const uint32_t width = rdna2_vgpr_write_count(in);
    if (!width) return;
    const bool relative = in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32;
    const bool preserve = vector_write_depends_on_old_word(in, full_exec);
    // CONFIDENCE: HIGH -- these reads are complete in their active lanes exactly as a buffer
    // load is; only this transfer function uses that fact (CFG DATA facts keep their own).
    const bool complete = published_interpolation || complete_full_word_write(in) ||
                          memory_read_completes_active_word(in);
    const int end = relative ? rs.max_vgpr + 1 : in.dst.value + static_cast<int>(width);
    for (int reg = in.dst.value; reg < end; ++reg) {
        const auto old = vector_word(rs, reg);
        ScalarSourceMarks result = source;
        // CONFIDENCE: HIGH -- RDNA2 70648's ordinary VALU writes retain inactive lanes; WRITELANE
        // replaces one selected lane regardless of EXEC. Partial/relative writes retain old bits.
        if (preserve) union_marks(result, old, &rs);
        // Only a complete write can grant new confinement. An incomplete lane-mapped write keeps
        // the existing meet of source and old-word confinement; selected-lane, relative and
        // cross-lane writes cannot retain that proof.
        const bool narrowed_write = rs.exec_narrowed && !rdna2_instruction_may_change_exec(in);
        if (!complete && !lane_mapped_vector_write(in)) {
            result.outside_exec = 0;
        } else if (narrowed_write && complete) {
            // Outside EXEC the old word survives; inside it the result is the source relayed.
            const bool image_proof = in.fmt == Rdna2Format::MIMG && result.hard;
            const bool defined_inside =
                vector_word_defined_in_exec(rs, source, rs.exec) &&
                (!image_proof || (rs.mimg_sources.pc == in.pc && rs.mimg_sources.enabled &&
                                  rs.mimg_sources.defined));
            result.outside_exec = defined_inside ? rs.exec : 0u;
            result.scoped_complete_word = defined_inside && !result.hard &&
                                          lane_mapped_vector_write(in) && !memory_result(in);
            if (result.scoped_complete_word)
                result.scoped_blanket_roots = source_roots_in_exec(rs, source, rs.exec);
            if (defined_inside && result.hard) {
                // A narrower write leaves genuine old lanes in the rest of the prior owner. Keep
                // that wider domain and its carried obligation; an equal-owner complete write
                // replaces the whole required domain and depends only on its current sources.
                const bool keep_wider = old.hard && old.outside_exec != 0 &&
                                        old.outside_exec != rs.exec &&
                                        exec_within(rs, rs.exec, old.outside_exec);
                result.blanket_roots = source_roots_in_exec(rs, source, rs.exec);
                if (image_proof)
                    result.blanket_roots.insert(rs.mimg_sources.roots.begin(),
                                                rs.mimg_sources.roots.end());
                // CONFIDENCE: HIGH -- the second arm of a partition: EXEC = whole & ~part with
                // part a proved subset of whole. An old word confined within part (its missing
                // lanes lie outside a superset of part) misses no lane of part, and this write
                // completes the rest of whole, so the result misses lanes only outside whole.
                const auto partition = rs.exec_complement_in.find(rs.exec);
                const bool completes_partition =
                    !keep_wider && old.hard && old.outside_exec != 0 &&
                    partition != rs.exec_complement_in.end() &&
                    exec_within(rs, partition->second.second, old.outside_exec);
                if (keep_wider || completes_partition) {
                    result.outside_exec = keep_wider ? old.outside_exec : partition->second.first;
                    result.blanket_roots.insert(old.blanket_roots.begin(), old.blanket_roots.end());
                }
            }
        }
        if (result.hard && !result.outside_exec) result.blanket_roots.clear();
        if (result.hard || !result.outside_exec) result.scoped_complete_word = false;
        if (!result.scoped_complete_word) result.scoped_blanket_roots.clear();
        set_word(rs.vreg_source_marks, reg, std::move(result));
    }
}

void join_vector_word_marks(VectorWordMarks& into, const VectorWordMarks& other,
                            const RegState* current) {
    for (const auto& [reg, source] : other) union_marks(into[reg], source, current);
}

void retire_vector_word_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& violated) {
    const auto invalid = [&](const auto& roots) {
        return std::any_of(roots.begin(), roots.end(),
                           [&](uint64_t root) { return violated.contains(root); });
    };
    if (invalid(marks.blanket_roots)) {
        const bool scoped = !marks.hard && marks.scoped_complete_word && marks.outside_exec &&
                            !invalid(marks.scoped_blanket_roots);
        marks.blanket_roots = scoped ? marks.scoped_blanket_roots : std::set<uint64_t>{};
        marks.hard = true;
        marks.scoped_complete_word = false;
        if (!scoped) marks.outside_exec = 0;
        marks.scoped_blanket_roots.clear();
    } else if (!marks.hard && marks.scoped_complete_word && invalid(marks.scoped_blanket_roots)) {
        marks.outside_exec = 0;
        marks.scoped_complete_word = false;
        marks.scoped_blanket_roots.clear();
    }
}

bool vector_word_defined_in_exec(const RegState& rs, const ScalarSourceMarks& marks,
                                 uint32_t exec) {
    if (!marks.placeholder) return true;
    if (!marks.hard) return !marks.blanket_roots.empty();
    return marks.outside_exec != 0 && exec_within(rs, exec, marks.outside_exec);
}

void begin_mimg_source_proof(RegState& rs, uint32_t pc) {
    rs.mimg_sources = {pc, rs.exec_narrowed, true, {}, {}};
}

void record_mimg_source_word(RegState& rs, int reg) {
    auto& proof = rs.mimg_sources;
    if (!proof.enabled || !proof.words.insert(reg).second) return;
    const auto value = rs.vreg.find(reg);
    const auto marks = vector_word(rs, reg);
    if (reg < 0 || reg > 255 || value == rs.vreg.end() || !value->second ||
        !vector_word_defined_in_exec(rs, marks, rs.exec))
        proof.defined = false;
    proof.roots.insert(marks.blanket_roots.begin(), marks.blanket_roots.end());
}

void record_mimg_address_operand(RegState& rs, const Rdna2Inst& in, uint32_t index) {
    // NSA's first VGPR is VADDR; subsequent register numbers come from the actual extra dwords.
    // Zero padding beyond the encoded instruction cannot invent an additional v0 operand.
    if (rs.mimg_sources.enabled && in.len_dwords > 2u && index > 0u &&
        2u + (index - 1u) / 4u >= in.len_dwords)
        rs.mimg_sources.defined = false;
}

uint32_t read_mimg_source_word(SpirvCompute& b, RegState& rs, int reg) {
    record_mimg_source_word(rs, reg);
    const auto value = rs.vreg.find(reg);
    return value == rs.vreg.end() ? b.uconst(0) : value->second;
}

int mimg_address_vgpr(RegState& rs, const Rdna2Inst& in, uint32_t index) {
    record_mimg_address_operand(rs, in, index);
    if (in.len_dwords <= 2u) return in.src[0].value + static_cast<int>(index);
    if (!index) return in.src[0].value;
    const uint32_t offset = index - 1u;
    return static_cast<int>((in.words[2u + offset / 4u] >> (8u * (offset % 4u))) & 0xffu);
}

void mark_vector_loop_carried(RegState& rs, const std::vector<Rdna2Inst>& ins, uint32_t lo,
                              uint32_t hi, std::map<int, std::set<uint64_t>>& carried,
                              std::map<int, uint32_t>& confined) {
    std::set<int> vectors, scalars;
    loop_written_regs(ins, lo, hi, vectors, scalars);
    for (int reg : vectors) {
        auto source = vector_word(rs, reg);
        // A new soft header generation needs its own backedge obligation; its global proof
        // survives, but a prior complete writer's active-domain proof does not bypass that edge.
        if (!source.hard) {
            source.outside_exec = 0;
            source.scoped_complete_word = false;
            source.scoped_blanket_roots.clear();
        }
        const int root_word = kVectorBlanketBase + reg;
        if (rs.vreg.contains(reg) && (!source.hard || source.outside_exec != 0)) {
            carried[root_word] = source.blanket_roots;
            if (source.hard) confined[root_word] = source.outside_exec;
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

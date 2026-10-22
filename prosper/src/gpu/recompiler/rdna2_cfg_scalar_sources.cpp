// Existing scalar input-mark propagation and saved-mask snapshots.
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {
void scalar_source_word_marks(const RegState& rs, ScalarSourceMarks& marks, int& r) {
    const bool fabricated = sreg_word_may_be_fabricated(rs, r);
    marks.placeholder = marks.placeholder || fabricated;
    // A scalar word is unconfined: it may be absent in every lane, so no EXEC confinement survives it.
    // Nor is it a never-loaded VGPR: scalar data stays hard under #5174 and #5185. A loop
    // assumption's word is not hard: like a soft vector word it adds only its roots, which an
    // arbitrary result keeps as obligations (#5193 companion rule).
    const auto blanket = rs.sreg_loop_blanket.find(r);
    if (fabricated) marks.outside_exec = 0;
    if (fabricated && blanket == rs.sreg_loop_blanket.end()) marks.arbitrary = false;
    if (fabricated && blanket == rs.sreg_loop_blanket.end())
        marks.hard = true;
    else if (fabricated)
        marks.blanket_roots.insert(blanket->second.begin(), blanket->second.end());
    marks.memory = marks.memory || rs.sreg_memory_pattern.contains(r);
}
}   // namespace

// Whether `in` is an s_movrels_b32 the lowering can emit (rdna2_movrels.cpp). What it writes is
// then a scalar word: the select over what each candidate holds. A candidate in another domain (a
// saved mask half, an entry-M0 token) refuses the whole instruction. A candidate with no tracked
// value does NOT refuse; it contributes the zero placeholder every scalar read uses, so the
// result is exactly as good as the candidate M0 selects, and wrong when that one was untracked.
//
// The dispatcher's scalar-word transfer asks this INSTEAD of charging the candidates, and the
// reason is what failing that transfer means. It is not a refusal: the destination is dropped
// from the words a case boundary reloads, and reads the zero placeholder afterwards. So dropping
// can only turn a right result into zero, never a wrong one into a refusal. Charged the base
// alone, a table whose first register was never written lost its result at the next join.
// Charged the whole range s[base..105], every result did, since almost no program defines all of
// those words. scalar_alu_source_words describes the read range for refusing proofs; this is
// for the one that drops.
bool s_movrels_b32_result_is_scalar_data(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovrelsB32 &&
           in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
           in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 && in.src[0].value <= 105;
}
// Scalar instructions whose encoded destination is also an implicit source. These reads must be
// observed before writer transfer functions invalidate the old lifetime. Every SOPK form whose SDST
// field is READ belongs here, not merely the arithmetic band: none of them decodes a src operand at
// all, so a caller scanning `n_src` sees nothing for any of them. SOPK has no decoded src
// operands at all; conditional moves preserve the old destination on one SCC outcome, comparisons
// consume SDST without writing it, and ADDK/MULK are ordinary read-modify-write operations. SOP1
// conditional moves and bitset operations have an explicit source too, but it is the replacement
// value/bit index rather than the old destination.
uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::SGPR) return 0;
    if (in.fmt == Rdna2Format::SOPK) {
        if (in.opcode == 0x02 ||   // s_cmovk_i32
            (in.opcode >= 0x03 && in.opcode <= 0x10) ||   // s_cmpk_*, s_addk, s_mulk
            // s_setreg_b32: the encoded SDST field is the SOURCE SGPR, which rdna2_emit_alu.cpp's
            // own handler says in as many words. It reads the register and writes none.
            in.opcode == 0x13 ||
            // s_subvector_loop_begin/end read SDST as the loop counter.
            in.opcode == 0x1b || in.opcode == 0x1c)
            return 1;
        return 0;
    }
    if (in.fmt != Rdna2Format::SOP1) return 0;
    switch (in.opcode) {
        case kSop1OpcodeCmovB32: return 1;
        case kSop1OpcodeCmovB64: return 2;
        case kSop1OpcodeBitset0B32: return 1;
        case 0x1c: return 2;   // s_bitset0_b64
        case kSop1OpcodeBitset1B32: return 1;
        case 0x1e: return 2;   // s_bitset1_b64
        default: return 0;
    }
}

// The input marks of one instruction (#4706; RegState::sreg_merge_placeholder). Defined here, not
// in rdna2_to_spirv_internal.hpp, because it reads the two width inventories above; every caller
// of snapshot_saved_b64_masks() includes this header.
ScalarSourceMarks scalar_source_marks(const RegState& rs, const Rdna2Inst& in) {
    ScalarSourceMarks marks = vector_source_marks(rs, in);
    const auto word = [&](int r) { scalar_source_word_marks(rs, marks, r); };
    if (in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG ||
        in.fmt == Rdna2Format::FLAT) {   // load addresses are not the loaded data
        marks.memory = true;
        return marks;
    }
    const bool lane_read = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360;
    if (lane_read) marks = vector_lane_read_marks(rs, in);
    // Numeric SCC copies must retain the current condition's missing-input mark, including
    // WRITELANE's data role before its early return. Loaded data remains separate from addresses.
    if (scalar_reads_scc(in) && (!rs.scc || rs.scc_merge_placeholder)) {
        ScalarSourceMarks condition;
        condition.placeholder = true;
        condition.hard = !rs.scc || rs.scc_loop_blanket.empty();
        condition.blanket_roots = rs.scc_loop_blanket;
        union_source_marks(marks, condition);
    }
    if (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x360 || in.opcode == 0x361)) {
        // A dynamic selector is a numeric input too. Its storage value does not prove which
        // peer a READLANE consumes or which peer a WRITELANE replaces.
        if (in.src[1].kind == OperandKind::SGPR ||
            (in.src[1].kind == OperandKind::Special && in.src[1].value <= 124))
            word(in.src[1].value);
        if (!lane_read && (in.src[0].kind == OperandKind::SGPR ||
                           (in.src[0].kind == OperandKind::Special && in.src[0].value <= 124))) {
            word(in.src[0].value);   // v_writelane's data word
        }
        return marks;
    }
    for (uint32_t k = 0; k < in.n_src && k < 3; ++k) {
        const Operand& o = in.src[k];
        // Every register whose marks propagation can set: SGPRs, and the special data words VCC,
        // ttmp0-15 and M0 (106-124).
        const bool scalar = o.kind == OperandKind::SGPR ||
                            (o.kind == OperandKind::Special && o.value >= 106 && o.value <= 124);
        if (!scalar) continue;
        // CNDMASK's third operand consumes the current Bool domain, not the saved mask's
        // physical scalar halves. A fresh predicate can be genuine while those halves cannot
        // be reconstructed. Retain actual merge poison and DATA roles; map membership alone
        // must not authenticate an absent, relayed or numerically overwritten mask.
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101 && k == 2 &&
            o.kind == OperandKind::SGPR && o.value >= 0 && o.value <= 105) {
            const auto mask = rs.sreg_bool.find(o.value);
            const uint32_t mask_words = rs.sreg_bool_b32.contains(o.value) ? 1u : 2u;
            bool genuine = mask != rs.sreg_bool.end() && mask->second;
            for (uint32_t w = 0; w < mask_words; ++w) {
                const int reg = o.value + static_cast<int>(w);
                genuine = genuine && !rs.sreg_merge_placeholder.contains(reg);
                marks.memory = marks.memory || rs.sreg_memory_pattern.contains(reg);
            }
            if (genuine) continue;
        }
        // A Wave32 lane mask is one SGPR (#5202), VCC_LO excepted (scalar_source_mark_words).
        uint32_t width = scalar_source_mark_words(in, k, rs.vector_wave_size);
        if (width == 0 || width == UINT32_MAX) width = 2;   // unknown: both words of a pair
        if (width > 2) width = static_cast<uint32_t>(std::max(0, 125 - o.value));   // a range
        for (uint32_t w = 0; w < width; ++w) word(o.value + static_cast<int>(w));
    }
    // A read-modify-write keeps the destination's old bits (s_bitset*, s_cmov*, s_addk, ...).
    const uint32_t implicit = scalar_implicit_destination_read_width(in);
    for (uint32_t w = 0; w < implicit; ++w) word(in.dst.value + static_cast<int>(w));
    return marks;
}

SavedB64MaskSnapshot snapshot_saved_b64_masks(const RegState& rs, const Rdna2Inst& in) {
    SavedB64MaskSnapshot snapshot;
    snapshot.source_marks = scalar_source_marks(rs, in);
    // The widest write form is deliberate: this set only FILTERS what record_scalar_write may
    // expire, and that function applies its own exact `effective_width`, so an extra candidate root
    // here can never widen the erase set.
    for_each_scalar_write(
        in,
        [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int root = base + static_cast<int>(word);
                for (int candidate : {root - 1, root}) {
                    if (candidate < 0 || candidate > 105 || rs.sreg_bool_b32.contains(candidate))
                        continue;
                    const auto mask = rs.sreg_bool.find(candidate);
                    if (mask != rs.sreg_bool.end())
                        snapshot.entries.emplace_back(candidate, mask->second);
                }
            }
        },
        /*wave32_one_word_masks*/ false);
    return snapshot;
}

// Explicit emission-domain snapshot: host load_state reconstruction is not an architectural write.
SavedB64MaskSnapshot snapshot_saved_b64_masks(SpirvCompute& b, const RegState& rs,
                                              const Rdna2Inst& in) {
    auto snapshot = snapshot_saved_b64_masks(rs, in);
    note_memory_address(b, rs, in, snapshot.source_marks);   // #5193, before the roots are cleared
    if (in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF) {
        // Address assumptions are read before an aliasing write. Keep them on the loaded value:
        // a weak back edge makes it arbitrary, so a later strong address consumer must charge
        // these same roots rather than treating the loaded value as unconditional DATA (#5204).
        auto& roots = snapshot.source_marks.blanket_roots;
        b.loop_blanket_roots_read.insert(roots.begin(), roots.end());
        // Scalar roots cannot hold weakly; retire their address-only dependencies as before.
        std::erase_if(roots, [](uint64_t root) {
            return static_cast<int>(root & 0xffffu) < kVectorBlanketBase;
        });
        if (!roots.empty()) snapshot.source_marks.placeholder = true;
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43)
        snapshot.source_marks = vector_relative_source_marks(b, rs, in);
    snapshot.portable_numeric_replacement = ordinary_portable_wave64_masks(b);
    return snapshot;
}

}   // namespace prosper::gpu

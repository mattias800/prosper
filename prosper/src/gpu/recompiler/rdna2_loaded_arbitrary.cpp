// See rdna2_loaded_arbitrary.hpp.
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"

#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace prosper::gpu {
namespace {

// MIMG opcodes below 0x20 (loads, stores, atomics, GET_RESINFO) take no sampler; the SAMPLE,
// GATHER4 and GET_LOD families from 0x20 up read their S# (ISA 70648 MIMG opcode table).
// CONFIDENCE: HIGH.
constexpr uint32_t kMimgFirstSamplerOpcode = 0x20;

// CONFIDENCE: MED -- a resource is fabricated when one of its words carries a mark: merged with an
// edge that never wrote it, relayed from fabricated data, or a saved-mask half that cannot be
// rebuilt. A word with no mark is the driver's own data: the emitter resolves the descriptor
// through the resource table (an unresolvable one is refused there), whether or not RegState
// holds a numeric value for it, so absence alone does not make it fabricated here.
bool scalar_word_genuine(const RegState& rs, int reg) {
    return reg >= 0 && reg <= 124 && !rs.unavailable_saved_mask_words.contains(reg) &&
           !rs.sreg_merge_placeholder.contains(reg);
}

bool scalar_words_genuine(const RegState& rs, int first, int words) {
    for (int word = 0; word < words; ++word)
        if (!scalar_word_genuine(rs, first + word)) return false;
    return true;
}

// A GLOBAL load the indirect-pointer relocation proof authorizes by exact PC and packet. Its
// lowering (SpirvCompute::relocated_indirect_*_load_dword) reads only inside a proved record and
// returns zero for any other address; a packet the lowering cannot take is refused there.
bool relocated_global_read(const SpirvCompute& b, const Rdna2Inst& in) {
    return dispatcher_arbitrary_read(in, {}, {}, false, b.indirect_pointer_proof) &&
           in.fmt == Rdna2Format::FLAT;
}

// CONFIDENCE: MED -- #5193 for a relocated GLOBAL read: every word of the address pair is defined
// in EXEC or arbitrary, so the bounded lowering loads a value hardware could have loaded (or the
// zero it returns outside every record). Its defined words' roots are read here, before the
// snapshot clears them, exactly as for a defined address.
void admit_relocated_read(const RegState& rs, const Rdna2Inst& in, ScalarSourceMarks& source) {
    if (in.src[0].kind != OperandKind::VGPR) return;
    std::set<uint64_t> roots;
    for (int word = 0; word < 2; ++word) {
        const int reg = in.src[0].value + word;
        const auto found = rs.vreg_source_marks.find(reg);
        if (reg < 0 || reg > 255) return;
        if (found == rs.vreg_source_marks.end()) continue;
        const auto& marks = found->second;
        const auto value = rs.vreg.find(reg);
        const bool defined = vector_word_defined_in_exec(rs, marks, rs.exec) &&
                             value != rs.vreg.end() && value->second;
        if (!defined && !(marks.placeholder && marks.hard && marks.arbitrary)) return;
        roots.insert(marks.blanket_roots.begin(), marks.blanket_roots.end());
    }
    source.placeholder = source.hard = source.arbitrary = true;
    source.outside_exec = 0;
    source.blanket_roots = std::move(roots);
}

}   // namespace

bool arbitrary_read_form(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::MUBUF:
            return !rdna2_instruction_may_write_memory(in) && !in.mubuf_lds &&
                   rdna2_vgpr_write_count(in) != 0;
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
            return !rdna2_instruction_may_write_memory(in) && rdna2_vgpr_write_count(in) != 0;
        case Rdna2Format::DS:
            if (in.ds_gds) return false;
            // CONFIDENCE: HIGH -- the plain LDS reads: each EXEC lane loads at its own VGPR address
            // (plus offsets) and nothing else. Not ADDTID (no VGPR address), not swizzle, permute,
            // bpermute or append/consume (their result depends on other lanes or on counters), not
            // any atomic or write.
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
        default: return false;
    }
}

bool arbitrary_read_resource_genuine(const RegState& rs, const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::DS: return true;   // LDS has no descriptor and no scalar address input
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: {
            if (in.src[1].kind != OperandKind::SGPR ||
                !scalar_words_genuine(rs, in.src[1].value, 4))
                return false;
            const auto& offset = in.src[2];
            if (offset.kind == OperandKind::SGPR ||
                (offset.kind == OperandKind::Special && offset.value >= 106 && offset.value <= 124))
                return scalar_word_genuine(rs, offset.value) &&
                       !rs.unavailable_saved_mask_words.contains(offset.value);
            if (offset.kind == OperandKind::Special && offset.value == 253)
                return rs.scc && !rs.scc_merge_placeholder;
            return true;   // an inline constant
        }
        case Rdna2Format::MIMG:
            if (in.src[1].kind != OperandKind::SGPR ||
                !scalar_words_genuine(rs, in.src[1].value, in.mimg_r128 ? 4 : 8))
                return false;
            return in.opcode < kMimgFirstSamplerOpcode ||
                   (in.src[2].kind == OperandKind::SGPR &&
                    scalar_words_genuine(rs, in.src[2].value, 4));
        default: return false;
    }
}

bool dispatcher_arbitrary_read(const Rdna2Inst& in, const std::set<int>& scalar_words,
                               const std::set<int>& relayed, bool definite_scc,
                               const IndirectPointerRelocationProof* relocation) {
    if (in.fmt == Rdna2Format::FLAT)
        return relocation && rdna2_indirect_pointer_access(*relocation, in) &&
               !rdna2_instruction_may_write_memory(in);
    if (!arbitrary_read_form(in)) return false;
    // As scalar_word_genuine: only a relayed (MAY fabricated) word is a hard descriptor word. A
    // scalar offset is DATA, which must also be a MUST-defined word, as the address charge needs.
    const auto genuine = [&](int first, int words) {
        for (int word = 0; word < words; ++word)
            if (first + word < 0 || first + word > 124 || relayed.contains(first + word))
                return false;
        return true;
    };
    switch (in.fmt) {
        case Rdna2Format::DS: return true;
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF: {
            if (in.src[1].kind != OperandKind::SGPR || !genuine(in.src[1].value, 4)) return false;
            const auto& offset = in.src[2];
            if (offset.kind == OperandKind::SGPR ||
                (offset.kind == OperandKind::Special && offset.value >= 106 && offset.value <= 124))
                return genuine(offset.value, 1) && scalar_words.contains(offset.value);
            if (offset.kind == OperandKind::Special && offset.value == 253) return definite_scc;
            return true;
        }
        // The dispatcher relays no image result: an image read keeps its old word's facts.
        default: return false;
    }
}

bool mimg_result_may_be_arbitrary(const RegState& rs, const Rdna2Inst& in) {
    const auto& proof = rs.mimg_sources;
    return in.fmt == Rdna2Format::MIMG && arbitrary_read_form(in) && proof.pc == in.pc &&
           proof.enabled && proof.no_hard && arbitrary_read_resource_genuine(rs, in);
}

void note_memory_address(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                         ScalarSourceMarks& source) {
    b.lds_bound_reads = false;
    const bool relocated = relocated_global_read(b, in);
    if (relocated && source.placeholder && source.hard) admit_relocated_read(rs, in, source);
    if (in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF) {
        // CONFIDENCE: HIGH -- a bounded read at an address that turns out arbitrary still loads a
        // value hardware could have loaded (#5193); FLAT has no bound, and a write or atomic at the
        // wrong address changes memory hardware did not. Those need the address DEFINED.
        if (!arbitrary_read_form(in) && !relocated)
            b.loop_blanket_roots_strong.insert(source.blanket_roots.begin(),
                                               source.blanket_roots.end());
        return;
    }
    if (in.fmt != Rdna2Format::DS || in.src[0].kind != OperandKind::VGPR) return;
    const auto found = rs.vreg_source_marks.find(in.src[0].value);
    if (found == rs.vreg_source_marks.end() || !found->second.placeholder) return;
    const auto& address = found->second;
    // CONFIDENCE: MED -- an LDS read whose address may hold an arbitrary lane bounds its index by
    // the declared allocation and returns zero outside it (SpirvCompute::lds_load). Workgroup
    // storage has no robust-access bound, so an unbounded index is undefined behaviour, not a
    // value. Zero is the out-of-range read ISA 70648 gives for LDS; for an arbitrary lane any
    // in-range word, or that zero, is a value hardware could have read, so the property this
    // admits does not rest on the exact out-of-range value. A hard address keeps a hard result
    // (vector_source_marks), so bounding it changes nothing it admits.
    if (address.hard && !vector_word_defined_in_exec(rs, address, rs.exec) &&
        arbitrary_read_form(in)) {
        b.lds_bound_reads = true;
        return;
    }
    // A loop assumption read through an unbounded LDS address: an arbitrary back edge would make
    // this access out of range, so the assumption must hold strictly (the address DEFINED).
    b.loop_blanket_roots_strong.insert(address.blanket_roots.begin(), address.blanket_roots.end());
    b.loop_blanket_roots_strong.insert(address.scoped_blanket_roots.begin(),
                                       address.scoped_blanket_roots.end());
}

uint32_t lds_index_inside_if_noted(SpirvCompute& b, uint32_t index) {
    if (!b.lds_bound_reads) return 0;
    const uint32_t dwords = b.is_compute ? b.lds_dwords : b.vertex_lds_dwords;
    return b.ucmp(Op_ULessThan, index, b.uconst(dwords));
}

}   // namespace prosper::gpu

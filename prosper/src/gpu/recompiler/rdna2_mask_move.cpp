#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <optional>

namespace prosper::gpu {
namespace {
uint32_t numeric_pair_mask_bit(SpirvCompute& b, uint32_t lo, uint32_t hi, uint32_t lane) {
    const uint32_t word = b.sel(b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)), hi, lo);
    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
    return b.ucmp(Op_INotEqual,
                  b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                  b.uconst(0));
}
} // namespace

// Consume the existing instruction-order CFG MUST domains, before a stale within-case Bool
// spelling can bypass a physical high-word overwrite. Scalar initialization also counts saved
// masks and is deliberately NOT this numeric-word proof.
const char* packet_b64_mask_move_source_gap(const SpirvCompute& b, const Rdna2Inst& in,
                                            const std::set<int>& masks,
                                            const std::set<int>& scalar_words,
                                            const std::set<int>& ambiguous) {
    const bool saveexec = in.opcode == kSop1OpcodeAndSaveexecB64;
    if (!b.is_fragment_packet() || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
        (!saveexec && (in.opcode != 0x04 || (in.dst.value != 126 && in.dst.value != 106))) ||
        in.src[0].kind != OperandKind::SGPR)
        return nullptr;
    const int source = in.src[0].value;
    const bool low = scalar_words.contains(source), high = scalar_words.contains(source + 1);
    if (low && high) return nullptr;
    if (!low && !high && masks.contains(source) && !ambiguous.contains(source)) return nullptr;
    return saveexec              ? "packet-saveexec-mask-source-words-unavailable"
           : in.dst.value == 126 ? "packet-exec-mask-source-words-unavailable"
                                 : "packet-vcc-mask-source-words-unavailable";
}

// RDNA2 70648 section12.3/page117: AND_SAVEEXEC reads the complete source and OLD EXEC before
// publishing either destination. Resolve a current numeric pair before any older Bool spelling;
// the CFG guard above proves both words on every reaching path. Neither sreg_input nor a missing
// physical half supplies current authority. A genuine zero pair still produces a predicate ID.
// Passing the existing fallback preserves native lowering and its emission order unchanged.
uint32_t packet_and_saveexec_source_mask(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                         uint32_t fallback) {
    if (!b.is_fragment_packet() || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
        in.opcode != kSop1OpcodeAndSaveexecB64 || in.src[0].kind != OperandKind::SGPR)
        return fallback;
    const auto low = rs.sreg.find(in.src[0].value), high = rs.sreg.find(in.src[0].value + 1);
    if (low == rs.sreg.end() || high == rs.sreg.end()) return fallback;
    return numeric_pair_mask_bit(b, low->second, high->second,
                                 b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63)));
}

// A complete scalar move INTO VCC defines both physical words, independently of EXEC, and
// preserves SCC (AMD RDNA2 70648 sections 3.3/12.3). Publish the exact logical-lane bit as well
// as emit_alu's unchanged DATA copy. Owned packets and exact-native compute retain a guest lane;
// an old predicate cannot supply this new value. Native words need independent uniformity and
// genuine provenance under their entry contract. Still-current driver words and loop assumptions
// retain their existing lifetime contract; each assumed root must pass the actual backedge check.
bool numeric_vcc_move_has_lane_context(const SpirvCompute& b) {
    const bool fragment = b.is_fragment && !b.is_compute && !b.is_vertex &&
                          b.guest_stage == GuestShaderStage::Fragment &&
                          b.physical_domain == PhysicalExecutionDomain::Invocation &&
                          !b.allow_b32_masks && !b.ngg_workgroup_shell && !b.ngg_one_lane &&
                          native_vcc_copy_execution_domain_matches(b);
    return b.is_fragment_packet() ||
           ((b.is_compute || fragment) && uses_native_saved_mask_words(b) &&
            saved_mask_physical_launch_matches_original(b));
}

uint32_t s_mov_b64_numeric_vcc_bit(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    if (!numeric_vcc_move_has_lane_context(b) || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
        in.opcode != kSop1OpcodeMovB64 || in.dst.value != 106)
        return 0;
    const auto& source = in.src[0];
    if (!b.is_fragment_packet()) {
        if (!b.native_uniform_numeric_vcc_move_pcs.contains(in.pc) ||
            in.pc >= b.saved_mask_original_words.size())
            return 0;
        auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                         b.saved_mask_original_words.size() - in.pc);
        original.pc = in.pc;
        if (in.synthetic_terminator || in != original) return 0;
    }
    uint32_t lo = 0, hi = 0;
    if (source.kind == OperandKind::SGPR ||
        (source.kind == OperandKind::Special && source.value >= 106 && source.value <= 123)) {
        if (!b.is_fragment_packet()) {
            for (int word = source.value; word <= source.value + 1; ++word)
                if (rs.unavailable_saved_mask_words.contains(word)) return 0;
            // This records only the soft roots actually read here. LoopBlanketCheck rejects
            // their entire module if a backedge violates them; hard/missing words still refuse.
            if (!scalar_words_projectable(b, rs, source.value, 2)) return 0;
        }
        const auto current_word = [&](int word) -> uint32_t {
            if (const auto value = rs.sreg.find(word); value != rs.sreg.end()) return value->second;
            if (!b.is_fragment_packet() && !rs.sreg_written.contains(word))
                if (const auto entry = rs.sreg_input.find(word); entry != rs.sreg_input.end())
                    return entry->second;
            return 0;
        };
        lo = current_word(source.value);
        hi = current_word(source.value + 1);
        if (!lo || !hi) return 0;
    } else
        return 0; // existing mask/inline lowering is unchanged
    return numeric_pair_mask_bit(b, lo, hi, b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63)));
}

// A physical numeric replacement ends the old mask spelling even when this stage cannot
// project the new pair. Keeping VCC's old mirror would silently restore departed mask bits.
void retire_data_only_b64_mask(RegState& rs, int destination) {
    rs.sreg_bool.erase(destination);
    rs.sreg_bool_narrowed.erase(destination);
    if (destination == 106 || destination == 107) rs.vcc = 0;
}

void retire_scalar_data_word_mask(RegState& rs, int destination) {
    rs.sreg_bool.erase(destination);
    rs.sreg_bool_narrowed.erase(destination);
    rs.sreg_bool_b32.erase(destination);
    if (destination == 106 || (destination == 107 && !rs.sreg_bool_b32.contains(106)))
        retire_data_only_b64_mask(rs, 106);
}

// Resolve the entire old pair before publishing either destination. A legal self-copy must
// retain both original values and descriptor offsets throughout publication.
bool copy_s_mov_b64_scalar_pair(RegState& rs, int source, int destination, bool source_mask) {
    std::array<std::optional<uint32_t>, 2> values, descriptors;
    for (int word = 0; word < 2; ++word) {
        const int reg = source + word;
        const auto current = rs.sreg.find(reg);
        if (current != rs.sreg.end())
            values[word] = current->second;
        else {
            const auto input = rs.sreg_input.find(reg);
            if (input != rs.sreg_input.end()) values[word] = input->second;
        }
        const auto descriptor = rs.sreg_srt.find(reg);
        if (descriptor != rs.sreg_srt.end()) descriptors[word] = descriptor->second;
    }
    for (int word = 0; word < 2; ++word) {
        const int reg = destination + word;
        const auto& value = values[word];
        const auto& descriptor = descriptors[word];
        if (value.has_value())
            rs.sreg[reg] = value.value();
        else
            rs.sreg.erase(reg);
        if (descriptor.has_value())
            rs.sreg_srt[reg] = descriptor.value();
        else
            rs.sreg_srt.erase(reg);
    }
    // Ordinary DATA moves keep the established absent-word behavior. A Bool alias still
    // needs both genuine physical values, rather than one copied word and a guessed sibling.
    return (values[0].has_value() && values[1].has_value()) || !source_mask;
}

// AMD RDNA2 70648 sections 3.3/12.3: S_MOV_B64 copies the complete scalar pair into EXEC,
// independently of old EXEC, and does not write SCC. A physical compute packet uses its owned
// logical lane, not a native fragment subgroup or the value of either source word as a Bool.
bool emit_s_mov_b64_exec(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    const auto& source = in.src[0];
    if (uses_native_saved_mask_words(b) && source.kind == OperandKind::SGPR &&
        (!rs.sreg_bool.contains(source.value) || rs.sreg_bool_b32.contains(source.value)) &&
        (rs.unavailable_saved_mask_words.contains(source.value) ||
         rs.unavailable_saved_mask_words.contains(source.value + 1))) {
        bool ok = true;
        refuse_unavailable_saved_mask_word(b, in, &ok);
        return false;
    }
    if (source.kind == OperandKind::InlineInt && source.value == -1) {
        rs.exec = b.btrue();
        rs.exec_narrowed = false;
        return true;
    }
    const auto src_mask = [&]() -> uint32_t {
        if (source.value == 106 || source.value == 107) return rs.vcc;
        if (source.value == 126 || source.value == 127) return rs.exec;
        if (source.kind == OperandKind::SGPR) {
            const auto mask = rs.sreg_bool.find(source.value);
            if (mask != rs.sreg_bool.end()) return mask->second;
        }
        if (source.kind == OperandKind::InlineInt) return inline_int_mask_bit(b, source.value);
        return 0;
    };
    if (ordinary_portable_wave64_masks(b) && source.kind == OperandKind::SGPR) {
        // CONFIDENCE: HIGH -- ordinary compute retains the guest lane identity without a ballot.
        // Current numeric words take precedence over a competing older Bool spelling. Both must
        // exist in the current scalar file and pass its provenance guard; an entry-time input or
        // the ordinary absent-word-as-zero spelling cannot restore an overwritten saved pair.
        const bool low = rs.sreg.contains(source.value), high = rs.sreg.contains(source.value + 1);
        if (low && high) {
            if (const auto mask = scalar_pair_lane_bit(b, rs, source)) {
                rs.exec = mask;
                rs.exec_narrowed = true;
                return true;
            }
        } else if (!low && !high && !rs.sreg_bool_b32.contains(source.value) && src_mask()) {
            // An unchanged complete Bool save/restore needs neither physical words nor a vote.
            rs.exec = src_mask();
            const auto narrowed = rs.sreg_bool_narrowed.find(source.value);
            rs.exec_narrowed = narrowed == rs.sreg_bool_narrowed.end() || narrowed->second;
            return true;
        }
        b.stage_reject_pc = in.pc;
        b.stage_reject_reason = "portable-exec-mask-source-words-unavailable";
        return false;
    }
    if (b.is_fragment_packet() && b.wave_size == 64 && source.kind == OperandKind::SGPR) {
        // Dispatcher reloads retain only MUST-reaching numeric words. The packet entry and
        // resource services publish current values in sreg; do NOT resurrect an entry-time
        // sreg_input after a later mask writer has overwritten its physical word.
        const auto low = rs.sreg.find(source.value);
        const auto high = rs.sreg.find(source.value + 1);
        const bool has_low = low != rs.sreg.end(), has_high = high != rs.sreg.end();
        if (has_low && has_high) {
            // Exact numeric words win over a competing old Bool spelling. In particular, a
            // high-word overwrite must not silently retain the saved pair's earlier mask.
            const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
            rs.exec = numeric_pair_mask_bit(b, low->second, high->second, lane);
            rs.exec_narrowed = true;
            return true;
        }
        if (has_low || has_high || !src_mask()) {
            // PacketRawMasks materializes admitted complete ordinary saved pairs BEFORE a half
            // overwrite. A remaining partial source has no reaching complete producer/words;
            // allocation, scalar initialization or a stale Bool cannot supply its absent half.
            b.stage_reject_pc = in.pc;
            b.stage_reject_reason = "packet-exec-mask-source-words-unavailable";
            return false;
        }
    }
    uint32_t m = src_mask();
    // A native saved pair can round-trip as genuine physical DATA through spill slots. Rebuild
    // only from its two current unmarked words; an old complete Bool needs no new ballot.
    if (!m && uses_native_saved_mask_words(b) && b.is_compute && source.kind == OperandKind::SGPR &&
        rs.sreg.contains(source.value) && rs.sreg.contains(source.value + 1))
        m = scalar_pair_lane_bit(b, rs, source);
    // Preserve the native-fragment route and its expression order. Its subgroup-local identity
    // records a required native Wave64 contract; the new owned packet route above never calls it.
    if (!m && b.is_fragment && b.wave_size == 64 &&
        (source.kind == OperandKind::SGPR ||
         (source.kind == OperandKind::Special && source.value >= 106 && source.value <= 123))) {
        const auto scalar_word = [&](int reg) -> uint32_t {
            const auto current = rs.sreg.find(reg);
            if (current != rs.sreg.end()) return current->second;
            const auto input = rs.sreg_input.find(reg);
            return input != rs.sreg_input.end() ? input->second : 0;
        };
        const uint32_t lo = scalar_word(source.value), hi = scalar_word(source.value + 1);
        if (lo && hi) {
            const uint32_t lane = b.ibin(Op_BitwiseAnd, b.subgroup_local_id(), b.uconst(63));
            m = numeric_pair_mask_bit(b, lo, hi, lane);
        }
    }
    if (!m) return false;
    const auto narrowed = rs.sreg_bool_narrowed.find(source.value);
    rs.exec = m;
    rs.exec_narrowed = narrowed != rs.sreg_bool_narrowed.end() ? narrowed->second : true;
    return true;
}

// s_cmov_b64 exec, S0: EXEC = SCC ? S0 : EXEC. SCC is wave-uniform scalar state, so per lane this is
// a select between the source's lane bit and the current EXEC bit; SCC is read, not written. The
// Pathless's merged ES prolog sets EXEC from its ES count with `s_bfm_b64 exec, s3, 0` and, when
// the count is 64 (`s_bitcmp1_b32 s3, 6`), widens it with `s_cmov_b64 exec, -1` (#4808). A mask,
// EXEC, VCC or inline source is modelled; a literal or a plain data pair refuses.
bool emit_s_cmov_b64_exec(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    const Operand& src = in.src[0];
    uint32_t source = 0;
    if (src.kind == OperandKind::Literal) return false;
    if (src.value == 126 || src.value == 127)
        source = rs.exec;
    else if (src.kind == OperandKind::InlineInt)
        source = inline_int_mask_bit(b, src.value);
    else if (src.value == 106 || src.value == 107)
        source = rs.vcc;
    else if (src.kind == OperandKind::SGPR) {
        const auto it = rs.sreg_bool.find(src.value);
        if (it != rs.sreg_bool.end()) source = it->second;
    }
    // A merge-filled SCC may be the fabricated bfalse, which would select EXEC on no real
    // condition (GPU-5, #4819 review): only a definite SCC selects.
    if (!source || !rs.scc || !rs.exec || rs.scc_merge_placeholder) return false;
    const uint32_t selected = b.id();
    b.put(b.code, Op_Select, {b.t_bool, selected, rs.scc, source, rs.exec});
    rs.exec = selected;
    rs.exec_narrowed = true;   // conservative: predication stays on
    return true;
}

bool emit_exec_cmov(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != kSop1OpcodeCmovB64 ||
        (in.dst.value != 126 && in.dst.value != 127))
        return false;
    ok = emit_s_cmov_b64_exec(b, rs, in);
    return true;
}

// Architectural DATA view (#4694's program selects it as data): D.u64 from the same width/offset
// the mask above used, so the two views agree bit-for-bit on every input by construction. Recorded
// only when both source words are defined DATA: #4749's mark (or a non-data operand kind) withholds
// it, and the mask path above proceeds exactly as before — so a fabricated word can never become a
// tracked zero here, while genuinely undefined sources keep today's behavior. Split with every
// emitted shift amount masked below 32: a 32-bit SPIR-V shift of 32 or more must select, never
// execute. sreg_srt stays erased by the clobber above (a rewritten value carries no provenance);
// sreg_bool keeps the mask view, which mask consumers read unchanged.
// CONFIDENCE: HIGH on the bitfield formula; MED on operand field widths above 31, shared verbatim
// with the mask preparation (one place to fix if the ISA's S0/S1 field widths say otherwise).
void record_s_bfm_b64_scalar_data(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                  uint32_t width, uint32_t offset, bool ok) {
    auto data_defined = [&](const Operand& o) -> bool {
        if (o.kind == OperandKind::InlineInt || o.kind == OperandKind::Literal) return true;
        if (o.kind == OperandKind::Special && o.value == 125)
            return true;   // SGPR_NULL: the one Special whose data is 0
        if (o.kind == OperandKind::SGPR) return !sreg_word_may_be_fabricated(rs, o.value);
        return false;   // VGPR, masks-as-data and everything else
    };
    if (!ok || !data_defined(in.src[0]) || !data_defined(in.src[1])) return;
    const uint32_t w_ge_32 = b.ucmp(Op_UGreaterThanEqual, width, b.uconst(32));
    const uint32_t w_sub =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, width, b.uconst(32)), b.uconst(31));
    const uint32_t w_low = b.ibin(Op_BitwiseAnd, width, b.uconst(31));
    const uint32_t ones_lo =
        b.sel(w_ge_32, b.uconst(0xFFFFFFFFu),
              b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), w_low), b.uconst(1)));
    const uint32_t ones_hi = b.sel(
        w_ge_32, b.ibin(Op_ISub, b.ibin(Op_ShiftLeftLogical, b.uconst(1), w_sub), b.uconst(1)),
        b.uconst(0));
    const uint32_t o_ge_32 = b.ucmp(Op_UGreaterThanEqual, offset, b.uconst(32));
    const uint32_t o_low = b.ibin(Op_BitwiseAnd, offset, b.uconst(31));
    const uint32_t o_sub =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, offset, b.uconst(32)), b.uconst(31));
    const uint32_t data_lo =
        b.sel(o_ge_32, b.uconst(0), b.ibin(Op_ShiftLeftLogical, ones_lo, o_low));
    // For offset < 32 the high word is the low word's carry-out (none at offset 0, where the
    // shift amount 32 - offset would wrap to 0) OR the high ones shifted by the same offset. The
    // second term must survive offset 0: bfm(33, 0) = 0x1_ffffffff has its high bit set there.
    const uint32_t o_is_zero = b.ucmp(Op_IEqual, offset, b.uconst(0));
    const uint32_t carry_out =
        b.sel(o_is_zero, b.uconst(0),
              b.ibin(Op_ShiftRightLogical, ones_lo,
                     b.ibin(Op_BitwiseAnd, b.ibin(Op_ISub, b.uconst(32), offset), b.uconst(31))));
    const uint32_t hi_low_part =
        b.ibin(Op_BitwiseOr, carry_out, b.ibin(Op_ShiftLeftLogical, ones_hi, o_low));
    const uint32_t data_hi = b.sel(o_ge_32,
                                   b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, ones_lo, o_sub),
                                          b.ibin(Op_ShiftLeftLogical, ones_hi, o_sub)),
                                   hi_low_part);
    rs.sreg[in.dst.value] = data_lo;
    rs.sreg[in.dst.value + 1] = data_hi;
}
}   // namespace prosper::gpu

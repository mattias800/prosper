#include "gpu/recompiler/rdna2_alu_support.hpp"

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
// as emit_alu's unchanged DATA copy. An absent entry predicate cannot supply this new value;
// nor may a supplied old predicate survive a genuine numeric replacement. Dispatcher reloads
// have already removed words without current MUST authority. Never resurrect sreg_input here.
uint32_t packet_s_mov_b64_numeric_vcc_bit(SpirvCompute& b, const RegState& rs,
                                          const Rdna2Inst& in) {
    if (!b.is_fragment_packet() || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
        in.opcode != kSop1OpcodeMovB64 || in.dst.value != 106)
        return 0;
    const auto& source = in.src[0];
    uint32_t lo = 0, hi = 0;
    if (source.kind == OperandKind::SGPR ||
        (source.kind == OperandKind::Special && source.value >= 106 && source.value <= 123)) {
        const auto low = rs.sreg.find(source.value), high = rs.sreg.find(source.value + 1);
        if (low == rs.sreg.end() || high == rs.sreg.end()) return 0;
        lo = low->second;
        hi = high->second;
    } else
        return 0; // existing mask/inline lowering is unchanged
    return numeric_pair_mask_bit(b, lo, hi, b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63)));
}

// AMD RDNA2 70648 sections 3.3/12.3: S_MOV_B64 copies the complete scalar pair into EXEC,
// independently of old EXEC, and does not write SCC. A physical compute packet uses its owned
// logical lane, not a native fragment subgroup or the value of either source word as a Bool.
bool emit_s_mov_b64_exec(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    const auto& source = in.src[0];
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
    if (!source || !rs.scc || !rs.exec) return false;
    const uint32_t selected = b.id();
    b.put(b.code, Op_Select, {b.t_bool, selected, rs.scc, source, rs.exec});
    rs.exec = selected;
    rs.exec_narrowed = true;   // conservative: predication stays on
    return true;
}
}   // namespace prosper::gpu

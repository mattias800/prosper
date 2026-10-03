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
const char* packet_exec_mask_source_gap(const SpirvCompute& b, const Rdna2Inst& in,
                                        const std::set<int>& masks,
                                        const std::set<int>& scalar_words,
                                        const std::set<int>& ambiguous) {
    if (!b.is_fragment_packet() || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
        in.opcode != 0x04 || in.dst.value != 126 || in.src[0].kind != OperandKind::SGPR)
        return nullptr;
    const int source = in.src[0].value;
    const bool low = scalar_words.contains(source), high = scalar_words.contains(source + 1);
    if (low && high) return nullptr;
    if (!low && !high && masks.contains(source) && !ambiguous.contains(source)) return nullptr;
    return "packet-exec-mask-source-words-unavailable";
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
            // A saved Bool mask is not raw numeric halves. A legal partial overwrite needs the
            // preserved genuine half materialized before this transfer; that service is still
            // missing. Allocation/scalar-initialization alone cannot supply it.
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
}   // namespace prosper::gpu

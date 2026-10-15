// sop1_data_routes.cpp — SOP1 scalar-data routes that are not plain u32 functions (getpc, flbit, bitset, bitreplicate, entry-M0).
// Relocated verbatim from emit_alu's SOP1 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop1Lowering::route_not_vcc_half_self() {
    if ((b.is_compute || b.is_fragment) && in.opcode == 0x07 &&
        (in.dst.value == 106 || in.dst.value == 107) &&
        in.src[0].value == in.dst.value &&
        !rs.sreg.contains(in.src[0].value)) {
        // s_not_b32 on one physical VCC half updates only those 32 guest lanes. Large
        // Wave64 compute kernels use `v_cmp ... vcc; s_not_b32 vcc_lo,vcc_lo;
        // s_mov_b64 exec,vcc` to select the complement for lanes 0..31 while retaining the
        // original predicate for lanes 32..63. The per-invocation Bool already represents
        // this lane's VCC bit, so select the inversion only in the addressed half.
        uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(),
                               b.uconst(b.wave_size - 1));
        const uint32_t in_half = in.dst.value == 106
            ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
            : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
        if (!rs.vcc) { ok = false; return true; }
        rs.vcc = b.bsel(in_half, b.logical_not(rs.vcc), rs.vcc);
        rs.sreg_bool[in.dst.value] = rs.vcc;
        rs.sreg_bool_narrowed[in.dst.value] = true;
        if (b.allow_b32_masks) rs.sreg_bool_b32.insert(in.dst.value);
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-half reduction.
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_getpc_b64() {
    if (in.opcode == 0x1f) {   // s_getpc_b64
        // Accepted ONLY when the pcrel pre-pass FOLDED an embedded-table load from this
        // shader — the pair then only feeds that folded chain. Otherwise the PC would flow
        // into unmodeled address math: keep rejecting.
        if (rs.mubuf_pcrel_tables.empty() && rs.smem_pcrel_tables.empty() &&
            rs.mtbuf_pcrel_tables.empty()) {
            ok = false; return true;
        }
        for (int k = 0; k < 2; k++) {
            rs.sreg.erase(in.dst.value + k); rs.sreg_srt.erase(in.dst.value + k);
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_bcnt1_b64_one_lane() {
    if (in.opcode == 0x10 && b.ngg_one_lane) { // s_bcnt1_i32_b64
        // NGG's final primitive packing counts active bits in a saved wave mask. A Vulkan
        // vertex invocation models one guest lane, so the population count of that lane's
        // Boolean mask is exactly 0 or 1. Keep the result in the ordinary scalar-data domain
        // for the following integer packing arithmetic.
        uint32_t mask = 0;
        if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
        else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
        else if (in.src[0].kind == OperandKind::SGPR) {
            auto it = rs.sreg_bool.find(in.src[0].value);
            if (it != rs.sreg_bool.end()) mask = it->second;
        }
        if (!mask) { ok = false; return true; }
        rs.sreg[in.dst.value] = b.sel(mask, b.uconst(1), b.uconst(0));
        rs.sreg_srt.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_ff1_b64_one_lane() {
    if (in.opcode == 0x14 && b.ngg_one_lane) { // s_ff1_i32_b64
        // RDNA2 returns the bit index of the first set bit, or -1 for an empty mask.
        // The exact Astro NGG projection represents guest lane zero only, so a tracked
        // mask has either that bit set (result 0) or no bits set (result 0xffffffff).
        uint32_t mask = 0;
        if (in.src[0].value == 106 || in.src[0].value == 107) mask = rs.vcc;
        else if (in.src[0].value == 126 || in.src[0].value == 127) mask = rs.exec;
        else if (in.src[0].kind == OperandKind::SGPR) {
            auto it = rs.sreg_bool.find(in.src[0].value);
            if (it != rs.sreg_bool.end()) mask = it->second;
        }
        if (!mask) { ok = false; return true; }
        rs.sreg[in.dst.value] = b.sel(mask, b.uconst(0), b.uconst(0xffffffffu));
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_flbit() {
    if (in.opcode == 0x15 || in.opcode == 0x16) {
        // s_flbit_i32_b32 / s_flbit_i32_b64: the count of leading zeros in the UNSIGNED
        // 32/64-bit source, or -1 when the source is zero. Unlike s_ff1/s_bcnt above, the
        // operand here is ordinary scalar DATA, not a wave mask — Sonic Racing:
        // CrossWorlds' downsample kernel does `s_flbit_i32_b64 vcc_lo, s[14:15]` and
        // immediately `s_sub_i32 vcc_lo, 64, vcc_lo`, i.e. it is computing the index of the
        // highest set bit (a log2 / mip count), then feeds that to v_ldexp_f32 (#2013).
        // VERIFIED(round-trip llvm-mc gfx1030): the live word `beea160e` is
        // `s_flbit_i32_b64 vcc_lo, s[14:15]`, and 0x15 is the same operation over one word.
        // The signed forms 0x17/0x18 (s_flbit_i32 / _i32_i64, which count leading SIGN
        // bits) are a different operation and keep rejecting until a title exercises them.
        if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
        auto scalar_word = [&](int reg, uint32_t& value) {
            auto current = rs.sreg.find(reg);
            if (current != rs.sreg.end()) { value = current->second; return true; }
            auto input = rs.sreg_input.find(reg);
            if (input != rs.sreg_input.end()) { value = input->second; return true; }
            return false;
        };
        const bool wide = in.opcode == 0x16;
        uint32_t low = 0, high = b.uconst(0);
        const Operand& source = in.src[0];
        if (source.kind == OperandKind::SGPR ||
            (source.kind == OperandKind::Special &&
             source.value >= 106 && source.value < 124)) {
            if (!scalar_word(source.value, low)) { ok = false; return true; }
            if (wide && !scalar_word(source.value + 1, high)) { ok = false; return true; }
        } else if (source.kind == OperandKind::InlineInt) {
            low = b.uconst(static_cast<uint32_t>(source.value));
            if (wide) high = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
        } else if (source.kind == OperandKind::Literal) {
            low = b.uconst(in.literal);
        } else {
            ok = false; return true;
        }
        // FindUMsb is undefined at zero, so never hand it a zero: OR in bit 0 (which cannot
        // change the highest set bit of a non-zero value) and select the zero answer after.
        auto leading_zeros = [&](uint32_t word) {
            return b.ibin(Op_ISub, b.uconst(31),
                          b.find_umsb(b.ibin(Op_BitwiseOr, word, b.uconst(1))));
        };
        const uint32_t none = b.uconst(0xffffffffu);
        uint32_t result;
        if (!wide) {
            result = b.sel(b.ucmp(Op_INotEqual, low, b.uconst(0)),
                           leading_zeros(low), none);
        } else {
            result = b.sel(
                b.ucmp(Op_INotEqual, high, b.uconst(0)), leading_zeros(high),
                b.sel(b.ucmp(Op_INotEqual, low, b.uconst(0)),
                      b.ibin(Op_IAdd, b.uconst(32), leading_zeros(low)), none));
        }
        rs.sreg[in.dst.value] = result;
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value);
        // The live destination is VCC_LO used purely as scalar data. Overwriting the
        // physical register destroys whatever predicate the pair held, so drop the tracked
        // mask instead of leaving a stale one: a later consumer of VCC then fails visibly
        // rather than branching on a value the guest has already overwritten.
        if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_bitset() {
    if (in.opcode == 0x1b || in.opcode == 0x1d) {
        // s_bitset{0,1}_b32 is an in-place scalar read/modify/write. The encoded source is
        // the bit index while SDST supplies both the old value and destination. Astro's
        // world-map kernel uses `s_bitset1_b32 s5,31` in a resource-table address path;
        // Plucky's post-Desk transition uses `s_bitset0_b32 vcc_hi,31`. Opcode 0x1c is the
        // distinct B64 clear form and must remain fail-visible until both words are modeled.
        if (in.dst.value == 126 || in.dst.value == 127) {
            ok = false; return true;
        }
        uint32_t old_value = 0;
        auto current = rs.sreg.find(in.dst.value);
        if (current != rs.sreg.end()) {
            old_value = current->second;
        } else {
            auto input = rs.sreg_input.find(in.dst.value);
            if (input == rs.sreg_input.end()) { ok = false; return true; }
            old_value = input->second;
        }
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd, val(in.src[0]), b.uconst(31));
        if (!ok) return true;
        const uint32_t mask = b.ibin(
            Op_ShiftLeftLogical, b.uconst(1), bit);
        rs.sreg[in.dst.value] = in.opcode == 0x1b
            ? b.ibin(Op_BitwiseAnd, old_value, b.iun(Op_Not, mask))
            : b.ibin(Op_BitwiseOr, old_value, mask);
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_brev_exec_lo_wave32() {
    if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
        in.opcode == kSop1OpcodeBrevB32) {
        const auto scalar_data_operand = [&](const Operand& source) {
            switch (source.kind) {
                case OperandKind::InlineInt:
                case OperandKind::InlineFloat:
                case OperandKind::Literal:
                    return true;
                case OperandKind::SGPR:
                    return rs.sreg.contains(source.value) ||
                        rs.sreg_input.contains(source.value);
                case OperandKind::Special:
                    if (source.value == 125) return true; // SGPR_NULL
                    if (source.value == 253) return rs.scc != 0;
                    return source.value >= 106 && source.value <= 124 &&
                        rs.sreg.contains(source.value);
                default:
                    return false;
            }
        };
        if (!scalar_data_operand(in.src[0])) { ok = false; return true; }
        // A scalar BREV result written to physical EXEC_LO is still the complete Wave32
        // execution mask. GTA V reverses inline -16 (0xfffffff0) into 0x0fffffff, enabling
        // lanes 0..27. Select the exact result bit per invocation; S_BREV_B32 leaves SCC.
        const uint32_t result = b.iun(Op_BitReverse, val(in.src[0]));
        if (!ok) return true;
        const uint32_t lane = b.ibin(
            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
        rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        rs.exec_narrowed = true;
        rs.sreg.erase(126);
        rs.sreg_srt.erase(126);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_bitreplicate_b64_b32() {
    if (in.opcode == kSop1OpcodeBitreplicateB64B32) { // 0x3b: s_bitreplicate_b64_b32
        const uint32_t a = val(in.src[0]);
        if (!ok) return true;
        const int dst = in.dst.value;
        auto replicate_16 = [&](uint32_t in_val) {
            uint32_t out_val = b.uconst(0);
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, in_val, b.uconst(i)),
                    b.uconst(1u));
                const uint32_t rep = b.ibin(
                    Op_BitwiseOr,
                    b.ibin(Op_ShiftLeftLogical, bit, b.uconst(2u * i)),
                    b.ibin(Op_ShiftLeftLogical, bit, b.uconst(2u * i + 1u)));
                out_val = b.ibin(Op_BitwiseOr, out_val, rep);
            }
            return out_val;
        };
        const uint32_t lo = replicate_16(a);
        const uint32_t hi = replicate_16(b.ibin(Op_ShiftRightLogical, a, b.uconst(16u)));
        rs.sreg_written.insert(dst);
        rs.sreg_written.insert(dst + 1);
        rs.sreg_srt.erase(dst); rs.sreg_srt.erase(dst + 1);
        rs.sreg_bool.erase(dst); rs.sreg_bool.erase(dst + 1);
        rs.sreg_bool_narrowed.erase(dst); rs.sreg_bool_narrowed.erase(dst + 1);
        rs.sreg_bool_b32.erase(dst); rs.sreg_bool_b32.erase(dst + 1);
        rs.sreg_entry_m0.erase(dst); rs.sreg_entry_m0.erase(dst + 1);

        const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63u));
        const uint32_t src_bit_index = b.ibin(Op_ShiftRightLogical, lane, b.uconst(1u));
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, a, src_bit_index),
            b.uconst(1u));
        const uint32_t as_bool = b.ucmp(Op_INotEqual, bit, b.uconst(0u));
        if (dst == 126 || dst == 127) {
            rs.exec = as_bool;
            rs.exec_narrowed = true;
            rs.sreg.erase(126);
            rs.sreg.erase(127);
        } else if (dst == 106 || dst == 107) {
            rs.vcc = as_bool;
            rs.sreg[dst] = lo;
            rs.sreg[dst + 1] = hi;
        } else {
            rs.sreg[dst] = lo;
            rs.sreg[dst + 1] = hi;
        }
        // s_bitreplicate_b64_b32 leaves SCC unmodified per ISA.
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_reject_exec_half_data() {
    // A 32-bit scalar DATA write into an EXEC half would leave the live per-lane mask
    // (rs.exec) stale — hardware updates EXEC (and EXECZ) immediately. No exercised title
    // writes EXEC halves via b32 scalar ops (wave64 compilers use the b64 forms), so
    // reject rather than model it. VCC_LO/HI (106/107) and M0 stay accepted as the
    // documented data-scratch round-trip (NGG preamble s_bfe_u32 vcc_lo, DOLL M0 moves).
    if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_entry_m0_save() {
    // #3133, the M0 entry-value round trip. Compiled code borrows M0 as scratch: it saves
    // the driver's entry value, overwrites M0 with the LDS base it wants, uses it, and puts
    // the original back.
    //
    //     s_mov_b32 s0, m0 | s_movk_i32 m0, 0 | ds_... | s_mov_b32 m0, s0
    //
    // Stray's `0x300c010000` -- a full-screen fragment pass -- does exactly that twice, and
    // the SAVE was an unresolved operand, so the whole stage failed and every draw bound to
    // it was discarded.
    //
    // Both halves are modelled as an OPAQUE TOKEN, never as a number. The save marks the
    // destination as holding entry-M0 and leaves it with no `sreg` value, so arithmetic on
    // it, a V_WRITELANE scalar source, an LDS index -- every consumer that is not the
    // restore -- still rejects exactly as an untracked read does. The restore consumes the
    // token and returns M0 to UNTRACKED, which is the state the movrel / ADDTID /
    // ds_append guards require and check for. No value is invented at any point, so there
    // is no "is zero the right constant" question and no way for a fabricated word to leak
    // into the data domain through a copy.
    //
    // The DESTINATION is gated as tightly as the source. `in.dst.value` is the raw SOP1 DST
    // field (`sgpr(w >> 16)`, 0..127), so it also names VCC_LO/HI (106/107), ttmp0..15
    // (108..123), M0 itself (124) and SGPR_NULL (125). Only an ordinary SGPR may hold the
    // token: those Special words are read back through the `OperandKind::Special` arm of
    // `operand_bits`, which never consults `sreg_entry_m0`, so tokenising one of them would
    // be worse than doing nothing. `s_mov_b32 vcc_lo, m0` is the case that bites -- the arm
    // erases `rs.sreg[106]` but not `rs.vcc`, and a later data read of VCC_LO then
    // materialises the STALE BALLOT as if it were entry-M0. Anything above s105 falls
    // through to the ordinary path below, where an untracked M0 source rejects loudly, i.e.
    // exactly the behaviour that predates this arm. (EXEC, 126/127, already rejected above.)
    if (in.opcode == 0x03 && in.dst.value <= 105 &&
        in.src[0].kind == OperandKind::Special &&
        in.src[0].value == 124 && !rs.sreg.contains(124)) {
        rs.sreg.erase(in.dst.value);
        rs.sreg_entry_m0.insert(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value);
        rs.sreg_written.insert(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_entry_m0_restore() {
    if (in.opcode == 0x03 && in.dst.value == 124 &&
        in.src[0].kind == OperandKind::SGPR &&
        rs.sreg_entry_m0.contains(in.src[0].value) &&
        !rs.sreg.contains(in.src[0].value)) {
        rs.sreg.erase(124);                 // M0 is untracked again, as it was at entry
        rs.sreg_entry_m0.erase(124);
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir

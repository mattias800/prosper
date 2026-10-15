// sop1_b64_mask_routes.cpp — SOP1 64-bit wave-mask routes: s_bcnt1_i32_b64 and the B64 mask/saveexec family.
// Relocated verbatim from emit_alu's SOP1 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop1Lowering::route_bcnt1_b64() {
    if (in.opcode == kSop1OpcodeBcnt1I32B64) {
        // s_bcnt1_i32_b64: popcount the complete scalar pair
        // A VOPC may write an arbitrary SGPR pair as a wave mask.  In the portable vertex
        // shell that pair contains the one represented guest lane, so its exact population
        // count is 0/1.  Ordinary data pairs retain full uint semantics and use two native
        // OpBitCount operations.  GTA's compute shaders also count EXEC, VCC, and saved
        // VOPC masks.  Those are exact only when one native subgroup is the complete guest
        // Wave64; a narrower or unknown host width would silently discard guest lanes.
        // EXEC is tracked separately from scalar DATA.  Until the scalar count can be
        // expanded into architectural EXEC bits, reject either EXEC destination for every
        // source domain rather than only for the native-wave reduction below.
        if (in.dst.value == 126 || in.dst.value == 127) {
            ok = false;
            return true;
        }
        uint32_t result = 0;
        bool reduced_wave_mask = false;
        auto mask = rs.sreg_bool.find(in.src[0].value);
        if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64) {
            const auto data_word_present = [&](int reg) {
                return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
            };
            const int source = in.src[0].value;
            uint32_t source_mask = 0;
            // Match S_FF1's EXEC rule above: canonical special:126 names architectural
            // EXEC, while structured state may also carry synthetic scalar placeholders
            // for the same physical words.  VCC and saved SGPR pairs can genuinely be
            // recycled as scalar data and therefore retain their ambiguity checks.
            if (in.src[0].kind == OperandKind::Special && source == 126) {
                source_mask = rs.exec;
            } else {
                const bool proven_saved_mask =
                    b.structured_wave64_mask_reduction_pcs.contains(in.pc);
                const bool competing_data = !proven_saved_mask &&
                    (data_word_present(source) || data_word_present(source + 1));
                if (!competing_data && source == 106) {
                    source_mask = rs.vcc;
                } else if (!competing_data && in.src[0].kind == OperandKind::SGPR &&
                           !rs.sreg_bool_b32.contains(source) &&
                           mask != rs.sreg_bool.end()) {
                    source_mask = mask->second;
                }
            }
            if (source_mask) {
                result = b.native_wave_popcount(source_mask);
                reduced_wave_mask = true;
            }
        }
        if (!reduced_wave_mask && b.ngg_workgroup_shell && b.wave_size == 64 &&
            b.local_count == 64 &&
            (allow_wave || b.ngg_uniform_wave_reduction_pcs.contains(in.pc)) &&
            in.src[0].kind == OperandKind::SGPR && mask != rs.sreg_bool.end() &&
            !rs.sreg.contains(in.src[0].value) && !rs.sreg.contains(in.src[0].value + 1) &&
            !rs.sreg_input.contains(in.src[0].value) &&
            !rs.sreg_input.contains(in.src[0].value + 1)) {
            // A one-wave merged-NGG workgroup shell holds one complete guest Wave64. Its
            // LDS reduction can count a saved VOPC mask without a native Wave64 subgroup.
            // The compact walk supplies allow_wave only for straight-line execution; the
            // dispatcher has a separate branch-free region proof. The shell's static ABI
            // admission (ngg_subgroup_abi) is what makes its launch state trustworthy.
            result = b.guest_wave_popcount(mask->second);
            reduced_wave_mask = true;
        }
        if (!reduced_wave_mask && !b.is_compute && !b.is_fragment && b.ngg_one_lane &&
                   mask != rs.sreg_bool.end()) {
            result = b.sel(mask->second, b.uconst(1), b.uconst(0));
        } else if (!reduced_wave_mask) {
            // A Boolean-domain pair is a whole wave mask, not two ordinary scalar dwords.
            // Only the exact native-wave and one-lane NGG projections above can reduce it.
            if (mask != rs.sreg_bool.end()) { ok = false; return true; }
            auto scalar_half = [&](int reg, uint32_t& value) {
                auto current = rs.sreg.find(reg);
                if (current != rs.sreg.end()) { value = current->second; return true; }
                auto input = rs.sreg_input.find(reg);
                if (input != rs.sreg_input.end()) { value = input->second; return true; }
                return false;
            };
            uint32_t lo = 0, hi = 0;
            if (in.src[0].kind == OperandKind::SGPR) {
                if (!scalar_half(in.src[0].value, lo) ||
                    !scalar_half(in.src[0].value + 1, hi)) {
                    ok = false; return true;
                }
            } else if (in.src[0].kind == OperandKind::InlineInt) {
                lo = b.uconst(static_cast<uint32_t>(in.src[0].value));
                hi = b.uconst(in.src[0].value < 0 ? UINT32_MAX : 0u);
            } else if (in.src[0].kind == OperandKind::Literal) {
                lo = b.uconst(in.literal); hi = b.uconst(0);
            } else {
                ok = false; return true;
            }
            result = b.ibin(Op_IAdd, b.iun(Op_BitCount, lo), b.iun(Op_BitCount, hi));
        }
        // The result is ordinary scalar DATA regardless of whether its source was a wave
        // mask or a scalar pair.  A one-dword write can overlap either half of a saved B64
        // mask, so terminate both possible Boolean-domain aliases.  VCC has the same
        // split-register hazard and must not retain an older implicit predicate.
        const auto erase_mask_alias = [&](int base) {
            rs.sreg_bool.erase(base);
            rs.sreg_bool_narrowed.erase(base);
            rs.sreg_bool_b32.erase(base);
        };
        erase_mask_alias(in.dst.value);
        if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
        if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
        rs.sreg[in.dst.value] = result;
        rs.sreg_srt.erase(in.dst.value);
        // A B32 write to VCC_LO also replaces virtual lane zero's architectural mask bit.
        // VCC_HI cannot affect that lane.  Other stages keep their multi-lane masks out of
        // the scalar domain and therefore never take this update.
        if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
            const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
            rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
            rs.sreg_bool_narrowed[106] = true;
        }
        rs.scc = b.ucmp(Op_INotEqual, result, b.uconst(0));
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_b64_mask_ops() {
    // 64-bit per-lane MASK ops (EXEC / VCC / saved masks). In our per-invocation model a wave
    // mask is a single bool for this lane. EXEC=SGPR 126/127, VCC=106/107; a saved mask lives
    // in sreg_bool. These implement divergent control flow (if/endif via saveexec + restore).
    if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a ||
        (in.opcode >= 0x24 && in.opcode <= 0x2b) ||
        in.opcode == 0x37 || in.opcode == 0x38) {
        // ISA: every op here EXCEPT s_mov_b64 writes SCC=(result!=0) — a cross-lane
        // reduction the per-invocation model cannot form. POISON the tracked SCC (SSA id 0)
        // so a later consumer (s_cselect/s_addc/SCC-source/scc-branch emission) rejects
        // fail-visibly instead of silently evaluating the value an OLDER s_cmp produced.
        // Any real SCC writer re-arms it. Branches the mask_test/waterfall linearizers
        // claim never read rs.scc, so the exercised adjacent shapes are unaffected.
        if (in.opcode != 0x04) rs.scc = 0;
        auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
        auto src_mask = [&](const Operand& o) -> uint32_t {
            if (o.value == 106 || o.value == 107) return rs.vcc;    // VCC
            if (o.value == 126 || o.value == 127) return rs.exec;   // EXEC
            if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second; }
            if (o.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, o.value);
            return 0;   // not a recognizable mask
        };
        if (in.opcode == 0x0a) {                    // s_wqm_b64: whole-quad-mode mask widen
            // WQM widens a lane mask to whole 2x2 quads so derivative/sample helper lanes stay
            // active. In our per-invocation scalar SPIR-V model each lane is one bool and helper
            // lanes are implicit in the fragment stage, so WQM-widening a mask is the IDENTITY.
            // VERIFIED(round-trip llvm-mc gfx1030): SOP1 op 0x0a. Common PS preamble around
            // image_sample (real game shaders 26-29,39), where it is `s_wqm_b64 exec,exec`.
            // exec_narrowed handling mirrors s_mov_b64 (0x04): the exec<-exec self case is a true
            // no-op (leave exec AND its narrowed flag untouched — so a later forward s_cbranch_execz
            // is not spuriously rejected); replacing exec with a *different* mask may narrow it, so
            // set exec_narrowed conservatively (else inactive-lane writes would escape predication).
            // SCC = (mask != 0) is a CROSS-lane reduction our per-lane model can't form —
            // rs.scc was POISONED above so later SCC consumers reject instead of misreading.
            uint32_t m = src_mask(in.src[0]);
            if (!m) { ok = false; return true; }
            if (is_exec(in.dst)) {
                if (is_exec(in.src[0])) { /* exec <- wqm(exec): identity; exec & narrowed unchanged */ }
                else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                    rs.exec = b.btrue(); rs.exec_narrowed = false;      // exec = all lanes on
                } else { rs.exec = m; rs.exec_narrowed = true; }        // replaced by a (maybe narrowed) mask
            } else { rs.sreg_bool[in.dst.value] = m; rs.sreg_bool_narrowed[in.dst.value] = true;
                mask_write_clobbers_pair(rs, in.dst.value);   // conservative: WQM widens
                // Into VCC this is a VCC write, as it is for s_mov_b64 and s_not_b64
                // below: branches and the e32 selects read rs.vcc, not sreg_bool[106].
                if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = m;
            }
            return true;
        }
        // Narrowed-state carried alongside a saved mask: restoring EXEC from a mask that was saved
        // while EXEC was all-on must clear exec_narrowed (else it stays stuck true past an if/endif
        // — which e.g. breaks a loop that must re-enter the header with full EXEC).
        auto saved_narrowed = [&](const Operand& o) -> bool {
            // Key by operand VALUE (VCC=106/107 and saved SGPR pairs alike; VCC decodes as Special,
            // not SGPR, so don't gate on kind). The flag is kept in sync with the mask at EVERY
            // writer (v_cmp/saveexec/s_mov/s_cselect all update it), so a lookup is accurate.
            // Unknown provenance -> conservatively narrowed (over-narrowing is a safe no-op).
            auto it = rs.sreg_bool_narrowed.find(o.value);
            return it != rs.sreg_bool_narrowed.end() ? it->second : true;
        };
        if (in.opcode == 0x08) {                    // s_not_b64
            const uint32_t m = src_mask(in.src[0]);
            if (!m) { ok = false; return true; }
            const uint32_t inverted = b.logical_not(m);
            if (is_exec(in.dst)) {
                rs.exec = inverted;
                rs.exec_narrowed = true;
            } else if (in.dst.value == 106 || in.dst.value == 107) {
                rs.vcc = inverted;
                rs.sreg_bool[in.dst.value] = inverted;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                mask_write_clobbers_pair(rs, in.dst.value);
            } else {
                rs.sreg_bool[in.dst.value] = inverted;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                mask_write_clobbers_pair(rs, in.dst.value);
            }
            return true;
        }
        if (in.opcode == 0x04) {                    // s_mov_b64
            if (is_exec(in.dst)) {   // set/restore EXEC
                ok = emit_s_mov_b64_exec(b, rs, in);
            } else {   // s_mov_b64 sDST, <mask-or-data> : save a mask / copy a pair
                uint32_t m = s_mov_b64_numeric_mask_bit(b, rs, in);
                if (!m) m = src_mask(in.src[0]);
                if (m && !validate_saved_mask_copy_words(b, rs, in)) {
                    ok = false;
                    return true;
                }
                if (m) { rs.sreg_bool[in.dst.value] = m;
                         rs.sreg_bool_narrowed[in.dst.value] = is_exec(in.src[0]) ? rs.exec_narrowed : saved_narrowed(in.src[0]);
                         // A move INTO VCC is a VCC write (DOLL's scalar-indexed unroll does
                         // `s_mov_b64 vcc, s[4:5]` before its vccz break, #273): the branch reads
                         // rs.vcc, so it must be updated too — sreg_bool[106] alone left the
                         // branch on a stale VCC. (The SOP2 mask ops already special-case 106.)
                         if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = m;
                }
                // s_mov_b64 ALSO moves a plain 64-bit VALUE (a descriptor pair, a scratch pair,
                // an inline constant) — compilers use it to shuffle T#/V# halves and constants,
                // not just wave masks. Copy the data SSA + descriptor provenance (sreg_srt) for
                // both halves so a later ALU read / buffer op resolves; an untracked source half
                // clears the stale dest entry rather than aliasing old data. Mask save (above)
                // and data copy coexist: reads pick their own domain. CONFIDENCE: HIGH.
                bool data_copied = false;
                if (in.src[0].kind == OperandKind::SGPR ||
                    (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 && in.src[0].value <= 123)) {
                    data_copied = copy_s_mov_b64_scalar_pair(rs, in.src[0].value,
                                                             in.dst.value, m != 0);
                } else if (in.src[0].kind == OperandKind::InlineInt) {   // 64-bit sign-extended constant
                    rs.sreg[in.dst.value]     = b.uconst((uint32_t)in.src[0].value);
                    rs.sreg[in.dst.value + 1] = b.uconst(in.src[0].value < 0 ? 0xFFFFFFFFu : 0u);
                    rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                    data_copied = true;
                } else if (in.src[0].kind == OperandKind::Literal) {
                    // 32-bit literal on a B64 move: zero-extended (the f64 high-placement rule is
                    // for double operands only; integer/untyped b64 literals zero-extend).
                    // CONFIDENCE: MED (matches llvm-mc's value validation for s_mov_b64 literals).
                    rs.sreg[in.dst.value]     = b.uconst(in.literal);
                    rs.sreg[in.dst.value + 1] = b.uconst(0);
                    rs.sreg_srt.erase(in.dst.value); rs.sreg_srt.erase(in.dst.value + 1);
                    data_copied = true;
                }
                // A fresh entry-site mask may publish its native ballot words. Older saved
                // pairs copy retained words; a late ballot cannot recover departed lanes.
                if (m && !data_copied && saved_mask_mov_defines_words(b, in, false, true)) {
                    rs.sreg[in.dst.value] = b.native_wave_ballot_half(m, 0);
                    rs.sreg[in.dst.value + 1] = b.native_wave_ballot_half(m, 1);
                    rs.sreg_srt.erase(in.dst.value);
                    rs.sreg_srt.erase(in.dst.value + 1);
                    data_copied = true;
                }
                if (data_copied && !m) {   // data-only move: drop any stale saved-mask alias
                    retire_data_only_b64_mask(rs, in.dst.value);
                }
                // Mask-only source (EXEC/a saved bool with no data half): the pair was still
                // architecturally overwritten — stale data/descriptor entries must not survive.
                if (m && !data_copied) mask_write_clobbers_pair(rs, in.dst.value);
                if (!m && !data_copied) ok = false;
            }
        } else {                                    // s_*_saveexec_b64 sDST, src
            // All ten SAVEEXEC forms read OLD EXEC and S0 before writing either destination.
            // Resolve S0 first because D may itself be VCC (Plucky's exact
            // `s_orn2_saveexec_b64 vcc, exec`); publishing the saved mask early would change
            // a VCC source into OLD EXEC and silently compute a different operation.
            const uint32_t old_exec = rs.exec;
            const bool old_exec_narrowed = rs.exec_narrowed;
            uint32_t m = packet_and_saveexec_source_mask(b, rs, in, src_mask(in.src[0]));
            if (!m) { ok = false; return true; }
            rs.sreg_bool[in.dst.value] = old_exec;  // D = OLD_EXEC
            rs.sreg_bool_narrowed[in.dst.value] = old_exec_narrowed;
            if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = old_exec;
            saveexec_write_clobbers_pair(rs, in.dst.value);
            const uint32_t ne = b.logical_not(old_exec);
            const uint32_t nm = b.logical_not(m);
            const uint32_t xor_em = b.bsel(old_exec, nm, m);
            rs.exec = in.opcode == 0x24 ? b.land(old_exec, m)       // AND
                    : in.opcode == 0x25 ? b.lor(old_exec, m)        // OR
                    : in.opcode == 0x26 ? xor_em                    // XOR
                    : in.opcode == 0x27 ? b.land(m, ne)             // ANDN2: S0 & ~EXEC
                    : in.opcode == 0x28 ? b.lor(m, ne)              // ORN2: S0 | ~EXEC
                    : in.opcode == 0x29 ? b.lor(ne, nm)             // NAND
                    : in.opcode == 0x2a ? b.land(ne, nm)            // NOR
                    : in.opcode == 0x2b ? b.logical_not(xor_em)     // XNOR
                    : in.opcode == 0x37 ? b.land(old_exec, nm)      // GFX10 ANDN1_SAVEEXEC
                                        : b.lor(old_exec, nm);      // GFX10 ORN1_SAVEEXEC
            if (in.opcode == 0x24 || in.opcode == 0x25 || in.opcode == 0x27 ||
                in.opcode == 0x37)
                record_exec_mask_subset(rs, rs.exec, in.opcode == 0x27 ? m : old_exec,
                                        in.opcode == 0x27 ? old_exec : m,
                                        in.opcode == 0x24   ? 0x0f
                                        : in.opcode == 0x25 ? 0x11
                                                            : 0x15,
                                        old_exec);
            // Every non-trivial mask expression may be narrower than full EXEC. Two self
            // identities are exactly all-on and must clear the flag or a later forward
            // EXECZ guard is conservatively (and incorrectly) rejected.
            const bool source_is_exec = in.src[0].value == 126 || in.src[0].value == 127;
            rs.exec_narrowed = !((in.opcode == 0x28 || in.opcode == 0x2b) && source_is_exec);
            // RE-ARM SCC for the FRAGMENT stage (#2418). The poison above is the right default
            // — SCC here is `(result != 0)` over the whole 64-lane mask, which one lane's bool
            // cannot express — but the fragment stage has an exact instrument for precisely
            // this reduction, and this file already uses it for the same purpose after v_cmpx
            // narrows EXEC (`rs.scc = b.is_fragment ? b.fragment_wave_any(rs.exec) : 0;`).
            // `fragment_wave_any`'s contract is "exact scalar wave vote for fragment control
            // flow AND MASK REDUCTIONS".
            //
            // Without it, the poisoned SCC makes every later consumer reject — `s_cselect_b32`
            // at the SOP2 switch (`if (!rs.scc) { ok = false; }`), `s_addc`, and SCC branch
            // emission — which is a large share of GTA V's 23,386 skipped draws, since
            // saveexec-then-`s_cbranch_scc0` is an ordinary shape in its pixel shaders.
            //
            // GATED ON `reads_scc`, and that gate is the whole point. `fragment_wave_any` sets
            // `fragment_required_subgroup_size`, so an unconditional re-arm would give every
            // shader that merely saves EXEC a wave64 requirement it does not need — and that
            // shader is then GATED on a 32-wide device where it works today. On a native-wave64
            // host that regression is invisible, which is exactly why it is guarded here rather
            // than discovered later on other hardware.
            //
            // This is a REDUCE in #2410's taxonomy: the value becomes a guest scalar consumed
            // as data, so forcing wave64 where it IS used is correct and must not be relaxed.
            // Compute and vertex keep the poison — compute has its own synchronized guest-wave
            // reduction paths, and the vertex stage has no equivalent exact vote at all.
            if (b.is_fragment && rs.reads_scc) rs.scc = b.fragment_wave_any(rs.exec);
        }
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
